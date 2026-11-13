"""WP2.6 P4: unit tests of the OHLCV store contract, the in-memory store and the SQL adapter.

No database: the in-memory store, ``prepare_extend`` and ``SqlOhlcvStore`` driven through a
scripted fake session (call order, parameters, error mapping), plus pins on the SQL text
(bound parameters only, exact coverage UPDATE, ON CONFLICT on the primary key, no UPDATE or
DELETE on the bars) and the layering. The real-PostgreSQL suite is
``tests/integration/test_wp26_ohlcv_store_pg.py`` (DSN-gated).
"""

from __future__ import annotations

import ast
import pickle
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql.elements import TextClause
from structlog.testing import capture_logs

import api.services.ohlcv_store as store_mod
from api.db.models import OhlcvBarORM, OhlcvCoverageORM
from api.services.ohlcv_store import DIVERGENT_METRIC, SqlOhlcvStore, store_error_code
from common.metrics import metrics
from common.types import TimeFrame
from data.ohlcv_cache import (
    MAX_EXTEND_BARS,
    STORE_CODES,
    Coverage,
    ExtendResult,
    InMemoryOhlcvStore,
    OhlcvStoreError,
    SeriesKey,
    StoredRead,
)
from data.ohlcv_integrity import Candle, OhlcvDataError, canonicalise_candle
from trading.strategy_engine import _TIMEFRAME_SECONDS

_REPO = Path(__file__).resolve().parents[2]
T0, H = 1_704_067_200_000, 3_600_000
NOW = T0 + 20_000 * H  # settle boundary: T0 + 19_999 H
KEY = SeriesKey("coinbase", "BTC/EUR", 3600)
_GOOD = (Decimal(100), Decimal(110), Decimal(90), Decimal(105), Decimal(1))


def bar(i: int, close: str = "105") -> Candle:
    return canonicalise_candle(T0 + i * H, "100", "110", "90", close, "1", timeframe_s=3600)[0]


def span(a: int, b: int) -> Coverage:
    return Coverage(T0 + a * H, T0 + b * H)


async def ext(store: Any, a: int, b: int, items: list[Candle] | None = None) -> ExtendResult:
    got = [bar(i) for i in range(a, b)] if items is None else items
    return await store.extend(KEY, span(a, b), got, fetched_at_ms=NOW)


# ---------------------------------------------------------------------------
# contract objects
# ---------------------------------------------------------------------------
def test_store_error_is_closed_fixed_text_and_picklable() -> None:
    assert STORE_CODES == (
        "missing_table",
        "unavailable",
        "timeout",
        "denied",
        "coverage_disjoint",
        "stored_row_invalid",
        "error",
    )
    err = pickle.loads(pickle.dumps(OhlcvStoreError("denied")))  # noqa: S301
    assert (err.code, str(err)) == ("denied", "ohlcv store failed (denied)")
    with pytest.raises(ValueError):
        OhlcvStoreError("boom")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "args",
    [("Coinbase", "BTC/EUR", 3600), ("coinbase", "BTCEUR", 3600), ("coinbase", "BTC/EUR", 604800)],
)
def test_series_key_validates_identifiers(args: tuple[str, str, int]) -> None:
    with pytest.raises(OhlcvDataError) as err:
        SeriesKey(*args)
    assert err.value.code == "bad_identifier"


@pytest.mark.parametrize("args", [(5, 3), (-1, 3), (True, 3), (1.0, 3)])
def test_coverage_must_be_an_ordered_int_span(args: tuple[Any, Any]) -> None:
    with pytest.raises(OhlcvDataError):
        Coverage(*args)


# ---------------------------------------------------------------------------
# in-memory store semantics
# ---------------------------------------------------------------------------
async def test_empty_then_extend_then_read_clipped_to_coverage() -> None:
    store = InMemoryOhlcvStore()
    assert await store.read(KEY, T0, T0 + 9 * H) == StoredRead(None, ())
    assert await ext(store, 2, 6) == ExtendResult(4, 0, 0, span(2, 6))
    got = await store.read(KEY, T0, T0 + 4 * H)  # window starts before the coverage
    assert got.coverage == span(2, 6)
    assert [c.open_ms for c in got.bars] == [T0 + 2 * H, T0 + 3 * H]
    assert (await store.read(KEY, T0 + 9 * H, T0 + 12 * H)).bars == ()


async def test_edges_grow_both_ways_and_never_shrink() -> None:
    store = InMemoryOhlcvStore()
    await ext(store, 2, 4)
    assert (await ext(store, 4, 6)).coverage == span(2, 6)  # touches the right edge
    assert (await ext(store, 0, 2)).coverage == span(0, 6)  # touches the left edge
    assert (await ext(store, 1, 3)).coverage == span(0, 6)  # contained: a no-op
    got = await store.read(KEY, T0, T0 + 6 * H)
    assert [c.open_ms for c in got.bars] == [T0 + i * H for i in range(6)]


async def test_disjoint_span_raises_and_changes_nothing() -> None:
    store = InMemoryOhlcvStore()
    await ext(store, 0, 2)
    with pytest.raises(OhlcvStoreError) as err:
        await ext(store, 3, 5)
    assert err.value.code == "coverage_disjoint"
    got = await store.read(KEY, T0, T0 + 9 * H)
    assert got.coverage == span(0, 2) and len(got.bars) == 2
    assert (await store.read_bars(KEY, T0, T0 + 9 * H)) == got.bars


async def test_first_write_wins_and_divergence_is_counted() -> None:
    store = InMemoryOhlcvStore()
    await ext(store, 0, 3)
    again = await ext(store, 0, 3, [bar(0), bar(1, "106"), bar(2)])
    assert (again.inserted, again.conflicts, again.divergent) == (0, 3, 1)
    stored = await store.read_bars(KEY, T0, T0 + 3 * H)
    assert [c.close for c in stored] == [Decimal(105)] * 3  # the stored value wins
    same = await ext(store, 0, 3)
    assert (same.inserted, same.conflicts, same.divergent) == (0, 3, 0)


async def test_in_batch_duplicates_are_not_divergence_and_input_order_is_free() -> None:
    store = InMemoryOhlcvStore()
    res = await ext(store, 0, 2, [bar(1), bar(0), bar(0, "106")])
    assert (res.inserted, res.conflicts, res.divergent) == (2, 0, 0)
    assert (await store.read_bars(KEY, T0, T0 + 2 * H))[0].close == Decimal(105)


async def test_series_keys_are_case_sensitive_and_separate() -> None:
    store, other = InMemoryOhlcvStore(), SeriesKey("coinbase", "btc/eur", 3600)
    await ext(store, 0, 2)
    assert (await store.read(other, T0, T0 + 2 * H)) == StoredRead(None, ())


async def test_read_bars_ignores_coverage() -> None:
    store = InMemoryOhlcvStore()
    await ext(store, 0, 4)
    assert len(await store.read_bars(KEY, T0 - 5 * H, T0 + 2 * H)) == 2


async def test_fail_with_hook_raises_on_every_call() -> None:
    store = InMemoryOhlcvStore()
    store.fail_with = OhlcvStoreError("unavailable")
    for call in (
        store.read(KEY, T0, T0 + H),
        store.read_bars(KEY, T0, T0 + H),
        ext(store, 0, 1),
    ):
        with pytest.raises(OhlcvStoreError):
            await call


# ---------------------------------------------------------------------------
# validate-before-write (CF-26-40): both stores, no I/O
# ---------------------------------------------------------------------------
class _NoIo:
    calls = 0

    def __call__(self) -> Any:
        _NoIo.calls += 1
        raise AssertionError("the database was touched")


def _bad(**kw: Any) -> Candle:
    base = dict(zip(("open_", "high", "low", "close", "volume"), _GOOD, strict=True))
    return Candle(kw.pop("open_ms", T0), **{**base, **kw})


_INVALID = [
    ("span_off_grid", Coverage(T0 + 1, T0 + H), [], "off_grid"),
    ("span_end_off_grid", Coverage(T0, T0 + H + 1), [], "off_grid"),
    ("past_settle", Coverage(T0, NOW), [], "forming_or_future"),
    ("bar_before_span", span(2, 4), [bar(1)], "out_of_range"),
    ("bar_at_span_end", span(2, 4), [bar(4)], "out_of_range"),
    ("loose_exponent", span(0, 2), [_bad(open_=Decimal("100.0"))], "not_canonical"),
    ("inverted_ohlc", span(0, 2), [_bad(high=Decimal(80))], "bad_ohlc"),
    ("negative_volume", span(0, 2), [_bad(volume=Decimal(-1))], "bad_volume"),
    ("off_grid_bar", span(0, 2), [_bad(open_ms=T0 + 1)], "off_grid"),
    ("mixed_types", span(0, 2), [_bad(open_ms="x")], "bad_type"),
    (
        "too_many",
        span(0, MAX_EXTEND_BARS + 1),
        [_bad(open_ms=T0 + i * H) for i in range(MAX_EXTEND_BARS + 1)],
        "out_of_range",
    ),
]


@pytest.mark.parametrize("sql", [False, True])
@pytest.mark.parametrize(("name", "cover", "items", "code"), _INVALID, ids=[c[0] for c in _INVALID])
async def test_invalid_extend_is_refused_before_any_io(
    sql: bool, name: str, cover: Coverage, items: list[Candle], code: str
) -> None:
    store: Any = SqlOhlcvStore(cast(Any, _NoIo())) if sql else InMemoryOhlcvStore()
    with pytest.raises(OhlcvDataError) as err:
        await store.extend(KEY, cover, items, fetched_at_ms=NOW)
    assert err.value.code == code
    if not sql:
        assert await store.read(KEY, T0, NOW) == StoredRead(None, ())


async def test_a_chunk_of_exactly_the_maximum_is_accepted() -> None:
    items = [bar(i) for i in range(MAX_EXTEND_BARS)]
    res = await InMemoryOhlcvStore().extend(KEY, span(0, MAX_EXTEND_BARS), items, fetched_at_ms=NOW)
    assert res.inserted == MAX_EXTEND_BARS


async def test_the_settle_boundary_itself_is_accepted() -> None:
    res = await InMemoryOhlcvStore().extend(KEY, Coverage(T0, NOW - H), [bar(0)], fetched_at_ms=NOW)
    assert res.coverage == Coverage(T0, NOW - H)


# ---------------------------------------------------------------------------
# SqlOhlcvStore through a scripted fake session
# ---------------------------------------------------------------------------
def _pg(state: str, text: str = "x") -> Exception:
    err = Exception(text)
    err.sqlstate = state  # type: ignore[attr-defined]
    return err


class _Result:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None, rowcount: int = 0) -> None:
        self._rows, self.rowcount = rows or [], rowcount

    def first(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def scalar_one(self) -> Any:
        return self._rows[0][0]


def _dt(ms: int) -> datetime:
    return store_mod._dt(ms)


_KNOWN_SQL: frozenset[str] = frozenset(
    {v.text for v in vars(store_mod).values() if isinstance(v, TextClause)}
    | {"SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY", "SET TRANSACTION READ ONLY"}
)


class _Db:
    """The fake server: scripted answers, a call log, a rollback flag."""

    def __init__(self, **kw: Any) -> None:
        self.cover: tuple[int, int] | None = kw.get("cover")
        self.rows: list[tuple[Any, ...]] = kw.get("rows", [])
        self.inserted: int | None = kw.get("inserted")
        self.divergent: int = kw.get("divergent", 0)
        self.disjoint: bool = kw.get("disjoint", False)
        self.error: Exception | None = kw.get("error")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rolled_back: bool | None = None

    def factory(self) -> Any:
        return self

    async def __aenter__(self) -> _Db:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def begin(self) -> Any:
        db = self

        class _Tx:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, exc_type: object, *_: object) -> None:
                db.rolled_back = exc_type is not None

        return _Tx()

    def store(self) -> SqlOhlcvStore:
        return SqlOhlcvStore(cast(Any, self.factory))

    async def execute(self, stmt: TextClause, params: dict[str, Any] | None = None) -> _Result:
        params = params or {}
        assert set(stmt.compile().params) == set(params), "every bind is given, none extra"
        assert stmt.text in _KNOWN_SQL, "only the module's literal statements are executed"
        sql = " ".join(str(stmt).split())
        self.calls.append((sql, params))
        if self.error is not None:
            raise self.error
        if sql.startswith("SELECT covered_from"):
            return _Result([(_dt(self.cover[0]), _dt(self.cover[1]))] if self.cover else [])
        if sql.startswith("INSERT INTO ohlcv_bars"):
            n = len(params["o"]) if self.inserted is None else self.inserted
            return _Result(rowcount=n)
        if sql.startswith("UPDATE ohlcv_coverage"):
            lo, hi = params["a"], params["b"]  # LEAST / GREATEST against the scripted coverage
            if self.cover:
                lo, hi = min(lo, _dt(self.cover[0])), max(hi, _dt(self.cover[1]))
            return _Result([] if self.disjoint else [(lo, hi)])
        if sql.startswith("SELECT count(*)"):
            return _Result([(self.divergent,)])
        if sql.startswith("SELECT open_time"):
            hit = [r for r in self.rows if params["lo"] <= r[0] < params["hi"]]
            return _Result(hit[: params["page"]])
        return _Result()

    def sqls(self) -> list[str]:
        return [s for s, _ in self.calls]


def _row(i: int, **kw: str) -> tuple[Any, ...]:
    vals = {"o": "100", "h": "110", "l": "90", "c": "105", "v": "1"} | kw
    return (_dt(T0 + i * H), *(Decimal(vals[k]).quantize(Decimal("1E-12")) for k in "ohlcv"))


async def test_extend_statement_order_and_parameters() -> None:
    db = _Db(cover=None)
    res = await db.store().extend(KEY, span(0, 3), [bar(2), bar(0), bar(1)], fetched_at_ms=NOW)
    assert res == ExtendResult(3, 0, 0, span(0, 3))
    order = [
        "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
        "SET LOCAL lock_timeout = '5s'",
        "SET LOCAL statement_timeout = '15s'",
        "INSERT INTO ohlcv_coverage",
        "SELECT covered_from",
        "INSERT INTO ohlcv_bars",
        "UPDATE ohlcv_coverage",
    ]
    assert len(db.sqls()) == 7
    assert all(s.startswith(p) for s, p in zip(db.sqls(), order, strict=True))
    assert db.sqls()[4].endswith("FOR UPDATE") and db.rolled_back is False
    anchor, insert, update = db.calls[3][1], db.calls[5][1], db.calls[6][1]
    assert anchor["anchor"] == _dt(T0) and anchor["timeframe"] == "1h"
    assert insert["open_time"] == [_dt(T0 + i * H) for i in range(3)]  # ascending
    assert insert["fetched_at"] == _dt(NOW) and insert["ingest_version"] == 1
    assert insert["o"] == [Decimal(100)] * 3
    assert (insert["h"], insert["l"]) == ([Decimal(110)] * 3, [Decimal(90)] * 3)
    assert (insert["c"], insert["v"]) == ([Decimal(105)] * 3, [Decimal(1)] * 3)
    assert (update["a"], update["b"], update["verified_at"]) == (_dt(T0), _dt(T0 + 3 * H), _dt(NOW))


async def test_conflicts_trigger_the_divergence_query_log_and_metric() -> None:
    metrics.reset()
    db = _Db(inserted=1, divergent=2)
    with capture_logs() as logs:
        res = await db.store().extend(
            KEY, span(0, 3), [bar(i) for i in range(3)], fetched_at_ms=NOW
        )
    assert (res.inserted, res.conflicts, res.divergent) == (1, 2, 2)
    assert db.sqls()[-2].startswith("UPDATE") and db.sqls()[-1].startswith("SELECT count(*)")
    assert logs == [
        {
            "event": "ohlcv_cache_divergent_bars",
            "log_level": "warning",
            "exchange": "coinbase",
            "symbol": "BTC/EUR",
            "timeframe_s": 3600,
            "divergent": 2,
            "conflicts": 2,
        }
    ]
    assert metrics.get_all()["counters"][DIVERGENT_METRIC] == 2
    assert DIVERGENT_METRIC == "ohlcv_cache_divergent_bars_total"  # SY-26-09


async def test_no_conflicts_means_no_divergence_query() -> None:
    db = _Db()
    await db.store().extend(KEY, span(0, 2), [bar(0), bar(1)], fetched_at_ms=NOW)
    assert not any(s.startswith("SELECT count(*)") for s in db.sqls())


async def test_zero_updated_rows_is_coverage_disjoint_and_rolls_back() -> None:
    db = _Db(disjoint=True)
    with pytest.raises(OhlcvStoreError) as err:
        await db.store().extend(KEY, span(0, 2), [bar(0)], fetched_at_ms=NOW)
    assert err.value.code == "coverage_disjoint" and db.rolled_back is True


async def test_extend_returns_the_merged_coverage_of_the_update() -> None:
    db = _Db(cover=(T0 + 2 * H, T0 + 4 * H))
    left = await db.store().extend(KEY, span(0, 2), [bar(0), bar(1)], fetched_at_ms=NOW)
    assert left.coverage == span(0, 4)
    right = await db.store().extend(KEY, span(4, 6), [bar(4), bar(5)], fetched_at_ms=NOW)
    assert right.coverage == span(2, 6)


async def test_read_is_one_repeatable_read_snapshot_clipped_to_coverage() -> None:
    db = _Db(cover=(T0 + 2 * H, T0 + 6 * H), rows=[_row(i) for i in range(8)])
    got = await db.store().read(KEY, T0, T0 + 8 * H)  # the window spans both coverage edges
    assert db.sqls()[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert db.sqls()[1] == "SET LOCAL statement_timeout = '30s'"
    assert got.coverage == span(2, 6)
    assert [c.open_ms for c in got.bars] == [T0 + i * H for i in range(2, 6)]
    assert got.bars[0].open_.as_tuple() == Decimal(100).as_tuple()  # canonical, not scale 12
    assert (await _Db().store().read(KEY, T0, T0 + H)) == StoredRead(None, ())


@pytest.mark.parametrize("count", [4, 5])
async def test_keyset_pages_equal_one_query(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(store_mod, "_PAGE", 2)
    db = _Db(cover=(T0, T0 + 9 * H), rows=[_row(i) for i in range(count)])
    got = await db.store().read(KEY, T0, T0 + 9 * H)
    assert [c.open_ms for c in got.bars] == [T0 + i * H for i in range(count)]
    los = [p["lo"] for s, p in db.calls if s.startswith("SELECT open_time")]
    assert los == [_dt(T0), _dt(T0 + 2 * H), _dt(T0 + 4 * H)]  # lo = last open + one bar


async def test_read_bars_ignores_coverage_and_is_read_only() -> None:
    db = _Db(rows=[_row(i) for i in range(4)])
    got = await db.store().read_bars(KEY, T0 + H, T0 + 3 * H)
    assert [c.open_ms for c in got] == [T0 + H, T0 + 2 * H]
    assert db.sqls()[:2] == ["SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = '30s'"]


async def test_stored_row_that_is_not_canonical_data_is_stored_row_invalid() -> None:
    db = _Db(cover=(T0, T0 + 2 * H), rows=[_row(0, l="0")])
    with pytest.raises(OhlcvStoreError) as err:
        await db.store().read(KEY, T0, T0 + 2 * H)
    assert err.value.code == "stored_row_invalid"


async def test_hostile_symbol_only_travels_as_a_bound_parameter() -> None:
    hostile = SeriesKey("coinbase", "X';DROP/EUR", 3600)
    db = _Db(cover=(T0, T0 + 2 * H), inserted=0)  # all conflicts: the divergence query runs
    await db.store().read(hostile, T0, T0 + 2 * H)
    await db.store().read_bars(hostile, T0, T0 + 2 * H)
    await db.store().extend(
        hostile, span(0, 2), [bar(0)], fetched_at_ms=NOW
    )  # symbol is not in bar
    assert any(s.startswith("SELECT count(*)") for s in db.sqls())
    assert all("DROP" not in s and "X'" not in s for s in db.sqls())
    assert all(p.get("symbol") == "X';DROP/EUR" for _, p in db.calls if "symbol" in p)


# ---------------------------------------------------------------------------
# error mapping and log hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("state", "code"),
    [
        ("42P01", "missing_table"),
        ("42501", "denied"),
        ("57014", "timeout"),
        ("55P03", "timeout"),
        ("08006", "unavailable"),
        ("57P01", "unavailable"),
        ("23514", "error"),
        ("22003", "error"),
        ("22001", "error"),
        ("XX000", "error"),
    ],
)
def test_sqlstate_maps_to_a_closed_code(state: str, code: str) -> None:
    assert store_error_code(DBAPIError("s", {}, _pg(state))) == code
    assert store_error_code(_pg(state)) == code


def test_type_based_codes() -> None:
    assert store_error_code(TimeoutError()) == "timeout"
    assert store_error_code(ConnectionRefusedError()) == "unavailable"
    assert store_error_code(RuntimeError("42P01")) == "error"  # text never decides


def test_invalidated_connection_and_chained_sqlstate() -> None:
    dropped = DBAPIError("s", {}, Exception(), connection_invalidated=True)
    assert store_error_code(dropped) == "unavailable"
    chained = RuntimeError()
    chained.__cause__ = _pg("42501")
    assert store_error_code(chained) == "denied"


@pytest.mark.parametrize("op", ["read", "read_bars", "extend"])
async def test_failure_logs_and_raises_no_text(op: str) -> None:
    secret = DBAPIError(
        "INSERT INTO t VALUES (:p)", {"p": "SECRET-PARAM"}, _pg("42501", "SECRET-MSG")
    )
    db = _Db(error=secret)
    store = db.store()
    calls = {
        "read": lambda: store.read(KEY, T0, T0 + H),
        "read_bars": lambda: store.read_bars(KEY, T0, T0 + H),
        "extend": lambda: store.extend(KEY, span(0, 1), [bar(0)], fetched_at_ms=NOW),
    }
    with capture_logs() as logs, pytest.raises(OhlcvStoreError) as err:
        await calls[op]()
    assert (err.value.code, str(err.value)) == ("denied", "ohlcv store failed (denied)")
    assert err.value.__suppress_context__ is True and err.value.__cause__ is None
    assert "SECRET" not in repr(logs) and "INSERT" not in repr(logs)
    assert logs[0]["event"] == "ohlcv_store_failed" and logs[0]["op"] == op
    assert set(logs[0]) <= {"event", "log_level", "op", "code", "error_type", "error_at"}


# ---------------------------------------------------------------------------
# SQL text pins
# ---------------------------------------------------------------------------
def _statements() -> dict[str, str]:
    found = {
        k: " ".join(v.text.split()) for k, v in vars(store_mod).items() if isinstance(v, TextClause)
    }
    return {k: v for k, v in found.items() if not v.startswith("SET ")}


def test_no_update_or_delete_on_the_bars_table() -> None:
    sql = _statements()
    assert len(sql) == 7
    for name, text in sql.items():
        assert not re.search(r"\b(DELETE|TRUNCATE|DROP|ALTER|DO UPDATE)\b", text, re.I), name
        assert not re.search(r"UPDATE\s+ohlcv_bars", text, re.I), name
    source = Path(store_mod.__file__).read_text()
    assert not re.search(r"UPDATE\s+ohlcv_bars|DELETE\s+FROM", source, re.I)


def test_coverage_update_text_is_pinned() -> None:
    assert _statements()["_UPDATE_SQL"] == (
        "UPDATE ohlcv_coverage SET covered_from = LEAST(covered_from, :a), "
        "covered_to = GREATEST(covered_to, :b), verified_at = :verified_at "
        "WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe "
        "AND :a <= covered_to AND :b >= covered_from RETURNING covered_from, covered_to"
    )


def test_conflict_targets_are_exactly_the_primary_keys() -> None:
    sql = _statements()
    for orm, name in ((OhlcvBarORM, "_INSERT_SQL"), (OhlcvCoverageORM, "_ANCHOR_SQL")):
        cols = ", ".join(c.name for c in orm.__table__.primary_key.columns)
        assert f"ON CONFLICT ({cols}) DO NOTHING" in sql[name]


def test_insert_is_ordered_and_divergence_compares_every_value_column() -> None:
    sql = _statements()
    assert "ORDER BY t.open_time ON CONFLICT" in sql["_INSERT_SQL"]  # DB-26-18 step 4
    for col in ("open", "high", "low", "close", "volume"):
        assert f"b.{col} IS DISTINCT FROM t." in sql["_DIVERGENCE_SQL"], col


def test_range_query_has_the_five_predicates_and_an_order() -> None:
    text = _statements()["_BARS_SQL"]
    for part in ("exchange = :exchange", "symbol = :symbol", "timeframe = :timeframe"):
        assert part in text
    assert "open_time >= :lo AND open_time < :hi ORDER BY open_time LIMIT :page" in text


def test_module_builds_no_sql_strings() -> None:
    tree = ast.parse(Path(store_mod.__file__).read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "text":
            assert isinstance(node.args[0], ast.Constant), "text() takes a literal only"


def test_asyncpg_renders_typed_array_parameters() -> None:
    sql = str(store_mod._INSERT_SQL.compile(dialect=pg_asyncpg.dialect()))
    assert sql.count("::NUMERIC[]") == 5 and sql.count("::TIMESTAMP WITH TIME ZONE[]") == 1


def test_timeframe_labels_equal_the_enum_minus_1w() -> None:
    expected = {s: tf.value for tf, s in _TIMEFRAME_SECONDS.items() if tf is not TimeFrame.ONE_WEEK}
    assert dict(store_mod._TF_LABEL) == expected


def test_millisecond_conversion_is_exact_and_utc() -> None:
    for ms in (0, 1, -1, T0 + 123, 253_402_300_799_999):
        assert store_mod._ms(store_mod._dt(ms)) == ms
    assert store_mod._dt(T0) == datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# layering: leaf-side module imports, no callers yet
# ---------------------------------------------------------------------------
def test_cache_module_imports_only_the_leaf_and_the_stdlib() -> None:
    tree = ast.parse((_REPO / "packages/data/ohlcv_cache.py").read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "typing",
        "data.ohlcv_integrity",
    }


def test_nothing_calls_the_store_yet() -> None:
    users = set()
    for root in ("apps", "packages"):
        for path in (_REPO / root).rglob("*.py"):
            if re.search(r"^\s*(from|import)\s.*ohlcv_(store|cache)", path.read_text(), re.M):
                users.add(path.relative_to(_REPO).as_posix())
    assert users == {"apps/api/services/ohlcv_store.py"}  # the adapter imports the contract
