"""WP2.4 P5: the point-in-time context store (T-36, T-37).

* T-36: writer idempotency (same batch twice, a changed value, backfill after recording), the
  SQL shape of the writer, and the pin that NO application code issues UPDATE or DELETE
  against ``context_observations``.
* T-37: ``load_window`` equals the pure reference, the digest is preserved, rows with
  ``available_at > end`` are never loaded, and a store exception becomes an unavailable
  window with a closed reason, never an exception into the bar loop.
* Also: row validation equals the JSONL lag floor (CF-SYN-01), 8-decimal HALF_UP
  quantisation (QR-11), and a clock-free / trading-free import pin on the module.

No database is used here (an optional Postgres test lives in tests/integration).
"""

from __future__ import annotations

import ast
import asyncio
import json
import random
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from api.services import context_store as cs
from api.services.context_store import (
    ContextRow,
    ContextWindow,
    InMemoryContextStore,
    SqlContextStore,
    load_window_safe,
    quantize_store_value,
    series_from_rows,
)
from data.point_in_time import (
    SERIES_RULES,
    PitError,
    available_at_estimated,
    available_at_vintage,
    parse_jsonl_lines,
    series_digest,
    to_ms,
)

_REPO = Path(__file__).resolve().parents[2]
_STORE_PATH = _REPO / "apps" / "api" / "services" / "context_store.py"
_DOM = "coingecko:btc_dominance"
_WHALE = "whale:net_flow_1h"
_FGI = "alternative_me:fgi"
_T10 = "fred:T10Y2Y"
_BASE = datetime(2024, 3, 1, tzinfo=UTC)
_HOUR = timedelta(hours=1)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _rec(
    sid: str, obs: datetime, avail: datetime, value: str, basis: str = "recorded"
) -> ContextRow:
    return ContextRow(sid, obs, avail, Decimal(value), basis)


# ---------------------------------------------------------------------------
# quantisation (QR-11)
# ---------------------------------------------------------------------------


class TestQuantize:
    def test_half_up_not_half_even(self) -> None:
        assert quantize_store_value(Decimal("0.123456785")) == Decimal("0.12345679")
        assert quantize_store_value(Decimal("0.123456775")) == Decimal("0.12345678")
        assert quantize_store_value(Decimal("-0.123456785")) == Decimal("-0.12345679")
        assert quantize_store_value("0.000000005") == Decimal("0.00000001")

    def test_float_goes_through_repr_never_decimal_of_float(self) -> None:
        assert quantize_store_value(55.1) == Decimal("55.10000000")
        assert quantize_store_value(0.1 + 0.2) == Decimal("0.30000000")
        assert quantize_store_value(1_234_567.891234567) == Decimal("1234567.89123457")
        assert quantize_store_value(-12) == Decimal("-12.00000000")

    def test_ambient_context_does_not_matter(self) -> None:
        with localcontext() as ctx:
            ctx.prec = 3
            assert quantize_store_value(Decimal("123456.123456785")) == Decimal("123456.12345679")

    def test_negative_zero_is_zero(self) -> None:
        out = quantize_store_value("-0.000000001")
        assert out == 0 and not out.is_signed()

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), float("inf"), -float("inf"), True, "NaN", "abc", "1e16", 1e16, None, [1]],
    )
    def test_rejects_non_finite_huge_and_junk(self, bad: Any) -> None:
        with pytest.raises(PitError) as info:
            quantize_store_value(bad)
        assert info.value.code == "pit_bad_value"

    def test_value_just_below_the_limit_is_kept(self) -> None:
        assert quantize_store_value("9999999999999999.99999999") == Decimal(
            "9999999999999999.99999999"
        )

    def test_rounding_up_to_the_limit_is_rejected(self) -> None:
        with pytest.raises(PitError):
            quantize_store_value("9999999999999999.999999999")

    def test_result_is_accepted_by_a_row(self) -> None:
        row = _rec(_DOM, _BASE, _BASE + _HOUR, "0")
        assert row.value == 0
        ContextRow(_DOM, _BASE, _BASE, quantize_store_value(55.123456789), "recorded")


# ---------------------------------------------------------------------------
# row validation == the JSONL lag floor (CF-SYN-01)
# ---------------------------------------------------------------------------


def _jsonl_accepts(sid: str, obs: datetime, avail: datetime, value: str, basis: str) -> bool:
    line = json.dumps(
        {
            "series_id": sid,
            "observed_at": obs.isoformat(),
            "available_at": avail.isoformat(),
            "value": value,
            "basis": basis,
        }
    )
    try:
        parse_jsonl_lines([line])
    except PitError:
        return False
    return True


def _row_accepts(sid: str, obs: datetime, avail: datetime, value: str, basis: str) -> bool:
    try:
        ContextRow(sid, obs, avail, Decimal(value), basis)
    except PitError:
        return False
    return True


class TestRowValidation:
    def test_floor_differential_against_the_jsonl_parser(self) -> None:
        """For every series and basis, sweep stamps around the floor: row == JSONL verdict."""
        checked = 0
        for day in range(0, 14):  # covers weekends for the T10Y2Y business-day rule
            observed = _BASE + timedelta(days=day)
            observed_ms = to_ms(observed)
            floors = {
                "estimated_lag": {
                    _FGI: available_at_estimated(_FGI, observed_ms),
                    _T10: available_at_estimated(_T10, observed_ms),
                },
                "vintage": {_T10: available_at_vintage(observed_ms)},
                "recorded": {_DOM: observed_ms, _WHALE: observed_ms, _FGI: observed_ms},
            }
            for basis, per_series in floors.items():
                for sid, floor_ms in per_series.items():
                    for delta in (-86_400_000, -1, 0, 1, 3_600_000):
                        avail = datetime.fromtimestamp((floor_ms + delta) / 1000, tz=UTC)
                        value = "50" if sid == _FGI else "1.5"
                        assert _row_accepts(sid, observed, avail, value, basis) == _jsonl_accepts(
                            sid, observed, avail, value, basis
                        )
                        checked += 1
        assert checked > 300

    def test_exact_floor_accepted_one_ms_below_rejected(self) -> None:
        floor = datetime.fromtimestamp(available_at_estimated(_FGI, to_ms(_BASE)) / 1000, tz=UTC)
        assert _row_accepts(_FGI, _BASE, floor, "50", "estimated_lag")
        assert not _row_accepts(
            _FGI, _BASE, floor - timedelta(milliseconds=1), "50", "estimated_lag"
        )

    def test_basis_allowlist_per_series(self) -> None:
        late = _BASE + timedelta(days=30)
        assert not _row_accepts(_T10, _BASE, late, "0.5", "recorded")  # QR-15 mislabel
        assert not _row_accepts(_DOM, _BASE, late, "55", "estimated_lag")
        assert not _row_accepts(_WHALE, _BASE, late, "1", "vintage")
        assert _row_accepts(_DOM, _BASE, late, "55", "recorded")

    def test_unknown_series_and_junk_basis(self) -> None:
        with pytest.raises(PitError) as info:
            ContextRow("fred:FEDFUNDS", _BASE, _BASE, Decimal("5"), "estimated_lag")
        assert info.value.code == "pit_unknown_series"
        assert not _row_accepts(_DOM, _BASE, _BASE, "1", "bogus")
        assert not _row_accepts("", _BASE, _BASE, "1", "recorded")

    def test_naive_and_sub_millisecond_timestamps_rejected(self) -> None:
        with pytest.raises(PitError) as info:
            ContextRow(_DOM, datetime(2024, 3, 1), _BASE, Decimal(1), "recorded")
        assert info.value.code == "pit_naive_time"
        micro = _BASE + timedelta(microseconds=1)
        assert not _row_accepts(_DOM, _BASE, micro, "1", "recorded")
        assert _row_accepts(_DOM, _BASE, _BASE + timedelta(milliseconds=7), "1", "recorded")

    def test_available_before_observed_rejected(self) -> None:
        with pytest.raises(PitError) as info:
            _rec(_DOM, _BASE, _BASE - timedelta(milliseconds=1), "1")
        assert info.value.code == "pit_available_before_observed"

    @pytest.mark.parametrize(
        "value",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            "sNaN",
            "1E+16",
            "-1E+16",
            "0.000000001",
            "1E-999999",
            "1E+999999",
            "1E+999999999",  # CR-05: the 1 GB format() shape
            "1E-999999999",
        ],
    )
    def test_bad_values_rejected_without_huge_text(self, value: str) -> None:
        with pytest.raises(PitError):
            ContextRow(_DOM, _BASE, _BASE, Decimal(value), "recorded")

    @pytest.mark.parametrize("value", ["1E+999999", "1E-999999", "1E+999999999", "1E-999999999"])
    def test_huge_shapes_are_refused_before_any_text_is_built(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CR-05: the guard must fire BEFORE ``format``; the leaf's later rejection is too late."""

        def _boom(*_: Any, **__: Any) -> str:
            raise AssertionError("format() reached for a huge shape")

        monkeypatch.setattr(cs, "format", _boom, raising=False)  # shadows the builtin here
        with pytest.raises(PitError) as info:
            ContextRow(_DOM, _BASE, _BASE, Decimal(value), "recorded")
        assert info.value.code == "pit_bad_value"

    def test_extreme_datetimes_raise_piterror_not_overflowerror(self) -> None:
        """CR-04: ``datetime.min`` / ``max`` with an offset cannot be normalised to UTC."""
        from datetime import timezone

        low = datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5)))
        high = datetime(9999, 12, 31, 23, tzinfo=timezone(-timedelta(hours=5)))
        for stamp in (low, high):
            for observed, available in ((stamp, _BASE), (_BASE, stamp), (stamp, stamp)):
                with pytest.raises(PitError) as info:
                    ContextRow(_DOM, observed, available, Decimal(1), "recorded")
                assert info.value.code == "pit_bad_value"

    def test_fgi_must_be_integral(self) -> None:
        late = _BASE + timedelta(days=2)
        assert _row_accepts(_FGI, _BASE, late, "50", "estimated_lag")
        assert not _row_accepts(_FGI, _BASE, late, "50.5", "estimated_lag")

    def test_wrong_types_rejected(self) -> None:
        with pytest.raises(PitError):
            ContextRow(_DOM, _BASE, _BASE, 1.5, "recorded")  # type: ignore[arg-type]
        with pytest.raises(PitError):
            ContextRow(_DOM, "2024-03-01T00:00:00+00:00", _BASE, Decimal(1), "recorded")  # type: ignore[arg-type]
        with pytest.raises(PitError):
            ContextRow(None, _BASE, _BASE, Decimal(1), "recorded")  # type: ignore[arg-type]

    def test_error_text_carries_no_data(self) -> None:
        with pytest.raises(PitError) as info:
            _rec(_DOM, _BASE, _BASE - _HOUR, "123456.789")
        assert "123456" not in str(info.value) and _DOM not in str(info.value)

    def test_row_is_normalised_to_utc_and_source_is_derived(self) -> None:
        from datetime import timezone

        plus2 = timezone(timedelta(hours=2))
        row = ContextRow(
            _DOM, _BASE.astimezone(plus2), _BASE.astimezone(plus2), Decimal(1), "recorded"
        )
        assert row.observed_at.utcoffset() == timedelta(0)
        assert row.source == "coingecko"
        assert {sid: cs.source_for(sid) for sid in SERIES_RULES} == {
            _FGI: "alternative_me",
            _T10: "fred",
            _DOM: "coingecko",
            _WHALE: "whale_alert",
        }
        assert max(len(cs.source_for(s)) for s in SERIES_RULES) <= 32  # VARCHAR(32)
        assert max(len(s) for s in SERIES_RULES) <= 64  # VARCHAR(64)


# ---------------------------------------------------------------------------
# T-36 idempotency
# ---------------------------------------------------------------------------


class TestT36Idempotency:
    def test_same_batch_twice_adds_no_rows(self) -> None:
        store = InMemoryContextStore()
        batch = [
            _rec(_DOM, _BASE, _BASE + _HOUR, "55.1"),
            _rec(_DOM, _BASE + _HOUR, _BASE + 2 * _HOUR, "55.2"),
            _rec(_WHALE, _BASE, _BASE + _HOUR, "-1200000.5"),
        ]
        assert _run(store.write_observations(batch)) == 3
        assert _run(store.write_observations(batch)) == 0
        assert len(store) == 3

    def test_repolling_an_unchanged_value_later_inserts_nothing(self) -> None:
        store = InMemoryContextStore()
        _run(store.write_observations([_rec(_DOM, _BASE, _BASE + _HOUR, "55.1")]))
        later = _rec(_DOM, _BASE, _BASE + 3 * _HOUR, "55.10000000")
        assert _run(store.write_observations([later])) == 0
        assert [r.available_at for r in store.rows()] == [_BASE + _HOUR]  # first sighting wins

    def test_a_changed_value_adds_exactly_one_vintage(self) -> None:
        store = InMemoryContextStore()
        _run(store.write_observations([_rec(_DOM, _BASE, _BASE + _HOUR, "55.1")]))
        revised = _rec(_DOM, _BASE, _BASE + 2 * _HOUR, "55.4")
        assert _run(store.write_observations([revised])) == 1
        assert _run(store.write_observations([revised])) == 0
        assert len(store) == 2

    def test_backfill_after_recording_does_not_duplicate(self) -> None:
        store = InMemoryContextStore()
        floor = datetime.fromtimestamp(available_at_estimated(_FGI, to_ms(_BASE)) / 1000, tz=UTC)
        recorded = _rec(_FGI, _BASE, floor + 6 * _HOUR, "62", "recorded")
        estimated = _rec(_FGI, _BASE, floor, "62", "estimated_lag")
        assert _run(store.write_observations([recorded])) == 1
        assert _run(store.write_observations([estimated])) == 0
        assert len(store) == 1
        assert store.rows()[0].basis == "recorded"  # first written wins (U-02)

    def test_recording_after_backfill_does_not_duplicate_either(self) -> None:
        store = InMemoryContextStore()
        floor = datetime.fromtimestamp(available_at_estimated(_FGI, to_ms(_BASE)) / 1000, tz=UTC)
        assert (
            _run(store.write_observations([_rec(_FGI, _BASE, floor, "62", "estimated_lag")])) == 1
        )
        assert (
            _run(store.write_observations([_rec(_FGI, _BASE, floor + _HOUR, "62", "recorded")]))
            == 0
        )
        assert len(store) == 1

    def test_duplicates_inside_one_batch_collapse(self) -> None:
        store = InMemoryContextStore()
        batch = [
            _rec(_DOM, _BASE, _BASE + _HOUR, "55.1"),
            _rec(_DOM, _BASE, _BASE + 2 * _HOUR, "55.1"),
        ]
        assert _run(store.write_observations(batch)) == 1
        assert _run(store.write_observations(list(reversed(batch)))) == 0

    def test_same_primary_key_with_another_value_conflicts_and_is_skipped(self) -> None:
        store = InMemoryContextStore()
        _run(store.write_observations([_rec(_DOM, _BASE, _BASE + _HOUR, "55.1")]))
        clash = _rec(_DOM, _BASE, _BASE + _HOUR, "56.0")
        assert _run(store.write_observations([clash])) == 0
        assert store.rows()[0].value == Decimal("55.1")

    def test_series_are_independent(self) -> None:
        store = InMemoryContextStore()
        a = _rec(_DOM, _BASE, _BASE + _HOUR, "1")
        b = _rec(_WHALE, _BASE, _BASE + _HOUR, "1")
        assert _run(store.write_observations([a, b])) == 2

    def test_empty_batch(self) -> None:
        assert _run(InMemoryContextStore().write_observations([])) == 0


class _Result:
    def __init__(self, rowcount: int = 0, rows: Sequence[tuple[Any, ...]] = ()) -> None:
        self.rowcount = rowcount
        self._rows = list(rows)

    def all(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _Txn:
    def __init__(self, log: list[Any]) -> None:
        self._log = log

    async def __aenter__(self) -> _Txn:
        self._log.append("BEGIN")
        return self

    async def __aexit__(self, exc_type: Any, *_: Any) -> bool:
        self._log.append("ROLLBACK" if exc_type else "COMMIT")
        return False


class _Session:
    def __init__(self, log: list[Any], results: list[_Result], boom: BaseException | None) -> None:
        self._log, self._results, self._boom = log, results, boom

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    def begin(self) -> _Txn:
        return _Txn(self._log)

    async def execute(self, stmt: Any, params: Any = None) -> _Result:
        self._log.append((stmt, params))
        if self._boom is not None and not str(stmt).startswith("SET"):
            raise self._boom
        return self._results.pop(0) if self._results else _Result()


class _Factory:
    def __init__(self, results: Sequence[_Result] = (), boom: BaseException | None = None) -> None:
        self.log: list[Any] = []
        self._results = list(results)
        self._boom = boom
        self.sessions = 0

    def __call__(self) -> _Session:
        self.sessions += 1
        return _Session(self.log, self._results, self._boom)

    def statements(self) -> list[tuple[str, Any]]:
        return [(str(e[0]), e[1]) for e in self.log if isinstance(e, tuple)]


class TestT36SqlWriter:
    def test_statement_shape(self) -> None:
        sql = " ".join(str(cs._INSERT_SQL).split())
        assert sql.startswith("INSERT INTO context_observations")
        assert "WHERE NOT EXISTS" in sql
        assert "series_id = :series_id AND observed_at = :observed_at" in sql
        assert "value = CAST(:value AS NUMERIC(24, 8))" in sql
        assert "ON CONFLICT (series_id, observed_at, available_at) DO NOTHING" in sql
        assert not re.search(r"\b(UPDATE|DELETE|TRUNCATE|MERGE)\b", sql, re.IGNORECASE)

    def test_write_uses_one_transaction_bound_params_and_sums_rowcounts(self) -> None:
        factory = _Factory([_Result(), _Result(1), _Result(0), _Result(1)])
        store = SqlContextStore(factory)  # type: ignore[arg-type]
        rows = [
            _rec(_DOM, _BASE + _HOUR, _BASE + 2 * _HOUR, "55.2"),
            _rec(_DOM, _BASE, _BASE + _HOUR, "55.1"),
            _rec(_WHALE, _BASE, _BASE + _HOUR, "9.5"),
        ]
        assert _run(store.write_observations(rows)) == 2
        assert factory.sessions == 1
        assert factory.log[0] == "BEGIN" and factory.log[-1] == "COMMIT"
        stmts = factory.statements()
        assert stmts[0][0] == "SET LOCAL statement_timeout = '5s'"
        inserts = stmts[1:]
        assert len(inserts) == 3
        assert all(s is cs._INSERT_SQL or s == str(cs._INSERT_SQL) for s, _ in inserts)
        # deterministic order (series, observed_at, available_at) and bound parameters only
        assert [p["series_id"] for _, p in inserts] == [_DOM, _DOM, _WHALE]
        assert [p["observed_at"] for _, p in inserts][:2] == [_BASE, _BASE + _HOUR]
        for _, params in inserts:
            assert set(params) == {
                "series_id",
                "observed_at",
                "available_at",
                "value",
                "source",
                "basis",
            }
            assert isinstance(params["value"], Decimal)
        assert inserts[0][1]["source"] == "coingecko" and inserts[2][1]["source"] == "whale_alert"

    def test_empty_batch_opens_no_session(self) -> None:
        factory = _Factory()
        assert _run(SqlContextStore(factory).write_observations([])) == 0  # type: ignore[arg-type]
        assert factory.sessions == 0

    def test_failure_rolls_back_and_propagates_the_original_error(self) -> None:
        factory = _Factory(boom=ConnectionError("db down"))
        store = SqlContextStore(factory)  # type: ignore[arg-type]
        with pytest.raises(ConnectionError):
            _run(store.write_observations([_rec(_DOM, _BASE, _BASE, "1")]))
        assert factory.log[-1] == "ROLLBACK"


class TestT36NoUpdateOrDelete:
    """Application code never UPDATEs or DELETEs ``context_observations`` (append-only)."""

    _VERBS = re.compile(
        r"\b(UPDATE\s+\S+\s+SET|DELETE\s+FROM|TRUNCATE|MERGE\s+INTO|DO\s+UPDATE|DROP\s+TABLE|ALTER\s+TABLE)\b",
        re.IGNORECASE,
    )
    _ORM_MUTATORS = frozenset({"update", "delete", "merge", "bulk_update_mappings"})

    @staticmethod
    def _docstring_nodes(tree: ast.AST) -> set[int]:
        skip: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    skip.add(id(body[0].value))
        return skip

    @classmethod
    def problems(cls, src: str) -> list[str]:
        """Violations in one source file that mentions the table or its ORM class."""
        tree = ast.parse(src)
        mentions = "context_observations" in src or "ContextObservationORM" in src
        if not mentions:
            return []
        out: list[str] = []
        docs = cls._docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
            ):
                if cls._VERBS.search(node.value):
                    out.append(f"sql verb in string at line {node.lineno}")
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in cls._ORM_MUTATORS:
                    out.append(f"mutator call {name} at line {node.lineno}")
        return out

    @staticmethod
    def _tree_files() -> list[Path]:
        files: list[Path] = []
        for root in ("apps", "packages", "scripts"):
            for path in sorted((_REPO / root).rglob("*.py")):
                parts = path.relative_to(_REPO).parts
                if "tests" in parts or "__pycache__" in parts or path.name.startswith("test_"):
                    continue
                files.append(path)
        return files

    def test_no_application_file_mutates_the_table(self) -> None:
        offenders: dict[str, list[str]] = {}
        referencing = 0
        for path in self._tree_files():
            src = path.read_text()
            found = self.problems(src)
            if "context_observations" in src or "ContextObservationORM" in src:
                referencing += 1
            if found:
                offenders[path.relative_to(_REPO).as_posix()] = found
        assert referencing >= 2  # models.py and the store: the scan does see the table
        # models.py holds other tables whose DDL words are not mutations of this one; it must
        # still be free of mutator calls on the ORM class.
        assert offenders == {}, offenders

    def test_the_scan_catches_mutations(self) -> None:
        for bad in (
            "S = 'UPDATE context_observations SET value = 1'\n",
            "S = 'DELETE FROM context_observations'\n",
            "S = 'TRUNCATE context_observations'\n",
            "S = 'INSERT INTO context_observations VALUES (1) ON CONFLICT DO UPDATE SET a = 1'\n",
            "from api.db.models import ContextObservationORM\n"
            "session.delete(ContextObservationORM)\n",
            "from api.db.models import ContextObservationORM as C\nq = update(C)\n",
            "x = 'context_observations'\nsession.merge(x)\n",
        ):
            assert self.problems(bad), bad

    def test_the_scan_ignores_docstrings_and_unrelated_files(self) -> None:
        assert (
            self.problems('"""no UPDATE foo SET and no DELETE FROM context_observations."""\n')
            == []
        )
        assert self.problems("S = 'DELETE FROM runs'\nsession.delete(x)\n") == []  # no mention

    def test_the_store_module_has_no_mutating_sql_or_calls(self) -> None:
        assert self.problems(_STORE_PATH.read_text()) == []
        sql = " ".join(str(cs._LOAD_SQL).split()) + " " + " ".join(str(cs._INSERT_SQL).split())
        assert not self._VERBS.search(sql)


# ---------------------------------------------------------------------------
# T-37 load_window
# ---------------------------------------------------------------------------


def _ms(dt: datetime) -> int:
    return to_ms(dt)


def _random_rows(rng: random.Random) -> list[ContextRow]:
    rows: list[ContextRow] = []
    for sid, step_h in ((_DOM, 1), (_WHALE, 2)):
        for h in range(rng.randint(3, 45)):
            observed = (
                _BASE + timedelta(hours=h * step_h) + timedelta(milliseconds=rng.randint(0, 999))
            )
            for _ in range(rng.choice([1, 1, 1, 2, 3])):
                lag = timedelta(milliseconds=rng.randint(0, 5 * 3_600_000))
                value = (Decimal(rng.randint(-10_000, 10_000)) / 100).quantize(
                    Decimal("0.00000001")
                )
                rows.append(ContextRow(sid, observed, observed + lag, value, "recorded"))
    for d in range(rng.randint(2, 9)):  # a daily series with a real lag floor
        observed = _BASE + timedelta(days=d)
        floor = datetime.fromtimestamp(available_at_estimated(_FGI, _ms(observed)) / 1000, tz=UTC)
        extra = timedelta(hours=rng.choice([0, 0, 5, 30]))
        rows.append(
            ContextRow(_FGI, observed, floor + extra, Decimal(rng.randint(0, 100)), "estimated_lag")
        )
    return rows


def _asof_all(ref: dict[str, Any], sid: str, t_ms: int) -> Decimal | None:
    series = ref.get(sid)
    return None if series is None else series.asof(t_ms, max_age_ms=SERIES_RULES[sid].max_age_ms)


class TestT37LoadWindow:
    def test_equals_the_pure_reference_on_random_data(self) -> None:
        rng = random.Random(20240301)  # noqa: S311
        compared = 0
        for _ in range(120):
            store = InMemoryContextStore()
            _run(store.write_observations(_random_rows(rng)))
            reference = series_from_rows(store.rows())  # ALL rows, even those after `end`
            start = _BASE + timedelta(milliseconds=rng.randint(-7_200_000, 60 * 3_600_000))
            end = start + timedelta(milliseconds=rng.randint(0, 60 * 3_600_000))
            ids = rng.sample([_DOM, _WHALE, _FGI], k=rng.randint(1, 3))
            window = _run(store.load_window(ids, start, end))
            assert window.ok
            for sid in ids:
                for _ in range(25):
                    t = rng.randint(_ms(start), _ms(end))
                    got = (
                        window.series[sid].asof(t, max_age_ms=SERIES_RULES[sid].max_age_ms)
                        if sid in window.series
                        else None
                    )
                    assert got == _asof_all(reference, sid, t), (sid, t)
                    compared += 1
            assert set(window.series) <= set(ids)
        assert compared > 2000

    def test_digest_is_preserved_and_ignores_rows_available_after_end(self) -> None:
        rng = random.Random(7)  # noqa: S311
        store = InMemoryContextStore()
        _run(store.write_observations(_random_rows(rng)))
        start, end = _BASE + 3 * _HOUR, _BASE + 20 * _HOUR
        first = _run(store.load_window([_DOM, _WHALE], start, end))
        digest = series_digest(first.series.values())
        again = _run(store.load_window([_WHALE, _DOM], start, end))
        assert series_digest(again.series.values()) == digest
        # appends AFTER the window never change the digest
        future = [
            _rec(_DOM, end + _HOUR, end + 2 * _HOUR, "77"),
            _rec(
                _DOM, _BASE + 4 * _HOUR, end + _HOUR, "78"
            ),  # a late revision, published after end
            _rec(_WHALE, end, end + timedelta(milliseconds=1), "79"),
        ]
        assert _run(store.write_observations(future)) == 3
        after = _run(store.load_window([_DOM, _WHALE], start, end))
        assert series_digest(after.series.values()) == digest
        assert after.row_count == first.row_count

    def test_rows_available_after_end_are_never_loaded(self) -> None:
        store = InMemoryContextStore()
        end = _BASE + 10 * _HOUR
        _run(
            store.write_observations(
                [
                    _rec(_DOM, _BASE, _BASE + _HOUR, "1"),
                    _rec(_DOM, _BASE + 9 * _HOUR, end, "2"),  # available exactly at end: loaded
                    _rec(_DOM, _BASE + 9 * _HOUR, end + timedelta(milliseconds=1), "3"),
                    _rec(_DOM, _BASE + 12 * _HOUR, _BASE + 13 * _HOUR, "4"),
                ]
            )
        )
        window = _run(store.load_window([_DOM], _BASE, end))
        end_ms = _ms(end)
        assert all(o.available_ms <= end_ms for o in window.series[_DOM].observations)
        assert window.row_count == 2
        assert window.series[_DOM].asof(end_ms, max_age_ms=6 * 3_600_000) == Decimal("2")

    def test_seed_row_and_its_late_revision_are_kept(self) -> None:
        store = InMemoryContextStore()
        seed_obs = _BASE
        start, end = _BASE + 5 * _HOUR, _BASE + 8 * _HOUR
        _run(
            store.write_observations(
                [
                    _rec(_DOM, seed_obs - 2 * _HOUR, seed_obs - _HOUR, "10"),  # older than the seed
                    _rec(_DOM, seed_obs, seed_obs + _HOUR, "20"),  # the seed (visible at start)
                    _rec(
                        _DOM, seed_obs, start + _HOUR, "21"
                    ),  # revision published inside the window
                ]
            )
        )
        window = _run(store.load_window([_DOM], start, end))
        series = window.series[_DOM]
        values = sorted((o.available_ms, o.value) for o in series.observations)
        assert [v for _, v in values] == [Decimal("20"), Decimal("21")]  # the older row is dropped
        max_age = 24 * 3_600_000
        assert series.asof(_ms(start), max_age_ms=max_age) == Decimal("20")
        assert series.asof(_ms(start + _HOUR), max_age_ms=max_age) == Decimal("21")

    def test_observation_published_inside_the_window_about_an_older_period(self) -> None:
        store = InMemoryContextStore()
        start, end = _BASE + 5 * _HOUR, _BASE + 9 * _HOUR
        _run(store.write_observations([_rec(_DOM, _BASE, start + _HOUR, "33")]))  # observed < start
        window = _run(store.load_window([_DOM], start, end))
        assert window.series[_DOM].asof(
            _ms(start + 2 * _HOUR), max_age_ms=6 * 3_600_000
        ) == Decimal("33")

    def test_empty_series_are_absent_and_the_window_is_still_ok(self) -> None:
        window = _run(InMemoryContextStore().load_window([_DOM], _BASE, _BASE + _HOUR))
        assert window.ok and dict(window.series) == {} and window.row_count == 0
        assert window.ingest_cutoff is None

    def test_window_arguments_are_validated(self) -> None:
        store = InMemoryContextStore()
        with pytest.raises(PitError):
            _run(store.load_window([_DOM], _BASE + _HOUR, _BASE))  # start > end
        with pytest.raises(PitError):
            _run(store.load_window([], _BASE, _BASE))
        with pytest.raises(PitError):
            _run(store.load_window(["fred:FEDFUNDS"], _BASE, _BASE))
        with pytest.raises(PitError):
            _run(store.load_window([_DOM], datetime(2024, 3, 1), _BASE))  # naive

    def test_window_is_immutable(self) -> None:
        window = _run(InMemoryContextStore().load_window([_DOM], _BASE, _BASE))
        with pytest.raises(AttributeError):
            window.reason = "x"  # type: ignore[misc]
        with pytest.raises(TypeError):
            window.series["x"] = None  # type: ignore[index]


class TestT37FailSoft:
    _SENTINEL = "SENTINEL-postgresql://user:hunter2@db/prod?api_key=abc"

    def test_store_exception_becomes_unavailable_with_a_closed_reason(self) -> None:
        store = InMemoryContextStore()
        store.fail_with = ConnectionError(self._SENTINEL)
        with capture_logs() as logs:
            window = _run(load_window_safe(store, [_DOM], _BASE, _BASE + _HOUR))
        assert isinstance(window, ContextWindow)
        assert not window.ok
        assert window.reason == "store_unavailable" == cs.REASON_STORE_UNAVAILABLE
        assert dict(window.series) == {} and window.row_count == 0
        blob = json.dumps(logs, default=str) + repr(window)
        assert "hunter2" not in blob and "api_key" not in blob and "SENTINEL" not in blob
        assert logs and logs[0]["error_type"] == "ConnectionError"
        assert logs[0]["event"] == "context_store_load_failed"

    def test_invalid_arguments_and_invalid_stored_rows_are_soft_too(self) -> None:
        store = InMemoryContextStore()
        window = _run(load_window_safe(store, [_DOM], _BASE + _HOUR, _BASE))
        assert window.reason == "store_unavailable"

    def test_missing_table_error_does_not_escape(self) -> None:
        factory = _Factory(boom=RuntimeError('relation "context_observations" does not exist'))
        with capture_logs() as logs:
            window = _run(load_window_safe(SqlContextStore(factory), [_DOM], _BASE, _BASE + _HOUR))  # type: ignore[arg-type]
        assert window.reason == "store_unavailable"
        assert "relation" not in json.dumps(logs, default=str)
        assert factory.log[-1] == "ROLLBACK"

    def test_success_passes_through(self) -> None:
        store = InMemoryContextStore()
        _run(store.write_observations([_rec(_DOM, _BASE, _BASE + _HOUR, "5")]))
        window = _run(load_window_safe(store, [_DOM], _BASE, _BASE + 2 * _HOUR))
        assert window.ok and window.row_count == 1

    def test_cancellation_is_not_swallowed(self) -> None:
        store = InMemoryContextStore()
        store.fail_with = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            _run(load_window_safe(store, [_DOM], _BASE, _BASE + _HOUR))


class TestT37SqlLoader:
    def _factory(self, rows: list[tuple[Any, ...]]) -> _Factory:
        return _Factory([_Result(), _Result(), _Result(rows=rows)])

    def test_one_read_only_query_with_bound_parameters(self) -> None:
        rows = [
            (_DOM, _BASE, _BASE + _HOUR, Decimal("55.10000000"), "recorded", _BASE + 3 * _HOUR),
            (
                _DOM,
                _BASE + _HOUR,
                _BASE + 2 * _HOUR,
                Decimal("55.20000000"),
                "recorded",
                _BASE + 4 * _HOUR,
            ),
        ]
        factory = self._factory(rows)
        window = _run(
            SqlContextStore(factory).load_window([_WHALE, _DOM, _DOM], _BASE, _BASE + 5 * _HOUR)  # type: ignore[arg-type]
        )
        stmts = factory.statements()
        assert [s for s, _ in stmts[:2]] == [
            "SET TRANSACTION READ ONLY",
            "SET LOCAL statement_timeout = '15s'",
        ]
        assert len(stmts) == 3 and factory.sessions == 1  # ONE data query
        assert stmts[2][1] == {"ids": [_DOM, _WHALE], "start": _BASE, "end": _BASE + 5 * _HOUR}
        assert window.row_count == 2
        assert window.ingest_cutoff == _BASE + 4 * _HOUR
        reference = series_from_rows([ContextRow(r[0], r[1], r[2], r[3], r[4]) for r in rows])
        assert series_digest(window.series.values()) == series_digest(reference.values())

    def test_load_statement_filters_on_available_at_and_uses_the_seed(self) -> None:
        sql = " ".join(str(cs._LOAD_SQL).split())
        assert "c.available_at <= CAST(:end AS TIMESTAMPTZ)" in sql
        assert "c.observed_at >= s.floor_observed" in sql
        assert "max(c.observed_at)" in sql and "'-infinity'" in sql
        assert "ORDER BY c.series_id, c.available_at, c.observed_at" in sql
        assert "ingested_at" in sql.split("FROM context_observations c JOIN")[0]
        assert "ingested_at" not in sql.split("WHERE", 2)[-1]  # audit only, never the join

    def test_a_stored_row_below_the_floor_fails_the_load_closed(self) -> None:
        bad = (_FGI, _BASE, _BASE + _HOUR, Decimal("50"), "estimated_lag", _BASE)  # 1h < 24h floor
        factory = self._factory([bad])
        store = SqlContextStore(factory)  # type: ignore[arg-type]
        with pytest.raises(PitError) as info:
            _run(store.load_window([_FGI], _BASE, _BASE + timedelta(days=3)))
        assert info.value.code == "pit_available_before_observed"
        soft = _run(
            load_window_safe(
                SqlContextStore(self._factory([bad])),  # type: ignore[arg-type]
                [_FGI],
                _BASE,
                _BASE + timedelta(days=3),
            )
        )
        assert soft.reason == "store_unavailable"

    def test_a_malformed_row_shape_fails_closed(self) -> None:
        factory = self._factory([(_DOM, _BASE, _BASE, Decimal(1), "recorded")])
        with pytest.raises(PitError):
            _run(SqlContextStore(factory).load_window([_DOM], _BASE, _BASE + _HOUR))  # type: ignore[arg-type]

    def test_a_non_decimal_stored_value_fails_closed(self) -> None:
        factory = self._factory([(_DOM, _BASE, _BASE, 1.5, "recorded", _BASE)])
        with pytest.raises(PitError):
            _run(SqlContextStore(factory).load_window([_DOM], _BASE, _BASE + _HOUR))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SY-01-07 API compatibility (CF-24-01 recorder and boot seed)
# ---------------------------------------------------------------------------


class TestRecorderCompatibility:
    def test_recorder_rows_for_the_two_recorded_series(self) -> None:
        store = InMemoryContextStore()
        fetch = _BASE + 17 * timedelta(minutes=1) + timedelta(milliseconds=123)
        bucket = fetch.replace(minute=0, second=0, microsecond=0)
        rows = [
            ContextRow(_DOM, bucket, fetch, quantize_store_value(54.123456789), "recorded"),
            ContextRow(_WHALE, bucket, fetch, quantize_store_value(-2_500_000.5), "recorded"),
        ]
        assert _run(store.write_observations(rows)) == 2
        assert _run(store.write_observations(rows)) == 0

    def test_boot_seed_keeps_only_rows_available_by_now(self) -> None:
        store = InMemoryContextStore()
        now = _BASE + 30 * _HOUR
        _run(
            store.write_observations(
                [
                    _rec(_DOM, _BASE + h * _HOUR, _BASE + h * _HOUR + timedelta(seconds=5), str(h))
                    for h in range(40)
                ]
            )
        )
        seven_days_plus_age = timedelta(days=7) + timedelta(hours=6)
        window = _run(load_window_safe(store, [_DOM], now - seven_days_plus_age, now))
        assert window.ok
        last = max(o.available_ms for o in window.series[_DOM].observations)
        assert last <= _ms(now)
        assert window.series[_DOM].asof(_ms(now), max_age_ms=6 * 3_600_000) == Decimal("29")

    def test_the_protocol_is_satisfied_by_both_implementations(self) -> None:
        def accept(store: cs.ContextStore) -> cs.ContextStore:
            return store

        accept(InMemoryContextStore())
        accept(SqlContextStore(_Factory()))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module pins
# ---------------------------------------------------------------------------


class TestModulePins:
    def _tree(self) -> ast.Module:
        return ast.parse(_STORE_PATH.read_text())

    def test_no_wall_clock_read(self) -> None:
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"now", "utcnow", "today", "time", "monotonic", "sleep"}, (
                    node.lineno
                )

    def test_imports_nothing_from_trading_and_never_names_the_signal_module(self) -> None:
        tree = self._tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "trading"
                assert "mtf_signals" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "trading" for a in node.names)
        idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert idents.isdisjoint(
            {
                "LIVE_SIGNALS",
                "LiveSignals",
                "BacktestSignalSource",
                "mtf_signals",
                "importlib",
                "__import__",
            }
        )

    def test_logs_use_exc_log_fields_only(self) -> None:
        src = _STORE_PATH.read_text()
        assert "str(exc)" not in src and "repr(exc)" not in src
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"debug", "info", "warning", "error", "exception", "critical"}:
                    assert [k.arg for k in node.keywords] == [None], (
                        node.lineno
                    )  # **exc_log_fields only

    def test_the_store_does_not_use_the_sync_leaf_ingestion_functions_directly(self) -> None:
        names = {n.id for n in ast.walk(self._tree()) if isinstance(n, ast.Name)}
        assert "available_at_estimated" not in names and "available_at_vintage" not in names

    def test_every_sql_statement_is_a_module_constant_without_formatting(self) -> None:
        tree = self._tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text_parts = "".join(
                    v.value
                    for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                # f-strings only for the two constant timeouts, never for data
                assert text_parts.startswith("SET LOCAL statement_timeout = '"), node.lineno


# ---------------------------------------------------------------------------
# code-critic r1 additions (CR-02 part a, CR-03): each test kills a named mutant
# ---------------------------------------------------------------------------


class TestCriticR1Additions:
    def test_float_ties_use_the_repr_text(self) -> None:
        """Kills ``Decimal(raw)`` for floats (QR-11: never ``Decimal(float)``)."""
        assert quantize_store_value(1.000000005) == Decimal("1.00000001")
        assert quantize_store_value(2.000000025) == Decimal("2.00000003")

    def test_same_value_at_a_new_observed_at_is_a_new_row(self) -> None:
        """Kills a fake dedupe that ignores ``observed_at`` (FGI 50 on two days)."""
        s = InMemoryContextStore()
        rows = [
            _rec(_DOM, _BASE, _BASE + _HOUR, "55"),
            _rec(_DOM, _BASE + _HOUR, _BASE + 2 * _HOUR, "55"),
        ]
        assert _run(s.write_observations(rows)) == 2

    def test_unsorted_batch_first_sighting_wins(self) -> None:
        """Kills a fake write without the PK sort (it must match the SQL writer)."""
        s = InMemoryContextStore()
        late = _rec(_DOM, _BASE, _BASE + 3 * _HOUR, "55")
        early = _rec(_DOM, _BASE, _BASE + _HOUR, "55")
        assert _run(s.write_observations([late, early])) == 1
        assert [r.available_at for r in s.rows()] == [early.available_at]

    def test_row_available_exactly_at_start_is_the_seed(self) -> None:
        """Kills the seed boundary ``<`` for ``<=`` in the fake."""
        s = InMemoryContextStore()
        start = _BASE + 5 * _HOUR
        _run(
            s.write_observations(
                [
                    _rec(_DOM, _BASE, _BASE + _HOUR, "1"),
                    _rec(_DOM, _BASE + 2 * _HOUR, start, "2"),
                ]
            )
        )
        # seed = the row available exactly at ``start``; the older observation is dropped
        assert _run(s.load_window([_DOM], start, start + _HOUR)).row_count == 1

    def test_digest_equals_the_jsonl_digest_for_mixed_bases(self) -> None:
        """Kills ``series_from_rows`` dropping or hard-coding ``basis``.

        The digest of a store window must equal the digest of the same rows parsed from
        JSONL by the leaf (the store was only ever compared with itself before).
        """
        day = timedelta(days=1)
        spec: list[tuple[str, datetime, datetime, str, str]] = []
        for d in range(3):
            obs = _BASE + d * day
            fgi_avail = datetime.fromtimestamp(
                available_at_estimated(_FGI, to_ms(obs)) / 1000, tz=UTC
            )
            t10_avail = datetime.fromtimestamp(
                available_at_estimated(_T10, to_ms(obs)) / 1000, tz=UTC
            )
            spec += [
                (_FGI, obs, fgi_avail, str(40 + d), "estimated_lag"),
                (_FGI, obs, fgi_avail + 2 * day, str(41 + d), "recorded"),  # a recorded vintage
                (_T10, obs, obs + 2 * day, f"0.{d + 1}", "vintage"),  # >= vintage floor (+1d)
                (_T10, obs + day, t10_avail + day, f"1.{d}", "estimated_lag"),
                (_DOM, obs, obs + _HOUR, f"5{d}.5", "recorded"),
                (_WHALE, obs, obs + 2 * _HOUR, f"-{d}.25", "recorded"),
            ]
        lines = [
            json.dumps(
                {
                    "series_id": sid,
                    "observed_at": obs.isoformat(),
                    "available_at": avail.isoformat(),
                    "value": value,
                    "basis": basis,
                }
            )
            for sid, obs, avail, value, basis in spec
        ]
        expected = series_digest(parse_jsonl_lines(lines).values())
        store = InMemoryContextStore()
        rows = [_rec(*item) for item in spec]
        assert _run(store.write_observations(rows)) == len(spec)
        window = _run(
            store.load_window([_FGI, _T10, _DOM, _WHALE], _BASE - 10 * day, _BASE + 30 * day)
        )
        assert window.row_count == len(spec)
        assert series_digest(window.series.values()) == expected
        assert {o.basis for s in window.series.values() for o in s.observations} == {
            "estimated_lag",
            "recorded",
            "vintage",
        }

    def test_sql_pins(self) -> None:
        """CR-02 part a. Kills the SQL-only mutants ``c.available_at < :start`` for the seed
        and ``JOIN seed s ON TRUE`` (a cross join across series)."""
        sql = " ".join(str(cs._LOAD_SQL).split())
        assert "c.available_at <= CAST(:start AS TIMESTAMPTZ)" in sql
        assert "JOIN seed s ON s.series_id = c.series_id" in sql
        assert "ON TRUE" not in sql.upper()

    def test_insert_order_is_the_primary_key_order(self) -> None:
        """Kills ``pk()`` ordered ``(series, available, observed)``: the writer sorts by the
        PK, so concurrent writers take row locks in one global order."""
        factory = _Factory([_Result(), _Result(1), _Result(1)])
        rows = [
            _rec(_DOM, _BASE + _HOUR, _BASE + 2 * _HOUR, "55.2"),
            _rec(_DOM, _BASE, _BASE + 5 * _HOUR, "55.1"),
        ]
        assert _run(SqlContextStore(factory).write_observations(rows)) == 2  # type: ignore[arg-type]
        inserts = factory.statements()[1:]
        assert [p["observed_at"] for _, p in inserts] == [_BASE, _BASE + _HOUR]
        assert [p["available_at"] for _, p in inserts] == [_BASE + 5 * _HOUR, _BASE + 2 * _HOUR]
        assert [r.pk() for r in sorted(rows, key=ContextRow.pk)] == sorted(r.pk() for r in rows)

    def test_error_text_contract_is_documented_on_the_writer(self) -> None:
        """CR-06: the log-hygiene and ``hide_parameters`` contract stays on the class."""
        for doc in (cs.SqlContextStore.__doc__, cs.SqlContextStore.write_observations.__doc__):
            assert doc is not None
            assert "exc_log_fields" in doc and "hide_parameters=True" in doc
