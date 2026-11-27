"""
tests/unit/test_cf2401_recorder.py
CF-24-01 P7: T-36 recorder, T-37 missing table, T-38 boot seed, ER3-03 offers via the real poller,
CF-2401-46/99/109, DO-P6-01. No network, no database: ``InMemoryContextStore``.
"""

from __future__ import annotations

import ast
import asyncio
import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest
from asyncpg.exceptions import InsufficientPrivilegeError, UndefinedTableError
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_connection as Conn
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_dbapi
from sqlalchemy.exc import DBAPIError
from structlog.testing import capture_logs

import api.services.context_recorder as rec
from api.prometheus import TradingBotCollector
from api.services.context_poller import ContextPoller
from api.services.context_recorder import ContextRecorder
from api.services.context_store import ContextRow, ContextWindow, InMemoryContextStore
from common.metrics import metrics
from data.live_context import LIVE_BOOK
from data.point_in_time import SERIES_RULES, Observation, PitSeries
from tests.unit import test_cf2401_lifespan as ls
from tests.unit import test_cf2401_poller as tp
from tests.unit.test_cf2401_alerts import _expr, _rules

DOM, FGI, WHALE, START, H, dt = tp.DOM, tp.FGI, tp.WHALE, tp.START, tp.H, tp.dt
ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "apps/api/services/context_recorder.py").read_text()
NOW = START + 7 * 24 * H + 30 * 60_000
CAP = SERIES_RULES[DOM].max_age_ms


@pytest.fixture(autouse=True)
def _fresh_metrics() -> None:
    metrics.reset()


def obs(value: str = "52.5", observed: int = START, avail: int = START + 300_000) -> Observation:
    return Observation(observed, avail, Decimal(value), "recorded")


def count(name: str, series: str = DOM) -> int:
    return int(metrics.get_all()["counters"].get(f'{name}{{series="{series}"}}', 0))


def listener(raises: bool = False) -> SimpleNamespace:
    err = RuntimeError("LISTENER-SECRET") if raises else None
    return SimpleNamespace(on_recorder_failing=AsyncMock(side_effect=err))


class Missing(Exception):
    sqlstate = "42P01"  # the shape of an asyncpg UndefinedTableError


def pg_error(driver: type[Exception]) -> Any:
    """DB-P7-02: SQLAlchemy 2's real asyncpg translation of a driver error, no server."""
    dbapi, up = AsyncAdapt_asyncpg_dbapi(asyncpg), SimpleNamespace(is_closed=lambda: False)
    with pytest.raises(dbapi.Error) as adapted:
        Conn._handle_exception(SimpleNamespace(dbapi=dbapi, _connection=up), driver("no table"))
    return DBAPIError.instance("SELECT x", {"p": "SECRET"}, adapted.value, dbapi.Error)


class Store(InMemoryContextStore):
    """Records each write attempt's batch size; ``hang`` makes reads and writes never return."""

    sizes: tuple[int, ...] = ()
    hang = False
    linger = 0.0  # seconds a hung write takes to finish its cancel (a session close)

    async def load_window(self, *a: Any) -> ContextWindow:
        if self.hang:
            await asyncio.Event().wait()
        return await super().load_window(*a)

    async def write_observations(self, rows: Any) -> int:
        self.sizes += (len(rows),)
        if self.hang:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(self.linger)
        return await super().write_observations(rows)


def poller(r: ContextRecorder) -> tuple[ContextPoller, Any]:
    clock = tp.FakeClock()
    p = ContextPoller(coingecko=tp.FakeClient(clock, [tp.dom_ok()]), sink=r, clock=clock)
    return p, p._providers[0]


def sight(p: tuple[ContextPoller, Any], value: float, fetch_ms: int) -> str:
    return p[0]._apply(p[1], tp.dom_ok(value).value, fetch_ms).outcome


def row(observed: int, avail: int, value: str) -> ContextRow:
    return ContextRow(DOM, dt(observed), dt(avail), Decimal(value), "recorded")


# T-36: the recorder
async def test_only_dominance_and_whale_are_recorded_and_a_second_write_is_idempotent() -> None:
    store = InMemoryContextStore()
    r = ContextRecorder(store)
    for sid in (FGI, "fred:T10Y2Y", "unknown"):
        r.offer(sid, obs())
    r.offer(DOM, obs())
    r.offer(WHALE, fake("-250.123456785"))
    await r.flush(5.0)
    assert {x.series_id for x in store.rows()} == {DOM, WHALE}
    assert not [k for k in metrics.get_all()["counters"] if k.startswith(rec.DROPPED)]  # no label
    assert Decimal("-250.12345679") in {x.value for x in store.rows()}  # 8 decimals, HALF_UP
    again = ContextRecorder(store)  # a restart without a seed writes the same rows again
    again.offer(DOM, obs())
    await again.flush(5.0)
    assert len(store) == 2  # idempotent


async def test_ten_sightings_per_bucket_record_one_row_equal_to_the_books() -> None:
    """ER3-03 via the real poller at 10x the poll rate; CF-2401-46: a new value is no new row."""
    store = InMemoryContextStore()
    r = ContextRecorder(store)
    p = poller(r)
    assert [sight(p, 50 + i, START + i * 360_000) for i in range(10)] == ["ok"] + ["unchanged"] * 9
    await r.flush(5.0)
    assert [(x.available_at, x.value) for x in store.rows()] == [(dt(START), Decimal("50"))]
    assert sight(p, 60, START + H) == "ok"
    await r.flush(5.0)
    assert len(store) == 2
    book = LIVE_BOOK.snapshot()[DOM]
    window = await store.load_window([DOM], dt(START - H), dt(START + 2 * H))
    for at in (START, START + 30 * 60_000, START + H, START + H + 1):
        assert window.series[DOM].asof(at, max_age_ms=CAP) == book.asof(at, max_age_ms=CAP)


async def test_the_daily_budget_binds_per_series_and_day() -> None:
    r = ContextRecorder(InMemoryContextStore())
    for i in range(50):  # a bucket bug: 50 distinct "buckets" inside one UTC day
        r.offer(DOM, obs(observed=START + i, avail=START + i))
    r.offer(DOM, obs(observed=START + 24 * H, avail=START + 24 * H))  # the next day is open
    assert r._queue.qsize() == 49 and count(rec.DROPPED) == 2


def fake(value: str, basis: str = "recorded", avail: int = START + 300_000) -> SimpleNamespace:
    """What ``Observation`` itself would refuse: the recorder is a second line of defence."""
    return SimpleNamespace(observed_ms=START, available_ms=avail, value=Decimal(value), basis=basis)


@pytest.mark.parametrize(
    ("series", "o"),
    [
        (DOM, fake("NaN")),
        (DOM, fake("0")),  # dominance is 0 < v <= 100
        (WHALE, fake("1e13")),  # |net flow| <= 1e12
        (WHALE, fake("1e16")),
        (DOM, fake("5", avail=START - 1)),  # available before observed
        (DOM, fake("5", basis="estimated_lag")),
    ],
)
def test_nothing_out_of_band_nan_or_before_its_observation_is_queued(series: str, o: Any) -> None:
    r = ContextRecorder(InMemoryContextStore())
    r.offer(series, o)
    assert r._queue.empty() and count(rec.DROPPED, series) == 1


def test_the_queue_holds_256_and_drops_the_oldest_with_a_counter() -> None:
    r = ContextRecorder(InMemoryContextStore())
    r.offer(WHALE, obs("5"))  # CM04: the evicted oldest row is counted under its series
    for i in range(300):
        r.offer(DOM, obs(observed=START + i * 24 * H, avail=START + i * 24 * H))
    assert r._queue.qsize() == 256 and (count(rec.DROPPED), count(rec.DROPPED, WHALE)) == (44, 1)
    assert r._queue.get_nowait().observed_at == dt(START + 44 * 24 * H)


@pytest.mark.parametrize("raises", [False, True])
async def test_a_database_failure_stays_off_the_poller_and_calls_the_listener(raises: bool) -> None:
    store = InMemoryContextStore()
    store.fail_with = RuntimeError("SQL-TEXT password=hunter2")
    spy = listener(raises)
    r = ContextRecorder(store, spy)
    r.start()
    p = poller(r)
    with capture_logs() as logs:
        assert sight(p, 50, START) == "ok"  # the poller is not told
        r.offer(WHALE, obs("5"))  # CM05: one mixed batch, a failure for each series
        for _ in range(20):
            await asyncio.sleep(0)
    assert (count(rec.FAILURES), count(rec.FAILURES, WHALE), count(rec.DROPPED)) == (1, 1, 1)
    assert spy.on_recorder_failing.await_count == 1
    assert LIVE_BOOK.freshness(START).series[DOM].fresh is True  # the book is unaffected
    (failed,) = [x for x in logs if x["event"] == "context_recorder.write_failed"]
    assert failed["error_type"] == "RuntimeError" and "hunter2" not in repr(logs)
    assert set(failed) == {"event", "log_level", "rows", "error_type", "error_at"}  # SEC-P7-04
    assert failed["rows"] == 2  # a count, never row values
    assert ("context_recorder.listener_failed" in repr(logs)) is raises  # by type, never its text
    assert "LISTENER-SECRET" not in repr(logs)
    store.fail_with = None
    r.offer(WHALE, obs("5", START + H, START + H))  # the writer survived
    await r.flush(5.0)
    assert len(store) == 1


SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|SELECT)\b")
_OK_CALLS = ("exc_log_fields", "_missing_table")


def _book(tree: ast.AST) -> list[str]:
    return [
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "LIVE_BOOK"
    ]


def recorder_problems(source: str) -> list[str]:
    """No SQL or sqlalchemy (the store owns it), one ``LIVE_BOOK.publish`` inside ``seed``, and an
    except-variable only as the argument of ``exc_log_fields`` / ``_missing_table``."""
    tree, found = ast.parse(source), []
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    handlers = {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import | ast.ImportFrom) and "sqlalchemy" in ast.unparse(n):
            found.append("sqlalchemy")
        if isinstance(n, ast.Constant) and SQL.search(str(n.value)):
            found.append("sql")
        if isinstance(n, ast.Name) and n.id in handlers:
            if getattr(getattr(parents[n], "func", None), "id", "") not in _OK_CALLS:
                found.append(f"exc:{n.id}")
    seed = next(n for n in ast.walk(tree) if getattr(n, "name", "") == "seed")
    return found + ([] if _book(tree) == _book(seed) == ["publish"] else ["book"])


def test_the_recorder_source_has_no_sql_one_publish_site_and_no_exception_text() -> None:
    assert recorder_problems(SRC) == []
    for extra in (
        "from sqlalchemy import text\n",
        "x = 'UPDATE t SET a = 1'\n",
        "LIVE_BOOK.heartbeat(1)\n",
        "try:\n    pass\nexcept Exception as boom:\n    print(boom)\n",
    ):
        assert recorder_problems(SRC + "\n" + extra) != []


async def test_the_failure_counter_starts_at_zero_and_the_alert_rule_reads_its_name() -> None:
    """CF-2401-109 / DO-P6-01: the name, its zero init and the rule expression agree."""
    r = ContextRecorder(InMemoryContextStore())
    metrics.increment(rec.FAILURES, labels={"series": DOM})  # SEC-P7-05: start never resets
    r.start()
    exported = "trading_mtf_context_recorder_write_failures_total"
    assert rec.FAILURES == exported.removeprefix("trading_")
    samples = {
        s.labels["series"]: s.value
        for f in TradingBotCollector(0.0).collect()
        for s in f.samples
        if s.name == exported
    }
    assert samples == {DOM: 1.0, WHALE: 0.0} and set(rec.SERIES) == set(samples)
    assert f"increase({exported}[15m])" in _expr(_rules()["mtf-context-recorder-failing"])
    await r.flush(5.0)


# T-37: a missing table disables the recorder
@pytest.mark.parametrize("shape", ["driver", "wrapped"])
async def test_a_missing_table_disables_with_one_info_line_and_rechecks_hourly(shape: str) -> None:
    store, spy, now = Store(), listener(), [1_000.0]
    store.fail_with = Missing() if shape == "driver" else pg_error(UndefinedTableError)
    r = ContextRecorder(store, spy, monotonic=lambda: now[0])
    p = poller(r)
    with capture_logs() as logs:
        for i in range(4):  # i=0 hits the missing table; 1..3 arrive inside the hour
            assert sight(p, 50 + i, START + i * H) == "ok"  # polling and freshness are unaffected
            await r.flush(5.0)
            now[0] += 1_000
        assert LIVE_BOOK.freshness(START + 3 * H).series[DOM].fresh is True
        now[0] += 3_600  # the re-check: still missing -> no second line
        r.offer(DOM, obs(observed=START + 9 * H, avail=START + 9 * H))
        await r.flush(5.0)
        store.fail_with = None
        now[0] += 3_600
        r.offer(DOM, obs(observed=START + 10 * H, avail=START + 10 * H))
        await r.flush(5.0)
    info = [x["event"] for x in logs if x["log_level"] == "info"]
    assert info == ["context_recorder.disabled", "context_recorder.enabled"]
    assert (count(rec.FAILURES), len(store), count(rec.DROPPED)) == (0, 1, 5)
    assert spy.on_recorder_failing.await_count == 0 and store.sizes == (1, 1, 1)  # 3 attempts
    assert "SECRET" not in repr(logs)


def test_the_real_asyncpg_shapes_are_classified_by_sqlstate_and_42501_is_a_failure() -> None:
    cause_only = RuntimeError("a wrapper without .orig")
    cause_only.__cause__ = pg_error(UndefinedTableError).orig  # CM02: the cause is read too
    assert rec._missing_table(pg_error(UndefinedTableError)) and rec._missing_table(cause_only)
    assert not rec._missing_table(pg_error(InsufficientPrivilegeError))  # S05, S15


# T-38: the boot seed
class Lying(InMemoryContextStore):
    """A store that returns a row the window excludes (AR4-07b: the seed pre-filters)."""

    async def load_window(self, series_ids: Any, start: Any, end: Any) -> ContextWindow:
        self.window = (start, end)
        bn = NOW - NOW % H
        bad = obs("150", bn - H, bn - H + 60_000)  # SEC-P7-01: out of band (K01: see the 60 row)
        series = PitSeries(DOM, (bad, obs("50", bn, bn + 60_000), obs("60", bn - H, NOW + H)))
        return ContextWindow(series={DOM: series}, row_count=2)


async def test_the_seed_restores_dominance_7d_with_the_recorded_stamps() -> None:
    b0, bn = NOW - 7 * 24 * H - 30 * 60_000, NOW - NOW % H
    store = InMemoryContextStore()
    await store.write_observations(
        [row(b0, b0 + 300_000, "50"), row(bn, bn + 60_000, "55"), row(bn, bn + 1_800_000, "99")]
    )
    r = ContextRecorder(store)
    await r.seed(NOW)
    book = LIVE_BOOK.snapshot()[DOM]
    assert [(o.observed_ms, o.available_ms, o.value) for o in book.observations] == [
        (b0, b0 + 300_000, Decimal("50")),
        (bn, bn + 60_000, Decimal("55")),  # the earliest sighting of the bucket
    ]
    assert book.asof(NOW, max_age_ms=CAP) == 55
    assert book.asof(NOW - 7 * 24 * H, max_age_ms=CAP) == 50  # dominance 7d restored
    assert book.asof(bn + 59_999, max_age_ms=CAP) is None  # not visible before it was available
    assert LIVE_BOOK.view(NOW)[DOM] is book
    r.offer(DOM, obs(observed=bn, avail=NOW))  # CF-2401-46: the seeded bucket is not recorded again
    assert r._queue.empty()


async def test_a_future_stamped_row_is_filtered_and_never_makes_the_seed_raise() -> None:
    store = Lying()
    await ContextRecorder(store).seed(NOW)
    assert [o.value for o in LIVE_BOOK.snapshot()[DOM].observations] == [Decimal("50")]
    assert store.window == (dt(NOW - 7 * 24 * H - CAP), dt(NOW))  # 7d + the largest max_age


async def test_a_failed_or_hung_store_leaves_the_book_empty_and_raises_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryContextStore()
    store.fail_with = RuntimeError("DSN-SECRET")
    with capture_logs() as logs:
        await ContextRecorder(store).seed(NOW)
    assert len(LIVE_BOOK.snapshot()) == 0 and "DSN-SECRET" not in repr(logs)
    assert rec.SEED_TIMEOUT_S == 3.0  # SEC-P7-02, RK-P7-03 (K04), CM14: the T-38 boot bound
    monkeypatch.setattr(rec, "SEED_TIMEOUT_S", 0.05)
    hung = Store()
    hung.hang = True
    await ContextRecorder(hung).seed(NOW)
    assert len(LIVE_BOOK.snapshot()) == 0


# Flush, shutdown order (CF-2401-99) and wiring
async def test_flush_writes_the_queue_stops_the_writer_and_is_bounded() -> None:
    store = InMemoryContextStore()
    r = ContextRecorder(store)
    r.start()
    writer = r._task
    r.offer(DOM, obs())
    await r.flush(5.0)
    assert len(store) == 1 and r._task is None and writer is not None and writer.done()

    hung = Store()
    hung.hang = True
    r = ContextRecorder(hung)
    for i in range(200):  # a batch of 64 is in flight when the bound hits, 136 are left over
        r.offer(DOM, obs(observed=START + i * 24 * H, avail=START + i * 24 * H))
    with pytest.raises(TimeoutError):
        await r.flush(0.05)
    assert hung.sizes == (64,) and count(rec.DROPPED) == 200 and r._queue.empty()
    hung.linger = 1.0  # CR-P7-01: the writer takes 1 s to cancel, the outer bound fires
    r.start()
    r.offer(WHALE, obs("5"))
    await asyncio.sleep(0.01)  # the writer is inside the hung write
    r.offer(WHALE, obs("6", START + H, START + H))
    await ContextPoller(sink=r).flush_sink(0.05)  # the poller counts the timeout too
    assert metrics.get_all()["counters"]["mtf_context_record_flush_timeout_total"] == 1
    assert r._queue.empty()  # the queued row is counted, not lost


def test_the_lifespan_wires_the_seed_before_the_poller_and_the_flag_before_the_flush() -> None:
    fn, line = ls._lifespan_tree(), ls._first_line
    seed, build = line(fn, "_recorder.seed("), line(fn, "ContextPoller(")
    assert line(fn, "_kill_switch.load(") < seed < build < line(fn, "_recorder.start()")
    assert line(fn, "_recorder.start()") < line(fn, "recover_orphaned_runs()")
    assert line(fn, "mark_shutdown_requested()") < line(fn, "_poller.stop()")  # CF-2401-99
    calls = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert "ContextRecorder(SqlContextStore(get_session_factory()), _alerts)" in calls
    assert "_recorder.seed(time.time_ns() // 1000000)" in calls  # RK-P7-03: the true now
    (built,) = [c for c in calls if c.startswith("ContextPoller(")]
    assert "sink=_recorder" in built and "listener=_alerts" in built


def test_the_real_lifespan_marks_the_orphan_flag_before_the_recorder_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.services.run_orchestrator as orchestrator

    ls._env(monkeypatch)
    events: list[str] = []
    real_flush = ContextRecorder.flush

    def flush(self: ContextRecorder, timeout_s: float) -> Any:
        events.append("flush")
        return real_flush(self, timeout_s)

    monkeypatch.setattr(ContextRecorder, "flush", flush)
    monkeypatch.setattr(orchestrator, "mark_shutdown_requested", lambda: events.append("mark"))
    ls._boot(monkeypatch)
    assert events == ["mark", "flush"]  # CF-2401-99


@pytest.mark.parametrize("target", ["load_window", "__init__", "start"])
def test_a_recorder_or_store_that_fails_never_blocks_the_boot(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from api.services.context_store import SqlContextStore

    async def broken_load(self: Any, *a: Any) -> None:
        raise RuntimeError("DSN-SECRET")

    def broken_init(self: Any, *a: Any) -> None:
        raise RuntimeError("DSN-SECRET")

    broken = {"load_window": broken_load, "__init__": broken_init, "start": broken_init}[target]
    monkeypatch.setattr(ContextRecorder if target == "start" else SqlContextStore, target, broken)
    ls._env(monkeypatch)
    order, _, seen = ls._boot(monkeypatch)
    assert order[:3] == ["kill_switch", "poller_start:4", "recovery"] and seen["book_empty"] is True
    assert seen["poller_running"] is True and seen["elapsed"] < 10
