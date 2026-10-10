"""
tests/migrations/test_wp15b_resume_hints.py
---------------------------------------------
WP1.5b T-32 (real PostgreSQL): the daily-risk-state seed hints read by
``apps.api.services.run_persistence.load_resume_snapshot(db, run, now=...)``.

MUST run against a real PostgreSQL instance -- same convention as the other
files in this directory (reads ``MIGRATION_TEST_DATABASE_URL``, skips when
unset; the shared teardown fixture in ``conftest.py`` empties ``runs``, and
``equity_snapshots`` cascades).

Cases:
1. ``day_start_equity_hint`` is the latest snapshot strictly before 00:00 UTC
   of the ``now`` day; a timestamp tie is broken by the higher ``bar_index``.
2. ``day_min_equity_hint`` / ``day_first_equity_hint`` come from the rows at or
   after 00:00 UTC: the minimum, and the earliest row (a tie is broken by the
   LOWER ``bar_index``).  A row exactly at 00:00:00 counts as today.
3. An empty run: hints ``None``, ``daily_hints_day`` and ``run_started_at`` set.
4. Rows of another run never leak in.
5. ``now``: a naive value is UTC; an aware non-UTC value is converted.
6. One of ``day_min`` / ``day_first`` is set if and only if the other is.
7. The values are exactly what the seed validator accepts (no false
   ``unknown`` on the normal path): ``Decimal`` equities, an aware start.
8. The hint statements are parameterised (no values in the SQL text) and the
   planner can serve each of them from ``ix_equity_snapshots_run_id_timestamp``.
9. T-46b (1), WP15B-CR-01 end to end: a first resume must not leave the
   ``from_fills`` rebuild point (positions at old fill prices) in
   ``equity_snapshots``; a second resume the same UTC day reads the real marks.
10. T-46b (2): ``bar_index`` stays gap-free and collision-free across two
    consecutive resumes against the real ``uq_equity_snapshots_run_bar``
    (regression pin), including an empty run (first row ``bar_index = 0``).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- these tests need a real "
        "Postgres instance and are skipped in environments without one. "
        "See test_wp18b_scan_and_import.py's module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

D = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
D_OPEN = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
RUN_STARTED = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def _alembic_config(database_url: str) -> Any:
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    return cfg


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


@pytest.fixture()
def pg_env(monkeypatch: pytest.MonkeyPatch) -> str:
    assert _MIGRATION_URL is not None  # guarded by pytestmark skipif
    monkeypatch.setenv("DATABASE_URL", _MIGRATION_URL)
    from api.config import get_settings

    get_settings.cache_clear()

    import api.db.session as session_module

    session_module._engine = None
    session_module._session_factory = None

    from alembic import command

    cfg = _alembic_config(_MIGRATION_URL)
    command.upgrade(cfg, "head")

    return _MIGRATION_URL


@pytest.fixture()
async def db_session(pg_env: str) -> Any:
    from api.db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.rollback()


async def _seed_run(run_id: uuid.UUID, *, started_at: datetime = RUN_STARTED) -> None:
    import asyncpg

    assert _MIGRATION_URL is not None
    conn = await asyncpg.connect(_asyncpg_dsn(_MIGRATION_URL))
    try:
        config = (
            '{"strategy_name": "grid_trading", "symbols": ["BTC/USD"], "timeframe": "1h",'
            ' "initial_capital": "10000", "strategy_params": {}}'
        )
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'resuming', $2::jsonb, $3, now(), now())
            """,
            run_id,
            config,
            started_at,
        )
    finally:
        await conn.close()


async def _seed_snapshots(run_id: uuid.UUID, rows: list[tuple[int, datetime, str]]) -> None:
    """``rows`` = ``(bar_index, timestamp, equity)``; inserted in the given order."""
    import asyncpg

    assert _MIGRATION_URL is not None
    conn = await asyncpg.connect(_asyncpg_dsn(_MIGRATION_URL))
    try:
        for bar_index, ts, equity in rows:
            await conn.execute(
                """
                INSERT INTO equity_snapshots
                    (run_id, equity, cash, unrealised_pnl, realised_pnl, drawdown_pct,
                     bar_index, "timestamp")
                VALUES ($1, $2, 0, 0, 0, 0, $3, $4)
                """,
                run_id,
                Decimal(equity),
                bar_index,
                ts,
            )
    finally:
        await conn.close()


async def _load(db: Any, run_id: uuid.UUID, now: datetime) -> Any:
    from sqlalchemy import select

    from api.db.models import RunORM
    from api.services.run_persistence import load_resume_snapshot

    run = (await db.execute(select(RunORM).where(RunORM.id == run_id))).scalar_one()
    return await load_resume_snapshot(db, run, now=now)


# ---------------------------------------------------------------------------
# 1. day_start_equity_hint
# ---------------------------------------------------------------------------


async def test_wp15_t32_day_start_is_the_latest_snapshot_before_midnight(db_session: Any) -> None:
    run_id = uuid.uuid4()
    await _seed_run(run_id)
    await _seed_snapshots(
        run_id,
        [
            (3, D_OPEN - timedelta(hours=5), "9800"),
            (4, D_OPEN - timedelta(hours=1), "9700"),  # the latest timestamp ...
            (5, D_OPEN - timedelta(hours=1), "9650"),  # ... tie -> the higher bar_index wins
            (6, D_OPEN - timedelta(hours=2), "9999"),  # an older timestamp, even with a high bar
        ],
    )
    snap = await _load(db_session, run_id, D)
    assert snap.day_start_equity_hint == Decimal("9650")
    assert snap.day_min_equity_hint is None and snap.day_first_equity_hint is None


async def test_wp15_t32_a_row_exactly_at_midnight_is_today_not_the_day_start(
    db_session: Any,
) -> None:
    run_id = uuid.uuid4()
    await _seed_run(run_id)
    await _seed_snapshots(
        run_id,
        [(1, D_OPEN - timedelta(seconds=1), "9500"), (2, D_OPEN, "9400")],
    )
    snap = await _load(db_session, run_id, D)
    assert snap.day_start_equity_hint == Decimal("9500")  # strictly before 00:00:00
    assert snap.day_first_equity_hint == Decimal("9400")  # at 00:00:00 -> today
    assert snap.day_min_equity_hint == Decimal("9400")


# ---------------------------------------------------------------------------
# 2. today's minimum and earliest row
# ---------------------------------------------------------------------------


async def test_wp15_t32_today_minimum_and_first_with_a_tie_on_the_lower_bar_index(
    db_session: Any,
) -> None:
    run_id = uuid.uuid4()
    await _seed_run(run_id)
    first_ts = D_OPEN + timedelta(minutes=5)
    await _seed_snapshots(
        run_id,
        [
            (9, first_ts, "9000"),  # inserted FIRST, but the higher bar_index at the same ts
            (8, first_ts, "9660"),  # the lower bar_index -> the earliest row
            (10, D_OPEN + timedelta(minutes=10), "9100"),
            (11, D_OPEN + timedelta(hours=1), "9500"),
            (2, D_OPEN - timedelta(hours=3), "8000"),  # yesterday: never in today's min
        ],
    )
    snap = await _load(db_session, run_id, D)
    assert snap.day_first_equity_hint == Decimal("9660")
    assert snap.day_min_equity_hint == Decimal("9000")
    assert snap.day_start_equity_hint == Decimal("8000")


# ---------------------------------------------------------------------------
# 3. an empty run
# ---------------------------------------------------------------------------


async def test_wp15_t32_an_empty_run_has_no_hints_but_the_day_and_the_start(
    db_session: Any,
) -> None:
    run_id = uuid.uuid4()
    started = datetime(2026, 3, 10, 7, 30, tzinfo=UTC)
    await _seed_run(run_id, started_at=started)
    snap = await _load(db_session, run_id, D)
    assert snap.day_start_equity_hint is None
    assert snap.day_min_equity_hint is None
    assert snap.day_first_equity_hint is None
    assert snap.daily_hints_day == date(2026, 3, 10)
    assert snap.run_started_at == started
    assert snap.daily_seed is not None  # never None once hints were read
    # the pre-existing fields are unchanged for an empty run
    assert snap.peak_equity_hint is None and snap.max_bar_index == -1


# ---------------------------------------------------------------------------
# 4. run scoping
# ---------------------------------------------------------------------------


async def test_wp15_t32_other_runs_snapshots_never_leak_in(db_session: Any) -> None:
    run_a, run_b = uuid.uuid4(), uuid.uuid4()
    await _seed_run(run_a)
    await _seed_run(run_b)
    await _seed_snapshots(
        run_a, [(1, D_OPEN - timedelta(hours=1), "9900"), (2, D_OPEN + timedelta(hours=1), "9800")]
    )
    await _seed_snapshots(
        run_b, [(1, D_OPEN - timedelta(hours=2), "1"), (2, D_OPEN + timedelta(hours=2), "2")]
    )
    snap = await _load(db_session, run_a, D)
    assert snap.day_start_equity_hint == Decimal("9900")
    assert snap.day_min_equity_hint == Decimal("9800")
    assert snap.day_first_equity_hint == Decimal("9800")


# ---------------------------------------------------------------------------
# 5. the `now` argument
# ---------------------------------------------------------------------------


async def test_wp15_t32_now_naive_is_utc_and_an_aware_offset_is_converted(
    db_session: Any,
) -> None:
    run_id = uuid.uuid4()
    await _seed_run(run_id)
    await _seed_snapshots(
        run_id, [(1, D_OPEN - timedelta(hours=1), "9900"), (2, D_OPEN + timedelta(hours=1), "9800")]
    )
    naive = await _load(db_session, run_id, datetime(2026, 3, 10, 12, 0))  # treated as UTC
    assert naive.daily_hints_day == date(2026, 3, 10)
    assert naive.day_min_equity_hint == Decimal("9800")

    # 2026-03-11 01:30 +02:00 == 2026-03-10 23:30 UTC -> still the 10th
    plus2 = await _load(
        db_session, run_id, datetime(2026, 3, 11, 1, 30, tzinfo=timezone(timedelta(hours=2)))
    )
    assert plus2.daily_hints_day == date(2026, 3, 10)
    assert plus2.day_min_equity_hint == Decimal("9800")

    # the next UTC day: both rows are "before today", nothing is "today"
    next_day = await _load(db_session, run_id, D + timedelta(days=1))
    assert next_day.daily_hints_day == date(2026, 3, 11)
    assert next_day.day_start_equity_hint == Decimal("9800")  # the latest before the 11th
    assert next_day.day_min_equity_hint is None and next_day.day_first_equity_hint is None


# ---------------------------------------------------------------------------
# 6 + 7. one set <=> the other; the values pass the strict seed validator
# ---------------------------------------------------------------------------


async def test_wp15_t32_min_and_first_are_set_together_and_pass_the_seed_validator(
    db_session: Any,
) -> None:
    from trading.portfolio import _invalid_seed_field

    with_rows, without_rows = uuid.uuid4(), uuid.uuid4()
    await _seed_run(with_rows)
    await _seed_run(without_rows)
    await _seed_snapshots(
        with_rows,
        [(1, D_OPEN - timedelta(hours=1), "9900"), (2, D_OPEN + timedelta(hours=1), "9800")],
    )
    await _seed_snapshots(without_rows, [(1, D_OPEN - timedelta(hours=1), "9900")])

    for run_id in (with_rows, without_rows):
        snap = await _load(db_session, run_id, D)
        assert (snap.day_min_equity_hint is None) == (snap.day_first_equity_hint is None)
        for hint in (
            snap.day_start_equity_hint,
            snap.day_min_equity_hint,
            snap.day_first_equity_hint,
        ):
            assert hint is None or type(hint) is Decimal
        assert snap.run_started_at is not None and snap.run_started_at.utcoffset() is not None
        assert _invalid_seed_field(snap.daily_seed) is None  # no false "unknown" on the DB path


# ---------------------------------------------------------------------------
# 8. parameterised + index-served
# ---------------------------------------------------------------------------


async def test_wp15_t32_the_hint_queries_are_parameterised_and_index_served(
    db_session: Any,
) -> None:
    import asyncpg
    from sqlalchemy import event

    run_id = uuid.uuid4()
    await _seed_run(run_id)
    await _seed_snapshots(run_id, [(1, D_OPEN - timedelta(hours=1), "9900")])

    captured: list[tuple[str, Any]] = []

    def _capture(conn: Any, cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        if "equity_snapshots" in statement:
            captured.append((statement, parameters))

    sync_engine = db_session.bind.sync_engine
    event.listen(sync_engine, "before_cursor_execute", _capture)
    try:
        await _load(db_session, run_id, D)
    finally:
        event.remove(sync_engine, "before_cursor_execute", _capture)

    # the day-start query and the (min, first) query -- the run_id-only peak / max-bar
    # queries have no timestamp predicate
    hint_queries = [(s, p) for s, p in captured if "timestamp" in s.split("WHERE", 1)[-1]]
    assert len(hint_queries) == 2, [s for s, _ in captured]

    assert _MIGRATION_URL is not None
    conn = await asyncpg.connect(_asyncpg_dsn(_MIGRATION_URL))
    try:
        await conn.execute("SET enable_seqscan = off")
        for statement, parameters in hint_queries:
            # parameterised: the run id and the day-open instant are bound, never inlined
            assert str(run_id) not in statement and "2026-03-10" not in statement
            assert "$1" in statement
            assert run_id in tuple(parameters) and D_OPEN in tuple(parameters)
            plan = "\n".join(r[0] for r in await conn.fetch("EXPLAIN " + statement, *parameters))
            assert "ix_equity_snapshots_run_id_timestamp" in plan, plan
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 9 / 10. WP15B-CR-01 -- the from_fills curve point is never persisted (T-46b)
# ---------------------------------------------------------------------------

_SYM = "BTC/USD"
_PRE_MIDNIGHT = D_OPEN - timedelta(hours=1)  # D-1 23:00
_RESUME_1 = D_OPEN + timedelta(hours=3)
_MARK_1 = D_OPEN + timedelta(hours=3, minutes=5)
_RESUME_2 = D_OPEN + timedelta(hours=6)


def _buy_25_at_100() -> Any:
    from tests.unit._wp15_support import make_fill

    return make_fill(
        quantity="25",
        price="100",
        symbol=_SYM,
        executed_at=D_OPEN - timedelta(days=5),
    )


async def _resume_and_flush(
    db: Any,
    run_id: uuid.UUID,
    *,
    now: datetime,
    marks: list[tuple[datetime, Decimal]],
    fills: list[Any],
    pre_fix_watermark: bool = False,
) -> Any:
    """One resume as the orchestrator sites do it, then one REAL flush per mark.

    ``pre_fix_watermark=True`` reproduces the 1.5b tree (``bar_index_offset =
    max_bar_index + 1``, nothing counted as flushed) -- control only.
    """
    from api.services.run_orchestrator import _IncrementalFlushState, flush_incremental
    from tests.unit._wp15_support import Clock
    from trading.portfolio import PortfolioAccounting

    snap = await _load(db, run_id, now)
    clock = Clock(now)
    portfolio = PortfolioAccounting.from_fills(
        snap.run_id,
        snap.initial_cash,
        fills,
        peak_equity_hint=snap.peak_equity_hint,
        now=now,
        clock=clock,
        daily_seed=snap.daily_seed,
    )
    state = _IncrementalFlushState()
    if pre_fix_watermark:
        state.bar_index_offset = snap.max_bar_index + 1
    else:
        # exactly the two resume sites in run_orchestrator (WP15B-CR-01)
        state.flushed_equity_count = len(portfolio.get_equity_curve())
        state.bar_index_offset = snap.max_bar_index
    state.peak_equity = snap.peak_equity_hint or Decimal("0")

    log = MagicMock()
    for at, price in marks:
        clock.at = at
        portfolio.update_market_prices({_SYM: price})
        await flush_incremental(
            run_id_str=snap.run_id,
            portfolio=portfolio,
            execution_engine=None,
            state=state,
            log=log,
        )
    # flush_incremental swallows DB errors (it only logs them): make that loud
    log.exception.assert_not_called()
    return snap


async def _equity_rows(run_id: uuid.UUID) -> list[tuple[int, Decimal]]:
    import asyncpg

    assert _MIGRATION_URL is not None
    conn = await asyncpg.connect(_asyncpg_dsn(_MIGRATION_URL))
    try:
        rows = await conn.fetch(
            "SELECT bar_index, equity FROM equity_snapshots WHERE run_id = $1 ORDER BY bar_index",
            run_id,
        )
    finally:
        await conn.close()
    return [(r["bar_index"], r["equity"]) for r in rows]


async def test_wp15_t46b_second_same_day_resume_reads_real_marks_not_the_rebuild_point(
    db_session: Any,
) -> None:
    """WP15B-CR-01 as the critic executed it: no false worst after a second restart."""
    from common.types import OrderSide
    from tests.unit._wp15_support import Clock
    from tests.unit.test_wp15a_resume_seed import _gate
    from trading.portfolio import PortfolioAccounting
    from trading.risk import RiskParameters
    from trading.risk_manager import DefaultRiskManager

    run_id = uuid.uuid4()
    await _seed_run(run_id)
    await _seed_snapshots(run_id, [(5, _PRE_MIDNIGHT, "11000")])
    fills = [_buy_25_at_100()]

    # resume #1 (03:00): rebuild (10 000 at the old fill price) then a real mark 140 -> 11 000
    await _resume_and_flush(
        db_session,
        run_id,
        now=_RESUME_1,
        marks=[(_MARK_1, Decimal("140"))],
        fills=fills,
    )
    # the rebuild point is not persisted: the only row of today is the real mark, bar 6
    assert await _equity_rows(run_id) == [(5, Decimal("11000")), (6, Decimal("11000"))]

    # resume #2 (06:00, same UTC day) reads the hints
    snap = await _load(db_session, run_id, _RESUME_2)
    assert snap.day_first_equity_hint == Decimal("11000")
    assert snap.day_min_equity_hint == Decimal("11000")
    assert snap.max_bar_index == 6  # previous max (5) + 1

    clock = Clock(_RESUME_2)
    p = PortfolioAccounting.from_fills(
        snap.run_id,
        snap.initial_cash,
        fills,
        peak_equity_hint=snap.peak_equity_hint,
        now=_RESUME_2,
        clock=clock,
        daily_seed=snap.daily_seed,
    )
    assert p._worst_nav_pnl == Decimal("0")  # the seed itself carries no loss
    # the engine marks every bar before the gate runs; the unmarked rebuild NAV
    # (positions at the old fill price, 10 000) is deliberately not the subject here
    clock.at = _RESUME_2 + timedelta(minutes=5)
    p.update_market_prices({_SYM: Decimal("140")})
    risk = p.risk_snapshot()
    assert risk.day_start_source == "snapshot"
    assert risk.worst_nav_pnl == Decimal("0")
    assert not risk.state_unknown
    manager = DefaultRiskManager(run_id="wp15b-t46b", params=RiskParameters())
    assert (
        manager._check_daily_loss(risk.gate_daily_pnl(risk.nav), risk.nav, risk.daily_loss_basis)
        is None
    )
    assert _gate(p, OrderSide.BUY) is True


async def test_wp15_t46b_bar_index_is_gap_free_across_two_resumes_on_the_real_unique_constraint(
    db_session: Any,
) -> None:
    """Regression pin (green at 1.5b too): ``uq_equity_snapshots_run_bar`` never trips."""
    cases = {
        "empty-run": ([], 0),  # max_bar_index -1 -> first persisted row is bar 0
        "existing-rows": ([(0, "9000"), (1, "9100"), (2, "9200")], 3),
    }
    for name, (seed_rows, first_new_bar) in cases.items():
        run_id = uuid.uuid4()
        await _seed_run(run_id)
        if seed_rows:
            base = D_OPEN - timedelta(days=2)
            await _seed_snapshots(
                run_id,
                [(b, base + timedelta(hours=b), eq) for b, eq in seed_rows],
            )
        fills = [_buy_25_at_100()]

        # resume #1: two marks -> two rows; resume #2 (a restart): two more marks
        await _resume_and_flush(
            db_session,
            run_id,
            now=_RESUME_1,
            marks=[
                (_MARK_1, Decimal("140")),
                (_MARK_1 + timedelta(minutes=1), Decimal("150")),
            ],
            fills=fills,
        )
        await _resume_and_flush(
            db_session,
            run_id,
            now=_RESUME_2,
            marks=[
                (_RESUME_2 + timedelta(minutes=5), Decimal("160")),
                (_RESUME_2 + timedelta(minutes=6), Decimal("170")),
            ],
            fills=fills,
        )

        rows = await _equity_rows(run_id)
        bars = [b for b, _ in rows]
        assert bars == list(range(len(seed_rows) + 4)), (name, bars)
        assert bars[len(seed_rows)] == first_new_bar, name
        # the four real marks, in order: 7500 + 25 x price
        assert [eq for _, eq in rows][len(seed_rows) :] == [
            Decimal("11000"),
            Decimal("11250"),
            Decimal("11500"),
            Decimal("11750"),
        ]
