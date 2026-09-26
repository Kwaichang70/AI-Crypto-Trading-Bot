"""
tests/unit/test_sprint24_run_recovery.py
-----------------------------------------
Unit tests for boot-time orphan recovery (`recover_orphaned_runs`).

WP1.8a note
-----------
This file originally covered Sprint 24's "copy to a new run_id, linked via
recovered_from_run_id" recovery chain.  WP1.8a (Verbeterplan v2 synthesis
spec S5/S6) replaces that behaviour entirely:

- LIVE runs (`status IN ('running', 'orphaned')`) are left/transitioned to
  `'orphaned'` with NO engine task started (O1) -- an operator must call
  `POST /runs/{id}/resume`.  No strategy/gate validation happens at boot
  for live any more, because boot never starts a live task either way.
- PAPER runs (`status IN ('running', 'orphaned')`) are rebuilt IN PLACE
  under the SAME `run_id` from persisted fill history and their task
  restarts immediately, subject to a bounded `config["resume_count"]`
  budget (> 3 -> `'error'`) and a fill/order integrity check
  (`check_fill_integrity`, O10/WP18-R-06).
- The Sprint 24 `recovered_from_run_id IS NULL` filter is dropped for both
  modes (S6) -- the query now keys off `status` alone.

Kept in this file (same name) so the historical Sprint 24 test-count
context survives in git history; the test bodies below are new.

Modules under test
-------------------
- apps/api/routers/runs.py -- recover_orphaned_runs(), _mark_run_error()
  (`_mark_orphan_error` is kept as a backwards-compat alias),
  `_orphan_live_run()`.

Design notes (mocking pattern, unchanged from Sprint 24)
---------------------------------------------------------
- recover_orphaned_runs() imports get_session_factory and RunORM lazily
  INSIDE the function body, so patch targets are the source modules:
    * "api.db.session.get_session_factory"
  _get_strategy_registry / _load_resume_snapshot / _run_paper_engine /
  _run_live_engine are module-level names in api.routers.runs:
    * "api.routers.runs._get_strategy_registry"
    * "api.routers.runs._load_resume_snapshot"
    * "api.routers.runs._run_paper_engine"
    * "api.routers.runs._run_live_engine"
- `_load_resume_snapshot` is mocked to return a canned `ResumeSnapshot`
  directly -- its own DB-query correctness is covered by
  `tests/unit/test_wp18a_recovery.py` (check_fill_integrity) and the
  integration/migration tests, not re-derived here via mock plumbing.
  An EMPTY snapshot (`fills=[], orders=[]`) trivially passes
  `check_fill_integrity`, so most tests below never need to think about it.
- `_build_factory` returns successive sessions per `factory()` call
  (reusing the last one if more calls happen than sessions supplied); each
  session's `scalar_one_or_none()` returns the SAME orphan `SimpleNamespace`
  so in-place mutations across sessions accumulate correctly, exactly as a
  real ORM object attached across statements in one logical flow would.
- asyncio.create_task is NOT patched -- the real event loop is used so
  _RUN_TASKS receives a real asyncio.Task; the coroutines themselves are
  AsyncMock so they resolve immediately (no I/O).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routers.runs as runs_module
from api.routers.runs import _mark_orphan_error, _orphan_live_run, recover_orphaned_runs
from common.types import OrderSide, OrderStatus, OrderType
from trading.models import Fill, Order
from trading.recovery import ResumeSnapshot

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_orphan(
    run_mode: str = "paper",
    config: dict | None = None,
    status: str = "running",
    strategy_name: str = "grid_trading",
) -> SimpleNamespace:
    """Build a SimpleNamespace that mimics a RunORM orphan/candidate row."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        run_mode=run_mode,
        status=status,
        config=config
        or {
            "strategy_name": strategy_name,
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
            "mode": run_mode,
        },
        started_at=datetime.now(UTC),
        stopped_at=None,
        updated_at=datetime.now(UTC),
    )


_EMPTY_SNAPSHOT = ResumeSnapshot(
    run_id="unused",
    initial_cash=Decimal("10000"),
    fills=[],
    orders=[],
    peak_equity_hint=None,
    max_bar_index=-1,
)


def _mismatched_snapshot() -> ResumeSnapshot:
    """A ResumeSnapshot whose fill/order history fails check_fill_integrity.

    filled_quantity (0.02) does not match the single persisted fill's
    quantity (0.01) within tolerance -- 'fill_history_partial' (S6/R-06:
    a lost fill in the 30s flush window).
    """
    order = Order(
        client_order_id=f"wp18a-{uuid.uuid4().hex[:12]}",
        run_id="mismatched",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.02"),
    )
    fill = Fill(
        order_id=order.order_id,
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
        fee=Decimal("0.5"),
        fee_currency="USD",
    )
    return ResumeSnapshot(
        run_id="mismatched",
        initial_cash=Decimal("10000"),
        fills=[fill],
        orders=[order],
        peak_equity_hint=None,
        max_bar_index=-1,
    )


def _make_session_for_select(orphans: list) -> AsyncMock:
    """Return a mock async session whose execute() returns *orphans* via scalars().all()."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    result = MagicMock()
    result.scalars.return_value.all.return_value = orphans
    session.execute = AsyncMock(return_value=result)
    return session


def _make_session_for_write(orphan: SimpleNamespace | None) -> AsyncMock:
    """Return a mock async session for write operations.

    scalar_one_or_none() returns *orphan* (used by _mark_run_error,
    _orphan_live_run, and the final paper-resume commit block).
    """
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    result = MagicMock()
    result.scalar_one_or_none.return_value = orphan
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


def _build_factory(*sessions: AsyncMock) -> MagicMock:
    """Build a factory mock whose successive calls return successive sessions.

    If more calls are made than sessions are provided, the last session is
    reused.
    """
    contexts = []
    for s in sessions:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=s)
        ctx.__aexit__ = AsyncMock(return_value=False)
        contexts.append(ctx)

    call_count = [-1]

    def _factory():
        call_count[0] += 1
        idx = min(call_count[0], len(contexts) - 1)
        return contexts[idx]

    return MagicMock(side_effect=_factory)


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_strategy_registry():
    original = runs_module._STRATEGY_REGISTRY
    runs_module._STRATEGY_REGISTRY = None
    yield
    runs_module._STRATEGY_REGISTRY = original


@pytest.fixture(autouse=True)
def _clean_run_tasks():
    runs_module._RUN_TASKS.clear()
    yield
    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()


def _patched_registry(*names: str):
    return patch(
        "api.routers.runs._get_strategy_registry",
        return_value={name: MagicMock() for name in names},
    )


# ---------------------------------------------------------------------------
# TestMarkRunError
# ---------------------------------------------------------------------------


class TestMarkRunError:
    """_mark_orphan_error (alias of _mark_run_error) marks a non-terminal run 'error'."""

    @pytest.mark.asyncio
    async def test_running_run_is_marked_error(self) -> None:
        orphan = _make_orphan(status="running")
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, orphan.id, log, reason="test")

        assert orphan.status == "error"
        assert orphan.stopped_at is not None
        write_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_orphaned_run_is_also_marked_error(self) -> None:
        """WP1.8a: 'orphaned' (not just 'running') is a valid source status."""
        orphan = _make_orphan(status="orphaned")
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, orphan.id, log, reason="test")

        assert orphan.status == "error"
        write_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminal_run_is_not_modified(self) -> None:
        orphan = _make_orphan(status="stopped")
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, orphan.id, log, reason="test")

        assert orphan.status == "stopped"
        write_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_run_is_a_no_op(self) -> None:
        write_session = _make_session_for_write(None)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, uuid.uuid4(), log, reason="test")

        write_session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestOrphanLiveRun
# ---------------------------------------------------------------------------


class TestOrphanLiveRun:
    """_orphan_live_run transitions a running live row to 'orphaned' + audits."""

    @pytest.mark.asyncio
    async def test_running_live_run_becomes_orphaned(self) -> None:
        run = _make_orphan(run_mode="live", status="running")
        write_session = _make_session_for_write(run)
        factory = _build_factory(write_session)
        log = MagicMock()

        with patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit:
            transitioned = await _orphan_live_run(factory, run.id, log)

        assert transitioned is True
        assert run.status == "orphaned"
        write_session.commit.assert_awaited_once()
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["event_type"] == "run_orphaned"

    @pytest.mark.asyncio
    async def test_already_orphaned_run_is_left_alone(self) -> None:
        run = _make_orphan(run_mode="live", status="orphaned")
        write_session = _make_session_for_write(run)
        factory = _build_factory(write_session)
        log = MagicMock()

        with patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit:
            transitioned = await _orphan_live_run(factory, run.id, log)

        assert transitioned is False
        assert run.status == "orphaned"
        write_session.commit.assert_not_awaited()
        audit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resuming_live_run_becomes_orphaned(self) -> None:
        """WP1.8a-round2 (S-04/C-06): a hard kill mid-resume leaves the row
        'resuming' (it never reached the final CAS to 'running'). Boot
        recovery must treat that exactly like a plain 'running' row -- no
        task, no exchange session survives a process restart either way --
        and the audit payload must record the pre-transition status so an
        operator can tell a mid-resume kill apart from a plain hard-kill.
        """
        run = _make_orphan(run_mode="live", status="resuming")
        write_session = _make_session_for_write(run)
        factory = _build_factory(write_session)
        log = MagicMock()

        with patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit:
            transitioned = await _orphan_live_run(factory, run.id, log)

        assert transitioned is True
        assert run.status == "orphaned"
        write_session.commit.assert_awaited_once()
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["event_type"] == "run_orphaned"
        assert audit.call_args.kwargs["payload"]["previous_status"] == "resuming"


# ---------------------------------------------------------------------------
# TestRecoverNoOrphans
# ---------------------------------------------------------------------------


class TestRecoverNoOrphans:
    @pytest.mark.asyncio
    async def test_empty_db_returns_zero(self) -> None:
        select_session = _make_session_for_select([])
        factory = _build_factory(select_session)

        with patch("api.db.session.get_session_factory", return_value=factory):
            result = await recover_orphaned_runs()

        assert result == 0

    @pytest.mark.asyncio
    async def test_empty_db_starts_no_tasks(self) -> None:
        select_session = _make_session_for_select([])
        factory = _build_factory(select_session)

        with patch("api.db.session.get_session_factory", return_value=factory):
            await recover_orphaned_runs()

        assert runs_module._RUN_TASKS == {}


# ---------------------------------------------------------------------------
# TestRecoverLiveOrphan (WP1.8a: orphan in place, never start a task)
# ---------------------------------------------------------------------------


class TestRecoverLiveOrphan:
    @pytest.mark.asyncio
    async def test_live_running_orphan_is_transitioned_with_no_task(self) -> None:
        orphan = _make_orphan(run_mode="live", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit,
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        # Live orphaning does not count toward the (paper-only) return value.
        assert result == 0
        assert orphan.status == "orphaned"
        assert runs_module._RUN_TASKS == {}, "O1: a live run never gets a task at boot"
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["event_type"] == "run_orphaned"

    @pytest.mark.asyncio
    async def test_live_already_orphaned_is_left_alone_no_duplicate_audit(self) -> None:
        orphan = _make_orphan(run_mode="live", status="orphaned")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit,
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "orphaned"
        assert runs_module._RUN_TASKS == {}
        audit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_resuming_orphan_is_transitioned(self) -> None:
        """WP1.8a-round2 (S-04/C-06): boot's candidate query must include
        'resuming' (not just 'running'/'orphaned') so a hard kill mid-resume
        is picked back up, not left invisibly stuck forever."""
        orphan = _make_orphan(run_mode="live", status="resuming")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.services.audit_log.record_audit_event", AsyncMock()) as audit,
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "orphaned"
        assert runs_module._RUN_TASKS == {}
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["payload"]["previous_status"] == "resuming"

    @pytest.mark.asyncio
    async def test_live_orphan_never_gate_checked_or_strategy_validated(self) -> None:
        """Boot never starts a live task, so it never needs the safety gate or
        strategy-availability lockdown any more -- ANY live candidate, even one
        with an unregistered/demoted strategy, is simply orphaned."""
        orphan = _make_orphan(
            run_mode="live",
            status="running",
            config={
                "strategy_name": "totally_unregistered_strategy",
                "symbols": [],
                "timeframe": "not-a-timeframe",
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.services.audit_log.record_audit_event", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "orphaned"
        assert runs_module._RUN_TASKS == {}


# ---------------------------------------------------------------------------
# TestRecoverPaperOrphan (WP1.8a: rebuild in place, same run_id)
# ---------------------------------------------------------------------------


class TestRecoverPaperOrphan:
    @pytest.mark.asyncio
    async def test_paper_orphan_resumes_under_the_same_run_id(self) -> None:
        orphan = _make_orphan(run_mode="paper", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()) as run_paper,
        ):
            result = await recover_orphaned_runs()

        assert result == 1
        assert orphan.status == "running", "the SAME row resumes -- no new run is created"
        assert str(orphan.id) in runs_module._RUN_TASKS
        assert run_paper.call_args.kwargs["run_id_str"] == str(orphan.id)
        assert run_paper.call_args.kwargs["resume"] is _EMPTY_SNAPSHOT

    @pytest.mark.asyncio
    async def test_paper_orphan_config_resume_count_increments(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="orphaned",
            config={
                "strategy_name": "grid_trading",
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "resume_count": 1,
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1
        assert orphan.config["resume_count"] == 2

    @pytest.mark.asyncio
    async def test_paper_orphan_resume_count_exceeded_marks_error(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "grid_trading",
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "resume_count": 3,  # next attempt is 4 -> exceeds the budget
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"
        assert runs_module._RUN_TASKS == {}

    @pytest.mark.asyncio
    async def test_paper_fill_mismatch_marks_error(self) -> None:
        orphan = _make_orphan(run_mode="paper", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_mismatched_snapshot()),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"
        assert runs_module._RUN_TASKS == {}

    @pytest.mark.asyncio
    async def test_paper_snapshot_load_exception_marks_error(self) -> None:
        """WP1.8a-round2 (S-06/C-04): a snapshot-load failure (e.g. a
        persisted row that fails to round-trip back through the pure
        Order/Fill Pydantic models) must fail this candidate closed with
        reason='snapshot_load_failed', not propagate up into the outer
        per-candidate except-Exception (which would only log it and leave
        the row stuck)."""
        orphan = _make_orphan(run_mode="paper", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(side_effect=ValueError("corrupt row")),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"
        assert runs_module._RUN_TASKS == {}

    @pytest.mark.asyncio
    async def test_paper_orphan_task_registered_in_run_tasks(self) -> None:
        orphan = _make_orphan(run_mode="paper", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            await recover_orphaned_runs()

        assert len(runs_module._RUN_TASKS) == 1
        task = next(iter(runs_module._RUN_TASKS.values()))
        assert isinstance(task, asyncio.Task)


# ---------------------------------------------------------------------------
# TestRecoverValidationSkips
# ---------------------------------------------------------------------------


class TestRecoverValidationSkips:
    @pytest.mark.asyncio
    async def test_missing_strategy_name_marks_error_and_skips(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={"symbols": ["BTC/USD"], "timeframe": "1h", "initial_capital": "10000"},
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_unknown_strategy_marks_error_and_skips(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "nonexistent_strategy",
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_empty_symbols_marks_error_and_skips(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "grid_trading",
                "symbols": [],
                "timeframe": "1h",
                "initial_capital": "10000",
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_invalid_timeframe_marks_error_and_skips(self) -> None:
        orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "grid_trading",
                "symbols": ["BTC/USD"],
                "timeframe": "3h",
                "initial_capital": "10000",
            },
        )
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"


# ---------------------------------------------------------------------------
# TestRecoverPerOrphanIsolation
# ---------------------------------------------------------------------------


class TestRecoverPerOrphanIsolation:
    @pytest.mark.asyncio
    async def test_valid_paper_orphan_recovers_even_when_neighbour_fails(self) -> None:
        bad_orphan = _make_orphan(
            run_mode="paper",
            status="running",
            config={"strategy_name": "does_not_exist", "symbols": ["BTC/USD"], "timeframe": "1h"},
        )
        good_orphan = _make_orphan(run_mode="paper", status="running")

        select_session = _make_session_for_select([bad_orphan, good_orphan])
        bad_write = _make_session_for_write(bad_orphan)
        good_write = _make_session_for_write(good_orphan)
        factory = _build_factory(select_session, bad_write, good_write, good_write)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1
        assert bad_orphan.status == "error"
        assert good_orphan.status == "running"
        assert len(runs_module._RUN_TASKS) == 1

    @pytest.mark.asyncio
    async def test_live_and_paper_candidates_both_processed_independently(self) -> None:
        live_orphan = _make_orphan(run_mode="live", status="running")
        paper_orphan = _make_orphan(run_mode="paper", status="running")

        select_session = _make_session_for_select([live_orphan, paper_orphan])
        live_write = _make_session_for_write(live_orphan)
        paper_write = _make_session_for_write(paper_orphan)
        factory = _build_factory(select_session, live_write, paper_write, paper_write)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.services.audit_log.record_audit_event", AsyncMock()),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1  # only the paper resume counts
        assert live_orphan.status == "orphaned"
        assert paper_orphan.status == "running"
        assert len(runs_module._RUN_TASKS) == 1
        assert str(paper_orphan.id) in runs_module._RUN_TASKS


# ---------------------------------------------------------------------------
# TestRecoverStrategyAvailabilityLockdown (paper-only at boot -- live never
# reaches this check any more, see TestRecoverLiveOrphan above)
# ---------------------------------------------------------------------------


class TestRecoverStrategyAvailabilityLockdown:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["ma_crossover", "breakout", "model_strategy"])
    async def test_demoted_paper_orphan_marked_error_and_not_restarted(self, strategy: str) -> None:
        orphan = _make_orphan(run_mode="paper", status="running", strategy_name=strategy)
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch(
                "api.routers.runs._get_strategy_registry",
                return_value={strategy: MagicMock()},
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_active_paper_orphan_still_recovers_under_lockdown(self) -> None:
        orphan = _make_orphan(run_mode="paper", status="running", strategy_name="grid_trading")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            _patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=_EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1
        assert len(runs_module._RUN_TASKS) == 1
