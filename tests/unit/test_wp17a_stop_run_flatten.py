"""
tests/unit/test_wp17a_stop_run_flatten.py
--------------------------------------------
WP1.7a: unit-level (mocked DB session) tests for the flatten-aware
DELETE /runs/{id}, POST /runs/{id}/emergency-stop, and
POST /runs/{id}/entries-latch/clear endpoints.

Mandatory per reports/vp2-wp1.7/synthesis-spec.md §6 (1.7a API tests):
  - Live DELETE without flatten -> 422; paper -> 200.
  - Incomplete flatten -> 409 and the run stays running.
  - Emergency stop needs no admin key (no require_admin dependency).
  - Per-run clear needs an admin key and writes an audit row (verified at
    the endpoint-function level: clear_entries_latch always writes one).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.db.models import RunORM
from trading.strategy_engine import FlattenResult, FlattenSymbolResult


def _make_request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


def _make_run(
    *,
    run_mode: str = "live",
    status: str = "running",
    entries_latch_reason: str | None = None,
) -> MagicMock:
    # MagicMock(spec=RunORM) instead of a bare MagicMock: from_attributes
    # validation (RunDetailResponse.model_validate) treats an
    # auto-vivified MagicMock sub-attribute (e.g. .confidence_flag, which
    # RunORM does not have) as real data instead of the missing-attribute
    # AttributeError a genuine RunORM instance raises -- spec= restricts
    # attribute access to RunORM's own declared columns.
    run = MagicMock(spec=RunORM)
    run.id = uuid.uuid4()
    run.run_mode = run_mode
    run.status = status
    run.config = {}
    run.entries_latch_reason = entries_latch_reason
    run.entries_latched_at = None
    run.started_at = datetime.now(tz=UTC)
    run.stopped_at = None
    run.created_at = datetime.now(tz=UTC)
    run.updated_at = datetime.now(tz=UTC)
    run.n_closed_trades = None
    run.metrics_v2_backfilled = False
    run.recovered_from_run_id = None
    run.promoted_from_run_id = None
    return run


def _make_engine(*, symbols: list[str], held: dict[str, Decimal]) -> MagicMock:
    engine = MagicMock()
    engine.symbols = symbols
    engine.risk_manager = MagicMock()
    engine.risk_manager.trigger_kill_switch = MagicMock()
    engine.risk_manager.kill_switch_reasons = frozenset()

    def _get_position(symbol: str) -> MagicMock | None:
        qty = held.get(symbol)
        if qty is None:
            return None
        pos = MagicMock()
        pos.quantity = qty
        pos.is_flat = qty <= Decimal("0")
        return pos

    engine.portfolio = MagicMock()
    engine.portfolio.get_position = MagicMock(side_effect=_get_position)
    return engine


def _db_with_single_result(run: MagicMock | None) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = run
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    return db


class TestStopRunFlattenDecision:
    async def test_live_running_without_flatten_returns_422_with_held_symbols(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="live", status="running")
        db = _db_with_single_result(run)
        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})

        with patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}):
            with pytest.raises(HTTPException) as exc_info:
                await stop_run(run.id, db, _make_request(), flatten=None)

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "flatten_decision_required"
        assert exc_info.value.detail["held_symbols"] == ["BTC/EUR"]

    async def test_paper_running_without_flatten_defaults_false_returns_200(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="paper", status="running")
        db = _db_with_single_result(run)

        with patch("api.routers.runs._RUN_TASKS", {}), patch("api.routers.runs._RUN_ENGINES", {}):
            response = await stop_run(run.id, db, _make_request(), flatten=None)

        assert run.status == "stopped"
        assert response.flatten is None


class TestStopRunFlattenIncomplete:
    async def test_incomplete_flatten_returns_409_and_stays_running(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="live", status="running")
        # First execute() call: the initial FOR UPDATE fetch. Second call
        # (after flatten): the re-fetch. Same row both times here.
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        incomplete_result = FlattenResult(
            run_id=str(run.id),
            outcome="partial",
            complete=False,
            symbols=[
                FlattenSymbolResult(
                    symbol="BTC/EUR",
                    status="partial",
                    cause="ledger_doubt",
                    held_before=Decimal("0.01"),
                    sold_qty=Decimal("0"),
                    remaining_qty=Decimal("0.01"),
                )
            ],
        )
        engine.flatten = AsyncMock(return_value=incomplete_result)

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await stop_run(run.id, db, _make_request(), flatten=True)

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "flatten_incomplete"
        # WP1.7a round 2 (S-08): stop_in_progress is latched first, then
        # -- only AFTER the flatten_incomplete latch is durably persisted
        # (commit succeeded) -- replaced by the durable flatten_incomplete
        # reason; reset_kill_switch("stop_in_progress") releases the
        # temporary one.
        trigger_calls = [c.args[0] for c in engine.risk_manager.trigger_kill_switch.call_args_list]
        assert trigger_calls == ["stop_in_progress", "flatten_incomplete"]
        engine.risk_manager.reset_kill_switch.assert_called_once_with("stop_in_progress")
        assert run.status == "running", "an incomplete flatten must never stop the run (I8)"
        assert run.entries_latch_reason == "flatten_incomplete"

    async def test_complete_flatten_stops_the_run(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0")})
        complete_result = FlattenResult(
            run_id=str(run.id),
            outcome="flattened",
            complete=True,
            symbols=[
                FlattenSymbolResult(
                    symbol="BTC/EUR",
                    status="flat",
                    cause=None,
                    held_before=Decimal("0.01"),
                    sold_qty=Decimal("0.01"),
                    remaining_qty=Decimal("0"),
                    order_ids=["abc"],
                )
            ],
        )
        engine.flatten = AsyncMock(return_value=complete_result)
        task = MagicMock()
        task.done.return_value = False

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run.id): task}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            response = await stop_run(run.id, db, _make_request(), flatten=True)

        assert run.status == "stopped"
        assert response.flatten.outcome == "flattened"
        task.cancel.assert_called_once()


class TestEmergencyStopNoAdminKey:
    def test_emergency_stop_route_has_no_require_admin_dependency(self) -> None:
        """SY-07: emergency-stop must stay reachable on X-API-Key alone --
        no require_admin dependency on the route."""
        from api.routers import runs as runs_module

        route = next(
            r for r in runs_module.router.routes
            if getattr(r, "path", "").endswith("/emergency-stop")
        )
        dependant_calls = [
            dep.call for dep in route.dependant.dependencies
        ]
        assert runs_module.require_admin not in dependant_calls

    async def test_emergency_stop_always_stops_even_flatten_incomplete(self) -> None:
        from api.routers.runs import emergency_stop_run

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        incomplete_result = FlattenResult(
            run_id=str(run.id),
            outcome="partial",
            complete=False,
            symbols=[
                FlattenSymbolResult(
                    symbol="BTC/EUR",
                    status="partial",
                    cause="ledger_doubt",
                    held_before=Decimal("0.01"),
                    sold_qty=Decimal("0"),
                    remaining_qty=Decimal("0.01"),
                )
            ],
        )
        engine.flatten = AsyncMock(return_value=incomplete_result)
        task = MagicMock()
        task.done.return_value = False

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run.id): task}),
            patch("api.routers.runs._LEARNING_INSTANCES", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            response = await emergency_stop_run(
                run.id, _make_request(), db, reason="incident", flatten=True
            )

        assert run.status == "stopped", "emergency-stop must ALWAYS stop (SY-07)"
        task.cancel.assert_called_once()
        assert response.flatten.complete is False
        assert len(response.unprotected_positions) == 1
        assert response.unprotected_positions[0].symbol == "BTC/EUR"


class TestClearEntriesLatch:
    async def test_not_latched_returns_409(self) -> None:
        from api.routers.runs import _ClearEntriesLatchRequest, clear_entries_latch

        run = _make_run(run_mode="paper", status="running", entries_latch_reason=None)
        db = _db_with_single_result(run)

        with pytest.raises(HTTPException) as exc_info:
            await clear_entries_latch(
                run.id,
                _ClearEntriesLatchRequest(reason="operator confirmed safe"),
                db,
                _make_request(),
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "not_latched"

    async def test_clears_latch_and_writes_audit_row(self) -> None:
        from api.routers.runs import _ClearEntriesLatchRequest, clear_entries_latch

        run = _make_run(
            run_mode="paper", status="running", entries_latch_reason="flatten_incomplete"
        )
        db = _db_with_single_result(run)
        audit_mock = AsyncMock()

        with (
            patch("api.services.audit_log.record_audit_event_strict", audit_mock),
            patch("api.routers.runs._RUN_ENGINES", {}),
        ):
            result = await clear_entries_latch(
                run.id,
                _ClearEntriesLatchRequest(reason="operator confirmed safe"),
                db,
                _make_request(),
            )

        assert run.entries_latch_reason is None
        assert result.cleared == "flatten_incomplete"
        audit_mock.assert_awaited_once()
        assert audit_mock.await_args.kwargs["event_type"] == "entries_latch_cleared"

    async def test_live_run_requires_confirm_token(self) -> None:
        from api.routers.runs import _ClearEntriesLatchRequest, clear_entries_latch

        run = _make_run(
            run_mode="live", status="running", entries_latch_reason="flatten_incomplete"
        )
        db = _db_with_single_result(run)

        with pytest.raises(HTTPException) as exc_info:
            await clear_entries_latch(
                run.id,
                _ClearEntriesLatchRequest(reason="operator confirmed safe"),
                db,
                _make_request(),
                x_live_confirm_token=None,
            )
        assert exc_info.value.status_code == 422
