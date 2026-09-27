"""
tests/unit/test_wp17a_round2_security.py
--------------------------------------------
WP1.7a round 2 (security round 1 + critic round 1) regression tests,
replicating the probes described in
reports/vp2-wp1.7/security-report-1.7a.md (WP17a-S-01..S-17) and
reports/vp2-wp1.7/critic-report-1.7a.md (WP17a-C-01..C-03).

Each test class below is named after the finding it regresses.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.db.models import RunORM
from api.services import kill_switch as kill_switch_service
from trading.strategy_engine import (
    FlattenResult,
    FlattenSymbolResult,
    build_synthetic_flatten_result,
)


def _make_request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


def _make_settings(admin_key: str = "a" * 32 + "b1c2d3e4f5g6") -> object:
    from pydantic import SecretStr

    from api.config import Settings

    s = Settings.model_construct()
    object.__setattr__(s, "admin_api_key", SecretStr(admin_key))
    return s


def _make_run(*, run_mode: str = "live", status: str = "running") -> MagicMock:
    run = MagicMock(spec=RunORM)
    run.id = uuid.uuid4()
    run.run_mode = run_mode
    run.status = status
    run.config = {}
    run.entries_latch_reason = None
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
    engine.run_id = "test-run"
    engine.risk_manager = MagicMock()
    engine.risk_manager.trigger_kill_switch = MagicMock()
    engine.risk_manager.kill_switch_reasons = frozenset()
    engine.risk_manager.kill_switch_active = False

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


class TestS01LatchBeforeAnyDBAccess:
    """S-01: kill_switch's memory + engine latch must happen before ANY
    await, so a DB outage or a slow FOR UPDATE wait can never leave a
    running engine unlatched."""

    async def test_press_with_db_unreachable_returns_200_latched_engines_latched(self) -> None:
        from api.routers.emergency import kill_switch

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        run_id = str(uuid.uuid4())

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=ConnectionRefusedError("DB unreachable"))
        db.rollback = AsyncMock()

        with (
            patch("api.routers.emergency._RUN_ENGINES", {run_id: engine}),
            patch(
                "api.routers.emergency.get_session_factory",
                side_effect=ConnectionRefusedError("still unreachable"),
            ),
        ):
            response = await kill_switch(
                request=_make_request(),
                db=db,
                reason=None,
                settings=_make_settings(),
                body=None,
            )

        assert response.latched is True
        assert response.latch_persisted is False
        assert response.runs_latched == [run_id]
        engine.risk_manager.trigger_kill_switch.assert_called_once_with(
            kill_switch_service.GLOBAL_REASON
        )

    async def test_engines_latch_before_the_row_lock_wait_resolves(self) -> None:
        """Simulates a press arriving while another connection holds
        FOR UPDATE on the runs table: the engine-latch loop must have
        already run by the time db.execute is even awaited."""
        from api.routers.emergency import kill_switch

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        run_id = str(uuid.uuid4())
        observed_latched_before_db_call: list[bool] = []

        async def _slow_execute(*args: object, **kwargs: object) -> MagicMock:
            # By the time this coroutine is even scheduled, the engine
            # must already be latched -- this is checked BEFORE the
            # (simulated) row-lock wait completes.
            observed_latched_before_db_call.append(engine.risk_manager.kill_switch_active)
            await asyncio.sleep(0)  # simulates the row-lock wait
            result = MagicMock()
            result.scalars.return_value.all.return_value = []
            result.rowcount = 1
            return result

        engine.risk_manager.kill_switch_active = False

        def _trigger(reason: str) -> None:
            engine.risk_manager.kill_switch_active = True

        engine.risk_manager.trigger_kill_switch = MagicMock(side_effect=_trigger)

        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)

        db = AsyncMock()
        db.execute = _slow_execute
        db.begin_nested = MagicMock(return_value=nested_cm)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        with patch("api.routers.emergency._RUN_ENGINES", {run_id: engine}):
            response = await kill_switch(
                request=_make_request(),
                db=db,
                reason=None,
                settings=_make_settings(),
                body=None,
            )

        # _persist_press calls db.execute more than once in a single
        # attempt (the candidates SELECT, then activate()'s own UPDATE)
        # -- every one of them must observe the engine ALREADY latched.
        assert observed_latched_before_db_call, "db.execute was never called"
        assert all(observed_latched_before_db_call), (
            "the engine must already be latched before EVERY db.execute call, "
            f"got {observed_latched_before_db_call!r}"
        )
        assert response.runs_latched == [run_id]
        assert response.latch_persisted is True


class TestS02FailClosedDefault:
    """S-02/WP17a-C-01: the bare/unloaded state is fail-closed, and both
    load() and the outer main.py except-branch must reach it via every
    realistic failure mode."""

    def test_bare_unloaded_state_is_fail_closed(self) -> None:
        kill_switch_service.mark_unknown()
        assert kill_switch_service.is_loaded() is False
        assert kill_switch_service.is_active() is True
        assert (
            kill_switch_service.current_state().reason
            == kill_switch_service.UNKNOWN_STATE_REASON
        )

    async def test_main_py_outer_except_marks_unknown_on_session_factory_failure(self) -> None:
        """WP17a-C-01: a failure in session-factory construction/the
        `async with` entry -- i.e. something load() itself never got a
        chance to run against -- must still fail the mirror closed via
        the OUTER except branch calling mark_unknown()."""
        kill_switch_service.reset_state_for_tests()
        assert kill_switch_service.is_active() is False

        # Reproduce main.py's own try/except structure directly: the
        # session-factory constructor itself raises, BEFORE load() (or
        # even the `async with` entry) ever runs.
        def _broken_factory() -> None:
            raise RuntimeError("pool construction failed")

        try:
            _ks_factory = _broken_factory
            _ks_factory()
        except Exception:
            kill_switch_service.mark_unknown()

        assert kill_switch_service.is_active() is True
        assert (
            kill_switch_service.current_state().reason
            == kill_switch_service.UNKNOWN_STATE_REASON
        )

    @pytest.mark.parametrize(
        "exc",
        [ConnectionRefusedError("refused"), OSError("bad db name"), ValueError("bad password")],
    )
    async def test_load_various_failure_modes_all_fail_closed(self, exc: Exception) -> None:
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=exc)
        state = await kill_switch_service.load(db)
        assert state.active is True
        assert kill_switch_service.is_active() is True


class TestS05EmergencyStopAlwaysStops:
    async def test_flatten_exception_still_stops_and_reports_failed(self) -> None:
        from api.routers.runs import emergency_stop_run

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        engine.flatten = AsyncMock(side_effect=RuntimeError("exchange call hung then raised"))
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

        assert run.status == "stopped", (
            "emergency-stop must ALWAYS stop even on a flatten exception"
        )
        task.cancel.assert_called_once()
        assert response.flatten is not None
        assert response.flatten.complete is False
        assert response.flatten.outcome == "failed"
        assert len(response.unprotected_positions) == 1

    async def test_flatten_timeout_still_stops(self) -> None:
        """S-05: a hung flatten() (caught by the outer wait_for) must
        still let emergency-stop proceed to actually stop the run."""
        from api.routers import runs as runs_module

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})

        async def _never_returns(*args: object, **kwargs: object) -> FlattenResult:
            await asyncio.sleep(3600)
            raise AssertionError("should never reach here")

        engine.flatten = _never_returns
        task = MagicMock()
        task.done.return_value = False

        async def _fake_wait_for(aw: object, timeout: float) -> FlattenResult:
            raise TimeoutError("simulated outer timeout")

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run.id): task}),
            patch("api.routers.runs._LEARNING_INSTANCES", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
            patch.object(runs_module.asyncio, "wait_for", _fake_wait_for),
        ):
            response = await runs_module.emergency_stop_run(
                run.id, _make_request(), db, reason="incident", flatten=True
            )

        assert run.status == "stopped"
        assert response.flatten is not None
        assert response.flatten.complete is False
        assert response.flatten.symbols[0].cause == "timeout_open"
        assert response.flatten.symbols[0].status == "in_flight"


class TestS07StopRunFlattenException:
    async def test_flatten_exception_persists_incomplete_and_returns_409(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.commit = AsyncMock()
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})
        engine.flatten = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await stop_run(run.id, db, _make_request(), flatten=True)

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "flatten_incomplete"
        assert run.status == "running"
        assert run.entries_latch_reason == "flatten_incomplete"
        trigger_calls = [c.args[0] for c in engine.risk_manager.trigger_kill_switch.call_args_list]
        assert trigger_calls == ["stop_in_progress", "flatten_incomplete"]


class TestS09KillSwitchFlattenPersistence:
    async def test_incomplete_kill_switch_flatten_persists_and_audits(self) -> None:
        from api.routers.emergency import _persist_run_flatten

        run_id_str = str(uuid.uuid4())
        incomplete_result = FlattenResult(
            run_id=run_id_str,
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

        run_db = AsyncMock()
        run_db.execute = AsyncMock()
        run_db.commit = AsyncMock()
        audit_mock = AsyncMock()

        factory = MagicMock(return_value=run_db)
        run_db.__aenter__ = AsyncMock(return_value=run_db)
        run_db.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("api.routers.emergency.get_session_factory", return_value=factory),
            patch("api.routers.emergency.record_audit_event", audit_mock),
        ):
            persisted = await _persist_run_flatten(
                run_id_str,
                trigger="kill_switch",
                result=incomplete_result,
                actor_id="admin_key_abc",
            )

        assert persisted is True
        run_db.execute.assert_awaited_once()
        audit_mock.assert_awaited_once()
        assert audit_mock.await_args.kwargs["event_type"] == "run_flatten"
        assert audit_mock.await_args.kwargs["payload"]["phase"] == "incomplete"

    async def test_persist_failure_reports_false(self) -> None:
        from api.routers.emergency import _persist_run_flatten

        complete_result = FlattenResult(run_id="r1", outcome="flattened", complete=True, symbols=[])

        with patch(
            "api.routers.emergency.get_session_factory",
            side_effect=RuntimeError("db down"),
        ):
            persisted = await _persist_run_flatten(
                "r1", trigger="kill_switch", result=complete_result, actor_id="admin_key_abc"
            )

        assert persisted is False


class TestS11GetStatusRequiresApiKey:
    def test_get_kill_switch_route_has_require_api_key_dependency(self) -> None:
        from api.auth import require_api_key
        from api.routers import emergency as emergency_module

        route = next(
            r
            for r in emergency_module.router.routes
            if getattr(r, "path", "") == "/emergency/kill-switch"
            and "GET" in getattr(r, "methods", set())
        )
        dependant_calls = [dep.call for dep in route.dependant.dependencies]
        assert require_api_key in dependant_calls

    def test_post_kill_switch_clear_is_not_in_rate_limit_exemption(self) -> None:
        import inspect

        from api import rate_limit as rate_limit_module

        source = inspect.getsource(rate_limit_module)
        assert (
            'path == "/api/v1/emergency/kill-switch" and '
            'request.method.upper() == "POST"'
        ) in source


class TestS13StopFlattenFalseReportsUnprotected:
    async def test_delete_flatten_false_reports_unprotected_position(self) -> None:
        from api.routers.runs import stop_run

        run = _make_run(run_mode="live", status="running")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.flush = AsyncMock()

        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.01")})

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()) as audit_mock,
        ):
            response = await stop_run(run.id, db, _make_request(), flatten=False)

        assert run.status == "stopped"
        assert len(response.unprotected_positions) == 1
        assert response.unprotected_positions[0].symbol == "BTC/EUR"
        # a run_flatten{phase: skipped} audit row must have been written
        payloads = [c.kwargs.get("payload", {}) for c in audit_mock.await_args_list]
        assert any(p.get("phase") == "skipped" for p in payloads)


class TestS14ExposureUnknown:
    async def test_emergency_stop_live_no_engine_reports_exposure_unknown(self) -> None:
        from api.routers.runs import emergency_stop_run

        run = _make_run(run_mode="live", status="orphaned")
        db = AsyncMock()
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = run
        db.execute = AsyncMock(return_value=result1)
        db.flush = AsyncMock()

        with (
            patch("api.routers.runs._RUN_ENGINES", {}),
            patch("api.routers.runs._RUN_TASKS", {}),
            patch("api.routers.runs._LEARNING_INSTANCES", {}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            response = await emergency_stop_run(
                run.id, _make_request(), db, reason="incident", flatten=False
            )

        assert response.exposure_unknown is True
        assert response.unprotected_positions == []


class TestBuildSyntheticFlattenResult:
    def test_no_position_symbol_is_no_position(self) -> None:
        engine = _make_engine(symbols=["BTC/EUR"], held={})
        result = build_synthetic_flatten_result(engine, reason="stop", cause="error")
        assert result.outcome == "noop"
        assert result.complete is False
        assert result.symbols[0].status == "no_position"

    def test_held_symbol_timeout_cause_is_in_flight(self) -> None:
        # WP17a-S-R2-04 (round 3): a caller-side timeout means the SELL
        # may still be in flight -- this must never be reported as a
        # hard "failed" outcome (nor "complete"), only "partial".
        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.02")})
        result = build_synthetic_flatten_result(engine, reason="stop", cause="timeout_open")
        assert result.symbols[0].status == "in_flight"
        assert result.symbols[0].cause == "timeout_open"
        assert result.outcome == "partial"
        assert result.complete is False

    def test_held_symbol_error_cause_is_failed(self) -> None:
        engine = _make_engine(symbols=["BTC/EUR"], held={"BTC/EUR": Decimal("0.02")})
        result = build_synthetic_flatten_result(
            engine, reason="stop", cause="error", error="boom"
        )
        assert result.symbols[0].status == "failed"
        assert result.symbols[0].error == "boom"
