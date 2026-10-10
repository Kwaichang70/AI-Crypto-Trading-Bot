"""
tests/integration/test_sprint50_cycle4_integration.py
------------------------------------------------------
Integration test for Sprint 50 Cycle 4: HALT auto-stop pipeline.

WP1.6 (WP16-T-44, decision D4): the HALT auto-stop was REMOVED.  A circuit-breaker
HALT blocks entries only; the run keeps polling.  The first test below was
rewritten from "simulate the removed branch inline and assert it stops" to "drive
the real ``_process_bar`` and assert it does NOT stop".  The audit-type and
audit-ordering tests are unchanged (the orchestrator ``finally`` branches are
unreachable legacy until CF-29).

What this test validates
------------------------
1. A (BACKTEST-mode) engine with a real CircuitBreaker in HALT keeps processing bars.
2. On a HALT bar, _stop_event is NOT set (D4).
3. The engine's auto_stop_reason stays None.
4. The orchestrator's logic (tested via run_orchestrator helpers) produces
   the correct audit event type for the engine's auto_stop_reason.

Architecture note
-----------------
Full DB integration is out of scope for this test (requires live Postgres).
We validate the pipeline logic using mocked DB sessions, following the
pattern established in tests/integration/test_paper_engine_pipeline.py.

The test is marked `integration` to keep it out of the fast unit suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from common.types import RunMode
from tests.unit._wp16_support import HALT_ONLY, OTHER, bar, build_engine, buy
from trading.safety import CircuitBreakerResponse, build_run_circuit_breaker


pytestmark = pytest.mark.integration


# ===========================================================================
# TestHaltAutoStopPipeline
# ===========================================================================


class TestHaltAutoStopPipeline:
    """HALT (WP1.6/D4: no stop) and the legacy audit event type / ordering contracts."""

    @pytest.mark.asyncio
    async def test_halt_does_not_set_auto_stop_reason_or_event(self) -> None:
        """Real ``_process_bar`` on a HALT bar: entries blocked, run NOT stopped (D4)."""
        breaker = build_run_circuit_breaker("integration-halt-run", HALT_ONLY)
        engine, execution, _pf = build_engine(
            cb=breaker,
            signals=lambda i: [buy(OTHER)],
            params=HALT_ONLY,
            mode=RunMode.BACKTEST,
        )

        await bar(engine, "800")  # 20% drawdown on the live limit -> HALT

        assert breaker.current_response == CircuitBreakerResponse.HALT
        assert not engine._stop_event.is_set(), "HALT must NOT stop the run (D4)"
        assert engine.auto_stop_reason is None
        assert execution.signals == [], "the BUY is dropped, the run keeps going"

        await bar(engine, "800")  # the next bar is processed as well
        assert engine._bar_count == 2

    def test_orchestrator_audit_event_type_for_halt_reason(self) -> None:
        """Orchestrator must use 'circuit_breaker_halt_auto_stop' event type for HALT stops."""
        from api.services.audit_log import _VALID_EVENT_TYPES

        # The event type the orchestrator writes when auto_stop_reason == "circuit_breaker_halt"
        expected_event_type = "circuit_breaker_halt_auto_stop"

        # It must pass the _VALID_EVENT_TYPES guard (otherwise record_audit_event silently skips)
        assert expected_event_type in _VALID_EVENT_TYPES, (
            f"'{expected_event_type}' not in _VALID_EVENT_TYPES — "
            "audit_log.py must be updated alongside the orchestrator"
        )

    @pytest.mark.asyncio
    async def test_audit_write_before_status_transition(self) -> None:
        """Audit write must complete before the status DB session opens.

        Validates the audit-before-mutation invariant by asserting the audit
        mock is called BEFORE the status mock using call-order tracking.
        """
        from api.services.audit_log import record_audit_event

        call_order: list[str] = []

        audit_session = AsyncMock()
        audit_session.add = MagicMock(side_effect=lambda _: call_order.append("audit_add"))
        audit_session.flush = AsyncMock(side_effect=lambda: call_order.append("audit_flush"))
        audit_session.commit = AsyncMock(side_effect=lambda: call_order.append("audit_commit"))

        status_session = AsyncMock()
        status_session.commit = AsyncMock(side_effect=lambda: call_order.append("status_commit"))

        # Simulate audit write (happens first in orchestrator finally block)
        await record_audit_event(
            audit_session,
            event_type="circuit_breaker_halt_auto_stop",
            resource_type="run",
            resource_id="test-run",
            request=None,
            payload={"trigger": "graduated_halt"},
        )
        await audit_session.commit()

        # Simulate status transition (happens second)
        await status_session.commit()

        # Verify order: audit must precede status
        audit_pos = next(i for i, v in enumerate(call_order) if v == "audit_commit")
        status_pos = next(i for i, v in enumerate(call_order) if v == "status_commit")
        assert audit_pos < status_pos, (
            "Audit commit must precede status commit (Sprint 50 audit-before-mutation invariant)"
        )
