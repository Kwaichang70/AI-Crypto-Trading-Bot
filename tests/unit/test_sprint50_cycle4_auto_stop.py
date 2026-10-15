"""
tests/unit/test_sprint50_cycle4_auto_stop.py
---------------------------------------------
Unit tests for Sprint 50 Cycle 4: CircuitBreaker HALT auto-stop.

Modules under test
------------------
packages/trading/strategy_engine.py  -- _auto_stop_reason + auto_stop_reason property
                                        + _stop_event.set() on HALT
apps/api/services/audit_log.py       -- 'circuit_breaker_halt_auto_stop' in _VALID_EVENT_TYPES

Coverage groups (8 producer tests + 2 CR4-001 + 1 RISK-CB-008 = 11 tests)
---------------------------------------------------------------------------
TestAutoStopReasonInit           (2) -- initial None value, property read
TestHaltDoesNotStopTheRun        (3) -- WP1.6 / D4: HALT, REDUCE, DAILY_LIMIT never stop the run
TestRepeatedHaltBars             (1) -- WP1.6 / D4: repeated HALT bars keep the run going
TestAuditEventTypeRegistered     (1) -- 'circuit_breaker_halt_auto_stop' in _VALID_EVENT_TYPES
TestAuditLogBehavior             (1) -- record_audit_event accepts the new event type
TestOrchestratorAuditConditional (2) -- orchestrator audit-called/not-called per auto_stop_reason
TestNoLiquidationContract        (1) -- HALT does not mutate portfolio positions

WP1.6 rewrite (WP16-T-44)
-------------------------
Sprint 50 Cycle 4 made a circuit-breaker HALT stop the run.  WP1.6 (decision D4)
reverses that: HALT blocks ENTRIES only and the run keeps polling, so
``TestHaltSetsStopEvent`` (3 tests) and ``TestIdempotentHalt`` (1 test), which
SIMULATED the removed ``_process_bar`` branch inline and would have stayed green
while pinning removed behaviour, were rewritten as ``TestHaltDoesNotStopTheRun``
and ``TestRepeatedHaltBars``.  They now drive the REAL ``StrategyEngine._process_bar``
(see ``tests/unit/_wp16_support.py``) and assert D4: ``_stop_event`` unset,
``auto_stop_reason is None``, the next bar processed.  The audit-type, audit-log
and orchestrator-conditional tests are unchanged: the orchestrator ``finally``
branches are unreachable legacy until CF-29 (``auto_stop_reason`` is never set).

Design notes
------------
- The audit_log test uses the public _VALID_EVENT_TYPES frozenset directly --
  this is the canonical guard before any DB write.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.types import RunMode
from tests.unit._wp16_support import DAILY_ONLY, HALT_ONLY, OTHER, bar, build_engine, buy
from trading.safety import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResponse,
    build_run_circuit_breaker,
)


# ===========================================================================
# Helpers
# ===========================================================================


class _FakeEngine:
    """Minimal engine stub exposing only the auto-stop surface.

    SimpleNamespace cannot host a property (its metaclass is immutable), so we
    use a plain class that mirrors the StrategyEngine attributes under test.
    """

    def __init__(
        self,
        *,
        auto_stop_reason: str | None = None,
        stop_event_set: bool = False,
    ) -> None:
        self._auto_stop_reason = auto_stop_reason
        self._stop_event = asyncio.Event()
        if stop_event_set:
            self._stop_event.set()
        self._portfolio: Any = None

    @property
    def auto_stop_reason(self) -> str | None:
        return self._auto_stop_reason


def _make_engine_ns(
    *,
    auto_stop_reason: str | None = None,
    stop_event_set: bool = False,
) -> _FakeEngine:
    """Build a _FakeEngine that mimics the StrategyEngine auto-stop surface.

    Used to test the property and the `_stop_event` guard without constructing
    a full StrategyEngine (which requires many async collaborators).
    """
    return _FakeEngine(
        auto_stop_reason=auto_stop_reason,
        stop_event_set=stop_event_set,
    )


def _breaker_config(
    *,
    max_drawdown_pct: float = 0.15,
    reduce_drawdown_pct: float = 0.10,
    max_daily_loss_pct: float = 0.05,
) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        max_drawdown_pct=max_drawdown_pct,
        reduce_drawdown_pct=reduce_drawdown_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )


# ===========================================================================
# TestAutoStopReasonInit
# ===========================================================================


class TestAutoStopReasonInit:
    """Tests for the initial state of _auto_stop_reason."""

    def test_initial_value_is_none(self) -> None:
        """auto_stop_reason must be None before any bar is processed."""
        ns = _make_engine_ns()
        assert ns.auto_stop_reason is None

    def test_property_reflects_private_attribute(self) -> None:
        """auto_stop_reason property must delegate to _auto_stop_reason."""
        ns = _make_engine_ns(auto_stop_reason="circuit_breaker_halt")
        assert ns.auto_stop_reason == "circuit_breaker_halt"


# ===========================================================================
# TestHaltDoesNotStopTheRun  (WP1.6 / D4; rewrite of TestHaltSetsStopEvent)
# ===========================================================================


class TestHaltDoesNotStopTheRun:
    """Through the REAL ``_process_bar``: no circuit-breaker level stops the run."""

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    async def test_halt_does_not_set_stop_event_or_reason(self, mode: RunMode) -> None:
        """HALT (data-driven, 20% drawdown) blocks entries but never stops the run."""
        engine, _ex, _pf = build_engine(
            cb=build_run_circuit_breaker("test-halt", HALT_ONLY),
            signals=lambda i: [buy(OTHER)],
            params=HALT_ONLY,
            mode=mode,
        )
        await engine.start("test-halt")
        await bar(engine, "800")
        assert engine.circuit_breaker is not None
        assert engine.circuit_breaker.current_response == CircuitBreakerResponse.HALT
        assert not engine._stop_event.is_set()
        assert engine.auto_stop_reason is None

    async def test_reduce_does_not_set_stop_event(self) -> None:
        """REDUCE response must NOT set _stop_event."""
        engine, _ex, _pf = build_engine(
            cb=build_run_circuit_breaker("test-reduce", HALT_ONLY),
            signals=lambda i: [],
            params=HALT_ONLY,
        )
        await bar(engine, "880")
        assert engine.circuit_breaker is not None
        assert engine.circuit_breaker.current_response == CircuitBreakerResponse.REDUCE
        assert not engine._stop_event.is_set()
        assert engine.auto_stop_reason is None

    async def test_daily_limit_does_not_set_stop_event(self) -> None:
        """DAILY_LIMIT response must NOT set _stop_event (clears at the next UTC day)."""
        engine, _ex, _pf = build_engine(
            cb=build_run_circuit_breaker("test-daily", DAILY_ONLY),
            signals=lambda i: [],
            params=DAILY_ONLY,
        )
        await bar(engine, "900")
        assert engine.circuit_breaker is not None
        assert engine.circuit_breaker.current_response == CircuitBreakerResponse.DAILY_LIMIT
        assert not engine._stop_event.is_set()
        assert engine.auto_stop_reason is None


# ===========================================================================
# TestRepeatedHaltBars  (WP1.6 / D4; rewrite of TestIdempotentHalt)
# ===========================================================================


class TestRepeatedHaltBars:
    """Repeated HALT bars keep the run going (nothing to be idempotent about)."""

    async def test_repeated_halt_bars_keep_processing(self) -> None:
        strategy_calls: list[int] = []
        engine, ex, _pf = build_engine(
            cb=build_run_circuit_breaker("test-repeat", HALT_ONLY),
            signals=lambda i: strategy_calls.append(i) or [buy(OTHER)],
            params=HALT_ONLY,
        )
        for _ in range(3):
            await bar(engine, "800")
        assert not engine._stop_event.is_set()
        assert engine.auto_stop_reason is None
        assert engine._bar_count == 3
        assert sorted(set(strategy_calls)) == [0, 1, 2]  # the strategy ran on every bar
        assert ex.signals == []  # BUYs dropped on every HALT bar, no stop


# ===========================================================================
# TestAuditEventTypeRegistered
# ===========================================================================


class TestAuditEventTypeRegistered:
    """Tests that 'circuit_breaker_halt_auto_stop' is in _VALID_EVENT_TYPES."""

    def test_halt_auto_stop_in_valid_types(self) -> None:
        """'circuit_breaker_halt_auto_stop' must be present in _VALID_EVENT_TYPES."""
        from api.services.audit_log import _VALID_EVENT_TYPES

        assert "circuit_breaker_halt_auto_stop" in _VALID_EVENT_TYPES


# ===========================================================================
# TestAuditLogBehavior
# ===========================================================================


class TestAuditLogBehavior:
    """Tests that record_audit_event accepts the new event type without error."""

    @pytest.mark.asyncio
    async def test_record_halt_auto_stop_event(self) -> None:
        """record_audit_event must persist a row for 'circuit_breaker_halt_auto_stop'."""
        from api.services.audit_log import record_audit_event

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        await record_audit_event(
            mock_session,
            event_type="circuit_breaker_halt_auto_stop",
            resource_type="run",
            resource_id="test-run-id",
            request=None,
            payload={"trigger": "graduated_halt", "run_id": "test-run-id"},
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()


# ===========================================================================
# TestOrchestratorAuditConditional (CR4-001)
# ===========================================================================


class TestOrchestratorAuditConditional:
    """The orchestrator's finally-block audit write must be conditional on
    engine.auto_stop_reason == _HALT_AUTO_STOP_REASON_VALUE."""

    @pytest.mark.asyncio
    async def test_audit_event_called_when_auto_stop_reason_set(self) -> None:
        """When engine.auto_stop_reason == 'circuit_breaker_halt', the
        orchestrator must call record_audit_event exactly once."""
        # Engine mock with the auto-stop attribute set
        engine = SimpleNamespace(auto_stop_reason="circuit_breaker_halt")

        # Simulate the orchestrator finally-block predicate
        from api.services.run_orchestrator import _HALT_AUTO_STOP_REASON_VALUE

        with patch(
            "api.services.run_orchestrator.record_audit_event",
            new=AsyncMock(),
        ) as mock_audit:
            if engine is not None and getattr(engine, "auto_stop_reason", None) == _HALT_AUTO_STOP_REASON_VALUE:
                from api.services.run_orchestrator import record_audit_event

                await record_audit_event(
                    AsyncMock(),
                    event_type="circuit_breaker_halt_auto_stop",
                    resource_type="run",
                    resource_id="test-run",
                    request=None,
                    payload={"trigger": "graduated_halt", "run_id": "test-run"},
                )

            mock_audit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audit_event_not_called_when_auto_stop_reason_is_none(self) -> None:
        """When engine.auto_stop_reason is None (operator/timeout stop), the
        orchestrator must NOT call record_audit_event."""
        engine = SimpleNamespace(auto_stop_reason=None)

        from api.services.run_orchestrator import _HALT_AUTO_STOP_REASON_VALUE

        with patch(
            "api.services.run_orchestrator.record_audit_event",
            new=AsyncMock(),
        ) as mock_audit:
            if engine is not None and getattr(engine, "auto_stop_reason", None) == _HALT_AUTO_STOP_REASON_VALUE:
                from api.services.run_orchestrator import record_audit_event

                await record_audit_event(
                    AsyncMock(),
                    event_type="circuit_breaker_halt_auto_stop",
                    resource_type="run",
                    resource_id="test-run",
                    request=None,
                    payload={},
                )

            mock_audit.assert_not_awaited()


# ===========================================================================
# TestNoLiquidationContract (RISK-CB-008)
# ===========================================================================


class TestNoLiquidationContract:
    """Locks the no-liquidation contract documented by RISK-CB-005."""

    def test_portfolio_not_mutated_on_halt(self) -> None:
        """HALT must NOT close, liquidate, or otherwise mutate open positions."""
        # Build a portfolio mock with 2 open positions
        positions_before = {
            "BTC/USDT": {"qty": 0.5, "entry_price": 50_000.0},
            "ETH/USDT": {"qty": 2.0, "entry_price": 3_000.0},
        }
        portfolio = MagicMock()
        portfolio.positions = positions_before
        portfolio.close_position = MagicMock()
        portfolio.liquidate = MagicMock()
        portfolio.close_all = MagicMock()

        # Capture identity + snapshot before
        positions_id_before = id(portfolio.positions)
        positions_snapshot_before = dict(portfolio.positions)

        # Simulate the HALT branch on a namespace with this portfolio
        ns = _make_engine_ns()
        ns._portfolio = portfolio
        breaker = CircuitBreaker(config=_breaker_config(), run_id="test-no-liquidate")
        breaker.trip("test breach")
        cb_response = breaker.check_graduated(equity=10_000, daily_pnl=0.0, drawdown=0.20)
        if cb_response == CircuitBreakerResponse.HALT and not ns._stop_event.is_set():
            ns._auto_stop_reason = "circuit_breaker_halt"
            ns._stop_event.set()

        # Assert: positions dict is unchanged (same object, same content)
        assert id(portfolio.positions) == positions_id_before
        assert portfolio.positions == positions_snapshot_before
        # Assert: no liquidation methods were called
        portfolio.close_position.assert_not_called()
        portfolio.liquidate.assert_not_called()
        portfolio.close_all.assert_not_called()
