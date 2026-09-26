"""
tests/unit/test_wp18a_shutdown_orphan.py
-------------------------------------------
Unit tests for the WP1.8a shutdown-vs-crash disambiguation in
``apps/api/services/run_orchestrator.py``:

- ``mark_shutdown_requested()`` / ``_SHUTDOWN_REQUESTED``.
- A cancelled ``run_paper_engine``/``run_live_engine`` coroutine writes
  ``status='orphaned'`` (+ a ``run_orphaned`` audit row) when the shutdown
  flag is set, vs. ``'stopped'`` when it is not (a user-initiated
  ``stop_run`` already wrote 'stopped' before cancelling in production --
  this file tests the orchestrator's own half of that contract in
  isolation from the router).

Every heavy internal component (StrategyEngine, execution engine,
portfolio, market data, risk manager) is mocked so the test exercises only
the orchestrator's own try/except/finally control flow, not a real
trading loop.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.services.run_orchestrator as orchestrator
from common.types import TimeFrame


def _make_run_row(status: str = "running") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        status=status,
        stopped_at=None,
        updated_at=None,
    )


def _session_factory_for(run_row: SimpleNamespace) -> MagicMock:
    """A get_session_factory() replacement whose every session's
    scalar_one_or_none() returns the SAME run_row (in-place mutation
    accumulates across the status-update and orphan-audit sessions)."""

    def _new_session() -> AsyncMock:
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        result = MagicMock()
        result.scalar_one_or_none.return_value = run_row
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    factory = MagicMock(side_effect=_new_session)
    return factory


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.exchange_api_key = None
    settings.exchange_api_secret = None
    settings.exchange_api_passphrase = None
    settings.max_run_duration_hours = 24.0
    settings.exchange_id = "coinbase"
    return settings


class _FakeEngine:
    """Stands in for StrategyEngine: start() succeeds, run_live_loop() is
    cancelled (simulating the lifespan cancelling _RUN_TASKS)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._stop_event = asyncio.Event()
        self.auto_stop_reason: str | None = None
        self.start = AsyncMock()
        self.run_live_loop = AsyncMock(side_effect=asyncio.CancelledError())
        self.stop = AsyncMock()


@pytest.fixture(autouse=True)
def _reset_shutdown_flag():
    """_SHUTDOWN_REQUESTED is a module global -- never let one test's flag
    leak into the next."""
    original = orchestrator._SHUTDOWN_REQUESTED
    orchestrator._SHUTDOWN_REQUESTED = False
    yield
    orchestrator._SHUTDOWN_REQUESTED = original


@pytest.fixture(autouse=True)
def _fast_side_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the background flush/auto-stop loops -- irrelevant to the
    shutdown-vs-crash status transition this file tests."""
    monkeypatch.setattr(orchestrator, "_flush_incremental", AsyncMock())
    monkeypatch.setattr(orchestrator, "_incremental_flush_loop", AsyncMock())
    monkeypatch.setattr(orchestrator, "_auto_stop_after", AsyncMock())


class TestMarkShutdownRequested:
    def test_flag_starts_false_and_flips_true(self) -> None:
        orchestrator._SHUTDOWN_REQUESTED = False
        orchestrator.mark_shutdown_requested()
        assert orchestrator._SHUTDOWN_REQUESTED is True


class TestPaperEngineShutdownOrphaning:
    @pytest.mark.asyncio
    async def test_cancel_during_shutdown_writes_orphaned_with_audit(self) -> None:
        run_row = _make_run_row(status="running")
        factory = _session_factory_for(run_row)
        orchestrator.mark_shutdown_requested()

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch.object(orchestrator, "record_audit_event", AsyncMock()) as audit,
            patch("trading.engines.paper.PaperExecutionEngine", MagicMock()),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.risk_manager.DefaultRiskManager", MagicMock()),
            patch("trading.strategy_engine.StrategyEngine", _FakeEngine),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await orchestrator.run_paper_engine(
                    run_id_str=str(run_row.id),
                    strategy_cls=MagicMock,
                    strategy_name="grid_trading",
                    strategy_params={},
                    symbols=["BTC/USD"],
                    timeframe=TimeFrame.ONE_HOUR,
                    initial_capital="10000",
                )

        assert run_row.status == "orphaned"
        assert run_row.stopped_at is None, "an orphaned run is not finished"
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["event_type"] == "run_orphaned"

    @pytest.mark.asyncio
    async def test_cancel_without_shutdown_flag_writes_stopped(self) -> None:
        """Mirrors a user stop_run: no shutdown flag, so a still-'running' row
        (this test never wrote 'stopped' first, unlike production's stop_run)
        gets the ORIGINAL default status, 'stopped' -- not 'orphaned'."""
        run_row = _make_run_row(status="running")
        factory = _session_factory_for(run_row)
        assert orchestrator._SHUTDOWN_REQUESTED is False

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch.object(orchestrator, "record_audit_event", AsyncMock()) as audit,
            patch("trading.engines.paper.PaperExecutionEngine", MagicMock()),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.risk_manager.DefaultRiskManager", MagicMock()),
            patch("trading.strategy_engine.StrategyEngine", _FakeEngine),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await orchestrator.run_paper_engine(
                    run_id_str=str(run_row.id),
                    strategy_cls=MagicMock,
                    strategy_name="grid_trading",
                    strategy_params={},
                    symbols=["BTC/USD"],
                    timeframe=TimeFrame.ONE_HOUR,
                    initial_capital="10000",
                )

        assert run_row.status == "stopped"
        audit.assert_not_awaited()


class TestLiveEngineShutdownOrphaning:
    @pytest.mark.asyncio
    async def test_cancel_during_shutdown_writes_orphaned_with_audit(self) -> None:
        run_row = _make_run_row(status="running")
        factory = _session_factory_for(run_row)
        orchestrator.mark_shutdown_requested()

        fake_exchange = MagicMock()
        fake_exchange.close = AsyncMock()

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch.object(orchestrator, "record_audit_event", AsyncMock()) as audit,
            patch("ccxt.async_support.coinbase", return_value=fake_exchange),
            patch("trading.engines.live.LiveExecutionEngine", MagicMock()),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.risk_manager.DefaultRiskManager", MagicMock()),
            patch("trading.strategy_engine.StrategyEngine", _FakeEngine),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await orchestrator.run_live_engine(
                    run_id_str=str(run_row.id),
                    strategy_cls=MagicMock,
                    strategy_name="grid_trading",
                    strategy_params={},
                    symbols=["BTC/USD"],
                    timeframe=TimeFrame.ONE_HOUR,
                    initial_capital="10000",
                )

        assert run_row.status == "orphaned"
        assert run_row.stopped_at is None
        audit.assert_awaited_once()
        assert audit.call_args.kwargs["event_type"] == "run_orphaned"
