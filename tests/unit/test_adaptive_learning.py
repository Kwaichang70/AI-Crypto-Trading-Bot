"""
tests/unit/test_adaptive_learning.py
--------------------------------------
Unit tests for AdaptiveLearningTask (Sprint 36).

Modules under test
------------------
packages/trading/adaptive_learning.py

Test coverage
-------------
TestIngest (3 tests)
- test_ingest_trade: add 1 trade, verify it's in _all_trades
- test_ingest_skipped: add 1 skipped, verify in _all_skipped
- test_ingest_bulk: add 10 trades, verify count

TestAnalysisCycle (5 tests)
- test_cycle_triggers_at_threshold: add 50 trades -> cycle runs -> cycle_count=1
- test_no_cycle_below_threshold: add 10 trades -> no cycle
- test_cycle_produces_report: after cycle -> last_analysis is not None
- test_dry_run_no_apply: auto_apply=False -> params unchanged
- test_auto_apply_updates_params: auto_apply=True, actionable adjustment -> params updated

TestRollback (4 tests)
- test_rollback_restores_params: mock check_rollback -> should_rollback=True -> params restored
- test_rollback_emits_alert: rollback -> PARAMETER_ROLLBACK alert emitted
- test_learning_disabled_alert: optimizer disabled after rollback -> LEARNING_DISABLED alert
- test_no_rollback_when_ok: PnL within threshold -> no rollback

TestDailyReport (3 tests)
- test_daily_report_generated_once_per_day: tick twice same day -> only 1 report
- test_daily_report_different_days: advance date -> generates again
- test_daily_report_content: verify trades_today filtered correctly

TestWeeklyReport (2 tests)
- test_weekly_report_on_monday: weekday=0 -> generates
- test_no_weekly_report_on_tuesday: weekday=1 -> skips

TestRunLoop (3 tests)
- test_run_stops_on_event: set stop_event -> loop exits cleanly
- test_run_ticks_multiple_times: run for multiple intervals -> cycles > 0
- test_exception_does_not_crash: tick raises -> loop continues
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.types import OrderSide
from trading.adaptive_learning import AdaptiveLearningTask
from trading.adaptive_optimizer import (
    AdaptiveOptimizer,
    ParameterAdjustment,
    ParameterChange,
    RollbackDecision,
)
from trading.models import SkippedTrade, TradeResult
from trading.performance_analyzer import PerformanceAnalyzer
from trading.reporting import AlertLevel, AlertType, ReportingService
from trading.strategy import BaseStrategy

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_BASE_ENTRY_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_BASE_EXIT_AT = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)


def _make_trade(
    symbol: str = "BTC/USD",
    entry_price: float = 100.0,
    exit_price: float = 105.0,
    quantity: float = 1.0,
    run_id: str = "run-1",
    exit_at: datetime | None = None,
) -> TradeResult:
    ep = Decimal(str(entry_price))
    xp = Decimal(str(exit_price))
    qty = Decimal(str(quantity))
    gross = (xp - ep) * qty
    fees = Decimal("0.001") * ep * qty
    realised = gross - fees
    return TradeResult(
        run_id=run_id,
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=ep,
        exit_price=xp,
        quantity=qty,
        realised_pnl=realised,
        total_fees=fees,
        entry_at=_BASE_ENTRY_AT,
        exit_at=exit_at or _BASE_EXIT_AT,
        strategy_id="rsi",
        mae_pct=-0.02,
        mfe_pct=0.05,
        exit_reason="signal_exit",
        regime_at_entry="NEUTRAL",
        signal_context={"rsi": 28.0},
    )


def _make_skipped(
    symbol: str = "BTC/USD",
    run_id: str = "run-1",
) -> SkippedTrade:
    return SkippedTrade(
        run_id=run_id,
        symbol=symbol,
        skip_reason="regime_risk_off",
        regime_at_skip="RISK_OFF",
        hypothetical_outcome_pct=None,
    )


def _make_strategy(params: dict[str, Any] | None = None) -> BaseStrategy:
    """Mock BaseStrategy with _params dict and update_params spy."""
    strategy = MagicMock(spec=BaseStrategy)
    strategy._params = params or {"oversold": 30.0, "overbought": 70.0}
    strategy.update_params = MagicMock()
    return strategy


def _make_task(
    auto_apply: bool = False,
    min_trades_per_cycle: int = 50,
    check_interval_seconds: float = 60.0,
    params: dict[str, Any] | None = None,
) -> AdaptiveLearningTask:
    """Create an AdaptiveLearningTask with a mock strategy and fast settings."""
    strategy = _make_strategy(params)
    return AdaptiveLearningTask(
        strategies=[strategy],
        check_interval_seconds=check_interval_seconds,
        min_trades_per_cycle=min_trades_per_cycle,
        auto_apply=auto_apply,
        original_params=params or {"oversold": 30.0, "overbought": 70.0},
    )


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_trade(self) -> None:
        task = _make_task()
        trade = _make_trade()
        task.ingest_trade(trade)
        assert len(task._all_trades) == 1
        assert task._all_trades[0] is trade

    def test_ingest_skipped(self) -> None:
        task = _make_task()
        skipped = _make_skipped()
        task.ingest_skipped(skipped)
        assert len(task._all_skipped) == 1
        assert task._all_skipped[0] is skipped

    def test_ingest_bulk(self) -> None:
        task = _make_task()
        trades = [_make_trade() for _ in range(10)]
        task.ingest_trades_bulk(trades)
        assert len(task._all_trades) == 10


# ---------------------------------------------------------------------------
# TestAnalysisCycle
# ---------------------------------------------------------------------------


class TestAnalysisCycle:
    @pytest.mark.asyncio
    async def test_cycle_triggers_at_threshold(self) -> None:
        task = _make_task(min_trades_per_cycle=50)
        for _ in range(50):
            task.ingest_trade(_make_trade())
        await task._tick()
        assert task.cycle_count == 1

    @pytest.mark.asyncio
    async def test_no_cycle_below_threshold(self) -> None:
        task = _make_task(min_trades_per_cycle=50)
        for _ in range(10):
            task.ingest_trade(_make_trade())
        await task._tick()
        assert task.cycle_count == 0

    @pytest.mark.asyncio
    async def test_cycle_produces_report(self) -> None:
        task = _make_task(min_trades_per_cycle=5)
        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._tick()
        assert task.last_analysis is not None

    @pytest.mark.asyncio
    async def test_dry_run_no_apply(self) -> None:
        """auto_apply=False -- update_params must never be called."""
        task = _make_task(auto_apply=False, min_trades_per_cycle=5)
        strategy = task._strategies[0]

        # Mock optimizer to return an actionable adjustment
        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="oversold: 30.0->28.0",
            actionable=True,
        )
        task._optimizer = MagicMock(spec=AdaptiveOptimizer)
        task._optimizer.propose_adjustments.return_value = mock_adj
        task._optimizer.state = MagicMock()
        task._optimizer.state.last_adjustment = None
        task._optimizer.is_enabled = True

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        strategy.update_params.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_apply_updates_params(self) -> None:
        """auto_apply=True -- update_params called with new params."""
        new_params = {"oversold": 28.0, "overbought": 70.0}
        task = _make_task(auto_apply=True, min_trades_per_cycle=5)
        strategy = task._strategies[0]

        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="oversold: 30.0->28.0",
            actionable=True,
        )
        task._optimizer = MagicMock(spec=AdaptiveOptimizer)
        task._optimizer.propose_adjustments.return_value = mock_adj
        task._optimizer.apply_adjustment.return_value = new_params
        task._optimizer.state = MagicMock()
        task._optimizer.state.last_adjustment = None
        task._optimizer.is_enabled = True

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        strategy.update_params.assert_called_once_with(new_params)


# ---------------------------------------------------------------------------
# TestRollback
# ---------------------------------------------------------------------------


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_restores_params(self) -> None:
        """When rollback triggered, restored params are passed to update_params."""
        restored_params = {"oversold": 30.0, "overbought": 70.0}
        task = _make_task(auto_apply=True, min_trades_per_cycle=5)
        strategy = task._strategies[0]

        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="test",
            actionable=True,
        )
        rollback_decision = RollbackDecision(
            should_rollback=True,
            reason="PnL degraded -6.00% since adjustment (threshold: -5.0%)",
            pnl_delta_pct=-6.0,
            hours_since_adjustment=2.0,
        )

        mock_optimizer = MagicMock(spec=AdaptiveOptimizer)
        mock_optimizer.propose_adjustments.return_value = mock_adj
        mock_optimizer.check_rollback.return_value = rollback_decision
        mock_optimizer.rollback.return_value = restored_params
        mock_optimizer.is_enabled = True
        mock_state = MagicMock()
        mock_state.last_adjustment = mock_adj  # previous adjustment exists
        mock_optimizer.state = mock_state

        task._optimizer = mock_optimizer

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        strategy.update_params.assert_called_once_with(restored_params)

    @pytest.mark.asyncio
    async def test_rollback_emits_alert(self) -> None:
        """Rollback must emit PARAMETER_ROLLBACK alert."""
        task = _make_task(auto_apply=True, min_trades_per_cycle=5)

        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="test",
            actionable=True,
        )
        rollback_decision = RollbackDecision(
            should_rollback=True,
            reason="PnL degraded",
            pnl_delta_pct=-6.0,
            hours_since_adjustment=2.0,
        )

        mock_optimizer = MagicMock(spec=AdaptiveOptimizer)
        mock_optimizer.propose_adjustments.return_value = mock_adj
        mock_optimizer.check_rollback.return_value = rollback_decision
        mock_optimizer.rollback.return_value = {"oversold": 30.0}
        mock_optimizer.is_enabled = True
        mock_state = MagicMock()
        mock_state.last_adjustment = mock_adj
        mock_optimizer.state = mock_state

        task._optimizer = mock_optimizer

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        alerts = task._reporter.get_alerts()
        alert_types = [a.alert_type for a in alerts]
        assert AlertType.PARAMETER_ROLLBACK in alert_types

    @pytest.mark.asyncio
    async def test_learning_disabled_alert(self) -> None:
        """When optimizer is disabled after rollback, LEARNING_DISABLED alert emitted."""
        task = _make_task(auto_apply=True, min_trades_per_cycle=5)

        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="test",
            actionable=True,
        )
        rollback_decision = RollbackDecision(
            should_rollback=True,
            reason="PnL degraded",
            pnl_delta_pct=-6.0,
            hours_since_adjustment=2.0,
        )

        mock_optimizer = MagicMock(spec=AdaptiveOptimizer)
        mock_optimizer.propose_adjustments.return_value = mock_adj
        mock_optimizer.check_rollback.return_value = rollback_decision
        mock_optimizer.rollback.return_value = {"oversold": 30.0}
        # Simulate optimizer disabled after this rollback
        mock_optimizer.is_enabled = False
        mock_state = MagicMock()
        mock_state.last_adjustment = mock_adj
        mock_optimizer.state = mock_state

        task._optimizer = mock_optimizer

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        alerts = task._reporter.get_alerts()
        alert_types = [a.alert_type for a in alerts]
        assert AlertType.LEARNING_DISABLED in alert_types

    @pytest.mark.asyncio
    async def test_no_rollback_when_ok(self) -> None:
        """When PnL is fine, no rollback occurs."""
        task = _make_task(auto_apply=True, min_trades_per_cycle=5)
        strategy = task._strategies[0]

        mock_change = ParameterChange(
            param_name="oversold",
            old_value=30.0,
            new_value=28.0,
            change_pct=-0.067,
            reason="test",
            confidence=0.8,
        )
        mock_adj = ParameterAdjustment(
            changes=[mock_change],
            overall_confidence=0.8,
            report_summary="test",
            actionable=True,
        )
        ok_decision = RollbackDecision(
            should_rollback=False,
            reason="Monitoring in progress",
            pnl_delta_pct=1.0,
            hours_since_adjustment=2.0,
        )
        new_params = {"oversold": 28.0, "overbought": 70.0}

        mock_optimizer = MagicMock(spec=AdaptiveOptimizer)
        mock_optimizer.propose_adjustments.return_value = mock_adj
        mock_optimizer.check_rollback.return_value = ok_decision
        mock_optimizer.apply_adjustment.return_value = new_params
        mock_optimizer.is_enabled = True
        mock_state = MagicMock()
        mock_state.last_adjustment = mock_adj
        mock_optimizer.state = mock_state

        task._optimizer = mock_optimizer

        for _ in range(5):
            task.ingest_trade(_make_trade())
        await task._run_analysis_cycle()

        # Should proceed to apply, not rollback
        strategy.update_params.assert_called_once_with(new_params)
        alerts = task._reporter.get_alerts()
        rollback_alerts = [a for a in alerts if a.alert_type == AlertType.PARAMETER_ROLLBACK]
        assert len(rollback_alerts) == 0


# ---------------------------------------------------------------------------
# TestDailyReport
# ---------------------------------------------------------------------------


class TestDailyReport:
    @pytest.mark.asyncio
    async def test_daily_report_generated_once_per_day(self) -> None:
        """Two ticks on same UTC day produce only 1 daily report call."""
        task = _make_task()
        mock_reporter = MagicMock(spec=ReportingService)
        mock_reporter._peak_equity = 1000.0
        mock_reporter.get_alerts.return_value = []
        task._reporter = mock_reporter

        today = date(2024, 6, 10)
        # Neither tick has enough trades, so no analysis cycle.
        # The daily report branch fires on both ticks unless guarded.
        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = today
            await task._tick()
            await task._tick()

        # Should be called exactly once (the second tick sees same date)
        assert mock_reporter.generate_daily_report.call_count == 1

    @pytest.mark.asyncio
    async def test_daily_report_different_days(self) -> None:
        """Different dates each produce a daily report call."""
        task = _make_task()
        mock_reporter = MagicMock(spec=ReportingService)
        mock_reporter._peak_equity = 1000.0
        task._reporter = mock_reporter

        day1 = date(2024, 6, 10)
        day2 = date(2024, 6, 11)

        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = day1
            await task._tick()

        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = day2
            await task._tick()

        assert mock_reporter.generate_daily_report.call_count == 2

    @pytest.mark.asyncio
    async def test_daily_report_content(self) -> None:
        """trades_today is filtered to only trades whose exit_at.date() == today."""
        task = _make_task()
        mock_reporter = MagicMock(spec=ReportingService)
        mock_reporter._peak_equity = 1000.0
        task._reporter = mock_reporter

        today = date(2024, 6, 10)
        yesterday = date(2024, 6, 9)

        trade_today = _make_trade(
            exit_at=datetime(2024, 6, 10, 10, 0, 0, tzinfo=UTC)
        )
        trade_yesterday = _make_trade(
            exit_at=datetime(2024, 6, 9, 10, 0, 0, tzinfo=UTC)
        )
        task.ingest_trade(trade_today)
        task.ingest_trade(trade_yesterday)

        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = today

            # Patch datetime.now inside _generate_daily_report indirectly
            # The method uses t.exit_at.date() comparison
            await task._tick()

        call_kwargs = mock_reporter.generate_daily_report.call_args
        assert call_kwargs is not None
        passed_trades = call_kwargs.kwargs.get("trades_today") or call_kwargs.args[0]
        # Only the trade from today should be passed
        assert len(passed_trades) == 1
        assert passed_trades[0] is trade_today


# ---------------------------------------------------------------------------
# TestWeeklyReport
# ---------------------------------------------------------------------------


class TestWeeklyReport:
    @pytest.mark.asyncio
    async def test_weekly_report_on_monday(self) -> None:
        """Monday (weekday=0) generates a weekly report."""
        task = _make_task()
        mock_reporter = MagicMock(spec=ReportingService)
        mock_reporter._peak_equity = 1000.0
        task._reporter = mock_reporter

        # 2024-06-10 is a Monday
        monday = date(2024, 6, 10)
        assert monday.weekday() == 0

        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = monday
            await task._tick()

        assert mock_reporter.generate_weekly_report.call_count == 1

    @pytest.mark.asyncio
    async def test_no_weekly_report_on_tuesday(self) -> None:
        """Non-Monday days do not generate a weekly report."""
        task = _make_task()
        mock_reporter = MagicMock(spec=ReportingService)
        mock_reporter._peak_equity = 1000.0
        task._reporter = mock_reporter

        # 2024-06-11 is a Tuesday
        tuesday = date(2024, 6, 11)
        assert tuesday.weekday() == 1

        with patch("trading.adaptive_learning.date") as mock_date:
            mock_date.today.return_value = tuesday
            await task._tick()

        assert mock_reporter.generate_weekly_report.call_count == 0


# ---------------------------------------------------------------------------
# TestRunLoop
# ---------------------------------------------------------------------------


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_run_stops_on_event(self) -> None:
        """Setting stop_event causes run() to exit cleanly."""
        task = _make_task(check_interval_seconds=0.05)
        stop_event = asyncio.Event()

        async def _set_stop() -> None:
            await asyncio.sleep(0.01)
            stop_event.set()

        await asyncio.gather(task.run(stop_event), _set_stop())
        # If we reach here without hanging, the test passes.

    @pytest.mark.asyncio
    async def test_run_ticks_multiple_times(self) -> None:
        """Loop fires _tick at least twice when run for multiple intervals."""
        task = _make_task(check_interval_seconds=0.02)
        stop_event = asyncio.Event()
        tick_count = 0
        original_tick = task._tick

        async def counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            await original_tick()

        task._tick = counting_tick  # type: ignore[method-assign]

        async def _set_stop() -> None:
            await asyncio.sleep(0.07)
            stop_event.set()

        await asyncio.gather(task.run(stop_event), _set_stop())
        assert tick_count >= 2

    @pytest.mark.asyncio
    async def test_exception_does_not_crash(self) -> None:
        """Exception inside _tick is caught and the loop continues."""
        task = _make_task(check_interval_seconds=0.02)
        stop_event = asyncio.Event()
        call_count = 0

        async def bad_tick() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated error")

        task._tick = bad_tick  # type: ignore[method-assign]

        async def _set_stop() -> None:
            # Give enough time for 2+ ticks
            await asyncio.sleep(0.07)
            stop_event.set()

        await asyncio.gather(task.run(stop_event), _set_stop())
        # After the first exception, the loop kept going
        assert call_count >= 2
