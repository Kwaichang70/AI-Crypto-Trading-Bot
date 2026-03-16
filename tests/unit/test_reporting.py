"""
tests/unit/test_reporting.py
------------------------------
Unit tests for ReportingService (Sprint 35).

Module under test
-----------------
packages/trading/reporting.py

Test coverage
-------------
TestAlertEvent (3 tests)
- test_alert_creation: basic field defaults
- test_alert_levels: all 3 levels accepted
- test_frozen: immutable after creation

TestDailyReport (5 tests)
- test_daily_report_basic: 5 trades → correct win/loss counts
- test_daily_report_no_trades: 0 trades → zeros
- test_param_drift: oversold 30→28 → drift=-0.0667
- test_ath_alert_emitted: equity above peak on second call → NEW_EQUITY_ATH
- test_equity_below_start_alert: equity below start → EQUITY_BELOW_START

TestWeeklyReport (5 tests)
- test_weekly_report_basic: trades + skipped → all sections populated
- test_recommendations_low_win_rate: win_rate < 0.4, 20+ trades → recommendation
- test_recommendations_high_drawdown: drawdown > 10% → recommendation
- test_param_changes_from_adjustments: 2 actionable adjustments → param_changes
- test_skip_quality_from_perf_report: correctly_skipped/missed_opportunities from RegimeStats

TestRegimeChange (3 tests)
- test_regime_change_alert: FEAR→GREED → REGIME_CHANGE alert emitted
- test_no_alert_on_first_regime: first regime set → no regime change alert
- test_no_alert_same_regime: same regime twice → no alert

TestEmitAlert (4 tests)
- test_emit_stores_alert: emit → get_alerts contains it
- test_alert_count: emit 3 → count=3
- test_clear_alerts: clear → count=0
- test_get_alerts_since: filter by datetime works

TestCircuitBreakerAlerts (3 tests)
- test_emit_circuit_breaker_halt: HALT alert → CRITICAL level
- test_emit_rollback: PARAMETER_ROLLBACK → WARNING level
- test_emit_learning_disabled: LEARNING_DISABLED → CRITICAL level

TestParamDrift (2 tests)
- test_drift_positive: param increased → positive drift
- test_drift_zero_original: original=0 → no drift key (no division by zero)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from common.types import OrderSide
from trading.adaptive_optimizer import (
    OptimizerState,
    ParameterAdjustment,
    ParameterChange,
)
from trading.models import SkippedTrade, TradeResult
from trading.performance_analyzer import (
    IndicatorAnalysis,
    PairAnalysis,
    ParameterAnalysis,
    PerformanceReport,
    RegimeAnalysis,
    RegimeStats,
)
from trading.reporting import (
    AlertEvent,
    AlertLevel,
    AlertType,
    DailyReport,
    ReportingService,
    WeeklyReport,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_BASE_ENTRY_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_BASE_EXIT_AT = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)


def _make_trade(
    entry_price: float = 100.0,
    exit_price: float = 105.0,
    quantity: float = 1.0,
    regime_at_entry: str | None = "NEUTRAL",
    exit_reason: str | None = "signal_exit",
    symbol: str = "BTC/USD",
    run_id: str = "run-1",
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
        exit_at=_BASE_EXIT_AT,
        strategy_id="rsi",
        exit_reason=exit_reason,
        regime_at_entry=regime_at_entry,
    )


def _make_losing_trade(
    entry_price: float = 100.0,
    exit_price: float = 95.0,
    quantity: float = 1.0,
) -> TradeResult:
    ep = Decimal(str(entry_price))
    xp = Decimal(str(exit_price))
    qty = Decimal(str(quantity))
    gross = (xp - ep) * qty
    fees = Decimal("0.001") * ep * qty
    realised = gross - fees
    return TradeResult(
        run_id="run-1",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        entry_price=ep,
        exit_price=xp,
        quantity=qty,
        realised_pnl=realised,
        total_fees=fees,
        entry_at=_BASE_ENTRY_AT,
        exit_at=_BASE_EXIT_AT,
        strategy_id="rsi",
    )


def _make_skipped(
    regime_at_skip: str | None = "RISK_OFF",
    hypothetical_outcome_pct: float | None = None,
) -> SkippedTrade:
    return SkippedTrade(
        run_id="run-1",
        symbol="BTC/USD",
        skip_reason="regime_risk_off",
        regime_at_skip=regime_at_skip,
        hypothetical_outcome_pct=hypothetical_outcome_pct,
    )


def _make_regime_analysis(
    correctly_skipped: int = 3,
    skipped_would_profit: int = 2,
) -> RegimeAnalysis:
    rs = RegimeStats(
        regime="NEUTRAL",
        trade_count=10,
        win_count=6,
        win_rate=0.6,
        avg_pnl_pct=0.02,
        total_pnl_pct=0.20,
        skipped_would_profit=skipped_would_profit,
        skipped_correctly=correctly_skipped,
        skipped_unknown=1,
    )
    return RegimeAnalysis(
        by_regime=[rs],
        best_regime="NEUTRAL",
        worst_regime="NEUTRAL",
        confidence=0.8,
    )


def _make_performance_report(
    correctly_skipped: int = 3,
    skipped_would_profit: int = 2,
    warnings: list[str] | None = None,
) -> PerformanceReport:
    regime = _make_regime_analysis(correctly_skipped, skipped_would_profit)
    indicator = IndicatorAnalysis(
        by_indicator=[],
        most_predictive=None,
        highest_false_signal_rate=None,
        confidence=0.8,
    )
    parameter = ParameterAnalysis(
        rsi_buckets=[],
        stop_loss_hit_rate=0.3,
        take_profit_hit_rate=0.4,
        trailing_stop_hit_rate=0.1,
        signal_exit_rate=0.2,
        avg_mfe_winners=0.05,
        avg_mae_losers=-0.02,
        avg_mfe_all=0.04,
        avg_mae_all=-0.01,
        mfe_beyond_tp_count=2,
        mfe_beyond_tp_rate=0.2,
        confidence=0.8,
    )
    pair = PairAnalysis(
        by_symbol=[],
        best_symbol=None,
        worst_symbol=None,
        confidence=0.8,
    )
    return PerformanceReport(
        generated_at=datetime.now(tz=UTC),
        analysis_window_start=None,
        analysis_window_end=None,
        total_trades=10,
        total_skipped=5,
        overall_win_rate=0.6,
        overall_avg_pnl_pct=0.02,
        overall_confidence=0.8,
        is_actionable=True,
        regime=regime,
        indicators=indicator,
        parameters=parameter,
        pairs=pair,
        warnings=warnings or [],
        safeguards_applied={},
    )


def _make_parameter_adjustment(actionable: bool = True) -> ParameterAdjustment:
    change = ParameterChange(
        param_name="oversold",
        old_value=30.0,
        new_value=28.0,
        change_pct=-0.0667,
        reason="Best RSI buy bucket Sharpe=1.2",
        confidence=0.75,
    )
    return ParameterAdjustment(
        changes=[change],
        overall_confidence=0.75,
        report_summary="oversold: 30→28",
        actionable=actionable,
        rejection_reason=None if actionable else "Low confidence",
    )


# ---------------------------------------------------------------------------
# TestAlertEvent
# ---------------------------------------------------------------------------


class TestAlertEvent:
    def test_alert_creation(self) -> None:
        alert = AlertEvent(
            alert_type=AlertType.CIRCUIT_BREAKER_HALT,
            level=AlertLevel.CRITICAL,
            message="Trading halted",
            details={"drawdown": 0.15},
        )
        assert alert.alert_type == AlertType.CIRCUIT_BREAKER_HALT
        assert alert.level == AlertLevel.CRITICAL
        assert alert.message == "Trading halted"
        assert alert.details == {"drawdown": 0.15}
        # UUID was auto-generated
        assert alert.alert_id is not None
        # Timestamp is recent
        assert (datetime.now(tz=UTC) - alert.created_at).total_seconds() < 5

    def test_alert_levels(self) -> None:
        for level in (AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL):
            a = AlertEvent(
                alert_type=AlertType.REGIME_CHANGE,
                level=level,
                message="test",
            )
            assert a.level == level

    def test_frozen(self) -> None:
        alert = AlertEvent(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.INFO,
            message="test",
        )
        with pytest.raises(Exception):
            alert.message = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDailyReport
# ---------------------------------------------------------------------------


class TestDailyReport:
    def test_daily_report_basic(self) -> None:
        svc = ReportingService()
        trades = [
            _make_trade(exit_price=105.0),  # win
            _make_trade(exit_price=105.0),  # win
            _make_trade(exit_price=105.0),  # win
            _make_losing_trade(),           # loss
            _make_losing_trade(),           # loss
        ]
        report = svc.generate_daily_report(
            trades_today=trades,
            total_equity=10_000.0,
            peak_equity=10_500.0,
            daily_pnl_usd=150.0,
        )
        assert isinstance(report, DailyReport)
        assert report.trades_today == 5
        assert report.wins_today == 3
        assert report.losses_today == 2
        assert report.total_equity == 10_000.0
        assert report.peak_equity == 10_500.0
        # Drawdown: (10500 - 10000) / 10500
        expected_dd = round((10_500.0 - 10_000.0) / 10_500.0, 6)
        assert abs(report.drawdown_pct - expected_dd) < 1e-6

    def test_daily_report_no_trades(self) -> None:
        svc = ReportingService()
        report = svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
        )
        assert report.trades_today == 0
        assert report.wins_today == 0
        assert report.losses_today == 0
        assert report.drawdown_pct == 0.0

    def test_param_drift(self) -> None:
        svc = ReportingService()
        report = svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            active_params={"oversold": 28},
            original_params={"oversold": 30},
        )
        # drift = (28 - 30) / 30 = -0.0667 (rounded to 4 dp)
        assert "oversold" in report.param_drift
        assert abs(report.param_drift["oversold"] - round(-2 / 30, 4)) < 1e-6

    def test_ath_alert_emitted(self) -> None:
        svc = ReportingService()
        # First call — sets _peak_equity to 10_000, no ATH alert
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
        )
        assert svc.alert_count == 0

        # Second call with higher equity → ATH alert
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_500.0,
            peak_equity=10_500.0,
            daily_pnl_usd=500.0,
        )
        ath_alerts = [
            a for a in svc.get_alerts() if a.alert_type == AlertType.NEW_EQUITY_ATH
        ]
        assert len(ath_alerts) == 1
        assert ath_alerts[0].level == AlertLevel.INFO

    def test_equity_below_start_alert(self) -> None:
        svc = ReportingService()
        # First call records start_equity = 10_000
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
        )
        # Second call with equity below start
        svc.generate_daily_report(
            trades_today=[],
            total_equity=9_500.0,
            peak_equity=10_000.0,
            daily_pnl_usd=-500.0,
        )
        below_alerts = [
            a
            for a in svc.get_alerts()
            if a.alert_type == AlertType.EQUITY_BELOW_START
        ]
        assert len(below_alerts) == 1
        assert below_alerts[0].level == AlertLevel.WARNING


# ---------------------------------------------------------------------------
# TestWeeklyReport
# ---------------------------------------------------------------------------


class TestWeeklyReport:
    def test_weekly_report_basic(self) -> None:
        svc = ReportingService()
        trades = [_make_trade() for _ in range(10)]
        skipped = [_make_skipped() for _ in range(3)]
        report = svc.generate_weekly_report(
            trades=trades,
            skipped=skipped,
            weekly_return_pct=0.05,
            max_drawdown_pct=0.03,
            sharpe_ratio=1.2,
        )
        assert isinstance(report, WeeklyReport)
        assert report.total_trades == 10
        assert report.weekly_return_pct == round(0.05, 6)
        assert report.max_drawdown_pct == round(0.03, 6)
        assert report.sharpe_ratio == round(1.2, 4)
        # week_start is Monday, week_end is today
        from datetime import date
        today = date.today()
        assert report.week_end == today
        assert report.week_start <= today

    def test_recommendations_low_win_rate(self) -> None:
        svc = ReportingService()
        # 5 wins out of 25 trades = 20% win rate — below 40% threshold
        winning = [_make_trade() for _ in range(5)]
        losing = [_make_losing_trade() for _ in range(20)]
        trades = winning + losing
        report = svc.generate_weekly_report(
            trades=trades,
            skipped=[],
        )
        assert any("Win rate below 40%" in r for r in report.recommendations)

    def test_recommendations_high_drawdown(self) -> None:
        svc = ReportingService()
        report = svc.generate_weekly_report(
            trades=[_make_trade()],
            skipped=[],
            max_drawdown_pct=0.15,
        )
        assert any("Max drawdown" in r for r in report.recommendations)
        assert any("15.0%" in r for r in report.recommendations)

    def test_param_changes_from_adjustments(self) -> None:
        svc = ReportingService()
        adj1 = _make_parameter_adjustment(actionable=True)
        adj2 = _make_parameter_adjustment(actionable=True)
        non_actionable = _make_parameter_adjustment(actionable=False)
        report = svc.generate_weekly_report(
            trades=[_make_trade()],
            skipped=[],
            adjustments=[adj1, adj2, non_actionable],
        )
        # Each actionable adjustment has 1 ParameterChange → 2 entries total
        assert report.adjustments_made == 2
        assert len(report.param_changes) == 2
        assert report.param_changes[0]["param"] == "oversold"

    def test_skip_quality_from_perf_report(self) -> None:
        svc = ReportingService()
        perf = _make_performance_report(correctly_skipped=4, skipped_would_profit=3)
        report = svc.generate_weekly_report(
            trades=[_make_trade()],
            skipped=[],
            performance_report=perf,
        )
        assert report.correctly_skipped == 4
        assert report.missed_opportunities == 3


# ---------------------------------------------------------------------------
# TestRegimeChange
# ---------------------------------------------------------------------------


class TestRegimeChange:
    def test_regime_change_alert(self) -> None:
        svc = ReportingService()
        # First call: FEAR — no alert (first regime observation)
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            current_regime="FEAR",
        )
        assert svc.alert_count == 0

        # Second call: GREED — regime change alert
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            current_regime="GREED",
        )
        regime_alerts = [
            a for a in svc.get_alerts() if a.alert_type == AlertType.REGIME_CHANGE
        ]
        assert len(regime_alerts) == 1
        assert regime_alerts[0].details["old"] == "FEAR"
        assert regime_alerts[0].details["new"] == "GREED"

    def test_no_alert_on_first_regime(self) -> None:
        svc = ReportingService()
        svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            current_regime="NEUTRAL",
        )
        regime_alerts = [
            a for a in svc.get_alerts() if a.alert_type == AlertType.REGIME_CHANGE
        ]
        assert len(regime_alerts) == 0

    def test_no_alert_same_regime(self) -> None:
        svc = ReportingService()
        for _ in range(3):
            svc.generate_daily_report(
                trades_today=[],
                total_equity=10_000.0,
                peak_equity=10_000.0,
                daily_pnl_usd=0.0,
                current_regime="NEUTRAL",
            )
        regime_alerts = [
            a for a in svc.get_alerts() if a.alert_type == AlertType.REGIME_CHANGE
        ]
        assert len(regime_alerts) == 0


# ---------------------------------------------------------------------------
# TestEmitAlert
# ---------------------------------------------------------------------------


class TestEmitAlert:
    def test_emit_stores_alert(self) -> None:
        svc = ReportingService()
        alert = svc.emit_alert(
            AlertType.CIRCUIT_BREAKER_HALT,
            AlertLevel.CRITICAL,
            "Halt triggered",
            {"drawdown": 0.2},
        )
        stored = svc.get_alerts()
        assert len(stored) == 1
        assert stored[0].alert_id == alert.alert_id

    def test_alert_count(self) -> None:
        svc = ReportingService()
        for _ in range(3):
            svc.emit_alert(AlertType.REGIME_CHANGE, AlertLevel.INFO, "test")
        assert svc.alert_count == 3

    def test_clear_alerts(self) -> None:
        svc = ReportingService()
        svc.emit_alert(AlertType.REGIME_CHANGE, AlertLevel.INFO, "test")
        svc.emit_alert(AlertType.REGIME_CHANGE, AlertLevel.INFO, "test2")
        svc.clear_alerts()
        assert svc.alert_count == 0
        assert svc.get_alerts() == []

    def test_get_alerts_since(self) -> None:
        svc = ReportingService()
        cutoff = datetime.now(tz=UTC)
        # Inject two alerts with manipulated timestamps: one before and one after cutoff
        old_alert = AlertEvent(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.INFO,
            message="old",
            created_at=cutoff - timedelta(hours=2),
        )
        new_alert = AlertEvent(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.INFO,
            message="new",
            created_at=cutoff + timedelta(seconds=1),
        )
        svc._alerts.extend([old_alert, new_alert])

        result = svc.get_alerts(since=cutoff)
        assert len(result) == 1
        assert result[0].message == "new"


# ---------------------------------------------------------------------------
# TestCircuitBreakerAlerts
# ---------------------------------------------------------------------------


class TestCircuitBreakerAlerts:
    def test_emit_circuit_breaker_halt(self) -> None:
        svc = ReportingService()
        alert = svc.emit_alert(
            AlertType.CIRCUIT_BREAKER_HALT,
            AlertLevel.CRITICAL,
            "Circuit breaker halted all trading",
            {"drawdown_pct": 0.18},
        )
        assert alert.alert_type == AlertType.CIRCUIT_BREAKER_HALT
        assert alert.level == AlertLevel.CRITICAL

    def test_emit_rollback(self) -> None:
        svc = ReportingService()
        alert = svc.emit_alert(
            AlertType.PARAMETER_ROLLBACK,
            AlertLevel.WARNING,
            "Parameters rolled back after PnL degradation",
            {"pnl_delta_pct": -0.06},
        )
        assert alert.alert_type == AlertType.PARAMETER_ROLLBACK
        assert alert.level == AlertLevel.WARNING

    def test_emit_learning_disabled(self) -> None:
        svc = ReportingService()
        alert = svc.emit_alert(
            AlertType.LEARNING_DISABLED,
            AlertLevel.CRITICAL,
            "Adaptive learning disabled: 3 rollbacks in 30 days",
            {"rollback_count": 3},
        )
        assert alert.alert_type == AlertType.LEARNING_DISABLED
        assert alert.level == AlertLevel.CRITICAL


# ---------------------------------------------------------------------------
# TestParamDrift
# ---------------------------------------------------------------------------


class TestParamDrift:
    def test_drift_positive(self) -> None:
        svc = ReportingService()
        # oversold raised from 30 → 33: positive drift = (33-30)/30 = 0.1
        report = svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            active_params={"oversold": 33},
            original_params={"oversold": 30},
        )
        assert "oversold" in report.param_drift
        assert abs(report.param_drift["oversold"] - round(3 / 30, 4)) < 1e-6

    def test_drift_zero_original(self) -> None:
        """When original value is 0, division is skipped — no KeyError."""
        svc = ReportingService()
        report = svc.generate_daily_report(
            trades_today=[],
            total_equity=10_000.0,
            peak_equity=10_000.0,
            daily_pnl_usd=0.0,
            active_params={"some_param": 5.0},
            original_params={"some_param": 0.0},
        )
        # Key must not appear in param_drift (zero original → skip)
        assert "some_param" not in report.param_drift
