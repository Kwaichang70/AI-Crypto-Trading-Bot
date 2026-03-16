"""
packages/trading/reporting.py
-------------------------------
Reporting and alerting module for the adaptive learning system (Sprint 35).

Generates daily and weekly performance reports and emits structured alert
events for significant system events such as circuit breaker trips, equity
milestones, regime changes, and optimizer state transitions.

Reports are logged via structlog at INFO level. Alerts are logged at their
appropriate severity level (INFO / WARNING / CRITICAL).

Webhook delivery is reserved for a future sprint (placeholder ``_webhook_url``
field exists but delivery is not yet implemented).

Usage example
-------------
::

    from trading.reporting import ReportingService, AlertType, AlertLevel

    svc = ReportingService()

    # Emit a manual circuit-breaker alert
    svc.emit_alert(
        AlertType.CIRCUIT_BREAKER_HALT,
        AlertLevel.CRITICAL,
        "Trading halted: drawdown exceeded 15%",
        {"drawdown_pct": 0.15},
    )

    # Generate a daily summary
    report = svc.generate_daily_report(
        trades_today=completed_trades,
        total_equity=98500.0,
        peak_equity=100000.0,
        daily_pnl_usd=-1500.0,
    )

Thread-safety
-------------
NOT thread-safe. Designed for single-threaded use within the StrategyEngine
or a dedicated reporting task.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from trading.adaptive_optimizer import OptimizerState, ParameterAdjustment
from trading.models import SkippedTrade, TradeResult
from trading.performance_analyzer import PerformanceReport

__all__ = [
    "ReportingService",
    "AlertEvent",
    "AlertLevel",
    "AlertType",
    "DailyReport",
    "WeeklyReport",
]


# ---------------------------------------------------------------------------
# Alert enumerations
# ---------------------------------------------------------------------------


class AlertLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(StrEnum):
    CIRCUIT_BREAKER_REDUCE = "circuit_breaker_reduce"
    CIRCUIT_BREAKER_HALT = "circuit_breaker_halt"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    PARAMETER_ROLLBACK = "parameter_rollback"
    LEARNING_DISABLED = "learning_disabled"
    REGIME_CHANGE = "regime_change"
    NEW_EQUITY_ATH = "new_equity_ath"
    EQUITY_BELOW_START = "equity_below_start"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"


# ---------------------------------------------------------------------------
# Frozen Pydantic data models
# ---------------------------------------------------------------------------


class AlertEvent(BaseModel):
    """A single alert event for logging and optional webhook delivery."""

    model_config = {"frozen": True}

    alert_id: UUID = Field(default_factory=uuid4)
    alert_type: AlertType
    level: AlertLevel
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class DailyReport(BaseModel):
    """Daily trading summary report."""

    model_config = {"frozen": True}

    report_date: date
    # Performance
    daily_pnl_pct: float
    daily_pnl_usd: float
    total_equity: float
    peak_equity: float
    drawdown_pct: float
    # Trading activity
    trades_today: int
    wins_today: int
    losses_today: int
    # Regime
    current_regime: str | None = None
    fear_greed_index: int | None = None
    # Safety status
    circuit_breaker_status: str = "ok"
    kill_switch_active: bool = False
    # Adaptive learning
    optimizer_enabled: bool = True
    active_params: dict[str, Any] = Field(default_factory=dict)
    original_params: dict[str, Any] = Field(default_factory=dict)
    param_drift: dict[str, float] = Field(default_factory=dict)


class WeeklyReport(BaseModel):
    """Weekly performance and learning cycle report."""

    model_config = {"frozen": True}

    week_start: date
    week_end: date
    # Performance summary
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    weekly_return_pct: float
    # Learning cycle
    adjustments_made: int
    rollbacks: int
    optimizer_enabled: bool
    # Parameter changes
    param_changes: list[dict[str, Any]] = Field(default_factory=list)
    current_vs_original: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Trade quality
    correctly_skipped: int = 0
    missed_opportunities: int = 0
    # Recommendations
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ReportingService
# ---------------------------------------------------------------------------


class ReportingService:
    """
    Generates daily/weekly reports and emits alert events.

    Reports are logged via structlog at INFO level.
    Alerts are logged at their appropriate level (INFO/WARNING/CRITICAL).
    Optional webhook delivery for alerts (not implemented in Sprint 35).
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._alerts: list[AlertEvent] = []
        self._last_regime: str | None = None
        self._peak_equity: float = 0.0
        self._start_equity: float | None = None
        self._log = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface — report generation
    # ------------------------------------------------------------------

    def generate_daily_report(
        self,
        trades_today: list[TradeResult],
        total_equity: float,
        peak_equity: float,
        daily_pnl_usd: float,
        circuit_breaker_status: str = "ok",
        kill_switch_active: bool = False,
        current_regime: str | None = None,
        fear_greed_index: int | None = None,
        optimizer_state: OptimizerState | None = None,
        active_params: dict[str, Any] | None = None,
        original_params: dict[str, Any] | None = None,
    ) -> DailyReport:
        """Build and log a daily report.

        Side effects
        ------------
        - Sets ``_start_equity`` on first call.
        - Updates ``_peak_equity`` and emits ``NEW_EQUITY_ATH`` when a new
          all-time high is reached (only after the first call so spurious ATH
          alerts on cold-start are suppressed).
        - Emits ``EQUITY_BELOW_START`` when equity falls below the starting
          value recorded on the first call.
        - Emits ``REGIME_CHANGE`` when ``current_regime`` differs from the
          previously seen regime (suppressed on first regime observation).
        """
        # Track start equity (first observation only)
        if self._start_equity is None:
            self._start_equity = total_equity

        # Track peak equity and emit ATH alert
        if total_equity > self._peak_equity:
            if self._peak_equity > 0:  # Suppress alert on first bar
                self.emit_alert(
                    AlertType.NEW_EQUITY_ATH,
                    AlertLevel.INFO,
                    f"New equity ATH: ${total_equity:.2f}",
                    {"equity": total_equity},
                )
            self._peak_equity = total_equity

        # Equity below start alert
        if self._start_equity is not None and total_equity < self._start_equity:
            self.emit_alert(
                AlertType.EQUITY_BELOW_START,
                AlertLevel.WARNING,
                f"Equity ${total_equity:.2f} below start ${self._start_equity:.2f}",
                {"equity": total_equity, "start": self._start_equity},
            )

        # Regime change detection (suppressed on first regime set)
        if current_regime is not None and current_regime != self._last_regime:
            if self._last_regime is not None:
                self.emit_alert(
                    AlertType.REGIME_CHANGE,
                    AlertLevel.INFO,
                    f"Regime changed: {self._last_regime} -> {current_regime}",
                    {"old": self._last_regime, "new": current_regime},
                )
            self._last_regime = current_regime

        # Compute parameter drift from original baseline
        param_drift: dict[str, float] = {}
        if active_params and original_params:
            for key in active_params:
                if key in original_params:
                    orig = float(original_params[key])
                    curr = float(active_params[key])
                    if orig != 0:
                        param_drift[key] = round((curr - orig) / orig, 4)

        wins = sum(1 for t in trades_today if t.realised_pnl > 0)
        losses = sum(1 for t in trades_today if t.realised_pnl < 0)
        daily_pnl_pct = (daily_pnl_usd / total_equity) if total_equity > 0 else 0.0
        dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0

        report = DailyReport(
            report_date=date.today(),
            daily_pnl_pct=round(daily_pnl_pct, 6),
            daily_pnl_usd=round(daily_pnl_usd, 2),
            total_equity=round(total_equity, 2),
            peak_equity=round(peak_equity, 2),
            drawdown_pct=round(dd, 6),
            trades_today=len(trades_today),
            wins_today=wins,
            losses_today=losses,
            current_regime=current_regime,
            fear_greed_index=fear_greed_index,
            circuit_breaker_status=circuit_breaker_status,
            kill_switch_active=kill_switch_active,
            optimizer_enabled=optimizer_state.is_enabled if optimizer_state else True,
            active_params=active_params or {},
            original_params=original_params or {},
            param_drift=param_drift,
        )

        self._log.info("reporting.daily_report", **report.model_dump(mode="json"))
        return report

    def generate_weekly_report(
        self,
        trades: list[TradeResult],
        skipped: list[SkippedTrade],
        performance_report: PerformanceReport | None = None,
        optimizer_state: OptimizerState | None = None,
        adjustments: list[ParameterAdjustment] | None = None,
        original_params: dict[str, Any] | None = None,
        current_params: dict[str, Any] | None = None,
        weekly_return_pct: float = 0.0,
        max_drawdown_pct: float = 0.0,
        sharpe_ratio: float = 0.0,
    ) -> WeeklyReport:
        """Build and log a weekly report.

        Computes win rate, collects actionable parameter changes from
        ``adjustments``, computes parameter drift vs ``original_params``,
        accumulates correctly-skipped and missed-opportunity counts from the
        ``performance_report.regime`` sub-analysis, and appends recommendation
        strings from the performance report warnings plus heuristic thresholds.
        """
        today = date.today()
        # ISO week starts on Monday
        week_start = today - timedelta(days=today.weekday())

        wins = sum(1 for t in trades if t.realised_pnl > 0)
        win_rate = wins / len(trades) if trades else 0.0

        # Collect parameter changes from actionable adjustments
        param_changes: list[dict[str, Any]] = []
        rollbacks = 0
        if adjustments:
            for adj in adjustments:
                if adj.actionable:
                    for ch in adj.changes:
                        param_changes.append(
                            {
                                "param": ch.param_name,
                                "old": ch.old_value,
                                "new": ch.new_value,
                                "reason": ch.reason,
                            }
                        )
        if optimizer_state:
            rollbacks = optimizer_state.rollback_count_30d

        # Current vs original parameter drift comparison
        current_vs_original: dict[str, dict[str, float]] = {}
        if current_params and original_params:
            for key in current_params:
                if key in original_params:
                    curr = float(current_params[key])
                    orig = float(original_params[key])
                    drift = round((curr - orig) / orig, 4) if orig != 0 else 0.0
                    current_vs_original[key] = {
                        "current": curr,
                        "original": orig,
                        "drift_pct": drift,
                    }

        # Skip quality metrics aggregated from regime sub-analysis
        correctly_skipped = 0
        missed_opportunities = 0
        if performance_report:
            for rs in performance_report.regime.by_regime:
                correctly_skipped += rs.skipped_correctly
                missed_opportunities += rs.skipped_would_profit

        # Build recommendation list
        recommendations: list[str] = []
        if performance_report and performance_report.warnings:
            recommendations.extend(performance_report.warnings[:5])
        if win_rate < 0.4 and len(trades) >= 20:
            recommendations.append(
                "Win rate below 40% — consider pausing and reviewing strategy params"
            )
        if max_drawdown_pct > 0.10:
            recommendations.append(
                f"Max drawdown {max_drawdown_pct:.1%} exceeds 10% — review position sizing"
            )

        report = WeeklyReport(
            week_start=week_start,
            week_end=today,
            total_trades=len(trades),
            win_rate=round(win_rate, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown_pct=round(max_drawdown_pct, 6),
            weekly_return_pct=round(weekly_return_pct, 6),
            adjustments_made=len(param_changes),
            rollbacks=rollbacks,
            optimizer_enabled=optimizer_state.is_enabled if optimizer_state else True,
            param_changes=param_changes,
            current_vs_original=current_vs_original,
            correctly_skipped=correctly_skipped,
            missed_opportunities=missed_opportunities,
            recommendations=recommendations,
        )

        self._log.info("reporting.weekly_report", **report.model_dump(mode="json"))
        return report

    # ------------------------------------------------------------------
    # Public interface — alert management
    # ------------------------------------------------------------------

    def emit_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> AlertEvent:
        """Create, log, and store an alert event.

        The alert is appended to the internal ``_alerts`` list and written to
        structlog at the severity that matches ``level``.

        Parameters
        ----------
        alert_type:
            Categorisation of the event.
        level:
            Severity level — determines the structlog log level used.
        message:
            Human-readable description of the event.
        details:
            Optional arbitrary key-value pairs logged alongside the alert.

        Returns
        -------
        AlertEvent
            The frozen event object (also stored internally).
        """
        alert = AlertEvent(
            alert_type=alert_type,
            level=level,
            message=message,
            details=details or {},
        )
        self._alerts.append(alert)

        log_method = {
            AlertLevel.INFO: self._log.info,
            AlertLevel.WARNING: self._log.warning,
            AlertLevel.CRITICAL: self._log.critical,
        }.get(level, self._log.info)

        log_method(
            "reporting.alert",
            alert_type=alert_type.value,
            level=level.value,
            message=message,
            **(details or {}),
        )

        return alert

    def get_alerts(self, since: datetime | None = None) -> list[AlertEvent]:
        """Return all stored alerts, optionally filtered by creation time.

        Parameters
        ----------
        since:
            When provided, only alerts with ``created_at >= since`` are
            returned.  Pass ``None`` to receive the full history.
        """
        if since is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.created_at >= since]

    def clear_alerts(self) -> None:
        """Discard all stored alerts."""
        self._alerts.clear()

    @property
    def alert_count(self) -> int:
        """Return the number of stored alerts."""
        return len(self._alerts)
