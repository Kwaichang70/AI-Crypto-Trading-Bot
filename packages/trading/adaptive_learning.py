"""
packages/trading/adaptive_learning.py
---------------------------------------
Background adaptive learning task that orchestrates the full pipeline:
TradeJournal -> PerformanceAnalyzer -> AdaptiveOptimizer -> ReportingService

Runs as a parallel asyncio.Task alongside the paper/live engine.

Sprint 36.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog

from trading.adaptive_optimizer import AdaptiveOptimizer, OptimizerState
from trading.models import SkippedTrade, TradeResult
from trading.performance_analyzer import PerformanceAnalyzer, PerformanceReport
from trading.reporting import AlertLevel, AlertType, ReportingService
from trading.strategy import BaseStrategy

__all__ = ["AdaptiveLearningTask"]

logger = structlog.get_logger(__name__)


class AdaptiveLearningTask:
    """
    Background adaptive learning pipeline for paper/live runs.

    Runs on a configurable interval (default: every 50 trades OR 60 minutes,
    whichever comes first). On each cycle:

    1. Collect trades + skipped trades since last cycle
    2. Run PerformanceAnalyzer
    3. Run AdaptiveOptimizer (propose + optionally apply)
    4. Check rollback conditions
    5. Generate daily report (once per UTC day)
    6. Generate weekly report (once per UTC week)
    7. Emit alerts for significant events

    Parameters
    ----------
    strategies : list[BaseStrategy]
        The strategy instances to tune (via update_params).
    analyzer : PerformanceAnalyzer | None
        Custom analyzer, or None for defaults.
    optimizer : AdaptiveOptimizer | None
        Custom optimizer, or None for defaults.
    reporter : ReportingService | None
        Custom reporter, or None for defaults.
    check_interval_seconds : float
        How often to check for new trades (default 60s -- checks every minute,
        but only runs analysis when trade count threshold is hit).
    min_trades_per_cycle : int
        Minimum new trades before triggering an analysis cycle (default 50).
    auto_apply : bool
        If True, automatically apply actionable adjustments.
        If False, only log proposals (dry-run mode). Default False for safety.
    original_params : dict[str, Any] | None
        Snapshot of the original strategy params for drift tracking.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        analyzer: PerformanceAnalyzer | None = None,
        optimizer: AdaptiveOptimizer | None = None,
        reporter: ReportingService | None = None,
        check_interval_seconds: float = 60.0,
        min_trades_per_cycle: int = 50,
        auto_apply: bool = False,
        original_params: dict[str, Any] | None = None,
    ) -> None:
        self._strategies = strategies
        self._analyzer = analyzer or PerformanceAnalyzer()
        self._optimizer = optimizer or AdaptiveOptimizer()
        self._reporter = reporter or ReportingService()
        self._check_interval = check_interval_seconds
        self._min_trades_per_cycle = min_trades_per_cycle
        self._auto_apply = auto_apply
        self._original_params = dict(original_params) if original_params else {}

        # State tracking
        self._all_trades: list[TradeResult] = []
        self._all_skipped: list[SkippedTrade] = []
        self._trades_at_last_cycle: int = 0
        self._last_daily_report_date: date | None = None
        self._last_weekly_report_date: date | None = None
        self._last_analysis: PerformanceReport | None = None
        self._pre_adjustment_pnl_pct: float = 0.0
        self._cycle_count: int = 0

        self._log = structlog.get_logger(__name__).bind(
            component="adaptive_learning",
            auto_apply=auto_apply,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def optimizer(self) -> AdaptiveOptimizer:
        return self._optimizer

    @property
    def reporter(self) -> ReportingService:
        return self._reporter

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_analysis(self) -> PerformanceReport | None:
        return self._last_analysis

    # ------------------------------------------------------------------
    # Public ingestion interface
    # ------------------------------------------------------------------

    def ingest_trade(self, trade: TradeResult) -> None:
        """Add a completed trade to the collection."""
        self._all_trades.append(trade)

    def ingest_skipped(self, skipped: SkippedTrade) -> None:
        """Add a skipped trade to the collection."""
        self._all_skipped.append(skipped)

    def ingest_trades_bulk(self, trades: list[TradeResult]) -> None:
        """Add multiple trades at once."""
        self._all_trades.extend(trades)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Main loop -- runs until stop_event is set.

        Checks for new trades every check_interval_seconds.
        Triggers analysis when min_trades_per_cycle new trades accumulated.
        """
        self._log.info(
            "adaptive_learning.started",
            check_interval=self._check_interval,
            min_trades=self._min_trades_per_cycle,
        )

        while not stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                self._log.exception("adaptive_learning.tick_error")

            # Sleep until next check, but wake early if stop_event fires
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self._check_interval
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal: continue loop

        self._log.info("adaptive_learning.stopped", cycles=self._cycle_count)

    # ------------------------------------------------------------------
    # Private tick and cycle logic
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """One tick of the learning loop."""
        # Use date.today() so tests can patch trading.adaptive_learning.date
        today = date.today()

        # Check if we should run an analysis cycle
        new_trades = len(self._all_trades) - self._trades_at_last_cycle
        if new_trades >= self._min_trades_per_cycle:
            await self._run_analysis_cycle()

        # Daily report (once per UTC day)
        if self._last_daily_report_date != today:
            self._generate_daily_report()
            self._last_daily_report_date = today

        # Weekly report (once per UTC week, on Monday)
        week_start = today - timedelta(days=today.weekday())
        if self._last_weekly_report_date != week_start and today.weekday() == 0:
            self._generate_weekly_report()
            self._last_weekly_report_date = week_start

    async def _run_analysis_cycle(self) -> None:
        """Run the full analysis -> optimize -> apply pipeline."""
        self._cycle_count += 1
        self._trades_at_last_cycle = len(self._all_trades)

        self._log.info(
            "adaptive_learning.cycle_start",
            cycle=self._cycle_count,
            total_trades=len(self._all_trades),
            total_skipped=len(self._all_skipped),
        )

        # 1. Analyze
        report = self._analyzer.analyze(
            trades=list(self._all_trades),
            skipped=list(self._all_skipped),
        )
        self._last_analysis = report

        self._log.info(
            "adaptive_learning.analysis_complete",
            overall_confidence=report.overall_confidence,
            warnings=len(report.warnings),
        )

        # Need at least one strategy to tune
        if not self._strategies:
            return

        # 2. Get current params from first strategy
        current_params = dict(self._strategies[0]._params)

        # 3. Propose adjustments
        adjustment = self._optimizer.propose_adjustments(report, current_params)

        self._log.info(
            "adaptive_learning.proposal",
            actionable=adjustment.actionable,
            changes=len(adjustment.changes),
            confidence=adjustment.overall_confidence,
            summary=adjustment.report_summary,
        )

        if not adjustment.actionable:
            return

        # 4. Check rollback on previous adjustment first
        if self._optimizer.state.last_adjustment is not None:
            current_pnl = self._compute_current_pnl_pct()
            decision = self._optimizer.check_rollback(
                current_pnl_pct=current_pnl,
                pre_adjustment_pnl_pct=self._pre_adjustment_pnl_pct,
            )
            if decision.should_rollback:
                restored = self._optimizer.rollback()
                if restored is not None:
                    if self._auto_apply:
                        for strategy in self._strategies:
                            strategy.update_params(restored)
                    self._reporter.emit_alert(
                        AlertType.PARAMETER_ROLLBACK,
                        AlertLevel.WARNING,
                        f"Parameter rollback after {decision.hours_since_adjustment:.1f}h: "
                        f"{decision.reason}",
                        {"pnl_delta": decision.pnl_delta_pct},
                    )
                    if not self._optimizer.is_enabled:
                        self._reporter.emit_alert(
                            AlertType.LEARNING_DISABLED,
                            AlertLevel.CRITICAL,
                            "Adaptive learning disabled after 3 rollbacks",
                        )
                self._log.warning(
                    "adaptive_learning.rollback",
                    reason=decision.reason,
                    pnl_delta=decision.pnl_delta_pct,
                )
                return

        # 5. Apply adjustment
        if self._auto_apply:
            self._pre_adjustment_pnl_pct = self._compute_current_pnl_pct()
            new_params = self._optimizer.apply_adjustment(adjustment, current_params)
            for strategy in self._strategies:
                strategy.update_params(new_params)
            self._reporter.emit_alert(
                AlertType.PARAMETER_ADJUSTMENT,
                AlertLevel.INFO,
                f"Applied {len(adjustment.changes)} parameter changes "
                f"(conf={adjustment.overall_confidence:.2f})",
                {"changes": [c.model_dump() for c in adjustment.changes]},
            )
            self._log.info(
                "adaptive_learning.adjustment_applied",
                changes=len(adjustment.changes),
            )
        else:
            self._log.info(
                "adaptive_learning.dry_run",
                msg="Adjustment proposed but auto_apply=False",
                changes=len(adjustment.changes),
            )

    # ------------------------------------------------------------------
    # PnL computation helper
    # ------------------------------------------------------------------

    def _compute_current_pnl_pct(self) -> float:
        """Compute total PnL percentage from all trades."""
        if not self._all_trades:
            return 0.0
        total_pnl = sum(float(t.realised_pnl) for t in self._all_trades)
        total_cost = sum(
            float(t.entry_price * t.quantity) for t in self._all_trades
        )
        if total_cost == 0:
            return 0.0
        return total_pnl / total_cost

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_daily_report(self) -> None:
        """Generate and log a daily report."""
        today = date.today()
        trades_today = [
            t for t in self._all_trades if t.exit_at.date() == today
        ]

        daily_pnl = sum(float(t.realised_pnl) for t in trades_today)
        total_pnl = sum(float(t.realised_pnl) for t in self._all_trades)

        # Attempt to read FGI / current regime (best-effort, optional dependency)
        current_regime: str | None = None
        fgi: int | None = None
        try:
            from data.sentiment import get_global_client  # noqa: PLC0415

            client = get_global_client()
            if client is not None:
                fgi_val = client.cached_value
                if fgi_val is not None:
                    fgi = fgi_val
                    from trading.strategy_engine import StrategyEngine  # noqa: PLC0415

                    current_regime = StrategyEngine._fgi_to_regime(fgi_val)
        except Exception:
            pass

        current_params = (
            dict(self._strategies[0]._params) if self._strategies else {}
        )

        self._reporter.generate_daily_report(
            trades_today=trades_today,
            total_equity=1000.0 + total_pnl,
            peak_equity=self._reporter._peak_equity or 1000.0,
            daily_pnl_usd=daily_pnl,
            current_regime=current_regime,
            fear_greed_index=fgi,
            optimizer_state=self._optimizer.state,
            active_params=current_params,
            original_params=self._original_params,
        )

    def _generate_weekly_report(self) -> None:
        """Generate and log a weekly report."""
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        weekly_trades = [
            t for t in self._all_trades if t.exit_at.date() >= week_start
        ]

        current_params = (
            dict(self._strategies[0]._params) if self._strategies else {}
        )

        self._reporter.generate_weekly_report(
            trades=weekly_trades,
            skipped=list(self._all_skipped),
            performance_report=self._last_analysis,
            optimizer_state=self._optimizer.state,
            original_params=self._original_params,
            current_params=current_params,
        )
