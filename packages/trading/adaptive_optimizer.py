"""
packages/trading/adaptive_optimizer.py
----------------------------------------
Conservative parameter tuning engine for live strategy adaptation (Sprint 34).

Consumes a frozen ``PerformanceReport`` produced by ``PerformanceAnalyzer``
(Sprint 33) and returns bounded ``ParameterAdjustment`` proposals.

Safety invariants
-----------------
S-1  SAFEGUARDS are bounded by absolute limits enforced in ``__init__``.
S-2  RSI oversold in [15, 40], overbought in [60, 85], gap >= 20.
S-3  Stop-loss in [1%, 8%], take-profit in [2%, 15%], TP > SL.
S-4  Position multiplier in [0.50, 1.20].
S-5  Max 20% parameter change per cycle, scaled by confidence.
S-6  No adaptation below 30 trades or 7 days.
S-7  No adaptation during rollback cooldown.
S-8  Auto-rollback trigger when PnL degrades >= 5% post-adjustment.
S-9  72-hour cooldown after every rollback.
S-10 3 rollbacks in 30 days → permanent disable (requires manual re-enable).
S-11 Optimizer never writes to strategy directly; caller applies changes.
S-12 RiskManager hard limits remain the backstop — optimizer cannot override.
S-13 Disabled state requires manual re-enable via ``state.is_enabled = True``.

Thread-safety
-------------
NOT thread-safe.  Designed for single-threaded use within the StrategyEngine
bar-loop or a dedicated AdaptiveLearningTask.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from trading.performance_analyzer import PerformanceReport

__all__ = [
    "AdaptiveOptimizer",
    "ParameterAdjustment",
    "ParameterChange",
    "RollbackDecision",
    "OptimizerState",
    "SAFEGUARDS",
]

# ---------------------------------------------------------------------------
# Module-level safeguard constants
# ---------------------------------------------------------------------------

SAFEGUARDS: dict[str, Any] = {
    "max_position_pct": 0.25,
    "max_total_exposure_pct": 0.60,
    "min_stop_loss_pct": 1.0,
    "max_stop_loss_pct": 8.0,
    "min_take_profit_pct": 2.0,
    "max_take_profit_pct": 15.0,
    "rsi_oversold_range": (15.0, 40.0),
    "rsi_overbought_range": (60.0, 85.0),
    "max_param_change_per_cycle": 0.20,
    "min_trades_before_adapt": 30,
    "min_days_before_adapt": 7,
    "rollback_threshold_pct": -5.0,
    "rollback_cooldown_hours": 72,
    "max_rollbacks_30d": 3,
    "confidence_threshold": 0.65,
}

# Absolute bounds — no caller-supplied override may exceed these limits.
# Maps safeguard key -> (absolute_min, absolute_max).
_ABSOLUTE_BOUNDS: dict[str, tuple[float, float]] = {
    "min_stop_loss_pct": (0.5, 10.0),
    "max_stop_loss_pct": (0.5, 10.0),
    "min_take_profit_pct": (1.0, 20.0),
    "max_take_profit_pct": (1.0, 20.0),
    "max_param_change_per_cycle": (0.01, 0.30),
    "min_trades_before_adapt": (10.0, 200.0),
    "min_days_before_adapt": (1.0, 90.0),
    "rollback_threshold_pct": (-20.0, -0.5),
    "rollback_cooldown_hours": (24.0, 168.0),
    "max_rollbacks_30d": (1.0, 10.0),
    "confidence_threshold": (0.50, 0.90),
}

# CR-002: Validate RSI range tuples
def _validate_rsi_range(key: str, value: tuple[float, float]) -> tuple[float, float]:
    """Ensure RSI range tuple is (lo, hi) with lo < hi and within [0, 100]."""
    lo, hi = value
    if lo >= hi:
        raise ValueError(f"{key}: lo ({lo}) must be < hi ({hi})")
    lo = max(0.0, min(100.0, lo))
    hi = max(0.0, min(100.0, hi))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Pydantic data models — immutable audit trail
# ---------------------------------------------------------------------------


class ParameterChange(BaseModel):
    """Single parameter adjustment with full audit trail."""

    model_config = {"frozen": True}

    param_name: str
    old_value: float
    new_value: float
    change_pct: float = Field(
        description="(new - old) / old, 0.0 if old == 0"
    )
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ParameterAdjustment(BaseModel):
    """Full adjustment proposal — immutable once created."""

    model_config = {"frozen": True}

    adjustment_id: UUID = Field(default_factory=uuid4)
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    changes: list[ParameterChange]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    report_summary: str
    actionable: bool
    rejection_reason: str | None = None


class RollbackDecision(BaseModel):
    """Result of a rollback evaluation."""

    model_config = {"frozen": True}

    should_rollback: bool
    reason: str
    pnl_delta_pct: float
    hours_since_adjustment: float


class OptimizerState(BaseModel):
    """
    Persistent optimizer state — serializable to JSON for crash recovery.

    Intentionally mutable (no frozen=True) so the optimizer can update it
    in place after each adjustment or rollback.
    """

    is_enabled: bool = True
    last_adjustment: ParameterAdjustment | None = None
    previous_params: dict[str, Any] | None = None
    pre_adjustment_pnl_pct: float | None = None
    rollback_count_30d: int = 0
    last_rollback_at: datetime | None = None
    cooldown_until: datetime | None = None
    disabled_reason: str | None = None


# ---------------------------------------------------------------------------
# AdaptiveOptimizer
# ---------------------------------------------------------------------------


class AdaptiveOptimizer:
    """
    Conservative parameter tuning engine for live strategy adaptation.

    Algorithm outline
    -----------------
    1. ``propose_adjustments(report, current_params)``
       — Gate checks → RSI optimization → Stop/TP optimization
       → Return a ``ParameterAdjustment`` (actionable or not).

    2. Caller checks ``adjustment.actionable``.
       If True, calls ``apply_adjustment(adjustment, current_params)``
       to persist previous_params, then applies changes to the strategy
       via ``strategy.update_params(new_params)``.

    3. On each subsequent bar, caller calls
       ``check_rollback(current_pnl_pct, pre_adjustment_pnl_pct)``.
       If ``RollbackDecision.should_rollback`` is True, caller calls
       ``rollback()`` to obtain restored params and passes them to
       ``strategy.update_params(restored_params)``.

    The optimizer never touches the strategy directly (S-11).
    """

    CONFIDENCE_THRESHOLD: float = 0.65
    MAX_CHANGE_PER_CYCLE: float = 0.20

    def __init__(
        self,
        safeguards: dict[str, Any] | None = None,
        state: OptimizerState | None = None,
    ) -> None:
        """
        Parameters
        ----------
        safeguards:
            Optional overrides for ``SAFEGUARDS``.  Each value is validated
            against ``_ABSOLUTE_BOUNDS`` — a ``ValueError`` is raised if any
            value falls outside the absolute limits.
        state:
            Optional pre-existing ``OptimizerState`` for crash recovery.
            If None a fresh state is created.
        """
        merged: dict[str, Any] = {**SAFEGUARDS, **(safeguards or {})}
        self._validate_safeguards(merged)
        # CR-002: validate RSI range tuples
        for rkey in ("rsi_oversold_range", "rsi_overbought_range"):
            if rkey in merged and isinstance(merged[rkey], tuple):
                merged[rkey] = _validate_rsi_range(rkey, merged[rkey])
        self._safeguards = merged
        self._state = state if state is not None else OptimizerState()
        self._log = structlog.get_logger(__name__).bind(
            component="adaptive_optimizer"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> OptimizerState:
        return self._state

    @property
    def is_enabled(self) -> bool:
        return self._state.is_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_adjustments(
        self,
        report: PerformanceReport,
        current_params: dict[str, Any],
    ) -> ParameterAdjustment:
        """
        Analyze ``PerformanceReport`` and propose bounded parameter changes.

        Returns a ``ParameterAdjustment`` with ``actionable=False`` if:
        - Optimizer is disabled
        - In rollback cooldown period
        - Report is not actionable (``report.is_actionable == False``)
        - Fewer than ``min_trades_before_adapt`` trades available
        - Report overall_confidence below ``CONFIDENCE_THRESHOLD``

        The optimizer never modifies any mutable state during this call.
        ``apply_adjustment`` must be called separately to persist changes.

        Parameters
        ----------
        report:
            Frozen ``PerformanceReport`` from ``PerformanceAnalyzer``.
        current_params:
            Current strategy parameter dictionary (read-only within this call).

        Returns
        -------
        ParameterAdjustment
            Immutable proposal.  Inspect ``.actionable`` before applying.
        """
        # --- Gate checks ---
        if not self._state.is_enabled:
            return self._empty_adjustment(
                f"Optimizer disabled: {self._state.disabled_reason}"
            )

        if self._is_in_cooldown():
            self._log.info(
                "adaptive_optimizer.cooldown_active",
                cooldown_until=str(self._state.cooldown_until),
            )
            return self._empty_adjustment(
                f"In rollback cooldown until {self._state.cooldown_until}"
            )

        if not report.is_actionable:
            return self._empty_adjustment(
                f"Report not actionable: {'; '.join(report.warnings[:3])}"
            )

        min_trades: int = int(self._safeguards["min_trades_before_adapt"])
        if report.total_trades < min_trades:
            return self._empty_adjustment(
                f"Insufficient trades: {report.total_trades} < {min_trades}"
            )

        # Check minimum analysis window
        min_days: float = float(self._safeguards["min_days_before_adapt"])
        if (
            report.analysis_window_start is not None
            and report.analysis_window_end is not None
        ):
            window_days = (
                report.analysis_window_end - report.analysis_window_start
            ).total_seconds() / 86400.0
            if window_days < min_days:
                return self._empty_adjustment(
                    f"Insufficient analysis window: {window_days:.1f}d < {min_days}d"
                )

        conf_threshold: float = float(self._safeguards["confidence_threshold"])
        if report.overall_confidence < conf_threshold:
            return self._empty_adjustment(
                f"Low overall confidence: {report.overall_confidence:.3f} "
                f"< {conf_threshold}"
            )

        # --- Parameter optimization ---
        changes: list[ParameterChange] = []

        rsi_changes = self._optimize_rsi(report, current_params)
        changes.extend(rsi_changes)

        stop_changes = self._optimize_stops(report, current_params)
        changes.extend(stop_changes)

        # Filter trivial changes (< 0.1% delta)
        changes = [c for c in changes if abs(c.change_pct) > 0.001]

        overall_conf = (
            min(c.confidence for c in changes) if changes else 0.0
        )
        actionable = (
            len(changes) > 0
            and overall_conf >= float(self._safeguards["confidence_threshold"])
        )

        summary = (
            "; ".join(
                f"{c.param_name}: {c.old_value:.3f}->{c.new_value:.3f}"
                for c in changes
            )
            if changes
            else "No significant parameter adjustments identified"
        )

        adj = ParameterAdjustment(
            adjustment_id=uuid4(),
            proposed_at=datetime.now(tz=UTC),
            changes=changes,
            overall_confidence=round(overall_conf, 4),
            report_summary=summary,
            actionable=actionable,
            rejection_reason=None if actionable else "Confidence below threshold",
        )

        self._log.info(
            "adaptive_optimizer.proposal_generated",
            adjustment_id=str(adj.adjustment_id),
            n_changes=len(changes),
            overall_confidence=adj.overall_confidence,
            actionable=actionable,
        )

        return adj

    def apply_adjustment(
        self,
        adjustment: ParameterAdjustment,
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Persist rollback state and compute the new parameter dictionary.

        DOES NOT apply params to the strategy.  The caller is responsible
        for calling ``strategy.update_params(new_params)`` with the returned
        dict (S-11).

        Parameters
        ----------
        adjustment:
            A ``ParameterAdjustment`` with ``actionable=True``.
        current_params:
            Current strategy parameter dict — a snapshot is saved for rollback.

        Returns
        -------
        dict[str, Any]
            New parameter dict with changes applied.

        Raises
        ------
        ValueError
            If ``adjustment.actionable`` is False or the optimizer is disabled.
        """
        if not adjustment.actionable:
            raise ValueError(
                "Cannot apply non-actionable adjustment "
                f"(reason: {adjustment.rejection_reason})"
            )
        if not self._state.is_enabled:
            raise ValueError(
                f"Optimizer is disabled: {self._state.disabled_reason}"
            )
        # CR-007: block apply during cooldown
        if self._is_in_cooldown():
            raise ValueError(
                "Cannot apply adjustment during rollback cooldown "
                f"(until {self._state.cooldown_until})"
            )

        # Persist rollback target
        self._state = OptimizerState(
            is_enabled=True,
            last_adjustment=adjustment,
            previous_params=dict(current_params),
            pre_adjustment_pnl_pct=self._state.pre_adjustment_pnl_pct,
            rollback_count_30d=self._state.rollback_count_30d,
            last_rollback_at=self._state.last_rollback_at,
            cooldown_until=self._state.cooldown_until,
            disabled_reason=None,
        )

        # Build new params
        new_params = dict(current_params)
        for change in adjustment.changes:
            new_params[change.param_name] = change.new_value

        self._log.info(
            "adaptive_optimizer.adjustment_applied",
            adjustment_id=str(adjustment.adjustment_id),
            changes=[
                {
                    "param": c.param_name,
                    "old": c.old_value,
                    "new": c.new_value,
                    "confidence": c.confidence,
                }
                for c in adjustment.changes
            ],
            overall_confidence=adjustment.overall_confidence,
        )

        return new_params

    def check_rollback(
        self,
        current_pnl_pct: float,
        pre_adjustment_pnl_pct: float,
    ) -> RollbackDecision:
        """
        Evaluate whether the latest adjustment should be rolled back.

        Should be called periodically (e.g. every bar or every hour) while
        an adjustment is active.

        Parameters
        ----------
        current_pnl_pct:
            Current equity return percentage since run start.
        pre_adjustment_pnl_pct:
            Equity return percentage at the moment the adjustment was applied.

        Returns
        -------
        RollbackDecision
            ``should_rollback=True`` triggers an immediate ``rollback()`` call.
        """
        if self._state.last_adjustment is None:
            return RollbackDecision(
                should_rollback=False,
                reason="No active adjustment",
                pnl_delta_pct=0.0,
                hours_since_adjustment=0.0,
            )

        now = datetime.now(tz=UTC)
        hours = (
            now - self._state.last_adjustment.proposed_at
        ).total_seconds() / 3600.0
        pnl_delta = current_pnl_pct - pre_adjustment_pnl_pct
        threshold: float = float(self._safeguards["rollback_threshold_pct"])
        cooldown_hours: float = float(self._safeguards["rollback_cooldown_hours"])

        self._log.debug(
            "adaptive_optimizer.rollback_check",
            pnl_delta_pct=round(pnl_delta, 4),
            hours_since_adjustment=round(hours, 2),
            should_rollback=pnl_delta <= threshold,
        )

        if pnl_delta <= threshold:
            return RollbackDecision(
                should_rollback=True,
                reason=(
                    f"PnL degraded {pnl_delta:.2f}% since adjustment "
                    f"(threshold: {threshold}%)"
                ),
                pnl_delta_pct=round(pnl_delta, 4),
                hours_since_adjustment=round(hours, 2),
            )

        # Monitoring period passed successfully — clear rollback target
        if hours >= cooldown_hours and pnl_delta > 0:
            self._state.last_adjustment = None
            self._state.previous_params = None
            self._state.pre_adjustment_pnl_pct = None
            return RollbackDecision(
                should_rollback=False,
                reason=(
                    f"Monitoring period passed ({hours:.1f}h >= "
                    f"{cooldown_hours}h), PnL positive"
                ),
                pnl_delta_pct=round(pnl_delta, 4),
                hours_since_adjustment=round(hours, 2),
            )

        return RollbackDecision(
            should_rollback=False,
            reason="Monitoring in progress",
            pnl_delta_pct=round(pnl_delta, 4),
            hours_since_adjustment=round(hours, 2),
        )

    def rollback(self) -> dict[str, Any] | None:
        """
        Roll back to the parameter set that was active before the last adjustment.

        Updates rollback counters, sets 72-hour cooldown, and permanently
        disables the optimizer if the 30-day rollback limit is exceeded.

        Returns
        -------
        dict[str, Any] | None
            Restored parameter dictionary, or None if no rollback target exists.
        """
        if self._state.previous_params is None:
            self._log.warning(
                "adaptive_optimizer.rollback_skipped",
                reason="No previous_params available",
            )
            return None

        restored = dict(self._state.previous_params)
        now = datetime.now(tz=UTC)
        cooldown_hours: float = float(self._safeguards["rollback_cooldown_hours"])
        max_rollbacks: int = int(self._safeguards["max_rollbacks_30d"])
        new_count = self._state.rollback_count_30d + 1

        is_enabled = new_count < max_rollbacks
        disabled_reason: str | None = (
            f"Adaptive learning disabled: {new_count} rollbacks in 30 days "
            f"(limit: {max_rollbacks}). Manual review required."
            if not is_enabled
            else None
        )

        adj_id = (
            str(self._state.last_adjustment.adjustment_id)
            if self._state.last_adjustment is not None
            else "unknown"
        )

        self._state = OptimizerState(
            is_enabled=is_enabled,
            last_adjustment=None,
            previous_params=None,
            pre_adjustment_pnl_pct=None,
            rollback_count_30d=new_count,
            last_rollback_at=now,
            cooldown_until=now + timedelta(hours=cooldown_hours),
            disabled_reason=disabled_reason,
        )

        if not is_enabled:
            self._log.critical(
                "adaptive_optimizer.learning_disabled",
                rollback_count=new_count,
                reason=disabled_reason,
                alert="ADAPTIVE_LEARNING_DISABLED",
            )
        else:
            self._log.warning(
                "adaptive_optimizer.rollback",
                adjustment_id=adj_id,
                rollback_count_30d=new_count,
                cooldown_until=str(self._state.cooldown_until),
            )

        return restored

    # ------------------------------------------------------------------
    # Private optimization methods
    # ------------------------------------------------------------------

    def _optimize_rsi(
        self,
        report: PerformanceReport,
        current_params: dict[str, Any],
    ) -> list[ParameterChange]:
        """
        Optimize RSI oversold/overbought thresholds from bucket Sharpe analysis.

        Algorithm (per design report section 4.1):
        - Oversold: find the buy-side bucket (bucket_high <= 50) with the
          highest per-trade Sharpe.  Use bucket_high as the new threshold.
        - Overbought: find the sell-side bucket (bucket_low >= 50) with the
          highest per-trade Sharpe.  Use bucket_low as the new threshold.
        - Both are clamped to their SAFEGUARDS ranges.
        - The ``_calculate_adjustment`` damping function is applied so that
          the change per cycle cannot exceed 20% of the current value.
        """
        pa = report.parameters
        conf_threshold = float(self._safeguards["confidence_threshold"])

        if not pa.rsi_buckets or pa.confidence < conf_threshold:
            return []

        changes: list[ParameterChange] = []
        current_oversold = float(current_params.get("oversold", 30.0))
        current_overbought = float(current_params.get("overbought", 70.0))

        # --- Oversold (BUY entry) ---
        buy_buckets = [
            b
            for b in pa.rsi_buckets
            if b.bucket_high <= 50 and b.trade_count >= 5
        ]
        if buy_buckets:
            # Highest Sharpe in the oversold zone
            best_buy = max(
                buy_buckets,
                key=lambda b: b.sharpe if b.sharpe != 0 else -999.0,
            )
            if best_buy.sharpe > 0:
                # Use upper bound of the best bucket as the threshold
                suggested_oversold = float(best_buy.bucket_high)
                lo, hi = self._safeguards["rsi_oversold_range"]
                suggested_oversold = max(float(lo), min(float(hi), suggested_oversold))
                new_oversold = self._calculate_adjustment(
                    current_oversold, suggested_oversold, pa.confidence
                )
                new_oversold = max(float(lo), min(float(hi), new_oversold))
                new_oversold = round(new_oversold, 2)

                if abs(new_oversold - current_oversold) > 0.01:
                    change_pct = (
                        (new_oversold - current_oversold) / current_oversold
                        if current_oversold != 0
                        else 0.0
                    )
                    changes.append(
                        ParameterChange(
                            param_name="oversold",
                            old_value=current_oversold,
                            new_value=new_oversold,
                            change_pct=round(change_pct, 4),
                            reason=(
                                f"Best RSI buy bucket "
                                f"[{best_buy.bucket_low:.0f}-"
                                f"{best_buy.bucket_high:.0f}] "
                                f"Sharpe={best_buy.sharpe:.3f}"
                            ),
                            confidence=round(pa.confidence, 4),
                        )
                    )

        # --- Overbought (SELL exit) ---
        sell_buckets = [
            b
            for b in pa.rsi_buckets
            if b.bucket_low >= 50 and b.trade_count >= 5
        ]
        if sell_buckets:
            best_sell = max(
                sell_buckets,
                key=lambda b: b.sharpe if b.sharpe != 0 else -999.0,
            )
            if best_sell.sharpe > 0:
                # Use lower bound of the best sell bucket as the new threshold
                suggested_overbought = float(best_sell.bucket_low)
                lo, hi = self._safeguards["rsi_overbought_range"]
                suggested_overbought = max(
                    float(lo), min(float(hi), suggested_overbought)
                )
                new_overbought = self._calculate_adjustment(
                    current_overbought, suggested_overbought, pa.confidence
                )
                new_overbought = max(float(lo), min(float(hi), new_overbought))
                new_overbought = round(new_overbought, 2)

                if abs(new_overbought - current_overbought) > 0.01:
                    change_pct = (
                        (new_overbought - current_overbought) / current_overbought
                        if current_overbought != 0
                        else 0.0
                    )
                    changes.append(
                        ParameterChange(
                            param_name="overbought",
                            old_value=current_overbought,
                            new_value=new_overbought,
                            change_pct=round(change_pct, 4),
                            reason=(
                                f"Best RSI sell bucket "
                                f"[{best_sell.bucket_low:.0f}-"
                                f"{best_sell.bucket_high:.0f}] "
                                f"Sharpe={best_sell.sharpe:.3f}"
                            ),
                            confidence=round(pa.confidence, 4),
                        )
                    )

        # Safety invariant S-2: ensure oversold < overbought with a gap >= 20
        # Resolve conflicts introduced by independent adjustments.
        final_oversold = next(
            (c.new_value for c in changes if c.param_name == "oversold"),
            current_oversold,
        )
        final_overbought = next(
            (c.new_value for c in changes if c.param_name == "overbought"),
            current_overbought,
        )
        if final_overbought - final_oversold < 20.0:
            # Conflict: drop both changes (safer to do nothing than violate invariant)
            changes = [
                c
                for c in changes
                if c.param_name not in ("oversold", "overbought")
            ]

        return changes

    def _optimize_stops(
        self,
        report: PerformanceReport,
        current_params: dict[str, Any],
    ) -> list[ParameterChange]:
        """
        Optimize stop-loss and take-profit levels from MAE/MFE analysis.

        Algorithm (per design report section 4.2):
        - Stop-loss: ``avg_mae_losers * 100`` as the baseline.
          Heuristic: widen by 15% if stop_loss_hit_rate > 50%,
          narrow by 10% if < 10%.
        - Take-profit: ``avg_mfe_winners * 0.85 * 100`` as the baseline.
          Heuristic: widen by 10% if mfe_beyond_tp_rate > 40%,
          narrow by 10% if take_profit_hit_rate < 10%.
        - Both are clamped to SAFEGUARDS bounds, and TP > SL is enforced.
        """
        pa = report.parameters
        conf_threshold = float(self._safeguards["confidence_threshold"])

        if pa.confidence < conf_threshold:
            return []

        changes: list[ParameterChange] = []

        min_sl = float(self._safeguards["min_stop_loss_pct"])
        max_sl = float(self._safeguards["max_stop_loss_pct"])
        min_tp = float(self._safeguards["min_take_profit_pct"])
        max_tp = float(self._safeguards["max_take_profit_pct"])

        current_sl = float(current_params.get("stop_loss_pct", 3.0))
        current_tp = float(current_params.get("take_profit_pct", 6.0))

        # --- Stop-loss from avg MAE of losers ---
        new_sl = current_sl
        if pa.avg_mae_losers != 0.0:
            # avg_mae_losers is stored as a fraction (e.g. -0.03 = -3%)
            suggested_sl = abs(pa.avg_mae_losers) * 100.0

            # Heuristics
            if pa.stop_loss_hit_rate > 0.50:
                suggested_sl *= 1.15  # stop too tight, widen
            elif pa.stop_loss_hit_rate < 0.10 and current_sl > 0:
                suggested_sl *= 0.90  # stop too wide, tighten

            suggested_sl = max(min_sl, min(max_sl, suggested_sl))
            # Use 60% confidence weight for stop (MAE is noisy)
            adj_sl = self._calculate_adjustment(
                current_sl, suggested_sl, pa.confidence * 0.6
            )
            new_sl = max(min_sl, min(max_sl, round(adj_sl, 2)))

            if abs(new_sl - current_sl) > 0.01:
                change_pct = (
                    (new_sl - current_sl) / current_sl
                    if current_sl != 0
                    else 0.0
                )
                changes.append(
                    ParameterChange(
                        param_name="stop_loss_pct",
                        old_value=current_sl,
                        new_value=new_sl,
                        change_pct=round(change_pct, 4),
                        reason=(
                            f"MAE analysis: avg loser MAE="
                            f"{pa.avg_mae_losers:.4f}, "
                            f"hit_rate={pa.stop_loss_hit_rate:.2f}"
                        ),
                        confidence=round(pa.confidence * 0.6, 4),
                    )
                )

        # --- Take-profit from avg MFE of winners ---
        new_tp = current_tp
        if pa.avg_mfe_winners > 0.0:
            # avg_mfe_winners is stored as a fraction (e.g. 0.06 = 6%)
            suggested_tp = pa.avg_mfe_winners * 0.85 * 100.0

            # Heuristics
            if pa.mfe_beyond_tp_rate > 0.40:
                suggested_tp *= 1.10  # TP too tight, widen
            elif pa.take_profit_hit_rate < 0.10 and current_tp > 0:
                suggested_tp *= 0.90  # TP too wide, tighten

            suggested_tp = max(min_tp, min(max_tp, suggested_tp))
            adj_tp = self._calculate_adjustment(
                current_tp, suggested_tp, pa.confidence * 0.6
            )
            new_tp = max(min_tp, min(max_tp, round(adj_tp, 2)))

            if abs(new_tp - current_tp) > 0.01:
                change_pct = (
                    (new_tp - current_tp) / current_tp
                    if current_tp != 0
                    else 0.0
                )
                changes.append(
                    ParameterChange(
                        param_name="take_profit_pct",
                        old_value=current_tp,
                        new_value=new_tp,
                        change_pct=round(change_pct, 4),
                        reason=(
                            f"MFE analysis: avg winner MFE="
                            f"{pa.avg_mfe_winners:.4f}, "
                            f"mfe_beyond_tp_rate={pa.mfe_beyond_tp_rate:.2f}"
                        ),
                        confidence=round(pa.confidence * 0.6, 4),
                    )
                )

        # Safety invariant S-3: TP must always exceed SL
        final_sl = next(
            (c.new_value for c in changes if c.param_name == "stop_loss_pct"),
            current_sl,
        )
        final_tp = next(
            (c.new_value for c in changes if c.param_name == "take_profit_pct"),
            current_tp,
        )
        if final_tp <= final_sl:
            # Drop the conflicting TP change (keep the SL change if present)
            changes = [
                c for c in changes if c.param_name != "take_profit_pct"
            ]

        return changes

    def _calculate_adjustment(
        self,
        current: float,
        suggested: float,
        confidence: float,
    ) -> float:
        """
        Apply bounded, confidence-scaled adjustment (S-5).

        The maximum delta is ``MAX_CHANGE_PER_CYCLE * abs(current)``.
        The raw delta is clamped to that range, then scaled by confidence
        so low-confidence suggestions produce smaller moves.

        Parameters
        ----------
        current:
            Current parameter value.
        suggested:
            Target parameter value suggested by optimization.
        confidence:
            Confidence scalar in [0.0, 1.0].

        Returns
        -------
        float
            New parameter value (NOT yet clamped to safeguard bounds — the
            caller is responsible for that final clamp).
        """
        max_change = float(self._safeguards["max_param_change_per_cycle"])

        if current == 0.0:
            # Avoid division-by-zero: cap absolute delta based on suggested value
            max_delta = abs(suggested) * max_change if suggested != 0.0 else 0.0
        else:
            max_delta = abs(current) * max_change

        raw_delta = suggested - current
        clamped_delta = max(-max_delta, min(max_delta, raw_delta))
        return current + clamped_delta * confidence

    def _is_in_cooldown(self) -> bool:
        """Return True if the optimizer is inside a rollback cooldown window."""
        if self._state.cooldown_until is None:
            return False
        return datetime.now(tz=UTC) < self._state.cooldown_until

    def _empty_adjustment(self, reason: str) -> ParameterAdjustment:
        """Construct a non-actionable ``ParameterAdjustment`` with the given reason."""
        self._log.info(
            "adaptive_optimizer.proposal_rejected",
            reason=reason,
        )
        return ParameterAdjustment(
            adjustment_id=uuid4(),
            proposed_at=datetime.now(tz=UTC),
            changes=[],
            overall_confidence=0.0,
            report_summary=reason,
            actionable=False,
            rejection_reason=reason,
        )

    # ------------------------------------------------------------------
    # Safeguard validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_safeguards(merged: dict[str, Any]) -> None:
        """
        Validate all numeric safeguard values against ``_ABSOLUTE_BOUNDS``.

        Raises
        ------
        ValueError
            If any value falls outside its absolute bounds.
        """
        for key, (abs_min, abs_max) in _ABSOLUTE_BOUNDS.items():
            if key not in merged:
                continue
            val = merged[key]
            if not isinstance(val, (int, float)):
                continue
            fval = float(val)
            if fval < abs_min or fval > abs_max:
                raise ValueError(
                    f"Safeguard {key!r} value {fval} is outside absolute "
                    f"bounds [{abs_min}, {abs_max}]"
                )
