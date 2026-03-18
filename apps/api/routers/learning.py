"""
apps/api/routers/learning.py
------------------------------
Adaptive learning state endpoint for running paper/live engines.

Reads in-memory state from the AdaptiveLearningTask instance registered
in _LEARNING_INSTANCES. Only available while the run is active.

Sprint 37 — Phase 2.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routers.runs import _LEARNING_INSTANCES

router = APIRouter(prefix="/api/v1/runs", tags=["learning"])


@router.get("/{run_id}/learning")
async def get_learning_state(run_id: str) -> dict[str, Any]:
    """
    Return the current adaptive learning state for a running engine.

    Only available while the run is active and has adaptive learning enabled.
    Returns optimizer state, cycle count, trades ingested, and last analysis summary.
    """
    learner = _LEARNING_INSTANCES.get(run_id)
    if learner is None:
        raise HTTPException(
            status_code=404,
            detail="No active adaptive learning task for this run. "
            "Either the run has stopped, or adaptive learning was not enabled.",
        )

    # Build optimizer state summary
    opt_state = learner.optimizer.state
    optimizer_summary: dict[str, Any] = {
        "isEnabled": opt_state.is_enabled,
        "rollbackCount30d": opt_state.rollback_count_30d,
        "cooldownUntil": opt_state.cooldown_until.isoformat() if opt_state.cooldown_until else None,
        "disabledReason": opt_state.disabled_reason,
        "preAdjustmentPnlPct": opt_state.pre_adjustment_pnl_pct,
    }

    # Last adjustment summary
    last_adj: dict[str, Any] | None = None
    if opt_state.last_adjustment is not None:
        adj = opt_state.last_adjustment
        last_adj = {
            "actionable": adj.actionable,
            "confidence": adj.confidence,
            "reason": adj.reason,
            "changes": [
                {
                    "paramName": c.param_name,
                    "oldValue": c.old_value,
                    "newValue": c.new_value,
                    "changePct": c.change_pct,
                }
                for c in adj.changes
            ],
        }

    # Last analysis summary
    last_analysis: dict[str, Any] | None = None
    if learner.last_analysis is not None:
        report = learner.last_analysis
        last_analysis = {
            "confidence": report.confidence,
            "isActionable": report.is_actionable,
            "totalTrades": report.total_trades,
            "totalSkipped": report.total_skipped,
        }
        # Add regime summary if available
        if report.regimes and report.regimes.by_regime:
            last_analysis["bestRegime"] = report.regimes.best_regime
            last_analysis["worstRegime"] = report.regimes.worst_regime
        # Add indicator summary if available
        if report.indicators and report.indicators.most_predictive:
            last_analysis["mostPredictiveIndicator"] = report.indicators.most_predictive

    return {
        "enabled": True,
        "autoApply": learner._auto_apply,
        "cycleCount": learner.cycle_count,
        "tradesIngested": len(learner._all_trades),
        "skippedIngested": len(learner._all_skipped),
        "tradesAtLastCycle": learner._trades_at_last_cycle,
        "minTradesPerCycle": learner._min_trades_per_cycle,
        "optimizerState": optimizer_summary,
        "lastAdjustment": last_adj,
        "lastAnalysis": last_analysis,
    }
