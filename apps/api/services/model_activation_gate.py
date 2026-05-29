"""
apps/api/services/model_activation_gate.py
-------------------------------------------
Walk-forward OOS gate for model activation (Sprint 50 Cycle 5 Sub-scope B).

The gate reads the persisted ``walk_forward_oos_skill_score`` from a
``ModelVersionORM`` row and compares it against the configured threshold.
NULL OOS skill score (pre-Cycle-5 models) passes with a diagnostic warning --
operators are informed but not blocked on legacy models.

IMPORTANT (Sprint 50 Cycle 6): the gate metric is now a REALIZED out-of-sample
trading Sharpe computed by a per-fold BacktestRunner mini-backtest (net Coinbase
60/40 bps fees + Sprint-42 slippage), persisted on
``ModelVersionORM.walk_forward_oos_skill_score`` (column name retained).  Legacy
models tagged ``metric_type=directional_zscore_proxy`` (Cycle 5) are not
comparable to the realized-Sharpe threshold and pass with a retrain warning.

This module is purposely free of routing logic; the router calls
``check_oos_eligibility()`` and decides whether to raise HTTP 422.

Gate logic (in order):
0. If JSONB ``extra["walk_forward"]["metric_type"] == "directional_zscore_proxy"``
   (legacy Cycle-5 model): pass with ``warning`` -- the z-score proxy is not
   comparable to the realized-Sharpe threshold; operator is told to retrain.
1. If ``walk_forward_oos_skill_score`` is NULL (pre-Cycle-5 model): pass with
   ``warning`` populated (backward-compatible pass-with-warning).
2. If JSONB ``extra["walk_forward"]["status"] == "insufficient_samples"``:
   reject with reason="insufficient_oos_samples".  Distinct from "skill is bad"
   -- it means "we cannot measure skill" (too few realized OOS trades).
3. If ``walk_forward_oos_skill_score`` < ``min_oos_skill_score`` (median realized
   OOS Sharpe gate): reject with reason="oos_skill_below_min".
4. If worst-fold Sharpe < ``min_worst_fold_skill_score`` floor: reject with
   reason="worst_fold_below_floor".
5. If any fold's OOS max drawdown > ``max_fold_drawdown`` (B1): reject with
   reason="fold_drawdown_exceeds_floor".
6. Otherwise: eligible.

CR5v2-002 (Option A): ``OOSEligibility`` has a separate ``detail`` field for
human-readable rejection messages (distinct from ``warning`` which is reserved
for pass-with-warning cases where ``eligible=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from api.db.models import ModelVersionORM

__all__ = ["OOSEligibility", "check_oos_eligibility"]

logger = structlog.get_logger(__name__)


@dataclass
class OOSEligibility:
    """Result of the OOS skill score gate check.

    Attributes
    ----------
    eligible:
        True when the model may be activated (OOS skill score above threshold
        or no OOS data available -- pass-with-warning).
    oos_skill_score:
        The persisted OOS skill score value (deflated median directional z-score
        proxy).  None for pre-Cycle-5 models.
    threshold:
        The configured ``min_oos_skill_score`` value.
    warning:
        Human-readable warning when ``eligible=True`` but OOS data is absent
        (i.e. the model passed the gate but without OOS evidence).
        EMPTY on ``eligible=False`` paths -- use ``detail`` instead.
    detail:
        Human-readable description when ``eligible=False`` explaining why
        the gate rejected the model.  Distinct from ``warning``:
        ``warning`` is for pass-with-warning cases; ``detail`` is for
        rejection cases.
    reason:
        Machine-readable failure code when ``eligible=False``.
        One of: "oos_skill_below_min", "worst_fold_below_floor",
        "insufficient_oos_samples", or empty string when eligible.
    worst_fold_skill_score:
        OOS skill score of the worst individual walk-forward fold.  None when
        fold metadata is unavailable.
    """

    eligible: bool
    oos_skill_score: float | None
    threshold: float
    warning: str = ""
    detail: str = ""
    reason: str = ""
    worst_fold_skill_score: float | None = None
    # Sprint 50 Cycle 6 (B1): worst per-fold OOS max drawdown read from JSONB
    # extra["walk_forward"]["worst_fold_drawdown"]. None when unavailable.
    worst_fold_drawdown_observed: float | None = None

    # ---------------------------------------------------------------------------
    # Backward-compatibility properties: old callers that read .oos_sharpe /
    # .worst_fold_sharpe still work without changes.  These are read-only aliases.
    # Remove in Cycle 6+ once all callers are updated.
    # ---------------------------------------------------------------------------
    @property
    def oos_sharpe(self) -> float | None:  # noqa: D401
        """Alias for oos_skill_score (backward compat)."""
        return self.oos_skill_score

    @property
    def worst_fold_sharpe(self) -> float | None:  # noqa: D401
        """Alias for worst_fold_skill_score (backward compat)."""
        return self.worst_fold_skill_score


def check_oos_eligibility(
    model_version: ModelVersionORM,
    *,
    min_oos_skill_score: float,
    min_worst_fold_skill_score: float = 0.0,
    min_trades_per_fold: int = 20,
    max_fold_drawdown: float = 0.25,
    # Backward-compat aliases (ml.py still passes these names via **kwargs)
    min_oos_sharpe: float | None = None,
    min_worst_fold_sharpe: float | None = None,
) -> OOSEligibility:
    """Evaluate OOS gate for a model version.

    Parameters
    ----------
    model_version:
        ORM row with walk_forward_oos_skill_score populated (or NULL).
        ``model_version.extra["walk_forward"]["oos_skill_worst"]`` is read
        for the worst-fold floor check.
        ``model_version.extra["walk_forward"]["status"]`` is read for the
        insufficient-samples check.
    min_oos_skill_score:
        Minimum DEFLATED MEDIAN OOS directional z-score skill score (from settings).
        This is a classification accuracy proxy -- NOT a trading Sharpe.
    min_worst_fold_skill_score:
        Minimum OOS skill score the worst individual fold must achieve.
        Default 0.0 -- no catastrophic-regime tolerance.
    min_trades_per_fold:
        Minimum OOS trades required in every fold for skill score metrics to
        be considered statistically meaningful.  Passed for the warning
        message only; the gate itself reads JSONB status "insufficient_samples"
        as written by the train endpoint.
    max_fold_drawdown:
        Sprint 50 Cycle 6 (B1).  Maximum tolerated per-fold OOS max drawdown
        (decimal fraction).  If JSONB ``worst_fold_drawdown`` exceeds this the
        gate rejects with reason="fold_drawdown_exceeds_floor".  Default 0.25.
    min_oos_sharpe:
        Deprecated alias for min_oos_skill_score (backward compat).
    min_worst_fold_sharpe:
        Deprecated alias for min_worst_fold_skill_score (backward compat).

    Returns
    -------
    OOSEligibility
        Gate result with diagnostic fields.
    """
    # Resolve backward-compat aliases (callers using old kwarg names still work).
    if min_oos_sharpe is not None:
        min_oos_skill_score = min_oos_sharpe
    if min_worst_fold_sharpe is not None:
        min_worst_fold_skill_score = min_worst_fold_sharpe

    # Read the JSONB walk_forward block once (used by several gates below).
    wf_meta: dict[str, Any] = {}
    if model_version.extra and isinstance(model_version.extra, dict):
        wf_meta = model_version.extra.get("walk_forward", {}) or {}

    # Gate -1 (Sprint 50 Cycle 6): legacy Cycle-5 z-score-proxy models are NOT
    # comparable to the realized-Sharpe threshold. Pass with a retrain warning
    # rather than gating them on a metric they were never measured against.
    if wf_meta.get("metric_type") == "directional_zscore_proxy":
        logger.warning(
            "model_activation_gate.legacy_zscore_proxy_passthrough",
            model_id=str(model_version.id),
            symbol=model_version.symbol,
            timeframe=model_version.timeframe,
        )
        raw_legacy = getattr(model_version, "walk_forward_oos_skill_score", None)
        return OOSEligibility(
            eligible=True,
            oos_skill_score=float(raw_legacy) if raw_legacy is not None else None,
            threshold=min_oos_skill_score,
            warning=(
                "This model was scored with the legacy Cycle-5 directional "
                "z-score proxy, which is NOT comparable to the realized-OOS-Sharpe "
                "gate (Sprint 50 Cycle 6). It is allowed to activate, but you "
                "should retrain via POST /ml/train to obtain a realized OOS Sharpe."
            ),
        )

    # Support both old (walk_forward_oos_sharpe) and new (walk_forward_oos_skill_score)
    # ORM attribute names during the transition window.
    raw_oos_col = getattr(model_version, "walk_forward_oos_skill_score", None)
    if raw_oos_col is None:
        raw_oos_col = getattr(model_version, "walk_forward_oos_sharpe", None)

    oos_skill_score = float(raw_oos_col) if raw_oos_col is not None else None

    # Gate 0: NULL = pre-Cycle-5 model: pass with warning (backward compatibility).
    if oos_skill_score is None:
        logger.warning(
            "model_activation_gate.no_oos_skill_score",
            model_id=str(model_version.id),
            symbol=model_version.symbol,
            timeframe=model_version.timeframe,
        )
        return OOSEligibility(
            eligible=True,
            oos_skill_score=None,
            threshold=min_oos_skill_score,
            warning=(
                "This model was trained before the walk-forward OOS gate was "
                "introduced (Sprint 50 Cycle 5). OOS skill score (directional "
                "z-score proxy) is unknown. "
                "Retrain via POST /ml/train to compute OOS metrics."
            ),
        )

    # Reject before skill check if OOS sample count was insufficient.
    # Distinct from "skill is bad" -- it means "we cannot measure skill".
    wf_status = wf_meta.get("status", "")

    if wf_status == "insufficient_samples":
        actual_min = min(wf_meta.get("fold_trade_counts", [0]))
        folds_below = wf_meta.get("folds_below_threshold", [])
        logger.warning(
            "model_activation_gate.insufficient_oos_samples",
            model_id=str(model_version.id),
            folds_below_threshold=folds_below,
            actual_min_fold_trades=actual_min,
        )
        return OOSEligibility(
            eligible=False,
            oos_skill_score=None,
            threshold=min_oos_skill_score,
            reason="insufficient_oos_samples",
            detail=(
                f"Walk-forward folds {folds_below} had fewer than "
                f"{min_trades_per_fold} OOS trades (minimum was {actual_min}). "
                "Train on more historical bars or reduce num_wf_folds."
            ),
        )

    # Gate 1: DEFLATED MEDIAN OOS skill score must meet threshold.
    if oos_skill_score < min_oos_skill_score:
        logger.warning(
            "model_activation_gate.oos_skill_below_threshold",
            model_id=str(model_version.id),
            oos_skill_score=oos_skill_score,
            threshold=min_oos_skill_score,
            note="OOS gate uses the realized OOS trading Sharpe (net fees + slippage)",
        )
        return OOSEligibility(
            eligible=False,
            oos_skill_score=oos_skill_score,
            threshold=min_oos_skill_score,
            reason="oos_skill_below_min",
            detail=(
                f"Median realized OOS Sharpe {oos_skill_score:.4f} is below the "
                f"configured minimum {min_oos_skill_score:.4f} "
                "(net Coinbase 60/40 fees + Sprint-42 slippage)."
            ),
        )

    # Gate 2: Worst-fold floor -- prevents catastrophic-regime models from
    # activating even when the median is acceptable.
    # Support both old (oos_sharpe_worst) and new (oos_skill_worst) JSONB keys.
    worst_fold_skill_score: float | None = wf_meta.get("oos_skill_worst")
    if worst_fold_skill_score is None:
        worst_fold_skill_score = wf_meta.get("oos_sharpe_worst")

    if worst_fold_skill_score is not None and worst_fold_skill_score < min_worst_fold_skill_score:
        logger.warning(
            "model_activation_gate.worst_fold_below_floor",
            model_id=str(model_version.id),
            worst_fold_skill_score=worst_fold_skill_score,
            floor=min_worst_fold_skill_score,
            note="OOS gate uses the realized OOS trading Sharpe (net fees + slippage)",
        )
        return OOSEligibility(
            eligible=False,
            oos_skill_score=oos_skill_score,
            threshold=min_oos_skill_score,
            reason="worst_fold_below_floor",
            detail=(
                f"Worst-fold realized OOS Sharpe {worst_fold_skill_score:.4f} is "
                f"below the configured floor {min_worst_fold_skill_score:.4f} "
                "(net fees + slippage)."
            ),
            worst_fold_skill_score=worst_fold_skill_score,
        )

    # Gate 3 (Sprint 50 Cycle 6 / B1): per-fold max-drawdown floor.
    # Read worst_fold_drawdown from JSONB; reject if it exceeds max_fold_drawdown.
    worst_fold_drawdown_raw = wf_meta.get("worst_fold_drawdown")
    worst_fold_drawdown: float | None = (
        float(worst_fold_drawdown_raw) if worst_fold_drawdown_raw is not None else None
    )
    if worst_fold_drawdown is not None and worst_fold_drawdown > max_fold_drawdown:
        logger.warning(
            "model_activation_gate.fold_drawdown_exceeds_floor",
            model_id=str(model_version.id),
            worst_fold_drawdown=worst_fold_drawdown,
            max_fold_drawdown=max_fold_drawdown,
        )
        return OOSEligibility(
            eligible=False,
            oos_skill_score=oos_skill_score,
            threshold=min_oos_skill_score,
            reason="fold_drawdown_exceeds_floor",
            detail=(
                f"Worst-fold OOS max drawdown {worst_fold_drawdown:.2%} exceeds "
                f"the configured floor {max_fold_drawdown:.2%}. At least one "
                "walk-forward fold suffered a catastrophic drawdown."
            ),
            worst_fold_skill_score=worst_fold_skill_score,
            worst_fold_drawdown_observed=worst_fold_drawdown,
        )

    logger.info(
        "model_activation_gate.eligible",
        model_id=str(model_version.id),
        oos_skill_score=oos_skill_score,
        threshold=min_oos_skill_score,
        note="OOS gate uses the realized OOS trading Sharpe (net fees + slippage)",
    )
    return OOSEligibility(
        eligible=True,
        oos_skill_score=oos_skill_score,
        threshold=min_oos_skill_score,
        worst_fold_skill_score=worst_fold_skill_score,
        worst_fold_drawdown_observed=worst_fold_drawdown,
    )
