"""
packages/trading/strategy_availability.py
------------------------------------------
Central, code-resident registry of which RunModes each strategy may run in
(Sprint 51 Cycle 2 — strategy-availability lockdown).

Single source of truth
-----------------------
Both run-creation (``POST /api/v1/runs``) and orphan recovery
(``recover_orphaned_runs``) consult :func:`is_mode_allowed` so a demoted
strategy can never be started in paper or live mode through any code path.
The paper->live promotion endpoint adds an outer guard on the same function.

Fail-closed
-----------
Any strategy name NOT present in ``_AVAILABILITY`` resolves to
``_DEFAULT_AVAILABILITY`` (backtest-only, DEMOTED).  A typo or a newly added
strategy that has not been explicitly classified therefore cannot reach
paper/live by accident.

No database migration is involved: this metadata lives in code and is
reviewed via the agent workflow.  The rationale for each demotion comes from
the Sprint 48 ModelAnalyse / strategy-performance diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from common.types import RunMode

__all__ = [
    "StrategyStatus",
    "StrategyAvailability",
    "get_availability",
    "is_mode_allowed",
]


class StrategyStatus(StrEnum):
    """Lifecycle status that governs which modes a strategy may run in."""

    ACTIVE = "active"
    DEMOTED = "demoted"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class StrategyAvailability:
    """Immutable availability record for a single strategy.

    Attributes
    ----------
    allowed_modes:
        The set of :class:`RunMode` values the strategy may run in.
    status:
        Lifecycle status (see :class:`StrategyStatus`).
    demotion_reason:
        Human-readable, concise reason a strategy is restricted.  Empty for
        ACTIVE strategies.
    promotion_requirements:
        Symbolic requirement codes that must be satisfied (out of band) before
        a DEMOTED/EXPERIMENTAL strategy may be re-promoted.  Empty for ACTIVE.
    """

    allowed_modes: frozenset[RunMode]
    status: StrategyStatus
    demotion_reason: str = ""
    promotion_requirements: list[str] = field(default_factory=list)


# All three live-eligible modes.
_ALL_MODES: frozenset[RunMode] = frozenset(
    {RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE}
)
# Backtest-only (the fail-closed / demoted restriction).
_BACKTEST_ONLY: frozenset[RunMode] = frozenset({RunMode.BACKTEST})


_AVAILABILITY: dict[str, StrategyAvailability] = {
    # ---- DEMOTED -> backtest-only (Sprint 48 diagnosis) -------------------
    "ma_crossover": StrategyAvailability(
        allowed_modes=_BACKTEST_ONLY,
        status=StrategyStatus.DEMOTED,
        demotion_reason=(
            "Trend-following underperformance in the observed non-trending "
            "regime (low win-rate, fee-cost dominance) per Sprint 48."
        ),
        promotion_requirements=[
            "oos_walk_forward_pass",
            "backtest_profit_factor_gt_1_2",
            "regime_filter_added",
        ],
    ),
    "breakout": StrategyAvailability(
        allowed_modes=_BACKTEST_ONLY,
        status=StrategyStatus.DEMOTED,
        demotion_reason=(
            "Breakout underperformance in the observed regime (frequent "
            "false breakouts, negative net PnL) per Sprint 48."
        ),
        promotion_requirements=[
            "oos_walk_forward_pass",
            "backtest_profit_factor_gt_1_2",
            "regime_filter_added",
        ],
    ),
    "model_strategy": StrategyAvailability(
        allowed_modes=_BACKTEST_ONLY,
        status=StrategyStatus.DEMOTED,
        demotion_reason=(
            "No reliable trained model artefact on disk; runs created with "
            "empty model_path and no sidecar schema per Sprint 48."
        ),
        promotion_requirements=[
            "trained_model_artefact_on_disk",
            "sidecar_schema_version_present",
            "end_to_end_smoke_test_pass",
            "oos_walk_forward_pass",
        ],
    ),
    # ---- ACTIVE -> all three modes ---------------------------------------
    "dca_rsi_hybrid": StrategyAvailability(
        allowed_modes=_ALL_MODES,
        status=StrategyStatus.ACTIVE,
    ),
    "grid_trading": StrategyAvailability(
        allowed_modes=_ALL_MODES,
        status=StrategyStatus.ACTIVE,
    ),
    "rsi_mean_reversion": StrategyAvailability(
        allowed_modes=_ALL_MODES,
        status=StrategyStatus.ACTIVE,
    ),
}


# Fail-closed default for any unlisted strategy: backtest-only, demoted.
_DEFAULT_AVAILABILITY: StrategyAvailability = StrategyAvailability(
    allowed_modes=_BACKTEST_ONLY,
    status=StrategyStatus.DEMOTED,
    demotion_reason=(
        "Strategy is not in the availability allow-list; defaulting to "
        "backtest-only (fail-closed)."
    ),
    promotion_requirements=["add_to_strategy_availability_registry"],
)


def get_availability(strategy_name: str) -> StrategyAvailability:
    """Return the availability record for ``strategy_name``.

    Returns :data:`_DEFAULT_AVAILABILITY` (backtest-only, DEMOTED) for any
    name not explicitly registered (fail-closed).
    """
    return _AVAILABILITY.get(strategy_name, _DEFAULT_AVAILABILITY)


def is_mode_allowed(strategy_name: str, mode: RunMode) -> bool:
    """Return True iff ``strategy_name`` is permitted to run in ``mode``.

    ``mode`` may be passed as a ``RunMode`` member or, because ``RunMode`` is a
    ``StrEnum``, as its equivalent string value — both compare and hash
    identically against the ``frozenset[RunMode]`` membership set.
    """
    return mode in get_availability(strategy_name).allowed_modes
