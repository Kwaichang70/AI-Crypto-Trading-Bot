"""
apps/api/services/promotion_gate.py
-------------------------------------
Paper->live promotion gate service (Sprint 50 Cycle 5).

Evaluates whether a stopped paper run has accumulated sufficient evidence
to be promoted to a live run.  Gate criteria (user-locked):

    trade_count >= settings.min_paper_trades_for_promotion  (default 50)
    runtime_days >= settings.min_paper_runtime_days         (default 7.0)

Performance metrics (Sharpe, drawdown, win rate) are deliberately excluded
from the gate check.  The operator decides performance acceptability;
the gate only enforces data-volume readiness so the live run is not
launched with insufficient signal.

Design notes
------------
- ``evaluate_paper_run_eligibility`` accepts a raw RunORM so callers can
  pass an already-fetched ORM row without a second DB round-trip.
- Eligibility check is read-only (SELECT only); the mutation (create live
  run + FK) lives in the router.
- ``reasons`` uses symbolic strings, not human sentences, so the frontend
  can localise the messages without string-parsing.

Notes
-----
Paper run ``n_closed_trades`` (Sprint 49 M3) is always NULL for live paper
runs -- the column is only backfilled post-stop.  This function therefore
always issues a COUNT query against :class:`TradeORM` to get the current
authoritative closed-trade count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import RunORM, TradeORM

__all__ = ["PromotionEligibility", "evaluate_paper_run_eligibility"]

logger = structlog.get_logger(__name__)


@dataclass
class PromotionEligibility:
    """Result of the paper->live promotion gate check.

    Attributes
    ----------
    eligible:
        True when ALL gate criteria are satisfied.
    trade_count:
        Actual number of closed trades recorded for this paper run.
    runtime_days:
        Actual runtime in fractional days (stopped_at - started_at).
        Zero when the run has not stopped yet.
    reasons:
        List of symbolic failure codes when ``eligible=False``.
        Empty when ``eligible=True``.
        Possible values:
            "run_not_found"             -- run_id does not exist
            "wrong_run_mode"            -- run is not mode='paper'
            "run_not_stopped"           -- run is still active (status='running')
            "trade_count_below_min"     -- fewer than min_paper_trades_for_promotion
            "runtime_below_min"         -- shorter than min_paper_runtime_days
    """

    eligible: bool
    trade_count: int
    runtime_days: float
    reasons: list[str] = field(default_factory=list)


async def evaluate_paper_run_eligibility(
    db: AsyncSession,
    run_orm: RunORM,
    *,
    min_trades: int,
    min_runtime_days: float,
) -> PromotionEligibility:
    """Evaluate gate criteria for a paper run.

    Parameters
    ----------
    db:
        Active async session (read-only; no writes performed here).
    run_orm:
        Pre-fetched RunORM row for the paper run to evaluate.
    min_trades:
        Minimum closed-trade count threshold.
    min_runtime_days:
        Minimum runtime threshold in fractional days.

    Returns
    -------
    PromotionEligibility
        Fully-populated result; ``eligible=True`` only when every criterion
        is satisfied.

    Notes
    -----
    Paper run ``n_closed_trades`` (Sprint 49 M3) is always NULL for live paper
    runs -- the column is only backfilled post-stop.  This function therefore
    always issues a COUNT query against :class:`TradeORM` to get the current
    authoritative closed-trade count.
    """
    reasons: list[str] = []

    # Gate: mode must be paper
    if run_orm.run_mode != "paper":
        reasons.append("wrong_run_mode")
        return PromotionEligibility(
            eligible=False,
            trade_count=0,
            runtime_days=0.0,
            reasons=reasons,
        )

    # Gate: run must be stopped (not still running, not in error).
    # §Fix-CR5-006: return early — continuing to evaluate threshold reasons
    # for an active run produces misleading output (partial trade count,
    # incomplete runtime).
    if run_orm.status not in ("stopped", "archived"):
        reasons.append("run_not_stopped")
        return PromotionEligibility(
            eligible=False,
            trade_count=0,
            runtime_days=0.0,
            reasons=reasons,
        )

    # Compute runtime
    if run_orm.stopped_at is not None:
        runtime_seconds = (
            run_orm.stopped_at - run_orm.started_at
        ).total_seconds()
    else:
        # Still running -- compute against now for informational value
        runtime_seconds = (
            datetime.now(tz=UTC) - run_orm.started_at
        ).total_seconds()
    runtime_days = runtime_seconds / 86_400.0

    # Count closed trades from the DB (source of truth; paper engine
    # flushes incrementally so the count is current even for long runs).
    # Every TradeORM row represents a completed round-trip (entry + exit),
    # so all rows for this run_id count as closed trades.
    trade_count_result = await db.execute(
        select(func.count(TradeORM.id)).where(
            TradeORM.run_id == run_orm.id,
        )
    )
    trade_count: int = trade_count_result.scalar_one()

    # Evaluate thresholds
    if trade_count < min_trades:
        reasons.append("trade_count_below_min")
    if runtime_days < min_runtime_days:
        reasons.append("runtime_below_min")

    eligible = len(reasons) == 0

    logger.debug(
        "promotion_gate.evaluated",
        run_id=str(run_orm.id),
        eligible=eligible,
        trade_count=trade_count,
        runtime_days=round(runtime_days, 1),
        reasons=reasons,
    )

    return PromotionEligibility(
        eligible=eligible,
        trade_count=trade_count,
        runtime_days=runtime_days,
        reasons=reasons,
    )
