"""
apps/api/routers/health.py
---------------------------
S47-6 (Sprint 47) -- background-task health diagnostics endpoint.

Surfaces the running state of every long-lived async task started by
the FastAPI lifespan so operators can confirm post-deploy that:

  * the live context poller (CF-24-01 P5) is alive and each provider series is
    ``fresh``, ``stale``, ``invalid``, ... (closed vocabulary, server-written)
  * RetrainingService is polling
  * the active-run task count matches expectations

NOTE: ``equity_prune_task`` (registered in
``BackgroundTaskRegistry.equity_prune_task``) is not surfaced here.
Add it in a follow-up sprint if operators need to monitor it
independently.

WP1.8b (P-06): ``orphan_repeater_task`` (S8's 15-minute
orphan-holding-unprotected alert loop, ``run_recovery.orphan_holding_repeater``)
IS surfaced -- a plain ``asyncio.Task`` with no richer diagnostics of its
own, so "is it still alive" (``running``) is the entire cheap signal
worth exposing here.

API-key auth required (operational endpoint -- mirrors /api/v1/runs).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from api.services.context_poller import OUTCOMES, REASONS, STATUSES, ContextPoller
from data.live_context import LIVE_BOOK

# CR-001 (Sprint 47): no ``tags=`` on the APIRouter itself -- the tag is
# applied exactly once at the ``include_router`` call in main.py.  Two
# sources cause duplicated entries in the OpenAPI schema and break
# Swagger UI grouping.
router = APIRouter(prefix="/health")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HistoryCacheWarmerHealth(BaseModel):
    """DEPRECATED compat key: the warmer was replaced by the context poller (CF-24-01 P5)
    and is always reported as not configured. Read ``context_poller`` instead."""

    configured: bool = Field(description="True if the warmer was instantiated at startup")
    running: bool = Field(description="True if the warmer's asyncio.Task is alive")
    last_run_at_unix: float | None = Field(
        default=None,
        description=(
            "Wall-clock Unix epoch (time.time()) of the last completed "
            "tick, or None if the warmer has not run since startup.  "
            "Comparable across process restarts."
        ),
    )
    last_fgi_points: int = Field(
        default=0,
        description="Count of FGI snapshots fetched on the last tick",
    )
    last_btc_dom_points: int = Field(
        default=0,
        description="Count of BTC dominance points fetched on the last tick",
    )


class ContextSeriesHealth(BaseModel):
    """One provider series of the live context book (CF-24-01 P5). Closed vocabularies only."""

    series: str
    status: str = Field(description="fresh | stale | invalid | unavailable | not_configured | ...")
    last_outcome: str | None = Field(default=None, description="closed poll outcome code")
    last_reason: str | None = Field(default=None, description="closed rejection reason code")
    rows: int = 0
    age_seconds: float | None = None
    failed_cycles: int = 0
    task_alive: bool = False

    @field_validator("status")
    @classmethod
    def _closed_status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError("status outside the closed vocabulary")
        return v

    @field_validator("last_outcome")
    @classmethod
    def _closed_outcome(cls, v: str | None) -> str | None:
        if v is not None and v not in OUTCOMES:
            raise ValueError("outcome outside the closed vocabulary")
        return v

    @field_validator("last_reason")
    @classmethod
    def _closed_reason(cls, v: str | None) -> str | None:
        if v is not None and v not in REASONS and v != "unknown":
            raise ValueError("reason outside the closed vocabulary")
        return v


class ContextPollerHealth(BaseModel):
    """The live context poller (CF-24-01 P5, RK-P4A-04: a dead poller is visible here)."""

    configured: bool = Field(description="True if the poller was started with a provider")
    running: bool = Field(description="True if the supervisor task is alive")
    alive: bool = Field(description="running AND a supervisor heartbeat within 3 ticks")
    heartbeat_age_seconds: float | None = None
    regressed: bool = Field(
        default=False,
        description=(
            "Wall clock behind the book's high-water mark by more than 5 min. The mark is "
            "raised by every engine read (consumer and non-consumer), so while this is true "
            "no series is reported fresh and engines read an empty book."
        ),
    )
    series: list[ContextSeriesHealth] = Field(default_factory=list)


class RetrainingServiceHealth(BaseModel):
    """Health snapshot for the ML retraining poller."""

    configured: bool = Field(description="True if the service was instantiated at startup")
    running: bool = Field(description="True if the poll task is alive")
    min_trades_for_retrain: int | None = Field(default=None)
    check_interval_seconds: int | None = Field(default=None)


class ActiveRunsHealth(BaseModel):
    """Aggregate snapshot of in-process trading runs."""

    count: int = Field(description="Number of active runs in the registry")
    run_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of active runs (UUIDs).",
    )


class OrphanRepeaterHealth(BaseModel):
    """Health snapshot for the WP1.8 S8 orphan-holding-unprotected alert
    repeater. A plain ``asyncio.Task`` (no internal diagnostics), so
    ``running`` is the only signal cheaply available here (P-06)."""

    configured: bool = Field(description="True if the repeater task was scheduled at startup")
    running: bool = Field(description="True if the repeater's asyncio.Task is alive")


class FxCacheWarmerHealth(BaseModel):
    """Health snapshot for the FX rate cache warmer (M6 MVP stub)."""

    configured: bool = Field(description="True if FxCacheWarmer was instantiated at startup")
    running: bool = Field(description="True if the warmer's asyncio.Task is alive")
    last_run_at_unix: float | None = Field(
        default=None,
        description="Wall-clock Unix epoch of the last tick, or None pre-first-tick.",
    )
    last_rates_cached: int = Field(
        default=0,
        description="Number of FX rate entries cached on the last tick (0 in M6 MVP stub).",
    )


class BackgroundHealthResponse(BaseModel):
    """Top-level response from GET /api/v1/health/background."""

    history_cache_warmer: HistoryCacheWarmerHealth  # deprecated, constant
    context_poller: ContextPollerHealth
    retraining_service: RetrainingServiceHealth
    active_runs: ActiveRunsHealth
    fx_cache_warmer: FxCacheWarmerHealth
    orphan_repeater: OrphanRepeaterHealth


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/background",
    response_model=BackgroundHealthResponse,
    summary="Background-task health snapshot",
)
async def background_health(request: Request) -> BackgroundHealthResponse:
    """Return running state + diagnostics for every long-lived background task."""
    container: Any = getattr(request.app.state, "container", None)

    # Deprecated compat key (CF-24-01 P5): the warmer no longer exists.
    warmer_health = HistoryCacheWarmerHealth(configured=False, running=False)

    # Live context poller (CF-24-01 P5). The book is read through ``freshness`` only: a
    # health probe must not move the high-water mark or count a regressed read. ``isinstance``
    # keeps a partial test double from being reported as a real poller.
    poller = container.background_tasks.context_poller if container is not None else None
    if isinstance(poller, ContextPoller):
        now_ms = poller.now_ms()  # an exact int (CF-2401-62)
        snap = poller.health(LIVE_BOOK.freshness(now_ms), now_ms)
        poller_health = ContextPollerHealth(
            configured=snap.configured,
            running=snap.running,
            alive=snap.alive,
            heartbeat_age_seconds=snap.heartbeat_age_seconds,
            regressed=snap.regressed,
            series=[
                ContextSeriesHealth(
                    series=row.series_id,
                    status=row.status,
                    last_outcome=row.last_outcome,
                    last_reason=row.last_reason,
                    rows=row.rows,
                    age_seconds=row.age_seconds,
                    failed_cycles=row.failed_cycles,
                    task_alive=row.task_alive,
                )
                for row in snap.series
            ],
        )
    else:
        poller_health = ContextPollerHealth(configured=False, running=False, alive=False)

    # RetrainingService (Sprint 23).  CR-003: read PUBLIC properties so
    # renaming the internal ``_min_trades`` attribute would be caught at
    # property-rename time rather than silently returning None.
    rsvc = container.services.retraining_service if container is not None else None
    if rsvc is not None:
        rsvc_health = RetrainingServiceHealth(
            configured=True,
            running=bool(getattr(rsvc, "running", False)),
            min_trades_for_retrain=getattr(rsvc, "min_trades_for_retrain", None),
            check_interval_seconds=getattr(rsvc, "check_interval_seconds", None),
        )
    else:
        rsvc_health = RetrainingServiceHealth(configured=False, running=False)

    # Active runs.
    run_registry = container.run_registry if container is not None else None
    if run_registry is not None:
        run_ids = list(run_registry.active_run_ids())
    else:
        run_ids = []
    active = ActiveRunsHealth(count=len(run_ids), run_ids=run_ids)

    # FX cache warmer (M6 Sprint 49).
    fx_warmer = (
        container.background_tasks.fx_cache_warmer
        if container is not None
        else None
    )
    if fx_warmer is not None:
        fx_warmer_health = FxCacheWarmerHealth(
            configured=True,
            running=bool(getattr(fx_warmer, "running", False)),
            last_run_at_unix=getattr(fx_warmer, "last_run_at", None),
            last_rates_cached=int(getattr(fx_warmer, "last_rates_cached", 0)),
        )
    else:
        fx_warmer_health = FxCacheWarmerHealth(configured=False, running=False)

    # Orphan-holding repeater (WP1.8b P-06). A bare asyncio.Task -- no
    # richer diagnostics object like the warmers above -- so "configured"
    # just means the container registered a task at all.
    orphan_repeater_task = (
        container.background_tasks.orphan_repeater_task if container is not None else None
    )
    if orphan_repeater_task is not None:
        orphan_repeater_health = OrphanRepeaterHealth(
            configured=True,
            running=not orphan_repeater_task.done(),
        )
    else:
        orphan_repeater_health = OrphanRepeaterHealth(configured=False, running=False)

    return BackgroundHealthResponse(
        history_cache_warmer=warmer_health,
        context_poller=poller_health,
        retraining_service=rsvc_health,
        active_runs=active,
        fx_cache_warmer=fx_warmer_health,
        orphan_repeater=orphan_repeater_health,
    )
