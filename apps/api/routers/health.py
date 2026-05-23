"""
apps/api/routers/health.py
---------------------------
S47-6 (Sprint 47) -- background-task health diagnostics endpoint.

Surfaces the running state of every long-lived async task started by
the FastAPI lifespan so operators can confirm post-deploy that:

  * the v2 feature pipeline's history cache is being warmed
    (HistoryCacheWarmer last_run_at_unix is recent)
  * RetrainingService is polling
  * the active-run task count matches expectations

NOTE: ``equity_prune_task`` (registered in
``BackgroundTaskRegistry.equity_prune_task``) is not surfaced here.
Add it in a follow-up sprint if operators need to monitor it
independently.

API-key auth required (operational endpoint -- mirrors /api/v1/runs).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

# CR-001 (Sprint 47): no ``tags=`` on the APIRouter itself -- the tag is
# applied exactly once at the ``include_router`` call in main.py.  Two
# sources cause duplicated entries in the OpenAPI schema and break
# Swagger UI grouping.
router = APIRouter(prefix="/health")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HistoryCacheWarmerHealth(BaseModel):
    """Health snapshot for the FGI + BTC-dominance history warmer."""

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

    history_cache_warmer: HistoryCacheWarmerHealth
    retraining_service: RetrainingServiceHealth
    active_runs: ActiveRunsHealth
    fx_cache_warmer: FxCacheWarmerHealth


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

    # History cache warmer (S47-1).  ``getattr`` guards survive partial
    # test fixtures where a MagicMock omits some of the diagnostics fields.
    warmer = (
        container.background_tasks.history_cache_warmer
        if container is not None
        else None
    )
    if warmer is not None:
        warmer_health = HistoryCacheWarmerHealth(
            configured=True,
            running=bool(getattr(warmer, "running", False)),
            last_run_at_unix=getattr(warmer, "last_run_at", None),
            last_fgi_points=int(getattr(warmer, "last_fgi_points", 0)),
            last_btc_dom_points=int(getattr(warmer, "last_btc_dom_points", 0)),
        )
    else:
        warmer_health = HistoryCacheWarmerHealth(configured=False, running=False)

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

    return BackgroundHealthResponse(
        history_cache_warmer=warmer_health,
        retraining_service=rsvc_health,
        active_runs=active,
        fx_cache_warmer=fx_warmer_health,
    )
