"""
apps/api/routers/circuit_breaker.py
------------------------------------
Circuit breaker endpoints.

GET  /api/v1/runs/{run_id}/circuit-breaker       -- current (typed) state
POST /api/v1/runs/{run_id}/circuit-breaker/reset -- RETIRED (410 Gone)

WP1.6 (U-1 / K-09): the breaker is data-derived and auto-recovering.  HALT,
REDUCE and DAILY_LIMIT clear by themselves as the drawdown recovers (HALT
after N bars below 0.9 x the limit, stepping down to REDUCE) or the UTC day
rolls, so there is nothing an operator may reset.  The reset route is kept
only so existing callers get an explicit 410 instead of a 404 / 405; it
performs no state change, needs no DB session and writes no audit row.  To
block entries manually use the kill switch; to exit use flatten.  The route
itself is removed in CF-29.

Authentication is applied at the router level in ``main.py``
(``require_api_key``) and is unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import structlog

from trading.safety import CircuitBreaker, CircuitBreakerState

router = APIRouter()
log = structlog.get_logger(__name__)

#: Fixed detail of the retired reset route (asserted verbatim by tests).
RESET_RETIRED_DETAIL: str = (
    "circuit_breaker_reset_retired: HALT auto-recovers (WP1.6/U-1); "
    "use the kill switch to block entries or flatten to exit"
)


def _get_circuit_breaker(run_id: str) -> CircuitBreaker:
    """Retrieve the circuit breaker for a running engine.

    Reads from :mod:`api.services.run_orchestrator` — the registry was moved
    there in Sprint 40 Stap 2b; ``api.routers.runs`` re-exports the dict but
    mypy strict does not see it as an explicit export.
    """
    from api.services.run_orchestrator import _RUN_ENGINES

    engine = _RUN_ENGINES.get(run_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"No running engine for run {run_id}")

    breaker: CircuitBreaker | None = getattr(engine, "circuit_breaker", None)
    if breaker is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} has no circuit breaker configured")

    return breaker


@router.get("/runs/{run_id}/circuit-breaker", response_model=CircuitBreakerState)
async def get_circuit_breaker_state(run_id: str) -> CircuitBreakerState:
    """Return the current circuit breaker state for a running engine.

    The read is atomic (single worker, no awaits in ``CircuitBreaker.state``).
    The payload is an allow-list: level, active conditions, hysteresis
    progress, thresholds and the last evaluation's run-level figures.
    """
    return _get_circuit_breaker(run_id).state


@router.post(
    "/runs/{run_id}/circuit-breaker/reset",
    status_code=410,
    response_model=None,
)
async def reset_circuit_breaker(run_id: str) -> None:
    """Retired (WP1.6): always 410 Gone; no state change, no audit row."""
    log.info("circuit_breaker.reset_retired_called", run_id=run_id)
    raise HTTPException(status_code=410, detail=RESET_RETIRED_DETAIL)
