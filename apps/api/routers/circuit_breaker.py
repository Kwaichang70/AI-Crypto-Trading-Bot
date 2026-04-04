"""
apps/api/routers/circuit_breaker.py
------------------------------------
Circuit breaker management endpoints.

GET  /api/v1/runs/{run_id}/circuit-breaker       — current state
POST /api/v1/runs/{run_id}/circuit-breaker/reset  — reset tripped breaker
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

import structlog

router = APIRouter()
log = structlog.get_logger(__name__)


def _get_circuit_breaker(run_id: str) -> Any:
    """Retrieve the circuit breaker for a running engine.

    Lazy-imports _RUN_ENGINES from runs.py to avoid circular imports.
    """
    from api.routers.runs import _RUN_ENGINES

    engine = _RUN_ENGINES.get(run_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"No running engine for run {run_id}")

    breaker = getattr(engine, "circuit_breaker", None)
    if breaker is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} has no circuit breaker configured")

    return breaker


@router.get("/runs/{run_id}/circuit-breaker")
async def get_circuit_breaker_state(run_id: str) -> dict[str, Any]:
    """Return the current circuit breaker state for a running engine."""
    breaker = _get_circuit_breaker(run_id)
    result: dict[str, Any] = breaker.state.model_dump(mode="json")
    return result


@router.post("/runs/{run_id}/circuit-breaker/reset")
async def reset_circuit_breaker(run_id: str) -> dict[str, Any]:
    """Reset a tripped circuit breaker.

    Returns 409 if the breaker is not currently tripped.
    """
    breaker = _get_circuit_breaker(run_id)

    if not breaker.is_tripped:
        raise HTTPException(
            status_code=409,
            detail="Circuit breaker is not tripped; nothing to reset.",
        )

    breaker.reset()
    log.warning(
        "circuit_breaker.reset_via_api",
        run_id=run_id,
    )
    result: dict[str, Any] = breaker.state.model_dump(mode="json")
    return result
