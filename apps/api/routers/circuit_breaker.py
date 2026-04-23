"""
apps/api/routers/circuit_breaker.py
------------------------------------
Circuit breaker management endpoints.

GET  /api/v1/runs/{run_id}/circuit-breaker       — current state
POST /api/v1/runs/{run_id}/circuit-breaker/reset  — reset tripped breaker
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from api.db.session import get_db

router = APIRouter()
log = structlog.get_logger(__name__)


def _get_circuit_breaker(run_id: str) -> Any:
    """Retrieve the circuit breaker for a running engine.

    Reads from :mod:`api.services.run_orchestrator` — the registry was moved
    there in Sprint 40 Stap 2b; ``api.routers.runs`` re-exports the dict but
    mypy strict does not see it as an explicit export.
    """
    from api.services.run_orchestrator import _RUN_ENGINES

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
async def reset_circuit_breaker(
    run_id: str,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
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

    # SEC-002: persist audit record — manually resetting a tripped breaker
    # re-enables order submission on a run that has already breached its
    # risk envelope, so the action must leave a durable forensic trail.
    from api.services.audit_log import record_audit_event

    await record_audit_event(
        db,
        event_type="circuit_breaker_reset",
        resource_type="circuit_breaker",
        resource_id=run_id,
        request=request,
        payload={"run_id": run_id},
    )

    result: dict[str, Any] = breaker.state.model_dump(mode="json")
    return result
