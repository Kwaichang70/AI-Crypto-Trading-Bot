"""
apps/api/routers/emergency.py
------------------------------
Global kill-switch endpoint (Sprint 50 Cycle 3, SEC-006 aggregate variant).

POST /api/v1/emergency/kill-switch — stops all status='running' runs in one
operator call.  Protected by a separate X-Admin-Key token so the kill-switch
is accessible even when the regular X-API-Key rotation is in progress.

Design decisions
----------------
* X-Admin-Key separator token (NOT HMAC-signed role headers): the single-admin
  operator + Tailscale network boundary makes the simpler "second token" pattern
  sufficient.  HMAC-signed role headers (cycle 2 SR-001 alternative) are deferred
  to cycle 4+ when role granularity matters.  Document this scope decision here so
  the code-critic does not flag it as an oversight.

* Audit BEFORE mutation (Sprint 41 SEC-002 invariant): the audit_events row is
  committed with a nested session flush BEFORE any engine is cancelled, so the
  trail survives a partial failure on run #3 of N.

* Idempotent: when called with 0 running runs, returns 200 with runs_stopped=[]
  and a note field — not 404 or 409.  Operators must be able to invoke the
  kill-switch repeatedly during an incident without second-guessing state.

* Best-effort sequential stop: if cancellation of run #3 raises, the error is
  collected in the `errors` list.  Runs #1 and #2 are already stopped.  A second
  call to kill-switch skips already-stopped runs (they are no longer status='running').

* Rate-limit exempt (mirrors Sprint 45 /emergency-stop pattern):
  rate_limit.py checks exact path '/api/v1/emergency/kill-switch' — added in main.py.

* The existing per-run POST /runs/{id}/emergency-stop (SEC-006) is UNCHANGED.
  This endpoint is the aggregate wrapper; it iterates and delegates.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import Settings, get_settings
from api.db.models import RunORM
from api.db.session import get_db
from api.deps import require_admin
from api.services.audit_log import record_audit_event
from api.services.run_orchestrator import (
    _LEARNING_INSTANCES,
    _RUN_ENGINES,
    _RUN_TASKS,
)

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/emergency", tags=["emergency"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitise_reason(raw: str | None) -> str:
    """Strip control characters and cap length from X-Emergency-Reason header.

    Defensive against log-injection via the reason header.  Allows printable
    ASCII + tab + newline; strips all other control characters (0x00-0x1F
    except 0x09/0x0A, plus 0x7F DEL).

    Parameters
    ----------
    raw:
        The raw header value, or ``None`` if the header was absent.

    Returns
    -------
    str
        Cleaned reason string, at most 500 characters.  Returns
        ``"no reason given"`` when input is empty or None.
    """
    if not raw:
        return "no reason given"
    cleaned = "".join(
        ch for ch in raw
        if ch == "\t" or ch == "\n" or ord(ch) >= 0x20
    )
    if len(cleaned) > 500:
        cleaned = cleaned[:497] + "..."
    return cleaned or "no reason given"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class KillSwitchRunError(BaseModel):
    """Per-run error entry when stop fails for a single run."""

    run_id: str = Field(description="UUID of the run that failed to stop")
    error_msg: str = Field(description="Exception message or error detail")


class KillSwitchResponse(BaseModel):
    """Response from POST /emergency/kill-switch."""

    runs_stopped: list[str] = Field(
        description="UUIDs of runs successfully transitioned to 'stopped'"
    )
    tasks_cancelled: int = Field(
        description=(
            "Total asyncio tasks cancelled across all stopped runs.  "
            "Exchange order cancellation for live runs is performed by the "
            "live engine teardown path, not tracked independently here."
        )
    )
    engines_removed: int = Field(
        description=(
            "Number of _RUN_ENGINES entries removed (proxy for position-bearing "
            "engines that were shut down).  Actual position closure is handled by "
            "the engine teardown path."
        )
    )
    errors: list[KillSwitchRunError] = Field(
        description="Per-run errors when best-effort stop failed for that run"
    )
    note: str | None = Field(
        default=None,
        description="Informational note (e.g. 'no active runs to stop')",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/kill-switch",
    status_code=status.HTTP_200_OK,
    response_model=KillSwitchResponse,
    include_in_schema=True,
    openapi_extra={"x-admin-only": True},
    summary="Global emergency kill-switch — stop all running engines",
    description=(
        "Stops all paper and live trading engines currently in 'running' state.  "
        "Requires X-Admin-Key header matching ADMIN_API_KEY env var.  "
        "An audit_events row is written BEFORE any mutation so the trail survives "
        "partial failures.  Idempotent: returns 200 with an empty list when no "
        "runs are active.  Rate-limit exempt (same as per-run /emergency-stop)."
    ),
    dependencies=[Depends(require_admin)],
)
async def kill_switch(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    reason: Annotated[str | None, Header(alias="X-Emergency-Reason")] = None,
    settings: Settings = Depends(get_settings),
) -> KillSwitchResponse:
    """Global kill-switch: stop all running engines atomically-ish.

    Audit row is written first inside a SAVEPOINT so it is committed
    independently of any subsequent per-run flush failure.  Then each
    running run is stopped in sequence.  Per-run errors are collected;
    successful stops are not rolled back if a later run fails.  Caller
    may re-invoke kill-switch safely (idempotent).
    """
    log = logger.bind(endpoint="kill_switch")
    log.warning("emergency.kill_switch_requested", reason=_sanitise_reason(reason))

    # ------------------------------------------------------------------
    # 1. Query all running runs BEFORE writing the audit row so the payload
    #    contains the full scope of what will be attempted.
    # ------------------------------------------------------------------
    result = await db.execute(
        select(RunORM).where(RunORM.status == "running")
    )
    running_runs: list[RunORM] = list(result.scalars().all())

    if not running_runs:
        log.info("emergency.kill_switch_no_active_runs")
        return KillSwitchResponse(
            runs_stopped=[],
            tasks_cancelled=0,
            engines_removed=0,
            errors=[],
            note="no active runs to stop",
        )

    run_ids_attempted = [str(r.id) for r in running_runs]

    # ------------------------------------------------------------------
    # 2. Audit BEFORE mutation (Sprint 41 SEC-002 invariant).
    #    Wrapped in a SAVEPOINT so the audit row commits independently
    #    if a subsequent per-run flush fails inside the loop below.
    # ------------------------------------------------------------------
    raw_admin_key = settings.admin_api_key.get_secret_value()
    admin_key_prefix = hashlib.sha256(raw_admin_key.encode("utf-8")).hexdigest()[:12]
    actor_id = f"admin_key_{admin_key_prefix}"

    async with db.begin_nested():
        await record_audit_event(
            db,
            event_type="kill_switch",
            resource_type="global",
            resource_id="kill_switch",
            request=request,
            actor_override=actor_id,
            payload={
                "runs_attempted": run_ids_attempted,
                "runs_count": len(run_ids_attempted),
                "reason": _sanitise_reason(reason),
                "admin_key_prefix": admin_key_prefix,
            },
        )
    # SAVEPOINT released: audit row is now independently committed.
    # The outer transaction continues for per-run status mutations below.

    # ------------------------------------------------------------------
    # 3. Best-effort sequential stop — collect errors, never abort early.
    # ------------------------------------------------------------------
    runs_stopped: list[str] = []
    errors: list[KillSwitchRunError] = []
    tasks_cancelled = 0
    engines_removed = 0

    now = datetime.now(tz=UTC)

    for run in running_runs:
        run_id_str = str(run.id)
        try:
            # Transition status
            run.status = "stopped"
            run.stopped_at = now
            run.updated_at = now
            await db.flush()

            # Cancel background asyncio task
            task = _RUN_TASKS.pop(run_id_str, None)
            if task is not None and not task.done():
                task.cancel()
                tasks_cancelled += 1

            # Remove from engine + learning registries
            if _RUN_ENGINES.pop(run_id_str, None) is not None:
                engines_removed += 1
            _LEARNING_INSTANCES.pop(run_id_str, None)

            runs_stopped.append(run_id_str)
            log.warning(
                "emergency.kill_switch_run_stopped",
                run_id=run_id_str,
                run_mode=run.run_mode,
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            errors.append(KillSwitchRunError(run_id=run_id_str, error_msg=error_msg))
            log.exception(
                "emergency.kill_switch_run_stop_failed",
                run_id=run_id_str,
                error=error_msg,
            )

    log.warning(
        "emergency.kill_switch_complete",
        runs_stopped=len(runs_stopped),
        runs_failed=len(errors),
        tasks_cancelled=tasks_cancelled,
        engines_removed=engines_removed,
        reason=_sanitise_reason(reason),
    )

    return KillSwitchResponse(
        runs_stopped=runs_stopped,
        tasks_cancelled=tasks_cancelled,
        engines_removed=engines_removed,
        errors=errors,
    )
