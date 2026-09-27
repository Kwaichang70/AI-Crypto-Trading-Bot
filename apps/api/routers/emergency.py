"""
apps/api/routers/emergency.py
------------------------------
Global kill-switch endpoints.

WP1.7a (reports/vp2-wp1.7/synthesis-spec.md, SY-01/SY-08/SY-09) rewrites
the Sprint 50 Cycle 3 kill switch from "stop every running run" into a
persisted LATCH that blocks new entries while every run keeps running
(D3): the task stays alive, BUYs are dropped, and exits (strategy SELLs,
brackets, trailing stops) keep protecting any open position exactly as
before. Nothing is cancelled and no engine is removed from the
registries -- ``trigger_kill_switch`` on each engine's risk manager is
the entire mutation.

Round 2 (security round 1, WP17a-S-01/S-04/S-09/S-11/S-12/S-17,
WP17a-C-02/C-03) hardens the ordering and auth guarantees:

* **S-01:** the handler's LITERAL FIRST statements (before any ``await``,
  before any DB access) are ``kill_switch.latch_in_memory(...)`` and a
  synchronous loop that latches every engine currently in
  ``_RUN_ENGINES`` -- a DB outage, a slow ``FOR UPDATE`` wait, or any
  other persistence failure below can then never leave a running engine
  unlatched. All DB work is wrapped in ``try/except Exception`` with a
  same-session rollback and a fresh-session retry (S-12); on total
  failure the response is still 200 with ``latch_persisted=false``,
  never a 500.
* **S-09:** the optional kill-switch flatten pass persists
  ``flatten_incomplete`` + a ``run_flatten`` audit row per run, in its
  own short transaction, and reports the TRUE per-run persistence result
  via ``FlattenResult.latch_persisted``.
* **S-11/C-02:** ``GET /kill-switch`` requires ``X-API-Key``
  (``require_api_key``); the rate-limit exemption in ``rate_limit.py``
  is POST-only.
* **S-12:** the clear audit row is written in the SAME transaction as
  the state clear, non-swallowing, before the single commit (see
  ``kill_switch.clear``'s ``before_commit`` hook) -- a failure anywhere
  rolls back and returns 503, never a partially-applied clear.
* **S-17:** both clear endpoints' operator-supplied ``reason`` is
  sanitised identically to the press header.
* **C-03:** every response model here shares the project-wide
  ``API_MODEL_CONFIG`` (camelCase) -- previously ``flatten_results``'
  nested ``FlattenResultResponse`` values were camelCase while every
  top-level field was snake_case in the SAME payload.

Round 3 (security round 2, WP17a-S-R2-01/S-R2-02) closes two fail-open
races found by re-auditing round 2 on real Postgres:

* **S-R2-01:** engine latching/unlatching around a press or clear now
  happens EXCLUSIVELY inside ``kill_switch.activate``/``kill_switch.clear``
  themselves (both under the SAME module lock as the mirror flip) --
  this router no longer has its own post-``clear()`` engine-unlatch
  loop, which is what let a concurrent press/clear pair leave an
  engine disagreeing with the mirror/DB. The router still does its
  own synchronous pre-latch (S-01) before any ``await``; that flip is
  provisional and always superseded by ``activate()``'s own latching
  once it acquires the lock.
* **S-R2-02:** the kill-switch flatten pass now latches
  ``'flatten_incomplete'`` on the run's in-memory engine unconditionally
  whenever the flatten result is incomplete -- even if
  ``_persist_run_flatten`` itself fails -- so a later global clear can
  never re-enable BUYs on a run whose flatten never finished.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import require_api_key
from api.config import Settings, get_settings
from api.db.models import RunORM
from api.db.session import get_db, get_session_factory
from api.deps import require_admin
from api.schemas import API_MODEL_CONFIG, FlattenResultResponse
from api.services import kill_switch as _kill_switch
from api.services.audit_log import (
    record_audit_event,
    record_audit_event_strict,
    sanitise_reason,
)
from api.services.run_orchestrator import _RUN_ENGINES
from trading.strategy_engine import FlattenResult, build_synthetic_flatten_result

__all__ = ["router"]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/emergency", tags=["emergency"])

#: WP1.7a round 2 (S-05, mirrored here for the kill-switch flatten pass):
#: the outer safety-net budget is the per-run flatten timeout plus a
#: fixed margin, so a hung flatten() can never keep this endpoint (or
#: the emergency-stop endpoint, which uses the same margin) open forever.
_FLATTEN_TIMEOUT_S = 30.0
_FLATTEN_OUTER_MARGIN_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_result_to_response(result: FlattenResult) -> FlattenResultResponse:
    return FlattenResultResponse.model_validate(result, from_attributes=True)


async def _persist_run_flatten(
    run_id_str: str,
    *,
    trigger: str,
    result: FlattenResult,
    actor_id: str,
) -> bool:
    """WP1.7a round 2 (S-09): persist a per-run outcome from the
    kill-switch's own flatten pass, in a SHORT, INDEPENDENT transaction
    (the runs this loop iterates over are not the same row the main
    handler's own ``db`` session may still be holding, and multiple runs
    are flattened concurrently via ``asyncio.gather`` -- each needs its
    own session).

    Always writes a ``run_flatten`` audit row (phase reflects
    ``result.complete``); when incomplete, also persists
    ``entries_latch_reason='flatten_incomplete'`` on that run's row.

    Returns
    -------
    bool:
        Whether this run's own persistence succeeded -- becomes
        ``result.latch_persisted`` for the caller.
    """
    try:
        factory = get_session_factory()
        async with factory() as run_db:
            if not result.complete:
                now = datetime.now(tz=UTC)
                await run_db.execute(
                    update(RunORM)
                    .where(RunORM.id == uuid.UUID(run_id_str))
                    .values(
                        entries_latch_reason="flatten_incomplete",
                        entries_latched_at=now,
                        updated_at=now,
                    )
                )
            await record_audit_event(
                run_db,
                event_type="run_flatten",
                resource_type="run",
                resource_id=run_id_str,
                request=None,
                actor_override=actor_id,
                payload={
                    "phase": "completed" if result.complete else "incomplete",
                    "trigger": trigger,
                    "outcome": result.outcome,
                    "symbols": [
                        {"symbol": r.symbol, "status": r.status, "cause": r.cause}
                        for r in result.symbols
                    ],
                },
            )
            await run_db.commit()
        return True
    except Exception:
        logger.critical(
            "emergency.kill_switch_flatten_persist_failed",
            run_id=run_id_str,
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Request / response schemas -- all share API_MODEL_CONFIG (C-03): every
# nested FlattenResultResponse already does, so the top level must too.
# ---------------------------------------------------------------------------


class KillSwitchRequest(BaseModel):
    """Optional body for POST /emergency/kill-switch."""

    model_config = API_MODEL_CONFIG

    flatten: bool = False


class KillSwitchRunError(BaseModel):
    """Per-run error entry when the optional flatten pass fails for a run."""

    model_config = API_MODEL_CONFIG

    run_id: str
    error: str


class KillSwitchResponse(BaseModel):
    """Response from POST /emergency/kill-switch (WP1.7a shape, SY-08)."""

    model_config = API_MODEL_CONFIG

    latched: bool
    latch_persisted: bool
    since: datetime | None
    runs_latched: list[str] = Field(default_factory=list)
    orphaned_live_run_ids: list[str] = Field(default_factory=list)
    resuming_run_ids: list[str] = Field(default_factory=list)
    flatten_results: dict[str, FlattenResultResponse] = Field(default_factory=dict)
    errors: list[KillSwitchRunError] = Field(default_factory=list)


class KillSwitchClearRequest(BaseModel):
    """Body for POST /emergency/kill-switch/clear."""

    model_config = API_MODEL_CONFIG

    reason: str = Field(min_length=3, max_length=500)

    @field_validator("reason")
    @classmethod
    def _sanitise(cls, v: str) -> str:
        return sanitise_reason(v)


class KillSwitchRunKeptLatched(BaseModel):
    model_config = API_MODEL_CONFIG

    run_id: str
    reasons: list[str]


class KillSwitchClearResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    was_latched: bool
    runs_unlatched: list[str] = Field(default_factory=list)
    runs_kept_latched: list[KillSwitchRunKeptLatched] = Field(default_factory=list)


class KillSwitchStatusResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    latched: bool
    since: datetime | None
    reason: str | None
    source: str  # "db" | "unknown"


# ---------------------------------------------------------------------------
# POST /emergency/kill-switch
# ---------------------------------------------------------------------------


@router.post(
    "/kill-switch",
    status_code=status.HTTP_200_OK,
    response_model=KillSwitchResponse,
    include_in_schema=True,
    openapi_extra={"x-admin-only": True},
    summary="Global kill switch — latch entries on every running engine",
    description=(
        "Latches every running paper/live engine (blocks new BUYs; exits "
        "keep running, D3) -- it stops nothing and cancels no task. "
        "Requires X-Admin-Key. The very first thing this endpoint does, "
        "before ANY database access, is latch memory and every "
        "registered engine directly (S-01) -- a database outage or a "
        "slow row-lock wait can delay persistence and the audit trail, "
        "but can never leave a running engine unlatched. Idempotent. "
        "Optional body {'flatten': true} additionally flattens every "
        "newly-latched running engine in parallel (WP17-R-05); 'flatten' "
        "defaults to false (D3: flatten is always an explicit, separate "
        "action). Rate-limit exempt."
    ),
    dependencies=[Depends(require_admin)],
)
async def kill_switch(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    reason: Annotated[str | None, Header(alias="X-Emergency-Reason")] = None,
    settings: Settings = Depends(get_settings),
    body: Annotated[KillSwitchRequest | None, Body()] = None,
) -> KillSwitchResponse:
    """Global kill-switch: latch every running engine (D3), never stop one."""
    log = logger.bind(endpoint="kill_switch")
    sanitised_reason = sanitise_reason(reason)
    log.warning("emergency.kill_switch_requested", reason=sanitised_reason)

    flatten_requested = bool(body and body.flatten)

    # ------------------------------------------------------------------
    # S-01: latch memory + every registered engine FIRST -- no ``await``
    # before this point. ``runs_latched`` is derived ENTIRELY from this
    # loop, never from the DB query below, so it is correct even if
    # every subsequent database step fails outright.
    # ------------------------------------------------------------------
    _kill_switch.latch_in_memory(sanitised_reason)
    runs_latched: list[str] = []
    for run_id_str, engine in list(_RUN_ENGINES.items()):
        engine.risk_manager.trigger_kill_switch(_kill_switch.GLOBAL_REASON)
        runs_latched.append(run_id_str)

    log.warning(
        "emergency.kill_switch_latched",
        runs_latched=len(runs_latched),
        reason=sanitised_reason,
    )

    raw_admin_key = settings.admin_api_key.get_secret_value()
    admin_key_prefix = hashlib.sha256(raw_admin_key.encode("utf-8")).hexdigest()[:12]
    actor_id = f"admin_key_{admin_key_prefix}"

    async def _persist_press(session: AsyncSession) -> tuple[bool, list[str], list[str]]:
        """The full DB sequence: candidates query, persist, audit, the
        resuming->orphaned safety moves, and commit -- all in ONE
        transaction on ``session``. Raises on any failure; the caller
        decides whether/how to retry."""
        result = await session.execute(
            select(RunORM)
            .where(RunORM.status.in_(["running", "orphaned", "resuming"]))
            .order_by(RunORM.id)
            .with_for_update()
        )
        candidate_runs: list[RunORM] = list(result.scalars().all())
        running_runs = [r for r in candidate_runs if r.status == "running"]
        resuming_runs = [r for r in candidate_runs if r.status == "resuming"]
        orphaned_ids = [
            str(r.id) for r in candidate_runs if r.status == "orphaned" and r.run_mode == "live"
        ]
        resuming_ids = [str(r.id) for r in resuming_runs]
        run_ids_attempted = [str(r.id) for r in running_runs]

        # WP17a-S-R2-01 (round 3): pass the CURRENT set of registered
        # engines so activate() can re-latch them atomically with the
        # mirror, under its own lock -- see kill_switch.py's docstring.
        persisted_ = await _kill_switch.activate(
            session,
            reason=sanitised_reason,
            actor=actor_id,
            engines=lambda: list(_RUN_ENGINES.values()),
        )

        try:
            async with session.begin_nested():
                await record_audit_event(
                    session,
                    event_type="kill_switch",
                    resource_type="global",
                    resource_id="kill_switch",
                    request=request,
                    actor_override=actor_id,
                    payload={
                        "runs_attempted": run_ids_attempted,
                        "runs_count": len(run_ids_attempted),
                        "orphaned_live_run_ids": orphaned_ids,
                        "resuming_run_ids": resuming_ids,
                        "reason": sanitised_reason,
                        "admin_key_prefix": admin_key_prefix,
                        "flatten_requested": flatten_requested,
                        "latch_persisted": persisted_,
                        "runs_latched": runs_latched,
                    },
                )
        except Exception:
            log.critical("emergency.kill_switch_audit_flush_failed", exc_info=True)

        resuming_now = datetime.now(tz=UTC)
        for run in resuming_runs:
            run.status = "orphaned"
            run.updated_at = resuming_now
            await session.flush()
            try:
                await record_audit_event(
                    session,
                    event_type="run_orphaned",
                    resource_type="run",
                    resource_id=str(run.id),
                    request=request,
                    actor_override=actor_id,
                    payload={"trigger": "kill_switch", "previous_status": "resuming"},
                )
            except Exception:
                log.critical(
                    "emergency.kill_switch_audit_flush_failed",
                    run_id=str(run.id),
                    exc_info=True,
                )

        await session.commit()
        return persisted_, orphaned_ids, resuming_ids

    # ------------------------------------------------------------------
    # S-01/S-12: all DB work is wrapped so a failure never surfaces as a
    # 500 -- on any exception, roll back and retry ONCE in a fresh
    # session (a transient outage that has already recovered by retry
    # time still gets its audit row and resuming->orphaned moves
    # durably applied).
    # ------------------------------------------------------------------
    latch_persisted = False
    orphaned_live_run_ids: list[str] = []
    resuming_run_ids: list[str] = []
    try:
        latch_persisted, orphaned_live_run_ids, resuming_run_ids = await _persist_press(db)
    except Exception:
        log.critical("emergency.kill_switch_db_work_failed", exc_info=True)
        try:
            await db.rollback()
        except Exception:
            log.warning("emergency.kill_switch_rollback_failed", exc_info=True)
        try:
            factory = get_session_factory()
            async with factory() as fresh_db:
                latch_persisted, orphaned_live_run_ids, resuming_run_ids = await _persist_press(
                    fresh_db
                )
        except Exception:
            log.critical("emergency.kill_switch_db_retry_failed", exc_info=True)
            latch_persisted = False

    # ------------------------------------------------------------------
    # Optional flatten pass, run in parallel across every newly-latched
    # running engine (WP17-A-03). Independent of whether the DB work
    # above succeeded -- runs_latched came from the engine loop, not
    # from the DB query.
    # ------------------------------------------------------------------
    flatten_results: dict[str, FlattenResultResponse] = {}
    errors: list[KillSwitchRunError] = []
    if flatten_requested and runs_latched:
        async def _flatten_one(run_id_str: str) -> tuple[str, FlattenResult | None, str | None]:
            engine = _RUN_ENGINES.get(run_id_str)
            if engine is None:
                return run_id_str, None, "engine_not_found"

            task = asyncio.create_task(
                engine.flatten(reason="kill_switch", timeout_s=_FLATTEN_TIMEOUT_S)
            )
            try:
                fr = await asyncio.wait_for(
                    asyncio.shield(task), timeout=_FLATTEN_TIMEOUT_S + _FLATTEN_OUTER_MARGIN_S
                )
            except TimeoutError:
                log.critical("flatten.timeout", run_id=run_id_str, reason="kill_switch")
                fr = build_synthetic_flatten_result(
                    engine, reason="kill_switch", cause="timeout_open"
                )
            except Exception as exc:
                log.exception("emergency.kill_switch_flatten_error", run_id=run_id_str)
                fr = build_synthetic_flatten_result(
                    engine, reason="kill_switch", cause="error", error=str(exc)
                )

            persisted_ = await _persist_run_flatten(
                run_id_str, trigger="kill_switch", result=fr, actor_id=actor_id
            )
            fr.latch_persisted = persisted_
            if not fr.complete:
                # WP17a-S-R2-02 (round 3): latch the in-memory
                # per-run reason unconditionally -- regardless of
                # whether _persist_run_flatten above succeeded --
                # so a later global clear (which only ever removes
                # GLOBAL_REASON/UNKNOWN_STATE_REASON) can never
                # re-enable BUYs on a run whose flatten never
                # finished (fail closed).
                engine.risk_manager.trigger_kill_switch("flatten_incomplete")
                log.critical(
                    "flatten.incomplete",
                    run_id=run_id_str,
                    reason="kill_switch",
                    outcome=fr.outcome,
                )
            return run_id_str, fr, None

        gathered = await asyncio.gather(*(_flatten_one(rid) for rid in runs_latched))
        for run_id_str, fr, err in gathered:
            if fr is not None:
                flatten_results[run_id_str] = _flatten_result_to_response(fr)
            elif err is not None:
                errors.append(KillSwitchRunError(run_id=run_id_str, error=err))

    return KillSwitchResponse(
        latched=True,
        latch_persisted=latch_persisted,
        since=_kill_switch.current_state().since,
        runs_latched=runs_latched,
        orphaned_live_run_ids=orphaned_live_run_ids,
        resuming_run_ids=resuming_run_ids,
        flatten_results=flatten_results,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# POST /emergency/kill-switch/clear
# ---------------------------------------------------------------------------


@router.post(
    "/kill-switch/clear",
    status_code=status.HTTP_200_OK,
    response_model=KillSwitchClearResponse,
    responses={503: {"description": "Persistence write failed -- latch stays active"}},
    summary="Clear the global kill switch",
    description=(
        "Clears the global latch. Requires X-Admin-Key plus a 3-500 "
        "character reason. Writes an audit row in the SAME transaction "
        "as the state clear, before the single commit (S-10/S-12); "
        "returns 503 'latch_clear_not_persisted' if either write fails, "
        "leaving the process latched. NOT rate-limit exempt (SY-09). "
        "Per-engine reasons other than the global one (e.g. a run's own "
        "persisted 'flatten_incomplete' latch) are left untouched -- "
        "clear those individually via POST /runs/{id}/entries-latch/clear."
    ),
    dependencies=[Depends(require_admin)],
)
async def kill_switch_clear(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    body: KillSwitchClearRequest,
) -> KillSwitchClearResponse:
    log = logger.bind(endpoint="kill_switch_clear")
    was_latched = _kill_switch.is_active()

    # get_settings() is called directly here (not injected via
    # Depends(get_settings) as a parameter default) to avoid a second
    # B008 lint instance alongside kill_switch()'s pre-existing one --
    # functionally identical (get_settings is itself lru_cache'd).
    settings = get_settings()
    raw_admin_key = settings.admin_api_key.get_secret_value()
    admin_key_prefix = hashlib.sha256(raw_admin_key.encode("utf-8")).hexdigest()[:12]
    actor_id = f"admin_key_{admin_key_prefix}"

    async def _write_audit() -> None:
        # S-10/S-12: non-swallowing -- a failure here propagates OUT of
        # kill_switch.clear()'s own try/except, which rolls back and
        # raises KillSwitchClearError, so the state clear is undone too.
        await record_audit_event_strict(
            db,
            event_type="kill_switch_cleared",
            resource_type="global",
            resource_id="kill_switch",
            request=request,
            actor_override=actor_id,
            payload={"reason": body.reason, "was_latched": was_latched},
        )

    try:
        # WP17a-S-R2-01 (round 3): engine unlatching now happens
        # INSIDE clear(), atomically with the mirror flip, under its
        # own lock -- passing engines here (rather than looping over
        # _RUN_ENGINES ourselves afterwards, as round 2 did) is what
        # closes the press/clear race (see kill_switch.py docstring).
        outcome = await _kill_switch.clear(
            db,
            reason=body.reason,
            actor=actor_id,
            before_commit=_write_audit,
            engines=lambda: list(_RUN_ENGINES.values()),
        )
    except _kill_switch.KillSwitchClearError as exc:
        log.critical("emergency.kill_switch_clear_persist_failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "latch_clear_not_persisted"},
        ) from exc

    # S-04: clear() removes BOTH reasons apply_latch() could ever have
    # added (GLOBAL_REASON and, for an engine that booted before load()
    # ever ran, UNKNOWN_STATE_REASON) from EVERY registered engine --
    # runs_kept_latched lists every engine still latched afterwards for
    # ANY reason (e.g. its own unrelated per-run flatten_incomplete).
    runs_unlatched = outcome.runs_unlatched
    runs_kept_latched = [
        KillSwitchRunKeptLatched(run_id=run_id_str, reasons=reasons)
        for run_id_str, reasons in outcome.runs_kept_latched
    ]

    log.warning(
        "emergency.kill_switch_cleared",
        was_latched=was_latched,
        runs_unlatched=len(runs_unlatched),
        runs_kept_latched=len(runs_kept_latched),
    )

    return KillSwitchClearResponse(
        was_latched=was_latched,
        runs_unlatched=runs_unlatched,
        runs_kept_latched=runs_kept_latched,
    )


# ---------------------------------------------------------------------------
# GET /emergency/kill-switch  -- status, for the S13 UI badge
# ---------------------------------------------------------------------------


@router.get(
    "/kill-switch",
    status_code=status.HTTP_200_OK,
    response_model=KillSwitchStatusResponse,
    summary="Global kill-switch status",
    description="Read-only status for the UI's latch badge. Requires X-API-Key (not admin).",
    dependencies=[Depends(require_api_key)],
)
async def kill_switch_status() -> KillSwitchStatusResponse:
    state = _kill_switch.current_state()
    source = "unknown" if state.reason == _kill_switch.UNKNOWN_STATE_REASON else "db"
    return KillSwitchStatusResponse(
        latched=state.active, since=state.since, reason=state.reason, source=source
    )
