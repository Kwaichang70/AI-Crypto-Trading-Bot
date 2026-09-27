"""
apps/api/services/kill_switch.py
---------------------------------
WP1.7a: process-wide, DB-backed global kill-switch latch (SY-01).

The single-row ``kill_switch_state`` table (migration 018) is the source
of truth; an in-process mirror (``_STATE`` / ``_loaded``) lets every hot
path read the latch synchronously (``StrategyEngine._process_bar`` reads
``risk_manager.kill_switch_active`` on every bar; ``create_run`` /
``promote_to_live`` / ``resume_run`` read :func:`is_active` on every
request) with no per-call DB round trip.

Round 2 (security round 1, WP17a-S-01/S-02/S-03/S-04) rewrites the
ordering guarantees:

* :func:`latch_in_memory` -- the synchronous, no-``await``, memory-only
  primitive.  The kill-switch router calls this (and latches every
  registered engine directly) as its literal first statement, before
  ANY database access -- a DB outage, a slow ``FOR UPDATE`` wait, or any
  other persistence failure can then never leave a running engine
  unlatched (S-01).
* :func:`activate` -- persists under the module lock (S-03); the very
  first statements inside the lock (before any ``await``) flip the
  mirror AND latch every engine the caller passes in (round 3,
  WP17a-S-R2-01) -- see below.
* :func:`clear` -- holds the SAME module lock for its whole run (S-03):
  reads the row ``FOR UPDATE``, requires exactly one row updated, runs
  the caller's ``before_commit`` hook (so an audit row lands in the SAME
  transaction, S-10/S-12) and commits, and ONLY THEN flips the mirror
  AND unlatches every engine the caller passes in (round 3, WP17a-S-R2-01)
  -- still inside the SAME lock section, so it can never interleave with
  a concurrent :func:`activate`. Any failure anywhere in that sequence
  rolls back and raises :class:`KillSwitchClearError` without ever
  touching the mirror or any engine.
* :func:`load` -- read at boot, before ``recover_orphaned_runs()`` (I4).
  Catches ``Exception`` broadly (S-02) -- a missing row, a bad DSN, an
  auth failure, anything -- and marks the mirror **unloaded**, never
  touching ``_STATE`` itself.
* :func:`mark_unknown` -- forces the mirror unloaded from OUTSIDE
  :func:`load` (S-02) -- called by ``main.py``'s lifespan ``except``
  branch when even session-factory construction (something *before*
  ``load()`` could ever run its own try/except) failed.
* :func:`is_active` / :func:`current_state` / :func:`apply_latch` all
  treat "not yet loaded" as latched with reason ``latch_state_unknown``
  (I4) -- this is now the BARE import-time default too (``_loaded =
  False``), not merely a transient failure state.  Tests get a clean,
  loaded, un-latched mirror via the autouse fixture in
  ``tests/conftest.py`` (:func:`reset_state_for_tests`).

Round 3 (security round 2, WP17a-S-R2-01) closes a residual race: round
2 had the KILL-SWITCH ROUTER itself latch/unlatch engines synchronously
around its calls to :func:`activate`/:func:`clear`, with :func:`activate`
re-asserting only the MIRROR (not engines) after acquiring the lock. A
concurrent press and clear that interleaved around the lock boundary
could then leave a running engine's OWN latch state disagreeing with the
mirror/DB (e.g. a clear's post-DB, post-lock-release engine-unlatch loop
running AFTER a racing press had already re-latched the same engine).
Engine mutation now happens EXCLUSIVELY inside :func:`activate`/:func:`clear`,
under ``_LOCK``, atomically with the mirror flip -- so whichever call's
critical section runs LAST while holding the lock is the one whose
engine state (and mirror state) becomes final, and the two can never
disagree. The router still does its OWN synchronous pre-latch (calling
:func:`latch_in_memory` directly, before any ``await``) purely for the
S-01 "protect immediately, even before the lock" guarantee -- that
pre-latch is provisional and always gets superseded by :func:`activate`'s
own latching once it acquires the lock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import KillSwitchStateORM

__all__ = [
    "GLOBAL_REASON",
    "UNKNOWN_STATE_REASON",
    "ClearOutcome",
    "KillSwitchClearError",
    "KillSwitchState",
    "activate",
    "apply_latch",
    "clear",
    "current_state",
    "is_active",
    "is_loaded",
    "latch_in_memory",
    "load",
    "mark_unknown",
]

logger = structlog.get_logger(__name__)

#: SY-03: the canonical global-latch reason string used everywhere a
#: caller does not supply a more specific one (e.g. a bare re-press).
GLOBAL_REASON = "global_kill_switch"
#: I4: the fail-closed reason used when the persisted state cannot be read.
UNKNOWN_STATE_REASON = "latch_state_unknown"


class _RiskManagerLike(Protocol):
    def trigger_kill_switch(self, reason: str) -> None: ...
    def reset_kill_switch(self, reason: str) -> None: ...

    @property
    def kill_switch_active(self) -> bool: ...

    @property
    def kill_switch_reasons(self) -> Iterable[str]: ...


class _EngineLike(Protocol):
    @property
    def risk_manager(self) -> _RiskManagerLike: ...

    @property
    def run_id(self) -> str | None: ...


#: Round 3 (WP17a-S-R2-01): both :func:`activate` and :func:`clear` accept
#: this same callable shape -- a zero-arg callable returning the CURRENT
#: set of registered engines at the moment the lock is held (never a
#: snapshot taken earlier), so a run that registers/unregisters mid-race
#: is still handled correctly.
EngineProvider = Callable[[], Iterable[_EngineLike]]


@dataclass(slots=True)
class KillSwitchState:
    """In-memory mirror of the ``kill_switch_state`` row."""

    active: bool
    reason: str | None = None
    since: datetime | None = None


@dataclass(slots=True)
class ClearOutcome:
    """Result of :func:`clear`'s (optional) engine-unlatch pass (round 3).

    ``runs_kept_latched`` pairs each still-latched engine's ``run_id``
    (falling back to ``"<unknown>"`` if the engine has not finished
    ``start()`` yet) with its full, sorted, remaining reason set -- e.g.
    an engine that also carries its own per-run ``flatten_incomplete``
    latch stays in this list even after the global reasons are removed.
    """

    runs_unlatched: list[str] = field(default_factory=list)
    runs_kept_latched: list[tuple[str, list[str]]] = field(default_factory=list)


class KillSwitchClearError(RuntimeError):
    """Raised by :func:`clear` when the DB write/commit fails (I3).

    The in-memory mirror is left latched (the caller never sees the
    clear applied) -- the router maps this to 503
    ``latch_clear_not_persisted``.
    """


# ---------------------------------------------------------------------------
# Module-level mirror + lock.
#
# ``_loaded=False`` is the bare import-time default (S-02): every reader
# (:func:`is_active`, :func:`current_state`, :func:`apply_latch`) treats
# "not loaded" as latched with ``latch_state_unknown`` -- fail-closed by
# construction, not merely on a caught exception. ``main.py``'s lifespan
# unconditionally calls :func:`load` before ``recover_orphaned_runs()``
# and before the app ever serves a request; the test suite gets a clean
# loaded+un-latched mirror via the autouse ``tests/conftest.py`` fixture
# (:func:`reset_state_for_tests`), so this default no longer needs to be
# "un-latched" for test compatibility the way round 1's did.
#
# ``_LOCK`` serialises :func:`activate` and :func:`clear` (S-03) so the
# in-memory mirror, the persisted row, AND every registered engine's own
# latch state (round 3, WP17a-S-R2-01) can never end up disagreeing after
# two concurrent calls interleave.
# ---------------------------------------------------------------------------
_STATE = KillSwitchState(active=False, reason=None, since=None)
_loaded: bool = False
_LOCK = asyncio.Lock()


def is_loaded() -> bool:
    """True once :func:`load` (or a test reset) has run successfully."""
    return _loaded


def is_active() -> bool:
    """True while the global kill switch is latched (in-memory mirror).

    Fail-closed (S-02): returns ``True`` whenever the mirror has not yet
    been loaded, regardless of what ``_STATE.active`` happens to hold.
    """
    if not _loaded:
        return True
    return _STATE.active


def current_state() -> KillSwitchState:
    """Snapshot of the in-memory mirror (for the status GET route).

    Fail-closed (S-02): reports ``active=True, reason=latch_state_unknown``
    whenever the mirror has not yet been loaded.
    """
    if not _loaded:
        return KillSwitchState(active=True, reason=UNKNOWN_STATE_REASON, since=None)
    return KillSwitchState(active=_STATE.active, reason=_STATE.reason, since=_STATE.since)


def reset_state_for_tests() -> None:
    """Test-only helper: reset the module-level mirror to a clean,
    LOADED, un-latched state.

    Production code never calls this -- the mirror is only ever
    supposed to change via :func:`load`/:func:`activate`/:func:`clear`/
    :func:`mark_unknown`.  Used by the autouse fixture in
    ``tests/conftest.py`` so the whole suite (which never calls
    :func:`load` against a real lifespan) is not fail-closed-latched by
    default -- tests that specifically want the fail-closed/unknown
    state call :func:`mark_unknown` themselves.
    """
    global _STATE, _loaded
    _STATE = KillSwitchState(active=False, reason=None, since=None)
    _loaded = True


def mark_unknown() -> None:
    """Force the mirror into the fail-closed 'unknown' state (S-02).

    Called by ``main.py``'s lifespan ``except`` branch when even
    session-factory construction / the ``async with`` context-manager
    entry raised -- i.e. something *outside* :func:`load`'s own
    try/except prevented it from ever running.  :func:`load` itself
    already leaves the mirror unloaded on any internal failure; this is
    the belt-and-suspenders call for the layer above it.
    """
    global _loaded
    _loaded = False
    logger.critical("kill_switch.marked_unknown")


def latch_in_memory(reason: str) -> None:
    """The synchronous, no-``await``, memory-only latch primitive (S-01).

    Callers that need the absolute strongest ordering guarantee (the
    kill-switch router) call this directly, as their literal first
    statement, before touching the database at all. This is a
    PROVISIONAL flip only (round 3): :func:`activate` re-imposes both the
    mirror and every registered engine's latch, atomically, once it
    acquires ``_LOCK`` -- so a concurrent :func:`clear` that raced to
    completion in the gap between this call and the lock can never
    silently win.
    """
    global _STATE, _loaded
    _STATE = KillSwitchState(active=True, reason=reason, since=datetime.now(tz=UTC))
    _loaded = True


async def load(db: AsyncSession) -> KillSwitchState:
    """Read the persisted latch at boot, before ``recover_orphaned_runs()``.

    Catches ``Exception`` broadly (S-02, not just ``SQLAlchemyError`` --
    an unreachable host, a bad DB name, or a bad password all raise
    ``asyncpg``/``OSError`` subtypes that are not ``SQLAlchemyError``).
    A read failure, or a missing singleton row, marks the mirror
    unloaded (I4) -- every reader then reports latched/unknown -- logged
    at critical so an operator notices a broken/uninitialised database
    instead of silently trading with a latch nobody can see.

    Round 3 (WP17a-S-R2-06): when the row reports ``active=False``,
    ``since`` is always ``None`` -- a cleared switch has no "since it was
    latched" timestamp to report, even though the row's own
    ``activated_at`` column still holds the timestamp of the most recent
    (now-cleared) activation for audit/history purposes.
    """
    global _STATE, _loaded
    try:
        result = await db.execute(
            select(KillSwitchStateORM).where(KillSwitchStateORM.id == 1)
        )
        row = result.scalar_one_or_none()
    except Exception:
        logger.critical("kill_switch.load_failed", exc_info=True)
        _loaded = False
        return current_state()

    if row is None:
        logger.critical("kill_switch.load_missing_row")
        _loaded = False
        return current_state()

    _STATE = KillSwitchState(
        active=row.active,
        reason=row.reason,
        since=row.activated_at if row.active else None,
    )
    _loaded = True
    logger.info("kill_switch.loaded", active=_STATE.active, reason=_STATE.reason)
    return current_state()


async def activate(
    db: AsyncSession,
    *,
    reason: str,
    actor: str,
    engines: EngineProvider | None = None,
) -> bool:
    """Latch the global kill switch and persist it.

    Round 3 (WP17a-S-R2-01): the first statements inside ``async with
    _LOCK:`` -- before any ``await`` -- are flipping the mirror to
    ``active=True`` AND latching every engine ``engines()`` currently
    returns with :data:`GLOBAL_REASON`. This closes the round-2 race: a
    concurrent :func:`clear` can never complete BETWEEN this flip and the
    persistence attempt below, because both this function and
    :func:`clear` hold the SAME lock for their entire body -- whichever
    of the two runs its critical section LAST while holding the lock is
    the one whose mirror-and-engine state becomes (and stays) final. The
    old "re-assert the mirror after persisting" step is gone: there is no
    longer a gap between the flip and the lock for anything to race
    against.

    Round 3 (WP17a-S-R2-03): the persistence UPDATE (and the defensive
    insert-if-missing fallback) run inside ``db.begin_nested()`` -- a
    SAVEPOINT -- so a persistence failure only rolls back that savepoint,
    not the caller's whole transaction (preserving any audit row/other
    work the caller's session already flushed in the same transaction).

    Never raises (S-01/S-02): every persistence exception is caught,
    logged at critical, and reported via the ``bool`` return so the
    caller's response can surface ``latch_persisted=false`` instead of a
    500 -- the mirror and every engine are latched in-memory regardless.

    Returns
    -------
    bool:
        ``True`` if the persistence write succeeded, ``False`` if it
        raised -- the process (and every engine) is still latched
        in-memory either way.
    """
    global _STATE, _loaded
    async with _LOCK:
        now = datetime.now(tz=UTC)
        _STATE = KillSwitchState(active=True, reason=reason, since=now)
        _loaded = True
        if engines is not None:
            for engine in engines():
                engine.risk_manager.trigger_kill_switch(GLOBAL_REASON)

        persisted = True
        try:
            async with db.begin_nested():
                result = await db.execute(
                    update(KillSwitchStateORM)
                    .where(KillSwitchStateORM.id == 1)
                    .values(
                        active=True,
                        reason=reason,
                        activated_at=now,
                        activated_by=actor,
                        cleared_at=None,
                        cleared_by=None,
                        clear_reason=None,
                    )
                )
                if result.rowcount == 0:  # type: ignore[attr-defined]
                    # Singleton row missing (should never happen post-018)
                    # -- insert it defensively rather than silently no-op.
                    db.add(
                        KillSwitchStateORM(
                            id=1,
                            active=True,
                            reason=reason,
                            activated_at=now,
                            activated_by=actor,
                        )
                    )
                await db.flush()
        except Exception:
            logger.critical("kill_switch.activate_persist_failed", exc_info=True)
            persisted = False

        return persisted


async def clear(
    db: AsyncSession,
    *,
    reason: str,
    actor: str,
    before_commit: Callable[[], Awaitable[None]] | None = None,
    engines: EngineProvider | None = None,
) -> ClearOutcome:
    """Clear the global kill switch.

    Holds the module lock for its whole run (S-03), so it can never
    interleave with a concurrent :func:`activate`. Reads the row ``FOR
    UPDATE`` and requires exactly one row updated (S-03). Runs the
    caller's ``before_commit`` awaitable (typically a non-swallowing
    audit-row write, S-10/S-12) INSIDE the same transaction, before the
    single commit, so the state clear and its audit row land atomically.

    Round 3 (WP17a-S-R2-01): only after that commit succeeds -- and
    STILL inside ``_LOCK`` -- does the mirror flip AND every engine
    ``engines()`` currently returns get unlatched (removing both
    :data:`GLOBAL_REASON` and :data:`UNKNOWN_STATE_REASON`, whichever it
    holds). Doing the engine mutation in the SAME lock section as the
    mirror flip (rather than in a separate, unlocked loop back in the
    router, as round 2 did) is what closes the race: a concurrent
    :func:`activate` cannot observe (or undo) a partially-applied clear,
    because it cannot even start its own critical section until this
    one -- mirror, engines, and all -- has fully finished and released
    the lock.

    Any failure anywhere in the sequence rolls back and raises
    :class:`KillSwitchClearError` without ever touching ``_STATE`` or any
    engine; the router maps that to 503 ``latch_clear_not_persisted`` and
    the process (and every engine) stays latched.

    Returns
    -------
    ClearOutcome:
        The per-engine unlatch result (empty lists when ``engines`` is
        ``None``, e.g. tests that only care about the mirror/DB).
    """
    global _STATE, _loaded
    now = datetime.now(tz=UTC)
    async with _LOCK:
        try:
            row_result = await db.execute(
                select(KillSwitchStateORM)
                .where(KillSwitchStateORM.id == 1)
                .with_for_update()
            )
            row = row_result.scalar_one_or_none()
            if row is None:
                raise KillSwitchClearError("kill_switch_state row missing")

            update_result = await db.execute(
                update(KillSwitchStateORM)
                .where(KillSwitchStateORM.id == 1)
                .values(
                    active=False,
                    cleared_at=now,
                    cleared_by=actor,
                    clear_reason=reason,
                )
            )
            updated_rowcount: int = update_result.rowcount  # type: ignore[attr-defined]
            if updated_rowcount != 1:
                raise KillSwitchClearError(
                    f"expected exactly 1 row updated, got {updated_rowcount!r}"
                )

            if before_commit is not None:
                await before_commit()

            await db.commit()
        except KillSwitchClearError:
            await _safe_rollback(db)
            raise
        except Exception as exc:
            await _safe_rollback(db)
            logger.critical("kill_switch.clear_persist_failed", exc_info=True)
            raise KillSwitchClearError("failed to persist kill-switch clear") from exc

        _STATE = KillSwitchState(active=False, reason=None, since=None)
        _loaded = True

        outcome = ClearOutcome()
        if engines is not None:
            for engine in engines():
                risk_manager = engine.risk_manager
                for r in (GLOBAL_REASON, UNKNOWN_STATE_REASON):
                    if r in risk_manager.kill_switch_reasons:
                        risk_manager.reset_kill_switch(r)
                run_id = engine.run_id or "<unknown>"
                if risk_manager.kill_switch_active:
                    outcome.runs_kept_latched.append(
                        (run_id, sorted(risk_manager.kill_switch_reasons))
                    )
                else:
                    outcome.runs_unlatched.append(run_id)
        return outcome


async def _safe_rollback(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception:
        logger.warning("kill_switch.rollback_failed", exc_info=True)


def apply_latch(
    engine: _EngineLike,
    *,
    entries_latch_reason: str | None = None,
) -> None:
    """Apply the current latch state(s) to a just-registered engine (I5).

    MUST be called synchronously, with no ``await`` in between, right
    after ``_RUN_ENGINES[run_id] = engine`` -- both at
    ``run_orchestrator.py``'s paper and live registration points, and by
    ``recover_orphaned_runs``/``resume_run`` when rebuilding an engine.

    Reads the in-memory mirror ONLY (never awaits the DB) -- that is the
    entire reason the mirror exists.  Only ever adds :data:`GLOBAL_REASON`
    or (when the mirror has not been loaded, S-02) :data:`UNKNOWN_STATE_REASON`
    -- never the operator's free-text reason string (S-04): that text is
    persisted to the DB ``reason`` column and to audit only, never copied
    into an engine's risk-manager reason set (where it would be
    un-removable by :func:`clear`, which only ever resets
    :data:`GLOBAL_REASON`/:data:`UNKNOWN_STATE_REASON`).  ``entries_latch_reason``
    is the per-run persisted latch (SY-02, only ever
    ``'flatten_incomplete'``) -- global state is never copied into it and
    vice versa (SY-01).
    """
    risk_manager = engine.risk_manager
    if not _loaded:
        risk_manager.trigger_kill_switch(UNKNOWN_STATE_REASON)
    elif _STATE.active:
        risk_manager.trigger_kill_switch(GLOBAL_REASON)
    if entries_latch_reason:
        risk_manager.trigger_kill_switch(entries_latch_reason)
