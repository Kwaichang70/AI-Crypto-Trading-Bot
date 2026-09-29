"""
apps/api/services/idempotency.py
----------------------------------
WP7.0 (reports/vp2-wp7.0/synthesis-spec.md) -- idempotent run creation and
promotion under a client-supplied ``Idempotency-Key`` header.

Design summary (binding spec SY-70-01..17)
--------------------------------------------
``Idempotency-Key`` is required (in all modes) on ``POST /api/v1/runs`` and
on ``POST /api/v1/runs/{id}/promote-to-live``. It is validated and claimed
at the "claim point" inside each handler -- i.e. AFTER every pre-existing
rejection (pydantic validation, WP1.3a exit-config, the live-trading gate,
the kill-switch check, the concurrency cap, and -- for promote -- the 404
source-run lookup). Presence/format failures never touch the store.

The claim algorithm (:func:`PostgresIdempotencyStore.claim`, SY-70-07) runs
in its own dedicated, short, READ COMMITTED transactions -- never inside
the caller's main request transaction -- using the database clock
(``now()``) for every staleness comparison. It loops at most 3 times and
fails closed (409) if it cannot resolve a race within that budget.

Completion (:func:`PostgresIdempotencyStore.complete`, SY-70-08) is the
one operation that runs IN the caller's main transaction, immediately
before ``await db.commit()`` -- it fences on
``(key, claimed_run_id, status='in_progress')`` so a slow original that
lost a race to a stale reclaim gets 0 rows back (:class:`OwnershipLost`)
and must roll back instead of returning a phantom 201 (this OVERRIDES an
earlier design note that said "still return 201" -- see synthesis-spec.md
G-2/C-5).

Marking a claim ``failed`` (:func:`PostgresIdempotencyStore.fail`, SY-70-09)
runs in yet another dedicated session and NEVER raises -- a failure to
record "failed" only means a future retry has to wait out the staleness
window instead of reclaiming immediately; it must never mask the original
exception that triggered it.

Security (I7): the raw header value is never logged. Every log event below
carries only ``key_prefix`` (the first 8 hex characters of the canonical,
lowercased key).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    pass

__all__ = [
    "CREATE_ENDPOINT",
    "PROMOTE_ENDPOINT",
    "IdempotencyStore",
    "Owned",
    "OwnershipLost",
    "PostgresIdempotencyStore",
    "Replay",
    "compute_fingerprint",
    "get_idempotency_store",
    "idempotency_http_error",
    "mark_run_error_after_ambiguous_commit",
    "parse_idempotency_key",
    "prune_expired_idempotency_keys",
    "reset_idempotency_store_for_tests",
]

logger = structlog.get_logger(__name__)

CREATE_ENDPOINT = "POST /runs"
PROMOTE_ENDPOINT = "POST /runs/{run_id}/promote-to-live"

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# ---------------------------------------------------------------------------
# §5 wire contract -- exact bodies (SY-70-04)
# ---------------------------------------------------------------------------
_CODE_INFO: dict[str, tuple[int, str]] = {
    "idempotency_key_required": (
        428,
        "Idempotency-Key header is required for this endpoint.",
    ),
    "idempotency_key_invalid_format": (
        400,
        "Idempotency-Key must be a UUID (8-4-4-4-12 hex).",
    ),
    "idempotency_key_reused": (
        422,
        "This Idempotency-Key was already used for a different request.",
    ),
    "idempotency_in_progress": (
        409,
        "A request with this Idempotency-Key is still being processed.",
    ),
}

# Codes for which the §5 table echoes the canonical key in the response
# header. 428/400 never had a valid key to echo.
_ECHO_HEADER_CODES = frozenset({"idempotency_key_reused", "idempotency_in_progress"})


def _key_prefix(key: uuid.UUID) -> str:
    """I7: the only form of the key ever logged -- first 8 hex chars."""
    return str(key)[:8]


def idempotency_http_error(code: str, *, key: uuid.UUID | None) -> HTTPException:
    """Build the exact §5 ``HTTPException`` for ``code``.

    The wire body is always ``{"detail": {"code": ..., "message": ...}}``
    (SY-70-04) -- ``ErrorResponse`` is not used; its flat shape does not
    match what every other structured error on this router already sends.
    """
    status_code, message = _CODE_INFO[code]
    headers: dict[str, str] | None = None
    if key is not None and code in _ECHO_HEADER_CODES:
        headers = {"Idempotency-Key": str(key)}
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=headers,
    )


def parse_idempotency_key(raw: str | None) -> uuid.UUID:
    """Validate the raw ``Idempotency-Key`` header value (SY-70-02/03).

    Order: missing -> 428 ``idempotency_key_required``; malformed -> 400
    ``idempotency_key_invalid_format``. The raw value is NEVER logged --
    the invalid-format path logs ``idempotency.invalid_key_format`` with
    ``length`` only. On success the value is canonicalised to lowercase
    (``str(uuid.UUID(v))`` is always lowercase, regardless of input case).
    """
    if raw is None:
        raise idempotency_http_error("idempotency_key_required", key=None)

    if len(raw) != 36 or not _UUID_RE.match(raw):
        logger.info("idempotency.invalid_key_format", length=len(raw))
        raise idempotency_http_error("idempotency_key_invalid_format", key=None)

    return uuid.UUID(raw)


def compute_fingerprint(
    endpoint: str, path_params: dict[str, Any], body: dict[str, Any]
) -> str:
    """sha256({"endpoint", "path_params", "body"}) (SY-70-06).

    Deterministic across key order via ``sort_keys=True`` and a fixed
    separator; ``default=str`` handles any residual non-JSON-native type
    (e.g. a stray ``Decimal``/``UUID``) without raising. Headers are never
    included in the fingerprint.
    """
    payload = {"endpoint": endpoint, "path_params": path_params, "body": body}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Claim outcomes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Owned:
    """This request now owns the key; ``claimed_run_id`` is the id to use
    for the ``runs`` row it is about to create."""

    claimed_run_id: uuid.UUID


@dataclass(frozen=True)
class Replay:
    """The key was already completed under the SAME fingerprint; return the
    existing run at its CURRENT state (SY-70-11) instead of creating a new
    one."""

    run_id: uuid.UUID
    status_code: int


class OwnershipLost(Exception):
    """Raised by :meth:`IdempotencyStore.complete` when the fencing UPDATE
    affects 0 rows -- a stale reclaim by another request won the race
    (SY-70-08). The caller MUST roll back its main transaction (undoing the
    run row and any audit row) and return 409 ``idempotency_in_progress``;
    it must NOT call :meth:`IdempotencyStore.fail` (there is no longer an
    owned claim to mark failed)."""


class IdempotencyStore(Protocol):
    """Injectable store contract (SY-70-13). Hermetic tests install
    ``tests.integration.fakes.idempotency_store.InMemoryIdempotencyStore``
    via ``dependency_overrides``; production uses
    :class:`PostgresIdempotencyStore`."""

    async def claim(
        self, *, key: uuid.UUID, endpoint: str, fingerprint: str
    ) -> Owned | Replay:
        """Resolve ``key`` to :class:`Owned` or :class:`Replay`.

        Raises
        ------
        HTTPException 422 ``idempotency_key_reused``
        HTTPException 409 ``idempotency_in_progress``
        """
        ...

    async def complete(
        self,
        db: AsyncSession,
        *,
        key: uuid.UUID,
        claimed_run_id: uuid.UUID,
        status_code: int = 201,
    ) -> None:
        """Fence-and-complete, IN the caller's own transaction/session,
        immediately before ``await db.commit()``.

        Raises
        ------
        OwnershipLost
            When the fencing UPDATE affects 0 rows.
        """
        ...

    async def fail(self, *, key: uuid.UUID, claimed_run_id: uuid.UUID) -> None:
        """Mark the claim ``failed`` so a future retry can reclaim it once
        stale. NEVER raises."""
        ...


# ---------------------------------------------------------------------------
# Store SQL (binding; §6). ``:stale`` is seconds. Every claim-phase
# statement below is its own committed transaction, on its OWN dedicated
# session -- never the caller's main session (SY-70-07/13).
# ---------------------------------------------------------------------------
_SQL_CLAIM_INSERT = text(
    "INSERT INTO idempotency_keys (key, endpoint, request_fingerprint, status, claimed_run_id) "
    "VALUES (:k, :e, :fp, 'in_progress', :cid) "
    "ON CONFLICT (key) DO NOTHING RETURNING key"
)
_SQL_CLAIM_READ = text(
    "SELECT request_fingerprint, status, run_id, response_status_code, "
    "(updated_at < now() - make_interval(secs => :stale)) AS is_stale "
    "FROM idempotency_keys WHERE key = :k"
)
_SQL_RECLAIM_FAILED = text(
    "UPDATE idempotency_keys SET status='in_progress', claimed_run_id=:cid, "
    "run_id=NULL, updated_at=now() "
    "WHERE key=:k AND status='failed' AND request_fingerprint=:fp "
    "RETURNING key"
)
_SQL_STALE_BACKFILL = text(
    "UPDATE idempotency_keys ik SET status='completed', run_id=ik.claimed_run_id, "
    "updated_at=now() "
    "WHERE ik.key=:k AND ik.status='in_progress' AND ik.request_fingerprint=:fp "
    "AND ik.updated_at < now() - make_interval(secs => :stale) "
    "AND EXISTS (SELECT 1 FROM runs r WHERE r.id = ik.claimed_run_id) "
    "RETURNING ik.run_id, ik.response_status_code"
)
_SQL_STALE_RECLAIM = text(
    "UPDATE idempotency_keys ik SET claimed_run_id=:cid, run_id=NULL, updated_at=now() "
    "WHERE ik.key=:k AND ik.status='in_progress' AND ik.request_fingerprint=:fp "
    "AND ik.updated_at < now() - make_interval(secs => :stale) "
    "AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.id = ik.claimed_run_id) "
    "RETURNING ik.key"
)
_SQL_COMPLETE = text(
    "UPDATE idempotency_keys SET status='completed', run_id=:cid, "
    "response_status_code=:sc, updated_at=now() "
    "WHERE key=:k AND claimed_run_id=:cid AND status='in_progress' "
    "RETURNING key"
)
_SQL_FAIL = text(
    "UPDATE idempotency_keys SET status='failed', updated_at=now() "
    "WHERE key=:k AND claimed_run_id=:cid AND status='in_progress'"
)
_SQL_PRUNE = text(
    "DELETE FROM idempotency_keys WHERE key IN ("
    "SELECT key FROM idempotency_keys WHERE updated_at < now() - make_interval(hours => :ttl) "
    "LIMIT :batch)"
)
# WP7.0 round 2 (WP70-S-01): a defensive, idempotent recovery UPDATE for
# the ambiguous-commit window (see mark_run_error_after_ambiguous_commit
# below). Deliberately touches ONLY ``runs``, never ``idempotency_keys`` --
# it is safe to run unconditionally: if the main transaction's commit
# never actually landed, no row with this id is visible to this dedicated
# session/connection at all, so the WHERE clause matches 0 rows and this
# is a pure no-op.
_SQL_MARK_RUN_ERROR = text(
    "UPDATE runs SET status='error', stopped_at=now(), updated_at=now() "
    "WHERE id=:rid AND status='running' RETURNING id"
)

_MAX_CLAIM_ITERATIONS = 3


class PostgresIdempotencyStore:
    """Production :class:`IdempotencyStore` backed by the ``idempotency_keys``
    table (SY-70-07/08/09)."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        stale_after_seconds: float,
    ) -> None:
        self._session_factory = session_factory
        self._stale_after_seconds = float(stale_after_seconds)

    async def claim(
        self, *, key: uuid.UUID, endpoint: str, fingerprint: str
    ) -> Owned | Replay:
        prefix = _key_prefix(key)

        for _ in range(_MAX_CLAIM_ITERATIONS):
            new_id = uuid.uuid4()

            # (1) INSERT ... ON CONFLICT DO NOTHING RETURNING key
            async with self._session_factory() as session:
                result = await session.execute(
                    _SQL_CLAIM_INSERT,
                    {"k": key, "e": endpoint, "fp": fingerprint, "cid": new_id},
                )
                inserted = result.first()
                await session.commit()

            if inserted is not None:
                logger.info("idempotency.claimed", key_prefix=prefix)
                return Owned(claimed_run_id=new_id)

            # (2) SELECT the existing row.
            async with self._session_factory() as session:
                result = await session.execute(
                    _SQL_CLAIM_READ,
                    {"k": key, "stale": self._stale_after_seconds},
                )
                srow = result.mappings().first()
                await session.commit()

            # (3) Row is gone (pruned between steps 1 and 2) -- retry.
            if srow is None:
                continue

            # (4) Fingerprint mismatch -> 422, whatever the status.
            if srow["request_fingerprint"] != fingerprint:
                logger.info("idempotency.reused", key_prefix=prefix)
                raise idempotency_http_error("idempotency_key_reused", key=key)

            row_status = srow["status"]

            # (5) completed -> Replay; a hard-deleted target run -> 422.
            if row_status == "completed":
                if srow["run_id"] is None:
                    logger.info(
                        "idempotency.reused",
                        key_prefix=prefix,
                        reason="run_deleted",
                    )
                    raise idempotency_http_error("idempotency_key_reused", key=key)
                logger.info("idempotency.replayed", key_prefix=prefix)
                return Replay(
                    run_id=srow["run_id"],
                    status_code=srow["response_status_code"],
                )

            # (6) failed -> reclaim.
            if row_status == "failed":
                async with self._session_factory() as session:
                    result = await session.execute(
                        _SQL_RECLAIM_FAILED,
                        {"k": key, "cid": new_id, "fp": fingerprint},
                    )
                    hit = result.first()
                    await session.commit()
                if hit is not None:
                    logger.info("idempotency.reclaimed_failed", key_prefix=prefix)
                    return Owned(claimed_run_id=new_id)
                continue

            # row_status == "in_progress"
            # (7) Fresh in_progress -> 409.
            if not srow["is_stale"]:
                logger.info("idempotency.in_progress", key_prefix=prefix)
                raise idempotency_http_error("idempotency_in_progress", key=key)

            # (8) Stale in_progress: Step A backfill, else Step B reclaim.
            async with self._session_factory() as session:
                result = await session.execute(
                    _SQL_STALE_BACKFILL,
                    {"k": key, "fp": fingerprint, "stale": self._stale_after_seconds},
                )
                arow = result.mappings().first()
                await session.commit()
            if arow is not None:
                logger.info("idempotency.backfilled_completed", key_prefix=prefix)
                return Replay(
                    run_id=arow["run_id"],
                    status_code=arow["response_status_code"],
                )

            async with self._session_factory() as session:
                result = await session.execute(
                    _SQL_STALE_RECLAIM,
                    {
                        "k": key,
                        "cid": new_id,
                        "fp": fingerprint,
                        "stale": self._stale_after_seconds,
                    },
                )
                brow = result.first()
                await session.commit()
            if brow is not None:
                logger.info("idempotency.reclaimed_stale", key_prefix=prefix)
                return Owned(claimed_run_id=new_id)

            continue

        # Fail-closed (I6): 3 iterations could not resolve the race.
        logger.warning("idempotency.claim_fail_closed", key_prefix=prefix)
        raise idempotency_http_error("idempotency_in_progress", key=key)

    async def complete(
        self,
        db: AsyncSession,
        *,
        key: uuid.UUID,
        claimed_run_id: uuid.UUID,
        status_code: int = 201,
    ) -> None:
        result = await db.execute(
            _SQL_COMPLETE,
            {"k": key, "cid": claimed_run_id, "sc": status_code},
        )
        row = result.first()
        if row is None:
            # WP7.0 round 2 (WP70-S-02): the exception message is part of
            # the traceback that a 500 handler or an unhandled-exception
            # log may render -- carry only the 8-char prefix, never the
            # full key (I7).
            raise OwnershipLost(_key_prefix(key))
        logger.info("idempotency.completed", key_prefix=_key_prefix(key))

    async def fail(self, *, key: uuid.UUID, claimed_run_id: uuid.UUID) -> None:
        prefix = _key_prefix(key)
        try:
            async with self._session_factory() as session:
                await session.execute(text("SET LOCAL lock_timeout = '2s'"))
                result = await session.execute(
                    _SQL_FAIL, {"k": key, "cid": claimed_run_id}
                )
                await session.commit()
            # WP7.0 round 2 (WP70-S-04): the UPDATE's own predicate
            # (status='in_progress') means 0 rows is an EXPECTED outcome
            # whenever the row already reached 'completed' (e.g. the main
            # transaction's commit actually succeeded before this handler
            # ever ran) -- logging "marked_failed" for that case would
            # mislead an incident review into thinking the run never
            # committed. Log strictly on what happened.
            rowcount = result.rowcount or 0  # type: ignore[attr-defined]
            if rowcount > 0:
                logger.info("idempotency.marked_failed", key_prefix=prefix)
            else:
                logger.info("idempotency.fail_noop", key_prefix=prefix)
        except Exception as exc:
            # Never mask the original exception that triggered this call --
            # a missed "failed" mark only costs a future retry the
            # staleness window before it can reclaim (SY-70-09).
            # WP7.0 round 2 (WP70-S-02): exc_info=False -- a bound-parameter
            # exception (e.g. a lock-timeout DBAPIError) would otherwise
            # render the full key via SQLAlchemy's "[parameters: ...]"
            # traceback segment. hide_parameters=True on the engine
            # (session.py) is the primary defence; exc_type-only logging
            # here is defence in depth for this specific call site.
            logger.warning(
                "idempotency.fail_mark_failed",
                key_prefix=prefix,
                exc_type=type(exc).__name__,
                exc_info=False,
            )


# ---------------------------------------------------------------------------
# Dependency singleton
# ---------------------------------------------------------------------------
_store_singleton: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    """FastAPI dependency -- lazy module singleton built from settings."""
    global _store_singleton
    if _store_singleton is None:
        from api.config import get_settings
        from api.db.session import get_session_factory

        settings = get_settings()
        _store_singleton = PostgresIdempotencyStore(
            get_session_factory(),
            stale_after_seconds=float(settings.idempotency_stale_after_seconds),
        )
    return _store_singleton


def reset_idempotency_store_for_tests() -> None:
    """Test-only: force :func:`get_idempotency_store` to rebuild on next
    call. Needed by real-Postgres fixtures that rebind
    ``api.db.session``'s module-level session-factory singleton to a fresh
    scratch-DB engine (mirrors ``api.services.kill_switch.reset_state_for_tests``)."""
    global _store_singleton
    _store_singleton = None


# ---------------------------------------------------------------------------
# Prune (SY-70-14 / DB-08). Batched deletes, staleness clock = updated_at.
# ---------------------------------------------------------------------------
async def prune_expired_idempotency_keys(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    ttl_hours: int,
    batch: int = 1000,
) -> int:
    """Delete every ``idempotency_keys`` row whose ``updated_at`` is older
    than ``ttl_hours``, in batches of ``batch`` rows per transaction. Returns
    the total number of rows deleted."""
    total = 0
    while True:
        async with session_factory() as session:
            result = await session.execute(
                _SQL_PRUNE, {"ttl": ttl_hours, "batch": batch}
            )
            deleted = result.rowcount or 0  # type: ignore[attr-defined]
            await session.commit()
        total += deleted
        if deleted < batch:
            break
    if total:
        logger.info("idempotency.pruned", count=total)
    return total


# ---------------------------------------------------------------------------
# WP7.0 round 2 (WP70-S-01): ambiguous-commit recovery.
# ---------------------------------------------------------------------------
async def mark_run_error_after_ambiguous_commit(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
) -> bool:
    """Defensively flip a possibly-committed ``running`` row to ``error``.

    Called ONLY from the ``except BaseException`` branch of
    ``create_run``/``promote_to_live``, and ONLY when that exception was
    raised AFTER :meth:`PostgresIdempotencyStore.complete` had already
    returned successfully in the main transaction -- i.e. the window where
    ``await db.commit()`` (or anything between it and the engine-task
    spawn) is what raised. At that point it is genuinely ambiguous whether
    the COMMIT reached and was applied by the server before the driver
    raised (a socket reset while reading the server's own reply, or a
    ``CancelledError`` delivered mid-await): if it did, the ``runs`` row
    and the idempotency claim are BOTH durably committed (they were the
    same transaction), but no engine task was ever created, since the
    spawn happens strictly after the commit (SY-70-12).

    Chosen remedy (security report WP70-S-01, option (b) -- the minimum
    fix, not the "re-query and maybe spawn an engine from inside an
    exception handler" option (a)):
    run a single, dedicated-session, fenced UPDATE that flips
    ``status='running' -> 'error'`` by id alone, with a short
    ``lock_timeout`` so it can never hang the request past a couple of
    seconds. This is deliberately UNCONDITIONAL and idempotent-safe:

    - If the ambiguous commit truly landed, this UPDATE finds the row
      (still ``running``, since nothing else touches it) and flips it to
      ``error`` -- ending the false-liveness signal (a replay of this key
      now returns the run in ``error`` state, not ``running``). The
      existing WP1.8 orphan/resume machinery is the intentional next
      owner of recovery for it: it is not ``orphaned`` (resume's own CAS
      requires that exact status, so a resume attempt correctly 409s
      instead of trying to adopt a run with no crash-recovery data), but
      an operator can see it in the UI exactly like any other errored
      backtest/paper/live run and re-submit a fresh create/promote.
    - If the commit never actually landed, no session anywhere (including
      this dedicated one) can see a row with this id, so the UPDATE
      matches 0 rows and is a pure no-op -- safe to call unconditionally
      from the ambiguous branch without first re-checking anything.

    Option (a) (re-query then conditionally spawn the engine from inside
    the exception handler) was rejected: it would duplicate create_run's
    and promote_to_live's own engine-construction code paths a second
    time, in a place that by definition has just seen an unexpected
    exception -- exactly the wrong place to add more speculative
    DB/engine work. Option (b) is a single, narrow, always-safe SQL
    statement that hands the actually-hard problem (deciding whether a
    now-``error`` run needs anything else done) to the machinery that
    already exists for it.

    Returns ``True`` iff a row was actually flipped (logged, run id only
    -- never the idempotency key, which this function never even sees).
    """
    try:
        async with session_factory() as session:
            await session.execute(text("SET LOCAL lock_timeout = '2s'"))
            result = await session.execute(_SQL_MARK_RUN_ERROR, {"rid": run_id})
            row = result.first()
            await session.commit()
    except Exception:
        logger.warning(
            "idempotency.ambiguous_commit_mark_error_failed",
            run_id=str(run_id),
            exc_info=False,
        )
        return False

    if row is not None:
        logger.warning(
            "idempotency.ambiguous_commit_marked_error",
            run_id=str(run_id),
        )
        return True
    return False
