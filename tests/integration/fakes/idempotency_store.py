"""
tests/integration/fakes/idempotency_store.py
------------------------------------------------
WP7.0 (SY-70-13): a hermetic, dict-backed :class:`IdempotencyStore` fake.

Implements exactly the same claim algorithm (SY-70-07/08/09) as
:class:`api.services.idempotency.PostgresIdempotencyStore`, over plain
in-memory dicts instead of real SQL, so ``tests/integration`` (which
overrides ``get_db`` with an ``AsyncMock`` and never touches a real
Postgres instance) can exercise ``create_run``/``promote_to_live`` without
a database.

Test-only surface (beyond the :class:`IdempotencyStore` protocol):
    - ``set_clock(fn)`` -- inject a deterministic clock for staleness tests.
    - ``seed_row(...)`` -- directly seed a row in an arbitrary state
      (used by the ST-21 contract test's stale/failed/completed cases and
      by "seeded stale in_progress" scenarios that are impractical to
      reach purely through ``claim()``/``complete()`` calls).
    - ``mark_run_exists(run_id)`` / ``rows`` -- inspect/seed the EXISTS()
      side of case-8 reconciliation, since there is no real ``runs`` table
      here.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from api.services.idempotency import (
    IdempotencyStore,
    Owned,
    OwnershipLost,
    Replay,
    idempotency_http_error,
)

__all__ = ["InMemoryIdempotencyStore", "Row"]

logger = structlog.get_logger(__name__)


def _key_prefix(key: uuid.UUID) -> str:
    return str(key)[:8]


@dataclass
class Row:
    key: uuid.UUID
    endpoint: str
    request_fingerprint: str
    status: str  # "in_progress" | "completed" | "failed"
    claimed_run_id: uuid.UUID
    run_id: uuid.UUID | None = None
    response_status_code: int = 201
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryIdempotencyStore(IdempotencyStore):
    """Hermetic fake mirroring ``PostgresIdempotencyStore``'s claim
    algorithm and fencing semantics over plain dicts."""

    def __init__(
        self,
        *,
        stale_after_seconds: float = 240.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._stale_after_seconds = float(stale_after_seconds)
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._rows: dict[uuid.UUID, Row] = {}
        # Mirrors the real `runs` table's existence, for the EXISTS() half
        # of case-8 reconciliation. complete() adds to this automatically.
        self._existing_run_ids: set[uuid.UUID] = set()

    # ------------------------------------------------------------------
    # Test-only helpers
    # ------------------------------------------------------------------
    def set_clock(self, fn: Callable[[], datetime]) -> None:
        self._clock = fn

    def mark_run_exists(self, run_id: uuid.UUID) -> None:
        self._existing_run_ids.add(run_id)

    def seed_row(
        self,
        key: uuid.UUID,
        *,
        endpoint: str,
        fingerprint: str,
        status: str,
        claimed_run_id: uuid.UUID,
        run_id: uuid.UUID | None = None,
        response_status_code: int = 201,
        updated_at: datetime | None = None,
    ) -> None:
        self._rows[key] = Row(
            key=key,
            endpoint=endpoint,
            request_fingerprint=fingerprint,
            status=status,
            claimed_run_id=claimed_run_id,
            run_id=run_id,
            response_status_code=response_status_code,
            updated_at=updated_at or self._clock(),
        )

    def row(self, key: uuid.UUID) -> Row | None:
        return self._rows.get(key)

    # ------------------------------------------------------------------
    # IdempotencyStore protocol
    # ------------------------------------------------------------------
    async def claim(
        self, *, key: uuid.UUID, endpoint: str, fingerprint: str
    ) -> Owned | Replay:
        prefix = _key_prefix(key)

        for _ in range(3):
            new_id = uuid.uuid4()
            now = self._clock()

            existing = self._rows.get(key)

            # (1) INSERT ... ON CONFLICT DO NOTHING
            if existing is None:
                self._rows[key] = Row(
                    key=key,
                    endpoint=endpoint,
                    request_fingerprint=fingerprint,
                    status="in_progress",
                    claimed_run_id=new_id,
                    updated_at=now,
                )
                logger.info("idempotency.claimed", key_prefix=prefix)
                return Owned(claimed_run_id=new_id)

            # (4) Fingerprint mismatch -> 422, whatever the status.
            if existing.request_fingerprint != fingerprint:
                logger.info("idempotency.reused", key_prefix=prefix)
                raise idempotency_http_error("idempotency_key_reused", key=key)

            if existing.status == "completed":
                if existing.run_id is None:
                    logger.info(
                        "idempotency.reused", key_prefix=prefix, reason="run_deleted"
                    )
                    raise idempotency_http_error("idempotency_key_reused", key=key)
                logger.info("idempotency.replayed", key_prefix=prefix)
                return Replay(
                    run_id=existing.run_id,
                    status_code=existing.response_status_code,
                )

            if existing.status == "failed":
                existing.status = "in_progress"
                existing.claimed_run_id = new_id
                existing.run_id = None
                existing.updated_at = now
                logger.info("idempotency.reclaimed_failed", key_prefix=prefix)
                return Owned(claimed_run_id=new_id)

            # status == "in_progress"
            is_stale = (now - existing.updated_at) >= timedelta(
                seconds=self._stale_after_seconds
            )
            if not is_stale:
                logger.info("idempotency.in_progress", key_prefix=prefix)
                raise idempotency_http_error("idempotency_in_progress", key=key)

            # Step A: stale backfill (target run exists).
            if existing.claimed_run_id in self._existing_run_ids:
                existing.status = "completed"
                existing.run_id = existing.claimed_run_id
                existing.updated_at = now
                logger.info("idempotency.backfilled_completed", key_prefix=prefix)
                return Replay(
                    run_id=existing.run_id,
                    status_code=existing.response_status_code,
                )

            # Step B: stale reclaim (target run missing).
            existing.claimed_run_id = new_id
            existing.run_id = None
            existing.updated_at = now
            logger.info("idempotency.reclaimed_stale", key_prefix=prefix)
            return Owned(claimed_run_id=new_id)

        logger.warning("idempotency.claim_fail_closed", key_prefix=prefix)
        raise idempotency_http_error("idempotency_in_progress", key=key)

    async def complete(
        self,
        db: Any,
        *,
        key: uuid.UUID,
        claimed_run_id: uuid.UUID,
        status_code: int = 201,
    ) -> None:
        row = self._rows.get(key)
        if (
            row is None
            or row.claimed_run_id != claimed_run_id
            or row.status != "in_progress"
        ):
            raise OwnershipLost(str(key))
        row.status = "completed"
        row.run_id = claimed_run_id
        row.response_status_code = status_code
        row.updated_at = self._clock()
        self._existing_run_ids.add(claimed_run_id)
        logger.info("idempotency.completed", key_prefix=_key_prefix(key))

    async def fail(self, *, key: uuid.UUID, claimed_run_id: uuid.UUID) -> None:
        row = self._rows.get(key)
        if (
            row is not None
            and row.claimed_run_id == claimed_run_id
            and row.status == "in_progress"
        ):
            row.status = "failed"
            row.updated_at = self._clock()
            logger.info("idempotency.marked_failed", key_prefix=_key_prefix(key))
