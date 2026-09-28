"""
tests/unit/test_wp70_idempotency_store_unit.py
--------------------------------------------------
WP7.0 round 2 (WP70-C-01, code-critic) -- a hermetic unit test for
``PostgresIdempotencyStore.complete()``'s exact SQL/bound-parameter
shape, using a stub session that never touches a real database.

IMPORTANT (also noted in ``tests/integration/test_wp70_idempotency_api.py``'s
own module docstring): this file proves the SHAPE of the fencing
statement -- which columns appear in its WHERE clause, and that a 0-row
result raises :class:`OwnershipLost` carrying only the key prefix. It does
NOT prove that Postgres's own row-level locking/MVCC actually serializes
two concurrent completions correctly under real concurrent load -- that
guarantee (SY-70-08's actual invariant) is proven ONLY by the real-Postgres
tests ``tests/migrations/test_wp70_idempotency_races.py::TestST42FencingRegression``
and ``::TestST49SlowOriginalVsStaleReclaim``. A green run of this file
alone (no ``MIGRATION_TEST_DATABASE_URL``) is not evidence that SY-70-08
holds in production -- see AC1.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from api.services.idempotency import OwnershipLost, PostgresIdempotencyStore


class _CapturingResult:
    def __init__(self, row: tuple[Any, ...] | None) -> None:
        self._row = row

    def first(self) -> tuple[Any, ...] | None:
        return self._row


class _CapturingSession:
    """Stub in place of ``AsyncSession``: records the exact statement text
    and bound parameters passed to ``execute()``, and returns a scripted
    result (present/absent row) without any real I/O."""

    def __init__(self, *, matched: bool) -> None:
        self._matched = matched
        self.captured_sql: str | None = None
        self.captured_params: dict[str, Any] | None = None

    async def execute(
        self, statement: Any, params: dict[str, Any]
    ) -> _CapturingResult:
        self.captured_sql = str(statement)
        self.captured_params = dict(params)
        return _CapturingResult(("key",) if self._matched else None)


@pytest.mark.asyncio
async def test_complete_fences_on_claimed_run_id_and_in_progress_status() -> None:
    """SY-70-08's fencing predicate: WHERE key=:k AND claimed_run_id=:cid
    AND status='in_progress'. This is what actually prevents a slow
    original from overwriting a stale reclaim's completion."""
    session = _CapturingSession(matched=True)
    # complete() never uses self._session_factory (it operates on the
    # caller's own `db` session) -- the factory passed here is never
    # invoked, so a bare lambda raising is deliberate: if a future
    # refactor ever makes complete() reach for its own session, this
    # test fails loudly instead of silently passing.
    store = PostgresIdempotencyStore(
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
        stale_after_seconds=240.0,
    )

    key = uuid.uuid4()
    claimed_run_id = uuid.uuid4()

    await store.complete(
        session,  # type: ignore[arg-type]
        key=key,
        claimed_run_id=claimed_run_id,
        status_code=201,
    )

    assert session.captured_sql is not None
    sql = session.captured_sql.lower()
    assert "update idempotency_keys" in sql
    assert "claimed_run_id=:cid" in sql
    assert "status='in_progress'" in sql
    assert "key=:k" in sql
    assert session.captured_params == {"k": key, "cid": claimed_run_id, "sc": 201}


@pytest.mark.asyncio
async def test_complete_zero_rows_raises_ownership_lost_with_prefix_only() -> None:
    """0 rows (fence lost to a stale reclaim, SY-70-08/G-2) must raise
    OwnershipLost -- and the exception must carry only the key prefix,
    never the full key (WP70-S-02: it can appear in a chained traceback)."""
    session = _CapturingSession(matched=False)
    store = PostgresIdempotencyStore(
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
        stale_after_seconds=240.0,
    )

    key = uuid.uuid4()
    claimed_run_id = uuid.uuid4()

    with pytest.raises(OwnershipLost) as exc_info:
        await store.complete(
            session,  # type: ignore[arg-type]
            key=key,
            claimed_run_id=claimed_run_id,
            status_code=201,
        )

    message = str(exc_info.value)
    assert str(key) not in message
    assert str(key)[:8] in message
