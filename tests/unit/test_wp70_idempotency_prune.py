"""
tests/unit/test_wp70_idempotency_prune.py
---------------------------------------------
WP7.0 (ST-22, SY-70-14/DB-08, G-11) -- unit coverage for
``api.services.idempotency.prune_expired_idempotency_keys`` against a
lightweight fake session factory (no real Postgres -- the real batched
DELETE SQL itself is exercised against real Postgres by
``tests/migrations/test_wp70_idempotency_races.py`` ST-45).

This file only proves the Python-side batching loop: it keeps issuing the
DELETE until a batch returns fewer rows than the batch size, and sums the
total across batches.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from api.services.idempotency import prune_expired_idempotency_keys


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    """One fake session per ``async with session_factory() as session`` --
    each call to ``execute`` pops the next scripted rowcount."""

    def __init__(self, rowcounts: list[int]) -> None:
        self._rowcounts = rowcounts
        self.commit = AsyncMock()

    async def execute(self, *_args: Any, **_kwargs: Any) -> _FakeResult:
        rowcount = self._rowcounts.pop(0)
        return _FakeResult(rowcount)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeSessionFactory:
    """Callable session factory -- each call yields a NEW ``_FakeSession``
    wrapping the next chunk of the scripted rowcount sequence (mirrors
    ``prune_expired_idempotency_keys`` opening one session per batch)."""

    def __init__(self, rowcounts: list[int]) -> None:
        self._rowcounts = list(rowcounts)
        self.call_count = 0

    def __call__(self) -> _FakeSession:
        self.call_count += 1
        # Each batch pops exactly one rowcount off the front.
        session = _FakeSession(self._rowcounts)
        return session


@pytest.mark.asyncio
async def test_single_batch_below_limit_stops_after_one_round() -> None:
    factory = _FakeSessionFactory([25])
    deleted = await prune_expired_idempotency_keys(factory, ttl_hours=24, batch=1000)
    assert deleted == 25
    assert factory.call_count == 1


@pytest.mark.asyncio
async def test_zero_rows_deleted_stops_after_one_round() -> None:
    factory = _FakeSessionFactory([0])
    deleted = await prune_expired_idempotency_keys(factory, ttl_hours=24, batch=1000)
    assert deleted == 0
    assert factory.call_count == 1


@pytest.mark.asyncio
async def test_multiple_full_batches_then_partial_sums_total() -> None:
    # 2500 rows deleted in 3 batches of 1000, 1000, 500 (G-review DB-T style).
    factory = _FakeSessionFactory([1000, 1000, 500])
    deleted = await prune_expired_idempotency_keys(factory, ttl_hours=24, batch=1000)
    assert deleted == 2500
    assert factory.call_count == 3


@pytest.mark.asyncio
async def test_exactly_batch_sized_round_requires_one_more_empty_round() -> None:
    # A batch that returns EXACTLY `batch` rows must loop again (it cannot
    # tell, from the rowcount alone, whether more rows remain) until a
    # round returns fewer than `batch`.
    factory = _FakeSessionFactory([1000, 0])
    deleted = await prune_expired_idempotency_keys(factory, ttl_hours=24, batch=1000)
    assert deleted == 1000
    assert factory.call_count == 2
