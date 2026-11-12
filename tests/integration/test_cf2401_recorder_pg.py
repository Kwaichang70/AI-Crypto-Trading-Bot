"""CF-2401-119: ContextRecorder against a REAL Postgres (optional, skipped without a DSN).

Set ``CONTEXT_STORE_TEST_DATABASE_URL`` to a SCRATCH database, as for
``test_wp24_context_store_pg.py``, with a role that is not a superuser (test 4 revokes INSERT and
fails if that has no effect). The table is built from the ORM metadata in a private schema
``cf2401_rec_<hex>`` (a migrated table is never touched) that is dropped again; a killed run may
leave one behind to drop by hand. The DSN is only read.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from structlog.testing import capture_logs

from api.db.models import ContextObservationORM
from api.services.context_recorder import FAILURES, ContextRecorder
from api.services.context_store import SqlContextStore
from common.metrics import metrics
from data.live_context import LIVE_BOOK
from data.point_in_time import Observation

_DSN = os.environ.get("CONTEXT_STORE_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DSN, reason="CONTEXT_STORE_TEST_DATABASE_URL not set"),
]

_SCHEMA = f"cf2401_rec_{uuid4().hex[:12]}"  # unique per process: no clash, a leak blocks nothing
_DOM, _WHALE = "coingecko:btc_dominance", "whale:net_flow_1h"
_T0, _NOW = 1_710_158_400_000, 1_710_158_400_000 + 3_600_000
_AV = _T0 + 300_123  # has milliseconds: a write or read that truncates to seconds fails


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    assert _DSN is not None
    eng = create_async_engine(_DSN, connect_args={"server_settings": {"search_path": _SCHEMA}})
    async with eng.begin() as conn:
        assert (await conn.execute(text("SHOW search_path"))).scalar() == _SCHEMA
        await conn.execute(text(f"CREATE SCHEMA {_SCHEMA}"))  # errors if it exists
    try:
        metrics.reset()
        LIVE_BOOK.reset()
        yield eng
    finally:
        LIVE_BOOK.reset()
        metrics.reset()
        async with eng.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {_SCHEMA} CASCADE"))
        await eng.dispose()


@pytest_asyncio.fixture
async def store(engine: AsyncEngine) -> SqlContextStore:
    async with engine.begin() as conn:
        await conn.run_sync(ContextObservationORM.__table__.create)
    return SqlContextStore(async_sessionmaker(engine, expire_on_commit=False))


def _obs(value: str, avail: int = _AV) -> Observation:
    return Observation(_T0, avail, Decimal(value), "recorded")


async def _life(store: SqlContextStore, *offers: tuple[str, Observation]) -> None:
    """One process life without a seed: a fresh recorder offers the rows and shuts down."""
    rec = ContextRecorder(store)
    for sid, o in offers:
        rec.offer(sid, o)
    await rec.flush(5.0)


async def _count(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return int(
            (await conn.execute(text("SELECT count(*) FROM context_observations"))).scalar_one()
        )


def _book(sid: str) -> list[tuple[int, int, Decimal]]:
    book = LIVE_BOOK.snapshot()[sid]
    return [(o.observed_ms, o.available_ms, o.value) for o in book.observations]


def _failures() -> int:
    return sum(v for k, v in metrics.get_all()["counters"].items() if k.startswith(FAILURES))


async def test_round_trip_and_seeded_restart_adds_no_rows(
    engine: AsyncEngine, store: SqlContextStore
) -> None:
    await _life(store, (_DOM, _obs("50")), (_WHALE, _obs("7")))
    assert await _count(engine) == 2
    rec = ContextRecorder(store)
    await rec.seed(_NOW)
    assert _book(_DOM) == [(_T0, _AV, Decimal(50))]  # the stored stamps
    assert _book(_WHALE) == [(_T0, _AV, Decimal(7))]
    rec.offer(_DOM, _obs("60", _T0 + 900_000))  # a restart: same buckets, even a new value
    rec.offer(_WHALE, _obs("7"))
    await rec.flush(5.0)
    assert await _count(engine) == 2
    assert _failures() == 0


async def test_changed_value_after_a_failed_seed_is_a_second_vintage(
    engine: AsyncEngine, store: SqlContextStore
) -> None:
    await _life(store, (_DOM, _obs("50")))
    await _life(store, (_DOM, _obs("60", _T0 + 900_000)))  # DB-P7-01: no seed, nothing held
    assert await _count(engine) == 2
    await ContextRecorder(store).seed(_NOW)
    assert _book(_DOM) == [(_T0, _AV, Decimal(50))]  # the earlier vintage


async def test_missing_table_is_one_info_line_and_no_failure(engine: AsyncEngine) -> None:
    rec = ContextRecorder(SqlContextStore(async_sessionmaker(engine, expire_on_commit=False)))
    with capture_logs() as logs:
        for sid, o in ((_DOM, _obs("50")), (_WHALE, _obs("7"))):  # two writes, one line
            rec.offer(sid, o)
            await rec.flush(5.0)
    assert [(x["log_level"], x["event"]) for x in logs] == [("info", "context_recorder.disabled")]
    assert _failures() == 0


async def test_insufficient_privilege_is_a_failure_not_a_missing_table(
    engine: AsyncEngine, store: SqlContextStore
) -> None:
    async with engine.begin() as conn:  # a real 42501 from the server, no stub
        await conn.execute(text("REVOKE INSERT ON context_observations FROM CURRENT_USER"))
        priv = "SELECT has_table_privilege('context_observations', 'INSERT')"
        assert not (await conn.execute(text(priv))).scalar(), "role bypasses the REVOKE"
    listener = SimpleNamespace(on_recorder_failing=AsyncMock())
    rec = ContextRecorder(store, listener)
    with capture_logs() as logs:
        for sid in (_DOM, _WHALE):  # a failure does not switch the recorder off
            rec.offer(sid, _obs("50"))
            await rec.flush(5.0)
    assert [x["event"] for x in logs] == ["context_recorder.write_failed"] * 2
    assert _failures() == 2
    assert listener.on_recorder_failing.await_count == 2
    assert await _count(engine) == 0
