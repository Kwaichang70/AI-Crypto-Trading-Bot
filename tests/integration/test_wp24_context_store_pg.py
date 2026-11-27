"""WP2.4 P5: SqlContextStore against a REAL Postgres (optional, skipped without a DSN).

Set ``CONTEXT_STORE_TEST_DATABASE_URL`` (``postgresql+asyncpg://user:pass@host:port/db``) to a
SCRATCH database (role and database names must contain ``scratch``; password from
``openssl rand``, never echoed; no pg_hba edits; trap-cleanup drops the DB, the role and stops
the cluster). The test creates ``context_observations`` from the ORM metadata only when it
does not exist yet, and drops it again only in that case.

It checks that the SQL writer and the SQL loader give the same answers as the in-memory fake
(the semantics the unit tests pin), and that the CHECK constraints reject bad rows.
"""

from __future__ import annotations

import os
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from api.db.models import ContextObservationORM
from api.services.context_store import (
    ContextRow,
    InMemoryContextStore,
    SqlContextStore,
)
from data.point_in_time import SERIES_RULES, series_digest, to_ms

_DSN = os.environ.get("CONTEXT_STORE_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DSN, reason="CONTEXT_STORE_TEST_DATABASE_URL not set"),
]

_BASE = datetime(2024, 3, 1, tzinfo=UTC)
_DOM = "coingecko:btc_dominance"
_WHALE = "whale:net_flow_1h"


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    assert _DSN is not None
    eng = create_async_engine(_DSN)
    created = False
    async with eng.begin() as conn:
        exists = (await conn.execute(text("SELECT to_regclass('context_observations')"))).scalar()
        if exists is not None:
            await eng.dispose()
            pytest.skip("context_observations already exists; refusing to touch it")
        await conn.run_sync(ContextObservationORM.__table__.create)
        created = True
    try:
        yield eng
    finally:
        if created:
            async with eng.begin() as conn:
                await conn.run_sync(ContextObservationORM.__table__.drop)
        await eng.dispose()


def _rows(rng: random.Random) -> list[ContextRow]:
    out: list[ContextRow] = []
    for sid, step in ((_DOM, 1), (_WHALE, 2)):
        for h in range(30):
            observed = _BASE + timedelta(hours=h * step)
            for _ in range(rng.choice([1, 1, 2])):
                avail = observed + timedelta(milliseconds=rng.randint(0, 4 * 3_600_000))
                value = Decimal(rng.randint(-5000, 5000)) / 100
                out.append(
                    ContextRow(sid, observed, avail, value.quantize(Decimal("1e-8")), "recorded")
                )
    return out


async def test_sql_store_matches_the_fake(engine: AsyncEngine) -> None:
    rng = random.Random(24)  # noqa: S311
    sql = SqlContextStore(async_sessionmaker(engine, expire_on_commit=False))
    fake = InMemoryContextStore()
    batch = _rows(rng)
    assert await sql.write_observations(batch) == await fake.write_observations(batch)
    assert await sql.write_observations(batch) == 0  # idempotent
    for _ in range(20):
        start = _BASE + timedelta(hours=rng.randint(0, 40))
        end = start + timedelta(hours=rng.randint(0, 30))
        got = await sql.load_window([_DOM, _WHALE], start, end)
        want = await fake.load_window([_DOM, _WHALE], start, end)
        assert got.row_count == want.row_count
        assert series_digest(got.series.values()) == series_digest(want.series.values())
        for sid in got.series:
            for t in range(to_ms(start), to_ms(end) + 1, 1_800_000):
                age = SERIES_RULES[sid].max_age_ms
                assert got.series[sid].asof(t, max_age_ms=age) == want.series[sid].asof(
                    t, max_age_ms=age
                )
    assert (
        await sql.load_window([_DOM], _BASE, _BASE + timedelta(days=3))
    ).ingest_cutoff is not None


async def test_check_constraints_reject_bad_rows(engine: AsyncEngine) -> None:
    insert = text(
        "INSERT INTO context_observations "
        "(series_id, observed_at, available_at, value, source, available_at_basis) "
        "VALUES (:s, :o, :a, CAST(:v AS NUMERIC), 'x', :b)"
    )
    bad = [
        {"s": _DOM, "o": _BASE, "a": _BASE - timedelta(seconds=1), "v": "1", "b": "recorded"},
        {"s": _DOM, "o": _BASE, "a": _BASE, "v": "NaN", "b": "recorded"},
        {"s": _DOM, "o": _BASE, "a": _BASE, "v": "1", "b": "guess"},
        {"s": "", "o": _BASE, "a": _BASE, "v": "1", "b": "recorded"},
    ]
    for params in bad:
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(insert, params)
