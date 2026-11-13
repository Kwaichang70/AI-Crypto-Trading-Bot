"""WP2.6 P4: ``SqlOhlcvStore`` against a REAL Postgres (optional, skipped without a DSN).

Set ``OHLCV_STORE_TEST_DATABASE_URL`` to a SCRATCH database (it may be the one used for
``CONTEXT_STORE_TEST_DATABASE_URL``) with a role that is not a superuser: the REVOKE test fails if
the role bypasses it, and a superuser fails the run instead of skipping. The tables are built from
the ORM metadata in a private schema ``wp26_ohlcv_<hex>`` that is dropped again (a killed run may
leave one behind to drop by hand). Tests insert and read bars; the DSN is otherwise only read.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

import api.services.ohlcv_store as store_mod
from api.db.models import OhlcvBarORM, OhlcvCoverageORM
from api.services.ohlcv_store import DIVERGENT_METRIC, SqlOhlcvStore
from common.metrics import metrics
from data.ohlcv_cache import Coverage, InMemoryOhlcvStore, OhlcvStoreError, SeriesKey
from data.ohlcv_integrity import (
    Candle,
    OhlcvDataError,
    canonicalise_candle,
    check_identifiers,
    series_digest,
)
from tests.unit.test_wp26_ident_parity import (
    EXCHANGE_ACCEPT,
    EXCHANGE_REJECT,
    SYMBOL_ACCEPT,
    SYMBOL_REJECT,
)

_DSN = os.environ.get("OHLCV_STORE_TEST_DATABASE_URL")
_APP_DSN = os.environ.get("DATABASE_URL")  # read at import, before any fixture patches it
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DSN, reason="OHLCV_STORE_TEST_DATABASE_URL not set"),
]

_SCHEMA = f"wp26_ohlcv_{uuid4().hex[:12]}"  # unique per process: no clash, a leak blocks nothing
T0, H = 1_704_067_200_000, 3_600_000  # 2024-01-01T00:00Z, one hour
NOW = T0 + 20_000 * H
KEY = SeriesKey("coinbase", "BTC/EUR", 3600)
_RAW = text(
    "INSERT INTO ohlcv_bars (exchange, symbol, timeframe, open_time, open, high, low, close, "
    "volume, fetched_at) VALUES (:e, :s, :tf, :t, 100, 110, 90, 105, 1, :f)"
)


def _engine(**server_settings: str) -> AsyncEngine:
    assert _DSN is not None
    settings = {"search_path": _SCHEMA, **server_settings}
    return create_async_engine(
        _DSN, hide_parameters=True, connect_args={"server_settings": settings}
    )


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    if _DSN == _APP_DSN:  # pytest.fail, not assert: a failed assert would print both DSNs
        pytest.fail("OHLCV_STORE_TEST_DATABASE_URL must not equal DATABASE_URL", pytrace=False)
    eng, created = _engine(), False
    try:
        async with eng.begin() as conn:
            assert (await conn.execute(text("SHOW search_path"))).scalar() == _SCHEMA
            who = "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
            assert (await conn.execute(text(who))).scalar() is False, (
                "a superuser must not run this"
            )
            await conn.execute(text(f"CREATE SCHEMA {_SCHEMA}"))  # errors if it exists
        created = True
        async with eng.begin() as conn:
            await conn.run_sync(OhlcvBarORM.__table__.create)
            await conn.run_sync(OhlcvCoverageORM.__table__.create)
        metrics.reset()
        yield eng
    finally:
        if created:
            async with eng.begin() as conn:
                await conn.execute(text(f"DROP SCHEMA {_SCHEMA} CASCADE"))
        await eng.dispose()


@pytest_asyncio.fixture
async def store(engine: AsyncEngine) -> SqlOhlcvStore:
    return SqlOhlcvStore(async_sessionmaker(engine, expire_on_commit=False))


def bar(i: int, close: str = "105") -> Candle:
    return canonicalise_candle(T0 + i * H, "100", "110", "90", close, "1", timeframe_s=3600)[0]


async def ext(
    store: Any, a: int, b: int, key: SeriesKey = KEY, items: list[Candle] | None = None
) -> Any:
    got = [bar(i) for i in range(a, b)] if items is None else items
    return await store.extend(key, Coverage(T0 + a * H, T0 + b * H), got, fetched_at_ms=NOW)


def leaf_accepts(exchange: str, symbol: str) -> bool:
    try:
        check_identifiers(exchange, symbol, 3600)
    except OhlcvDataError:
        return False
    return True


async def scalar(engine: AsyncEngine, sql: str) -> Any:
    async with engine.connect() as conn:
        return (await conn.execute(text(sql))).scalar_one()


def state(err: DBAPIError) -> str | None:
    return getattr(err.orig, "sqlstate", None)


async def test_round_trip_cold_equals_warm_and_equals_the_fake(store: SqlOhlcvStore) -> None:
    fake = InMemoryOhlcvStore()
    assert await ext(store, 0, 50) == await ext(fake, 0, 50)
    sql_read = await store.read(KEY, T0 + 10 * H, T0 + 40 * H)
    assert sql_read == await fake.read(KEY, T0 + 10 * H, T0 + 40 * H)
    assert len(sql_read.bars) == 30
    first = sql_read.bars[0]  # canonical Decimals (exponent 0), not the scale-12 column text
    assert first.open_.as_tuple() == bar(10).open_.as_tuple()
    assert series_digest("coinbase", "BTC/EUR", 3600, sql_read.bars)  # canonical or it raises
    other = SeriesKey("coinbase", "ETH/EUR", 3600)  # edges: right, left, contained, overlap
    for a, b in ((20, 30), (30, 34), (14, 20), (16, 32), (12, 18)):
        assert await ext(store, a, b, other) == await ext(fake, a, b, other), (a, b)
    for a, b in ((0, 5), (40, 44)):  # disjoint on either side, the same code from both
        for s in (store, fake):
            with pytest.raises(OhlcvStoreError) as err:
                await ext(s, a, b, other)
            assert err.value.code == "coverage_disjoint", (a, b)
    for lo, hi in ((0, 50), (13, 20), (30, 60)):
        window = (other, T0 + lo * H, T0 + hi * H)
        assert await store.read(*window) == await fake.read(*window)
        assert await store.read_bars(*window) == await fake.read_bars(*window)


async def test_rewrite_is_idempotent_and_divergence_keeps_the_stored_value(
    store: SqlOhlcvStore,
) -> None:
    await ext(store, 0, 10)
    same = await ext(store, 0, 10)
    assert (same.inserted, same.conflicts, same.divergent) == (0, 10, 0)
    changed = await ext(store, 0, 10, items=[bar(i, "106" if i == 3 else "105") for i in range(10)])
    assert (changed.inserted, changed.conflicts, changed.divergent) == (0, 10, 1)
    stored = await store.read_bars(KEY, T0, T0 + 10 * H)
    assert [c.close for c in stored] == [bar(0).close] * 10  # the first write wins
    assert metrics.get_all()["counters"][DIVERGENT_METRIC] == 1
    dup = await ext(store, 10, 12, items=[bar(10), bar(10, "107"), bar(11)])
    assert (dup.inserted, dup.conflicts, dup.divergent) == (2, 0, 0)  # in-batch duplicate


async def test_two_concurrent_writers_end_with_the_union(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    same = await asyncio.gather(ext(store, 0, 200), ext(store, 0, 200))
    assert sum(r.inserted for r in same) == 200
    await asyncio.gather(ext(store, 150, 250), ext(store, 200, 300))
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_bars") == 300
    assert (await store.read(KEY, T0, T0 + 300 * H)).coverage == Coverage(T0, T0 + 300 * H)


async def test_a_blocked_writer_gives_up_after_the_lock_timeout(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    await ext(store, 0, 2)
    async with engine.connect() as holder:  # holds the coverage row lock (the 5 s lock_timeout)
        await holder.execute(text("SELECT 1 FROM ohlcv_coverage FOR UPDATE"))
        started = time.monotonic()
        with pytest.raises(OhlcvStoreError) as err:
            await ext(store, 2, 4)
        waited = time.monotonic() - started
        await holder.rollback()
    assert err.value.code == "timeout" and 4.5 < waited < 10, waited  # 5 s lock, not 15 s statement
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_bars") == 2


async def test_the_coverage_row_is_locked_before_the_bars_are_written(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    await ext(store, 0, 2)
    async with engine.connect() as holder:
        await holder.execute(text("SELECT 1 FROM ohlcv_coverage FOR UPDATE"))
        task = asyncio.create_task(ext(store, 2, 4))
        await asyncio.sleep(1)
        waiting = await scalar(
            engine,
            "SELECT query FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND datname = current_database() AND pid <> pg_backend_pid()",
        )
        await holder.rollback()
    await task
    assert waiting.startswith("SELECT covered_from") and waiting.endswith("FOR UPDATE"), waiting


async def test_disjoint_span_is_refused_and_rolled_back(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    await ext(store, 0, 2)
    with pytest.raises(OhlcvStoreError) as err:
        await ext(store, 5, 8)
    assert err.value.code == "coverage_disjoint"
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_bars") == 2
    assert (await store.read(KEY, T0, T0 + 9 * H)).coverage == Coverage(T0, T0 + 2 * H)


async def test_extension_never_shrinks_coverage(store: SqlOhlcvStore) -> None:
    await ext(store, 5, 10)
    await ext(store, 0, 6)  # left extension that overlaps: the right edge must stay
    assert (await store.read(KEY, T0, T0 + 20 * H)).coverage == Coverage(T0, T0 + 10 * H)
    await ext(store, 6, 8)  # contained span: nothing shrinks
    assert (await store.read(KEY, T0, T0 + 20 * H)).coverage == Coverage(T0, T0 + 10 * H)


async def test_audit_columns_are_written_explicitly(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    await ext(store, 0, 3)
    want = store_mod._dt(NOW)  # fetched_at_ms, never the now() default
    assert await scalar(engine, "SELECT verified_at FROM ohlcv_coverage") == want
    assert await scalar(engine, "SELECT array_agg(DISTINCT fetched_at) FROM ohlcv_bars") == [want]
    ver = "SELECT min(ingest_version) = 1 AND max(ingest_version) = 1 FROM ohlcv_bars"
    assert await scalar(engine, ver)


async def test_a_read_only_transaction_rejects_writes(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        with pytest.raises(DBAPIError) as err:
            await conn.execute(
                _RAW,
                {"e": "a", "s": "A/B", "tf": "1h", "t": store_mod._dt(T0), "f": store_mod._dt(NOW)},
            )
    assert state(err.value) == "25006"


@pytest.mark.parametrize("page", [7, 10])
async def test_keyset_pages_equal_one_query(
    store: SqlOhlcvStore, monkeypatch: pytest.MonkeyPatch, page: int
) -> None:
    await ext(store, 0, 50)
    whole = await store.read(KEY, T0, T0 + 50 * H)
    monkeypatch.setattr(store_mod, "_PAGE", page)
    assert await store.read(KEY, T0, T0 + 50 * H) == whole
    assert len(whole.bars) == 50


async def test_series_hash_is_independent_of_order_chunking_and_time_zone(
    store: SqlOhlcvStore,
) -> None:
    one, chunked = SeriesKey("coinbase", "AAA/EUR", 3600), SeriesKey("coinbase", "BBB/EUR", 3600)
    await ext(store, 0, 60, one)
    for a in range(0, 60, 7):  # ascending chunks, each chunk's bars in reverse order
        b = min(a + 7, 60)
        await ext(store, a, b, chunked, items=[bar(i) for i in reversed(range(a, b))])

    async def digest(s: SqlOhlcvStore, key: SeriesKey) -> str:
        bars = (await s.read(key, T0, T0 + 60 * H)).bars
        return series_digest("coinbase", "BTC/EUR", 3600, bars)

    la = _engine(TimeZone="America/Los_Angeles")
    try:
        async with la.connect() as conn:
            assert (await conn.execute(text("SHOW TimeZone"))).scalar() == "America/Los_Angeles"
        zoned = SqlOhlcvStore(async_sessionmaker(la, expire_on_commit=False))
        assert await digest(store, one) == await digest(store, chunked) == await digest(zoned, one)
    finally:
        await la.dispose()


async def test_revoked_insert_is_denied_not_missing_table(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    async with engine.begin() as conn:  # a real 42501 from the server, no stub
        await conn.execute(text("REVOKE INSERT ON ohlcv_bars FROM CURRENT_USER"))
        priv = "SELECT has_table_privilege('ohlcv_bars', 'INSERT')"
        assert not (await conn.execute(text(priv))).scalar(), "role bypasses the REVOKE"
    with pytest.raises(OhlcvStoreError) as err:
        await ext(store, 0, 2)
    assert err.value.code == "denied"
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_coverage") == 0  # anchor rolled back


async def test_missing_table_is_missing_table(engine: AsyncEngine, store: SqlOhlcvStore) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE ohlcv_coverage"))
    for call in (store.read(KEY, T0, T0 + H), ext(store, 0, 1)):
        with pytest.raises(OhlcvStoreError) as err:
            await call
        assert err.value.code == "missing_table"


async def test_hostile_symbol_is_inert(engine: AsyncEngine, store: SqlOhlcvStore) -> None:
    hostile = SeriesKey("coinbase", "X';DROPTABLEx;--/EUR", 3600)
    await ext(store, 0, 3, hostile)
    assert len((await store.read(hostile, T0, T0 + 3 * H)).bars) == 3
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_bars") == 3
    await ext(store, 0, 1, SeriesKey("coinbase", "x';droptablex;--/eur", 3600))  # case: new series
    assert await scalar(engine, "SELECT count(*) FROM ohlcv_bars") == 4


async def test_range_query_uses_the_primary_key_index(
    engine: AsyncEngine, store: SqlOhlcvStore
) -> None:
    for sym in ("AAA/EUR", "BBB/EUR"):  # the plan is only meaningful on a populated table
        await ext(store, 0, 10_000, SeriesKey("coinbase", sym, 3600))
    async with engine.begin() as conn:
        await conn.execute(text("ANALYZE ohlcv_bars"))
        params = {"exchange": "coinbase", "symbol": "AAA/EUR", "timeframe": "1h", "page": 50_000}
        params |= {"lo": store_mod._dt(T0 + 500 * H), "hi": store_mod._dt(T0 + 600 * H)}
        rows = await conn.execute(text("EXPLAIN " + store_mod._BARS_SQL.text), params)
        plan = "\n".join(r[0] for r in rows)
    cond = " ".join(line for line in plan.splitlines() if "Index Cond" in line)
    assert "pk_ohlcv_bars" in plan and "Seq Scan" not in plan, plan  # not "no Sort" (CF-26-42)
    for part in ("exchange", "symbol", "timeframe", "open_time >=", "open_time <"):
        assert part in cond, plan


async def test_identifier_vectors_agree_with_the_ddl(engine: AsyncEngine) -> None:
    async def insert(exchange: str, symbol: str, tf: str) -> str | None:
        args = {"e": exchange, "s": symbol, "tf": tf, "t": store_mod._dt(T0)}
        args["f"] = store_mod._dt(T0 + 2 * 86_400_000)
        try:
            async with engine.begin() as conn:
                await conn.execute(_RAW, args)
        except DBAPIError as err:
            return state(err)
        return None

    for symbol in SYMBOL_ACCEPT + SYMBOL_REJECT:
        got = await insert("coinbase", symbol, "1h")
        want = None if symbol in SYMBOL_ACCEPT else ("22001" if len(symbol) > 32 else "23514")
        assert got == want, symbol
        assert (got is None) == leaf_accepts("coinbase", symbol), symbol
    for exchange in EXCHANGE_ACCEPT + EXCHANGE_REJECT:  # 1d: a distinct primary key
        got = await insert(exchange, "BTC/EUR", "1d")
        want = None if exchange in EXCHANGE_ACCEPT else ("22001" if len(exchange) > 32 else "23514")
        assert got == want and (got is None) == leaf_accepts(exchange, "BTC/EUR"), exchange
