"""WP2.6 P6b (SY-26-105, CF-26-93): G-06, the loop stall of a 500k-bar history request (DSN-gated).

Optional: skipped without ``OHLCV_STORE_TEST_DATABASE_URL`` (fixtures of
``test_wp26_ohlcv_store_pg.py``).  A cold request (empty store, scripted exchange) and a warm one
run through ``fetch_history_bars`` and ``SqlOhlcvStore`` next to a task that sleeps 5 ms and
records its worst overrun.  The first test asserts what the offload must never change (bars,
block, ``data_hash``) and prints the numbers (stall, wall time, peak RSS).  The second pins the
250 ms gate as a strict xfail: G-06 FAILS today (generation-2 garbage collections and inline loops
in ``ohlcv_cache.py``); P6c closes it and removes the marker.  The repeatable probe with spreads
is ``reports/vp2-wp26/p6b-g06-probe.py``.
"""

from __future__ import annotations

import asyncio
import bisect
import os
import resource
import time
from datetime import datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

import api.services.ohlcv_history as hist
import api.services.ohlcv_store as store_mod
import tests.integration.test_wp26_ohlcv_store_pg as spg
from common.types import TimeFrame
from data.ohlcv_cache import InMemoryOhlcvStore, OhlcvStoreError
from data.services.ccxt_market_data import RangeFetch
from tests.integration.test_wp26_ohlcv_store_pg import engine, store  # noqa: F401  (fixtures)
from tests.unit.test_wp26_api import EXCH, NOW, SYM, T0, Log, mk

_DSN = os.environ.get("OHLCV_STORE_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DSN, reason="OHLCV_STORE_TEST_DATABASE_URL not set"),
]
N, MINUTE, GATE_S = 500_000, timedelta(minutes=1), 0.25
_RUNS: list[tuple[float, Any]] = []  # (worst stall in s, HistoryBars) of cold and warm


async def _stall(call: Any) -> tuple[float, Any]:
    worst, stop = 0.0, asyncio.Event()

    async def lag() -> None:
        nonlocal worst
        while not stop.is_set():
            before = time.perf_counter()
            await asyncio.sleep(0.005)
            worst = max(worst, time.perf_counter() - before - 0.005)

    task = asyncio.create_task(lag())
    out = await call
    stop.set()
    await task
    return worst, out


async def _cold_and_warm(eng: AsyncEngine, mp: pytest.MonkeyPatch) -> list[tuple[float, Any]]:
    if _RUNS:
        return _RUNS
    bars = mk(SYM, N, TimeFrame.ONE_MINUTE, MINUTE)
    stamps = [b.timestamp for b in bars]

    class Svc:
        async def connect(self) -> None: ...

        def check_series(self, symbol: str, timeframe: TimeFrame) -> None: ...

        async def fetch_ohlcv_range_ex(
            self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime
        ) -> RangeFetch:
            part = bars[bisect.bisect_left(stamps, start) : bisect.bisect_left(stamps, end)]
            return RangeFetch(part, True, "stop", 1, 0)

    sessions = async_sessionmaker(eng, expire_on_commit=False)
    mp.setattr(hist, "_store_session_factory", lambda: sessions)
    window, svc, log = (T0, T0 + N * MINUTE), cast(Any, Svc()), cast(Any, Log())
    for name in ("cold", "warm"):
        began = time.perf_counter()
        call = hist.fetch_history_bars(
            svc, "read_through", EXCH, [SYM], TimeFrame.ONE_MINUTE, window, NOW, log
        )
        _RUNS.append(await _stall(call))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        wall = time.perf_counter() - began
        print(f"G-06 {name}: stall {_RUNS[-1][0] * 1000:.0f} ms, wall {wall:.1f} s, RSS {rss} MB")
    return _RUNS


def _data(bars: Any) -> dict[str, Any]:
    # the block without ``source``, a fact about this run's fetches, not about the data
    block = dict(bars.ohlcv_data)
    block["series"] = [{k: v for k, v in s.items() if k != "source"} for s in block["series"]]
    return block


async def test_cold_and_warm_return_identical_bars_block_and_data_hash(
    engine: AsyncEngine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (_, cold), (_, warm) = await _cold_and_warm(engine, monkeypatch)
    assert len(warm[SYM]) == N and cold[SYM] == warm[SYM]
    assert _data(cold) == _data(warm) and cold.ohlcv_data["data_hash"] is not None


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="CF-26-93: G-06 fails; P6c")
async def test_the_loop_never_stalls_250_ms_cold_or_warm(
    engine: AsyncEngine,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = await _cold_and_warm(engine, monkeypatch)
    assert all(worst < GATE_S for worst, _ in runs)


async def test_a_multi_slice_read_on_worker_threads_equals_the_fake(
    store: store_mod.SqlOhlcvStore,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (("INLINE_MAX_ROWS", 5), ("_SLICE", 7), ("_PAGE", 11)):  # 50 rows: 8 slices
        monkeypatch.setattr(store_mod, name, value)
    fake = InMemoryOhlcvStore()
    assert await spg.ext(store, 0, 50) == await spg.ext(fake, 0, 50)
    for lo, hi in ((0, 50), (13, 20), (3, 4), (30, 50)):
        window = (spg.KEY, spg.T0 + lo * spg.H, spg.T0 + hi * spg.H)
        assert await store.read(*window) == await fake.read(*window)
        assert await store.read_bars(*window) == await fake.read_bars(*window)


async def test_a_corrupt_stored_row_is_stored_row_invalid_after_the_transaction(
    engine: AsyncEngine,  # noqa: F811
    store: store_mod.SqlOhlcvStore,  # noqa: F811
) -> None:
    await spg.ext(store, 0, 3)
    async with engine.begin() as conn:  # private schema: drop the CHECK to plant a bad row
        await conn.execute(text("ALTER TABLE ohlcv_bars DROP CONSTRAINT ck_ohlcv_bars_ohlc"))
        stamp = {"t": store_mod._dt(spg.T0 + spg.H)}
        await conn.execute(text("UPDATE ohlcv_bars SET low = 0 WHERE open_time = :t"), stamp)
    span = (spg.KEY, spg.T0, spg.T0 + 3 * spg.H)
    for call in (store.read(*span), store.read_bars(*span)):
        with pytest.raises(OhlcvStoreError) as err:
            await call
        assert err.value.code == "stored_row_invalid"
