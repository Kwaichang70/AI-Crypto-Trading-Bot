"""WP2.6 P6a (SY-26-87, DoD 2/3/4): the history wiring through ``SqlOhlcvStore`` on a REAL Postgres.

Optional: skipped without ``OHLCV_STORE_TEST_DATABASE_URL`` (a SCRATCH database, a role that is
not a superuser).  The ``engine`` and ``store`` fixtures are those of
``test_wp26_ohlcv_store_pg.py`` (private schema, ``search_path`` asserted first, non-superuser
second, ``pytest.fail`` if the DSN equals ``DATABASE_URL``).  The helper runs against the same
scripted exchange as the unit tests; the run tables are not needed.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

import api.routers.runs as runs
import data.ohlcv_integrity as leaf
from api.services.ohlcv_store import SqlOhlcvStore
from common.types import TimeFrame
from tests.integration.test_wp26_ohlcv_store_pg import (  # noqa: F401  (fixtures)
    engine,
    store,
)
from tests.unit.test_wp26_api import ETH, EXCH, SYM, T0, H, World, canon, mk, world  # noqa: F401
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE
from trading.safety import BACKTEST_CIRCUIT_BREAKER_DEFAULT

_DSN = os.environ.get("OHLCV_STORE_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DSN, reason="OHLCV_STORE_TEST_DATABASE_URL not set"),
]


@pytest.fixture
def pg(world: World, store: SqlOhlcvStore) -> World:  # noqa: F811
    world.store, world.series = store, {SYM: mk(SYM, 400), ETH: mk(ETH, 400)}  # type: ignore[assignment]
    return world


def digests(w: World) -> list[str]:
    return [leaf.summarise_series(EXCH, s, 3600, canon(w.series[s])).digest for s in (SYM, ETH)]


async def metrics(bars: dict[str, Any]) -> dict[str, Any]:
    """The metrics and the trade list of one fixed backtest over ``bars`` (seeded)."""
    strategy = runs._get_strategy_registry()["ma_crossover"]
    runner = BacktestRunner(
        strategies=[strategy(strategy_id="wp26", params={"fast_period": 5, "slow_period": 20})],
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        cost_profile=DEFAULT_COST_PROFILE,
        enable_circuit_breaker=BACKTEST_CIRCUIT_BREAKER_DEFAULT,
    )
    result = await runner.run({SYM: list(bars[SYM])}, requested_window=(T0, T0 + 400 * H))
    dump = result.model_dump(exclude={"run_id"})
    dump["trades"] = [
        {k: v for k, v in t.items() if k not in ("trade_id", "run_id")} for t in dump["trades"]
    ]
    return dump


async def test_cold_then_warm_gives_equal_hash_digests_bars_and_metrics(pg: World) -> None:
    cold = await pg.fetch("read_through", [SYM, ETH])
    warm = await pg.fetch("read_through", [SYM, ETH])
    assert [s["source"] for s in pg.block(cold)["series"]] == ["exchange", "exchange"]
    assert [s["source"] for s in pg.block(warm)["series"]] == ["cache", "cache"]
    for run in (cold, warm):
        assert [s["digest"] for s in pg.block(run)["series"]] == digests(pg)
        assert pg.block(run)["data_hash"] == pg.block(cold)["data_hash"]
    assert dict(cold) == dict(warm)
    off = await metrics(pg.series)  # the exchange's own bars, as mode off hands them over
    assert await metrics(cold) == await metrics(warm) == off
    assert off["total_trades"] > 0  # not vacuous


async def test_a_store_that_is_down_gives_the_same_hash_and_metrics_fail_open(pg: World) -> None:
    healthy = await pg.fetch("read_through", [SYM, ETH])
    dead = create_async_engine(
        _DSN or "",
        hide_parameters=True,
        connect_args={"server_settings": {"search_path": "wp26_api_absent_schema"}},
    )
    try:
        pg.store = SqlOhlcvStore(async_sessionmaker(dead, expire_on_commit=False))  # type: ignore[assignment]
        down = await pg.fetch("read_through", [SYM, ETH])
        block = pg.block(down)
        assert (block["store"], block["store_reason"]) == ("unavailable", "missing_table")
        assert block["data_hash"] == pg.block(healthy)["data_hash"]
        assert await metrics(down) == await metrics(healthy)
        with pytest.raises(HTTPException) as info:  # fail closed
            await pg.fetch("require", [SYM, ETH])
        assert (info.value.status_code, info.value.detail) == (
            503,
            {"code": "ohlcv_store_unavailable"},
        )
    finally:
        await dead.dispose()


async def test_require_succeeds_with_every_series_stored_and_complete(pg: World) -> None:
    block = pg.block(await pg.fetch("require", [SYM, ETH]))
    assert block["store"] == "ok" and block["mode"] == "require"
    assert all(s["fetch_complete"] and s["volatile_tail_bars"] == 0 for s in block["series"])


async def test_a_revised_stored_bar_changes_the_hash_and_the_data_hash_decides_the_metrics(
    pg: World,
    engine: AsyncEngine,  # noqa: F811
) -> None:
    first = await pg.fetch("read_through", [SYM, ETH])
    again = await pg.fetch("read_through", [SYM, ETH])
    assert pg.block(first)["data_hash"] == pg.block(again)["data_hash"]  # equal hash ...
    assert await metrics(first) == await metrics(again)  # ... equal results
    revise = text(
        "UPDATE ohlcv_bars SET close = close + 1, high = high + 1 WHERE symbol = :s "
        "AND open_time = (SELECT max(open_time) FROM ohlcv_bars WHERE symbol = :s)"
    )
    async with engine.begin() as conn:
        await conn.execute(revise, {"s": SYM})
    revised = await pg.fetch("read_through", [SYM, ETH])
    assert pg.block(revised)["data_hash"] != pg.block(first)["data_hash"]


async def test_two_concurrent_cold_helpers_agree_and_leave_one_contiguous_coverage(
    pg: World,
    engine: AsyncEngine,  # noqa: F811
) -> None:
    one, two = await asyncio.gather(
        pg.fetch("read_through", [SYM, ETH]), pg.fetch("read_through", [SYM, ETH])
    )
    assert pg.block(one)["data_hash"] == pg.block(two)["data_hash"]
    assert pg.block(one)["store"] == pg.block(two)["store"] == "ok"
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT symbol, covered_from, covered_to FROM ohlcv_coverage ORDER BY symbol")
            )
        ).all()
    assert [r[0] for r in rows] == [SYM, ETH] and len({(r[1], r[2]) for r in rows}) == 1
    assert (rows[0][2] - rows[0][1]).total_seconds() == 400 * 3600
