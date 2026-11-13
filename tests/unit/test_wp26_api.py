"""WP2.6 P6a (SY-26-80..89): the history-cache wiring of backtest and optimize.

Hermetic: a scripted fake of ``CCXTMarketDataService`` and the in-memory store, no Postgres, no
network, no wall clock (``runs._window_now`` is patched).  ``World`` is the one shared fixture:
it records every call the helper makes, so mode ``off`` is pinned to today's exact calls.
The DSN-gated half is ``tests/integration/test_wp26_api_pg.py``.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import math
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import api.routers.runs as runs
import api.services.ohlcv_history as hist
import data.ohlcv_cache as oc
import data.ohlcv_integrity as leaf
from api.config import Settings, get_settings
from api.services.cost_profiles import MAX_BACKTEST_BARS
from common.models import OHLCVBar
from common.types import TimeFrame
from data.market_data import DataNotAvailableError
from data.services.ccxt_market_data import RangeFetch
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _BACKTEST,
    _C,
    _OPT_BODY,
    _clean_run_tasks,
    _hdr,
    app,
)

_REPO = Path(__file__).resolve().parents[2]
T0 = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday, on every grid
NOW = datetime(2026, 3, 10, 12, 0, 30, tzinfo=UTC)
H = timedelta(hours=1)
SYM, ETH, EXCH = "BTC/EUR", "ETH/EUR", "kraken"
LOG = "runs.backtest_ohlcv_rejected"


def ms(moment: datetime) -> int:
    return int(moment.timestamp()) * 1000  # independent of the module under test


def mk(symbol: str, n: int, tf: TimeFrame = TimeFrame.ONE_HOUR, step: timedelta = H) -> list[Any]:
    closes = [
        Decimal(str(round(100 + 12 * math.sin(i / 5) + (i * 37 % 11) / 10, 2))) for i in range(n)
    ]
    out, prev = [], closes[0]
    for i, close in enumerate(closes):
        high, low = max(prev, close), min(prev, close)
        out.append(
            OHLCVBar(
                symbol=symbol,
                timeframe=tf,
                timestamp=T0 + i * step,
                open=prev,
                high=high,
                low=low,
                close=close,
                volume=Decimal("10"),
            )
        )
        prev = close
    return out


def canon(bars: list[Any], tf_s: int = 3600) -> list[leaf.Candle]:
    return [
        leaf.canonicalise_candle(
            ms(b.timestamp), b.open, b.high, b.low, b.close, b.volume, timeframe_s=tf_s
        )[0]
        for b in bars
    ]


class Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    warning = error = info


class World:
    """Fake exchange service + store + settings, with a call log shared by every test."""

    def __init__(self, mp: pytest.MonkeyPatch) -> None:
        self.mp, self.events, self.series = mp, [], {}
        self.complete, self.store, self.built = True, oc.InMemoryOhlcvStore(), 0
        self.log = Log()

    def rows(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime) -> list[Any]:
        step = hist._TIMEFRAME_SECONDS[timeframe]
        stop = min(end.timestamp(), (int(NOW.timestamp()) // step) * step + step)
        return [
            b
            for b in self.series[symbol]
            if start <= b.timestamp and b.timestamp.timestamp() < stop
        ]

    async def fetch(
        self,
        mode: str,
        symbols: list[str],
        *,
        tf: TimeFrame = TimeFrame.ONE_HOUR,
        start: datetime = T0,
        end: datetime = T0 + 400 * H,
        now: datetime | None = NOW,
    ) -> Any:
        ns = SimpleNamespace(
            exchange_api_key=None,
            exchange_api_secret=None,
            exchange_api_passphrase=None,
            exchange_id=EXCH,
            ohlcv_cache_mode=mode,
        )
        self.mp.setattr("api.config.get_settings", lambda: ns)
        return await runs._fetch_bars_for_backtest(symbols, tf, start, end, self.log, now=now)

    def block(self, bars: Any) -> dict[str, Any]:
        block = bars.ohlcv_data
        assert isinstance(block, dict)
        return block


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = World(monkeypatch)

    class Svc:
        def __init__(self, **kwargs: Any) -> None:
            w.events.append(("init", kwargs))

        async def connect(self) -> None:
            w.events.append("connect")

        async def close(self) -> None:
            w.events.append("close")

        def check_series(self, symbol: str, timeframe: TimeFrame) -> None:
            w.events.append(("check", symbol))
            if symbol not in w.series:
                raise DataNotAvailableError("unknown symbol")

        async def fetch_ohlcv_range(
            self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime
        ) -> list[Any]:
            w.events.append(("range", symbol, start, end))
            await asyncio.sleep(0)
            return w.rows(symbol, timeframe, start, end)

        async def fetch_ohlcv_range_ex(
            self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime
        ) -> RangeFetch:
            w.events.append(("ex", symbol))
            await asyncio.sleep(0)
            rows = w.rows(symbol, timeframe, start, end)
            w.events.append(("done", symbol))
            return RangeFetch(rows, w.complete, "stop" if w.complete else "page_cap", 1, 0)

    def build(_sessions: object) -> oc.InMemoryOhlcvStore:
        w.built += 1
        return w.store

    monkeypatch.setattr("data.services.ccxt_market_data.CCXTMarketDataService", Svc)
    monkeypatch.setattr(hist, "SqlOhlcvStore", build)
    monkeypatch.setattr(hist, "_store_session_factory", lambda: object())
    monkeypatch.setattr(runs, "_window_now", lambda: NOW)
    return w


# ---------------------------------------------------------------------------
# DoD 1: mode off (and no clock) is today's path, byte for byte
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("mode", "now"), [("off", NOW), ("read_through", None)])
async def test_off_or_no_clock_makes_exactly_todays_calls_and_builds_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, mode: str, now: datetime | None
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("no source and no store in this mode")

    monkeypatch.setattr(oc, "HistoricalBarSource", boom)
    monkeypatch.setattr(hist, "SqlOhlcvStore", boom)
    world.series = {SYM: mk(SYM, 5), ETH: mk(ETH, 5)}
    end = T0 + 5 * H
    out = await world.fetch(mode, [ETH, SYM], end=end, now=now)
    init = {
        "exchange_id": EXCH,
        "api_key": None,
        "api_secret": None,
        "api_passphrase": None,
        "cache_ttl_seconds": 0,
    }
    assert world.events == [
        ("init", init),
        "connect",
        ("range", ETH, T0, end),
        ("range", SYM, T0, end),
        "close",
    ]
    assert type(out) is dict and list(out) == [ETH, SYM] and out[SYM] == world.series[SYM]
    assert getattr(out, "ohlcv_data", None) is None
    assert [e for e, _ in world.log.events] == ["runs.backtest_fetching_bars"] + [
        "runs.backtest_bars_fetched"
    ] * 2


# ---------------------------------------------------------------------------
# B1-B4, G1, DoD 2/3 (unit half): the block, cold = warm, a revised stored bar
# ---------------------------------------------------------------------------
async def test_cold_equals_warm_and_the_block_has_exactly_this_shape(world: World) -> None:
    world.series = {SYM: mk(SYM, 400), ETH: mk(ETH, 400)}
    cold = await world.fetch("read_through", [ETH, SYM])
    sums = [leaf.summarise_series(EXCH, s, 3600, canon(world.series[s])) for s in (SYM, ETH)]
    line = {
        "n_bars": 400,
        "first_ms": ms(T0),
        "last_ms": ms(T0 + 399 * H),
        "from_ms": ms(T0),
        "to_ms": ms(T0 + 400 * H),
        "source": "exchange",
        "volatile_tail_bars": 0,
        "fetch_complete": True,
        "bypass_disjoint": False,
    }
    assert world.block(cold) == {
        "version": "wp26-1",
        "data_hash": leaf.run_digest(sums),
        "mode": "read_through",
        "store": "ok",
        "store_reason": "none",
        "exchange": EXCH,
        "timeframe_s": 3600,
        "settle_bars": 1,
        "rounded_values": 0,
        "divergent_bars": 0,
        "requested_from_ms": ms(T0),
        "requested_to_ms": ms(T0 + 400 * H),
        "series": [
            {**line, "symbol": s, "digest": d.digest} for s, d in zip((SYM, ETH), sums, strict=True)
        ],
    }
    assert list(cold) == [ETH, SYM]  # the caller's order, the series list is sorted
    info = {"mode": "read_through", "store": "ok", "store_reason": "none"}
    assert world.log.events[-1] == (
        "runs.backtest_ohlcv_data",
        {**info, "data_hash": world.block(cold)["data_hash"]},
    )
    key = oc.SeriesKey(EXCH, SYM, 3600)
    stored = (await world.store.read(key, ms(T0), ms(T0 + 400 * H))).bars
    for bar, candle in zip(cold[SYM], stored, strict=True):  # G1: Candle -> OHLCVBar, five fields
        assert ms(bar.timestamp) == candle.open_ms
        for got, want in zip(
            (bar.open, bar.high, bar.low, bar.close, bar.volume),
            (candle.open_, candle.high, candle.low, candle.close, candle.volume),
            strict=True,
        ):
            assert got.as_tuple() == want.as_tuple()
    warm = await world.fetch("read_through", [ETH, SYM])
    drop = ("source", "rounded_values", "divergent_bars")  # run properties, not data (CF-26-72)

    def data_view(b: Any) -> dict[str, Any]:
        blk = dict(world.block(b))
        blk["series"] = [{k: v for k, v in s.items() if k not in drop} for s in blk["series"]]
        return {k: v for k, v in blk.items() if k not in drop}

    assert data_view(warm) == data_view(cold) and dict(warm) == dict(cold)
    assert {s["source"] for s in world.block(warm)["series"]} == {"cache"}
    first = world.store._bars[key][ms(T0)]  # a revised stored bar changes the hash
    world.store._bars[key][ms(T0)] = dataclasses.replace(first, volume=first.volume + 1)
    revised = await world.fetch("read_through", [ETH, SYM])
    assert world.block(revised)["data_hash"] != world.block(cold)["data_hash"]


async def test_the_series_list_is_the_requested_symbols_including_an_empty_one(
    world: World,
) -> None:
    world.series = {SYM: mk(SYM, 50), ETH: []}
    out = await world.fetch("read_through", [SYM, ETH], end=T0 + 50 * H)
    lines = world.block(out)["series"]
    assert [x["symbol"] for x in lines] == [SYM, ETH]
    assert (lines[1]["n_bars"], lines[1]["first_ms"], lines[1]["last_ms"]) == (0, 0, 0)
    both = [leaf.summarise_series(EXCH, s, 3600, canon(world.series[s])) for s in (SYM, ETH)]
    assert world.block(out)["data_hash"] == leaf.run_digest(both)
    assert world.block(out)["data_hash"] != leaf.run_digest(both[:1])  # n = 0 is in the hash


@pytest.mark.parametrize(
    ("start", "end", "to_ms", "n"),
    [
        (T0, NOW + timedelta(days=10), ms(datetime(2026, 3, 10, 12, tzinfo=UTC)), 400),  # B6
        (NOW + 5 * H, NOW + 10 * H, ms(NOW + 6 * H - timedelta(seconds=30)), 0),  # CF-26-69
    ],
    ids=["future-end", "start-after-the-closed-end"],
)
async def test_a_window_past_now_persists_floor_now_and_the_request(
    world: World, start: datetime, end: datetime, to_ms: int, n: int
) -> None:
    world.series = {SYM: mk(SYM, 400)}
    out = await world.fetch("read_through", [SYM], start=start, end=end)
    block, (line,) = world.block(out), world.block(out)["series"]
    assert (block["requested_from_ms"], block["requested_to_ms"]) == (ms(start), ms(end))
    assert (line["n_bars"], line["to_ms"]) == (n, to_ms)
    assert line["to_ms"] >= line["from_ms"]


@pytest.mark.parametrize(
    ("tf", "step"),
    [(TimeFrame.FIFTEEN_MINUTES, timedelta(minutes=15)), (TimeFrame.ONE_DAY, timedelta(days=1))],
)
async def test_the_other_cacheable_timeframes_use_their_own_grid(
    world: World, tf: TimeFrame, step: timedelta
) -> None:
    world.series = {SYM: mk(SYM, 20, tf, step)}
    block = world.block(out := await world.fetch("read_through", [SYM], tf=tf, end=T0 + 20 * step))
    assert (block["store"], block["timeframe_s"]) == ("ok", int(step.total_seconds()))
    assert block["series"][0]["n_bars"] == 20 and {b.timeframe for b in out[SYM]} == {tf}


# ---------------------------------------------------------------------------
# C1-C3: error mapping, by type and code only
# ---------------------------------------------------------------------------
_INTERNAL = {"bad_type", "bad_digest", "not_canonical"}  # SY-26-33, hand-listed on purpose
_CASES = (
    [
        (
            leaf.OhlcvDataError(c),
            500 if c in _INTERNAL else 422,
            "internal_ohlcv_invariant" if c in _INTERNAL else c,
        )
        for c in leaf.OHLCV_CODES
    ]
    + [(oc.OhlcvInvariantError(c), 500, "internal_ohlcv_invariant") for c in leaf.OHLCV_CODES]
    + [(oc.OhlcvStoreError(c), 500, "internal_ohlcv_invariant") for c in oc.STORE_CODES]
    + [
        (oc.OhlcvRequireError("ohlcv_store_unavailable"), 503, "ohlcv_store_unavailable"),
        (oc.OhlcvRequireError("ohlcv_timeframe_not_cached"), 422, "ohlcv_timeframe_not_cached"),
    ]
)


@pytest.mark.parametrize(("exc", "http", "code"), _CASES, ids=lambda x: getattr(x, "code", x))
async def test_every_ohlcv_error_is_mapped_by_type_and_code_with_no_text(
    world: World, monkeypatch: pytest.MonkeyPatch, exc: Exception, http: int, code: str
) -> None:
    class Failing:
        def __init__(self, *_a: object, **_k: object) -> None: ...

        async def connect(self) -> None: ...

        async def get_ranges(self, *_a: object) -> list[Any]:
            raise exc

    monkeypatch.setattr(oc, "HistoricalBarSource", Failing)
    world.series = {SYM: mk(SYM, 5)}
    with pytest.raises(HTTPException) as info:
        await world.fetch("read_through", [SYM], end=T0 + 5 * H)
    err = info.value
    assert (err.status_code, err.detail) == (http, {"code": code})
    assert err.__cause__ is None and err.__suppress_context__ and world.events[-1] == "close"
    assert world.log.events[-1] == (LOG, {"code": exc.code, "error_type": type(exc).__name__})  # type: ignore[attr-defined]
    assert str(exc) not in repr(world.log.events) + repr(err.detail)


@pytest.mark.parametrize("tf", [TimeFrame.ONE_HOUR, TimeFrame.ONE_WEEK])
@pytest.mark.parametrize("mode", ["read_through", "require"])
async def test_a_sub_millisecond_window_is_a_closed_422_never_a_value_error(
    world: World, mode: str, tf: TimeFrame
) -> None:
    world.series = {SYM: mk(SYM, 5)}
    start, end = T0 + timedelta(microseconds=100), T0 + timedelta(microseconds=900)
    with pytest.raises(HTTPException) as info:
        await world.fetch(mode, [SYM], tf=tf, start=start, end=end)
    assert (info.value.status_code, info.value.detail) == (422, {"code": "out_of_range"})
    assert world.events[-1] == "close" and world.built == 0


async def test_a_row_error_is_a_422_never_a_500_and_the_store_is_not_written(world: World) -> None:
    bad = mk(SYM, 3)[0].model_copy(
        update={
            "open": Decimal("1E-13"),
            "high": Decimal("1E-13"),
            "low": Decimal("1E-13"),
            "close": Decimal("1E-13"),
        }
    )
    world.series = {SYM: [bad, *mk(SYM, 3)[1:]]}
    for mode in ("read_through", "require"):
        with pytest.raises(HTTPException) as info:
            await world.fetch(mode, [SYM], end=T0 + 3 * H)
        assert (info.value.status_code, info.value.detail) == (422, {"code": "bad_ohlc"})
    assert world.store._coverage == {}


async def test_an_invariant_broken_inside_the_store_call_is_a_500(world: World) -> None:
    world.series = {SYM: mk(SYM, 5)}
    world.store.fail_with = leaf.OhlcvDataError("out_of_range")
    with pytest.raises(HTTPException) as info:
        await world.fetch("read_through", [SYM], end=T0 + 5 * H)
    assert (info.value.status_code, info.value.detail) == (
        500,
        {"code": "internal_ohlcv_invariant"},
    )


async def test_store_down_is_fail_open_in_read_through_and_fail_closed_in_require(
    world: World,
) -> None:
    world.series = {SYM: mk(SYM, 50)}
    healthy = await world.fetch("read_through", [SYM], end=T0 + 50 * H)
    world.store = oc.InMemoryOhlcvStore()
    world.store.fail_with = oc.OhlcvStoreError("missing_table")
    down = await world.fetch("read_through", [SYM], end=T0 + 50 * H)
    assert (world.block(down)["store"], world.block(down)["store_reason"]) == (
        "unavailable",
        "missing_table",
    )
    assert world.block(down)["data_hash"] == world.block(healthy)["data_hash"] and dict(
        down
    ) == dict(healthy)
    with pytest.raises(HTTPException) as info:
        await world.fetch("require", [SYM], end=T0 + 50 * H)
    assert (info.value.status_code, info.value.detail) == (503, {"code": "ohlcv_store_unavailable"})


async def test_require_with_a_request_far_from_the_cached_range_is_a_503_read_through_serves(
    world: World,
) -> None:
    world.series = {SYM: mk(SYM, 400)}
    await world.fetch("read_through", [SYM], end=T0 + 100 * H)  # coverage [0, 100)
    far = {"start": T0 + 300 * H, "end": T0 + 400 * H}
    with pytest.raises(HTTPException) as info:
        await world.fetch("require", [SYM], **far)
    assert (info.value.status_code, info.value.detail) == (503, {"code": "ohlcv_store_unavailable"})
    out = await world.fetch("read_through", [SYM], **far)
    assert world.block(out)["series"][0]["bypass_disjoint"] is True and len(out[SYM]) == 100


async def test_one_failing_series_makes_the_store_partial_with_the_first_reason_in_symbol_order(
    world: World,
) -> None:
    class Flaky(oc.InMemoryOhlcvStore):
        async def read(self, key: oc.SeriesKey, start_ms: int, end_ms: int) -> oc.StoredRead:
            if key.symbol in (ETH, SYM):
                raise oc.OhlcvStoreError("denied" if key.symbol == SYM else "timeout")
            return await super().read(key, start_ms, end_ms)

    world.store = Flaky()
    world.series = {s: mk(s, 5) for s in (SYM, ETH, "ADA/EUR")}
    block = world.block(await world.fetch("read_through", [ETH, SYM, "ADA/EUR"], end=T0 + 5 * H))
    assert (block["store"], block["store_reason"]) == ("partial", "denied")
    assert world.log.events[-1][0] == "runs.backtest_ohlcv_data"
    assert world.log.events[-1][1] == {
        "mode": "read_through",
        "store": "partial",
        "store_reason": "denied",
        "data_hash": block["data_hash"],
    }


async def test_the_counters_describe_the_run_not_the_data(world: World) -> None:
    odd = mk(SYM, 110)
    odd[0] = odd[0].model_copy(
        update={k: Decimal("100.1234567890123") for k in ("open", "high", "low", "close")}
    )
    world.series = {SYM: odd}
    cold = await world.fetch("read_through", [SYM], end=T0 + 100 * H)
    assert world.block(cold)["rounded_values"] > 0 and world.block(cold)["divergent_bars"] == 0
    world.series[SYM][99] = odd[99].model_copy(update={"volume": Decimal("11")})  # revised upstream
    again = await world.fetch("read_through", [SYM], end=T0 + 110 * H)  # re-fetches the edge bar
    assert (world.block(again)["rounded_values"], world.block(again)["divergent_bars"]) == (0, 1)
    assert again[SYM][99].volume == Decimal("10")  # the stored value wins (keep-first)


async def test_a_cut_short_fetch_is_marked_and_require_refuses_it(world: World) -> None:
    world.series, world.complete = {SYM: mk(SYM, 5)}, False
    out = await world.fetch("read_through", [SYM], end=T0 + 5 * H)
    assert world.block(out)["series"][0]["fetch_complete"] is False
    with pytest.raises(HTTPException) as info:
        await world.fetch("require", [SYM], end=T0 + 5 * H)
    assert info.value.status_code == 503


async def test_a_unique_error_site_is_mapped_for_the_run_digest_and_an_unknown_symbol_is_a_400(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.series = {SYM: mk(SYM, 5)}
    with pytest.raises(HTTPException) as unknown:
        await world.fetch("read_through", [SYM, "ZZZ/EUR"], end=T0 + 5 * H)
    assert unknown.value.status_code == 400
    monkeypatch.setattr(
        leaf,
        "run_digest",
        lambda _s: (_ for _ in ()).throw(leaf.OhlcvDataError("duplicate_series")),
    )
    with pytest.raises(HTTPException) as info:
        await world.fetch("read_through", [SYM], end=T0 + 5 * H)
    assert (info.value.status_code, info.value.detail) == (
        500,
        {"code": "internal_ohlcv_invariant"},
    )


# ---------------------------------------------------------------------------
# B5: 1w (the leaf does not hold it): today's bars, a block without hashes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["read_through", "require"])
async def test_a_weekly_timeframe_is_bypassed_or_refused_after_validation(
    world: World, mode: str
) -> None:
    world.series = {SYM: mk(SYM, 6, TimeFrame.ONE_WEEK, timedelta(weeks=1))}
    world.complete = False
    args: dict[str, Any] = {"tf": TimeFrame.ONE_WEEK, "end": T0 + timedelta(weeks=6)}
    with pytest.raises(HTTPException) as unknown:  # today's 400 first, in both modes
        await world.fetch(mode, [ETH], **args)
    assert unknown.value.status_code == 400
    assert not [e for e in world.events if e[0] == "ex"]
    if mode == "require":
        with pytest.raises(HTTPException) as info:
            await world.fetch(mode, [SYM], **args)
        assert (info.value.status_code, info.value.detail) == (
            422,
            {"code": "ohlcv_timeframe_not_cached"},
        )
    else:
        out = await world.fetch(mode, [SYM], **args)
        (line,) = world.block(out)["series"]
        assert out[SYM] == world.series[SYM]  # today's bars, pre-trim
        assert line == {
            "symbol": SYM,
            "n_bars": 6,
            "first_ms": ms(T0),
            "last_ms": ms(T0 + timedelta(weeks=5)),
            "from_ms": ms(T0),
            "to_ms": ms(T0 + timedelta(weeks=6)),
            "digest": None,
            "source": "exchange",
            "volatile_tail_bars": 0,
            "fetch_complete": False,
            "bypass_disjoint": False,
        }
        block = world.block(out)
        assert (block["data_hash"], block["store"], block["store_reason"]) == (
            None,
            "bypass",
            "timeframe_not_cached",
        )
        assert (block["rounded_values"], block["divergent_bars"], block["timeframe_s"]) == (
            0,
            0,
            604800,
        )
    assert world.built == 0 and world.events.count("connect") == world.events.count("close") > 0


# ---------------------------------------------------------------------------
# A1-A5: adapter arithmetic, construction, order
# ---------------------------------------------------------------------------
async def test_the_adapter_uses_integer_ms_in_both_directions(world: World) -> None:
    bar = mk(SYM, 1)[0].model_copy(
        update={"timestamp": T0 + timedelta(milliseconds=999, microseconds=999)}
    )
    world.series = {SYM: [bar]}
    up = hist._Upstream(await _svc(world), TimeFrame.ONE_HOUR)
    got = await up.fetch_range(SYM, 3600, ms(T0), ms(T0 + H))
    assert got.complete and [r.open_ms for r in got.rows] == [ms(T0) + 999]  # floor, not round
    assert got.rows[0].values == (bar.open, bar.high, bar.low, bar.close, bar.volume)
    assert [e for e in world.events if e[0] == "range" or e[0] == "ex"] == [("ex", SYM)]
    assert hist._ms(datetime(2024, 1, 1, 0, 0, 0, 999_999)) == ms(T0) + 999  # naive means UTC


async def _svc(world: World) -> Any:
    import data.services.ccxt_market_data as cmd

    return cmd.CCXTMarketDataService()


async def test_the_source_is_built_once_with_the_routers_clock_and_the_request_cap(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real = oc.HistoricalBarSource

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(oc, "HistoricalBarSource", spy)
    world.series = {SYM: mk(SYM, 5)}
    await world.fetch("require", [SYM], end=T0 + 5 * H)
    (kw,) = seen
    assert (kw["exchange"], kw["now_ms"], kw["mode"], kw["cap_bars"]) == (
        EXCH,
        ms(NOW),
        "require",
        MAX_BACKTEST_BARS,
    )
    assert kw["offload"] is asyncio.to_thread and callable(kw["log"].warning)
    assert oc.REQUEST_CAP_BARS == MAX_BACKTEST_BARS
    assert world.built == 1


async def test_connect_comes_first_and_series_are_fetched_one_at_a_time_in_sorted_order(
    world: World,
) -> None:
    world.series = {SYM: mk(SYM, 5), ETH: mk(ETH, 5), "ADA/EUR": mk("ADA/EUR", 5)}
    await world.fetch("read_through", [ETH, SYM, "ADA/EUR"], end=T0 + 5 * H)
    calls = [e for e in world.events if e[0] in ("ex", "done")]
    assert [e[1] for e in calls] == ["ADA/EUR", "ADA/EUR", SYM, SYM, ETH, ETH]  # no overlap
    assert [e[0] for e in calls] == ["ex", "done"] * 3
    assert world.events[1] == "connect" and world.events[-1] == "close"


def _calls(fn: ast.AST, name: str) -> list[int]:
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name
    ]


@pytest.mark.parametrize(("path", "fn"), [("runs", "create_run"), ("optimize", "run_optimization")])
def test_the_bounds_check_precedes_the_fetch_and_the_fetch_gets_the_routers_clock(
    path: str, fn: str
) -> None:
    tree = ast.parse((_REPO / f"apps/api/routers/{path}.py").read_text())
    (func,) = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == fn]
    fetch = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_fetch_bars_for_backtest"
    ]
    (call,) = fetch
    assert _calls(func, "check_backtest_request_bounds")[0] < call.lineno
    assert [k.arg for k in call.keywords if k.arg == "now"] == ["now"]


# ---------------------------------------------------------------------------
# A6: nothing live or paper can reach the cache; the module is not imported on the off path
# ---------------------------------------------------------------------------
_BANNED = ("ohlcv_cache", "ohlcv_history", "ohlcv_store")


def _imported(path: Path) -> list[str]:
    names: list[str] = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom):
            names += [n.module or "", *(a.name for a in n.names)]
        elif isinstance(n, ast.Import):
            names += [a.name for a in n.names]
    return [x for x in names if any(b in x for b in _BANNED)]


def test_live_and_paper_code_never_imports_the_history_cache() -> None:
    allowed = {"ohlcv_history.py", "ohlcv_store.py"}
    watched = [p for p in (_REPO / "apps/api/services").glob("*.py") if p.name not in allowed]
    watched += [
        _REPO / "packages/trading/strategy_engine.py",
        *(_REPO / "packages/trading").rglob("*.py"),
    ]
    assert len(watched) > 20 and (_REPO / "apps/api/services/run_orchestrator.py") in watched
    assert {p.name: _imported(p) for p in watched if _imported(p)} == {}
    assert _imported(_REPO / "apps/api/services/ohlcv_history.py")  # the scan sees imports


def test_runs_imports_the_wiring_only_inside_the_helper_and_optimize_not_at_all() -> None:
    runs_tree = ast.parse((_REPO / "apps/api/routers/runs.py").read_text())
    assert not [
        n
        for n in runs_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom)) and "ohlcv" in ast.dump(n)
    ]
    (helper,) = [
        n
        for n in ast.walk(runs_tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_fetch_bars_for_backtest"
    ]
    assert "ohlcv_history" in ast.dump(helper)
    assert not _imported(_REPO / "apps/api/routers/optimize.py")


# ---------------------------------------------------------------------------
# D1, F2
# ---------------------------------------------------------------------------
def test_the_setting_defaults_to_off_and_refuses_other_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OHLCV_CACHE_MODE", raising=False)
    base = {"database_url": "postgresql+asyncpg://t:t@localhost:5432/t"}
    assert Settings(_env_file=None, **base).ohlcv_cache_mode == "off"  # type: ignore[arg-type]
    for good in ("read_through", "require"):
        assert Settings(_env_file=None, ohlcv_cache_mode=good, **base).ohlcv_cache_mode == good  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ohlcv_cache_mode="REQUIRE", **base)  # type: ignore[arg-type]


async def test_the_store_runs_on_its_own_engine_that_hides_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.db.session as session_mod

    ns = SimpleNamespace(
        database_url=SimpleNamespace(
            get_secret_value=lambda: "postgresql+asyncpg://u:p@127.0.0.1:1/d"
        ),
        db_pool_size=1,
        db_max_overflow=0,
        db_pool_timeout=1.0,
        debug=False,
    )
    monkeypatch.setattr(session_mod, "get_settings", lambda: ns)
    monkeypatch.setattr(session_mod, "_engine", None)
    monkeypatch.setattr(session_mod, "_session_factory", None)
    engine = hist._store_session_factory().kw["bind"]
    try:
        assert engine.sync_engine.hide_parameters is True
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# HTTP: write sites, mode off equals cache, require fail-closed (DoD 1, 3, 4)
# ---------------------------------------------------------------------------
@pytest.fixture()
def c(app: Any) -> Generator[_C, None, None]:  # noqa: F811
    client = _C(app)
    yield client
    client.close()


def _use_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("OHLCV_CACHE_MODE", mode)
    get_settings.cache_clear()  # the ``app`` fixture clears it again at teardown


def _post(c: _C, **overrides: Any) -> Any:
    return c.client.post("/api/v1/runs", json={**_BACKTEST, "seed": 7, **overrides}, headers=_hdr())


@pytest.mark.parametrize("router", ["runs", "optimize"])
@pytest.mark.parametrize("with_block", [True, False])
def test_the_routers_persist_the_block_only_when_the_helper_returns_one_and_pass_now(
    c: _C, router: str, with_block: bool
) -> None:
    body = _BACKTEST if router == "runs" else _OPT_BODY
    bars = {body["symbols"][0]: mk(body["symbols"][0], 120 if router == "runs" else 300)}
    out = hist.HistoryBars(bars) if with_block else dict(bars)
    if with_block:
        out.ohlcv_data = {"version": "wp26-1", "series": []}  # type: ignore[union-attr]
    fetch = AsyncMock(return_value=out)
    with (
        patch.object(runs, "_window_now", lambda: NOW),
        patch(f"api.routers.{router}._fetch_bars_for_backtest", new=fetch),
    ):
        if router == "runs":
            assert _post(c).status_code == 201
            meta = c.created_run().config
        else:
            assert c.client.post("/api/v1/optimize", json=body).status_code == 200
            meta = next(
                x.args[0]
                for x in c.db.add.call_args_list
                if type(x.args[0]).__name__ == "OptimizationRunORM"
            ).run_metadata
    assert fetch.call_args.kwargs["now"] == NOW
    assert meta.get("ohlcv_data") == ({"version": "wp26-1", "series": []} if with_block else None)


def test_a_backtest_through_the_cache_equals_mode_off_and_persists_the_block(
    c: _C, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    sym = _BACKTEST["symbols"][0]
    world.series = {sym: mk(sym, 120)}
    configs: list[dict[str, Any]] = []
    for mode in ("off", "read_through", "read_through"):  # off, cold, warm
        _use_mode(monkeypatch, mode)
        assert _post(c).status_code == 201
        configs.append(dict(c.created_run().config))
        c.db.add.reset_mock()
    off, cold, warm = configs
    assert "ohlcv_data" not in off
    for run in (cold, warm):
        block = run.pop("ohlcv_data")
        assert block["store"] == "ok" and len(block["data_hash"]) == 64
    assert cold == warm == off  # equal metrics, trades, alignment: only the block differs


def test_require_with_the_store_down_is_a_503_and_the_run_row_is_an_error(
    c: _C, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    sym = _BACKTEST["symbols"][0]
    world.series = {sym: mk(sym, 120)}
    world.store.fail_with = oc.OhlcvStoreError("unavailable")
    _use_mode(monkeypatch, "require")
    resp = _post(c)
    assert resp.status_code == 503 and resp.json()["detail"] == {"code": "ohlcv_store_unavailable"}
    run = c.created_run()
    assert run.status == "error" and "ohlcv_data" not in run.config
    world.store.fail_with = None  # and on success every series is stored and complete
    c.db.add.reset_mock()
    assert _post(c).status_code == 201
    (line,) = c.created_run().config["ohlcv_data"]["series"]
    assert (c.created_run().config["ohlcv_data"]["store"], line["fetch_complete"]) == ("ok", True)


def test_an_optimize_run_writes_the_same_block_shape(
    c: _C, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    sym = _OPT_BODY["symbols"][0]
    world.series = {sym: mk(sym, 300)}
    _use_mode(monkeypatch, "read_through")
    with patch.object(runs, "_window_now", lambda: NOW):
        resp = c.client.post("/api/v1/optimize", json=_OPT_BODY)
    assert resp.status_code == 200, resp.text
    run = next(
        x.args[0]
        for x in c.db.add.call_args_list
        if type(x.args[0]).__name__ == "OptimizationRunORM"
    )
    block = run.run_metadata["ohlcv_data"]
    assert (block["version"], block["store"], block["series"][0]["n_bars"]) == ("wp26-1", "ok", 300)
