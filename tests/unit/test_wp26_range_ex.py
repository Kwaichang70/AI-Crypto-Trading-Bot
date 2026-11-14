"""
WP2.6 P5 (SY-26-15): ``fetch_ohlcv_range_ex`` and the unchanged ``fetch_ohlcv_range``.

``fetch_ohlcv_range`` now delegates to ``fetch_ohlcv_range_ex(...).bars``.  The delegation
must change nothing for its callers (``strategy_engine`` trailing gap, ``runs.py``): every
scenario below is replayed on a scripted fake ccxt exchange with a fake clock, and the full
trace (ccxt call arguments, sleeps, log events, result or error) is hashed (``error_at`` is a code
location inside this file and is left out).  The expected
hashes in ``GOLDEN`` were produced by the pre-P5 module (``p5-differential.py`` replays both
and shows them equal), so a drift in call sequence, throttling, retry or logging fails here.
No network; the clock and ``random`` are fixed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from contextlib import ExitStack
from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import ccxt.async_support as ccxt_async
import pytest
from structlog.testing import capture_logs

from common.types import TimeFrame
from data.services import ccxt_market_data as cmd

H = 3_600_000
T0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
NOW = T0 + 1000 * H + H // 2  # fixed clock: the forming bar is slot 1000
SYMBOL = "BTC/USDT"


def row(i: int, close: float = 102.0) -> list[float]:
    return [T0 + i * H, close, close + 1.0, close - 1.0, close, 1_000.0]


class _Ex:
    """Scripted ccxt exchange: ``script[n]`` is call n's rows or exception; else the slots."""

    def __init__(self, sc: dict[str, Any]) -> None:
        self.sc, self.n = sc, 0
        self.timeframes = {"1h": "1 hour"}
        self.load_markets = AsyncMock(return_value={SYMBOL: {}})
        self.fetch_ohlcv = AsyncMock(side_effect=self._fetch)
        self.close = AsyncMock()

    async def _fetch(self, symbol: str, timeframe: str, since: int, limit: int) -> Any:
        k, self.n = self.n, self.n + 1
        out = self.sc.get("script", {}).get(k)
        if isinstance(out, BaseException):
            raise out
        if out is not None:
            return out
        first = -(-(since - T0) // H)  # first slot index with ts >= since
        n, slots = min(limit, self.sc.get("max", 300)), self.sc.get("slots", set())
        if self.sc.get("mode", "window") == "window":
            idx = [i for i in range(first, first + n) if i in slots]
        else:
            idx = sorted(i for i in slots if i >= first)[:n]
        return [row(i) for i in idx]


_RL, _NET = ccxt_async.RateLimitExceeded("x"), ccxt_async.NetworkError("x")
_ALL, _GAPS = set(range(2000)), set(range(1000)) - set(range(300, 700))
_FAIL = dict.fromkeys((2, 3, 4, 5), _NET)


def _sc(start: float, end: float, reason: str | None, **kw: Any) -> dict[str, Any]:
    return {"start": start, "end": end, "reason": reason, **kw}


#: name -> scenario.  ``start``/``end`` are slot indexes; ``reason`` is the new ``end_reason``.
SCENARIOS: dict[str, dict[str, Any]] = {
    "dense": _sc(0, 700, "stop", slots=set(range(700))),
    "dense_past_end": _sc(0, 700, "reached_end", slots=_ALL),
    "windowed_gaps": _sc(0, 1000, "stop", slots=_GAPS),
    "all_empty": _sc(0, 700, "stop", slots=set()),
    "future_end": _sc(900, 5000, "reached_end", slots=_ALL),
    "count_mode_short": _sc(0, 500, "stop", slots=set(range(500)), mode="count", max=120),
    "page_cap": _sc(0, 2000, "page_cap", slots=_ALL, mode="count", max=1),
    "no_progress": _sc(0, 700, "no_progress", slots=_ALL, script={1: [row(0)]}),
    "unaligned_start": _sc(0.5, 50, "reached_end", slots=_ALL),
    "zero_window": _sc(1100, 1200, "stop", slots=_ALL),
    "end_before_start": _sc(10, 10, None, slots=_ALL),
    "retry_then_ok": _sc(0, 50, "reached_end", slots=_ALL, script={0: _RL, 1: _NET}),
    "retries_exhausted": _sc(0, 50, None, script=dict.fromkeys(range(4), _RL)),
    "network_exhausted": _sc(0, 50, None, script=dict.fromkeys(range(4), _NET)),
    "bad_symbol": _sc(0, 50, None, script={0: ccxt_async.BadSymbol("x")}),
    "bad_request": _sc(0, 50, None, script={0: ccxt_async.BadRequest("x")}),
    "unavailable": _sc(0, 5, None, script={0: ccxt_async.ExchangeNotAvailable("x")}),
    "exchange_error": _sc(0, 50, None, script={0: ccxt_async.ExchangeError("x")}),
    "error_on_page_3": _sc(0, 900, None, slots=_ALL, script=_FAIL),
    "invalid_row": _sc(0, 50, None, script={0: [[T0, 5.0, 4.0, 6.0, 5.0, 1.0]]}),
}
#: sha256 of the trace, recorded from the module at 2857bd7 (before P5) by ``p5-differential.py``.
GOLDEN: dict[str, str] = {
    "all_empty": "090130ae25eaaf123a46b96cf128276c1ef7bbe0cd49092c89520a0a8f1863ff",
    "bad_request": "ae8cf7f52d1ae239b33da65470f76dbf96a89f8d3e098c99dffda677de714082",
    "bad_symbol": "46656652996cc16f5e0e365e5f49d0734eef535aae0720a259b29d00e0d5bdd9",
    "count_mode_short": "f8a29e7f8a23d86c2030819ac399627a9782f8e5bbed02c064ea423b8d065d07",
    "dense": "3ede4d8ff5260480776c0239c30451ad00284b58fc0b0f99eeb8bf16dc3ab4ed",
    "dense_past_end": "fcbd6af0dfc4264b865a19e5b0c2bb6c0542ca4721d856f0f7ee383a7ec1496f",
    "end_before_start": "2ffa0906f3ca6932f81c69e3c83fa6783bb83e43820c542a9e3579d30be8fedc",
    "error_on_page_3": "1e19d078fe25240b652b051f5a48d57cf5249e48bdfe304e03577fd5a183bbd2",
    "exchange_error": "e0f0cbfb8f38cd8a7f8d0d0a446e89e16da0e1c52a878e2a6ff62b07b9508bed",
    "future_end": "7eac0945fad3d49ebb5317636d7a062be8a0d36999d054b52491ee00adccb208",
    "invalid_row": "543b63971189fffa38e3a3467a025c705386ccb4dfd95be421d26447d52302b2",
    "network_exhausted": "a96f0aab7771384609e40422525003bcb77d5bcb54634bf33531e6dc0de4ec9d",
    "no_progress": "7b08103ae2d4ebeb3b1ee4f1b7979aa261ce3dc61916cf02c7291d4f79a4c5c7",
    "page_cap": "0816bb77cada1e4da0910bf806879862eca827f0530043ad5c8d5275f3f21be6",
    "retries_exhausted": "f555877d7a78cabff46acfc89417b35e317dfef4c65594da12999041123ec19e",
    "retry_then_ok": "c0ef6b22b16821315063e7743f5ef7b1ff3864d7c011d154967c41d06eec9332",
    "unaligned_start": "a3202067e5723aad8e036ae6230ef06d0402686c7af175e623eb8dd771eb374f",
    "unavailable": "d293dd1ae0f2f84bfb4de2a2b5872c45999dc799f887d12a7605d0f911d6fe5f",
    "windowed_gaps": "8fc9262356599db39aca971f4584567da2ae58fc96e8791f872704a26c6ac92d",
    "zero_window": "5b375aa41164899417cd3b0db286e7da6c6684b30e4ed46fada089cc090ca9ba",
}


async def trace(mod: ModuleType, name: str, *, ex_method: bool = False) -> dict[str, Any]:
    """Run one scenario on ``mod`` (old or new) and return its observable trace."""
    sc, clock, sleeps = SCENARIOS[name], [1000.0], []
    ex = _Ex(sc)

    async def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        clock[0] += seconds

    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: NOW / 1000)
    with ExitStack() as stack, capture_logs() as logs:
        stack.enter_context(patch.object(mod, "_now_ms", lambda: NOW))
        stack.enter_context(patch.object(mod, "time", fake_time))
        stack.enter_context(patch.object(mod.asyncio, "sleep", sleep))
        random.seed(0)
        svc = mod.CCXTMarketDataService(
            "binance", cache_ttl_seconds=0, page_limit=sc.get("page", 300)
        )
        svc._exchange = ex
        await svc.connect()
        logs.clear()
        start = datetime.fromtimestamp((T0 + sc["start"] * H) / 1000, UTC)
        end = datetime.fromtimestamp((T0 + sc["end"] * H) / 1000, UTC)
        call = svc.fetch_ohlcv_range_ex if ex_method else svc.fetch_ohlcv_range
        try:
            got = await call(SYMBOL, TimeFrame.ONE_HOUR, start, end)
            bars = got.bars if ex_method else got
            outcome: list[Any] = [
                "ok",
                [[int(b.timestamp.timestamp() * 1000), str(b.close)] for b in bars],
            ]
        except Exception as exc:
            got, outcome = None, ["err", type(exc).__name__, str(exc)]
    return {
        "calls": [
            [c.kwargs[k] for k in ("symbol", "timeframe", "since", "limit")]
            for c in ex.fetch_ohlcv.call_args_list
        ],
        "sleeps": sleeps,
        "logs": [
            [
                e["log_level"],
                e["event"],
                sorted(
                    (k, str(v)) for k, v in e.items() if k not in ("event", "log_level", "error_at")
                ),
            ]
            for e in logs
        ],
        "outcome": outcome,
        "got": got,
    }


def digest(t: dict[str, Any]) -> str:
    body = {k: v for k, v in t.items() if k != "got"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_fetch_ohlcv_range_trace_equals_the_pre_p5_trace(name: str) -> None:
    assert digest(await trace(cmd, name)) == GOLDEN[name], name


@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_ex_and_delegate_share_one_trace_and_the_end_reason(name: str) -> None:
    plain, ex = await trace(cmd, name), await trace(cmd, name, ex_method=True)
    assert digest(plain) == digest(ex), (
        "fetch_ohlcv_range is exactly fetch_ohlcv_range_ex(...).bars"
    )
    result = ex["got"]
    reason = SCENARIOS[name]["reason"]
    if reason is None:
        assert result is None and ex["outcome"][0] == "err"  # exceptions propagate unchanged
        return
    assert isinstance(result, cmd.RangeFetch) and result.end_reason == reason
    assert result.complete is (reason in ("stop", "reached_end"))
    done = [e for e in ex["logs"] if e[1] == "fetch_ohlcv_range_complete"]
    fields = dict(done[0][2])
    assert (fields["pages"], fields["empty_pages"]) == (str(result.pages), str(result.empty_pages))
    assert len(ex["calls"]) == result.pages + (2 if name == "retry_then_ok" else 0)  # retries
    assert [b.timestamp for b in result.bars] == sorted(b.timestamp for b in result.bars)


async def test_pages_and_empty_pages_are_counted() -> None:
    got = (await trace(cmd, "windowed_gaps", ex_method=True))["got"]
    assert (got.pages, got.empty_pages) == (4, 1) and len(got.bars) == 600
    empty = (await trace(cmd, "all_empty", ex_method=True))["got"]
    assert (empty.bars, empty.pages, empty.empty_pages, empty.complete) == ([], 3, 3, True)


def test_range_fetch_is_a_frozen_value_with_the_documented_fields() -> None:
    got = cmd.RangeFetch([], True, "stop", 0, 0)
    assert list(cmd.RangeFetch.__dataclass_fields__) == [
        "bars",
        "complete",
        "end_reason",
        "pages",
        "empty_pages",
    ]
    with pytest.raises(AttributeError):
        got.complete = False  # type: ignore[misc]
    assert set(cmd.EndReason.__args__) == {"stop", "reached_end", "page_cap", "no_progress"}  # type: ignore[attr-defined]


def test_fetch_ohlcv_range_keeps_its_signature() -> None:
    sig = inspect.signature(cmd.CCXTMarketDataService.fetch_ohlcv_range)
    assert list(sig.parameters) == ["self", "symbol", "timeframe", "start", "end"]
    assert sig.return_annotation == "list[OHLCVBar]"
    assert inspect.iscoroutinefunction(cmd.CCXTMarketDataService.fetch_ohlcv_range)


async def test_check_series_runs_the_same_validation_without_a_request() -> None:
    svc = cmd.CCXTMarketDataService("binance", cache_ttl_seconds=0)
    ex = _Ex({})
    svc._exchange = ex
    await svc.connect()
    svc.check_series(SYMBOL, TimeFrame.ONE_HOUR)
    for bad in (("ETH/USDT", TimeFrame.ONE_HOUR), (SYMBOL, TimeFrame.ONE_DAY)):
        with pytest.raises(cmd.DataNotAvailableError):
            svc.check_series(*bad)
    ex.fetch_ohlcv.assert_not_called()


# ---------------------------------------------------------------------------
# P6b (CF-26-63): a cancelled or timed-out delegate releases the semaphore and the inner call
# ---------------------------------------------------------------------------
asyncio = cmd.asyncio  # the module under test's asyncio (the header imports stay untouched)


class _BlockEx(_Ex):
    """Serves every page, but call number ``block_at`` never returns (until it is cancelled)."""

    def __init__(self, block_at: int) -> None:
        super().__init__({"slots": _ALL})
        self.block_at, self.started, self.cancelled = block_at, asyncio.Event(), False

    async def _fetch(self, symbol: str, timeframe: str, since: int, limit: int) -> Any:
        if self.n == self.block_at:
            self.n += 1
            self.started.set()
            try:
                await asyncio.get_running_loop().create_future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super()._fetch(symbol, timeframe, since, limit)


@pytest.mark.parametrize(
    ("block_at", "timeout"), [(2, None), (1, 0.05)], ids=["cancel_p3", "wait_for_p2"]
)
async def test_the_delegate_releases_the_semaphore_when_cancelled_or_timed_out(
    block_at: int, timeout: float | None
) -> None:
    ex, clock = _BlockEx(block_at), [1000.0]

    async def sleep(seconds: float) -> None:
        clock[0] += seconds

    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: NOW / 1000)
    with patch.object(cmd, "_now_ms", lambda: NOW), patch.object(cmd, "time", fake_time):
        with patch.object(cmd.asyncio, "sleep", sleep):
            svc = cmd.CCXTMarketDataService("binance", cache_ttl_seconds=0)
            svc._exchange = ex
            await svc.connect()
            free = svc._semaphore._value
            start, end = (datetime.fromtimestamp((T0 + i * H) / 1000, UTC) for i in (0, 900))
            call = svc.fetch_ohlcv_range(SYMBOL, TimeFrame.ONE_HOUR, start, end)  # the delegate
            if timeout is not None:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(call, timeout)
            else:
                task = asyncio.ensure_future(call)
                await ex.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
    assert ex.cancelled and ex.n == block_at + 1  # the blocked request was cancelled, no retry
    assert svc._semaphore._value == free and not svc._semaphore._waiters
