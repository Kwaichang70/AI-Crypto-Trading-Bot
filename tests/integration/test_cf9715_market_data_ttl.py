"""CF-97-15: paper and live runs build their market-data service with ``cache_ttl_seconds=0``.

With a 60 s L1 TTL and a 60 s poll on 1m, the cache entry (expiry = fetch end + 60 s) is still valid
at the next poll whenever the fetch round trip exceeds the wake-up overhead, so every other closed
bar is served stale and dropped by the engine's dedup.  T-9715-01: the real service built with the
TTL the orchestrator passes (captured in memory, never the credentials), a fake exchange and a fake
clock.  T-9715-02: the real paper and live stacks over ``FakeCCXTExchange`` on 1m.  Each has a
TTL-60 control that proves the test can see the defect.  No timing assertion anywhere.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from structlog.testing import capture_logs

import data.services.ccxt_market_data as cmd
from common.types import TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    EXCHANGE_ATTR,
    build_live_stack,
    build_paper_stack,
    patch_exchange_factory,
    start_and_warmup,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.integration.test_wp15b_resume_seed_wiring import _session_factory
from tests.integration.test_wp16_cb_api_orchestrator import (  # noqa: F401
    _CapturingEngine,
    _quiet_side_tasks,
    _settings,
)
from trading.costs import DEFAULT_COST_PROFILE

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_RTT = 0.2  # a fetch round trip, longer than the wake-up overhead (0 here)
_SYMBOL = "BTC/EUR"
_ONE_MINUTE_POLL = 60.0


class _Recorder:
    """Stands in for ``CCXTMarketDataService`` in the orchestrator: keeps the TTL, no secret."""

    seen: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        _Recorder.seen.append((kwargs["exchange_id"], kwargs.get("cache_ttl_seconds", 60)))


async def _orchestrator_ttl(mode: str) -> Any:
    """The ``cache_ttl_seconds`` that ``run_<mode>_engine`` passes (the default 60 if omitted)."""
    from api.services import run_orchestrator as orch

    _Recorder.seen.clear()
    _CapturingEngine.instances.clear()
    common: dict[str, Any] = {
        "run_id_str": "cf9715-run", "strategy_name": "grid_trading", "strategy_params": {},
        "strategy_cls": MagicMock(**{"mtf_signal_fields.return_value": frozenset()}),
        "symbols": [_SYMBOL], "timeframe": TimeFrame.ONE_MINUTE, "initial_capital": "10000",
        "cost_profile": DEFAULT_COST_PROFILE, "resume": None,
    }  # fmt: skip
    patches = [
        patch("api.db.session.get_session_factory", _session_factory()),
        patch("api.config.get_settings", return_value=_settings()),
        patch("data.services.ccxt_market_data.CCXTMarketDataService", _Recorder),
        patch("trading.portfolio.PortfolioAccounting", MagicMock()),
        patch("trading.strategy_engine.StrategyEngine", _CapturingEngine),
        patch.object(orch, "make_circuit_breaker_alert_listener", MagicMock()),
        patch.object(orch, "record_audit_event", MagicMock()),
    ]
    if mode == "paper":
        patches += [patch("trading.engines.paper.PaperExecutionEngine", MagicMock()),
                    patch.object(orch, "_auto_retry_paper_run", MagicMock())]  # fmt: skip
    else:
        patches += [patch.object(orch, "build_live_ccxt_exchange", MagicMock()),
                    patch("trading.engines.live.LiveExecutionEngine", MagicMock())]  # fmt: skip
        common["protective_mode"] = False
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with pytest.raises(asyncio.CancelledError):
            await getattr(orch, f"run_{mode}_engine")(**common)
    assert len(_Recorder.seen) == 1, "the run builds exactly one market-data service"
    return _Recorder.seen[0][1]


class _Clock:
    t = 0.0  # seconds; monotonic and wall time are the same fake


def _fake_time(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    monkeypatch.setattr(
        cmd, "time", SimpleNamespace(monotonic=lambda: clock.t, time=lambda: clock.t)
    )
    monkeypatch.setattr(cmd, "_now_ms", lambda: int(clock.t * 1000))


async def _skipped_bars(monkeypatch: pytest.MonkeyPatch, ttl: Any, polls: int = 60) -> int:
    clock = _Clock()
    _fake_time(monkeypatch, clock)

    class _Exchange:
        timeframes = {"1m": "1m"}  # noqa: RUF012

        async def load_markets(self) -> dict[str, object]:
            return {_SYMBOL: {}}

        async def fetch_ohlcv(
            self, symbol: str, timeframe: str, since: Any, limit: int
        ) -> list[Any]:
            clock.t += _RTT
            slot = int(clock.t // 60)  # the forming candle; the closed ones come before it
            return [[int((_T0 + timedelta(seconds=60 * s)).timestamp() * 1000), 1, 1, 1, 1, 1]
                    for s in range(slot - limit + 1, slot + 1)]  # fmt: skip

        async def close(self) -> None: ...

    service = CCXTMarketDataService("binance", cache_ttl_seconds=ttl)
    service._exchange = _Exchange()
    await service.connect()
    clock.t, seen = 10 * 60 + 0.5, []
    for _ in range(
        polls
    ):  # the engine's schedule: the next poll starts 60 s after this one started
        start = clock.t
        bar = await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_MINUTE)
        if not seen or bar.timestamp > seen[-1]:
            seen.append(bar.timestamp)
        clock.t = start + max(_ONE_MINUTE_POLL, clock.t - start)
    return int((seen[-1] - seen[0]).total_seconds() // 60) + 1 - len(seen)


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_orchestrator_ttl_skips_no_closed_bar_on_1m(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ttl = await _orchestrator_ttl(mode)
    assert ttl == 0
    assert await _skipped_bars(monkeypatch, ttl) == 0
    assert (
        await _skipped_bars(monkeypatch, 60) > 0
    )  # the control: the old TTL skips about every other bar


# --- T-9715-02: the real stacks over the fake exchange on 1m --------------------------------------


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant)


async def _bar_counts(
    mode: str, ttl: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[int], list[str]]:
    exchange = FakeCCXTExchange()
    exchange.register_market(_SYMBOL, base="BTC", quote="EUR")
    exchange.set_balance("EUR", Decimal(100000))
    exchange.seed_flat_bars(_SYMBOL, count=100, price=Decimal(100), timeframe="1m")
    patch_exchange_factory(monkeypatch, exchange)
    clock = _Clock()
    _fake_time(monkeypatch, clock)
    original = exchange.fetch_ohlcv

    async def with_rtt(*args: Any, **kwargs: Any) -> list[list[Any]]:
        clock.t += _RTT
        return await original(*args, **kwargs)

    monkeypatch.setattr(exchange, "fetch_ohlcv", with_rtt)
    build = build_paper_stack if mode == "paper" else build_live_stack
    stack = await build(
        exchange=exchange, symbol=_SYMBOL,
        strategy=ScriptedSequenceStrategy("cf9715", {"script": {}}),
        timeframe=TimeFrame.ONE_MINUTE, initial_capital=Decimal(1000), run_id="cf9715",
    )  # fmt: skip
    stack.engine._market_data = CCXTMarketDataService(
        exchange_id=EXCHANGE_ATTR, cache_ttl_seconds=ttl
    )
    await start_and_warmup(stack, "cf9715")
    counts: list[int] = []
    with capture_logs() as logs:
        for k in range(1, 5):
            exchange.push_bar(_SYMBOL, Decimal(100 + k), timeframe="1m")
            start = clock.t
            await stack.engine._poll_and_process()
            clock.t = start + max(stack.engine._poll_interval, clock.t - start)
            counts.append(stack.engine.bar_count)
    return counts, [e["event"] for e in logs if e["event"] == "engine.poll_fetch_failed"]


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_every_pushed_1m_bar_is_evaluated_once_in_order(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _bar_counts(mode, await _orchestrator_ttl(mode), monkeypatch) == ([1, 2, 3, 4], [])


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_the_old_ttl_lags_on_the_same_stack(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts, _ = await _bar_counts(
        mode, 60, monkeypatch
    )  # the control: a stale poll evaluates no bar
    assert counts != [1, 2, 3, 4] and counts[1] == 1
