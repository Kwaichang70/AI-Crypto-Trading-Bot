"""
tests/integration/fakes/live_harness.py
------------------------------------------
Builder helpers for the WP1.0 live protective-path integration harness.

Wires together the *real* production classes -- ``StrategyEngine``,
``LiveExecutionEngine``, ``PortfolioAccounting``, ``DefaultRiskManager``,
``CCXTMarketDataService`` -- exactly as
``apps/api/services/run_orchestrator.py::run_live_engine`` does, with only
the CCXT exchange class replaced by ``FakeCCXTExchange``
(``tests/integration/fakes/fake_ccxt_exchange.py``).

Deviations from ``run_live_engine`` (documented; see the producer report
for the full rationale of each):

1. No database session / ``RunORM`` writes, no ``_RUN_TASKS`` /
   ``_RUN_ENGINES`` global-registry mutation -- this harness never touches
   Postgres or module-global state.
2. No incremental-flush task, no ``_auto_stop_after`` task, no adaptive
   learning task -- irrelevant to protective-exit correctness and would
   otherwise require a live event loop running forever.
3. ``CCXTMarketDataService(cache_ttl_seconds=0)`` instead of production's
   ``60`` -- the 60s L1 cache is keyed by ``(symbol, timeframe, since,
   limit)`` and would return a stale bar across every bar-by-bar test step
   (all of which happen within milliseconds of wall-clock time). Disabling
   the cache is a pure performance optimisation with zero behavioural
   effect on the code paths this WP verifies.
4. The harness drives bars by calling ``StrategyEngine._poll_and_process()``
   and ``StrategyEngine._warmup_bar_windows()`` directly instead of
   ``run_live_loop()`` -- ``run_live_loop()`` sleeps ``poll_interval_seconds``
   between iterations and loops until a stop event, neither of which is
   compatible with a deterministic, sub-15s test file. Both private methods
   are the *exact* per-bar logic ``run_live_loop()`` calls each iteration;
   nothing about signal/order/fill processing is bypassed or reimplemented.
5. ``asyncio.sleep`` is monkeypatched to return immediately for the whole
   test module (see ``test_live_protective_paths.py``'s ``_fast_sleep``
   autouse fixture), neutralising ``LiveExecutionEngine.process_signal``'s
   real ``await asyncio.sleep(2)`` post-submit wait.
6. Both ``LiveExecutionEngine``'s exchange handle and
   ``CCXTMarketDataService``'s internally-constructed exchange handle
   resolve to the *same* ``FakeCCXTExchange`` instance via a monkeypatched
   ``ccxt.async_support.coinbase`` factory (see ``patch_exchange_factory``)
   -- in production these are two independently-constructed instances of
   the same real CCXT class hitting the same real exchange; sharing one
   fake instance is the in-memory equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt_async
import pytest

from common.types import RunMode, TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from trading.engines.live import LiveExecutionEngine
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

__all__ = [
    "EXCHANGE_ATTR",
    "LiveStack",
    "build_live_stack",
    "patch_exchange_factory",
    "start_and_warmup",
    "step_bar",
]

#: Attribute name patched on ``ccxt.async_support`` -- must be a real CCXT
#: exchange id so ``CCXTMarketDataService.__init__``'s
#: ``getattr(ccxt_async, exchange_id, None)`` lookup succeeds before the
#: monkeypatch replaces the class with our factory.
EXCHANGE_ATTR = "coinbase"


@dataclass
class LiveStack:
    """The full set of real engine-stack components built for one scenario."""

    engine: StrategyEngine
    execution: LiveExecutionEngine
    portfolio: PortfolioAccounting
    risk_manager: DefaultRiskManager
    market_data: CCXTMarketDataService
    exchange: FakeCCXTExchange


def patch_exchange_factory(monkeypatch: pytest.MonkeyPatch, exchange: FakeCCXTExchange) -> None:
    """Make every ``ccxt.async_support.coinbase(config)`` call return ``exchange``.

    Mirrors both construction call-sites in production
    (``run_orchestrator.run_live_engine``'s ``exchange_cls(exchange_config)``
    and ``CCXTMarketDataService.__init__``'s ``exchange_cls(ccxt_config)``)
    without needing a real network-capable CCXT class.
    """

    def _factory(config: dict[str, Any] | None = None) -> FakeCCXTExchange:
        return exchange

    monkeypatch.setattr(ccxt_async, EXCHANGE_ATTR, _factory)


async def build_live_stack(
    *,
    exchange: FakeCCXTExchange,
    strategy: BaseStrategy,
    symbol: str,
    timeframe: TimeFrame,
    initial_capital: Decimal,
    run_id: str,
    engine_config: dict[str, object] | None = None,
    risk_params: RiskParameters | None = None,
) -> LiveStack:
    """Construct one real StrategyEngine(LIVE) stack against ``exchange``.

    Mirrors ``run_orchestrator.run_live_engine`` lines ~1111-1185. Call
    ``patch_exchange_factory`` first so the two internal
    ``ccxt_async.coinbase(...)`` construction calls below resolve to the
    shared fake instance.
    """
    risk_manager = DefaultRiskManager(run_id=run_id, params=risk_params)

    # Mirrors run_orchestrator.py:1111-1123.
    live_exchange_handle = getattr(ccxt_async, EXCHANGE_ATTR)({"enableRateLimit": True})

    execution = LiveExecutionEngine(
        run_id=run_id,
        risk_manager=risk_manager,
        exchange=live_exchange_handle,
        enable_live_trading=True,
    )

    # Mirrors run_orchestrator.py:1126-1132, minus credentials (the fake
    # needs none) and with cache_ttl_seconds=0 (deviation #3 above).
    market_data = CCXTMarketDataService(exchange_id=EXCHANGE_ATTR, cache_ttl_seconds=0)

    portfolio = PortfolioAccounting(run_id=run_id, initial_cash=initial_capital)

    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=[symbol],
        timeframe=timeframe,
        run_mode=RunMode.LIVE,
        config=engine_config,
        # circuit_breaker intentionally omitted: production never
        # instantiates one either (finding C2) -- passing None here is
        # therefore high-fidelity, not a simplification.
    )

    return LiveStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
    )


async def start_and_warmup(stack: LiveStack, run_id: str) -> None:
    """Start the engine and load its initial bar window.

    Calls the same two steps ``run_live_loop()`` performs before entering
    its polling ``while`` loop (``engine.start()`` then
    ``engine._warmup_bar_windows()``) -- see deviation #4 in the module
    docstring for why we call the second one directly instead of entering
    the full sleeping loop.
    """
    await stack.engine.start(run_id)
    await stack.engine._warmup_bar_windows()


async def step_bar(
    stack: LiveStack,
    symbol: str,
    close: Decimal,
    *,
    timeframe: str = "1h",
) -> None:
    """Advance the fake exchange's price by one bar and process it.

    Calls ``StrategyEngine._poll_and_process()`` directly -- the exact
    per-iteration body of ``run_live_loop()`` minus the sleep -- so every
    scenario step exercises the real fetch -> dedupe -> ``_process_bar()``
    pipeline (signals, risk gate, order submission, fills, bracket/trailing
    checks) with no shortcuts.
    """
    stack.exchange.push_bar(symbol, close, timeframe=timeframe)
    await stack.engine._poll_and_process()
