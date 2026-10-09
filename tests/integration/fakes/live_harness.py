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

WP1.5a additions
----------------
- ``clock``: an injectable risk-day clock (``ManualClock``) passed to
  ``PortfolioAccounting(clock=...)`` -- the paper/live risk day is the wall
  clock UTC, so a test moves it by hand across 00:00 UTC.
- ``risk_params`` defaults to ``live_risk_parameters()`` (R-02: live drawdown
  limit 20%) in the live builders, exactly as ``run_live_engine`` does; the
  paper builder defaults to ``RiskParameters()`` (30%).
- ``build_paper_stack``: the paper-mode twin of ``build_live_stack``
  (``PaperExecutionEngine`` + ``CCXTMarketDataService`` over the same fake).
- ``daily_seed``: forwarded to ``PortfolioAccounting.from_fills`` by
  ``build_resumed_live_stack`` (1.5b wiring uses it; 1.5a only passes it on).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt_async
import pytest

from common.types import RunMode, TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine
from trading.models import Fill
from trading.portfolio import PortfolioAccounting
from trading.recovery import ResumeDailySeed
from trading.risk import RiskParameters, live_risk_parameters
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

__all__ = [
    "EXCHANGE_ATTR",
    "LiveStack",
    "ManualClock",
    "PaperStack",
    "build_live_stack",
    "build_paper_stack",
    "build_resumed_live_stack",
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


@dataclass
class PaperStack:
    """The paper-mode twin of :class:`LiveStack` (WP1.5a)."""

    engine: StrategyEngine
    execution: PaperExecutionEngine
    portfolio: PortfolioAccounting
    risk_manager: DefaultRiskManager
    market_data: CCXTMarketDataService
    exchange: FakeCCXTExchange


class ManualClock:
    """A risk-day clock a test moves by hand (WP1.5a).

    Pass it as ``clock=`` to the stack builders; call ``set`` to jump to an
    absolute UTC time (e.g. across 00:00 UTC between two bars).
    """

    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def set(self, at: datetime) -> None:
        self.at = at


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
    clock: Callable[[], datetime] | None = None,
) -> LiveStack:
    """Construct one real StrategyEngine(LIVE) stack against ``exchange``.

    Mirrors ``run_orchestrator.run_live_engine`` lines ~1111-1185. Call
    ``patch_exchange_factory`` first so the two internal
    ``ccxt_async.coinbase(...)`` construction calls below resolve to the
    shared fake instance.

    WP1.5a: ``risk_params=None`` means ``live_risk_parameters()`` (R-02 --
    what production's live site passes), not ``RiskParameters()``; ``clock``
    is the portfolio's risk-day clock (wall clock UTC when ``None``).
    """
    risk_manager = DefaultRiskManager(
        run_id=run_id, params=risk_params if risk_params is not None else live_risk_parameters()
    )

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

    portfolio = PortfolioAccounting(run_id=run_id, initial_cash=initial_capital, clock=clock)

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


async def build_paper_stack(
    *,
    exchange: FakeCCXTExchange,
    strategy: BaseStrategy,
    symbol: str,
    timeframe: TimeFrame,
    initial_capital: Decimal,
    run_id: str,
    engine_config: dict[str, object] | None = None,
    risk_params: RiskParameters | None = None,
    clock: Callable[[], datetime] | None = None,
) -> PaperStack:
    """Construct one real StrategyEngine(PAPER) stack against ``exchange``.

    Mirrors ``run_orchestrator.run_paper_engine``: ``DefaultRiskManager``
    with the default (30% drawdown) parameters, ``PaperExecutionEngine``,
    ``PortfolioAccounting``, and a ``CCXTMarketDataService`` that resolves to
    the shared fake (``patch_exchange_factory``).  Same deviations as the
    live builder (no DB, no flush task, ``cache_ttl_seconds=0``).
    """
    risk_manager = DefaultRiskManager(run_id=run_id, params=risk_params)
    execution = PaperExecutionEngine(
        run_id=run_id,
        risk_manager=risk_manager,
        initial_cash=initial_capital,
    )
    market_data = CCXTMarketDataService(exchange_id=EXCHANGE_ATTR, cache_ttl_seconds=0)
    portfolio = PortfolioAccounting(run_id=run_id, initial_cash=initial_capital, clock=clock)
    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=[symbol],
        timeframe=timeframe,
        run_mode=RunMode.PAPER,
        config=engine_config,
    )
    return PaperStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
    )


async def build_resumed_live_stack(
    *,
    exchange: FakeCCXTExchange,
    strategy: BaseStrategy,
    symbol: str,
    timeframe: TimeFrame,
    initial_capital: Decimal,
    run_id: str,
    fills: Sequence[Fill],
    engine_config: dict[str, object] | None = None,
    risk_params: RiskParameters | None = None,
    protective_mode: bool = False,
    peak_equity_hint: Decimal | None = None,
    clock: Callable[[], datetime] | None = None,
    daily_seed: ResumeDailySeed | None = None,
) -> LiveStack:
    """Build a SECOND engine stack under the SAME ``run_id``, with its
    portfolio rebuilt from ``fills`` via the production
    ``PortfolioAccounting.from_fills`` (WP1.8a S6/S7) -- the in-memory
    equivalent of a live resume after an API restart.

    Mirrors ``build_live_stack`` in every other respect: a fresh
    ``LiveExecutionEngine`` against the SAME exchange instance (same
    balances, same price history -- a real process restart does not
    reset either), no order adoption (WP1.8b scope, out here by design:
    the 1.8a ``scan_and_import`` stub never runs in this harness either).

    ``StrategyEngine.__init__`` wires the rebuilt ``portfolio`` in as the
    execution engine's ``LivePositionSource`` (WP1.1), so
    ``on_start()``'s ``sync_positions()`` call inside :func:`start_and_warmup`
    naturally compares the correctly-rebuilt "own" quantity against the
    exchange balance -- a genuine mismatch flags per WP1.1 D2 (S9), exactly
    as it would in production.

    ``peak_equity_hint`` (WP1.4/H4): forwarded verbatim to
    ``PortfolioAccounting.from_fills`` -- the persisted-peak resume seed
    (WP1.8) that the live engine itself no longer overwrites (WP1.4 S-03).
    ``None`` (default) preserves every pre-WP1.4 caller's behaviour
    unchanged.

    WP1.5a: ``risk_params=None`` means ``live_risk_parameters()`` (R-02);
    ``clock`` and ``daily_seed`` are forwarded to ``from_fills``.
    """
    risk_manager = DefaultRiskManager(
        run_id=run_id, params=risk_params if risk_params is not None else live_risk_parameters()
    )
    if protective_mode:
        # WP1.8 S-10: risk-layer defence-in-depth mirror of the
        # strategy-layer protective_mode filter below -- production wires
        # this identically in run_orchestrator._run_live_engine.
        risk_manager.set_protective_mode(True)

    # Mirrors run_orchestrator.py:1111-1123 (a fresh exchange handle, same
    # FakeCCXTExchange instance behind it via patch_exchange_factory).
    live_exchange_handle = getattr(ccxt_async, EXCHANGE_ATTR)({"enableRateLimit": True})

    execution = LiveExecutionEngine(
        run_id=run_id,
        risk_manager=risk_manager,
        exchange=live_exchange_handle,
        enable_live_trading=True,
    )

    market_data = CCXTMarketDataService(exchange_id=EXCHANGE_ATTR, cache_ttl_seconds=0)

    portfolio = PortfolioAccounting.from_fills(
        run_id=run_id,
        initial_cash=initial_capital,
        fills=fills,
        peak_equity_hint=peak_equity_hint,
        clock=clock,
        daily_seed=daily_seed,
    )

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
        protective_mode=protective_mode,
    )

    return LiveStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
    )


async def start_and_warmup(stack: LiveStack | PaperStack, run_id: str) -> None:
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
    stack: LiveStack | PaperStack,
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
