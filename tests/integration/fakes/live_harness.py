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

WP1.6 addition (K-15)
---------------------
- ``circuit_breaker: bool = True`` on the three builders: the stack gets the SAME
  breaker production builds (``build_run_circuit_breaker(run_id, risk_manager.params)``
  -- live 20% / paper 30% drawdown, 8% daily loss), without the Telegram listener.
  The pre-WP1.6 comment "production never instantiates one (finding C2)" no longer
  holds: ``run_paper_engine`` / ``run_live_engine`` now wire it.  Passing
  ``circuit_breaker=False`` is an explicit opt-out and REQUIRES a one-line written
  reason in the calling test's docstring.  ``stack.circuit_breaker`` exposes it.

WP1.3b-b additions (live intra-bar exit check)
----------------------------------------------
- ``IntrabarClock`` / ``install_intrabar_clock(stack)``: a VIRTUAL clock for the
  engine's intra-bar task.  It replaces ``engine._intrabar_wait`` (the only place
  the task sleeps, stop-interruptible in production) and ``engine._intrabar_clock``,
  so a 300 s interval and a 10 s confirm delay cost no wall time.  Exchange
  mutations (``set_ticker``, ``push_bar``) can be scheduled at virtual times with
  ``clock.at(t, fn)``; ``clock.create_order_times`` records the virtual time of
  every ``create_order``.
- ``run_intrabar_tick(stack)``: one ``engine._intrabar_exit_tick()`` call.
- ``run_intrabar_loop(stack)``: run the REAL ``_intrabar_exit_loop``
  task body until the virtual clock reaches its ``horizon``.

WP2.1 addition
--------------
- ``cost_profile: CostProfile | None = None`` on the three builders: forwarded to
  the execution engine (paper: fee / slippage model; live: log-only drift
  estimate) and to ``StrategyEngine`` (exit-cost floor), exactly as
  ``run_orchestrator.run_paper_engine`` / ``run_live_engine`` now do.  ``None``
  keeps every pre-WP2.1 caller on the engine defaults (paper/strategy: the
  platform default profile; live: no drift checks).

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

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt_async
import pytest

from common.types import RunMode, TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from trading.costs import CostProfile, default_cost_profile
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine
from trading.models import Fill, Order
from trading.portfolio import PortfolioAccounting
from trading.recovery import ResumeDailySeed
from trading.risk import RiskParameters, live_risk_parameters
from trading.risk_manager import DefaultRiskManager
from trading.safety import CircuitBreaker, build_run_circuit_breaker
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

__all__ = [
    "EXCHANGE_ATTR",
    "IntrabarClock",
    "LiveStack",
    "ManualClock",
    "PaperStack",
    "build_live_stack",
    "build_paper_stack",
    "build_resumed_live_stack",
    "install_intrabar_clock",
    "patch_exchange_factory",
    "run_intrabar_loop",
    "run_intrabar_tick",
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
    circuit_breaker: CircuitBreaker | None = None


@dataclass
class PaperStack:
    """The paper-mode twin of :class:`LiveStack` (WP1.5a)."""

    engine: StrategyEngine
    execution: PaperExecutionEngine
    portfolio: PortfolioAccounting
    risk_manager: DefaultRiskManager
    market_data: CCXTMarketDataService
    exchange: FakeCCXTExchange
    circuit_breaker: CircuitBreaker | None = None


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
    circuit_breaker: bool = True,
    symbols: Sequence[str] | None = None,
    cost_profile: CostProfile | None = None,
) -> LiveStack:
    """Construct one real StrategyEngine(LIVE) stack against ``exchange``.

    WP1.3b-b: ``symbols`` (default ``[symbol]``) lets a test run several symbols in
    one engine (the intra-bar tick must keep checking the others).

    Mirrors ``run_orchestrator.run_live_engine`` lines ~1111-1185. Call
    ``patch_exchange_factory`` first so the two internal
    ``ccxt_async.coinbase(...)`` construction calls below resolve to the
    shared fake instance.

    WP1.5a: ``risk_params=None`` means ``live_risk_parameters()`` (R-02 --
    what production's live site passes), not ``RiskParameters()``; ``clock``
    is the portfolio's risk-day clock (wall clock UTC when ``None``).

    WP1.6 (K-15): ``circuit_breaker=True`` (default) builds the production
    breaker from the run's own risk parameters; ``False`` is an opt-out that
    needs a written reason in the calling test.
    """
    risk_manager = DefaultRiskManager(
        run_id=run_id, params=risk_params if risk_params is not None else live_risk_parameters()
    )
    cb = build_run_circuit_breaker(run_id, risk_manager.params) if circuit_breaker else None

    # Mirrors run_orchestrator.py:1111-1123.
    live_exchange_handle = getattr(ccxt_async, EXCHANGE_ATTR)({"enableRateLimit": True})

    execution = LiveExecutionEngine(
        run_id=run_id,
        risk_manager=risk_manager,
        exchange=live_exchange_handle,
        enable_live_trading=True,
        cost_profile=cost_profile,
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
        symbols=list(symbols) if symbols is not None else [symbol],
        timeframe=timeframe,
        run_mode=RunMode.LIVE,
        config=engine_config,
        # WP1.6 (C2 closed): production wires the breaker, so the harness does
        # too (same helper, same thresholds; no Telegram listener).
        circuit_breaker=cb,
        cost_profile=cost_profile,
    )

    return LiveStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
        circuit_breaker=cb,
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
    circuit_breaker: bool = True,
    cost_profile: CostProfile | None = None,
) -> PaperStack:
    """Construct one real StrategyEngine(PAPER) stack against ``exchange``.

    Mirrors ``run_orchestrator.run_paper_engine``: ``DefaultRiskManager``
    with the default (30% drawdown) parameters, ``PaperExecutionEngine``,
    ``PortfolioAccounting``, and a ``CCXTMarketDataService`` that resolves to
    the shared fake (``patch_exchange_factory``).  Same deviations as the
    live builder (no DB, no flush task, ``cache_ttl_seconds=0``).

    WP1.6 (K-15): ``circuit_breaker=True`` (default) builds the production
    paper breaker (30% drawdown); ``False`` needs a written reason.
    """
    risk_manager = DefaultRiskManager(run_id=run_id, params=risk_params)
    cb = build_run_circuit_breaker(run_id, risk_manager.params) if circuit_breaker else None
    execution = PaperExecutionEngine(
        run_id=run_id,
        risk_manager=risk_manager,
        cost_profile=cost_profile if cost_profile is not None else default_cost_profile(),
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
        circuit_breaker=cb,
        cost_profile=cost_profile,
    )
    return PaperStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
        circuit_breaker=cb,
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
    circuit_breaker: bool = True,
    orders: Sequence[Order] = (),
    cost_profile: CostProfile | None = None,
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

    WP1.6 (K-15): ``circuit_breaker=True`` (default) builds a FRESH breaker, as a
    real resume does (nothing is persisted); ``False`` needs a written reason.

    WP1.3b-a: ``orders`` are the persisted orders of the run (``ResumeSnapshot
    .orders``); like production (``run_live_engine`` / ``run_paper_engine``) the
    builder hands ``fills`` + ``orders`` to ``engine.prime_resume_exit_state``
    BEFORE ``start`` so the warmup recomputes the bracket anchor / trailing mark.
    """
    risk_manager = DefaultRiskManager(
        run_id=run_id, params=risk_params if risk_params is not None else live_risk_parameters()
    )
    cb = build_run_circuit_breaker(run_id, risk_manager.params) if circuit_breaker else None
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
        cost_profile=cost_profile,
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
        circuit_breaker=cb,
        cost_profile=cost_profile,
    )
    engine.prime_resume_exit_state(fills, orders)

    return LiveStack(
        engine=engine,
        execution=execution,
        portfolio=portfolio,
        risk_manager=risk_manager,
        market_data=market_data,
        exchange=exchange,
        circuit_breaker=cb,
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


class IntrabarClock:
    """Virtual time for the intra-bar task (WP1.3b-b).

    ``wait(seconds)`` advances ``now`` by ``seconds`` (running every action
    scheduled with :meth:`at` that falls inside the interval, in time order, with
    ``now`` set to the action's time while it runs) and returns ``True`` -- "stop"
    -- once ``now`` reaches ``horizon`` or the engine's stop event is set.
    ``waits`` records every requested wait, ``wake_times`` the virtual time after
    each one.
    """

    def __init__(self, engine: StrategyEngine, *, horizon: float = float("inf")) -> None:
        self.engine = engine
        self.now: float = 0.0
        self.horizon = horizon
        self.waits: list[float] = []
        self.wake_times: list[float] = []
        self.create_order_times: list[float] = []
        self._actions: list[tuple[float, int, Callable[[], Awaitable[None] | None]]] = []
        self._seq = 0

    def time(self) -> float:
        return self.now

    def at(self, t: float, action: Callable[[], Awaitable[None] | None]) -> None:
        """Run ``action`` when virtual time reaches ``t`` (during a ``wait``).  An
        ``async`` action is awaited -- e.g. a whole bar poll that falls INSIDE the
        confirm wait."""
        self._seq += 1
        self._actions.append((t, self._seq, action))
        self._actions.sort(key=lambda item: (item[0], item[1]))

    async def wait(self, seconds: float) -> bool:
        if self.engine._stop_event.is_set():
            return True
        self.waits.append(seconds)
        target = self.now + max(0.0, seconds)
        while self._actions and self._actions[0][0] <= target:
            t, _seq, action = self._actions.pop(0)
            self.now = max(self.now, t)
            outcome = action()
            if outcome is not None:
                await outcome
        if target >= self.horizon:
            self.now = self.horizon
            self.wake_times.append(self.now)
            return True
        self.now = target
        self.wake_times.append(self.now)
        # A REAL yield to the event loop (test modules patch ``asyncio.sleep`` to
        # return instantly, which would not yield).
        loop = asyncio.get_running_loop()
        gate: asyncio.Future[None] = loop.create_future()
        loop.call_soon(gate.set_result, None)
        await gate
        return self.engine._stop_event.is_set()

    def record_create_orders(self, exchange: FakeCCXTExchange) -> None:
        """Wrap ``exchange.create_order`` so its virtual call time is recorded."""
        original = exchange.create_order

        async def _recording(*args: Any, **kwargs: Any) -> dict[str, Any]:
            self.create_order_times.append(self.now)
            return await original(*args, **kwargs)

        exchange.create_order = _recording  # type: ignore[method-assign]


def install_intrabar_clock(
    stack: LiveStack, *, horizon: float = float("inf"), record_orders: bool = True
) -> IntrabarClock:
    """Swap the engine's intra-bar sleeper and clock for a virtual :class:`IntrabarClock`."""
    clock = IntrabarClock(stack.engine, horizon=horizon)
    stack.engine._intrabar_wait = clock.wait  # type: ignore[method-assign]
    stack.engine._intrabar_clock = clock.time
    if record_orders:
        clock.record_create_orders(stack.exchange)
    return clock


async def run_intrabar_tick(stack: LiveStack) -> None:
    """One intra-bar pass (``StrategyEngine._intrabar_exit_tick``)."""
    await stack.engine._intrabar_exit_tick()


async def run_intrabar_loop(stack: LiveStack) -> None:
    """Run the REAL intra-bar task body until the installed clock's horizon (virtual time)."""
    await stack.engine._intrabar_exit_loop()
