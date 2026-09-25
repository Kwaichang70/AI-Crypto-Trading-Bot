"""
tests/integration/test_live_protective_paths.py
---------------------------------------------------
WP1.0 (Verbeterplan v2, Documentation/Verbeterplan-v2-2026-09.md §4 Fase 1,
§3 "Herstart-protocol") -- live protective-path regression gate.

Drives the *real* ``StrategyEngine(run_mode=LIVE)`` + the *real*
``LiveExecutionEngine`` (``packages/trading/engines/live.py``) + the *real*
``PortfolioAccounting`` + the *real* ``DefaultRiskManager``, with only the
CCXT exchange client replaced by an in-memory
``FakeCCXTExchange`` (``tests/integration/fakes/fake_ccxt_exchange.py``).

This file is the regression gate the herstart-protocol (§3) and the Fase 1
minimum-set (§4) require before any live run may restart. It proves finding
**C1** ("live-engine kan geen SELL uitvoeren" -- ``LiveExecutionEngine``'s
``_positions`` dict is only ever written by ``sync_positions()``, which no
production code path calls) *without* ever writing to
``execution._positions`` from the test itself -- doing that would hide the
bug rather than prove it.

Scenario table (see the producer report,
``reports/vp2-wp1.0/producer-report.md``, for today's pass/xfail result and
the captured failure reason for every row):

    buy_only                        MUST PASS today (proves harness validity)
    sell_without_position_rejected  MUST PASS today and after every fix
    buy_then_stop_loss              xfail today -- C1 (WP1.1)
    buy_then_take_profit            xfail today -- C1 (WP1.1)
    buy_then_trailing_stop          xfail today -- C1 (WP1.1)
    buy_then_kill_switch            xfail today -- C6 (WP1.2), also needs C1/WP1.1
    buy_restart_reconcile_stop_loss xfail today -- C7 (WP1.8), also needs C1/WP1.1
    external_holdings_not_sold      xfail today -- C1 (WP1.1); C22 unreachable until C1 lands

Runtime target: < 15s for the whole file (no real sleeps, no real network
I/O -- see the ``_fast_sleep`` fixture and ``FakeCCXTExchange``).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from common.types import TimeFrame
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSignalStrategy

# ---------------------------------------------------------------------------
# Fixed scenario constants -- deterministic, no randomness, no wall clock.
# ---------------------------------------------------------------------------

SYMBOL = "BTC/EUR"
BASE = "BTC"
QUOTE = "EUR"
TIMEFRAME = TimeFrame.ONE_HOUR
TIMEFRAME_STR = "1h"
START_PRICE = Decimal("50000")
INITIAL_CAPITAL = Decimal("1000")
TARGET_NOTIONAL = Decimal("100")  # 10% of equity -- comfortably under every
# default RiskParameters cap (15% concentration, 60% exposure, 40% cluster),
# so the BUY fills at exactly target_notional / price with no risk-driven
# quantity reduction (verified analytically in the producer report).

_QTY_TOLERANCE = Decimal("0.00000001")


def _buy_params(call_index: int = 0) -> dict[str, object]:
    """Shorthand for a ScriptedSignalStrategy BUY-params dict."""
    return {
        "direction": "buy",
        "call_index": call_index,
        "target_notional": str(TARGET_NOTIONAL),
    }


def _approx_eq(a: Decimal, b: Decimal, tol: Decimal = _QTY_TOLERANCE) -> bool:
    """Compare two Decimals allowing for CCXT's float-JSON round-trip.

    Every fill quantity that flows through ``FakeCCXTExchange.fetch_my_trades``
    is deliberately round-tripped through ``float`` (mirroring the real
    CCXT wire format: real exchanges return JSON floats, not Decimals), so
    comparisons against a value derived from that path use a tolerance
    instead of ``==``. Values read directly from ``FakeCCXTExchange``'s own
    internal ledger (``balance_of``, ``order_log``) never go through that
    round-trip and are compared with exact equality instead.
    """
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every real sleep in the live path so the file runs in <15s.

    Patches the shared ``asyncio`` module's ``sleep`` attribute, which is
    what every affected call site actually looks up at call time (all of
    them do ``import asyncio`` then ``asyncio.sleep(...)``, never
    ``from asyncio import sleep``):

    - ``LiveExecutionEngine.process_signal``'s post-submit
      ``await asyncio.sleep(2)`` (packages/trading/engines/live.py).
    - ``CCXTMarketDataService``'s internal throttle / backoff sleeps
      (packages/data/services/ccxt_market_data.py) -- not expected to fire
      against ``FakeCCXTExchange`` (it never raises the retryable CCXT
      errors), but neutralised defensively.
    """

    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    """A FakeCCXTExchange with the default market set and quote balance seeded.

    Registers BTC/EUR, BTC/USD and ETH/EUR (requirement: "markets with
    multiple quotes per base") even though only BTC/EUR is traded in this
    file -- WP1.1's ``sync_positions`` base->market mapping fix needs a
    fixture that already exercises the ambiguous-market case.
    """
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE)
    ex.register_market("BTC/USD", base="BTC", quote="USD")
    ex.register_market("ETH/EUR", base="ETH", quote="EUR")
    ex.set_balance(QUOTE, INITIAL_CAPITAL)
    ex.seed_flat_bars(SYMBOL, count=100, price=START_PRICE, timeframe=TIMEFRAME_STR)
    patch_exchange_factory(monkeypatch, ex)
    return ex


async def _build_and_warm(
    exchange: FakeCCXTExchange,
    strategy: ScriptedSignalStrategy,
    *,
    engine_config: dict[str, object] | None = None,
    run_id: str = "wp10-test-run",
) -> LiveStack:
    stack = await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL_CAPITAL,
        run_id=run_id,
        engine_config=engine_config,
    )
    await start_and_warmup(stack, run_id)
    return stack


# ---------------------------------------------------------------------------
# buy_only -- MUST PASS today: proves the harness itself is valid before any
# xfail scenario below is trusted to be failing for the right reason.
# ---------------------------------------------------------------------------


async def test_buy_only_fills_and_shows_position(exchange: FakeCCXTExchange) -> None:
    strategy = ScriptedSignalStrategy("buy-only", _buy_params())
    stack = await _build_and_warm(exchange, strategy)

    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1, f"expected exactly one BUY order; got {exchange.order_log!r}"
    bought_qty = buy_orders[0]["amount"]
    assert bought_qty == TARGET_NOTIONAL / START_PRICE

    # Fake-exchange balances reflect the fill (quote debited, base credited).
    fee = bought_qty * START_PRICE * exchange.taker_fee_pct
    assert exchange.balance_of(BASE) == bought_qty
    assert exchange.balance_of(QUOTE) == INITIAL_CAPITAL - bought_qty * START_PRICE - fee

    # PortfolioAccounting -- the real source of truth the bracket/trailing
    # managers read from -- shows the same position.
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat
    assert _approx_eq(position.quantity, bought_qty)
    assert stack.portfolio.get_summary()["open_positions"] == 1

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# sell_without_position_rejected -- MUST PASS today AND after every fix in
# this plan: a SELL with genuinely no position must always be rejected.
# ---------------------------------------------------------------------------


async def test_sell_without_position_rejected(exchange: FakeCCXTExchange) -> None:
    strategy = ScriptedSignalStrategy("sell-no-position", {"direction": "sell", "call_index": 0})
    stack = await _build_and_warm(exchange, strategy)

    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)

    reject_events = [e for e in cap if e.get("event") == "live.sell_no_position"]
    assert reject_events, f"expected live.sell_no_position; got {cap!r}"

    assert exchange.order_log == []
    assert stack.execution.positions.get(SYMBOL) is None

    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_then_stop_loss / buy_then_take_profit -- xfail today: C1 (WP1.1).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C1 (Verbeterplan v2 WP1.1): LiveExecutionEngine._positions is only "
        "ever written by sync_positions(), which no production call site "
        "invokes -- BUY fills never populate it. The bracket manager "
        "correctly reads the open position from PortfolioAccounting and "
        "emits the exit SELL signal, but LiveExecutionEngine.process_signal "
        "rejects it one call later with 'live.sell_no_position' because its "
        "own _positions dict is empty. Fixed by WP1.1."
    ),
)
@pytest.mark.parametrize(
    ("trigger", "breach_multiplier", "config_key"),
    [
        pytest.param(
            "stop_loss", Decimal("0.80"), "bracket_stop_loss_pct", id="buy_then_stop_loss"
        ),
        pytest.param(
            "take_profit", Decimal("1.20"), "bracket_take_profit_pct", id="buy_then_take_profit"
        ),
    ],
)
async def test_buy_then_bracket_exit_blocked_by_c1(
    exchange: FakeCCXTExchange,
    trigger: str,
    breach_multiplier: Decimal,
    config_key: str,
) -> None:
    strategy = ScriptedSignalStrategy(f"buy-{trigger}", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={config_key: 0.05})

    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    position_after_buy = stack.portfolio.get_position(SYMBOL)
    assert position_after_buy is not None and not position_after_buy.is_flat

    breach_price = START_PRICE * breach_multiplier
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    reject_events = [e for e in cap if e.get("event") == "live.sell_no_position"]
    assert reject_events, (
        f"expected the {trigger} exit to be rejected with "
        f"'live.sell_no_position' (C1); captured log events: {cap!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, (
        f"expected a {trigger} SELL order to reach the exchange once C1 "
        "(WP1.1) is fixed; got none -- the bracket manager emitted the "
        "exit signal (see the captured live.sell_no_position event above) "
        "but LiveExecutionEngine rejected it before submission."
    )

    position_after_breach = stack.portfolio.get_position(SYMBOL)
    assert position_after_breach is not None and position_after_breach.is_flat, (
        f"the {trigger} exit should have fully closed the position; "
        "instead the bot is left holding it with no working exit (C1)"
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_then_trailing_stop -- xfail today: C1 (WP1.1).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C1 (Verbeterplan v2 WP1.1): same root cause as "
        "test_buy_then_bracket_exit_blocked_by_c1, exercised via "
        "TrailingStopManager instead of BracketExitManager. Fixed by WP1.1."
    ),
)
async def test_buy_then_trailing_stop_blocked_by_c1(exchange: FakeCCXTExchange) -> None:
    strategy = ScriptedSignalStrategy("buy-trailing", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={"trailing_stop_pct": 0.05})

    # bar0: BUY, peak seeds at 50000. bar1: new peak (55000), no trigger yet.
    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    await step_bar(stack, SYMBOL, Decimal("55000"), timeframe=TIMEFRAME_STR)
    position_after_peak = stack.portfolio.get_position(SYMBOL)
    assert position_after_peak is not None and not position_after_peak.is_flat

    # 55000 * (1 - 0.05) = 52250 -- 52000 breaches it.
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, Decimal("52000"), timeframe=TIMEFRAME_STR)

    trigger_events = [e for e in cap if e.get("event") == "trailing_stop.triggered"]
    assert trigger_events, f"expected trailing_stop.triggered; got {cap!r}"

    reject_events = [e for e in cap if e.get("event") == "live.sell_no_position"]
    assert reject_events, (
        f"expected the trailing-stop exit to be rejected with "
        f"'live.sell_no_position' (C1); captured log events: {cap!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, (
        "expected the trailing-stop exit to reach the exchange once C1 "
        "(WP1.1) is fixed; got none -- TrailingStopManager correctly read "
        "the open position from PortfolioAccounting and emitted the exit "
        "signal, but LiveExecutionEngine rejected it before submission."
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_then_kill_switch -- xfail today: C6 (WP1.2). Also needs C1 (WP1.1).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C6 (Verbeterplan v2 WP1.2): in LIVE mode, StrategyEngine._process_bar "
        "returns immediately when the kill switch is active (before the "
        "bracket/trailing sections run at all), so a protective stop-loss "
        "is never even evaluated while the switch is on -- target semantics "
        "per plan decision D3 is 'block new entries, protective exits keep "
        "running'. Note this scenario ALSO requires C1 (WP1.1) to be fixed "
        "before it can pass end-to-end: once the early return is removed, "
        "the resulting SELL would still hit the live.sell_no_position "
        "rejection today."
    ),
)
async def test_buy_then_kill_switch_blocks_protective_exit(exchange: FakeCCXTExchange) -> None:
    strategy = ScriptedSignalStrategy("buy-kill-switch", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={"bracket_stop_loss_pct": 0.05})

    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    position_after_buy = stack.portfolio.get_position(SYMBOL)
    assert position_after_buy is not None and not position_after_buy.is_flat

    stack.risk_manager.trigger_kill_switch("wp10-harness-test")

    breach_price = START_PRICE * Decimal("0.80")
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    skip_events = [e for e in cap if e.get("event") == "engine.bar_skipped_kill_switch"]
    assert skip_events, (
        "expected _process_bar to log engine.bar_skipped_kill_switch and "
        f"return before evaluating the bracket exit (C6); got: {cap!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, (
        "expected the protective stop-loss to still exit while the kill "
        "switch blocks new entries (D3 / WP1.2); today the bracket/trailing "
        "sections are never reached at all while the kill switch is active "
        "(C6), so the stop-loss is never evaluated -- let alone rejected by "
        "C1."
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_restart_reconcile_stop_loss -- xfail today: C7 (WP1.8).
# Also needs C1 (WP1.1).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C7 (Verbeterplan v2 WP1.8): orphan-recovery restarts a live run "
        "today with a brand-new PortfolioAccounting and never calls "
        "sync_positions() on boot, so a fresh engine stack has no memory "
        "of a position the bot itself opened before the restart even "
        "though the exchange still holds it. The bracket manager therefore "
        "never attempts an exit at all (position=None), independently of "
        "C1. Fixed by WP1.8 (position reconciliation on resume); also "
        "needs C1 (WP1.1) for the resulting SELL to actually reach the "
        "exchange."
    ),
)
async def test_buy_restart_reconcile_stop_loss(exchange: FakeCCXTExchange) -> None:
    strategy_before = ScriptedSignalStrategy("buy-restart-entry", _buy_params())
    stack_before = await _build_and_warm(
        exchange,
        strategy_before,
        engine_config={"bracket_stop_loss_pct": 0.05},
        run_id="wp10-restart-before",
    )

    await step_bar(stack_before, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    position_before_restart = stack_before.portfolio.get_position(SYMBOL)
    assert position_before_restart is not None and not position_before_restart.is_flat
    bought_qty = exchange.order_log[-1]["amount"]
    assert exchange.balance_of(BASE) == bought_qty  # the bot's BTC really sits on the exchange

    # --- Simulate an API restart: a brand-new engine stack against the
    # SAME exchange (same balances, same price history), exactly as
    # orphan-recovery does *today* -- a fresh PortfolioAccounting, a fresh
    # LiveExecutionEngine, no sync_positions() call (C7). stack_before is
    # deliberately never stopped: a real process restart does not get a
    # graceful shutdown either.
    strategy_after = ScriptedSignalStrategy(
        "buy-restart-noop", {"direction": "buy", "call_index": 999}
    )
    stack_after = await _build_and_warm(
        exchange,
        strategy_after,
        engine_config={"bracket_stop_loss_pct": 0.05},
        run_id="wp10-restart-after",
    )

    position_after_restart = stack_after.portfolio.get_position(SYMBOL)
    assert position_after_restart is not None and not position_after_restart.is_flat, (
        "the entry recorded before the simulated restart should still be "
        "known to the new engine stack; today it is NOT (C7): "
        "PortfolioAccounting is rebuilt empty and sync_positions() is "
        "never called on boot, so the bot has no memory of the position "
        "it actually holds on the exchange."
    )

    breach_price = START_PRICE * Decimal("0.80")
    with capture_logs() as cap:
        await step_bar(stack_after, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, (
        "expected the stop-loss to fire against the entry price recorded "
        "before the restart; today no SELL is even attempted because the "
        f"new engine's bracket manager sees position=None (C7); captured "
        f"log events: {cap!r}"
    )
    assert sell_orders[-1]["amount"] == bought_qty

    await stack_after.engine.stop()


# ---------------------------------------------------------------------------
# external_holdings_not_sold -- xfail today: C1 (WP1.1). C22 is dormant.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C1 (Verbeterplan v2 WP1.1) blocks this scenario before C22 (D15) "
        "can even be observed: no production code path calls "
        "sync_positions(), so the pre-existing 0.5 BTC external balance is "
        "not (yet) visible to LiveExecutionEngine at all -- the only "
        "reachable failure today is the same live.sell_no_position "
        "rejection as every other SELL scenario. WP1.1's plan explicitly "
        "bundles the C22 fix (bot position = own fills, bounded by "
        "min(own, exchange)) with the C1 fix, so this scenario is expected "
        "to flip green together with the bracket/trailing scenarios above, "
        "not as a separate step."
    ),
)
async def test_external_holdings_not_sold(exchange: FakeCCXTExchange) -> None:
    exchange.set_balance(BASE, Decimal("0.5"))  # pre-existing holdings the bot never bought

    strategy = ScriptedSignalStrategy("buy-external-holdings", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={"bracket_stop_loss_pct": 0.05})

    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    bot_bought_qty = exchange.order_log[-1]["amount"]
    assert exchange.balance_of(BASE) == Decimal("0.5") + bot_bought_qty

    breach_price = START_PRICE * Decimal("0.80")
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders and sell_orders[-1]["amount"] == bot_bought_qty, (
        f"expected the exit SELL to close exactly the bot's own quantity "
        f"({bot_bought_qty}), never touching the pre-existing 0.5 BTC "
        f"external holding (C22 / D15); today NO sell order reaches the "
        f"exchange at all -- C1 rejects it before C22's over-sell behaviour "
        f"could even be observed through this harness "
        f"(sell_orders={sell_orders!r}, captured logs={cap!r})"
    )

    assert exchange.balance_of(BASE) == Decimal("0.5"), (
        "external holdings must be untouched once the bot's own position is closed"
    )

    await stack.engine.stop()
