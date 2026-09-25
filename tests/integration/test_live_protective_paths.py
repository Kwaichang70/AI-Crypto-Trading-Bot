"""
tests/integration/test_live_protective_paths.py
---------------------------------------------------
WP1.0/WP1.1 (Verbeterplan v2, Documentation/Verbeterplan-v2-2026-09.md §4
Fase 1, §3 "Herstart-protocol") -- live protective-path regression gate.

Drives the *real* ``StrategyEngine(run_mode=LIVE)`` + the *real*
``LiveExecutionEngine`` (``packages/trading/engines/live.py``) + the *real*
``PortfolioAccounting`` + the *real* ``DefaultRiskManager``, with only the
CCXT exchange client replaced by an in-memory
``FakeCCXTExchange`` (``tests/integration/fakes/fake_ccxt_exchange.py``).

This file is the regression gate the herstart-protocol (§3) and the Fase 1
minimum-set (§4) require before any live run may restart. WP1.0 proved
finding **C1** ("live-engine kan geen SELL uitvoeren") with 4 of these
scenarios xfailing; WP1.1 (`reports/vp2-wp1.1/synthesis-spec.md`) fixes C1
by wiring ``PortfolioAccounting`` in as the engine's
``LivePositionSource`` (D1) and flips those 4 scenarios green:

    buy_only                        PASSES (proves harness validity)
    sell_without_position_rejected  PASSES (SELL genuinely without a
                                     position must always be rejected)
    buy_then_stop_loss              PASSES (WP1.1)
    buy_then_take_profit            PASSES (WP1.1)
    buy_then_trailing_stop          PASSES (WP1.1)
    buy_then_kill_switch            xfail -- C6 (WP1.2); C1 no longer applies
    buy_restart_reconcile_stop_loss xfail -- I2/D15 by design (WP1.8)
    external_holdings_not_sold      PASSES (WP1.1, R-21)

Two new WP1.1 scenarios cover R-22 (startup sync with an external balance
never fabricates a position / false take-profit) and R-23 (the fake
exchange's fee-in-base and locked-balance extensions, proving the SELL cap
uses ``free`` -- not ``total`` -- per D9/I1).

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
# other scenario below is trusted to be failing/passing for the right reason.
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
# buy_then_stop_loss / buy_then_take_profit -- WP1.1 fixes C1: the bracket
# manager's exit SELL now reaches the exchange instead of being rejected by
# LiveExecutionEngine.process_signal's own (previously always-empty)
# _positions guard.
# ---------------------------------------------------------------------------


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
async def test_buy_then_bracket_exit(
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
    bought_qty = exchange.order_log[-1]["amount"]

    breach_price = START_PRICE * breach_multiplier
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    reject_events = [e for e in cap if e.get("event") == "live.sell_no_position"]
    assert not reject_events, (
        f"WP1.1: the {trigger} exit must no longer be rejected as "
        f"'live.sell_no_position' -- C1 is fixed; captured logs: {cap!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, f"expected a {trigger} SELL order to reach the exchange"
    assert sell_orders[-1]["amount"] == bought_qty

    position_after_breach = stack.portfolio.get_position(SYMBOL)
    assert position_after_breach is not None and position_after_breach.is_flat, (
        f"the {trigger} exit should have fully closed the position"
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_then_trailing_stop -- WP1.1 fixes C1 via TrailingStopManager instead
# of BracketExitManager (same root cause, same fix).
# ---------------------------------------------------------------------------


async def test_buy_then_trailing_stop(exchange: FakeCCXTExchange) -> None:
    strategy = ScriptedSignalStrategy("buy-trailing", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={"trailing_stop_pct": 0.05})

    # bar0: BUY, peak seeds at 50000. bar1: new peak (55000), no trigger yet.
    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)
    bought_qty = exchange.order_log[-1]["amount"]
    await step_bar(stack, SYMBOL, Decimal("55000"), timeframe=TIMEFRAME_STR)
    position_after_peak = stack.portfolio.get_position(SYMBOL)
    assert position_after_peak is not None and not position_after_peak.is_flat

    # 55000 * (1 - 0.05) = 52250 -- 52000 breaches it.
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, Decimal("52000"), timeframe=TIMEFRAME_STR)

    trigger_events = [e for e in cap if e.get("event") == "trailing_stop.triggered"]
    assert trigger_events, f"expected trailing_stop.triggered; got {cap!r}"

    reject_events = [e for e in cap if e.get("event") == "live.sell_no_position"]
    assert not reject_events, (
        f"WP1.1: the trailing-stop exit must no longer be rejected as "
        f"'live.sell_no_position' -- C1 is fixed; captured logs: {cap!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, "expected the trailing-stop exit to reach the exchange"
    assert sell_orders[-1]["amount"] == bought_qty

    position_after_exit = stack.portfolio.get_position(SYMBOL)
    assert position_after_exit is not None and position_after_exit.is_flat

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_then_kill_switch -- still xfail: C6 (WP1.2). WP1.1 fixed C1, but the
# kill-switch early-return in LIVE mode (before the bracket/trailing
# sections even run) is unchanged and is out of this WP's scope.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C6 (Verbeterplan v2 WP1.2): in LIVE mode, StrategyEngine._process_bar "
        "returns immediately when the kill switch is active (before the "
        "bracket/trailing sections run at all), so a protective stop-loss "
        "is never even evaluated while the switch is on -- target semantics "
        "per plan decision D3 is 'block new entries, protective exits keep "
        "running'. C1 (WP1.1) is fixed: the resulting SELL would now reach "
        "the exchange if the bracket/trailing sections ran, but they never "
        "get the chance to. Fixed by WP1.2."
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
        "(C6), so the stop-loss is never evaluated."
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# buy_restart_reconcile_stop_loss -- still xfail, but the reason has
# changed: sync_positions() now runs on boot (WP11-A-07), but by design
# (I2/D15) it NEVER creates a Position from the exchange balance -- own
# quantity comes only from the run's own fills. So a restarted run's
# portfolio stays empty (the pre-restart position now looks identical to an
# external holding) until WP1.8 rebuilds it from persisted fill history.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I2/D15 (Verbeterplan v2 WP1.1 design, deferred to WP1.8): "
        "sync_positions() now runs on every engine boot (WP11-A-07), but it "
        "deliberately never creates a Position from the exchange balance -- "
        "own quantity comes only from fills routed into THIS run's own "
        "PortfolioAccounting (I2). A brand-new engine stack (orphan-recovery "
        "restart) therefore has an empty portfolio even though the exchange "
        "still holds the bot's own BTC from before the restart -- "
        "sync_positions logs live.external_holdings_ignored for it (it is "
        "now indistinguishable from a genuinely external holding) instead "
        "of fabricating a Position with a fake entry price (which is what "
        "the pre-WP1.1 code would have done, immediately mis-triggering a "
        "bracket exit at entry=0). No SELL is even attempted: safe (no "
        "false exit), but there is no protective stop until WP1.8 rebuilds "
        "the portfolio from persisted fill history on resume."
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
    # LiveExecutionEngine. stack_before is deliberately never stopped: a
    # real process restart does not get a graceful shutdown either.
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
        "known to the new engine stack; today it is NOT (I2/D15): "
        "PortfolioAccounting is rebuilt empty and sync_positions() "
        "deliberately never fabricates a Position from the balance, so the "
        "bot's own pre-restart holding now looks external."
    )

    breach_price = START_PRICE * Decimal("0.80")
    with capture_logs() as cap:
        await step_bar(stack_after, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, (
        "expected the stop-loss to fire against the entry price recorded "
        f"before the restart; today no SELL is even attempted because the "
        f"new engine's bracket manager sees position=None (I2/D15); "
        f"captured log events: {cap!r}"
    )
    assert sell_orders[-1]["amount"] == bought_qty

    await stack_after.engine.stop()


# ---------------------------------------------------------------------------
# external_holdings_not_sold (R-21) -- WP1.1 fixes C1 together with the C22
# over-sell bound (D15): the exit SELL closes exactly the bot's own
# quantity, never touching the pre-existing 0.5 BTC external holding.
# ---------------------------------------------------------------------------


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
        f"external holding (I5/D15/R-21); "
        f"(sell_orders={sell_orders!r}, captured logs={cap!r})"
    )

    assert exchange.balance_of(BASE) == Decimal("0.5"), (
        "external holdings must be untouched once the bot's own position is closed"
    )

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# R-22 -- startup sync with a pre-existing external balance must never
# fabricate a Position (and therefore never trigger a false take-profit
# on the very first bar, before the bot has even bought anything).
# ---------------------------------------------------------------------------


async def test_startup_sync_with_external_balance_causes_no_false_take_profit(
    exchange: FakeCCXTExchange,
) -> None:
    # A pre-existing external balance, seeded BEFORE the engine boots -- the
    # pre-WP1.1 sync_positions() would have fabricated a Position for this
    # with average_entry_price=0, so ANY positive price immediately breaches
    # a take-profit level computed as entry * (1 + tp_pct) == 0.
    exchange.set_balance(BASE, Decimal("0.5"))

    # Never emits a signal: isolates sync_positions()'s own behaviour at
    # boot from anything the strategy itself might do.
    strategy = ScriptedSignalStrategy("never-fires", {"direction": "buy", "call_index": 999})
    stack = await _build_and_warm(
        exchange, strategy, engine_config={"bracket_take_profit_pct": 0.05}
    )

    # sync_positions() has already run once inside on_start() (WP11-A-07),
    # against the pre-existing 0.5 BTC balance, before this first bar.
    assert stack.portfolio.get_position(SYMBOL) is None, (
        "sync_positions() must never fabricate a Position from the exchange "
        "balance (I2) -- an external holding stays untracked, not a "
        "zero-entry position waiting to falsely take-profit"
    )
    assert stack.execution.positions.get(SYMBOL) is None
    assert not stack.execution.reconcile_required

    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)

    # No bracket exit can have fired -- there is no position to exit, bought
    # or otherwise.
    assert exchange.order_log == []
    tp_events = [e for e in cap if e.get("event") == "bracket_exit.take_profit"]
    assert not tp_events, f"unexpected false take-profit on first bar: {cap!r}"
    assert exchange.balance_of(BASE) == Decimal("0.5"), "external holding must be untouched"

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# R-23 -- fake-exchange extensions (fee charged in base currency, a balance
# partially locked in another open order) prove the SELL cap uses `free`,
# never `total` (D9/I1): the bracket exit closes only what is actually
# free, not the bot's full own quantity.
# ---------------------------------------------------------------------------


async def test_sell_capped_at_free_when_balance_partially_locked(
    exchange: FakeCCXTExchange,
) -> None:
    exchange.set_fee_currency(SYMBOL, "base")

    strategy = ScriptedSignalStrategy("buy-fee-in-base", _buy_params())
    stack = await _build_and_warm(exchange, strategy, engine_config={"bracket_stop_loss_pct": 0.05})

    await step_bar(stack, SYMBOL, START_PRICE, timeframe=TIMEFRAME_STR)

    bought_qty = TARGET_NOTIONAL / START_PRICE  # 0.002 BTC
    fee_base = (bought_qty * exchange.taker_fee_pct).quantize(Decimal("0.00000001"))
    own_qty = bought_qty - fee_base  # 0.001988 BTC -- net of the base-currency fee

    position_after_buy = stack.portfolio.get_position(SYMBOL)
    assert position_after_buy is not None
    assert _approx_eq(position_after_buy.quantity, own_qty), (
        "WP11-A-05: a fee charged in the base currency must net out of the "
        f"routed fill quantity; got {position_after_buy.quantity}, expected {own_qty}"
    )
    assert exchange.balance_of(BASE) == own_qty  # engine's own qty == exchange's actual balance

    # Lock 90% of the bot's own BTC away in a (simulated) other open order:
    # `total` is unaffected, only `free` shrinks. own == total here (no
    # other external BTC balance), so the I8 mismatch flag must NOT fire --
    # only the free-based cap engages (D9).
    locked = (own_qty * Decimal("0.9")).quantize(Decimal("0.00000001"))
    free = own_qty - locked
    exchange.lock_balance(BASE, locked)

    breach_price = START_PRICE * Decimal("0.80")
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    assert not stack.execution.reconcile_required, (
        f"own == total (only `free` is reduced) -- the I8 mismatch flag "
        f"must not fire; got {dict(stack.execution.reconcile_required)!r}"
    )

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert sell_orders, f"expected a capped SELL to reach the exchange; captured logs: {cap!r}"
    assert sell_orders[-1]["amount"] == free, (
        f"the SELL must be capped at free ({free}), not own/total ({own_qty}) -- "
        f"got {sell_orders[-1]['amount']}"
    )

    await stack.engine.stop()
