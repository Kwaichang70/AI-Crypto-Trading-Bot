"""
tests/integration/test_smoke_roundtrip_live.py
--------------------------------------------------
WP-SMOKE (reports/vp2-smoke/synthesis-spec.md section 9, SMK-T-11..24) --
live harness coverage for ``SmokeRoundtripStrategy`` driven through the
*real* ``StrategyEngine`` + ``LiveExecutionEngine`` + ``PortfolioAccounting``
+ ``DefaultRiskManager`` stack, against an in-memory ``FakeCCXTExchange``.
Mirrors ``tests/integration/test_live_protective_paths.py`` and
``tests/integration/test_live_nav_sizing_harness.py`` exactly (same harness,
same ``_fast_sleep`` rationale, same "no shortcuts" design).

Scope note (producer report, Producer A; updated by WP-SMOKE fix F-6)
-------------------------------------------------------------------------
SMK-T-15's original wording ("sub-min_cost remainder stays as flagged
dust") describes an engine-internal rounding/tolerance edge case that
``FakeCCXTExchange`` resolves automatically (partial fills always
complete on the second poll); SMK-T-15 is implemented against what the
fixture actually supports (a partial-then-complete BUY poll -- I9's real
mechanics).

SMK-T-20 ("off-grid residual -> engine.flatten returns incomplete") IS
now implemented (WP-SMOKE fix F-6, reports/vp2-smoke/final-synthesis-smoke.md
section 5): rather than trying to reach the off-grid state through a BUY
fill (SMK-T-19 shows the base-fee variant always nets to a step-aligned
quantity, never an off-grid one), the ledger position is seeded directly
via ``PortfolioAccounting.from_fills`` / ``build_resumed_live_stack`` --
the SAME production rebuild path SMK-T-22's protective resume already
exercises -- with one synthetic BUY fill for an off-grid quantity. See
``test_smk_t_20_precondition_off_grid_residual`` (plain, asserts the
precondition) and ``test_smk_t_20_flatten_off_grid_residual_completes``
(``xfail(strict=True)``, pinned to CF-SMK-S1: today ``flatten()`` cannot
clear a residual between the 1e-8 dust tolerance and one amount step; it
XPASSes once CF-SMK-S1's step-based dust tolerance lands, at which point
``strict=True`` turns that XPASS into a failure, forcing the flip).
"""
from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, TimeFrame
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_live_stack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from trading.models import Fill
from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

SYMBOL = "XRP/EUR"
BASE = "XRP"
QUOTE = "EUR"
TIMEFRAME = TimeFrame.FIVE_MINUTES
TIMEFRAME_STR = "5m"
PRICE = Decimal("0.50")  # ~18 XRP for a EUR 9 BUY -- comfortably above every step/min-amount
CAPITAL = Decimal("65")
DEFAULT_TAKER = Decimal("0.012")  # D-SMK-4: the higher of the two plausible tiers


def _strategy(
    strategy_id: str = "smoke_roundtrip-live-test", **params: object
) -> SmokeRoundtripStrategy:
    merged = {"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4, **params}
    return SmokeRoundtripStrategy(strategy_id=strategy_id, params=merged)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every real sleep in the live path (see
    test_live_protective_paths.py's identical fixture for the full
    rationale -- same call sites, same reason)."""

    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


def _make_exchange(
    *,
    free_eur: Decimal = Decimal("70"),
    taker_fee_pct: Decimal = DEFAULT_TAKER,
    min_cost: str | None = "1",
) -> FakeCCXTExchange:
    ex = FakeCCXTExchange(taker_fee_pct=taker_fee_pct)
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE, min_cost=min_cost, min_amount="0.01")
    ex.set_balance(QUOTE, free_eur)
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TIMEFRAME_STR)
    return ex


async def _build_and_warm(
    exchange: FakeCCXTExchange,
    strategy: SmokeRoundtripStrategy,
    *,
    initial_capital: Decimal = CAPITAL,
    run_id: str = "smoke-live-test-run",
    engine_config: dict[str, object] | None = None,
) -> LiveStack:
    merged_config: dict[str, object] = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    if engine_config:
        merged_config.update(engine_config)
    stack = await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=initial_capital,
        run_id=run_id,
        engine_config=merged_config,
    )
    await start_and_warmup(stack, run_id)
    return stack


# ===========================================================================
# SMK-T-11: happy path.
# ===========================================================================


async def test_smk_t_11_happy_path_buy_then_sell_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange(free_eur=Decimal("70"), taker_fee_pct=Decimal("0.012"))
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)

    with capture_logs() as cap:
        stack = await _build_and_warm(exchange, strategy, run_id="smk-t11-run")

    assert not any(e.get("event") == "live.initial_capital_exceeds_free_quote" for e in cap)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1
    notional = buy_orders[0]["amount"] * buy_orders[0]["price"]
    assert Decimal("8.00") <= notional <= Decimal("9.60")
    for order in stack.execution.get_all_orders():
        assert order.client_order_id.startswith("smk-t11-run-"), order.client_order_id

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit window bar 1
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1

    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat
    assert exchange.balance_of(BASE) == Decimal("0")

    await stack.execution.sync_positions()
    assert stack.execution.reconcile_required == {}

    for _ in range(30):
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    assert len(exchange.order_log) == 2, f"expected no further orders; got {exchange.order_log!r}"

    await stack.engine.stop()


# ===========================================================================
# SMK-T-12: below-min-cost silent drop vs a fill.
# ===========================================================================


async def test_smk_t_12_below_min_cost_dropped_capital_10(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange(free_eur=Decimal("70"), min_cost="2")
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy()
    # G-4/G-5 would reject capital=10 at the API layer -- this bypasses the
    # guard entirely (harness-only) to prove the engine's own min-cost
    # floor is a second, independent line of defence (V5).
    with capture_logs() as cap:
        stack = await _build_and_warm(exchange, strategy, initial_capital=Decimal("10"))
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    assert exchange.order_log == []
    assert any(e.get("event") == "live.below_min_cost" for e in cap)
    await stack.engine.stop()


async def test_smk_t_12_fills_at_capital_65(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange(free_eur=Decimal("70"), min_cost="2")
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy()
    stack = await _build_and_warm(exchange, strategy, initial_capital=CAPITAL)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    assert len([o for o in exchange.order_log if o["side"] == "buy"]) == 1
    await stack.engine.stop()


# ===========================================================================
# SMK-T-13: REJECTED SELL -> next bar SELL fills; ambiguous SELL -> no
# double SELL.
# ===========================================================================


async def test_smk_t_13_rejected_sell_retried_next_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    import ccxt

    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t13-rejected")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    assert stack.portfolio.get_position(SYMBOL) is not None

    exchange.queue_order_error(SYMBOL, ccxt.InvalidOrder(f"{SYMBOL} rejected"))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit attempt 1: rejected
    assert [o for o in exchange.order_log if o["side"] == "sell"] == []
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit attempt 2: fills
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1

    await stack.engine.stop()


async def test_smk_t_13_ambiguous_sell_no_double_sell(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t13-ambiguous")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry

    # The exchange actually accepts and places the SELL (balance mutated,
    # order/trade recorded) but the response never reaches the engine --
    # an ambiguous submit (ccxt.RequestTimeout by default).
    exchange.queue_accept_then_timeout(SYMBOL)
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit attempt 1: ambiguous
    sell_orders_after_1 = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders_after_1) == 1, "the exchange DID place the order despite the timeout"

    with capture_logs() as cap2:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit attempt 2
    sell_orders_after_2 = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders_after_2) == 1, (
        f"expected no double SELL; got {sell_orders_after_2!r}; logs={cap + cap2!r}"
    )

    await stack.engine.stop()


# ===========================================================================
# SMK-T-14: a BUY that reports 0-filled on its first exchange poll and
# only fully routes on the retry -- proves the entry (and the strategy's
# subsequent SELL window) is never sized off a stale zero/partial figure.
#
# Scope note (Producer A): LiveExecutionEngine's own post-submit
# reconcile loop resolves this fully INSIDE the same process_signal call
# (the harness's ``_fast_sleep`` fixture collapses its internal wait to
# zero, exactly like every other scenario in this module and in
# test_live_protective_paths.py) -- so this does not reproduce a fill
# that only routes on a bar AFTER the exit window has already started
# (the synthesis spec's literal "a later window bar sells" framing).
# Reproducing that would need to defer FakeCCXTExchange's own poll
# resolution across ``_poll_and_process()`` calls, which risks
# masking real reconcile-loop behaviour rather than testing it. What
# this test does prove: the BUY's eventual routed quantity (not the
# stale first-poll figure) is what the strategy's own SELL sells.
# ===========================================================================


async def test_smk_t_14_buy_zero_filled_first_poll_still_routes_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t14-zero-first-poll")

    # The BUY's first fetch_order poll reports 0 filled (still "open");
    # FakeCCXTExchange resolves it fully on the very next poll (I9).
    exchange.queue_partial_fill(SYMBOL, Decimal("0"))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    bought_qty = next(o for o in exchange.order_log if o["side"] == "buy")["amount"]
    assert bought_qty > Decimal("0"), "the BUY must route to its real qty, not the 0 first poll"

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0]["amount"] == bought_qty

    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack.engine.stop()


# ===========================================================================
# SMK-T-15: a partial-then-complete BUY poll (I9's real partial-fill /
# idempotent-routing mechanics) -- the eventual SELL equals the fully
# routed quantity, never a stale partial figure.
# ===========================================================================


async def test_smk_t_15_partial_then_complete_buy_sell_matches_routed_qty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t15-partial-buy")

    exchange.queue_partial_fill(SYMBOL, Decimal("0.5"))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    bought_qty = next(o for o in exchange.order_log if o["side"] == "buy")["amount"]

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0]["amount"] == bought_qty, (
        "the SELL must match the fully-routed BUY quantity, not the stale "
        "50% figure reported by the first poll"
    )
    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack.engine.stop()


# ===========================================================================
# SMK-T-19: a base-currency fee variant (D6/A-05 fee normalisation) -- the
# strategy's SELL still nets the position to flat, with any residual
# under one amount step and no reconcile flag.
# ===========================================================================


async def test_smk_t_19_base_currency_fee_variant_nets_to_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    exchange.set_fee_currency(SYMBOL, "base")
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t19-base-fee")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1

    residual = exchange.balance_of(BASE)
    assert residual < Decimal("0.01"), f"residual base balance too large: {residual}"
    assert stack.execution.reconcile_required == {}
    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack.engine.stop()


# ===========================================================================
# SMK-T-16: -8% gap during the hold -- strategy SELL fires first; the SL
# check on the same bar is a no-op; exactly one SELL.
# ===========================================================================


async def test_smk_t_16_gap_down_strategy_sell_wins_over_sl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t16-gap")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry

    gap_price = (PRICE * Decimal("0.92")).quantize(Decimal("0.0001"))  # -8%, breaches the 5% SL
    await step_bar(stack, SYMBOL, gap_price, timeframe=TIMEFRAME_STR)  # exit window bar 1

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1, f"expected exactly one SELL; got {exchange.order_log!r}"
    position = stack.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack.engine.stop()


# ===========================================================================
# SMK-T-17: on_bar raises from the second call onward -> the bracket SL
# still fires independently, exit_reason=stop_loss.
# ===========================================================================


async def test_smk_t_17_strategy_exception_does_not_block_bracket_sl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=6, exit_retry_bars=4)  # long hold so the SL fires first
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t17-raises")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry (call 1, no raise)
    assert stack.portfolio.get_position(SYMBOL) is not None

    call_count = {"n": 1}
    real_on_bar = strategy.on_bar

    def _raising_on_bar(*args: object, **kwargs: object) -> list[object]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("smoke_roundtrip.on_bar boom (test-only fault injection)")
        return real_on_bar(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(strategy, "on_bar", _raising_on_bar)

    breach_price = (PRICE * Decimal("0.90")).quantize(Decimal("0.0001"))  # -10%, breaches 5% SL
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    assert any(e.get("event") == "engine.strategy_on_bar_error" for e in cap), cap
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1, (
        f"expected the bracket SL to fire despite the exception; got {exchange.order_log!r}"
    )

    trades = stack.portfolio.get_trade_history()
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_loss"

    await stack.engine.stop()


# ===========================================================================
# SMK-T-18: external holding of 50 XRP -- only own qty sold; no reconcile
# flag.
# ===========================================================================


async def test_smk_t_18_external_holding_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange()
    exchange.set_balance(BASE, Decimal("50"))  # pre-existing external holding
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=1, exit_retry_bars=4)

    with capture_logs() as cap:
        stack = await _build_and_warm(exchange, strategy, run_id="smk-t18-external")
    assert any(e.get("event") == "live.external_holdings_ignored" for e in cap)
    assert stack.execution.reconcile_required == {}

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    bought_qty = next(o for o in exchange.order_log if o["side"] == "buy")["amount"]
    total_after_buy = exchange.balance_of(BASE)
    assert total_after_buy == Decimal("50") + bought_qty

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # exit
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0]["amount"] == bought_qty, (
        "must sell only the run's own qty, never the external 50"
    )
    assert exchange.balance_of(BASE) == Decimal("50"), "the external 50 must be left untouched"
    assert stack.execution.reconcile_required == {}

    await stack.engine.stop()


# ===========================================================================
# SMK-T-21: global kill switch latched at start -> BUY dropped, 0 orders.
# ===========================================================================


async def test_smk_t_21_kill_switch_at_start_drops_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy()
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t21-kill")

    stack.risk_manager.trigger_kill_switch("wp-smoke-harness-test")
    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    assert exchange.order_log == []
    assert any(e.get("event") == "engine.kill_switch_entries_blocked" for e in cap)

    await stack.engine.stop()


# ===========================================================================
# SMK-T-22: BUY -> restart -> protective resume -> the fresh instance's
# BUY is dropped -> the strategy's own SELL (or the SL) closes -> flat;
# exactly 1 BUY in order_log.
# ===========================================================================


async def test_smk_t_22_protective_resume_drops_second_buy_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    run_id = "smk-t22-resume"

    strategy_before = _strategy(
        strategy_id="smoke_roundtrip-before", hold_bars=6, exit_retry_bars=4
    )
    stack_before = await _build_and_warm(exchange, strategy_before, run_id=run_id)
    await step_bar(stack_before, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    bought_qty = next(o for o in exchange.order_log if o["side"] == "buy")["amount"]
    persisted_fills = stack_before.execution.get_all_fills()

    # A brand-new instance -- IDLE -> would BUY again on its first call,
    # but protective resume must drop that entry.
    strategy_after = _strategy(strategy_id="smoke_roundtrip-after", hold_bars=1, exit_retry_bars=4)
    stack_after = await build_resumed_live_stack(
        exchange=exchange,
        strategy=strategy_after,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=CAPITAL,
        run_id=run_id,
        fills=persisted_fills,
        engine_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
        protective_mode=True,
    )
    await start_and_warmup(stack_after, run_id)

    orders_before = len(exchange.order_log)
    await step_bar(stack_after, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # fresh call 1: BUY dropped
    await step_bar(stack_after, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # fresh call 2: SELL

    new_orders = exchange.order_log[orders_before:]
    buy_orders = [o for o in new_orders if o["side"] == "buy"]
    sell_orders = [o for o in new_orders if o["side"] == "sell"]
    assert buy_orders == [], "protective mode must drop the fresh instance's entry"
    assert sell_orders, "the strategy's own SELL (or the SL) must still close the rebuilt position"
    assert sell_orders[-1]["amount"] == bought_qty

    assert len([o for o in exchange.order_log if o["side"] == "buy"]) == 1, (
        f"expected exactly one BUY across the whole run; got {exchange.order_log!r}"
    )
    position = stack_after.portfolio.get_position(SYMBOL)
    assert position is None or position.is_flat

    await stack_after.engine.stop()


# ===========================================================================
# SMK-T-23: Run B -- kill switch + engine.flatten() while holding.
# ===========================================================================


async def test_smk_t_23_flatten_while_holding(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy(hold_bars=6, exit_retry_bars=4)  # Run B: a longer hold
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t23-run-b")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry
    bought_qty = next(o for o in exchange.order_log if o["side"] == "buy")["amount"]
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat

    stack.risk_manager.trigger_kill_switch("stop_in_progress")
    result = await stack.engine.flatten(reason="operator_stop", timeout_s=30.0)

    assert result.outcome == "flattened"
    assert result.complete is True
    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[-1]["amount"] == bought_qty

    trades = stack.portfolio.get_trade_history()
    assert len(trades) == 1
    assert trades[0].strategy_id == "operator_flatten"

    # No BUY afterwards, even if bars keep coming.
    orders_before = len(exchange.order_log)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    assert len(exchange.order_log) == orders_before

    await stack.engine.stop()


# ===========================================================================
# SMK-T-24: free EUR below initial_capital -- the abort-trigger warning
# fires; a very low free EUR caps the BUY.
# ===========================================================================


async def test_smk_t_24_free_eur_below_capital_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange(free_eur=Decimal("30"))  # < capital 65
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy()

    with capture_logs() as cap:
        stack = await _build_and_warm(exchange, strategy, run_id="smk-t24-low-free")

    assert any(e.get("event") == "live.initial_capital_exceeds_free_quote" for e in cap)
    await stack.engine.stop()


async def test_smk_t_24_very_low_free_eur_caps_the_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = _make_exchange(free_eur=Decimal("5"))
    patch_exchange_factory(monkeypatch, exchange)
    strategy = _strategy()
    stack = await _build_and_warm(exchange, strategy, run_id="smk-t24-capped")

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1
    notional = buy_orders[0]["amount"] * buy_orders[0]["price"]
    # Capped to ~ free_eur / (1 + taker + slippage), never the full EUR 9 target.
    assert notional < Decimal("5")

    await stack.engine.stop()


# ===========================================================================
# WP-SMOKE fix F-6 (SMK-SEC-07) / SMK-T-20: an off-grid ledger residual
# (between the 1e-8 dust tolerance and one amount step) makes
# engine.flatten() end "partial", not "flattened" -- pinned to CF-SMK-S1.
# ===========================================================================

_OFF_GRID_STEP = Decimal("0.01")
#: 0.005 above the nearest 0.01-step multiple -- genuinely off-grid: the
#: residual is well above the 1e-8 dust tolerance _flatten_symbol checks,
#: but strictly below one amount step, so flooring any SELL of it to the
#: step yields exactly zero and the residual can never shrink.
_OFF_GRID_HELD_QTY = Decimal("12.925")


def _off_grid_market(exchange: FakeCCXTExchange) -> None:
    """Re-register SYMBOL with a coarse 2-decimal (0.01) amount step --
    _make_exchange()'s default is 8 decimals, too fine to ever leave an
    off-grid residual."""
    exchange.register_market(
        SYMBOL, base=BASE, quote=QUOTE, min_cost="1", min_amount="0.0001", amount_precision=2
    )


def _seed_off_grid_fill(run_id: str) -> Fill:
    """One synthetic BUY fill giving the run an off-grid ledger quantity
    directly, via the SAME production rebuild path SMK-T-22's protective
    resume already exercises (``PortfolioAccounting.from_fills``) --
    SMK-T-19 shows a real BUY-fill path always nets to a step-aligned
    quantity, so this is the only deterministic way to reach the
    precondition (F-6 step 2)."""
    return Fill(
        order_id=uuid.uuid4(),
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=_OFF_GRID_HELD_QTY,
        price=PRICE,
        fee=Decimal("0"),
        fee_currency=QUOTE,
    )


async def _build_off_grid_stack(monkeypatch: pytest.MonkeyPatch, run_id: str) -> LiveStack:
    exchange = _make_exchange()
    _off_grid_market(exchange)
    # The exchange's own base balance must equal the seeded ledger qty --
    # a real restart does not change either.
    exchange.set_balance(BASE, _OFF_GRID_HELD_QTY)
    patch_exchange_factory(monkeypatch, exchange)

    strategy = SmokeRoundtripStrategy(
        strategy_id=f"smoke_roundtrip-{run_id}",
        params={"notional_quote": 9.0, "hold_bars": 6, "exit_retry_bars": 4},
    )
    stack = await build_resumed_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=CAPITAL,
        run_id=run_id,
        fills=[_seed_off_grid_fill(run_id)],
        engine_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
    )
    await start_and_warmup(stack, run_id)
    return stack


async def test_smk_t_20_precondition_off_grid_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precondition only (not xfail): the seeded ledger position is
    genuinely off-grid -- strictly between the 1e-8 dust tolerance and
    one amount step (0.01) past the nearest step multiple."""
    stack = await _build_off_grid_stack(monkeypatch, "smk-t20-precondition")

    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None
    held = position.quantity
    assert held == _OFF_GRID_HELD_QTY

    residual = held % _OFF_GRID_STEP
    assert 0 < residual
    assert residual > Decimal("0.00000001")  # the flatten dust tolerance (1e-8)
    assert residual < _OFF_GRID_STEP

    await stack.engine.stop()


@pytest.mark.xfail(
    strict=True,
    reason="CF-SMK-S1: flatten dust tolerance is 1e-8, not one amount step",
)
async def test_smk_t_20_flatten_off_grid_residual_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned to CF-SMK-S1 (synthesis-spec.md section 15). Observed today
    (recorded here per F-6 step 5): ``flatten()`` submits exactly one SELL,
    which fills for the floor of 12.925 to the 0.01 step -- 12.92 -- and
    leaves an 0.005 residual that can never shrink (flooring 0.005 to the
    same 0.01 step yields exactly 0, so every subsequent attempt sells
    nothing). The result today is ``outcome="partial"``,
    ``complete=False``, and the symbol's own ``status="partial"`` with
    ``remaining_qty==Decimal("0.005")`` -- which the API's stop endpoint
    turns into a 409 (spec section 10, Run A stop step 1 / section 12 step
    2 already document the dust-409 handling this exercises). This test
    asserts the CORRECT (post-CF-SMK-S1) behaviour, so it fails today by
    design; once CF-SMK-S1 lands (a step-based dust tolerance, or a
    dedicated 'dust' outcome), it XPASSes and ``strict=True`` turns that
    XPASS into a failure here, forcing this assertion to be updated --
    the flip F-6/CF-SMK-S1 describes.
    """
    stack = await _build_off_grid_stack(monkeypatch, "smk-t20-flatten")

    stack.risk_manager.trigger_kill_switch("stop_in_progress")
    result = await stack.engine.flatten(reason="stop", timeout_s=5.0)

    assert result.complete is True
    assert result.symbols[0].status in ("flat", "dust")

    await stack.engine.stop()
