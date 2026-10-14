"""
tests/integration/test_wp13b_anchor_harness.py
------------------------------------------------
WP1.3b-a (anchoring in all modes, trailing semantics, R-04, resume recompute)
through the REAL live stack -- ``StrategyEngine`` + ``LiveExecutionEngine`` +
``PortfolioAccounting`` + ``CCXTMarketDataService`` over ``FakeCCXTExchange`` --
and the real ``BacktestRunner``.  WP13B-T-17/18 (live), T-20..T-30, T-32.

Clock model.  The fake stamps fills with its notion of "now"
(``set_now_ms``).  A real fill happens AFTER the signal bar closed, so every
step below moves "now" to the bar CLOSE + 2 s (``_step``); that is what makes
``anchor_at = min(order.created_at, fill.executed_at)`` select the signal bar.

    T-17  HWM_0 is the exchange fill, not the signal close            (live, red)
    T-18  the same-bar check does not ratchet with the signal close   (live, red)
    T-20  R-04: an exit that was not submitted is retried next bar    (red)
    T-21  R-04 negative: an accepted (unknown) SELL keeps the guard
    T-22  resume equality: the recomputed anchor == the live anchor
    T-23  a lot older than the warmup window -> `since` fetch; fetch fails -> fallback
    T-24  WP1.8b imported fills anchor on the rule bar
    T-25  trailing mark, fresh vs resumed (gap fetch); fetch fails -> seed_partial
    T-26  end-to-end ATR immutability (live)                           (red)
    T-27  an opening-order fill that arrives via check_resting_orders  (red)
    T-28  boundary race: provisional -> finalised == resume anchor
    T-29  gapped window: provisional -> bounded fetch == resume anchor; failure
    T-30  backtest determinism + byte-identical golden (fixed, no trailing/adds)
    T-32  reconcile_required symbol: the bar-close SL still fires
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import ccxt
import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, OrderType, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_live_stack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
)
from tests.integration.fakes.scripted_strategy import (
    ScriptedSequenceStrategy,
)
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE, CostProfile, get_cost_profile
from trading.models import Fill, Order

SYMBOL = "BTC/EUR"
TF = "1h"
TF_MS = 3_600_000
RUN_ID = "wp13b-run"
INITIAL = Decimal("1000")

FIXED_CFG: dict[str, object] = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
ATR_CFG: dict[str, object] = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_tp_multiplier": 3.0,
    "bracket_atr_period": 14,
}
ATR_SL_ONLY: dict[str, object] = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_period": 14,
}


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("100000"))
    patch_exchange_factory(monkeypatch, ex)
    return ex


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wavy(ex: FakeCCXTExchange, n: int = 100, amp: int = 2) -> None:
    """``n`` closed bars alternating 100 / 100+amp (ATR ~ amp)."""
    for i in range(n):
        ex.push_bar(SYMBOL, Decimal(100 + (i % 2) * amp), timeframe=TF)


def _last_ts_ms(ex: FakeCCXTExchange) -> int:
    return int(ex._closed_bars[SYMBOL][-1][0])


def _set_now_after_close(
    ex: FakeCCXTExchange, *, bars_after: int = 0, latency_ms: int = 2000
) -> None:
    ex.set_now_ms(_last_ts_ms(ex) + (1 + bars_after) * TF_MS + latency_ms)


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


async def _step(stack: LiveStack, close: int | str | Decimal) -> None:
    """Push one closed bar, move 'now' to its close + 2 s, run the real bar pipeline."""
    stack.exchange.push_bar(SYMBOL, Decimal(str(close)), timeframe=TF)
    _set_now_after_close(stack.exchange)
    await stack.engine._poll_and_process()


async def _live(
    ex: FakeCCXTExchange,
    cfg: dict[str, object],
    *,
    buy_at: int | None = 0,
    script: dict[int, str] | None = None,
    targets: dict[int, str] | None = None,
    start: bool = True,
) -> LiveStack:
    if script is None:
        script = {} if buy_at is None else {buy_at: "buy"}
    strategy = ScriptedSequenceStrategy(
        "wp13b", {"script": script, "target_notional": "100", "targets": targets or {}}
    )
    stack = await build_live_stack(
        exchange=ex,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        engine_config=cfg,
    )
    if start:
        await start_and_warmup(stack, RUN_ID)
    return stack


async def _resume(
    ex: FakeCCXTExchange,
    cfg: dict[str, object],
    fills: list[Fill],
    orders: list[Order],
    *,
    pad_forming_bar: bool = False,
) -> LiveStack:
    ex.pad_forming_bar = pad_forming_bar
    stack = await build_resumed_live_stack(
        exchange=ex,
        strategy=ScriptedSequenceStrategy("wp13b-r", {"script": {}}),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        fills=fills,
        orders=orders,
        engine_config=cfg,
    )
    await start_and_warmup(stack, RUN_ID)
    ex.pad_forming_bar = True
    return stack


def _sells(ex: FakeCCXTExchange) -> list[dict[str, Any]]:
    return [o for o in ex.order_log if o["side"] == "sell"]


def _fail_since_fetches(
    ex: FakeCCXTExchange,
    monkeypatch: pytest.MonkeyPatch,
    message: str = "simulated since-fetch outage",
) -> None:
    """Every ``fetch_ohlcv(since=...)`` raises (the warmup ``since=None`` call is untouched)."""
    original = ex.fetch_ohlcv

    async def _flaky(
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        if since is not None:
            raise ccxt.NetworkError(message)
        return await original(symbol, timeframe, since, limit, params)

    monkeypatch.setattr(ex, "fetch_ohlcv", _flaky)


def _persisted(stack: LiveStack) -> tuple[list[Fill], list[Order]]:
    return stack.execution.get_all_fills(), stack.execution.get_all_orders()


# ---------------------------------------------------------------------------
# T-17 / T-18  trailing mark seeded from the exchange fill (live)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("slippage", "label"), [("0.01", "above"), ("-0.01", "below")])
async def test_t17_t18_trailing_hwm0_is_the_exchange_fill_and_the_signal_close_never_ratchets(
    exchange: FakeCCXTExchange, slippage: str, label: str
) -> None:
    """A live BUY fills at the exchange's price, 1% above (T-17) / below (T-18) the
    signal close of 100.  HEAD seeded the peak from the signal bar's close (100) and
    the same-bar 5b check ratcheted with it; WP1.3b-a seeds the gross fill price and
    only later bars ratchet."""
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(exchange, {"trailing_stop_pct": 0.10})
    exchange.set_fill_slippage_pct(SYMBOL, Decimal(slippage))
    await _step(stack, 100)
    fills = stack.execution.get_all_fills()
    assert len(fills) == 1
    fill_price = fills[0].price
    assert fill_price == Decimal("100") * (Decimal("1") + Decimal(slippage)), label
    trailing = stack.engine._trailing_stop
    assert trailing is not None
    assert trailing.peak_prices[SYMBOL] == fill_price  # HEAD: 100 (the signal close)
    # the next bar's close does ratchet it
    await _step(stack, 103)
    assert trailing.peak_prices[SYMBOL] == Decimal("103")


# ---------------------------------------------------------------------------
# T-20 / T-21  R-04
# ---------------------------------------------------------------------------


async def _entered_fixed(ex: FakeCCXTExchange) -> LiveStack:
    ex.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(ex, FIXED_CFG)
    await _step(stack, 100)  # BUY; stop = 100.6 * 0.95 = 95.57
    assert stack.portfolio.get_position(SYMBOL) is not None
    return stack


@pytest.mark.parametrize("variant", ["empty_list", "rejected", "raised"])
async def test_t20_r04_an_exit_that_was_not_submitted_is_retried_on_the_next_bar(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """HEAD marks the symbol pending before submission and keeps it for 3 bars even when
    nothing reached the exchange.  Now: [] / [REJECTED] / raised -> pending cleared at once,
    `engine.exit_not_submitted` logged, and the next bar's SELL succeeds."""
    stack = await _entered_fixed(exchange)
    bracket = stack.engine._bracket_exit
    assert bracket is not None

    if variant == "empty_list":  # ticker failure inside process_signal -> []
        outage = {"on": True}
        original = exchange.fetch_ticker

        async def _ticker(symbol: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
            if outage["on"]:
                raise ccxt.NetworkError("simulated ticker outage")
            return await original(symbol, params)

        monkeypatch.setattr(exchange, "fetch_ticker", _ticker)
    elif variant == "rejected":
        exchange.queue_order_error(SYMBOL, ccxt.InvalidOrder("simulated rejection"))
    else:
        real = stack.execution.process_signal
        state = {"raise": True}

        async def _boom(signal: Any) -> Any:
            if state["raise"]:
                state["raise"] = False
                raise RuntimeError("simulated submit crash")
            return await real(signal)

        monkeypatch.setattr(stack.execution, "process_signal", _boom)

    with capture_logs() as first:
        await _step(stack, 90)  # below the stop: the exit is emitted, nothing is submitted
    assert any(e["event"] == "engine.exit_not_submitted" for e in first), variant
    assert bracket.pending_exit_symbols == set()
    assert _sells(exchange) == []
    assert not stack.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]

    if variant == "empty_list":
        outage["on"] = False
    with capture_logs() as second:
        await _step(stack, 89)  # retried at once -- HEAD waits out the 3-bar TTL
    assert len(_sells(exchange)) == 1, variant
    assert stack.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]
    assert sum(1 for e in second if e["event"] == "bracket_exit.bracket_stop_loss") == 1


async def test_t20d_crash_after_the_exchange_accepted_never_sells_twice(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strong form of the R-04 'raised' variant (WP13BA-CR-08): ``process_signal``
    crashes AFTER the exchange accepted (and filled) the SELL.  R-04 clears the guard and
    retries on the next bars; the in-flight / evidence-based reservation and the position
    check must keep it at exactly ONE SELL."""
    stack = await _entered_fixed(exchange)
    real = stack.execution.process_signal
    state = {"raise": True}

    async def _crash_after_submit(signal: Any) -> Any:
        orders = await real(signal)
        if state["raise"]:
            state["raise"] = False
            raise RuntimeError("simulated crash after the exchange accepted the order")
        return orders

    monkeypatch.setattr(stack.execution, "process_signal", _crash_after_submit)
    await _step(stack, 90)  # the exit is submitted and filled, then the call raises
    for close in (89, 88, 87):
        await _step(stack, close)
    assert len(_sells(exchange)) == 1


async def test_t20e_retry_after_rejection_never_sells_external_holdings(
    exchange: FakeCCXTExchange,
) -> None:
    """The R-04 retry after a rejected exit sells only the run's own quantity; coins the run
    does not own (WP1.11a cap) stay untouched (WP13BA-CR-08)."""
    stack = await _entered_fixed(exchange)
    own = stack.portfolio.get_position(SYMBOL).quantity  # type: ignore[union-attr]
    exchange.set_balance("BTC", exchange.balance_of("BTC") + Decimal("0.5"))
    exchange.queue_order_error(SYMBOL, ccxt.InvalidOrder("simulated rejection"))
    await _step(stack, 90)  # first exit rejected
    assert _sells(exchange) == []
    await _step(stack, 89)  # retried at once
    sells = _sells(exchange)
    assert len(sells) == 1
    assert Decimal(str(sells[0]["amount"])) <= own
    assert exchange.balance_of("BTC") == Decimal("0.5")



async def test_t21_r04_negative_an_accepted_unknown_sell_keeps_the_pending_guard(
    exchange: FakeCCXTExchange,
) -> None:
    """The SELL reached the exchange but the response timed out (WP1.4b unknown submit):
    accepted -> the guard stays, nothing is re-emitted before the TTL, and every
    re-emission after it is blocked by the live in-flight reservation (one create_order)."""
    stack = await _entered_fixed(exchange)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    exchange.queue_accept_then_timeout(SYMBOL, ccxt.RequestTimeout("simulated timeout"))

    with capture_logs() as logs:
        await _step(stack, 90)
    assert not any(e["event"] == "engine.exit_not_submitted" for e in logs)
    assert bracket.pending_exit_symbols == {SYMBOL}
    assert len(_sells(exchange)) == 1

    emitted_before_ttl = 0
    for close in (89, 88):  # ages 1 and 2 < ttl 3
        with capture_logs() as logs:
            await _step(stack, close)
        emitted_before_ttl += sum(1 for e in logs if e["event"] == "bracket_exit.bracket_stop_loss")
        assert len(_sells(exchange)) == 1
    assert emitted_before_ttl == 0
    for close in (87, 86, 85):  # TTL expired: re-emitted, but never a second create_order
        await _step(stack, close)
        assert len(_sells(exchange)) == 1


# ---------------------------------------------------------------------------
# T-22  resume equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fixed", "atr"])
async def test_t22_resume_recompute_equals_the_live_entry_anchor(
    exchange: FakeCCXTExchange, mode: str
) -> None:
    cfg = FIXED_CFG if mode == "fixed" else ATR_CFG
    _wavy(exchange)
    live_stack = await _live(exchange, cfg)
    await _step(live_stack, 101)
    live = live_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert live is not None and live.source == "entry" and live.final
    fills, orders = _persisted(live_stack)

    resumed_stack = await _resume(exchange, cfg, fills, orders)
    resumed = resumed_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert resumed is not None
    assert resumed.source == "resume_recomputed" and resumed.final
    # Decimal ==, not approx: AC-13B-04
    assert resumed.entry_basis == live.entry_basis
    assert resumed.atr == live.atr
    assert resumed.stop_price == live.stop_price
    assert resumed.take_profit_price == live.take_profit_price
    assert resumed.anchor_bar_ts == live.anchor_bar_ts
    assert resumed.opening_order_id == live.opening_order_id
    if mode == "atr":
        assert live.atr is not None  # the ATR equality above is not vacuous

    # the SL fires at the same level as on the run without a restart
    with capture_logs() as logs:
        await _step(resumed_stack, 85)
    fired = next(e for e in logs if e["event"] == "bracket_exit.bracket_stop_loss")
    assert Decimal(fired["level_price"]) == live.stop_price
    assert len(_sells(exchange)) == 1


# ---------------------------------------------------------------------------
# T-23  lot older than the warmup window
# ---------------------------------------------------------------------------


async def _old_lot(
    exchange: FakeCCXTExchange, cfg: dict[str, object]
) -> tuple[LiveStack, list[Fill], list[Order]]:
    _wavy(exchange)
    stack = await _live(exchange, cfg)
    await _step(stack, 101)
    for i in range(130):  # hold: wavy closes stay inside the bracket
        await _step(stack, 100 + (i % 2) * 2)
    fills, orders = _persisted(stack)
    return stack, fills, orders


async def test_t23_lot_older_than_warmup_window_uses_a_since_fetch(
    exchange: FakeCCXTExchange,
) -> None:
    live_stack, fills, orders = await _old_lot(exchange, ATR_CFG)
    live = live_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert live is not None
    exchange.ohlcv_calls.clear()
    resumed_stack = await _resume(exchange, ATR_CFG, fills, orders)
    resumed = resumed_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert resumed is not None and resumed.source == "resume_recomputed"
    assert resumed.atr == live.atr and resumed.stop_price == live.stop_price
    since_calls = [c for c in exchange.ohlcv_calls if c[2] is not None]
    assert since_calls, "a bounded `since` fetch was expected"


async def test_t23_fetch_failure_falls_back_to_the_current_atr_and_still_protects(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP1.3b-a FX-03: the resume fallback is the INTERIM ATR and NOT final; the exact window
    is retried on the next passes (attempts 2 and 3), then it settles, tagged and logged."""
    _live_stack, fills, orders = await _old_lot(exchange, ATR_CFG)
    _fail_since_fetches(exchange, monkeypatch)
    with capture_logs() as logs:
        resumed_stack = await _resume(exchange, ATR_CFG, fills, orders)
    resumed = resumed_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert resumed is not None
    assert resumed.source == "resume_fallback_current_atr" and not resumed.final
    assert resumed.stop_price is not None  # protected
    armed = [e for e in logs if e["event"] == "bracket_exit.interim_armed"]
    assert armed and armed[0]["log_level"] == "warning"
    assert armed[0]["source"] == "resume_fallback_current_atr"
    # attempts 2 and 3 happen on the next two passes; then the interim levels are final
    with capture_logs() as later:
        await _step(resumed_stack, 100)
        mid = resumed_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
        assert mid is not None and not mid.final
        await _step(resumed_stack, 100)
    settled = resumed_stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert settled is not None and settled.final
    assert settled.source == "resume_fallback_current_atr"
    assert (settled.stop_price, settled.atr) == (resumed.stop_price, resumed.atr)
    err = [e for e in later if e["event"] == "bracket_exit.resume_fallback_final"]
    assert err and err[0]["log_level"] == "error"
    # and the position really is protected: a crash below the stop exits
    await _step(resumed_stack, 70)
    assert len(_sells(exchange)) == 1


async def test_t23_metric_is_tagged_with_the_anchor_source(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_stack, fills, orders = await _old_lot(exchange, ATR_CFG)
    _fail_since_fetches(exchange, monkeypatch)
    seen: list[tuple[str, dict[str, str]]] = []
    from common.metrics import metrics as registry

    original = registry.increment

    def _spy(name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        seen.append((name, dict(labels or {})))
        original(name, value, labels)

    monkeypatch.setattr(registry, "increment", _spy)
    await _resume(exchange, ATR_CFG, fills, orders)
    assert any(
        n == "bracket_anchor_source_total" and lab.get("source") == "resume_fallback_current_atr"
        for n, lab in seen
    )


# ---------------------------------------------------------------------------
# T-24  WP1.8b imported fills
# ---------------------------------------------------------------------------


async def test_t24_imported_fill_anchors_on_the_rule_bar_and_the_sl_fires(
    exchange: FakeCCXTExchange,
) -> None:
    _wavy(exchange)
    exchange.set_balance("BTC", Decimal("1"))
    bought_at = _ts(_last_ts_ms(exchange) + TF_MS + 5000)  # 5 s after the last bar closed
    order = Order(
        client_order_id=f"{RUN_ID}-imported",
        run_id=RUN_ID,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        # an imported order is stamped at import time -- LATER than its fill
        created_at=bought_at + timedelta(days=2),
    )
    fill = Fill(
        order_id=order.order_id,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.6"),
        fee_currency="EUR",
        executed_at=bought_at,
    )
    stack = await _resume(exchange, ATR_CFG, [fill], [order])
    anchor = stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert anchor is not None and anchor.source == "resume_recomputed"
    assert anchor.anchor_bar_ts == _ts(_last_ts_ms(exchange))  # the last closed bar
    assert anchor.anchor_at == bought_at  # min(created_at, executed_at)
    assert anchor.entry_basis == Decimal("100.6")
    with capture_logs() as logs:
        await _step(stack, 90)  # through the SL
    fired = next(e for e in logs if e["event"] == "bracket_exit.bracket_stop_loss")
    assert Decimal(fired["level_price"]) == anchor.stop_price
    assert len(_sells(exchange)) == 1


# ---------------------------------------------------------------------------
# T-25  trailing mark, fresh vs resumed
# ---------------------------------------------------------------------------


async def _trailing_run(exchange: FakeCCXTExchange) -> tuple[LiveStack, list[Fill], list[Order]]:
    _wavy(exchange)
    stack = await _live(exchange, {"trailing_stop_pct": 0.20})
    await _step(stack, 101)  # BUY at 101
    for i in range(130):
        # one spike early on (inside the part of the history the warmup window will miss)
        await _step(stack, 108 if i == 20 else 100 + (i % 2) * 2)
    fills, orders = _persisted(stack)
    return stack, fills, orders


async def test_t25_trailing_mark_fresh_equals_resumed_including_the_gap_fetch(
    exchange: FakeCCXTExchange,
) -> None:
    stack, fills, orders = await _trailing_run(exchange)
    fresh_peak = stack.engine._trailing_stop.peak_prices[SYMBOL]  # type: ignore[union-attr]
    assert fresh_peak == Decimal("108")
    exchange.ohlcv_calls.clear()
    resumed = await _resume(exchange, {"trailing_stop_pct": 0.20}, fills, orders)
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] == fresh_peak  # type: ignore[union-attr]
    assert [c for c in exchange.ohlcv_calls if c[2] is not None], "the gap must be fetched once"


async def test_t25_gap_fetch_failure_seeds_from_the_available_closes_and_says_so(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stack, fills, orders = await _trailing_run(exchange)
    _fail_since_fetches(exchange, monkeypatch)
    with capture_logs() as logs:
        resumed = await _resume(exchange, {"trailing_stop_pct": 0.20}, fills, orders)
    assert any(e["event"] == "trailing_stop.seed_partial" for e in logs)
    peak = resumed.engine._trailing_stop.peak_prices[SYMBOL]  # type: ignore[union-attr]
    assert Decimal("101") <= peak < Decimal("108")  # seeded from the closes the window had


# ---------------------------------------------------------------------------
# T-26  end-to-end ATR immutability
# ---------------------------------------------------------------------------


async def test_t26_widening_ranges_never_move_the_stop_and_a_close_between_exits(
    exchange: FakeCCXTExchange,
) -> None:
    _wavy(exchange)
    stack = await _live(exchange, ATR_SL_ONLY)
    await _step(stack, 101)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    anchored = bracket.get_anchor(SYMBOL)
    assert anchored is not None and anchored.stop_price is not None
    stop = anchored.stop_price
    assert stop > Decimal("97")

    # ranges 10, 12, 14, ... : the CURRENT ATR rises well above ATR0, so a recomputed
    # stop (HEAD) would slide far below the anchored one -- closes stay above the stop
    for close in (111, 101, 114, 101, 116, 101):
        await _step(stack, close)
        assert bracket.get_anchor(SYMBOL) == anchored
    current_atr = stack.engine._compute_atr_for_symbol(
        SYMBOL, {SYMBOL: stack.engine._bar_windows[SYMBOL]}
    )
    assert current_atr is not None and current_atr > anchored.atr * 2  # type: ignore[operator]
    drifted_stop = anchored.entry_basis - Decimal("1.5") * current_atr
    assert drifted_stop < stop - 1  # HEAD's level was far lower
    assert _sells(exchange) == []

    close_between = (stop + Decimal("0.5")).quantize(Decimal("0.01")) - Decimal("1")
    assert drifted_stop < close_between <= stop
    await _step(stack, close_between)
    assert len(_sells(exchange)) == 1  # exits on the anchored level (HEAD would hold)


# ---------------------------------------------------------------------------
# T-27  an opening-order fill arriving late
# ---------------------------------------------------------------------------


async def test_t27_late_opening_fill_via_check_resting_orders_anchors_on_the_signal_bar(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The BUY fills on the exchange at bar N but its fill is not routed until bar N+1
    (``check_resting_orders`` picks it up).  The anchor bar is still the SIGNAL bar N
    (the rule is on ``min(created_at, executed_at)``), so ATR0 is the ATR of the window
    ending at N -- not the (much larger) ATR after the violent bar N+1."""
    _wavy(exchange)
    stack = await _live(exchange, ATR_SL_ONLY)  # no TP: the violent bar must not exit
    real_route = stack.engine._route_exit_fills

    async def _not_routed(orders: Any, bars: Any, signal: Any) -> tuple[int, Decimal]:
        return 0, Decimal(0)

    real_resting = stack.engine._check_resting_orders

    async def _no_resting(bars: Any) -> None:
        return None

    monkeypatch.setattr(stack.engine, "_route_exit_fills", _not_routed)
    monkeypatch.setattr(stack.engine, "_check_resting_orders", _no_resting)
    await _step(stack, 101)  # signal bar N: order filled at the exchange, fill NOT routed
    monkeypatch.setattr(stack.engine, "_route_exit_fills", real_route)
    monkeypatch.setattr(stack.engine, "_check_resting_orders", real_resting)
    bracket = stack.engine._bracket_exit
    assert bracket is not None and bracket.get_anchor(SYMBOL) is None
    assert exchange.order_log and exchange.order_log[0]["side"] == "buy"
    signal_bar_ts = _ts(_last_ts_ms(exchange))
    window_at_n = list(stack.engine._bar_windows[SYMBOL])

    await _step(stack, 130)  # bar N+1 (violent): the late fill is routed here
    anchor = bracket.get_anchor(SYMBOL)
    assert anchor is not None and anchor.source == "entry" and anchor.final
    assert anchor.anchor_bar_ts == signal_bar_ts
    from trading.strategy_engine import atr_for_bars

    assert anchor.atr == atr_for_bars(window_at_n[-85:], 14)  # ATR(N), frozen at the signal bar
    window_at_n1 = stack.engine._bar_windows[SYMBOL]
    current = atr_for_bars(window_at_n1[-85:], 14)
    assert current is not None and anchor.atr is not None and current > anchor.atr * 2


# ---------------------------------------------------------------------------
# T-28 / T-29  provisional anchors
# ---------------------------------------------------------------------------


async def test_t28_boundary_race_provisional_then_finalised_equals_the_resume_anchor(
    exchange: FakeCCXTExchange,
) -> None:
    _wavy(exchange)
    stack = await _live(exchange, ATR_CFG)
    exchange.push_bar(SYMBOL, Decimal("101"), timeframe=TF)
    _set_now_after_close(exchange, bars_after=1)  # the order is built AFTER the next boundary
    await stack.engine._poll_and_process()
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    prov = bracket.get_anchor(SYMBOL)
    assert prov is not None and prov.source == "entry_provisional" and not prov.final
    assert prov.stop_price is not None  # protected at once

    with capture_logs() as logs:
        await _step(stack, 101)  # bar N+1 -- the rule bar -- arrives and is processed
    final = bracket.get_anchor(SYMBOL)
    assert final is not None and final.final and final.source == "entry"
    assert final.anchor_bar_ts == _ts(_last_ts_ms(exchange))
    assert [e["event"] for e in logs].count("bracket_exit.anchored") == 1  # finalised once

    fills, orders = _persisted(stack)
    resumed = await _resume(exchange, ATR_CFG, fills, orders)
    again = resumed.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert again is not None
    assert (again.atr, again.stop_price, again.take_profit_price) == (
        final.atr,
        final.stop_price,
        final.take_profit_price,
    )
    # a third pass never touches a final anchor
    await _step(stack, 100)
    assert bracket.get_anchor(SYMBOL) == final


async def _gapped_entry(exchange: FakeCCXTExchange) -> LiveStack:
    _wavy(exchange)
    stack = await _live(exchange, ATR_CFG)
    exchange.push_bar(SYMBOL, Decimal("102"), timeframe=TF)  # bar N: the poll is missed
    await _step(stack, 101)  # bar N+1: processed -> BUY; the window has a hole at N
    return stack


async def test_t29_gapped_window_provisional_then_bounded_fetch_equals_the_resume_anchor(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _gapped_entry(exchange)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    prov = bracket.get_anchor(SYMBOL)
    assert prov is not None and prov.source == "entry_provisional" and not prov.final
    exchange.ohlcv_calls.clear()
    await _step(stack, 100)
    final = bracket.get_anchor(SYMBOL)
    assert final is not None and final.final and final.source == "entry"
    assert len([c for c in exchange.ohlcv_calls if c[2] is not None]) == 1  # ONE bounded fetch
    fills, orders = _persisted(stack)
    resumed = await _resume(exchange, ATR_CFG, fills, orders)
    again = resumed.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert again is not None
    assert (again.atr, again.stop_price, again.take_profit_price) == (
        final.atr,
        final.stop_price,
        final.take_profit_price,
    )


async def test_t29_fetch_failure_keeps_the_levels_as_entry_unverified(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP1.3b-a FX-03: three bounded attempts (one per pass), THEN the interim levels are kept
    as `entry_unverified`; nothing is fetched afterwards."""
    stack = await _gapped_entry(exchange)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    prov = bracket.get_anchor(SYMBOL)
    assert prov is not None
    _fail_since_fetches(exchange, monkeypatch)
    for _ in range(2):  # attempts 1 and 2: still pending, still the interim levels
        await _step(stack, 100)
        still = bracket.get_anchor(SYMBOL)
        assert still is not None and not still.final and still.source == "entry_provisional"
    with capture_logs() as logs:
        await _step(stack, 100)  # attempt 3 -> verification over
    final = bracket.get_anchor(SYMBOL)
    assert final is not None and final.final and final.source == "entry_unverified"
    assert (final.stop_price, final.take_profit_price) == (prov.stop_price, prov.take_profit_price)
    err = [e for e in logs if e["event"] == "bracket_exit.entry_unverified"]
    assert err and err[0]["log_level"] == "error"
    calls_before = len(exchange.ohlcv_calls)
    await _step(stack, 100)  # exactly 3 attempts: no further finalisation attempt
    assert [c for c in exchange.ohlcv_calls[calls_before:] if c[2] is not None] == []


# ---------------------------------------------------------------------------
# T-30  backtests
# ---------------------------------------------------------------------------

BT_SYMBOL = "BTC/USDT"


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def _backtest(
    closes: list[int],
    script: dict[int, str],
    *,
    bracket: dict[str, Any] | None = None,
    trailing: float | None = None,
    pyramiding: bool = False,
    targets: dict[int, str] | None = None,
    cost_profile: CostProfile | None = None,
) -> dict[str, Any]:
    strategy = ScriptedSequenceStrategy(
        "wp13b-bt", {"script": script, "target_notional": "3000", "targets": targets or {}}
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[BT_SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=bracket,
        trailing_stop_pct=trailing,
        allow_pyramiding=pyramiding,
        cost_profile=cost_profile,
    )
    bars = scripted_path(
        closes,
        first_open=datetime(2026, 3, 10, 10, 0, tzinfo=UTC),
        symbol=BT_SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
    )
    result = await runner.run({BT_SYMBOL: bars})
    assert runner.last_execution_engine is not None
    return {
        "fills": [
            (f.side.value, str(f.quantity), str(f.price))
            for f in runner.last_execution_engine.get_all_fills()
        ],
        "curve": [str(p.equity) for p in result.equity_curve],
        "final_equity": str(result.final_equity),
    }


#: entry at 100, +5% TP leg, re-entry at 100, -3% SL leg (fixed bracket 2% / 4% would differ)
_FIXED_CLOSES = [100, 101, 102, 104, 105, 100, 99, 98, 97, 96, 100, 101]
_FIXED_BRACKET = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.02,
    "bracket_take_profit_pct": 0.04,
}
_FIXED_SCRIPT = {0: "buy", 5: "buy"}

# Golden digest of the fixed-bracket reference backtest (no trailing, no adds), taken at
# HEAD bf2b27f BEFORE any WP1.3b change.  If it fails the default fixed path changed
# (AC-13B-08).  WP2.1: the constant is UNCHANGED; the test now pins the pre-WP2.1 backtest
# costs (``legacy_backtest_v0`` = maker 10 / taker 15 / slippage 5 bps) explicitly, because
# the default profile moved to coinbase_advanced_1 (60/120 + 10 bps).
GOLDEN_FIXED_DIGEST = "94015f75e15178a7d197488360f8cc72a0f6243cbf8dfde63c88fd46be3d3a5c"


async def test_t30_fixed_bracket_backtest_is_byte_identical_to_head() -> None:
    out = await _backtest(
        _FIXED_CLOSES,
        _FIXED_SCRIPT,
        bracket=_FIXED_BRACKET,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
    )
    assert len(out["fills"]) >= 4  # two round trips actually happened
    assert _digest(out) == GOLDEN_FIXED_DIGEST


async def test_t30_atr_backtest_is_deterministic_run_to_run() -> None:
    closes = [100 + (i % 7) * 2 - (i % 3) for i in range(80)]
    bracket = {
        "bracket_mode": "atr",
        "bracket_atr_sl_multiplier": 1.5,
        "bracket_atr_tp_multiplier": 3.0,
    }
    first = await _backtest(closes, {0: "buy", 12: "buy", 30: "buy"}, bracket=bracket)
    second = await _backtest(closes, {0: "buy", 12: "buy", 30: "buy"}, bracket=bracket)
    assert first == second
    assert len(first["fills"]) >= 2


async def test_t30_backtest_add_keeps_levels_after_a_dip_that_does_not_stop_out() -> None:
    """Entry 100 (stop 98.0 on E ~100.2); add at 99 lowers the average to ~99.7.
    HEAD would then trail the stop down to ~97.7; anchored it stays at ~98.2, so a close
    of 98.0 exits only with anchoring (red at HEAD)."""
    closes = [100, 99, 99, 99, 98, 98, 98]
    out = await _backtest(
        closes + [98] * 3,
        {0: "buy", 1: "buy"},
        bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.02},
        pyramiding=True,
        targets={0: "700", 1: "700"},
    )
    sides = [f[0] for f in out["fills"]]
    assert sides.count("buy") == 2
    assert "sell" in sides, "the anchored stop of the FIRST entry must fire at a close of 98"


# ---------------------------------------------------------------------------
# T-32  reconcile_required does not silence the bar-close SL
# ---------------------------------------------------------------------------


async def test_t32_reconcile_required_symbol_still_exits_at_the_bar_close(
    exchange: FakeCCXTExchange,
) -> None:
    """ARB-13b-01 / invariant I4: ``reconcile_required`` blocks BUYs only; a flagged
    symbol must never leave a protective exit unable to fire (capped by WP1.11a)."""
    stack = await _entered_fixed(exchange)
    stack.execution._flag_reconcile(SYMBOL, "wp13b_test_flag")
    assert SYMBOL in stack.execution.reconcile_required
    await _step(stack, 90)
    sells = _sells(exchange)
    assert len(sells) == 1 and sells[0]["amount"] <= exchange.balance_of("BTC") + sells[0]["amount"]
    assert stack.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# AC-13B-18  log hygiene of the new fetch paths
# ---------------------------------------------------------------------------

_LEAKY_ERROR_TEXT = (
    "https://api.exchange.example/v3?api_key=" + "SECRET-KEY-123" + "&sign=" + "SECRET-SIG-456"
)
_NEW_SITES = ("engine.exit_", "engine.anchor_", "bracket_exit.", "trailing_stop.", "entry.")


def _new_site_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Log records of the WP1.3b-a sites.  (The market-data service's own pre-existing
    ``network_error_retrying`` site logs ``error=str(exc)``: that is CF-36, not 1.3b.)"""
    return [e for e in logs if str(e["event"]).startswith(_NEW_SITES)]


async def test_ac13b18_new_fetch_failure_logs_carry_the_error_type_only(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume ATR fetch and the trailing gap fetch fail with an exception whose text
    embeds a URL + credentials; none of it may appear in a WP1.3b-a log record."""
    cfg = {**ATR_CFG, "trailing_stop_pct": 0.2}
    _live_stack, fills, orders = await _old_lot(exchange, cfg)
    _fail_since_fetches(exchange, monkeypatch, _LEAKY_ERROR_TEXT)
    with capture_logs() as logs:
        await _resume(exchange, cfg, fills, orders)
    mine = _new_site_logs(logs)
    events = {e["event"] for e in mine}
    assert {"engine.exit_window_fetch_failed", "engine.exit_gap_fetch_failed"} <= events
    assert all("error_type" in e for e in mine if e["event"].endswith("fetch_failed"))
    rendered = json.dumps(mine, default=str)
    assert "SECRET" not in rendered and "api.exchange.example" not in rendered


async def test_ac13b18_finalisation_fetch_failure_logs_no_exception_text(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = await _gapped_entry(exchange)
    _fail_since_fetches(exchange, monkeypatch, _LEAKY_ERROR_TEXT)
    with capture_logs() as logs:
        await _step(stack, 100)
    mine = _new_site_logs(logs)
    assert any(e["event"] == "engine.exit_window_fetch_failed" for e in mine)
    rendered = json.dumps(mine, default=str)
    assert "SECRET" not in rendered and "api.exchange.example" not in rendered


# ---------------------------------------------------------------------------
# T-33  the orchestrator primes the engine before start() at both resume sites
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Stands in for ``StrategyEngine``: records the order of ``prime``/``start``."""

    instances: ClassVar[list[_RecordingEngine]] = []

    def __init__(self, *args: object, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._stop_event = asyncio.Event()
        self.auto_stop_reason: str | None = None
        self.risk_manager = MagicMock()
        self.stop = AsyncMock()
        self.run_live_loop = AsyncMock(side_effect=asyncio.CancelledError())
        _RecordingEngine.instances.append(self)

    def prime_resume_exit_state(self, *args: Any) -> None:
        self.calls.append(("prime", args))

    async def start(self, *args: Any) -> None:
        self.calls.append(("start", args))


def _orch_settings() -> MagicMock:
    settings = MagicMock()
    settings.exchange_api_key = None
    settings.exchange_api_secret = None
    settings.exchange_api_passphrase = None
    settings.max_run_duration_hours = 24.0
    settings.live_exit_check_interval_s = 300  # WP1.3b-b: a real Settings value, not a MagicMock
    settings.exchange_id = "coinbase"
    return settings


async def _drive_orchestrator(mode: str, resume: Any) -> _RecordingEngine:
    import contextlib
    from unittest.mock import patch

    from api.services import run_orchestrator as orch
    from tests.integration.test_wp15b_resume_seed_wiring import _session_factory

    _RecordingEngine.instances.clear()
    common: dict[str, Any] = {
        "run_id_str": str(uuid4()),
        "strategy_cls": MagicMock,
        "strategy_name": "grid_trading",
        "strategy_params": {},
        "symbols": [SYMBOL],
        "timeframe": TimeFrame.ONE_HOUR,
        "initial_capital": "10000",
        "resume": resume,
        "cost_profile": DEFAULT_COST_PROFILE,  # WP2.1: required keyword
    }
    patches: list[Any] = [
        patch("api.db.session.get_session_factory", _session_factory()),
        patch("api.config.get_settings", return_value=_orch_settings()),
        patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        patch("trading.strategy_engine.StrategyEngine", _RecordingEngine),
        patch.object(orch, "make_circuit_breaker_alert_listener", MagicMock()),
        patch.object(orch, "record_audit_event", AsyncMock()),
        patch.object(orch, "_flush_incremental", AsyncMock()),
        patch.object(orch, "_incremental_flush_loop", AsyncMock()),
        patch.object(orch, "_auto_stop_after", AsyncMock()),
        patch.object(orch, "_SHUTDOWN_REQUESTED", False),
    ]
    if mode == "paper":
        patches.append(patch.object(orch, "_auto_retry_paper_run", AsyncMock()))
        runner = orch.run_paper_engine
    else:
        patches.append(patch.object(orch, "build_live_ccxt_exchange", MagicMock()))
        patches.append(patch("trading.engines.live.LiveExecutionEngine", MagicMock()))
        runner = orch.run_live_engine
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with pytest.raises(asyncio.CancelledError):
            await runner(**common)
    assert len(_RecordingEngine.instances) == 1
    return _RecordingEngine.instances[0]


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_t33_resume_sites_prime_the_engine_before_start(mode: str) -> None:
    from trading.recovery import ResumeSnapshot

    oid = uuid4()
    fill = Fill(
        order_id=oid,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.6"),
        fee_currency="EUR",
    )
    order = Order(
        order_id=oid,
        client_order_id="r-1",
        run_id="r",
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
    )
    snapshot = ResumeSnapshot(
        run_id=str(uuid4()),
        initial_cash=Decimal("10000"),
        fills=[fill],
        orders=[order],
        peak_equity_hint=None,
        max_bar_index=-1,
    )
    engine = await _drive_orchestrator(mode, snapshot)
    assert [name for name, _ in engine.calls] == ["prime", "start"]
    prime_args = engine.calls[0][1]
    assert list(prime_args[0]) == [fill] and list(prime_args[1]) == [order]


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_t33_a_fresh_run_is_not_primed(mode: str) -> None:
    engine = await _drive_orchestrator(mode, None)
    assert [name for name, _ in engine.calls] == ["start"]
