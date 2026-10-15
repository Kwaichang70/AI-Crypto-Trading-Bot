"""
tests/integration/test_wp13b_intrabar_harness.py
--------------------------------------------------
WP1.3b-b (live intra-bar exit check) through the REAL live stack --
``StrategyEngine`` + ``LiveExecutionEngine`` + ``PortfolioAccounting`` +
``CCXTMarketDataService`` over ``FakeCCXTExchange``.  WP13B-T-40, T-42, T-43,
T-47..T-64, T-66, T-68 and the 1.3b-a carry-forwards routed here (CF-13b-20/22/23).

Clock model.  The intra-bar task never sleeps for real: ``install_intrabar_clock``
swaps ``engine._intrabar_wait`` for a VIRTUAL clock (300 s interval, 10 s confirm
cost no wall time) and lets a test schedule exchange mutations at virtual times.
``asyncio.sleep`` is patched instant (the 2 s post-submit wait of ``process_signal``).

    T-40  flat run: zero ticker calls; held run: calls per tick = held (+ candidates)
    T-42  CF-04 with an exit: the fill is the EXCHANGE's, the mark is the last closed close
    T-43  CF-30: no breaker / cooldown / strategy / excursion / bar-count activity
    T-47  SL broken mid-bar -> SELL within one interval + 10 s                (red)
    T-48  trailing broken mid-bar -> SELL within one interval + 10 s           (red)
    T-49  a single bad tick never fires
    T-50  invalid / missing readings never fire
    T-51  last vs bid
    T-52  ticker outage: degraded / recovered; the bar-close SL still fires
    T-53  429: the tick is abandoned, the interval doubles (capped), restored
    T-54  the lock is not held across the confirm wait
    T-55  flatten() holds the lock
    T-56  an OPEN / unknown intra-bar SELL then a bar close: one SELL
    T-57  partial fill + external balance
    T-58  accept-then-timeout: no second create_order
    T-59  exit not submitted: pending cleared, retried                         (red)
    T-60  kill switch / protective / HALT / DAILY_LIMIT: the SL fires
    T-61  TP is bar-close only
    T-62  paper / no key: no task, no ticks
    T-63  reconcile_required symbol fires
    T-64  a breach that recovered before the tick: no exit
    T-66  resume during a breach
    T-68  log hygiene
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import ccxt
import pytest
from structlog.testing import capture_logs

from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    IntrabarClock,
    LiveStack,
    build_live_stack,
    build_resumed_live_stack,
    install_intrabar_clock,
    patch_exchange_factory,
    run_intrabar_loop,
    run_intrabar_tick,
    start_and_warmup,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading import strategy_engine as se
from trading.models import Fill
from trading.risk import live_risk_parameters
from trading.safety import CircuitBreakerResponse

SYMBOL = "BTC/EUR"
TF = "1h"
TF_MS = 3_600_000
RUN_ID = "wp13bb-run"
INITIAL = Decimal("1000")
INTERVAL = 300
CONFIRM = 10.0

FIXED: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.05,
    "exit_check_interval_s": INTERVAL,
}
FIXED_SL_TP: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.05,
    "bracket_take_profit_pct": 0.10,
    "exit_check_interval_s": INTERVAL,
}
ATR_SL: dict[str, object] = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_period": 14,
    "exit_check_interval_s": INTERVAL,
}
TRAIL: dict[str, object] = {"trailing_stop_pct": 0.10, "exit_check_interval_s": INTERVAL}


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture(autouse=True)
def _short_lock_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tick waits 5 s for ``_cycle_lock`` in production; tests that make it busy must not."""
    monkeypatch.setattr(se, "_INTRABAR_LOCK_TIMEOUT_S", 0.05, raising=False)


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
    for i in range(n):
        ex.push_bar(SYMBOL, Decimal(100 + (i % 2) * amp), timeframe=TF)


def _last_ts_ms(ex: FakeCCXTExchange) -> int:
    return int(ex._closed_bars[SYMBOL][-1][0])


def _set_now_after_close(ex: FakeCCXTExchange, latency_ms: int = 2000) -> None:
    ex.set_now_ms(_last_ts_ms(ex) + TF_MS + latency_ms)


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
    risk_params: Any = None,
    start: bool = True,
    notional: str = "100",
) -> LiveStack:
    script = {} if buy_at is None else {buy_at: "buy"}
    strategy = ScriptedSequenceStrategy("wp13bb", {"script": script, "target_notional": notional})
    stack = await build_live_stack(
        exchange=ex,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        engine_config=cfg,
        risk_params=risk_params,
    )
    if start:
        await start_and_warmup(stack, RUN_ID)
    return stack


async def _entered(
    ex: FakeCCXTExchange,
    cfg: dict[str, object] = FIXED,
    *,
    risk_params: Any = None,
    notional: str = "100",
) -> LiveStack:
    """Flat history at 100, BUY on the first bar (fills at 100, all-in basis 100.6)."""
    ex.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(ex, cfg, risk_params=risk_params, notional=notional)
    await _step(stack, 100)
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat
    return stack


def _sells(ex: FakeCCXTExchange) -> list[dict[str, Any]]:
    return [o for o in ex.order_log if o["side"] == "sell"]


def _stop(stack: LiveStack) -> Decimal:
    anchor = stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert anchor is not None and anchor.stop_price is not None
    return anchor.stop_price


def _flat(stack: LiveStack) -> bool:
    position = stack.portfolio.get_position(SYMBOL)
    return position is None or position.is_flat


def _below(level: Decimal, by: str = "1") -> Decimal:
    return (level - Decimal(by)).quantize(Decimal("0.01"))


def _spy_signals(stack: LiveStack) -> list[Any]:
    """Record every signal that reaches the live execution engine."""
    seen: list[Any] = []
    real = stack.execution.process_signal

    async def _spy(signal: Any) -> Any:
        seen.append(signal)
        return await real(signal)

    stack.execution.process_signal = _spy  # type: ignore[method-assign]
    return seen


def _spy_metrics(stack: LiveStack) -> list[tuple[str, dict[str, str]]]:
    seen: list[tuple[str, dict[str, str]]] = []
    real = stack.engine._metric

    def _spy(name: str, **labels: str) -> None:
        seen.append((name, dict(labels)))
        real(name, **labels)

    stack.engine._metric = _spy  # type: ignore[method-assign]
    return seen


def _events(logs: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == name]


def _risk(stack: LiveStack) -> str:
    """The typed risk snapshot (peak, latch, day-start basis, NAV) minus its wall-clock stamp."""
    snapshot = stack.portfolio.risk_snapshot()
    return repr(dataclasses.replace(snapshot, as_of=None))


def _state(stack: LiveStack) -> dict[str, Any]:
    """Everything a tick must not change when it submits nothing."""
    bracket = stack.engine._bracket_exit
    trailing = stack.engine._trailing_stop
    position = stack.portfolio.get_position(SYMBOL)
    return {
        "anchor": bracket.get_anchor(SYMBOL) if bracket is not None else None,
        "bracket_pending": bracket.pending_exit_symbols if bracket is not None else None,
        "peaks": trailing.peak_prices if trailing is not None else None,
        "trailing_pending": trailing.pending_stop_symbols if trailing is not None else None,
        "position": position.model_copy() if position is not None else None,
        "curve": len(stack.portfolio.get_equity_curve()),
        "summary": repr(stack.portfolio.get_summary()),
        "bar_count": stack.engine.bar_count,
        "n_sells": len(_sells(stack.exchange)),
    }


async def _loop(
    stack: LiveStack,
    *,
    horizon: float,
    schedule: list[tuple[float, Callable[[], Any]]] | None = None,
    timeout: float = 20.0,
) -> IntrabarClock:
    """Run the REAL intra-bar task body to ``horizon`` virtual seconds."""
    clock = install_intrabar_clock(stack, horizon=horizon)
    for at, action in schedule or []:
        clock.at(at, action)
    await asyncio.wait_for(run_intrabar_loop(stack), timeout=timeout)
    return clock


# ---------------------------------------------------------------------------
# T-47 / T-48  the headline: broken mid-bar -> SELL within one interval + confirm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("breach_at", [100.0, 61.0, 359.0])
@pytest.mark.parametrize("variant", ["fixed", "atr"])
async def test_t47_sl_broken_mid_bar_sells_within_one_interval_plus_the_confirm(
    exchange: FakeCCXTExchange, variant: str, breach_at: float
) -> None:
    """BUY at bar N; at virtual time ``breach_at`` the BID drops below the anchored SL
    (no new bar closes).  The REAL ``_intrabar_exit_loop`` reads it at its next tick,
    confirms ~10 s later and sells ONCE: SELL at <= breach + I + 10 s + 1 s."""
    if variant == "fixed":
        stack = await _entered(exchange, FIXED)
    else:
        _wavy(exchange)
        stack = await _live(exchange, ATR_SL)
        await _step(stack, 101)
    assert not _flat(stack)
    signals = _spy_signals(stack)
    stop = _stop(stack)
    bid = _below(stop, "2")

    with capture_logs() as logs:
        clock = await _loop(
            stack,
            horizon=900.0,
            schedule=[(breach_at, lambda: exchange.set_ticker(SYMBOL, bid=bid, last=bid))],
        )

    sells = _sells(exchange)
    assert len(sells) == 1
    assert len(clock.create_order_times) == 1
    sold_at = clock.create_order_times[0]
    assert breach_at <= sold_at <= breach_at + INTERVAL + CONFIRM + 1.0
    assert _flat(stack)
    # the cadence the spec promises: first tick at +60 s, then every 300 s
    assert clock.waits[0] == 60.0 and clock.waits[1] == 300.0
    # classification + provenance
    exits = [s for s in signals if s.metadata.get("trigger") == "bracket_stop_loss"]
    assert len(exits) == 1
    meta = exits[0].metadata
    assert meta["exit_path"] == "intrabar" and meta["trigger_source"] == "bid"
    assert meta["trigger_price"] == str(bid) and meta["confirm_bids"] == [str(bid), str(bid)]
    assert meta["stop_loss"] is True and "take_profit" not in meta
    trade = stack.portfolio.get_trade_history()[-1]
    assert trade.exit_reason == "stop_loss"
    assert _events(logs, "engine.trade_recorded")[0]["exit_path"] == "intrabar"


@pytest.mark.parametrize("breach_at", [100.0, 61.0])
async def test_t48_trailing_broken_mid_bar_sells_within_one_interval_plus_the_confirm(
    exchange: FakeCCXTExchange, breach_at: float
) -> None:
    stack = await _entered(exchange, TRAIL)
    await _step(stack, 110)  # the mark ratchets at the bar close: trail = 110 * 0.9 = 99
    trailing = stack.engine._trailing_stop
    assert trailing is not None and trailing.peak_prices[SYMBOL] == Decimal("110")
    signals = _spy_signals(stack)
    bid = Decimal("97.5")

    clock = await _loop(
        stack,
        horizon=900.0,
        schedule=[(breach_at, lambda: exchange.set_ticker(SYMBOL, bid=bid, last=bid))],
    )
    assert len(_sells(exchange)) == 1
    sold_at = clock.create_order_times[0]
    assert breach_at <= sold_at <= breach_at + INTERVAL + CONFIRM + 1.0
    assert _flat(stack)
    meta = signals[-1].metadata
    assert meta["trigger"] == "trailing_stop" and meta["exit_path"] == "intrabar"
    assert meta["trigger_source"] == "bid" and Decimal(meta["peak_price"]) == Decimal("110")
    assert stack.portfolio.get_trade_history()[-1].exit_reason == "trailing_stop"


# ---------------------------------------------------------------------------
# T-49 / T-50 / T-51  never fire on one reading, on missing data, or on `last`
# ---------------------------------------------------------------------------


async def test_t49_a_single_bad_tick_never_fires_and_changes_nothing(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    stop = _stop(stack)
    low = _below(stop, "2")
    seen = _spy_metrics(stack)
    before = _state(stack)
    clock = install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    # between reading 1 and reading 2 the book recovers
    clock.at(5.0, lambda: exchange.set_ticker(SYMBOL, bid=Decimal("100"), last=Decimal("100")))

    with capture_logs() as logs:
        await run_intrabar_tick(stack)

    assert _sells(exchange) == []
    assert _state(stack) == before
    assert ("intrabar_breach_unconfirmed_total", {}) in seen
    assert _events(logs, "engine.intrabar_breach_unconfirmed")
    assert clock.waits == [CONFIRM]


@pytest.mark.parametrize(
    "bad_bid",
    [None, float("nan"), 0, -5, "", float("inf")],
    ids=["none", "nan", "zero", "negative", "empty_string", "inf"],
)
async def test_t50_an_invalid_first_reading_never_fires(
    exchange: FakeCCXTExchange, bad_bid: object
) -> None:
    stack = await _entered(exchange)
    before = _state(stack)
    install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=bad_bid, ask=Decimal("100"), last=Decimal("10"))
    await run_intrabar_tick(stack)
    assert _sells(exchange) == []
    assert _state(stack) == before


async def test_t50_a_crossed_book_is_invalid(exchange: FakeCCXTExchange) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=low, ask=low - Decimal("1"), last=low)  # ask < bid
    await run_intrabar_tick(stack)
    assert _sells(exchange) == []


@pytest.mark.parametrize("second", ["error", "none", "nan", "crossed"])
async def test_t50_an_invalid_confirming_reading_never_confirms_and_never_fires(
    exchange: FakeCCXTExchange, second: str
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    seen = _spy_metrics(stack)
    before = _state(stack)
    clock = install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=low, last=low)

    def _break_second() -> None:
        if second == "error":
            exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"))
        elif second == "none":
            exchange.set_ticker(SYMBOL, bid=None, ask=low, last=low)
        elif second == "nan":
            exchange.set_ticker(SYMBOL, bid=float("nan"), ask=low, last=low)
        else:
            exchange.set_ticker(SYMBOL, bid=low, ask=low - Decimal("1"), last=low)

    clock.at(5.0, _break_second)
    await run_intrabar_tick(stack)
    assert _sells(exchange) == []
    assert _state(stack) == before
    assert any(name == "intrabar_confirm_failed_total" for name, _ in seen)


async def test_t51_last_at_or_below_the_sl_with_the_bid_above_does_not_fire(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "5")
    install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=Decimal("100"), ask=Decimal("100.1"), last=low)
    await run_intrabar_tick(stack)
    assert _sells(exchange) == [] and not _flat(stack)


async def test_t51_the_bid_at_or_below_the_sl_fires_even_when_last_is_above(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=low, ask=Decimal("101"), last=Decimal("100"))
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1 and _flat(stack)


async def test_t64_a_breach_that_recovered_before_the_tick_is_not_acted_on(
    exchange: FakeCCXTExchange,
) -> None:
    """No look-back: the book dipped below the SL at t=100 and recovered at t=200; the
    tick at t=360 sees a normal book."""
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    clock = await _loop(
        stack,
        horizon=700.0,
        schedule=[
            (100.0, lambda: exchange.set_ticker(SYMBOL, bid=low, last=low)),
            (200.0, lambda: exchange.set_ticker(SYMBOL, bid=Decimal("100"), last=Decimal("100"))),
        ],
    )
    assert _sells(exchange) == [] and not _flat(stack)
    assert CONFIRM not in clock.waits  # never even a candidate


# ---------------------------------------------------------------------------
# T-40  rate budget
# ---------------------------------------------------------------------------


async def test_t40_a_flat_run_makes_zero_ticker_calls(exchange: FakeCCXTExchange) -> None:
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(exchange, FIXED, buy_at=None)
    install_intrabar_clock(stack)
    before = exchange.ticker_calls
    for _ in range(3):
        await run_intrabar_tick(stack)
    assert exchange.ticker_calls == before


async def test_t40_a_held_run_costs_one_reading_per_symbol_plus_one_per_candidate(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    install_intrabar_clock(stack)
    start = exchange.ticker_calls
    await run_intrabar_tick(stack)  # held, no candidate: exactly the first reading
    assert exchange.ticker_calls - start == 1
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.queue_order_error(SYMBOL, ccxt.InvalidOrder("keep the position for the count"))
    mid = exchange.ticker_calls
    await run_intrabar_tick(stack)  # candidate: first + confirming reading (+ process_signal's own)
    readings = list(exchange.ticker_call_log[mid:])
    assert len(readings) >= 2


# ---------------------------------------------------------------------------
# T-41 / T-42 / T-43  CF-04 and CF-30
# ---------------------------------------------------------------------------


def _install_cf_spies(stack: LiveStack) -> dict[str, int]:
    counts = {"update_market_prices": 0, "evaluate": 0, "tick_cooldown": 0, "exc_on_bar": 0}
    portfolio, engine = stack.portfolio, stack.engine
    real_ump = portfolio.update_market_prices

    def _ump(prices: dict[str, Decimal]) -> None:
        counts["update_market_prices"] += 1
        real_ump(prices)

    portfolio.update_market_prices = _ump  # type: ignore[method-assign]
    if stack.circuit_breaker is not None:
        real_eval = stack.circuit_breaker.evaluate

        def _eval(snapshot: Any) -> Any:
            counts["evaluate"] += 1
            return real_eval(snapshot)

        stack.circuit_breaker.evaluate = _eval  # type: ignore[method-assign]
    real_cd = stack.risk_manager.tick_cooldown

    def _cd() -> None:
        counts["tick_cooldown"] += 1
        real_cd()

    stack.risk_manager.tick_cooldown = _cd  # type: ignore[method-assign]
    real_exc = engine._excursion_tracker.on_bar

    def _exc(**kwargs: Any) -> None:
        counts["exc_on_bar"] += 1
        real_exc(**kwargs)

    engine._excursion_tracker.on_bar = _exc  # type: ignore[method-assign]
    return counts


async def test_t41_ticks_without_an_exit_leave_portfolio_state_untouched(
    exchange: FakeCCXTExchange,
) -> None:
    """CF-04 (WP15-T-19 style): single-reading breach, unconfirmed breach and a ticker outage
    change none of NAV, peak, day-start, marks, equity curve, cash, positions, anchors, HWM,
    pending guards; ``update_market_prices`` is never called."""
    stack = await _entered(exchange)
    await _step(stack, 101)  # a mark and a curve point exist
    counts = _install_cf_spies(stack)
    before = _state(stack)
    risk_before = _risk(stack)
    low = _below(_stop(stack), "2")
    clock = install_intrabar_clock(stack)

    exchange.set_ticker(SYMBOL, bid=Decimal("101"), last=Decimal("101"))
    await run_intrabar_tick(stack)  # nothing near the SL
    exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"))
    await run_intrabar_tick(stack)  # outage
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    clock.at(5.0, lambda: exchange.set_ticker(SYMBOL, bid=Decimal("101"), last=Decimal("101")))
    await run_intrabar_tick(stack)  # unconfirmed breach

    assert _state(stack) == before
    assert _risk(stack) == risk_before
    assert counts["update_market_prices"] == 0


async def test_t42_with_an_exit_the_fill_is_the_exchanges_and_the_mark_is_the_last_closed_close(
    exchange: FakeCCXTExchange,
) -> None:
    """CF-04 / SY-13b-13: cash, realised PnL, the trade's exit price and the fee use the
    exchange fill (``last`` = 93.5); the bid reading (94) never enters the portfolio, and
    ``update_position``'s mark is the last CLOSED bar close (100)."""
    stack = await _entered(exchange)
    stop = _stop(stack)
    assert stop > Decimal("94")
    bid, last = Decimal("94"), Decimal("93.5")
    marks: list[tuple[Fill, Decimal]] = []
    real_up = stack.portfolio.update_position

    def _up(fill: Fill, current_price: Decimal) -> Any:
        marks.append((fill, current_price))
        return real_up(fill, current_price)

    stack.portfolio.update_position = _up  # type: ignore[method-assign]
    counts = _install_cf_spies(stack)
    install_intrabar_clock(stack)
    exchange.set_ticker(SYMBOL, bid=bid, ask=Decimal("94.1"), last=last)

    await run_intrabar_tick(stack)

    assert len(marks) == 1
    fill, mark = marks[0]
    assert fill.price == last and mark == Decimal("100")  # the last closed close
    trade = stack.portfolio.get_trade_history()[-1]
    assert trade.exit_price == last and trade.exit_price != bid
    assert fill.fee > 0
    assert counts["update_market_prices"] == 0
    assert _flat(stack)


async def test_t43_ticks_never_evaluate_the_breaker_tick_cooldown_a_strategy_or_count_a_bar(
    exchange: FakeCCXTExchange,
) -> None:
    """CF-30: counts are unchanged across ticks, including the tick that exits."""
    stack = await _entered(exchange)
    counts = _install_cf_spies(stack)
    strategy = stack.engine._strategies[0]
    calls_before = strategy._call_count  # type: ignore[attr-defined]
    bars_before = stack.engine.bar_count
    clock = install_intrabar_clock(stack)
    for _ in range(2):
        await run_intrabar_tick(stack)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    await run_intrabar_tick(stack)  # exits
    assert len(_sells(exchange)) == 1 and _flat(stack)
    assert counts == {"update_market_prices": 0, "evaluate": 0, "tick_cooldown": 0, "exc_on_bar": 0}
    assert strategy._call_count == calls_before  # type: ignore[attr-defined]
    assert stack.engine.bar_count == bars_before
    assert clock.waits == [CONFIRM]


# ---------------------------------------------------------------------------
# T-52 / T-53  outage and rate limits
# ---------------------------------------------------------------------------


async def test_t52_a_ticker_outage_degrades_loudly_never_fires_and_the_bar_close_still_exits(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    install_intrabar_clock(stack)
    exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"), times=3)
    seen = _spy_metrics(stack)
    with capture_logs() as logs:
        for _ in range(3):
            await run_intrabar_tick(stack)
        degraded_at_three = list(_events(logs, "exit_check.degraded"))
        await run_intrabar_tick(stack)  # the outage is over: first valid reading
    assert len(degraded_at_three) == 1 and degraded_at_three[0]["log_level"] == "error"
    assert degraded_at_three[0]["symbol"] == SYMBOL
    assert len(_events(logs, "exit_check.recovered")) == 1
    assert _sells(exchange) == []
    assert sum(1 for name, _ in seen if name == "intrabar_ticker_errors_total") == 3
    # the bar-close SL is untouched by any of this
    await _step(stack, 90)
    assert len(_sells(exchange)) == 1 and _flat(stack)


async def test_t52_degraded_repeats_every_15_minutes_only(exchange: FakeCCXTExchange) -> None:
    stack = await _entered(exchange)
    clock = install_intrabar_clock(stack)
    exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"), times=50)
    with capture_logs() as logs:
        for _ in range(5):  # ticks 3, 4, 5 are degraded, 300 s apart
            await run_intrabar_tick(stack)
            clock.now += 300.0
    assert len(_events(logs, "exit_check.degraded")) == 1
    with capture_logs() as later:
        clock.now += 900.0
        await run_intrabar_tick(stack)
    assert len(_events(later, "exit_check.degraded")) == 1  # >= 15 min since the last one


@pytest.mark.parametrize("exc_type", [ccxt.RateLimitExceeded, ccxt.DDoSProtection])
async def test_t53_a_rate_limit_abandons_the_tick_doubles_the_interval_and_is_restored(
    exchange: FakeCCXTExchange, exc_type: type[Exception]
) -> None:
    stack = await _entered(exchange)
    before = exchange.ticker_calls
    exchange.queue_ticker_error(SYMBOL, exc_type("simulated 429"), times=1)
    clock = await _loop(stack, horizon=2000.0)
    # tick 1 (t=60): 429 -> abandoned after ONE call, no retry; the next wait doubles
    assert clock.waits[:4] == [60.0, 600.0, 300.0, 300.0]
    assert exchange.ticker_calls - before == 1 + (
        len(clock.waits) - 1 - 1
    )  # 1 per tick, none extra
    assert _sells(exchange) == []


async def test_t53_repeated_rate_limits_double_up_to_the_cap_and_not_beyond(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    exchange.queue_ticker_error(SYMBOL, ccxt.RateLimitExceeded("simulated 429"), times=6)
    clock = await _loop(stack, horizon=8000.0)
    assert clock.waits[:6] == [60.0, 600.0, 900.0, 900.0, 900.0, 900.0]  # cap max(300, 900)
    assert max(clock.waits) <= 900.0 < float(stack.engine._tf_s)


# ---------------------------------------------------------------------------
# T-54 / T-55  the lock
# ---------------------------------------------------------------------------


async def test_t54_the_lock_is_not_held_across_the_confirm_wait_and_a_bar_exit_wins_cleanly(
    exchange: FakeCCXTExchange,
) -> None:
    """A whole bar poll runs INSIDE the confirm wait (it would deadlock if the tick held the
    lock) and exits at the bar close; the tick then sees a flat position: ONE SELL."""
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    clock = install_intrabar_clock(stack)
    observed: dict[str, Any] = {}

    async def _bar_poll_inside_the_wait() -> None:
        observed["locked_during_wait"] = stack.engine._cycle_lock.locked()
        await asyncio.wait_for(_step(stack, 90), timeout=5.0)
        observed["flat_after_bar"] = _flat(stack)

    clock.at(5.0, _bar_poll_inside_the_wait)
    await run_intrabar_tick(stack)
    assert observed == {"locked_during_wait": False, "flat_after_bar": True}
    assert len(_sells(exchange)) == 1


async def test_t55_flatten_holding_the_lock_makes_the_tick_abandon_and_flatten_stays_complete(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    clock = install_intrabar_clock(stack)
    gate = asyncio.Event()
    real = stack.execution.process_signal

    async def _gated(signal: Any) -> Any:
        if signal.strategy_id == "operator_flatten":
            await gate.wait()
        return await real(signal)

    stack.execution.process_signal = _gated  # type: ignore[method-assign]
    stack.risk_manager.trigger_kill_switch("stop_in_progress")
    flatten_task: asyncio.Task[Any] | None = None

    async def _start_flatten() -> None:
        nonlocal flatten_task
        flatten_task = asyncio.create_task(stack.engine.flatten("wp13bb-test", timeout_s=30.0))
        for _ in range(5):  # let it take the lock and park in the gated process_signal
            await asyncio.get_running_loop().run_in_executor(None, lambda: None)

    clock.at(5.0, _start_flatten)
    seen = _spy_metrics(stack)
    with capture_logs() as logs:
        await run_intrabar_tick(stack)
    assert stack.engine._cycle_lock.locked()
    assert _events(logs, "engine.intrabar_lock_busy")
    assert ("intrabar_lock_busy_total", {}) in seen
    assert stack.engine._intrabar_retry_soon is True
    assert _sells(exchange) == []  # the tick submitted nothing

    gate.set()
    assert flatten_task is not None
    result = await asyncio.wait_for(flatten_task, timeout=5.0)
    assert result.complete and len(_sells(exchange)) == 1 and _flat(stack)
    calls = exchange.ticker_calls
    await run_intrabar_tick(stack)  # the fast retry finds a flat run: no exchange call at all
    assert exchange.ticker_calls == calls and len(_sells(exchange)) == 1


async def test_t55_the_fast_retry_wait_is_30_seconds_then_the_normal_cadence(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    clock = install_intrabar_clock(stack, horizon=1000.0)
    await stack.engine._cycle_lock.acquire()  # a bar pass that never ends
    try:
        await asyncio.wait_for(run_intrabar_loop(stack), timeout=20.0)
    finally:
        stack.engine._cycle_lock.release()
    # tick@60 (confirm 10, lock busy) -> ONE fast retry after 30 s (confirm 10, busy again)
    # -> back to the regular 300 s cadence
    assert clock.waits[:6] == [60.0, CONFIRM, 30.0, CONFIRM, 300.0, CONFIRM]
    assert _sells(exchange) == []


# ---------------------------------------------------------------------------
# T-56 / T-57 / T-58  in-flight SELLs and partial fills
# ---------------------------------------------------------------------------


async def test_t58_accepted_then_timed_out_intrabar_sell_is_never_sent_twice(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.queue_accept_then_timeout(SYMBOL, ccxt.RequestTimeout("simulated timeout"))
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1
    bracket = stack.engine._bracket_exit
    assert bracket is not None and bracket.pending_exit_symbols == {SYMBOL}  # accepted: guard kept

    calls = exchange.ticker_calls
    await run_intrabar_tick(stack)  # next tick: the symbol is not even read
    assert exchange.ticker_calls == calls and len(_sells(exchange)) == 1
    for close in (89, 88, 87, 86):  # bar closes below the SL, incl. past the TTL
        await _step(stack, close)
        assert len(_sells(exchange)) == 1


async def test_t56_an_open_intrabar_sell_then_a_bar_close_below_the_sl_sells_once(
    exchange: FakeCCXTExchange,
) -> None:
    """The intra-bar SELL reached the exchange but its answer was lost (WP1.4b unknown
    submit: in flight, reserved).  Bar closes below the SL must not send a second SELL, and
    once the cid lookup adopts the first one the run is flat (routed once)."""
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.queue_accept_then_timeout(SYMBOL, ccxt.RequestTimeout("simulated timeout"))
    # the cid lookup filters on WALL-CLOCK time: stamp the fake's orders with it
    exchange.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    exchange.hide_from_listing(SYMBOL, calls=4)  # 3 inline lookups + the first bar's resolver
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1 and not _flat(stack)  # left in flight (unknown submit)
    flat_after: list[bool] = []
    for close in (90, 89, 88, 87):
        exchange.push_bar(SYMBOL, Decimal(close), timeframe=TF)
        await stack.engine._poll_and_process()
        assert len(_sells(exchange)) == 1  # the bar-close SL never sends a second one
        flat_after.append(_flat(stack))
    assert flat_after[0] is False and flat_after[-1] is True  # adopted + routed exactly once


async def test_t57_a_partially_executed_intrabar_sell_reprotects_the_residual_without_overselling(
    exchange: FakeCCXTExchange,
) -> None:
    """A REAL partial execution (40% of the SELL executes, the rest rests on the book) with
    0.5 external BTC on the account.  The residual is protected by a new intra-bar signal, but the
    live in-flight reservation (WP1.11a) blocks a second ``create_order``; once the rest executes
    the run is flat, the sum sold equals the run's own quantity and the external coins are
    untouched."""
    stack = await _entered(exchange)
    own = stack.portfolio.get_position(SYMBOL).quantity  # type: ignore[union-attr]
    external = Decimal("0.5")
    exchange.set_balance("BTC", exchange.balance_of("BTC") + external)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.queue_partial_execution(SYMBOL, Decimal("0.4"))
    signals = _spy_signals(stack)
    install_intrabar_clock(stack)

    await run_intrabar_tick(stack)
    residual = stack.portfolio.get_position(SYMBOL)
    assert residual is not None and residual.quantity == own * Decimal("0.6")
    assert len(_sells(exchange)) == 1 and stack.engine._bracket_exit.pending_exit_symbols == {
        SYMBOL
    }  # type: ignore[union-attr]

    with capture_logs() as logs:
        await run_intrabar_tick(stack)  # the quantity changed: the residual is re-protected ...
    assert len(signals) == 2 and signals[1].metadata["exit_path"] == "intrabar"
    assert len(_sells(exchange)) == 1  # ... but the reservation blocks a second create_order
    blocked = _events(logs, "engine.exit_not_submitted")
    assert [e["reason"] for e in blocked] == ["inflight_exit_pending"]
    assert blocked[0]["log_level"] == "info"
    assert SYMBOL not in stack.engine._exit_not_submitted_streak

    exchange.complete_partial_execution("fake-order-2")  # the rest of the SELL executes
    exchange.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    for close in (90, 89):
        exchange.push_bar(SYMBOL, Decimal(close), timeframe=TF)
        await stack.engine._poll_and_process()
    assert _flat(stack)
    assert len(_sells(exchange)) == 1
    assert exchange.balance_of("BTC") == external  # nothing but the run's own coins was sold


# ---------------------------------------------------------------------------
# T-59  not submitted -> pending cleared, retried at the next tick, loud at 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["rejected", "ticker_failure", "raised"])
async def test_t59_an_intrabar_exit_that_was_not_submitted_is_retried_at_the_next_tick(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    install_intrabar_clock(stack)
    if variant == "rejected":
        exchange.queue_order_error(SYMBOL, ccxt.InvalidOrder("simulated rejection"))
    elif variant == "ticker_failure":
        # let the two intra-bar readings through; process_signal's own ticker fetch fails
        # on all three ccxt_retry attempts
        exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"), times=3, after=2)
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
        await run_intrabar_tick(stack)
    assert _events(first, "engine.exit_not_submitted"), variant
    assert bracket.pending_exit_symbols == set()
    assert _sells(exchange) == [] and not _flat(stack)

    await run_intrabar_tick(stack)  # the next tick retries at once (no TTL wait)
    assert len(_sells(exchange)) == 1 and _flat(stack)


async def test_t59_the_not_submitted_streak_escalates_to_error_at_three_and_resets(
    exchange: FakeCCXTExchange,
) -> None:
    """CF-13b-22: warning for streak 1..2, error at 3 (then every 10th); an accepted
    order resets the streak."""
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    install_intrabar_clock(stack)
    real = stack.execution.process_signal
    state = {"empty": 3}

    async def _flaky(signal: Any) -> Any:
        if state["empty"] > 0:
            state["empty"] -= 1
            return []
        return await real(signal)

    stack.execution.process_signal = _flaky  # type: ignore[method-assign]
    seen = _spy_metrics(stack)
    levels: list[str] = []
    for _ in range(3):
        with capture_logs() as logs:
            await run_intrabar_tick(stack)
        levels.append(_events(logs, "engine.exit_not_submitted")[0]["log_level"])
    assert levels == ["warning", "warning", "error"]
    assert sum(1 for n, _ in seen if n == "exit_not_submitted_total") == 3
    assert stack.engine._exit_not_submitted_streak[SYMBOL] == 3
    await run_intrabar_tick(stack)  # accepted this time
    assert len(_sells(exchange)) == 1 and _flat(stack)
    assert SYMBOL not in stack.engine._exit_not_submitted_streak


async def test_t59_a_sell_already_in_flight_is_info_level_and_outside_the_streak(
    exchange: FakeCCXTExchange,
) -> None:
    """CF-13b-22 / SEC-05: when the live engine reports a pending SELL quantity the empty
    answer is ``inflight_exit_pending`` at info level, never a streak or an error."""
    stack = await _entered(exchange)
    exchange.queue_accept_then_timeout(SYMBOL, ccxt.RequestTimeout("simulated timeout"))
    await _step(stack, 90)  # bar-close SL: accepted but unknown -> a SELL is in flight
    assert len(_sells(exchange)) == 1 and stack.execution.pending_sell_quantity(SYMBOL) > 0
    levels: list[str] = []
    reasons: list[str] = []
    for close in (89, 88, 87, 86):  # past the 3-bar TTL the exit is re-emitted
        with capture_logs() as logs:
            await _step(stack, close)
        for e in _events(logs, "engine.exit_not_submitted"):
            levels.append(e["log_level"])
            reasons.append(e["reason"])
    assert reasons and set(reasons) == {"inflight_exit_pending"}
    assert set(levels) == {"info"}
    assert SYMBOL not in stack.engine._exit_not_submitted_streak
    assert len(_sells(exchange)) == 1


# ---------------------------------------------------------------------------
# T-60 / T-61 / T-62 / T-63  interactions
# ---------------------------------------------------------------------------


async def _assert_intrabar_sl_fires(stack: LiveStack, exchange: FakeCCXTExchange) -> None:
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1 and _flat(stack)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]  # no BUY anywhere


async def test_t60_kill_switch_does_not_stop_the_intrabar_exit(exchange: FakeCCXTExchange) -> None:
    stack = await _entered(exchange)
    stack.risk_manager.trigger_kill_switch("wp13bb-test")
    await _assert_intrabar_sl_fires(stack, exchange)


async def test_t60_protective_mode_does_not_stop_the_intrabar_exit(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    stack.engine._protective_mode = True
    stack.risk_manager.set_protective_mode(True)
    await _assert_intrabar_sl_fires(stack, exchange)


_ROOM = {
    "max_position_size_pct": 0.5,
    "max_portfolio_exposure_pct": 1.0,
    "max_cluster_exposure_pct": 1.0,
}


@pytest.mark.parametrize("level", ["halt", "daily_limit"])
async def test_t60_breaker_halt_and_daily_limit_do_not_stop_the_intrabar_exit(
    exchange: FakeCCXTExchange, level: str
) -> None:
    if level == "halt":
        params = dataclasses.replace(live_risk_parameters(), max_drawdown_pct=0.01, **_ROOM)
        expected = CircuitBreakerResponse.HALT
    else:
        params = dataclasses.replace(
            live_risk_parameters(), max_drawdown_pct=0.5, max_daily_loss_pct=0.01, **_ROOM
        )
        expected = CircuitBreakerResponse.DAILY_LIMIT
    cfg = {**FIXED, "bracket_stop_loss_pct": 0.20, "allow_pyramiding": True}
    stack = await _entered(exchange, cfg, risk_params=params, notional="150")
    # a dip that trips the breaker but stays above the 20% stop
    await _step(stack, 93)
    assert stack.circuit_breaker is not None
    assert stack.circuit_breaker.current_response == expected
    counts = _install_cf_spies(stack)
    await _assert_intrabar_sl_fires(stack, exchange)
    assert counts["evaluate"] == 0  # ticks never consult the breaker


async def test_t61_tp_is_bar_close_only(exchange: FakeCCXTExchange) -> None:
    stack = await _entered(exchange, FIXED_SL_TP)
    anchor = stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert anchor is not None and anchor.take_profit_price is not None
    high = anchor.take_profit_price + Decimal("5")
    exchange.set_ticker(SYMBOL, bid=high, last=high)
    clock = install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert _sells(exchange) == [] and clock.waits == []  # not even a candidate
    exchange.push_bar(SYMBOL, Decimal(str(high)), timeframe=TF)
    _set_now_after_close(exchange)
    signals = _spy_signals(stack)
    await stack.engine._poll_and_process()  # the bar close takes the profit
    assert len(_sells(exchange)) == 1
    assert signals[-1].metadata["trigger"] == "bracket_take_profit"
    assert "exit_path" not in signals[-1].metadata  # a missing exit_path means bar close


async def test_t62_no_key_means_no_task_and_no_ticks(exchange: FakeCCXTExchange) -> None:
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(exchange, {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05})
    assert stack.engine.exit_check_interval_s is None
    stack.engine._start_intrabar_task()
    assert stack.engine._intrabar_task is None
    await stack.engine._intrabar_exit_loop()  # returns at once
    assert exchange.ticker_calls == 0


async def test_t63_a_reconcile_required_symbol_is_evaluated_and_fires_capped(
    exchange: FakeCCXTExchange,
) -> None:
    """ARB-13b-01 / invariant I4: ``reconcile_required`` blocks BUYs only."""
    stack = await _entered(exchange)
    own = stack.portfolio.get_position(SYMBOL).quantity  # type: ignore[union-attr]
    exchange.set_balance("BTC", exchange.balance_of("BTC") + Decimal("0.5"))
    stack.execution._flag_reconcile(SYMBOL, "wp13bb_test_flag")
    assert SYMBOL in stack.execution.reconcile_required
    await _assert_intrabar_sl_fires(stack, exchange)
    assert _sells(exchange)[0]["amount"] <= own  # WP1.11a cap: external coins stay
    assert exchange.balance_of("BTC") == Decimal("0.5")


# ---------------------------------------------------------------------------
# T-66  resume during a breach
# ---------------------------------------------------------------------------


async def test_t66_resume_during_a_breach_restores_the_anchor_before_the_first_tick_which_fires(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    fills, orders = stack.execution.get_all_fills(), stack.execution.get_all_orders()
    live_stop = _stop(stack)
    exchange.pad_forming_bar = False
    resumed = await build_resumed_live_stack(
        exchange=exchange,
        strategy=ScriptedSequenceStrategy("wp13bb-r", {"script": {}}),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        fills=fills,
        orders=orders,
        engine_config=dict(FIXED),
    )
    await start_and_warmup(resumed, RUN_ID)
    exchange.pad_forming_bar = True
    # restored BEFORE any task / tick exists
    assert resumed.engine._intrabar_task is None
    assert _stop(resumed) == live_stop
    bid = _below(live_stop, "2")
    clock = await _loop(
        resumed,
        horizon=500.0,
        schedule=[(0.0, lambda: exchange.set_ticker(SYMBOL, bid=bid, last=bid))],
    )
    assert len(_sells(exchange)) == 1 and _flat(resumed)
    assert clock.waits[0] == 60.0
    assert clock.create_order_times[0] <= 60.0 + CONFIRM + 1.0  # the FIRST tick fires


# ---------------------------------------------------------------------------
# T-68  log hygiene (AC-13B-18 / CF-36)
# ---------------------------------------------------------------------------

_LEAK = "https://api.exchange.example/v3/market/products/BTC-EUR/ticker?api_key=" + "SECRET-KEY-123"


async def test_t68_intrabar_logs_carry_no_exception_text_url_or_raw_ticker(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    install_intrabar_clock(stack)
    low = _below(_stop(stack), "2")
    with capture_logs() as logs:
        # outage text, 429 text, an invalid reading, an unconfirmed breach, a confirm failure
        exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError(_LEAK))
        await run_intrabar_tick(stack)
        exchange.queue_ticker_error(SYMBOL, ccxt.RateLimitExceeded(_LEAK))
        await run_intrabar_tick(stack)
        exchange.set_ticker(SYMBOL, bid="SECRET-BID-TEXT", last=Decimal("100"))
        await run_intrabar_tick(stack)
        exchange.set_ticker(SYMBOL, bid=low, last=low)
        clock = stack.engine._intrabar_wait.__self__  # type: ignore[attr-defined]
        clock.at(
            clock.now + 5.0, lambda: exchange.queue_ticker_error(SYMBOL, ccxt.BadRequest(_LEAK))
        )
        await run_intrabar_tick(stack)
    rendered = json.dumps(logs, default=str)
    assert "SECRET" not in rendered and "api.exchange.example" not in rendered
    assert "ticker" not in {k for e in logs for k in e}  # no raw ticker dict as a field
    mine = [e for e in logs if str(e["event"]).startswith(("engine.intrabar_", "exit_check."))]
    assert mine, "the scenario must have produced intra-bar log records"
    assert all("error" not in e for e in mine)


async def test_sec03_the_exit_log_carries_both_asks_and_quote_ages_as_numbers(
    exchange: FakeCCXTExchange,
) -> None:
    """WP13BB-SEC-03 / C-2: forensic context on ``engine.intrabar_exit_signal`` (spread and
    the age of the exchange timestamp); logging only, the SELL is unchanged."""
    stack = await _entered(exchange)
    clock = install_intrabar_clock(stack)
    low = _below(_stop(stack), "2")
    ask1, ask2 = low + Decimal("0.5"), low + Decimal("0.7")
    stamp = int(datetime.now(UTC).timestamp() * 1000) - 7_000
    exchange.set_ticker(SYMBOL, bid=low, ask=ask1, last=low, timestamp_ms=stamp)
    clock.at(  # the second reading sees a wider book
        5.0, lambda: exchange.set_ticker(SYMBOL, bid=low, ask=ask2, last=low, timestamp_ms=stamp)
    )
    with capture_logs() as logs:
        await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1
    (event,) = _events(logs, "engine.intrabar_exit_signal")
    assert (event["ask1"], event["ask2"]) == (str(ask1), str(ask2))
    for key in ("quote_age_s1", "quote_age_s2"):
        assert isinstance(event[key], str) and 6.0 <= float(event[key]) < 120.0
    assert "SECRET" not in json.dumps(logs, default=str)


async def test_sec03_a_missing_ask_or_timestamp_is_logged_as_none(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    install_intrabar_clock(stack)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, ask=None, last=low, timestamp_ms=None)
    with capture_logs() as logs:
        await run_intrabar_tick(stack)
    (event,) = _events(logs, "engine.intrabar_exit_signal")
    assert [event[k] for k in ("ask1", "ask2", "quote_age_s1", "quote_age_s2")] == [None] * 4


# ---------------------------------------------------------------------------
# T-50 / T-38 / precedence: several symbols, stop during the confirm, SL before trailing
# ---------------------------------------------------------------------------

SYMBOL_B = "ETH/EUR"


async def _two_symbol_stack(ex: FakeCCXTExchange) -> LiveStack:
    ex.register_market(SYMBOL_B, base="ETH", quote="EUR")
    for sym in (SYMBOL, SYMBOL_B):
        ex.seed_flat_bars(sym, count=100, price=Decimal("100"), timeframe=TF)
    strategy = ScriptedSequenceStrategy(
        "wp13bb-2", {"script": {0: "buy", 1: "buy"}, "target_notional": "100"}
    )
    stack = await build_live_stack(
        exchange=ex,
        strategy=strategy,
        symbol=SYMBOL,
        symbols=[SYMBOL, SYMBOL_B],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        engine_config=dict(FIXED),
    )
    await start_and_warmup(stack, RUN_ID)
    for sym in (SYMBOL, SYMBOL_B):
        ex.push_bar(sym, Decimal("100"), timeframe=TF)
    _set_now_after_close(ex)
    await stack.engine._poll_and_process()
    for sym in (SYMBOL, SYMBOL_B):
        position = stack.portfolio.get_position(sym)
        assert position is not None and not position.is_flat
    return stack


async def test_t50_a_failing_symbol_never_blocks_the_others(exchange: FakeCCXTExchange) -> None:
    """Symbol A's first reading fails, symbol B is below its SL for both readings:
    B is sold, A is untouched and keeps its failure streak."""
    stack = await _two_symbol_stack(exchange)
    low = Decimal("90")
    exchange.set_ticker(SYMBOL_B, bid=low, last=low)
    exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("simulated outage"))
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    sells = _sells(exchange)
    assert [o["symbol"] for o in sells] == [SYMBOL_B]
    assert stack.engine._intrabar_failure_streak == {SYMBOL: 1}
    pos_a = stack.portfolio.get_position(SYMBOL)
    assert pos_a is not None and not pos_a.is_flat


async def test_t50_a_failing_confirmation_of_one_symbol_never_blocks_the_other(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _two_symbol_stack(exchange)
    low = Decimal("90")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.set_ticker(SYMBOL_B, bid=low, last=low)
    clock = install_intrabar_clock(stack)
    # reading 2 of symbol A (the first one read after the wait) fails; B's is fine
    clock.at(5.0, lambda: exchange.queue_ticker_error(SYMBOL, ccxt.NetworkError("outage")))
    await run_intrabar_tick(stack)
    assert [o["symbol"] for o in _sells(exchange)] == [SYMBOL_B]


async def test_t38_a_stop_during_the_confirm_wait_ends_the_tick_without_a_sell(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    clock = install_intrabar_clock(stack)
    clock.at(5.0, stack.engine._stop_event.set)
    before = exchange.ticker_calls
    await run_intrabar_tick(stack)
    assert exchange.ticker_calls - before == 1  # reading 1 only
    assert _sells(exchange) == [] and not _flat(stack)
    # and a later tick after the stop does nothing at all
    await run_intrabar_tick(stack)
    assert exchange.ticker_calls - before == 1


async def test_t38_a_stop_while_waiting_for_the_lock_submits_nothing(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP13BB-CR-01: the stop arrives while the confirmed tick is BLOCKED on the lock; the
    lock is then released, the tick acquires it and must still submit nothing (the
    ``_stop_event`` guard at the top of ``_intrabar_fire``)."""
    monkeypatch.setattr(se, "_INTRABAR_LOCK_TIMEOUT_S", 2.0)  # long enough to get the lock
    stack = await _entered(exchange)
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    install_intrabar_clock(stack)
    lock = stack.engine._cycle_lock
    await lock.acquire()  # a bar pass is running
    tick = asyncio.create_task(run_intrabar_tick(stack))
    try:
        for _ in range(200):  # until the tick is parked in the lock's waiter queue
            if lock._waiters:  # type: ignore[attr-defined]
                break
            await asyncio.get_running_loop().run_in_executor(None, lambda: None)
        assert lock._waiters, "the tick never reached the lock"  # type: ignore[attr-defined]
        stack.engine._stop_event.set()
    finally:
        lock.release()
    await asyncio.wait_for(tick, timeout=5.0)
    assert _sells(exchange) == [] and not _flat(stack)
    assert lock.locked() is False  # released in the tick's finally


async def test_the_bracket_stop_wins_over_the_trailing_stop_and_one_sell_goes_out(
    exchange: FakeCCXTExchange,
) -> None:
    cfg = {**FIXED, "trailing_stop_pct": 0.10}
    stack = await _entered(exchange, cfg)
    signals = _spy_signals(stack)
    low = Decimal("80")  # below the bracket stop (95.57) AND the trail (90)
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1
    assert [s.metadata["trigger"] for s in signals if s.metadata.get("exit_path")] == [
        "bracket_stop_loss"
    ]
    assert stack.engine._trailing_stop.pending_stop_symbols == set()  # type: ignore[union-attr]


async def test_a_held_symbol_without_a_protective_level_costs_no_ticker_call(
    exchange: FakeCCXTExchange,
) -> None:
    """A take-profit-only bracket has no stop to watch: the lot is held but nothing is read."""
    tp_only = {
        "bracket_mode": "fixed",
        "bracket_take_profit_pct": 0.10,
        "exit_check_interval_s": INTERVAL,
    }
    stack = await _entered(exchange, tp_only)
    assert stack.engine._intrabar_held_symbols() == []
    install_intrabar_clock(stack)
    before = exchange.ticker_calls
    await run_intrabar_tick(stack)
    assert exchange.ticker_calls == before and _sells(exchange) == []


async def test_an_exit_already_in_flight_is_not_even_read(exchange: FakeCCXTExchange) -> None:
    """A pending guard with an unchanged quantity (the SELL is in flight) takes the symbol out
    of the tick: no ticker call, no second signal."""
    stack = await _entered(exchange)
    bracket = stack.engine._bracket_exit
    assert bracket is not None
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    position = stack.portfolio.get_position(SYMBOL)
    bracket.check(SYMBOL, low, position)  # bar-close style emission: marks the symbol pending
    assert bracket.has_unchanged_pending(SYMBOL, position)
    install_intrabar_clock(stack)
    before = exchange.ticker_calls
    await run_intrabar_tick(stack)
    assert exchange.ticker_calls == before and _sells(exchange) == []


# ---------------------------------------------------------------------------
# T-62  paper / smoke / disabled: no ticks
# ---------------------------------------------------------------------------


async def test_t62_a_paper_stack_refuses_the_key_and_has_no_task(
    exchange: FakeCCXTExchange,
) -> None:
    from tests.integration.fakes.live_harness import build_paper_stack

    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    with pytest.raises(ValueError, match="only valid for LIVE"):
        await build_paper_stack(
            exchange=exchange,
            strategy=ScriptedSequenceStrategy("p", {"script": {}}),
            symbol=SYMBOL,
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=INITIAL,
            run_id=RUN_ID,
            engine_config=dict(FIXED),
        )
    paper = await build_paper_stack(
        exchange=exchange,
        strategy=ScriptedSequenceStrategy("p", {"script": {}}),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        engine_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
    )
    paper.engine._start_intrabar_task()
    assert paper.engine._intrabar_task is None and exchange.ticker_calls == 0


async def test_t62_the_real_run_live_loop_starts_and_stops_the_task_exactly_once(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _entered(exchange)
    engine = stack.engine
    clock = install_intrabar_clock(stack, horizon=10_000.0)
    started: list[asyncio.Task[None]] = []
    real_start = engine._start_intrabar_task

    def _start() -> None:
        real_start()
        assert engine._intrabar_task is not None
        started.append(engine._intrabar_task)

    engine._start_intrabar_task = _start  # type: ignore[method-assign]

    async def _poll_then_stop() -> None:
        await asyncio.sleep(0)
        await asyncio.get_running_loop().run_in_executor(None, lambda: None)
        engine._stop_event.set()

    engine._poll_and_process = _poll_then_stop  # type: ignore[method-assign]
    await asyncio.wait_for(engine.run_live_loop(), timeout=10.0)
    assert len(started) == 1 and started[0].done()
    assert engine._intrabar_task is None
    assert clock.waits and clock.waits[0] == 60.0


# ---------------------------------------------------------------------------
# CF-13b-20  the trailing gap is paginated
# ---------------------------------------------------------------------------


async def _gap_resume(
    ex: FakeCCXTExchange, *, extra_bars: int, page_cap: int | None
) -> tuple[LiveStack, list[dict[str, Any]]]:
    ex.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe=TF)
    stack = await _live(ex, {"trailing_stop_pct": 0.20})
    await _step(stack, 100)
    fills, orders = stack.execution.get_all_fills(), stack.execution.get_all_orders()
    for i in range(extra_bars):  # the engine is "down": the market keeps printing bars
        ex.push_bar(SYMBOL, Decimal(108 if i == 20 else 100 + (i % 2) * 2), timeframe=TF)
    ex.max_ohlcv_limit = page_cap
    ex.pad_forming_bar = False
    ex.ohlcv_calls.clear()
    with capture_logs() as logs:
        resumed = await build_resumed_live_stack(
            exchange=ex,
            strategy=ScriptedSequenceStrategy("r", {"script": {}}),
            symbol=SYMBOL,
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=INITIAL,
            run_id=RUN_ID,
            fills=fills,
            orders=orders,
            engine_config={"trailing_stop_pct": 0.20},
        )
        await start_and_warmup(resumed, RUN_ID)
    return resumed, logs


@pytest.mark.parametrize(("extra_bars", "min_pages"), [(450, 2), (1100, 4)])
async def test_cf20_a_gap_of_hundreds_of_bars_is_paginated_not_seeded_partially(
    exchange: FakeCCXTExchange, extra_bars: int, min_pages: int
) -> None:
    """The exchange answers at most 300 bars per call (ccxt Coinbase).  Gaps of 301..1000 bars
    used to end in ``seed_partial``; they now go through ``fetch_ohlcv_range`` (<= 4 pages)."""
    resumed, logs = await _gap_resume(exchange, extra_bars=extra_bars, page_cap=300)
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] == Decimal("108")  # type: ignore[union-attr]
    assert not [e for e in logs if e["event"] == "trailing_stop.seed_partial"]
    pages = [c for c in exchange.ohlcv_calls if c[2] is not None]
    assert min_pages <= len(pages) <= 4


async def test_cf20_beyond_the_1000_bar_bound_the_seed_is_partial_and_loud(
    exchange: FakeCCXTExchange,
) -> None:
    _resumed, logs = await _gap_resume(exchange, extra_bars=1150, page_cap=300)
    assert [e for e in logs if e["event"] == "trailing_stop.seed_partial"]


async def test_cf20_a_failing_page_falls_back_to_the_available_closes_and_says_so(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = exchange.fetch_ohlcv

    async def _flaky(
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        if since is not None:
            raise ccxt.NetworkError("https://api.example/v3?api_key=" + "SECRET-KEY-9")
        return await original(symbol, timeframe, since, limit, params)

    monkeypatch.setattr(exchange, "fetch_ohlcv", _flaky)
    resumed, logs = await _gap_resume(exchange, extra_bars=450, page_cap=300)
    assert [e for e in logs if e["event"] == "trailing_stop.seed_partial"]
    gap_errors = [e for e in logs if e["event"] == "engine.exit_gap_fetch_failed"]
    assert gap_errors and gap_errors[0]["error_type"] == "MarketDataError"
    assert "SECRET" not in json.dumps(gap_errors, default=str)
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] >= Decimal("100")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# CF-13b-23 / CF-13b-27
# ---------------------------------------------------------------------------


async def test_cf23_the_anchor_source_metric_tells_a_pending_anchor_from_a_final_one(
    exchange: FakeCCXTExchange,
) -> None:
    _wavy(exchange)
    stack = await _live(exchange, ATR_SL)
    seen = _spy_metrics(stack)
    await _step(stack, 101)
    anchors = [lab for name, lab in seen if name == "bracket_anchor_source_total"]
    assert anchors and anchors[0] == {"source": "entry", "final": "true"}


async def test_cf27_the_resume_origin_tag_survives_an_exception_after_the_install(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading.bracket_exit import BracketExitManager

    _wavy(exchange)
    stack = await _live(exchange, ATR_SL)
    await _step(stack, 101)
    for i in range(130):
        await _step(stack, 100 + (i % 2) * 2)
    fills, orders = stack.execution.get_all_fills(), stack.execution.get_all_orders()
    original = exchange.fetch_ohlcv

    async def _no_since(
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        if since is not None:
            raise ccxt.NetworkError("outage")
        return await original(symbol, timeframe, since, limit, params)

    monkeypatch.setattr(exchange, "fetch_ohlcv", _no_since)  # forces a non-final fallback anchor
    real_restore = BracketExitManager.restore_anchor

    def _install_then_raise(self: BracketExitManager, anchor: Any) -> None:
        real_restore(self, anchor)
        raise RuntimeError("fault after the install")

    monkeypatch.setattr(BracketExitManager, "restore_anchor", _install_then_raise)
    exchange.pad_forming_bar = False
    resumed = await build_resumed_live_stack(
        exchange=exchange,
        strategy=ScriptedSequenceStrategy("r", {"script": {}}),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        fills=fills,
        orders=orders,
        engine_config={k: v for k, v in ATR_SL.items() if k != "exit_check_interval_s"},
    )
    await start_and_warmup(resumed, RUN_ID)
    anchor = resumed.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]
    assert anchor is not None and not anchor.final  # the interim anchor was installed
    assert resumed.engine._anchor_origin.get(SYMBOL) == "resume"  # tag set BEFORE the install


# ---------------------------------------------------------------------------
# WP1.3b-b fix delta, Part B (test-only): CR-03 cancellation, CR-04 boundary and provenance,
# CR-06 trailing pending
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "fires"), [("0", True), ("0.01", False)], ids=["on_the_level", "one_cent_above"]
)
async def test_cr04_a_bid_exactly_on_the_level_fires_and_one_cent_above_does_not(
    exchange: FakeCCXTExchange, offset: str, fires: bool
) -> None:
    """WP13BB-CR-04 (a), mutants M01 / M02: the engine's ``bid1 <= level`` (candidate) and
    ``price <= level`` (confirmation) are inclusive, like the manager's own boundary."""
    stack = await _entered(exchange)
    level = _stop(stack)
    bid = level + Decimal(offset)
    exchange.set_ticker(SYMBOL, bid=bid, last=bid)
    signals = _spy_signals(stack)
    install_intrabar_clock(stack)
    before = exchange.ticker_calls
    await run_intrabar_tick(stack)
    exits = [s for s in signals if s.metadata.get("exit_path") == "intrabar"]
    if fires:
        assert len(_sells(exchange)) == 1 and _flat(stack)
        assert len(exits) == 1 and exits[0].metadata["trigger_price"] == str(level)
    else:
        assert _sells(exchange) == [] and not _flat(stack) and exits == []
        assert exchange.ticker_calls - before == 1  # reading 1 only: not even a candidate


@pytest.mark.parametrize("first_is_lower", [True, False], ids=["bid1_lower", "bid2_lower"])
async def test_cr04_the_trigger_price_is_the_higher_of_the_two_confirming_bids(
    exchange: FakeCCXTExchange, first_is_lower: bool
) -> None:
    """WP13BB-CR-04 (b), mutant M04 (``min`` instead of ``max``), and any "one reading
    only" variant: both bids are below the level, they differ, and the signal carries both
    plus the HIGHER one as its trigger price."""
    stack = await _entered(exchange)
    level = _stop(stack)
    deep, shallow = _below(level, "2"), _below(level, "0.5")
    bid1, bid2 = (deep, shallow) if first_is_lower else (shallow, deep)
    exchange.set_ticker(SYMBOL, bid=bid1, last=bid1)
    clock = install_intrabar_clock(stack)
    clock.at(5.0, lambda: exchange.set_ticker(SYMBOL, bid=bid2, last=bid2))
    signals = _spy_signals(stack)
    await run_intrabar_tick(stack)
    (exit_signal,) = [s for s in signals if s.metadata.get("exit_path") == "intrabar"]
    assert exit_signal.metadata["trigger_price"] == str(shallow)
    assert exit_signal.metadata["confirm_bids"] == [str(bid1), str(bid2)]
    assert len(_sells(exchange)) == 1


async def test_cr04_a_bar_close_stop_loss_exit_is_recorded_as_a_bar_close_trade(
    exchange: FakeCCXTExchange,
) -> None:
    """WP13BB-CR-04 (c), mutant M41: a missing ``exit_path`` in the signal metadata means
    bar close, and ``engine.trade_recorded`` says so."""
    stack = await _entered(exchange)
    with capture_logs() as logs:
        await _step(stack, 90)  # the close is below the SL: the bar pass sells
    assert len(_sells(exchange)) == 1 and _flat(stack)
    recorded = _events(logs, "engine.trade_recorded")
    assert len(recorded) == 1 and recorded[0]["exit_path"] == "bar_close"


async def test_cr06_a_trailing_exit_in_flight_with_an_unchanged_quantity_is_not_read_again(
    exchange: FakeCCXTExchange,
) -> None:
    """WP13BB-CR-06 (mutant M45): the trailing manager's ``has_unchanged_pending`` keeps a
    trailing-only run from reading the ticker of a symbol whose SELL is already in flight."""
    stack = await _entered(exchange, TRAIL)
    trailing = stack.engine._trailing_stop
    assert trailing is not None and stack.engine._bracket_exit is None
    low = Decimal("85")  # below the 10 % trail of the 100 peak
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    exchange.queue_accept_then_timeout(SYMBOL, ccxt.RequestTimeout("simulated timeout"))
    install_intrabar_clock(stack)
    await run_intrabar_tick(stack)
    assert len(_sells(exchange)) == 1 and trailing.pending_stop_symbols == {SYMBOL}
    calls = exchange.ticker_calls
    await run_intrabar_tick(stack)  # unchanged quantity, exit in flight: not even read
    assert exchange.ticker_calls == calls and len(_sells(exchange)) == 1


def _real_time_intrabar_task(
    stack: LiveStack,
    monkeypatch: pytest.MonkeyPatch,
    *,
    confirm_delay_s: float,
    join_timeout_s: float,
) -> None:
    """Prepare ``run_live_loop`` on REAL time: the first tick at once, a bar poll that
    never returns (``asyncio.sleep`` is patched instant in this module, so it blocks on an
    Event), the real ``_intrabar_wait`` (no virtual clock)."""
    monkeypatch.setattr(se, "_INTRABAR_FIRST_TICK_S", 0.0)
    monkeypatch.setattr(se, "_INTRABAR_JOIN_TIMEOUT_S", join_timeout_s)
    stack.engine._intrabar_confirm_delay_s = confirm_delay_s
    never = asyncio.Event()

    async def _poll_forever() -> None:
        await never.wait()

    stack.engine._poll_and_process = _poll_forever  # type: ignore[method-assign]


async def _cancel_and_settle(loop_task: asyncio.Task[None]) -> None:
    """Cancel the bar loop and wait (bounded) for its ``CancelledError``."""
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(loop_task, timeout=5.0)


async def test_cr03_cancelling_the_bar_loop_mid_fire_cancels_the_sibling_and_frees_the_lock(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP13BB-CR-03 (1): the likely shutdown path.  The sibling is parked inside a hung
    ``process_signal`` while holding ``_cycle_lock``; cancelling ``run_live_loop`` joins it
    for the (patched) 0.2 s, then cancels it (mutant: no ``task.cancel()`` after the join
    timeout hangs here)."""
    stack = await _entered(exchange)
    engine = stack.engine
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    _real_time_intrabar_task(stack, monkeypatch, confirm_delay_s=0.0, join_timeout_s=0.2)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def _hung_process_signal(signal: Any) -> Any:
        entered.set()
        await never.wait()

    stack.execution.process_signal = _hung_process_signal  # type: ignore[method-assign]
    loop_task = asyncio.create_task(engine.run_live_loop())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        sibling = engine._intrabar_task
        assert sibling is not None and not sibling.done()
        assert engine._cycle_lock.locked()  # the sibling is inside the fire section
        await _cancel_and_settle(loop_task)
    finally:
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
    assert sibling.done() and sibling.cancelled()
    assert not engine._cycle_lock.locked()
    assert engine._intrabar_task is None
    assert _sells(exchange) == []


async def test_cr03_cancelling_the_bar_loop_during_the_real_confirm_wait_ends_the_sibling_at_once(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP13BB-CR-03 (2): a real 30 s confirm wait.  The stop event interrupts it, so the
    sibling ends by itself well inside the join (not cancelled), nothing is sold and the
    lock is free (mutant: ``_intrabar_wait`` ignores the stop event)."""
    stack = await _entered(exchange)
    engine = stack.engine
    low = _below(_stop(stack), "2")
    exchange.set_ticker(SYMBOL, bid=low, last=low)
    _real_time_intrabar_task(stack, monkeypatch, confirm_delay_s=30.0, join_timeout_s=2.0)
    in_confirm = asyncio.Event()
    real_wait = engine._intrabar_wait

    async def _watch(seconds: float) -> bool:
        if seconds == 30.0:
            in_confirm.set()
        return await real_wait(seconds)

    engine._intrabar_wait = _watch  # type: ignore[method-assign]
    before = exchange.ticker_calls
    loop_task = asyncio.create_task(engine.run_live_loop())
    try:
        await asyncio.wait_for(in_confirm.wait(), timeout=5.0)
        sibling = engine._intrabar_task
        assert sibling is not None and not sibling.done()
        assert exchange.ticker_calls - before == 1  # reading 1 only; reading 2 is 30 s away
        await _cancel_and_settle(loop_task)
    finally:
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
    assert sibling.done() and not sibling.cancelled()  # it saw the stop event and returned
    assert not engine._cycle_lock.locked()
    assert engine._intrabar_task is None
    assert _sells(exchange) == [] and exchange.ticker_calls - before == 1
