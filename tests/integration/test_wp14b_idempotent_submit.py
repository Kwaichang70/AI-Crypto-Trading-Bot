"""
tests/integration/test_wp14b_idempotent_submit.py
------------------------------------------------------
WP1.4b (idempotent-submit spec) mandatory tests T1 and T4-T11: a real
``LiveExecutionEngine`` + a real ``PortfolioAccounting`` (as its
``LivePositionSource``) against ``FakeCCXTExchange`` (A-13 extensions).

T1 additionally drives a full ``StrategyEngine`` stack (via
``tests.integration.fakes.live_harness``) since it needs multiple bars'
worth of signal processing plus the resting-order/get_fills routing path
StrategyEngine owns.

Every test asserts the number of orders on the exchange side
(``exchange.order_log``), per the spec's blanket requirement.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import ccxt
import pytest

from common.models import MultiTimeframeContext
from common.types import OrderSide, OrderStatus, OrderType, SignalDirection, TimeFrame
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    build_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from trading.engines.live import LiveExecutionEngine
from trading.models import Order, RiskCheckResult, Signal
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters
from trading.strategy import BaseStrategy, StrategyMetadata

_SYMBOL = "BTC/EUR"
_BASE = "BTC"
_QUOTE = "EUR"
_SYMBOL2 = "ETH/EUR"
_BASE2 = "ETH"
_PRICE = Decimal("50000")
_RUN_ID = "wp14b-idem-run"


# ---------------------------------------------------------------------------
# Shared factory helpers (direct-engine tests: T4-T11)
# ---------------------------------------------------------------------------


def _make_risk_manager_mock() -> MagicMock:
    mock = MagicMock()

    def _pre_trade_check(*, order: Any, **_: Any) -> RiskCheckResult:
        return RiskCheckResult(
            approved=True, adjusted_quantity=order.quantity,
            rejection_reasons=[], warnings=[],
        )

    mock.pre_trade_check.side_effect = _pre_trade_check
    mock.calculate_position_size.return_value = Decimal("999")
    return mock


async def _make_engine_with_fake(
    *,
    initial_cash: Decimal = Decimal("100000"),
    unknown_submit_settle_s: float = 120.0,
    extra_symbol: bool = False,
) -> tuple[LiveExecutionEngine, PortfolioAccounting, FakeCCXTExchange]:
    ex = FakeCCXTExchange(exchange_id="coinbase")
    ex.register_market(_SYMBOL, base=_BASE, quote=_QUOTE)
    ex.seed_flat_bars(_SYMBOL, count=5, price=_PRICE)
    symbols = [_SYMBOL]
    if extra_symbol:
        # R-02(a): a SECOND, unrelated symbol -- proves the stale-unknown-
        # SELL BUY block is run-wide, not just on the flagged symbol.
        ex.register_market(_SYMBOL2, base=_BASE2, quote=_QUOTE)
        ex.seed_flat_bars(_SYMBOL2, count=5, price=_PRICE)
        symbols.append(_SYMBOL2)
    await ex.load_markets()
    # A-13: align the fake's order timestamps with real wall-clock time --
    # the cid lookup's `since` filter is computed from datetime.now(UTC),
    # which the fake's default synthetic bar-time epoch would fail.
    ex.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    ex.set_balance(_QUOTE, initial_cash)

    rm = _make_risk_manager_mock()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
        unknown_submit_settle_s=unknown_submit_settle_s,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=initial_cash)
    engine.attach_position_source(portfolio, symbols=symbols)
    return engine, portfolio, ex


def _make_signal(
    *,
    direction: SignalDirection,
    target: Decimal = Decimal("500"),
    symbol: str = _SYMBOL,
) -> Signal:
    return Signal(
        strategy_id="wp14b-test-strategy", symbol=symbol, direction=direction,
        target_position=target, confidence=1.0,
    )


# ---------------------------------------------------------------------------
# T1 (probe PG): a repeating-BUY strategy for the full-harness scenario.
# ---------------------------------------------------------------------------


class _RepeatingBuyStrategy(BaseStrategy):
    """Emits a BUY signal on each of the first ``max_fires`` ``on_bar``
    calls, then HOLDs forever -- lets T1 script "the first BUY, then N
    more BUY signals" deterministically."""

    metadata = StrategyMetadata(
        name="wp14b_repeating_buy_test_strategy",
        description="WP1.4b T1 harness-only strategy.",
        tags=["test-only"],
    )

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._max_fires = int(self._params.get("max_fires", 1))
        self._target_notional = Decimal(str(self._params.get("target_notional", "500")))
        self._call_count = -1

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Any, *, mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        self._call_count += 1
        if not bars or self._call_count >= self._max_fires:
            return []
        symbol = bars[-1].symbol
        return [
            Signal(
                strategy_id=self.strategy_id, symbol=symbol, direction=SignalDirection.BUY,
                target_position=self._target_notional, confidence=1.0,
            )
        ]


@pytest.mark.asyncio
async def test_t1_probe_pg_accept_then_timeout_exactly_one_exchange_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1 (probe PG): a EUR 1,000 run on a EUR 100,000 account. The first
    BUY is accepted by the exchange but the caller sees a timeout; three
    MORE BUY signals arrive before the outcome resolves (the exchange
    listing is deliberately hidden long enough to force this). Expected:
    exactly 1 exchange order, ending FILLED with its exchange id, and run
    cash reduced by exactly cost + fee -- never double-spent by a second
    real order."""
    async def _instant_sleep(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    exchange = FakeCCXTExchange(exchange_id="coinbase")
    exchange.register_market(_SYMBOL, base=_BASE, quote=_QUOTE)
    exchange.seed_flat_bars(_SYMBOL, count=5, price=_PRICE, timeframe="1h")
    exchange.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    exchange.set_balance(_QUOTE, Decimal("100000"))
    patch_exchange_factory(monkeypatch, exchange)

    # Accept-then-timeout on the FIRST create_order call; hide the order
    # from every listing call until well after the 4th signal's own
    # resting-order check would have looked for it (3 inline lookups + 3
    # per-bar resolver calls during bars 1-3 = 6; a generous margin).
    exchange.queue_accept_then_timeout(_SYMBOL)
    exchange.hide_from_listing(_SYMBOL, calls=10)

    strategy = _RepeatingBuyStrategy(
        "probe-pg", {"max_fires": 4, "target_notional": "500"},
    )
    stack = await build_live_stack(
        exchange=exchange, strategy=strategy, symbol=_SYMBOL,
        timeframe=TimeFrame.ONE_HOUR, initial_capital=Decimal("1000"),
        run_id=_RUN_ID, risk_params=RiskParameters(),
    )
    await start_and_warmup(stack, _RUN_ID)

    # Bars 1-4: the first BUY (accept-then-timeout) plus 3 more BUY
    # signals, all while the order stays hidden/unresolved.
    for _ in range(4):
        await step_bar(stack, _SYMBOL, _PRICE, timeframe="1h")

    assert len(exchange.order_log) == 1, "no second real order was ever placed"

    live_orders = list(stack.execution.get_all_orders())
    assert len(live_orders) == 1
    stuck_order = live_orders[0]
    assert stuck_order.status == OrderStatus.PENDING_SUBMIT
    assert stuck_order.order_id in stack.execution._unknown_submits

    # Reveal the order and let ONE more bar's resting-order check resolve
    # (adopt + reconcile-to-FILLED + route the fill) it via the normal
    # production path -- no special-cased test-only resolution.
    exchange.hide_from_listing(_SYMBOL, calls=0)
    await step_bar(stack, _SYMBOL, _PRICE, timeframe="1h")

    resolved_order = stack.execution.get_all_orders()[0]
    assert resolved_order.status == OrderStatus.FILLED
    assert resolved_order.exchange_order_id == exchange.order_log[0]["id"]
    assert len(exchange.order_log) == 1, "still exactly one exchange order"

    fills = stack.execution.get_all_fills()
    assert len(fills) == 1
    fill = fills[0]
    fee_in_quote = fill.fee if fill.fee_currency == _QUOTE else Decimal("0")
    expected_cash = Decimal("1000") - (fill.quantity * fill.price) - fee_in_quote
    assert stack.portfolio.cash == expected_cash

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T4: timeout-then-absent -> blocked; the 10s and 120s boundaries; REJECTED
# never_placed; flag cleared; next BUY sent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t4_boundaries_then_never_placed_unblocks_next_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4, rewritten for WP1.4b round 2 (S-01a): only a POST-settle absent
    lookup counts towards D7's evidence -- the 3 inline lookups (all well
    within the 120s settle window) must count for nothing."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("timed out -- never reached the exchange"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert len(orders) == 1
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    assert order.order_id in engine._unknown_submits
    entry = engine._unknown_submits[order.order_id]
    # S-01a: the 3 inline lookups are all pre-settle -- none count.
    assert entry.absent_count == 0

    # W5: blocked run-wide while unresolved.
    blocked = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert blocked == []

    # Fast-forward past the settle window -- the FIRST post-settle absent.
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=130)
    await engine._resolve_unknown_submits(_SYMBOL)
    assert entry.absent_count == 1
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT

    # Boundary 1: a SECOND post-settle absent, but < 10s after the first --
    # count reaches 2, but the gap boundary is not met.
    await engine._resolve_unknown_submits(_SYMBOL)
    assert entry.absent_count == 2
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT

    # Boundary 2: push the recorded first-absent back so the gap crosses
    # 10s -- now every D7 condition is met -> REJECTED never_placed, flag
    # cleared.
    entry.first_absent_at = datetime.now(UTC) - timedelta(seconds=15)
    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.REJECTED
    assert order.order_id not in engine._unknown_submits
    assert engine.reconcile_required.get(_SYMBOL) is None

    next_orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert len(next_orders) == 1
    assert next_orders[0].status in (OrderStatus.OPEN, OrderStatus.FILLED)
    assert len(ex.order_log) == 1  # the only REAL order this whole test ever placed


# ---------------------------------------------------------------------------
# T5: listing lag -> adopted on a later bar; a second get_fills call
# returns nothing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t5_listing_lag_adopted_later_get_fills_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.hide_from_listing(_SYMBOL, calls=3)  # hides exactly the 3 inline lookups
    ex.queue_accept_then_timeout(_SYMBOL)

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    assert order.order_id in engine._unknown_submits
    assert len(ex.order_log) == 1

    # "a later bar": the per-bar resting-order check resolves it now that
    # the listing is no longer hidden.
    candidates = await engine.check_resting_orders(_SYMBOL, _PRICE)
    assert any(o.order_id == order.order_id for o in candidates)
    assert order.order_id not in engine._unknown_submits

    fills_first = await engine.get_fills(order.order_id)
    assert len(fills_first) == 1

    fills_second = await engine.get_fills(order.order_id)
    assert fills_second == []
    assert len(ex.order_log) == 1


# ---------------------------------------------------------------------------
# T6: SELL with unknown state -- reserved, second SELL capped, no timer
# release.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t6_unknown_sell_reserved_second_sell_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T6, extended for WP1.4b round 2 (R-06): external holdings, a
    PARTIAL unknown-SELL remainder still sellable, and release by
    adoption (never a timer)."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, portfolio, ex = await _make_engine_with_fake()
    # R-06: external holdings -- the exchange reports 1.05 BTC total, but
    # the bot's own ledger only owns 0.05 of it; 1.0 is a holding this run
    # must never touch.
    ex.set_balance(_BASE, Decimal("1.05"))

    from trading.models import Position

    portfolio._position_snapshots[_SYMBOL] = Position(
        symbol=_SYMBOL, run_id=_RUN_ID, quantity=Decimal("0.05"),
        average_entry_price=_PRICE, current_price=_PRICE,
    )

    # R-06: a PARTIAL SELL (0.02 of the 0.05 owned), genuinely accepted by
    # the exchange but seen by us as a timeout -- hidden from listing for
    # the 3 inline lookups so it stays genuinely unresolved afterward.
    ex.hide_from_listing(_SYMBOL, calls=3)
    ex.queue_accept_then_timeout(_SYMBOL)
    partial_signal = _make_signal(direction=SignalDirection.SELL, target=Decimal("0.02") * _PRICE)
    first = await engine.process_signal(partial_signal)
    assert len(first) == 1
    stuck = first[0]
    assert stuck.status == OrderStatus.PENDING_SUBMIT
    assert stuck.side == OrderSide.SELL
    assert stuck.quantity == Decimal("0.02")
    assert len(ex.order_log) == 1

    # I5b(a)/R-06: only the SOLD 0.02 is reserved -- the remaining 0.03 is
    # still sellable (a partial remainder, not an all-or-nothing block).
    remainder_signal = _make_signal(direction=SignalDirection.SELL, target=Decimal("0"))
    second = await engine.process_signal(remainder_signal)
    assert len(second) == 1
    assert second[0].quantity == Decimal("0.03")
    assert len(ex.order_log) == 2

    # Now own_avail is back to 0 (0.05 owned - 0.02 unknown - 0.03 just
    # sold) -- a further SELL is blocked.
    third = await engine.process_signal(remainder_signal)
    assert third == []
    assert len(ex.order_log) == 2

    # The hold on the unknown 0.02 does not release on a timer -- even
    # "long after" submit, with no lookup evidence at all, it is still
    # reserved.
    entry = engine._unknown_submits[stuck.order_id]
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=99999)
    fourth = await engine.process_signal(remainder_signal)
    assert fourth == []

    # R-06: release happens by ADOPTION (the exchange really did accept
    # it) -- never a timer.
    await engine._resolve_unknown_submits(_SYMBOL)
    assert engine._orders[stuck.order_id].status in (
        OrderStatus.OPEN, OrderStatus.PARTIAL, OrderStatus.FILLED,
    )
    assert stuck.order_id not in engine._unknown_submits

    # External holdings are still exactly what they should be: 1.0 BTC
    # (1.05 total minus the two REAL sells this run made, 0.02 + 0.03).
    assert ex.balance_of(_BASE) == Decimal("1.00")


# ---------------------------------------------------------------------------
# T7: lookups keep failing -> stays unknown, no release, stale critical
# alert logged once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t7_lookup_always_fails_stays_unknown_stale_alert_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("timeout"))
    ex.queue_fetch_orders_error(_SYMBOL, RuntimeError("lookup boom"), times=100)

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    entry = engine._unknown_submits[order.order_id]
    assert entry.absent_count == 0  # W4: every failed lookup, never "absent"

    for _ in range(3):
        await engine._resolve_unknown_submits(_SYMBOL)
        assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT
        assert entry.absent_count == 0

    entry.submit_at = datetime.now(UTC) - timedelta(seconds=901)  # > default 900s
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)
        await engine._resolve_unknown_submits(_SYMBOL)

    stale_events = [e for e in cap if e.get("event") == "live.order_submit_state_unknown_stale"]
    assert len(stale_events) == 1
    assert stale_events[0].get("log_level") == "critical"
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT


@pytest.mark.asyncio
async def test_t7_sell_variant_300s_stale_alert_and_run_wide_buy_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T7 SELL variant (WP1.4b round 2, R-02): an unknown SELL whose
    lookups keep failing stays unresolved -- the reservation is still
    held, its own D16 alert fires at 300s (not the 900s BUY threshold,
    R-02(b)), and once older than 300s it blocks every new BUY run-wide,
    even on a completely unrelated symbol (R-02(a))."""
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, portfolio, ex = await _make_engine_with_fake(extra_symbol=True)
    ex.set_balance(_BASE, Decimal("0.05"))

    from trading.models import Position

    portfolio._position_snapshots[_SYMBOL] = Position(
        symbol=_SYMBOL, run_id=_RUN_ID, quantity=Decimal("0.05"),
        average_entry_price=_PRICE, current_price=_PRICE,
    )

    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("timeout"))
    ex.queue_fetch_orders_error(_SYMBOL, RuntimeError("lookup boom"), times=100)

    orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target=Decimal("0"))
    )
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    assert order.side == OrderSide.SELL
    entry = engine._unknown_submits[order.order_id]

    # Not yet 300s -- neither symbol's BUY is blocked by this SELL alone
    # (BTC/EUR has no position source symbol overlap issue since it's a
    # SELL; ETH/EUR has no unknown submit of its own yet).
    unblocked = await engine.process_signal(
        _make_signal(direction=SignalDirection.BUY, symbol=_SYMBOL2, target=Decimal("100"))
    )
    assert len(unblocked) == 1

    # R-02(b): the SELL's own stale alert fires at 300s, not 900s.
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=301)
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)
    stale_events = [e for e in cap if e.get("event") == "live.order_submit_state_unknown_stale"]
    assert len(stale_events) == 1
    assert stale_events[0].get("log_level") == "critical"
    assert stale_events[0].get("side") == "sell"

    # R-02(a): now blocks a BUY run-wide, including on the UNRELATED
    # symbol that never had any unknown submit of its own.
    #
    # WP1.4b round 3 (R-07): wrap this call in capture_logs() and assert
    # the SPECIFIC "live.buy_blocked_stale_unknown_sell" event fires --
    # the earlier ETH/EUR BUY above (line ~475-478) was allowed through
    # and never routed, so a plain in-flight-BUY check alone would ALSO
    # block this final call; without asserting the specific event, this
    # test would pass even if R-02's stale-unknown-SELL block were
    # deleted entirely.
    with capture_logs() as cap:
        blocked_other_symbol = await engine.process_signal(
            _make_signal(direction=SignalDirection.BUY, symbol=_SYMBOL2, target=Decimal("100"))
        )
    assert blocked_other_symbol == []
    blocked_events = [e for e in cap if e.get("event") == "live.buy_blocked_stale_unknown_sell"]
    assert len(blocked_events) == 1
    assert blocked_events[0].get("blocked_symbol") == _SYMBOL


# ---------------------------------------------------------------------------
# T8: partial fill while unknown -> imported once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t8_partial_fill_while_unknown_imported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.hide_from_listing(_SYMBOL, calls=3)
    ex.queue_accept_then_timeout(_SYMBOL)
    ex.queue_partial_fill(_SYMBOL, Decimal("0.4"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT

    # Adopt -> OPEN (fetch_orders listing never reports "closed" on its
    # own). The first reconcile (inside check_resting_orders) then reports
    # the queued partial fill.
    candidates = await engine.check_resting_orders(_SYMBOL, _PRICE)
    resolved = next(o for o in candidates if o.order_id == order.order_id)
    assert resolved.status == OrderStatus.PARTIAL
    assert Decimal("0") < resolved.filled_quantity < resolved.quantity

    # The fake records one trade (the full fill amount) at create_order
    # time -- the first get_fills call already routes it (I9), even
    # though the STATUS only caught up to PARTIAL on this poll.
    fills_partial = await engine.get_fills(order.order_id)
    assert len(fills_partial) == 1
    assert fills_partial[0].quantity == resolved.quantity

    # Second bar: the queued partial-fill fraction is consumed, so this
    # poll reports the order fully closed/filled -- but the fill itself
    # was already imported above, so nothing new is routed (I9: imported
    # exactly once, never re-delivered).
    candidates_2 = await engine.check_resting_orders(_SYMBOL, _PRICE)
    resolved_2 = next(o for o in candidates_2 if o.order_id == order.order_id)
    assert resolved_2.status == OrderStatus.FILLED

    fills_final = await engine.get_fills(order.order_id)
    assert fills_final == []
    assert len(ex.order_log) == 1


# ---------------------------------------------------------------------------
# T9: duplicate-cid reply, both as the existing order and as ExchangeError
# -> resolved, never REJECTED.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid_order", "exchange_error", "return_existing"])
async def test_t9_duplicate_cid_resolved_never_rejected(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()

    cid = f"{_RUN_ID}-{'a' * 12}"
    ex.seed_exchange_order(
        client_order_id=cid, symbol=_SYMBOL, side="buy",
        amount=Decimal("0.01"), price=_PRICE, status="closed", filled=Decimal("0.01"),
    )
    ex.queue_duplicate_cid(_SYMBOL, mode=mode)

    order = Order(
        client_order_id=cid, run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )
    result = await engine.submit_order(order)

    assert result.status != OrderStatus.REJECTED
    assert result.exchange_order_id is not None
    assert len(ex.order_log) == 0  # never a NEW real order


# ---------------------------------------------------------------------------
# T10: CancelledError -> registered unknown, flag set, error re-raised.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t10_cancelled_error_registers_unknown_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _portfolio, ex = await _make_engine_with_fake()
    monkeypatch.setattr(ex, "create_order", AsyncMock(side_effect=asyncio.CancelledError()))

    order = Order(
        client_order_id=f"{_RUN_ID}-{'b' * 12}", run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )

    with pytest.raises(asyncio.CancelledError):
        await engine.submit_order(order)

    assert order.order_id in engine._unknown_submits
    assert engine.reconcile_required.get(_SYMBOL) == "buy_submit_unknown"
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT


@pytest.mark.asyncio
async def test_t10b_cancelled_during_inline_sleep_registers_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T10b (WP1.4b round 2, S-03): a CancelledError raised during one of
    the inline (D5) sleeps -- AFTER create_order already returned an
    ambiguous error -- must also register the order as unknown before
    propagating; D15 alone only covers cancellation during the
    create_order call itself."""
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.ExchangeError("ambiguous, then cancelled mid-lookup"))

    async def _cancel_on_sleep(*_a: object, **_k: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _cancel_on_sleep)

    order = Order(
        client_order_id=f"{_RUN_ID}-{'d' * 12}", run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )

    with pytest.raises(asyncio.CancelledError):
        await engine.submit_order(order)

    assert order.order_id in engine._unknown_submits
    assert engine.reconcile_required.get(_SYMBOL) == "buy_submit_unknown"
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT
    assert len(ex.order_log) == 0


# ---------------------------------------------------------------------------
# T11: on_stop does not cancel unresolved orders locally.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t11_on_stop_leaves_unresolved_submit_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("never reached the exchange"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT

    with capture_logs() as cap:
        await engine.on_stop()

    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT
    assert order.order_id in engine._unknown_submits

    # WP1.4b round 2 (S-07): critical, with cid/side/quantity -- a user
    # stop means no resume scan will ever reconcile this order.
    pending_events = [
        e for e in cap if e.get("event") == "live.unresolved_submit_left_pending_on_stop"
    ]
    assert len(pending_events) == 1
    assert pending_events[0].get("log_level") == "critical"
    assert pending_events[0].get("cid") == order.client_order_id
    assert pending_events[0].get("side") == "buy"
    assert pending_events[0].get("quantity") == str(order.quantity)


# ===========================================================================
# WP1.4b round 2 (security/risk/critic review) -- additional tests.
# ===========================================================================


@pytest.mark.asyncio
async def test_s01a_single_flaky_absent_after_settle_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (S-01, "single-flaky" probe): the order WAS accepted
    by the exchange; its listing lags through the 3 inline lookups; then
    lookups fail outright for a while; then exactly ONE empty page (a
    single flaky "absent") arrives at +132s (post-settle). This single
    absent must NOT be enough to conclude ``never_placed`` -- and the
    order is still correctly adopted once the listing catches up."""
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.hide_from_listing(_SYMBOL, calls=3)  # hides exactly the 3 inline lookups
    ex.queue_accept_then_timeout(_SYMBOL)

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    entry = engine._unknown_submits[order.order_id]
    assert entry.absent_count == 0  # S-01a: pre-settle, never counted

    # ~2 minutes of failing lookups (never "absent") -- simulated by
    # fast-forwarding submit_at into the settle window and feeding the
    # resolver a run of lookup FAILURES.
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=125)
    ex.queue_fetch_orders_error(_SYMBOL, RuntimeError("flaky"), times=5)
    for _ in range(5):
        await engine._resolve_unknown_submits(_SYMBOL)
    assert entry.absent_count == 0
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT

    # ONE empty page (the order still exists, but THIS ONE listing call
    # omits it) at +132s -- a single flaky "absent" must NOT be enough.
    ex.hide_from_listing(_SYMBOL, calls=1)
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert entry.absent_count == 1
    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT
    assert not any(e.get("event") == "live.order_submit_never_placed" for e in cap)

    # The order is still correctly adopted once the listing catches up.
    await engine._resolve_unknown_submits(_SYMBOL)
    assert engine._orders[order.order_id].status in (OrderStatus.OPEN, OrderStatus.FILLED)
    assert order.order_id not in engine._unknown_submits
    assert len(ex.order_log) == 1


@pytest.mark.asyncio
async def test_s01c_never_placed_found_later_blocks_buys_run_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (S-01c, the >120s listing-lag probe): a genuinely
    accepted order is wrongly concluded ``never_placed`` because the
    listing stayed hidden through the whole D7 evidence window. Once the
    listing catches up, the watch list must catch the contradiction, flag
    the symbol, halt every future BUY run-wide, and log critical -- never
    silently let a second BUY go out."""
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.hide_from_listing(_SYMBOL, calls=100)  # long listing lag
    ex.queue_accept_then_timeout(_SYMBOL)  # order really placed

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    entry = engine._unknown_submits[order.order_id]

    # Force D7 evidence (2 post-settle absents >= 10s apart) while STILL
    # hidden -- REJECTED never_placed (wrongly, as it turns out).
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=130)
    await engine._resolve_unknown_submits(_SYMBOL)
    entry.first_absent_at = datetime.now(UTC) - timedelta(seconds=15)
    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.REJECTED
    assert order.order_id in engine._never_placed_watch

    # Reveal the listing -- the watch pass now finds the (really-placed)
    # order and raises the alarm.
    ex.hide_from_listing(_SYMBOL, calls=0)
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id not in engine._never_placed_watch
    assert engine.reconcile_required.get(_SYMBOL) == "never_placed_contradicted"
    assert engine._run_buy_block == "never_placed_contradicted"
    critical_events = [
        e for e in cap if e.get("event") == "live.order_never_placed_found_later"
    ]
    assert len(critical_events) == 1
    assert critical_events[0].get("log_level") == "critical"
    assert critical_events[0].get("cid") == order.client_order_id

    # No second BUY is ever placed.
    second = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert second == []
    assert len(ex.order_log) == 1


@pytest.mark.asyncio
async def test_s_r2_04_never_placed_watch_anchors_on_submit_at_not_rejected_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 3 (S-R2-04): ``_NeverPlacedWatch``'s own cid lookup
    must anchor on ``submit_at`` (the ORIGINAL submit time), not
    ``rejected_at`` (when the never_placed verdict itself lands) --
    proven with a submit-to-reject gap (~2h10s here) wide enough that
    anchoring on ``rejected_at`` would put the watch's own 1h lookback
    window entirely AFTER the real exchange order's timestamp,
    permanently missing it."""
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()

    now = datetime.now(UTC)
    # The exchange's OWN clock stamps the about-to-be-placed order 2h in
    # the past relative to "now" -- this fake's set_now_ms is the tool
    # this harness gives us to control that deterministically (a real
    # exchange never reports a future timestamp for an order it accepts
    # "now"; the scenario this proves is a long gap between the real
    # accept and this engine concluding never_placed, not a literal
    # backdated exchange clock).
    ex.set_now_ms(int((now - timedelta(hours=2)).timestamp() * 1000))
    ex.queue_accept_then_timeout(_SYMBOL)
    ex.hide_from_listing(_SYMBOL, calls=100)  # long listing lag through D7 settling

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    entry = engine._unknown_submits[order.order_id]

    # The LOCAL submit_at (this run's own bookkeeping clock) is itself
    # ~2h10s in the past -- easily satisfies the D7 settle window, and is
    # exactly what the FIXED watch must anchor its own lookup on.
    entry.submit_at = now - timedelta(hours=2, seconds=10)
    await engine._resolve_unknown_submits(_SYMBOL)
    entry.first_absent_at = now - timedelta(seconds=15)
    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.REJECTED
    watch = engine._never_placed_watch[order.order_id]
    assert watch.submit_at == entry.submit_at

    # Reveal the listing -- the FIXED watch (anchored on submit_at, whose
    # own -1h window reaches back to ~now-3h10s) must now find the real
    # (exchange-clock-2h-old) order and raise the alarm. Anchored on
    # rejected_at (~now) instead, its -1h window would only reach back to
    # ~now-1h -- entirely after the order's now-2h timestamp -- and it
    # would stay blind forever.
    ex.hide_from_listing(_SYMBOL, calls=0)
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id not in engine._never_placed_watch
    assert engine.reconcile_required.get(_SYMBOL) == "never_placed_contradicted"
    assert engine._run_buy_block == "never_placed_contradicted"
    critical_events = [
        e for e in cap if e.get("event") == "live.order_never_placed_found_later"
    ]
    assert len(critical_events) == 1


@pytest.mark.asyncio
async def test_s_r2_05_contradicted_sell_reserve_caps_second_sell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 3 (S-R2-05): once the never-placed watch finds a
    contradicting order for a SELL (the watch's own lookup proves the
    exchange DID place it, even though the engine concluded
    never_placed), that quantity must stay reserved for the ENGINE's
    LIFETIME -- no order in PENDING_SUBMIT/OPEN/PARTIAL represents it any
    more (it is REJECTED), so without an explicit reserve a second SELL
    would spend the external coins this reservation protects."""
    from structlog.testing import capture_logs

    from trading.models import Position

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, portfolio, ex = await _make_engine_with_fake()
    # External holdings: exchange reports 1.05 BTC total, the bot's own
    # ledger only owns 0.05 of it.
    ex.set_balance(_BASE, Decimal("1.05"))
    portfolio._position_snapshots[_SYMBOL] = Position(
        symbol=_SYMBOL, run_id=_RUN_ID, quantity=Decimal("0.05"),
        average_entry_price=_PRICE, current_price=_PRICE,
    )

    # A 0.02 SELL genuinely accepted by the exchange (queue_accept_then_
    # timeout mutates the real balance) but seen by us as ambiguous, then
    # -- wrongly, since the listing stays hidden through the whole D7
    # evidence window -- concluded never_placed.
    ex.hide_from_listing(_SYMBOL, calls=100)
    ex.queue_accept_then_timeout(_SYMBOL)
    first = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target=Decimal("0.02") * _PRICE)
    )
    stuck = first[0]
    assert stuck.quantity == Decimal("0.02")
    entry = engine._unknown_submits[stuck.order_id]

    entry.submit_at = datetime.now(UTC) - timedelta(seconds=130)
    await engine._resolve_unknown_submits(_SYMBOL)
    entry.first_absent_at = datetime.now(UTC) - timedelta(seconds=15)
    await engine._resolve_unknown_submits(_SYMBOL)
    assert engine._orders[stuck.order_id].status == OrderStatus.REJECTED
    assert stuck.order_id in engine._never_placed_watch

    # Reveal the listing -- the watch finds the real order and reserves
    # its quantity.
    ex.hide_from_listing(_SYMBOL, calls=0)
    with capture_logs():
        await engine._resolve_unknown_submits(_SYMBOL)
    assert stuck.order_id not in engine._never_placed_watch
    assert engine._contradicted_sell_reserve.get(_SYMBOL) == Decimal("0.02")

    # A second SELL is capped to exactly the remainder (0.05 - 0.02 =
    # 0.03) -- the reserved 0.02 is never touched a second time.
    remainder_signal = _make_signal(direction=SignalDirection.SELL, target=Decimal("0"))
    second = await engine.process_signal(remainder_signal)
    assert len(second) == 1
    assert second[0].quantity == Decimal("0.03")

    # A third SELL, once the remainder is also spoken for, is blocked
    # entirely.
    third = await engine.process_signal(remainder_signal)
    assert third == []

    # External holdings are exactly what they should be: 1.00 BTC (1.05
    # total minus the two REAL sells, 0.02 [contradicted-but-real] + 0.03).
    assert ex.balance_of(_BASE) == Decimal("1.00")

    # The reserve persists across a further resolver pass -- no timer or
    # adoption ever clears it.
    await engine._resolve_unknown_submits(_SYMBOL)
    assert engine._contradicted_sell_reserve.get(_SYMBOL) == Decimal("0.02")


@pytest.mark.asyncio
async def test_s_r2_08_half_applied_adoption_critical_alert_fires_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 3 (S-R2-08): the ``submit_adoption_failed`` critical
    alert must fire exactly ONCE per order, not on every resolver pass --
    the reconcile flag itself is still re-asserted every pass (I4), but an
    operator paged once should not then be paged again forever for as
    long as the entry stays half-applied."""
    from structlog.testing import capture_logs

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.ExchangeError("ambiguous"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.order_id in engine._unknown_submits

    # Simulate a half-applied adoption: the exchange id got recorded, but
    # the order itself is still (artificially, for this defensive test)
    # PENDING_SUBMIT.
    engine._exchange_order_map[order.order_id] = "half-applied-id"

    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)
        await engine._resolve_unknown_submits(_SYMBOL)
        await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id in engine._unknown_submits  # still never cleared
    assert engine.reconcile_required.get(_SYMBOL) == "submit_adoption_failed"
    critical_events = [
        e for e in cap if e.get("event") == "live.order_submit_adoption_failed"
    ]
    assert len(critical_events) == 1  # not 3


@pytest.mark.asyncio
async def test_s_r2_07_never_placed_watch_capped_at_50_evicts_oldest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 3 (S-R2-07, optional hardening): the never-placed
    watch list is capped at 50 entries -- once exceeded, the OLDEST entry
    (by ``rejected_at``) is evicted with a critical log, never silently
    dropped."""
    from uuid import uuid4

    from structlog.testing import capture_logs

    from trading.engines.live import _NeverPlacedWatch

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()

    now = datetime.now(UTC)
    for i in range(50):
        oid = uuid4()
        engine._never_placed_watch[oid] = _NeverPlacedWatch(
            symbol=_SYMBOL, side=OrderSide.BUY, client_order_id=f"cid-{i}",
            submit_at=now, rejected_at=now - timedelta(seconds=50 - i),
        )
    oldest_id = min(
        engine._never_placed_watch,
        key=lambda k: engine._never_placed_watch[k].rejected_at,
    )
    assert len(engine._never_placed_watch) == 50

    # Trigger one more rejection via the real path -- the 51st entry.
    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("timeout"))
    ex.hide_from_listing(_SYMBOL, calls=100)
    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    entry = engine._unknown_submits[order.order_id]
    entry.submit_at = now - timedelta(seconds=130)
    await engine._resolve_unknown_submits(_SYMBOL)
    entry.first_absent_at = now - timedelta(seconds=15)
    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.REJECTED
    assert len(engine._never_placed_watch) == 50  # capped, not 51
    assert oldest_id not in engine._never_placed_watch  # oldest evicted
    assert order.order_id in engine._never_placed_watch  # newest kept

    evicted_events = [
        e for e in cap if e.get("event") == "live.never_placed_watch_evicted"
    ]
    assert len(evicted_events) == 1
    assert evicted_events[0].get("log_level") == "critical"


@pytest.mark.asyncio
async def test_s04_duplicate_cid_evidence_blocks_never_placed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (S-04): a DuplicateOrderId reply is positive evidence
    the cid exists -- even once D7's usual evidence bar (2 post-settle
    absents >= 10s apart) is fully met, ``never_placed`` must not fire."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.hide_from_listing(_SYMBOL, calls=100)  # never found by any lookup
    ex.queue_order_error(_SYMBOL, ccxt.DuplicateOrderId("duplicate client_order_id"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    entry = engine._unknown_submits[order.order_id]
    assert entry.exists_evidence is True

    # Even with full D7 evidence satisfied, never_placed must NOT fire.
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=130)
    await engine._resolve_unknown_submits(_SYMBOL)
    entry.first_absent_at = datetime.now(UTC) - timedelta(seconds=15)
    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.PENDING_SUBMIT
    assert order.order_id in engine._unknown_submits


@pytest.mark.asyncio
async def test_s05_adopted_expired_order_via_open_not_stuck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (S-05): a lookup match whose ccxt status is
    "expired" must adopt cleanly through OPEN first (PENDING_SUBMIT ->
    EXPIRED is not a legal transition on its own)."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.ExchangeError("ambiguous"))
    ex.hide_from_listing(_SYMBOL, calls=100)

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT

    ex.hide_from_listing(_SYMBOL, calls=0)
    # Directly seed a raw "expired" match for this cid (the exchange really
    # did create it, but it expired before it could fill).
    ex.seed_exchange_order(
        client_order_id=order.client_order_id, symbol=_SYMBOL, side="buy",
        amount=Decimal("0.01"), price=_PRICE, status="expired", filled=Decimal("0"),
    )

    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[order.order_id].status == OrderStatus.EXPIRED
    assert order.order_id not in engine._unknown_submits


@pytest.mark.asyncio
async def test_s05_half_applied_adoption_flags_and_keeps_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (S-05): if the exchange id was recorded but the
    order never actually left PENDING_SUBMIT (a defensive scenario -- not
    reachable via Coinbase/ccxt 4.5.40 today, per the security report),
    the resolver must NOT silently clear the entry -- it must flag
    ``submit_adoption_failed``, log critical, and keep the entry so an
    operator (or a future D16-style repeat) is not silently starved."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.ExchangeError("ambiguous"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    assert order.order_id in engine._unknown_submits

    # Simulate a half-applied adoption: the exchange id got recorded, but
    # the order itself is still (artificially, for this defensive test)
    # PENDING_SUBMIT.
    engine._exchange_order_map[order.order_id] = "half-applied-id"

    from structlog.testing import capture_logs

    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id in engine._unknown_submits  # NOT cleared
    assert engine.reconcile_required.get(_SYMBOL) == "submit_adoption_failed"
    critical_events = [
        e for e in cap if e.get("event") == "live.order_submit_adoption_failed"
    ]
    assert len(critical_events) == 1
    assert critical_events[0].get("log_level") == "critical"


@pytest.mark.asyncio
async def test_r01_balance_unavailable_outage_re_flags_submit_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (R-01): a ``balance_unavailable`` cycle elsewhere
    must never leave a genuinely-unresolved unknown submit unflagged -- the
    resolver re-asserts ``buy_submit_unknown``/``sell_submit_unknown`` on
    every non-resolving outcome."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()
    ex.queue_order_error(_SYMBOL, ccxt.RequestTimeout("never reached the exchange"))

    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT
    assert engine.reconcile_required.get(_SYMBOL) == "buy_submit_unknown"

    # A balance outage elsewhere in the engine overwrites the flag (I4's
    # ``_flag_reconcile`` always overwrites) ...
    engine._flag_reconcile(_SYMBOL, "balance_unavailable")
    assert engine.reconcile_required.get(_SYMBOL) == "balance_unavailable"

    # ... and "recovers" (I4's own self-clearing rule for THAT one reason).
    engine._maybe_clear_balance_unavailable(_SYMBOL)
    assert _SYMBOL not in engine.reconcile_required

    # Without R-01, the unknown submit's own flag would now be gone even
    # though the order is still genuinely unresolved. The next resolver
    # pass must re-set it.
    await engine._resolve_unknown_submits(_SYMBOL)
    assert engine.reconcile_required.get(_SYMBOL) == "buy_submit_unknown"

    # And a BUY is still blocked.
    blocked = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert blocked == []


@pytest.mark.asyncio
async def test_r03_adoption_refused_still_logs_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 2 (R-03/C-02): when a lookup finds a match but adoption
    is refused (the exchange id already belongs to a DIFFERENT local
    order, D5's reverse-map conflict guard), the order stays unresolved
    and forever un-alertable unless the resolver still calls
    ``_maybe_log_stale_unknown`` for it."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, ex = await _make_engine_with_fake()

    # A DIFFERENT local order already owns this exchange id.
    other_order_id = Order(
        client_order_id=f"{_RUN_ID}-{'e' * 12}", run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    ).order_id
    engine._reverse_order_map["conflicting-exch-id"] = other_order_id

    ex.queue_order_error(_SYMBOL, ccxt.ExchangeError("ambiguous"))
    orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    order = orders[0]
    assert order.status == OrderStatus.PENDING_SUBMIT

    ex.seed_exchange_order(
        client_order_id=order.client_order_id, symbol=_SYMBOL, side="buy",
        amount=Decimal("0.01"), price=_PRICE, status="closed", filled=Decimal("0.01"),
    )
    # Make the SEEDED order's exchange id collide with the one already
    # claimed above.
    seeded_id = next(iter(ex._orders))
    ex._orders[seeded_id]["id"] = "conflicting-exch-id"

    entry = engine._unknown_submits[order.order_id]
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=901)

    from structlog.testing import capture_logs

    with capture_logs() as cap:
        await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id in engine._unknown_submits  # still unresolved
    assert engine.reconcile_required.get(_SYMBOL) == "submit_adoption_conflict"
    stale_events = [
        e for e in cap if e.get("event") == "live.order_submit_state_unknown_stale"
    ]
    assert len(stale_events) == 1
    assert stale_events[0].get("log_level") == "critical"


@pytest.mark.asyncio
async def test_c02_adoption_conflict_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """WP14b-C-02: ``_try_adopt`` refuses to adopt when ``_reverse_order_map``
    already points the found exchange id at a DIFFERENT local order --
    flags ``submit_adoption_conflict``, never silently reassigns the id."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, _portfolio, _ex = await _make_engine_with_fake()

    other_order_id = Order(
        client_order_id=f"{_RUN_ID}-{'f' * 12}", run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    ).order_id
    engine._reverse_order_map["already-owned-id"] = other_order_id

    raw_match = {
        "id": "already-owned-id",
        "clientOrderId": "some-cid",
        "symbol": _SYMBOL,
        "side": "buy",
        "status": "closed",
        "filled": "0.01",
        "average": "50000",
    }
    order = Order(
        client_order_id="some-cid", run_id=_RUN_ID, symbol=_SYMBOL,
        side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )
    engine._orders[order.order_id] = order

    result = engine._try_adopt(order, raw_match)

    assert result is None
    assert engine.reconcile_required.get(_SYMBOL) == "submit_adoption_conflict"
    # The original owner is untouched.
    assert engine._reverse_order_map["already-owned-id"] == other_order_id
    assert order.order_id not in engine._exchange_order_map
