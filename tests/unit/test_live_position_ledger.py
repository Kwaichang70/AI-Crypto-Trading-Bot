"""
tests/unit/test_live_position_ledger.py
------------------------------------------
WP1.1 (Verbeterplan v2, `reports/vp2-wp1.1/synthesis-spec.md` §5) -- unit
tests for the live position ledger fix (C1/C22).

Module under test
------------------
    packages/trading/engines/live.py -- LiveExecutionEngine, now reading
    its own held quantity, open positions, and daily PnL from an attached
    ``LivePositionSource`` (``PortfolioAccounting`` in production) instead
    of the never-populated legacy ``_positions`` dict.

Test IDs mirror the spec's §5 table exactly (R-01..R-12, S-01..S-07) so a
reviewer can cross-reference the spec directly; each test's docstring
repeats its ID and the invariant it proves.

Design notes
------------
- ``_make_engine_with_source()`` builds a real ``LiveExecutionEngine`` +
  a real ``PortfolioAccounting`` (used as the ``LivePositionSource`` --
  there is no reason to mock the class WP1.1 chose as the single source of
  truth, D1) + a mocked CCXT exchange, and attaches the portfolio exactly
  as ``StrategyEngine.__init__`` does.
- Fill routing (``portfolio.update_position(fill, price)``) is called
  directly in these tests to simulate the one line
  ``StrategyEngine._process_bar`` runs after every ``get_fills()`` call --
  these are unit tests of the engine+portfolio contract, not of
  ``StrategyEngine`` itself (that path is covered by
  ``tests/integration/test_live_protective_paths.py``).
- No test injects ``engine._positions[...]`` directly except S-03, whose
  entire point is to prove that legacy path still works when no source is
  attached (D8).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.engines.live import _INFLIGHT_SELL_MAX_AGE_S, LiveExecutionEngine
from trading.models import Order, RiskCheckResult, Signal
from trading.portfolio import PortfolioAccounting

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_BASE = "BTC"
_QUOTE = "USDT"
_RUN_ID = "ledger-test-run"
_LAST_PRICE = Decimal("50000")
_QTY_PRECISION = Decimal("0.00000001")

_DEFAULT_MARKET: dict[str, Any] = {
    "base": _BASE,
    "quote": _QUOTE,
    "precision": {"amount": 8, "price": 2},
}

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_mock_exchange(
    *,
    markets: dict[str, Any] | None = None,
    fetch_balance_response: dict[str, Any] | None = None,
    fetch_order_trades_response: list[dict[str, Any]] | None = None,
    has_fetch_order_trades: bool = True,
) -> MagicMock:
    exchange = MagicMock()
    exchange.id = "mock-exchange"
    exchange.markets = markets if markets is not None else {_SYMBOL: dict(_DEFAULT_MARKET)}

    exchange.create_order = AsyncMock(
        return_value={
            "id": "exch-001",
            "status": "open",
            "filled": "0",
            "average": None,
            "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_order = AsyncMock(
        return_value={
            "id": "exch-001",
            "status": "closed",
            "filled": None,
            "average": None,
            "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_ticker = AsyncMock(return_value={"last": str(_LAST_PRICE)})
    exchange.fetch_balance = AsyncMock(
        return_value=fetch_balance_response
        or {"total": {_QUOTE: 100000.0}, "free": {_QUOTE: 100000.0}}
    )
    exchange.load_markets = AsyncMock(return_value=exchange.markets)
    exchange.cancel_order = AsyncMock(return_value={"id": "exch-001", "status": "canceled"})
    exchange.close = AsyncMock(return_value=None)
    exchange.has = {"fetchOrderTrades": has_fetch_order_trades, "fetchMyTrades": True}
    exchange.fetch_order_trades = AsyncMock(return_value=fetch_order_trades_response or [])
    exchange.fetch_my_trades = AsyncMock(return_value=fetch_order_trades_response or [])
    # WP1.4b: the idempotent-submit cid lookup calls fetch_orders whenever
    # create_order's outcome is ambiguous -- default to "nothing found".
    exchange.fetch_orders = AsyncMock(return_value=[])
    return exchange


def _make_risk_manager_mock(
    *, approved: bool = True, adjusted_quantity: Decimal | None = None
) -> MagicMock:
    mock = MagicMock()

    def _pre_trade_check(*, order: Order, **_: Any) -> RiskCheckResult:
        qty = adjusted_quantity if adjusted_quantity is not None else order.quantity
        return RiskCheckResult(
            approved=approved,
            adjusted_quantity=qty if approved else Decimal("0"),
            rejection_reasons=[] if approved else ["test rejection"],
            warnings=[],
        )

    mock.pre_trade_check.side_effect = _pre_trade_check
    mock.calculate_position_size.return_value = Decimal("1.0")
    return mock


def _make_engine_with_source(
    *,
    markets: dict[str, Any] | None = None,
    fetch_balance_response: dict[str, Any] | None = None,
    fetch_order_trades_response: list[dict[str, Any]] | None = None,
    has_fetch_order_trades: bool = True,
    approved: bool = True,
    symbols: list[str] | None = None,
    initial_cash: Decimal = Decimal("100000"),
) -> tuple[LiveExecutionEngine, PortfolioAccounting, MagicMock, MagicMock]:
    """Build an engine with a *real* PortfolioAccounting attached as its
    LivePositionSource, exactly as StrategyEngine.__init__ does."""
    rm = _make_risk_manager_mock(approved=approved)
    ex = _make_mock_exchange(
        markets=markets,
        fetch_balance_response=fetch_balance_response,
        fetch_order_trades_response=fetch_order_trades_response,
        has_fetch_order_trades=has_fetch_order_trades,
    )
    engine = LiveExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        exchange=ex,
        enable_live_trading=True,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=initial_cash)
    engine.attach_position_source(portfolio, symbols=symbols or [_SYMBOL])
    return engine, portfolio, rm, ex


def _make_signal(
    *,
    direction: SignalDirection = SignalDirection.BUY,
    symbol: str = _SYMBOL,
    target_position: Decimal = Decimal("1000"),
    confidence: float = 1.0,
) -> Signal:
    return Signal(
        strategy_id="ledger-test-strategy",
        symbol=symbol,
        direction=direction,
        target_position=target_position,
        confidence=confidence,
    )


def _make_ccxt_trade(
    *,
    amount: str = "0.02",
    price: str = "50000",
    fee_cost: str = "0.6",
    fee_currency: str = _QUOTE,
    timestamp_ms: int | None = None,
    trade_id: str | None = None,
    order_id: str = "exch-001",
) -> dict[str, Any]:
    if timestamp_ms is None:
        timestamp_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    trade: dict[str, Any] = {
        "order": order_id,
        "amount": amount,
        "price": price,
        "fee": {"cost": fee_cost, "currency": fee_currency},
        "takerOrMaker": "taker",
        "timestamp": timestamp_ms,
    }
    if trade_id is not None:
        trade["id"] = trade_id
    return trade


async def _submit_buy(engine: LiveExecutionEngine, *, quantity: Decimal = Decimal("0.02")) -> Order:
    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
    )
    return await engine.submit_order(order)


# ===========================================================================
# R-01: BUY fill sets own qty and entry price with no `_positions` injection.
# ===========================================================================


@pytest.mark.asyncio
async def test_r01_buy_fill_sets_own_qty_and_all_in_entry_price() -> None:
    """R-01: a routed BUY fill sets PortfolioAccounting's own qty and an
    all-in (fee-inclusive) entry price -- with no `_positions` injection
    anywhere in this test."""
    trade = _make_ccxt_trade(amount="0.02", price="50000", fee_cost="0.6")
    engine, portfolio, _, _ = _make_engine_with_source(fetch_order_trades_response=[trade])

    order = await _submit_buy(engine)
    fills = await engine.get_fills(order.order_id)
    assert len(fills) == 1

    portfolio.update_position(fills[0], current_price=_LAST_PRICE)

    position = portfolio.get_position(_SYMBOL)
    assert position is not None
    assert position.quantity == Decimal("0.02")
    # All-in cost basis: (price*qty + fee) / qty = (50000*0.02 + 0.6) / 0.02
    expected_entry = (Decimal("50000") * Decimal("0.02") + Decimal("0.6")) / Decimal("0.02")
    assert position.average_entry_price == expected_entry.quantize(_QTY_PRECISION)


# ===========================================================================
# R-02: two partial deltas + a repeated poll -> correct VWAP, counted once.
# ===========================================================================


@pytest.mark.asyncio
async def test_r02_partial_deltas_vwap_counted_once() -> None:
    """R-02: two partial fills across two polls produce the correct VWAP
    and a third (repeated) poll routes nothing new (I9 idempotency)."""
    trade1 = _make_ccxt_trade(amount="0.01", price="50000", fee_cost="0.3", trade_id="t1")
    engine, portfolio, _, ex = _make_engine_with_source(fetch_order_trades_response=[trade1])

    order = await _submit_buy(engine, quantity=Decimal("0.02"))

    fills_1 = await engine.get_fills(order.order_id)
    assert len(fills_1) == 1
    portfolio.update_position(fills_1[0], current_price=_LAST_PRICE)

    # A second real trade has now appeared server-side alongside the first.
    trade2 = _make_ccxt_trade(amount="0.01", price="51000", fee_cost="0.306", trade_id="t2")
    ex.fetch_order_trades.return_value = [trade1, trade2]

    fills_2 = await engine.get_fills(order.order_id)
    assert len(fills_2) == 1, "trade1 must not be re-routed (I9)"
    assert fills_2[0].price == Decimal("51000")
    portfolio.update_position(fills_2[0], current_price=_LAST_PRICE)

    position = portfolio.get_position(_SYMBOL)
    assert position is not None
    assert position.quantity == Decimal("0.02")
    # VWAP all-in: ((50000*0.01+0.3) + (51000*0.01+0.306)) / 0.02
    expected_vwap = (
        (Decimal("50000") * Decimal("0.01") + Decimal("0.3"))
        + (Decimal("51000") * Decimal("0.01") + Decimal("0.306"))
    ) / Decimal("0.02")
    assert position.average_entry_price == expected_vwap.quantize(_QTY_PRECISION)

    # Repeated poll (nothing changed server-side): no new fills routed.
    fills_3 = await engine.get_fills(order.order_id)
    assert fills_3 == []


# ===========================================================================
# R-03: fee in base gives the net quantity; a full close never oversells.
# ===========================================================================


@pytest.mark.asyncio
async def test_r03_fee_in_base_nets_quantity_and_full_close_never_oversells() -> None:
    """R-03: a fee charged in the base currency nets out of the routed fill
    quantity (WP11-A-05), and the resulting full-close SELL quantity never
    exceeds that net (own) quantity -- the exchange-level InsufficientFunds
    guarantee is proven end-to-end by
    ``test_sell_capped_at_free_when_balance_partially_locked`` in the
    integration harness (R-23)."""
    # 0.02 BTC bought, 0.0002 BTC fee (base currency) -> net 0.0198 BTC.
    trade = _make_ccxt_trade(amount="0.02", price="50000", fee_cost="0.0002", fee_currency=_BASE)
    engine, portfolio, _, _ = _make_engine_with_source(fetch_order_trades_response=[trade])

    order = await _submit_buy(engine)
    fills = await engine.get_fills(order.order_id)
    assert len(fills) == 1
    assert fills[0].quantity == Decimal("0.0198")
    assert fills[0].fee_currency == _QUOTE  # normalised to quote (D6)

    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    own = portfolio.get_position(_SYMBOL).quantity
    assert own == Decimal("0.0198")

    sell_signal = _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    sell_orders = await engine.process_signal(sell_signal)
    assert len(sell_orders) == 1
    assert sell_orders[0].quantity <= own


# ===========================================================================
# R-04: SELL floors correctly in both decimal-places and step-size modes.
# ===========================================================================


def test_r04_floor_to_amount_precision_decimal_places_mode() -> None:
    """R-04a: DECIMAL_PLACES market (``precision.amount`` is an int) floors
    to that many decimals."""
    engine, _, _, _ = _make_engine_with_source(
        markets={_SYMBOL: {"base": _BASE, "quote": _QUOTE, "precision": {"amount": 4}}}
    )
    floored = engine._floor_to_amount_precision(_SYMBOL, Decimal("0.123456789"))
    assert floored == Decimal("0.1234")


def test_r04_floor_to_amount_precision_tick_size_mode() -> None:
    """R-04b (WP11-S-05): TICK_SIZE market. Real ccxt (4.5.40) reports
    ``precisionMode`` on the *exchange* object, not the market dict --
    Binance and Coinbase are both mode 4 (TICK_SIZE). Uses a realistic
    amount step of 1.0 (whole-unit lots): 5.37 floors to 5."""
    engine, _, _, ex = _make_engine_with_source(
        markets={_SYMBOL: {"base": _BASE, "quote": _QUOTE, "precision": {"amount": 1.0}}}
    )
    ex.precisionMode = 4

    floored = engine._floor_to_amount_precision(_SYMBOL, Decimal("5.37"))
    assert floored == Decimal("5")


@pytest.mark.asyncio
async def test_r04_sell_end_to_end_floors_in_tick_size_mode() -> None:
    """R-04c: the floor is actually applied on the SELL order path, not
    just in the helper. ``precisionMode`` lives on the exchange object
    (WP11-S-05), not the market dict."""
    engine, portfolio, _, ex = _make_engine_with_source(
        markets={
            _SYMBOL: {
                "base": _BASE,
                "quote": _QUOTE,
                "precision": {"amount": 0.001},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
            }
        },
        fetch_balance_response={"total": {_BASE: 1.0}, "free": {_BASE: 1.0}},
    )
    ex.precisionMode = 4
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.1237"))

    sell_signal = _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    orders = await engine.process_signal(sell_signal)
    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.123")


def _open_position(quantity: Decimal) -> Any:
    from trading.models import Position

    return Position(
        symbol=_SYMBOL,
        run_id=_RUN_ID,
        quantity=quantity,
        average_entry_price=_LAST_PRICE,
        current_price=_LAST_PRICE,
    )


# ===========================================================================
# R-05: only BTC/EUR (the run's own symbol) is touched when BTC/USD also
# exists -- no base-to-market scan (I6).
# ===========================================================================


@pytest.mark.asyncio
async def test_r05_sync_positions_only_touches_run_symbol_not_other_quote() -> None:
    """R-05: with BTC/USD and BTC/EUR both registered but the run only
    trading BTC/EUR, sync_positions must resolve the base asset via an
    exact ``markets["BTC/EUR"]`` lookup, never scanning BTC/USD."""
    markets = {
        "BTC/EUR": {"base": "BTC", "quote": "EUR", "precision": {"amount": 8}},
        "BTC/USD": {"base": "BTC", "quote": "USD", "precision": {"amount": 8}},
    }
    engine, portfolio, _, ex = _make_engine_with_source(
        markets=markets,
        fetch_balance_response={"total": {"BTC": 0.02}, "free": {"BTC": 0.02}},
        symbols=["BTC/EUR"],
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(Decimal("0.02")).model_copy(
        update={"symbol": "BTC/EUR"}
    )

    await engine.sync_positions()

    # sync_positions never creates entries from the balance (I2); the run's
    # own qty (0.02) equals the exchange total (0.02): no mismatch.
    assert not engine.reconcile_required
    assert "BTC/USD" not in engine.reconcile_required
    ex.fetch_balance.assert_called()


# ===========================================================================
# R-06: a foreign balance with no own fills creates no position; SELL
# rejected.
# ===========================================================================


@pytest.mark.asyncio
async def test_r06_foreign_balance_creates_no_position_and_sell_rejected() -> None:
    """R-06: a pre-existing exchange balance with no own fills never
    becomes a Position (I2), and a SELL against it is rejected exactly
    like a genuine no-position SELL."""
    engine, portfolio, _, _ = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.3}, "free": {_BASE: 0.3}},
    )

    await engine.sync_positions()
    assert portfolio.get_position(_SYMBOL) is None

    with capture_logs() as cap:
        orders = await engine.process_signal(
            _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
        )
    assert orders == []
    assert any(e.get("event") == "live.sell_no_position" for e in cap)


# ===========================================================================
# R-07: a fill priced 0/None flags the symbol, is not routed, BUYs rejected.
# ===========================================================================


@pytest.mark.asyncio
async def test_r07_zero_price_fill_flags_symbol_not_routed_and_blocks_buy() -> None:
    """R-07: a trade record with price <= 0 is never turned into a Fill,
    flags reconcile_required, and a subsequent BUY on that symbol is
    rejected."""
    bad_trade = _make_ccxt_trade(amount="0.02", price="0")
    engine, portfolio, _, _ = _make_engine_with_source(fetch_order_trades_response=[bad_trade])

    order = await _submit_buy(engine)
    with capture_logs() as cap:
        fills = await engine.get_fills(order.order_id)

    assert fills == []
    assert portfolio.get_position(_SYMBOL) is None
    assert engine.reconcile_required.get(_SYMBOL) == "fill_price_invalid"
    assert any(e.get("event") == "live.invalid_fill_price" for e in cap)

    buy_orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert buy_orders == []


# ===========================================================================
# R-08: own > total flags the symbol; own is never raised; dust doesn't
# flag.
# ===========================================================================


@pytest.mark.asyncio
async def test_r08_own_exceeds_total_flags_but_own_never_raised_and_dust_ignored() -> None:
    """R-08: own > exchange total (beyond tolerance) flags
    reconcile_required; sync_positions never raises own from the balance
    (I2); a sub-tolerance (dust) discrepancy never flags."""
    engine, portfolio, _, _ex = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.01}, "free": {_BASE: 0.01}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))  # own > total

    await engine.sync_positions()

    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"
    # own is never raised by sync_positions (I2) -- still 0.02, not 0.01.
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.02")

    # Dust: own exceeds total by less than one amount step (1e-8 default) --
    # must not flag.
    engine2, portfolio2, _, _ = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.01999999}, "free": {_BASE: 0.01999999}},
    )
    portfolio2._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    await engine2.sync_positions()
    assert _SYMBOL not in engine2.reconcile_required


# ===========================================================================
# R-10: the balance cache is invalidated after a fill.
# ===========================================================================


@pytest.mark.asyncio
async def test_r10_balance_cache_invalidated_after_fill() -> None:
    """R-10: a BUY's create_order call invalidates the 10s balance cache so
    a same-bar SELL cap reads a fresh (post-fill) balance."""
    engine, _, _, ex = _make_engine_with_source()

    await engine._fetch_balance_cached()
    assert ex.fetch_balance.call_count == 1
    await engine._fetch_balance_cached()
    assert ex.fetch_balance.call_count == 1  # served from cache

    await _submit_buy(engine)

    await engine._fetch_balance_cached()
    assert ex.fetch_balance.call_count == 2  # cache was invalidated by the fill


# ===========================================================================
# R-11: balance fetch fails -> SELL still goes out, floored.
# ===========================================================================


@pytest.mark.asyncio
async def test_r11_balance_fetch_failure_still_allows_floored_sell() -> None:
    """R-11: when the balance fetch fails, the SELL is never blocked --
    `capped` falls back to `own` (I1's "never block the exit"), floored to
    the market precision. Uses a half-close signal so both the
    balance-failure fallback and the precision floor are exercised
    together."""
    engine, portfolio, _, ex = _make_engine_with_source(
        markets={_SYMBOL: {"base": _BASE, "quote": _QUOTE, "precision": {"amount": 4}}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.12345678"))
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    half_close = _make_signal(
        direction=SignalDirection.SELL,
        target_position=Decimal("3086.41945"),
        confidence=1.0,
    )  # (3086.41945 / 50000) ≈ 0.0617283890 -- about half of 0.12345678
    orders = await engine.process_signal(half_close)

    assert len(orders) == 1
    assert orders[0].quantity <= Decimal("0.12345678")
    assert orders[0].quantity == orders[0].quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)


# ===========================================================================
# R-12: trades fail on a FILLED order -> one synthesised fill, no double
# count later.
# ===========================================================================


@pytest.mark.asyncio
async def test_r12_synthesized_fill_on_filled_order_when_trades_unavailable() -> None:
    """R-12: fetch_order_trades raising on a locally-FILLED order
    synthesises exactly one fill for the unrouted quantity at
    average_fill_price with fee 0, and permanently ignores any later
    (real) trade record for that order."""
    engine, portfolio, _, ex = _make_engine_with_source()

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
    )
    order = await engine.submit_order(order)
    # Force the order into FILLED with a known filled_quantity/average, as
    # LiveExecutionEngine.submit_order/_reconcile_order would after a real
    # Coinbase fill.
    order = order.model_copy(
        update={
            "status": OrderStatus.FILLED,
            "filled_quantity": Decimal("0.02"),
            "average_fill_price": Decimal("50000"),
        }
    )
    engine._orders[order.order_id] = order

    ex.fetch_order_trades.side_effect = Exception("trades endpoint down")

    with capture_logs() as cap:
        fills = await engine.get_fills(order.order_id)

    assert len(fills) == 1
    assert fills[0].quantity == Decimal("0.02")
    assert fills[0].price == Decimal("50000")
    assert fills[0].fee == Decimal("0")
    assert any(e.get("event") == "live.fill_synthesized" for e in cap)

    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.02")

    # The trades endpoint later "recovers" and would return a matching real
    # trade for the same order -- it must be permanently ignored.
    ex.fetch_order_trades.side_effect = None
    ex.fetch_order_trades.return_value = [
        _make_ccxt_trade(amount="0.02", price="50000", fee_cost="0.6")
    ]
    fills_again = await engine.get_fills(order.order_id)
    assert fills_again == []


# ===========================================================================
# S-01: a late fill is routed once through check_resting_orders.
# ===========================================================================


@pytest.mark.asyncio
async def test_s01_late_fill_routed_once_via_check_resting_orders() -> None:
    """S-01: an order still OPEN after process_signal's own wait is picked
    up by check_resting_orders on a later bar, and its fill is routed
    exactly once."""
    engine, portfolio, _, ex = _make_engine_with_source(fetch_order_trades_response=[])

    order = await _submit_buy(engine)
    assert order.status == OrderStatus.OPEN  # still open -- no trades yet

    # No fill yet: fetch_order still reports open, no trades.
    ex.fetch_order.return_value = {
        "id": "exch-001",
        "status": "open",
        "filled": None,
        "average": None,
    }
    resting = await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)
    assert order.order_id in {o.order_id for o in resting}
    fills = await engine.get_fills(order.order_id)
    assert fills == []  # exchange still has no trades -- nothing to route yet

    # A later poll: the exchange now reports the order closed, with a trade.
    ex.fetch_order.return_value = {
        "id": "exch-001",
        "status": "closed",
        "filled": "0.02",
        "average": "50000",
    }
    trade = _make_ccxt_trade(amount="0.02", price="50000", fee_cost="0.6")
    ex.fetch_order_trades.return_value = [trade]

    resting_2 = await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)
    matched = [o for o in resting_2 if o.order_id == order.order_id]
    assert matched, "the now-FILLED order must be offered for a get_fills check"

    fills_2 = await engine.get_fills(order.order_id)
    assert len(fills_2) == 1
    portfolio.update_position(fills_2[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.02")

    # A third poll must not re-route the same trade (I9).
    resting_3 = await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)
    assert order.order_id not in {o.order_id for o in resting_3}


# ===========================================================================
# S-02: a flagged symbol still SELLs, capped at min(own, free).
# ===========================================================================


@pytest.mark.asyncio
async def test_s02_flagged_symbol_still_sells_capped_at_min_own_free() -> None:
    """S-02: reconcile_required blocks BUYs only -- a flagged symbol's
    protective SELL still reaches the exchange, capped at min(own, free)."""
    engine, portfolio, _, _ = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.015}, "free": {_BASE: 0.01}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    engine._flag_reconcile(_SYMBOL, "own_exceeds_exchange_total")

    sell_orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )
    assert len(sell_orders) == 1
    assert sell_orders[0].quantity == Decimal("0.01")  # capped at free, not own

    buy_orders = await engine.process_signal(_make_signal(direction=SignalDirection.BUY))
    assert buy_orders == []  # still blocked


# ===========================================================================
# S-03: with no source attached, the legacy `_positions` path still works.
# ===========================================================================


@pytest.mark.asyncio
async def test_s03_no_source_attached_legacy_positions_path_still_works() -> None:
    """S-03 (D8): with no LivePositionSource attached, injecting
    `engine._positions[symbol]` directly still drives SELL sizing exactly
    as before WP1.1 -- the 8 pre-existing unit-test injection sites in
    test_live_execution.py / test_sprint51_cycle1_sizing_contract.py must
    keep passing unmodified."""
    rm = _make_risk_manager_mock()
    ex = _make_mock_exchange()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        exchange=ex,
        enable_live_trading=True,
    )
    # No attach_position_source call.
    from trading.models import Position

    engine._positions[_SYMBOL] = Position(
        symbol=_SYMBOL,
        run_id=_RUN_ID,
        quantity=Decimal("0.5"),
        average_entry_price=Decimal("48000"),
        current_price=_LAST_PRICE,
    )

    orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL


# ===========================================================================
# S-04: daily PnL comes from the portfolio.
# ===========================================================================


def test_s04_daily_pnl_delegates_to_portfolio() -> None:
    """S-04: _calculate_daily_pnl() delegates to the attached source's
    get_daily_pnl() rather than summing the (always-empty) legacy
    `_positions` cache."""
    engine, portfolio, _, _ = _make_engine_with_source()
    portfolio._daily_pnl = Decimal("42.5")

    assert engine._calculate_daily_pnl() == Decimal("42.5")
    assert engine._calculate_daily_pnl() == portfolio.get_daily_pnl()


# ===========================================================================
# S-05: a missing fee currency defaults to the quote currency.
# ===========================================================================


def test_s05_missing_fee_currency_defaults_to_quote() -> None:
    """S-05 (D6): a trade with no fee.currency defaults to the market's
    quote currency, not a hardcoded "USDT"."""
    engine, _, _, _ = _make_engine_with_source(
        markets={_SYMBOL: {"base": _BASE, "quote": "EUR", "precision": {"amount": 8}}},
    )
    trade = {"fee": {"cost": "0.5"}}  # no "currency" key
    _, fee_currency = engine._extract_fee_from_ccxt(trade, _SYMBOL)
    assert fee_currency == "EUR"


# ===========================================================================
# S-06: live enabled, no source attached -> critical log.
# ===========================================================================


@pytest.mark.asyncio
async def test_s06_no_source_attached_logs_critical_and_fails_closed_on_start() -> None:
    """S-06 (D8) + WP11-S-09 (round 2): on_start() with live trading
    enabled and no LivePositionSource attached logs
    live.position_source_missing at critical level AND raises RuntimeError
    (fail-closed) -- a real live run must never fall back to the legacy
    (never fill-populated) _positions cache; that fallback is for unit
    tests that never call on_start() (S-03)."""
    rm = _make_risk_manager_mock()
    ex = _make_mock_exchange()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        exchange=ex,
        enable_live_trading=True,
    )

    with capture_logs() as cap:
        with pytest.raises(RuntimeError, match="LivePositionSource"):
            await engine.on_start()

    critical_events = [
        e
        for e in cap
        if e.get("event") == "live.position_source_missing" and e.get("log_level") == "critical"
    ]
    assert critical_events, f"expected a critical live.position_source_missing log; got {cap!r}"
    # Fail-closed happens BEFORE load_markets -- no exchange call was made.
    ex.load_markets.assert_not_called()


# ===========================================================================
# S-07: balance_unavailable clears itself; a mismatch flag stays.
# ===========================================================================


@pytest.mark.asyncio
async def test_s07_balance_unavailable_clears_but_mismatch_flag_persists() -> None:
    """S-07 (I4): balance_unavailable is cleared automatically on the next
    successful sync; every other reconcile_required reason (e.g. an
    own/total mismatch) persists even after the underlying condition
    resolves -- only an operator or WP1.8 clears it."""
    engine, portfolio, _, ex = _make_engine_with_source()
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    await engine.sync_positions()
    assert engine.reconcile_required.get(_SYMBOL) == "balance_unavailable"

    ex.fetch_balance.side_effect = None
    ex.fetch_balance.return_value = {"total": {_BASE: 0.0}, "free": {_BASE: 0.0}}
    await engine.sync_positions()
    assert _SYMBOL not in engine.reconcile_required  # cleared automatically

    # Now create a genuine mismatch, then resolve the underlying condition.
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    ex.fetch_balance.return_value = {"total": {_BASE: 0.0}, "free": {_BASE: 0.0}}
    await engine.sync_positions()
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"

    # The exchange now shows enough balance -- the flag must NOT clear
    # itself (I4: only balance_unavailable does).
    ex.fetch_balance.return_value = {"total": {_BASE: 0.02}, "free": {_BASE: 0.02}}
    await engine.sync_positions()
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"


# ===========================================================================
# S-07 (extra): an unparseable balance VALUE (not a fetch exception) must
# be treated as unavailable, never coerced to zero.
# ===========================================================================


@pytest.mark.asyncio
async def test_s07_unparseable_balance_value_treated_as_unavailable_not_zero() -> None:
    """S-07 (WP11-S-07): an unparseable balance value (e.g. a malformed
    string from a misbehaving exchange) must be treated as unavailable
    (own is returned uncapped), never coerced to zero (which would look
    like an external-holdings-only balance and could wrongly flag
    own_exceeds_exchange_total)."""
    engine, portfolio, _, _ex = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: "not-a-number"}, "free": {_BASE: "also-bad"}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )

    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.02")  # capped == own (balance treated unavailable)
    assert not engine.reconcile_required  # no false mismatch from a coerced-zero balance


# ===========================================================================
# Round 2 security regression tests (WP11-S-01..09)
# ===========================================================================


@pytest.mark.asyncio
async def test_p1_partial_order_trade_fetch_fails_no_duplicate() -> None:
    """P1 (WP11-S-01, CRITICAL): a PARTIAL order whose trade fetch fails
    must not have its already-routed fill handed back again.

    Pre-fix: ``_synthesize_unrouted_fill`` returned the order's cached
    fill history for any non-FILLED status, so this returned the SAME
    0.01 fill a second (and third) time."""
    trade = _make_ccxt_trade(amount="0.01", price="50000", fee_cost="0.3", trade_id="t1")
    engine, portfolio, _, ex = _make_engine_with_source(fetch_order_trades_response=[trade])

    order = await _submit_buy(engine, quantity=Decimal("0.02"))
    fills = await engine.get_fills(order.order_id)
    assert len(fills) == 1
    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.01")

    # Order is genuinely PARTIAL (0.01 of 0.02 filled); the trade fetch now
    # starts failing (e.g. a rate limit / transient network error).
    engine._orders[order.order_id] = order.model_copy(
        update={"status": OrderStatus.PARTIAL, "filled_quantity": Decimal("0.01")}
    )
    ex.fetch_order_trades.side_effect = Exception("rate limited")

    for _ in range(3):
        fills_again = await engine.get_fills(order.order_id)
        assert fills_again == [], "must never re-deliver the already-routed fill"

    # No caller ever received a second fill to route -- own stays 0.01.
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.01")


@pytest.mark.asyncio
async def test_p2_canceled_partial_empty_trades_three_bars_own_stays_bounded() -> None:
    """P2 (WP11-S-01, CRITICAL): a CANCELED order carrying a partial fill,
    polled across 3 bars with empty trades each time, must never grow own
    quantity beyond what was genuinely routed once.

    Pre-fix: own went 0.01 -> 0.04 after 3 bars (the same 0.01 fill handed
    back on every poll, with no bound)."""
    trade = _make_ccxt_trade(amount="0.01", price="50000", fee_cost="0.3", trade_id="t1")
    engine, portfolio, _, ex = _make_engine_with_source(fetch_order_trades_response=[trade])

    order = await _submit_buy(engine, quantity=Decimal("0.02"))
    fills = await engine.get_fills(order.order_id)
    assert len(fills) == 1
    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.01")

    # The order gets canceled while carrying that 0.01 partial fill.
    engine._orders[order.order_id] = order.model_copy(
        update={"status": OrderStatus.CANCELED, "filled_quantity": Decimal("0.01")}
    )
    ex.fetch_order_trades.return_value = []  # trades endpoint now returns nothing

    for bar in range(3):
        fills_again = await engine.get_fills(order.order_id)
        assert fills_again == [], f"bar {bar}: must not re-synthesise an already-routed fill"

    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.01"), (
        "own quantity must stay bounded at the genuinely-routed 0.01, not "
        "grow with every empty-trades poll (pre-fix: 0.01 -> 0.04 over 3 bars)"
    )


@pytest.mark.asyncio
async def test_p3_fee_in_base_no_phantom_fill_and_stops_being_polled() -> None:
    """P3 (WP11-S-02, HIGH): a base-currency fee must not make an order
    look permanently under-routed.

    Pre-fix: the "routed quantity" check compared NET routed quantity
    (``sum(Fill.quantity)``, fee already subtracted) against the GROSS
    ``filled_quantity``, so the order was re-offered forever and a phantom
    fill was eventually synthesised for the fee gap (own 0.01998 ->
    0.02000)."""
    trade = _make_ccxt_trade(
        amount="0.02", price="50000", fee_cost="0.0002", fee_currency=_BASE, trade_id="t1"
    )
    engine, portfolio, _, ex = _make_engine_with_source(fetch_order_trades_response=[trade])

    order = await _submit_buy(engine, quantity=Decimal("0.02"))
    fills = await engine.get_fills(order.order_id)
    assert len(fills) == 1
    assert fills[0].quantity == Decimal("0.0198")  # net of the 0.0002 base fee
    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.0198")

    # Mark the order FILLED at its true gross quantity, as a real
    # reconcile would (order.filled_quantity is always gross on the
    # exchange side).
    engine._orders[order.order_id] = order.model_copy(
        update={
            "status": OrderStatus.FILLED,
            "filled_quantity": Decimal("0.02"),
            "average_fill_price": _LAST_PRICE,
        }
    )

    # check_resting_orders must NOT re-offer this order: its gross-routed
    # quantity (0.02, tracked from the raw trade amount) already equals
    # filled_quantity (0.02) even though the NET fill was only 0.0198.
    candidates = await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)
    assert order.order_id not in {o.order_id for o in candidates}

    # Even if get_fills were called again (trades now empty), no phantom
    # fill for the fee gap is synthesised.
    ex.fetch_order_trades.return_value = []
    fills_again = await engine.get_fills(order.order_id)
    assert fills_again == []
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.0198"), (
        "own must stay at the true net quantity -- no phantom fill for the fee gap"
    )


@pytest.mark.asyncio
async def test_p4_second_sell_blocked_while_first_still_in_flight() -> None:
    """P4 (WP11-S-03, HIGH): with the first SELL still OPEN (unrouted) and
    a 0.5 BTC external holding on the exchange, a second SELL signal for
    the same symbol must be rejected (own_avail == 0), never taken from
    the external holding, and must not raise a false
    own_exceeds_exchange_total flag from comparing stale ``own`` against
    the exchange total while the first SELL hasn't settled yet."""
    engine, portfolio, _, ex = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.52}, "free": {_BASE: 0.52}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))  # bot's own qty
    # 0.5 BTC external + the bot's own 0.02 = 0.52 total/free on the exchange.

    # First SELL order: submitted, but the exchange has not reported a
    # fill yet (still OPEN) -- nothing routed into _routed_gross_qty for it.
    first_order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
    )
    first_order = first_order.model_copy(update={"status": OrderStatus.OPEN})
    engine._orders[first_order.order_id] = first_order

    with capture_logs() as cap:
        second_sell_orders = await engine.process_signal(
            _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
        )

    assert second_sell_orders == [], (
        "must not sell the same quantity twice while the first SELL is in flight"
    )
    assert any(e.get("event") == "live.sell_inflight_pending" for e in cap)
    assert not engine.reconcile_required, (
        f"own_avail (0) must be used for the mismatch check, not stale own "
        f"(0.02) vs total (0.52) -- got {dict(engine.reconcile_required)!r}"
    )
    assert ex.fetch_ticker.await_count == 0  # never got past the held-quantity guard


@pytest.mark.asyncio
async def test_p6_get_fills_atomic_commit_on_per_trade_parse_error() -> None:
    """P6 (WP11-S-04): a batch of 2 trades where the SECOND one has an
    unparseable field must not lose the FIRST trade's fill -- both are
    parsed into a local buffer and only committed once the whole batch
    finishes; the bad trade is flagged ``fill_parse_failed`` and left
    unrouted (retried on the next call), never lost."""
    good_trade = _make_ccxt_trade(amount="0.01", price="50000", fee_cost="0.3", trade_id="t-good")
    bad_trade = _make_ccxt_trade(
        amount="not-a-number", price="50000", fee_cost="0.3", trade_id="t-bad"
    )
    engine, portfolio, _, ex = _make_engine_with_source(
        fetch_order_trades_response=[good_trade, bad_trade]
    )

    order = await _submit_buy(engine, quantity=Decimal("0.02"))

    with capture_logs() as cap:
        fills = await engine.get_fills(order.order_id)

    assert len(fills) == 1, "the good trade must still be delivered despite the bad one"
    assert fills[0].quantity == Decimal("0.01")
    assert any(e.get("event") == "live.fill_parse_failed" for e in cap)
    assert engine.reconcile_required.get(_SYMBOL) == "fill_parse_failed"

    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.01")

    # The bad trade was never marked routed -- fixing it upstream and
    # polling again must deliver it (retried, not lost).
    fixed_trade = _make_ccxt_trade(amount="0.01", price="50000", fee_cost="0.3", trade_id="t-bad")
    ex.fetch_order_trades.return_value = [good_trade, fixed_trade]
    fills_retry = await engine.get_fills(order.order_id)
    assert len(fills_retry) == 1
    assert fills_retry[0].quantity == Decimal("0.01")


# ===========================================================================
# Round 3 security regression tests (WP11-S-R2-01..03)
# ===========================================================================


@pytest.mark.asyncio
async def test_p7_unknown_sell_reserved_until_never_placed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P7 rewritten for WP1.4b (I5b(a), supersedes WP11-S-R2-01): a SELL
    whose create_order outcome is still AMBIGUOUS (here, a ValueError with
    no matching order ever found by the cid lookup) now RESERVES its full
    quantity -- unlike the pre-WP1.4b behaviour (which stopped reserving
    it immediately). A second SELL signal is therefore capped to 0 and
    blocked; re-selling the same quantity out of a possibly-external
    holding would breach I5. The reservation is released ONLY once
    ``_resolve_unknown_submits`` gathers D7's evidence (two absent lookups
    >= 10s apart, the last >= the settle window after submit) and
    confirms ``never_placed`` -- never on a timer (W3)."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, portfolio, _, ex = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    ex.create_order.side_effect = ValueError("exchange rejected the order shape")
    stuck = await engine.submit_order(
        Order(
            client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
            run_id=_RUN_ID,
            symbol=_SYMBOL,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.02"),
        )
    )
    assert stuck.status == OrderStatus.PENDING_SUBMIT
    assert stuck.order_id not in engine._exchange_order_map
    assert engine.reconcile_required.get(_SYMBOL) == "sell_submit_unknown"
    assert stuck.order_id in engine._unknown_submits

    # I5b(a): the unknown SELL's full quantity is reserved -- a second
    # SELL is capped to 0 and blocked, never re-selling the same quantity.
    second_orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )
    assert second_orders == []

    # Simulate D7 evidence: age the entry past the settle window with two
    # absent lookups >= 10s apart (real time was never actually waited --
    # asyncio.sleep is mocked above -- so the clock is moved directly,
    # exactly as a per-bar resolver running much later would observe).
    entry = engine._unknown_submits[stuck.order_id]
    entry.submit_at = datetime.now(UTC) - timedelta(seconds=130)
    entry.first_absent_at = entry.submit_at + timedelta(seconds=1)
    entry.last_absent_at = entry.submit_at + timedelta(seconds=5)
    entry.absent_count = 1

    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine._orders[stuck.order_id].status == OrderStatus.REJECTED
    assert stuck.order_id not in engine._unknown_submits
    assert engine.reconcile_required.get(_SYMBOL) is None

    ex.create_order.side_effect = None  # the next attempt reaches the exchange normally
    third_orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )
    assert len(third_orders) == 1, "the next SELL is submitted once the reservation releases"


@pytest.mark.asyncio
async def test_p8_stale_open_sell_with_unreconcilable_state_not_reserved() -> None:
    """P8 (WP11-S-R2-02, MEDIUM regression): an OPEN SELL whose state can
    no longer be verified (fetch_order persistently raising OrderNotFound,
    so ``updated_at`` never advances) must stop being reserved once it is
    older than ``_INFLIGHT_SELL_MAX_AGE_S``."""
    engine, portfolio, _, ex = _make_engine_with_source(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    stale_order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
    )
    stale_order = stale_order.model_copy(
        update={
            "status": OrderStatus.OPEN,
            "updated_at": datetime.now(UTC) - timedelta(seconds=_INFLIGHT_SELL_MAX_AGE_S + 1),
        }
    )
    engine._orders[stale_order.order_id] = stale_order
    engine._exchange_order_map[stale_order.order_id] = "exch-stale-sell"
    ex.fetch_order.side_effect = Exception("OrderNotFound: no such order")

    second_orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
    )

    assert len(second_orders) == 1, "the next SELL must still be submitted"
    assert engine.reconcile_required.get(_SYMBOL) == "sell_order_state_unknown"


@pytest.mark.asyncio
async def test_p9_negative_fee_clamped_to_zero_does_not_fail_batch() -> None:
    """P9 (WP11-S-R2-03, LOW): a negative fee (a maker rebate) must not
    crash Fill's ``ge=0`` constraint or abort the whole atomic-commit
    batch -- it is clamped to 0, logged, and the fill is still booked."""
    rebate_trade = _make_ccxt_trade(
        amount="0.02", price="50000", fee_cost="-0.05", trade_id="t-rebate"
    )
    engine, portfolio, _, _ = _make_engine_with_source(fetch_order_trades_response=[rebate_trade])

    order = await _submit_buy(engine, quantity=Decimal("0.02"))

    with capture_logs() as cap:
        fills = await engine.get_fills(order.order_id)

    assert len(fills) == 1
    assert fills[0].fee == Decimal("0")
    assert any(e.get("event") == "live.fee_rebate_ignored" for e in cap)

    portfolio.update_position(fills[0], current_price=_LAST_PRICE)
    assert portfolio.get_position(_SYMBOL).quantity == Decimal("0.02")
