"""
tests/unit/test_paper_execution.py
------------------------------------
Comprehensive unit tests for PaperExecutionEngine.

Module under test
-----------------
    packages/trading/engines/paper.py -- PaperExecutionEngine

Coverage groups
---------------
1.  TestConstructorAndProperties  -- Constructor defaults, custom params, property access (5 tests)
2.  TestMarketOrderSubmission     -- MARKET order fills, slippage, fees, cash/position updates (8 tests)
3.  TestLimitOrderSubmission      -- LIMIT order immediate fill vs resting, check_resting_orders (6 tests)
4.  TestCancelOrders              -- Cancel open, filled, and unknown orders (3 tests)
5.  TestSignalProcessing          -- BUY/SELL/HOLD signals, risk rejection, SELL without position (5 tests)
6.  TestPositionTracking          -- New position, average-up, partial sell, full close, unrealised PnL (5 tests)
7.  TestCashAndEquity             -- Equity calculation, fee deductions, cumulative cash tracking (4 tests)
8.  TestLifecycle                 -- on_start, on_stop (cancels open orders), set_last_price, get_open_orders (4 tests)

Design notes
------------
- _make_engine() factory returns (PaperExecutionEngine, MagicMock) tuple.
  The mock has .params.taker_fee_pct = 0.001, .params.maker_fee_pct = 0.0005 to
  match RiskParameters defaults and enable fee assertions.
- All price/quantity/fee assertions use Decimal arithmetic to avoid float imprecision.
- asyncio_mode = "auto" is set in pyproject.toml; @pytest.mark.asyncio is included
  for explicitness but not strictly required.
- Tests never mock internal engine methods; only the injected BaseRiskManager is mocked.
- The MagicMock for risk_manager is configured so pre_trade_check() returns an approved
  RiskCheckResult by default. Individual tests override this to test rejection paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.engines.paper import PaperExecutionEngine
from trading.execution import InvalidOrderTransitionError
from trading.models import Order, RiskCheckResult, Signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_RUN_ID = "paper-test-run-001"
_INITIAL_CASH = Decimal("10000")
_LAST_PRICE = Decimal("50000")
_SLIPPAGE_BPS = 5
_TAKER_FEE_PCT = 0.006   # 0.60% (Coinbase Advanced Trade lowest tier)
_MAKER_FEE_PCT = 0.004   # 0.40%

# Precision constant matching paper.py internal constant
_PRICE_PRECISION = Decimal("0.00000001")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_risk_manager_mock(
    *,
    approved: bool = True,
    adjusted_quantity: Decimal | None = None,
    position_size: Decimal = Decimal("0.02"),
) -> MagicMock:
    """
    Build a fully-configured MagicMock for BaseRiskManager.

    Configures:
    - params.taker_fee_pct / maker_fee_pct matching RiskParameters defaults (0.60%/0.40%)
    - pre_trade_check() returning an approved RiskCheckResult by default
    - calculate_position_size() returning `position_size`
    """
    mock = MagicMock()

    # Params used by _calculate_fee()
    mock.params.taker_fee_pct = _TAKER_FEE_PCT
    mock.params.maker_fee_pct = _MAKER_FEE_PCT

    # pre_trade_check default: approved, quantity unchanged
    qty = adjusted_quantity if adjusted_quantity is not None else position_size
    mock.pre_trade_check.return_value = RiskCheckResult(
        approved=approved,
        adjusted_quantity=qty,
        rejection_reasons=[] if approved else ["test rejection"],
        warnings=[],
    )

    # calculate_position_size default
    mock.calculate_position_size.return_value = position_size

    return mock


def _make_engine(
    *,
    initial_cash: Decimal = _INITIAL_CASH,
    slippage_bps: int = _SLIPPAGE_BPS,
    fill_latency_ms: int = 0,
    approved: bool = True,
    position_size: Decimal = Decimal("0.02"),
) -> tuple[PaperExecutionEngine, MagicMock]:
    """
    Factory that returns (engine, risk_manager_mock).

    The engine is fully constructed and ready for use.  Set `last_price`
    via engine.set_last_price() before submitting orders.
    """
    rm = _make_risk_manager_mock(
        approved=approved,
        position_size=position_size,
        adjusted_quantity=position_size,
    )
    engine = PaperExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        fill_latency_ms=fill_latency_ms,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
    )
    return engine, rm


def _make_market_order(
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.1"),
    symbol: str = _SYMBOL,
) -> Order:
    """Create a MARKET order in NEW status."""
    return Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
    )


def _make_limit_order(
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.1"),
    price: Decimal = Decimal("50000"),
    symbol: str = _SYMBOL,
) -> Order:
    """Create a LIMIT order in NEW status."""
    return Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
    )


def _make_buy_signal(
    *,
    confidence: float = 1.0,
    symbol: str = _SYMBOL,
) -> Signal:
    """Create a BUY signal."""
    return Signal(
        strategy_id="test-strategy",
        symbol=symbol,
        direction=SignalDirection.BUY,
        target_position=Decimal("1000"),
        confidence=confidence,
    )


def _make_sell_signal(*, symbol: str = _SYMBOL) -> Signal:
    """Create a SELL signal."""
    return Signal(
        strategy_id="test-strategy",
        symbol=symbol,
        direction=SignalDirection.SELL,
        target_position=Decimal("0"),
        confidence=1.0,
    )


def _make_hold_signal(*, symbol: str = _SYMBOL) -> Signal:
    """Create a HOLD signal."""
    return Signal(
        strategy_id="test-strategy",
        symbol=symbol,
        direction=SignalDirection.HOLD,
        target_position=Decimal("0"),
        confidence=1.0,
    )


def _expected_slippage_price(price: Decimal, side: OrderSide, bps: int = _SLIPPAGE_BPS) -> Decimal:
    """Compute expected slippage-adjusted price matching paper.py formula."""
    factor = Decimal(bps) / Decimal("10000")
    if side == OrderSide.BUY:
        adjusted = price * (Decimal("1") + factor)
    else:
        adjusted = price * (Decimal("1") - factor)
    return adjusted.quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)


def _expected_fee(quantity: Decimal, price: Decimal, is_maker: bool) -> Decimal:
    """Compute expected fee matching paper.py formula."""
    notional = quantity * price
    fee_rate = Decimal(str(_MAKER_FEE_PCT if is_maker else _TAKER_FEE_PCT))
    return (notional * fee_rate).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)


# ===========================================================================
# Group 1: Constructor & Properties
# ===========================================================================


class TestConstructorAndProperties:
    """Verify that PaperExecutionEngine stores all constructor arguments correctly."""

    def test_default_construction_stores_risk_manager(self) -> None:
        """risk_manager property returns the exact instance injected at construction."""
        engine, rm = _make_engine()
        assert engine.risk_manager is rm

    def test_custom_initial_cash_reflected_in_cash_property(self) -> None:
        """Custom initial_cash is immediately readable through .cash."""
        custom_cash = Decimal("25000")
        engine, _ = _make_engine(initial_cash=custom_cash)
        assert engine.cash == custom_cash

    def test_positions_empty_on_construction(self) -> None:
        """The positions dict must be empty when no orders have been placed."""
        engine, _ = _make_engine()
        assert engine.positions == {}

    def test_slippage_bps_stored_correctly(self) -> None:
        """Custom slippage_bps is stored and influences fill price calculations."""
        engine, _ = _make_engine(slippage_bps=10)
        assert engine._slippage_bps == 10

    def test_fill_latency_ms_stored_correctly(self) -> None:
        """Custom fill_latency_ms is stored on the engine instance."""
        engine, _ = _make_engine(fill_latency_ms=50)
        assert engine._fill_latency_ms == 50


# ===========================================================================
# Group 2: MARKET Order Submission
# ===========================================================================


class TestMarketOrderSubmission:
    """Verify MARKET order fill simulation: price, fees, cash, and position updates."""

    @pytest.mark.asyncio
    async def test_market_buy_order_status_becomes_filled(self) -> None:
        """A submitted MARKET BUY order must transition to FILLED status."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))

        result = await engine.submit_order(order)

        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_market_sell_order_status_becomes_filled(self) -> None:
        """A MARKET SELL order also transitions to FILLED."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # First, create a position by buying
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await engine.submit_order(buy_order)

        sell_order = _make_market_order(side=OrderSide.SELL, quantity=Decimal("0.1"))
        result = await engine.submit_order(sell_order)

        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_market_buy_fill_price_includes_slippage(self) -> None:
        """BUY fill price = last_price * (1 + slippage_bps/10000), rounded to 8dp."""
        engine, _ = _make_engine(slippage_bps=5)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))

        result = await engine.submit_order(order)

        expected = _expected_slippage_price(_LAST_PRICE, OrderSide.BUY, bps=5)
        assert result.average_fill_price == expected

    @pytest.mark.asyncio
    async def test_market_sell_fill_price_decremented_by_slippage(self) -> None:
        """SELL fill price = last_price * (1 - slippage_bps/10000), rounded to 8dp."""
        engine, _ = _make_engine(slippage_bps=5)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Build a position first
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await engine.submit_order(buy_order)

        sell_order = _make_market_order(side=OrderSide.SELL, quantity=Decimal("0.1"))
        result = await engine.submit_order(sell_order)

        expected = _expected_slippage_price(_LAST_PRICE, OrderSide.SELL, bps=5)
        assert result.average_fill_price == expected

    @pytest.mark.asyncio
    async def test_market_buy_fill_fee_calculated_at_taker_rate(self) -> None:
        """Fill fee for a MARKET BUY equals notional * taker_fee_pct, rounded to 8dp."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)

        result = await engine.submit_order(order)

        fill_price = result.average_fill_price
        fills = await engine.get_fills(result.order_id)
        assert len(fills) == 1
        expected_fee = _expected_fee(qty, fill_price, is_maker=False)
        assert fills[0].fee == expected_fee

    @pytest.mark.asyncio
    async def test_market_buy_decrements_cash(self) -> None:
        """After a BUY fill, cash decreases by (quantity * fill_price + fee)."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)

        result = await engine.submit_order(order)

        fills = await engine.get_fills(result.order_id)
        fill = fills[0]
        expected_cash = _INITIAL_CASH - (qty * fill.price + fill.fee)
        assert engine.cash == expected_cash

    @pytest.mark.asyncio
    async def test_market_sell_increments_cash(self) -> None:
        """After a SELL fill, cash increases by (quantity * fill_price - fee)."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")

        # First buy to open position
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        await engine.submit_order(buy_order)
        cash_after_buy = engine.cash

        sell_order = _make_market_order(side=OrderSide.SELL, quantity=qty)
        result = await engine.submit_order(sell_order)

        fills = await engine.get_fills(result.order_id)
        fill = fills[0]
        expected_cash = cash_after_buy + (qty * fill.price - fill.fee)
        assert engine.cash == expected_cash

    @pytest.mark.asyncio
    async def test_market_buy_creates_position_with_correct_quantity(self) -> None:
        """A filled BUY creates a position in _positions with correct quantity."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)

        await engine.submit_order(order)

        assert _SYMBOL in engine.positions
        assert engine.positions[_SYMBOL].quantity == qty

    @pytest.mark.asyncio
    async def test_market_order_raises_when_no_price_set(self) -> None:
        """Submitting an order before set_last_price() raises ValueError."""
        engine, _ = _make_engine()
        # Deliberately do NOT call set_last_price()
        order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))

        with pytest.raises(ValueError, match="No last price set"):
            await engine.submit_order(order)


# ===========================================================================
# Group 3: LIMIT Order Submission
# ===========================================================================


class TestLimitOrderSubmission:
    """Verify LIMIT order fill logic: immediate fill vs resting, and check_resting_orders."""

    @pytest.mark.asyncio
    async def test_limit_buy_at_market_fills_immediately(self) -> None:
        """A LIMIT BUY at exactly the current market price fills immediately."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # BUY limit at last_price: condition is last_price <= limit_price, so fills
        order = _make_limit_order(
            side=OrderSide.BUY, quantity=Decimal("0.1"), price=_LAST_PRICE
        )

        result = await engine.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.average_fill_price == _LAST_PRICE

    @pytest.mark.asyncio
    async def test_limit_buy_below_market_becomes_resting(self) -> None:
        """A LIMIT BUY below the current price is not filled immediately (OPEN status)."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # Limit price below last price: last_price (50000) > limit (49000) → no fill
        order = _make_limit_order(
            side=OrderSide.BUY, quantity=Decimal("0.1"), price=Decimal("49000")
        )

        result = await engine.submit_order(order)

        assert result.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_limit_sell_at_market_fills_immediately(self) -> None:
        """A LIMIT SELL at exactly the current price fills immediately."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Open a position first
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await engine.submit_order(buy_order)

        # SELL limit at last_price: condition is last_price >= limit_price, so fills
        sell_order = _make_limit_order(
            side=OrderSide.SELL, quantity=Decimal("0.1"), price=_LAST_PRICE
        )
        result = await engine.submit_order(sell_order)

        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_limit_sell_above_market_becomes_resting(self) -> None:
        """A LIMIT SELL above the current price remains OPEN (resting)."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Open a position first
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await engine.submit_order(buy_order)

        # Sell limit above last price: last_price (50000) < limit (51000) → no fill
        sell_order = _make_limit_order(
            side=OrderSide.SELL, quantity=Decimal("0.1"), price=Decimal("51000")
        )
        result = await engine.submit_order(sell_order)

        assert result.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_check_resting_orders_fills_limit_buy_when_price_drops(self) -> None:
        """
        check_resting_orders() fills a resting LIMIT BUY when the new price
        drops to or below the limit price.
        """
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # Place a resting limit buy at 49000 (below current 50000)
        order = _make_limit_order(
            side=OrderSide.BUY, quantity=Decimal("0.1"), price=Decimal("49000")
        )
        resting = await engine.submit_order(order)
        assert resting.status == OrderStatus.OPEN

        # Simulate price dropping to 49000 — trigger the resting order
        filled_orders = await engine.check_resting_orders(_SYMBOL, Decimal("49000"))

        assert len(filled_orders) == 1
        filled = await engine.get_order(resting.order_id)
        assert filled.status == OrderStatus.FILLED

    # CR-004 (HIGH) — SELL limit resting fill
    @pytest.mark.asyncio
    async def test_check_resting_orders_fills_limit_sell_when_price_rises(self) -> None:
        """Resting SELL limit triggers when market price rises to limit level."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)  # 50000

        # Open a long position first so the SELL has something to close
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        await engine.submit_order(buy_order)

        # Place a resting SELL limit at 51000 (above current 50000 — will not fill immediately)
        sell_limit_price = Decimal("51000")
        sell_order = _make_limit_order(
            side=OrderSide.SELL, quantity=Decimal("0.1"), price=sell_limit_price
        )
        resting = await engine.submit_order(sell_order)
        assert resting.status == OrderStatus.OPEN

        # Simulate price rising to 51000 — should trigger the resting SELL
        filled_orders = await engine.check_resting_orders(_SYMBOL, sell_limit_price)

        assert len(filled_orders) == 1
        filled = await engine.get_order(resting.order_id)
        assert filled.status == OrderStatus.FILLED

        # Fill price must equal the limit price (no slippage for maker orders)
        fills = await engine.get_fills(resting.order_id)
        assert len(fills) == 1
        assert fills[0].price == sell_limit_price

    @pytest.mark.asyncio
    async def test_limit_order_filled_at_limit_price_not_market_price(self) -> None:
        """
        When a resting limit order is filled via check_resting_orders, the fill
        price is the limit price (not the triggering bar price).
        """
        engine, _ = _make_engine()
        limit_price = Decimal("49000")
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        order = _make_limit_order(
            side=OrderSide.BUY, quantity=Decimal("0.1"), price=limit_price
        )
        resting = await engine.submit_order(order)

        # Price drops further to 48500 — order still fills at limit_price
        await engine.check_resting_orders(_SYMBOL, Decimal("48500"))

        fills = await engine.get_fills(resting.order_id)
        assert len(fills) == 1
        assert fills[0].price == limit_price


# ===========================================================================
# Group 4: Cancel Orders
# ===========================================================================


class TestCancelOrders:
    """Verify cancel_order() transitions and error conditions."""

    @pytest.mark.asyncio
    async def test_cancel_open_order_transitions_to_canceled(self) -> None:
        """Cancelling an OPEN resting order sets its status to CANCELED."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # Submit a resting LIMIT order (won't fill immediately)
        order = _make_limit_order(
            side=OrderSide.BUY, quantity=Decimal("0.1"), price=Decimal("49000")
        )
        resting = await engine.submit_order(order)
        assert resting.status == OrderStatus.OPEN

        canceled = await engine.cancel_order(resting.order_id)

        assert canceled.status == OrderStatus.CANCELED

    @pytest.mark.asyncio
    async def test_cancel_filled_order_raises_invalid_transition(self) -> None:
        """Attempting to cancel a FILLED order raises InvalidOrderTransitionError."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        order = _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.1"))
        filled = await engine.submit_order(order)
        assert filled.status == OrderStatus.FILLED

        with pytest.raises(InvalidOrderTransitionError):
            await engine.cancel_order(filled.order_id)

    @pytest.mark.asyncio
    async def test_cancel_unknown_order_id_raises_key_error(self) -> None:
        """Cancelling a non-existent order_id raises KeyError."""
        engine, _ = _make_engine()
        fake_id = uuid4()

        with pytest.raises(KeyError):
            await engine.cancel_order(fake_id)


# ===========================================================================
# Group 5: Signal Processing
# ===========================================================================


class TestSignalProcessing:
    """Verify process_signal() correctly converts signals into orders or returns empty."""

    @pytest.mark.asyncio
    async def test_buy_signal_with_sufficient_cash_creates_market_order(self) -> None:
        """A BUY signal with sufficient cash returns a list containing one FILLED order."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        signal = _make_buy_signal()

        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET
        assert orders[0].status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_sell_signal_with_existing_position_creates_order(self) -> None:
        """A SELL signal when a position exists produces a FILLED SELL order."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # First, open a position via a BUY signal
        buy_signal = _make_buy_signal()
        await engine.process_signal(buy_signal)
        assert _SYMBOL in engine.positions

        sell_signal = _make_sell_signal()
        orders = await engine.process_signal(sell_signal)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_risk_manager_rejection_prevents_order_submission(self) -> None:
        """When pre_trade_check() returns approved=False, process_signal returns empty."""
        engine, rm = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # Override pre_trade_check to reject
        rm.pre_trade_check.return_value = RiskCheckResult(
            approved=False,
            adjusted_quantity=Decimal("0"),
            rejection_reasons=["test: max positions reached"],
            warnings=[],
        )

        signal = _make_buy_signal()
        orders = await engine.process_signal(signal)

        assert orders == []

    @pytest.mark.asyncio
    async def test_hold_signal_returns_empty_list(self) -> None:
        """A HOLD signal must produce no orders."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        signal = _make_hold_signal()

        orders = await engine.process_signal(signal)

        assert orders == []

    @pytest.mark.asyncio
    async def test_sell_signal_without_position_returns_empty_list(self) -> None:
        """A SELL signal when no position exists returns an empty list (nothing to sell)."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        # No position opened — sell signal should be a no-op
        signal = _make_sell_signal()

        orders = await engine.process_signal(signal)

        assert orders == []

    # CR-002 (HIGH) — Zero-quantity guard
    @pytest.mark.asyncio
    async def test_zero_position_size_returns_empty_list(self) -> None:
        """When risk manager calculates zero position size, no order is created."""
        engine, rm = _make_engine()
        engine.set_last_price("BTC/USDT", Decimal("50000"))
        rm.calculate_position_size.return_value = Decimal("0")

        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
            confidence=0.8,
            generated_at=datetime.now(tz=UTC),
            metadata={},
        )
        orders = await engine.process_signal(signal)
        assert orders == []
        rm.pre_trade_check.assert_not_called()

    # CR-003 (HIGH) — Quantity-capping by pre_trade_check adjusted_quantity
    @pytest.mark.asyncio
    async def test_pre_trade_check_adjusted_quantity_caps_order(self) -> None:
        """
        When pre_trade_check approves but returns a smaller adjusted_quantity,
        the submitted order uses the capped quantity, not the originally proposed size.

        Setup: calculate_position_size returns 0.1, but pre_trade_check caps it to 0.05.
        The filled order quantity must equal 0.05.
        """
        # Build engine with calculate_position_size returning 0.1
        engine, rm = _make_engine(position_size=Decimal("0.1"))
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Override pre_trade_check to approve but with a smaller adjusted_quantity
        capped_qty = Decimal("0.05")
        rm.pre_trade_check.return_value = RiskCheckResult(
            approved=True,
            adjusted_quantity=capped_qty,
            rejection_reasons=[],
            warnings=[],
        )

        signal = _make_buy_signal()
        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].quantity == capped_qty
        assert orders[0].status == OrderStatus.FILLED


# ===========================================================================
# Group 6: Position Tracking
# ===========================================================================


class TestPositionTracking:
    """Verify position tracking across buys, sells, and PnL calculations."""

    @pytest.mark.asyncio
    async def test_buy_fill_creates_new_position_with_correct_entry_price(self) -> None:
        """
        After a MARKET BUY, a Position is created whose average_entry_price includes
        the all-in cost (price + fee spread across qty).
        """
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)

        result = await engine.submit_order(order)
        fills = await engine.get_fills(result.order_id)
        fill = fills[0]

        position = engine.positions[_SYMBOL]
        # all-in entry = (price * qty + fee) / qty
        expected_entry = (
            (fill.price * qty + fill.fee) / qty
        ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

        assert position.quantity == qty
        assert position.average_entry_price == expected_entry

    @pytest.mark.asyncio
    async def test_second_buy_updates_average_entry_price(self) -> None:
        """
        A second BUY on the same symbol recalculates average_entry_price using
        the weighted average of both fills' all-in costs.
        """
        engine, _ = _make_engine()
        qty1 = Decimal("0.1")
        qty2 = Decimal("0.05")

        # First buy at 50000
        engine.set_last_price(_SYMBOL, Decimal("50000"))
        order1 = _make_market_order(side=OrderSide.BUY, quantity=qty1)
        r1 = await engine.submit_order(order1)

        # Snapshot the position BEFORE the second buy so old_cost uses the
        # all-in entry price computed by paper.py for the first fill only.
        pos_after_first_buy = engine.positions[_SYMBOL]

        # Second buy at 52000
        engine.set_last_price(_SYMBOL, Decimal("52000"))
        order2 = _make_market_order(side=OrderSide.BUY, quantity=qty2)
        r2 = await engine.submit_order(order2)
        fills2 = await engine.get_fills(r2.order_id)
        f2 = fills2[0]

        position = engine.positions[_SYMBOL]
        # paper.py formula: old_cost = position.quantity * position.average_entry_price
        # where position is the state BEFORE the second fill.
        old_cost = qty1 * pos_after_first_buy.average_entry_price
        # new_cost = fill.quantity * fill.price + fill.fee  (from paper.py)
        new_cost = qty2 * f2.price + f2.fee
        total_qty = qty1 + qty2
        expected_avg = ((old_cost + new_cost) / total_qty).quantize(
            _PRICE_PRECISION, rounding=ROUND_HALF_UP
        )

        assert position.quantity == total_qty
        assert position.average_entry_price == expected_avg

    @pytest.mark.asyncio
    async def test_partial_sell_reduces_position_quantity(self) -> None:
        """Selling half a position reduces its quantity by exactly the sold amount."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.2")

        buy_order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        await engine.submit_order(buy_order)

        sell_qty = Decimal("0.1")
        sell_order = _make_market_order(side=OrderSide.SELL, quantity=sell_qty)
        await engine.submit_order(sell_order)

        remaining = engine.positions[_SYMBOL].quantity
        assert remaining == qty - sell_qty

    @pytest.mark.asyncio
    async def test_full_sell_closes_position(self) -> None:
        """Selling the full position quantity sets position.quantity to zero (is_flat)."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")

        buy_order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        await engine.submit_order(buy_order)

        sell_order = _make_market_order(side=OrderSide.SELL, quantity=qty)
        await engine.submit_order(sell_order)

        assert engine.positions[_SYMBOL].is_flat

    # CR-005 (MEDIUM) — Realised PnL assertion
    @pytest.mark.asyncio
    async def test_realised_pnl_after_full_close_matches_independent_calculation(self) -> None:
        """
        Buy then sell the full position; realised_pnl must equal
        (sell_fill_price - avg_entry_price) * qty - sell_fee,
        where all values are computed independently from paper.py's formulas.
        """
        engine, _ = _make_engine()
        buy_market_price = Decimal("50000")
        sell_market_price = Decimal("51000")
        qty = Decimal("0.1")

        # --- Step 1: Open position ---
        engine.set_last_price(_SYMBOL, buy_market_price)
        buy_order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        buy_result = await engine.submit_order(buy_order)
        buy_fills = await engine.get_fills(buy_result.order_id)
        buy_fill = buy_fills[0]

        # Independently compute all-in average entry price
        # paper.py: all_in_entry = (fill.price * fill.quantity + fill.fee) / fill.quantity
        expected_avg_entry = (
            (buy_fill.price * qty + buy_fill.fee) / qty
        ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

        # --- Step 2: Close position ---
        engine.set_last_price(_SYMBOL, sell_market_price)
        sell_order = _make_market_order(side=OrderSide.SELL, quantity=qty)
        sell_result = await engine.submit_order(sell_order)
        sell_fills = await engine.get_fills(sell_result.order_id)
        sell_fill = sell_fills[0]

        # --- Step 3: Independently compute expected realised PnL ---
        # paper.py: pnl = (fill.price - position.average_entry_price) * sell_qty - fill.fee
        expected_pnl = (sell_fill.price - expected_avg_entry) * qty - sell_fill.fee

        position = engine.positions[_SYMBOL]
        assert position.realised_pnl == expected_pnl

    # CR-006 (MEDIUM) — Non-tautological unrealised PnL test
    @pytest.mark.asyncio
    async def test_unrealised_pnl_reflects_current_price_change(self) -> None:
        """
        After opening a position and triggering a second fill at a higher price,
        unrealised_pnl equals (second_fill_price - new_avg_entry) * total_qty.
        All inputs are derived independently via helper functions — never read
        from the position object being asserted on.
        """
        engine, _ = _make_engine()
        qty1 = Decimal("0.1")
        qty2 = Decimal("0.001")
        price1 = Decimal("50000")
        price2 = Decimal("52000")

        # --- First buy ---
        engine.set_last_price(_SYMBOL, price1)
        order1 = _make_market_order(side=OrderSide.BUY, quantity=qty1)
        r1 = await engine.submit_order(order1)
        fills1 = await engine.get_fills(r1.order_id)
        f1 = fills1[0]

        # Independently compute all-in entry price for first buy
        # paper.py: all_in_entry = (f1.price * qty1 + f1.fee) / qty1
        avg_entry1 = (
            (f1.price * qty1 + f1.fee) / qty1
        ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

        # --- Second buy at higher price (updates current_price and unrealised_pnl) ---
        engine.set_last_price(_SYMBOL, price2)
        order2 = _make_market_order(side=OrderSide.BUY, quantity=qty2)
        r2 = await engine.submit_order(order2)
        fills2 = await engine.get_fills(r2.order_id)
        f2 = fills2[0]

        # Independently compute new average entry price
        # paper.py: new_avg = (old_cost + new_cost) / total_qty
        #   old_cost = qty1 * avg_entry1
        #   new_cost = qty2 * f2.price + f2.fee
        old_cost = qty1 * avg_entry1
        new_cost = qty2 * f2.price + f2.fee
        total_qty = qty1 + qty2
        expected_avg_entry2 = (
            (old_cost + new_cost) / total_qty
        ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

        # Independently compute expected unrealised PnL
        # paper.py: unrealised_pnl = (current_price - new_avg_price) * total_qty
        # current_price == f2.price (the fill price of the second buy, set as current_price)
        expected_unrealised = (f2.price - expected_avg_entry2) * total_qty

        pos = engine.positions[_SYMBOL]
        assert pos.unrealised_pnl == expected_unrealised


# ===========================================================================
# Group 7: Cash & Equity
# ===========================================================================


class TestCashAndEquity:
    """Verify equity calculations and cumulative cash tracking across multiple trades."""

    def test_initial_equity_equals_initial_cash(self) -> None:
        """Before any trades, total equity must equal initial_cash."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        equity = engine._get_current_equity()
        assert equity == _INITIAL_CASH

    @pytest.mark.asyncio
    async def test_equity_equals_cash_plus_position_value(self) -> None:
        """After a BUY, equity = cash + (position.quantity * position.current_price)."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        await engine.submit_order(order)

        equity = engine._get_current_equity()
        pos = engine.positions[_SYMBOL]
        expected_equity = engine.cash + pos.notional_value
        assert equity == expected_equity

    @pytest.mark.asyncio
    async def test_fees_correctly_deducted_from_cash(self) -> None:
        """Total cash deduction for a BUY must equal notional + fee (no rounding loss)."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.1")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        result = await engine.submit_order(order)

        fills = await engine.get_fills(result.order_id)
        fill = fills[0]
        expected_cash = _INITIAL_CASH - (qty * fill.price + fill.fee)
        assert engine.cash == expected_cash

    @pytest.mark.asyncio
    async def test_multiple_trades_correctly_track_cumulative_cash(self) -> None:
        """Cash balance is correctly updated after two round-trip trades."""
        engine, _ = _make_engine(initial_cash=_INITIAL_CASH)
        engine.set_last_price(_SYMBOL, _LAST_PRICE)
        qty = Decimal("0.05")

        # Trade 1: Buy
        b1 = await engine.submit_order(_make_market_order(side=OrderSide.BUY, quantity=qty))
        f1_fills = await engine.get_fills(b1.order_id)
        f1 = f1_fills[0]
        expected_cash = _INITIAL_CASH - (qty * f1.price + f1.fee)

        # Trade 1: Sell
        s1 = await engine.submit_order(_make_market_order(side=OrderSide.SELL, quantity=qty))
        s1_fills = await engine.get_fills(s1.order_id)
        sf1 = s1_fills[0]
        expected_cash += qty * sf1.price - sf1.fee

        # Trade 2: Buy
        b2 = await engine.submit_order(_make_market_order(side=OrderSide.BUY, quantity=qty))
        f2_fills = await engine.get_fills(b2.order_id)
        f2 = f2_fills[0]
        expected_cash -= qty * f2.price + f2.fee

        assert engine.cash == expected_cash


# ===========================================================================
# Group 8: Lifecycle
# ===========================================================================


class TestLifecycle:
    """Verify on_start, on_stop, set_last_price, and get_open_orders behavior."""

    @pytest.mark.asyncio
    async def test_on_start_succeeds_without_error(self) -> None:
        """on_start() must complete without raising any exception."""
        engine, _ = _make_engine()
        await engine.on_start()  # Must not raise

    @pytest.mark.asyncio
    async def test_on_stop_cancels_all_open_orders(self) -> None:
        """on_stop() must cancel all resting (OPEN) orders before shutdown."""
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Place two resting limit orders
        o1 = await engine.submit_order(
            _make_limit_order(side=OrderSide.BUY, quantity=Decimal("0.1"), price=Decimal("49000"))
        )
        o2 = await engine.submit_order(
            _make_limit_order(side=OrderSide.BUY, quantity=Decimal("0.05"), price=Decimal("48000"))
        )
        assert o1.status == OrderStatus.OPEN
        assert o2.status == OrderStatus.OPEN

        await engine.on_stop()

        # Both orders must now be CANCELED
        r1 = await engine.get_order(o1.order_id)
        r2 = await engine.get_order(o2.order_id)
        assert r1.status == OrderStatus.CANCELED
        assert r2.status == OrderStatus.CANCELED

    def test_set_last_price_updates_price_registry(self) -> None:
        """set_last_price() stores the price so _get_last_price() returns it."""
        engine, _ = _make_engine()
        price = Decimal("55000")
        engine.set_last_price(_SYMBOL, price)
        assert engine._get_last_price(_SYMBOL) == price

    @pytest.mark.asyncio
    async def test_get_open_orders_returns_only_non_terminal_orders(self) -> None:
        """
        get_open_orders() returns resting (OPEN) orders and excludes FILLED
        and CANCELED orders.
        """
        engine, _ = _make_engine()
        engine.set_last_price(_SYMBOL, _LAST_PRICE)

        # Submit a MARKET order (will be FILLED)
        filled_order = await engine.submit_order(
            _make_market_order(side=OrderSide.BUY, quantity=Decimal("0.05"))
        )
        assert filled_order.status == OrderStatus.FILLED

        # Submit a resting LIMIT order (will be OPEN)
        resting_order = await engine.submit_order(
            _make_limit_order(side=OrderSide.BUY, quantity=Decimal("0.05"), price=Decimal("49000"))
        )
        assert resting_order.status == OrderStatus.OPEN

        open_orders = engine.get_open_orders()

        order_ids = [o.order_id for o in open_orders]
        assert resting_order.order_id in order_ids
        assert filled_order.order_id not in order_ids
