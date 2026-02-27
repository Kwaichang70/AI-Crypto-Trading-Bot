"""
packages/trading/engines/paper.py
-----------------------------------
Paper-trading execution engine with simulated fills.

Provides realistic order execution simulation including:
- Configurable slippage model (basis-point spread)
- Configurable fill latency (not enforced as real delay, but recorded)
- Fee calculation using risk-manager fee parameters
- Full order state-machine compliance
- Position tracking keyed by symbol

Design notes
------------
- MARKET orders fill immediately at the slippage-adjusted last known price.
- LIMIT orders fill only if the last known price would satisfy the limit
  (BUY: last_price <= limit_price; SELL: last_price >= limit_price).
- Partial fills are not simulated in MVP; every fill is a complete fill.
  The state machine supports PARTIAL for future extension.
- All fills are recorded in ``_fills`` for auditability and replay.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import UUID, uuid4

import structlog

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.execution import BaseExecutionEngine
from trading.models import Fill, Order, Position, Signal
from trading.risk import BaseRiskManager

__all__ = ["PaperExecutionEngine"]

logger = structlog.get_logger(__name__)

# Precision for monetary rounding (8 decimal places, standard for crypto)
_PRICE_PRECISION = Decimal("0.00000001")
_QTY_PRECISION = Decimal("0.00000001")


class PaperExecutionEngine(BaseExecutionEngine):
    """
    Simulated execution engine for backtesting and paper trading.

    All order execution is deterministic given the same input sequence,
    which is critical for reproducible backtests.

    Parameters
    ----------
    run_id:
        Unique identifier for the trading run.
    risk_manager:
        Injected risk manager used for pre-trade checks and position sizing.
    fill_latency_ms:
        Simulated fill latency in milliseconds. Recorded on fills but does
        not introduce real async delay. Default 0.
    slippage_bps:
        Slippage in basis points applied to MARKET orders.
        BUY: price * (1 + bps/10000). SELL: price * (1 - bps/10000).
        Default 5 (0.05%).
    initial_cash:
        Starting cash balance in quote currency. Used for portfolio
        equity tracking. Default 10000.
    """

    def __init__(
        self,
        run_id: str,
        risk_manager: BaseRiskManager,
        *,
        fill_latency_ms: int = 0,
        slippage_bps: int = 5,
        initial_cash: Decimal = Decimal("10000"),
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(run_id=run_id, config=config)
        self._risk_manager = risk_manager
        self._fill_latency_ms = fill_latency_ms
        self._slippage_bps = slippage_bps
        self._initial_cash = initial_cash
        self._cash = initial_cash

        # Fill registry: order_id -> list of Fill objects
        self._fills: dict[UUID, list[Fill]] = {}

        # Position tracking: symbol -> Position
        self._positions: dict[str, Position] = {}

        # Last known prices: symbol -> Decimal (updated via set_last_price)
        self._last_prices: dict[str, Decimal] = {}

        self._log = self._log.bind(
            engine="paper",
            slippage_bps=slippage_bps,
            fill_latency_ms=fill_latency_ms,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def risk_manager(self) -> BaseRiskManager:
        return self._risk_manager

    @property
    def positions(self) -> dict[str, Position]:
        """Return a copy of current positions."""
        return dict(self._positions)

    @property
    def cash(self) -> Decimal:
        """Current cash balance in quote currency."""
        return self._cash

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def set_last_price(self, symbol: str, price: Decimal) -> None:
        """
        Update the last known price for a symbol.

        This must be called before processing signals or submitting orders
        for a given symbol. In backtesting, this is set from the bar close
        price before each on_bar call.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. 'BTC/USDT'.
        price:
            Last known price in quote currency.
        """
        self._last_prices[symbol] = price

    def _get_last_price(self, symbol: str) -> Decimal:
        """
        Get the last known price for a symbol.

        Raises
        ------
        ValueError
            If no price has been set for the symbol.
        """
        price = self._last_prices.get(symbol)
        if price is None:
            raise ValueError(
                f"No last price set for {symbol}. "
                f"Call set_last_price() before processing orders."
            )
        return price

    # ------------------------------------------------------------------
    # Slippage model
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """
        Apply slippage to a market-order execution price.

        For BUY: price increases (we pay more).
        For SELL: price decreases (we receive less).

        Parameters
        ----------
        price:
            Raw market price before slippage.
        side:
            Order side (BUY or SELL).

        Returns
        -------
        Decimal:
            Slippage-adjusted execution price.
        """
        slippage_factor = Decimal(self._slippage_bps) / Decimal("10000")
        if side == OrderSide.BUY:
            adjusted = price * (Decimal("1") + slippage_factor)
        else:
            adjusted = price * (Decimal("1") - slippage_factor)
        return adjusted.quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Fee calculation
    # ------------------------------------------------------------------

    def _calculate_fee(
        self,
        quantity: Decimal,
        price: Decimal,
        is_maker: bool,
    ) -> Decimal:
        """
        Calculate the trading fee for a fill.

        Parameters
        ----------
        quantity:
            Fill quantity in base asset.
        price:
            Execution price per unit.
        is_maker:
            True if the order was a resting limit order (maker fee).

        Returns
        -------
        Decimal:
            Fee amount in quote currency.
        """
        notional = quantity * price
        fee_rate = (
            Decimal(str(self._risk_manager.params.maker_fee_pct))
            if is_maker
            else Decimal(str(self._risk_manager.params.taker_fee_pct))
        )
        return (notional * fee_rate).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        order: Order,
        fill_price: Decimal,
        is_maker: bool,
    ) -> Fill:
        """
        Create a Fill object for a simulated execution.

        Parameters
        ----------
        order:
            The order being filled.
        fill_price:
            The execution price (already slippage-adjusted for MARKET).
        is_maker:
            Whether this fill qualifies for maker fee.

        Returns
        -------
        Fill:
            The generated fill event.
        """
        fee = self._calculate_fee(order.quantity, fill_price, is_maker)

        # Extract quote currency from symbol (e.g. 'BTC/USDT' -> 'USDT')
        parts = order.symbol.split("/")
        fee_currency = parts[1] if len(parts) > 1 else "USDT"

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            fee_currency=fee_currency,
            is_maker=is_maker,
            executed_at=datetime.now(tz=UTC),
        )

        # Record fill
        if order.order_id not in self._fills:
            self._fills[order.order_id] = []
        self._fills[order.order_id].append(fill)

        self._log.info(
            "paper.fill",
            order_id=str(order.order_id),
            symbol=order.symbol,
            side=order.side.value,
            quantity=str(order.quantity),
            price=str(fill_price),
            fee=str(fee),
            is_maker=is_maker,
        )

        return fill

    # ------------------------------------------------------------------
    # Position update
    # ------------------------------------------------------------------

    def _update_position(self, fill: Fill) -> None:
        """
        Update the position tracker and cash balance after a fill.

        For BUY fills: increase position quantity, decrease cash.
        For SELL fills: decrease position quantity, increase cash,
        and realize PnL based on the average entry price.

        Parameters
        ----------
        fill:
            The fill event to process.
        """
        symbol = fill.symbol
        current_price = fill.price

        position = self._positions.get(symbol)
        now = datetime.now(tz=UTC)

        if fill.side == OrderSide.BUY:
            # Decrease cash by (quantity * price + fee)
            cost = fill.quantity * fill.price + fill.fee
            self._cash -= cost

            if position is None or position.is_flat:
                # Open new position -- use all-in cost basis (includes entry fee)
                all_in_entry_price = (
                    (fill.price * fill.quantity + fill.fee) / fill.quantity
                ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)
                self._positions[symbol] = Position(
                    symbol=symbol,
                    run_id=self._run_id,
                    quantity=fill.quantity,
                    average_entry_price=all_in_entry_price,
                    current_price=current_price,
                    realised_pnl=Decimal("0"),
                    unrealised_pnl=Decimal("0"),
                    total_fees_paid=fill.fee,
                    opened_at=now,
                    updated_at=now,
                )
            else:
                # Add to existing position (average up/down)
                old_cost = position.quantity * position.average_entry_price
                new_cost = fill.quantity * fill.price + fill.fee
                total_qty = position.quantity + fill.quantity
                new_avg_price = (
                    (old_cost + new_cost) / total_qty
                ).quantize(_PRICE_PRECISION, rounding=ROUND_HALF_UP)

                self._positions[symbol] = position.model_copy(update={
                    "quantity": total_qty,
                    "average_entry_price": new_avg_price,
                    "current_price": current_price,
                    "unrealised_pnl": (current_price - new_avg_price) * total_qty,
                    "total_fees_paid": position.total_fees_paid + fill.fee,
                    "updated_at": now,
                })

        elif fill.side == OrderSide.SELL:
            if position is None or position.is_flat:
                self._log.warning(
                    "paper.sell_without_position",
                    symbol=symbol,
                    quantity=str(fill.quantity),
                )
                return

            # Calculate realized PnL on the sold quantity
            sell_qty = min(fill.quantity, position.quantity)
            pnl = (fill.price - position.average_entry_price) * sell_qty - fill.fee
            remaining_qty = position.quantity - sell_qty

            # Increase cash by (quantity * price - fee)
            proceeds = fill.quantity * fill.price - fill.fee
            self._cash += proceeds

            if remaining_qty <= Decimal("0"):
                # Position fully closed
                self._positions[symbol] = position.model_copy(update={
                    "quantity": Decimal("0"),
                    "current_price": current_price,
                    "realised_pnl": position.realised_pnl + pnl,
                    "unrealised_pnl": Decimal("0"),
                    "total_fees_paid": position.total_fees_paid + fill.fee,
                    "updated_at": now,
                })
            else:
                # Partial close
                self._positions[symbol] = position.model_copy(update={
                    "quantity": remaining_qty,
                    "current_price": current_price,
                    "realised_pnl": position.realised_pnl + pnl,
                    "unrealised_pnl": (
                        (current_price - position.average_entry_price) * remaining_qty
                    ),
                    "total_fees_paid": position.total_fees_paid + fill.fee,
                    "updated_at": now,
                })

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the paper-trading fill simulator.

        MARKET orders are filled immediately at the slippage-adjusted
        last known price. LIMIT orders are checked against the last
        known price and filled if the limit condition is met; otherwise
        the order remains OPEN.

        Parameters
        ----------
        order:
            A fully validated Order with status=NEW.

        Returns
        -------
        Order:
            Updated order reflecting the fill outcome.
        """
        # Register the order
        self._orders[order.order_id] = order

        # NEW -> PENDING_SUBMIT
        order = self._transition(order, OrderStatus.PENDING_SUBMIT)

        # PENDING_SUBMIT -> OPEN
        order = self._transition(order, OrderStatus.OPEN)

        symbol = order.symbol
        last_price = self._get_last_price(symbol)

        if order.order_type == OrderType.MARKET:
            # Apply slippage for market orders
            fill_price = self._apply_slippage(last_price, order.side)
            is_maker = False

            fill = self._simulate_fill(order, fill_price, is_maker)

            # Update order with fill data
            order = order.model_copy(update={
                "filled_quantity": order.quantity,
                "average_fill_price": fill_price,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[order.order_id] = order

            # OPEN -> FILLED
            order = self._transition(order, OrderStatus.FILLED)

            # Update position tracking
            self._update_position(fill)

        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError(
                    f"LIMIT order {order.order_id} has no price -- "
                    "this is a bug in order construction."
                )
            is_maker = True

            # Check if limit price would trigger immediate fill
            should_fill = False
            if order.side == OrderSide.BUY and last_price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and last_price >= order.price:
                should_fill = True

            if should_fill:
                # Fill at limit price (no slippage for limit orders)
                fill_price = order.price
                fill = self._simulate_fill(order, fill_price, is_maker)

                order = order.model_copy(update={
                    "filled_quantity": order.quantity,
                    "average_fill_price": fill_price,
                    "updated_at": datetime.now(tz=UTC),
                })
                self._orders[order.order_id] = order

                # OPEN -> FILLED
                order = self._transition(order, OrderStatus.FILLED)

                self._update_position(fill)
            else:
                # Order remains OPEN, waiting for price to reach limit
                self._log.debug(
                    "paper.limit_order_resting",
                    order_id=str(order.order_id),
                    symbol=symbol,
                    side=order.side.value,
                    limit_price=str(order.price),
                    last_price=str(last_price),
                )

        return order

    async def cancel_order(self, order_id: UUID) -> Order:
        """
        Cancel an open or partially-filled order.

        Parameters
        ----------
        order_id:
            Internal UUID of the order to cancel.

        Returns
        -------
        Order:
            Updated order with status CANCELED.

        Raises
        ------
        KeyError:
            If the order_id is not found.
        InvalidOrderTransitionError:
            If the order is in a terminal state.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        order = self._transition(order, OrderStatus.CANCELED)

        self._log.info(
            "paper.order_canceled",
            order_id=str(order_id),
            symbol=order.symbol,
        )

        return order

    async def get_order(self, order_id: UUID) -> Order:
        """
        Fetch the current state of an order from the in-memory registry.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        Order:
            The current order record.

        Raises
        ------
        KeyError:
            If the order_id is not found.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")
        return order

    async def process_signal(self, signal: Signal) -> list[Order]:
        """
        Convert a trading Signal into zero or more submitted Orders.

        Flow:
        1. If signal direction is HOLD, return empty (no action).
        2. Determine order side from signal direction.
        3. Use risk_manager.calculate_position_size() to determine quantity.
        4. Build a proposed Order.
        5. Run risk_manager.pre_trade_check() for approval.
        6. If approved, submit the order.

        Parameters
        ----------
        signal:
            The Signal emitted by a strategy's on_bar call.

        Returns
        -------
        list[Order]:
            Orders created and submitted (may be empty if signal is
            rejected or is HOLD).
        """
        if signal.direction == SignalDirection.HOLD:
            self._log.debug(
                "paper.signal_hold",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
            )
            return []

        # Map signal direction to order side
        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.BUY
            else OrderSide.SELL
        )

        # For SELL signals, check if we have a position to sell
        if side == OrderSide.SELL:
            position = self._positions.get(signal.symbol)
            if position is None or position.is_flat:
                self._log.debug(
                    "paper.sell_no_position",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                )
                return []

        symbol = signal.symbol
        last_price = self._get_last_price(symbol)

        # Calculate position size via risk manager
        quantity = self._risk_manager.calculate_position_size(
            equity=self._get_current_equity(),
            entry_price=last_price,
            stop_loss_price=None,
            confidence=signal.confidence,
        )

        if quantity <= Decimal("0"):
            self._log.debug(
                "paper.zero_quantity",
                strategy_id=signal.strategy_id,
                symbol=symbol,
            )
            return []

        # For SELL, cap quantity at current position size
        if side == OrderSide.SELL:
            position = self._positions[signal.symbol]
            quantity = min(quantity, position.quantity)

        # Build the proposed order
        client_order_id = f"{self._run_id}-{uuid4().hex[:12]}"
        proposed_order = Order(
            client_order_id=client_order_id,
            run_id=self._run_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity.quantize(_QTY_PRECISION, rounding=ROUND_HALF_UP),
        )

        # Pre-trade risk check
        open_positions = [
            p for p in self._positions.values() if not p.is_flat
        ]
        daily_pnl = self._get_daily_pnl()
        peak_equity = self._get_peak_equity()

        risk_result = self._risk_manager.pre_trade_check(
            order=proposed_order,
            current_equity=self._get_current_equity(),
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            peak_equity=peak_equity,
            market_price=last_price,
        )

        if not risk_result.approved:
            self._log.warning(
                "paper.signal_rejected",
                strategy_id=signal.strategy_id,
                symbol=symbol,
                reasons=risk_result.rejection_reasons,
            )
            return []

        # Apply adjusted quantity from risk check (only when approved;
        # on rejection adjusted_quantity is 0 and we already returned above)
        if risk_result.approved and risk_result.adjusted_quantity < proposed_order.quantity:
            proposed_order = proposed_order.model_copy(update={
                "quantity": risk_result.adjusted_quantity,
            })

        if risk_result.warnings:
            self._log.warning(
                "paper.risk_warnings",
                strategy_id=signal.strategy_id,
                symbol=symbol,
                warnings=risk_result.warnings,
            )

        # Submit the order
        filled_order = await self.submit_order(proposed_order)

        return [filled_order]

    async def get_fills(self, order_id: UUID) -> list[Fill]:
        """
        Return all fills associated with an order, sorted by executed_at.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        list[Fill]:
            Fills sorted by executed_at ascending. Empty list if none.
        """
        fills = self._fills.get(order_id, [])
        return sorted(fills, key=lambda f: f.executed_at)

    # ------------------------------------------------------------------
    # Resting limit-order check (called on each new bar)
    # ------------------------------------------------------------------

    async def check_resting_orders(self, symbol: str, price: Decimal) -> list[Order]:
        """
        Check all resting LIMIT orders for a symbol against a new price.

        Called on each bar update to determine if any resting limit orders
        should now be filled. This method is not part of the abstract
        interface but is essential for paper-trading LIMIT order support.

        Parameters
        ----------
        symbol:
            The trading pair to check.
        price:
            The new bar price to check limit orders against.

        Returns
        -------
        list[Order]:
            List of orders that were filled during this check.
        """
        filled_orders: list[Order] = []

        for order in self.get_open_orders():
            if order.symbol != symbol:
                continue
            if order.order_type != OrderType.LIMIT:
                continue
            if order.price is None:
                raise ValueError(
                    f"LIMIT order {order.order_id} has no price -- "
                    "this is a bug in order construction."
                )

            should_fill = False
            if order.side == OrderSide.BUY and price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and price >= order.price:
                should_fill = True

            if should_fill:
                fill_price = order.price  # Fill at limit price
                fill = self._simulate_fill(order, fill_price, is_maker=True)

                order = order.model_copy(update={
                    "filled_quantity": order.quantity,
                    "average_fill_price": fill_price,
                    "updated_at": datetime.now(tz=UTC),
                })
                self._orders[order.order_id] = order
                order = self._transition(order, OrderStatus.FILLED)

                self._update_position(fill)
                filled_orders.append(order)

        return filled_orders

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    def _get_current_equity(self) -> Decimal:
        """
        Calculate current total equity (cash + unrealised position value).

        Returns
        -------
        Decimal:
            Total portfolio equity in quote currency.
        """
        position_value = Decimal("0")
        for pos in self._positions.values():
            if not pos.is_flat:
                position_value += pos.notional_value
        return self._cash + position_value

    def _get_peak_equity(self) -> Decimal:
        """
        Return peak equity. In a full implementation this would be tracked
        by PortfolioAccounting; for now we return initial_cash as a
        conservative baseline. PortfolioAccounting.get_peak_equity() should
        be used when that component is wired in.

        Returns
        -------
        Decimal:
            Peak equity value.
        """
        # Simplistic: max of initial and current
        return max(self._initial_cash, self._get_current_equity())

    def _get_daily_pnl(self) -> Decimal:
        """
        Return daily PnL. In a full implementation this would be tracked
        by PortfolioAccounting; for now we compute from realised PnL
        across positions.

        Returns
        -------
        Decimal:
            Net daily PnL in quote currency.
        """
        total_pnl = Decimal("0")
        for pos in self._positions.values():
            total_pnl += pos.realised_pnl
        return total_pnl

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """Initialize the paper trading engine."""
        await super().on_start()
        self._log.info(
            "paper.engine_started",
            initial_cash=str(self._initial_cash),
            slippage_bps=self._slippage_bps,
            fill_latency_ms=self._fill_latency_ms,
        )

    async def on_stop(self) -> None:
        """
        Shut down the paper trading engine.

        Cancels all resting orders and logs final portfolio state.
        """
        # Cancel all open orders
        open_orders = self.get_open_orders()
        for order in open_orders:
            try:
                await self.cancel_order(order.order_id)
            except Exception:
                self._log.warning(
                    "paper.cancel_failed_on_stop",
                    order_id=str(order.order_id),
                )

        equity = self._get_current_equity()
        total_return = (
            (equity - self._initial_cash) / self._initial_cash
            if self._initial_cash > Decimal("0")
            else Decimal("0")
        )

        self._log.info(
            "paper.engine_stopped",
            final_equity=str(equity),
            initial_cash=str(self._initial_cash),
            total_return=f"{total_return:.4%}",
            total_orders=len(self._orders),
            total_fills=sum(len(f) for f in self._fills.values()),
            open_positions=len(
                [p for p in self._positions.values() if not p.is_flat]
            ),
        )

        await super().on_stop()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PaperExecutionEngine("
            f"run_id={self._run_id!r}, "
            f"orders={len(self._orders)}, "
            f"cash={self._cash}, "
            f"positions={len(self._positions)})"
        )
