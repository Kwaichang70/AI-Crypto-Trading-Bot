"""
packages/trading/engines/live.py
----------------------------------
Live execution engine using CCXT for real exchange order placement.

Safety contract
---------------
- ``enable_live_trading`` must be explicitly True at construction time.
  Any attempt to submit an order when the gate is False raises RuntimeError.
- All orders pass through the RiskManager pre-trade check before submission.
- Exchange API calls are wrapped with error handling and logging.
- The kill-switch on the RiskManager can halt all trading at any time.

Stub policy
-----------
CCXT API calls are structurally complete but marked with TODO comments where
full error recovery (retries, rate-limit handling) is not yet implemented.
The order state machine, signal processing, and risk-check flow are fully
functional.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import UUID, uuid4

import ccxt.async_support as ccxt_async
import structlog

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.execution import BaseExecutionEngine
from trading.ccxt_errors import translate_ccxt_error
from trading.ccxt_retry import ccxt_retry
from trading.models import Fill, Order, Position, Signal
from trading.risk import BaseRiskManager

__all__ = ["LiveExecutionEngine"]

logger = structlog.get_logger(__name__)

_PRICE_PRECISION = Decimal("0.00000001")
_QTY_PRECISION = Decimal("0.00000001")


class LiveExecutionEngine(BaseExecutionEngine):
    """
    Live execution engine that places real orders via CCXT.

    This engine is the production-path for real capital deployment.
    It requires an explicit enable gate and a CCXT async exchange instance.

    Parameters
    ----------
    run_id:
        Unique identifier for the trading run.
    risk_manager:
        Injected risk manager for pre-trade checks and position sizing.
    exchange:
        A CCXT async exchange instance (e.g. ccxt.pro.binance()).
        Must already be configured with API credentials.
    enable_live_trading:
        Explicit gate. If False (default), any call to submit_order
        raises RuntimeError. This is the outermost safety gate.
    """

    def __init__(
        self,
        run_id: str,
        risk_manager: BaseRiskManager,
        exchange: Any,  # ccxt.async_support.Exchange — typed as Any to avoid hard import
        *,
        enable_live_trading: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(run_id=run_id, config=config)
        self._risk_manager = risk_manager
        self._exchange = exchange
        self._enable_live_trading = enable_live_trading

        # Fill registry: order_id -> list of Fill objects
        self._fills: dict[UUID, list[Fill]] = {}

        # Position tracking: symbol -> Position
        self._positions: dict[str, Position] = {}

        # Peak equity tracker for drawdown calculation
        self._peak_equity: Decimal = Decimal("0")

        # Map internal order_id -> exchange order ID for reconciliation
        self._exchange_order_map: dict[UUID, str] = {}

        # Reverse map: exchange order ID -> internal order_id
        self._reverse_order_map: dict[str, UUID] = {}

        self._log = self._log.bind(
            engine="live",
            exchange=getattr(exchange, "id", "unknown"),
            live_trading_enabled=enable_live_trading,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def risk_manager(self) -> BaseRiskManager:
        return self._risk_manager

    @property
    def exchange(self) -> Any:
        """Return the underlying CCXT exchange instance."""
        return self._exchange

    @property
    def is_live_enabled(self) -> bool:
        return self._enable_live_trading

    @property
    def positions(self) -> dict[str, Position]:
        """Return a copy of current positions."""
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Safety gate
    # ------------------------------------------------------------------

    def _enforce_live_gate(self) -> None:
        """
        Check the live-trading safety gate.

        Raises
        ------
        RuntimeError
            If live trading is not enabled.
        """
        if not self._enable_live_trading:
            raise RuntimeError(
                "Live trading is not enabled. "
                "Set enable_live_trading=True and provide valid API credentials "
                "to place real orders. This is a safety gate to prevent "
                "accidental capital deployment."
            )

    # ------------------------------------------------------------------
    # CCXT response mapping
    # ------------------------------------------------------------------

    def _map_ccxt_order_status(self, ccxt_status: str) -> OrderStatus:
        """
        Map a CCXT order status string to our internal OrderStatus enum.

        CCXT statuses: 'open', 'closed', 'canceled', 'expired', 'rejected'

        Parameters
        ----------
        ccxt_status:
            Status string from the CCXT order response.

        Returns
        -------
        OrderStatus:
            Mapped internal status.
        """
        mapping: dict[str, OrderStatus] = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(ccxt_status, OrderStatus.OPEN)

    def _extract_fee_from_ccxt(
        self,
        ccxt_trade: dict[str, Any],
    ) -> tuple[Decimal, str]:
        """
        Extract fee amount and currency from a CCXT trade dict.

        Parameters
        ----------
        ccxt_trade:
            A single trade dict from exchange.fetch_order_trades().

        Returns
        -------
        tuple[Decimal, str]:
            (fee_amount, fee_currency)
        """
        fee_info = ccxt_trade.get("fee") or {}
        fee_cost = Decimal(str(fee_info.get("cost", "0")))
        fee_currency = fee_info.get("currency", "USDT")
        return fee_cost, fee_currency

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the live exchange via CCXT.

        The full flow:
        1. Enforce the live-trading safety gate.
        2. Transition NEW -> PENDING_SUBMIT.
        3. Call exchange.create_order() via CCXT.
        4. On success: transition to OPEN and record exchange_order_id.
        5. On failure: transition to REJECTED with error details.

        Parameters
        ----------
        order:
            A fully validated Order with status=NEW.

        Returns
        -------
        Order:
            Updated order reflecting the submission outcome.

        Raises
        ------
        RuntimeError:
            If live trading is not enabled.
        """
        self._enforce_live_gate()

        # Register the order
        self._orders[order.order_id] = order

        # NEW -> PENDING_SUBMIT
        order = self._transition(order, OrderStatus.PENDING_SUBMIT)

        try:
            # Build CCXT order parameters
            ccxt_order_type = order.order_type.value  # 'market' or 'limit'
            ccxt_side = order.side.value  # 'buy' or 'sell'
            price_param = str(order.price) if order.price is not None else None

            # Coinbase requires a price for market BUY orders on spot markets
            # to calculate total cost (amount * price). Fetch last price if needed.
            if (
                price_param is None
                and ccxt_order_type == "market"
                and ccxt_side == "buy"
                and self._exchange.id == "coinbase"
            ):
                try:
                    ticker = await ccxt_retry(
                        self._exchange.fetch_ticker, order.symbol,
                        max_retries=1, base_delay=0.5, operation=f"fetch_ticker_for_buy({order.symbol})",
                    )
                    price_param = str(ticker.get("last", "0"))
                except Exception:
                    self._log.warning(
                        "live.market_buy_price_fallback_failed",
                        symbol=order.symbol,
                    )

            params: dict[str, Any] = {}
            if order.client_order_id:
                params["clientOrderId"] = order.client_order_id

            ccxt_response = await ccxt_retry(
                self._exchange.create_order,
                order.symbol,
                ccxt_order_type,
                ccxt_side,
                str(order.quantity),
                price_param,
                max_retries=2, base_delay=1.0, operation=f"create_order({order.symbol})",
                params=params,
            )

            # Extract exchange order ID
            exchange_order_id = str(ccxt_response.get("id", ""))

            # Record the mapping
            self._exchange_order_map[order.order_id] = exchange_order_id
            self._reverse_order_map[exchange_order_id] = order.order_id

            # Update order with exchange info (do NOT store yet -- wait for fill data)
            order = order.model_copy(update={
                "exchange_order_id": exchange_order_id,
                "updated_at": datetime.now(tz=UTC),
            })

            # Determine initial status from exchange response
            ccxt_status = ccxt_response.get("status", "open")
            mapped_status = self._map_ccxt_order_status(ccxt_status)

            # PENDING_SUBMIT -> OPEN (or directly to FILLED for instant fills)
            if mapped_status == OrderStatus.FILLED:
                order = self._transition(order, OrderStatus.OPEN)

                # Extract fill data from response
                filled_qty = Decimal(
                    str(ccxt_response.get("filled", order.quantity))
                )
                avg_price = Decimal(
                    str(ccxt_response.get("average", ccxt_response.get("price", "0")))
                )

                # Apply fill data and store atomically before terminal transition
                order = order.model_copy(update={
                    "filled_quantity": filled_qty,
                    "average_fill_price": avg_price,
                    "updated_at": datetime.now(tz=UTC),
                })
                self._orders[order.order_id] = order

                order = self._transition(order, OrderStatus.FILLED)
            else:
                order = self._transition(order, mapped_status)

                # If partially filled on creation
                filled_qty = Decimal(str(ccxt_response.get("filled", "0")))
                if filled_qty > Decimal("0"):
                    avg_price = Decimal(
                        str(ccxt_response.get("average", "0"))
                    )
                    order = order.model_copy(update={
                        "filled_quantity": filled_qty,
                        "average_fill_price": avg_price if avg_price > 0 else None,
                        "updated_at": datetime.now(tz=UTC),
                    })
                    self._orders[order.order_id] = order

            self._log.info(
                "live.order_submitted",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                symbol=order.symbol,
                side=order.side.value,
                type=order.order_type.value,
                quantity=str(order.quantity),
                status=order.status.value,
            )

        except ccxt_async.NetworkError as exc:
            # Transient network error -- reject for now, retry logic is TODO
            self._log.error(
                "live.order_submission_network_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.AuthenticationError as exc:
            self._log.error(
                "live.order_submission_auth_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.InsufficientFunds as exc:
            self._log.error(
                "live.order_submission_insufficient_funds",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.ExchangeError as exc:
            self._log.error(
                "live.order_submission_exchange_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except Exception as exc:
            # Unexpected error -- do NOT silently absorb. Log and re-raise.
            self._log.critical(
                "live.order_submission_unexpected_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        return order

    async def cancel_order(self, order_id: UUID) -> Order:
        """
        Cancel an open or partially-filled order on the exchange.

        Parameters
        ----------
        order_id:
            Internal UUID of the order to cancel.

        Returns
        -------
        Order:
            Updated order with status CANCELED (or FILLED if race occurred).

        Raises
        ------
        RuntimeError:
            If live trading is not enabled.
        KeyError:
            If the order_id is not found.
        """
        self._enforce_live_gate()

        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        exchange_order_id = self._exchange_order_map.get(order_id)

        try:
            if exchange_order_id:
                # TODO: implement full exchange integration
                # - Handle 'order already filled' race condition gracefully
                # - Add retry for transient errors
                await self._exchange.cancel_order(
                    id=exchange_order_id,
                    symbol=order.symbol,
                )

            order = self._transition(order, OrderStatus.CANCELED)

            self._log.info(
                "live.order_canceled",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                symbol=order.symbol,
            )

        except Exception as exc:
            # The order may have already been filled on the exchange
            self._log.warning(
                "live.cancel_failed",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            # Reconcile: fetch actual state from exchange
            order = await self._reconcile_order(order)

        return order

    async def get_order(self, order_id: UUID) -> Order:
        """
        Fetch the current state of an order, reconciling with the exchange.

        For orders with an exchange_order_id, this calls exchange.fetch_order()
        and reconciles the local state with the exchange state.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        Order:
            The most up-to-date order record.

        Raises
        ------
        KeyError:
            If the order_id is not found.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        # If the order is in a terminal state, no need to reconcile
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        if order.status in terminal:
            return order

        return await self._reconcile_order(order)

    async def process_signal(self, signal: Signal) -> list[Order]:
        """
        Convert a trading Signal into zero or more submitted Orders.

        Same risk-gated flow as the paper engine, but using real
        exchange submission.

        Parameters
        ----------
        signal:
            The Signal emitted by a strategy's on_bar call.

        Returns
        -------
        list[Order]:
            Orders created and submitted (may be empty).
        """
        self._enforce_live_gate()

        if signal.direction == SignalDirection.HOLD:
            return []

        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.BUY
            else OrderSide.SELL
        )

        # For SELL signals, verify we hold a position
        if side == OrderSide.SELL:
            position = self._positions.get(signal.symbol)
            if position is None or position.is_flat:
                self._log.info(
                    "live.sell_no_position",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                )
                return []

        # Fetch current ticker for position sizing
        # TODO: implement full exchange integration
        # - Use cached ticker data to avoid rate-limit pressure
        # - Fall back to last known price if ticker fetch fails
        try:
            ticker = await ccxt_retry(
                self._exchange.fetch_ticker, signal.symbol,
                max_retries=2, base_delay=1.0,
                operation=f"fetch_ticker({signal.symbol})",
            )
            last_price = Decimal(str(ticker.get("last", "0")))
            if last_price <= Decimal("0"):
                self._log.error(
                    "live.invalid_ticker_price",
                    symbol=signal.symbol,
                    ticker=ticker,
                )
                return []
        except Exception as exc:
            self._log.error(
                "live.ticker_fetch_failed",
                symbol=signal.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            return []

        # Calculate position size
        equity = await self._fetch_equity()
        quantity = self._risk_manager.calculate_position_size(
            equity=equity,
            entry_price=last_price,
            stop_loss_price=None,
            confidence=signal.confidence,
        )

        if quantity <= Decimal("0"):
            self._log.warning(
                "live.zero_quantity",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                equity=str(equity),
                last_price=str(last_price),
                confidence=signal.confidence,
            )
            return []

        # For SELL, cap at current position
        if side == OrderSide.SELL:
            position = self._positions[signal.symbol]
            quantity = min(quantity, position.quantity)

        # ------------------------------------------------------------------
        # Exchange minimum order size validation
        # Checked after all quantity adjustments (position cap, etc.) so we
        # evaluate the final quantity that would actually be submitted.
        # markets is populated by load_markets() in on_start(); if markets is
        # not yet loaded (e.g. in disabled-gate mode) the dict will be empty
        # and the guard is skipped gracefully.
        # ------------------------------------------------------------------
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(signal.symbol)
        if market:
            limits: dict[str, Any] = market.get("limits") or {}
            amount_limits: dict[str, Any] = limits.get("amount") or {}
            cost_limits: dict[str, Any] = limits.get("cost") or {}

            min_amount = amount_limits.get("min")
            min_cost = cost_limits.get("min")

            notional = quantity * last_price

            if min_amount is not None and float(quantity) < float(min_amount):
                self._log.warning(
                    "live.below_min_amount",
                    symbol=signal.symbol,
                    quantity=str(quantity),
                    min_amount=str(min_amount),
                    msg=(
                        f"Order quantity {quantity} below exchange minimum "
                        f"{min_amount} for {signal.symbol}"
                    ),
                )
                return []

            if min_cost is not None and float(notional) < float(min_cost):
                self._log.warning(
                    "live.below_min_cost",
                    symbol=signal.symbol,
                    notional=str(notional),
                    min_cost=str(min_cost),
                    msg=(
                        f"Order notional {notional} below exchange minimum cost "
                        f"{min_cost} for {signal.symbol}"
                    ),
                )
                return []

        # Build the proposed order
        client_order_id = f"{self._run_id}-{uuid4().hex[:12]}"
        proposed_order = Order(
            client_order_id=client_order_id,
            run_id=self._run_id,
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity.quantize(_QTY_PRECISION, rounding=ROUND_HALF_UP),
        )

        # Pre-trade risk check
        open_positions = [
            p for p in self._positions.values() if not p.is_flat
        ]
        daily_pnl = self._calculate_daily_pnl()
        peak_equity = await self._fetch_peak_equity()

        risk_result = self._risk_manager.pre_trade_check(
            order=proposed_order,
            current_equity=equity,
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            peak_equity=peak_equity,
            market_price=last_price,
        )

        if not risk_result.approved:
            self._log.warning(
                "live.signal_rejected",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
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
                "live.risk_warnings",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                warnings=risk_result.warnings,
            )

        # Submit
        submitted_order = await self.submit_order(proposed_order)
        return [submitted_order]

    async def get_fills(self, order_id: UUID) -> list[Fill]:
        """
        Return all fills for an order by querying the exchange.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        list[Fill]:
            Fills sorted by executed_at ascending.
        """
        # Check local cache first
        cached_fills = self._fills.get(order_id, [])

        exchange_order_id = self._exchange_order_map.get(order_id)
        order = self._orders.get(order_id)

        if exchange_order_id is None or order is None:
            return sorted(cached_fills, key=lambda f: f.executed_at)

        try:
            # Use exchange.has to pick the best fills-fetching method.
            # Binance supports fetchOrderTrades; Coinbase does not.
            if self._exchange.has.get("fetchOrderTrades"):
                ccxt_trades = await self._exchange.fetch_order_trades(
                    id=exchange_order_id,
                    symbol=order.symbol,
                )
            else:
                # Fallback: fetch all recent trades and filter by order ID.
                all_trades = await self._exchange.fetch_my_trades(
                    symbol=order.symbol,
                )
                ccxt_trades = [
                    t for t in all_trades
                    if t.get("order") == exchange_order_id
                ]

            fills: list[Fill] = []
            for trade in ccxt_trades:
                fee_amount, fee_currency = self._extract_fee_from_ccxt(trade)
                fill = Fill(
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=Decimal(str(trade.get("amount", "0"))),
                    price=Decimal(str(trade.get("price", "0"))),
                    fee=fee_amount,
                    fee_currency=fee_currency,
                    is_maker=trade.get("takerOrMaker") == "maker",
                    executed_at=datetime.fromtimestamp(
                        trade.get("timestamp", 0) / 1000, tz=UTC
                    ),
                )
                fills.append(fill)

            # Update local cache
            self._fills[order_id] = fills

            return sorted(fills, key=lambda f: f.executed_at)

        except Exception as exc:
            self._log.warning(
                "live.fetch_fills_failed",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            return sorted(cached_fills, key=lambda f: f.executed_at)

    def get_all_fills(self) -> list[Fill]:
        """Return all fills across all orders, sorted by executed_at.

        Note: returns only locally cached fills. Fills are populated when
        ``get_fills(order_id)`` is called during execution or reconciliation.
        If some orders' fills were never fetched from the exchange, they will
        not be included.
        """
        all_fills: list[Fill] = []
        for fills in self._fills.values():
            all_fills.extend(fills)
        return sorted(all_fills, key=lambda f: f.executed_at)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _reconcile_order(self, order: Order) -> Order:
        """
        Reconcile local order state with the exchange.

        Fetches the current order state from the exchange and updates
        the local record to match, respecting state-machine transitions.

        Parameters
        ----------
        order:
            The local order to reconcile.

        Returns
        -------
        Order:
            Updated order reflecting the exchange state.
        """
        exchange_order_id = self._exchange_order_map.get(order.order_id)
        if exchange_order_id is None:
            return order

        try:
            # TODO: implement full exchange integration
            # - Handle exchange-specific response formats
            # - Retry on transient failures
            ccxt_order = await self._exchange.fetch_order(
                id=exchange_order_id,
                symbol=order.symbol,
            )

            # Update fill data
            filled_qty = Decimal(str(ccxt_order.get("filled", "0")))
            avg_price_raw = ccxt_order.get("average") or ccxt_order.get("price")
            avg_price = (
                Decimal(str(avg_price_raw))
                if avg_price_raw is not None
                else None
            )

            order = order.model_copy(update={
                "filled_quantity": filled_qty,
                "average_fill_price": avg_price,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[order.order_id] = order

            # Determine target status
            ccxt_status = ccxt_order.get("status", "open")
            target_status = self._map_ccxt_order_status(ccxt_status)

            # Handle partial fills
            if (
                target_status == OrderStatus.OPEN
                and filled_qty > Decimal("0")
                and filled_qty < order.quantity
            ):
                target_status = OrderStatus.PARTIAL

            # Only transition if the status actually changed
            if target_status != order.status:
                try:
                    order = self._transition(order, target_status)
                except Exception as exc:
                    self._log.warning(
                        "live.reconciliation_transition_failed",
                        order_id=str(order.order_id),
                        from_status=order.status.value,
                        to_status=target_status.value,
                        error=str(exc),
                    )

            self._log.debug(
                "live.order_reconciled",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                status=order.status.value,
                filled_quantity=str(order.filled_quantity),
            )

        except Exception as exc:
            self._log.warning(
                "live.reconciliation_failed",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )

        return order

    # ------------------------------------------------------------------
    # Equity helpers (exchange-aware)
    # ------------------------------------------------------------------

    async def _fetch_equity(self) -> Decimal:
        """
        Fetch current account equity from the exchange.

        Falls back to local position tracking if exchange call fails.

        Returns
        -------
        Decimal:
            Total account equity in quote currency.
        """
        try:
            # TODO: implement full exchange integration
            # - Cache balance to avoid excessive API calls
            # - Use the appropriate quote currency balance
            balance = await self._exchange.fetch_balance()
            total = balance.get("total", {})
            # Try common quote currencies (including EUR for European exchanges)
            for quote in ("EUR", "USDT", "BUSD", "USD", "USDC"):
                if quote in total and total[quote] is not None and float(total[quote]) > 0:
                    return Decimal(str(total[quote]))

            # Fallback: sum all position values
            self._log.warning("live.equity_fallback_to_local")
        except Exception as exc:
            self._log.warning(
                "live.balance_fetch_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )

        # Fallback: use peak_equity (last known good value) or position tracking
        if self._peak_equity > Decimal("0"):
            self._log.debug("live.equity_fallback_to_peak", peak=str(self._peak_equity))
            return self._peak_equity

        self._log.warning(
            "live.equity_fallback_to_positions",
            msg="Using position-only equity -- cash balance unknown.",
        )
        total = Decimal("0")
        for pos in self._positions.values():
            if not pos.is_flat:
                total += pos.notional_value
        return total

    async def _fetch_peak_equity(self) -> Decimal:
        """
        Return the highest equity observed during this run.

        Uses the INTERNAL peak tracker (seeded at startup from current
        exchange balance). This ensures drawdown is calculated relative
        to THIS run's starting equity, not a historical account high
        from before this run started.

        Returns
        -------
        Decimal:
            Peak equity (highest equity seen since engine start).
        """
        current = await self._fetch_equity()
        # If peak is 0 (not yet seeded), initialize to current equity
        if self._peak_equity <= Decimal("0"):
            self._peak_equity = current
        elif current > self._peak_equity:
            self._peak_equity = current
        return self._peak_equity

    def _calculate_daily_pnl(self) -> Decimal:
        """
        Calculate daily PnL from local position tracking.

        Returns
        -------
        Decimal:
            Net daily PnL in quote currency.
        """
        # TODO: implement full exchange integration
        # - Wire into PortfolioAccounting for accurate daily PnL
        total_pnl = Decimal("0")
        for pos in self._positions.values():
            total_pnl += pos.realised_pnl
        return total_pnl

    # ------------------------------------------------------------------
    # Position sync
    # ------------------------------------------------------------------

    async def sync_positions(self) -> dict[str, Position]:
        """
        Synchronize local position state with exchange balances.

        This should be called on startup and periodically to ensure
        the local position tracker reflects reality.

        Returns
        -------
        dict[str, Position]:
            Updated positions keyed by symbol.
        """
        # TODO: implement full exchange integration
        # - Fetch exchange balances
        # - Reconcile with local position state
        # - Handle positions opened outside the bot
        self._log.info("live.sync_positions_stub")
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """
        Initialize the live trading engine.

        Validates exchange connectivity and loads initial state.
        """
        await super().on_start()

        if not self._enable_live_trading:
            self._log.warning(
                "live.engine_started_disabled",
                msg="Live trading gate is OFF. Orders will be rejected.",
            )
            return

        try:
            # TODO: implement full exchange integration
            # - Validate API key permissions (trade, read)
            # - Load exchange markets for symbol validation
            # - Sync existing positions
            await ccxt_retry(
                self._exchange.load_markets,
                max_retries=3, base_delay=2.0, operation="load_markets",
            )
            # Seed peak equity so the first drawdown check has a baseline.
            self._peak_equity = await self._fetch_equity()
            self._log.info(
                "live.engine_started",
                exchange=self._exchange.id,
                markets_loaded=len(self._exchange.markets),
            )
        except Exception as exc:
            self._log.error(
                "live.engine_start_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            raise

    async def on_stop(self) -> None:
        """
        Shut down the live trading engine.

        Cancels open orders and closes the exchange connection.
        """
        # Cancel all open orders directly via exchange (bypass live gate for shutdown)
        open_orders = self.get_open_orders()
        for order in open_orders:
            try:
                exchange_order_id = self._exchange_order_map.get(order.order_id)
                if exchange_order_id:
                    await self._exchange.cancel_order(
                        id=exchange_order_id,
                        symbol=order.symbol,
                    )
                self._transition(order, OrderStatus.CANCELED)
            except Exception:
                self._log.warning(
                    "live.cancel_failed_on_stop",
                    order_id=str(order.order_id),
                )

        # Close exchange connection
        try:
            # TODO: implement full exchange integration
            # - Graceful connection shutdown
            if hasattr(self._exchange, "close"):
                await self._exchange.close()
        except Exception as exc:
            self._log.warning(
                "live.exchange_close_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )

        self._log.info(
            "live.engine_stopped",
            total_orders=len(self._orders),
            total_fills=sum(len(f) for f in self._fills.values()),
        )

        await super().on_stop()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LiveExecutionEngine("
            f"run_id={self._run_id!r}, "
            f"exchange={getattr(self._exchange, 'id', 'unknown')!r}, "
            f"live_enabled={self._enable_live_trading}, "
            f"orders={len(self._orders)})"
        )
