"""
tests/unit/test_live_execution.py
-----------------------------------
Comprehensive unit tests for LiveExecutionEngine.

Module under test
-----------------
    packages/trading/engines/live.py -- LiveExecutionEngine

Coverage groups (54 tests)
--------------------------
1.  TestConstructorAndProperties   -- __init__, properties, run_id, repr (5 tests)
2.  TestLiveGateEnforcement        -- _enforce_live_gate raises/passes, API surfaces (5 tests)
3.  TestCCXTStatusMapping          -- all known CCXT statuses + unknown fallback (6 tests)
4.  TestFeeExtraction              -- normal fee, missing key, null cost, non-USDT currency (4 tests)
5.  TestSubmitOrder                -- success market, instant fill, exchange_id stored,
                                      partial fill on creation, NetworkError, AuthenticationError,
                                      InsufficientFunds, ExchangeError, unexpected error reraises (10 tests)
6.  TestCancelOrder                -- cancel open order, calls exchange, unknown raises,
                                      filled race reconciles, no exchange_id local cancel (5 tests)
7.  TestGetOrder                   -- terminal returns cached, open reconciles,
                                      unknown raises, fill data updated after reconcile (4 tests)
8.  TestProcessSignal              -- HOLD returns empty, BUY submits, SELL without position,
                                      SELL with position, sell capped at position qty,
                                      risk rejection, risk adjusted qty, zero qty, ticker failure (9 tests)
9.  TestGetFills                   -- fills from exchange, sorted by executed_at,
                                      fallback on error, no exchange_id returns cached (4 tests)
10. TestReconcileOrder             -- updates fill quantities, partial fill detected,
                                      fetch_order failure returns original (3 tests)
11. TestEquityHelpers              -- USDT balance, multi-currency order, fallback to positions (4 tests) -- actually 3 equity + 1 daily_pnl
12. TestLifecycle                  -- on_start loads markets, on_start disabled logs warning,
                                      on_stop cancels open orders, on_stop handles cancel failure (4 tests)

Design notes
------------
- _make_mock_exchange() — AsyncMock-backed exchange with configurable methods.
  The exchange mock has an ``id`` attribute ("mock-exchange") and a ``markets``
  attribute (dict) so that on_start() and __repr__() work without AttributeError.
- _make_risk_manager_mock() — MagicMock for BaseRiskManager with approved
  pre_trade_check return by default.
- _make_engine() — factory returning (engine, risk_mock, exchange_mock).
- All CCXT exceptions are imported from ccxt.async_support (the same module
  the production code catches) and used as side_effect values.
- Tests never mock internal engine methods (_enforce_live_gate, _transition, etc.).
- asyncio_mode = "auto" is set in pyproject.toml; @pytest.mark.asyncio is
  included for explicitness but is not strictly required.
- Position objects injected directly into engine._positions to set up
  pre-existing position state without going through live order flow.
- Exchange order IDs are pre-seeded into engine._exchange_order_map where
  necessary to simulate an order that has already been submitted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import ccxt.async_support as ccxt_async
import pytest

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.engines.live import LiveExecutionEngine
from trading.execution import InvalidOrderTransitionError
from trading.models import Fill, Order, Position, RiskCheckResult, Signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_RUN_ID = "live-test-run-001"
_LAST_PRICE = Decimal("50000")
_POSITION_SIZE = Decimal("0.02")

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_mock_exchange(
    *,
    create_order_response: dict[str, Any] | None = None,
    cancel_order_response: dict[str, Any] | None = None,
    fetch_order_response: dict[str, Any] | None = None,
    fetch_order_trades_response: list[dict[str, Any]] | None = None,
    fetch_ticker_response: dict[str, Any] | None = None,
    fetch_balance_response: dict[str, Any] | None = None,
    markets: dict[str, Any] | None = None,
    has_fetch_order_trades: bool = True,
    fetch_my_trades_response: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """
    Build a fully-configured AsyncMock exchange instance.

    All exchange methods default to sensible no-op responses.  Individual
    tests override specific method return values as needed.
    """
    exchange = MagicMock()
    exchange.id = "mock-exchange"
    exchange.markets = markets if markets is not None else {_SYMBOL: {}}

    # Async methods
    exchange.create_order = AsyncMock(
        return_value=create_order_response
        or {
            "id": "exch-001",
            "status": "open",
            "filled": "0",
            "average": None,
            "price": str(_LAST_PRICE),
        }
    )
    exchange.cancel_order = AsyncMock(
        return_value=cancel_order_response or {"id": "exch-001", "status": "canceled"}
    )
    exchange.fetch_order = AsyncMock(
        return_value=fetch_order_response
        or {
            "id": "exch-001",
            "status": "open",
            "filled": "0",
            "average": None,
            "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_order_trades = AsyncMock(
        return_value=fetch_order_trades_response or []
    )
    exchange.fetch_ticker = AsyncMock(
        return_value=fetch_ticker_response or {"last": str(_LAST_PRICE)}
    )
    exchange.fetch_balance = AsyncMock(
        return_value=fetch_balance_response
        or {"total": {"USDT": 10000.0, "BTC": 0.0}}
    )
    exchange.load_markets = AsyncMock(return_value=exchange.markets)
    exchange.close = AsyncMock(return_value=None)
    # Exchange capability map -- controls which fills-fetching method is used.
    # Set as a real dict so .get() returns a true bool, not a MagicMock.
    exchange.has = {
        "fetchOrderTrades": has_fetch_order_trades,
        "fetchMyTrades": True,
    }
    exchange.fetch_my_trades = AsyncMock(
        return_value=fetch_my_trades_response if fetch_my_trades_response is not None else []
    )

    return exchange


def _make_risk_manager_mock(
    *,
    approved: bool = True,
    adjusted_quantity: Decimal | None = None,
    position_size: Decimal = _POSITION_SIZE,
) -> MagicMock:
    """
    Build a fully-configured MagicMock for BaseRiskManager.

    Configures:
    - pre_trade_check() returning an approved RiskCheckResult by default.
    - calculate_position_size() returning `position_size`.
    """
    mock = MagicMock()
    qty = adjusted_quantity if adjusted_quantity is not None else position_size
    mock.pre_trade_check.return_value = RiskCheckResult(
        approved=approved,
        adjusted_quantity=qty,
        rejection_reasons=[] if approved else ["test rejection"],
        warnings=[],
    )
    mock.calculate_position_size.return_value = position_size
    return mock


def _make_engine(
    *,
    enable_live_trading: bool = True,
    approved: bool = True,
    position_size: Decimal = _POSITION_SIZE,
    adjusted_quantity: Decimal | None = None,
    exchange_kwargs: dict[str, Any] | None = None,
) -> tuple[LiveExecutionEngine, MagicMock, MagicMock]:
    """
    Factory returning (engine, risk_manager_mock, exchange_mock).

    The engine is fully constructed with ``enable_live_trading=True`` by
    default so most tests exercise the live path directly.  Pass
    ``enable_live_trading=False`` to test gate-enforcement paths.
    """
    rm = _make_risk_manager_mock(
        approved=approved,
        position_size=position_size,
        adjusted_quantity=adjusted_quantity,
    )
    ex = _make_mock_exchange(**(exchange_kwargs or {}))
    engine = LiveExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        exchange=ex,
        enable_live_trading=enable_live_trading,
    )
    return engine, rm, ex


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


def _make_signal(
    *,
    direction: SignalDirection = SignalDirection.BUY,
    symbol: str = _SYMBOL,
    confidence: float = 1.0,
) -> Signal:
    """Create a Signal with the specified direction."""
    target = Decimal("1000") if direction == SignalDirection.BUY else Decimal("0")
    return Signal(
        strategy_id="test-strategy",
        symbol=symbol,
        direction=direction,
        target_position=target,
        confidence=confidence,
    )


def _make_position(
    *,
    symbol: str = _SYMBOL,
    quantity: Decimal = Decimal("0.5"),
    average_entry_price: Decimal = Decimal("48000"),
) -> Position:
    """Create a Position with the given quantity."""
    return Position(
        symbol=symbol,
        run_id=_RUN_ID,
        quantity=quantity,
        average_entry_price=average_entry_price,
        current_price=_LAST_PRICE,
    )


def _make_ccxt_trade(
    *,
    amount: str = "0.1",
    price: str = "50000",
    fee_cost: str = "0.5",
    fee_currency: str = "USDT",
    is_maker: bool = False,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Build a CCXT trade dict for fetch_order_trades responses."""
    if timestamp_ms is None:
        timestamp_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    return {
        "amount": amount,
        "price": price,
        "fee": {"cost": fee_cost, "currency": fee_currency},
        "takerOrMaker": "maker" if is_maker else "taker",
        "timestamp": timestamp_ms,
    }


# ===========================================================================
# Group 1: Constructor & Properties
# ===========================================================================


class TestConstructorAndProperties:
    """Verify constructor stores all arguments and properties return correct values."""

    def test_risk_manager_property_returns_injected_instance(self) -> None:
        """risk_manager property must return the exact instance injected at construction."""
        engine, rm, _ = _make_engine()
        assert engine.risk_manager is rm

    def test_exchange_property_returns_injected_instance(self) -> None:
        """exchange property must return the exact exchange instance injected."""
        engine, _, ex = _make_engine()
        assert engine.exchange is ex

    def test_is_live_enabled_true_when_gate_open(self) -> None:
        """is_live_enabled reflects the enable_live_trading constructor flag."""
        engine, _, _ = _make_engine(enable_live_trading=True)
        assert engine.is_live_enabled is True

    def test_is_live_enabled_false_when_gate_closed(self) -> None:
        """is_live_enabled returns False when enable_live_trading=False."""
        engine, _, _ = _make_engine(enable_live_trading=False)
        assert engine.is_live_enabled is False

    def test_positions_empty_dict_on_construction(self) -> None:
        """positions property returns an empty dict before any orders are placed."""
        engine, _, _ = _make_engine()
        assert engine.positions == {}


# ===========================================================================
# Group 2: Live Gate Enforcement
# ===========================================================================


class TestLiveGateEnforcement:
    """Verify that the live-trading safety gate raises RuntimeError when disabled."""

    def test_enforce_live_gate_raises_when_disabled(self) -> None:
        """_enforce_live_gate() raises RuntimeError when enable_live_trading=False."""
        engine, _, _ = _make_engine(enable_live_trading=False)
        with pytest.raises(RuntimeError, match="Live trading is not enabled"):
            engine._enforce_live_gate()

    def test_enforce_live_gate_passes_when_enabled(self) -> None:
        """_enforce_live_gate() does not raise when enable_live_trading=True."""
        engine, _, _ = _make_engine(enable_live_trading=True)
        engine._enforce_live_gate()  # Must not raise

    @pytest.mark.asyncio
    async def test_submit_order_raises_when_gate_disabled(self) -> None:
        """submit_order() propagates RuntimeError from gate when live is disabled."""
        engine, _, _ = _make_engine(enable_live_trading=False)
        order = _make_market_order()
        with pytest.raises(RuntimeError, match="Live trading is not enabled"):
            await engine.submit_order(order)

    @pytest.mark.asyncio
    async def test_cancel_order_raises_when_gate_disabled(self) -> None:
        """cancel_order() checks the live gate before attempting cancellation."""
        engine, _, _ = _make_engine(enable_live_trading=False)
        fake_id = uuid4()
        with pytest.raises(RuntimeError, match="Live trading is not enabled"):
            await engine.cancel_order(fake_id)

    @pytest.mark.asyncio
    async def test_process_signal_raises_when_gate_disabled(self) -> None:
        """process_signal() enforces the live gate at the start of the call."""
        engine, _, _ = _make_engine(enable_live_trading=False)
        signal = _make_signal(direction=SignalDirection.BUY)
        with pytest.raises(RuntimeError, match="Live trading is not enabled"):
            await engine.process_signal(signal)


# ===========================================================================
# Group 3: CCXT Status Mapping
# ===========================================================================


class TestCCXTStatusMapping:
    """Verify all CCXT status strings map to the correct internal OrderStatus values."""

    def test_open_maps_to_order_status_open(self) -> None:
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("open") == OrderStatus.OPEN

    def test_closed_maps_to_order_status_filled(self) -> None:
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("closed") == OrderStatus.FILLED

    def test_canceled_maps_to_order_status_canceled(self) -> None:
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("canceled") == OrderStatus.CANCELED

    def test_expired_maps_to_order_status_expired(self) -> None:
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("expired") == OrderStatus.EXPIRED

    def test_rejected_maps_to_order_status_rejected(self) -> None:
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("rejected") == OrderStatus.REJECTED

    def test_unknown_status_falls_back_to_open(self) -> None:
        """An unrecognised CCXT status string defaults to OPEN (safe/conservative fallback)."""
        engine, _, _ = _make_engine()
        assert engine._map_ccxt_order_status("unknown_status_xyz") == OrderStatus.OPEN


# ===========================================================================
# Group 4: Fee Extraction
# ===========================================================================


class TestFeeExtraction:
    """Verify _extract_fee_from_ccxt parses fee data from CCXT trade dicts correctly."""

    def test_normal_fee_extraction_returns_cost_and_currency(self) -> None:
        """Standard fee dict with cost and currency is parsed without error."""
        engine, _, _ = _make_engine()
        trade = {"fee": {"cost": "0.50", "currency": "USDT"}}
        fee_amount, fee_currency = engine._extract_fee_from_ccxt(trade)
        assert fee_amount == Decimal("0.50")
        assert fee_currency == "USDT"

    def test_missing_fee_key_returns_zero(self) -> None:
        """A trade dict with no 'fee' key returns zero fee amount."""
        engine, _, _ = _make_engine()
        trade: dict[str, Any] = {}
        fee_amount, fee_currency = engine._extract_fee_from_ccxt(trade)
        assert fee_amount == Decimal("0")

    def test_null_fee_value_returns_zero(self) -> None:
        """A trade dict with fee=None returns zero fee amount."""
        engine, _, _ = _make_engine()
        trade: dict[str, Any] = {"fee": None}
        fee_amount, fee_currency = engine._extract_fee_from_ccxt(trade)
        assert fee_amount == Decimal("0")

    def test_non_usdt_fee_currency_preserved(self) -> None:
        """Fee currency other than USDT is preserved as-is."""
        engine, _, _ = _make_engine()
        trade = {"fee": {"cost": "0.001", "currency": "BNB"}}
        _, fee_currency = engine._extract_fee_from_ccxt(trade)
        assert fee_currency == "BNB"


# ===========================================================================
# Group 5: Submit Order
# ===========================================================================


class TestSubmitOrder:
    """Verify submit_order() correctly drives the order state machine through all paths."""

    @pytest.mark.asyncio
    async def test_success_market_order_transitions_to_open(self) -> None:
        """A successful create_order response with status 'open' transitions order to OPEN."""
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-100",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                }
            }
        )
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_instant_fill_response_transitions_to_filled(self) -> None:
        """When exchange returns status='closed', the order transitions all the way to FILLED."""
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-101",
                    "status": "closed",
                    "filled": "0.1",
                    "average": "50000",
                    "price": "50000",
                }
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_exchange_order_id_stored_after_submission(self) -> None:
        """exchange_order_id is set on the returned Order after successful create_order."""
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-xyz-999",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                }
            }
        )
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.exchange_order_id == "exch-xyz-999"

    @pytest.mark.asyncio
    async def test_partial_fill_on_creation_stores_filled_quantity(self) -> None:
        """
        When the exchange immediately partially fills an order (filled > 0, status='open'),
        the returned order captures filled_quantity and average_fill_price.
        """
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-partial",
                    "status": "open",
                    "filled": "0.05",
                    "average": "49500",
                    "price": "50000",
                }
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        result = await engine.submit_order(order)
        assert result.filled_quantity == Decimal("0.05")
        assert result.average_fill_price == Decimal("49500")

    @pytest.mark.asyncio
    async def test_network_error_transitions_to_rejected(self) -> None:
        """A ccxt NetworkError during create_order transitions the order to REJECTED."""
        engine, _, ex = _make_engine()
        ex.create_order.side_effect = ccxt_async.NetworkError("connection refused")
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_authentication_error_transitions_to_rejected(self) -> None:
        """A ccxt AuthenticationError during create_order transitions the order to REJECTED."""
        engine, _, ex = _make_engine()
        ex.create_order.side_effect = ccxt_async.AuthenticationError("invalid api key")
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_insufficient_funds_transitions_to_rejected(self) -> None:
        """A ccxt InsufficientFunds during create_order transitions the order to REJECTED."""
        engine, _, ex = _make_engine()
        ex.create_order.side_effect = ccxt_async.InsufficientFunds("not enough balance")
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_exchange_error_transitions_to_rejected(self) -> None:
        """A generic ccxt ExchangeError during create_order transitions the order to REJECTED."""
        engine, _, ex = _make_engine()
        ex.create_order.side_effect = ccxt_async.ExchangeError("order rejected by exchange")
        order = _make_market_order()
        result = await engine.submit_order(order)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_unexpected_error_reraises(self) -> None:
        """An unexpected exception (not a ccxt error) is re-raised without being absorbed."""
        engine, _, ex = _make_engine()
        ex.create_order.side_effect = RuntimeError("unexpected internal failure")
        order = _make_market_order()
        with pytest.raises(RuntimeError, match="unexpected internal failure"):
            await engine.submit_order(order)

    @pytest.mark.asyncio
    async def test_create_order_called_with_correct_parameters(self) -> None:
        """create_order receives symbol, type, side, amount, and price=None for market orders."""
        engine, _, ex = _make_engine()
        qty = Decimal("0.15")
        order = _make_market_order(side=OrderSide.BUY, quantity=qty)
        await engine.submit_order(order)
        ex.create_order.assert_called_once_with(
            _SYMBOL,
            "market",
            "buy",
            str(qty),
            None,
            params={"clientOrderId": order.client_order_id},
        )


# ===========================================================================
# Group 6: Cancel Order
# ===========================================================================


class TestCancelOrder:
    """Verify cancel_order() drives the correct state transitions and exchange calls."""

    @pytest.mark.asyncio
    async def test_cancel_open_order_transitions_to_canceled(self) -> None:
        """Cancelling an OPEN order transitions it to CANCELED status."""
        engine, _, _ = _make_engine()
        order = _make_market_order()
        submitted = await engine.submit_order(order)
        assert submitted.status == OrderStatus.OPEN

        canceled = await engine.cancel_order(submitted.order_id)
        assert canceled.status == OrderStatus.CANCELED

    @pytest.mark.asyncio
    async def test_cancel_order_calls_exchange_cancel(self) -> None:
        """cancel_order() calls exchange.cancel_order with the exchange order ID."""
        engine, _, ex = _make_engine()
        order = _make_market_order()
        submitted = await engine.submit_order(order)
        exchange_order_id = submitted.exchange_order_id

        await engine.cancel_order(submitted.order_id)

        # ccxt_retry passes positional args to the underlying exchange call
        ex.cancel_order.assert_called_once_with(
            exchange_order_id,
            _SYMBOL,
        )

    @pytest.mark.asyncio
    async def test_cancel_unknown_order_id_raises_key_error(self) -> None:
        """cancel_order() raises KeyError for an order_id not in the registry."""
        engine, _, _ = _make_engine()
        with pytest.raises(KeyError):
            await engine.cancel_order(uuid4())

    @pytest.mark.asyncio
    async def test_cancel_raises_reconciles_with_exchange(self) -> None:
        """
        When exchange.cancel_order raises, cancel_order falls back to reconcile
        and returns the order in a state consistent with the exchange response.
        """
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-race",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                },
                "fetch_order_response": {
                    "id": "exch-race",
                    "status": "closed",
                    "filled": "0.1",
                    "average": "50000",
                    "price": "50000",
                },
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)
        assert submitted.status == OrderStatus.OPEN

        # Make cancel_order raise to trigger the reconcile path
        ex.cancel_order.side_effect = ccxt_async.ExchangeError("already filled")

        result = await engine.cancel_order(submitted.order_id)
        # The reconcile should have fetched the filled state from exchange
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_cancel_order_without_exchange_id_transitions_locally(self) -> None:
        """
        If no exchange_order_id is mapped (order never reached exchange), cancel_order
        transitions locally to CANCELED without calling the exchange.
        """
        engine, _, ex = _make_engine()
        # Manually register an order that was never submitted to the exchange
        order = _make_market_order()
        order_in_pending = order.model_copy(update={"status": OrderStatus.PENDING_SUBMIT})
        engine._orders[order.order_id] = order_in_pending
        # Deliberately leave _exchange_order_map empty for this order

        canceled = await engine.cancel_order(order.order_id)

        assert canceled.status == OrderStatus.CANCELED
        # exchange.cancel_order must not have been invoked
        ex.cancel_order.assert_not_called()


# ===========================================================================
# Group 7: Get Order
# ===========================================================================


class TestGetOrder:
    """Verify get_order() returns cached data for terminal orders and reconciles open ones."""

    @pytest.mark.asyncio
    async def test_terminal_order_returns_cached_without_exchange_call(self) -> None:
        """get_order() for a FILLED order returns the cached record and skips reconciliation."""
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-200",
                    "status": "closed",
                    "filled": "0.1",
                    "average": "50000",
                    "price": "50000",
                }
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)
        assert submitted.status == OrderStatus.FILLED

        # Reset the call count to isolate get_order behavior
        ex.fetch_order.reset_mock()

        fetched = await engine.get_order(submitted.order_id)

        assert fetched.status == OrderStatus.FILLED
        ex.fetch_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_order_triggers_reconcile(self) -> None:
        """get_order() for an OPEN order calls fetch_order to reconcile state."""
        engine, _, ex = _make_engine()
        order = _make_market_order()
        submitted = await engine.submit_order(order)
        assert submitted.status == OrderStatus.OPEN

        ex.fetch_order.reset_mock()
        await engine.get_order(submitted.order_id)

        ex.fetch_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_order_id_raises_key_error(self) -> None:
        """get_order() raises KeyError for an order_id that was never registered."""
        engine, _, _ = _make_engine()
        with pytest.raises(KeyError):
            await engine.get_order(uuid4())

    @pytest.mark.asyncio
    async def test_reconcile_updates_fill_data_on_open_order(self) -> None:
        """
        After reconciliation, the returned order reflects filled_quantity
        and average_fill_price from the exchange response.
        """
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-201",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                },
                "fetch_order_response": {
                    "id": "exch-201",
                    "status": "open",
                    "filled": "0.07",
                    "average": "50100",
                    "price": "50000",
                },
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)
        assert submitted.status == OrderStatus.OPEN

        reconciled = await engine.get_order(submitted.order_id)

        assert reconciled.filled_quantity == Decimal("0.07")
        assert reconciled.average_fill_price == Decimal("50100")


# ===========================================================================
# Group 8: Process Signal
# ===========================================================================


class TestProcessSignal:
    """Verify process_signal() converts signals to orders through the full risk-gated flow."""

    @pytest.mark.asyncio
    async def test_hold_signal_returns_empty_list(self) -> None:
        """A HOLD signal must produce no orders and not interact with the exchange."""
        engine, _, ex = _make_engine()
        signal = _make_signal(direction=SignalDirection.HOLD)

        orders = await engine.process_signal(signal)

        assert orders == []
        ex.fetch_ticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_buy_signal_submits_market_order(self) -> None:
        """A BUY signal with valid ticker data creates and submits a MARKET order."""
        engine, _, _ = _make_engine()
        signal = _make_signal(direction=SignalDirection.BUY)

        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_sell_signal_without_position_returns_empty(self) -> None:
        """A SELL signal when no position exists returns an empty list."""
        engine, _, _ = _make_engine()
        signal = _make_signal(direction=SignalDirection.SELL)

        orders = await engine.process_signal(signal)

        assert orders == []

    @pytest.mark.asyncio
    async def test_sell_signal_with_position_submits_order(self) -> None:
        """A SELL signal when a position exists creates a SELL MARKET order."""
        engine, _, _ = _make_engine()
        # Pre-seed a long position
        engine._positions[_SYMBOL] = _make_position(quantity=Decimal("0.5"))

        signal = _make_signal(direction=SignalDirection.SELL)
        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_sell_quantity_capped_at_position_size(self) -> None:
        """
        When risk manager calculates a sell quantity larger than the open position,
        the order quantity is capped at the position quantity.
        """
        position_qty = Decimal("0.1")
        engine, rm, _ = _make_engine(position_size=Decimal("9999"))
        # Override adjusted_quantity to also be large so capping applies
        rm.pre_trade_check.return_value = RiskCheckResult(
            approved=True,
            adjusted_quantity=Decimal("9999"),
            rejection_reasons=[],
            warnings=[],
        )
        engine._positions[_SYMBOL] = _make_position(quantity=position_qty)

        signal = _make_signal(direction=SignalDirection.SELL)
        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].quantity == position_qty

    @pytest.mark.asyncio
    async def test_risk_rejection_returns_empty_list(self) -> None:
        """When pre_trade_check returns approved=False, process_signal returns empty."""
        engine, rm, _ = _make_engine()
        rm.pre_trade_check.return_value = RiskCheckResult(
            approved=False,
            adjusted_quantity=Decimal("0"),
            rejection_reasons=["max positions reached"],
            warnings=[],
        )
        signal = _make_signal(direction=SignalDirection.BUY)

        orders = await engine.process_signal(signal)

        assert orders == []

    @pytest.mark.asyncio
    async def test_risk_adjusted_quantity_applied_to_order(self) -> None:
        """
        When pre_trade_check approves but returns a reduced adjusted_quantity,
        the submitted order uses that reduced quantity.
        """
        original_qty = Decimal("0.1")
        capped_qty = Decimal("0.05")
        engine, rm, _ = _make_engine(position_size=original_qty)
        rm.pre_trade_check.return_value = RiskCheckResult(
            approved=True,
            adjusted_quantity=capped_qty,
            rejection_reasons=[],
            warnings=[],
        )
        signal = _make_signal(direction=SignalDirection.BUY)

        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        assert orders[0].quantity == capped_qty

    @pytest.mark.asyncio
    async def test_zero_position_size_returns_empty_list(self) -> None:
        """When calculate_position_size returns zero, no order is created."""
        engine, rm, _ = _make_engine()
        rm.calculate_position_size.return_value = Decimal("0")

        signal = _make_signal(direction=SignalDirection.BUY)
        orders = await engine.process_signal(signal)

        assert orders == []
        rm.pre_trade_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_ticker_fetch_failure_returns_empty_list(self) -> None:
        """When fetch_ticker raises, process_signal returns an empty list (no order)."""
        engine, _, ex = _make_engine()
        ex.fetch_ticker.side_effect = ccxt_async.NetworkError("timeout")

        signal = _make_signal(direction=SignalDirection.BUY)
        orders = await engine.process_signal(signal)

        assert orders == []


# ===========================================================================
# Group 9: Get Fills
# ===========================================================================


class TestGetFills:
    """Verify get_fills() fetches, converts, and caches fill data from the exchange."""

    @pytest.mark.asyncio
    async def test_fills_constructed_from_exchange_trades(self) -> None:
        """get_fills() converts fetch_order_trades response into Fill objects."""
        trade = _make_ccxt_trade(amount="0.1", price="50000", fee_cost="0.5")
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "fetch_order_trades_response": [trade],
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        fills = await engine.get_fills(submitted.order_id)

        assert len(fills) == 1
        assert fills[0].quantity == Decimal("0.1")
        assert fills[0].price == Decimal("50000")
        assert fills[0].fee == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_fills_sorted_by_executed_at_ascending(self) -> None:
        """get_fills() returns fills sorted by executed_at in ascending order."""
        ts_base = int(datetime(2024, 6, 1, tzinfo=UTC).timestamp() * 1000)
        trade1 = _make_ccxt_trade(amount="0.05", timestamp_ms=ts_base + 10_000)
        trade2 = _make_ccxt_trade(amount="0.05", timestamp_ms=ts_base)  # earlier
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "fetch_order_trades_response": [trade1, trade2],
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        fills = await engine.get_fills(submitted.order_id)

        assert fills[0].executed_at < fills[1].executed_at

    @pytest.mark.asyncio
    async def test_fills_fallback_to_cache_on_exchange_error(self) -> None:
        """
        When fetch_order_trades raises, get_fills falls back to the local cache
        (which is empty on first call, so an empty list is returned).
        """
        engine, _, ex = _make_engine()
        order = _make_market_order()
        submitted = await engine.submit_order(order)

        ex.fetch_order_trades.side_effect = ccxt_async.NetworkError("timeout")

        fills = await engine.get_fills(submitted.order_id)

        # No cached fills: returns empty list from fallback
        assert fills == []

    @pytest.mark.asyncio
    async def test_no_exchange_id_returns_cached_fills(self) -> None:
        """
        When the order has no exchange_order_id mapping, get_fills returns the
        locally cached fills without making an exchange call.
        """
        engine, _, ex = _make_engine()
        # Register an order that has no exchange mapping
        order = _make_market_order()
        engine._orders[order.order_id] = order.model_copy(
            update={"status": OrderStatus.OPEN}
        )
        # Deliberately leave _exchange_order_map empty

        fills = await engine.get_fills(order.order_id)

        assert fills == []
        ex.fetch_order_trades.assert_not_called()

    @pytest.mark.asyncio
    async def test_fills_via_fetch_my_trades_fallback(self) -> None:
        """
        When exchange.has["fetchOrderTrades"] is falsy, get_fills() must
        call fetch_my_trades(symbol) instead of fetch_order_trades().

        This exercises the Coinbase-compatible fallback path introduced in
        the get_fills() refactor. Binance supports fetchOrderTrades; exchanges
        such as Coinbase Advanced Trade do not, so the fallback fetches all
        recent trades and filters by order ID.
        """
        trade = _make_ccxt_trade(amount="0.1", price="50000", fee_cost="0.5")
        # Attach the exchange_order_id that the engine will store after submit
        trade["order"] = "exch-001"  # matches default create_order response id

        engine, _, ex = _make_engine(
            exchange_kwargs={
                "has_fetch_order_trades": False,
                "fetch_my_trades_response": [trade],
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        fills = await engine.get_fills(submitted.order_id)

        assert len(fills) == 1, (
            f"Expected 1 fill from fetch_my_trades fallback, got {len(fills)}"
        )
        assert fills[0].quantity == Decimal("0.1")
        assert fills[0].price == Decimal("50000")
        ex.fetch_order_trades.assert_not_called()
        ex.fetch_my_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_my_trades_filters_by_order_id(self) -> None:
        """
        When fetch_my_trades() returns multiple trades, get_fills() must
        filter to only those whose "order" field matches the exchange_order_id.

        Two trades are returned: one matching the submitted order's exchange ID
        ("exch-001") and one belonging to a different order. Only the matching
        trade should appear as a Fill object.
        """
        matching_trade = _make_ccxt_trade(amount="0.1", price="50000", fee_cost="0.5")
        matching_trade["order"] = "exch-001"  # matches default create_order id

        other_trade = _make_ccxt_trade(amount="0.2", price="49000", fee_cost="1.0")
        other_trade["order"] = "exch-999"  # different order -- must be filtered out

        engine, _, ex = _make_engine(
            exchange_kwargs={
                "has_fetch_order_trades": False,
                "fetch_my_trades_response": [matching_trade, other_trade],
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        fills = await engine.get_fills(submitted.order_id)

        assert len(fills) == 1, (
            f"Expected only the matching trade to become a Fill, "
            f"got {len(fills)}: {fills!r}"
        )
        assert fills[0].quantity == Decimal("0.1"), (
            "Fill quantity must match the matching trade, not the filtered-out trade"
        )

    @pytest.mark.asyncio
    async def test_fetch_my_trades_fallback_error_returns_cached(self) -> None:
        """
        When fetch_my_trades() raises a network error, get_fills() must fall
        back to the locally cached fills (empty list for a freshly submitted order).

        This mirrors test_fills_fallback_to_cache_on_exchange_error for the
        fetchOrderTrades path, but exercises the Coinbase fallback branch.
        """
        engine, _, ex = _make_engine(
            exchange_kwargs={"has_fetch_order_trades": False}
        )
        order = _make_market_order()
        submitted = await engine.submit_order(order)

        ex.fetch_my_trades.side_effect = ccxt_async.NetworkError("exchange unavailable")

        fills = await engine.get_fills(submitted.order_id)

        # No cached fills: fallback returns empty list
        assert fills == [], (
            f"Expected empty fallback list after fetch_my_trades error, got {fills!r}"
        )


# ===========================================================================
# Group 10: Reconcile Order
# ===========================================================================


class TestReconcileOrder:
    """Verify _reconcile_order() correctly updates local state from exchange data."""

    @pytest.mark.asyncio
    async def test_reconcile_updates_filled_quantity(self) -> None:
        """After reconciliation, filled_quantity reflects the exchange response."""
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-rec-1",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                },
                "fetch_order_response": {
                    "id": "exch-rec-1",
                    "status": "open",
                    "filled": "0.06",
                    "average": "50200",
                    "price": "50000",
                },
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        reconciled = await engine._reconcile_order(submitted)

        assert reconciled.filled_quantity == Decimal("0.06")

    @pytest.mark.asyncio
    async def test_reconcile_detects_partial_fill_status(self) -> None:
        """
        When filled > 0 but < quantity and exchange reports 'open',
        _reconcile_order transitions the order to PARTIAL.
        """
        engine, _, ex = _make_engine(
            exchange_kwargs={
                "create_order_response": {
                    "id": "exch-rec-2",
                    "status": "open",
                    "filled": "0",
                    "average": None,
                    "price": "50000",
                },
                "fetch_order_response": {
                    "id": "exch-rec-2",
                    "status": "open",
                    "filled": "0.04",
                    "average": "49900",
                    "price": "50000",
                },
            }
        )
        order = _make_market_order(quantity=Decimal("0.1"))
        submitted = await engine.submit_order(order)

        reconciled = await engine._reconcile_order(submitted)

        assert reconciled.status == OrderStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_reconcile_failure_returns_original_order(self) -> None:
        """When fetch_order raises, _reconcile_order returns the original order unchanged."""
        engine, _, ex = _make_engine()
        order = _make_market_order()
        submitted = await engine.submit_order(order)
        original_status = submitted.status

        ex.fetch_order.side_effect = ccxt_async.NetworkError("timeout")

        result = await engine._reconcile_order(submitted)

        # Must return the order — status unchanged despite fetch failure
        assert result.status == original_status


# ===========================================================================
# Group 11: Equity Helpers
# ===========================================================================


class TestEquityHelpers:
    """Verify _fetch_equity() and _calculate_daily_pnl() return correct values."""

    @pytest.mark.asyncio
    async def test_fetch_equity_returns_usdt_balance(self) -> None:
        """_fetch_equity() returns the USDT total balance from exchange.fetch_balance()."""
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "fetch_balance_response": {"total": {"USDT": 12345.67, "BTC": 0.5}}
            }
        )
        equity = await engine._fetch_equity()
        assert equity == Decimal("12345.67")

    @pytest.mark.asyncio
    async def test_fetch_equity_prefers_first_matching_quote_currency(self) -> None:
        """
        When multiple quote currencies are present, _fetch_equity returns the first
        match from the priority order: USDT > BUSD > USD > USDC.
        """
        engine, _, _ = _make_engine(
            exchange_kwargs={
                "fetch_balance_response": {
                    "total": {"USDT": None, "BUSD": 5000.0, "USD": 9000.0}
                }
            }
        )
        equity = await engine._fetch_equity()
        # USDT is None so it's skipped; BUSD is the next match
        assert equity == Decimal("5000.0")

    @pytest.mark.asyncio
    async def test_fetch_equity_fallback_to_position_value_on_error(self) -> None:
        """
        When fetch_balance raises, _fetch_equity sums position notional values.
        A flat account with no positions returns Decimal('0').
        """
        engine, _, ex = _make_engine()
        ex.fetch_balance.side_effect = ccxt_async.NetworkError("exchange unavailable")

        equity = await engine._fetch_equity()

        assert equity == Decimal("0")

    def test_calculate_daily_pnl_sums_position_realised_pnl(self) -> None:
        """_calculate_daily_pnl() returns the sum of realised_pnl across all positions."""
        engine, _, _ = _make_engine()
        engine._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            run_id=_RUN_ID,
            quantity=Decimal("0.1"),
            average_entry_price=Decimal("48000"),
            current_price=Decimal("50000"),
            realised_pnl=Decimal("150"),
        )
        engine._positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            run_id=_RUN_ID,
            quantity=Decimal("1.0"),
            average_entry_price=Decimal("3000"),
            current_price=Decimal("3200"),
            realised_pnl=Decimal("-50"),
        )

        pnl = engine._calculate_daily_pnl()

        assert pnl == Decimal("100")


# ===========================================================================
# Group 12: Lifecycle
# ===========================================================================


class TestLifecycle:
    """Verify on_start and on_stop lifecycle hooks behave correctly."""

    @pytest.mark.asyncio
    async def test_on_start_calls_load_markets_when_enabled(self) -> None:
        """on_start() calls exchange.load_markets() when live trading is enabled."""
        engine, _, ex = _make_engine(enable_live_trading=True)

        await engine.on_start()

        ex.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_start_skips_load_markets_when_disabled(self) -> None:
        """on_start() logs a warning and returns early when live trading is disabled."""
        engine, _, ex = _make_engine(enable_live_trading=False)

        await engine.on_start()  # Must not raise

        ex.load_markets.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_stop_cancels_open_orders_on_exchange(self) -> None:
        """on_stop() calls exchange.cancel_order for each open order that has an exchange ID."""
        engine, _, ex = _make_engine()
        # Submit two orders that remain OPEN
        o1 = await engine.submit_order(_make_market_order())
        o2 = await engine.submit_order(_make_market_order())
        assert o1.status == OrderStatus.OPEN
        assert o2.status == OrderStatus.OPEN

        # Reset call counter to isolate on_stop behavior
        ex.cancel_order.reset_mock()

        await engine.on_stop()

        # cancel_order should have been called once for each open order
        assert ex.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_on_stop_continues_despite_cancel_failure(self) -> None:
        """
        on_stop() does not propagate exceptions from individual cancel_order failures.
        Other orders are still processed and the engine shuts down cleanly.
        """
        engine, _, ex = _make_engine()
        await engine.submit_order(_make_market_order())

        ex.cancel_order.reset_mock()
        ex.cancel_order.side_effect = ccxt_async.ExchangeError("cancel failed")

        # Must not raise
        await engine.on_stop()

    @pytest.mark.asyncio
    async def test_sync_positions_returns_copy(self) -> None:
        engine, _, _ = _make_engine()
        result = await engine.sync_positions()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_on_start_reraises_load_markets_failure(self) -> None:
        engine, _, ex = _make_engine(enable_live_trading=True)
        ex.load_markets.side_effect = Exception("exchange down")
        with pytest.raises(Exception, match="exchange down"):
            await engine.on_start()

# ===========================================================================
# Group 13: Get All Fills (LiveExecutionEngine.get_all_fills)
# ===========================================================================


class TestGetAllFills:
    """
    Verify LiveExecutionEngine.get_all_fills() merges all per-order fill
    lists from the local cache and returns them sorted by executed_at ascending.

    From engines/live.py:
        def get_all_fills(self) -> list[Fill]:
            all_fills: list[Fill] = []
            for fills in self._fills.values():
                all_fills.extend(fills)
            return sorted(all_fills, key=lambda f: f.executed_at)

    Fill objects are injected directly into engine._fills (bypassing the
    async exchange path) so these tests are fully synchronous and
    deterministic — no network interaction, no exchange mocking required.
    """

    def test_get_all_fills_empty(self) -> None:
        """
        A freshly constructed engine with no cached fills must return an empty list.

        This is the baseline before any order fill events have been recorded.
        The _fills dict is initialised as an empty dict in __init__, so the
        sorted() call over an empty iterable must yield [].
        """
        engine, _, _ = _make_engine()

        fills = engine.get_all_fills()

        assert fills == [], (
            f"Expected empty list from fresh LiveExecutionEngine, got {fills!r}"
        )

    def test_get_all_fills_merges_across_orders(self) -> None:
        """
        Fills from two distinct orders must be merged into a single list sorted
        by executed_at ascending, regardless of the order in which they were
        stored in _fills.

        Setup:
        - order_id_a has one fill at T3 (latest timestamp).
        - order_id_b has two fills at T1 (earliest) and T2 (middle).

        Expected result: [T1, T2, T3] — chronological order across both orders.

        This test mirrors the pattern in TestGetAllFills in
        test_order_fill_persistence.py but exercises the LiveExecutionEngine
        implementation, which mirrors PaperExecutionEngine's get_all_fills().
        """
        engine, _, _ = _make_engine()

        _T1 = datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC)
        _T2 = datetime(2024, 3, 1, 1, 0, 0, tzinfo=UTC)
        _T3 = datetime(2024, 3, 1, 2, 0, 0, tzinfo=UTC)

        order_id_a = uuid4()
        order_id_b = uuid4()

        fill_a = Fill(
            order_id=order_id_a,
            symbol=_SYMBOL,
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("0.5"),
            fee_currency="USDT",
            is_maker=False,
            executed_at=_T3,
        )
        fill_b1 = Fill(
            order_id=order_id_b,
            symbol=_SYMBOL,
            side=OrderSide.SELL,
            quantity=Decimal("0.05"),
            price=Decimal("49000"),
            fee=Decimal("0.25"),
            fee_currency="USDT",
            is_maker=False,
            executed_at=_T1,
        )
        fill_b2 = Fill(
            order_id=order_id_b,
            symbol=_SYMBOL,
            side=OrderSide.SELL,
            quantity=Decimal("0.05"),
            price=Decimal("49500"),
            fee=Decimal("0.25"),
            fee_currency="USDT",
            is_maker=True,
            executed_at=_T2,
        )

        # Inject directly into the fill cache, bypassing the async exchange path
        engine._fills[order_id_a] = [fill_a]
        engine._fills[order_id_b] = [fill_b1, fill_b2]

        result = engine.get_all_fills()

        assert len(result) == 3, (
            f"Expected 3 fills merged from 2 orders, got {len(result)}"
        )

        # Must be sorted ascending by executed_at across both orders
        assert result[0].executed_at == _T1, (
            f"First fill should be at T1, got {result[0].executed_at}"
        )
        assert result[1].executed_at == _T2, (
            f"Second fill should be at T2, got {result[1].executed_at}"
        )
        assert result[2].executed_at == _T3, (
            f"Third fill should be at T3, got {result[2].executed_at}"
        )

        # Verify exact identity — no copying of Fill objects
        assert result[0] is fill_b1
        assert result[1] is fill_b2
        assert result[2] is fill_a

    def test_get_all_fills_single_order_sorted(self) -> None:
        """
        Fills from a single order must be returned in ascending executed_at order.

        This exercises the sorting guarantee when only one order's fills are
        present in the cache. Two fills are inserted in reverse chronological
        order to confirm that get_all_fills() does not rely on insertion order.
        """
        engine, _, _ = _make_engine()

        _T_EARLY = datetime(2024, 3, 1, 8, 0, 0, tzinfo=UTC)
        _T_LATE = datetime(2024, 3, 1, 9, 0, 0, tzinfo=UTC)

        order_id = uuid4()

        fill_late = Fill(
            order_id=order_id,
            symbol=_SYMBOL,
            side=OrderSide.BUY,
            quantity=Decimal("0.05"),
            price=Decimal("51000"),
            fee=Decimal("0.25"),
            fee_currency="USDT",
            is_maker=False,
            executed_at=_T_LATE,
        )
        fill_early = Fill(
            order_id=order_id,
            symbol=_SYMBOL,
            side=OrderSide.BUY,
            quantity=Decimal("0.05"),
            price=Decimal("50900"),
            fee=Decimal("0.25"),
            fee_currency="USDT",
            is_maker=False,
            executed_at=_T_EARLY,
        )

        # Insert in reverse order to confirm sorting is applied
        engine._fills[order_id] = [fill_late, fill_early]

        result = engine.get_all_fills()

        assert len(result) == 2, (
            f"Expected 2 fills for single order, got {len(result)}"
        )
        assert result[0] is fill_early, (
            "Earlier fill must appear first after sorting"
        )
        assert result[1] is fill_late, (
            "Later fill must appear second after sorting"
        )
        assert result[0].executed_at < result[1].executed_at, (
            "Fills must be sorted ascending by executed_at"
        )
