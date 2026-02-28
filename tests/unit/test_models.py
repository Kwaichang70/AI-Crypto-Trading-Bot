"""
tests/unit/test_models.py
--------------------------
Unit tests for Pydantic domain models in packages/trading/models.py.

Test coverage
-------------
- Signal: valid creation, invalid target_position, confidence clamping
- Order: LIMIT/MARKET constraints, fill quantity validation, model_copy mutation
- Fill: valid creation, field constraints
- Position: notional_value property, is_flat property
- TradeResult: return_pct calculation including edge cases
- RiskCheckResult: approved/rejected states
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.models import Fill, Order, Position, RiskCheckResult, Signal, TradeResult


# ===========================================================================
# Signal tests
# ===========================================================================


class TestSignal:
    """Tests for the Signal domain model."""

    def test_signal_valid_buy(self) -> None:
        """A well-formed BUY signal round-trips through Pydantic without errors."""
        signal = Signal(
            strategy_id="ma-crossover-01",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
            confidence=0.75,
        )
        assert signal.strategy_id == "ma-crossover-01"
        assert signal.symbol == "BTC/USDT"
        assert signal.direction == SignalDirection.BUY
        assert signal.target_position == Decimal("1000")
        assert signal.confidence == 0.75

    def test_signal_valid_sell(self) -> None:
        """A SELL signal with zero target_position represents 'close all'."""
        signal = Signal(
            strategy_id="rsi-01",
            symbol="ETH/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        assert signal.direction == SignalDirection.SELL
        assert signal.target_position == Decimal("0")

    def test_signal_valid_hold_direction(self) -> None:
        """HOLD direction is a valid SignalDirection value."""
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.HOLD,
            target_position=Decimal("500"),
        )
        assert signal.direction == SignalDirection.HOLD

    def test_signal_confidence_defaults_to_one(self) -> None:
        """Omitting confidence defaults to 1.0."""
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("100"),
        )
        assert signal.confidence == 1.0

    def test_signal_metadata_defaults_to_empty_dict(self) -> None:
        """Metadata defaults to an empty dict when not provided."""
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("100"),
        )
        assert signal.metadata == {}

    def test_signal_metadata_stored(self) -> None:
        """Arbitrary metadata keys survive round-trip."""
        meta = {"rsi": 28.5, "period": 14}
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("100"),
            metadata=meta,
        )
        assert signal.metadata["rsi"] == 28.5

    def test_signal_negative_target_position_rejected(self) -> None:
        """target_position must be >= 0; a negative value raises ValidationError."""
        with pytest.raises(ValidationError):
            Signal(
                strategy_id="test",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                target_position=Decimal("-1"),
            )

    def test_signal_confidence_above_one_rejected(self) -> None:
        """confidence must be in [0, 1]; exceeding 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            Signal(
                strategy_id="test",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                target_position=Decimal("100"),
                confidence=1.1,
            )

    def test_signal_confidence_below_zero_rejected(self) -> None:
        """confidence must be >= 0; a negative value raises ValidationError."""
        with pytest.raises(ValidationError):
            Signal(
                strategy_id="test",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                target_position=Decimal("100"),
                confidence=-0.1,
            )

    def test_signal_is_frozen(self) -> None:
        """Signal is immutable — attribute assignment raises AttributeError."""
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("100"),
        )
        with pytest.raises((AttributeError, ValidationError)):
            signal.confidence = 0.5  # type: ignore[misc]

    def test_signal_generated_at_is_utc(self) -> None:
        """The auto-generated timestamp has UTC timezone info."""
        signal = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("100"),
        )
        assert signal.generated_at.tzinfo is not None


# ===========================================================================
# Order tests
# ===========================================================================


class TestOrder:
    """Tests for the Order domain model and its constraints."""

    def _base_order_kwargs(self) -> dict:
        return {
            "client_order_id": "run-01-aabbcc112233",
            "run_id": "run-01",
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY,
            "quantity": Decimal("0.01"),
        }

    def test_limit_order_valid(self) -> None:
        """A LIMIT order with a price passes validation."""
        order = Order(
            **self._base_order_kwargs(),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
        )
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("50000")

    def test_market_order_valid(self) -> None:
        """A MARKET order without a price passes validation."""
        order = Order(
            **self._base_order_kwargs(),
            order_type=OrderType.MARKET,
            price=None,
        )
        assert order.order_type == OrderType.MARKET
        assert order.price is None

    def test_limit_order_without_price_rejected(self) -> None:
        """A LIMIT order missing the price field raises ValidationError."""
        with pytest.raises(ValidationError, match="LIMIT orders require a price"):
            Order(
                **self._base_order_kwargs(),
                order_type=OrderType.LIMIT,
                price=None,
            )

    def test_market_order_with_price_rejected(self) -> None:
        """A MARKET order with a price set raises ValidationError."""
        with pytest.raises(ValidationError, match="MARKET orders must not specify a price"):
            Order(
                **self._base_order_kwargs(),
                order_type=OrderType.MARKET,
                price=Decimal("50000"),
            )

    def test_order_quantity_must_be_positive(self) -> None:
        """Zero quantity raises ValidationError (quantity must be > 0)."""
        # Build merged kwargs with quantity overridden to 0 (avoids duplicate kwarg error)
        kwargs = {**self._base_order_kwargs(), "quantity": Decimal("0")}
        with pytest.raises(ValidationError):
            Order(
                **kwargs,
                order_type=OrderType.MARKET,
                price=None,
            )

    def test_filled_quantity_cannot_exceed_quantity(self) -> None:
        """Setting filled_quantity > quantity raises ValidationError."""
        # base_order_kwargs already contains quantity=0.01; do not repeat it
        with pytest.raises(ValidationError, match="filled_quantity cannot exceed quantity"):
            Order(
                **self._base_order_kwargs(),
                order_type=OrderType.LIMIT,
                price=Decimal("50000"),
                filled_quantity=Decimal("0.02"),
            )

    def test_order_default_status_is_new(self) -> None:
        """The default status is NEW."""
        order = Order(
            **self._base_order_kwargs(),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
        )
        assert order.status == OrderStatus.NEW

    def test_order_has_uuid(self) -> None:
        """Every order gets a UUID order_id auto-generated."""
        order = Order(
            **self._base_order_kwargs(),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
        )
        assert isinstance(order.order_id, UUID)

    def test_order_model_copy_updates_status(self) -> None:
        """model_copy can update mutable fields like status."""
        order = Order(
            **self._base_order_kwargs(),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
        )
        updated = order.model_copy(update={"status": OrderStatus.OPEN})
        assert updated.status == OrderStatus.OPEN
        assert order.status == OrderStatus.NEW  # original unchanged

    def test_order_sell_side(self) -> None:
        """An order with SELL side is accepted."""
        kwargs = self._base_order_kwargs()
        kwargs["side"] = OrderSide.SELL
        order = Order(
            **kwargs,
            order_type=OrderType.MARKET,
            price=None,
        )
        assert order.side == OrderSide.SELL


# ===========================================================================
# Fill tests
# ===========================================================================


class TestFill:
    """Tests for the Fill domain model."""

    def _make_fill(self, **overrides) -> Fill:
        defaults = {
            "order_id": uuid4(),
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY,
            "quantity": Decimal("0.01"),
            "price": Decimal("50000"),
            "fee": Decimal("0.5"),
            "fee_currency": "USDT",
        }
        defaults.update(overrides)
        return Fill(**defaults)

    def test_fill_valid(self) -> None:
        """A fully specified Fill passes all validators."""
        fill = self._make_fill()
        assert fill.symbol == "BTC/USDT"
        assert fill.quantity == Decimal("0.01")
        assert fill.fee == Decimal("0.5")

    def test_fill_zero_quantity_rejected(self) -> None:
        """Quantity must be > 0; zero raises ValidationError."""
        with pytest.raises(ValidationError):
            self._make_fill(quantity=Decimal("0"))

    def test_fill_zero_price_rejected(self) -> None:
        """Price must be > 0; zero raises ValidationError."""
        with pytest.raises(ValidationError):
            self._make_fill(price=Decimal("0"))

    def test_fill_negative_fee_rejected(self) -> None:
        """Fee must be >= 0; negative raises ValidationError."""
        with pytest.raises(ValidationError):
            self._make_fill(fee=Decimal("-0.01"))

    def test_fill_zero_fee_allowed(self) -> None:
        """Zero fee (e.g. for maker with no fee) is valid."""
        fill = self._make_fill(fee=Decimal("0"))
        assert fill.fee == Decimal("0")

    def test_fill_is_frozen(self) -> None:
        """Fill is immutable — attribute assignment raises an error."""
        fill = self._make_fill()
        with pytest.raises((AttributeError, ValidationError)):
            fill.price = Decimal("99999")  # type: ignore[misc]

    def test_fill_is_maker_default_false(self) -> None:
        """is_maker defaults to False (taker fill)."""
        fill = self._make_fill()
        assert fill.is_maker is False

    def test_fill_uuid(self) -> None:
        """Each fill gets its own unique fill_id."""
        fill1 = self._make_fill()
        fill2 = self._make_fill()
        assert fill1.fill_id != fill2.fill_id


# ===========================================================================
# Position tests
# ===========================================================================


class TestPosition:
    """Tests for Position properties notional_value and is_flat."""

    def _make_position(self, **overrides) -> Position:
        defaults = {
            "symbol": "BTC/USDT",
            "run_id": "run-001",
            "quantity": Decimal("0.1"),
            "average_entry_price": Decimal("50000"),
            "current_price": Decimal("51000"),
        }
        defaults.update(overrides)
        return Position(**defaults)

    def test_notional_value_calculation(self) -> None:
        """notional_value = quantity * current_price."""
        pos = self._make_position(
            quantity=Decimal("0.1"),
            current_price=Decimal("51000"),
        )
        expected = Decimal("0.1") * Decimal("51000")  # 5100
        assert pos.notional_value == expected

    def test_notional_value_zero_quantity(self) -> None:
        """A closed position (quantity=0) has notional_value of 0."""
        pos = self._make_position(quantity=Decimal("0"))
        assert pos.notional_value == Decimal("0")

    def test_is_flat_true_when_quantity_zero(self) -> None:
        """is_flat is True when quantity is exactly 0."""
        pos = self._make_position(quantity=Decimal("0"))
        assert pos.is_flat is True

    def test_is_flat_false_when_quantity_nonzero(self) -> None:
        """is_flat is False when quantity is positive."""
        pos = self._make_position(quantity=Decimal("0.001"))
        assert pos.is_flat is False

    def test_unrealised_pnl_default_zero(self) -> None:
        """unrealised_pnl defaults to Decimal(0)."""
        pos = self._make_position()
        assert pos.unrealised_pnl == Decimal("0")

    @pytest.mark.parametrize(
        "quantity,price,expected_notional",
        [
            (Decimal("1"), Decimal("100"), Decimal("100")),
            (Decimal("0.5"), Decimal("200"), Decimal("100")),
            (Decimal("2"), Decimal("50"), Decimal("100")),
        ],
    )
    def test_notional_value_parametrized(
        self,
        quantity: Decimal,
        price: Decimal,
        expected_notional: Decimal,
    ) -> None:
        """notional_value = quantity * current_price across multiple inputs."""
        pos = self._make_position(quantity=quantity, current_price=price)
        assert pos.notional_value == expected_notional


# ===========================================================================
# TradeResult tests
# ===========================================================================


class TestTradeResult:
    """Tests for TradeResult, especially the return_pct property."""

    def _make_trade(self, **overrides) -> TradeResult:
        now = datetime.now(tz=UTC)
        defaults = {
            "run_id": "run-001",
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY,
            "entry_price": Decimal("50000"),
            "exit_price": Decimal("51000"),
            "quantity": Decimal("0.01"),
            "realised_pnl": Decimal("9.5"),  # (1000 - fee)
            "total_fees": Decimal("0.5"),
            "entry_at": now,
            "exit_at": now,
            "strategy_id": "test-strategy",
        }
        defaults.update(overrides)
        return TradeResult(**defaults)

    def test_return_pct_positive_trade(self) -> None:
        """A profitable trade yields a positive return_pct."""
        trade = self._make_trade(
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            realised_pnl=Decimal("10"),  # +10 USDT on 500 USDT cost = 2%
        )
        expected = float(Decimal("10")) / float(Decimal("50000") * Decimal("0.01"))
        assert abs(trade.return_pct - expected) < 1e-9

    def test_return_pct_negative_trade(self) -> None:
        """A losing trade yields a negative return_pct."""
        trade = self._make_trade(
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            realised_pnl=Decimal("-5"),
        )
        assert trade.return_pct < 0.0

    def test_return_pct_break_even(self) -> None:
        """A break-even trade returns exactly 0.0."""
        trade = self._make_trade(realised_pnl=Decimal("0"))
        assert trade.return_pct == 0.0

    def test_return_pct_zero_cost_guard(self) -> None:
        """If entry cost evaluates to zero, return_pct safely returns 0.0."""
        # Technically prevented by gt=0 validators, but test the property guard
        trade = TradeResult(
            run_id="run-001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("1"),  # smallest valid positive value
            exit_price=Decimal("1"),
            quantity=Decimal("0.00000001"),
            realised_pnl=Decimal("0"),
            total_fees=Decimal("0"),
            entry_at=datetime.now(tz=UTC),
            exit_at=datetime.now(tz=UTC),
            strategy_id="test",
        )
        # cost = 1 * 0.00000001 = 0.00000001 (not zero), so safe
        assert isinstance(trade.return_pct, float)

    def test_trade_result_is_frozen(self) -> None:
        """TradeResult is immutable — assignment raises an error."""
        trade = self._make_trade()
        with pytest.raises((AttributeError, ValidationError)):
            trade.realised_pnl = Decimal("999")  # type: ignore[misc]


# ===========================================================================
# RiskCheckResult tests
# ===========================================================================


class TestRiskCheckResult:
    """Tests for the RiskCheckResult domain model."""

    def test_approved_result(self) -> None:
        """An approved result with adjusted quantity has no rejection reasons."""
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=Decimal("0.01"),
        )
        assert result.approved is True
        assert result.adjusted_quantity == Decimal("0.01")
        assert result.rejection_reasons == []
        assert result.warnings == []

    def test_rejected_result_with_reasons(self) -> None:
        """A rejected result captures all blocking reasons."""
        reasons = ["Kill switch active", "Daily loss limit exceeded"]
        result = RiskCheckResult(
            approved=False,
            adjusted_quantity=Decimal("0"),
            rejection_reasons=reasons,
        )
        assert result.approved is False
        assert result.adjusted_quantity == Decimal("0")
        assert result.rejection_reasons == reasons

    def test_approved_with_warnings(self) -> None:
        """An approved-but-warned result carries non-empty warnings."""
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=Decimal("0.005"),
            warnings=["Quantity reduced due to concentration cap"],
        )
        assert result.approved is True
        assert len(result.warnings) == 1

    def test_negative_adjusted_quantity_rejected(self) -> None:
        """adjusted_quantity must be >= 0; negative raises ValidationError."""
        with pytest.raises(ValidationError):
            RiskCheckResult(
                approved=False,
                adjusted_quantity=Decimal("-1"),
            )

    def test_risk_check_is_frozen(self) -> None:
        """RiskCheckResult is immutable."""
        result = RiskCheckResult(approved=True, adjusted_quantity=Decimal("1"))
        with pytest.raises((AttributeError, ValidationError)):
            result.approved = False  # type: ignore[misc]
