"""
packages/trading/models.py
--------------------------
Pydantic domain models for the trading layer.
These are in-memory value objects — not ORM models.
Database persistence models live in apps/api/db/models.py (to be created).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from common.models import OHLCVBar as OHLCVBar  # noqa: F401
from common.types import OrderSide, OrderStatus, OrderType, SignalDirection, TimeFrame

__all__ = [
    "Signal",
    "Order",
    "Fill",
    "Position",
    "TradeResult",
    "SkippedTrade",
    "OHLCVBar",  # re-exported from common.models
    "RiskCheckResult",
]


class Signal(BaseModel):
    """
    Trading signal emitted by a strategy on each bar.

    ``target_position`` is the desired notional position size in quote currency.
    A value of 0.0 with direction=SELL means "close the position entirely".
    ``confidence`` is a float in [0, 1] used by risk sizing models.
    """

    model_config = {"frozen": True}

    strategy_id: str = Field(description="Unique identifier of the originating strategy")
    symbol: str = Field(description="Trading pair the signal applies to")
    direction: SignalDirection = Field(description="BUY / SELL / HOLD")
    target_position: Decimal = Field(
        ge=Decimal(0),
        description="Target notional size in quote currency (0 = flat)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strategy confidence in this signal, used for fractional sizing",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp when the signal was generated",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary strategy-specific context (indicator values, etc.)",
    )


class Order(BaseModel):
    """
    Represents a single order in the order state machine.

    The state machine transitions are enforced by the ExecutionEngine.
    ``client_order_id`` is our idempotency key; ``exchange_order_id`` is
    set after the exchange acknowledges the order.

    Intentionally mutable (no ``frozen=True``) so the ExecutionEngine can
    update ``status``, ``filled_quantity``, and ``updated_at`` in place via
    ``model_copy``. ``validate_assignment=True`` ensures field-level
    validators still run on every mutation.
    """

    model_config = {"validate_assignment": True}

    order_id: UUID = Field(default_factory=uuid4, description="Internal order UUID")
    client_order_id: str = Field(
        description="Idempotency key sent to the exchange. Format: <run_id>-<uuid4_hex[:12]>"
    )
    run_id: str = Field(description="Run that generated this order")
    symbol: str = Field(description="Trading pair")
    side: OrderSide = Field(description="BUY or SELL")
    order_type: OrderType = Field(description="MARKET or LIMIT")
    quantity: Decimal = Field(gt=Decimal(0), description="Order size in base asset")
    price: Decimal | None = Field(
        default=None,
        description="Limit price. None for MARKET orders.",
    )
    status: OrderStatus = Field(
        default=OrderStatus.NEW,
        description="Current state in the order state machine",
    )
    filled_quantity: Decimal = Field(
        default=Decimal(0),
        ge=Decimal(0),
        description="Quantity filled so far",
    )
    average_fill_price: Decimal | None = Field(
        default=None,
        description="Volume-weighted average fill price. None if unfilled.",
    )
    exchange_order_id: str | None = Field(
        default=None,
        description="Exchange-assigned order ID. None until acknowledged.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @model_validator(mode="after")
    def validate_order_constraints(self) -> Order:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT orders require a price")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("MARKET orders must not specify a price")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity")
        return self


class Fill(BaseModel):
    """
    A single execution fill event, potentially partial.

    Multiple Fill records may exist for one Order (partial fills).
    """

    model_config = {"frozen": True}

    fill_id: UUID = Field(default_factory=uuid4)
    order_id: UUID = Field(description="Parent order UUID")
    symbol: str
    side: OrderSide
    quantity: Decimal = Field(gt=Decimal(0), description="Filled quantity in base asset")
    price: Decimal = Field(gt=Decimal(0), description="Execution price")
    fee: Decimal = Field(ge=Decimal(0), description="Fee paid in quote asset")
    fee_currency: str = Field(description="Currency in which fee was paid, e.g. 'USDT'")
    is_maker: bool = Field(
        default=False,
        description="True if this fill earned maker fee (resting limit order)",
    )
    executed_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class Position(BaseModel):
    """
    Current open position for a symbol within a run.

    ``quantity`` is in base asset. Positive = long, negative would be
    short (not used in MVP spot-only mode — quantity >= 0 always).
    ``unrealised_pnl`` is computed on each bar update; realised_pnl
    accumulates as fills close the position.
    """

    symbol: str
    run_id: str
    quantity: Decimal = Field(ge=Decimal(0), description="Open quantity in base asset")
    average_entry_price: Decimal = Field(
        ge=Decimal(0),
        description="Volume-weighted average entry price",
    )
    current_price: Decimal = Field(ge=Decimal(0), description="Latest market price")
    realised_pnl: Decimal = Field(default=Decimal(0), description="Cumulative realised PnL in quote")
    unrealised_pnl: Decimal = Field(default=Decimal(0), description="Current unrealised PnL in quote")
    total_fees_paid: Decimal = Field(default=Decimal(0), ge=Decimal(0))
    opened_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def notional_value(self) -> Decimal:
        """Current market value of the position in quote currency."""
        return self.quantity * self.current_price

    @property
    def is_flat(self) -> bool:
        """True when the position has been fully closed."""
        return self.quantity == Decimal(0)


class TradeResult(BaseModel):
    """
    Completed round-trip trade record (entry + exit fills aggregated).
    Written to the database when a position is fully closed.
    """

    model_config = {"frozen": True}

    trade_id: UUID = Field(default_factory=uuid4)
    run_id: str
    symbol: str
    side: OrderSide = Field(description="Side of the opening fill")
    entry_price: Decimal = Field(gt=Decimal(0))
    exit_price: Decimal = Field(gt=Decimal(0))
    quantity: Decimal = Field(gt=Decimal(0))
    realised_pnl: Decimal = Field(description="Net PnL after fees in quote currency")
    total_fees: Decimal = Field(ge=Decimal(0))
    entry_at: datetime
    exit_at: datetime
    strategy_id: str = Field(description="Strategy that generated the opening signal")

    # Sprint 32: Adaptive learning fields
    mae_pct: float | None = Field(
        default=None,
        description="Maximum Adverse Excursion as pct of entry price",
    )
    mfe_pct: float | None = Field(
        default=None,
        description="Maximum Favorable Excursion as pct of entry price",
    )
    exit_reason: str | None = Field(
        default=None,
        description="Exit trigger: take_profit|stop_loss|trailing_stop|signal_exit|regime_change|manual",
    )
    regime_at_entry: str | None = Field(
        default=None,
        description="Market regime at open: RISK_ON|NEUTRAL|RISK_OFF",
    )
    signal_context: dict[str, Any] | None = Field(
        default=None,
        description="Indicator snapshot at entry",
    )

    @property
    def return_pct(self) -> float:
        """Percentage return relative to entry cost."""
        cost = float(self.entry_price * self.quantity)
        if cost == 0:
            return 0.0
        return float(self.realised_pnl) / cost


class SkippedTrade(BaseModel):
    """Trade that was evaluated but not taken, logged for adaptive learning."""

    model_config = {"frozen": True}

    skip_id: UUID = Field(default_factory=uuid4)
    run_id: str
    symbol: str
    skip_reason: str
    regime_at_skip: str | None = Field(default=None)
    signal_context: dict[str, Any] | None = Field(default=None)
    hypothetical_entry_price: Decimal | None = Field(default=None)
    hypothetical_outcome_pct: float | None = Field(default=None)
    skipped_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class RiskCheckResult(BaseModel):
    """
    Result of a pre-trade risk check.

    ``approved`` False means the order must not be submitted.
    ``adjusted_quantity`` may be smaller than the requested quantity
    when the risk manager applies position-size caps.
    """

    model_config = {"frozen": True}

    approved: bool
    adjusted_quantity: Decimal = Field(ge=Decimal(0))
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
