"""
apps/api/schemas.py
--------------------
Pydantic request/response schemas for the AI Crypto Trading Bot REST API.

Design decisions
----------------
- All monetary values are serialised as ``str`` to preserve Decimal precision
  across JSON boundaries (JavaScript cannot represent IEEE-754 64-bit floats
  with full Decimal fidelity).
- CamelCase field names on JSON output via ``alias_generator=to_camel`` so
  the Next.js dashboard consumes idiomatic JavaScript objects.
- ``model_config`` uses ``populate_by_name=True`` so internal Python code can
  construct models using snake_case names while the API emits camelCase.
- Pagination is standardised with ``PaginationParams`` (offset/limit query
  params) and a ``total`` count on every list response.
- Error responses carry a ``code`` field alongside ``detail`` to enable
  programmatic error handling in the frontend without string-matching.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic.alias_generators import to_camel

from common.types import OrderSide, OrderStatus, OrderType, RunMode, TimeFrame

__all__ = [
    # Pagination
    "PaginationParams",
    "PaginatedResponse",
    # Runs
    "RunCreateRequest",
    "RunResponse",
    "RunListResponse",
    # Orders
    "OrderResponse",
    "OrderListResponse",
    # Fills
    "FillResponse",
    "FillListResponse",
    # Trades
    "TradeResponse",
    "TradeListResponse",
    # Portfolio
    "PortfolioResponse",
    "EquityPointResponse",
    "EquityCurveResponse",
    "PositionResponse",
    "PositionListResponse",
    # Strategies
    "StrategyInfoResponse",
    "StrategyListResponse",
    # Error
    "ErrorResponse",
]

# ---------------------------------------------------------------------------
# Shared model configuration
# All API-facing schemas use camelCase serialisation.
# ---------------------------------------------------------------------------

_API_MODEL_CONFIG = ConfigDict(
    alias_generator=to_camel,
    populate_by_name=True,   # Allow construction with snake_case in Python code
    from_attributes=True,    # Enable ORM-mode (from SQLAlchemy ORM objects)
    use_enum_values=True,    # Serialise StrEnum members as their string value
    # Strict decimal string handling — monetary fields use explicit serialisers
)


# ---------------------------------------------------------------------------
# Generic Paginated wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PaginationParams(BaseModel):
    """
    Query parameter bag for paginated list endpoints.

    Parameters
    ----------
    offset:
        Number of records to skip. Default 0.
    limit:
        Maximum number of records to return. Capped at 500 to guard against
        runaway queries. Default 50.
    """

    model_config = ConfigDict(populate_by_name=True)

    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=50, ge=1, le=500, description="Max records to return")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic wrapper for paginated list responses.

    Parameters
    ----------
    total:
        Total number of records matching the query (before pagination).
    offset:
        The offset applied to this page.
    limit:
        The limit applied to this page.
    items:
        The records on this page.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    total: int = Field(description="Total record count across all pages")
    offset: int = Field(description="Offset used for this page")
    limit: int = Field(description="Limit used for this page")
    items: list[T] = Field(description="Records on this page")


# ---------------------------------------------------------------------------
# Run schemas
# ---------------------------------------------------------------------------

class RunCreateRequest(BaseModel):
    """
    Request body for POST /api/v1/runs — start a new trading run.

    Strategy parameters are validated against the named strategy's
    ``parameter_schema()`` at the endpoint level, not here. This schema only
    enforces structural correctness.

    Parameters
    ----------
    strategy_name:
        One of: "ma_crossover", "rsi_mean_reversion", "breakout".
    strategy_params:
        Key-value parameters forwarded to the strategy constructor.
        Must conform to the strategy's ``parameter_schema()``.
    symbols:
        List of CCXT-format trading pairs, e.g. ["BTC/USDT", "ETH/USDT"].
        At least one symbol required.
    timeframe:
        OHLCV candle timeframe. Must be a valid ``TimeFrame`` value.
    mode:
        Run mode: "backtest", "paper", or "live".
    initial_capital:
        Starting capital in quote currency. Accepted as a string to preserve
        precision; internally converted to Decimal for validation.
    backtest_start:
        ISO-8601 datetime string for backtest start (required when mode=backtest).
    backtest_end:
        ISO-8601 datetime string for backtest end (required when mode=backtest).
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        use_enum_values=True,
    )

    strategy_name: str = Field(
        description="Strategy identifier: ma_crossover | rsi_mean_reversion | breakout"
    )
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameter overrides",
    )
    symbols: list[str] = Field(
        min_length=1,
        description="CCXT trading pairs, e.g. ['BTC/USDT']",
    )
    timeframe: TimeFrame = Field(
        description="Candle timeframe: 1m | 5m | 15m | 1h | 4h | 1d | etc.",
    )
    mode: RunMode = Field(
        description="Execution mode: backtest | paper | live",
    )
    initial_capital: str = Field(
        description="Starting capital as a decimal string, e.g. '10000.00'",
    )
    backtest_start: datetime | None = Field(
        default=None,
        description="Backtest start timestamp (required for mode=backtest)",
    )
    backtest_end: datetime | None = Field(
        default=None,
        description="Backtest end timestamp (required for mode=backtest)",
    )

    @field_validator("initial_capital")
    @classmethod
    def validate_initial_capital(cls, v: str) -> str:
        """Ensure initial_capital is a positive decimal string."""
        try:
            amount = Decimal(v)
        except Exception as exc:
            raise ValueError(
                f"initial_capital must be a valid decimal string, got: {v!r}"
            ) from exc
        if amount <= Decimal("0"):
            raise ValueError("initial_capital must be greater than zero")
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Ensure symbols follow CCXT format BASE/QUOTE."""
        for symbol in v:
            if "/" not in symbol:
                raise ValueError(
                    f"Symbol {symbol!r} must follow CCXT format 'BASE/QUOTE' "
                    f"(e.g. 'BTC/USDT')"
                )
        return v


class RunResponse(BaseModel):
    """
    Response model for a single trading run.

    Monetary fields (``config`` may embed monetary values) are not extracted
    into top-level typed fields — the config JSONB blob is returned as-is
    since its shape varies by strategy.
    """

    model_config = _API_MODEL_CONFIG

    id: uuid.UUID = Field(description="Unique run identifier")
    run_mode: str = Field(description="Execution mode: backtest | paper | live")
    status: str = Field(description="Run state: running | stopped | error")
    config: dict[str, Any] = Field(description="Immutable strategy config snapshot")
    started_at: datetime = Field(description="UTC timestamp when the run started")
    stopped_at: datetime | None = Field(description="UTC timestamp when the run stopped")
    created_at: datetime = Field(description="Row creation timestamp")
    updated_at: datetime = Field(description="Row last-update timestamp")


class RunListResponse(PaginatedResponse[RunResponse]):
    """Paginated list of trading runs."""
    pass


# ---------------------------------------------------------------------------
# Order schemas
# ---------------------------------------------------------------------------

class OrderResponse(BaseModel):
    """
    Response model for a single order.

    ``quantity``, ``filled_quantity``, ``price``, and ``average_fill_price``
    are returned as strings to preserve Decimal precision.
    """

    model_config = _API_MODEL_CONFIG

    id: uuid.UUID = Field(description="Internal order UUID")
    client_order_id: str = Field(description="Idempotency key")
    run_id: uuid.UUID = Field(description="Parent run UUID")
    symbol: str = Field(description="Trading pair, e.g. BTC/USDT")
    side: str = Field(description="buy | sell")
    order_type: str = Field(description="market | limit | stop_limit | stop_market")
    quantity: str = Field(description="Requested order size (base asset)")
    price: str | None = Field(description="Limit price. Null for market orders")
    status: str = Field(description="Order state machine status")
    filled_quantity: str = Field(description="Cumulative quantity filled")
    average_fill_price: str | None = Field(description="VWAP fill price. Null if unfilled")
    exchange_order_id: str | None = Field(description="Exchange-assigned order ID")
    created_at: datetime = Field(description="Row creation timestamp")
    updated_at: datetime = Field(description="Row last-update timestamp")

    @field_serializer("quantity", "filled_quantity")
    def serialise_decimal(self, v: Decimal | str) -> str:
        return str(v)

    @field_serializer("price", "average_fill_price")
    def serialise_optional_decimal(self, v: Decimal | str | None) -> str | None:
        return str(v) if v is not None else None


class OrderListResponse(PaginatedResponse[OrderResponse]):
    """Paginated list of orders for a run."""
    pass


# ---------------------------------------------------------------------------
# Fill schemas
# ---------------------------------------------------------------------------

class FillResponse(BaseModel):
    """
    Response model for a single execution fill.

    ``quantity``, ``price``, and ``fee`` are strings for precision safety.
    """

    model_config = _API_MODEL_CONFIG

    id: uuid.UUID = Field(description="Fill UUID")
    order_id: uuid.UUID = Field(description="Parent order UUID")
    symbol: str = Field(description="Trading pair")
    side: str = Field(description="buy | sell")
    quantity: str = Field(description="Filled quantity (base asset)")
    price: str = Field(description="Execution price")
    fee: str = Field(description="Fee paid in quote asset")
    fee_currency: str = Field(description="Currency the fee was paid in")
    is_maker: bool = Field(description="True if this fill earned maker fee rate")
    executed_at: datetime = Field(description="UTC timestamp of fill execution")

    @field_serializer("quantity", "price", "fee")
    def serialise_decimal(self, v: Decimal | str) -> str:
        return str(v)


class FillListResponse(PaginatedResponse[FillResponse]):
    """Paginated list of fills for a run."""
    pass


# ---------------------------------------------------------------------------
# Trade schemas
# ---------------------------------------------------------------------------

class TradeResponse(BaseModel):
    """
    Response model for a completed round-trip trade.

    All monetary fields are returned as strings.
    """

    model_config = _API_MODEL_CONFIG

    id: uuid.UUID = Field(description="Trade UUID")
    run_id: uuid.UUID = Field(description="Parent run UUID")
    symbol: str = Field(description="Trading pair")
    side: str = Field(description="Side of the opening fill: buy | sell")
    entry_price: str = Field(description="VWAP entry price")
    exit_price: str = Field(description="VWAP exit price")
    quantity: str = Field(description="Total traded quantity (base asset)")
    realised_pnl: str = Field(description="Net realised PnL after fees (quote)")
    total_fees: str = Field(description="Total fees paid across all fills (quote)")
    entry_at: datetime = Field(description="UTC timestamp of first entry fill")
    exit_at: datetime = Field(description="UTC timestamp of final exit fill")
    strategy_id: str = Field(description="Strategy that generated the opening signal")

    @field_serializer(
        "entry_price", "exit_price", "quantity",
        "realised_pnl", "total_fees",
    )
    def serialise_decimal(self, v: Decimal | str) -> str:
        return str(v)


class TradeListResponse(PaginatedResponse[TradeResponse]):
    """Paginated list of completed trades for a run."""
    pass


# ---------------------------------------------------------------------------
# Portfolio schemas
# ---------------------------------------------------------------------------

class PortfolioResponse(BaseModel):
    """
    Portfolio summary for a trading run.

    Provides a snapshot of the complete financial state: equity, cash,
    PnL metrics, drawdown, and win/loss statistics. All monetary values
    are strings.

    Parameters
    ----------
    run_id:
        The run this summary belongs to.
    initial_cash:
        Starting capital.
    current_cash:
        Current uninvested cash balance.
    current_equity:
        Total portfolio value (cash + open position value).
    peak_equity:
        Highest equity recorded during the run.
    total_return_pct:
        Return relative to initial capital as a decimal fraction.
    total_realised_pnl:
        Net realised PnL from all closed trades.
    total_fees_paid:
        All fees paid during the run.
    daily_pnl:
        Realised PnL for the current trading day.
    drawdown_pct:
        Current drawdown from peak equity (0.0 to 1.0).
    max_drawdown_pct:
        Maximum drawdown recorded during the run.
    total_trades:
        Number of completed round-trips.
    winning_trades:
        Trades with positive realised PnL.
    losing_trades:
        Trades with negative realised PnL.
    win_rate:
        Fraction of winning trades.
    open_positions:
        Count of symbols with non-zero position.
    equity_curve_length:
        Number of data points in the stored equity curve.
    """

    model_config = _API_MODEL_CONFIG

    run_id: str = Field(description="Run identifier")
    initial_cash: str = Field(description="Starting capital (quote currency)")
    current_cash: str = Field(description="Current cash balance (quote currency)")
    current_equity: str = Field(description="Total portfolio equity (quote currency)")
    peak_equity: str = Field(description="Highest equity recorded")
    total_return_pct: float = Field(description="Total return as decimal fraction")
    total_realised_pnl: str = Field(description="Cumulative realised PnL")
    total_fees_paid: str = Field(description="Cumulative fees paid")
    daily_pnl: str = Field(description="Realised PnL today")
    drawdown_pct: float = Field(description="Current drawdown from peak (0.0 to 1.0)")
    max_drawdown_pct: float = Field(description="Maximum drawdown recorded")
    total_trades: int = Field(description="Completed round-trip trade count")
    winning_trades: int = Field(description="Winning trade count")
    losing_trades: int = Field(description="Losing trade count")
    win_rate: float = Field(description="Fraction of winning trades")
    open_positions: int = Field(description="Number of open positions")
    equity_curve_length: int = Field(description="Equity curve data point count")


class EquityPointResponse(BaseModel):
    """
    A single data point on the equity curve.

    Parameters
    ----------
    timestamp:
        UTC datetime for the bar or fill event that produced this snapshot.
    equity:
        Portfolio equity at this point in time (string for precision).
    cash:
        Cash balance at this snapshot.
    unrealised_pnl:
        Unrealised PnL across all open positions.
    realised_pnl:
        Cumulative realised PnL to this point.
    drawdown_pct:
        Drawdown fraction at this point.
    bar_index:
        Zero-based bar number (ordering key in backtest).
    """

    model_config = _API_MODEL_CONFIG

    timestamp: datetime = Field(description="UTC bar timestamp")
    equity: str = Field(description="Portfolio equity at this point")
    cash: str = Field(description="Cash balance at this point")
    unrealised_pnl: str = Field(description="Unrealised PnL at this point")
    realised_pnl: str = Field(description="Cumulative realised PnL at this point")
    drawdown_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Drawdown fraction at this point (0.0 to 1.0)",
    )
    bar_index: int = Field(description="Zero-based bar number")

    @field_serializer("equity", "cash", "unrealised_pnl", "realised_pnl")
    def serialise_decimal(self, v: Decimal | str) -> str:
        return str(v)


class EquityCurveResponse(BaseModel):
    """
    Full equity curve for a run with metadata.

    Parameters
    ----------
    run_id:
        The run this curve belongs to.
    total_points:
        Total number of data points (before any pagination).
    points:
        The equity curve data points (ordered by bar_index ascending).
    """

    model_config = _API_MODEL_CONFIG

    run_id: uuid.UUID = Field(description="Parent run UUID")
    total_points: int = Field(description="Total equity curve data points")
    points: list[EquityPointResponse] = Field(description="Equity curve data points")


class PositionResponse(BaseModel):
    """
    Current open position for a symbol within a run.

    All monetary values are strings. Only non-flat positions are returned
    by default. An empty position list indicates no open positions.
    """

    model_config = _API_MODEL_CONFIG

    symbol: str = Field(description="Trading pair")
    run_id: str = Field(description="Parent run identifier")
    quantity: str = Field(description="Open quantity (base asset)")
    average_entry_price: str = Field(description="VWAP entry price")
    current_price: str = Field(description="Latest market price")
    realised_pnl: str = Field(description="Realised PnL from partial closes")
    unrealised_pnl: str = Field(description="Current unrealised PnL")
    total_fees_paid: str = Field(description="Total fees paid on this position")
    notional_value: str = Field(description="Current market value (quantity * price)")
    opened_at: datetime = Field(description="Timestamp when position was opened")
    updated_at: datetime = Field(description="Timestamp of last position update")

    @field_serializer(
        "quantity", "average_entry_price", "current_price",
        "realised_pnl", "unrealised_pnl", "total_fees_paid", "notional_value",
    )
    def serialise_decimal(self, v: Decimal | str) -> str:
        return str(v)


class PositionListResponse(BaseModel):
    """
    List of current open positions for a run.

    Parameters
    ----------
    run_id:
        The run these positions belong to.
    positions:
        All non-flat open positions.
    count:
        Number of open positions.
    """

    model_config = _API_MODEL_CONFIG

    run_id: uuid.UUID = Field(description="Parent run UUID")
    positions: list[PositionResponse] = Field(description="Open positions")
    count: int = Field(description="Number of open positions")


# ---------------------------------------------------------------------------
# Strategy schemas
# ---------------------------------------------------------------------------

class StrategyInfoResponse(BaseModel):
    """
    Metadata and parameter schema for a single available strategy.

    Parameters
    ----------
    name:
        Canonical strategy identifier used in ``RunCreateRequest.strategy_name``.
    display_name:
        Human-readable name from ``StrategyMetadata.name``.
    version:
        Semantic version of the strategy implementation.
    description:
        Human-readable description of the strategy's logic.
    tags:
        Classification tags (e.g. ["trend-following", "sma"]).
    parameter_schema:
        JSON Schema object describing accepted parameters. Use this to
        build dynamic configuration forms in the frontend.
    """

    model_config = _API_MODEL_CONFIG

    name: str = Field(description="Strategy identifier used in RunCreateRequest")
    display_name: str = Field(description="Human-readable strategy name")
    version: str = Field(description="Semantic version of the strategy")
    description: str = Field(description="What this strategy does")
    tags: list[str] = Field(description="Classification tags")
    parameter_schema: dict[str, Any] = Field(
        description="JSON Schema for strategy parameters"
    )


class StrategyListResponse(BaseModel):
    """
    Response listing all available strategies.

    Parameters
    ----------
    strategies:
        One entry per available strategy.
    total:
        Total number of available strategies.
    """

    model_config = _API_MODEL_CONFIG

    strategies: list[StrategyInfoResponse] = Field(
        description="Available strategy definitions"
    )
    total: int = Field(description="Total strategy count")


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """
    Structured error response body.

    All non-2xx API responses use this schema. The ``code`` field enables
    programmatic error handling in the frontend without brittle string-matching
    on the ``detail`` message.

    Parameters
    ----------
    detail:
        Human-readable error description.
    code:
        Machine-readable error code, e.g. "RUN_NOT_FOUND".
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    detail: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")
