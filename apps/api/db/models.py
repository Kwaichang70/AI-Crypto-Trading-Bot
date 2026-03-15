"""
apps/api/db/models.py
---------------------
SQLAlchemy 2.0 ORM models for the AI Crypto Trading Bot persistence layer.

Design principles:
- All monetary values use Numeric(20, 8)  -- never Float  -- to avoid IEEE-754 rounding
  errors that are catastrophic in financial accounting.
- Enum columns use String storage, not PostgreSQL native ENUM types. This avoids
  the painful ALTER TYPE migrations required when adding enum members (e.g., new
  OrderStatus states post-MVP).
- JSONB columns (config, metadata) give schemaless flexibility for strategy params
  and signal context without sacrificing PostgreSQL's indexing capabilities.
- All timestamps are stored WITH TIME ZONE. UTC is enforced at the application
  layer via Pydantic validators; the database stores the offset for correctness.
- Composite and partial indexes are placed deliberately  -- every query pattern that
  appears in the execution hot-path has a covering index.

Relationship to Pydantic models (packages/trading/models.py):
- These ORM models mirror the Pydantic domain models but are NOT the same objects.
- Conversion helpers belong in the service layer, not here.
- Do not import from trading.models here  -- that would create a circular dependency.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

__all__ = [
    "Base",
    "RunORM",
    "OrderORM",
    "FillORM",
    "TradeORM",
    "SkippedTradeORM",
    "EquitySnapshotORM",
    "SignalORM",
    "PositionSnapshotORM",
    "ModelVersionORM",
    "OptimizationRunORM",
    "OptimizationEntryORM",
]

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    All tables inherit from this class so Alembic autogenerate can discover
    them via a single ``Base.metadata`` import in env.py.
    """

    pass


# ---------------------------------------------------------------------------
# Monetary precision constant  -- never use Float for money
# ---------------------------------------------------------------------------
_MONEY = Numeric(precision=20, scale=8)


# ---------------------------------------------------------------------------
# 1. runs  -- Trading session records
# ---------------------------------------------------------------------------

class RunORM(Base):
    """
    Represents a single trading session (backtest, paper, or live run).

    Every order, fill, trade, signal, and equity snapshot belongs to exactly
    one run. The ``config`` JSONB column stores the immutable strategy
    parameters, symbol list, and timeframe that were active for the session.

    Status transitions:
        RUNNING -> STOPPED (normal shutdown)
        RUNNING -> ERROR   (unhandled exception or circuit breaker)

    Recovery chain (Sprint 24):
        When the API restarts, orphaned RUNNING paper/live runs are marked
        as ERROR and a new run is created from the same config.  The new
        run's ``recovered_from_run_id`` points back to the original orphan,
        forming a recoverable audit trail.
    """

    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "run_mode IN ('backtest', 'paper', 'live')",
            name="ck_runs_run_mode",
        ),
        CheckConstraint(
            "status IN ('running', 'stopped', 'error')",
            name="ck_runs_status",
        ),
        CheckConstraint(
            "stopped_at IS NULL OR stopped_at >= started_at",
            name="ck_runs_stopped_after_started",
        ),
    )

    # Primary key  -- UUID v4 generated at the application layer so callers
    # know the run_id before the INSERT completes (important for async flows).
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique run identifier (UUID v4)",
    )

    # Enum stored as String  -- see module docstring for rationale.
    run_mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Execution mode: backtest | paper | live",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="running",
        comment="Current run state: running | stopped | error",
    )

    # Strategy configuration snapshot  -- stored at run creation time.
    # Includes: strategy_id, symbols, timeframe, risk params, etc.
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Immutable strategy configuration snapshot for this run",
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC timestamp when the run was started",
    )
    stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when the run was stopped or errored. NULL = still running",
    )

    # Recovery self-reference (Sprint 24)  -- set on the NEW run that replaced
    # an orphaned run after an API restart.  NULL for all non-recovered runs.
    recovered_from_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id"),
        nullable=True,
        default=None,
        comment="If this run was auto-recovered, the ID of the original orphaned run",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Row creation time (immutable after INSERT)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        # NOTE: updated_at is maintained by the trigger_set_updated_at()
        # PostgreSQL trigger (defined in migration 001). Do not add onupdate=
        # here  -- it conflicts with the trigger and causes double-write overhead.
        comment="Row last-update time  -- maintained by DB trigger",
    )

    # Relationships  -- lazy="select" is the 2.0-safe default.
    # Use selectin_load() in query code for N+1 prevention.
    orders: Mapped[list[OrderORM]] = relationship(
        "OrderORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    trades: Mapped[list[TradeORM]] = relationship(
        "TradeORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    skipped_trades: Mapped[list[SkippedTradeORM]] = relationship(
        "SkippedTradeORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    equity_snapshots: Mapped[list[EquitySnapshotORM]] = relationship(
        "EquitySnapshotORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="EquitySnapshotORM.bar_index",
    )
    signals: Mapped[list[SignalORM]] = relationship(
        "SignalORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    position_snapshots: Mapped[list[PositionSnapshotORM]] = relationship(
        "PositionSnapshotORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<RunORM id={self.id} mode={self.run_mode} status={self.status}>"


# ---------------------------------------------------------------------------
# 2. orders  -- Order state machine records
# ---------------------------------------------------------------------------

class OrderORM(Base):
    """
    Persists the full lifecycle of a trading order.

    Maps directly to the Pydantic ``Order`` domain model. The ``status``
    column tracks the state machine: NEW -> PENDING_SUBMIT -> OPEN ->
    PARTIAL -> FILLED (or CANCELED / REJECTED / EXPIRED).

    Index strategy:
    - (run_id)  -- fetch all orders for a run
    - (client_order_id) UNIQUE  -- idempotency enforcement at the DB layer
    - (exchange_order_id)  -- reconciliation lookups from exchange callbacks
    - (run_id, symbol)  -- symbol-level P&L queries within a run
    - (status) PARTIAL  -- monitoring open/partial orders (hot path for the execution engine)
    """

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        CheckConstraint(
            "side IN ('buy', 'sell')",
            name="ck_orders_side",
        ),
        CheckConstraint(
            "order_type IN ('market', 'limit', 'stop_limit', 'stop_market')",
            name="ck_orders_order_type",
        ),
        CheckConstraint(
            "status IN ('new', 'pending_submit', 'open', 'partial', 'filled', 'canceled', 'rejected', 'expired')",
            name="ck_orders_status",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_orders_quantity_positive",
        ),
        CheckConstraint(
            "filled_quantity >= 0 AND filled_quantity <= quantity",
            name="ck_orders_filled_quantity_range",
        ),
        CheckConstraint(
            "price IS NULL OR price > 0",
            name="ck_orders_price_positive",
        ),
        Index("ix_orders_run_id", "run_id"),
        Index("ix_orders_exchange_order_id", "exchange_order_id"),
        Index("ix_orders_run_id_symbol", "run_id", "symbol"),
        # Partial index  -- only index non-terminal orders to keep the index small.
        # The execution engine polls this frequently; terminal orders are rarely queried.
        Index(
            "ix_orders_status_active",
            "status",
            postgresql_where="status IN ('new', 'pending_submit', 'open', 'partial')",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Internal order UUID (maps to Order.order_id in domain model)",
    )
    client_order_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Idempotency key sent to exchange. Format: <run_id>-<uuid4_hex[:12]>",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run that generated this order",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair, e.g. BTC/USDT",
    )
    side: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Order direction: buy | sell",
    )
    order_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Execution type: market | limit | stop_limit | stop_market",
    )
    quantity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Requested order size in base asset",
    )
    price: Mapped[Decimal | None] = mapped_column(
        _MONEY,
        nullable=True,
        comment="Limit price. NULL for MARKET orders",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="new",
        comment="Current state in the order state machine",
    )
    filled_quantity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        default=Decimal("0"),
        comment="Cumulative quantity filled so far",
    )
    average_fill_price: Mapped[Decimal | None] = mapped_column(
        _MONEY,
        nullable=True,
        comment="Volume-weighted average fill price. NULL when no fills yet",
    )
    exchange_order_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Exchange-assigned order ID. NULL until exchange acknowledges",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        # NOTE: Maintained by trigger_set_updated_at() DB trigger (migration 001).
        comment="Row last-update time  -- maintained by DB trigger",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="orders")
    fills: Mapped[list[FillORM]] = relationship(
        "FillORM",
        back_populates="order",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<OrderORM id={self.id} symbol={self.symbol} "
            f"side={self.side} status={self.status}>"
        )


# ---------------------------------------------------------------------------
# 3. fills  -- Execution fill events
# ---------------------------------------------------------------------------

class FillORM(Base):
    """
    Records individual execution fill events for an order.

    A single order may produce multiple fill rows (partial fills). The
    ``is_maker`` flag differentiates maker/taker fee tiers.

    Index strategy:
    - (order_id)  -- retrieve all fills for an order (join from orders)
    - (order_id, symbol) composite  -- P&L aggregation queries
    - (executed_at)  -- time-range queries for reporting
    """

    __tablename__ = "fills"
    __table_args__ = (
        CheckConstraint(
            "side IN ('buy', 'sell')",
            name="ck_fills_side",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_fills_quantity_positive",
        ),
        CheckConstraint(
            "price > 0",
            name="ck_fills_price_positive",
        ),
        CheckConstraint(
            "fee >= 0",
            name="ck_fills_fee_non_negative",
        ),
        Index("ix_fills_order_id", "order_id"),
        Index("ix_fills_order_id_symbol", "order_id", "symbol"),
        Index("ix_fills_executed_at", "executed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Fill UUID (maps to Fill.fill_id in domain model)",
    )
    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the parent order",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair, e.g. BTC/USDT",
    )
    side: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Fill direction: buy | sell",
    )
    quantity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Filled quantity in base asset",
    )
    price: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Execution price per unit of base asset",
    )
    fee: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Fee paid for this fill in quote asset (or fee_currency)",
    )
    fee_currency: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Currency in which fee was paid, e.g. USDT",
    )
    is_maker: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this fill earned maker (resting-order) fee rate",
    )
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp when the fill was executed on the exchange",
    )

    # Relationships
    order: Mapped[OrderORM] = relationship("OrderORM", back_populates="fills")

    def __repr__(self) -> str:
        return (
            f"<FillORM id={self.id} order_id={self.order_id} "
            f"symbol={self.symbol} qty={self.quantity} price={self.price}>"
        )


# ---------------------------------------------------------------------------
# 4. trades  -- Completed round-trip trade records
# ---------------------------------------------------------------------------

class TradeORM(Base):
    """
    Aggregated record of a completed round-trip trade (entry + exit fills).

    Written to the database when a position is fully closed by the portfolio
    accounting service. Maps to the ``TradeResult`` Pydantic domain model.

    Sprint 32 additions:
    - mae_pct, mfe_pct: Maximum Adverse/Favorable Excursion as fraction of entry
    - exit_reason: Classified exit trigger (trailing_stop, signal_exit, etc.)
    - regime_at_entry: Market regime label at position open
    - signal_context: JSONB snapshot of indicator values at entry

    Index strategy:
    - (run_id)  -- all trades for a run
    - (run_id, symbol)  -- per-symbol P&L within a run
    - (strategy_id)  -- aggregate performance by strategy
    - (entry_at, exit_at)  -- time-range queries for backtest reporting
    """

    __tablename__ = "trades"
    __table_args__ = (
        CheckConstraint(
            "side IN ('buy', 'sell')",
            name="ck_trades_side",
        ),
        CheckConstraint(
            "entry_price > 0 AND exit_price > 0",
            name="ck_trades_prices_positive",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_trades_quantity_positive",
        ),
        CheckConstraint(
            "total_fees >= 0",
            name="ck_trades_fees_non_negative",
        ),
        CheckConstraint(
            "exit_at >= entry_at",
            name="ck_trades_exit_after_entry",
        ),
        CheckConstraint(
            "exit_reason IS NULL OR exit_reason IN "
            "('take_profit', 'stop_loss', 'trailing_stop', 'signal_exit', 'regime_change', 'manual')",
            name="ck_trades_exit_reason",
        ),
        Index("ix_trades_run_id", "run_id"),
        Index("ix_trades_run_id_symbol", "run_id", "symbol"),
        Index("ix_trades_strategy_id", "strategy_id"),
        Index("ix_trades_entry_at", "entry_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Trade UUID (maps to TradeResult.trade_id in domain model)",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run that produced this trade",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair, e.g. BTC/USDT",
    )
    side: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Side of the opening fill: buy | sell",
    )
    entry_price: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Volume-weighted average entry price",
    )
    exit_price: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Volume-weighted average exit price",
    )
    quantity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Total traded quantity in base asset",
    )
    realised_pnl: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Net realised PnL after all fees, in quote currency",
    )
    total_fees: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Total fees paid across all fills for this round-trip",
    )
    entry_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp of the first entry fill",
    )
    exit_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp of the final exit fill",
    )
    strategy_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Identifier of the strategy that generated the opening signal",
    )

    # Sprint 32: Adaptive learning fields
    mae_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment="Maximum Adverse Excursion as fraction of entry price (Sprint 32)",
    )
    mfe_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment="Maximum Favorable Excursion as fraction of entry price (Sprint 32)",
    )
    exit_reason: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Exit trigger: take_profit|stop_loss|trailing_stop|signal_exit|regime_change|manual",
    )
    regime_at_entry: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="Market regime at position open: RISK_ON|NEUTRAL|RISK_OFF",
    )
    signal_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Indicator snapshot at entry time for adaptive learning",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="trades")

    def __repr__(self) -> str:
        return (
            f"<TradeORM id={self.id} symbol={self.symbol} "
            f"side={self.side} pnl={self.realised_pnl}>"
        )


# ---------------------------------------------------------------------------
# 4b. skipped_trades  -- Trades evaluated but not taken (Sprint 32)
# ---------------------------------------------------------------------------

class SkippedTradeORM(Base):
    """
    Records trades that were evaluated but not taken.

    Persisted for adaptive learning analysis. Allows comparison between
    hypothetical outcomes of risk-blocked trades and actual outcomes of
    trades that were taken.

    Index strategy:
    - (run_id)  -- all skipped trades for a run
    - (run_id, symbol)  -- per-symbol skip analysis
    - (skipped_at)  -- time-range queries
    """

    __tablename__ = "skipped_trades"
    __table_args__ = (
        Index("ix_skipped_trades_run_id", "run_id"),
        Index("ix_skipped_trades_run_id_symbol", "run_id", "symbol"),
        Index("ix_skipped_trades_skipped_at", "skipped_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Skipped trade UUID (maps to SkippedTrade.skip_id in domain model)",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run that produced this skip event",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair that was evaluated, e.g. BTC/USDT",
    )
    skip_reason: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Human-readable reason the trade was not taken (e.g. risk rule name)",
    )
    regime_at_skip: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="Market regime label at skip time",
    )
    signal_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Indicator snapshot at skip time for adaptive learning",
    )
    hypothetical_entry_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8),
        nullable=True,
        comment="Price at which the trade would have been entered",
    )
    hypothetical_outcome_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment="Hypothetical return pct if the trade had been taken (filled post-hoc)",
    )
    hypothetical_outcome_filled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when the hypothetical outcome was computed",
    )
    skipped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC timestamp when the skip event occurred",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="skipped_trades")

    def __repr__(self) -> str:
        return (
            f"<SkippedTradeORM id={self.id} symbol={self.symbol} "
            f"reason={self.skip_reason}>"
        )


# ---------------------------------------------------------------------------
# 5. equity_snapshots  -- Equity curve data points
# ---------------------------------------------------------------------------

class EquitySnapshotORM(Base):
    """
    Time-series record of portfolio equity state at each bar close.

    Written by the portfolio accounting service on every bar. Forms the
    equity curve used by the backtesting metrics engine (Sharpe, Sortino,
    max drawdown calculations).

    Uses a BigInteger serial PK rather than UUID for insert performance  --
    these rows are written at very high frequency during backtests.

    Index strategy:
    - (run_id, bar_index) UNIQUE  -- prevents duplicate snapshots per bar
    - (run_id, timestamp)  -- time-range queries for equity curve plotting
    """

    __tablename__ = "equity_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "bar_index", name="uq_equity_snapshots_run_bar"),
        CheckConstraint(
            "equity >= 0",
            name="ck_equity_snapshots_equity_non_negative",
        ),
        CheckConstraint(
            "drawdown_pct >= 0 AND drawdown_pct <= 1",
            name="ck_equity_snapshots_drawdown_range",
        ),
        Index("ix_equity_snapshots_run_id", "run_id"),
        Index("ix_equity_snapshots_run_id_timestamp", "run_id", "timestamp"),
    )

    # Serial integer PK  -- Alembic will map this to BIGSERIAL in the DDL
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing surrogate key for insert performance",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run this snapshot belongs to",
    )
    equity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Total portfolio equity (cash + unrealised position value) in quote",
    )
    cash: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Available cash balance in quote currency",
    )
    unrealised_pnl: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Current unrealised profit/loss across all open positions",
    )
    realised_pnl: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Cumulative realised profit/loss since run start",
    )
    drawdown_pct: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=8),
        nullable=False,
        comment="Current drawdown as a fraction of peak equity (0.0 to 1.0)",
    )
    bar_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Zero-based bar number within the run (ordering key)",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp of the bar close that triggered this snapshot",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="equity_snapshots")

    def __repr__(self) -> str:
        return (
            f"<EquitySnapshotORM run={self.run_id} bar={self.bar_index} "
            f"equity={self.equity} dd={self.drawdown_pct}>"
        )


# ---------------------------------------------------------------------------
# 6. signals  -- Strategy signal log
# ---------------------------------------------------------------------------

class SignalORM(Base):
    """
    Immutable log of every trading signal emitted by a strategy.

    Signals are the causal link between strategy logic and order creation.
    Persisting them enables post-hoc analysis of why specific orders were
    placed and provides the raw data for strategy attribution.

    The ``metadata`` JSONB column stores arbitrary strategy context
    (indicator values, model outputs, etc.) without requiring schema changes.

    Index strategy:
    - (run_id)  -- all signals for a run
    - (run_id, symbol)  -- per-symbol signal history
    - (strategy_id)  -- cross-run strategy analysis
    - (generated_at)  -- time-range queries
    """

    __tablename__ = "signals"
    __table_args__ = (
        CheckConstraint(
            "direction IN ('buy', 'sell', 'hold')",
            name="ck_signals_direction",
        ),
        CheckConstraint(
            "target_position >= 0",
            name="ck_signals_target_position_non_negative",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_signals_confidence_range",
        ),
        Index("ix_signals_run_id", "run_id"),
        Index("ix_signals_run_id_symbol", "run_id", "symbol"),
        Index("ix_signals_strategy_id", "strategy_id"),
        Index("ix_signals_generated_at", "generated_at"),
    )

    # Serial PK for insert performance (signals are high-frequency during backtests)
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing surrogate key",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run that produced this signal",
    )
    strategy_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Unique identifier of the strategy that generated this signal",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair this signal applies to",
    )
    direction: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Signal direction: buy | sell | hold",
    )
    target_position: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Desired notional position size in quote currency (0 = flat)",
    )
    confidence: Mapped[float] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=False,
        default=1.0,
        comment="Strategy confidence score in [0.0, 1.0]",
    )
    signal_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Arbitrary strategy context: indicator values, model outputs, etc.",
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp when the signal was generated",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="signals")

    def __repr__(self) -> str:
        return (
            f"<SignalORM id={self.id} run={self.run_id} "
            f"strategy={self.strategy_id} symbol={self.symbol} dir={self.direction}>"
        )


# ---------------------------------------------------------------------------
# 7. position_snapshots  -- Final position state at run termination
# ---------------------------------------------------------------------------

class PositionSnapshotORM(Base):
    """
    Final position state snapshot, persisted when a run stops.

    One row per symbol per run. Stores the position state at run
    termination (for backtests: last bar; for paper runs: when stopped).

    Index strategy:
    - (run_id)  -- all position snapshots for a run
    - (run_id, symbol) UNIQUE  -- enforces one snapshot per symbol per run
    """

    __tablename__ = "position_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "symbol", name="uq_position_snapshots_run_symbol"),
        CheckConstraint("quantity >= 0", name="ck_position_snapshots_quantity_non_negative"),
        CheckConstraint("average_entry_price >= 0", name="ck_position_snapshots_entry_price_non_negative"),
        CheckConstraint("current_price >= 0", name="ck_position_snapshots_current_price_non_negative"),
        Index("ix_position_snapshots_run_id", "run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Position snapshot UUID",
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the run this snapshot belongs to",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair, e.g. BTC/USDT",
    )
    quantity: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Open position size in base asset at snapshot time",
    )
    average_entry_price: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Volume-weighted average entry price of the open position",
    )
    current_price: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Last known mark price used to compute unrealised PnL",
    )
    unrealised_pnl: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Unrealised profit/loss at snapshot time, in quote currency",
    )
    realised_pnl: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        comment="Cumulative realised profit/loss for this symbol within the run",
    )
    total_fees_paid: Mapped[Decimal] = mapped_column(
        _MONEY,
        nullable=False,
        default=Decimal("0"),
        comment="Total fees paid across all fills for this symbol within the run",
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp when the position was first opened",
    )
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC timestamp when the snapshot was recorded (run stop time)",
    )

    # Relationships
    run: Mapped[RunORM] = relationship("RunORM", back_populates="position_snapshots")

    def __repr__(self) -> str:
        return f"<PositionSnapshotORM run={self.run_id} symbol={self.symbol} qty={self.quantity}>"


# ---------------------------------------------------------------------------
# 8. model_versions  -- ML model version registry
# ---------------------------------------------------------------------------

class ModelVersionORM(Base):
    """
    Registry of trained ML model versions.

    Each row represents one trained RandomForestClassifier, tagged with the
    symbol/timeframe pair, training provenance, and accuracy metrics. At most
    one row per (symbol, timeframe) pair has is_active=True  -- the invariant
    is enforced at the application layer in RetrainingService.

    The model_path column holds the filesystem path to the .joblib file.
    RetrainingService prunes old versions when the count exceeds ml_max_model_versions.

    Index strategy:
    - (symbol, timeframe)  -- primary lookup for active model per symbol/timeframe
    - (trained_at)  -- prune query ORDER BY trained_at DESC
    - partial on (symbol, timeframe) WHERE is_active=true  -- fast active model lookup
      (created via op.execute() in the Alembic migration; SQLAlchemy ORM does not
      support partial index WHERE clauses on Index() objects).
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        CheckConstraint(
            "accuracy >= 0 AND accuracy <= 1",
            name="ck_model_versions_accuracy_range",
        ),
        CheckConstraint(
            "n_trades_used >= 0",
            name="ck_model_versions_n_trades_non_negative",
        ),
        CheckConstraint(
            "n_bars_used >= 0",
            name="ck_model_versions_n_bars_non_negative",
        ),
        CheckConstraint(
            "label_method IN ('trade_outcome', 'future_return')",
            name="ck_model_versions_label_method",
        ),
        CheckConstraint(
            "trigger IN ('manual', 'auto')",
            name="ck_model_versions_trigger",
        ),
        Index("ix_model_versions_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_model_versions_trained_at", "trained_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Model version UUID (app-generated v4)",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Trading pair this model was trained for, e.g. BTC/USD",
    )
    timeframe: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Candle timeframe this model was trained on, e.g. 1h",
    )
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="UTC wall-clock when training completed",
    )
    accuracy: Mapped[float] = mapped_column(
        Numeric(precision=6, scale=4),
        nullable=False,
        comment="Test-set accuracy in [0, 1]",
    )
    n_trades_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Count of closed trades used for training",
    )
    n_bars_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Count of OHLCV bars fetched for the training window",
    )
    label_method: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Labeling scheme: trade_outcome (PnL-based) or future_return (horizon-based)",
    )
    model_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Filesystem path to the .joblib model file",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if this is the currently active model for its (symbol, timeframe) pair",
    )
    trigger: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="manual",
        comment="What triggered training: manual (API call) or auto (RetrainingService)",
    )
    extra: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Supplementary metadata: feature importances, class distribution, etc.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Row creation time  -- immutable after INSERT",
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersionORM id={self.id} symbol={self.symbol} "
            f"tf={self.timeframe} acc={self.accuracy} active={self.is_active}>"
        )


# ---------------------------------------------------------------------------
# 9. optimization_runs  -- Parameter grid search session records
# ---------------------------------------------------------------------------

class OptimizationRunORM(Base):
    """
    Represents one completed parameter grid search (optimization) run.

    Written atomically when POST /api/v1/optimize completes successfully.
    Stores the top-N ranked parameter combinations in the related
    ``OptimizationEntryORM`` rows.

    Index strategy:
    - (created_at DESC)  -- list endpoint ORDER BY, newest-first pagination
    """

    __tablename__ = "optimization_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Optimization run UUID (app-generated v4)",
    )
    strategy_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Strategy identifier used for this search, e.g. ma_crossover",
    )
    # JSONB list of symbol strings  -- uses list[Any] per project JSONB convention
    # (SQLAlchemy returns list[Any] at runtime; Pydantic layer enforces list[str]).
    symbols: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment='CCXT-format trading pairs searched, e.g. ["BTC/USD"]',
    )
    timeframe: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Candle timeframe used for the search, e.g. 1h",
    )
    rank_by: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Metric used to rank results, e.g. sharpe_ratio",
    )
    total_combinations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Total number of parameter combinations in the grid",
    )
    completed_combinations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of combinations that completed without error",
    )
    failed_combinations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of combinations that raised an exception",
    )
    elapsed_seconds: Mapped[float] = mapped_column(
        Numeric(10, 3),
        nullable=False,
        comment="Wall-clock time for the full grid search in seconds",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC timestamp when the optimization run was persisted",
    )

    # Relationship  -- ordered by rank ascending so entry[0] is the best result
    entries: Mapped[list[OptimizationEntryORM]] = relationship(
        "OptimizationEntryORM",
        back_populates="optimization_run",
        cascade="all, delete-orphan",
        order_by="OptimizationEntryORM.rank",
    )

    def __repr__(self) -> str:
        return (
            f"<OptimizationRunORM id={self.id} strategy={self.strategy_name} "
            f"combinations={self.total_combinations}>"
        )


# ---------------------------------------------------------------------------
# 10. optimization_entries  -- Individual ranked parameter combination results
# ---------------------------------------------------------------------------

class OptimizationEntryORM(Base):
    """
    One ranked parameter combination result within an optimization run.

    Each row stores the parameter dict, the computed metrics dict, and the
    rank (1 = best) for a single backtest combination.

    Index strategy:
    - (optimization_run_id)  -- retrieve all entries for a run
    - (optimization_run_id, rank) UNIQUE  -- enforces one entry per rank per run
      and supports fast ORDER BY rank queries on the entries page.
    """

    __tablename__ = "optimization_entries"
    __table_args__ = (
        UniqueConstraint(
            "optimization_run_id",
            "rank",
            name="uq_optimization_entries_run_rank",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Entry UUID (app-generated v4)",
    )
    optimization_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False,
        comment="FK to the parent optimization run",
    )
    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Rank of this parameter combination (1 = best by rank_by metric)",
    )
    # JSONB parameter dict  -- uses dict[str, Any] per project JSONB convention.
    # The Pydantic response layer (OptimizeEntryResponse) enforces the actual types.
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment='Parameter combination for this entry, e.g. {"fast_period": 10}',
    )
    # JSONB metrics dict  -- uses dict[str, Any] per project JSONB convention.
    # Metrics are floats at runtime; the Pydantic layer enforces dict[str, float].
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment='Computed backtest metrics for this combination, e.g. {"sharpe_ratio": 1.2}',
    )

    # Relationship
    optimization_run: Mapped[OptimizationRunORM] = relationship(
        "OptimizationRunORM",
        back_populates="entries",
    )

    def __repr__(self) -> str:
        return (
            f"<OptimizationEntryORM run={self.optimization_run_id} "
            f"rank={self.rank} params={self.params}>"
        )
