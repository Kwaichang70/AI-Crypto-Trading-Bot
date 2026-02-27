"""Initial schema — runs, orders, fills, trades, equity_snapshots, signals

Revision ID: 001
Revises: None
Create Date: 2026-02-27 00:00:00.000000 UTC

Description
-----------
Creates the complete initial schema for the AI Crypto Trading Bot persistence
layer. This revision establishes six tables:

  runs            — Trading session records (backtest / paper / live)
  orders          — Order state machine records with full audit trail
  fills           — Individual execution fill events (partial-fill support)
  trades          — Completed round-trip trade records for P&L reporting
  equity_snapshots — Bar-by-bar equity curve data for metrics calculations
  signals         — Strategy signal log for attribution analysis

All monetary columns use NUMERIC(20, 8) to avoid IEEE-754 floating-point
rounding errors. Enum values are stored as VARCHAR to allow adding new enum
members without ALTER TYPE migrations.

Rollback strategy
-----------------
``downgrade()`` drops all tables in reverse dependency order using CASCADE.
No data migration is required — this is the initial schema, so downgrade
is only safe on an empty database or in development/test environments.

Performance considerations
--------------------------
All indexes are created with the standard blocking approach (not CONCURRENTLY)
because this is the initial schema with no existing data. For future index
additions on populated tables, use ``CREATE INDEX CONCURRENTLY`` to avoid
table locks:

    op.execute("CREATE INDEX CONCURRENTLY ix_name ON table (col)")

The composite index on equity_snapshots (run_id, bar_index) with a UNIQUE
constraint prevents duplicate snapshot writes, which is important for
idempotent backtest reruns.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Upgrade — create all tables
# ---------------------------------------------------------------------------

def upgrade() -> None:
    """
    Create all six tables in dependency order.

    Dependency order (FK constraints require parents before children):
        1. runs          (no foreign keys)
        2. orders        (FK -> runs)
        3. fills         (FK -> orders)
        4. trades        (FK -> runs)
        5. equity_snapshots (FK -> runs)
        6. signals       (FK -> runs)
    """

    # ------------------------------------------------------------------
    # 1. runs
    # ------------------------------------------------------------------
    op.create_table(
        "runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Unique run identifier (UUID v4)",
        ),
        sa.Column(
            "run_mode",
            sa.String(length=16),
            nullable=False,
            comment="Execution mode: backtest | paper | live",
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="running",
            comment="Current run state: running | stopped | error",
        ),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
            comment="Immutable strategy configuration snapshot for this run",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="UTC timestamp when the run was started",
        ),
        sa.Column(
            "stopped_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="UTC timestamp when the run was stopped or errored",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "run_mode IN ('backtest', 'paper', 'live')",
            name="ck_runs_run_mode",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'stopped', 'error')",
            name="ck_runs_status",
        ),
        sa.CheckConstraint(
            "stopped_at IS NULL OR stopped_at >= started_at",
            name="ck_runs_stopped_after_started",
        ),
    )
    # GIN index on config JSONB for fast containment queries:
    # SELECT * FROM runs WHERE config @> '{"strategy_id": "ma_crossover"}'
    op.create_index(
        "ix_runs_config_gin",
        "runs",
        ["config"],
        postgresql_using="gin",
    )
    # Index on status for monitoring queries: "show all running runs"
    op.create_index("ix_runs_status", "runs", ["status"])

    # ------------------------------------------------------------------
    # 2. orders
    # ------------------------------------------------------------------
    op.create_table(
        "orders",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Internal order UUID",
        ),
        sa.Column(
            "client_order_id",
            sa.String(length=64),
            nullable=False,
            comment="Idempotency key sent to exchange",
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE", name="fk_orders_run_id"),
            nullable=False,
        ),
        sa.Column(
            "symbol",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "side",
            sa.String(length=8),
            nullable=False,
        ),
        sa.Column(
            "order_type",
            sa.String(length=16),
            nullable=False,
        ),
        sa.Column(
            "quantity",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "price",
            sa.Numeric(precision=20, scale=8),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="new",
        ),
        sa.Column(
            "filled_quantity",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "average_fill_price",
            sa.Numeric(precision=20, scale=8),
            nullable=True,
        ),
        sa.Column(
            "exchange_order_id",
            sa.String(length=128),
            nullable=True,
            comment="Exchange-assigned order ID. NULL until exchange acknowledges",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="ck_orders_side"),
        sa.CheckConstraint(
            "order_type IN ('market', 'limit', 'stop_limit', 'stop_market')",
            name="ck_orders_order_type",
        ),
        sa.CheckConstraint(
            "status IN ('new', 'pending_submit', 'open', 'partial', 'filled', 'canceled', 'rejected', 'expired')",
            name="ck_orders_status",
        ),
        sa.CheckConstraint("quantity > 0", name="ck_orders_quantity_positive"),
        sa.CheckConstraint(
            "filled_quantity >= 0 AND filled_quantity <= quantity",
            name="ck_orders_filled_quantity_range",
        ),
        sa.CheckConstraint(
            "price IS NULL OR price > 0",
            name="ck_orders_price_positive",
        ),
    )
    op.create_index("ix_orders_run_id", "orders", ["run_id"])
    op.create_index(
        "ix_orders_client_order_id", "orders", ["client_order_id"]
    )
    op.create_index(
        "ix_orders_exchange_order_id", "orders", ["exchange_order_id"]
    )
    op.create_index("ix_orders_run_id_symbol", "orders", ["run_id", "symbol"])
    # Partial index — only index active (non-terminal) orders.
    # This keeps the index small and fast for the execution engine's
    # frequent polling of open/partial orders.
    op.create_index(
        "ix_orders_status_active",
        "orders",
        ["status"],
        postgresql_where=sa.text(
            "status IN ('new', 'pending_submit', 'open', 'partial')"
        ),
    )

    # ------------------------------------------------------------------
    # 3. fills
    # ------------------------------------------------------------------
    op.create_table(
        "fills",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "order_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("orders.id", ondelete="CASCADE", name="fk_fills_order_id"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column(
            "quantity",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "price",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "fee",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column("fee_currency", sa.String(length=16), nullable=False),
        sa.Column(
            "is_maker",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="ck_fills_side"),
        sa.CheckConstraint("quantity > 0", name="ck_fills_quantity_positive"),
        sa.CheckConstraint("price > 0", name="ck_fills_price_positive"),
        sa.CheckConstraint("fee >= 0", name="ck_fills_fee_non_negative"),
    )
    op.create_index("ix_fills_order_id", "fills", ["order_id"])
    op.create_index("ix_fills_order_id_symbol", "fills", ["order_id", "symbol"])
    op.create_index("ix_fills_executed_at", "fills", ["executed_at"])

    # ------------------------------------------------------------------
    # 4. trades
    # ------------------------------------------------------------------
    op.create_table(
        "trades",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE", name="fk_trades_run_id"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column(
            "entry_price",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "exit_price",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "quantity",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "realised_pnl",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "total_fees",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "entry_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "exit_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="ck_trades_side"),
        sa.CheckConstraint(
            "entry_price > 0 AND exit_price > 0",
            name="ck_trades_prices_positive",
        ),
        sa.CheckConstraint("quantity > 0", name="ck_trades_quantity_positive"),
        sa.CheckConstraint(
            "total_fees >= 0", name="ck_trades_fees_non_negative"
        ),
        sa.CheckConstraint(
            "exit_at >= entry_at", name="ck_trades_exit_after_entry"
        ),
    )
    op.create_index("ix_trades_run_id", "trades", ["run_id"])
    op.create_index("ix_trades_run_id_symbol", "trades", ["run_id", "symbol"])
    op.create_index("ix_trades_strategy_id", "trades", ["strategy_id"])
    op.create_index("ix_trades_entry_at", "trades", ["entry_at"])

    # ------------------------------------------------------------------
    # 5. equity_snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "equity_snapshots",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            comment="Auto-incrementing surrogate key for insert performance",
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "runs.id",
                ondelete="CASCADE",
                name="fk_equity_snapshots_run_id",
            ),
            nullable=False,
        ),
        sa.Column(
            "equity",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "cash",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "unrealised_pnl",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "realised_pnl",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "drawdown_pct",
            sa.Numeric(precision=10, scale=8),
            nullable=False,
        ),
        sa.Column(
            "bar_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "run_id",
            "bar_index",
            name="uq_equity_snapshots_run_bar",
        ),
        sa.CheckConstraint(
            "equity >= 0",
            name="ck_equity_snapshots_equity_non_negative",
        ),
        sa.CheckConstraint(
            "drawdown_pct >= 0 AND drawdown_pct <= 1",
            name="ck_equity_snapshots_drawdown_range",
        ),
    )
    op.create_index(
        "ix_equity_snapshots_run_id", "equity_snapshots", ["run_id"]
    )
    op.create_index(
        "ix_equity_snapshots_run_id_timestamp",
        "equity_snapshots",
        ["run_id", "timestamp"],
    )

    # ------------------------------------------------------------------
    # 6. signals
    # ------------------------------------------------------------------
    op.create_table(
        "signals",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "runs.id", ondelete="CASCADE", name="fk_signals_run_id"
            ),
            nullable=False,
        ),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column(
            "target_position",
            sa.Numeric(precision=20, scale=8),
            nullable=False,
        ),
        sa.Column(
            "confidence",
            sa.Numeric(precision=5, scale=4),
            nullable=False,
            server_default="1.0",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "direction IN ('buy', 'sell', 'hold')",
            name="ck_signals_direction",
        ),
        sa.CheckConstraint(
            "target_position >= 0",
            name="ck_signals_target_position_non_negative",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_signals_confidence_range",
        ),
    )
    op.create_index("ix_signals_run_id", "signals", ["run_id"])
    op.create_index(
        "ix_signals_run_id_symbol", "signals", ["run_id", "symbol"]
    )
    op.create_index("ix_signals_strategy_id", "signals", ["strategy_id"])
    op.create_index("ix_signals_generated_at", "signals", ["generated_at"])

    # ------------------------------------------------------------------
    # Trigger function: auto-update updated_at on row modification
    # ------------------------------------------------------------------
    # PostgreSQL function that sets updated_at = now() on any UPDATE.
    # Applied to tables that have an updated_at column.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION trigger_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Apply the trigger to runs and orders (the two tables with updated_at)
    for table_name in ("runs", "orders"):
        op.execute(
            f"""
            CREATE TRIGGER set_updated_at_{table_name}
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION trigger_set_updated_at();
            """
        )


# ---------------------------------------------------------------------------
# Downgrade — drop all tables in reverse dependency order
# ---------------------------------------------------------------------------

def downgrade() -> None:
    """
    Drop all tables created by this migration.

    CAUTION: This is a destructive operation. All data is permanently lost.
    Only run this in development or test environments, or after taking a
    full database backup.

    Tables are dropped in reverse FK dependency order to satisfy constraints.
    CASCADE is used on the trigger drops for safety.
    """
    # Drop triggers first (they reference the tables)
    for table_name in ("orders", "runs"):
        op.execute(
            f"DROP TRIGGER IF EXISTS set_updated_at_{table_name} ON {table_name};"
        )

    # Drop the trigger function
    op.execute(
        "DROP FUNCTION IF EXISTS trigger_set_updated_at() CASCADE;"
    )

    # Drop tables in reverse dependency order
    # (children before parents to satisfy FK constraints)
    op.drop_table("signals")
    op.drop_table("equity_snapshots")
    op.drop_table("trades")
    op.drop_table("fills")
    op.drop_table("orders")
    op.drop_table("runs")
