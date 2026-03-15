"""Add trade journal fields and skipped_trades table (Sprint 32)

Revision ID: 006
Revises: 005
Create Date: 2026-03-15 00:00:00.000000 UTC

Description
-----------
Sprint 32 — Adaptive Learning System foundation.

``trades`` table additions:
    Five nullable columns for adaptive learning metadata:
    - mae_pct       Numeric(10,6) — Maximum Adverse Excursion as fraction of entry
    - mfe_pct       Numeric(10,6) — Maximum Favorable Excursion as fraction of entry
    - exit_reason   String(32)    — Exit trigger classification
    - regime_at_entry String(16)  — Market regime at position open
    - signal_context JSONB        — Indicator snapshot at entry

    A CHECK constraint enforces valid exit_reason values:
    take_profit | stop_loss | trailing_stop | signal_exit | regime_change | manual

``skipped_trades`` table:
    New table for recording trades that were evaluated but not taken.
    Used by the adaptive learning system to compare hypothetical outcomes
    against actual risk blocks.

FK safety
---------
upgrade(): ALTER TABLE trades first (no FK), then CREATE TABLE skipped_trades.
downgrade(): DROP TABLE skipped_trades first, then ALTER TABLE trades.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Add 5 nullable adaptive learning columns to trades
    # ------------------------------------------------------------------
    op.add_column(
        "trades",
        sa.Column(
            "mae_pct",
            sa.Numeric(10, 6),
            nullable=True,
            comment="Maximum Adverse Excursion as fraction of entry price (Sprint 32)",
        ),
    )
    op.add_column(
        "trades",
        sa.Column(
            "mfe_pct",
            sa.Numeric(10, 6),
            nullable=True,
            comment="Maximum Favorable Excursion as fraction of entry price (Sprint 32)",
        ),
    )
    op.add_column(
        "trades",
        sa.Column(
            "exit_reason",
            sa.String(32),
            nullable=True,
            comment="Exit trigger: take_profit|stop_loss|trailing_stop|signal_exit|regime_change|manual",
        ),
    )
    op.add_column(
        "trades",
        sa.Column(
            "regime_at_entry",
            sa.String(16),
            nullable=True,
            comment="Market regime at position open: RISK_ON|NEUTRAL|RISK_OFF",
        ),
    )
    op.add_column(
        "trades",
        sa.Column(
            "signal_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Indicator snapshot at entry time for adaptive learning",
        ),
    )

    # CHECK constraint for exit_reason valid values
    op.create_check_constraint(
        "ck_trades_exit_reason",
        "trades",
        "exit_reason IS NULL OR exit_reason IN "
        "('take_profit', 'stop_loss', 'trailing_stop', 'signal_exit', 'regime_change', 'manual')",
    )

    # ------------------------------------------------------------------
    # 2. Create skipped_trades table
    # ------------------------------------------------------------------
    op.create_table(
        "skipped_trades",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Skipped trade UUID (app-generated v4)",
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
            comment="FK to the run that produced this skip event",
        ),
        sa.Column(
            "symbol",
            sa.String(32),
            nullable=False,
            comment="Trading pair that was evaluated, e.g. BTC/USDT",
        ),
        sa.Column(
            "skip_reason",
            sa.String(64),
            nullable=False,
            comment="Human-readable reason the trade was not taken (e.g. risk rule name)",
        ),
        sa.Column(
            "regime_at_skip",
            sa.String(16),
            nullable=True,
            comment="Market regime label at skip time",
        ),
        sa.Column(
            "signal_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Indicator snapshot at skip time for adaptive learning",
        ),
        sa.Column(
            "hypothetical_entry_price",
            sa.Numeric(20, 8),
            nullable=True,
            comment="Price at which the trade would have been entered",
        ),
        sa.Column(
            "hypothetical_outcome_pct",
            sa.Numeric(10, 6),
            nullable=True,
            comment="Hypothetical return pct if the trade had been taken (filled post-hoc)",
        ),
        sa.Column(
            "hypothetical_outcome_filled_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="UTC timestamp when the hypothetical outcome was computed",
        ),
        sa.Column(
            "skipped_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
            comment="UTC timestamp when the skip event occurred",
        ),
    )

    # Indexes for query patterns on skipped_trades
    op.create_index(
        "ix_skipped_trades_run_id",
        "skipped_trades",
        ["run_id"],
    )
    op.create_index(
        "ix_skipped_trades_run_id_symbol",
        "skipped_trades",
        ["run_id", "symbol"],
    )
    op.create_index(
        "ix_skipped_trades_skipped_at",
        "skipped_trades",
        ["skipped_at"],
    )


def downgrade() -> None:
    # Drop skipped_trades table first (FK child)
    op.drop_index("ix_skipped_trades_skipped_at", table_name="skipped_trades")
    op.drop_index("ix_skipped_trades_run_id_symbol", table_name="skipped_trades")
    op.drop_index("ix_skipped_trades_run_id", table_name="skipped_trades")
    op.drop_table("skipped_trades")

    # Remove trade journal columns from trades
    op.drop_constraint("ck_trades_exit_reason", "trades", type_="check")
    op.drop_column("trades", "signal_context")
    op.drop_column("trades", "regime_at_entry")
    op.drop_column("trades", "exit_reason")
    op.drop_column("trades", "mfe_pct")
    op.drop_column("trades", "mae_pct")
