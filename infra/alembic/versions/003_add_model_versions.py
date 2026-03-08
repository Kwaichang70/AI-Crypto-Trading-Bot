"""Add model_versions table

Revision ID: 003
Revises: 002
Create Date: 2026-03-07 00:00:00.000000 UTC

Description
-----------
Creates the model_versions table for tracking ML model training history
and version management (Sprint 23 — Adaptive Learning).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("timeframe", sa.String(8), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accuracy", sa.Numeric(6, 4), nullable=False),
        sa.Column("n_trades_used", sa.Integer, nullable=False),
        sa.Column("n_bars_used", sa.Integer, nullable=False),
        sa.Column("label_method", sa.String(32), nullable=False),
        sa.Column("model_path", sa.String(512), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "trigger",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'manual'"),
        ),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Check constraints
        sa.CheckConstraint(
            "accuracy >= 0 AND accuracy <= 1",
            name="ck_model_versions_accuracy_range",
        ),
        sa.CheckConstraint(
            "n_trades_used >= 0",
            name="ck_model_versions_n_trades_non_negative",
        ),
        sa.CheckConstraint(
            "n_bars_used >= 0",
            name="ck_model_versions_n_bars_non_negative",
        ),
        sa.CheckConstraint(
            "label_method IN ('trade_outcome', 'future_return')",
            name="ck_model_versions_label_method",
        ),
        sa.CheckConstraint(
            "trigger IN ('manual', 'auto')",
            name="ck_model_versions_trigger",
        ),
    )

    # Standard B-tree indexes
    op.create_index(
        "ix_model_versions_symbol_timeframe",
        "model_versions",
        ["symbol", "timeframe"],
    )
    op.create_index(
        "ix_model_versions_trained_at",
        "model_versions",
        ["trained_at"],
    )

    # Partial index — at most one active model per (symbol, timeframe)
    # Cannot express WHERE clauses via op.create_index(); use raw DDL.
    op.execute(
        """
        CREATE INDEX ix_model_versions_active_per_symbol
        ON model_versions (symbol, timeframe)
        WHERE is_active = true
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_model_versions_active_per_symbol")
    op.drop_index("ix_model_versions_trained_at", table_name="model_versions")
    op.drop_index("ix_model_versions_symbol_timeframe", table_name="model_versions")
    op.drop_table("model_versions")
