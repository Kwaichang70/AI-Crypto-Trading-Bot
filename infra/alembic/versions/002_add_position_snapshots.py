"""Add position_snapshots table

Revision ID: 002
Revises: 001
Create Date: 2026-03-04 00:00:00.000000 UTC

Description
-----------
Creates the position_snapshots table for persisting final position state
at run termination (Sprint 17).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "position_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("average_entry_price", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("current_price", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("unrealised_pnl", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("realised_pnl", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("total_fees_paid", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "symbol", name="uq_position_snapshots_run_symbol"),
        sa.CheckConstraint("quantity >= 0", name="ck_position_snapshots_quantity_non_negative"),
        sa.CheckConstraint("average_entry_price >= 0", name="ck_position_snapshots_entry_price_non_negative"),
        sa.CheckConstraint("current_price >= 0", name="ck_position_snapshots_current_price_non_negative"),
    )
    op.create_index("ix_position_snapshots_run_id", "position_snapshots", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_position_snapshots_run_id", table_name="position_snapshots")
    op.drop_table("position_snapshots")
