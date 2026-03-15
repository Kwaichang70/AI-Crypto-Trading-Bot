"""Add optimization_runs and optimization_entries tables

Revision ID: 005
Revises: 004
Create Date: 2026-03-15 00:00:00.000000 UTC

Description
-----------
Adds two tables for the Sprint 31 parameter optimization persistence feature:

``optimization_runs``
    One row per completed POST /api/v1/optimize call. Stores the grid search
    metadata (strategy, symbols, timeframe, rank_by, combination counts, elapsed
    time) and acts as the parent for all ranked parameter entries.

``optimization_entries``
    One row per ranked parameter combination within an optimization run. Stores
    the parameter dict (JSONB), the computed metrics dict (JSONB), and the rank
    (1 = best). A unique constraint on (optimization_run_id, rank) prevents
    duplicate entries and makes ORDER BY rank queries efficient.

FK safety
---------
upgrade(): optimization_runs created before optimization_entries (FK parent first).
downgrade(): optimization_entries dropped before optimization_runs (FK child first).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. optimization_runs — parent table
    # ------------------------------------------------------------------
    op.create_table(
        "optimization_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Optimization run UUID (app-generated v4)",
        ),
        sa.Column(
            "strategy_name",
            sa.String(64),
            nullable=False,
            comment="Strategy identifier used for this search, e.g. ma_crossover",
        ),
        sa.Column(
            "symbols",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment='CCXT-format trading pairs searched, e.g. ["BTC/USD"]',
        ),
        sa.Column(
            "timeframe",
            sa.String(8),
            nullable=False,
            comment="Candle timeframe used for the search, e.g. 1h",
        ),
        sa.Column(
            "rank_by",
            sa.String(64),
            nullable=False,
            comment="Metric used to rank results, e.g. sharpe_ratio",
        ),
        sa.Column(
            "total_combinations",
            sa.Integer(),
            nullable=False,
            comment="Total number of parameter combinations in the grid",
        ),
        sa.Column(
            "completed_combinations",
            sa.Integer(),
            nullable=False,
            comment="Number of combinations that completed without error",
        ),
        sa.Column(
            "failed_combinations",
            sa.Integer(),
            nullable=False,
            comment="Number of combinations that raised an exception",
        ),
        sa.Column(
            "elapsed_seconds",
            sa.Numeric(10, 3),
            nullable=False,
            comment="Wall-clock time for the full grid search in seconds",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
            comment="UTC timestamp when the optimization run was persisted",
        ),
    )

    # Index for the list endpoint: ORDER BY created_at DESC
    op.create_index(
        "ix_optimization_runs_created_at",
        "optimization_runs",
        ["created_at"],
    )

    # ------------------------------------------------------------------
    # 2. optimization_entries — child table
    # ------------------------------------------------------------------
    op.create_table(
        "optimization_entries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            comment="Entry UUID (app-generated v4)",
        ),
        sa.Column(
            "optimization_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("optimization_runs.id", ondelete="CASCADE"),
            nullable=False,
            comment="FK to the parent optimization run",
        ),
        sa.Column(
            "rank",
            sa.Integer(),
            nullable=False,
            comment="Rank of this parameter combination (1 = best by rank_by metric)",
        ),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment='Parameter combination for this entry, e.g. {"fast_period": 10}',
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment='Computed backtest metrics for this combination, e.g. {"sharpe_ratio": 1.2}',
        ),
        sa.UniqueConstraint(
            "optimization_run_id",
            "rank",
            name="uq_optimization_entries_run_rank",
        ),
    )

    # FK index on optimization_entries.optimization_run_id — supports JOIN queries
    # and ON DELETE CASCADE lookups.  Created via migration (not ORM Index()) to
    # match the established project pattern (see ModelVersionORM, migration 003).
    op.create_index(
        "ix_optimization_entries_run_id",
        "optimization_entries",
        ["optimization_run_id"],
    )


def downgrade() -> None:
    # Drop child table first (FK constraint)
    op.drop_index("ix_optimization_entries_run_id", table_name="optimization_entries")
    op.drop_table("optimization_entries")

    # Drop parent table second
    op.drop_index("ix_optimization_runs_created_at", table_name="optimization_runs")
    op.drop_table("optimization_runs")
