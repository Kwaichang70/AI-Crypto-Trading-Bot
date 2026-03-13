"""Add recovered_from_run_id to runs table

Revision ID: 004
Revises: 003
Create Date: 2026-03-10 00:00:00.000000 UTC

Description
-----------
Adds ``recovered_from_run_id`` nullable UUID self-referencing FK column to
the ``runs`` table to support Sprint 24 run-recovery on API restart.

When the API container restarts, any paper/live run that was in ``status='running'``
is orphaned — its asyncio.Task has been killed.  The recovery logic in
``api.routers.runs.recover_orphaned_runs()`` marks the original as ``status='error'``
and creates a replacement run that points back to the original via this column.

This enables the UI to show the recovery chain and allows the operator to
trace which recovered run was spawned from which orphan.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "recovered_from_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id"),
            nullable=True,
            comment="If this run was auto-recovered, the ID of the original orphaned run",
        ),
    )

    # Index to support fast lookup of all recovery children for a given original run
    op.create_index(
        "ix_runs_recovered_from_run_id",
        "runs",
        ["recovered_from_run_id"],
        postgresql_where=sa.text("recovered_from_run_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_runs_recovered_from_run_id", table_name="runs")
    op.drop_column("runs", "recovered_from_run_id")
