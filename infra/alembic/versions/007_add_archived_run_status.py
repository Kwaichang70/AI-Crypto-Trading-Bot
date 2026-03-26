"""Extend runs.status check constraint to include 'archived' value

Revision ID: 007
Revises: 006
Create Date: 2026-03-26 00:00:00.000000 UTC

Description
-----------
Adds 'archived' as a valid value for the ``runs.status`` column so that
completed runs can be soft-deleted (hidden from the default listing) without
physical deletion.

The existing ``ck_runs_status`` check constraint only allowed
'running' | 'stopped' | 'error'.  This migration drops that constraint and
recreates it with the additional 'archived' value.

Downgrade path
--------------
Rows with status='archived' must be updated to 'stopped' before the
downgrade can succeed; otherwise the recreated narrower constraint will
reject existing rows.  The downgrade() function converts archived rows to
'stopped' automatically before reinstating the original constraint.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old constraint and recreate with 'archived' included.
    # PostgreSQL requires DROP CONSTRAINT before a new CHECK can be added.
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('running', 'stopped', 'error', 'archived')",
    )


def downgrade() -> None:
    # Convert archived rows to 'stopped' so the narrower constraint is
    # satisfied after the downgrade.
    op.execute(
        "UPDATE runs SET status = 'stopped' WHERE status = 'archived'"
    )
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('running', 'stopped', 'error')",
    )
