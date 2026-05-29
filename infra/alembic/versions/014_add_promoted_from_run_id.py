"""Add promoted_from_run_id FK to runs (Sprint 50 Cycle 5 Sub-scope A)

Revision ID: 014
Revises: 013
Create Date: 2026-05-29 00:00:00.000000 UTC

Description
-----------
Adds a self-referential FK column ``promoted_from_run_id`` to the ``runs``
table.  When an operator promotes a stopped paper run to live, the new live
run's ``promoted_from_run_id`` points back to the source paper run,
providing a full promotion audit trail.

Pattern mirrors migration 004 (``recovered_from_run_id``).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "promoted_from_run_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("runs.id"),
            nullable=True,
            comment=(
                "If this live run was created by promoting a paper run, "
                "the ID of the source paper run. NULL for all non-promoted runs."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("runs", "promoted_from_run_id")
