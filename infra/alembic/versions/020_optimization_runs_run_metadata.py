"""Add optimization_runs.run_metadata JSONB column (WP2.1b, SY-21-08)

Revision ID: 020
Revises: 019
Create Date: 2026-10-16 00:00:00.000000 UTC

Description
-----------
WP2.1b (reports/vp2-wp2.1/synthesis-spec.md, SY-21-08 / AG-12) adds one
additive, nullable ``run_metadata JSONB`` column to ``optimization_runs``.
It holds ``{"cost_profile": ..., "engine_semantics": ...}`` so an
optimizer ranking is attributable to the cost profile it was produced
under (the optimize endpoint lets the caller choose any profile, U-2).

* One extensible column instead of separate ``cost_profile`` /
  ``engine_semantics`` columns, so later additions (for example
  ``engine_semantics.backtest_circuit_breaker`` in 2.1c, or ``data_hash``
  in WP2.6) need no further migration.
* Named ``run_metadata``, not ``metadata``: ``metadata`` is a reserved
  declarative attribute on SQLAlchemy models.
* No server default and no backfill. NULL (or a missing key) means a
  legacy row whose profile was never persisted -- truthful for every row
  written before 2.1b.

ORM contract
------------
This migration matches the ORM declaration on ``OptimizationRunORM``
exactly::

    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

(that is, ``mapped_column(JSONB, nullable=True)`` on
``OptimizationRunORM``). No DB-level column comment is set so the DB and
the ORM metadata stay identical.

Lock/scan characteristics
--------------------------
``ALTER TABLE ... ADD COLUMN`` of a nullable column with no default is a
catalog-only change on PostgreSQL (no table rewrite, no scan), but it does
need a brief ACCESS EXCLUSIVE lock on ``optimization_runs``. ``SET LOCAL
lock_timeout = '5s'`` (same convention as 017/018/019) makes that fail
fast instead of queueing behind a long-running transaction.

Deployment order
----------------
Apply this migration BEFORE the application code that reads or writes
``optimization_runs.run_metadata`` (mirrors 017/018/019). Old application
code that does not know the column keeps working against the migrated
schema.

Downgrade path
---------------
The column is dropped outright. Data loss on downgrade is limited to the
persisted cost-profile / engine-semantics attribution of optimization
runs; the rest of each row is untouched.

Rollback caveat: an image older than 020 cannot start on a DB at 020
(unknown revision); rollback = stop api -> downgrade to 019 with the new
image -> deploy the older image.

References
----------
SY-21-08, AG-12, AC-14, WP21-T-34 (reports/vp2-wp2.1/synthesis-spec.md).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.add_column(
        "optimization_runs",
        sa.Column(
            "run_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.drop_column("optimization_runs", "run_metadata")
