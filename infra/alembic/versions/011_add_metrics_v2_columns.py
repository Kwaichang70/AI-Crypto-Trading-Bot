"""add metrics_v2 columns: n_closed_trades + metrics_v2_backfilled

Revision ID: 011
Revises: 010
Create Date: 2026-05-22 00:00:00.000000 UTC

Sprint 49 M3 -- INF-3 closed/open split. Adds two columns to ``runs``:

``n_closed_trades`` (INTEGER NULL): populated by persist_backtest_results
for new backtest runs from Sprint 49 onward. NULL for pre-Sprint-49 runs
and for paper/live runs. Backfill script populates historical rows.

``metrics_v2_backfilled`` (BOOLEAN DEFAULT FALSE): one-shot idempotency
flag for the backfill script. After scripts/backfill_metrics_v2.py
confirms 100% coverage on production (metrics_v2_backfilled = TRUE for
all backtest rows), schedule a future migration to drop this column.
It is intentionally a one-shot operational flag, not permanent schema.

A partial index on (n_closed_trades) WHERE run_mode = 'backtest'
accelerates the M5 leaderboard hot-path. Plain CREATE INDEX (no
CONCURRENTLY) is used because:
- Production table has ~150 rows; plain index takes <5ms with negligible
  lock contention
- env.py uses async (asyncpg) engine + run_sync inside a
  context.begin_transaction() wrapper; op.execute("COMMIT") / op.execute("BEGIN")
  around CONCURRENTLY may commit a SAVEPOINT rather than the outer
  transaction (asyncpg nested-transaction semantics), leaving CREATE INDEX
  CONCURRENTLY still inside the outer transaction and causing the migration
  to fail with 'CREATE INDEX CONCURRENTLY cannot run inside a transaction block'.
- CONCURRENTLY's value at this table size is microseconds, not worth the
  async-path migration risk.
- This migration has zero precedent for transactional_ddl = False across
  the 10 prior migrations; plain op.create_index() is the established pattern.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "n_closed_trades",
            sa.Integer(),
            nullable=True,
            comment="Closed round-trip trade count. NULL for pre-M3 or non-backtest runs.",
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "metrics_v2_backfilled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment="True after backfill_metrics_v2.py has populated n_closed_trades.",
        ),
    )
    op.create_index(
        "ix_runs_n_closed_trades_backtest",
        "runs",
        ["n_closed_trades"],
        postgresql_where=sa.text("run_mode = 'backtest'"),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_runs_n_closed_trades_backtest", table_name="runs")
    op.drop_column("runs", "metrics_v2_backfilled")
    op.drop_column("runs", "n_closed_trades")
