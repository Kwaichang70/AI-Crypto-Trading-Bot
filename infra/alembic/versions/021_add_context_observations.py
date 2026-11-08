"""Point-in-time context observations: append-only ``context_observations`` (WP2.4 P5)

Revision ID: 021
Revises: 020
Create Date: 2026-11-08 00:00:00.000000 UTC

Description
-----------
WP2.4 (reports/vp2-wp2.4/synthesis-spec.md, SY-24-09) adds ONE append-only table that
stores context series (Fear & Greed, T10Y2Y, BTC dominance, whale net flow) together
with the instant each value became available to a trader. Backtests join on
``available_at <= close_i``; nothing else decides visibility.

Columns
-------
* ``series_id`` -- namespaced id, e.g. ``coingecko:btc_dominance``.
* ``observed_at`` -- the period the value describes.
* ``available_at`` -- first instant a trader could have known the value. A revision is
  a NEW row with a later ``available_at`` (primary key ``series_id, observed_at,
  available_at``); there is no revision counter.
* ``value`` -- ``NUMERIC(24,8)``: exact; the canonical digest text is derived from it.
* ``source`` -- provider label.
* ``available_at_basis`` -- ``recorded`` (our fetch time at first sighting),
  ``estimated_lag`` (backfill with a conservative lag) or ``vintage`` (provider stamp).
* ``ingested_at`` -- audit only (feeds ``ingest_cutoff``, never the join).

Constraints: ``available_at >= observed_at``, ``value <> 'NaN'`` (NUMERIC allows NaN),
the basis set, ``series_id <> ''``. No foreign keys (series are global, not per run), so
the migration takes no lock on any existing table. The store (apps/api/services/
context_store.py) additionally enforces the per-series lag floor (CF-SYN-01).

Lock/scan characteristics
--------------------------
Pure ``CREATE TABLE``: metadata only, instant. ``SET LOCAL lock_timeout = '5s'`` follows
the 019 convention. Deploy this migration BEFORE application code that reads or writes
the table (the reader and the recorder also fail soft on a missing table).

Downgrade path
---------------
``recorded`` rows cannot be re-fetched from any provider, so the downgrade REFUSES while
any ``recorded`` row exists, unless the operator passes ``-x allow_drop_context=1``. The
check runs under ``LOCK TABLE ... IN ACCESS EXCLUSIVE MODE`` so no recorder can add a
``recorded`` row between the check and the drop.
Backfilled rows (``estimated_lag``, ``vintage``) are reproducible and do not block.
In offline (``--sql``) mode the table cannot be inspected, so the downgrade requires
the flag as well. Rollback: stop api -> ``pg_dump -t context_observations`` -> downgrade
-> redeploy the old image. No UPDATE or DELETE exists in application code and no
retention applies (a run's recorded digest must stay reproducible).

Revision chain: ``down_revision = "020"``. If WP2.6 lands first, whichever migration
merges second rebases its ``down_revision`` (single-head rule).

References
----------
SY-24-09, CF-SYN-01, reports/vp2-wp2.4/db-design.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "021"
down_revision: str | None = "020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "context_observations"
_ALLOW_DROP_FLAG = "allow_drop_context"


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.create_table(
        _TABLE,
        sa.Column(
            "series_id",
            sa.String(length=64),
            nullable=False,
            comment="Namespaced series id, e.g. 'coingecko:btc_dominance'.",
        ),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Period the value describes.",
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="First instant a trader could have known the value (the join key).",
        ),
        sa.Column(
            "value",
            sa.Numeric(precision=24, scale=8),
            nullable=False,
            comment="Exact value; canonical digest text derives from it.",
        ),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "available_at_basis",
            sa.String(length=16),
            nullable=False,
            comment="recorded | estimated_lag | vintage.",
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="Audit only; feeds ingest_cutoff, never the join.",
        ),
        sa.PrimaryKeyConstraint(
            "series_id",
            "observed_at",
            "available_at",
            name="pk_context_observations",
        ),
        sa.CheckConstraint(
            "available_at >= observed_at",
            name="ck_context_obs_avail_after_obs",
        ),
        sa.CheckConstraint("value <> 'NaN'", name="ck_context_obs_value_finite"),
        sa.CheckConstraint(
            "available_at_basis IN ('recorded', 'estimated_lag', 'vintage')",
            name="ck_context_obs_basis",
        ),
        sa.CheckConstraint("series_id <> ''", name="ck_context_obs_series_id"),
    )


def _drop_allowed() -> bool:
    args = context.get_x_argument(as_dictionary=True)
    return args.get(_ALLOW_DROP_FLAG) == "1"


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    if not _drop_allowed():
        if context.is_offline_mode():
            raise RuntimeError(
                "refusing to drop context_observations in offline mode: the table cannot "
                "be inspected for recorded rows; pass -x allow_drop_context=1 to override"
            )
        # DB-R-06: close the check-then-drop window. A recorder cannot insert a
        # 'recorded' row between the EXISTS check and the DROP (lock_timeout bounds the wait).
        op.execute("LOCK TABLE context_observations IN ACCESS EXCLUSIVE MODE")
        recorded = op.get_bind().execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM context_observations "
                "WHERE available_at_basis = 'recorded')"
            )
        )
        if recorded.scalar():
            raise RuntimeError(
                "refusing to drop context_observations: it holds 'recorded' rows that "
                "cannot be re-fetched; pg_dump the table first, then rerun with "
                "-x allow_drop_context=1"
            )
    op.drop_table(_TABLE)
