"""Idempotency-key dedup table for run creation and promotion (WP7.0, S12)

Revision ID: 019
Revises: 018
Create Date: 2026-09-28 00:00:00.000000 UTC

Description
-----------
WP7.0 (reports/vp2-wp7.0/synthesis-spec.md, SY-70-05) adds a dedicated
``idempotency_keys`` table so ``POST /api/v1/runs`` (all modes) and
``POST /api/v1/runs/{id}/promote-to-live`` can be made idempotent under a
client-supplied ``Idempotency-Key`` header (a UUID). This is pure
operational dedup state -- it is not an audit record.

Columns
-------
* ``key`` (PK) -- the client-supplied UUID, canonicalised to lowercase.
* ``endpoint`` / ``request_fingerprint`` -- identify which logical request
  this key was claimed for; a second request with the same key but a
  different fingerprint is rejected (422 ``idempotency_key_reused``).
* ``status`` -- ``in_progress`` | ``completed`` | ``failed``.
* ``claimed_run_id`` -- the provisional run id written at claim time,
  BEFORE the corresponding ``runs`` row is guaranteed to exist (or exist
  at all, if the claimant crashes first). Deliberately has **no** foreign
  key -- see database-architect review, ``reports/vp2-wp7.0/db-review.md``
  WP70-DB-01: enforcing a FK here would require the referenced row to
  already exist, which is exactly the ordering constraint this column is
  meant to relax. Existence is checked explicitly, when needed, via
  ``EXISTS (SELECT 1 FROM runs WHERE id = claimed_run_id)`` in the
  stale-claim reconciliation queries (SY-70-07 case 8).
* ``run_id`` -- the CONFIRMED run id. Written only inside ``complete()``,
  in the same transaction as (immediately before committing) the ``runs``
  INSERT itself -- at that point same-transaction MVCC guarantees the
  target row is visible, so a real ``REFERENCES runs(id) ON DELETE
  SET NULL`` foreign key is safe and enforced here.
* ``updated_at`` -- the staleness clock for case-8 reconciliation AND the
  TTL-prune clock (SY-70-14). Deliberately NOT ``created_at``, which never
  changes and would make a just-reclaimed row look permanently stale to
  the next reader, or let the daily prune delete a row that was just
  reclaimed (WP70 synthesis G-11).

Lock/scan characteristics
--------------------------
This is mostly a metadata-only ``CREATE TABLE`` plus one index -- fast
regardless of existing row counts. The one exception: the
``run_id REFERENCES runs(id)`` foreign key briefly takes a SHARE ROW
EXCLUSIVE lock on ``runs`` while PostgreSQL installs the RI triggers
(corrects an earlier claim in the design docs that this migration takes
"no lock on runs" -- see synthesis-spec.md G-15). ``SET LOCAL
lock_timeout = '5s'`` makes that fail fast and loudly instead of hanging
indefinitely; deploy with no long-running transaction held open against
``runs`` at the same time.

Deployment order
----------------
Deploy and apply this migration BEFORE the application code that reads
or writes ``idempotency_keys`` (mirrors 017/018's own ordering
requirement).

Downgrade path
---------------
The table (and its index) is dropped outright -- operational dedup state,
not an audit record worth preserving, exactly like 018's
``kill_switch_state`` downgrade rationale.

References
----------
SY-70-05, WP70-DB-01 (reports/vp2-wp7.0/{synthesis-spec,db-review}.md).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.create_table(
        "idempotency_keys",
        sa.Column("key", UUID(as_uuid=True), primary_key=True),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="in_progress",
        ),
        sa.Column("claimed_run_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "response_status_code",
            sa.Integer(),
            nullable=False,
            server_default="201",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('in_progress', 'completed', 'failed')",
            name="ck_idempotency_keys_status",
        ),
    )
    op.create_index(
        "ix_idempotency_keys_updated_at",
        "idempotency_keys",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_idempotency_keys_updated_at", table_name="idempotency_keys")
    op.drop_table("idempotency_keys")
