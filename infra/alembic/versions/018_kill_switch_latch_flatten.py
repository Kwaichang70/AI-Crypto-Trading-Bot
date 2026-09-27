"""Kill-switch latch table + per-run entries latch + flatten audit events (WP1.7a)

Revision ID: 018
Revises: 017
Create Date: 2026-09-27 00:00:00.000000 UTC

Description
-----------
WP1.7a (Verbeterplan v2, reports/vp2-wp1.7/synthesis-spec.md) turns the
global kill switch into a persisted, fail-closed LATCH that blocks new
entries without stopping any run (D3), and adds a per-run latch that
survives an incomplete flatten:

1. New singleton table ``kill_switch_state`` (SY-01): a one-row table
   (``id`` fixed at 1 via CHECK) recording whether the global kill switch
   is active, when/by-whom it was last activated, and when/by-whom/why it
   was last cleared.  Read at boot by
   ``api.services.kill_switch.load()`` -- a read failure or a missing row
   leaves the process latched (fail-closed, I4).  Seeded with exactly one
   inactive row so a fresh deploy boots un-latched.

   Deriving the latch from ``audit_events`` instead (the arch-design
   alternative) was rejected: ``record_audit_event`` swallows flush
   failures (``audit_log.py``), and the kill-switch endpoint catches them
   again -- after a restart the latch would silently read as "off" with
   no trace of the failure.

2. Two new nullable ``runs`` columns (SY-02): ``entries_latch_reason``
   (the only value ever stored is ``'flatten_incomplete'``, enforced by
   ``ck_runs_entries_latch_reason``) and ``entries_latched_at``.  Set when
   a normal stop's flatten does not complete (I8/I9): the run keeps
   running with entries blocked until an operator clears it via
   ``POST /runs/{id}/entries-latch/clear``.  The global latch is never
   copied into these columns (SY-01).

3. Three new ``audit_events.event_type`` values: ``kill_switch_cleared``
   (global clear), ``run_flatten`` (flatten requested/completed/incomplete
   phases, written by ``StrategyEngine.flatten`` callers), and
   ``entries_latch_cleared`` (per-run latch clear).

Lock/scan characteristics
--------------------------
Same pattern as migrations 007/015/017: ``DROP CONSTRAINT`` + non-``NOT
VALID`` ``CREATE CHECK CONSTRAINT`` on ``audit_events`` takes an ACCESS
EXCLUSIVE lock and re-validates every existing row -- expected to be brief
on this deployment's row counts (see 017's docstring for the full
rationale). ``SET LOCAL lock_timeout`` bounds the wait so a stuck
migration fails fast and loudly instead of hanging indefinitely.  Adding
two nullable columns with no default to ``runs`` and creating the new
singleton table are both fast, metadata-only operations.

Deployment order
----------------
Deploy and apply this migration BEFORE the application code that depends
on it (``kill_switch.py``, the rewritten ``/emergency/kill-switch``
endpoint, ``StrategyEngine.flatten``) -- exactly like 017's own ordering
requirement.

Downgrade path
--------------
``kill_switch_state`` and the two ``runs`` columns are dropped outright
(no historical data worth preserving -- the latch is operational state,
not an audit record).  Audit rows carrying one of the three new
``event_type`` values are relabelled to ``'emergency_stop'`` with the
original value preserved under ``payload['original_event_type']``,
mirroring 017's own downgrade for its four resume-lifecycle event types.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ALL_EVENT_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop', "
    "'paper_promoted_to_live', 'model_oos_gate_bypassed', "
    "'run_orphaned', 'run_resumed', 'run_resume_rejected', "
    "'resume_orders_imported', "
    "'kill_switch_cleared', 'run_flatten', 'entries_latch_cleared'"
)
_PREV_EVENT_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop', "
    "'paper_promoted_to_live', 'model_oos_gate_bypassed', "
    "'run_orphaned', 'run_resumed', 'run_resume_rejected', "
    "'resume_orders_imported'"
)



def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")

    # ------------------------------------------------------------------
    # 1. kill_switch_state -- singleton table, seeded inactive.
    # ------------------------------------------------------------------
    op.create_table(
        "kill_switch_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_by", sa.String(length=128), nullable=True),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cleared_by", sa.String(length=128), nullable=True),
        sa.Column("clear_reason", sa.Text(), nullable=True),
        sa.CheckConstraint("id = 1", name="ck_kill_switch_state_singleton"),
    )
    op.execute(
        "INSERT INTO kill_switch_state (id, active, reason) "
        "VALUES (1, false, NULL)"
    )

    # ------------------------------------------------------------------
    # 2. runs.entries_latch_reason / entries_latched_at.
    # ------------------------------------------------------------------
    op.add_column(
        "runs",
        sa.Column("entries_latch_reason", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("entries_latched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_runs_entries_latch_reason",
        "runs",
        "entries_latch_reason IS NULL OR entries_latch_reason = 'flatten_incomplete'",
    )

    # ------------------------------------------------------------------
    # 3. audit_events.event_type: three new values.
    # ------------------------------------------------------------------
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_ALL_EVENT_TYPES})",
    )


def downgrade() -> None:
    op.drop_constraint("ck_runs_entries_latch_reason", "runs", type_="check")
    op.drop_column("runs", "entries_latched_at")
    op.drop_column("runs", "entries_latch_reason")

    op.drop_table("kill_switch_state")

    op.execute(
        "UPDATE audit_events "
        "SET payload = COALESCE(payload, '{}'::jsonb) "
        "       || jsonb_build_object('original_event_type', event_type), "
        "    event_type = 'emergency_stop' "
        "WHERE event_type IN ("
        "'kill_switch_cleared', 'run_flatten', 'entries_latch_cleared')"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_PREV_EVENT_TYPES})",
    )
