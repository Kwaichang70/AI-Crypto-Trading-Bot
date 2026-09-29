"""Add orphaned/resuming run status and resume audit event types (WP1.8a)

Revision ID: 017
Revises: 016
Create Date: 2026-09-25 00:00:00.000000 UTC

Description
-----------
WP1.8a (orphan recovery + resume, Verbeterplan v2) needs two additions:

1. ``runs.status`` gains:
   - ``'orphaned'``: a live run whose engine task is gone (API restart or
     a graceful shutdown that caught the process mid-run) and now needs an
     operator-issued ``POST /runs/{id}/resume`` before it trades again.
   - ``'resuming'``: a short-lived compare-and-set lock held while a resume
     request is preparing the engine, so two concurrent resume requests
     for the same run cannot both succeed (WP18-R-02).
2. ``audit_events.event_type`` gains four resume-lifecycle events:
   ``'run_orphaned'``, ``'run_resumed'``, ``'run_resume_rejected'``,
   ``'resume_orders_imported'``.

Same ``DROP CONSTRAINT`` + ``CREATE CHECK`` pattern as migrations 007 and
015 -- a low-cardinality ``String`` column with a named CHECK, not a
native PG enum (module docstring in ``apps/api/db/models.py``).

Lock/scan characteristics (WP1.8a-round2 S-11 correction)
-----------------------------------------------------------
This is **not** an O(1) metadata-only change. ``DROP CONSTRAINT`` +
``CREATE CHECK CONSTRAINT`` (without ``NOT VALID``) takes an
``ACCESS EXCLUSIVE`` lock on the table for the duration of the operation
-- Postgres must re-validate the new CHECK against every existing row, a
full table scan, before the constraint is considered valid, and no other
transaction may read or write ``runs``/``audit_events`` while that scan
runs. On a table with a very large row count this can be a real,
user-visible stall, not a no-op. ``runs``/``audit_events`` are expected to
be small/moderate in this deployment (not append-only high-volume
tables), so the stall is expected to be brief, but this migration does
**not** use the safer ``ADD CONSTRAINT ... NOT VALID`` + ``VALIDATE
CONSTRAINT`` two-step (which only takes a brief lock to add the
constraint and validates without blocking concurrent writes) -- a future
migration touching a larger table should prefer that pattern instead.
``upgrade()`` sets ``SET LOCAL lock_timeout = '5s'`` so a stuck migration
fails fast and loudly (a clear error) rather than holding an
``ACCESS EXCLUSIVE`` lock indefinitely behind a long-running query on
either table.

Deployment order (WP1.8a-round2 S-11)
----------------------------------------
This migration **must be deployed and applied BEFORE the application code
that depends on it** (boot recovery writing ``status='orphaned'``, the
resume endpoint's ``'resuming'`` CAS, and the ``run_orphaned`` /
``run_resumed`` / ``run_resume_rejected`` / ``resume_orders_imported``
audit event types). Deploying the code first would immediately violate
the pre-017 ``CHECK`` constraints on the very first orphan/resume,
turning every attempted write into a hard database error instead of a
graceful application-level 409/500.

Downgrade path
---------------
Rows with ``status IN ('orphaned', 'resuming')`` are relabelled to
``'error'`` before the narrower ``ck_runs_status`` is restored (mirrors
007's archived->stopped relabel-then-narrow).

Audit rows carrying one of the four new ``event_type`` values are
relabelled to ``'emergency_stop'`` -- the closest semantic neighbour among
the pre-existing types (an operator-visible stop/intervention event) --
with the original value preserved under
``payload['original_event_type']`` so the relabel is forensically
reversible, before the narrower ``ck_audit_events_event_type`` is
restored (mirrors 015's promotion-event downgrade).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ALL_STATUSES = "'running', 'stopped', 'error', 'archived', 'orphaned', 'resuming'"
_PREV_STATUSES = "'running', 'stopped', 'error', 'archived'"

_ALL_EVENT_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop', "
    "'paper_promoted_to_live', 'model_oos_gate_bypassed', "
    "'run_orphaned', 'run_resumed', 'run_resume_rejected', "
    "'resume_orders_imported'"
)
_PREV_EVENT_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop', "
    "'paper_promoted_to_live', 'model_oos_gate_bypassed'"
)


def upgrade() -> None:
    # WP1.8a-round2 (S-11): fail fast and loudly on a stuck ACCESS
    # EXCLUSIVE lock acquisition (e.g. a long-running query holding a
    # weaker lock on the same table) instead of hanging indefinitely.
    # SET LOCAL is transaction-scoped -- Alembic runs each migration in
    # its own transaction, so this never leaks to other sessions.
    op.execute("SET LOCAL lock_timeout = '5s'")

    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        f"status IN ({_ALL_STATUSES})",
    )

    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_ALL_EVENT_TYPES})",
    )


def downgrade() -> None:
    op.execute("UPDATE runs SET status = 'error' WHERE status IN ('orphaned', 'resuming')")
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        f"status IN ({_PREV_STATUSES})",
    )

    op.execute(
        "UPDATE audit_events "
        "SET payload = COALESCE(payload, '{}'::jsonb) "
        "       || jsonb_build_object('original_event_type', event_type), "
        "    event_type = 'emergency_stop' "
        "WHERE event_type IN ("
        "'run_orphaned', 'run_resumed', 'run_resume_rejected', "
        "'resume_orders_imported')"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_PREV_EVENT_TYPES})",
    )
