"""Extend audit_events.event_type CHECK to include 'circuit_breaker_halt_auto_stop' (Sprint 50 Cycle 4)

Revision ID: 013
Revises: 012
Create Date: 2026-05-28 00:00:00.000000 UTC

Description
-----------
Sprint 50 Cycle 4 — graduated circuit breaker HALT now triggers an autonomous
engine shutdown and writes an audit row with
``event_type='circuit_breaker_halt_auto_stop'``.  The current constraint
(migration 012) allows five values: live_trading_enabled / model_activated /
circuit_breaker_reset / emergency_stop / kill_switch.  Adding a sixth requires
DROP + CREATE of the named constraint.

PostgreSQL requires DROP CONSTRAINT before CREATE CONSTRAINT.  Alembic wraps
the migration in a transaction so the ordering is safe.

Downgrade safety: existing 'circuit_breaker_halt_auto_stop' rows are relabelled
to 'emergency_stop' (closest semantic neighbour — both represent unplanned stops)
before the constraint is narrowed back to five values.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        "event_type IN ("
        "'live_trading_enabled', 'model_activated', "
        "'circuit_breaker_reset', 'emergency_stop', "
        "'kill_switch', 'circuit_breaker_halt_auto_stop'"
        ")",
    )


def downgrade() -> None:
    # Relabel halt-auto-stop rows so the narrower constraint is satisfied
    # after downgrade.  Operator running the downgrade gets this safety net.
    op.execute(
        "UPDATE audit_events SET event_type = 'emergency_stop' "
        "WHERE event_type = 'circuit_breaker_halt_auto_stop'"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        "event_type IN ("
        "'live_trading_enabled', 'model_activated', "
        "'circuit_breaker_reset', 'emergency_stop', "
        "'kill_switch'"
        ")",
    )
