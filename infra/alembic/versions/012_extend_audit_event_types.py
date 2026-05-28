"""Extend audit_events.event_type CHECK to include 'kill_switch' (Sprint 50 Cycle 3)

Revision ID: 012
Revises: 011
Create Date: 2026-05-27 00:00:00.000000 UTC

Description
-----------
Sprint 50 Cycle 3 — global kill-switch endpoint writes an audit row with
``event_type='kill_switch'``.  The current constraint (migration 010) allows
four values: live_trading_enabled / model_activated / circuit_breaker_reset /
emergency_stop.  Adding a fifth requires DROP + CREATE of the named constraint.

PostgreSQL requires DROP CONSTRAINT before CREATE CONSTRAINT in the same DDL
statement.  Alembic wraps the migration in a transaction so the ordering is
safe (DROP succeeds before CREATE races).

Downgrade safety: existing 'kill_switch' rows are relabelled to 'emergency_stop'
(closest semantic neighbour) before the constraint is narrowed.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "012"
down_revision: Union[str, None] = "011"
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
        "'kill_switch'"
        ")",
    )


def downgrade() -> None:
    # Relabel kill_switch rows so the narrower constraint is satisfied after
    # downgrade.  Operator running the downgrade after production rows have
    # accumulated gets this safety net automatically.
    op.execute(
        "UPDATE audit_events SET event_type = 'emergency_stop' "
        "WHERE event_type = 'kill_switch'"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        "event_type IN ("
        "'live_trading_enabled', 'model_activated', "
        "'circuit_breaker_reset', 'emergency_stop'"
        ")",
    )
