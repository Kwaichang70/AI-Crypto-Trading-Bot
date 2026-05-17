"""Extend audit_events.event_type CHECK to include 'emergency_stop' (SEC-006)

Revision ID: 010
Revises: 009
Create Date: 2026-05-17 00:00:00.000000 UTC

Description
-----------
Sprint 45 SEC-006 — dedicated emergency-stop endpoint writes an audit
row with ``event_type='emergency_stop'``.  The Sprint 41 migration 008
created ``ck_audit_events_event_type`` with only three allowed values
(live_trading_enabled / model_activated / circuit_breaker_reset);
adding a fourth value requires dropping + recreating the constraint.

PostgreSQL constraint-name conflicts: DROP CONSTRAINT must succeed
before CREATE CONSTRAINT in the same transaction, otherwise the new
constraint would race against the old one.  Alembic wraps the migration
in a transaction by default so this ordering is safe.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        "event_type IN ("
        "'live_trading_enabled', 'model_activated', "
        "'circuit_breaker_reset', 'emergency_stop'"
        ")",
    )


def downgrade() -> None:
    # Convert any emergency_stop rows to circuit_breaker_reset (closest
    # semantic neighbour) so the narrower constraint is satisfied after
    # the downgrade.  An operator running the downgrade manually after
    # production rows have accumulated needs this safety net.
    op.execute(
        "UPDATE audit_events SET event_type = 'circuit_breaker_reset' "
        "WHERE event_type = 'emergency_stop'"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        "event_type IN ("
        "'live_trading_enabled', 'model_activated', 'circuit_breaker_reset'"
        ")",
    )
