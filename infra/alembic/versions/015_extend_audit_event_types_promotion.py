"""Extend audit_events.event_type CHECK to include promotion + OOS bypass types
(Sprint 50 Cycle 5)

Revision ID: 015
Revises: 014
Create Date: 2026-05-29 00:00:00.000000 UTC

Description
-----------
Adds two new event_type values:
  - 'paper_promoted_to_live': paper run promoted to a live run by operator.
  - 'model_oos_gate_bypassed': OOS Sharpe gate bypassed via X-Override-OOS-Gate.

Current constraint (migration 013) allows six values.
Adding two more requires DROP + CREATE of the named constraint.
Downgrade relabels the two new types to 'model_activated' (closest semantic
neighbour) before narrowing the constraint.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ALL_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop', "
    "'paper_promoted_to_live', 'model_oos_gate_bypassed'"
)

_PREV_TYPES = (
    "'live_trading_enabled', 'model_activated', "
    "'circuit_breaker_reset', 'emergency_stop', "
    "'kill_switch', 'circuit_breaker_halt_auto_stop'"
)


def upgrade() -> None:
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_ALL_TYPES})",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE audit_events SET event_type = 'model_activated' "
        "WHERE event_type IN ('paper_promoted_to_live', 'model_oos_gate_bypassed')"
    )
    op.drop_constraint("ck_audit_events_event_type", "audit_events", type_="check")
    op.create_check_constraint(
        "ck_audit_events_event_type",
        "audit_events",
        f"event_type IN ({_PREV_TYPES})",
    )
