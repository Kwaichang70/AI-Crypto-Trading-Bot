"""Add audit_events table (Sprint 41 SEC-002)

Revision ID: 008
Revises: 007
Create Date: 2026-04-23 00:00:00.000000 UTC

Description
-----------
Sprint 41 SEC-002 — Tamper-evident audit log for security-sensitive API
mutations (live-trading gate pass, model activation, circuit-breaker reset).

Schema matches :class:`api.db.models.AuditEventORM`:
    id            UUID PK
    timestamp     TIMESTAMP WITH TIME ZONE  (indexed, default now())
    actor         VARCHAR(128)               (API key hash prefix or 'system')
    event_type    VARCHAR(64)                (indexed, CHECK constraint)
    resource_type VARCHAR(32)
    resource_id   VARCHAR(64)                (indexed)
    ip_address    VARCHAR(64)    NULL
    user_agent    VARCHAR(256)   NULL
    payload       JSONB          NULL

Indexes
-------
    ix_audit_events_timestamp            (default on DateTime(index=True))
    ix_audit_events_event_type           (default on String(index=True))
    ix_audit_events_resource_id          (default on String(index=True))
    ix_audit_events_timestamp_event_type (explicit composite in __table_args__)

CHECK constraint
----------------
``ck_audit_events_event_type`` limits event_type to the three values
enumerated in the ORM class docstring.  Adding a new event type in a
future sprint requires a migration that drops + recreates this constraint.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="UTC timestamp of the audited action",
        ),
        sa.Column(
            "actor",
            sa.String(length=128),
            nullable=False,
            comment=(
                "Actor identifier — API key hash prefix (first 12 chars of "
                "SHA-256), 'system' for lifespan-driven events, or "
                "'unknown' when require_api_auth=false."
            ),
        ),
        sa.Column(
            "event_type",
            sa.String(length=64),
            nullable=False,
            comment="Event classifier (see AuditEventORM docstring for valid values)",
        ),
        sa.Column(
            "resource_type",
            sa.String(length=32),
            nullable=False,
            comment="Resource classifier: 'run', 'model_version', 'circuit_breaker'",
        ),
        sa.Column(
            "resource_id",
            sa.String(length=64),
            nullable=False,
            comment="Resource identifier (typically a UUID string)",
        ),
        sa.Column(
            "ip_address",
            sa.String(length=64),
            nullable=True,
            comment="Client IP address captured from the request (when available)",
        ),
        sa.Column(
            "user_agent",
            sa.String(length=256),
            nullable=True,
            comment="Client User-Agent header (when available)",
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Event-specific context (model version id on activation, "
                "failed-layer names on rejected gate, etc.).  Never include "
                "secret material — confirm tokens / API keys MUST be "
                "hashed or omitted before persisting."
            ),
        ),
        sa.CheckConstraint(
            "event_type IN ("
            "'live_trading_enabled', 'model_activated', 'circuit_breaker_reset'"
            ")",
            name="ck_audit_events_event_type",
        ),
    )
    op.create_index(
        "ix_audit_events_timestamp",
        "audit_events",
        ["timestamp"],
    )
    op.create_index(
        "ix_audit_events_event_type",
        "audit_events",
        ["event_type"],
    )
    op.create_index(
        "ix_audit_events_resource_id",
        "audit_events",
        ["resource_id"],
    )
    op.create_index(
        "ix_audit_events_timestamp_event_type",
        "audit_events",
        ["timestamp", "event_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_timestamp_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_resource_id", table_name="audit_events")
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_timestamp", table_name="audit_events")
    op.drop_table("audit_events")
