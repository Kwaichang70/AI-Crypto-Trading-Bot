"""
apps/api/services/audit_log.py
-------------------------------
Helper service for appending to the ``audit_events`` table (Sprint 41 SEC-002).

The service isolates the audit-log write path from the mutating business
logic so routers never need to know the ORM column layout.  Failures to
persist an audit event never block the user-facing request — audit loss
is logged at WARNING and swallowed so the functional request still
succeeds (availability > perfect auditability on this sprint).

Design notes
------------
* Actor resolution is best-effort.  When ``require_api_auth=true`` the
  router has already validated the key before calling us; we capture the
  first 12 hex chars of the presented ``X-API-Key`` hash so the audit
  row is diagnostic without storing the raw key.  When auth is off the
  actor is recorded as ``"unknown"``.
* IP + User-Agent come directly from the ``Request`` object; both are
  optional because not every caller context has a request (e.g. lifespan
  hooks).  The caller may pass ``request=None`` to record ``"system"``
  events with empty transport metadata.
* Payload must be safe for JSONB serialisation — typically a shallow
  dict of primitive values.  Never include the confirm-token, the API
  key, or any exchange secret.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import AuditEventORM

__all__ = ["record_audit_event"]

logger = structlog.get_logger(__name__)


_VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "live_trading_enabled",
        "model_activated",
        "circuit_breaker_reset",
    }
)


def _resolve_actor(request: Request | None) -> str:
    """Return a best-effort actor identifier for the audit row.

    Captures the first 12 hex chars of ``sha256(X-API-Key)`` when the
    header is present, falling back to ``"unknown"`` when auth is
    disabled and ``"system"`` when no request is available.

    Defensive: returns ``"unknown"`` when the header lookup returns a
    non-string (e.g. a ``MagicMock`` from a router-level unit test that
    synthesises a request).  Treat audit actor resolution as best-effort —
    a type surprise should never crash the audit write.
    """
    if request is None:
        return "system"
    try:
        raw_key = request.headers.get("X-API-Key")
    except Exception:
        return "unknown"
    if not isinstance(raw_key, str) or not raw_key:
        return "unknown"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return f"api_key_{digest[:12]}"


def _resolve_transport(request: Request | None) -> tuple[str | None, str | None]:
    """Extract (ip_address, user_agent) from ``request`` when available.

    Defensive: returns ``(None, None)`` when attribute / header lookups
    raise or yield non-strings (unit-test mocks), matching the same
    "audit resolution is best-effort" philosophy as ``_resolve_actor``.
    """
    if request is None:
        return None, None
    ip: str | None
    try:
        client = request.client
        ip = client.host if client and isinstance(client.host, str) else None
    except Exception:
        ip = None
    try:
        ua = request.headers.get("user-agent")
        user_agent: str | None = ua if isinstance(ua, str) else None
    except Exception:
        user_agent = None
    return ip, user_agent


async def record_audit_event(
    db: AsyncSession,
    *,
    event_type: str,
    resource_type: str,
    resource_id: str,
    request: Request | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Persist a single audit event.

    Parameters
    ----------
    db:
        Active SQLAlchemy async session.  The caller controls the
        transaction lifecycle; this function only calls ``add()`` + ``flush()``.
    event_type:
        One of the values in :data:`_VALID_EVENT_TYPES`.  Mismatched
        values are logged and the write is skipped — ``ck_audit_events_event_type``
        would reject them at the DB level, but failing early keeps the
        session usable.
    resource_type:
        Classifier for the resource the event concerns (``"run"``,
        ``"model_version"``, ``"circuit_breaker"``).
    resource_id:
        Primary key string of the resource.
    request:
        Optional FastAPI ``Request`` for actor + transport extraction.
        Pass ``None`` for system-initiated events (lifespan recovery etc).
    payload:
        Optional JSONB payload.  MUST NOT include secret material.
    """
    if event_type not in _VALID_EVENT_TYPES:
        logger.warning(
            "audit_log.invalid_event_type",
            event_type=event_type,
            valid_types=sorted(_VALID_EVENT_TYPES),
        )
        return

    actor = _resolve_actor(request)
    ip, user_agent = _resolve_transport(request)

    event = AuditEventORM(
        id=uuid.uuid4(),
        timestamp=datetime.now(tz=UTC),
        actor=actor,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip,
        user_agent=user_agent,
        payload=payload,
    )

    try:
        db.add(event)
        await db.flush()
    except Exception:
        # Audit-log writes must never block the functional request — log
        # the failure so operators notice gaps in the trail.
        logger.exception(
            "audit_log.persist_failed",
            event_type=event_type,
            resource_id=resource_id,
        )
