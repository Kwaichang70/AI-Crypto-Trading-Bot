"""
tests/unit/test_audit_log.py
-----------------------------
Unit tests for :mod:`api.services.audit_log` (Sprint 41 SEC-002).

Covered invariants
------------------
- Valid event types are persisted; invalid ones are skipped with a log
  line (DB level CHECK would reject them anyway — early skip keeps the
  session usable for the surrounding business transaction).
- Actor resolution prefers the X-API-Key hash prefix, falls back to
  "unknown" when auth is off, and returns "system" when no request is
  available.
- IP + User-Agent are extracted from the request when present.
- Persistence errors are swallowed — audit loss never blocks the
  functional request.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services.audit_log import record_audit_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    *,
    api_key: str | None = None,
    ip: str | None = "10.0.0.1",
    user_agent: str | None = "pytest/1.0",
) -> Any:
    headers: dict[str, str] = {}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    if user_agent is not None:
        headers["user-agent"] = user_agent
    client = SimpleNamespace(host=ip) if ip is not None else None

    # Headers mimic starlette's case-insensitive dict just well enough for the
    # service's .get("X-API-Key") and .get("user-agent") calls.
    class _Headers(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:  # type: ignore[override]
            return dict.get(self, key, dict.get(self, key.lower(), default))

    return SimpleNamespace(headers=_Headers(headers), client=client)


def _make_db() -> Any:
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# Event-type validation
# ---------------------------------------------------------------------------


class TestEventTypeValidation:
    async def test_invalid_event_type_is_skipped_silently(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="not_a_real_event",
            resource_type="run",
            resource_id="abc",
        )
        db.add.assert_not_called()
        db.flush.assert_not_called()

    async def test_live_trading_enabled_is_persisted(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="run-1",
        )
        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    async def test_model_activated_is_persisted(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="model_activated",
            resource_type="model_version",
            resource_id="mv-1",
        )
        db.add.assert_called_once()

    async def test_circuit_breaker_reset_is_persisted(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="circuit_breaker_reset",
            resource_type="circuit_breaker",
            resource_id="run-1",
        )
        db.add.assert_called_once()


# ---------------------------------------------------------------------------
# Actor resolution
# ---------------------------------------------------------------------------


class TestActorResolution:
    async def test_actor_is_system_when_no_request(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="r1",
        )
        event = db.add.call_args.args[0]
        assert event.actor == "system"

    async def test_actor_is_unknown_when_no_api_key_header(self) -> None:
        db = _make_db()
        req = _make_request(api_key=None)
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="r1",
            request=req,
        )
        event = db.add.call_args.args[0]
        assert event.actor == "unknown"

    async def test_actor_is_api_key_hash_prefix_when_key_present(self) -> None:
        api_key = "live-test-key-42"
        expected_prefix = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]

        db = _make_db()
        req = _make_request(api_key=api_key)
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="r1",
            request=req,
        )
        event = db.add.call_args.args[0]
        assert event.actor == f"api_key_{expected_prefix}"


# ---------------------------------------------------------------------------
# Transport metadata extraction
# ---------------------------------------------------------------------------


class TestTransportExtraction:
    async def test_ip_and_user_agent_captured_when_present(self) -> None:
        db = _make_db()
        req = _make_request(ip="192.168.1.7", user_agent="curl/8.0.0")
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="r1",
            request=req,
        )
        event = db.add.call_args.args[0]
        assert event.ip_address == "192.168.1.7"
        assert event.user_agent == "curl/8.0.0"

    async def test_ip_and_user_agent_none_without_request(self) -> None:
        db = _make_db()
        await record_audit_event(
            db,
            event_type="model_activated",
            resource_type="model_version",
            resource_id="mv-1",
        )
        event = db.add.call_args.args[0]
        assert event.ip_address is None
        assert event.user_agent is None


# ---------------------------------------------------------------------------
# Payload handling
# ---------------------------------------------------------------------------


class TestPayloadHandling:
    async def test_payload_is_stored_verbatim(self) -> None:
        db = _make_db()
        payload = {"symbols": ["BTC/USDT"], "initial_capital": "10000"}
        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id="r1",
            payload=payload,
        )
        event = db.add.call_args.args[0]
        assert event.payload == payload


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    async def test_flush_error_is_swallowed(self) -> None:
        """Audit persistence must never propagate an exception back to the
        router — a DB hiccup would otherwise abort the mutating request."""
        db = _make_db()
        db.flush = AsyncMock(side_effect=RuntimeError("simulated DB failure"))

        # Must not raise
        await record_audit_event(
            db,
            event_type="circuit_breaker_reset",
            resource_type="circuit_breaker",
            resource_id="r1",
        )
