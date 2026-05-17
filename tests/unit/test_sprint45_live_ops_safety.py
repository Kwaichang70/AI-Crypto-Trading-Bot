"""
tests/unit/test_sprint45_live_ops_safety.py
--------------------------------------------
Sprint 45 — Live Operations Safety items:
  SEC-006   Emergency-stop endpoint exemption + audit_event row.
  AR-006    Max-concurrent-runs cap (503 on cap-hit).
  SEC-003   API key rotation: primary + secondary hash both authenticate.
"""

from __future__ import annotations

import hashlib

import pytest

from api.auth import _hash_key, verify_api_key
from api.config import Settings


# ---------------------------------------------------------------------------
# SEC-003: API key rotation window
# ---------------------------------------------------------------------------


class TestSEC003KeyRotation:
    def test_primary_only_matches(self) -> None:
        key = "primary-key-2026"
        h = _hash_key(key)
        assert verify_api_key(key, h) is True

    def test_wrong_key_rejected(self) -> None:
        h = _hash_key("primary")
        assert verify_api_key("secondary", h) is False

    def test_secondary_hash_matches(self) -> None:
        """During rotation, the secondary key must also authenticate."""
        primary = _hash_key("new-primary-2026")
        secondary = _hash_key("old-primary-2025")
        assert verify_api_key("old-primary-2025", primary, secondary) is True

    def test_primary_still_matches_with_secondary_set(self) -> None:
        primary = _hash_key("new-primary-2026")
        secondary = _hash_key("old-primary-2025")
        assert verify_api_key("new-primary-2026", primary, secondary) is True

    def test_third_key_rejected_with_both_set(self) -> None:
        primary = _hash_key("p")
        secondary = _hash_key("s")
        assert verify_api_key("forged", primary, secondary) is False

    def test_empty_secondary_disables_path(self) -> None:
        """Empty string secondary must NOT accidentally match anything."""
        h = _hash_key("real")
        assert verify_api_key("real", h, "") is True
        assert verify_api_key("fake", h, "") is False

    def test_settings_rejects_identical_primary_and_secondary(self) -> None:
        """SEC-003 model validator: identical hashes are a no-op rotation."""
        h = hashlib.sha256(b"k").hexdigest()
        with pytest.raises(ValueError, match="must be different hashes"):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                api_key_hash=h,
                api_key_hash_secondary=h,
            )

    def test_settings_accepts_different_primary_and_secondary(self) -> None:
        h1 = hashlib.sha256(b"k1").hexdigest()
        h2 = hashlib.sha256(b"k2").hexdigest()
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            api_key_hash=h1,
            api_key_hash_secondary=h2,
        )
        assert s.api_key_hash == h1
        assert s.api_key_hash_secondary == h2

    def test_settings_validates_secondary_hash_length(self) -> None:
        """Secondary hash must satisfy the same 64-char SHA-256 invariant."""
        with pytest.raises(ValueError, match="64-character"):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                api_key_hash_secondary="abc123",
            )


# ---------------------------------------------------------------------------
# AR-006: Max-concurrent-runs cap
# ---------------------------------------------------------------------------


class TestAR006ConcurrencyCap:
    def test_default_value_is_twenty(self) -> None:
        s = Settings(database_url="postgresql+asyncpg://u:p@h:5432/d")
        assert s.max_concurrent_runs == 20

    def test_can_be_overridden(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            max_concurrent_runs=50,
        )
        assert s.max_concurrent_runs == 50

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                max_concurrent_runs=0,
            )

    def test_rejects_above_one_hundred(self) -> None:
        with pytest.raises(ValueError):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                max_concurrent_runs=101,
            )


# ---------------------------------------------------------------------------
# SEC-006: emergency-stop event_type accepted by audit_log service
# ---------------------------------------------------------------------------


class TestSEC006EmergencyStopAuditType:
    """End-to-end Request-bound integration test for the emergency-stop
    endpoint is heavy; this unit test asserts the audit service accepts
    the new event_type so SEC-006 + SEC-002 stay aligned."""

    async def test_emergency_stop_event_type_persisted(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from api.services.audit_log import record_audit_event

        db = MagicMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        await record_audit_event(
            db,
            event_type="emergency_stop",
            resource_type="run",
            resource_id="run-1",
            payload={"reason": "manual override"},
        )
        db.add.assert_called_once()
        event = db.add.call_args.args[0]
        assert event.event_type == "emergency_stop"
        assert event.payload == {"reason": "manual override"}

    async def test_other_invalid_event_types_still_rejected(self) -> None:
        """The new valid type didn't accidentally weaken the gate."""
        from unittest.mock import AsyncMock, MagicMock

        from api.services.audit_log import record_audit_event

        db = MagicMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        await record_audit_event(
            db,
            event_type="not_a_real_event",
            resource_type="run",
            resource_id="r",
        )
        db.add.assert_not_called()
