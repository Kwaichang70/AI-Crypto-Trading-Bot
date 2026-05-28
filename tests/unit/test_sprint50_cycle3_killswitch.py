"""Unit tests for Sprint 50 Cycle 3 global kill-switch.

Tests:
  1. require_admin — 401 when header absent
  2. require_admin — 403 when header wrong
  3. require_admin — 401 when admin_api_key not configured (§Fix-J: was 503)
  4. require_admin — passes (no exception) when key matches
  5. kill_switch — 0 running runs returns 200 + note
  6. kill_switch — 2 running runs: stops both, returns summary
  7. kill_switch — 1 of 2 fails: returns stopped=1 + errors=[1]
  8. kill_switch — audit row written BEFORE stop (verify via mock)
  TestAdminApiKeyValidator — 6 validator cases (§Fix-H)
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from api.deps import require_admin
from api.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(admin_key: str = "test-admin-key-abc123") -> Settings:
    """Return a minimal Settings instance with admin_api_key set."""
    s = Settings.model_construct()
    object.__setattr__(s, "admin_api_key", SecretStr(admin_key))
    return s


def _make_request(headers: dict[str, str] | None = None) -> MagicMock:
    req = MagicMock()
    req.headers = headers or {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


# ---------------------------------------------------------------------------
# require_admin tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_require_admin_absent_header_returns_401() -> None:
    settings = _make_settings()
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(
            request=_make_request(),
            header_key=None,
            settings=settings,
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_admin_wrong_key_returns_403() -> None:
    settings = _make_settings(admin_key="correct-key-with-numbers-12345678901")
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(
            request=_make_request(),
            header_key="wrong-key",
            settings=settings,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_admin_not_configured_returns_401() -> None:
    """When ADMIN_API_KEY is empty, require_admin must return 401 (not 503)."""
    settings = _make_settings(admin_key="")
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(
            request=_make_request(),
            header_key="any-key",
            settings=settings,
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_admin_correct_key_passes() -> None:
    settings = _make_settings(admin_key="secret-key-with-numbers-12345678901")
    # Should not raise
    result = await require_admin(
        request=_make_request(),
        header_key="secret-key-with-numbers-12345678901",
        settings=settings,
    )
    assert result is None  # dependency returns None on success


# ---------------------------------------------------------------------------
# kill_switch endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_switch_no_active_runs() -> None:
    """Returns 200 with note when no runs are running."""
    from api.routers.emergency import kill_switch

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    response = await kill_switch(
        request=_make_request(),
        db=mock_db,
        reason=None,
        settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
    )

    assert response.runs_stopped == []
    assert response.note == "no active runs to stop"
    assert response.errors == []


@pytest.mark.asyncio
async def test_kill_switch_two_runs_stopped() -> None:
    """Stops 2 running runs, returns both UUIDs in runs_stopped."""
    from api.routers.emergency import kill_switch

    run_a = MagicMock()
    run_a.id = uuid.uuid4()
    run_a.run_mode = "paper"
    run_a.status = "running"

    run_b = MagicMock()
    run_b.id = uuid.uuid4()
    run_b.run_mode = "live"
    run_b.status = "running"

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_a, run_b]
    mock_db.execute = AsyncMock(return_value=mock_result)

    # Set up begin_nested as async context manager
    nested_cm = MagicMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    mock_db.begin_nested = MagicMock(return_value=nested_cm)

    with patch("api.routers.emergency._RUN_TASKS", {}), \
         patch("api.routers.emergency._RUN_ENGINES", {}), \
         patch("api.routers.emergency._LEARNING_INSTANCES", {}), \
         patch("api.routers.emergency.record_audit_event", new=AsyncMock()):

        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason="unit test",
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
        )

    assert str(run_a.id) in response.runs_stopped
    assert str(run_b.id) in response.runs_stopped
    assert response.errors == []
    assert response.note is None


@pytest.mark.asyncio
async def test_kill_switch_partial_failure() -> None:
    """One run fails to stop; other is stopped; both recorded correctly.

    Uses an explicit response list so the flush failure is deterministic:
      flush #1 = run_ok status update (succeeds)
      flush #2 = run_bad status update (raises RuntimeError)

    The audit write is inside begin_nested() and record_audit_event is
    patched, so it does NOT trigger a bare db.flush() call.
    """
    from api.routers.emergency import kill_switch

    run_ok = MagicMock()
    run_ok.id = uuid.uuid4()
    run_ok.run_mode = "paper"
    run_ok.status = "running"

    run_bad = MagicMock()
    run_bad.id = uuid.uuid4()
    run_bad.run_mode = "live"
    run_bad.status = "running"

    # Set up begin_nested as async context manager
    nested_cm = MagicMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)

    # flush_responses: 2 entries (1 per run in the loop)
    flush_responses: list[Exception | None] = [None, RuntimeError("DB connection lost")]

    async def _flush_side_effect() -> None:
        response = flush_responses.pop(0)
        if isinstance(response, Exception):
            raise response

    mock_db = AsyncMock()
    mock_db.begin_nested = MagicMock(return_value=nested_cm)
    mock_db.flush = _flush_side_effect
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_ok, run_bad]
    mock_db.execute = AsyncMock(return_value=mock_result)

    with patch("api.routers.emergency._RUN_TASKS", {}), \
         patch("api.routers.emergency._RUN_ENGINES", {}), \
         patch("api.routers.emergency._LEARNING_INSTANCES", {}), \
         patch("api.routers.emergency.record_audit_event", new=AsyncMock()):

        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
        )

    assert str(run_ok.id) in response.runs_stopped
    assert len(response.errors) == 1
    assert response.errors[0].run_id == str(run_bad.id)
    assert "DB connection lost" in response.errors[0].error_msg
    # Verify flush response list was fully consumed (both scheduled flushes fired)
    assert flush_responses == [], "Expected all 2 flush slots to be consumed"


@pytest.mark.asyncio
async def test_kill_switch_audit_written_before_stop() -> None:
    """Audit row must be written even when all stop attempts fail."""
    from api.routers.emergency import kill_switch

    run_x = MagicMock()
    run_x.id = uuid.uuid4()
    run_x.run_mode = "paper"
    run_x.status = "running"

    audit_called = False

    # Set up begin_nested as async context manager
    nested_cm = MagicMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)

    async def _audit_side_effect(*args: object, **kwargs: object) -> None:
        nonlocal audit_called
        audit_called = True
        # After audit is called, subsequent flush will raise
        mock_db.flush.side_effect = RuntimeError("forced failure")

    mock_db = AsyncMock()
    mock_db.begin_nested = MagicMock(return_value=nested_cm)
    mock_db.flush = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_x]
    mock_db.execute = AsyncMock(return_value=mock_result)

    with patch("api.routers.emergency._RUN_TASKS", {}), \
         patch("api.routers.emergency._RUN_ENGINES", {}), \
         patch("api.routers.emergency._LEARNING_INSTANCES", {}), \
         patch("api.routers.emergency.record_audit_event", side_effect=_audit_side_effect):

        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
        )

    assert audit_called, "Audit event must be called before stop attempts"
    # Run failed to stop (flush raised after audit), so errors list is non-empty
    assert len(response.errors) == 1


# ---------------------------------------------------------------------------
# _validate_admin_api_key validator tests (§Fix-H)
# ---------------------------------------------------------------------------

class TestAdminApiKeyValidator:
    """Tests for the _validate_admin_api_key field_validator in Settings."""

    def _make_settings_with_key(self, key: str) -> Settings:
        import os
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"
        return Settings(admin_api_key=key)  # type: ignore[arg-type]

    def test_empty_sentinel_allowed(self) -> None:
        """Empty string is the 'not configured' sentinel — must not raise."""
        s = self._make_settings_with_key("")
        assert s.admin_api_key.get_secret_value() == ""

    def test_placeholder_rejected(self) -> None:
        """Values starting with REPLACE_ME (any case) must raise ValueError."""
        with pytest.raises(Exception, match="REPLACE_ME"):
            self._make_settings_with_key("REPLACE_ME_admin_key_here")

    def test_short_key_rejected(self) -> None:
        """Keys shorter than 32 chars must raise ValueError."""
        with pytest.raises(Exception, match="32 characters"):
            self._make_settings_with_key("short")

    def test_all_alpha_rejected(self) -> None:
        """Keys that are entirely alphabetic must raise ValueError."""
        with pytest.raises(Exception, match="alphabetic"):
            self._make_settings_with_key("a" * 32)

    def test_all_numeric_rejected(self) -> None:
        """Keys that are entirely numeric must raise ValueError."""
        with pytest.raises(Exception, match="numeric"):
            self._make_settings_with_key("1" * 32)

    def test_valid_hex_key_accepted(self) -> None:
        """A valid 64-char hex key (openssl rand -hex 32 output) must pass."""
        valid_key = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
        s = self._make_settings_with_key(valid_key)
        assert s.admin_api_key.get_secret_value() == valid_key
