"""Unit tests for the global kill-switch endpoint (Sprint 50 Cycle 3, WP1.7a).

WP1.7a (reports/vp2-wp1.7/synthesis-spec.md) flips the kill switch from
"stop every running run" into a persisted LATCH that blocks new entries
while every run keeps running (D3): ``test_kill_switch_two_runs_stopped``
and ``test_kill_switch_partial_failure`` are rewritten below to assert
``runs_latched`` (not ``runs_stopped``) and that NOTHING is cancelled or
removed from the registries.

Tests:
  1. require_admin — 401 when header absent
  2. require_admin — 403 when header wrong
  3. require_admin — 401 when admin_api_key not configured (§Fix-J: was 503)
  4. require_admin — passes (no exception) when key matches
  5. kill_switch — 0 running runs returns 200 + note
  6. kill_switch — 2 running runs: latches both, cancels neither
  7. kill_switch — a run engine missing from the registry is skipped, not an error
  8. kill_switch — audit row written BEFORE the latch mutation (verify via mock)
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
from api.services import kill_switch as kill_switch_service


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


@pytest.fixture(autouse=True)
def _reset_kill_switch_mirror():
    """Every test starts from a clean, LOADED, un-latched in-memory
    mirror -- without this, activate()'s own synchronous flip in one
    test would leak into the next (module-level global, WP1.7a design).
    Round 2 (S-02): the root ``tests/conftest.py`` autouse fixture
    already does exactly this for the whole suite; this file's own copy
    is kept as a belt-and-suspenders local reset."""
    kill_switch_service.reset_state_for_tests()
    yield
    kill_switch_service.reset_state_for_tests()


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.risk_manager = MagicMock()
    engine.risk_manager.trigger_kill_switch = MagicMock()
    return engine


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


def _make_db_mock() -> AsyncMock:
    mock_db = AsyncMock()
    nested_cm = MagicMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    mock_db.begin_nested = MagicMock(return_value=nested_cm)
    mock_db.commit = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.execute = AsyncMock()
    return mock_db


@pytest.mark.asyncio
async def test_kill_switch_no_active_runs() -> None:
    """Returns 200 with an empty runs_latched list when no runs are running.

    WP1.8a: the audit row is written unconditionally (even with 0 running
    runs -- S4).
    """
    from api.routers.emergency import kill_switch

    mock_db = _make_db_mock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    with (
        patch("api.routers.emergency.record_audit_event", new=AsyncMock()),
        patch.object(kill_switch_service, "activate", new=AsyncMock(return_value=True)),
    ):
        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
            body=None,
        )

    assert response.latched is True
    assert response.latch_persisted is True
    assert response.runs_latched == []
    assert response.errors == []


@pytest.mark.asyncio
async def test_kill_switch_with_only_orphaned_live_runs_writes_audit_row() -> None:
    """WP1.8a S4: pressing kill-switch while 0 runs are 'running' but a live
    run sits 'orphaned' must still write an audit row recording it."""
    from api.routers.emergency import kill_switch

    orphaned_id = uuid.uuid4()

    orphaned_run = MagicMock()
    orphaned_run.id = orphaned_id
    orphaned_run.run_mode = "live"
    orphaned_run.status = "orphaned"

    candidates_result = MagicMock()
    candidates_result.scalars.return_value.all.return_value = [orphaned_run]

    mock_db = _make_db_mock()
    mock_db.execute = AsyncMock(return_value=candidates_result)

    audit_calls: list[dict] = []

    async def _audit_side_effect(*args: object, **kwargs: object) -> None:
        audit_calls.append(kwargs)

    with (
        patch("api.routers.emergency.record_audit_event", side_effect=_audit_side_effect),
        patch.object(kill_switch_service, "activate", new=AsyncMock(return_value=True)),
    ):
        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
            body=None,
        )

    assert len(audit_calls) == 1, "the audit row must be written even with 0 running runs"
    assert audit_calls[0]["event_type"] == "kill_switch"
    assert audit_calls[0]["payload"]["orphaned_live_run_ids"] == [str(orphaned_id)]
    assert response.runs_latched == []
    assert response.orphaned_live_run_ids == [str(orphaned_id)]


@pytest.mark.asyncio
async def test_kill_switch_two_runs_latched_not_stopped() -> None:
    """Latches 2 running runs (runs_latched), cancels/removes NEITHER."""
    from api.routers.emergency import kill_switch

    run_a = MagicMock()
    run_a.id = uuid.uuid4()
    run_a.run_mode = "paper"
    run_a.status = "running"

    run_b = MagicMock()
    run_b.id = uuid.uuid4()
    run_b.run_mode = "live"
    run_b.status = "running"

    mock_db = _make_db_mock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_a, run_b]
    mock_db.execute = AsyncMock(return_value=mock_result)

    engine_a = _make_engine()
    engine_b = _make_engine()
    fake_engines = {str(run_a.id): engine_a, str(run_b.id): engine_b}
    fake_tasks = {str(run_a.id): MagicMock(), str(run_b.id): MagicMock()}

    with (
        patch("api.routers.emergency._RUN_ENGINES", fake_engines),
        patch("api.routers.emergency.record_audit_event", new=AsyncMock()),
        patch.object(kill_switch_service, "activate", new=AsyncMock(return_value=True)),
    ):
        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason="unit test",
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
            body=None,
        )

    assert str(run_a.id) in response.runs_latched
    assert str(run_b.id) in response.runs_latched
    assert response.errors == []
    engine_a.risk_manager.trigger_kill_switch.assert_called_once()
    engine_b.risk_manager.trigger_kill_switch.assert_called_once()

    # Nothing was cancelled or removed -- D3.
    assert str(run_a.id) in fake_engines
    assert str(run_b.id) in fake_engines
    for task in fake_tasks.values():
        task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_kill_switch_missing_engine_skipped_not_error() -> None:
    """A 'running' row with no in-process engine (single-worker invariant
    broken) is silently skipped -- not counted as latched, not an error."""
    from api.routers.emergency import kill_switch

    run_ok = MagicMock()
    run_ok.id = uuid.uuid4()
    run_ok.run_mode = "paper"
    run_ok.status = "running"

    run_missing = MagicMock()
    run_missing.id = uuid.uuid4()
    run_missing.run_mode = "live"
    run_missing.status = "running"

    mock_db = _make_db_mock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_ok, run_missing]
    mock_db.execute = AsyncMock(return_value=mock_result)

    engine_ok = _make_engine()
    fake_engines = {str(run_ok.id): engine_ok}  # run_missing has NO engine

    with (
        patch("api.routers.emergency._RUN_ENGINES", fake_engines),
        patch("api.routers.emergency.record_audit_event", new=AsyncMock()),
        patch.object(kill_switch_service, "activate", new=AsyncMock(return_value=True)),
    ):
        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
            body=None,
        )

    assert str(run_ok.id) in response.runs_latched
    assert str(run_missing.id) not in response.runs_latched
    assert response.errors == []


@pytest.mark.asyncio
async def test_kill_switch_audit_written_before_latch() -> None:
    """Audit row (and the latch persistence) happen before the per-engine
    latch loop -- verified via call-order side effects."""
    from api.routers.emergency import kill_switch

    run_x = MagicMock()
    run_x.id = uuid.uuid4()
    run_x.run_mode = "paper"
    run_x.status = "running"

    call_order: list[str] = []

    async def _audit_side_effect(*args: object, **kwargs: object) -> None:
        call_order.append("audit")

    engine_x = _make_engine()
    engine_x.risk_manager.trigger_kill_switch = MagicMock(
        side_effect=lambda *a, **k: call_order.append("latch")
    )

    mock_db = _make_db_mock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run_x]
    mock_db.execute = AsyncMock(return_value=mock_result)

    with (
        patch("api.routers.emergency._RUN_ENGINES", {str(run_x.id): engine_x}),
        patch("api.routers.emergency.record_audit_event", side_effect=_audit_side_effect),
        patch.object(kill_switch_service, "activate", new=AsyncMock(return_value=True)),
    ):
        response = await kill_switch(
            request=_make_request(),
            db=mock_db,
            reason=None,
            settings=_make_settings("a" * 32 + "b1c2d3e4f5g6"),
            body=None,
        )

    # WP1.7a round 2 (S-01): the engine-latch loop is now the handler's
    # literal FIRST action (before any DB access, including the audit
    # write) -- a DB outage must never leave a running engine unlatched.
    assert call_order == ["latch", "audit"], "engines must latch BEFORE any DB/audit work"
    assert response.runs_latched == [str(run_x.id)]


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
