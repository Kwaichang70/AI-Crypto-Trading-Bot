"""
tests/integration/test_sprint50_cycle3_integration.py
------------------------------------------------------
Integration coverage for POST /api/v1/emergency/kill-switch (Sprint 50 Cycle 3).

Uses the established conftest.py fixture pattern (client_dev / client_prod /
mock_db_session) — no real PostgreSQL required.

Endpoints under test
--------------------
- POST /api/v1/emergency/kill-switch
    401 when X-Admin-Key absent
    403 when X-Admin-Key wrong
    200 when X-Admin-Key correct + 0 running runs (idempotent no-op path)
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_ADMIN_KEY = "test-admin-key-hex32-abcdef0123456789"


# ---------------------------------------------------------------------------
# App fixture with admin key configured
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_with_admin(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    """Create a dev-mode app with ADMIN_API_KEY set and auth disabled."""
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ADMIN_API_KEY", _TEST_ADMIN_KEY)
    get_settings.cache_clear()
    from api.main import create_app
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture()
def client_admin(app_with_admin: Any) -> Generator[TestClient, None, None]:
    """TestClient bound to a dev-mode app with admin key configured."""
    with TestClient(app_with_admin, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Mock AsyncSession that returns an empty scalars result by default."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    # Default: no running runs
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)
    return session


@pytest.fixture()
def client_admin_with_db(
    app_with_admin: Any,
    mock_db_session: AsyncMock,
) -> Generator[TestClient, None, None]:
    """Client + DB override so kill-switch can complete without real Postgres."""
    from api.db.session import get_db

    async def _override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_db_session

    app_with_admin.dependency_overrides[get_db] = _override_get_db
    with TestClient(app_with_admin, raise_server_exceptions=False) as c:
        yield c
    app_with_admin.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKillSwitchAuth:
    def test_kill_switch_401_when_admin_header_absent(
        self, client_admin: TestClient
    ) -> None:
        """Missing X-Admin-Key must return 401, not 403 or 500."""
        response = client_admin.post("/api/v1/emergency/kill-switch")
        assert response.status_code == 401
        body = response.json()
        assert "detail" in body
        assert "X-Admin-Key" in body["detail"]

    def test_kill_switch_403_when_admin_header_wrong(
        self, client_admin: TestClient
    ) -> None:
        """Wrong X-Admin-Key must return 403, distinct from absent-key 401."""
        response = client_admin.post(
            "/api/v1/emergency/kill-switch",
            headers={"X-Admin-Key": "definitely-wrong-key"},
        )
        assert response.status_code == 403
        body = response.json()
        assert "detail" in body

    def test_kill_switch_200_no_runs_is_idempotent(
        self, client_admin_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """Correct X-Admin-Key + 0 running runs returns 200 with idempotent note.

        Covers the 0-runs early-return path: no audit row is written (no running
        runs means no scope to record), response contains note='no active runs to
        stop', and all numeric fields are zero.  Audit-written-before-stop coverage
        is handled by the unit test test_kill_switch_audit_written_before_stop.
        """
        with patch("api.routers.emergency.record_audit_event", new=AsyncMock()) as mock_audit:
            response = client_admin_with_db.post(
                "/api/v1/emergency/kill-switch",
                headers={"X-Admin-Key": _TEST_ADMIN_KEY},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["runs_stopped"] == []
        assert body["tasks_cancelled"] == 0
        assert body["engines_removed"] == 0
        assert body["errors"] == []
        assert body["note"] == "no active runs to stop"
        # Audit must NOT be called when there are no running runs (early-return path)
        mock_audit.assert_not_called()
