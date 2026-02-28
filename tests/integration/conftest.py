"""
tests/integration/conftest.py
------------------------------
Shared fixtures for API integration tests.

These tests exercise the FastAPI application through httpx TestClient without
requiring a real PostgreSQL or Redis instance. Database-dependent endpoints use
FastAPI dependency overrides to inject mock sessions.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncGenerator
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings


# ---------------------------------------------------------------------------
# Deterministic test API key
# ---------------------------------------------------------------------------

TEST_RAW_KEY = "test-integration-api-key-2026"
TEST_KEY_HASH = hashlib.sha256(TEST_RAW_KEY.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _set_dev_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure environment for dev mode (no auth, no rate limiting)."""
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")


def _set_prod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure environment for production mode (auth enabled)."""
    monkeypatch.setenv("REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("API_KEY_HASH", TEST_KEY_HASH)
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")  # Disable rate limiting for auth tests
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")


# ---------------------------------------------------------------------------
# App factory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_dev_mode(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create a FastAPI app in dev mode (no auth, no rate limiting)."""
    _set_dev_env(monkeypatch)
    get_settings.cache_clear()
    from api.main import create_app
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture()
def client_dev(app_dev_mode: Any) -> Generator[TestClient, None, None]:
    """TestClient bound to a dev-mode app."""
    with TestClient(app_dev_mode, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def app_prod_mode(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create a FastAPI app in production mode (auth required)."""
    _set_prod_env(monkeypatch)
    get_settings.cache_clear()
    from api.main import create_app
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture()
def client_prod(app_prod_mode: Any) -> Generator[TestClient, None, None]:
    """TestClient bound to a production-mode app (auth required)."""
    with TestClient(app_prod_mode, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Headers with a valid API key for production-mode requests."""
    return {"X-API-Key": TEST_RAW_KEY}


# ---------------------------------------------------------------------------
# DB mock fixtures (used by test_runs_endpoints and any future DB tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """
    Mock AsyncSession for DB-dependent endpoint tests.

    Provides a realistic AsyncMock surface matching the SQLAlchemy
    AsyncSession interface used by the runs router:

    - execute()     -- AsyncMock; configure side_effect or return_value per test
    - add()         -- no-op MagicMock (synchronous in SQLAlchemy)
    - add_all()     -- no-op MagicMock (synchronous in SQLAlchemy)
    - flush()       -- AsyncMock no-op (awaited in handlers)
    - commit()      -- AsyncMock no-op (awaited by get_db dependency)
    - rollback()    -- AsyncMock no-op (awaited on exception paths)

    Usage in tests:
    ---------------
    For list_runs (two execute() calls):
        mock_db_session.execute.side_effect = [
            _make_scalar_result(count),       # COUNT(*) query
            _make_scalars_result([run_orm]),  # page query
        ]

    For get_run / stop_run (one execute() call):
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

    The fixture is function-scoped so every test starts with a fresh mock
    with no call history or pre-configured side effects.
    """
    session = AsyncMock()

    # Synchronous session operations (SQLAlchemy does not await these)
    session.add = MagicMock()
    session.add_all = MagicMock()

    # Async session operations
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    # execute() is AsyncMock by default from AsyncMock() construction;
    # tests configure its return_value or side_effect individually.
    # Reset to a clean AsyncMock to be explicit about the type.
    session.execute = AsyncMock()

    return session


@pytest.fixture()
def client_dev_with_db(
    app_dev_mode: Any,
    mock_db_session: AsyncMock,
) -> Generator[TestClient, None, None]:
    """
    TestClient wired to a dev-mode app with the get_db dependency overridden.

    The override replaces the real AsyncSession (which would attempt to
    connect to PostgreSQL) with the mock_db_session fixture.  Tests that
    use this fixture can inspect mock_db_session call history after the
    request is made.

    Cleanup: the dependency override is removed from the app after the test
    to prevent cross-test contamination (important since app_dev_mode is
    a function-scoped fixture that creates a fresh app per test, but the
    override dict persists on the object for the fixture's lifetime).
    """
    from api.db.session import get_db

    async def _override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_db_session

    app_dev_mode.dependency_overrides[get_db] = _override_get_db

    with TestClient(app_dev_mode, raise_server_exceptions=False) as c:
        yield c

    # Teardown: remove override so subsequent fixtures see a clean app
    app_dev_mode.dependency_overrides.pop(get_db, None)
