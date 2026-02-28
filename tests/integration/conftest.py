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
