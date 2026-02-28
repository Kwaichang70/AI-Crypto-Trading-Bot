"""
tests/integration/test_auth.py
-------------------------------
Integration tests for API key authentication (apps/api/auth.py).

Tests cover:
- Dev mode bypass (require_api_auth=False)
- Production mode: missing key, invalid key, valid key (header + query)
- Misconfiguration: auth required but no hash configured
- Public endpoints remain accessible in all modes
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from tests.integration.conftest import TEST_KEY_HASH, TEST_RAW_KEY


@pytest.mark.integration
class TestAuthDevMode:
    """Authentication behaviour when require_api_auth=False."""

    def test_health_accessible_without_key(self, client_dev: TestClient) -> None:
        """GET /health should return 200 without any API key."""
        resp = client_dev.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_metrics_accessible_without_key(self, client_dev: TestClient) -> None:
        """GET /api/v1/metrics should return 200 without any API key."""
        resp = client_dev.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_protected_endpoint_accessible_in_dev_mode(self, client_dev: TestClient) -> None:
        """Protected endpoints should pass without key when auth is disabled."""
        resp = client_dev.get("/api/v1/runs")
        # May return 500 (no DB) but NOT 401 — that's the key assertion
        assert resp.status_code != 401


@pytest.mark.integration
class TestAuthProdMode:
    """Authentication behaviour when require_api_auth=True."""

    def test_missing_key_returns_401(self, client_prod: TestClient) -> None:
        """Request without API key should return 401."""
        resp = client_prod.get("/api/v1/runs")
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    def test_missing_key_has_www_authenticate_header(self, client_prod: TestClient) -> None:
        """401 response should include WWW-Authenticate header."""
        resp = client_prod.get("/api/v1/runs")
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "ApiKey"

    def test_invalid_key_returns_401(self, client_prod: TestClient) -> None:
        """Request with wrong API key should return 401."""
        resp = client_prod.get(
            "/api/v1/runs",
            headers={"X-API-Key": "wrong-key-completely-invalid"},
        )
        assert resp.status_code == 401

    def test_valid_key_via_header(self, client_prod: TestClient, auth_headers: dict) -> None:
        """Valid API key via X-API-Key header should pass authentication."""
        resp = client_prod.get("/api/v1/runs", headers=auth_headers)
        # Should NOT be 401 — may be 500 (no DB) but auth passed
        assert resp.status_code != 401

    def test_valid_key_via_query_param(self, client_prod: TestClient) -> None:
        """Valid API key via ?api_key= query param should pass authentication."""
        resp = client_prod.get(f"/api/v1/runs?api_key={TEST_RAW_KEY}")
        assert resp.status_code != 401

    def test_health_public_in_prod_mode(self, client_prod: TestClient) -> None:
        """GET /health should always be accessible even in production mode."""
        resp = client_prod.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_metrics_public_in_prod_mode(self, client_prod: TestClient) -> None:
        """GET /api/v1/metrics should always be public."""
        resp = client_prod.get("/api/v1/metrics")
        assert resp.status_code == 200


@pytest.mark.integration
class TestAuthMisconfigured:
    """Auth misconfiguration detection."""

    def test_auth_enabled_but_no_hash_returns_500(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When require_api_auth=True but api_key_hash is empty, return 500."""
        monkeypatch.setenv("REQUIRE_API_AUTH", "true")
        monkeypatch.setenv("API_KEY_HASH", "")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost:5432/x")
        monkeypatch.setenv("DEBUG", "true")
        get_settings.cache_clear()

        from api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/api/v1/runs",
                headers={"X-API-Key": "any-key"},
            )
            assert resp.status_code == 500
            assert "misconfigured" in resp.json()["detail"].lower()

        get_settings.cache_clear()
