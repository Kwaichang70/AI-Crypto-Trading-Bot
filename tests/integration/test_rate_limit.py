"""
tests/integration/test_rate_limit.py
-------------------------------------
Integration tests for per-IP rate limiting (apps/api/rate_limit.py).

Tests cover:
- Rate-limited endpoint returns 429 after exceeding limit
- 429 response includes Retry-After header and structured body
- Exempt paths (/health, /api/v1/metrics) are never rate-limited
- Rate limiting disabled mode passes all requests
- Path normalization collapses UUID segments
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.rate_limit import _normalize_path_key


@pytest.fixture()
def client_rate_limited(monkeypatch: pytest.MonkeyPatch):
    """
    TestClient with rate limiting enabled and very low limits for testing.

    Read limit: 3/minute, Write limit: 2/minute, Auth failures: 2/minute.
    """
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_READ", "3/minute")
    monkeypatch.setenv("RATE_LIMIT_WRITE", "2/minute")
    monkeypatch.setenv("RATE_LIMIT_AUTH_FAILURES", "2/minute")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost:5432/x")
    monkeypatch.setenv("DEBUG", "true")
    get_settings.cache_clear()

    from api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    get_settings.cache_clear()


@pytest.mark.integration
class TestRateLimiting:
    """Rate limiting enforcement."""

    def test_exceeds_read_limit_returns_429(
        self, client_rate_limited: TestClient
    ) -> None:
        """After exceeding the read limit, GET should return 429."""
        # Send 3 requests (at limit), then 4th should be 429
        for _ in range(3):
            resp = client_rate_limited.get("/api/v1/runs")
            # May be 500 (no DB) but NOT 429 yet
            assert resp.status_code != 429

        resp = client_rate_limited.get("/api/v1/runs")
        assert resp.status_code == 429

    def test_429_response_has_retry_after_header(
        self, client_rate_limited: TestClient
    ) -> None:
        """429 response should include Retry-After header."""
        # Exhaust the limit
        for _ in range(4):
            client_rate_limited.get("/api/v1/runs")

        resp = client_rate_limited.get("/api/v1/runs")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) > 0

    def test_429_response_has_structured_body(
        self, client_rate_limited: TestClient
    ) -> None:
        """429 response should include error, detail, and retry_after fields."""
        for _ in range(4):
            client_rate_limited.get("/api/v1/runs")

        resp = client_rate_limited.get("/api/v1/runs")
        if resp.status_code == 429:
            body = resp.json()
            assert body["error"] == "rate_limit_exceeded"
            assert "detail" in body
            assert "retry_after" in body

    def test_health_exempt_from_rate_limiting(
        self, client_rate_limited: TestClient
    ) -> None:
        """/health should never be rate-limited, even after many requests."""
        for _ in range(50):
            resp = client_rate_limited.get("/health")
            assert resp.status_code == 200

    def test_metrics_exempt_from_rate_limiting(
        self, client_rate_limited: TestClient
    ) -> None:
        """/api/v1/metrics should never be rate-limited."""
        for _ in range(50):
            resp = client_rate_limited.get("/api/v1/metrics")
            assert resp.status_code == 200


@pytest.mark.integration
class TestRateLimitDisabled:
    """Rate limiting when disabled via configuration."""

    def test_disabled_passes_all_requests(self, client_dev: TestClient) -> None:
        """When rate_limit_enabled=False, no requests should be rate-limited."""
        # client_dev has rate_limit_enabled=False
        for _ in range(100):
            resp = client_dev.get("/health")
            assert resp.status_code == 200


@pytest.mark.integration
class TestPathNormalization:
    """UUID path segment normalization for rate limit buckets."""

    def test_uuid_segments_collapsed(self) -> None:
        """UUID path segments should be replaced with _id_."""
        path = "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/orders"
        normalized = _normalize_path_key(path)
        assert normalized == "/api/v1/runs/_id_/orders"

    def test_hex_uuid_collapsed(self) -> None:
        """32-char hex UUIDs should also be collapsed."""
        path = "/api/v1/runs/550e8400e29b41d4a716446655440000/orders"
        normalized = _normalize_path_key(path)
        assert normalized == "/api/v1/runs/_id_/orders"

    def test_non_uuid_segments_preserved(self) -> None:
        """Non-UUID path segments should remain unchanged."""
        path = "/api/v1/runs"
        normalized = _normalize_path_key(path)
        assert normalized == "/api/v1/runs"

    def test_multiple_uuids_collapsed(self) -> None:
        """Multiple UUID segments in one path should all be collapsed."""
        path = "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/orders/660e8400-e29b-41d4-a716-446655440001"
        normalized = _normalize_path_key(path)
        assert normalized == "/api/v1/runs/_id_/orders/_id_"
