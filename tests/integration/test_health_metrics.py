"""
tests/integration/test_health_metrics.py
-----------------------------------------
Integration tests for health check and metrics endpoints.

Tests cover:
- GET /health returns 200 with status, uptime, version, timestamp
- GET /api/v1/metrics returns 200 with metrics snapshot
- Both endpoints are always public (no auth) even in production mode
- X-Process-Time header is present on responses
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestHealthEndpoint:
    """GET /health integration tests."""

    def test_health_returns_200(self, client_dev: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        resp = client_dev.get("/health")
        assert resp.status_code == 200

    def test_health_contains_required_fields(self, client_dev: TestClient) -> None:
        """Health response should contain status, uptime_seconds, version, timestamp."""
        resp = client_dev.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert "version" in data
        assert "timestamp" in data

    def test_health_public_with_auth_enabled(self, client_prod: TestClient) -> None:
        """Health endpoint should be accessible without auth even in prod mode."""
        resp = client_prod.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.integration
class TestMetricsEndpoint:
    """GET /api/v1/metrics integration tests."""

    def test_metrics_returns_200(self, client_dev: TestClient) -> None:
        """Metrics endpoint should return 200 OK."""
        resp = client_dev.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_contains_required_fields(self, client_dev: TestClient) -> None:
        """Metrics response should contain metrics and timestamp."""
        resp = client_dev.get("/api/v1/metrics")
        data = resp.json()
        assert "metrics" in data
        assert "timestamp" in data
        metrics = data["metrics"]
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics

    def test_metrics_has_default_counters(self, client_dev: TestClient) -> None:
        """Default counters should be present even before trading activity."""
        resp = client_dev.get("/api/v1/metrics")
        counters = resp.json()["metrics"]["counters"]
        assert "bars_processed_total" in counters
        assert "signals_generated_total" in counters
        assert "orders_submitted_total" in counters
        assert "fills_executed_total" in counters

    def test_metrics_public_with_auth_enabled(self, client_prod: TestClient) -> None:
        """Metrics endpoint should be accessible without auth in prod mode."""
        resp = client_prod.get("/api/v1/metrics")
        assert resp.status_code == 200


@pytest.mark.integration
class TestMiddleware:
    """General middleware integration tests."""

    def test_x_process_time_header_present(self, client_dev: TestClient) -> None:
        """X-Process-Time header should be present on all responses."""
        resp = client_dev.get("/health")
        assert "X-Process-Time" in resp.headers
        # Should be a valid number (milliseconds)
        elapsed = float(resp.headers["X-Process-Time"])
        assert elapsed >= 0
