"""
tests/unit/test_health_background.py
-------------------------------------
S47-6 (Sprint 47) -- /api/v1/health/background endpoint.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.health import router as health_router


def _make_app(container: Any | None) -> TestClient:
    """Build a minimal FastAPI app with the health router and a container
    attached to app.state.  No API-key auth -- that's tested separately at
    the main app level."""
    app = FastAPI()
    app.state.container = container
    app.include_router(health_router, prefix="/api/v1")
    return TestClient(app)


class TestBackgroundHealthEndpoint:

    def test_no_container_returns_all_false(self) -> None:
        client = _make_app(container=None)
        r = client.get("/api/v1/health/background")
        assert r.status_code == 200
        body = r.json()
        assert body["history_cache_warmer"]["configured"] is False
        assert body["history_cache_warmer"]["running"] is False
        assert body["retraining_service"]["configured"] is False
        assert body["retraining_service"]["running"] is False
        assert body["active_runs"]["count"] == 0
        assert body["active_runs"]["run_ids"] == []

    def test_warmer_diagnostics_surfaced(self) -> None:
        warmer = MagicMock()
        warmer.running = True
        warmer.last_run_at = 1747655400.0   # wall-clock unix epoch
        warmer.last_fgi_points = 30
        warmer.last_btc_dom_points = 25
        container = MagicMock()
        container.background_tasks.history_cache_warmer = warmer
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])
        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["history_cache_warmer"]["configured"] is True
        assert body["history_cache_warmer"]["running"] is True
        assert body["history_cache_warmer"]["last_run_at_unix"] == 1747655400.0
        assert body["history_cache_warmer"]["last_fgi_points"] == 30
        assert body["history_cache_warmer"]["last_btc_dom_points"] == 25

    def test_retraining_diagnostics_surfaced(self) -> None:
        rsvc = MagicMock()
        rsvc.running = True
        rsvc.min_trades_for_retrain = 50
        rsvc.check_interval_seconds = 3600
        container = MagicMock()
        container.background_tasks.history_cache_warmer = None
        container.services.retraining_service = rsvc
        container.run_registry.active_run_ids = MagicMock(return_value=[])
        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["retraining_service"]["configured"] is True
        assert body["retraining_service"]["running"] is True
        assert body["retraining_service"]["min_trades_for_retrain"] == 50
        assert body["retraining_service"]["check_interval_seconds"] == 3600

    def test_active_runs_count_and_ids(self) -> None:
        container = MagicMock()
        container.background_tasks.history_cache_warmer = None
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(
            return_value=["run-abc-001", "run-def-002", "run-ghi-003"]
        )
        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["active_runs"]["count"] == 3
        assert body["active_runs"]["run_ids"] == ["run-abc-001", "run-def-002", "run-ghi-003"]

    def test_warmer_not_running_is_distinguishable_from_unconfigured(self) -> None:
        """A configured-but-stopped warmer must report configured=True,
        running=False so operators can spot a crashed task vs a missing one."""
        warmer = MagicMock()
        warmer.running = False
        warmer.last_run_at = None
        warmer.last_fgi_points = 0
        warmer.last_btc_dom_points = 0
        container = MagicMock()
        container.background_tasks.history_cache_warmer = warmer
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])
        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["history_cache_warmer"]["configured"] is True
        assert body["history_cache_warmer"]["running"] is False

    def test_response_shape_matches_schema(self) -> None:
        """CR-007 remediation: assert nested key sets so removing a
        Pydantic field is caught at test time."""
        client = _make_app(container=None)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert set(body.keys()) == {
            "history_cache_warmer", "retraining_service", "active_runs",
        }
        assert set(body["history_cache_warmer"].keys()) == {
            "configured", "running", "last_run_at_unix",
            "last_fgi_points", "last_btc_dom_points",
        }
        assert set(body["retraining_service"].keys()) == {
            "configured", "running",
            "min_trades_for_retrain", "check_interval_seconds",
        }
        assert set(body["active_runs"].keys()) == {"count", "run_ids"}
