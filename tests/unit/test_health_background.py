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
        assert body["context_poller"]["configured"] is False
        assert body["context_poller"]["alive"] is False
        assert body["retraining_service"]["configured"] is False
        assert body["retraining_service"]["running"] is False
        assert body["active_runs"]["count"] == 0
        assert body["active_runs"]["run_ids"] == []

    def test_warmer_key_is_a_constant_deprecated_compat_key(self) -> None:
        """CF-24-01 P5: the warmer is gone; the key stays, always unconfigured."""
        container = MagicMock()
        container.background_tasks.context_poller = None
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])
        body = _make_app(container=container).get("/api/v1/health/background").json()
        assert body["history_cache_warmer"] == {
            "configured": False,
            "running": False,
            "last_run_at_unix": None,
            "last_fgi_points": 0,
            "last_btc_dom_points": 0,
        }

    def test_retraining_diagnostics_surfaced(self) -> None:
        rsvc = MagicMock()
        rsvc.running = True
        rsvc.min_trades_for_retrain = 50
        rsvc.check_interval_seconds = 3600
        container = MagicMock()
        container.background_tasks.context_poller = None
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
        container.background_tasks.context_poller = None
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(
            return_value=["run-abc-001", "run-def-002", "run-ghi-003"]
        )
        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["active_runs"]["count"] == 3
        assert body["active_runs"]["run_ids"] == ["run-abc-001", "run-def-002", "run-ghi-003"]

    def test_a_partial_double_is_never_reported_as_a_poller(self) -> None:
        """A MagicMock in the poller slot is not a ContextPoller: reported unconfigured."""
        container = MagicMock()
        container.background_tasks.context_poller = MagicMock()
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])
        body = _make_app(container=container).get("/api/v1/health/background").json()
        assert body["context_poller"]["configured"] is False
        assert body["context_poller"]["series"] == []

    def test_response_shape_matches_schema(self) -> None:
        """CR-007 remediation: assert nested key sets so removing a
        Pydantic field is caught at test time."""
        client = _make_app(container=None)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert set(body.keys()) == {
            "history_cache_warmer", "context_poller", "retraining_service", "active_runs",
            "fx_cache_warmer",  # M6 (Sprint 49)
            "orphan_repeater",  # WP1.8b P-06
        }
        assert set(body["history_cache_warmer"].keys()) == {
            "configured", "running", "last_run_at_unix",
            "last_fgi_points", "last_btc_dom_points",
        }
        assert set(body["context_poller"].keys()) == {  # CF-24-01 P5
            "configured", "running", "alive", "heartbeat_age_seconds", "regressed", "series",
        }
        assert set(body["retraining_service"].keys()) == {
            "configured", "running",
            "min_trades_for_retrain", "check_interval_seconds",
        }
        assert set(body["active_runs"].keys()) == {"count", "run_ids"}
        assert set(body["fx_cache_warmer"].keys()) == {  # M6 (Sprint 49)
            "configured", "running", "last_run_at_unix", "last_rates_cached",
        }

    def test_orphan_repeater_diagnostics_surfaced(self) -> None:
        """WP1.8b (P-06): the S8 orphan-holding-unprotected alert repeater
        task's alive/dead state is surfaced."""
        container = MagicMock()
        container.background_tasks.context_poller = None
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])

        real_task = MagicMock()
        real_task.done.return_value = False
        container.background_tasks.orphan_repeater_task = real_task

        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["orphan_repeater"]["configured"] is True
        assert body["orphan_repeater"]["running"] is True

    def test_orphan_repeater_not_configured_when_no_container(self) -> None:
        client = _make_app(container=None)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["orphan_repeater"]["configured"] is False
        assert body["orphan_repeater"]["running"] is False

    def test_orphan_repeater_reports_not_running_when_task_done(self) -> None:
        container = MagicMock()
        container.background_tasks.context_poller = None
        container.services.retraining_service = None
        container.run_registry.active_run_ids = MagicMock(return_value=[])

        finished_task = MagicMock()
        finished_task.done.return_value = True
        container.background_tasks.orphan_repeater_task = finished_task

        client = _make_app(container=container)
        r = client.get("/api/v1/health/background")
        body = r.json()
        assert body["orphan_repeater"]["configured"] is True
        assert body["orphan_repeater"]["running"] is False
