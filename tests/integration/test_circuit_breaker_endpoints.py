"""
tests/integration/test_circuit_breaker_endpoints.py
----------------------------------------------------
Integration coverage for apps/api/routers/circuit_breaker.py (Sprint 41 TO-001).

Endpoints under test
--------------------
- GET  /api/v1/runs/{run_id}/circuit-breaker       — current state (404 paths here;
  the typed 200 payload is WP16-T-28 in ``test_wp16_cb_api_orchestrator.py``)
- POST /api/v1/runs/{run_id}/circuit-breaker/reset — RETIRED in WP1.6 (410 Gone)

WP1.6 (WP16-T-44): the breaker is auto-recovering and data-derived (U-1), so the
reset route is retired.  ``TestResetCircuitBreaker`` was rewritten from the
pre-WP1.6 409/200 "operator reset" assertions to 410 assertions, and the stub-based
GET-200 test (a ``MagicMock`` state that no longer validates against
``response_model=CircuitBreakerState``, SYN-04) was replaced by WP16-T-28 which
uses a real ``CircuitBreaker``.

The router reads the live engine from ``_RUN_ENGINES``.  Tests register a
lightweight stub engine and assert HTTP-level behaviour through ``TestClient``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.routers.circuit_breaker import RESET_RETIRED_DETAIL

_RUN_ID = "ce9bea50-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_engines_registry() -> Generator[dict[str, Any], None, None]:
    """Reset ``_RUN_ENGINES`` around each test so registrations do not leak."""
    from api.routers.runs import _RUN_ENGINES

    _RUN_ENGINES.clear()
    yield _RUN_ENGINES
    _RUN_ENGINES.clear()


def _make_engine(*, is_tripped: bool = False) -> Any:
    """Build a minimal stub engine whose breaker would record a ``reset`` call."""
    breaker = SimpleNamespace(
        state=MagicMock(),
        is_tripped=is_tripped,
        reset=MagicMock(),
    )
    return SimpleNamespace(circuit_breaker=breaker)


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/circuit-breaker
# ---------------------------------------------------------------------------


class TestGetCircuitBreakerState:
    def test_returns_404_when_run_not_in_registry(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/circuit-breaker")
        assert response.status_code == 404
        assert "No running engine" in response.json()["detail"]

    def test_returns_404_when_engine_has_no_circuit_breaker(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        # Engine registered but without a circuit_breaker attribute set
        stub_engines_registry[_RUN_ID] = SimpleNamespace(circuit_breaker=None)

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/circuit-breaker")
        assert response.status_code == 404
        assert "has no circuit breaker" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/v1/runs/{run_id}/circuit-breaker/reset  (retired: 410 Gone)
# ---------------------------------------------------------------------------


class TestResetCircuitBreaker:
    def test_returns_410_when_run_not_in_registry(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        # No breaker lookup at all: 410 even for an unknown run.
        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 410
        assert response.json()["detail"] == RESET_RETIRED_DETAIL

    def test_returns_410_when_breaker_not_tripped(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        stub_engines_registry[_RUN_ID] = _make_engine(is_tripped=False)

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 410
        assert "auto-recovers" in response.json()["detail"]

    def test_returns_410_and_never_invokes_reset_when_tripped(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        engine = _make_engine(is_tripped=True)
        stub_engines_registry[_RUN_ID] = engine

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 410
        engine.circuit_breaker.reset.assert_not_called()

    def test_reset_response_is_the_fixed_retirement_detail(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        stub_engines_registry[_RUN_ID] = _make_engine(is_tripped=True)

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 410
        assert response.json() == {"detail": RESET_RETIRED_DETAIL}
