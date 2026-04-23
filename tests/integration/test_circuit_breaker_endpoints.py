"""
tests/integration/test_circuit_breaker_endpoints.py
----------------------------------------------------
Integration coverage for apps/api/routers/circuit_breaker.py (Sprint 41 TO-001).

Endpoints under test
--------------------
- GET  /api/v1/runs/{run_id}/circuit-breaker       — current state
- POST /api/v1/runs/{run_id}/circuit-breaker/reset — reset tripped breaker

The router reads the live engine from ``_RUN_ENGINES`` (currently still in
:mod:`api.routers.runs` — Sprint 41 does not migrate this; see Sprint 40
Stap 1d deferred scope).  Tests register a lightweight stub engine and
assert HTTP-level behaviour through ``TestClient``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


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


def _make_breaker_state(
    *,
    is_tripped: bool = False,
    response_level: str = "OK",
) -> Any:
    """Build a stub state object exposing the fields the router serialises."""
    state = MagicMock()
    state.model_dump = MagicMock(
        return_value={
            "is_tripped": is_tripped,
            "response_level": response_level,
            "daily_pnl_pct": 0.0,
            "consecutive_losses": 0,
        }
    )
    return state


def _make_engine(*, is_tripped: bool = False, response_level: str = "OK") -> Any:
    """Build a minimal stub engine with a ``circuit_breaker`` attribute."""
    breaker = SimpleNamespace(
        state=_make_breaker_state(is_tripped=is_tripped, response_level=response_level),
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

    def test_returns_200_with_state_when_breaker_configured(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        stub_engines_registry[_RUN_ID] = _make_engine(
            is_tripped=False, response_level="REDUCE"
        )

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/circuit-breaker")
        assert response.status_code == 200
        body = response.json()
        assert body["is_tripped"] is False
        assert body["response_level"] == "REDUCE"


# ---------------------------------------------------------------------------
# POST /api/v1/runs/{run_id}/circuit-breaker/reset
# ---------------------------------------------------------------------------


class TestResetCircuitBreaker:
    def test_returns_404_when_run_not_in_registry(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 404

    def test_returns_409_when_breaker_not_tripped(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        stub_engines_registry[_RUN_ID] = _make_engine(is_tripped=False)

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 409
        assert "not tripped" in response.json()["detail"]

    def test_returns_200_and_invokes_reset_when_tripped(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        engine = _make_engine(is_tripped=True, response_level="HALT")
        stub_engines_registry[_RUN_ID] = engine

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 200
        # The stub engine's reset() must have been called exactly once.
        engine.circuit_breaker.reset.assert_called_once()

    def test_reset_response_serialises_breaker_state(
        self, client_dev: TestClient, stub_engines_registry: dict[str, Any]
    ) -> None:
        stub_engines_registry[_RUN_ID] = _make_engine(
            is_tripped=True, response_level="HALT"
        )

        response = client_dev.post(f"/api/v1/runs/{_RUN_ID}/circuit-breaker/reset")
        assert response.status_code == 200
        body = response.json()
        # Router returns the post-reset state snapshot — fields must be present
        assert "is_tripped" in body
        assert "response_level" in body
