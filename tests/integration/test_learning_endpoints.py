"""
tests/integration/test_learning_endpoints.py
---------------------------------------------
Integration coverage for apps/api/routers/learning.py (Sprint 41 TO-001).

Endpoint under test
-------------------
- GET /api/v1/runs/{run_id}/learning — adaptive learning state for a
  running engine (returns 404 when no ``AdaptiveLearningTask`` is
  registered in ``_LEARNING_INSTANCES`` for the run).

The router reads the learner from ``_LEARNING_INSTANCES`` on
``api.routers.runs``; Sprint 41 does not migrate this to ``AppContainer``
(deferred to Sprint 41's RunRegistry migration sub-step).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient


_RUN_ID = "le91a050-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_learning_registry() -> Generator[dict[str, Any], None, None]:
    from api.routers.runs import _LEARNING_INSTANCES

    _LEARNING_INSTANCES.clear()
    yield _LEARNING_INSTANCES
    _LEARNING_INSTANCES.clear()


def _make_learner(
    *,
    cycle_count: int = 3,
    trades_ingested: int = 42,
    skipped_ingested: int = 7,
    trades_at_last_cycle: int = 40,
    min_trades_per_cycle: int = 50,
    is_enabled: bool = True,
    rollback_count_30d: int = 0,
    cooldown_until: Any = None,
    disabled_reason: str | None = None,
    pre_adjustment_pnl_pct: float | None = None,
    last_adjustment: Any = None,
    last_analysis: Any = None,
    auto_apply: bool = False,
) -> Any:
    """Build a stub ``AdaptiveLearningTask`` exposing the attributes read
    by :func:`api.routers.learning.get_learning_state`."""
    optimizer_state = SimpleNamespace(
        is_enabled=is_enabled,
        rollback_count_30d=rollback_count_30d,
        cooldown_until=cooldown_until,
        disabled_reason=disabled_reason,
        pre_adjustment_pnl_pct=pre_adjustment_pnl_pct,
        last_adjustment=last_adjustment,
    )
    optimizer = SimpleNamespace(state=optimizer_state)

    return SimpleNamespace(
        optimizer=optimizer,
        cycle_count=cycle_count,
        _all_trades=list(range(trades_ingested)),
        _all_skipped=list(range(skipped_ingested)),
        _trades_at_last_cycle=trades_at_last_cycle,
        _min_trades_per_cycle=min_trades_per_cycle,
        last_analysis=last_analysis,
        _auto_apply=auto_apply,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/learning
# ---------------------------------------------------------------------------


class TestGetLearningState:
    def test_returns_404_when_run_not_in_registry(
        self,
        client_dev: TestClient,
        stub_learning_registry: dict[str, Any],
    ) -> None:
        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/learning")
        assert response.status_code == 404
        assert "No active adaptive learning task" in response.json()["detail"]

    def test_returns_200_with_minimal_state(
        self,
        client_dev: TestClient,
        stub_learning_registry: dict[str, Any],
    ) -> None:
        """Happy path when no last_adjustment / last_analysis is present."""
        stub_learning_registry[_RUN_ID] = _make_learner()

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/learning")
        assert response.status_code == 200
        body = response.json()

        assert body["enabled"] is True
        assert body["autoApply"] is False
        assert body["cycleCount"] == 3
        assert body["tradesIngested"] == 42
        assert body["skippedIngested"] == 7
        assert body["tradesAtLastCycle"] == 40
        assert body["minTradesPerCycle"] == 50
        assert body["optimizerState"]["isEnabled"] is True
        assert body["optimizerState"]["rollbackCount30d"] == 0
        assert body["optimizerState"]["cooldownUntil"] is None
        assert body["lastAdjustment"] is None
        assert body["lastAnalysis"] is None

    def test_returns_200_with_last_adjustment_payload(
        self,
        client_dev: TestClient,
        stub_learning_registry: dict[str, Any],
    ) -> None:
        change = SimpleNamespace(
            param_name="rsi_period",
            old_value=14,
            new_value=17,
            change_pct=0.214,
        )
        adjustment = SimpleNamespace(
            actionable=True,
            confidence=0.78,
            reason="higher-win-rate-in-bucket",
            changes=[change],
        )
        stub_learning_registry[_RUN_ID] = _make_learner(
            last_adjustment=adjustment,
        )

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/learning")
        assert response.status_code == 200
        body = response.json()

        assert body["lastAdjustment"] is not None
        assert body["lastAdjustment"]["actionable"] is True
        assert body["lastAdjustment"]["confidence"] == pytest.approx(0.78)
        assert body["lastAdjustment"]["reason"] == "higher-win-rate-in-bucket"
        assert len(body["lastAdjustment"]["changes"]) == 1
        change_body = body["lastAdjustment"]["changes"][0]
        assert change_body["paramName"] == "rsi_period"
        assert change_body["oldValue"] == 14
        assert change_body["newValue"] == 17

    def test_returns_200_with_last_analysis_and_regime_fields(
        self,
        client_dev: TestClient,
        stub_learning_registry: dict[str, Any],
    ) -> None:
        regimes = SimpleNamespace(
            by_regime={"bull": 1.0, "bear": -0.5},
            best_regime="bull",
            worst_regime="bear",
        )
        indicators = SimpleNamespace(most_predictive="rsi_14")
        analysis = SimpleNamespace(
            confidence=0.9,
            is_actionable=True,
            total_trades=120,
            total_skipped=5,
            regimes=regimes,
            indicators=indicators,
        )
        stub_learning_registry[_RUN_ID] = _make_learner(last_analysis=analysis)

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/learning")
        assert response.status_code == 200
        body = response.json()

        assert body["lastAnalysis"] is not None
        assert body["lastAnalysis"]["confidence"] == pytest.approx(0.9)
        assert body["lastAnalysis"]["isActionable"] is True
        assert body["lastAnalysis"]["totalTrades"] == 120
        assert body["lastAnalysis"]["totalSkipped"] == 5
        assert body["lastAnalysis"]["bestRegime"] == "bull"
        assert body["lastAnalysis"]["worstRegime"] == "bear"
        assert body["lastAnalysis"]["mostPredictiveIndicator"] == "rsi_14"

    def test_returns_200_auto_apply_flag_reflects_learner_state(
        self,
        client_dev: TestClient,
        stub_learning_registry: dict[str, Any],
    ) -> None:
        stub_learning_registry[_RUN_ID] = _make_learner(auto_apply=True)

        response = client_dev.get(f"/api/v1/runs/{_RUN_ID}/learning")
        assert response.status_code == 200
        assert response.json()["autoApply"] is True
