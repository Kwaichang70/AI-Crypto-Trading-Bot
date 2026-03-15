"""
tests/integration/test_optimize_endpoint.py
--------------------------------------------
Integration tests for the POST /api/v1/optimize endpoint.

Endpoint under test
-------------------
POST /api/v1/optimize

Design notes
------------
- The optimizer calls ``_fetch_bars_for_backtest`` (imported from runs.py)
  to fetch OHLCV data.  We patch it with synthetic bars so the tests
  run without a live exchange connection.
- Each BacktestRunner run is deterministic (seed=42 is set inside
  ParameterOptimizer).
- Auth is disabled via ``client_dev`` fixture (REQUIRE_API_AUTH=false).
- JSON response keys are camelCase (alias_generator=to_camel).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_bars
from common.types import TimeFrame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_URL = "/api/v1/optimize"

_SYMBOL = "BTC/USD"
_TF = TimeFrame.ONE_HOUR

# Provide enough bars to satisfy warmup for fast_period=5, slow_period=20
_BARS = make_bars(300, symbol=_SYMBOL, timeframe=_TF)
_BARS_BY_SYMBOL: dict[str, list] = {_SYMBOL: _BARS}

_VALID_BODY: dict = {
    "strategyName": "ma_crossover",
    "paramGrid": {"fast_period": [5, 10], "slow_period": [20, 30]},
    "symbols": [_SYMBOL],
    "timeframe": "1h",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-06-01T00:00:00Z",
    "rankBy": "sharpe_ratio",
    "topN": 10,
    "maxCombinations": 50,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_fetch_bars():
    """Return an AsyncMock that yields synthetic bars without hitting exchange."""
    return AsyncMock(return_value=_BARS_BY_SYMBOL)


# ---------------------------------------------------------------------------
# TestOptimizeEndpoint
# ---------------------------------------------------------------------------


class TestOptimizeEndpoint:
    """Happy-path and error-path tests for POST /api/v1/optimize."""

    def test_happy_path_returns_ranked_entries(self, client_dev: TestClient) -> None:
        """
        A valid request with a 4-combination MA grid (2×2) must return HTTP 200
        with completedCombinations=4 and entries ranked by sharpe_ratio.
        """
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest",
            new=_mock_fetch_bars(),
        ):
            resp = client_dev.post(_URL, json=_VALID_BODY)

        assert resp.status_code == 200
        body = resp.json()

        assert body["totalCombinations"] == 4
        assert body["completedCombinations"] == 4
        assert body["failedCombinations"] == 0
        assert body["rankBy"] == "sharpe_ratio"
        assert len(body["entries"]) == 4

        # Entries must be in descending sharpe order
        sharpes = [e["metrics"]["sharpe_ratio"] for e in body["entries"]]
        assert sharpes == sorted(sharpes, reverse=True)

        # Every entry must have rank and params
        for i, entry in enumerate(body["entries"]):
            assert entry["rank"] == i + 1
            assert "fast_period" in entry["params"]
            assert "slow_period" in entry["params"]

    def test_top_n_limits_entries(self, client_dev: TestClient) -> None:
        """topN=2 on a 4-combination grid must return exactly 2 entries."""
        body = {**_VALID_BODY, "topN": 2}
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest",
            new=_mock_fetch_bars(),
        ):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 200
        assert len(resp.json()["entries"]) == 2

    def test_unknown_strategy_returns_400(self, client_dev: TestClient) -> None:
        """An unrecognised strategy name must return HTTP 400."""
        body = {**_VALID_BODY, "strategyName": "nonexistent_strategy"}
        resp = client_dev.post(_URL, json=body)
        assert resp.status_code == 400
        assert "Unknown strategy" in resp.json()["detail"]

    def test_bad_rank_by_returns_400(self, client_dev: TestClient) -> None:
        """An unsupported rankBy metric must return HTTP 400."""
        body = {**_VALID_BODY, "rankBy": "not_a_metric"}
        resp = client_dev.post(_URL, json=body)
        assert resp.status_code == 400
        assert "Unsupported rank_by" in resp.json()["detail"]

    def test_start_after_end_returns_400(self, client_dev: TestClient) -> None:
        """backtestStart >= backtestEnd must return HTTP 400."""
        body = {
            **_VALID_BODY,
            "backtestStart": "2024-06-01T00:00:00Z",
            "backtestEnd": "2024-01-01T00:00:00Z",
        }
        resp = client_dev.post(_URL, json=body)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "backtest_start must be before backtest_end"

    def test_grid_exceeds_max_combinations_returns_400(
        self, client_dev: TestClient
    ) -> None:
        """A grid producing more combinations than maxCombinations must return 400."""
        body = {
            **_VALID_BODY,
            # 6 × 6 = 36 > maxCombinations=10
            "paramGrid": {
                "fast_period": [5, 10, 15, 20, 25, 30],
                "slow_period": [40, 50, 60, 70, 80, 90],
            },
            "maxCombinations": 10,
        }
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest",
            new=_mock_fetch_bars(),
        ):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 400
        assert "exceeding max_combinations" in resp.json()["detail"]
