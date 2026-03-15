"""
tests/integration/test_optimize_endpoint.py
--------------------------------------------
Integration tests for the POST /api/v1/optimize endpoint and the new
GET /api/v1/optimize and GET /api/v1/optimize/{id} endpoints.

Endpoint under test
-------------------
POST /api/v1/optimize
GET  /api/v1/optimize
GET  /api/v1/optimize/{id}

Design notes
------------
- The optimizer calls ``_fetch_bars_for_backtest`` (imported from runs.py)
  to fetch OHLCV data.  We patch it with synthetic bars so the tests
  run without a live exchange connection.
- Each BacktestRunner run is deterministic (seed=42 is set inside
  ParameterOptimizer).
- Auth is disabled via ``client_dev`` fixture (REQUIRE_API_AUTH=false).
- JSON response keys are camelCase (alias_generator=to_camel).
- The ``run_optimization`` endpoint injects ``db: AsyncSession = Depends(get_db)``.
  All test classes use an ``override_db`` autouse fixture that replaces ``get_db``
  with a mock AsyncSession so no real PostgreSQL connection is required.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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
# Helper: build a mock db session for optimize endpoint tests
# ---------------------------------------------------------------------------

def _make_db_session_mock() -> AsyncMock:
    """
    Build an AsyncMock that satisfies the SQLAlchemy AsyncSession interface
    used by the optimize endpoints.

    Covers:
    - add()         — synchronous, no-op
    - flush()       — awaited (assign PKs within transaction)
    - commit()      — awaited (persist to DB)
    - execute()     — awaited, returns a mock result with .scalars().all() = []
    """
    session = AsyncMock()

    # Synchronous operations (SQLAlchemy does not await these)
    session.add = MagicMock()

    # Async operations
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    # execute() returns an object whose .scalars().all() returns [] by default.
    # Individual tests may override session.execute.return_value to simulate data.
    scalars_result = MagicMock()
    scalars_result.all.return_value = []
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_result
    execute_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=execute_result)

    return session


def _mock_fetch_bars() -> AsyncMock:
    """Return an AsyncMock that yields synthetic bars without hitting exchange."""
    return AsyncMock(return_value=_BARS_BY_SYMBOL)


# ---------------------------------------------------------------------------
# TestOptimizeEndpoint — happy-path and error-path tests for POST
# ---------------------------------------------------------------------------


class TestOptimizeEndpoint:
    """Happy-path and error-path tests for POST /api/v1/optimize."""

    @pytest.fixture(autouse=True)
    def override_db(self, app_dev_mode: object) -> AsyncGenerator[None, None]:
        """
        Override the get_db dependency for every test in this class.

        Replaces the real AsyncSession (which would attempt a PostgreSQL
        connection) with a mock session so the optimizer logic can reach
        db.add / db.flush / db.commit without a live database.
        """
        from api.db.session import get_db

        db_mock = _make_db_session_mock()

        async def _override() -> AsyncGenerator[AsyncMock, None]:
            yield db_mock

        app_dev_mode.dependency_overrides[get_db] = _override  # type: ignore[union-attr]
        yield
        app_dev_mode.dependency_overrides.pop(get_db, None)  # type: ignore[union-attr]

    def test_happy_path_returns_ranked_entries(self, client_dev: TestClient) -> None:
        """
        A valid request with a 4-combination MA grid (2x2) must return HTTP 200
        with completedCombinations=4, entries ranked by sharpe_ratio, and a
        non-None optimizationRunId UUID.
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

        # optimizationRunId must be present and parseable as UUID
        assert "optimizationRunId" in body
        assert body["optimizationRunId"] is not None
        # Validate it is a well-formed UUID string
        from uuid import UUID
        UUID(body["optimizationRunId"])  # raises ValueError if malformed

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
            # 6 x 6 = 36 > maxCombinations=10
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


# ---------------------------------------------------------------------------
# TestListOptimizationRuns — GET /api/v1/optimize
# ---------------------------------------------------------------------------


class TestListOptimizationRuns:
    """Tests for GET /api/v1/optimize."""

    @pytest.fixture(autouse=True)
    def override_db(self, app_dev_mode: object) -> AsyncGenerator[None, None]:
        """Override get_db with empty-result mock for list endpoint tests."""
        from api.db.session import get_db

        db_mock = _make_db_session_mock()

        async def _override() -> AsyncGenerator[AsyncMock, None]:
            yield db_mock

        app_dev_mode.dependency_overrides[get_db] = _override  # type: ignore[union-attr]
        yield
        app_dev_mode.dependency_overrides.pop(get_db, None)  # type: ignore[union-attr]

    def test_list_optimization_runs_empty_returns_200(
        self, client_dev: TestClient
    ) -> None:
        """
        GET /api/v1/optimize with no persisted runs must return HTTP 200
        with an empty JSON array.
        """
        resp = client_dev.get(_URL)

        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# TestGetOptimizationRun — GET /api/v1/optimize/{id}
# ---------------------------------------------------------------------------


class TestGetOptimizationRun:
    """Tests for GET /api/v1/optimize/{optimization_run_id}."""

    @pytest.fixture(autouse=True)
    def override_db(self, app_dev_mode: object) -> AsyncGenerator[None, None]:
        """Override get_db with not-found mock for detail endpoint tests."""
        from api.db.session import get_db

        db_mock = _make_db_session_mock()
        # scalar_one_or_none() returns None by default (set in _make_db_session_mock)
        # so the endpoint will return 404 for any UUID.

        async def _override() -> AsyncGenerator[AsyncMock, None]:
            yield db_mock

        app_dev_mode.dependency_overrides[get_db] = _override  # type: ignore[union-attr]
        yield
        app_dev_mode.dependency_overrides.pop(get_db, None)  # type: ignore[union-attr]

    def test_get_optimization_run_not_found_returns_404(
        self, client_dev: TestClient
    ) -> None:
        """
        GET /api/v1/optimize/{unknown_uuid} must return HTTP 404.
        The mock db returns None from scalar_one_or_none(), simulating a missing row.
        """
        unknown_id = uuid4()
        resp = client_dev.get(f"{_URL}/{unknown_id}")

        assert resp.status_code == 404
        assert str(unknown_id) in resp.json()["detail"]
