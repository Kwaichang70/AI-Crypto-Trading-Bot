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

import time
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
# WP13a-C-02 (critic round 2, ST-19): the exit-config 422 contract on
# POST /api/v1/optimize, asserted against the real TestClient response
# body -- this is the layer that hid WP13a-C-01 (the optimizer's errors[]
# lacking combo_index/params) in round 1: a library-level unit test on
# validate_param_grid_exit_config alone would not have caught a bug in
# apps.api.routers.optimize's own envelope construction.
# ---------------------------------------------------------------------------

# Small lookback/trend so 300 synthetic bars comfortably cover warm-up.
_MOMENTUM_BODY: dict = {
    "strategyName": "momentum_breakout",
    "paramGrid": {"lookback": [5], "trend_sma_period": [5], "position_size": [100.0]},
    "symbols": [_SYMBOL],
    "timeframe": "1h",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-06-01T00:00:00Z",
    "rankBy": "sharpe_ratio",
    "topN": 10,
    "maxCombinations": 50,
}


@pytest.fixture()
def client_dev_large_body_limit(monkeypatch: pytest.MonkeyPatch):
    """WP1.3a round 6: a dev-mode TestClient with
    ``MAX_REQUEST_BODY_BYTES`` raised well above the default 1 MiB.

    The round-6 ``BodySizeLimitMiddleware`` now rejects any request over
    1 MiB before it ever reaches routing -- several PRE-EXISTING
    amplification-regression tests (rounds 4-5) deliberately send
    multi-megabyte bodies to exercise ``trading.exit_config``'s OWN
    per-field bounds specifically. This fixture raises the transport-layer
    cap for just those tests so they keep testing what they were written
    to test, while every other test in this module still exercises the
    real (smaller, default) transport-layer cap via ``client_dev``.
    """
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "104857600")  # 100 MiB

    from api.config import get_settings

    get_settings.cache_clear()
    from api.db.session import get_db
    from api.main import create_app
    from api.services import kill_switch as _kill_switch

    app = create_app()
    db_mock = _make_db_session_mock()

    async def _override():
        yield db_mock

    app.dependency_overrides[get_db] = _override
    with TestClient(app, raise_server_exceptions=False) as c:
        _kill_switch.reset_state_for_tests()
        yield c
    get_settings.cache_clear()


class TestOptimizeExitConfigWP13a:
    @pytest.fixture(autouse=True)
    def override_db(self, app_dev_mode: object) -> AsyncGenerator[None, None]:
        from api.db.session import get_db

        db_mock = _make_db_session_mock()

        async def _override() -> AsyncGenerator[AsyncMock, None]:
            yield db_mock

        app_dev_mode.dependency_overrides[get_db] = _override  # type: ignore[union-attr]
        yield
        app_dev_mode.dependency_overrides.pop(get_db, None)  # type: ignore[union-attr]

    def test_momentum_grid_no_bracket_keys_422_before_bar_fetch(
        self, client_dev: TestClient
    ) -> None:
        """A momentum_breakout grid with no bracket_*/trailing_stop_pct keys
        at all -> 422 exit_manager_required, and the bar fetch (a real
        exchange round trip) is never reached."""
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=_MOMENTUM_BODY)

        assert resp.status_code == 422, resp.text
        body = resp.json()["detail"]
        assert body["code"] == "exit_manager_required"
        assert body["total_invalid"] == 1
        assert len(body["errors"]) == 1
        issue = body["errors"][0]
        assert issue["combo_index"] == 0
        # Only bracket_*/trailing_stop_pct/allow_pyramiding grid dimensions
        # are tracked by the exit-config pre-check (SY-13a-17) -- none are
        # present in this grid, so the single synthetic "no bracket at
        # all" combination carries an empty params dict.
        assert issue["params"] == {}
        fetch_spy.assert_not_called()

    def test_one_invalid_combo_422_with_combo_index_and_params(
        self, client_dev: TestClient
    ) -> None:
        """A grid with one valid and one invalid (out-of-range) SL combo ->
        422, with the SECOND combination named by combo_index/params (not
        a composite 'combo[1].field' string, WP13a-C-01)."""
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                **_MOMENTUM_BODY["paramGrid"],
                "bracket_mode": ["fixed"],
                "bracket_stop_loss_pct": [0.05, 0.51],  # second combo out of range
                "bracket_take_profit_pct": [0.2],
            },
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert detail["total_invalid"] == 1
        assert len(detail["errors"]) == 1
        issue = detail["errors"][0]
        assert issue["combo_index"] == 1
        assert issue["params"]["bracket_stop_loss_pct"] == 0.51
        assert issue["field"] == "bracket_stop_loss_pct"
        assert issue["reason"] == "out_of_range"
        fetch_spy.assert_not_called()

    @pytest.mark.parametrize("unhashable_value", [[[0.05]], [{"a": 1}]])
    def test_unhashable_exit_key_grid_value_422_not_500(
        self, client_dev: TestClient, unhashable_value: list
    ) -> None:
        """WP13a-S-R3-02 (security round 3): a list/dict grid value on an
        exit key (e.g. a nested list or dict inside
        ``bracket_stop_loss_pct``'s grid values) must still return 422
        ``invalid_exit_config``/``invalid_type`` -- NOT an uncaught 500 from
        the dedupe cache trying to hash an unhashable value (AC3: no
        malformed exit input produces a 500 on optimize)."""
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                **_MOMENTUM_BODY["paramGrid"],
                "bracket_mode": ["fixed"],
                "bracket_stop_loss_pct": unhashable_value,
                "bracket_take_profit_pct": [0.2],
            },
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert len(detail["errors"]) == 1
        issue = detail["errors"][0]
        assert issue["field"] == "bracket_stop_loss_pct"
        assert issue["reason"] == "invalid_type"
        fetch_spy.assert_not_called()

    def test_empty_grid_value_list_400_before_cap_check(
        self, client_dev: TestClient
    ) -> None:
        """WP13a-S-R3-03 (security round 3, Info): an empty value list on
        any paramGrid key must be rejected with a clear 400 -- NOT a
        vacuous 200 with zero combinations run (an empty list makes the
        full cross-product 0, which trivially satisfies the
        max_combinations cap and would otherwise sail through to the bar
        fetch and an empty result)."""
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {**_MOMENTUM_BODY["paramGrid"], "position_size": []},
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 400, resp.text
        assert "position_size" in resp.json()["detail"]
        fetch_spy.assert_not_called()

    def test_large_unhashable_value_amplification_stays_fast(
        self, client_dev_large_body_limit: TestClient
    ) -> None:
        """WP13a-S-R4-03 (security round 4): a single ~7MB unhashable grid
        value (a 1,000,000-element list on an exit key), repeated across
        1000 full-grid positions via an unrelated non-exit dimension
        (``lookback``), must still return 422 ``invalid_type`` quickly --
        NOT block the event loop for ~43s the way an unbounded ``_trunc``
        plus an uncacheable-by-value dedupe cache did before this fix."""
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                "lookback": list(range(1000)),
                "bracket_stop_loss_pct": [list(range(1_000_000))],
            },
            "maxCombinations": 1000,
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            started = time.perf_counter()
            resp = client_dev_large_body_limit.post(_URL, json=body)
            elapsed = time.perf_counter() - started

        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert detail["errors"][0]["reason"] == "invalid_type"
        assert detail["errors"][0]["field"] == "bracket_stop_loss_pct"
        fetch_spy.assert_not_called()
        assert elapsed < 0.5, f"amplification case took {elapsed:.3f}s, expected < 0.5s"

    def test_large_invalid_string_amplification_response_stays_small_and_fast(
        self, client_dev_large_body_limit: TestClient
    ) -> None:
        """WP13a-S-R5-02 (security round 5): a single ~7MB invalid STRING
        grid value, repeated across 1000 full-grid positions via an
        unrelated non-exit dimension, must give a response body under
        50KB in well under 0.5s -- NOT echo the raw 7MB string into each
        of up to 20 issues (which previously produced a ~140MB response
        and ~0.95s)."""
        huge_string = "x" * 7_000_000
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                "lookback": list(range(1000)),
                "bracket_stop_loss_pct": [huge_string],
            },
            "maxCombinations": 1000,
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            started = time.perf_counter()
            resp = client_dev_large_body_limit.post(_URL, json=body)
            elapsed = time.perf_counter() - started

        assert resp.status_code == 422, resp.text
        response_bytes = len(resp.content)
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert detail["errors"][0]["reason"] == "not_a_number"
        assert len(detail["errors"][0]["params"]["bracket_stop_loss_pct"]) <= 64
        fetch_spy.assert_not_called()
        assert elapsed < 0.5, f"large-string case took {elapsed:.3f}s, expected < 0.5s"
        assert response_bytes < 50_000, (
            f"response was {response_bytes} bytes, expected < 50000"
        )

    def test_huge_int_gives_not_finite_not_500(self, client_dev: TestClient) -> None:
        """WP13a-S-R5-03 (security round 5, AC3): a 4000-digit int on an
        exit key must classify as a clean 422 ``not_finite`` -- NOT an
        uncaught ``OverflowError`` surfacing as a 500."""
        huge_int = 10**3999
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                **_MOMENTUM_BODY["paramGrid"],
                "bracket_mode": ["fixed"],
                "bracket_stop_loss_pct": [huge_int],
                "bracket_take_profit_pct": [0.2],
            },
        }
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert detail["errors"][0]["reason"] == "not_finite"
        fetch_spy.assert_not_called()

    def test_valid_bracket_grid_backtest_runner_receives_bracket_and_pyramiding(
        self, client_dev: TestClient
    ) -> None:
        """A valid bracket grid must reach BacktestRunner with bracket_config
        and allow_pyramiding populated (fixes A-07 -- previously only
        trailing_stop_pct was ever forwarded)."""
        from trading.backtest import BacktestRunner

        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                **_MOMENTUM_BODY["paramGrid"],
                "bracket_mode": ["fixed"],
                "bracket_stop_loss_pct": [0.05],
                "bracket_take_profit_pct": [0.2],
            },
        }
        runner_spy = MagicMock(wraps=BacktestRunner)
        with (
            patch("api.routers.optimize._fetch_bars_for_backtest", new=_mock_fetch_bars()),
            patch("trading.optimizer.BacktestRunner", new=runner_spy),
        ):
            resp = client_dev.post(_URL, json=body)

        assert resp.status_code == 200, resp.text
        runner_spy.assert_called_once()
        call_kwargs = runner_spy.call_args.kwargs
        assert call_kwargs["bracket_config"] == {
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.05,
            "bracket_take_profit_pct": 0.2,
        }
        # allow_pyramiding IS forwarded as a kwarg (previously A-07: it was
        # never forwarded at all) -- None here because the grid has no
        # allow_pyramiding dimension; BacktestRunner's own __init__ resolves
        # None to the strategy's default (momentum_breakout -> False).
        assert "allow_pyramiding" in call_kwargs
        assert call_kwargs["allow_pyramiding"] is None

    def test_exit_config_error_never_maps_to_400(self, client_dev: TestClient) -> None:
        """ExitConfigError is a ValueError subclass -- it must be caught
        BEFORE the generic ``except ValueError -> 400`` mapping (G-3)."""
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(_URL, json=_MOMENTUM_BODY)
        assert resp.status_code == 422
        assert resp.status_code != 400

    def test_oversized_grid_rejected_before_exit_config_validation(
        self, client_dev: TestClient
    ) -> None:
        """WP13a-S-R2-01: a grid whose full cross-product exceeds
        ``max_combinations`` must be rejected -- with the same 400 status
        and message ``ParameterOptimizer`` uses today -- BEFORE
        ``validate_param_grid_exit_config`` is ever called (it is O(grid
        size) per WP13a-C-01/C-02, and must not run unbounded on the event
        loop), and it must do so quickly."""
        body = {
            **_MOMENTUM_BODY,
            "paramGrid": {
                # 51 x 1 x 1 = 51 combinations > the default maxCombinations=50.
                "lookback": list(range(5, 56)),
                "trend_sma_period": [5],
                "position_size": [100.0],
            },
        }
        validate_spy = MagicMock(side_effect=AssertionError("must not be called"))
        fetch_spy = AsyncMock(side_effect=AssertionError("bar fetch must not be reached"))
        with (
            patch("api.routers.optimize.validate_param_grid_exit_config", new=validate_spy),
            patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy),
        ):
            started = time.perf_counter()
            resp = client_dev.post(_URL, json=body)
            elapsed = time.perf_counter() - started

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == (
            "Parameter grid produces 51 combinations, exceeding "
            "max_combinations=50. Reduce the grid or increase max_combinations."
        )
        validate_spy.assert_not_called()
        fetch_spy.assert_not_called()
        assert elapsed < 0.5, f"oversized-grid rejection took {elapsed:.3f}s, expected < 0.5s"


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
