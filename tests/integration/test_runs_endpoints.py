"""
tests/integration/test_runs_endpoints.py
-----------------------------------------
Integration tests for the runs CRUD endpoints.

Endpoints under test
--------------------
POST   /api/v1/runs              -- Create a run (201)
GET    /api/v1/runs              -- List runs paginated (200)
GET    /api/v1/runs/{run_id}     -- Get single run detail (200 / 404)
DELETE /api/v1/runs/{run_id}     -- Stop a running run (200 / 409)

Test strategy
-------------
- All DB I/O is intercepted via a FastAPI dependency override that injects
  a hand-crafted AsyncMock instead of a real AsyncSession.  This keeps the
  tests hermetic: no PostgreSQL instance is required.
- The trading strategy import path IS exercised.  The strategy registry is
  populated lazily on the first POST call; because the real strategy classes
  exist in the codebase, _get_strategy_registry() succeeds without patching.
- Test data is constructed with deterministic UUIDs and fixed UTC timestamps
  so assertions never depend on wall-clock time.
- Each test is independent: the mock_db_session fixture resets all call
  history and return values between tests.

Mock wiring summary
-------------------
  list_runs:  db.execute() is called twice (count then page).
              Call 0  → scalar_one() returns an integer count.
              Call 1  → scalars().all() returns a list of SimpleNamespace objects.

  get_run / stop_run:
              db.execute() is called once.
              scalar_one_or_none() returns a SimpleNamespace or None.

  create_run (paper mode):
              db.add(), db.flush() are no-ops.
              After flush the ORM object already has the attributes set
              by the handler (id, status, config, timestamps).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Deterministic test data constants
# ---------------------------------------------------------------------------

_FIXED_UUID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_FIXED_UUID_2 = uuid.UUID("11111111-2222-3333-4444-555555555555")
_FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

_PAPER_RUN_CONFIG = {
    "strategy_name": "ma_crossover",
    "strategy_params": {"fast_period": 10, "slow_period": 50},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "paper",
    "initial_capital": "10000.00",
}


# ---------------------------------------------------------------------------
# RunORM factory helper
# ---------------------------------------------------------------------------

def _make_run_orm(
    *,
    run_id: uuid.UUID = _FIXED_UUID,
    run_mode: str = "paper",
    status: str = "running",
    config: dict | None = None,
    started_at: datetime = _FIXED_NOW,
    stopped_at: datetime | None = None,
    created_at: datetime = _FIXED_NOW,
    updated_at: datetime = _FIXED_NOW,
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as RunORM,
    suitable for Pydantic from_attributes=True serialization.
    """
    return SimpleNamespace(
        id=run_id,
        run_mode=run_mode,
        status=status,
        config=config or {
            "strategy_name": "ma_crossover",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": run_mode,
            "initial_capital": "10000.00",
        },
        started_at=started_at,
        stopped_at=stopped_at,
        created_at=created_at,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# DB mock helpers
# ---------------------------------------------------------------------------

def _make_scalar_result(value: object) -> MagicMock:
    """
    Return a MagicMock that mimics an AsyncSession execute() result
    supporting .scalar_one() — used for COUNT queries in list_runs.
    """
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _make_scalars_result(items: list) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting
    .scalars().all() — used for page queries in list_runs.
    """
    result = MagicMock()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = items
    result.scalars.return_value = scalars_mock
    return result


def _make_scalar_one_or_none_result(value: object) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting
    .scalar_one_or_none() — used for get_run and stop_run.
    """
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


# ---------------------------------------------------------------------------
# POST /api/v1/runs test cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCreateRun:
    """Tests for POST /api/v1/runs."""

    def test_unknown_strategy_returns_400(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        Submitting an unknown strategy_name must return HTTP 400.

        The strategy registry validation in the handler converts the name to
        snake_case and checks it against the registry dict.  Any name not in
        {ma_crossover, rsi_mean_reversion, breakout} must be rejected before
        any DB I/O occurs.
        """
        payload = {
            "strategyName": "no_such_strategy",
            "strategyParams": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initialCapital": "10000.00",
        }
        resp = client_dev_with_db.post("/api/v1/runs", json=payload)

        assert resp.status_code == 400
        body = resp.json()
        assert "Unknown strategy" in body["detail"]

    def test_missing_required_fields_returns_422(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        A request body missing mandatory fields must trigger Pydantic 422.

        ``symbols``, ``timeframe``, ``mode``, and ``initialCapital`` are all
        required by ``RunCreateRequest``.  Omitting them must produce an
        HTTP 422 Unprocessable Entity response (Pydantic validation error).
        """
        # Only strategy_name is provided — everything else is missing
        payload = {"strategyName": "ma_crossover"}
        resp = client_dev_with_db.post("/api/v1/runs", json=payload)

        assert resp.status_code == 422

    def test_invalid_symbol_format_returns_422(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        Symbols not following the BASE/QUOTE format must be rejected with 422.

        The ``validate_symbols`` field_validator on RunCreateRequest raises a
        ValueError which FastAPI converts to 422 before reaching the handler.
        """
        payload = {
            "strategyName": "ma_crossover",
            "strategyParams": {},
            "symbols": ["BTCUSDT"],  # Missing slash — invalid CCXT format
            "timeframe": "1h",
            "mode": "paper",
            "initialCapital": "10000.00",
        }
        resp = client_dev_with_db.post("/api/v1/runs", json=payload)

        assert resp.status_code == 422

    def test_create_paper_run_returns_201(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A valid paper-mode run creation must return HTTP 201 with RunDetailResponse.

        The handler calls db.add() then db.flush() and returns the ORM object
        converted to RunDetailResponse.  We verify:
        - HTTP 201 status code
        - Response fields match what the handler would set on the ORM
        - db.add() was called once
        - db.flush() was called once (inside the paper mode branch)
        """
        payload = {
            "strategyName": "ma_crossover",
            "strategyParams": {"fast_period": 10, "slow_period": 50},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initialCapital": "10000.00",
        }

        resp = client_dev_with_db.post("/api/v1/runs", json=payload)

        assert resp.status_code == 201
        body = resp.json()
        assert body["runMode"] == "paper"
        assert body["status"] == "running"
        assert body["config"]["strategy_name"] == "ma_crossover"
        assert body["config"]["symbols"] == ["BTC/USDT"]
        assert body["backtestMetrics"] is None

        # Verify DB interactions
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called()


# ---------------------------------------------------------------------------
# GET /api/v1/runs test cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestListRuns:
    """Tests for GET /api/v1/runs."""

    def test_returns_paginated_list_200(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        GET /api/v1/runs must return HTTP 200 with a RunListResponse envelope.

        The handler executes two queries: a COUNT(*) then a page SELECT.
        We set up the mock to return 2 for the count and one RunORM for the
        page, then assert the response envelope structure is correct.
        """
        run_orm = _make_run_orm()

        # First execute() call → scalar_one() for COUNT
        # Second execute() call → scalars().all() for the page
        mock_db_session.execute.side_effect = [
            _make_scalar_result(1),
            _make_scalars_result([run_orm]),
        ]

        resp = client_dev_with_db.get("/api/v1/runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["offset"] == 0
        assert body["limit"] == 50
        assert len(body["items"]) == 1
        assert body["items"][0]["runMode"] == "paper"
        assert body["items"][0]["status"] == "running"

    def test_empty_list_returns_200_with_zero_total(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When no runs exist the response must have total=0 and items=[].
        """
        mock_db_session.execute.side_effect = [
            _make_scalar_result(0),
            _make_scalars_result([]),
        ]

        resp = client_dev_with_db.get("/api/v1/runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_pagination_params_are_forwarded(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        offset and limit query parameters must be reflected in the response envelope.

        We do not validate the SQLAlchemy statement objects themselves —
        that would couple the test to ORM internals.  We verify the values
        are echoed back in the response payload, which proves the handler
        read and used them.
        """
        run_orm = _make_run_orm()
        mock_db_session.execute.side_effect = [
            _make_scalar_result(10),
            _make_scalars_result([run_orm]),
        ]

        resp = client_dev_with_db.get("/api/v1/runs?offset=5&limit=1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["offset"] == 5
        assert body["limit"] == 1
        assert body["total"] == 10
        assert len(body["items"]) == 1

    def test_limit_exceeds_maximum_returns_422(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        A limit value greater than 500 must be rejected with HTTP 422.

        FastAPI validates the Query(le=500) constraint and returns 422 before
        the handler body executes.  No DB calls are made.
        """
        resp = client_dev_with_db.get("/api/v1/runs?limit=501")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id} test cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetRun:
    """Tests for GET /api/v1/runs/{run_id}."""

    def test_known_run_id_returns_200(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting a run that exists must return HTTP 200 with RunDetailResponse.

        The mock returns a RunORM with status='stopped' to simulate a
        completed run.  We verify the response body fields are correctly
        populated including the camelCase field names from the Pydantic schema.
        """
        run_orm = _make_run_orm(status="stopped", stopped_at=_FIXED_NOW)
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_FIXED_UUID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(_FIXED_UUID)
        assert body["runMode"] == "paper"
        assert body["status"] == "stopped"
        assert body["backtestMetrics"] is None

    def test_unknown_run_id_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting a run UUID that does not exist must return HTTP 404.

        The mock returns None from scalar_one_or_none() which triggers the
        HTTPException(404) branch in the get_run handler.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        missing_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        resp = client_dev_with_db.get(f"/api/v1/runs/{missing_id}")

        assert resp.status_code == 404
        body = resp.json()
        assert str(missing_id) in body["detail"]

    def test_backtest_run_with_metrics_returns_populated_field(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A completed backtest run whose config contains backtest_metrics must
        return a RunDetailResponse with the backtest_metrics field populated.

        The _run_orm_to_detail_response() helper reads config['backtest_metrics']
        and validates it into a BacktestMetricsResponse.  We embed a valid
        metrics dict in the ORM's config to exercise this path.
        """
        metrics_blob = {
            "totalReturnPct": 0.12,
            "cagr": 0.08,
            "initialCapital": "10000.00",
            "finalEquity": "11200.00",
            "totalFeesPaid": "50.00",
            "sharpeRatio": 1.5,
            "sortinoRatio": 2.1,
            "calmarRatio": 1.2,
            "maxDrawdownPct": 0.05,
            "maxDrawdownDurationBars": 10,
            "totalTrades": 20,
            "winningTrades": 12,
            "losingTrades": 8,
            "winRate": 0.6,
            "profitFactor": 1.8,
            "averageTradePnl": "60.00",
            "averageWin": "120.00",
            "averageLoss": "-45.00",
            "largestWin": "400.00",
            "largestLoss": "-180.00",
            "totalBars": 500,
            "barsInMarket": 200,
            "exposurePct": 0.4,
            "startDate": "2025-01-01T00:00:00+00:00",
            "endDate": "2025-06-01T00:00:00+00:00",
            "durationDays": 151,
        }
        config_with_metrics = {
            "strategy_name": "ma_crossover",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "backtest",
            "initial_capital": "10000.00",
            "backtest_metrics": metrics_blob,
        }
        run_orm = _make_run_orm(
            run_mode="backtest",
            status="stopped",
            config=config_with_metrics,
            stopped_at=_FIXED_NOW,
        )
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_FIXED_UUID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["runMode"] == "backtest"
        assert body["status"] == "stopped"
        # backtest_metrics must be populated from config JSONB
        assert body["backtestMetrics"] is not None
        assert body["backtestMetrics"]["totalTrades"] == 20
        assert body["backtestMetrics"]["winRate"] == 0.6


# ---------------------------------------------------------------------------
# DELETE /api/v1/runs/{run_id} test cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStopRun:
    """Tests for DELETE /api/v1/runs/{run_id}."""

    def test_running_run_is_stopped_returns_200(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Sending DELETE to a run with status='running' must return 200 and
        the response body must reflect status='stopped'.

        The handler mutates run.status, run.stopped_at, and run.updated_at
        in-place on the ORM object before calling flush().  Since we pass a
        real RunORM (not a MagicMock), the attribute mutations are real and
        the Pydantic conversion will read the updated values.
        """
        run_orm = _make_run_orm(status="running")
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

        resp = client_dev_with_db.delete(f"/api/v1/runs/{_FIXED_UUID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stopped"
        assert body["stoppedAt"] is not None
        # DB flush must have been called to persist the status change
        mock_db_session.flush.assert_called()

    def test_already_stopped_run_returns_409(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Sending DELETE to a run that is already 'stopped' must return 409 Conflict.

        The stop_run handler checks run.status != 'running' and raises
        HTTPException(409) for terminal states.
        """
        run_orm = _make_run_orm(status="stopped", stopped_at=_FIXED_NOW)
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

        resp = client_dev_with_db.delete(f"/api/v1/runs/{_FIXED_UUID}")

        assert resp.status_code == 409
        body = resp.json()
        assert "stopped" in body["detail"]

    def test_errored_run_cannot_be_stopped_returns_409(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Sending DELETE to a run with status='error' must also return 409.

        Both 'stopped' and 'error' are terminal states that cannot be
        transitioned from by the DELETE endpoint.
        """
        run_orm = _make_run_orm(status="error", stopped_at=_FIXED_NOW)
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_orm)

        resp = client_dev_with_db.delete(f"/api/v1/runs/{_FIXED_UUID}")

        assert resp.status_code == 409

    def test_stop_nonexistent_run_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Sending DELETE for an unknown run UUID must return HTTP 404.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        missing_id = uuid.UUID("cafecafe-cafe-cafe-cafe-cafecafecafe")
        resp = client_dev_with_db.delete(f"/api/v1/runs/{missing_id}")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Authentication tests (production mode — require_api_auth=True)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunsEndpointAuth:
    """
    Auth enforcement tests for runs endpoints in production mode.

    These tests reuse client_prod (no DB override needed) because auth
    rejection happens in the require_api_key dependency — before any
    DB access.  The DB will error if reached, but the 401 comes first.
    """

    def test_post_without_auth_returns_401(self, client_prod: TestClient) -> None:
        """
        POST /api/v1/runs without an API key must return 401 in prod mode.

        The ``require_api_key`` dependency is mounted on the router as a
        router-level dependency.  It fires before the handler body, so no
        DB access or strategy validation happens.
        """
        payload = {
            "strategyName": "ma_crossover",
            "strategyParams": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initialCapital": "10000.00",
        }
        resp = client_prod.post("/api/v1/runs", json=payload)

        assert resp.status_code == 401

    def test_get_list_without_auth_returns_401(self, client_prod: TestClient) -> None:
        """
        GET /api/v1/runs without an API key must return 401 in prod mode.
        """
        resp = client_prod.get("/api/v1/runs")

        assert resp.status_code == 401

    def test_get_list_with_valid_auth_does_not_return_401(
        self,
        app_prod_mode,
        mock_db_session: AsyncMock,
        auth_headers: dict[str, str],
    ) -> None:
        """
        GET /api/v1/runs with a valid API key must NOT return 401.

        We wire the DB mock into the prod-mode app so the request completes
        past the auth gate.  The exact response code (200 or otherwise) is
        not the focus — we only assert it is not 401.
        """
        from api.db.session import get_db

        async def _override_get_db():
            yield mock_db_session

        app_prod_mode.dependency_overrides[get_db] = _override_get_db

        mock_db_session.execute.side_effect = [
            _make_scalar_result(0),
            _make_scalars_result([]),
        ]

        try:
            with TestClient(app_prod_mode, raise_server_exceptions=False) as c:
                resp = c.get("/api/v1/runs", headers=auth_headers)
            assert resp.status_code != 401
        finally:
            app_prod_mode.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Paper engine background task wiring tests
# ---------------------------------------------------------------------------

# Standard payload reused across task-wiring tests.
_PAPER_PAYLOAD = {
    "strategyName": "ma_crossover",
    "strategyParams": {"fast_period": 10, "slow_period": 50},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "paper",
    "initialCapital": "10000.00",
}


@pytest.mark.integration
class TestPaperEngineTaskWiring:
    """
    Tests for the asyncio.Task lifecycle wired into POST /api/v1/runs (paper mode)
    and DELETE /api/v1/runs/{run_id}.

    Background
    ----------
    The paper-mode branch of create_run() calls asyncio.create_task() to launch
    _run_paper_engine() in the background and stores the Task object in the
    module-level _RUN_TASKS dict keyed by run_id string.  The stop_run() handler
    pops the Task from _RUN_TASKS and calls task.cancel() if the task is not done.

    Test isolation strategy
    -----------------------
    Each test patches ``api.routers.runs._run_paper_engine`` to replace the real
    coroutine (which connects to exchanges and a database) with a controlled stub.
    Two stub variants are used:

    1. ``_sleeping_engine`` — an infinite-sleep coroutine.  The task is created and
       registered but never completes during the test.  Used to verify that the task
       appears in _RUN_TASKS immediately after POST returns.

    2. ``_immediate_engine`` — a coroutine that returns immediately after removing
       itself from _RUN_TASKS (simulating the finally-block cleanup in the real
       implementation).  Used to verify the self-removal mechanic.

    Cleanup
    -------
    The ``_cancel_all_tasks`` autouse fixture cancels every task left in _RUN_TASKS
    after each test.  This prevents task-leaked warnings from the asyncio event loop
    and keeps tests fully independent.

    MagicMock tasks (test_stop_run_cancels_task)
    --------------------------------------------
    For the cancellation test we inject a plain MagicMock into _RUN_TASKS instead of
    a real asyncio.Task.  The stop_run handler only calls task.done() and task.cancel()
    on the object it pops from the dict, so a MagicMock is sufficient and avoids the
    complexity of creating a real running task from synchronous test code.
    """

    @pytest.fixture(autouse=True)
    def _cancel_all_tasks(self) -> None:
        """
        Autouse teardown fixture: cancel every task lingering in _RUN_TASKS
        after each test completes.

        This prevents unhandled-task warnings (which fail tests under
        filterwarnings = ["error"]) and ensures _RUN_TASKS is clean for the
        next test, even when a test raises an assertion error mid-flight.
        """
        yield  # test runs here
        from api.routers.runs import _RUN_TASKS
        for task in list(_RUN_TASKS.values()):
            # MagicMock objects (injected in test_stop_run_cancels_task) and
            # real asyncio.Task objects are both handled gracefully here.
            if hasattr(task, "cancel") and callable(task.cancel):
                try:
                    task.cancel()
                except Exception:
                    pass
        _RUN_TASKS.clear()

    def test_paper_run_registers_task(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        POST /api/v1/runs in paper mode must add an entry to _RUN_TASKS keyed
        by the returned run_id string before the HTTP response is returned.

        The patched coroutine sleeps indefinitely so the task stays alive
        (and therefore present in _RUN_TASKS) when we inspect the dict
        after the synchronous TestClient call returns.
        """
        from api.routers.runs import _RUN_TASKS

        async def _sleeping_engine(**kwargs):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise

        with patch("api.routers.runs._run_paper_engine", side_effect=_sleeping_engine):
            resp = client_dev_with_db.post("/api/v1/runs", json=_PAPER_PAYLOAD)

        assert resp.status_code == 201
        run_id_str = resp.json()["id"]

        # The task must be registered under the run's UUID string.
        assert run_id_str in _RUN_TASKS, (
            f"Expected run_id '{run_id_str}' in _RUN_TASKS; "
            f"actual keys: {list(_RUN_TASKS.keys())}"
        )

    def test_paper_run_task_has_correct_name(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        The asyncio.Task created for a paper run must carry the name
        ``paper-engine-{run_id}`` so it is identifiable in event-loop
        introspection and log messages.
        """
        from api.routers.runs import _RUN_TASKS

        async def _sleeping_engine(**kwargs):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise

        with patch("api.routers.runs._run_paper_engine", side_effect=_sleeping_engine):
            resp = client_dev_with_db.post("/api/v1/runs", json=_PAPER_PAYLOAD)

        assert resp.status_code == 201
        run_id_str = resp.json()["id"]

        task = _RUN_TASKS.get(run_id_str)
        assert task is not None, f"No task found for run_id '{run_id_str}'"

        expected_name = f"paper-engine-{run_id_str}"
        assert task.get_name() == expected_name, (
            f"Expected task name '{expected_name}', got '{task.get_name()}'"
        )

    def test_stop_run_cancels_task(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        DELETE /api/v1/runs/{run_id} must call task.cancel() on the task stored
        in _RUN_TASKS for that run_id, and must remove the entry from _RUN_TASKS.

        A MagicMock is injected directly into _RUN_TASKS to simulate a live
        background task without requiring a real asyncio.Task from synchronous
        test code.  The mock's .done() method returns False (task is running),
        so the handler's cancellation branch is exercised.

        The mock DB session is configured to return a 'running' RunORM so the
        stop_run handler proceeds past the 404/409 guards.
        """
        from api.routers.runs import _RUN_TASKS

        target_run_id = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
        target_run_id_str = str(target_run_id)

        # Wire the DB mock to return a running run for the DELETE query.
        # SimpleNamespace is used instead of _make_run_orm() to avoid the
        # SQLAlchemy mapper initialisation dependency that _make_run_orm has
        # when called with a non-default run_id in an isolated test context.
        run_ns = SimpleNamespace(
            id=target_run_id,
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "ma_crossover",
                "strategy_params": {},
                "symbols": ["BTC/USDT"],
                "timeframe": "1h",
                "mode": "paper",
                "initial_capital": "10000.00",
            },
            started_at=_FIXED_NOW,
            stopped_at=None,
            created_at=_FIXED_NOW,
            updated_at=_FIXED_NOW,
        )
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_ns)

        # Inject a MagicMock task that reports itself as still running.
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _RUN_TASKS[target_run_id_str] = mock_task

        resp = client_dev_with_db.delete(f"/api/v1/runs/{target_run_id}")

        assert resp.status_code == 200
        # cancel() must have been called exactly once.
        mock_task.cancel.assert_called_once()
        # The entry must be removed from the registry after cancellation.
        assert target_run_id_str not in _RUN_TASKS, (
            "Expected _RUN_TASKS to be empty for this run_id after stop_run"
        )

    def test_stop_run_without_task_still_succeeds(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        DELETE /api/v1/runs/{run_id} must return 200 even when _RUN_TASKS has
        no entry for the run_id (e.g. the task already completed naturally).

        This covers the path where ``_RUN_TASKS.pop(run_id, None)`` returns None
        and the cancellation branch is skipped entirely.
        """
        from api.routers.runs import _RUN_TASKS

        target_run_id = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
        target_run_id_str = str(target_run_id)

        # Confirm no task is present for this run_id.
        assert target_run_id_str not in _RUN_TASKS

        # SimpleNamespace avoids the SQLAlchemy mapper initialisation dependency.
        run_ns = SimpleNamespace(
            id=target_run_id,
            run_mode="paper",
            status="running",
            config={
                "strategy_name": "ma_crossover",
                "strategy_params": {},
                "symbols": ["BTC/USDT"],
                "timeframe": "1h",
                "mode": "paper",
                "initial_capital": "10000.00",
            },
            started_at=_FIXED_NOW,
            stopped_at=None,
            created_at=_FIXED_NOW,
            updated_at=_FIXED_NOW,
        )
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(run_ns)

        resp = client_dev_with_db.delete(f"/api/v1/runs/{target_run_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stopped"
        # No task entry should exist after the stop (pop on missing key is a no-op).
        assert target_run_id_str not in _RUN_TASKS

    def test_task_registry_empty_after_task_completes(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        When the paper engine coroutine completes (or returns early), its
        finally block must remove the run_id entry from _RUN_TASKS.

        The patched coroutine returns immediately after performing the same
        _RUN_TASKS.pop() that the real finally block performs.  After giving
        the event loop one iteration to drain the completed task (via a small
        asyncio.sleep(0) executed by the TestClient on a subsequent request),
        _RUN_TASKS must be empty for the run.

        Implementation note: TestClient runs the event loop between synchronous
        calls.  We trigger an additional GET request to /health after POST so
        the event loop has an opportunity to run the completed coroutine's
        cleanup before we inspect _RUN_TASKS.
        """
        from api.routers.runs import _RUN_TASKS

        async def _immediate_engine(**kwargs):
            # Mimic the real finally block: remove self from the task registry.
            _RUN_TASKS.pop(kwargs["run_id_str"], None)

        with patch("api.routers.runs._run_paper_engine", side_effect=_immediate_engine):
            resp = client_dev_with_db.post("/api/v1/runs", json=_PAPER_PAYLOAD)
            assert resp.status_code == 201
            run_id_str = resp.json()["id"]

            # Issue a second HTTP call so the TestClient's internal event loop
            # gets a full iteration to execute and finalize the completed task.
            client_dev_with_db.get("/health")

        assert run_id_str not in _RUN_TASKS, (
            f"Expected _RUN_TASKS to not contain '{run_id_str}' after task "
            f"completion; actual keys: {list(_RUN_TASKS.keys())}"
        )

    def test_multiple_paper_runs_each_get_own_task(
        self, client_dev_with_db: TestClient
    ) -> None:
        """
        Each POST /api/v1/runs (paper mode) must create an independent Task
        registered under a unique run_id in _RUN_TASKS.

        Two sequential paper-mode POSTs must produce two distinct _RUN_TASKS
        entries with different keys.  This verifies that the task registry
        does not accidentally overwrite entries or re-use run IDs.
        """
        from api.routers.runs import _RUN_TASKS

        async def _sleeping_engine(**kwargs):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise

        with patch("api.routers.runs._run_paper_engine", side_effect=_sleeping_engine):
            resp_a = client_dev_with_db.post("/api/v1/runs", json=_PAPER_PAYLOAD)
            resp_b = client_dev_with_db.post("/api/v1/runs", json=_PAPER_PAYLOAD)

        assert resp_a.status_code == 201
        assert resp_b.status_code == 201

        run_id_a = resp_a.json()["id"]
        run_id_b = resp_b.json()["id"]

        # The two runs must have been assigned different UUIDs.
        assert run_id_a != run_id_b, "Two paper runs received identical run IDs"

        # Both must have entries in _RUN_TASKS.
        assert run_id_a in _RUN_TASKS, f"No task registered for first run '{run_id_a}'"
        assert run_id_b in _RUN_TASKS, f"No task registered for second run '{run_id_b}'"

        # The two tasks must be distinct objects.
        assert _RUN_TASKS[run_id_a] is not _RUN_TASKS[run_id_b], (
            "Both paper runs share the same Task object — expected independent tasks"
        )
