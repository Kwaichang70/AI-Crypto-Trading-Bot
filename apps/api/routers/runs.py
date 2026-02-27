"""
apps/api/routers/runs.py
------------------------
Run management endpoints for the AI Crypto Trading Bot API.

Endpoints
---------
POST   /api/v1/runs              -- Start a new trading run
GET    /api/v1/runs              -- List all runs (paginated)
GET    /api/v1/runs/{run_id}     -- Get a single run's details
DELETE /api/v1/runs/{run_id}     -- Stop a running run

MVP notes
---------
- Backtest mode runs synchronously in the POST handler (fast enough for MVP).
- Paper/Live run creation is stub-level: the run record is created with
  status="running", but the engine is not wired up until Sprint 2.
- Strategy parameter validation occurs at request time via ``parameter_schema()``.
- The ``config`` JSONB snapshot captures all run parameters at creation time
  so historical runs are fully self-contained even if strategy defaults change.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import RunORM
from api.db.session import get_db
from api.schemas import (
    ErrorResponse,
    PaginationParams,
    RunCreateRequest,
    RunListResponse,
    RunResponse,
)
from common.types import RunMode

__all__ = ["router"]

router = APIRouter(prefix="/runs", tags=["runs"])

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry — maps API names to strategy classes
# Imported lazily inside the handler to avoid circular import issues.
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, Any] | None = None


def _get_strategy_registry() -> dict[str, Any]:
    """
    Return the lazy-loaded strategy name -> class mapping.

    Returns
    -------
    dict[str, Any]
        Mapping of strategy identifier to strategy class.
    """
    global _STRATEGY_REGISTRY
    if _STRATEGY_REGISTRY is None:
        from trading.strategies import (
            BreakoutStrategy,
            MACrossoverStrategy,
            RSIMeanReversionStrategy,
        )

        _STRATEGY_REGISTRY = {
            "ma_crossover": MACrossoverStrategy,
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "breakout": BreakoutStrategy,
        }
    return _STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Helper: ORM -> response model conversion
# ---------------------------------------------------------------------------

def _run_orm_to_response(run: RunORM) -> RunResponse:
    """
    Convert a ``RunORM`` instance to a ``RunResponse`` Pydantic model.

    Parameters
    ----------
    run:
        The ORM model instance to convert.

    Returns
    -------
    RunResponse
        The API response model.
    """
    return RunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# POST /api/v1/runs — start a new trading run
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=RunResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request (unknown strategy, bad params)"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
    summary="Start a new trading run",
    description=(
        "Create and start a new backtest, paper, or live trading run. "
        "Backtest runs execute synchronously and complete before the response is returned. "
        "Paper and live runs are created in the database with status='running'; "
        "the live engine wiring is Sprint 2."
    ),
)
async def create_run(
    body: RunCreateRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunResponse:
    """
    Start a new trading run.

    Parameters
    ----------
    body:
        Run configuration from the request body.
    db:
        Injected async database session.

    Returns
    -------
    RunResponse
        The newly created run record.

    Raises
    ------
    HTTPException 400:
        When the strategy name is unknown or strategy parameters fail
        schema validation.
    """
    log = logger.bind(
        endpoint="create_run",
        strategy_name=body.strategy_name,
        mode=body.mode,
        symbols=body.symbols,
        timeframe=body.timeframe,
    )
    log.info("runs.create_requested")

    registry = _get_strategy_registry()

    # Validate strategy name
    strategy_name = body.strategy_name.lower().replace("-", "_")
    if strategy_name not in registry:
        log.warning("runs.unknown_strategy", strategy_name=strategy_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown strategy: {body.strategy_name!r}. "
                f"Available: {sorted(registry.keys())}"
            ),
        )

    strategy_cls = registry[strategy_name]

    # Validate strategy parameters against the declared parameter_schema
    schema = strategy_cls.parameter_schema()
    param_errors = _validate_params_against_schema(body.strategy_params, schema)
    if param_errors:
        log.warning(
            "runs.invalid_strategy_params",
            errors=param_errors,
            strategy=strategy_name,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy parameters: {'; '.join(param_errors)}",
        )

    # Additional validation for backtest mode
    if body.mode == RunMode.BACKTEST or body.mode == "backtest":
        if body.backtest_start is None or body.backtest_end is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="backtest_start and backtest_end are required for backtest mode",
            )
        if body.backtest_start >= body.backtest_end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="backtest_start must be before backtest_end",
            )

    # Enforce live trading safety gate
    if body.mode == RunMode.LIVE or body.mode == "live":
        from api.config import get_settings
        settings = get_settings()
        if not settings.enable_live_trading:
            log.warning("runs.live_trading_disabled")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Live trading is disabled. "
                    "Set ENABLE_LIVE_TRADING=true and provide a confirm token."
                ),
            )

    # Build the config snapshot stored immutably on the run record
    run_id = uuid.uuid4()
    config_snapshot: dict[str, Any] = {
        "strategy_name": strategy_name,
        "strategy_params": body.strategy_params,
        "symbols": body.symbols,
        "timeframe": str(body.timeframe) if hasattr(body.timeframe, "value") else body.timeframe,
        "mode": str(body.mode) if hasattr(body.mode, "value") else body.mode,
        "initial_capital": body.initial_capital,
    }
    if body.backtest_start is not None:
        config_snapshot["backtest_start"] = body.backtest_start.isoformat()
    if body.backtest_end is not None:
        config_snapshot["backtest_end"] = body.backtest_end.isoformat()

    now = datetime.now(tz=UTC)

    # Determine mode string for ORM
    mode_value = body.mode if isinstance(body.mode, str) else body.mode.value

    run_orm = RunORM(
        id=run_id,
        run_mode=mode_value,
        status="running",
        config=config_snapshot,
        started_at=now,
        created_at=now,
        updated_at=now,
    )

    db.add(run_orm)
    await db.flush()  # Assign the PK within the transaction without committing

    # TODO Sprint 2 — BACKTEST: Call BacktestRunner.run() here (synchronously
    # or via an asyncio background task) with the bars fetched from the data
    # pipeline, then update run.status to "stopped" on success or "error" on
    # failure, and call await db.flush() again before returning.  Until this is
    # wired, every backtest request creates a "zombie" run record that remains
    # in status="running" permanently.
    #
    # TODO Sprint 2 — PAPER MODE: Launch the StrategyEngine in paper mode as a
    # long-running background task and update run.status to "stopped" or
    # "error" when the task terminates.  The record returned here is a stub.
    #
    # TODO Sprint 2 — LIVE MODE: Same as paper mode stub above.  Additionally,
    # enforce the live_trading_confirm_token from settings before allowing
    # execution (the 3-layer activation gate is currently only 1 layer deep).

    log.info(
        "runs.created",
        run_id=str(run_id),
        mode=mode_value,
        strategy=strategy_name,
    )

    return _run_orm_to_response(run_orm)


# ---------------------------------------------------------------------------
# GET /api/v1/runs — list all runs
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=RunListResponse,
    summary="List all trading runs",
    description="Returns a paginated list of all runs, ordered by creation time descending.",
)
async def list_runs(
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 50,
) -> RunListResponse:
    """
    List all trading runs with pagination.

    Parameters
    ----------
    db:
        Injected async database session.
    offset:
        Number of records to skip.
    limit:
        Maximum records to return.

    Returns
    -------
    RunListResponse
        Paginated list of run records.
    """
    log = logger.bind(endpoint="list_runs", offset=offset, limit=limit)
    log.info("runs.list_requested")

    # Count total matching rows
    count_stmt = select(func.count()).select_from(RunORM)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Fetch the page
    page_stmt = (
        select(RunORM)
        .order_by(RunORM.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(page_stmt)
    runs = list(result.scalars().all())

    log.info("runs.listed", total=total, returned=len(runs))

    return RunListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[_run_orm_to_response(r) for r in runs],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id} — get a single run
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}",
    response_model=RunResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get a single run's details",
)
async def get_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunResponse:
    """
    Retrieve details of a specific trading run.

    Parameters
    ----------
    run_id:
        UUID of the run to retrieve.
    db:
        Injected async database session.

    Returns
    -------
    RunResponse
        The run record.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    log = logger.bind(endpoint="get_run", run_id=str(run_id))
    log.info("runs.get_requested")

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    log.info("runs.found", status=run.status)
    return _run_orm_to_response(run)


# ---------------------------------------------------------------------------
# DELETE /api/v1/runs/{run_id} — stop a running run
# ---------------------------------------------------------------------------

@router.delete(
    "/{run_id}",
    status_code=status.HTTP_200_OK,
    response_model=RunResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        409: {"model": ErrorResponse, "description": "Run is not in a stoppable state"},
    },
    summary="Stop a running trading run",
    description=(
        "Transitions a run from 'running' to 'stopped'. "
        "Returns 409 if the run is already stopped or errored."
    ),
)
async def stop_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunResponse:
    """
    Stop a running trading run.

    Parameters
    ----------
    run_id:
        UUID of the run to stop.
    db:
        Injected async database session.

    Returns
    -------
    RunResponse
        The updated run record with status='stopped'.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    HTTPException 409:
        When the run is already in a terminal state (stopped/error).
    """
    log = logger.bind(endpoint="stop_run", run_id=str(run_id))
    log.info("runs.stop_requested")

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.status != "running":
        log.warning("runs.not_stoppable", current_status=run.status)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot stop run {run_id}: "
                f"current status is '{run.status}'. Only 'running' runs can be stopped."
            ),
        )

    now = datetime.now(tz=UTC)
    run.status = "stopped"
    run.stopped_at = now
    run.updated_at = now

    await db.flush()

    log.info("runs.stopped", run_id=str(run_id))
    return _run_orm_to_response(run)


# ---------------------------------------------------------------------------
# Parameter schema validation helper
# ---------------------------------------------------------------------------

def _validate_params_against_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """
    Perform lightweight JSON-Schema-style validation of strategy parameters.

    Only validates ``required`` fields and known ``properties`` types.
    Full JSON Schema validation (jsonschema library) is deferred to Sprint 2
    when strategies gain more complex parameter constraints.

    Parameters
    ----------
    params:
        The parameters submitted in the request.
    schema:
        JSON Schema dict from ``BaseStrategy.parameter_schema()``.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list = valid.
    """
    errors: list[str] = []

    required_fields: list[str] = schema.get("required", [])
    for field in required_fields:
        if field not in params:
            errors.append(f"Required parameter missing: '{field}'")

    properties: dict[str, Any] = schema.get("properties", {})
    for param_name, param_value in params.items():
        if param_name not in properties:
            if not schema.get("additionalProperties", True):
                errors.append(f"Unknown parameter: '{param_name}'")
            continue

        prop_schema = properties[param_name]
        expected_type = prop_schema.get("type")

        if expected_type == "integer" and not isinstance(param_value, int):
            errors.append(
                f"Parameter '{param_name}' must be an integer, "
                f"got {type(param_value).__name__}"
            )
        elif expected_type == "number" and not isinstance(param_value, (int, float)):
            errors.append(
                f"Parameter '{param_name}' must be a number, "
                f"got {type(param_value).__name__}"
            )
        elif expected_type == "string" and not isinstance(param_value, str):
            errors.append(
                f"Parameter '{param_name}' must be a string, "
                f"got {type(param_value).__name__}"
            )

        minimum = prop_schema.get("minimum")
        if minimum is not None and isinstance(param_value, (int, float)):
            if param_value < minimum:
                errors.append(
                    f"Parameter '{param_name}' must be >= {minimum}, "
                    f"got {param_value}"
                )

        maximum = prop_schema.get("maximum")
        if maximum is not None and isinstance(param_value, (int, float)):
            if param_value > maximum:
                errors.append(
                    f"Parameter '{param_name}' must be <= {maximum}, "
                    f"got {param_value}"
                )

    return errors
