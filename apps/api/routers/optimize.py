"""
apps/api/routers/optimize.py
------------------------------
Parameter optimization endpoints.

POST /api/v1/optimize       — Run a grid search over strategy parameters,
                              execute sequential backtests, return ranked results,
                              and persist the run + entries to the database.
GET  /api/v1/optimize       — List all past optimization runs (summary, no entries).
GET  /api/v1/optimize/{id}  — Retrieve one optimization run with its full entry list.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.db import get_db
from api.db.models import OptimizationEntryORM, OptimizationRunORM
from common.types import TimeFrame
# TODO(sprint-30): extract to api.services.market_data / api.services.strategy_registry
from api.routers.runs import _fetch_bars_for_backtest, _get_strategy_registry
from trading.optimizer import SUPPORTED_METRICS, ParameterOptimizer

__all__ = ["router"]

router = APIRouter(prefix="/optimize", tags=["optimize"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """Request body for POST /api/v1/optimize."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    strategy_name: str = Field(
        description="Strategy identifier: ma_crossover | rsi_mean_reversion | breakout"
    )
    param_grid: dict[str, list[Any]] = Field(
        description=(
            "Parameter name -> list of values to search. "
            'Example: {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]}'
        )
    )
    symbols: list[str] = Field(
        min_length=1,
        description="CCXT-format trading pairs, e.g. ['BTC/USD']",
    )
    timeframe: str = Field(
        default="1h",
        description="OHLCV candle timeframe",
    )
    backtest_start: datetime = Field(
        description="Backtest start datetime (ISO-8601, UTC)",
    )
    backtest_end: datetime = Field(
        description="Backtest end datetime (ISO-8601, UTC)",
    )
    initial_capital: str = Field(
        default="10000",
        description="Starting capital in quote currency",
    )
    rank_by: str = Field(
        default="sharpe_ratio",
        description="Metric to rank results by",
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to return",
    )
    max_combinations: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Hard cap on total parameter combinations",
    )


class OptimizeEntryResponse(BaseModel):
    """One optimization result entry."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    rank: int
    params: dict[str, Any]
    metrics: dict[str, float]


class OptimizationRunSummary(BaseModel):
    """Summary of one past optimization run (no entries). Used by GET /api/v1/optimize."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    optimization_run_id: uuid.UUID
    strategy_name: str
    symbols: list[str]
    timeframe: str
    rank_by: str
    total_combinations: int
    completed_combinations: int
    failed_combinations: int
    elapsed_seconds: float
    created_at: datetime


class OptimizeResponse(BaseModel):
    """Response for POST /api/v1/optimize and GET /api/v1/optimize/{id}."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    optimization_run_id: uuid.UUID
    strategy_name: str
    symbols: list[str]
    timeframe: str
    rank_by: str
    total_combinations: int
    completed_combinations: int
    failed_combinations: int
    elapsed_seconds: float
    entries: list[OptimizeEntryResponse]


# ---------------------------------------------------------------------------
# POST /api/v1/optimize — run grid search + persist
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run parameter grid search optimization",
    responses={
        400: {"description": "Invalid strategy, params, or date range"},
        502: {"description": "Exchange unreachable"},
    },
)
async def run_optimization(
    body: OptimizeRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OptimizeResponse:
    """
    Execute a grid search over the strategy parameter space.

    Fetches OHLCV data once, then runs a backtest for each parameter
    combination. Returns ranked results by the chosen metric and persists
    the run + entries to the database for later retrieval.
    """
    log = logger.bind(
        endpoint="optimize",
        strategy_name=body.strategy_name,
        symbols=body.symbols,
    )
    log.info("optimize.requested")

    # Validate strategy name
    registry = _get_strategy_registry()
    strategy_name = body.strategy_name.lower().replace("-", "_")
    if strategy_name not in registry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown strategy: {body.strategy_name!r}. "
            f"Available: {sorted(registry.keys())}",
        )

    strategy_cls = registry[strategy_name]

    # Validate rank_by metric
    if body.rank_by not in SUPPORTED_METRICS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported rank_by: {body.rank_by!r}. "
            f"Supported: {sorted(SUPPORTED_METRICS)}",
        )

    # Validate date range
    if body.backtest_start >= body.backtest_end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="backtest_start must be before backtest_end",
        )

    timeframe = TimeFrame(body.timeframe)

    # Step 1: Fetch OHLCV data ONCE
    bars_by_symbol = await _fetch_bars_for_backtest(
        symbols=body.symbols,
        timeframe=timeframe,
        start=body.backtest_start,
        end=body.backtest_end,
        log=log,
    )

    # Step 2: Build and run optimizer
    try:
        optimizer = ParameterOptimizer(
            strategy_cls=strategy_cls,
            symbols=body.symbols,
            timeframe=timeframe,
            param_grid=body.param_grid,
            initial_capital=Decimal(body.initial_capital),
            rank_by=body.rank_by,
            top_n=body.top_n,
            max_combinations=body.max_combinations,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    log.info(
        "optimize.starting",
        total_combinations=optimizer.total_combinations,
        rank_by=body.rank_by,
    )

    result = await optimizer.run(bars_by_symbol)

    # Step 3: Build response entry list
    entries = [
        OptimizeEntryResponse(
            rank=e.rank,
            params=e.params,
            metrics=e.metrics,
        )
        for e in result.entries
    ]

    # Step 4: Persist optimization run + entries to DB
    orm_run = OptimizationRunORM(
        id=uuid.uuid4(),
        strategy_name=result.strategy_name,
        symbols=result.symbols,
        timeframe=result.timeframe,
        rank_by=result.rank_by,
        total_combinations=result.total_combinations,
        completed_combinations=result.completed_combinations,
        failed_combinations=result.failed_combinations,
        elapsed_seconds=float(result.elapsed_seconds),
    )
    db.add(orm_run)
    await db.flush()  # Assign PK before inserting child rows

    for e in result.entries:
        db.add(
            OptimizationEntryORM(
                id=uuid.uuid4(),
                optimization_run_id=orm_run.id,
                rank=e.rank,
                params=e.params,
                metrics=e.metrics,
            )
        )

    await db.commit()

    log.info(
        "optimize.persisted",
        optimization_run_id=str(orm_run.id),
        entries=len(result.entries),
    )

    return OptimizeResponse(
        optimization_run_id=orm_run.id,
        strategy_name=result.strategy_name,
        symbols=result.symbols,
        timeframe=result.timeframe,
        rank_by=result.rank_by,
        total_combinations=result.total_combinations,
        completed_combinations=result.completed_combinations,
        failed_combinations=result.failed_combinations,
        elapsed_seconds=result.elapsed_seconds,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/optimize — list all past optimization runs
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[OptimizationRunSummary],
    status_code=status.HTTP_200_OK,
    summary="List all past optimization runs",
    description=(
        "Returns a list of all persisted optimization runs ordered by creation "
        "time descending. Entry details are not included; use GET /{id} for those."
    ),
)
async def list_optimization_runs(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[OptimizationRunSummary]:
    """
    List all persisted optimization runs, newest first.

    Parameters
    ----------
    db:
        Injected async database session.

    Returns
    -------
    list[OptimizationRunSummary]
        Summary records for all optimization runs. Empty list when none exist.
    """
    log = logger.bind(endpoint="list_optimization_runs")
    log.info("optimize.list_requested")

    stmt = select(OptimizationRunORM).order_by(OptimizationRunORM.created_at.desc())
    rows = (await db.execute(stmt)).scalars().all()

    log.info("optimize.listed", count=len(rows))

    return [
        OptimizationRunSummary(
            optimization_run_id=row.id,
            strategy_name=row.strategy_name,
            symbols=list(row.symbols),
            timeframe=row.timeframe,
            rank_by=row.rank_by,
            total_combinations=row.total_combinations,
            completed_combinations=row.completed_combinations,
            failed_combinations=row.failed_combinations,
            elapsed_seconds=float(row.elapsed_seconds),
            created_at=row.created_at,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/optimize/{optimization_run_id} — get one run with entries
# ---------------------------------------------------------------------------


@router.get(
    "/{optimization_run_id}",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a single optimization run with its ranked entries",
    responses={
        404: {"description": "Optimization run not found"},
    },
)
async def get_optimization_run(
    optimization_run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OptimizeResponse:
    """
    Retrieve a persisted optimization run with its full ranked entry list.

    Parameters
    ----------
    optimization_run_id:
        UUID of the optimization run to retrieve.
    db:
        Injected async database session.

    Returns
    -------
    OptimizeResponse
        The run record with all ranked parameter entries.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    log = logger.bind(
        endpoint="get_optimization_run",
        optimization_run_id=str(optimization_run_id),
    )
    log.info("optimize.get_requested")

    stmt = (
        select(OptimizationRunORM)
        .where(OptimizationRunORM.id == optimization_run_id)
        .options(selectinload(OptimizationRunORM.entries))
    )
    row = (await db.execute(stmt)).scalar_one_or_none()

    if row is None:
        log.warning("optimize.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization run {optimization_run_id} not found",
        )

    log.info("optimize.found", entries=len(row.entries))

    entries = [
        OptimizeEntryResponse(
            rank=e.rank,
            params=e.params,
            metrics={k: float(v) for k, v in e.metrics.items()},
        )
        for e in row.entries
    ]

    return OptimizeResponse(
        optimization_run_id=row.id,
        strategy_name=row.strategy_name,
        symbols=list(row.symbols),
        timeframe=row.timeframe,
        rank_by=row.rank_by,
        total_combinations=row.total_combinations,
        completed_combinations=row.completed_combinations,
        failed_combinations=row.failed_combinations,
        elapsed_seconds=float(row.elapsed_seconds),
        entries=entries,
    )
