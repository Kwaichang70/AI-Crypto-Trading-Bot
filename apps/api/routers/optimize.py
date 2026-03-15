"""
apps/api/routers/optimize.py
------------------------------
Parameter optimization endpoint.

POST /api/v1/optimize — Run a grid search over strategy parameters,
executing sequential backtests and returning ranked results.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

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


class OptimizeResponse(BaseModel):
    """Response for POST /api/v1/optimize."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

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
# Endpoint
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
async def run_optimization(body: OptimizeRequest) -> OptimizeResponse:
    """
    Execute a grid search over the strategy parameter space.

    Fetches OHLCV data once, then runs a backtest for each parameter
    combination. Returns ranked results by the chosen metric.
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

    # Step 3: Convert to response
    entries = [
        OptimizeEntryResponse(
            rank=e.rank,
            params=e.params,
            metrics=e.metrics,
        )
        for e in result.entries
    ]

    return OptimizeResponse(
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
