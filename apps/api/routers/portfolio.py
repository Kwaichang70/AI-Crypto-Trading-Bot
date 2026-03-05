"""
apps/api/routers/portfolio.py
------------------------------
Portfolio endpoints for the AI Crypto Trading Bot API.

Endpoints
---------
GET /api/v1/portfolio/summary             -- Aggregate cross-run portfolio summary
GET /api/v1/runs/{run_id}/portfolio       -- Portfolio summary snapshot
GET /api/v1/runs/{run_id}/equity-curve    -- Equity curve time series
GET /api/v1/runs/{run_id}/trades          -- Completed round-trip trades
GET /api/v1/runs/{run_id}/positions       -- Current open positions

Design notes
------------
- Portfolio summary and position data are derived from the database
  (EquitySnapshotORM for equity curve, TradeORM for trades, and the
  ``config`` JSONB on RunORM for initial capital).
- For running paper/live runs, the in-memory PortfolioAccounting state
  is the source of truth for live metrics; the DB contains checkpointed
  snapshots written on each bar close.
- For completed backtest runs, all data is fully in the DB.
- Equity curve endpoint supports both a paginated view (for large backtests)
  and a full-download mode (limit=0) for the frontend charting component.
- Positions are read from the dedicated ``position_snapshots`` table which
  is populated when a run stops. Non-flat positions (quantity > 0) are
  returned; flat/closed positions are excluded.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import EquitySnapshotORM, PositionSnapshotORM, RunORM, TradeORM
from api.db.session import get_db
from api.schemas import (
    AggregatePortfolioResponse,
    EquityCurveResponse,
    EquityPointResponse,
    ErrorResponse,
    PortfolioResponse,
    PositionListResponse,
    PositionResponse,
    TradeListResponse,
    TradeResponse,
)

__all__ = ["router", "summary_router"]

router = APIRouter(prefix="/runs", tags=["portfolio"])

summary_router = APIRouter(prefix="/portfolio", tags=["portfolio"])

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_run_or_404(run_id: uuid.UUID, db: AsyncSession) -> RunORM:
    """
    Fetch a RunORM or raise HTTP 404.

    Parameters
    ----------
    run_id:
        The run UUID to fetch.
    db:
        Async database session.

    Returns
    -------
    RunORM
        The fetched run record.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    return run


def _trade_orm_to_response(trade: TradeORM) -> TradeResponse:
    """
    Convert a ``TradeORM`` instance to ``TradeResponse``.

    Parameters
    ----------
    trade:
        ORM model instance.

    Returns
    -------
    TradeResponse
        API response model.
    """
    return TradeResponse(
        id=trade.id,
        run_id=trade.run_id,
        symbol=trade.symbol,
        side=trade.side,
        entry_price=str(trade.entry_price),
        exit_price=str(trade.exit_price),
        quantity=str(trade.quantity),
        realised_pnl=str(trade.realised_pnl),
        total_fees=str(trade.total_fees),
        entry_at=trade.entry_at,
        exit_at=trade.exit_at,
        strategy_id=trade.strategy_id,
    )


def _snapshot_orm_to_equity_point(snap: EquitySnapshotORM) -> EquityPointResponse:
    """
    Convert an ``EquitySnapshotORM`` to an ``EquityPointResponse``.

    Parameters
    ----------
    snap:
        ORM model instance.

    Returns
    -------
    EquityPointResponse
        API response model.
    """
    return EquityPointResponse(
        timestamp=snap.timestamp,
        equity=str(snap.equity),
        cash=str(snap.cash),
        unrealised_pnl=str(snap.unrealised_pnl),
        realised_pnl=str(snap.realised_pnl),
        drawdown_pct=str(snap.drawdown_pct),
        bar_index=snap.bar_index,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/portfolio — portfolio summary
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/portfolio",
    response_model=PortfolioResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get portfolio summary for a run",
    description=(
        "Returns a comprehensive snapshot of the portfolio: equity, cash, PnL metrics, "
        "drawdown, win/loss statistics, and open position count. "
        "For completed runs, data is sourced from the final equity snapshot. "
        "For running runs, the latest snapshot provides the data."
    ),
)
async def get_portfolio(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioResponse:
    """
    Get portfolio summary for a trading run.

    Parameters
    ----------
    run_id:
        UUID of the run.
    db:
        Injected async database session.

    Returns
    -------
    PortfolioResponse
        Complete portfolio summary.

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(endpoint="get_portfolio", run_id=str(run_id))
    log.info("portfolio.get_requested")

    run = await _get_run_or_404(run_id, db)

    # Extract initial capital from config snapshot
    initial_capital_str: str = run.config.get("initial_capital", "10000")
    initial_capital = Decimal(initial_capital_str)

    # Fetch aggregate trade statistics for this run
    trade_stats_stmt = select(
        func.count(TradeORM.id).label("total_trades"),
        func.sum(TradeORM.realised_pnl).label("total_realised_pnl"),
        func.sum(TradeORM.total_fees).label("total_fees_paid"),
    ).where(TradeORM.run_id == run_id)
    trade_stats = (await db.execute(trade_stats_stmt)).one()

    total_trades: int = trade_stats.total_trades or 0
    total_realised_pnl = Decimal(str(trade_stats.total_realised_pnl or "0"))
    total_fees_paid = Decimal(str(trade_stats.total_fees_paid or "0"))

    # Count winning/losing trades
    winning_stmt = (
        select(func.count(TradeORM.id))
        .where(TradeORM.run_id == run_id, TradeORM.realised_pnl > Decimal("0"))
    )
    losing_stmt = (
        select(func.count(TradeORM.id))
        .where(TradeORM.run_id == run_id, TradeORM.realised_pnl < Decimal("0"))
    )
    winning_trades: int = (await db.execute(winning_stmt)).scalar_one() or 0
    losing_trades: int = (await db.execute(losing_stmt)).scalar_one() or 0

    # Fetch the latest equity snapshot for current state
    latest_snap_stmt = (
        select(EquitySnapshotORM)
        .where(EquitySnapshotORM.run_id == run_id)
        .order_by(EquitySnapshotORM.bar_index.desc())
        .limit(1)
    )
    snap_result = await db.execute(latest_snap_stmt)
    latest_snap = snap_result.scalar_one_or_none()

    if latest_snap is not None:
        current_equity = latest_snap.equity
        current_cash = latest_snap.cash
        unrealised_pnl = latest_snap.unrealised_pnl
        drawdown_pct = float(latest_snap.drawdown_pct)
    else:
        # No snapshots yet — run just started or is empty
        current_equity = initial_capital
        current_cash = initial_capital
        unrealised_pnl = Decimal("0")
        drawdown_pct = 0.0

    # Compute peak equity from equity curve
    peak_equity_stmt = (
        select(func.max(EquitySnapshotORM.equity))
        .where(EquitySnapshotORM.run_id == run_id)
    )
    peak_equity_raw = (await db.execute(peak_equity_stmt)).scalar_one_or_none()
    peak_equity = (
        Decimal(str(peak_equity_raw)) if peak_equity_raw is not None else current_equity
    )

    # Total return
    total_return_pct: float = 0.0
    if initial_capital > Decimal("0"):
        total_return_pct = float(
            (current_equity - initial_capital) / initial_capital
        )

    # Max drawdown across all snapshots
    # Using 1 - (equity / peak) would require a window function; we approximate
    # via the max drawdown_pct stored in EquitySnapshotORM for MVP
    max_dd_stmt = (
        select(func.max(EquitySnapshotORM.drawdown_pct))
        .where(EquitySnapshotORM.run_id == run_id)
    )
    max_dd_raw = (await db.execute(max_dd_stmt)).scalar_one_or_none()
    max_drawdown_pct = float(max_dd_raw) if max_dd_raw is not None else 0.0

    # Count equity curve length
    eq_count_stmt = (
        select(func.count())
        .select_from(EquitySnapshotORM)
        .where(EquitySnapshotORM.run_id == run_id)
    )
    equity_curve_length: int = (await db.execute(eq_count_stmt)).scalar_one() or 0

    # Compute daily PnL: sum of realised_pnl from trades exiting today (UTC)
    # Use datetime.now(tz=UTC) to ensure UTC-day boundary regardless of server TZ
    now_utc = datetime.now(tz=UTC)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_pnl_stmt = (
        select(func.sum(TradeORM.realised_pnl))
        .where(TradeORM.run_id == run_id, TradeORM.exit_at >= today_start)
    )
    daily_pnl_raw = (await db.execute(daily_pnl_stmt)).scalar_one_or_none()
    daily_pnl = Decimal(str(daily_pnl_raw)) if daily_pnl_raw is not None else Decimal("0")

    # Query real position count from position_snapshots table
    pos_count_stmt = (
        select(func.count())
        .select_from(PositionSnapshotORM)
        .where(
            PositionSnapshotORM.run_id == run_id,
            PositionSnapshotORM.quantity > 0,
        )
    )
    pos_count_result = await db.execute(pos_count_stmt)
    open_positions = pos_count_result.scalar_one()

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    log.info(
        "portfolio.fetched",
        total_trades=total_trades,
        current_equity=str(current_equity),
        drawdown_pct=drawdown_pct,
    )

    return PortfolioResponse(
        run_id=str(run_id),
        initial_cash=str(initial_capital),
        current_cash=str(current_cash),
        current_equity=str(current_equity),
        peak_equity=str(peak_equity),
        total_return_pct=total_return_pct,
        total_realised_pnl=str(total_realised_pnl),
        total_fees_paid=str(total_fees_paid),
        daily_pnl=str(daily_pnl),
        drawdown_pct=drawdown_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        open_positions=open_positions,
        equity_curve_length=equity_curve_length,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/equity-curve — equity curve time series
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/equity-curve",
    response_model=EquityCurveResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get equity curve for a run",
    description=(
        "Returns the equity curve as a time series of portfolio snapshots. "
        "Results are ordered by bar_index ascending. "
        "Use offset/limit to page through large backtests."
    ),
)
async def get_equity_curve(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=10000, description="Max points to return")] = 1000,
) -> EquityCurveResponse:
    """
    Get equity curve data for a trading run.

    Parameters
    ----------
    run_id:
        UUID of the run.
    db:
        Injected async database session.
    offset:
        Number of data points to skip.
    limit:
        Maximum data points to return (capped at 10,000 for large backtests).

    Returns
    -------
    EquityCurveResponse
        Equity curve with total point count and paginated data.

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(
        endpoint="get_equity_curve",
        run_id=str(run_id),
        offset=offset,
        limit=limit,
    )
    log.info("equity_curve.get_requested")

    await _get_run_or_404(run_id, db)

    # Count total
    count_stmt = (
        select(func.count())
        .select_from(EquitySnapshotORM)
        .where(EquitySnapshotORM.run_id == run_id)
    )
    total: int = (await db.execute(count_stmt)).scalar_one() or 0

    # Fetch page ordered by bar_index ascending (chronological order)
    page_stmt = (
        select(EquitySnapshotORM)
        .where(EquitySnapshotORM.run_id == run_id)
        .order_by(EquitySnapshotORM.bar_index.asc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(page_stmt)
    snapshots = list(result.scalars().all())

    log.info("equity_curve.fetched", total=total, returned=len(snapshots))

    return EquityCurveResponse(
        run_id=run_id,
        total_points=total,
        points=[_snapshot_orm_to_equity_point(s) for s in snapshots],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/trades — completed trades
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/trades",
    response_model=TradeListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="List completed trades for a run",
    description=(
        "Returns all completed round-trip trades for the given run. "
        "Results are ordered by exit time descending. "
        "Supports optional filtering by symbol."
    ),
)
async def list_trades(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 50,
    symbol: Annotated[
        str | None,
        Query(description="Filter by trading pair, e.g. BTC/USDT"),
    ] = None,
    strategy_id: Annotated[
        str | None,
        Query(description="Filter by strategy identifier"),
    ] = None,
) -> TradeListResponse:
    """
    List completed round-trip trades for a run.

    Parameters
    ----------
    run_id:
        UUID of the parent run.
    db:
        Injected async database session.
    offset:
        Records to skip for pagination.
    limit:
        Maximum records to return.
    symbol:
        Optional filter on trading pair.
    strategy_id:
        Optional filter on strategy identifier.

    Returns
    -------
    TradeListResponse
        Paginated list of completed trades.

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(
        endpoint="list_trades",
        run_id=str(run_id),
        symbol=symbol,
        strategy_id=strategy_id,
    )
    log.info("trades.list_requested")

    await _get_run_or_404(run_id, db)

    filters = [TradeORM.run_id == run_id]
    if symbol is not None:
        filters.append(TradeORM.symbol == symbol)
    if strategy_id is not None:
        filters.append(TradeORM.strategy_id == strategy_id)

    count_stmt = (
        select(func.count())
        .select_from(TradeORM)
        .where(*filters)
    )
    total: int = (await db.execute(count_stmt)).scalar_one() or 0

    page_stmt = (
        select(TradeORM)
        .where(*filters)
        .order_by(TradeORM.exit_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(page_stmt)
    trades = list(result.scalars().all())

    log.info("trades.listed", total=total, returned=len(trades))

    return TradeListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[_trade_orm_to_response(t) for t in trades],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/positions — current open positions
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/positions",
    response_model=PositionListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get current open positions for a run",
    description=(
        "Returns all non-flat open positions for the given run. "
        "Positions are read from the position_snapshots table populated "
        "when a run stops. Only non-flat positions (quantity > 0) are returned."
    ),
)
async def get_positions(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PositionListResponse:
    """
    Return persisted position snapshots for a run.

    Queries ``PositionSnapshotORM`` for the given run, filtering to rows
    where ``quantity > 0`` (non-flat positions only). Returns an empty list
    for runs where all positions have been closed or no snapshots exist yet.

    Parameters
    ----------
    run_id:
        UUID of the run.
    db:
        Injected async database session.

    Returns
    -------
    PositionListResponse
        List of open positions (may be empty for completed/flat runs).

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(endpoint="get_positions", run_id=str(run_id))
    log.info("positions.get_requested")

    await _get_run_or_404(run_id, db)

    # Query persisted position snapshots (non-flat only)
    stmt = (
        select(PositionSnapshotORM)
        .where(
            PositionSnapshotORM.run_id == run_id,
            PositionSnapshotORM.quantity > 0,
        )
    )
    result = await db.execute(stmt)
    snapshots = list(result.scalars().all())

    positions = [
        PositionResponse(
            symbol=s.symbol,
            run_id=str(s.run_id),
            quantity=str(s.quantity),
            average_entry_price=str(s.average_entry_price),
            current_price=str(s.current_price),
            realised_pnl=str(s.realised_pnl),
            unrealised_pnl=str(s.unrealised_pnl),
            total_fees_paid=str(s.total_fees_paid),
            notional_value=str(s.quantity * s.current_price),
            opened_at=s.opened_at,
            updated_at=s.snapshot_at,
        )
        for s in snapshots
    ]

    log.info("positions.returned", count=len(positions))

    return PositionListResponse(
        run_id=run_id,
        positions=positions,
        count=len(positions),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/portfolio/summary — cross-run aggregate summary
# ---------------------------------------------------------------------------

@summary_router.get(
    "/summary",
    response_model=AggregatePortfolioResponse,
    summary="Get aggregate portfolio summary across all runs",
    description=(
        "Returns a cross-run aggregate view: run counts by status, total "
        "trade win/loss statistics, cumulative realised PnL and fees, "
        "best/worst backtest return percentages, and total capital deployed."
    ),
)
async def get_aggregate_portfolio(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AggregatePortfolioResponse:
    """Return a cross-run aggregate portfolio summary."""
    log = logger.bind(endpoint="get_aggregate_portfolio")
    log.info("aggregate_portfolio.get_requested")

    # ------------------------------------------------------------------
    # Query 1: Run status counts
    # ------------------------------------------------------------------
    status_stmt = (
        select(RunORM.status, func.count(RunORM.id).label("cnt"))
        .group_by(RunORM.status)
    )
    status_result = await db.execute(status_stmt)
    status_counts: dict[str, int] = {
        row.status: row.cnt for row in status_result.all()
    }
    total_runs = sum(status_counts.values())

    # ------------------------------------------------------------------
    # Query 2: Trade aggregate statistics (single round-trip)
    # Uses CASE-based conditional counting for cross-DB compatibility.
    # ------------------------------------------------------------------
    trade_stmt = select(
        func.count(TradeORM.id).label("total"),
        func.count(
            case((TradeORM.realised_pnl > Decimal("0"), TradeORM.id), else_=None)
        ).label("wins"),
        func.count(
            case((TradeORM.realised_pnl < Decimal("0"), TradeORM.id), else_=None)
        ).label("losses"),
        func.coalesce(func.sum(TradeORM.realised_pnl), Decimal("0")).label("pnl"),
        func.coalesce(func.sum(TradeORM.total_fees), Decimal("0")).label("fees"),
    )
    trade_result = await db.execute(trade_stmt)
    trade_row = trade_result.one()

    total_trades: int = trade_row.total or 0
    winning_trades: int = trade_row.wins or 0
    losing_trades: int = trade_row.losses or 0
    total_realised_pnl = Decimal(str(trade_row.pnl or "0"))
    total_fees_paid = Decimal(str(trade_row.fees or "0"))
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # ------------------------------------------------------------------
    # Query 3: Config blobs — initial_capital + backtest_metrics
    # Python-level extraction from JSONB for safety and maintainability.
    # ------------------------------------------------------------------
    configs_stmt = select(RunORM.config)
    configs_result = await db.execute(configs_stmt)
    config_rows: list[dict[str, Any]] = [
        row[0] for row in configs_result.all() if row[0]
    ]

    total_initial_capital = Decimal("0")
    return_pcts: list[float] = []

    for cfg in config_rows:
        # Sum initial capital (stored as string e.g. "10000.00")
        raw_capital = cfg.get("initial_capital")
        if raw_capital is not None:
            try:
                total_initial_capital += Decimal(str(raw_capital))
            except Exception:
                pass

        # Collect backtest return percentages (snake_case keys in JSONB)
        raw_metrics = cfg.get("backtest_metrics")
        if isinstance(raw_metrics, dict):
            ret_pct = raw_metrics.get("total_return_pct")
            if ret_pct is not None:
                try:
                    return_pcts.append(float(ret_pct))
                except (TypeError, ValueError):
                    pass

    best_return: float | None = max(return_pcts) if return_pcts else None
    worst_return: float | None = min(return_pcts) if return_pcts else None

    log.info(
        "aggregate_portfolio.fetched",
        total_runs=total_runs,
        total_trades=total_trades,
    )

    return AggregatePortfolioResponse(
        total_runs=total_runs,
        running_runs=status_counts.get("running", 0),
        stopped_runs=status_counts.get("stopped", 0),
        error_runs=status_counts.get("error", 0),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_realised_pnl=str(total_realised_pnl),
        total_fees_paid=str(total_fees_paid),
        best_run_return_pct=best_return,
        worst_run_return_pct=worst_return,
        total_initial_capital=str(total_initial_capital),
    )
