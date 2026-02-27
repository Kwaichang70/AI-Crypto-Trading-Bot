"""
apps/api/routers/orders.py
--------------------------
Order and fill query endpoints for the AI Crypto Trading Bot API.

Endpoints
---------
GET /api/v1/runs/{run_id}/orders            -- List orders for a run
GET /api/v1/runs/{run_id}/orders/{order_id} -- Get a single order
GET /api/v1/runs/{run_id}/fills             -- List fills for a run

Design notes
------------
- All endpoints are scoped under a run_id path parameter, enforcing that
  callers always operate within the context of a specific run.
- Fills are fetched via a JOIN on the orders table so the run_id filter is
  respected (FillORM has no direct run_id FK — it routes through OrderORM).
- Status filtering is supported on the orders list endpoint for the
  execution engine monitoring use case (active order polling).
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import FillORM, OrderORM, RunORM
from api.db.session import get_db
from api.schemas import (
    ErrorResponse,
    FillListResponse,
    FillResponse,
    OrderListResponse,
    OrderResponse,
)

__all__ = ["router"]

router = APIRouter(prefix="/runs", tags=["orders"])

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _assert_run_exists(run_id: uuid.UUID, db: AsyncSession) -> None:
    """
    Raise HTTP 404 if the run does not exist.

    Parameters
    ----------
    run_id:
        The run UUID to check.
    db:
        Async database session.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    stmt = select(func.count()).select_from(RunORM).where(RunORM.id == run_id)
    count: int = (await db.execute(stmt)).scalar_one()
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )


def _order_orm_to_response(order: OrderORM) -> OrderResponse:
    """
    Convert an ``OrderORM`` instance to ``OrderResponse``.

    Parameters
    ----------
    order:
        ORM model instance.

    Returns
    -------
    OrderResponse
        API response model.
    """
    return OrderResponse(
        id=order.id,
        client_order_id=order.client_order_id,
        run_id=order.run_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=str(order.quantity),
        price=str(order.price) if order.price is not None else None,
        status=order.status,
        filled_quantity=str(order.filled_quantity),
        average_fill_price=(
            str(order.average_fill_price)
            if order.average_fill_price is not None
            else None
        ),
        exchange_order_id=order.exchange_order_id,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def _fill_orm_to_response(fill: FillORM) -> FillResponse:
    """
    Convert a ``FillORM`` instance to ``FillResponse``.

    Parameters
    ----------
    fill:
        ORM model instance.

    Returns
    -------
    FillResponse
        API response model.
    """
    return FillResponse(
        id=fill.id,
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        quantity=str(fill.quantity),
        price=str(fill.price),
        fee=str(fill.fee),
        fee_currency=fill.fee_currency,
        is_maker=fill.is_maker,
        executed_at=fill.executed_at,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/orders — list orders for a run
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/orders",
    response_model=OrderListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="List orders for a run",
    description=(
        "Returns all orders for the given run, optionally filtered by status. "
        "Results are ordered by creation time descending."
    ),
)
async def list_orders(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 50,
    order_status: Annotated[
        str | None,
        Query(
            alias="status",
            description="Filter by order status: new | pending_submit | open | partial | filled | canceled | rejected | expired",
        ),
    ] = None,
    symbol: Annotated[
        str | None,
        Query(description="Filter by trading pair, e.g. BTC/USDT"),
    ] = None,
) -> OrderListResponse:
    """
    List orders for a trading run.

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
    order_status:
        Optional filter on order status.
    symbol:
        Optional filter on trading pair.

    Returns
    -------
    OrderListResponse
        Paginated list of orders.

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(
        endpoint="list_orders",
        run_id=str(run_id),
        order_status=order_status,
        symbol=symbol,
    )
    log.info("orders.list_requested")

    await _assert_run_exists(run_id, db)

    # Build base filter
    filters = [OrderORM.run_id == run_id]
    if order_status is not None:
        filters.append(OrderORM.status == order_status)
    if symbol is not None:
        filters.append(OrderORM.symbol == symbol)

    # Count
    count_stmt = (
        select(func.count())
        .select_from(OrderORM)
        .where(*filters)
    )
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Fetch page
    page_stmt = (
        select(OrderORM)
        .where(*filters)
        .order_by(OrderORM.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(page_stmt)
    orders = list(result.scalars().all())

    log.info("orders.listed", total=total, returned=len(orders))

    return OrderListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[_order_orm_to_response(o) for o in orders],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/orders/{order_id} — get a single order
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/orders/{order_id}",
    response_model=OrderResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run or order not found"},
    },
    summary="Get a single order",
)
async def get_order(
    run_id: uuid.UUID,
    order_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrderResponse:
    """
    Retrieve details of a specific order.

    Parameters
    ----------
    run_id:
        UUID of the parent run (used for scoping/ownership check).
    order_id:
        UUID of the order to retrieve.
    db:
        Injected async database session.

    Returns
    -------
    OrderResponse
        The order record.

    Raises
    ------
    HTTPException 404:
        When the run or order does not exist, or when the order does not
        belong to the specified run.
    """
    log = logger.bind(
        endpoint="get_order",
        run_id=str(run_id),
        order_id=str(order_id),
    )
    log.info("orders.get_requested")

    await _assert_run_exists(run_id, db)

    stmt = select(OrderORM).where(
        OrderORM.id == order_id,
        OrderORM.run_id == run_id,
    )
    result = await db.execute(stmt)
    order = result.scalar_one_or_none()

    if order is None:
        log.warning("orders.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found in run {run_id}",
        )

    log.info("orders.found", symbol=order.symbol, order_status=order.status)
    return _order_orm_to_response(order)


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/fills — list fills for a run
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/fills",
    response_model=FillListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="List fills for a run",
    description=(
        "Returns all execution fills for the given run. "
        "Fills are fetched via a join through the orders table. "
        "Results are ordered by execution time descending."
    ),
)
async def list_fills(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 50,
    symbol: Annotated[
        str | None,
        Query(description="Filter by trading pair, e.g. BTC/USDT"),
    ] = None,
) -> FillListResponse:
    """
    List execution fills for a trading run.

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

    Returns
    -------
    FillListResponse
        Paginated list of fills.

    Raises
    ------
    HTTPException 404:
        When the run does not exist.
    """
    log = logger.bind(
        endpoint="list_fills",
        run_id=str(run_id),
        symbol=symbol,
    )
    log.info("fills.list_requested")

    await _assert_run_exists(run_id, db)

    # FillORM joins through OrderORM for run scoping
    fill_filters = [OrderORM.run_id == run_id]
    if symbol is not None:
        fill_filters.append(FillORM.symbol == symbol)

    count_stmt = (
        select(func.count())
        .select_from(FillORM)
        .join(OrderORM, FillORM.order_id == OrderORM.id)
        .where(*fill_filters)
    )
    total: int = (await db.execute(count_stmt)).scalar_one()

    page_stmt = (
        select(FillORM)
        .join(OrderORM, FillORM.order_id == OrderORM.id)
        .where(*fill_filters)
        .order_by(FillORM.executed_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(page_stmt)
    fills = list(result.scalars().all())

    log.info("fills.listed", total=total, returned=len(fills))

    return FillListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[_fill_orm_to_response(f) for f in fills],
    )
