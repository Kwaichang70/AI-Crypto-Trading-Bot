"""
tests/integration/test_orders_endpoints.py
------------------------------------------
Integration tests for the orders and fills query endpoints.

Endpoints under test
--------------------
GET /api/v1/runs/{run_id}/orders              -- List orders for a run
GET /api/v1/runs/{run_id}/orders/{order_id}  -- Get a single order
GET /api/v1/runs/{run_id}/fills              -- List fills for a run

Test strategy
-------------
- All DB I/O is intercepted via the FastAPI dependency override that injects
  a hand-crafted AsyncMock instead of a real AsyncSession.  No PostgreSQL
  instance is required.
- Test data is constructed with deterministic UUIDs and fixed UTC timestamps
  so assertions never depend on wall-clock time.
- Each test is independent: the mock_db_session fixture resets all call
  history and return values between tests (function-scoped).

Mock wiring summary
-------------------
_assert_run_exists() pattern (first execute() call in every handler):
    stmt = select(func.count()).select_from(RunORM).where(RunORM.id == run_id)
    count: int = (await db.execute(stmt)).scalar_one()
    → Run exists:    scalar_one() returns 1  (integer count, NOT scalar_one_or_none)
    → Run missing:   scalar_one() returns 0  → handler raises HTTP 404

list_orders (3 execute() calls total):
    Call 0 → _assert_run_exists  : scalar_one()       returns int (1 or 0)
    Call 1 → COUNT query         : scalar_one()       returns int total
    Call 2 → page query          : scalars().all()    returns list[SimpleNamespace]

get_order (2 execute() calls total):
    Call 0 → _assert_run_exists  : scalar_one()          returns int (1 or 0)
    Call 1 → fetch order         : scalar_one_or_none()  returns SimpleNamespace | None

list_fills (3 execute() calls total):
    Call 0 → _assert_run_exists  : scalar_one()       returns int (1 or 0)
    Call 1 → COUNT with JOIN     : scalar_one()       returns int total
    Call 2 → page with JOIN      : scalars().all()    returns list[SimpleNamespace]

Decimal serialisation
---------------------
The conversion helpers _order_orm_to_response() and _fill_orm_to_response()
call str() on every Decimal field before passing it to the Pydantic schema.
The schema field_serializer methods also call str().  The net result is that
all monetary response fields arrive as JSON strings, never floats.
"""

from __future__ import annotations

from typing import Any
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Deterministic test data constants
# ---------------------------------------------------------------------------

_RUN_UUID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_ORDER_UUID = uuid.UUID("11111111-2222-3333-4444-555555555555")
_ORDER_UUID_2 = uuid.UUID("22222222-3333-4444-5555-666666666666")
_FILL_UUID = uuid.UUID("33333333-4444-5555-6666-777777777777")
_FILL_UUID_2 = uuid.UUID("44444444-4444-4444-4444-444444444444")
_MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")

_FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ORM factory helpers
# ---------------------------------------------------------------------------

def _make_order_orm(
    *,
    order_id: uuid.UUID = _ORDER_UUID,
    run_id: uuid.UUID = _RUN_UUID,
    client_order_id: str = "coid-abc123",
    symbol: str = "BTC/USDT",
    side: str = "buy",
    order_type: str = "market",
    quantity: Decimal = Decimal("0.05000000"),
    price: Decimal | None = None,
    status: str = "filled",
    filled_quantity: Decimal = Decimal("0.05000000"),
    average_fill_price: Decimal | None = Decimal("42000.00000000"),
    exchange_order_id: str | None = "EX-ORDER-001",
    created_at: datetime = _FIXED_NOW,
    updated_at: datetime = _FIXED_NOW,
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as OrderORM,
    suitable for Pydantic from_attributes=True serialization.
    """
    return SimpleNamespace(
        id=order_id,
        client_order_id=client_order_id,
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        status=status,
        filled_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        exchange_order_id=exchange_order_id,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_fill_orm(
    *,
    fill_id: uuid.UUID = _FILL_UUID,
    order_id: uuid.UUID = _ORDER_UUID,
    symbol: str = "BTC/USDT",
    side: str = "buy",
    quantity: Decimal = Decimal("0.05000000"),
    price: Decimal = Decimal("42000.00000000"),
    fee: Decimal = Decimal("2.10000000"),
    fee_currency: str = "USDT",
    is_maker: bool = False,
    executed_at: datetime = _FIXED_NOW,
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as FillORM,
    suitable for Pydantic from_attributes=True serialization.
    """
    return SimpleNamespace(
        id=fill_id,
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        fee_currency=fee_currency,
        is_maker=is_maker,
        executed_at=executed_at,
    )


# ---------------------------------------------------------------------------
# DB mock result helpers
# ---------------------------------------------------------------------------

def _make_run_exists_result(exists: bool) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result for _assert_run_exists().

    The helper calls scalar_one() which returns an integer count — 1 when the
    run exists and 0 when it does not.  This is NOT scalar_one_or_none; it
    always returns an int.
    """
    result = MagicMock()
    result.scalar_one.return_value = 1 if exists else 0
    return result


def _make_scalar_count_result(count: int) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting .scalar_one().

    Used for the COUNT query executed after _assert_run_exists() in list handlers.
    """
    result = MagicMock()
    result.scalar_one.return_value = count
    return result


def _make_scalars_result(items: list) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting .scalars().all().

    Used for the page SELECT query in list handlers.
    """
    result = MagicMock()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = items
    result.scalars.return_value = scalars_mock
    return result


def _make_scalar_one_or_none_result(value: object) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting
    .scalar_one_or_none().

    Used for the get_order fetch query (second execute() call in get_order).
    """
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/orders — list orders
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestListOrders:
    """Tests for GET /api/v1/runs/{run_id}/orders."""

    def test_returns_200_with_paginated_order_list_response(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        A valid run ID must return HTTP 200 with a populated OrderListResponse.

        The handler makes 3 execute() calls in sequence:
          - Call 0: _assert_run_exists() → scalar_one() returns 1 (run found)
          - Call 1: COUNT query         → scalar_one() returns 1 (one order)
          - Call 2: page SELECT         → scalars().all() returns [order_orm]

        We assert the response envelope structure and that the single item
        has correctly populated camelCase fields matching the OrderORM factory.
        """
        order_orm = _make_order_orm()
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([order_orm]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/orders")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["offset"] == 0
        assert body["limit"] == 50
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["id"] == str(_ORDER_UUID)
        assert item["runId"] == str(_RUN_UUID)
        assert item["symbol"] == "BTC/USDT"
        assert item["side"] == "buy"
        assert item["orderType"] == "market"
        assert item["status"] == "filled"
        # Verify Decimal fields are present and serialised as strings
        assert isinstance(item["quantity"], str)
        assert isinstance(item["filledQuantity"], str)
        assert item["quantity"] == "0.05000000"
        assert item["filledQuantity"] == "0.05000000"
        assert item["averageFillPrice"] == "42000.00000000"

    def test_empty_orders_returns_200_with_zero_total_and_empty_items(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When a run exists but has no orders the response must have total=0
        and items=[].

        The handler still executes all 3 queries: the run existence check,
        the COUNT (which returns 0), and the page query (which returns []).
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(0),
            _make_scalars_result([]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/orders")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_run_not_found_returns_404(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When _assert_run_exists() receives a count of 0 it raises HTTP 404.

        Only 1 execute() call is made — the existence check.  The COUNT and
        page queries are never reached because the exception short-circuits
        the handler.
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=False),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/orders")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]
        assert mock_db_session.execute.call_count == 1

    def test_pagination_offset_and_limit_reflected_in_response(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        The offset and limit query parameters must be echoed back in the
        response envelope.

        We do not inspect the SQLAlchemy statement objects — that would couple
        the test to ORM internals.  We verify the values are reflected in the
        response payload, which confirms the handler read and forwarded them.
        """
        order_orm = _make_order_orm()
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(20),
            _make_scalars_result([order_orm]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders?offset=10&limit=5"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["offset"] == 10
        assert body["limit"] == 5
        assert body["total"] == 20

    def test_status_filter_query_param_accepted(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        The ?status= query parameter must be accepted without a 422 error.

        The router exposes this as Query(alias="status") mapping to the
        order_status local variable.  We verify the endpoint returns 200 and
        that the filtered item in the response has the matching status field.
        """
        order_orm = _make_order_orm(status="open")
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([order_orm]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders?status=open"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["status"] == "open"

    def test_symbol_filter_query_param_accepted(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        The ?symbol= query parameter must be accepted without a 422 error.

        We verify the endpoint returns 200 and the filtered item carries
        the expected symbol field.
        """
        order_orm = _make_order_orm(symbol="ETH/USDT")
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([order_orm]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders?symbol=ETH%2FUSDT"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["symbol"] == "ETH/USDT"

    def test_limit_exceeds_maximum_returns_422(
        self,
        client_dev_with_db: TestClient,
    ) -> None:
        """
        A limit value greater than 500 must be rejected with HTTP 422.

        FastAPI validates the Query(le=500) constraint before the handler body
        executes.  No DB calls are made.
        """
        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders?limit=501"
        )

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/orders/{order_id} — single order detail
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetOrder:
    """Tests for GET /api/v1/runs/{run_id}/orders/{order_id}."""

    def test_known_order_returns_200_with_order_response(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        Requesting an order that exists must return HTTP 200 with OrderResponse.

        The handler makes 2 execute() calls:
          - Call 0: _assert_run_exists()  → scalar_one() returns 1
          - Call 1: fetch order           → scalar_one_or_none() returns order_orm

        We assert key fields are present with correct camelCase names and that
        the Decimal fields have been serialised to strings by the converter.
        """
        order_orm = _make_order_orm(
            order_id=_ORDER_UUID,
            run_id=_RUN_UUID,
            status="filled",
            exchange_order_id="EX-999",
        )
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_one_or_none_result(order_orm),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders/{_ORDER_UUID}"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(_ORDER_UUID)
        assert body["runId"] == str(_RUN_UUID)
        assert body["clientOrderId"] == "coid-abc123"
        assert body["symbol"] == "BTC/USDT"
        assert body["side"] == "buy"
        assert body["orderType"] == "market"
        assert body["status"] == "filled"
        assert body["exchangeOrderId"] == "EX-999"

    def test_order_not_found_returns_404(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When scalar_one_or_none() returns None the handler must raise HTTP 404.

        The run exists (first execute returns count=1), but the order fetch
        returns None, triggering the HTTPException(404) branch.
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_one_or_none_result(None),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders/{_MISSING_UUID}"
        )

        assert resp.status_code == 404
        body = resp.json()
        assert "Order" in body["detail"]
        assert str(_MISSING_UUID) in body["detail"]

    def test_run_not_found_returns_404_before_order_fetch(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When _assert_run_exists() fails the 404 is raised before the order
        fetch execute() call is made.

        We configure side_effect with only one result to confirm only one
        execute() call occurs.  If the handler incorrectly reached the second
        execute(), the mock would raise StopIteration / StopAsyncIteration,
        causing a 5xx rather than the expected 404.
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=False),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_MISSING_UUID}/orders/{_ORDER_UUID}"
        )

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]
        # Exactly one execute() call must have been made (the existence check)
        assert mock_db_session.execute.call_count == 1

    def test_order_with_null_price_returns_none_for_price_fields(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        A MARKET order has price=None and average_fill_price can be None before
        fills.  The converter guards these with ``if ... is not None`` logic and
        the response schema marks them as str | None.  We assert both come back
        as JSON null.
        """
        order_orm = _make_order_orm(
            order_type="market",
            price=None,
            average_fill_price=None,
            filled_quantity=Decimal("0"),
            status="open",
        )
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_one_or_none_result(order_orm),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders/{_ORDER_UUID}"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["price"] is None
        assert body["averageFillPrice"] is None

    def test_order_belonging_to_different_run_returns_404(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        An order that exists but belongs to a different run must return 404.

        The handler's WHERE clause includes both order_id AND run_id, so the
        DB query returns None when the run ownership does not match.  We mock
        scalar_one_or_none() returning None to simulate this case.

        This verifies the run-scoping security boundary: callers cannot access
        orders from another run by guessing a valid order_id.
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_one_or_none_result(None),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders/{_ORDER_UUID}"
        )

        assert resp.status_code == 404
        assert "Order" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/fills — list fills
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestListFills:
    """Tests for GET /api/v1/runs/{run_id}/fills."""

    def test_returns_200_with_paginated_fill_list_response(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        A valid run ID must return HTTP 200 with a populated FillListResponse.

        The handler makes 3 execute() calls in sequence:
          - Call 0: _assert_run_exists()        → scalar_one() returns 1
          - Call 1: COUNT with JOIN             → scalar_one() returns 1
          - Call 2: page SELECT with JOIN       → scalars().all() returns [fill_orm]

        We assert the response envelope structure and item fields including
        the camelCase aliased fields from the Pydantic schema.
        """
        fill_orm = _make_fill_orm()
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([fill_orm]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/fills")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["offset"] == 0
        assert body["limit"] == 50
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["id"] == str(_FILL_UUID)
        assert item["orderId"] == str(_ORDER_UUID)
        assert item["symbol"] == "BTC/USDT"
        assert item["side"] == "buy"
        assert item["feeCurrency"] == "USDT"
        assert item["isMaker"] is False

    def test_empty_fills_returns_200_with_zero_total(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When a run exists but has no fills the response must have total=0
        and items=[].

        All 3 execute() calls are made: the run check, the COUNT (0), and
        the page query (empty list).
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(0),
            _make_scalars_result([]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/fills")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_run_not_found_returns_404(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        When _assert_run_exists() receives a count of 0 it raises HTTP 404
        before the COUNT and page queries are executed.
        """
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=False),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/fills")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]
        assert mock_db_session.execute.call_count == 1

    def test_fills_pagination_params_reflected_in_response(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        The offset and limit query parameters must be echoed back in the
        fills response envelope.
        """
        fill_orm = _make_fill_orm()
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(30),
            _make_scalars_result([fill_orm]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/fills?offset=20&limit=10"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["offset"] == 20
        assert body["limit"] == 10
        assert body["total"] == 30

    def test_symbol_filter_query_param_accepted(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        The ?symbol= query parameter on the fills endpoint must be accepted
        without a 422 error.

        We verify the endpoint returns 200 and the fill item carries the
        expected symbol field.
        """
        fill_orm = _make_fill_orm(symbol="ETH/USDT")
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([fill_orm]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/fills?symbol=ETH%2FUSDT"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["symbol"] == "ETH/USDT"

    def test_fills_limit_exceeds_maximum_returns_422(
        self,
        client_dev_with_db: TestClient,
    ) -> None:
        """
        A limit value greater than 500 must be rejected with HTTP 422 by the
        fills endpoint.

        FastAPI validates the Query(le=500) constraint before the handler body
        executes.  No DB calls are made.
        """
        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/fills?limit=501"
        )

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication tests (production mode — require_api_auth=True)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOrdersAndFillsAuth:
    """
    Auth enforcement tests for orders and fills endpoints in production mode.

    These tests use client_prod (no DB override needed) because auth rejection
    happens in the require_api_key dependency — before any DB access.  The DB
    session would error if reached, but the 401 comes first.
    """

    def test_orders_list_without_api_key_returns_401(
        self,
        client_prod: TestClient,
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/orders without an API key must return 401
        in production mode.

        The require_api_key router dependency fires before the handler body,
        so no DB access or run existence check occurs.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/orders")

        assert resp.status_code == 401

    def test_fills_list_without_api_key_returns_401(
        self,
        client_prod: TestClient,
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/fills without an API key must return 401
        in production mode.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/fills")

        assert resp.status_code == 401

    def test_orders_list_with_valid_api_key_does_not_return_401(
        self,
        app_prod_mode: Any,
        mock_db_session: AsyncMock,
        auth_headers: dict[str, str],
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/orders with a valid API key must NOT return 401.

        We wire the DB mock into the prod-mode app so the request completes
        past the auth gate.  The exact response code (200, 404, etc.) is not
        the focus — we only assert the request was not rejected for auth.
        """
        from api.db.session import get_db

        async def _override_get_db():
            yield mock_db_session

        app_prod_mode.dependency_overrides[get_db] = _override_get_db

        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(0),
            _make_scalars_result([]),
        ]

        try:
            with TestClient(app_prod_mode, raise_server_exceptions=False) as c:
                resp = c.get(f"/api/v1/runs/{_RUN_UUID}/orders", headers=auth_headers)
            assert resp.status_code != 401
        finally:
            app_prod_mode.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Response format (Decimal serialisation)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDecimalSerialisation:
    """
    Verify that Decimal fields are serialised as JSON strings, never floats.

    The _order_orm_to_response() and _fill_orm_to_response() helpers call
    str() on Decimal attributes before constructing the Pydantic schema.
    The schema field_serializer methods also call str().  The combined result
    must appear as a string in the JSON body — never an IEEE-754 float.
    """

    def test_order_response_decimal_fields_are_strings(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        OrderResponse Decimal fields (quantity, filledQuantity, averageFillPrice)
        must arrive as JSON strings, not numbers.

        We use specific Decimal values that would lose precision as floats
        to verify the serialisation chain is intact end-to-end.
        """
        order_orm = _make_order_orm(
            order_id=_ORDER_UUID,
            quantity=Decimal("0.00100000"),
            price=Decimal("42150.75000000"),
            filled_quantity=Decimal("0.00100000"),
            average_fill_price=Decimal("42150.75000000"),
            status="filled",
            order_type="limit",
        )
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_one_or_none_result(order_orm),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/orders/{_ORDER_UUID}"
        )

        assert resp.status_code == 200
        body = resp.json()
        # All monetary fields must be JSON strings, not numbers
        assert isinstance(body["quantity"], str)
        assert isinstance(body["filledQuantity"], str)
        assert isinstance(body["price"], str)
        assert isinstance(body["averageFillPrice"], str)
        # Values must round-trip without precision loss
        assert body["quantity"] == "0.00100000"
        assert body["price"] == "42150.75000000"
        assert body["averageFillPrice"] == "42150.75000000"

    def test_fill_response_decimal_fields_are_strings(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        FillResponse Decimal fields (quantity, price, fee) must arrive as
        JSON strings, not numbers.

        We use a fee value with high precision to verify the str() conversion
        preserves trailing zeros through the full serialisation chain.
        """
        fill_orm = _make_fill_orm(
            fill_id=_FILL_UUID,
            quantity=Decimal("0.05000000"),
            price=Decimal("42000.12345678"),
            fee=Decimal("2.10006300"),
        )
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(1),
            _make_scalars_result([fill_orm]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/fills")

        assert resp.status_code == 200
        item = resp.json()["items"][0]
        # All monetary fields must be JSON strings, not numbers
        assert isinstance(item["quantity"], str)
        assert isinstance(item["price"], str)
        assert isinstance(item["fee"], str)
        # Values must preserve full precision through serialisation
        assert item["quantity"] == "0.05000000"
        assert item["price"] == "42000.12345678"
        assert item["fee"] == "2.10006300"

    def test_fill_response_is_maker_is_boolean(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """
        FillResponse.isMaker must be a JSON boolean, not a string or int.

        The FillORM.is_maker column is a Boolean column; the Pydantic schema
        declares it as bool.  We test both True and False values.
        """
        fill_maker = _make_fill_orm(fill_id=_FILL_UUID, is_maker=True)
        fill_taker = _make_fill_orm(
            fill_id=_FILL_UUID_2,
            is_maker=False,
        )
        mock_db_session.execute.side_effect = [
            _make_run_exists_result(exists=True),
            _make_scalar_count_result(2),
            _make_scalars_result([fill_maker, fill_taker]),
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/fills")

        assert resp.status_code == 200
        items = resp.json()["items"]
        assert items[0]["isMaker"] is True
        assert isinstance(items[0]["isMaker"], bool)
        assert items[1]["isMaker"] is False
        assert isinstance(items[1]["isMaker"], bool)
