"""
tests/unit/test_order_fill_persistence.py
------------------------------------------
Unit tests for Sprint 16: Persist Orders & Fills to Database.

Modules under test
------------------
    packages/trading/execution.py        -- BaseExecutionEngine.get_all_orders()
    packages/trading/engines/paper.py    -- PaperExecutionEngine.get_all_fills()
    apps/api/routers/runs.py             -- _persist_paper_results() order/fill path
    packages/trading/backtest.py         -- BacktestRunner.last_execution_engine property

Coverage groups
---------------
1. TestGetAllOrders          -- get_all_orders() returns all orders regardless of status (3 tests)
2. TestGetAllFills           -- get_all_fills() merges fills from all orders sorted by executed_at (3 tests)
3. TestOrderFillPersistence  -- _persist_paper_results() writes OrderORM + FillORM with correct fields
                               and correct flush ordering (4 tests)
4. TestBacktestOrderPersistence -- BacktestRunner.last_execution_engine property lifecycle (2 tests)

Design notes
------------
- All async tests use @pytest.mark.asyncio for explicitness.  asyncio_mode = "auto" in
  pyproject.toml means the decorator is not strictly required but is included for clarity.
- For TestGetAllOrders and TestGetAllFills, Order and Fill domain objects are injected
  directly into engine._orders and engine._fills to avoid the async risk-check path.
  PaperExecutionEngine is used as the concrete BaseExecutionEngine subclass since
  BaseExecutionEngine is abstract.
- For TestOrderFillPersistence, a MagicMock execution_engine with configured
  get_all_orders() and get_all_fills() is passed to _persist_paper_results().
- The DB flush-ordering test uses db_mock.method_calls to verify that add_all(order_orms)
  is followed by flush() before add_all(fill_orms) is called.  This ensures FK
  constraints (fills.order_id -> orders.id) are satisfied at the DB layer.
- For TestBacktestOrderPersistence, BacktestRunner is exercised end-to-end with a
  real no-signal strategy and deterministic OHLCV bars.  No internal components are
  mocked: the runner wires PaperExecutionEngine, PortfolioAccounting, DefaultRiskManager,
  and StrategyEngine internally.
- The patch target for get_session_factory is "api.db.session.get_session_factory"
  because _persist_paper_results imports it lazily from the source module's namespace.
- StrEnum auto() produces lowercase string values: OrderSide.BUY.value == "buy", etc.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from api.routers.runs import _persist_paper_results
from common.models import MultiTimeframeContext, OHLCVBar
from common.types import OrderSide, OrderStatus, OrderType, TimeFrame
from trading.backtest import BacktestRunner
from trading.engines.paper import PaperExecutionEngine
from trading.models import Fill, Order, Signal
from trading.strategy import BaseStrategy, StrategyMetadata

# ---------------------------------------------------------------------------
# Constants shared across all test classes
# ---------------------------------------------------------------------------

_RUN_ID_STR = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_RUN_UUID = uuid.UUID(_RUN_ID_STR)

_T1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_T2 = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
_T3 = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)

# get_session_factory is imported lazily inside _persist_paper_results from
# api.db.session; patch the attribute on the source module, not on api.routers.runs.
_PATCH_TARGET = "api.db.session.get_session_factory"


# ---------------------------------------------------------------------------
# Shared helpers (mirroring the patterns from test_paper_engine_persistence.py)
# ---------------------------------------------------------------------------


def _make_portfolio(
    equity_curve: list[tuple[datetime, Decimal]] | None = None,
    trade_history: list[Any] | None = None,
) -> MagicMock:
    """
    Build a MagicMock that satisfies the PortfolioAccounting protocol used by
    _persist_paper_results.  Only get_equity_curve() and get_trade_history()
    are called by the function under test.
    """
    portfolio = MagicMock()
    portfolio.get_equity_curve.return_value = (
        equity_curve if equity_curve is not None else []
    )
    portfolio.get_trade_history.return_value = (
        trade_history if trade_history is not None else []
    )
    return portfolio


def _make_log() -> MagicMock:
    """
    Build a MagicMock for a structlog bound-logger.

    Structlog loggers use attribute access for level methods.  MagicMock's
    default attribute creation handles this; we assign explicit MagicMocks
    for determinism so assert_called() works reliably.
    """
    log = MagicMock()
    log.info = MagicMock()
    log.debug = MagicMock()
    log.warning = MagicMock()
    log.exception = MagicMock()
    return log


def _make_session_factory(
    db_mock: MagicMock | None = None,
    raise_on_commit: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Build a (get_session_factory_mock, db_session) pair for patching.

    The function under test calls:
        factory = get_session_factory()
        async with factory() as db:
            db.add_all(...)
            await db.flush()
            await db.commit()

    We model this as:
        get_session_factory()   -> factory_fn   (plain MagicMock)
        factory_fn()            -> async_ctx    (supports __aenter__ / __aexit__)
        async_ctx.__aenter__()  -> db_mock

    Parameters
    ----------
    db_mock:
        Pre-built session mock.  If None, a fresh MagicMock is created.
    raise_on_commit:
        When provided, db.commit() raises this exception (tests error paths).

    Returns
    -------
    (get_session_factory_mock, db_mock)
    """
    if db_mock is None:
        db_mock = MagicMock()

    if raise_on_commit is not None:
        async def _commit_raiser() -> None:
            raise raise_on_commit

        db_mock.commit = _commit_raiser
    else:
        db_mock.commit = AsyncMock()

    db_mock.rollback = AsyncMock()
    db_mock.flush = AsyncMock()
    db_mock.add_all = MagicMock()

    async_ctx = MagicMock()

    async def _aenter(self: Any = None) -> MagicMock:  # noqa: ANN401
        return db_mock

    async def _aexit(self: Any = None, *args: Any) -> bool:  # noqa: ANN401
        return False

    async_ctx.__aenter__ = _aenter
    async_ctx.__aexit__ = _aexit

    factory_fn = MagicMock(return_value=async_ctx)
    get_session_factory_mock = MagicMock(return_value=factory_fn)

    return get_session_factory_mock, db_mock


def _make_order(
    *,
    run_id: str = _RUN_ID_STR,
    client_order_id: str | None = None,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("0.1"),
    status: OrderStatus = OrderStatus.FILLED,
    filled_quantity: Decimal = Decimal("0.1"),
    average_fill_price: Decimal | None = Decimal("50000"),
    created_at: datetime = _T1,
    updated_at: datetime = _T1,
) -> Order:
    """
    Build a minimal MARKET Order domain object with controllable fields.

    The client_order_id defaults to a fixed suffix for determinism.
    MARKET orders must not have a price field (enforced by Order's model_validator).
    """
    coid = client_order_id or f"{run_id}-aabbcc112233"
    return Order(
        client_order_id=coid,
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=None,
        status=status,
        filled_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        exchange_order_id=None,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_limit_order(
    *,
    run_id: str = _RUN_ID_STR,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.1"),
    price: Decimal = Decimal("48000"),
    status: OrderStatus = OrderStatus.OPEN,
    filled_quantity: Decimal = Decimal("0"),
    average_fill_price: Decimal | None = None,
    created_at: datetime = _T1,
    updated_at: datetime = _T1,
) -> Order:
    """
    Build a LIMIT Order domain object for tests requiring a resting order.
    """
    return Order(
        client_order_id=f"{run_id}-limit-{uuid4().hex[:8]}",
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        status=status,
        filled_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        exchange_order_id=None,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_fill(
    *,
    order_id: UUID | None = None,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.1"),
    price: Decimal = Decimal("50000"),
    fee: Decimal = Decimal("0.5"),
    fee_currency: str = "USDT",
    is_maker: bool = False,
    executed_at: datetime = _T1,
) -> Fill:
    """
    Build a Fill domain object with controllable fields.
    """
    return Fill(
        order_id=order_id if order_id is not None else uuid4(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        fee_currency=fee_currency,
        is_maker=is_maker,
        executed_at=executed_at,
    )


def _make_paper_engine(run_id: str = _RUN_ID_STR) -> PaperExecutionEngine:
    """
    Build a PaperExecutionEngine with a MagicMock risk manager.

    The risk manager is never called in tests that only exercise get_all_orders()
    and get_all_fills() via direct dictionary injection.
    """
    risk_manager = MagicMock()
    return PaperExecutionEngine(run_id=run_id, risk_manager=risk_manager)


def _collect_orms_of_type(db_mock: MagicMock, orm_cls: type) -> list[Any]:
    """
    Iterate over all add_all() calls on db_mock and return every ORM object
    whose type matches orm_cls.
    """
    result = []
    for call_args in db_mock.add_all.call_args_list:
        items = call_args[0][0]
        result.extend(item for item in items if isinstance(item, orm_cls))
    return result


# ---------------------------------------------------------------------------
# Test strategy stub for BacktestRunner tests
# ---------------------------------------------------------------------------


class _AlwaysHoldStrategy(BaseStrategy):
    """
    Minimal strategy that always returns an empty signal list.

    Used for BacktestRunner integration tests where we care only about the
    infrastructure wiring (last_execution_engine property), not trading logic.
    """

    metadata = StrategyMetadata(
        name="always_hold_persistence_test",
        version="1.0.0",
        description="Stub strategy for order/fill persistence tests",
    )

    def on_bar(self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None) -> list[Signal]:
        return []

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}


def _make_btc_bars(n: int = 200, seed: int = 42) -> list[OHLCVBar]:
    """
    Create n deterministic BTC/USDT hourly bars using a seeded random walk.

    Timestamps are spaced 1 hour apart starting from 2024-01-01 00:00 UTC
    so they are strictly ascending (BacktestRunner validation requirement).
    The local random.Random instance avoids polluting the global random state.
    """
    rng = random.Random(seed)  # noqa: S311 — deliberate seeded PRNG for tests
    bars: list[OHLCVBar] = []
    price = 50000.0
    base_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    for i in range(n):
        change_pct = rng.uniform(-0.02, 0.02)
        close = max(1.0, price * (1 + change_pct))
        high = close * rng.uniform(1.000, 1.005)
        low = close * rng.uniform(0.995, 1.000)
        open_ = rng.uniform(low, high)
        volume = rng.uniform(5.0, 50.0)

        bars.append(
            OHLCVBar(
                symbol="BTC/USDT",
                timeframe=TimeFrame.ONE_HOUR,
                timestamp=base_ts + timedelta(hours=i),
                open=Decimal(str(round(open_, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(close, 2))),
                volume=Decimal(str(round(volume, 4))),
            )
        )
        price = close

    return bars


# ---------------------------------------------------------------------------
# Class 1: TestGetAllOrders
# ---------------------------------------------------------------------------


class TestGetAllOrders:
    """
    Verify BaseExecutionEngine.get_all_orders() returns all orders regardless
    of their status, including terminal states.

    The method is defined on BaseExecutionEngine (execution.py) and returns
    list(self._orders.values()).  PaperExecutionEngine is used as the concrete
    subclass because BaseExecutionEngine is abstract.

    We inject Order objects directly into engine._orders to avoid the async
    submit_order path, which requires a live risk manager and price feed.
    """

    def test_get_all_orders_empty(self) -> None:
        """
        Engine with no orders registered must return an empty list.

        This exercises the base case where get_all_orders() is called before
        any orders have been submitted.
        """
        engine = _make_paper_engine()

        orders = engine.get_all_orders()

        assert orders == [], (
            f"Expected empty list from fresh engine, got {orders}"
        )

    def test_get_all_orders_mixed_states(self) -> None:
        """
        Engine with orders in FILLED, CANCELED, and OPEN states must return
        ALL of them.

        get_all_orders() must differ from get_open_orders() (which excludes
        terminal states).  We verify that all three orders are returned and
        that none are filtered out.
        """
        engine = _make_paper_engine()

        order_filled = _make_order(
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.1"),
            client_order_id=f"{_RUN_ID_STR}-filled-001",
        )
        order_canceled = Order(
            client_order_id=f"{_RUN_ID_STR}-canceled-001",
            run_id=_RUN_ID_STR,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.05"),
            price=None,
            status=OrderStatus.CANCELED,
            filled_quantity=Decimal("0"),
            average_fill_price=None,
        )
        order_open = _make_limit_order(status=OrderStatus.OPEN)

        # Inject directly into the internal order registry
        engine._orders[order_filled.order_id] = order_filled
        engine._orders[order_canceled.order_id] = order_canceled
        engine._orders[order_open.order_id] = order_open

        orders = engine.get_all_orders()

        assert len(orders) == 3, (
            f"Expected 3 orders from get_all_orders(), got {len(orders)}"
        )

        returned_ids = {o.order_id for o in orders}
        assert order_filled.order_id in returned_ids, "FILLED order missing from get_all_orders()"
        assert order_canceled.order_id in returned_ids, "CANCELED order missing from get_all_orders()"
        assert order_open.order_id in returned_ids, "OPEN order missing from get_all_orders()"

    def test_get_all_orders_returns_all_terminal_states(self) -> None:
        """
        Terminal orders (FILLED, CANCELED, REJECTED, EXPIRED) must all be
        included in get_all_orders(), while get_open_orders() must exclude them.

        This is the key behavioural contract: get_all_orders() is the complete
        order log used for persistence, not just the live order book.
        """
        engine = _make_paper_engine()

        terminal_statuses = [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

        for i, status in enumerate(terminal_statuses):
            order = Order(
                client_order_id=f"{_RUN_ID_STR}-terminal-{i}",
                run_id=_RUN_ID_STR,
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                price=None,
                status=status,
                filled_quantity=Decimal("0"),
                average_fill_price=None,
            )
            engine._orders[order.order_id] = order

        # get_open_orders() filters terminal orders — must return 0
        open_orders = engine.get_open_orders()
        assert len(open_orders) == 0, (
            f"get_open_orders() should return 0 for all-terminal engine, "
            f"got {len(open_orders)}"
        )

        # get_all_orders() must return all of them regardless of terminal status
        all_orders = engine.get_all_orders()
        assert len(all_orders) == len(terminal_statuses), (
            f"get_all_orders() should return {len(terminal_statuses)} orders, "
            f"got {len(all_orders)}"
        )


# ---------------------------------------------------------------------------
# Class 2: TestGetAllFills
# ---------------------------------------------------------------------------


class TestGetAllFills:
    """
    Verify PaperExecutionEngine.get_all_fills() merges all per-order fill
    lists and returns them sorted by executed_at ascending.

    The method is defined on PaperExecutionEngine (engines/paper.py):
        def get_all_fills(self) -> list[Fill]:
            all_fills = []
            for fills in self._fills.values():
                all_fills.extend(fills)
            return sorted(all_fills, key=lambda f: f.executed_at)

    We inject Fill objects directly into engine._fills to avoid the async
    submit_order / _simulate_fill path.
    """

    def test_get_all_fills_empty(self) -> None:
        """
        Engine with no fills registered must return an empty list.

        This is the base case before any orders have been filled.
        """
        engine = _make_paper_engine()

        fills = engine.get_all_fills()

        assert fills == [], (
            f"Expected empty list from fresh engine, got {fills}"
        )

    def test_get_all_fills_multi_order_sorted(self) -> None:
        """
        Fills from multiple orders must be merged into a single list sorted
        by executed_at ascending.

        We create two orders with fills at staggered timestamps and verify
        that get_all_fills() interleaves them in chronological order,
        regardless of dictionary insertion order.
        """
        engine = _make_paper_engine()

        order_id_1 = uuid4()
        order_id_2 = uuid4()

        # Fill for order 1 at T3 (latest)
        fill_1_late = _make_fill(order_id=order_id_1, executed_at=_T3, price=Decimal("50000"))

        # Fills for order 2 at T1 (earliest) and T2 (middle)
        fill_2_early = _make_fill(order_id=order_id_2, executed_at=_T1, price=Decimal("49000"))
        fill_2_mid = _make_fill(order_id=order_id_2, executed_at=_T2, price=Decimal("49500"))

        engine._fills[order_id_1] = [fill_1_late]
        engine._fills[order_id_2] = [fill_2_early, fill_2_mid]

        all_fills = engine.get_all_fills()

        assert len(all_fills) == 3, (
            f"Expected 3 total fills, got {len(all_fills)}"
        )

        # Must be sorted ascending by executed_at
        assert all_fills[0].executed_at == _T1, (
            f"First fill should be at T1, got {all_fills[0].executed_at}"
        )
        assert all_fills[1].executed_at == _T2, (
            f"Second fill should be at T2, got {all_fills[1].executed_at}"
        )
        assert all_fills[2].executed_at == _T3, (
            f"Third fill should be at T3, got {all_fills[2].executed_at}"
        )

        # Verify the fill objects are the exact instances injected (no copying)
        assert all_fills[0] is fill_2_early
        assert all_fills[1] is fill_2_mid
        assert all_fills[2] is fill_1_late

    def test_get_all_fills_consistent_with_get_fills(self) -> None:
        """
        The union of get_fills(order_id) across all orders must contain the
        same fills as get_all_fills(), and get_all_fills() must be sorted.

        This verifies that get_all_fills() is a consistent aggregate of the
        per-order fill lists exposed by get_fills().
        """
        import asyncio

        engine = _make_paper_engine()

        order_id_a = uuid4()
        order_id_b = uuid4()

        fills_a = [
            _make_fill(order_id=order_id_a, executed_at=_T1),
            _make_fill(order_id=order_id_a, executed_at=_T3),
        ]
        fills_b = [
            _make_fill(order_id=order_id_b, executed_at=_T2),
        ]

        engine._fills[order_id_a] = fills_a
        engine._fills[order_id_b] = fills_b

        async def _collect_per_order() -> list[Fill]:
            result: list[Fill] = []
            for oid in [order_id_a, order_id_b]:
                result.extend(await engine.get_fills(oid))
            return result

        per_order_fills = asyncio.get_event_loop().run_until_complete(_collect_per_order())
        all_fills = engine.get_all_fills()

        # Both collections must contain the same fill IDs
        assert set(f.fill_id for f in per_order_fills) == set(f.fill_id for f in all_fills), (
            "get_all_fills() and union of get_fills() must contain the same fills"
        )

        # get_all_fills() must be sorted ascending by executed_at
        assert all_fills == sorted(all_fills, key=lambda f: f.executed_at), (
            "get_all_fills() result must be sorted by executed_at ascending"
        )


# ---------------------------------------------------------------------------
# Class 3: TestOrderFillPersistence
# ---------------------------------------------------------------------------


class TestOrderFillPersistence:
    """
    Verify that _persist_paper_results() correctly maps Order and Fill domain
    objects to OrderORM and FillORM rows and writes them in the correct order.

    The function signature (Sprint 16) is:
        async def _persist_paper_results(
            *,
            run_id_str: str,
            portfolio: Any,
            execution_engine: Any | None = None,
            log: Any,
        ) -> None

    We pass a MagicMock execution_engine with configurable get_all_orders()
    and get_all_fills() return values.  The portfolio provides a minimal
    equity curve (one point) so the function does not early-exit on the
    "no data" guard clause:
        has_orders = execution_engine is not None and bool(execution_engine.get_all_orders())
        if not equity_curve and not trade_history and not has_orders:
            return
    """

    def _minimal_portfolio(self) -> MagicMock:
        """Minimal portfolio with a single equity point to bypass the early-exit guard."""
        return _make_portfolio(
            equity_curve=[(_T1, Decimal("10000"))],
            trade_history=[],
        )

    def _mock_engine_with(
        self,
        orders: list[Order] | None = None,
        fills: list[Fill] | None = None,
    ) -> MagicMock:
        """Build a mock execution engine returning controlled orders/fills."""
        engine = MagicMock()
        engine.get_all_orders.return_value = orders if orders is not None else []
        engine.get_all_fills.return_value = fills if fills is not None else []
        return engine

    @pytest.mark.asyncio
    async def test_order_orm_field_mapping(self) -> None:
        """
        Each field on OrderORM must match the corresponding field on the
        domain Order object.

        Field mapping enforced by _persist_paper_results (runs.py):
            id                  = order.order_id
            client_order_id     = order.client_order_id
            run_id              = UUID(run_id_str)
            symbol              = order.symbol
            side                = order.side.value      (StrEnum -> lowercase str)
            order_type          = order.order_type.value
            quantity            = order.quantity
            price               = order.price           (None for MARKET)
            status              = order.status.value
            filled_quantity     = order.filled_quantity
            average_fill_price  = order.average_fill_price
            exchange_order_id   = None                  (paper engine does not use this)
            created_at          = order.created_at
            updated_at          = order.updated_at
        """
        order = _make_order(
            run_id=_RUN_ID_STR,
            client_order_id=f"{_RUN_ID_STR}-eth-sell-001",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("1.5"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.5"),
            average_fill_price=Decimal("3000"),
            created_at=_T1,
            updated_at=_T2,
        )

        portfolio = self._minimal_portfolio()
        log = _make_log()
        engine = self._mock_engine_with(orders=[order], fills=[])
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=engine,
                log=log,
            )

        from api.db.models import OrderORM

        order_orms: list[OrderORM] = _collect_orms_of_type(db_mock, OrderORM)

        assert len(order_orms) == 1, (
            f"Expected 1 OrderORM row, got {len(order_orms)}"
        )

        orm = order_orms[0]

        # Primary key maps to the domain Order UUID
        assert orm.id == order.order_id, (
            f"id mismatch: {orm.id} != {order.order_id}"
        )
        assert orm.client_order_id == order.client_order_id

        # run_id must be the UUID parsed from the string parameter
        assert orm.run_id == _RUN_UUID, (
            f"run_id UUID mismatch: {orm.run_id} != {_RUN_UUID}"
        )
        assert orm.symbol == "ETH/USDT"

        # StrEnum values: OrderSide.SELL.value == "sell"
        assert orm.side == "sell", (
            f"OrderSide.SELL.value should be 'sell', got {orm.side!r}"
        )
        assert orm.order_type == "market", (
            f"OrderType.MARKET.value should be 'market', got {orm.order_type!r}"
        )
        assert orm.quantity == Decimal("1.5")
        assert orm.price is None, "MARKET orders must map to price=None in the ORM"

        # StrEnum: OrderStatus.FILLED.value == "filled"
        assert orm.status == "filled", (
            f"OrderStatus.FILLED.value should be 'filled', got {orm.status!r}"
        )
        assert orm.filled_quantity == Decimal("1.5")
        assert orm.average_fill_price == Decimal("3000")

        # Paper engine never sets exchange_order_id
        assert orm.exchange_order_id is None, (
            "Paper engine must always write exchange_order_id=None"
        )
        assert orm.created_at == _T1
        assert orm.updated_at == _T2

    @pytest.mark.asyncio
    async def test_fill_orm_field_mapping(self) -> None:
        """
        Each field on FillORM must match the corresponding field on the
        domain Fill object.

        Field mapping enforced by _persist_paper_results (runs.py):
            id            = fill.fill_id
            order_id      = fill.order_id
            symbol        = fill.symbol
            side          = fill.side.value   (StrEnum -> lowercase str)
            quantity      = fill.quantity
            price         = fill.price
            fee           = fill.fee
            fee_currency  = fill.fee_currency
            is_maker      = fill.is_maker
            executed_at   = fill.executed_at
        """
        fill_order_id = uuid4()
        fill = _make_fill(
            order_id=fill_order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.25"),
            price=Decimal("45000"),
            fee=Decimal("11.25"),
            fee_currency="USDT",
            is_maker=True,
            executed_at=_T2,
        )

        portfolio = self._minimal_portfolio()
        log = _make_log()
        engine = self._mock_engine_with(orders=[], fills=[fill])
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=engine,
                log=log,
            )

        from api.db.models import FillORM

        fill_orms: list[FillORM] = _collect_orms_of_type(db_mock, FillORM)

        assert len(fill_orms) == 1, (
            f"Expected 1 FillORM row, got {len(fill_orms)}"
        )

        orm = fill_orms[0]

        # Primary key maps to the domain Fill UUID
        assert orm.id == fill.fill_id, (
            f"id mismatch: {orm.id} != {fill.fill_id}"
        )
        assert orm.order_id == fill_order_id, (
            f"order_id mismatch: {orm.order_id} != {fill_order_id}"
        )
        assert orm.symbol == "BTC/USDT"

        # StrEnum: OrderSide.BUY.value == "buy"
        assert orm.side == "buy", (
            f"OrderSide.BUY.value should be 'buy', got {orm.side!r}"
        )
        assert orm.quantity == Decimal("0.25")
        assert orm.price == Decimal("45000")
        assert orm.fee == Decimal("11.25")
        assert orm.fee_currency == "USDT"
        assert orm.is_maker is True
        assert orm.executed_at == _T2

    @pytest.mark.asyncio
    async def test_orders_flushed_before_fills(self) -> None:
        """
        db.flush() must be called after order_orms are added but before
        fill_orms are added.

        This ordering satisfies the FK constraint fills.order_id -> orders.id
        at the DB layer: the flush() makes the order rows visible within the
        transaction before the fill rows reference them.

        Verification strategy: inspect db_mock.method_calls to confirm the
        sequence:
            1. add_all([order_orms])  -- orders inserted first
            2. flush()                -- flush so fills can reference orders
            3. add_all([fill_orms])   -- fills inserted after flush

        We find the flush call index in the method_calls list and verify that
        at least one add_all() precedes it (orders) and at least one follows it
        (fills).  We further verify that the pre-flush add_all contains
        OrderORM and the post-flush add_all contains FillORM.
        """
        from api.db.models import FillORM, OrderORM

        order = _make_order(
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.1"),
            client_order_id=f"{_RUN_ID_STR}-flush-order",
        )
        fill = _make_fill(
            order_id=order.order_id,
            executed_at=_T1,
        )

        portfolio = self._minimal_portfolio()
        log = _make_log()
        engine = self._mock_engine_with(orders=[order], fills=[fill])
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=engine,
                log=log,
            )

        # Inspect the ordered list of method names called on the DB session
        call_names = [c[0] for c in db_mock.method_calls]

        # flush must have been called at all
        assert "flush" in call_names, (
            f"db.flush() was never called; method_calls = {db_mock.method_calls}"
        )

        flush_index = call_names.index("flush")
        add_all_indices = [i for i, name in enumerate(call_names) if name == "add_all"]

        add_all_before_flush = [i for i in add_all_indices if i < flush_index]
        add_all_after_flush = [i for i in add_all_indices if i > flush_index]

        assert add_all_before_flush, (
            "At least one add_all() call (for orders) must precede db.flush(). "
            f"call sequence = {call_names}"
        )
        assert add_all_after_flush, (
            "At least one add_all() call (for fills) must follow db.flush(). "
            f"call sequence = {call_names}"
        )

        # The add_all before flush must contain OrderORM objects
        pre_flush_call = db_mock.method_calls[add_all_before_flush[-1]]
        pre_flush_items = pre_flush_call[1][0]  # positional args[0] = the list passed to add_all
        assert any(isinstance(item, OrderORM) for item in pre_flush_items), (
            "add_all() before flush() must contain OrderORM objects"
        )

        # The add_all after flush must contain FillORM objects
        post_flush_call = db_mock.method_calls[add_all_after_flush[0]]
        post_flush_items = post_flush_call[1][0]
        assert any(isinstance(item, FillORM) for item in post_flush_items), (
            "add_all() after flush() must contain FillORM objects"
        )

    @pytest.mark.asyncio
    async def test_no_engine_skips_orders_fills(self) -> None:
        """
        Passing execution_engine=None must result in zero OrderORM and FillORM rows.

        The guard in _persist_paper_results:
            if execution_engine is not None:
                for order in execution_engine.get_all_orders(): ...
                for fill in execution_engine.get_all_fills(): ...

        When execution_engine is None, only equity snapshots and trade ORM rows
        are written.  The equity curve data from the portfolio is still persisted.
        """
        from api.db.models import FillORM, OrderORM

        portfolio = self._minimal_portfolio()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=None,
                log=log,
            )

        order_orms: list[OrderORM] = _collect_orms_of_type(db_mock, OrderORM)
        fill_orms: list[FillORM] = _collect_orms_of_type(db_mock, FillORM)

        assert len(order_orms) == 0, (
            f"Expected 0 OrderORM rows when execution_engine=None, "
            f"got {len(order_orms)}"
        )
        assert len(fill_orms) == 0, (
            f"Expected 0 FillORM rows when execution_engine=None, "
            f"got {len(fill_orms)}"
        )

        # The DB session must still have been used for equity snapshots
        assert db_mock.add_all.called, (
            "add_all() should have been called for equity snapshots even with engine=None"
        )


    @pytest.mark.asyncio
    async def test_persist_uses_exchange_order_id_from_order(self) -> None:
        """
        When an Order has exchange_order_id set (e.g. from a live run), the
        OrderORM row written by _persist_paper_results must carry that value.

        The field mapping in _persist_paper_results (runs.py):
            exchange_order_id = order.exchange_order_id

        For paper runs this is always None.  For live runs, the Order model is
        populated with the exchange-assigned ID (e.g. "exch-123") at submit
        time.  This test verifies that non-None exchange_order_id values flow
        through to the ORM without being silently dropped or overwritten.

        Verification strategy: construct an Order with exchange_order_id="exch-123",
        run _persist_paper_results, collect OrderORM objects from add_all() calls,
        and assert orm.exchange_order_id == "exch-123".
        """
        from api.db.models import OrderORM

        order = Order(
            client_order_id=f"{_RUN_ID_STR}-live-exch-order",
            run_id=_RUN_ID_STR,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.2"),
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.2"),
            average_fill_price=Decimal("50000"),
            exchange_order_id="exch-123",
            created_at=_T1,
            updated_at=_T2,
        )

        portfolio = self._minimal_portfolio()
        log = _make_log()
        engine = self._mock_engine_with(orders=[order], fills=[])
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=engine,
                log=log,
            )

        order_orms: list[OrderORM] = _collect_orms_of_type(db_mock, OrderORM)

        assert len(order_orms) == 1, (
            f"Expected 1 OrderORM row, got {len(order_orms)}"
        )

        orm = order_orms[0]

        assert orm.exchange_order_id == "exch-123", (
            f"OrderORM.exchange_order_id must be 'exch-123' from order.exchange_order_id, "
            f"got {orm.exchange_order_id!r}"
        )


# ---------------------------------------------------------------------------
# Class 4: TestBacktestOrderPersistence
# ---------------------------------------------------------------------------


class TestBacktestOrderPersistence:
    """
    Verify the BacktestRunner.last_execution_engine property lifecycle.

    Sprint 16 requires BacktestRunner to expose the PaperExecutionEngine used
    during the most recent run so that _persist_backtest_results() can call
    get_all_orders() and get_all_fills() on it.

    From backtest.py:
        # __init__:
        self._last_execution_engine: PaperExecutionEngine | None = None

        # run():
        _, _, _, exec_engine = self._build_engine(run_id)
        ...
        self._last_execution_engine = exec_engine
        return result

        # property:
        @property
        def last_execution_engine(self) -> PaperExecutionEngine | None:
            return self._last_execution_engine
    """

    def test_runner_last_execution_engine_initially_none(self) -> None:
        """
        Before any run, last_execution_engine must return None.

        The property is initialized in __init__ and must not be set until
        a run() call completes successfully.
        """
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy("hold_test")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
        )

        assert runner.last_execution_engine is None, (
            f"last_execution_engine should be None before run(), "
            f"got {runner.last_execution_engine!r}"
        )

    @pytest.mark.asyncio
    async def test_runner_last_execution_engine_populated(self) -> None:
        """
        After a successful BacktestRunner.run(), last_execution_engine must be
        a non-None PaperExecutionEngine instance.

        The test uses _AlwaysHoldStrategy (no signals, no trades) with 200
        hourly bars, which exceeds the 50-bar warmup requirement
        (warmup = max(0 * 2, 50) = 50 for a strategy with min_bars_required=0).

        Post-run checks:
        1. The property must not be None.
        2. The property must be a PaperExecutionEngine instance.
        3. get_all_orders() and get_all_fills() must be callable and return lists.
        4. With _AlwaysHoldStrategy (no signals), both lists must be empty.
        """
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy("hold_test")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=42,
        )

        bars_by_symbol = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}

        await runner.run(bars_by_symbol)

        engine = runner.last_execution_engine

        assert engine is not None, (
            "last_execution_engine must be non-None after a completed run()"
        )
        assert isinstance(engine, PaperExecutionEngine), (
            f"last_execution_engine must be PaperExecutionEngine, "
            f"got {type(engine).__name__}"
        )

        # Duck-type check: the persistence layer calls these two methods
        all_orders = engine.get_all_orders()
        all_fills = engine.get_all_fills()

        assert isinstance(all_orders, list), (
            f"get_all_orders() must return a list, got {type(all_orders).__name__}"
        )
        assert isinstance(all_fills, list), (
            f"get_all_fills() must return a list, got {type(all_fills).__name__}"
        )

        # _AlwaysHoldStrategy emits no signals, so no orders or fills are generated
        assert all_orders == [], (
            f"AlwaysHold strategy should produce no orders, got {len(all_orders)}"
        )
        assert all_fills == [], (
            f"AlwaysHold strategy should produce no fills, got {len(all_fills)}"
        )
