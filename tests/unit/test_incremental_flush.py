"""
tests/unit/test_incremental_flush.py
-------------------------------------
Unit tests for the incremental flush feature in apps/api/routers/runs.py.

Modules under test
------------------
    apps/api/routers/runs.py  --  _IncrementalFlushState (dataclass)
                                  _flush_incremental()    (one DB flush cycle)
                                  _incremental_flush_loop() (periodic loop)

Coverage groups
---------------
1. TestIncrementalFlushState         -- dataclass initial state and mutation (2 tests)
2. TestFlushIncrementalDelta         -- delta computation and DB write paths (6 tests)
3. TestFlushIncrementalErrorHandling -- DB error paths and watermark safety (2 tests)
4. TestIncrementalFlushLoop          -- periodic loop scheduling and error resilience (2 tests)

Design notes
------------
- All async tests use @pytest.mark.asyncio for explicitness.
- Mock objects use SimpleNamespace so attribute access mirrors production code.
- get_session_factory patch target: api.db.session.get_session_factory
  (imported lazily inside _flush_incremental, same as test_order_fill_persistence.py).
- Order ORM rows use db.merge() (status updates); fills use db.add_all() (new only).
- Watermarks advance ONLY after successful db.commit().
- Skip guard: not new_equity and not new_trades and not new_orders
                and not new_fills and not has_positions
  where has_positions = hasattr(portfolio, _position_snapshots).
- Loop sleeps first, then flushes, then repeats.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from api.routers.runs import (
    _IncrementalFlushState,
    _flush_incremental,
    _incremental_flush_loop,
)

_RUN_ID_STR = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
_PATCH_TARGET = "api.db.session.get_session_factory"

_T1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_T2 = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
_T3 = datetime(2024, 1, 3, 0, 0, 0, tzinfo=UTC)
_T4 = datetime(2024, 1, 4, 0, 0, 0, tzinfo=UTC)


def _make_log():
    log = MagicMock()
    log.debug = MagicMock()
    log.info = MagicMock()
    log.warning = MagicMock()
    log.exception = MagicMock()
    return log


def _make_portfolio(equity_curve=None, trade_history=None, positions=None, include_position_snapshots=True):
    p = SimpleNamespace()
    _ec = equity_curve if equity_curve is not None else []
    _th = trade_history if trade_history is not None else []
    p.get_equity_curve = lambda: _ec
    p.get_trade_history = lambda: _th
    if include_position_snapshots:
        p._position_snapshots = positions if positions is not None else {}
    return p


def _make_trade(trade_id=None, symbol="BTC/USDT", side_value="buy"):
    t = SimpleNamespace()
    t.trade_id = trade_id or uuid.uuid4()
    t.symbol = symbol
    t.side = SimpleNamespace(value=side_value)
    t.entry_price = Decimal("50000")
    t.exit_price = Decimal("51000")
    t.quantity = Decimal("0.1")
    t.realised_pnl = Decimal("100")
    t.total_fees = Decimal("1")
    t.entry_at = _T1
    t.exit_at = _T2
    t.strategy_id = "test-strategy"
    return t


def _make_order(order_id=None, side_value="buy", status_value="open"):
    o = SimpleNamespace()
    o.order_id = order_id or uuid.uuid4()
    o.client_order_id = str(uuid.uuid4())
    o.symbol = "BTC/USDT"
    o.side = SimpleNamespace(value=side_value)
    o.order_type = SimpleNamespace(value="market")
    o.quantity = Decimal("0.1")
    o.price = None
    o.status = SimpleNamespace(value=status_value)
    o.filled_quantity = Decimal("0.1")
    o.average_fill_price = Decimal("50000")
    o.exchange_order_id = None
    o.created_at = _T1
    o.updated_at = _T1
    return o


def _make_fill(fill_id=None, order_id=None):
    f = SimpleNamespace()
    f.fill_id = fill_id or uuid.uuid4()
    f.order_id = order_id or uuid.uuid4()
    f.symbol = "BTC/USDT"
    f.side = SimpleNamespace(value="buy")
    f.quantity = Decimal("0.1")
    f.price = Decimal("50000")
    f.fee = Decimal("0.5")
    f.fee_currency = "USDT"
    f.is_maker = False
    f.executed_at = _T1
    return f


def _make_position(symbol="BTC/USDT", quantity=Decimal("0.1")):
    pos = SimpleNamespace()
    pos.symbol = symbol
    pos.quantity = quantity
    pos.average_entry_price = Decimal("50000")
    pos.current_price = Decimal("51000")
    pos.unrealised_pnl = Decimal("100")
    pos.realised_pnl = Decimal("0")
    pos.total_fees_paid = Decimal("0.5")
    pos.opened_at = _T1
    return pos


def _make_engine(orders=None, fills=None):
    _orders = orders if orders is not None else []
    _fills = fills if fills is not None else []
    e = SimpleNamespace()
    e.get_all_orders = lambda: _orders
    e.get_all_fills = lambda: _fills
    return e


def _make_session_factory(raise_on_commit=None, raise_on_factory=None):
    db_mock = MagicMock()
    db_mock.merge = AsyncMock()
    db_mock.flush = AsyncMock()
    db_mock.add_all = MagicMock()
    db_mock.execute = AsyncMock()
    db_mock.rollback = AsyncMock()

    if raise_on_commit is not None:
        _exc = raise_on_commit

        async def _commit_raiser():
            raise _exc

        db_mock.commit = _commit_raiser
    else:
        db_mock.commit = AsyncMock()

    async_ctx = MagicMock()

    async def _aenter(self=None):
        return db_mock

    async def _aexit(self=None, *args):
        return False

    async_ctx.__aenter__ = _aenter
    async_ctx.__aexit__ = _aexit

    factory_fn = MagicMock(return_value=async_ctx)

    if raise_on_factory is not None:
        gsf_mock = MagicMock(side_effect=raise_on_factory)
    else:
        gsf_mock = MagicMock(return_value=factory_fn)

    return gsf_mock, db_mock


class TestIncrementalFlushState:
    """
    Verify _IncrementalFlushState dataclass defaults and mutation isolation.
    The dataclass uses field(default_factory=set) for mutable set fields.
    """

    def test_initial_state(self):
        """
        A freshly constructed _IncrementalFlushState must have all counters at 0,
        both sets empty, and peak_equity == Decimal("0").
        """
        state = _IncrementalFlushState()

        assert state.flushed_equity_count == 0, (
            f"flushed_equity_count must start at 0, got {state.flushed_equity_count}"
        )
        assert state.flushed_trade_count == 0, (
            f"flushed_trade_count must start at 0, got {state.flushed_trade_count}"
        )
        assert isinstance(state.flushed_order_ids, set), (
            f"flushed_order_ids must be a set, got {type(state.flushed_order_ids).__name__}"
        )
        assert len(state.flushed_order_ids) == 0, (
            f"flushed_order_ids must be empty, has {len(state.flushed_order_ids)} items"
        )
        assert isinstance(state.flushed_fill_ids, set), (
            f"flushed_fill_ids must be a set, got {type(state.flushed_fill_ids).__name__}"
        )
        assert len(state.flushed_fill_ids) == 0, (
            f"flushed_fill_ids must be empty, has {len(state.flushed_fill_ids)} items"
        )
        assert state.peak_equity == Decimal("0"), (
            f"peak_equity must start at Decimal('0'), got {state.peak_equity}"
        )

    def test_mutable_fields_are_independent_instances(self):
        """
        Two _IncrementalFlushState instances must not share set objects.
        Mutations on state_a must not bleed into state_b.
        """
        state_a = _IncrementalFlushState()
        state_b = _IncrementalFlushState()

        order_id = uuid.uuid4()
        fill_id = uuid.uuid4()

        state_a.flushed_equity_count = 7
        state_a.flushed_trade_count = 3
        state_a.flushed_order_ids.add(order_id)
        state_a.flushed_fill_ids.add(fill_id)
        state_a.peak_equity = Decimal("99999.99")

        assert state_a.flushed_equity_count == 7
        assert state_a.flushed_trade_count == 3
        assert order_id in state_a.flushed_order_ids
        assert fill_id in state_a.flushed_fill_ids
        assert state_a.peak_equity == Decimal("99999.99")

        assert state_b.flushed_equity_count == 0
        assert state_b.flushed_trade_count == 0
        assert state_b.flushed_order_ids == set()
        assert state_b.flushed_fill_ids == set()
        assert state_b.peak_equity == Decimal("0")


class TestFlushIncrementalDelta:
    """
    Verify _flush_incremental() delta computation, DB write routing, and
    watermark advancement. Contracts exercised:
    - First flush: all data written; all watermarks advance from 0.
    - Second flush: only delta rows written (watermarks filter old data).
    - Nothing new + no _position_snapshots: DB session never opened.
    - Order status changes applied via merge() on ALL orders each cycle.
    - Flat positions (quantity <= 0) skipped before writing PositionSnapshotORM.
    - execution_engine=None: orders/fills skipped; equity still written.
    """

    @pytest.mark.asyncio
    async def test_first_flush_writes_all_data(self):
        """
        First flush: 3 equity points + 1 trade + 1 order + 1 fill.
        All watermarks must advance; db.merge() called once; db.commit() once.
        """
        order = _make_order()
        fill = _make_fill()
        trade = _make_trade()
        equity_curve = [
            (_T1, Decimal("10000")),
            (_T2, Decimal("10100")),
            (_T3, Decimal("10050")),
        ]
        portfolio = _make_portfolio(equity_curve=equity_curve, trade_history=[trade])
        engine = _make_engine(orders=[order], fills=[fill])
        state = _IncrementalFlushState()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                execution_engine=engine,
                state=state,
                log=log,
            )

        assert state.flushed_equity_count == 3
        assert state.flushed_trade_count == 1
        assert order.order_id in state.flushed_order_ids
        assert fill.fill_id in state.flushed_fill_ids
        assert db_mock.merge.call_count == 1
        db_mock.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_flush_writes_only_new_data(self):
        """
        After first flush of 2 equity points, second flush with 2 more points
        writes exactly 2 new EquitySnapshotORM rows (bar_index 2 and 3).
        """
        from api.db.models import EquitySnapshotORM

        first_equity = [(_T1, Decimal("10000")), (_T2, Decimal("10100"))]
        portfolio = _make_portfolio(equity_curve=first_equity, include_position_snapshots=False)
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )
        assert state.flushed_equity_count == 2

        extended = first_equity + [(_T3, Decimal("10200")), (_T4, Decimal("10150"))]
        portfolio.get_equity_curve = lambda: extended
        db_mock.add_all.reset_mock()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )

        assert state.flushed_equity_count == 4

        second_equity_orms = []
        for c in db_mock.add_all.call_args_list:
            second_equity_orms.extend(i for i in c[0][0] if isinstance(i, EquitySnapshotORM))

        assert len(second_equity_orms) == 2, (
            f"Second flush must write 2 equity rows, got {len(second_equity_orms)}"
        )
        bar_indices = {o.bar_index for o in second_equity_orms}
        assert bar_indices == {2, 3}, f"bar_index values must be {{2,3}}, got {bar_indices}"

    @pytest.mark.asyncio
    async def test_skip_when_no_new_data(self):
        """
        When watermarks match all data and portfolio lacks _position_snapshots,
        _flush_incremental() must return early without opening a DB session.
        """
        equity_curve = [(_T1, Decimal("10000")), (_T2, Decimal("10100"))]
        portfolio = _make_portfolio(equity_curve=equity_curve, include_position_snapshots=False)
        engine = _make_engine()
        state = _IncrementalFlushState()
        state.flushed_equity_count = 2
        log = _make_log()
        get_sf_mock, _ = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )

        get_sf_mock.assert_not_called()
        log.debug.assert_called_once()
        debug_event = log.debug.call_args[0][0]
        assert "runs.incremental_flush_skipped" in debug_event

    @pytest.mark.asyncio
    async def test_order_status_update_via_merge(self):
        """
        An order in flushed_order_ids must still be sent to db.merge() when
        its status changes. All orders are merged each cycle (not just new ones).

        First flush: order status=open. Second flush: same order status=filled
        + 1 new equity point (to avoid skip guard).
        Expected: db.merge() called on second flush with status=filled.
        """
        order_id = uuid.uuid4()
        order_open = _make_order(order_id=order_id, status_value="open")

        first_equity = [(_T1, Decimal("10000"))]
        portfolio = _make_portfolio(equity_curve=first_equity, include_position_snapshots=False)
        engine_open = _make_engine(orders=[order_open])
        state = _IncrementalFlushState()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine_open, state=state, log=log,
            )

        assert db_mock.merge.call_count == 1
        assert order_id in state.flushed_order_ids

        order_filled = _make_order(order_id=order_id, status_value="filled")
        engine_filled = _make_engine(orders=[order_filled])
        portfolio.get_equity_curve = lambda: first_equity + [(_T2, Decimal("10100"))]
        db_mock.merge.reset_mock()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine_filled, state=state, log=log,
            )

        assert db_mock.merge.call_count == 1, (
            f"Second flush must call db.merge() for status-updated order, "
            f"got {db_mock.merge.call_count}"
        )
        merged_orm = db_mock.merge.call_args[0][0]
        assert merged_orm.status == "filled", (
            f"Merged OrderORM status must be filled, got {merged_orm.status!r}"
        )

    @pytest.mark.asyncio
    async def test_flat_positions_filtered(self):
        """
        Positions with quantity <= 0 must be skipped. With BTC/USDT open
        (quantity=0.5) and ETH/USDT flat (quantity=0), only 1 PositionSnapshotORM
        row must be written.
        """
        from api.db.models import PositionSnapshotORM

        open_pos = _make_position(symbol="BTC/USDT", quantity=Decimal("0.5"))
        flat_pos = _make_position(symbol="ETH/USDT", quantity=Decimal("0"))

        portfolio = _make_portfolio(
            equity_curve=[(_T1, Decimal("10000"))],
            positions={"BTC/USDT": open_pos, "ETH/USDT": flat_pos},
        )
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )

        position_orms = []
        for c in db_mock.add_all.call_args_list:
            position_orms.extend(i for i in c[0][0] if isinstance(i, PositionSnapshotORM))

        assert len(position_orms) == 1, f"Only 1 non-flat position, got {len(position_orms)}"
        assert position_orms[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_flush_with_no_execution_engine(self):
        """
        When execution_engine=None: no db.merge(), no OrderORM or FillORM rows,
        but EquitySnapshotORM rows are still written and flushed_equity_count advances.
        """
        from api.db.models import EquitySnapshotORM, FillORM, OrderORM

        portfolio = _make_portfolio(
            equity_curve=[(_T1, Decimal("10000")), (_T2, Decimal("10100"))],
            include_position_snapshots=False,
        )
        state = _IncrementalFlushState()
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=None, state=state, log=log,
            )

        db_mock.merge.assert_not_called()

        all_added = []
        for c in db_mock.add_all.call_args_list:
            all_added.extend(c[0][0])

        assert len([i for i in all_added if isinstance(i, OrderORM)]) == 0
        assert len([i for i in all_added if isinstance(i, FillORM)]) == 0
        assert len([i for i in all_added if isinstance(i, EquitySnapshotORM)]) == 2
        assert state.flushed_equity_count == 2


class TestFlushIncrementalErrorHandling:
    """
    Verify _flush_incremental() handles errors safely:
    - Watermarks not advanced when commit fails.
    - log.exception called with correct event key.
    - Exceptions never propagate to caller.
    """

    @pytest.mark.asyncio
    async def test_watermarks_not_advanced_on_db_error(self):
        """
        When db.commit() raises: db.rollback() called, watermarks stay at 0,
        log.exception called with runs.incremental_flush_db_error, no propagation.
        """
        equity_curve = [(_T1, Decimal("10000"))]
        portfolio = _make_portfolio(equity_curve=equity_curve, include_position_snapshots=False)
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()

        db_error = RuntimeError("simulated commit failure")
        get_sf_mock, db_mock = _make_session_factory(raise_on_commit=db_error)

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )

        assert state.flushed_equity_count == 0, (
            f"flushed_equity_count must stay 0 on error, got {state.flushed_equity_count}"
        )
        assert state.flushed_trade_count == 0
        assert state.flushed_order_ids == set()
        assert state.flushed_fill_ids == set()
        db_mock.rollback.assert_called_once()

        log.exception.assert_called()
        event = log.exception.call_args[0][0]
        assert "runs.incremental_flush_db_error" in event, (
            f"log.exception event must contain runs.incremental_flush_db_error, got {event!r}"
        )

    @pytest.mark.asyncio
    async def test_session_factory_error_logged(self):
        """
        When get_session_factory() raises: outer except catches it, logs with
        runs.incremental_flush_session_failed, no propagation, watermarks unchanged.
        """
        equity_curve = [(_T1, Decimal("10000"))]
        portfolio = _make_portfolio(equity_curve=equity_curve, include_position_snapshots=False)
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()

        factory_error = RuntimeError("DB session pool unavailable")
        get_sf_mock, _ = _make_session_factory(raise_on_factory=factory_error)

        with patch(_PATCH_TARGET, get_sf_mock):
            await _flush_incremental(
                run_id_str=_RUN_ID_STR, portfolio=portfolio,
                execution_engine=engine, state=state, log=log,
            )

        log.exception.assert_called()
        event = log.exception.call_args[0][0]
        assert "runs.incremental_flush_session_failed" in event, (
            f"log.exception event must contain runs.incremental_flush_session_failed, got {event!r}"
        )
        assert state.flushed_equity_count == 0


class TestIncrementalFlushLoop:
    """
    Verify _incremental_flush_loop() scheduling behaviour.
    Loop: sleep -> flush -> repeat; CancelledError exits cleanly.
    RuntimeError from flush is caught, logged, loop continues.
    """

    @pytest.mark.asyncio
    async def test_loop_calls_flush_periodically(self):
        """
        Run loop for 3 iterations (CancelledError on 3rd flush call).
        Verify _flush_incremental called 3 times; asyncio.sleep called
        >= 3 times each with flush_interval.
        """
        call_count = 0
        flush_interval = 30.0

        async def _flush_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError()

        portfolio = _make_portfolio()
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()

        with (
            patch("api.routers.runs._flush_incremental", side_effect=_flush_side_effect) as mock_flush,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            try:
                await _incremental_flush_loop(
                    run_id_str=_RUN_ID_STR, portfolio=portfolio,
                    execution_engine=engine, state=state,
                    flush_interval=flush_interval, log=log,
                )
            except asyncio.CancelledError:
                pass

        assert mock_flush.call_count == 3, (
            f"_flush_incremental must be called 3 times, got {mock_flush.call_count}"
        )
        assert mock_sleep.call_count >= 3, (
            f"asyncio.sleep must be called >= 3 times, got {mock_sleep.call_count}"
        )
        for sc in mock_sleep.call_args_list[:3]:
            assert sc == call(flush_interval), (
                f"asyncio.sleep must be called with {flush_interval}, got {sc}"
            )

    @pytest.mark.asyncio
    async def test_loop_continues_after_flush_error(self):
        """
        RuntimeError from _flush_incremental: loop logs error, continues.
        Simulation: RuntimeError on 1st call, CancelledError on 2nd.
        Verify: 2 flush calls, log.exception with runs.incremental_flush_error,
        asyncio.sleep called >= 2 times.
        """
        flush_call_count = 0

        async def _flush_side_effect(**kwargs):
            nonlocal flush_call_count
            flush_call_count += 1
            if flush_call_count == 1:
                raise RuntimeError("simulated transient flush error")
            raise asyncio.CancelledError()

        portfolio = _make_portfolio()
        engine = _make_engine()
        state = _IncrementalFlushState()
        log = _make_log()

        with (
            patch("api.routers.runs._flush_incremental", side_effect=_flush_side_effect) as mock_flush,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            try:
                await _incremental_flush_loop(
                    run_id_str=_RUN_ID_STR, portfolio=portfolio,
                    execution_engine=engine, state=state,
                    flush_interval=5.0, log=log,
                )
            except asyncio.CancelledError:
                pass

        assert mock_flush.call_count == 2, (
            f"_flush_incremental must be called 2 times, got {mock_flush.call_count}"
        )
        log.exception.assert_called()
        event = log.exception.call_args[0][0]
        assert "runs.incremental_flush_error" in event, (
            f"log.exception event must contain runs.incremental_flush_error, got {event!r}"
        )
        assert mock_sleep.call_count >= 2, (
            f"asyncio.sleep must be called >= 2 times, got {mock_sleep.call_count}"
        )
