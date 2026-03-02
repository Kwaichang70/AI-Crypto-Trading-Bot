"""
tests/unit/test_paper_engine_persistence.py
-------------------------------------------
Unit tests for the _persist_paper_results() private coroutine in
apps/api/routers/runs.py.

Module under test
-----------------
    apps/api/routers/runs.py  --  _persist_paper_results()

Coverage groups
---------------
1. TestEquityCurveOrmConstruction   -- EquitySnapshotORM fields: run_id, equity, cash,
                                       unrealised_pnl, realised_pnl, bar_index, timestamp (1 test)
2. TestEarlyExitWhenNoData          -- Empty equity curve + empty trade history: DB session never
                                       created (1 test)
3. TestDrawdownPeakTracking         -- Peak-tracking drawdown: 4-bar curve, verify per-bar
                                       drawdown_pct values (1 test)
4. TestDrawdownClamped              -- drawdown_pct clamped to [0, 1] when equity < 0 (1 test)
5. TestTradeOrmConstruction         -- TradeORM rows written via session.add_all() with correct
                                       fields from TradeResult objects (1 test)
6. TestSessionErrorCaught           -- DB session raises on commit; exception caught, not
                                       propagated to caller (1 test)

Design notes
------------
- _persist_paper_results is an async function; all tests use pytest.mark.asyncio.
  pyproject.toml configures asyncio_mode = "auto" — the decorator is included for
  explicitness but is not strictly required.
- get_session_factory is imported lazily inside _persist_paper_results using
  `from api.db.session import get_session_factory`.  Because the import statement
  re-resolves the name from api.db.session on every call, the correct patch target
  is "api.db.session.get_session_factory" (the attribute on the source module),
  NOT "api.routers.runs.get_session_factory" (which is never bound in that namespace).
- The DB session is used as an async context manager (async with factory() as db).
  We wire this via MagicMock + __aenter__/__aexit__ rather than AsyncMock so we
  have fine-grained control over add_all() and commit() call verification.
- Portfolio is passed as a plain MagicMock with get_equity_curve() and
  get_trade_history() configured per test.  No real PortfolioAccounting is
  instantiated — the function only calls these two protocol methods.
- The log parameter is a MagicMock; we never assert on log calls but need to
  supply a valid object so structlog-style attribute access does not raise.
- TradeResult objects from trading.models are constructed directly (no mocking)
  to exercise the field-mapping path in _persist_paper_results faithfully.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.routers.runs import _persist_paper_results
from common.types import OrderSide
from trading.models import TradeResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID_STR = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_RUN_UUID = uuid.UUID(_RUN_ID_STR)

_T1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_T2 = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
_T3 = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)
_T4 = datetime(2024, 1, 1, 3, 0, 0, tzinfo=UTC)

# Patch target: get_session_factory is imported lazily inside the function body
# with `from api.db.session import get_session_factory`, so we patch the attribute
# on its source module — api.db.session — not on api.routers.runs.
_PATCH_TARGET = "api.db.session.get_session_factory"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(
    equity_curve: list[tuple[datetime, Decimal]],
    trade_history: list[Any] | None = None,
) -> MagicMock:
    """
    Build a MagicMock that mimics the PortfolioAccounting protocol used by
    _persist_paper_results.  Only get_equity_curve() and get_trade_history()
    are called by the function under test.
    """
    portfolio = MagicMock()
    portfolio.get_equity_curve.return_value = equity_curve
    portfolio.get_trade_history.return_value = trade_history or []
    return portfolio


def _make_log() -> MagicMock:
    """
    Build a MagicMock for a structlog bound-logger.

    Structlog loggers use attribute access for level methods (log.info, log.debug,
    log.exception).  MagicMock's default attribute creation handles this without
    explicit configuration; we assign explicit MagicMocks for clarity and so that
    assert_called() works deterministically.
    """
    log = MagicMock()
    log.info = MagicMock()
    log.debug = MagicMock()
    log.exception = MagicMock()
    return log


def _make_session_factory(
    db_mock: MagicMock | None = None,
    raise_on_commit: Exception | None = None,
    raise_on_factory_call: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Build a (get_session_factory_mock, db_session) pair for patching.

    The function under test calls:
        factory = get_session_factory()
        async with factory() as db:
            db.add_all(...)
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
    raise_on_factory_call:
        When provided, the outer factory() call raises this exception.

    Returns
    -------
    (get_session_factory_mock, db_mock):
        get_session_factory_mock is used as the patch replacement for the
        api.db.session.get_session_factory attribute.  db_mock is the session
        object yielded by the async context manager.
    """
    if db_mock is None:
        db_mock = MagicMock()

    # commit() must be awaitable
    if raise_on_commit is not None:
        async def _commit_raiser() -> None:
            raise raise_on_commit

        db_mock.commit = _commit_raiser
    else:
        db_mock.commit = AsyncMock()

    db_mock.rollback = AsyncMock()
    db_mock.add_all = MagicMock()

    # Build the async context manager returned by factory()
    async_ctx = MagicMock()

    async def _aenter(self: Any = None) -> MagicMock:  # noqa: ANN401
        return db_mock

    async def _aexit(self: Any = None, *args: Any) -> bool:  # noqa: ANN401
        return False

    async_ctx.__aenter__ = _aenter
    async_ctx.__aexit__ = _aexit

    # The callable returned by get_session_factory()
    factory_fn = MagicMock()
    if raise_on_factory_call is not None:
        factory_fn.side_effect = raise_on_factory_call
    else:
        factory_fn.return_value = async_ctx

    # get_session_factory() itself
    get_session_factory_mock = MagicMock(return_value=factory_fn)

    return get_session_factory_mock, db_mock


def _make_trade_result(
    *,
    run_id: str = _RUN_ID_STR,
    symbol: str = "BTC/USDT",
    entry_price: Decimal = Decimal("50000"),
    exit_price: Decimal = Decimal("55000"),
    quantity: Decimal = Decimal("0.1"),
    realised_pnl: Decimal = Decimal("500"),
    total_fees: Decimal = Decimal("10"),
    strategy_id: str = "ma_crossover",
) -> TradeResult:
    """
    Build a minimal TradeResult with controllable fields.

    entry_at is pinned to _T1, exit_at to _T2 for determinism.
    """
    return TradeResult(
        run_id=run_id,
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        realised_pnl=realised_pnl,
        total_fees=total_fees,
        entry_at=_T1,
        exit_at=_T2,
        strategy_id=strategy_id,
    )


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
# Test 1 — Equity curve -> EquitySnapshotORM construction
# ---------------------------------------------------------------------------


class TestEquityCurveOrmConstruction:
    """
    Verify _persist_paper_results writes EquitySnapshotORM rows with the
    correct field values when the portfolio has an equity curve.

    The function hard-codes cash=0, unrealised_pnl=0, realised_pnl=0 (MVP
    note in source).  run_id must match the UUID parsed from run_id_str.
    bar_index must be the enumeration index (0-based).  timestamp must match
    the tuple's first element.
    """

    @pytest.mark.asyncio
    async def test_persist_equity_curve_writes_snapshot_orms(self) -> None:
        equity_curve = [
            (_T1, Decimal("10000")),
            (_T2, Decimal("10500")),
        ]
        portfolio = _make_portfolio(equity_curve=equity_curve)
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        assert db_mock.add_all.called, "add_all() was never called on the DB session"

        from api.db.models import EquitySnapshotORM

        equity_orms: list[EquitySnapshotORM] = _collect_orms_of_type(
            db_mock, EquitySnapshotORM
        )

        assert len(equity_orms) == 2, (
            f"Expected 2 EquitySnapshotORM rows, got {len(equity_orms)}"
        )

        first, second = equity_orms

        # run_id parsed correctly from the run_id_str string
        assert first.run_id == _RUN_UUID
        assert second.run_id == _RUN_UUID

        # equity values mirror the equity curve tuple
        assert first.equity == Decimal("10000")
        assert second.equity == Decimal("10500")

        # MVP hard-coded zero fields
        assert first.cash == Decimal("0")
        assert first.unrealised_pnl == Decimal("0")
        assert first.realised_pnl == Decimal("0")
        assert second.cash == Decimal("0")
        assert second.unrealised_pnl == Decimal("0")
        assert second.realised_pnl == Decimal("0")

        # bar_index is the zero-based enumeration index
        assert first.bar_index == 0
        assert second.bar_index == 1

        # timestamps match the equity curve tuple's first element
        assert first.timestamp == _T1
        assert second.timestamp == _T2

        # Strictly rising curve — no drawdown at any bar
        assert first.drawdown_pct == Decimal("0")
        assert second.drawdown_pct == Decimal("0")


# ---------------------------------------------------------------------------
# Test 2 — Early exit when both equity curve and trade history are empty
# ---------------------------------------------------------------------------


class TestEarlyExitWhenNoData:
    """
    When get_equity_curve() and get_trade_history() both return empty lists
    the function must return immediately without touching any DB session.

    This guards the guard clause at the top of _persist_paper_results:
        if not equity_curve and not trade_history:
            log.debug("runs.paper_persist_skipped", reason="no_data")
            return

    The lazy `from api.db.session import get_session_factory` import lives
    below the guard, so get_session_factory is never called in this path.
    """

    @pytest.mark.asyncio
    async def test_persist_empty_equity_curve_skips_db(self) -> None:
        portfolio = _make_portfolio(equity_curve=[], trade_history=[])
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        # get_session_factory must never have been invoked
        get_sf_mock.assert_not_called()

        # No DB operations should have occurred
        db_mock.add_all.assert_not_called()
        db_mock.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — Drawdown peak tracking across a 4-bar equity curve
# ---------------------------------------------------------------------------


class TestDrawdownPeakTracking:
    """
    Verify that peak-tracking drawdown is computed correctly bar by bar.

    Input equity curve: [(t1, 100), (t2, 120), (t3, 90), (t4, 110)]

    Expected drawdown_pct per bar:
        bar 0 (equity=100): peak=100, dd = (100-100)/100 = 0.0
        bar 1 (equity=120): peak=120, dd = (120-120)/120 = 0.0
        bar 2 (equity= 90): peak=120, dd = (120- 90)/120 = 0.25
        bar 3 (equity=110): peak=120, dd = (120-110)/120 = 0.08333...

    The function uses Decimal arithmetic quantized to 8 decimal places.
    We compare with a tolerance of 1e-6 to accommodate quantization.
    """

    @pytest.mark.asyncio
    async def test_persist_drawdown_peak_tracking(self) -> None:
        equity_curve = [
            (_T1, Decimal("100")),
            (_T2, Decimal("120")),
            (_T3, Decimal("90")),
            (_T4, Decimal("110")),
        ]
        portfolio = _make_portfolio(equity_curve=equity_curve)
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        from api.db.models import EquitySnapshotORM

        equity_orms: list[EquitySnapshotORM] = _collect_orms_of_type(
            db_mock, EquitySnapshotORM
        )

        assert len(equity_orms) == 4

        dd_values = [float(orm.drawdown_pct) for orm in equity_orms]

        expected = [
            0.0,          # bar 0: peak=100, equity=100, dd=0
            0.0,          # bar 1: peak=120, equity=120, dd=0
            0.25,         # bar 2: peak=120, equity= 90, dd=(120-90)/120
            1.0 / 12.0,   # bar 3: peak=120, equity=110, dd=(120-110)/120=10/120
        ]

        for bar_idx, (actual, want) in enumerate(zip(dd_values, expected)):
            assert abs(actual - want) < 1e-6, (
                f"bar {bar_idx}: drawdown_pct={actual:.8f}, expected {want:.8f}"
            )


# ---------------------------------------------------------------------------
# Test 4 — Drawdown clamped to [0, 1]
# ---------------------------------------------------------------------------


class TestDrawdownClamped:
    """
    The DB has a CHECK constraint: drawdown_pct BETWEEN 0 AND 1.

    _persist_paper_results applies an explicit clamp:
        dd_pct = max(Decimal("0"), min(dd_pct, Decimal("1")))

    We trigger the upper-clamp branch by feeding a curve where equity drops
    below zero (pathological but arithmetically possible in floating-point
    contexts if a Decimal equity is negative).

    Curve: peak=100, then equity=-50.
        raw dd = (100 - (-50)) / 100 = 1.5  => clamped to 1.0

    The lower clamp (dd < 0) is unreachable in normal peak-tracking logic
    (dd is always >= 0 when equity <= peak), but the code guards both sides.
    We verify the upper clamp, which is the safety-critical DB-constraint path.
    """

    @pytest.mark.asyncio
    async def test_persist_drawdown_clamped_to_zero_one(self) -> None:
        equity_curve = [
            (_T1, Decimal("100")),
            (_T2, Decimal("-50")),  # negative equity -> raw dd = 1.5 > 1
        ]
        portfolio = _make_portfolio(equity_curve=equity_curve)
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        from api.db.models import EquitySnapshotORM

        equity_orms: list[EquitySnapshotORM] = _collect_orms_of_type(
            db_mock, EquitySnapshotORM
        )

        assert len(equity_orms) == 2

        # bar 0: equity == peak, no drawdown
        assert equity_orms[0].drawdown_pct == Decimal("0")

        # bar 1: raw drawdown = (100 - (-50)) / 100 = 1.5, clamped to 1.0
        assert equity_orms[1].drawdown_pct == Decimal("1"), (
            f"Expected drawdown_pct clamped to 1, got {equity_orms[1].drawdown_pct}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Trade history -> TradeORM construction
# ---------------------------------------------------------------------------


class TestTradeOrmConstruction:
    """
    Verify that trade_history entries produce TradeORM rows with correctly
    mapped fields and that they are written via session.add_all().

    The function maps TradeResult fields to TradeORM columns as follows:
        id          = trade.trade_id
        run_id      = uuid.UUID(run_id_str)
        symbol      = trade.symbol
        side        = trade.side.value  (OrderSide enum -> "buy"/"sell" str)
        entry_price = trade.entry_price
        exit_price  = trade.exit_price
        quantity    = trade.quantity
        realised_pnl= trade.realised_pnl
        total_fees  = trade.total_fees
        entry_at    = trade.entry_at
        exit_at     = trade.exit_at
        strategy_id = trade.strategy_id or "unknown"

    We use a real TradeResult (not a mock) so enum serialization is exercised.
    A minimal equity curve is included so the function does not early-exit.
    """

    @pytest.mark.asyncio
    async def test_persist_trades_when_available(self) -> None:
        trade = _make_trade_result(
            symbol="ETH/USDT",
            entry_price=Decimal("3000"),
            exit_price=Decimal("3300"),
            quantity=Decimal("1"),
            realised_pnl=Decimal("295"),
            total_fees=Decimal("5"),
            strategy_id="rsi_mean_reversion",
        )

        # Provide a minimal equity curve so the guard clause does not short-circuit
        equity_curve = [(_T1, Decimal("10000"))]
        portfolio = _make_portfolio(
            equity_curve=equity_curve,
            trade_history=[trade],
        )
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        from api.db.models import TradeORM

        trade_orms: list[TradeORM] = _collect_orms_of_type(db_mock, TradeORM)

        assert len(trade_orms) == 1, (
            f"Expected 1 TradeORM row, got {len(trade_orms)}"
        )

        orm = trade_orms[0]
        assert orm.id == trade.trade_id
        assert orm.run_id == _RUN_UUID
        assert orm.symbol == "ETH/USDT"
        assert orm.side == "buy"              # OrderSide.BUY.value
        assert orm.entry_price == Decimal("3000")
        assert orm.exit_price == Decimal("3300")
        assert orm.quantity == Decimal("1")
        assert orm.realised_pnl == Decimal("295")
        assert orm.total_fees == Decimal("5")
        assert orm.entry_at == _T1
        assert orm.exit_at == _T2
        assert orm.strategy_id == "rsi_mean_reversion"


# ---------------------------------------------------------------------------
# Test 6 — DB session error is caught and not propagated
# ---------------------------------------------------------------------------


class TestSessionErrorCaught:
    """
    When db.commit() raises an exception inside the async context manager,
    _persist_paper_results must:
        1. Not propagate the exception to the caller (swallow it entirely).
        2. Call db.rollback() to clean up the partial transaction.
        3. Call log.exception() to record the failure.

    The function wraps the entire session block in two nested try/excepts:
        outer try: catches factory session failures
        inner try: catches commit/add_all failures, calls rollback + log.exception

    We test the inner path (commit raises after add_all) because that is the
    most common production failure scenario (constraint violation, timeout).
    The commit must fail while the session context manager is still active,
    so we configure raise_on_commit on the db_mock.
    """

    @pytest.mark.asyncio
    async def test_persist_session_error_caught(self) -> None:
        equity_curve = [(_T1, Decimal("10000"))]
        portfolio = _make_portfolio(equity_curve=equity_curve)
        log = _make_log()

        db_error = RuntimeError("DB constraint violation: equity < 0")
        get_sf_mock, db_mock = _make_session_factory(raise_on_commit=db_error)

        with patch(_PATCH_TARGET, get_sf_mock):
            # Must NOT raise — _persist_paper_results absorbs all DB errors
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        # rollback() must have been awaited to clean up the failed transaction
        db_mock.rollback.assert_awaited_once()

        # log.exception() must have been called to record the DB error
        log.exception.assert_called()
