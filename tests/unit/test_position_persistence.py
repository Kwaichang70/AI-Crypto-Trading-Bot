"""
tests/unit/test_position_persistence.py
-----------------------------------------
Unit tests for position snapshot persistence.

Modules under test
------------------
    apps/api/routers/runs.py   -- _persist_paper_results() PositionSnapshotORM path
    packages/trading/backtest.py -- BacktestRunner.last_portfolio property

Coverage groups
---------------
1. TestPositionSnapshotPersistence (4 tests)
   1a. test_persist_paper_includes_position_snapshots
       -- portfolio with 2 positions produces 2 PositionSnapshotORM rows
   1b. test_position_field_mapping
       -- all fields map correctly from Position domain object to PositionSnapshotORM
   1c. test_no_positions_no_error
       -- empty _position_snapshots dict produces no PositionSnapshotORM and no crash
   1d. test_no_portfolio_snapshots_attr
       -- portfolio without _position_snapshots attribute does not crash (hasattr guard)

2. TestBacktestRunnerLastPortfolio (2 tests)
   2a. test_runner_last_portfolio_initially_none
       -- last_portfolio is None before any run() call
   2b. test_runner_last_portfolio_populated
       -- last_portfolio is a PortfolioAccounting instance after a completed run()

Design notes
------------
- _persist_paper_results is an async function; all persistence tests use
  @pytest.mark.asyncio.  asyncio_mode = "auto" in pyproject.toml makes the
  decorator optional but it is included for explicit documentation.
- get_session_factory is imported lazily inside _persist_paper_results via
  `from api.db.session import get_session_factory`.  Because the import
  re-resolves the name from api.db.session on every call, the correct patch
  target is "api.db.session.get_session_factory" (the source-module attribute),
  NOT "api.routers.runs.get_session_factory" (never bound in that namespace).
- Position domain objects are constructed directly (no mocking) so the
  Pydantic field validators are exercised and the mapping assertions are
  faithful to the real data path.
- The early-exit guard in _persist_paper_results short-circuits when both
  equity_curve and trade_history are empty AND there are no orders:
      has_orders = execution_engine is not None and bool(execution_engine.get_all_orders())
      if not equity_curve and not trade_history and not has_orders:
          return
  Position-only tests therefore must provide a minimal equity curve (one point)
  to ensure the DB session is reached.
- BacktestRunner tests use a real _AlwaysHoldStrategy (no signals) and
  deterministic seeded bars to exercise the full wiring without mocking
  internal components.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routers.runs import _persist_paper_results
from common.models import MultiTimeframeContext, OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.models import Position
from trading.portfolio import PortfolioAccounting
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.models import Signal

# ---------------------------------------------------------------------------
# Constants shared across all test classes
# ---------------------------------------------------------------------------

_RUN_ID_STR = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_RUN_UUID = uuid.UUID(_RUN_ID_STR)

_T1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_T2 = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
_T3 = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)

# get_session_factory is imported lazily inside _persist_paper_results from
# api.db.session; patch the attribute on the source module, not api.routers.runs.
_PATCH_TARGET = "api.db.session.get_session_factory"


# ---------------------------------------------------------------------------
# Helpers — mirroring patterns from test_paper_engine_persistence.py
# ---------------------------------------------------------------------------


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


def _make_portfolio_with_positions(
    equity_curve: list[tuple[datetime, Decimal]],
    positions: dict[str, Position] | None = None,
) -> MagicMock:
    """
    Build a MagicMock that satisfies the PortfolioAccounting protocol used by
    _persist_paper_results, with configurable _position_snapshots.

    The function under test accesses:
        portfolio.get_equity_curve()
        portfolio.get_trade_history()
        portfolio._position_snapshots   (via hasattr + .values() iteration)

    Parameters
    ----------
    equity_curve:
        List of (timestamp, equity) tuples returned by get_equity_curve().
        Provide at least one point to bypass the early-exit guard when also
        providing positions.
    positions:
        Dict of symbol -> Position to assign to _position_snapshots.
        If None, defaults to an empty dict.
    """
    portfolio = MagicMock()
    portfolio.get_equity_curve.return_value = equity_curve
    portfolio.get_trade_history.return_value = []
    portfolio._position_snapshots = positions or {}
    return portfolio


def _make_log() -> MagicMock:
    """
    Build a MagicMock for a structlog bound-logger.

    Structlog loggers use attribute access for level methods (log.info, log.debug,
    log.exception).  MagicMock's default attribute creation handles this; we
    assign explicit MagicMocks for determinism so assert_called() works reliably.
    """
    log = MagicMock()
    log.info = MagicMock()
    log.debug = MagicMock()
    log.warning = MagicMock()
    log.exception = MagicMock()
    return log


def _collect_orms_of_type(db_mock: MagicMock, orm_cls: type) -> list[Any]:
    """
    Iterate over all add_all() calls on db_mock and return every ORM object
    whose type matches orm_cls.

    Parameters
    ----------
    db_mock:
        The MagicMock representing the DB session.
    orm_cls:
        The ORM class to filter by (e.g. PositionSnapshotORM).

    Returns
    -------
    list:
        All matching ORM instances that were passed to add_all().
    """
    result = []
    for call_args in db_mock.add_all.call_args_list:
        items = call_args[0][0]
        result.extend(item for item in items if isinstance(item, orm_cls))
    return result


def _make_position(
    *,
    symbol: str = "BTC/USDT",
    run_id: str = _RUN_ID_STR,
    quantity: Decimal = Decimal("0.5"),
    average_entry_price: Decimal = Decimal("50000"),
    current_price: Decimal = Decimal("52000"),
    realised_pnl: Decimal = Decimal("0"),
    unrealised_pnl: Decimal = Decimal("1000"),
    total_fees_paid: Decimal = Decimal("25"),
    opened_at: datetime = _T1,
    updated_at: datetime = _T2,
) -> Position:
    """
    Build a minimal Position domain object with controllable fields.

    Timestamps are pinned by default for determinism.
    """
    return Position(
        symbol=symbol,
        run_id=run_id,
        quantity=quantity,
        average_entry_price=average_entry_price,
        current_price=current_price,
        realised_pnl=realised_pnl,
        unrealised_pnl=unrealised_pnl,
        total_fees_paid=total_fees_paid,
        opened_at=opened_at,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# Strategy stub for BacktestRunner tests (mirrors test_order_fill_persistence.py)
# ---------------------------------------------------------------------------


class _AlwaysHoldStrategy(BaseStrategy):
    """
    Minimal strategy that never emits signals.

    Used for BacktestRunner tests where only the infrastructure property
    (last_portfolio) is exercised, not trading logic.
    """

    metadata = StrategyMetadata(
        name="always_hold_position_persistence_test",
        version="1.0.0",
        description="Stub strategy for position persistence tests",
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
# Class 1: TestPositionSnapshotPersistence
# ---------------------------------------------------------------------------


class TestPositionSnapshotPersistence:
    """
    Verify that _persist_paper_results() correctly maps Position domain objects
    to PositionSnapshotORM rows and writes them via session.add_all().

    The function reads positions from portfolio._position_snapshots (a dict of
    symbol -> Position) and constructs one PositionSnapshotORM row per entry.
    The mapping is protected by a hasattr() guard for safety.

    All tests supply a minimal equity curve (one point) to bypass the
    early-exit guard that fires when equity_curve, trade_history, and
    execution_engine all yield no data.
    """

    def _minimal_equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """One equity point — enough to pass the early-exit guard."""
        return [(_T1, Decimal("10000"))]

    @pytest.mark.asyncio
    async def test_persist_paper_includes_position_snapshots(self) -> None:
        """
        A portfolio with 2 positions (BTC/USDT and ETH/USDT) in
        _position_snapshots must produce exactly 2 PositionSnapshotORM rows
        written via db.add_all().

        Verify:
        - add_all() was called at all (DB session reached).
        - Exactly 2 PositionSnapshotORM rows are collected.
        - Each row's symbol matches one of the two positions.
        - run_id on each row is the parsed UUID, not the raw string.
        """
        from api.db.models import PositionSnapshotORM

        btc_pos = _make_position(symbol="BTC/USDT", quantity=Decimal("0.5"))
        eth_pos = _make_position(
            symbol="ETH/USDT",
            quantity=Decimal("2.0"),
            average_entry_price=Decimal("3000"),
            current_price=Decimal("3200"),
            unrealised_pnl=Decimal("400"),
        )

        portfolio = _make_portfolio_with_positions(
            equity_curve=self._minimal_equity_curve(),
            positions={"BTC/USDT": btc_pos, "ETH/USDT": eth_pos},
        )
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        assert db_mock.add_all.called, "add_all() was never called on the DB session"

        position_orms: list[PositionSnapshotORM] = _collect_orms_of_type(
            db_mock, PositionSnapshotORM
        )

        assert len(position_orms) == 2, (
            f"Expected 2 PositionSnapshotORM rows, got {len(position_orms)}"
        )

        symbols_persisted = {orm.symbol for orm in position_orms}
        assert symbols_persisted == {"BTC/USDT", "ETH/USDT"}, (
            f"Expected symbols {{'BTC/USDT', 'ETH/USDT'}}, got {symbols_persisted}"
        )

        for orm in position_orms:
            assert orm.run_id == _RUN_UUID, (
                f"run_id UUID mismatch on {orm.symbol}: "
                f"{orm.run_id!r} != {_RUN_UUID!r}"
            )

    @pytest.mark.asyncio
    async def test_position_field_mapping(self) -> None:
        """
        All fields on PositionSnapshotORM must match the corresponding field
        on the source Position domain object.

        Field mapping enforced by _persist_paper_results (runs.py):
            run_id              = uuid.UUID(run_id_str)
            symbol              = pos.symbol
            quantity            = pos.quantity
            average_entry_price = pos.average_entry_price
            current_price       = pos.current_price
            unrealised_pnl      = pos.unrealised_pnl
            realised_pnl        = pos.realised_pnl
            total_fees_paid     = pos.total_fees_paid
            opened_at           = pos.opened_at
            snapshot_at         = datetime.now(tz=UTC)  (capture time, not pos field)

        We verify all domain-mapped fields exactly and assert snapshot_at is
        a timezone-aware datetime (it records the exact moment of persistence).
        """
        from api.db.models import PositionSnapshotORM

        position = _make_position(
            symbol="BTC/USDT",
            run_id=_RUN_ID_STR,
            quantity=Decimal("0.25"),
            average_entry_price=Decimal("48000"),
            current_price=Decimal("51000"),
            realised_pnl=Decimal("150"),
            unrealised_pnl=Decimal("750"),
            total_fees_paid=Decimal("12"),
            opened_at=_T1,
            updated_at=_T2,
        )

        portfolio = _make_portfolio_with_positions(
            equity_curve=self._minimal_equity_curve(),
            positions={"BTC/USDT": position},
        )
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        position_orms: list[PositionSnapshotORM] = _collect_orms_of_type(
            db_mock, PositionSnapshotORM
        )

        assert len(position_orms) == 1, (
            f"Expected 1 PositionSnapshotORM row, got {len(position_orms)}"
        )

        orm = position_orms[0]

        # run_id must be the UUID parsed from the run_id_str parameter
        assert orm.run_id == _RUN_UUID, (
            f"run_id mismatch: {orm.run_id!r} != {_RUN_UUID!r}"
        )

        # Domain-mapped fields must exactly match the Position attributes
        assert orm.symbol == "BTC/USDT", (
            f"symbol mismatch: {orm.symbol!r} != 'BTC/USDT'"
        )
        assert orm.quantity == Decimal("0.25"), (
            f"quantity mismatch: {orm.quantity} != 0.25"
        )
        assert orm.average_entry_price == Decimal("48000"), (
            f"average_entry_price mismatch: {orm.average_entry_price} != 48000"
        )
        assert orm.current_price == Decimal("51000"), (
            f"current_price mismatch: {orm.current_price} != 51000"
        )
        assert orm.unrealised_pnl == Decimal("750"), (
            f"unrealised_pnl mismatch: {orm.unrealised_pnl} != 750"
        )
        assert orm.realised_pnl == Decimal("150"), (
            f"realised_pnl mismatch: {orm.realised_pnl} != 150"
        )
        assert orm.total_fees_paid == Decimal("12"), (
            f"total_fees_paid mismatch: {orm.total_fees_paid} != 12"
        )
        assert orm.opened_at == _T1, (
            f"opened_at mismatch: {orm.opened_at} != {_T1}"
        )

        # snapshot_at is set to datetime.now(tz=UTC) inside the function — it
        # must be a timezone-aware datetime and must not equal opened_at
        # (which was pinned to 2024-01-01, well before the test runs).
        assert isinstance(orm.snapshot_at, datetime), (
            f"snapshot_at must be a datetime, got {type(orm.snapshot_at).__name__}"
        )
        assert orm.snapshot_at.tzinfo is not None, (
            "snapshot_at must be timezone-aware (UTC)"
        )

    @pytest.mark.asyncio
    async def test_no_positions_no_error(self) -> None:
        """
        A portfolio with an empty _position_snapshots dict must produce zero
        PositionSnapshotORM rows and must not raise.

        This is the common case immediately after engine start — before any
        fills have been processed, _position_snapshots is empty.  The function
        must iterate over an empty dict without crashing and without calling
        add_all() with an empty position list (add_all([]) is a no-op but we
        verify the specific absence of PositionSnapshotORM rows).
        """
        from api.db.models import PositionSnapshotORM

        portfolio = _make_portfolio_with_positions(
            equity_curve=self._minimal_equity_curve(),
            positions={},  # explicitly empty
        )
        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            # Must NOT raise even when positions dict is empty
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        position_orms: list[PositionSnapshotORM] = _collect_orms_of_type(
            db_mock, PositionSnapshotORM
        )

        assert len(position_orms) == 0, (
            f"Expected 0 PositionSnapshotORM rows for empty positions, "
            f"got {len(position_orms)}"
        )

    @pytest.mark.asyncio
    async def test_no_portfolio_snapshots_attr(self) -> None:
        """
        A portfolio MagicMock WITHOUT a _position_snapshots attribute must not
        crash _persist_paper_results.

        The production code guards with:
            if hasattr(portfolio, '_position_snapshots'):
                for pos in portfolio._position_snapshots.values(): ...

        This test verifies the guard works correctly when the portfolio object
        does not expose the internal attribute (e.g., a future protocol
        implementation or a plain MagicMock that has not been configured).

        We use spec=object to ensure _position_snapshots is genuinely absent
        rather than auto-created by MagicMock's default attribute machinery.
        """
        from api.db.models import PositionSnapshotORM

        # A plain MagicMock auto-creates attributes on access; use spec=object
        # to produce a mock that raises AttributeError on unknown attributes,
        # which is what hasattr() catches.
        portfolio = MagicMock(spec=object)

        # We need get_equity_curve and get_trade_history to be callable so the
        # function does not fail before reaching the position section.
        # Attach them explicitly as MagicMocks returning appropriate values.
        portfolio.get_equity_curve = MagicMock(
            return_value=self._minimal_equity_curve()
        )
        portfolio.get_trade_history = MagicMock(return_value=[])

        log = _make_log()
        get_sf_mock, db_mock = _make_session_factory()

        with patch(_PATCH_TARGET, get_sf_mock):
            # Must NOT raise — hasattr guard must protect against missing attr
            await _persist_paper_results(
                run_id_str=_RUN_ID_STR,
                portfolio=portfolio,
                log=log,
            )

        position_orms: list[PositionSnapshotORM] = _collect_orms_of_type(
            db_mock, PositionSnapshotORM
        )

        assert len(position_orms) == 0, (
            f"Expected 0 PositionSnapshotORM rows when _position_snapshots "
            f"attribute is absent, got {len(position_orms)}"
        )


# ---------------------------------------------------------------------------
# Class 2: TestBacktestRunnerLastPortfolio
# ---------------------------------------------------------------------------


class TestBacktestRunnerLastPortfolio:
    """
    Verify the BacktestRunner.last_portfolio property lifecycle.

    BacktestRunner exposes last_portfolio to allow _persist_backtest_results()
    to access _position_snapshots without requiring a return value change in
    BacktestRunner.run().

    From backtest.py:
        # __init__:
        self._last_portfolio: PortfolioAccounting | None = None

        # run():
        ...
        self._last_portfolio = portfolio
        return result

        # property:
        @property
        def last_portfolio(self) -> PortfolioAccounting | None:
            return self._last_portfolio

    The tests use _AlwaysHoldStrategy (no signals, no trades) with 200
    deterministic seeded hourly bars.  200 bars exceeds the 50-bar warmup
    requirement (warmup = max(0 * 2, 50) = 50 for min_bars_required=0).
    """

    def test_runner_last_portfolio_initially_none(self) -> None:
        """
        Before any run() call, last_portfolio must be None.

        The property is initialized to None in __init__ and must not be
        populated until a run() call completes.  This test verifies the
        initial state without executing any backtest.
        """
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy("hold_position_test")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
        )

        assert runner.last_portfolio is None, (
            f"last_portfolio should be None before run(), "
            f"got {runner.last_portfolio!r}"
        )

    @pytest.mark.asyncio
    async def test_runner_last_portfolio_populated(self) -> None:
        """
        After a successful BacktestRunner.run(), last_portfolio must be a
        non-None PortfolioAccounting instance.

        Post-run assertions:
        1. last_portfolio is not None.
        2. last_portfolio is a PortfolioAccounting instance (not a mock or stub).
        3. get_equity_curve() is callable and returns a non-empty list
           (the runner always records at least one equity point for the
           initial capital at run start).
        4. _position_snapshots is a dict (the attribute exists and is accessible).
        5. With _AlwaysHoldStrategy (no signals), _position_snapshots is empty
           because no fills were processed.
        """
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy("hold_position_test")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=42,
        )

        bars_by_symbol = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}

        await runner.run(bars_by_symbol)

        portfolio = runner.last_portfolio

        assert portfolio is not None, (
            "last_portfolio must be non-None after a completed run()"
        )
        assert isinstance(portfolio, PortfolioAccounting), (
            f"last_portfolio must be a PortfolioAccounting instance, "
            f"got {type(portfolio).__name__}"
        )

        # get_equity_curve() must work and return at least the initial capital point
        equity_curve = portfolio.get_equity_curve()
        assert isinstance(equity_curve, list), (
            f"get_equity_curve() must return a list, got {type(equity_curve).__name__}"
        )
        assert len(equity_curve) >= 1, (
            "get_equity_curve() must have at least one point (initial capital)"
        )

        # _position_snapshots must be present and be a dict
        assert hasattr(portfolio, "_position_snapshots"), (
            "PortfolioAccounting must have _position_snapshots attribute"
        )
        assert isinstance(portfolio._position_snapshots, dict), (
            f"_position_snapshots must be a dict, "
            f"got {type(portfolio._position_snapshots).__name__}"
        )

        # _AlwaysHoldStrategy emits no signals -> no fills -> no positions recorded
        assert portfolio._position_snapshots == {}, (
            f"AlwaysHold strategy should produce no position snapshots, "
            f"got {list(portfolio._position_snapshots.keys())}"
        )
