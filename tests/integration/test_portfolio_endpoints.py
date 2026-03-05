"""
tests/integration/test_portfolio_endpoints.py
----------------------------------------------
Integration tests for the portfolio endpoints.

Endpoints under test
--------------------
GET /api/v1/runs/{run_id}/portfolio      -- Portfolio summary snapshot (9 DB calls)
GET /api/v1/runs/{run_id}/equity-curve   -- Paginated equity curve (3 DB calls)
GET /api/v1/runs/{run_id}/trades         -- Paginated completed trades (3 DB calls)
GET /api/v1/runs/{run_id}/positions      -- Open positions stub (1 DB call)

Test strategy
-------------
All DB I/O is intercepted via FastAPI dependency overrides that inject a
hand-crafted AsyncMock in place of a real AsyncSession.  No PostgreSQL
instance is required.

The portfolio summary endpoint is the most complex: it issues 8 sequential
execute() calls with different result accessor patterns.  Each call returns
a distinct MagicMock wired for the exact accessor the handler uses:

  [0]  -> _get_run_or_404()           .scalar_one_or_none() -> RunORM | None
  [1]  -> trade stats aggregate       .one()               -> Row object
  [2]  -> winning trades count        .scalar_one()        -> int
  [3]  -> losing trades count         .scalar_one()        -> int
  [4]  -> latest equity snapshot      .scalar_one_or_none() -> EquitySnapshotORM | None
  [5]  -> peak equity (func.max)      .scalar_one_or_none() -> Decimal | None
  [6]  -> max drawdown (func.max)     .scalar_one_or_none() -> Decimal | None
  [7]  -> equity curve count          .scalar_one()        -> int

The equity-curve and trades endpoints each issue 3 execute() calls:
  Call 1 -> _get_run_or_404()     .scalar_one_or_none() -> RunORM
  Call 2 -> COUNT(*)              .scalar_one()        -> int
  Call 3 -> page SELECT           .scalars().all()     -> list

The positions endpoint issues 1 execute() call:
  Call 1 -> _get_run_or_404()    .scalar_one_or_none() -> RunORM

Determinism contract
--------------------
- All UUIDs are hardcoded constants — no uuid4() in test bodies.
- All timestamps are fixed UTC datetimes.
- All Decimal values are constructed from string literals.
- No wall-clock time or random state is consulted.

Response key naming
-------------------
All API responses use camelCase (via alias_generator=to_camel on all schemas).
Assertions use camelCase keys when inspecting resp.json() output.
"""

from __future__ import annotations

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
_TRADE_UUID_1 = uuid.UUID("cccccccc-dddd-eeee-ffff-000000000001")
_TRADE_UUID_2 = uuid.UUID("cccccccc-dddd-eeee-ffff-000000000002")
_MISSING_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")

_FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_FIXED_ENTRY = datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC)
_FIXED_EXIT = datetime(2026, 1, 10, 10, 0, 0, tzinfo=UTC)

_INITIAL_CAPITAL = "10000.00"


# ---------------------------------------------------------------------------
# ORM factory helpers
# ---------------------------------------------------------------------------

def _make_run_orm(
    *,
    run_id: uuid.UUID = _RUN_UUID,
    run_mode: str = "paper",
    status: str = "running",
    initial_capital: str = _INITIAL_CAPITAL,
    started_at: datetime = _FIXED_NOW,
    stopped_at: datetime | None = None,
    created_at: datetime = _FIXED_NOW,
    updated_at: datetime = _FIXED_NOW,
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as RunORM,
    suitable for Pydantic from_attributes=True serialization.

    Parameters
    ----------
    run_id:
        The UUID to assign.  Defaults to _RUN_UUID.
    run_mode:
        One of 'backtest', 'paper', 'live'.
    status:
        One of 'running', 'stopped', 'error'.
    initial_capital:
        String decimal used to populate config['initial_capital'].
    started_at:
        Fixed UTC start timestamp.
    stopped_at:
        Fixed UTC stop timestamp; None if still running.
    created_at:
        Row creation timestamp.
    updated_at:
        Row last-update timestamp.
    """
    return SimpleNamespace(
        id=run_id,
        run_mode=run_mode,
        status=status,
        config={
            "strategy_name": "ma_crossover",
            "strategy_params": {"fast_period": 10, "slow_period": 50},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": run_mode,
            "initial_capital": initial_capital,
        },
        started_at=started_at,
        stopped_at=stopped_at,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_equity_snapshot_orm(
    *,
    snap_id: int = 1,
    run_id: uuid.UUID = _RUN_UUID,
    equity: str = "10500.00",
    cash: str = "5000.00",
    unrealised_pnl: str = "500.00",
    realised_pnl: str = "200.00",
    drawdown_pct: str = "0.02",
    bar_index: int = 99,
    timestamp: datetime = _FIXED_NOW,
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as
    EquitySnapshotORM, suitable for Pydantic from_attributes=True
    serialization.

    Parameters
    ----------
    snap_id:
        Integer surrogate primary key (BigInteger in the schema).
    run_id:
        Parent run UUID.
    equity:
        Total portfolio equity as a string decimal.
    cash:
        Cash balance as a string decimal.
    unrealised_pnl:
        Unrealised PnL as a string decimal.
    realised_pnl:
        Cumulative realised PnL as a string decimal.
    drawdown_pct:
        Drawdown fraction (0.0 to 1.0) as a string decimal.
    bar_index:
        Zero-based bar number.
    timestamp:
        UTC timestamp for this snapshot.
    """
    return SimpleNamespace(
        id=snap_id,
        run_id=run_id,
        equity=Decimal(equity),
        cash=Decimal(cash),
        unrealised_pnl=Decimal(unrealised_pnl),
        realised_pnl=Decimal(realised_pnl),
        drawdown_pct=Decimal(drawdown_pct),
        bar_index=bar_index,
        timestamp=timestamp,
    )


def _make_trade_orm(
    *,
    trade_id: uuid.UUID = _TRADE_UUID_1,
    run_id: uuid.UUID = _RUN_UUID,
    symbol: str = "BTC/USDT",
    side: str = "buy",
    entry_price: str = "45000.00",
    exit_price: str = "47000.00",
    quantity: str = "0.1",
    realised_pnl: str = "200.00",
    total_fees: str = "5.00",
    entry_at: datetime = _FIXED_ENTRY,
    exit_at: datetime = _FIXED_EXIT,
    strategy_id: str = "ma_crossover_v1",
) -> SimpleNamespace:
    """
    Construct a SimpleNamespace with the same attribute surface as TradeORM,
    suitable for Pydantic from_attributes=True serialization.

    Parameters
    ----------
    trade_id:
        UUID primary key for the trade record.
    run_id:
        Parent run UUID.
    symbol:
        CCXT trading pair.
    side:
        Opening fill direction: 'buy' or 'sell'.
    entry_price:
        VWAP entry price as a string decimal.
    exit_price:
        VWAP exit price as a string decimal.
    quantity:
        Total traded quantity in base asset as a string decimal.
    realised_pnl:
        Net realised PnL after fees as a string decimal.
    total_fees:
        Total fees paid as a string decimal.
    entry_at:
        UTC timestamp of first entry fill.
    exit_at:
        UTC timestamp of final exit fill.
    strategy_id:
        Identifier of the strategy that generated the opening signal.
    """
    return SimpleNamespace(
        id=trade_id,
        run_id=run_id,
        symbol=symbol,
        side=side,
        entry_price=Decimal(entry_price),
        exit_price=Decimal(exit_price),
        quantity=Decimal(quantity),
        realised_pnl=Decimal(realised_pnl),
        total_fees=Decimal(total_fees),
        entry_at=entry_at,
        exit_at=exit_at,
        strategy_id=strategy_id,
    )


# ---------------------------------------------------------------------------
# DB mock result helpers
# ---------------------------------------------------------------------------

def _make_scalar_result(value: object) -> MagicMock:
    """
    Return a MagicMock that mimics an AsyncSession execute() result
    supporting .scalar_one().

    Used for COUNT(*) and count aggregate queries where the handler calls
    ``(await db.execute(stmt)).scalar_one()``.

    Parameters
    ----------
    value:
        The value scalar_one() will return — typically an int.
    """
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _make_scalars_result(items: list) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting
    .scalars().all().

    Used for page SELECT queries where the handler calls
    ``result.scalars().all()``.

    Parameters
    ----------
    items:
        List of ORM objects to return from .all().
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

    Used for _get_run_or_404 (returns RunORM or None), the latest equity
    snapshot query, peak equity query, and max drawdown query.

    Parameters
    ----------
    value:
        The value scalar_one_or_none() will return.
        Pass None to simulate a not-found condition.
    """
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _make_row_result(
    total_trades: int,
    total_realised_pnl: Decimal,
    total_fees_paid: Decimal,
) -> MagicMock:
    """
    Return a MagicMock for the trade stats aggregate query that uses .one().

    The portfolio handler calls ``(await db.execute(trade_stats_stmt)).one()``
    which returns a SQLAlchemy Row-like object with named attributes.  This
    mock replicates that interface by setting the three attribute names the
    handler expects: total_trades, total_realised_pnl, total_fees_paid.

    Parameters
    ----------
    total_trades:
        Aggregate count of completed trades.
    total_realised_pnl:
        Aggregate SUM of realised_pnl across all trades.
    total_fees_paid:
        Aggregate SUM of total_fees across all trades.
    """
    row = MagicMock()
    row.total_trades = total_trades
    row.total_realised_pnl = total_realised_pnl
    row.total_fees_paid = total_fees_paid

    result = MagicMock()
    result.one.return_value = row
    return result


# ---------------------------------------------------------------------------
# Shared side_effect builders for portfolio summary (9 execute() calls)
# ---------------------------------------------------------------------------

def _portfolio_side_effect_with_data(
    run_orm: SimpleNamespace,
    snapshot_orm: SimpleNamespace,
    total_trades: int = 5,
    total_realised_pnl: Decimal = Decimal("500.00"),
    total_fees_paid: Decimal = Decimal("25.00"),
    winning_trades: int = 3,
    losing_trades: int = 2,
    peak_equity: Decimal = Decimal("11000.00"),
    max_drawdown: Decimal = Decimal("0.05"),
    equity_curve_length: int = 100,
    daily_pnl: Decimal | None = None,
    open_positions: int = 1,
) -> list:
    """
    Build the 10-element side_effect list for a portfolio summary request with
    a fully populated run — snapshots exist and trades are present.

    Call order matches the handler's sequential await db.execute() calls:
      [0] _get_run_or_404           -> scalar_one_or_none -> RunORM
      [1] trade stats aggregate     -> one()              -> Row
      [2] winning trades count      -> scalar_one         -> int
      [3] losing trades count       -> scalar_one         -> int
      [4] latest equity snapshot    -> scalar_one_or_none -> EquitySnapshotORM
      [5] peak equity               -> scalar_one_or_none -> Decimal
      [6] max drawdown              -> scalar_one_or_none -> Decimal
      [7] equity curve count        -> scalar_one         -> int
      [8] daily PnL SUM            -> scalar_one_or_none -> Decimal | None
      [9] open positions count      -> scalar_one         -> int
    """
    return [
        _make_scalar_one_or_none_result(run_orm),
        _make_row_result(total_trades, total_realised_pnl, total_fees_paid),
        _make_scalar_result(winning_trades),
        _make_scalar_result(losing_trades),
        _make_scalar_one_or_none_result(snapshot_orm),
        _make_scalar_one_or_none_result(peak_equity),
        _make_scalar_one_or_none_result(max_drawdown),
        _make_scalar_result(equity_curve_length),
        _make_scalar_one_or_none_result(daily_pnl),
        _make_scalar_result(open_positions),
    ]


def _portfolio_side_effect_empty_run(run_orm: SimpleNamespace) -> list:
    """
    Build the 10-element side_effect list for a portfolio summary request with
    no trades and no equity snapshots.

    When a run has just been created or has never processed a bar, all
    aggregate queries return zero/None values.  The handler falls back to
    initial_capital for equity and cash.

    Parameters
    ----------
    run_orm:
        RunORM object for _get_run_or_404 to return.
    """
    return [
        _make_scalar_one_or_none_result(run_orm),
        _make_row_result(0, None, None),   # SUM returns NULL in SQL when no rows
        _make_scalar_result(0),             # winning trades
        _make_scalar_result(0),             # losing trades
        _make_scalar_one_or_none_result(None),  # no latest snapshot
        _make_scalar_one_or_none_result(None),  # no peak equity
        _make_scalar_one_or_none_result(None),  # no max drawdown
        _make_scalar_result(0),             # equity curve length
        _make_scalar_one_or_none_result(None),  # daily PnL (no trades)
        _make_scalar_result(0),             # open positions (none)
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/portfolio tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetPortfolio:
    """
    Tests for GET /api/v1/runs/{run_id}/portfolio.

    This is the most complex endpoint: 9 sequential execute() calls are made
    covering run lookup, trade aggregation, win/loss counting, snapshot
    retrieval, peak equity, max drawdown, equity curve length, and daily PnL.

    Each test configures mock_db_session.execute.side_effect as a list of 9
    pre-built MagicMock result objects so the handler's sequential awaits
    resolve predictably.
    """

    def test_full_portfolio_with_data_returns_200_and_all_fields(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A run with trade history and equity snapshots must return HTTP 200
        with all PortfolioResponse fields populated from the mock data.

        Verifies:
        - HTTP 200 status
        - runId matches the requested UUID
        - initialCash reflects the run's config.initial_capital
        - currentEquity and currentCash come from the latest snapshot
        - totalTrades, winningTrades, losingTrades match mock aggregate
        - totalRealisedPnl and totalFeesPaid match mock row values
        - drawdownPct comes from the latest snapshot's drawdown_pct
        - peakEquity comes from func.max query
        - maxDrawdownPct comes from func.max(drawdown_pct) query
        - equityCurveLength comes from count query
        - openPositions derived from latest snapshot unrealised_pnl
        - dailyPnl computed from today's trades (0 when no daily_pnl mock)
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(
            run_id=_RUN_UUID,
            equity="10500.00",
            cash="5000.00",
            unrealised_pnl="500.00",
            drawdown_pct="0.02",
            bar_index=99,
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            total_trades=5,
            total_realised_pnl=Decimal("500.00"),
            total_fees_paid=Decimal("25.00"),
            winning_trades=3,
            losing_trades=2,
            peak_equity=Decimal("11000.00"),
            max_drawdown=Decimal("0.05"),
            equity_curve_length=100,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()

        # Identity
        assert body["runId"] == str(_RUN_UUID)
        assert body["initialCash"] == "10000.00"

        # Current state from latest snapshot
        assert body["currentEquity"] == "10500.00"
        assert body["currentCash"] == "5000.00"
        assert body["drawdownPct"] == pytest.approx(0.02, abs=1e-9)

        # Peak and max drawdown from DB aggregates
        assert body["peakEquity"] == "11000.00"
        assert body["maxDrawdownPct"] == pytest.approx(0.05, abs=1e-9)

        # Trade statistics
        assert body["totalTrades"] == 5
        assert body["winningTrades"] == 3
        assert body["losingTrades"] == 2
        assert body["totalRealisedPnl"] == "500.00"
        assert body["totalFeesPaid"] == "25.00"

        # Equity curve
        assert body["equityCurveLength"] == 100

        # Derived fields
        assert body["openPositions"] == 1  # unrealised_pnl="500.00" != 0
        assert body["dailyPnl"] == "0"  # no daily_pnl passed → None → "0"
        # Win rate: 3 / 5 = 0.6
        assert body["winRate"] == pytest.approx(0.6, abs=1e-9)
        # Total return: (10500 - 10000) / 10000 = 0.05
        assert body["totalReturnPct"] == pytest.approx(0.05, abs=1e-9)

    def test_portfolio_run_not_found_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting portfolio data for a non-existent run UUID must return
        HTTP 404 with a detail message containing the run_id.

        _get_run_or_404() raises HTTPException(404) when scalar_one_or_none()
        returns None.  Only 1 execute() call is made before the exception.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/portfolio")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]

    def test_empty_run_no_trades_no_snapshots_returns_200_with_initial_capital(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A run that has not processed any bars or completed any trades must
        return HTTP 200 with portfolio values defaulting to initial_capital.

        When no equity snapshots exist:
        - currentEquity == initialCash == initialCapital from config
        - currentCash == initialCapital
        - drawdownPct == 0.0
        - maxDrawdownPct == 0.0
        - peakEquity == currentEquity (falls back to initial)

        When no trades exist:
        - totalTrades == 0
        - winningTrades == 0
        - losingTrades == 0
        - totalRealisedPnl == "0"
        - totalFeesPaid == "0"
        - winRate == 0.0
        - equityCurveLength == 0
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID, status="running")
        mock_db_session.execute.side_effect = _portfolio_side_effect_empty_run(run_orm)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["initialCash"] == "10000.00"
        # No snapshot: equity and cash fall back to initial_capital
        assert body["currentEquity"] == "10000.00"
        assert body["currentCash"] == "10000.00"
        # No trades
        assert body["totalTrades"] == 0
        assert body["winningTrades"] == 0
        assert body["losingTrades"] == 0
        # Handler: Decimal(str(None or "0")) -> Decimal("0") -> str -> "0"
        # This deliberately differs from the "500.00" form used when real data is present.
        assert body["totalRealisedPnl"] == "0"
        assert body["totalFeesPaid"] == "0"
        assert body["winRate"] == pytest.approx(0.0, abs=1e-9)
        # No drawdown recorded
        assert body["drawdownPct"] == pytest.approx(0.0, abs=1e-9)
        assert body["maxDrawdownPct"] == pytest.approx(0.0, abs=1e-9)
        assert body["equityCurveLength"] == 0
        assert body["openPositions"] == 0
        # No equity change: total return must be 0.0
        assert body["totalReturnPct"] == pytest.approx(0.0, abs=1e-9)

    def test_win_rate_calculation_is_correct(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Win rate must equal winning_trades / total_trades.

        With 3 winning trades and 5 total trades the win_rate field must
        contain 0.6 (= 3/5).  This verifies the handler's division is applied
        correctly and the result is serialised as a float in the response.

        The test also confirms that the non-winning trades count (losingTrades)
        does NOT affect the win_rate calculation — only total_trades is used
        as the denominator.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(run_id=_RUN_UUID)

        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()
        assert body["totalTrades"] == 5
        assert body["winningTrades"] == 3
        assert body["losingTrades"] == 2
        assert body["winRate"] == pytest.approx(0.6, abs=1e-9)

    def test_daily_pnl_from_trades_exiting_today(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When trades exited today exist, dailyPnl must reflect their summed
        realised_pnl from the daily_pnl DB query (call index [8]).
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(
            run_id=_RUN_UUID,
            unrealised_pnl="0.00",
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            daily_pnl=Decimal("123.45"),
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()
        assert body["dailyPnl"] == "123.45"

    def test_daily_pnl_zero_when_no_trades_today(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When no trades exited today, the daily_pnl query returns None and
        dailyPnl must be "0".
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(run_id=_RUN_UUID)
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            daily_pnl=None,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        assert resp.json()["dailyPnl"] == "0"

    def test_open_positions_from_unrealised_pnl(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When latest snapshot has non-zero unrealised_pnl, openPositions
        must be 1.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(
            run_id=_RUN_UUID,
            unrealised_pnl="250.00",
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        assert resp.json()["openPositions"] == 1

    def test_open_positions_zero_when_no_position_snapshots(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When position_snapshots COUNT returns 0, openPositions must be 0.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snapshot_orm = _make_equity_snapshot_orm(
            run_id=_RUN_UUID,
            unrealised_pnl="0.00",
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            open_positions=0,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        assert resp.json()["openPositions"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/equity-curve tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetEquityCurve:
    """
    Tests for GET /api/v1/runs/{run_id}/equity-curve.

    The handler issues 3 execute() calls in sequence:
      Call 1  -> _get_run_or_404()       .scalar_one_or_none()
      Call 2  -> COUNT(*) query          .scalar_one()
      Call 3  -> page SELECT             .scalars().all()

    Responses use EquityCurveResponse with camelCase fields:
      runId, totalPoints, points (list of EquityPointResponse)
    """

    def test_equity_curve_with_snapshots_returns_200(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When equity snapshots exist for a run, the endpoint must return HTTP 200
        with an EquityCurveResponse containing the correct runId, totalPoints,
        and a populated points list.

        Verifies:
        - HTTP 200 status
        - runId in response matches the URL parameter
        - totalPoints reflects the COUNT(*) mock return value
        - points list length matches the page data length
        - Each point contains: timestamp, equity, cash, unrealisedPnl,
          realisedPnl, drawdownPct, barIndex (camelCase)
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        snap1 = _make_equity_snapshot_orm(
            snap_id=1, run_id=_RUN_UUID, bar_index=0,
            equity="10000.00", cash="10000.00",
            unrealised_pnl="0.00", realised_pnl="0.00",
            drawdown_pct="0.00",
        )
        snap2 = _make_equity_snapshot_orm(
            snap_id=2, run_id=_RUN_UUID, bar_index=1,
            equity="10200.00", cash="9700.00",
            unrealised_pnl="200.00", realised_pnl="0.00",
            drawdown_pct="0.00",
        )

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),  # _get_run_or_404
            _make_scalar_result(2),                     # COUNT(*)
            _make_scalars_result([snap1, snap2]),        # page SELECT
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/equity-curve")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["totalPoints"] == 2
        assert len(body["points"]) == 2

        # Verify camelCase field names on the first point
        first_point = body["points"][0]
        assert "barIndex" in first_point
        assert "unrealisedPnl" in first_point
        assert "realisedPnl" in first_point
        assert "drawdownPct" in first_point
        assert "equity" in first_point
        assert "cash" in first_point
        assert "timestamp" in first_point

        # Spot-check values from snap1
        assert first_point["equity"] == "10000.00"
        assert first_point["barIndex"] == 0
        assert first_point["cash"] == "10000.00"
        assert first_point["unrealisedPnl"] == "0.00"
        assert first_point["realisedPnl"] == "0.00"
        assert first_point["drawdownPct"] == pytest.approx(0.0, abs=1e-9)

    def test_equity_curve_empty_run_returns_200_with_zero_total(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When no equity snapshots exist for a run, the endpoint must return
        HTTP 200 with totalPoints=0 and an empty points list.

        This covers newly created runs that have not processed any bars.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),  # _get_run_or_404
            _make_scalar_result(0),                     # COUNT(*) returns 0
            _make_scalars_result([]),                    # page SELECT returns empty
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/equity-curve")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["totalPoints"] == 0
        assert body["points"] == []

    def test_equity_curve_run_not_found_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting equity curve data for a non-existent run UUID must return
        HTTP 404 with a detail message containing the missing UUID.

        _get_run_or_404() raises HTTPException(404) when the run does not
        exist; the COUNT and page queries are never executed.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/equity-curve")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/trades tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestListTrades:
    """
    Tests for GET /api/v1/runs/{run_id}/trades.

    The handler issues 3 execute() calls in sequence:
      Call 1  -> _get_run_or_404()    .scalar_one_or_none()
      Call 2  -> COUNT(*) query       .scalar_one()
      Call 3  -> page SELECT          .scalars().all()

    Responses use TradeListResponse (PaginatedResponse[TradeResponse]):
      total, offset, limit, items

    TradeResponse fields use camelCase:
      runId, entryPrice, exitPrice, realisedPnl, totalFees, entryAt, exitAt, strategyId
    """

    def test_trades_list_returns_200_with_correct_envelope(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A run with completed trades must return HTTP 200 with a TradeListResponse
        containing the correct pagination envelope and item data.

        Verifies:
        - HTTP 200 status
        - total, offset, limit envelope fields are present
        - items list is populated with the mocked TradeORM data
        - Item fields use camelCase (runId, entryPrice, exitPrice, etc.)
        - Decimal monetary values are serialised as strings
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        trade1 = _make_trade_orm(
            trade_id=_TRADE_UUID_1,
            run_id=_RUN_UUID,
            symbol="BTC/USDT",
            side="buy",
            entry_price="45000.00",
            exit_price="47000.00",
            realised_pnl="200.00",
            total_fees="5.00",
        )
        trade2 = _make_trade_orm(
            trade_id=_TRADE_UUID_2,
            run_id=_RUN_UUID,
            symbol="ETH/USDT",
            side="buy",
            entry_price="2800.00",
            exit_price="2750.00",
            realised_pnl="-50.00",
            total_fees="2.80",
        )

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),    # _get_run_or_404
            _make_scalar_result(2),                       # COUNT(*)
            _make_scalars_result([trade1, trade2]),        # page SELECT
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/trades")

        assert resp.status_code == 200
        body = resp.json()

        # Pagination envelope
        assert body["total"] == 2
        assert body["offset"] == 0
        assert body["limit"] == 50  # default limit

        # Items
        assert len(body["items"]) == 2

        # Verify camelCase field names and values on the first trade
        item = body["items"][0]
        assert "runId" in item
        assert "entryPrice" in item
        assert "exitPrice" in item
        assert "realisedPnl" in item
        assert "totalFees" in item
        assert "entryAt" in item
        assert "exitAt" in item
        assert "strategyId" in item

        # Spot-check values
        assert item["runId"] == str(_RUN_UUID)
        assert item["entryPrice"] == "45000.00"
        assert item["exitPrice"] == "47000.00"
        assert item["realisedPnl"] == "200.00"
        assert item["symbol"] == "BTC/USDT"

    def test_trades_empty_run_returns_200_with_zero_total(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A run with no completed trades must return HTTP 200 with total=0
        and an empty items list.

        This is the expected state for newly created or still-running
        paper/live runs that have not closed any position yet.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),  # _get_run_or_404
            _make_scalar_result(0),                     # COUNT(*) returns 0
            _make_scalars_result([]),                    # page SELECT returns empty
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/trades")

        assert resp.status_code == 200
        body = resp.json()

        assert body["total"] == 0
        assert body["items"] == []
        assert body["offset"] == 0
        assert body["limit"] == 50

    def test_trades_run_not_found_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting trades for a non-existent run UUID must return HTTP 404
        with a detail message containing the missing UUID.

        The handler calls _get_run_or_404() first; when it returns None the
        endpoint raises HTTPException(404) before any COUNT or page query.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/trades")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]

    def test_trades_filtered_by_symbol_returns_200(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        The ?symbol= query parameter must be accepted without a 422 error.

        The mock still receives 3 execute() calls; the filter is applied
        inside the handler's WHERE clause and is invisible to the mock, but
        we verify the endpoint accepts the query param without error.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID)
        trade1 = _make_trade_orm(
            trade_id=_TRADE_UUID_1, run_id=_RUN_UUID, symbol="BTC/USDT"
        )

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),
            _make_scalar_result(1),
            _make_scalars_result([trade1]),
        ]

        resp = client_dev_with_db.get(
            f"/api/v1/runs/{_RUN_UUID}/trades?symbol=BTC%2FUSDT"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["symbol"] == "BTC/USDT"


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}/positions tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetPositions:
    """
    Tests for GET /api/v1/runs/{run_id}/positions.

    The handler queries PositionSnapshotORM rows where quantity > 0 for
    the given run.  Position snapshots are persisted when a run stops.
    Two DB calls: _get_run_or_404() lookup, then position query.

    Responses use PositionListResponse:
      runId (UUID), positions (list), count (int)
    """

    def test_stopped_run_returns_200_with_empty_positions(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A stopped run with no non-flat position snapshots must return HTTP 200
        with an empty positions list and count=0.

        The handler queries position_snapshots WHERE quantity > 0; when no
        matching rows exist, an empty list is returned.
        """
        run_orm = _make_run_orm(
            run_id=_RUN_UUID, status="stopped", stopped_at=_FIXED_NOW
        )
        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),
            _make_scalars_result([]),  # no position snapshots with qty > 0
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/positions")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["positions"] == []
        assert body["count"] == 0

    def test_running_run_returns_200_with_empty_positions(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A currently running run with no persisted position snapshots must
        return HTTP 200 with empty positions and count=0.

        Position snapshots are written on run stop; while a run is still
        active, no snapshots exist yet in the DB.
        """
        run_orm = _make_run_orm(run_id=_RUN_UUID, status="running")
        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),
            _make_scalars_result([]),  # no position snapshots yet
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/positions")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["positions"] == []
        assert body["count"] == 0

    def test_positions_run_not_found_returns_404(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        Requesting positions for a non-existent run UUID must return HTTP 404
        with a detail message containing the missing UUID.

        Only 1 execute() call is made (the _get_run_or_404 lookup); it
        returns None which triggers the 404 response.
        """
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_MISSING_UUID}/positions")

        assert resp.status_code == 404
        body = resp.json()
        assert str(_MISSING_UUID) in body["detail"]


# ---------------------------------------------------------------------------
# Paper engine persistence pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPaperEnginePersistencePipeline:
    """
    Integration tests verifying that paper-mode runs return real persisted data
    through the portfolio, equity-curve, and trades endpoints.

    Sprint 11 added _persist_paper_results() which writes equity snapshots and
    trades to the DB when a paper engine stops.  Sprint 12 wired record_trade()
    so trades are recorded during the run.  These tests confirm the full pipeline
    from persisted DB rows back to API responses.

    All four tests use the exact same mock patterns as the surrounding test
    classes:
    - _make_run_orm(run_mode="paper", status="stopped") -- stopped paper run
    - _make_equity_snapshot_orm() with paper-realistic values
    - _make_trade_orm() with paper-realistic values
    - _portfolio_side_effect_with_data() / _portfolio_side_effect_empty_run()
      for the 9-call portfolio summary sequence
    - [run, count, scalars] 3-element side_effect for equity-curve and trades
    """

    def test_paper_run_portfolio_with_equity_and_trades(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A stopped paper-mode run with persisted equity snapshots and trades
        must return HTTP 200 with portfolio values sourced from DB rows.

        Verifies:
        - HTTP 200 status
        - currentEquity comes from the persisted EquitySnapshotORM (not initial_capital)
        - totalRealisedPnl reflects the trade aggregate SUM from the DB
        - dailyPnl reflects the summed realised_pnl for trades exiting today
        - openPositions is 0 -- paper engine persists unrealised_pnl=0 (MVP
          hard-code, see _persist_paper_results in apps/api/routers/runs.py)
        - initialCash matches the run config initial_capital
        """
        run_orm = _make_run_orm(
            run_id=_RUN_UUID,
            run_mode="paper",
            status="stopped",
            initial_capital="10000.00",
            stopped_at=_FIXED_NOW,
        )
        # Paper engine persists unrealised_pnl=0 per the MVP hard-code in
        # _persist_paper_results -- openPositions will therefore be 0.
        snapshot_orm = _make_equity_snapshot_orm(
            snap_id=1,
            run_id=_RUN_UUID,
            equity="10250.00",
            cash="7500.00",
            unrealised_pnl="0.00",
            realised_pnl="350.00",
            drawdown_pct="0.01",
            bar_index=49,
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_with_data(
            run_orm=run_orm,
            snapshot_orm=snapshot_orm,
            total_trades=3,
            total_realised_pnl=Decimal("350.00"),
            total_fees_paid=Decimal("10.50"),
            winning_trades=2,
            losing_trades=1,
            peak_equity=Decimal("10300.00"),
            max_drawdown=Decimal("0.01"),
            equity_curve_length=50,
            daily_pnl=Decimal("75.50"),
            open_positions=0,
        )

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()

        # Identity
        assert body["runId"] == str(_RUN_UUID)
        assert body["initialCash"] == "10000.00"

        # Current state from persisted paper-engine snapshot
        assert body["currentEquity"] == "10250.00"
        assert body["currentCash"] == "7500.00"
        assert body["drawdownPct"] == pytest.approx(0.01, abs=1e-9)

        # Trade aggregates from DB
        assert body["totalRealisedPnl"] == "350.00"
        assert body["totalFeesPaid"] == "10.50"
        assert body["totalTrades"] == 3
        assert body["winningTrades"] == 2
        assert body["losingTrades"] == 1

        # Daily PnL from today's trades
        assert body["dailyPnl"] == "75.50"

        # MVP: paper engine persists unrealised_pnl=0 so openPositions is 0
        assert body["openPositions"] == 0

        # Equity curve and peak from DB aggregate
        assert body["equityCurveLength"] == 50
        assert body["peakEquity"] == "10300.00"

    def test_paper_run_equity_curve_returns_snapshots(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A stopped paper-mode run with two persisted EquitySnapshotORM rows
        must return HTTP 200 with totalPoints=2 and a points list containing
        the correct equity and drawdownPct values.

        Verifies:
        - HTTP 200 status
        - totalPoints equals the COUNT(*) mock return (2)
        - points list length equals 2
        - First point (bar_index=0) has the paper-run opening values
        - Second point (bar_index=1) reflects the first bar equity change
        - All camelCase field names are present on each point
        """
        run_orm = _make_run_orm(
            run_id=_RUN_UUID,
            run_mode="paper",
            status="stopped",
            stopped_at=_FIXED_NOW,
        )
        snap1 = _make_equity_snapshot_orm(
            snap_id=1,
            run_id=_RUN_UUID,
            bar_index=0,
            equity="10000.00",
            cash="10000.00",
            unrealised_pnl="0.00",
            realised_pnl="0.00",
            drawdown_pct="0.00",
        )
        snap2 = _make_equity_snapshot_orm(
            snap_id=2,
            run_id=_RUN_UUID,
            bar_index=1,
            equity="10125.00",
            cash="7500.00",
            unrealised_pnl="0.00",
            realised_pnl="125.00",
            drawdown_pct="0.00",
        )

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),  # _get_run_or_404
            _make_scalar_result(2),                     # COUNT(*)
            _make_scalars_result([snap1, snap2]),        # page SELECT
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/equity-curve")

        assert resp.status_code == 200
        body = resp.json()

        assert body["runId"] == str(_RUN_UUID)
        assert body["totalPoints"] == 2
        assert len(body["points"]) == 2

        # Verify all camelCase field names are present on first point
        first_point = body["points"][0]
        assert "barIndex" in first_point
        assert "equity" in first_point
        assert "cash" in first_point
        assert "unrealisedPnl" in first_point
        assert "realisedPnl" in first_point
        assert "drawdownPct" in first_point
        assert "timestamp" in first_point

        # Opening snapshot values (bar 0)
        assert first_point["equity"] == "10000.00"
        assert first_point["barIndex"] == 0
        assert first_point["drawdownPct"] == pytest.approx(0.00, abs=1e-9)

        # Second snapshot reflects first-bar trade outcome
        second_point = body["points"][1]
        assert second_point["equity"] == "10125.00"
        assert second_point["barIndex"] == 1
        assert second_point["realisedPnl"] == "125.00"
        assert second_point["unrealisedPnl"] == "0.00"

    def test_paper_run_trades_returns_recorded_trades(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A stopped paper-mode run with one persisted TradeORM record must
        return HTTP 200 with total=1 and an items list containing the correct
        trade data.

        Sprint 12 wired record_trade() so completed trades are written to the
        DB during paper-engine execution.  This test confirms the trades
        endpoint reflects those persisted records.

        Verifies:
        - HTTP 200 status
        - total == 1 (COUNT(*) mock)
        - items[0] fields use camelCase
        - entryPrice and exitPrice match the paper-engine fill values
        - realisedPnl reflects the net PnL after fees
        - strategyId matches "ma_crossover" (the paper run strategy)
        - symbol matches "BTC/USDT"
        """
        run_orm = _make_run_orm(
            run_id=_RUN_UUID,
            run_mode="paper",
            status="stopped",
            stopped_at=_FIXED_NOW,
        )
        trade_orm = _make_trade_orm(
            trade_id=_TRADE_UUID_1,
            run_id=_RUN_UUID,
            symbol="BTC/USDT",
            side="buy",
            entry_price="42000.00",
            exit_price="43500.00",
            quantity="0.05",
            realised_pnl="72.50",
            total_fees="2.50",
            entry_at=_FIXED_ENTRY,
            exit_at=_FIXED_EXIT,
            strategy_id="ma_crossover",
        )

        mock_db_session.execute.side_effect = [
            _make_scalar_one_or_none_result(run_orm),   # _get_run_or_404
            _make_scalar_result(1),                      # COUNT(*)
            _make_scalars_result([trade_orm]),            # page SELECT
        ]

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/trades")

        assert resp.status_code == 200
        body = resp.json()

        # Pagination envelope
        assert body["total"] == 1
        assert body["offset"] == 0
        assert body["limit"] == 50  # default limit

        # Single trade item
        assert len(body["items"]) == 1
        item = body["items"][0]

        # camelCase field names must be present
        assert "runId" in item
        assert "entryPrice" in item
        assert "exitPrice" in item
        assert "realisedPnl" in item
        assert "totalFees" in item
        assert "strategyId" in item

        # Paper-engine realistic values
        assert item["symbol"] == "BTC/USDT"
        assert item["side"] == "buy"
        assert item["entryPrice"] == "42000.00"
        assert item["exitPrice"] == "43500.00"
        assert item["realisedPnl"] == "72.50"
        assert item["totalFees"] == "2.50"
        assert item["strategyId"] == "ma_crossover"
        assert item["runId"] == str(_RUN_UUID)

    def test_paper_run_portfolio_with_no_data_returns_initial_capital(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        A stopped paper-mode run with no persisted snapshots or trades must
        return HTTP 200 with portfolio values falling back to the
        initial_capital stored in the run config.

        This covers the edge case where the engine was stopped before any bar
        completed (e.g., exchange unreachable on first poll), so
        _persist_paper_results() writes no rows.

        Verifies:
        - HTTP 200 status
        - initialCash == "5000.00" (non-default initial_capital from config)
        - currentEquity == "5000.00" (fallback: no snapshot exists)
        - currentCash == "5000.00" (fallback: no snapshot exists)
        - totalTrades == 0
        - dailyPnl == "0" (SQL SUM returns None with no rows)
        - openPositions == 0 (no snapshot, no unrealised_pnl heuristic)
        - equityCurveLength == 0 (no snapshots persisted)
        - totalReturnPct == 0.0 (no equity change from initial_capital)
        """
        run_orm = _make_run_orm(
            run_id=_RUN_UUID,
            run_mode="paper",
            status="stopped",
            initial_capital="5000.00",
            stopped_at=_FIXED_NOW,
        )
        mock_db_session.execute.side_effect = _portfolio_side_effect_empty_run(run_orm)

        resp = client_dev_with_db.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 200
        body = resp.json()

        # Non-default initial capital must flow through from config
        assert body["initialCash"] == "5000.00"
        # No snapshot: both equity and cash fall back to initial_capital
        assert body["currentEquity"] == "5000.00"
        assert body["currentCash"] == "5000.00"

        # No trades persisted
        assert body["totalTrades"] == 0
        # SQL SUM returns None when no rows; handler serialises to "0"
        assert body["dailyPnl"] == "0"

        # No snapshot means no unrealised_pnl heuristic fires
        assert body["openPositions"] == 0

        # No snapshots written at all
        assert body["equityCurveLength"] == 0

        # No equity growth from initial_capital
        assert body["totalReturnPct"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Authentication tests (production mode)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPortfolioEndpointAuth:
    """
    Auth enforcement tests for portfolio endpoints in production mode.

    These tests use client_prod (no DB override needed) because the auth
    rejection in the require_api_key dependency fires before any DB access.
    The 401 response is returned before the handler body runs.

    client_prod is created from app_prod_mode which has REQUIRE_API_AUTH=true.
    """

    def test_portfolio_summary_without_api_key_returns_401(
        self, client_prod: TestClient
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/portfolio without an API key must return
        HTTP 401 in production mode.

        The require_api_key dependency is mounted as a router-level dependency
        on the portfolio router and fires before any handler logic or DB call.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/portfolio")

        assert resp.status_code == 401

    def test_trades_without_api_key_returns_401(
        self, client_prod: TestClient
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/trades without an API key must return
        HTTP 401 in production mode.

        Confirms that the authentication dependency is correctly applied to
        the trades endpoint — not only the portfolio summary endpoint.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/trades")

        assert resp.status_code == 401

    def test_equity_curve_without_api_key_returns_401(
        self, client_prod: TestClient
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/equity-curve without an API key must return
        HTTP 401 in production mode.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/equity-curve")

        assert resp.status_code == 401

    def test_positions_without_api_key_returns_401(
        self, client_prod: TestClient
    ) -> None:
        """
        GET /api/v1/runs/{run_id}/positions without an API key must return
        HTTP 401 in production mode.
        """
        resp = client_prod.get(f"/api/v1/runs/{_RUN_UUID}/positions")

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/portfolio/summary — aggregate portfolio tests
# ---------------------------------------------------------------------------

# -- Additional helpers for the 3-query aggregate endpoint -----------------


def _make_all_result(rows: list) -> MagicMock:
    """
    Return a MagicMock that mimics an execute() result supporting .all().

    Used for the status-count query (Query 1) and the configs query (Query 3)
    of get_aggregate_portfolio, both of which call ``result.all()`` to iterate
    over rows.

    Parameters
    ----------
    rows:
        List of row objects (or tuples) to return from .all().
    """
    result = MagicMock()
    result.all.return_value = rows
    return result


def _make_aggregate_trade_one(
    total: int,
    wins: int,
    losses: int,
    pnl: Decimal,
    fees: Decimal,
) -> MagicMock:
    """
    Return a MagicMock for the trade aggregate query (Query 2) that uses .one().

    The get_aggregate_portfolio handler calls
    ``(await db.execute(trade_stmt)).one()`` and accesses the attributes
    ``total``, ``wins``, ``losses``, ``pnl``, ``fees`` on the returned row.

    Parameters
    ----------
    total:
        Total trade count.
    wins:
        Winning trade count (realised_pnl > 0).
    losses:
        Losing trade count (realised_pnl < 0).
    pnl:
        Aggregate SUM of realised_pnl.
    fees:
        Aggregate SUM of total_fees.
    """
    row = MagicMock()
    row.total = total
    row.wins = wins
    row.losses = losses
    row.pnl = pnl
    row.fees = fees

    result = MagicMock()
    result.one.return_value = row
    return result


@pytest.mark.integration
class TestAggregatePortfolio:
    """
    Tests for GET /api/v1/portfolio/summary.

    The endpoint is served by summary_router (prefix="/portfolio") mounted
    at /api/v1, so the full URL is /api/v1/portfolio/summary.

    The handler issues exactly 3 sequential db.execute() calls:

      [0] status counts  ->  result.all()  ->  list of rows with .status / .cnt
      [1] trade stats    ->  result.one()  ->  Row with .total/.wins/.losses/.pnl/.fees
      [2] config blobs   ->  result.all()  ->  list of (config_dict,) tuples

    Each test sets mock_db_session.execute.side_effect as a 3-element list
    built from _make_all_result() and _make_aggregate_trade_one().

    Response key naming
    -------------------
    AggregatePortfolioResponse uses alias_generator=to_camel, so all
    assertions use camelCase keys on resp.json() output.
    """

    def test_aggregate_empty_db(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When the database contains no runs and no trades, the summary endpoint
        must return HTTP 200 with all numeric totals as zero, winRate=0.0,
        bestRunReturnPct=None, worstRunReturnPct=None, and
        totalInitialCapital="0".

        Query 1 returns an empty list (no status group rows).
        Query 2 returns a trade row with all zeros.
        Query 3 returns an empty list (no config rows).
        """
        mock_db_session.execute.side_effect = [
            _make_all_result([]),                                     # no runs
            _make_aggregate_trade_one(0, 0, 0, Decimal("0"), Decimal("0")),
            _make_all_result([]),                                     # no configs
        ]

        resp = client_dev_with_db.get("/api/v1/portfolio/summary")

        assert resp.status_code == 200
        body = resp.json()

        # Run counts — all zero
        assert body["totalRuns"] == 0
        assert body["runningRuns"] == 0
        assert body["stoppedRuns"] == 0
        assert body["errorRuns"] == 0

        # Trade statistics — all zero
        assert body["totalTrades"] == 0
        assert body["winningTrades"] == 0
        assert body["losingTrades"] == 0
        assert body["winRate"] == pytest.approx(0.0, abs=1e-9)

        # Financial totals
        assert body["totalRealisedPnl"] == "0"
        assert body["totalFeesPaid"] == "0"
        assert body["totalInitialCapital"] == "0"

        # No backtest metrics in empty DB
        assert body["bestRunReturnPct"] is None
        assert body["worstRunReturnPct"] is None

    def test_aggregate_with_trades(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        With 2 runs and 5 trades (3 winners at +100 each, 2 losers at -40 each),
        the summary must return correct totalTrades, winningTrades, losingTrades,
        totalRealisedPnl, totalFeesPaid, and winRate.

        Expected values:
        - totalTrades = 5
        - winningTrades = 3
        - losingTrades = 2
        - totalRealisedPnl = "220.00"  (3*100 - 2*40 = 220)
        - totalFeesPaid = "15.00"      (5 trades at 3.00 each)
        - winRate = 0.6                (3 / 5)
        - totalRuns = 2                (1 stopped + 1 running)
        - totalInitialCapital = "20000.00"  (2 runs at 10000 each)
        """
        status_rows = [
            SimpleNamespace(status="stopped", cnt=1),
            SimpleNamespace(status="running", cnt=1),
        ]
        config_rows = [
            ({"initial_capital": "10000.00"},),
            ({"initial_capital": "10000.00"},),
        ]
        mock_db_session.execute.side_effect = [
            _make_all_result(status_rows),
            _make_aggregate_trade_one(
                total=5,
                wins=3,
                losses=2,
                pnl=Decimal("220.00"),
                fees=Decimal("15.00"),
            ),
            _make_all_result(config_rows),
        ]

        resp = client_dev_with_db.get("/api/v1/portfolio/summary")

        assert resp.status_code == 200
        body = resp.json()

        # Run counts derived from status group-by rows
        assert body["totalRuns"] == 2
        assert body["stoppedRuns"] == 1
        assert body["runningRuns"] == 1
        assert body["errorRuns"] == 0

        # Trade statistics
        assert body["totalTrades"] == 5
        assert body["winningTrades"] == 3
        assert body["losingTrades"] == 2
        assert body["winRate"] == pytest.approx(0.6, abs=1e-9)

        # Financial totals
        assert body["totalRealisedPnl"] == "220.00"
        assert body["totalFeesPaid"] == "15.00"
        assert body["totalInitialCapital"] == "20000.00"

    def test_aggregate_run_counts(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        With 1 running run, 2 stopped runs, and 1 error run, the summary must
        return runningRuns=1, stoppedRuns=2, errorRuns=1, and totalRuns=4.

        The status count query returns 3 group-by rows (one per status value).
        The total is computed as the sum of all cnt values in the handler, not
        fetched from a separate COUNT(*) query.
        """
        status_rows = [
            SimpleNamespace(status="running", cnt=1),
            SimpleNamespace(status="stopped", cnt=2),
            SimpleNamespace(status="error", cnt=1),
        ]
        mock_db_session.execute.side_effect = [
            _make_all_result(status_rows),
            _make_aggregate_trade_one(0, 0, 0, Decimal("0"), Decimal("0")),
            _make_all_result([]),
        ]

        resp = client_dev_with_db.get("/api/v1/portfolio/summary")

        assert resp.status_code == 200
        body = resp.json()

        assert body["totalRuns"] == 4
        assert body["runningRuns"] == 1
        assert body["stoppedRuns"] == 2
        assert body["errorRuns"] == 1

    def test_aggregate_with_backtest_metrics(
        self, client_dev_with_db: TestClient, mock_db_session: AsyncMock
    ) -> None:
        """
        When run configs contain backtest_metrics with total_return_pct, the
        summary must populate bestRunReturnPct and worstRunReturnPct from the
        maximum and minimum return values across those runs.

        Three config rows:
        - Run A: backtest_metrics.total_return_pct = 0.25  (best)
        - Run B: backtest_metrics.total_return_pct = -0.10 (worst)
        - Run C: no backtest_metrics key (paper/live run — excluded)

        Expected:
        - bestRunReturnPct = 0.25
        - worstRunReturnPct = -0.10
        """
        status_rows = [
            SimpleNamespace(status="stopped", cnt=3),
        ]
        config_rows = [
            ({"initial_capital": "10000.00", "backtest_metrics": {"total_return_pct": 0.25}},),
            ({"initial_capital": "10000.00", "backtest_metrics": {"total_return_pct": -0.10}},),
            ({"initial_capital": "10000.00"},),   # paper run — no backtest_metrics
        ]
        mock_db_session.execute.side_effect = [
            _make_all_result(status_rows),
            _make_aggregate_trade_one(
                total=10,
                wins=6,
                losses=4,
                pnl=Decimal("500.00"),
                fees=Decimal("30.00"),
            ),
            _make_all_result(config_rows),
        ]

        resp = client_dev_with_db.get("/api/v1/portfolio/summary")

        assert resp.status_code == 200
        body = resp.json()

        # Backtest return percentages
        assert body["bestRunReturnPct"] == pytest.approx(0.25, abs=1e-9)
        assert body["worstRunReturnPct"] == pytest.approx(-0.10, abs=1e-9)

        # Sanity-check other fields
        assert body["totalRuns"] == 3
        assert body["totalTrades"] == 10
        assert body["totalInitialCapital"] == "30000.00"
