"""
apps/api/services/run_persistence.py
-------------------------------------
Run persistence and ORM <-> response mapping helpers.

Extracted from :mod:`api.routers.runs` in Sprint 40 Stap 2a as part of the
AR-001 decomposition. These functions contain no routing logic; they only
translate between ORM models, Pydantic response schemas, and the database.

Public API
----------
- :func:`run_orm_to_response` -- convert ``RunORM`` to ``RunResponse``.
- :func:`run_orm_to_detail_response` -- convert ``RunORM`` to
  ``RunDetailResponse`` (adds ``backtest_metrics`` if present in config).
- :func:`build_backtest_metrics` -- convert ``BacktestResult`` to
  ``BacktestMetricsResponse``.
- :func:`persist_paper_results` -- write paper-engine equity curve, trades,
  orders, fills, and position snapshots to DB.
- :func:`persist_backtest_results` -- write backtest trades, equity,
  orders, fills, positions, and metrics to DB.

Backwards compatibility
-----------------------
``api.routers.runs`` re-exports each function under its original
underscore-prefixed name (``_run_orm_to_response``, etc.) so existing
internal callers inside the router module keep working unchanged.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import (
    EquitySnapshotORM,
    FillORM,
    OrderORM,
    PositionSnapshotORM,
    RunORM,
    TradeORM,
)
from api.schemas import BacktestMetricsResponse, RunDetailResponse, RunResponse

__all__ = [
    "build_backtest_metrics",
    "persist_backtest_results",
    "persist_paper_results",
    "run_orm_to_detail_response",
    "run_orm_to_response",
]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ORM -> response model conversion
# ---------------------------------------------------------------------------


def run_orm_to_response(run: RunORM) -> RunResponse:
    """Convert a ``RunORM`` instance to a ``RunResponse`` Pydantic model.

    Parameters
    ----------
    run:
        The ORM model instance to convert.

    Returns
    -------
    RunResponse
        The API response model.
    """
    return RunResponse.model_validate(run)


def run_orm_to_detail_response(run: RunORM) -> RunDetailResponse:
    """Convert a ``RunORM`` instance to a ``RunDetailResponse`` Pydantic model.

    Extracts ``backtest_metrics`` from ``run.config`` when present.

    Parameters
    ----------
    run:
        The ORM model instance to convert.

    Returns
    -------
    RunDetailResponse
        The extended API response model with optional backtest metrics.
    """
    base = RunDetailResponse.model_validate(run)

    # Attempt to populate backtest_metrics from the JSONB config blob
    raw_metrics: dict[str, Any] | None = (run.config or {}).get("backtest_metrics")
    if raw_metrics is not None:
        try:
            base = base.model_copy(
                update={"backtest_metrics": BacktestMetricsResponse.model_validate(raw_metrics)}
            )
        except Exception:  # noqa: BLE001  -- best-effort; never fail a GET on bad stored data
            logger.warning(
                "runs.backtest_metrics_parse_error",
                run_id=str(run.id),
                exc_info=True,
            )

    return base


# ---------------------------------------------------------------------------
# Backtest result -> metrics response
# ---------------------------------------------------------------------------


def build_backtest_metrics(result: Any) -> BacktestMetricsResponse:
    """Convert a ``BacktestResult`` into a ``BacktestMetricsResponse``.

    Parameters
    ----------
    result:
        The fully-populated ``BacktestResult`` returned by ``BacktestRunner.run()``.

    Returns
    -------
    BacktestMetricsResponse
        Typed schema ready for API serialisation.
    """
    return BacktestMetricsResponse(
        total_return_pct=result.total_return_pct,
        cagr=result.cagr,
        initial_capital=str(result.initial_capital),
        final_equity=str(result.final_equity),
        total_fees_paid=str(result.total_fees_paid),
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        calmar_ratio=result.calmar_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        max_drawdown_duration_bars=result.max_drawdown_duration_bars,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        average_trade_pnl=str(result.average_trade_pnl),
        average_win=str(result.average_win),
        average_loss=str(result.average_loss),
        largest_win=str(result.largest_win),
        largest_loss=str(result.largest_loss),
        total_bars=result.total_bars,
        bars_in_market=result.bars_in_market,
        exposure_pct=result.exposure_pct,
        start_date=result.start_date,
        end_date=result.end_date,
        duration_days=result.duration_days,
    )


# ---------------------------------------------------------------------------
# Persist paper engine results (uses isolated DB session)
# ---------------------------------------------------------------------------


async def persist_paper_results(
    *,
    run_id_str: str,
    portfolio: Any,
    execution_engine: Any | None = None,
    log: Any,
) -> None:
    """Persist paper engine equity curve, completed trades, orders, and fills to DB.

    Called from ``_run_live_engine``'s ``finally`` block. Uses its own
    isolated DB session. Skipped when equity curve, trade history, and
    order list are all empty (engine stopped before generating any data).
    """
    from api.db.session import get_session_factory  # noqa: E402

    equity_curve = portfolio.get_equity_curve()
    trade_history = portfolio.get_trade_history()

    has_orders = execution_engine is not None and bool(execution_engine.get_all_orders())
    if not equity_curve and not trade_history and not has_orders:
        log.debug("runs.paper_persist_skipped", reason="no_data")
        return

    run_id = uuid.UUID(run_id_str)

    # Build EquitySnapshotORM rows with peak-tracking drawdown
    equity_orms: list[EquitySnapshotORM] = []
    peak = Decimal("0")
    for bar_index, (timestamp, equity) in enumerate(equity_curve):
        if equity > peak:
            peak = equity
        if peak > Decimal("0"):
            dd_pct = ((peak - equity) / peak).quantize(Decimal("0.00000001"))
        else:
            dd_pct = Decimal("0")
        # Clamp to [0, 1] to satisfy DB CHECK constraint
        dd_pct = max(Decimal("0"), min(dd_pct, Decimal("1")))

        # Clamp equity to 0 to satisfy ck_equity_snapshots_equity_non_negative
        if equity < Decimal("0"):
            log.warning(
                "runs.paper_persist_negative_equity_clamped",
                bar_index=bar_index,
                raw_equity=str(equity),
            )
            equity = Decimal("0")

        equity_orms.append(
            EquitySnapshotORM(
                run_id=run_id,
                equity=equity,
                cash=Decimal("0"),           # MVP: per-bar cash not tracked
                unrealised_pnl=Decimal("0"), # MVP: per-bar unrealised not tracked
                realised_pnl=Decimal("0"),   # MVP: per-bar realised not tracked
                drawdown_pct=dd_pct,
                bar_index=bar_index,
                timestamp=timestamp,
            )
        )

    # Build TradeORM rows from completed round-trips
    trade_orms: list[TradeORM] = []
    for trade in trade_history:
        trade_orms.append(
            TradeORM(
                id=trade.trade_id,
                run_id=run_id,
                symbol=trade.symbol,
                side=(
                    trade.side.value
                    if hasattr(trade.side, "value")
                    else str(trade.side)
                ),
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                realised_pnl=trade.realised_pnl,
                total_fees=trade.total_fees,
                entry_at=trade.entry_at,
                exit_at=trade.exit_at,
                strategy_id=trade.strategy_id or "unknown",
                # Sprint 32: adaptive learning fields (getattr for backward compat)
                mae_pct=getattr(trade, "mae_pct", None),
                mfe_pct=getattr(trade, "mfe_pct", None),
                exit_reason=getattr(trade, "exit_reason", None),
                regime_at_entry=getattr(trade, "regime_at_entry", None),
                signal_context=getattr(trade, "signal_context", None),
            )
        )

    # Build OrderORM and FillORM rows
    order_orms: list[OrderORM] = []
    fill_orms: list[FillORM] = []

    if execution_engine is not None:
        for order in execution_engine.get_all_orders():
            order_orms.append(
                OrderORM(
                    id=order.order_id,
                    client_order_id=order.client_order_id,
                    run_id=run_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    quantity=order.quantity,
                    price=order.price,
                    status=order.status.value,
                    filled_quantity=order.filled_quantity,
                    average_fill_price=order.average_fill_price,
                    exchange_order_id=order.exchange_order_id,
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                )
            )

        for fill in execution_engine.get_all_fills():
            fill_orms.append(
                FillORM(
                    id=fill.fill_id,
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side.value,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                    is_maker=fill.is_maker,
                    executed_at=fill.executed_at,
                )
            )

    # Build PositionSnapshotORM rows from final position state
    position_orms: list[PositionSnapshotORM] = []
    now = datetime.now(tz=UTC)
    if hasattr(portfolio, '_position_snapshots'):
        for pos in portfolio._position_snapshots.values():
            position_orms.append(
                PositionSnapshotORM(
                    run_id=run_id,
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_entry_price=pos.average_entry_price,
                    current_price=pos.current_price,
                    unrealised_pnl=pos.unrealised_pnl,
                    realised_pnl=pos.realised_pnl,
                    total_fees_paid=pos.total_fees_paid,
                    opened_at=pos.opened_at,
                    snapshot_at=now,
                )
            )

    # Persist in isolated session
    try:
        factory = get_session_factory()
        async with factory() as db:
            try:
                if trade_orms:
                    db.add_all(trade_orms)
                if equity_orms:
                    db.add_all(equity_orms)
                if order_orms:
                    db.add_all(order_orms)
                    await db.flush()
                if fill_orms:
                    db.add_all(fill_orms)
                if position_orms:
                    db.add_all(position_orms)
                await db.commit()
                log.info(
                    "runs.paper_results_persisted",
                    equity_snapshots=len(equity_orms),
                    trades=len(trade_orms),
                    orders=len(order_orms),
                    fills=len(fill_orms),
                    positions=len(position_orms),
                )
            except Exception:
                await db.rollback()
                log.exception("runs.paper_persist_db_error")
    except Exception:
        log.exception("runs.paper_persist_session_failed")


# ---------------------------------------------------------------------------
# Persist backtest results (reuses provided AsyncSession)
# ---------------------------------------------------------------------------


async def persist_backtest_results(
    db: AsyncSession,
    run_id: uuid.UUID,
    run_orm: RunORM,
    result: Any,
    log: Any,
    execution_engine: Any | None = None,
    portfolio: Any | None = None,
) -> None:
    """Write all backtest result records to the database.

    Persists:
    - ``TradeORM`` rows for every completed round-trip trade.
    - ``EquitySnapshotORM`` rows for every equity curve point.
    - Backtest metrics summary into ``run_orm.config["backtest_metrics"]``.

    Parameters
    ----------
    db:
        Active async SQLAlchemy session.
    run_id:
        UUID of the run being persisted.
    run_orm:
        The ``RunORM`` instance to update with metrics.
    result:
        The ``BacktestResult`` from ``BacktestRunner.run()``.
    log:
        Bound structlog logger.
    """
    from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

    # --- Persist trades ---
    trade_orms: list[TradeORM] = []
    for trade in result.trades:
        trade_orm = TradeORM(
            id=trade.trade_id,
            run_id=run_id,
            symbol=trade.symbol,
            side=trade.side.value if hasattr(trade.side, "value") else str(trade.side),
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            quantity=trade.quantity,
            realised_pnl=trade.realised_pnl,
            total_fees=trade.total_fees,
            entry_at=trade.entry_at,
            exit_at=trade.exit_at,
            strategy_id=trade.strategy_id,
            # Sprint 32: adaptive learning fields (getattr for backward compat)
            mae_pct=getattr(trade, "mae_pct", None),
            mfe_pct=getattr(trade, "mfe_pct", None),
            exit_reason=getattr(trade, "exit_reason", None),
            regime_at_entry=getattr(trade, "regime_at_entry", None),
            signal_context=getattr(trade, "signal_context", None),
        )
        trade_orms.append(trade_orm)

    if trade_orms:
        db.add_all(trade_orms)
        log.info("runs.backtest_trades_inserted", count=len(trade_orms))

    # --- Persist equity curve ---
    equity_orms: list[EquitySnapshotORM] = []
    for bar_index, point in enumerate(result.equity_curve):
        snapshot = EquitySnapshotORM(
            run_id=run_id,
            equity=point.equity,
            # MVP approximation: cash and unrealised_pnl are not individually
            # tracked per bar in the current EquityCurvePoint model.
            # equity = cash + unrealised_pnl; we store equity as cash and
            # 0 for unrealised/realised until Sprint 2 enhances the model.
            cash=Decimal("0"),  # MVP: per-bar cash not tracked; see Sprint 2
            unrealised_pnl=Decimal("0"),
            realised_pnl=Decimal("0"),
            drawdown_pct=Decimal(str(point.drawdown_pct)),
            bar_index=bar_index,
            timestamp=point.timestamp,
        )
        equity_orms.append(snapshot)

    if equity_orms:
        db.add_all(equity_orms)
        log.info("runs.backtest_equity_snapshots_inserted", count=len(equity_orms))

    # --- Persist orders and fills ---
    order_orms: list[OrderORM] = []
    fill_orms: list[FillORM] = []

    if execution_engine is not None:
        for order in execution_engine.get_all_orders():
            order_orms.append(
                OrderORM(
                    id=order.order_id,
                    client_order_id=order.client_order_id,
                    run_id=run_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    quantity=order.quantity,
                    price=order.price,
                    status=order.status.value,
                    filled_quantity=order.filled_quantity,
                    average_fill_price=order.average_fill_price,
                    exchange_order_id=order.exchange_order_id,
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                )
            )

        for fill in execution_engine.get_all_fills():
            fill_orms.append(
                FillORM(
                    id=fill.fill_id,
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side.value,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                    is_maker=fill.is_maker,
                    executed_at=fill.executed_at,
                )
            )

    if order_orms:
        db.add_all(order_orms)
        await db.flush()
    if fill_orms:
        db.add_all(fill_orms)
    if order_orms or fill_orms:
        log.info(
            "runs.backtest_orders_fills_inserted",
            orders=len(order_orms),
            fills=len(fill_orms),
        )

    # --- Persist position snapshots ---
    position_orms: list[PositionSnapshotORM] = []
    if portfolio is not None and hasattr(portfolio, '_position_snapshots'):
        now = datetime.now(tz=UTC)
        for pos in portfolio._position_snapshots.values():
            position_orms.append(
                PositionSnapshotORM(
                    run_id=run_id,
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_entry_price=pos.average_entry_price,
                    current_price=pos.current_price,
                    unrealised_pnl=pos.unrealised_pnl,
                    realised_pnl=pos.realised_pnl,
                    total_fees_paid=pos.total_fees_paid,
                    opened_at=pos.opened_at,
                    snapshot_at=now,
                )
            )

    if position_orms:
        db.add_all(position_orms)
        log.info("runs.backtest_positions_inserted", count=len(position_orms))

    # --- Merge metrics into run.config JSONB ---
    metrics_response = build_backtest_metrics(result)
    updated_config = dict(run_orm.config or {})
    metrics_dict = metrics_response.model_dump(mode="json")
    # PostgreSQL JSONB rejects Infinity/NaN  -- replace with None
    for k, v in metrics_dict.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            metrics_dict[k] = None
    updated_config["backtest_metrics"] = metrics_dict
    run_orm.config = updated_config
    # Explicitly flag the JSONB column as modified so SQLAlchemy tracks the
    # in-place dict mutation through its change-detection mechanism.
    flag_modified(run_orm, "config")

    log.info(
        "runs.backtest_results_persisted",
        trades=len(trade_orms),
        equity_points=len(equity_orms),
        total_return=f"{result.total_return_pct:.4%}",
        sharpe=f"{result.sharpe_ratio:.3f}",
    )
