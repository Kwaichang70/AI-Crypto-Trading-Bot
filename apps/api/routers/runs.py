"""
apps/api/routers/runs.py
------------------------
Run management endpoints for the AI Crypto Trading Bot API.

Endpoints
---------
POST   /api/v1/runs              -- Start a new trading run
GET    /api/v1/runs              -- List all runs (paginated)
GET    /api/v1/runs/{run_id}     -- Get a single run's details
DELETE /api/v1/runs/{run_id}     -- Stop a running run

MVP notes
---------
- Backtest mode runs synchronously in the POST handler (fast enough for MVP).
  The BacktestRunner is wired up and results are persisted before returning.
- Paper and Live modes run as background asyncio.Tasks via _run_paper_engine
  and _run_live_engine coroutines respectively.
- Strategy parameter validation occurs at request time via ``parameter_schema()``.
- The ``config`` JSONB snapshot captures all run parameters at creation time
  so historical runs are fully self-contained even if strategy defaults change.
- Backtest metrics are written into ``config["backtest_metrics"]`` so they are
  available on ``GET /runs/{run_id}`` without a schema migration.
- LIVE mode requires passing the 3-layer safety gate:
  (1) ENABLE_LIVE_TRADING=true, (2) exchange API keys configured,
  (3) valid confirm_token matching LIVE_TRADING_CONFIRM_TOKEN.
- Paper runs emit periodic incremental DB flushes every 30 seconds via
  _flush_incremental / _incremental_flush_loop so equity, trades, orders,
  fills, and positions are visible while the run is still active.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import String, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from api.db.models import EquitySnapshotORM, FillORM, OrderORM, PositionSnapshotORM, RunORM, SkippedTradeORM, TradeORM
from api.db.session import get_db
from api.schemas import (
    BacktestMetricsResponse,
    ErrorResponse,
    PaginationParams,
    RunCreateRequest,
    RunDetailResponse,
    RunListResponse,
    RunResponse,
)
from common.types import TimeFrame

__all__ = ["router", "recover_orphaned_runs"]

router = APIRouter(prefix="/runs", tags=["runs"])

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Background task registry for paper/live trading engines
# ---------------------------------------------------------------------------
_RUN_TASKS: dict[str, asyncio.Task[None]] = {}

# Engine registry -- keyed by run_id for circuit breaker and live introspection
_RUN_ENGINES: dict[str, Any] = {}

# Adaptive learning task instances -- keyed by run_id for API state queries
_LEARNING_INSTANCES: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Telegram trade notification helper
# ---------------------------------------------------------------------------

async def _notify_trade_telegram(trade_orm: TradeORM) -> None:
    """
    Fire-and-forget Telegram notification for a newly persisted trade.

    Never raises -- failure to notify must never impact the flush path.

    Parameters
    ----------
    trade_orm:
        The persisted TradeORM row whose details will be formatted for Telegram.
    """
    try:
        from api.main import get_telegram_notifier
        notifier = get_telegram_notifier()
        if notifier is None:
            return
        await notifier.send_trade(
            symbol=trade_orm.symbol,
            side=trade_orm.side,
            quantity=str(trade_orm.quantity),
            price=str(trade_orm.exit_price),
            pnl=str(trade_orm.realised_pnl) if trade_orm.realised_pnl is not None else None,
            run_id=str(trade_orm.run_id),
        )
    except Exception:
        pass  # Never fail the flush path on notification errors


@dataclass
class _IncrementalFlushState:
    """Watermark state for incremental DB persistence during paper runs."""

    flushed_equity_count: int = 0
    flushed_trade_count: int = 0
    flushed_order_ids: set[uuid.UUID] = field(default_factory=set)
    flushed_fill_ids: set[uuid.UUID] = field(default_factory=set)
    peak_equity: Decimal = field(default_factory=lambda: Decimal("0"))


# ---------------------------------------------------------------------------
# Strategy registry  -- maps API names to strategy classes
# Imported lazily inside the handler to avoid circular import issues.
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, Any] | None = None


def _get_strategy_registry() -> dict[str, Any]:
    """
    Return the lazy-loaded strategy name -> class mapping.

    Returns
    -------
    dict[str, Any]
        Mapping of strategy identifier to strategy class.
    """
    global _STRATEGY_REGISTRY
    if _STRATEGY_REGISTRY is None:
        from trading.strategies import (
            BreakoutStrategy,
            DCARSIHybridStrategy,
            GridTradingStrategy,
            MACrossoverStrategy,
            ModelStrategy,
            RSIMeanReversionStrategy,
        )

        _STRATEGY_REGISTRY = {
            "ma_crossover": MACrossoverStrategy,
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "model_strategy": ModelStrategy,
            "dca_rsi_hybrid": DCARSIHybridStrategy,
            "grid_trading": GridTradingStrategy,
        }
    return _STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Helper: ORM -> response model conversion
# ---------------------------------------------------------------------------

def _run_orm_to_response(run: RunORM) -> RunResponse:
    """
    Convert a ``RunORM`` instance to a ``RunResponse`` Pydantic model.

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


# ---------------------------------------------------------------------------
# Helper: normalize exchange secret for CCXT compatibility
# ---------------------------------------------------------------------------

def _normalize_exchange_secret(secret: str) -> str:
    """Normalize exchange API secret for CCXT compatibility.

    Coinbase CDP Ed25519 keys are often stored in PEM format
    (-----BEGIN EC PRIVATE KEY-----\\n<base64>\\n-----END EC PRIVATE KEY-----)
    but the key body is raw Ed25519 (64 bytes / 88 base64 chars), NOT ECDSA
    SEC1 DER. CCXT's coinbase driver calls ecdsa.SigningKey.from_pem() which
    fails on Ed25519 keys wrapped in EC PEM headers.

    Solution: strip PEM headers and return raw base64 so CCXT can use the
    key directly for EdDSA signing.
    """
    # Convert literal backslash-n to real newlines
    normalized = secret.replace("\\n", "\n")
    # Strip PEM headers if present — Ed25519 keys must NOT have PEM wrapping
    lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip() and "-----" not in line
    ]
    if lines:
        return "".join(lines)
    return normalized


def _run_orm_to_detail_response(run: RunORM) -> RunDetailResponse:
    """
    Convert a ``RunORM`` instance to a ``RunDetailResponse`` Pydantic model.

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
# Helper: incremental flush of paper engine data to DB during active run
# ---------------------------------------------------------------------------

async def _flush_incremental(
    *,
    run_id_str: str,
    portfolio: Any,
    execution_engine: Any | None,
    state: _IncrementalFlushState,
    log: Any,
) -> None:
    """
    Perform one incremental flush cycle for an active paper run.

    Reads in-memory state from portfolio and execution_engine, computes
    deltas using watermarks stored in ``state``, and writes only new data
    to the database.  Uses an isolated DB session.  Safe to call concurrently
    with the engine loop  -- reads are non-destructive snapshots.

    Parameters
    ----------
    run_id_str:
        String representation of the run UUID.
    portfolio:
        ``PortfolioAccounting`` instance from the active paper engine.
    execution_engine:
        ``PaperExecutionEngine`` instance, or ``None`` if not yet created.
    state:
        Mutable watermark state tracking what has already been flushed.
    log:
        Bound structlog logger for contextual logging.
    """
    from api.db.session import get_session_factory

    run_id = uuid.UUID(run_id_str)

    # ------------------------------------------------------------------
    # Compute deltas
    # ------------------------------------------------------------------
    equity_curve = portfolio.get_equity_curve()
    new_equity_points = equity_curve[state.flushed_equity_count:]

    trade_history = portfolio.get_trade_history()
    new_trades = trade_history[state.flushed_trade_count:]

    all_orders = execution_engine.get_all_orders() if execution_engine is not None else []
    all_fills = execution_engine.get_all_fills() if execution_engine is not None else []
    new_fills = [f for f in all_fills if f.fill_id not in state.flushed_fill_ids]

    # Check position data availability
    has_positions = hasattr(portfolio, "_position_snapshots")

    new_orders = [o for o in all_orders if o.order_id not in state.flushed_order_ids]

    # Skip entirely when there is nothing to persist
    if (
        not new_equity_points
        and not new_trades
        and not new_orders
        and not new_fills
        and not has_positions
    ):
        log.debug("runs.incremental_flush_skipped", reason="no_new_data")
        return

    # ------------------------------------------------------------------
    # Build ORM rows
    # ------------------------------------------------------------------

    # Equity snapshots  -- use peak from state for consistent drawdown tracking
    equity_orms: list[EquitySnapshotORM] = []
    for i, (timestamp, equity) in enumerate(new_equity_points):
        if equity > state.peak_equity:
            state.peak_equity = equity
        if state.peak_equity > Decimal("0"):
            dd_pct = ((state.peak_equity - equity) / state.peak_equity).quantize(
                Decimal("0.00000001")
            )
        else:
            dd_pct = Decimal("0")
        # Clamp to [0, 1] to satisfy DB CHECK constraint
        dd_pct = max(Decimal("0"), min(dd_pct, Decimal("1")))

        # Clamp equity to 0 to satisfy ck_equity_snapshots_equity_non_negative
        if equity < Decimal("0"):
            log.warning(
                "runs.incremental_flush_negative_equity_clamped",
                bar_index=state.flushed_equity_count + i,
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
                bar_index=state.flushed_equity_count + i,
                timestamp=timestamp,
            )
        )

    # Trade rows
    trade_orms: list[TradeORM] = []
    for trade in new_trades:
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

    # Order rows  -- use merge() so existing rows are updated when status changes
    order_orms: list[OrderORM] = []
    for order in all_orders:
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

    # Fill rows  -- insert only new fills (watermarked by fill_id)
    fill_orms: list[FillORM] = []
    new_fill_ids: list[uuid.UUID] = []
    for fill in new_fills:
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
        new_fill_ids.append(fill.fill_id)

    # Position snapshot rows  -- delete existing then insert fresh
    position_orms: list[PositionSnapshotORM] = []
    now = datetime.now(tz=UTC)
    if has_positions:
        for pos in portfolio._position_snapshots.values():
            # Skip flat positions (fully closed) to avoid DB clutter
            if pos.quantity <= Decimal("0"):
                continue
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

    # ------------------------------------------------------------------
    # Persist in isolated session
    # ------------------------------------------------------------------
    try:
        factory = get_session_factory()
        async with factory() as db:
            try:
                # Orders first (FK parent of fills); use merge() so status
                # updates on already-persisted orders are applied correctly
                for order_orm in order_orms:
                    await db.merge(order_orm)
                if order_orms:
                    await db.flush()

                # New fills only (already-seen fills skipped above)
                if fill_orms:
                    db.add_all(fill_orms)

                # Equity and trade rows are always new (watermarked by index / id)
                if equity_orms:
                    db.add_all(equity_orms)
                if trade_orms:
                    db.add_all(trade_orms)

                # Positions: delete all existing snapshots for this run then
                # insert the current state (idempotent refresh)
                if has_positions:
                    await db.execute(
                        delete(PositionSnapshotORM).where(
                            PositionSnapshotORM.run_id == run_id
                        )
                    )
                    if position_orms:
                        db.add_all(position_orms)

                await db.commit()

                log.info(
                    "runs.incremental_flush_committed",
                    new_equity=len(equity_orms),
                    new_trades=len(trade_orms),
                    orders_merged=len(order_orms),
                    new_fills=len(fill_orms),
                    positions=len(position_orms),
                )

                # Advance watermarks only after a successful commit
                state.flushed_equity_count += len(equity_orms)
                state.flushed_trade_count += len(trade_orms)
                state.flushed_fill_ids.update(new_fill_ids)
                state.flushed_order_ids.update(o.order_id for o in all_orders)

            except Exception:
                await db.rollback()
                log.exception("runs.incremental_flush_db_error")
                trade_orms = []  # Do not notify on failed commit

        # Fire-and-forget Telegram trade notifications (outside DB session)
        for _trade_orm in trade_orms:
            asyncio.create_task(_notify_trade_telegram(_trade_orm))
    except Exception:
        log.exception("runs.incremental_flush_session_failed")


async def _incremental_flush_loop(
    *,
    run_id_str: str,
    portfolio: Any,
    execution_engine: Any | None,
    state: _IncrementalFlushState,
    flush_interval: float,
    log: Any,
) -> None:
    """Periodic incremental flush loop  -- runs as parallel asyncio.Task."""
    log.info("runs.incremental_flush_started", flush_interval=flush_interval)
    try:
        while True:
            await asyncio.sleep(flush_interval)
            try:
                await _flush_incremental(
                    run_id_str=run_id_str,
                    portfolio=portfolio,
                    execution_engine=execution_engine,
                    state=state,
                    log=log,
                )
            except Exception:
                log.exception("runs.incremental_flush_error")
    except asyncio.CancelledError:
        log.info("runs.incremental_flush_stopped")



# ---------------------------------------------------------------------------
# Helper: auto-stop task -- fires when max run duration is exceeded
# ---------------------------------------------------------------------------

async def _auto_stop_after(
    stop_event: asyncio.Event,
    max_seconds: float,
    run_id: str,
    log: Any,
) -> None:
    """
    Sleep for ``max_seconds`` then set ``stop_event`` to trigger a clean
    engine shutdown.

    Designed to run as a parallel asyncio.Task alongside the engine loop.
    When cancelled (normal stop path), it exits silently without setting
    the event so the engine's own stop logic remains authoritative.

    Parameters
    ----------
    stop_event:
        The StrategyEngine's internal stop event.  Setting it initiates
        a graceful shutdown of ``run_live_loop()``.
    max_seconds:
        Duration to wait before auto-stopping.
    run_id:
        String UUID used only for structured log context.
    log:
        Bound structlog logger for contextual logging.
    """
    try:
        await asyncio.sleep(max_seconds)
        log.warning(
            "runs.auto_stop_max_duration",
            run_id=run_id,
            max_hours=max_seconds / 3600,
        )
        stop_event.set()
    except asyncio.CancelledError:
        # Cancelled by the engine's finally block on normal/early shutdown.
        pass


# ---------------------------------------------------------------------------
# Helper: persist paper engine results to DB
# ---------------------------------------------------------------------------

async def _persist_paper_results(
    *,
    run_id_str: str,
    portfolio: Any,
    execution_engine: Any | None = None,
    log: Any,
) -> None:
    """
    Persist paper engine equity curve, completed trades, orders, and fills to DB.

    Called from ``_run_live_engine``'s ``finally`` block. Uses its own
    isolated DB session. Skipped when equity curve, trade history, and
    order list are all empty (engine stopped before generating any data).
    """
    from api.db.session import get_session_factory

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


async def _run_paper_engine(
    *,
    run_id_str: str,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: str,
    trailing_stop_pct: float | None = None,
    enable_adaptive_learning: bool = False,
    auto_apply_learning: bool = False,
) -> None:
    """
    Background coroutine that runs a paper trading engine for a single run.

    Creates all trading components, starts the StrategyEngine, and runs the
    live loop until stopped or errored. On exit, updates the run record in
    the database with the final status.

    This function uses its own database session (not the request session)
    because the POST handler's session is closed before this coroutine runs.

    An incremental flush task runs in parallel every 30 seconds to persist
    equity snapshots, trades, orders, fills, and position data while the run
    is still active.  A final flush is performed after the engine stops to
    capture any remaining data not covered by the last periodic flush.

    Parameters
    ----------
    run_id_str:
        String representation of the run UUID.
    strategy_cls:
        Strategy class to instantiate.
    strategy_name:
        Human-readable strategy name (used for strategy_id construction).
    strategy_params:
        Parameters to pass to the strategy constructor.
    symbols:
        CCXT-format trading pairs to trade.
    timeframe:
        Candle timeframe for the strategy.
    initial_capital:
        Starting capital in quote currency (will be converted to Decimal).
    """
    from api.config import get_settings
    from api.db.models import RunORM
    from api.db.session import get_session_factory
    from common.types import RunMode
    from data.services.ccxt_market_data import CCXTMarketDataService
    from trading.engines.paper import PaperExecutionEngine
    from trading.portfolio import PortfolioAccounting
    from trading.risk_manager import DefaultRiskManager
    from trading.strategy_engine import StrategyEngine

    log = logger.bind(run_id=run_id_str, mode="paper")
    log.info("runs.paper_engine_starting")

    final_status = "stopped"
    engine: StrategyEngine | None = None
    portfolio: Any = None
    execution: Any = None

    # Incremental flush state  -- declared at function scope so except/finally
    # blocks can access flush_task for cancellation regardless of where in
    # the try block an error occurs.
    flush_state = _IncrementalFlushState()
    flush_task: asyncio.Task[None] | None = None

    # Adaptive learning task -- opt-in per run
    learning_task: asyncio.Task[None] | None = None
    learning_stop_event: asyncio.Event | None = None

    # Auto-stop timeout task -- cancelled on normal stop (Feature: max_run_duration)
    auto_stop_task: asyncio.Task[None] | None = None

    try:
        settings = get_settings()
        capital = Decimal(initial_capital)

        # Extract exchange credentials
        api_key: str | None = None
        api_secret: str | None = None
        if settings.exchange_api_key is not None:
            api_key = settings.exchange_api_key.get_secret_value()
        if settings.exchange_api_secret is not None:
            api_secret = settings.exchange_api_secret.get_secret_value()
        if api_secret is not None:
            api_secret = _normalize_exchange_secret(api_secret)
        api_passphrase: str | None = None
        if settings.exchange_api_passphrase is not None:
            api_passphrase = settings.exchange_api_passphrase.get_secret_value()

        # Instantiate components
        market_data = CCXTMarketDataService(
            exchange_id=settings.exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            cache_ttl_seconds=60,
        )
        risk_manager = DefaultRiskManager(run_id=run_id_str)
        execution = PaperExecutionEngine(
            run_id=run_id_str,
            risk_manager=risk_manager,
            initial_cash=capital,
        )
        portfolio = PortfolioAccounting(
            run_id=run_id_str,
            initial_cash=capital,
        )
        strategy_instance = strategy_cls(
            strategy_id=f"{strategy_name}-{run_id_str.replace('-', '')[:8]}",
            params=strategy_params,
        )

        # Adaptive learning pipeline (opt-in per run)
        if enable_adaptive_learning:
            from trading.adaptive_learning import AdaptiveLearningTask

            learning_stop_event = asyncio.Event()
            adaptive_learner = AdaptiveLearningTask(
                strategies=[strategy_instance],
                auto_apply=auto_apply_learning,
                original_params=dict(strategy_params),
                check_interval_seconds=60.0,
                min_trades_per_cycle=50,
            )
            portfolio.on_trade_recorded = adaptive_learner.ingest_trade
            _LEARNING_INSTANCES[run_id_str] = adaptive_learner
            log.info(
                "runs.adaptive_learning_enabled",
                auto_apply=auto_apply_learning,
            )

        # Build engine config  -- include trailing stop if configured
        engine_config: dict[str, object] = {}
        if trailing_stop_pct is not None:
            engine_config["trailing_stop_pct"] = trailing_stop_pct

        engine = StrategyEngine(
            strategies=[strategy_instance],
            execution_engine=execution,
            risk_manager=risk_manager,
            market_data=market_data,
            portfolio=portfolio,
            symbols=symbols,
            timeframe=timeframe,
            run_mode=RunMode.PAPER,
            config=engine_config if engine_config else None,
        )

        _RUN_ENGINES[run_id_str] = engine
        await engine.start(run_id_str)
        log.info("runs.paper_engine_running")

        # Start periodic incremental flush as a parallel background task
        assert execution is not None, (
            "flush task must be created after PaperExecutionEngine is initialized"
        )
        flush_task = asyncio.create_task(
            _incremental_flush_loop(
                run_id_str=run_id_str,
                portfolio=portfolio,
                execution_engine=execution,
                state=flush_state,
                flush_interval=30.0,
                log=log,
            )
        )

        # Auto-stop timeout task -- cancels itself on normal stop
        auto_stop_task = asyncio.create_task(
            _auto_stop_after(
                stop_event=engine._stop_event,
                max_seconds=settings.max_run_duration_hours * 3600.0,
                run_id=run_id_str,
                log=log,
            )
        )

        # Start adaptive learning as parallel background task
        if learning_stop_event is not None:
            learning_task = asyncio.create_task(
                adaptive_learner.run(learning_stop_event)
            )

        await engine.run_live_loop()

    except asyncio.CancelledError:
        log.info("runs.paper_engine_cancelled")
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")
        raise  # Must re-raise CancelledError for asyncio bookkeeping

    except Exception:
        final_status = "error"
        log.exception("runs.paper_engine_error")
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")

    finally:
        # Cancel auto-stop timeout task (normal stop path)
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass

        # Belt-and-suspenders: cancel learning task if it somehow survived
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass

        # Belt-and-suspenders: cancel flush task if it somehow survived
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass

        # Remove from task registry
        _RUN_TASKS.pop(run_id_str, None)
        _RUN_ENGINES.pop(run_id_str, None)
        _LEARNING_INSTANCES.pop(run_id_str, None)

        # Final incremental flush captures any remaining data not covered by
        # the last periodic flush cycle  -- must run BEFORE status update so
        # clients see complete data when the run transitions to 'stopped'
        if portfolio is not None:
            await _flush_incremental(
                run_id_str=run_id_str,
                portfolio=portfolio,
                execution_engine=execution,
                state=flush_state,
                log=log,
            )

        # Update run status in DB using an isolated session
        try:
            factory = get_session_factory()
            async with factory() as db:
                try:
                    result = await db.execute(
                        select(RunORM).where(RunORM.id == uuid.UUID(run_id_str))
                    )
                    run = result.scalar_one_or_none()
                    if run is not None and run.status == "running":
                        now = datetime.now(tz=UTC)
                        run.status = final_status
                        run.stopped_at = now
                        run.updated_at = now
                        await db.commit()
                        log.info(
                            "runs.paper_engine_status_updated",
                            final_status=final_status,
                        )
                except Exception:
                    await db.rollback()
                    log.exception("runs.paper_engine_db_update_failed")
        except Exception:
            log.exception("runs.paper_engine_db_session_failed")


async def _run_live_engine(
    *,
    run_id_str: str,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: str,
    trailing_stop_pct: float | None = None,
    enable_adaptive_learning: bool = False,
) -> None:
    """
    Background coroutine that runs a live trading engine for a single run.

    Creates all trading components with a real CCXT exchange connection,
    starts the StrategyEngine, and runs the live loop until stopped or
    errored. On exit, updates the run record in the database with the
    final status and persists results.

    This function uses its own database session (not the request session)
    because the POST handler's session is closed before this coroutine runs.
    """
    import ccxt.async_support as ccxt_async

    from api.config import get_settings
    from api.db.models import RunORM
    from api.db.session import get_session_factory
    from common.types import RunMode
    from data.services.ccxt_market_data import CCXTMarketDataService
    from trading.engines.live import LiveExecutionEngine
    from trading.portfolio import PortfolioAccounting
    from trading.risk_manager import DefaultRiskManager
    from trading.strategy_engine import StrategyEngine

    log = logger.bind(run_id=run_id_str, mode="live")
    log.info("runs.live_engine_starting")

    final_status = "stopped"
    engine: StrategyEngine | None = None
    portfolio: Any = None
    execution: Any = None
    exchange: Any = None

    # Incremental flush state  -- mirrors paper engine pattern (Sprint 25)
    flush_state = _IncrementalFlushState()
    flush_task: asyncio.Task[None] | None = None

    # Adaptive learning task -- opt-in, auto_apply always False for live
    learning_task: asyncio.Task[None] | None = None
    learning_stop_event: asyncio.Event | None = None

    # Auto-stop timeout task -- cancelled on normal stop (Feature: max_run_duration)
    auto_stop_task: asyncio.Task[None] | None = None

    try:
        settings = get_settings()
        capital = Decimal(initial_capital)

        # Extract exchange credentials
        api_key: str | None = None
        api_secret: str | None = None
        if settings.exchange_api_key is not None:
            api_key = settings.exchange_api_key.get_secret_value()
        if settings.exchange_api_secret is not None:
            api_secret = settings.exchange_api_secret.get_secret_value()
        if api_secret is not None:
            api_secret = _normalize_exchange_secret(api_secret)
        api_passphrase: str | None = None
        if settings.exchange_api_passphrase is not None:
            api_passphrase = settings.exchange_api_passphrase.get_secret_value()

        # Build CCXT async exchange instance
        exchange_cls = getattr(ccxt_async, settings.exchange_id, None)
        if exchange_cls is None:
            raise RuntimeError(f"Unsupported CCXT exchange: {settings.exchange_id!r}")
        exchange_config: dict[str, Any] = {
            "enableRateLimit": True,
        }
        if api_key is not None:
            exchange_config["apiKey"] = api_key
        if api_secret is not None:
            exchange_config["secret"] = api_secret
        if api_passphrase is not None:
            exchange_config["password"] = api_passphrase
        exchange = exchange_cls(exchange_config)

        # Instantiate components
        market_data = CCXTMarketDataService(
            exchange_id=settings.exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            cache_ttl_seconds=60,
        )
        risk_manager = DefaultRiskManager(run_id=run_id_str)
        execution = LiveExecutionEngine(
            run_id=run_id_str,
            risk_manager=risk_manager,
            exchange=exchange,
            # Gate already enforced by LiveTradingGate in POST handler
            enable_live_trading=True,
        )
        portfolio = PortfolioAccounting(
            run_id=run_id_str,
            initial_cash=capital,
        )
        strategy_instance = strategy_cls(
            strategy_id=f"{strategy_name}-{run_id_str.replace('-', '')[:8]}",
            params=strategy_params,
        )

        # Adaptive learning pipeline (opt-in, auto_apply always False for live)
        # Safety invariant: auto_apply is never enabled for live mode
        if enable_adaptive_learning:
            from trading.adaptive_learning import AdaptiveLearningTask

            learning_stop_event = asyncio.Event()
            adaptive_learner = AdaptiveLearningTask(
                strategies=[strategy_instance],
                auto_apply=False,  # Safety: never auto-apply in live mode
                original_params=dict(strategy_params),
                check_interval_seconds=60.0,
                min_trades_per_cycle=50,
            )
            portfolio.on_trade_recorded = adaptive_learner.ingest_trade
            _LEARNING_INSTANCES[run_id_str] = adaptive_learner
            log.info("runs.adaptive_learning_enabled", auto_apply=False)

        # Build engine config  -- include trailing stop if configured
        live_engine_config: dict[str, object] = {}
        if trailing_stop_pct is not None:
            live_engine_config["trailing_stop_pct"] = trailing_stop_pct

        engine = StrategyEngine(
            strategies=[strategy_instance],
            execution_engine=execution,
            risk_manager=risk_manager,
            market_data=market_data,
            portfolio=portfolio,
            symbols=symbols,
            timeframe=timeframe,
            run_mode=RunMode.LIVE,
            config=live_engine_config if live_engine_config else None,
        )

        _RUN_ENGINES[run_id_str] = engine
        await engine.start(run_id_str)
        log.info("runs.live_engine_running")

        # Start periodic incremental flush (Sprint 25  -- mirrors paper engine)
        flush_task = asyncio.create_task(
            _incremental_flush_loop(
                run_id_str=run_id_str,
                portfolio=portfolio,
                execution_engine=execution,
                state=flush_state,
                flush_interval=30.0,
                log=log,
            )
        )

        # Auto-stop timeout task -- cancels itself on normal stop
        auto_stop_task = asyncio.create_task(
            _auto_stop_after(
                stop_event=engine._stop_event,
                max_seconds=settings.max_run_duration_hours * 3600.0,
                run_id=run_id_str,
                log=log,
            )
        )

        # Start adaptive learning as parallel background task
        if learning_stop_event is not None:
            learning_task = asyncio.create_task(
                adaptive_learner.run(learning_stop_event)
            )

        await engine.run_live_loop()

    except asyncio.CancelledError:
        log.info("runs.live_engine_cancelled")
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.live_engine_stop_error")
        raise  # Must re-raise CancelledError for asyncio bookkeeping

    except Exception:
        final_status = "error"
        log.exception("runs.live_engine_error")
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.live_engine_stop_error")

    finally:
        # Cancel auto-stop timeout task (normal stop path)
        if auto_stop_task is not None and not auto_stop_task.done():
            auto_stop_task.cancel()
            try:
                await auto_stop_task
            except (asyncio.CancelledError, Exception):
                pass

        # Belt-and-suspenders: cancel learning task if it somehow survived
        if learning_stop_event is not None:
            learning_stop_event.set()
        if learning_task is not None and not learning_task.done():
            learning_task.cancel()
            try:
                await learning_task
            except (asyncio.CancelledError, Exception):
                pass

        # Belt-and-suspenders: cancel flush task if it somehow survived
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            try:
                await flush_task
            except (asyncio.CancelledError, Exception):
                pass

        # Remove from task registry
        _RUN_TASKS.pop(run_id_str, None)
        _RUN_ENGINES.pop(run_id_str, None)
        _LEARNING_INSTANCES.pop(run_id_str, None)

        # Final incremental flush captures any remaining data not covered by
        # the last periodic flush cycle  -- must run BEFORE status update so
        # clients see complete data when the run transitions to 'stopped'
        if portfolio is not None:
            await _flush_incremental(
                run_id_str=run_id_str,
                portfolio=portfolio,
                execution_engine=execution,
                state=flush_state,
                log=log,
            )

        # Update run status in DB using an isolated session
        try:
            factory = get_session_factory()
            async with factory() as db:
                try:
                    result = await db.execute(
                        select(RunORM).where(RunORM.id == uuid.UUID(run_id_str))
                    )
                    run = result.scalar_one_or_none()
                    if run is not None and run.status == "running":
                        now = datetime.now(tz=UTC)
                        run.status = final_status
                        run.stopped_at = now
                        run.updated_at = now
                        await db.commit()
                        log.info(
                            "runs.live_engine_status_updated",
                            final_status=final_status,
                        )
                except Exception:
                    await db.rollback()
                    log.exception("runs.live_engine_db_update_failed")
        except Exception:
            log.exception("runs.live_engine_db_session_failed")

        # Close exchange connection (belt-and-suspenders; LiveExecutionEngine.on_stop
        # also closes it, but this covers cases where on_stop was never reached)
        if exchange is not None:
            try:
                await exchange.close()
            except Exception:
                log.warning("runs.live_engine_exchange_close_failed")


# ---------------------------------------------------------------------------
# Helper: fetch historical bars via CCXTMarketDataService
# ---------------------------------------------------------------------------

async def _fetch_bars_for_backtest(
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
    log: Any,
) -> dict[str, list[Any]]:
    """
    Fetch historical OHLCV bars for all symbols in the requested date range.

    Creates a transient ``CCXTMarketDataService`` instance, fetches bars for
    all symbols concurrently (within the service's semaphore limit), and
    closes the connection in a ``finally`` block.

    Parameters
    ----------
    symbols:
        CCXT-format trading pairs to fetch.
    timeframe:
        Candle timeframe.
    start:
        Inclusive start datetime (UTC).
    end:
        Exclusive end datetime (UTC).
    log:
        Bound structlog logger for contextual logging.

    Returns
    -------
    dict[str, list[OHLCVBar]]
        Bars keyed by symbol, sorted ascending by timestamp.

    Raises
    ------
    HTTPException 502:
        When the exchange is unreachable or returns an error.
    HTTPException 400:
        When a symbol is not supported by the configured exchange.
    """
    from api.config import get_settings
    from data.market_data import DataNotAvailableError, MarketDataError
    from data.services.ccxt_market_data import CCXTMarketDataService

    settings = get_settings()

    api_key: str | None = None
    api_secret: str | None = None
    if settings.exchange_api_key is not None:
        api_key = settings.exchange_api_key.get_secret_value()
    if settings.exchange_api_secret is not None:
        api_secret = settings.exchange_api_secret.get_secret_value()
    if api_secret is not None:
        api_secret = _normalize_exchange_secret(api_secret)
    api_passphrase: str | None = None
    if settings.exchange_api_passphrase is not None:
        api_passphrase = settings.exchange_api_passphrase.get_secret_value()

    service = CCXTMarketDataService(
        exchange_id=settings.exchange_id,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        cache_ttl_seconds=0,  # No caching for backtest data fetches
    )

    log.info(
        "runs.backtest_fetching_bars",
        exchange=settings.exchange_id,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    try:
        await service.connect()

        # Fetch all symbols concurrently using asyncio.gather.
        # CCXTMarketDataService's internal semaphore already throttles
        # concurrent exchange requests safely.
        tasks = [
            service.fetch_ohlcv_range(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)

        bars_by_symbol: dict[str, list[Any]] = {
            symbol: bars
            for symbol, bars in zip(symbols, results, strict=True)
        }

        for symbol, bars in bars_by_symbol.items():
            log.info(
                "runs.backtest_bars_fetched",
                symbol=symbol,
                bar_count=len(bars),
            )

        return bars_by_symbol

    except DataNotAvailableError as exc:
        log.warning("runs.backtest_data_not_available", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data not available for the requested range: {exc}",
        ) from exc
    except MarketDataError as exc:
        log.error("runs.backtest_market_data_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Exchange error fetching historical data: {exc}",
        ) from exc
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# Helper: build BacktestMetricsResponse from BacktestResult
# ---------------------------------------------------------------------------

def _build_backtest_metrics(result: Any) -> BacktestMetricsResponse:
    """
    Convert a ``BacktestResult`` into a ``BacktestMetricsResponse``.

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
# Helper: persist backtest results to DB
# ---------------------------------------------------------------------------

async def _persist_backtest_results(
    db: AsyncSession,
    run_id: uuid.UUID,
    run_orm: RunORM,
    result: Any,
    log: Any,
    execution_engine: Any | None = None,
    portfolio: Any | None = None,
) -> None:
    """
    Write all backtest result records to the database.

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
    from sqlalchemy.orm.attributes import flag_modified

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
    metrics_response = _build_backtest_metrics(result)
    updated_config = dict(run_orm.config or {})
    metrics_dict = metrics_response.model_dump(mode="json")
    # PostgreSQL JSONB rejects Infinity/NaN  -- replace with None
    import math
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


# ---------------------------------------------------------------------------
# POST /api/v1/runs  -- start a new trading run
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=RunDetailResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request (unknown strategy, bad params)"},
        403: {"description": "Live trading gate check failed (one or more safety layers not satisfied)"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Exchange unreachable (backtest data fetch)"},
    },
    summary="Start a new trading run",
    description=(
        "Create and start a new backtest, paper, or live trading run. "
        "Backtest runs execute synchronously, persist results, and complete "
        "before the response is returned. "
        "Paper and live runs are created in the database with status='running'; "
        "the live engine wiring is Sprint 2. "
        "LIVE mode requires passing the 3-layer safety gate: "
        "(1) ENABLE_LIVE_TRADING=true, (2) exchange API keys configured, "
        "(3) valid confirm_token matching LIVE_TRADING_CONFIRM_TOKEN."
    ),
)
async def create_run(
    body: RunCreateRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunDetailResponse:
    """
    Start a new trading run.

    Parameters
    ----------
    body:
        Run configuration from the request body.
    db:
        Injected async database session.

    Returns
    -------
    RunDetailResponse
        The newly created run record, including backtest metrics for
        completed backtest runs.

    Raises
    ------
    HTTPException 400:
        When the strategy name is unknown, strategy parameters fail schema
        validation, or backtest data is unavailable for the date range.
    HTTPException 403:
        When live trading gate check fails (one or more safety layers not satisfied).
    HTTPException 502:
        When the configured exchange cannot be reached to fetch historical data.
    """
    log = logger.bind(
        endpoint="create_run",
        strategy_name=body.strategy_name,
        mode=body.mode,
        symbols=body.symbols,
        timeframe=body.timeframe,
    )
    log.info("runs.create_requested")

    registry = _get_strategy_registry()

    # Validate strategy name
    strategy_name = body.strategy_name.lower().replace("-", "_")
    if strategy_name not in registry:
        log.warning("runs.unknown_strategy", strategy_name=strategy_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown strategy: {body.strategy_name!r}. "
                f"Available: {sorted(registry.keys())}"
            ),
        )

    strategy_cls = registry[strategy_name]

    # Extract trailing_stop_pct from strategy params BEFORE schema validation.
    # The UI may submit trailing_stop_pct as an empty string when the field is
    # left blank.  If it reaches the Pydantic schema validator while still a
    # string, validation raises HTTP 400 because the strategy schemas expect a
    # float.  Stripping it here ensures the validator sees a clean params dict.
    _trailing_stop_pct: float | None = None
    if "trailing_stop_pct" in body.strategy_params:
        raw_tsp = body.strategy_params.get("trailing_stop_pct")
        if raw_tsp is not None and raw_tsp != "":
            _trailing_stop_pct = float(raw_tsp)
        else:
            del body.strategy_params["trailing_stop_pct"]

    # Validate strategy parameters against the declared parameter_schema
    schema = strategy_cls.parameter_schema()
    param_errors = _validate_params_against_schema(body.strategy_params, schema)
    if param_errors:
        log.warning(
            "runs.invalid_strategy_params",
            errors=param_errors,
            strategy=strategy_name,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy parameters: {'; '.join(param_errors)}",
        )

    # Additional validation for backtest mode
    is_backtest = body.mode == "backtest"
    if is_backtest:
        if body.backtest_start is None or body.backtest_end is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="backtest_start and backtest_end are required for backtest mode",
            )
        if body.backtest_start >= body.backtest_end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="backtest_start must be before backtest_end",
            )

    # ------------------------------------------------------------------
    # 3-Layer Live Trading Safety Gate (SEC-003)
    # ------------------------------------------------------------------
    # All three layers must pass before a LIVE mode run is permitted:
    #
    #   Layer 1  -- Environment: ENABLE_LIVE_TRADING must be True.
    #   Layer 2  -- API Keys: EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be non-empty.
    #   Layer 3  -- Confirmation Token: A runtime token provided in the request body
    #             must match LIVE_TRADING_CONFIRM_TOKEN (hmac.compare_digest).
    #
    # If any layer fails, the endpoint returns HTTP 403 with a structured
    # response identifying which layer(s) failed.
    # ------------------------------------------------------------------
    if body.mode == "live":
        from api.config import get_settings
        from trading.safety import LiveTradingGate

        settings = get_settings()
        gate = LiveTradingGate()
        gate_result = gate.check_gate(
            settings=settings,
            confirm_token=body.confirm_token or "",
        )

        if not gate_result.passed:
            log.warning(
                "runs.live_trading_gate_failed",
                failures=gate_result.failures,
                layer_results=gate_result.layer_results,
            )
            failed_layers = [
                layer.name for layer in gate_result.layers if not layer.passed
            ]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Live trading gate check failed. "
                    f"Failed layers: {', '.join(failed_layers)}. "
                    "See server logs for details."
                ),
            )

        log.info(
            "runs.live_trading_gate_passed",
            layer_results=gate_result.layer_results,
        )

    timeframe = TimeFrame(str(body.timeframe))

    # Build the config snapshot stored immutably on the run record
    run_id = uuid.uuid4()
    config_snapshot: dict[str, Any] = {
        "strategy_name": strategy_name,
        "strategy_params": body.strategy_params,
        "symbols": body.symbols,
        "timeframe": str(body.timeframe) if hasattr(body.timeframe, "value") else body.timeframe,
        "mode": str(body.mode) if hasattr(body.mode, "value") else body.mode,
        "initial_capital": body.initial_capital,
    }
    if body.backtest_start is not None:
        config_snapshot["backtest_start"] = body.backtest_start.isoformat()
    if body.backtest_end is not None:
        config_snapshot["backtest_end"] = body.backtest_end.isoformat()

    now = datetime.now(tz=UTC)

    # Determine mode string for ORM
    mode_value = str(body.mode)

    run_orm = RunORM(
        id=run_id,
        run_mode=mode_value,
        status="running",
        config=config_snapshot,
        started_at=now,
        created_at=now,
        updated_at=now,
    )

    db.add(run_orm)
    await db.flush()  # Assign the PK within the transaction without committing

    log.info(
        "runs.created",
        run_id=str(run_id),
        mode=mode_value,
        strategy=strategy_name,
    )

    # ------------------------------------------------------------------
    # BACKTEST MODE  -- execute synchronously, persist results, finish run
    # ------------------------------------------------------------------
    if is_backtest:
        try:
            # Step 1: Fetch historical OHLCV bars
            bars_by_symbol = await _fetch_bars_for_backtest(
                symbols=body.symbols,
                timeframe=timeframe,
                start=body.backtest_start,  # type: ignore[arg-type]
                end=body.backtest_end,       # type: ignore[arg-type]
                log=log,
            )

            # Step 2: Instantiate strategy
            strategy_instance = strategy_cls(
                strategy_id=f"{strategy_name}-{run_id.hex[:8]}",
                params=body.strategy_params,
            )

            # Step 3: Instantiate and run BacktestRunner
            from trading.backtest import BacktestRunner

            runner = BacktestRunner(
                strategies=[strategy_instance],
                symbols=body.symbols,
                timeframe=timeframe,
                initial_capital=Decimal(body.initial_capital),
                trailing_stop_pct=_trailing_stop_pct,
            )

            log.info("runs.backtest_execution_starting", run_id=str(run_id))
            result = await runner.run(bars_by_symbol)

            # Step 4: Persist results (trades + equity curve + metrics in config)
            await _persist_backtest_results(
                db=db,
                run_id=run_id,
                run_orm=run_orm,
                result=result,
                log=log,
                execution_engine=runner.last_execution_engine,
                portfolio=runner.last_portfolio,
            )

            # Step 5: Mark run as stopped
            finish_time = datetime.now(tz=UTC)
            run_orm.status = "stopped"
            run_orm.stopped_at = finish_time
            run_orm.updated_at = finish_time

            await db.flush()

            log.info(
                "runs.backtest_completed",
                run_id=str(run_id),
                total_return=f"{result.total_return_pct:.4%}",
                sharpe=f"{result.sharpe_ratio:.3f}",
                total_trades=result.total_trades,
            )

        except HTTPException:
            # Data fetch errors (400, 502)  -- mark run as error and re-raise
            error_time = datetime.now(tz=UTC)
            run_orm.status = "error"
            run_orm.stopped_at = error_time
            run_orm.updated_at = error_time
            await db.flush()
            raise

        except ValueError as exc:
            # BacktestRunner._validate_bars raised a data-quality error
            # (empty bars, insufficient warm-up, non-chronological data).
            error_time = datetime.now(tz=UTC)
            run_orm.status = "error"
            run_orm.stopped_at = error_time
            run_orm.updated_at = error_time
            await db.flush()

            log.warning(
                "runs.backtest_data_quality_error",
                run_id=str(run_id),
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Backtest data quality check failed: {exc}. "
                    "Verify your date range provides sufficient bars "
                    "for the requested strategy."
                ),
            ) from exc

        except Exception as exc:
            # Unexpected backtest execution errors
            error_time = datetime.now(tz=UTC)
            run_orm.status = "error"
            run_orm.stopped_at = error_time
            run_orm.updated_at = error_time
            await db.flush()

            log.error(
                "runs.backtest_execution_error",
                run_id=str(run_id),
                error=str(exc),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Backtest execution failed. See server logs for details.",
            ) from exc

    # ------------------------------------------------------------------
    # PAPER MODE  -- launch StrategyEngine as a background asyncio.Task
    # ------------------------------------------------------------------
    elif mode_value == "paper":
        task = asyncio.create_task(
            _run_paper_engine(
                run_id_str=str(run_id),
                strategy_cls=strategy_cls,
                strategy_name=strategy_name,
                strategy_params=body.strategy_params,
                symbols=body.symbols,
                timeframe=timeframe,
                initial_capital=body.initial_capital,
                trailing_stop_pct=_trailing_stop_pct,
                enable_adaptive_learning=body.enable_adaptive_learning,
                auto_apply_learning=body.auto_apply_learning,
            ),
            name=f"paper-engine-{run_id}",
        )
        _RUN_TASKS[str(run_id)] = task
        log.info("runs.paper_engine_task_created", run_id=str(run_id))

    # ------------------------------------------------------------------
    # LIVE MODE  -- launch LiveExecutionEngine as a background asyncio.Task
    # The 3-layer LiveTradingGate is enforced above before reaching here.
    # ------------------------------------------------------------------
    elif mode_value == "live":
        task = asyncio.create_task(
            _run_live_engine(
                run_id_str=str(run_id),
                strategy_cls=strategy_cls,
                strategy_name=strategy_name,
                strategy_params=body.strategy_params,
                symbols=body.symbols,
                timeframe=timeframe,
                initial_capital=body.initial_capital,
                trailing_stop_pct=_trailing_stop_pct,
                enable_adaptive_learning=body.enable_adaptive_learning,
            ),
            name=f"live-engine-{run_id}",
        )
        _RUN_TASKS[str(run_id)] = task
        log.info("runs.live_engine_task_created", run_id=str(run_id))

    return _run_orm_to_detail_response(run_orm)


# ---------------------------------------------------------------------------
# GET /api/v1/runs  -- list all runs
# ---------------------------------------------------------------------------

_VALID_MODES: frozenset[str] = frozenset({"backtest", "paper", "live"})
_VALID_STATUSES: frozenset[str] = frozenset({"running", "stopped", "error", "archived"})


@router.get(
    "",
    response_model=RunListResponse,
    summary="List all trading runs",
    description="Returns a paginated list of all runs, ordered by creation time descending.",
)
async def list_runs(
    db: Annotated[AsyncSession, Depends(get_db)],
    offset: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 50,
    mode: Annotated[
        str | None,
        Query(description="Filter by run mode: backtest, paper, live"),
    ] = None,
    run_status: Annotated[
        str | None,
        Query(alias="status", description="Filter by status: running, stopped, error"),
    ] = None,
    strategy: Annotated[
        str | None,
        Query(description="Filter by strategy name (exact match on config JSONB)"),
    ] = None,
    symbol: Annotated[
        str | None,
        Query(description="Filter by symbol (runs containing this symbol)"),
    ] = None,
    created_after: Annotated[
        str | None,
        Query(description="Filter runs created after this ISO date"),
    ] = None,
    created_before: Annotated[
        str | None,
        Query(description="Filter runs created before this ISO date"),
    ] = None,
    include_archived: Annotated[
        bool,
        Query(description="When true, include archived runs in results (default: false)"),
    ] = False,
) -> RunListResponse:
    """
    List all trading runs with pagination and optional server-side filtering.

    Parameters
    ----------
    db:
        Injected async database session.
    offset:
        Number of records to skip.
    limit:
        Maximum records to return.
    mode:
        Optional filter by run mode.  Must be one of backtest, paper,
        or live when supplied.
    run_status:
        Optional filter by run status (query param name: status).  Must be
        one of running, stopped, or error when supplied.
    strategy:
        Optional exact match on the strategy_name key inside the config JSONB.
    symbol:
        Optional substring match against the symbols array in config JSONB.
    created_after:
        Optional ISO-8601 lower bound on created_at (inclusive).
    created_before:
        Optional ISO-8601 upper bound on created_at (inclusive).

    Returns
    -------
    RunListResponse
        Paginated list of run records matching the supplied filters.
    """
    log = logger.bind(
        endpoint="list_runs",
        offset=offset,
        limit=limit,
        mode=mode,
        status=run_status,
        strategy=strategy,
        symbol=symbol,
        created_after=created_after,
        created_before=created_before,
    )
    log.info("runs.list_requested")

    # Validate optional filter values
    if mode is not None and mode not in _VALID_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}",
        )
    if run_status is not None and run_status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status '{run_status}'. Must be one of: {sorted(_VALID_STATUSES)}",
        )

    # Build filter conditions
    filters = []
    if mode is not None:
        filters.append(RunORM.run_mode == mode)
    if run_status is not None:
        filters.append(RunORM.status == run_status)

    # Exclude archived runs by default -- callers must opt-in to see them
    if not include_archived:
        filters.append(RunORM.status != "archived")

    # Strategy filter — exact match on config->strategy_name (JSONB text extraction)
    if strategy is not None:
        filters.append(RunORM.config["strategy_name"].astext == strategy)

    # Symbol filter — substring match against the JSON-serialised symbols array
    if symbol is not None:
        filters.append(cast(RunORM.config["symbols"], String).contains(symbol))

    # Date range filters — parse ISO-8601, raise 422 for malformed input
    if created_after is not None:
        try:
            dt_after = datetime.fromisoformat(created_after.replace("Z", "+00:00"))
            filters.append(RunORM.created_at >= dt_after)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid created_after date: {created_after}",
            )

    if created_before is not None:
        try:
            dt_before = datetime.fromisoformat(created_before.replace("Z", "+00:00"))
            filters.append(RunORM.created_at <= dt_before)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid created_before date: {created_before}",
            )

    # Count total matching rows
    count_stmt = select(func.count()).select_from(RunORM)
    if filters:
        count_stmt = count_stmt.where(*filters)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Fetch the page
    # noload() explicitly prevents any lazy relationship traversal on the
    # list response (RunResponse only uses scalar columns).  This is a
    # defensive N+1 guard: if a future serializer accidentally iterates a
    # relationship, SQLAlchemy raises an error instead of silently firing
    # one query per row (2.0-safe lazy-load hygiene, CR-N1-001).
    page_stmt = select(RunORM).options(
        noload(RunORM.trades),
        noload(RunORM.orders),
        noload(RunORM.equity_snapshots),
        noload(RunORM.position_snapshots),
        noload(RunORM.signals),
        noload(RunORM.skipped_trades),
    )
    if filters:
        page_stmt = page_stmt.where(*filters)
    page_stmt = page_stmt.order_by(RunORM.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(page_stmt)
    runs = list(result.scalars().all())

    log.info("runs.listed", total=total, returned=len(runs))

    return RunListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[_run_orm_to_response(r) for r in runs],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{run_id}  -- get a single run
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get a single run's details",
    description=(
        "Returns full run details. For completed backtest runs the response "
        "includes a ``backtest_metrics`` object with all performance metrics."
    ),
)
async def get_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunDetailResponse:
    """
    Retrieve details of a specific trading run.

    Parameters
    ----------
    run_id:
        UUID of the run to retrieve.
    db:
        Injected async database session.

    Returns
    -------
    RunDetailResponse
        The run record, with backtest_metrics populated for backtest runs.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    log = logger.bind(endpoint="get_run", run_id=str(run_id))
    log.info("runs.get_requested")

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    log.info("runs.found", status=run.status)
    return _run_orm_to_detail_response(run)


# ---------------------------------------------------------------------------
# DELETE /api/v1/runs/{run_id}  -- stop a running run
# ---------------------------------------------------------------------------

@router.delete(
    "/{run_id}",
    status_code=status.HTTP_200_OK,
    response_model=RunDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        409: {"model": ErrorResponse, "description": "Run is not in a stoppable state"},
    },
    summary="Stop a running trading run",
    description=(
        "Transitions a run from 'running' to 'stopped'. "
        "Returns 409 if the run is already stopped or errored."
    ),
)
async def stop_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunDetailResponse:
    """
    Stop a running trading run.

    Parameters
    ----------
    run_id:
        UUID of the run to stop.
    db:
        Injected async database session.

    Returns
    -------
    RunDetailResponse
        The updated run record with status='stopped'.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    HTTPException 409:
        When the run is already in a terminal state (stopped/error).
    """
    log = logger.bind(endpoint="stop_run", run_id=str(run_id))
    log.info("runs.stop_requested")

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.status != "running":
        log.warning("runs.not_stoppable", current_status=run.status)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot stop run {run_id}: "
                f"current status is '{run.status}'. Only 'running' runs can be stopped."
            ),
        )

    now = datetime.now(tz=UTC)
    run.status = "stopped"
    run.stopped_at = now
    run.updated_at = now

    await db.flush()

    # Cancel the background task if one exists for this run
    task = _RUN_TASKS.pop(str(run_id), None)
    _RUN_ENGINES.pop(str(run_id), None)
    if task is not None and not task.done():
        task.cancel()
        log.info("runs.engine_task_cancelled", run_id=str(run_id))

    log.info("runs.stopped", run_id=str(run_id))
    return _run_orm_to_detail_response(run)



# ---------------------------------------------------------------------------
# PATCH /api/v1/runs/{run_id}/archive  -- soft-archive a finished run
# ---------------------------------------------------------------------------

@router.patch(
    "/{run_id}/archive",
    response_model=RunDetailResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Run is still running"},
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Archive a stopped or error run",
    description=(
        "Transitions a run to \'archived\' status, hiding it from the default "
        "list view.  Only runs that are already stopped or in error state may "
        "be archived.  Use GET /runs?include_archived=true to retrieve archived runs."
    ),
)
async def archive_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RunDetailResponse:
    """
    Archive a finished trading run.

    Archiving is a soft operation: the run record is retained in the database
    with status='archived' and is excluded from the default listing.  It
    remains accessible via GET /runs/{run_id} and via
    GET /runs?include_archived=true.

    Parameters
    ----------
    run_id:
        UUID of the run to archive.
    db:
        Injected async database session.

    Returns
    -------
    RunDetailResponse
        The updated run record with status='archived'.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    HTTPException 400:
        When the run is currently running (must be stopped first).
    """
    log = logger.bind(endpoint="archive_run", run_id=str(run_id))
    log.info("runs.archive_requested")

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.status == "running":
        log.warning("runs.archive_blocked_running", current_status=run.status)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot archive run {run_id}: run is currently active. "
                "Stop it first, then archive."
            ),
        )

    run.status = "archived"
    run.updated_at = datetime.now(tz=UTC)
    await db.commit()
    await db.refresh(run)

    log.info("runs.archived", run_id=str(run_id))
    return _run_orm_to_detail_response(run)


# ---------------------------------------------------------------------------
# Live diagnostics endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}/diagnostics",
    summary="Get live diagnostics for a running run",
)
async def get_diagnostics(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """
    Return current indicator values and engine state for a running run.

    Provides a lightweight status snapshot: current equity, drawdown, trade/order
    counts, and the latest Fear & Greed Index reading.  The endpoint is read-only
    and works for runs in any status, but the equity values are most meaningful
    while the run is in the *running* state.

    Parameters
    ----------
    run_id:
        UUID of the run to inspect.
    db:
        Injected async database session.

    Returns
    -------
    dict
        JSON object with run metadata and real-time diagnostic values.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    """
    result = await db.execute(select(RunORM).where(RunORM.id == run_id))
    run: RunORM | None = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # Most-recent equity snapshot (ordered by timestamp DESC)
    eq_result = await db.execute(
        select(EquitySnapshotORM)
        .where(EquitySnapshotORM.run_id == run_id)
        .order_by(EquitySnapshotORM.timestamp.desc())
        .limit(1)
    )
    latest_equity: EquitySnapshotORM | None = eq_result.scalar_one_or_none()

    # Trade count
    trade_count_result = await db.execute(
        select(func.count()).select_from(TradeORM).where(TradeORM.run_id == run_id)
    )
    trade_count: int = trade_count_result.scalar() or 0

    # Order count
    order_count_result = await db.execute(
        select(func.count()).select_from(OrderORM).where(OrderORM.run_id == run_id)
    )
    order_count: int = order_count_result.scalar() or 0

    # Fear & Greed Index (best-effort - None when FGI client not available)
    fgi_value: float | None = None
    fgi_regime: str | None = None
    try:
        from data.sentiment import get_global_client

        client = get_global_client()
        if client is not None:
            fgi_value = client.cached_value
            if fgi_value is not None:
                if fgi_value < 25:
                    fgi_regime = "EXTREME_FEAR"
                elif fgi_value < 45:
                    fgi_regime = "FEAR"
                elif fgi_value <= 55:
                    fgi_regime = "NEUTRAL"
                elif fgi_value <= 75:
                    fgi_regime = "GREED"
                else:
                    fgi_regime = "EXTREME_GREED"
    except Exception:  # noqa: BLE001 - best-effort; FGI must never break diagnostics
        pass

    return {
        "runId": str(run_id),
        "status": run.status,
        "mode": run.run_mode,
        "strategy": run.config.get("strategy_name") if run.config else None,
        "symbols": run.config.get("symbols", []) if run.config else [],
        "timeframe": run.config.get("timeframe") if run.config else None,
        "currentEquity": str(latest_equity.equity) if latest_equity else None,
        "drawdownPct": float(latest_equity.drawdown_pct) if latest_equity else None,
        "lastUpdated": latest_equity.timestamp.isoformat() if latest_equity else None,
        "tradeCount": trade_count,
        "orderCount": order_count,
        "fearGreedIndex": fgi_value,
        "fearGreedRegime": fgi_regime,
        "isRunning": run.status == "running",
    }


# ---------------------------------------------------------------------------
# Parameter schema validation helper
# ---------------------------------------------------------------------------

def _validate_params_against_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """
    Perform lightweight JSON-Schema-style validation of strategy parameters.

    Only validates ``required`` fields and known ``properties`` types.
    Full JSON Schema validation (jsonschema library) is deferred to Sprint 2
    when strategies gain more complex parameter constraints.

    Parameters
    ----------
    params:
        The parameters submitted in the request.
    schema:
        JSON Schema dict from ``BaseStrategy.parameter_schema()``.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list = valid.
    """
    errors: list[str] = []

    required_fields: list[str] = schema.get("required", [])
    for field_name in required_fields:
        if field_name not in params:
            errors.append(f"Required parameter missing: '{field_name}'")

    properties: dict[str, Any] = schema.get("properties", {})
    for param_name, param_value in params.items():
        if param_name not in properties:
            if not schema.get("additionalProperties", True):
                errors.append(f"Unknown parameter: '{param_name}'")
            continue

        prop_schema = properties[param_name]
        expected_type = prop_schema.get("type")

        if expected_type == "integer" and not isinstance(param_value, int):
            errors.append(
                f"Parameter '{param_name}' must be an integer, "
                f"got {type(param_value).__name__}"
            )
        elif expected_type == "number" and not isinstance(param_value, (int, float)):
            errors.append(
                f"Parameter '{param_name}' must be a number, "
                f"got {type(param_value).__name__}"
            )
        elif expected_type == "string" and not isinstance(param_value, str):
            errors.append(
                f"Parameter '{param_name}' must be a string, "
                f"got {type(param_value).__name__}"
            )

        minimum = prop_schema.get("minimum")
        if minimum is not None and isinstance(param_value, (int, float)):
            if param_value < minimum:
                errors.append(
                    f"Parameter '{param_name}' must be >= {minimum}, "
                    f"got {param_value}"
                )

        maximum = prop_schema.get("maximum")
        if maximum is not None and isinstance(param_value, (int, float)):
            if param_value > maximum:
                errors.append(
                    f"Parameter '{param_name}' must be <= {maximum}, "
                    f"got {param_value}"
                )

    return errors

# ---------------------------------------------------------------------------
# Startup helper: recover orphaned paper/live runs (Sprint 24)
# ---------------------------------------------------------------------------


async def _mark_orphan_error(
    factory: Any,
    run_id: uuid.UUID,
    log: Any,
) -> None:
    """Mark an orphaned run as error so it does not stay running forever."""
    try:
        async with factory() as session:
            result = await session.execute(
                select(RunORM).where(RunORM.id == run_id)
            )
            stale = result.scalar_one_or_none()
            if stale is not None and stale.status == "running":
                now = datetime.now(tz=UTC)
                stale.status = "error"
                stale.stopped_at = now
                stale.updated_at = now
                await session.commit()
    except Exception:
        log.exception("recovery.mark_error_failed", run_id=str(run_id))


async def recover_orphaned_runs() -> int:
    """Recover orphaned paper/live runs on API startup.

    When the API container restarts, any paper or live run that was in
    ``status='running'`` is an orphan -- its asyncio.Task has been killed and
    will never update the DB again.  This function:

    1. Queries the DB for all runs with status='running' and run_mode in
       ('paper', 'live').
    2. For each orphan:
       a. Marks the original as status='error', stopped_at=now().
       b. Creates a new RunORM with a fresh UUID, copying config verbatim,
          and setting ``recovered_from_run_id`` to the orphan's ID.
       c. Starts the appropriate background coroutine (_run_paper_engine or
          _run_live_engine) and registers the new task in _RUN_TASKS.
    3. For live-mode orphans, re-checks that layers 1 and 2 of the safety
       gate are satisfied (env flag + API keys). Layer 3 (confirm_token) is
       a runtime-only gate and is skipped here -- the operator already proved
       intent when the original run was created.  If layers 1/2 fail the
       live orphan is skipped (marked error only; no new run is created).

    Each orphan is processed in its own try/except so a single bad run does
    not prevent the others from being recovered.  The entire function is
    wrapped in a top-level try/except so a DB error at startup does not crash
    the API process.

    Returns
    -------
    int
        Number of runs successfully recovered (new tasks started).
    """
    from api.config import get_settings
    from api.db.models import RunORM
    from api.db.session import get_session_factory

    log = logger.bind(component="recovery")
    recovered_count = 0

    try:
        factory = get_session_factory()

        # --- Step 1: find orphaned runs ---
        # Exclude runs that were themselves recovered (non-null recovered_from_run_id)
        # to prevent ever-deepening recovery chains on repeated restarts (CR-001).
        async with factory() as session:
            result = await session.execute(
                select(RunORM).where(
                    RunORM.status == "running",
                    RunORM.run_mode.in_(["paper", "live"]),
                    RunORM.recovered_from_run_id.is_(None),
                )
            )
            orphans = list(result.scalars().all())

        if not orphans:
            log.debug("recovery.no_orphans_found")
            return 0

        log.info("recovery.orphans_found", count=len(orphans))

        settings = get_settings()

        for orphan in orphans:
            orphan_id = str(orphan.id)
            orphan_mode = orphan.run_mode
            orphan_config = dict(orphan.config or {})
            log.info(
                "recovery.found_orphan",
                run_id=orphan_id,
                mode=orphan_mode,
                strategy=orphan_config.get("strategy_name"),
            )

            try:
                # --- Extract config fields ---
                strategy_name: str | None = orphan_config.get("strategy_name")
                symbols: list[str] = orphan_config.get("symbols") or []
                timeframe_str: str = orphan_config.get("timeframe", "1h")
                initial_capital: str = orphan_config.get("initial_capital", "10000")
                strategy_params: dict[str, Any] = orphan_config.get("strategy_params") or {}

                # Validate strategy is still registered
                if not strategy_name:
                    log.warning("recovery.orphan_skipped", run_id=orphan_id, reason="missing_strategy_name")
                    await _mark_orphan_error(factory, orphan.id, log)
                    continue

                registry = _get_strategy_registry()
                strategy_cls = registry.get(strategy_name)
                if strategy_cls is None:
                    log.warning("recovery.orphan_skipped", run_id=orphan_id, reason="unknown_strategy", strategy_name=strategy_name)
                    await _mark_orphan_error(factory, orphan.id, log)
                    continue

                if not symbols:
                    log.warning("recovery.orphan_skipped", run_id=orphan_id, reason="empty_symbols")
                    await _mark_orphan_error(factory, orphan.id, log)
                    continue

                # Validate timeframe before DB write (CR-005)
                try:
                    timeframe = TimeFrame(timeframe_str)
                except ValueError:
                    log.warning("recovery.orphan_skipped", run_id=orphan_id, reason="invalid_timeframe", timeframe=timeframe_str)
                    await _mark_orphan_error(factory, orphan.id, log)
                    continue

                # --- Live-mode safety gate re-check (layers 1 + 2 only) ---
                if orphan_mode == "live":
                    env_ok = settings.enable_live_trading
                    keys_ok = (
                        settings.exchange_api_key is not None
                        and settings.exchange_api_secret is not None
                        and settings.exchange_api_key.get_secret_value().strip() != ""
                        and settings.exchange_api_secret.get_secret_value().strip() != ""
                    )
                    if not env_ok or not keys_ok:
                        log.warning("recovery.live_orphan_skipped_gate", run_id=orphan_id, env_ok=env_ok, keys_ok=keys_ok)
                        await _mark_orphan_error(factory, orphan.id, log)
                        continue

                # --- Atomically mark original as error and create recovery run ---
                new_run_id = uuid.uuid4()
                new_run_id_str = str(new_run_id)

                async with factory() as session:
                    result2 = await session.execute(
                        select(RunORM).where(RunORM.id == orphan.id)
                    )
                    stale = result2.scalar_one_or_none()
                    if stale is None or stale.status != "running":
                        log.debug("recovery.orphan_already_handled", run_id=orphan_id)
                        continue

                    now = datetime.now(tz=UTC)
                    stale.status = "error"
                    stale.stopped_at = now
                    stale.updated_at = now

                    new_run = RunORM(
                        id=new_run_id,
                        run_mode=orphan_mode,
                        status="running",
                        config=orphan_config,
                        started_at=now,
                        recovered_from_run_id=orphan.id,
                    )
                    session.add(new_run)
                    await session.commit()

                log.info("recovery.db_records_written", original_run_id=orphan_id, new_run_id=new_run_id_str)

                # --- Start background engine task ---
                # Extract trailing_stop_pct from saved strategy params (Sprint 27)
                recovery_trailing_pct: float | None = None
                raw_tsp = strategy_params.get("trailing_stop_pct")
                if raw_tsp is not None:
                    recovery_trailing_pct = float(raw_tsp)

                coro = _run_paper_engine if orphan_mode == "paper" else _run_live_engine
                task = asyncio.create_task(
                    coro(
                        run_id_str=new_run_id_str,
                        strategy_cls=strategy_cls,
                        strategy_name=strategy_name,
                        strategy_params=strategy_params,
                        symbols=symbols,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        trailing_stop_pct=recovery_trailing_pct,
                    ),
                    name=f"recovery-{orphan_mode}-{new_run_id_str[:8]}",
                )
                _RUN_TASKS[new_run_id_str] = task

                log.info(
                    "recovery.run_recovered",
                    original_run_id=orphan_id,
                    new_run_id=new_run_id_str,
                    mode=orphan_mode,
                )
                recovered_count += 1

            except Exception:
                log.exception(
                    "recovery.run_failed",
                    run_id=orphan_id,
                )
                # Continue to next orphan -- one bad run must not block others

    except Exception:
        log.exception("recovery.fatal_error")

    return recovered_count
