"""
apps/api/services/run_orchestrator.py
--------------------------------------
Paper and live engine orchestration extracted from :mod:`api.routers.runs`
in Sprint 40 Stap 2b as part of the AR-001 decomposition.

Public API
----------
- :data:`_RUN_TASKS` / :data:`_RUN_ENGINES` / :data:`_LEARNING_INSTANCES`
  module-level dicts tracking active run resources.  Stap 2c migrates these
  to :class:`api.run_registry.RunRegistry` on the ``AppContainer``.
- :class:`_IncrementalFlushState` -- watermark dataclass.
- :func:`notify_trade_telegram` -- fire-and-forget Telegram notifier.
- :func:`flush_incremental` -- one incremental DB flush cycle.
- :func:`incremental_flush_loop` -- periodic 30s flush task.
- :func:`auto_stop_after` -- max-duration guard task.
- :func:`run_paper_engine` -- paper engine lifecycle coroutine.
- :func:`run_live_engine` -- live engine lifecycle coroutine.

Backwards compatibility
-----------------------
``api.routers.runs`` re-exports each function under its original
underscore-prefixed name so existing internal callers and external
consumers (circuit_breaker.py reads ``_RUN_ENGINES`` from runs.py, etc.)
keep working unchanged.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import delete, select

from api.db.models import (
    EquitySnapshotORM,
    FillORM,
    OrderORM,
    PositionSnapshotORM,
    TradeORM,
)
from api.services.audit_log import record_audit_event
from api.services.run_persistence import persist_paper_results as _persist_paper_results
from common.types import TimeFrame

__all__ = [
    "_IncrementalFlushState",
    "_LEARNING_INSTANCES",
    "_RUN_ENGINES",
    "_RUN_TASKS",
    "auto_stop_after",
    "flush_incremental",
    "incremental_flush_loop",
    "notify_trade_telegram",
    "run_live_engine",
    "run_paper_engine",
]

logger = structlog.get_logger(__name__)


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


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Interval (seconds) between incremental DB flushes during paper/live runs.
#: Tune to balance DB write pressure vs dashboard freshness — 30 s keeps
#: Grafana panels current without hammering the connection pool.
_PAPER_FLUSH_INTERVAL_SECONDS: float = 30.0

# Mirrors trading.strategy_engine._HALT_AUTO_STOP_REASON — duplicated to
# avoid an api→trading→api import cycle. Both must stay in sync.
_HALT_AUTO_STOP_REASON_VALUE: str = "circuit_breaker_halt"


# ---------------------------------------------------------------------------
# Shared helper: cancel engine side-tasks on engine teardown (CR-001)
# ---------------------------------------------------------------------------

async def _cancel_engine_side_tasks(
    *,
    auto_stop_task: asyncio.Task[None] | None,
    learning_task: asyncio.Task[None] | None,
    learning_stop_event: asyncio.Event | None,
    flush_task: asyncio.Task[None] | None,
) -> None:
    """Cancel every long-lived side task spawned by the paper/live engine loop.

    Called from the ``except asyncio.CancelledError``, ``except Exception`` and
    ``finally`` branches of ``run_paper_engine`` and ``run_live_engine``.
    Each task is cancelled only when it is still pending; the cancellation is
    awaited so cleanup coroutines finish before the engine tears down, and
    both ``CancelledError`` and other exceptions are swallowed so the
    teardown never re-raises.
    """
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

async def notify_trade_telegram(trade_orm: TradeORM) -> None:
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

# ---------------------------------------------------------------------------
# Helper: incremental flush of paper engine data to DB during active run
# ---------------------------------------------------------------------------

async def flush_incremental(
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
                # QT-009: persist execution-quality telemetry when present
                expected_price=getattr(fill, "expected_price", None),
                slippage_bps_realized=getattr(fill, "slippage_bps_realized", None),
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


async def incremental_flush_loop(
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

            # Reconcile open orders with the exchange before flushing.
            # Coinbase processes market orders asynchronously — orders may
            # stay OPEN after submission.  Reconciliation polls the exchange
            # and transitions them to FILLED so position tracking is correct.
            if execution_engine is not None and hasattr(execution_engine, "reconcile_open_orders"):
                try:
                    await execution_engine.reconcile_open_orders()
                except Exception:
                    log.exception("runs.reconcile_open_orders_error")

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

async def auto_stop_after(
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
# ``_persist_paper_results`` lives in ``api.services.run_persistence`` since
# Sprint 40 Stap 2a; the top-level import block re-exports it under the
# original underscore-prefixed name for backwards compatibility.


# ---------------------------------------------------------------------------
# Auto-retry for crashed PAPER runs (Sprint 48 D2 design)
# ---------------------------------------------------------------------------
# A paper engine task that dies with an exception used to stay dead until the
# next API restart picked it up via orphan recovery.  Auto-retry recreates the
# run (new RunORM row linked via recovered_from_run_id, like recovery) after
# an exponential backoff, up to a bounded number of attempts.
#
# Deliberately PAPER-ONLY: a crashed LIVE engine requires operator attention
# (alerts fire via the safety gauges); silently relaunching real-money trading
# is not acceptable.  Backoff: 60s, 120s, 240s for attempts 0->1, 1->2, 2->3.
_AUTO_RETRY_MAX_ATTEMPTS = 3
_AUTO_RETRY_BASE_DELAY_SECONDS = 60.0


async def _auto_retry_paper_run(
    *,
    crashed_run_id: str,
    auto_retry_attempt: int,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: str,
    trailing_stop_pct: float | None,
    bracket_config: dict[str, object] | None,
    enable_adaptive_learning: bool,
    auto_apply_learning: bool,
) -> None:
    """Recreate a crashed paper run after a backoff (bounded attempts).

    ``auto_retry_attempt`` is the attempt counter of the CRASHED run (0 for
    an original run); the new run carries ``auto_retry_attempt + 1``.

    Notes
    -----
    Retry fires on ANY unhandled engine exception, including deterministic
    strategy bugs — monitor ``runs.auto_retry_exhausted`` as the signal that
    a run needs human attention.  The bounded-attempts guarantee is
    per-process-lifetime: orphan recovery re-reads the counter from the run
    config at API restart (CR-002), but a recovery-created run starts a
    fresh lineage.
    """
    from api.db.models import RunORM
    from api.db.session import get_session_factory

    log = logger.bind(crashed_run_id=crashed_run_id)
    next_attempt = auto_retry_attempt + 1
    if next_attempt > _AUTO_RETRY_MAX_ATTEMPTS:
        log.warning(
            "runs.auto_retry_exhausted",
            attempts=auto_retry_attempt,
            max_attempts=_AUTO_RETRY_MAX_ATTEMPTS,
        )
        return

    delay = _AUTO_RETRY_BASE_DELAY_SECONDS * (2**auto_retry_attempt)
    log.info(
        "runs.auto_retry_scheduled",
        next_attempt=next_attempt,
        delay_seconds=delay,
    )
    await asyncio.sleep(delay)

    try:
        new_run_id = uuid.uuid4()
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(RunORM).where(RunORM.id == uuid.UUID(crashed_run_id))
            )
            crashed = result.scalar_one_or_none()
            if crashed is None:
                log.warning("runs.auto_retry_source_missing")
                return
            # CR-001 guard: only retry a run that is still in 'error'.  If an
            # operator manually restarted/archived it during the backoff
            # window, creating a second engine would double exposure.
            if crashed.status != "error":
                log.info(
                    "runs.auto_retry_superseded",
                    crashed_status=crashed.status,
                )
                return
            new_config = dict(crashed.config or {})
            # Observability only — orphan recovery reads this back explicitly
            # at its call site; nothing else consumes it (CR-006).
            new_config["auto_retry_attempt"] = next_attempt
            new_config["auto_retried_from"] = crashed_run_id
            now = datetime.now(tz=UTC)
            db.add(
                RunORM(
                    id=new_run_id,
                    run_mode="paper",
                    status="running",
                    config=new_config,
                    started_at=now,
                    recovered_from_run_id=crashed.id,
                )
            )
            await db.commit()

        task = asyncio.create_task(
            run_paper_engine(
                run_id_str=str(new_run_id),
                strategy_cls=strategy_cls,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                symbols=symbols,
                timeframe=timeframe,
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_stop_pct,
                bracket_config=bracket_config,
                enable_adaptive_learning=enable_adaptive_learning,
                auto_apply_learning=auto_apply_learning,
                auto_retry_attempt=next_attempt,
            ),
            name=f"paper-engine-retry-{new_run_id}",
        )
        _RUN_TASKS[str(new_run_id)] = task
        log.info(
            "runs.auto_retry_started",
            new_run_id=str(new_run_id),
            attempt=next_attempt,
        )
    except Exception:
        # Best-effort: a failed retry must never take the API down.
        log.exception("runs.auto_retry_failed")


async def run_paper_engine(
    *,
    run_id_str: str,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: str,
    trailing_stop_pct: float | None = None,
    bracket_config: dict[str, object] | None = None,
    enable_adaptive_learning: bool = False,
    auto_apply_learning: bool = False,
    auto_retry_attempt: int = 0,
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
    trailing_stop_pct:
        Optional trailing-stop percentage (e.g. 0.02 for 2%).  When set the
        engine composes a ``TrailingStopManager`` and emits SELL signals
        whenever price drops ``trailing_stop_pct`` below the position's peak.
    enable_adaptive_learning:
        When ``True`` spawn an :class:`AdaptiveLearningTask` alongside the
        engine so trade outcomes are analysed and strategy parameters may be
        tuned during the run (see Sprint 36).
    auto_apply_learning:
        When ``True`` the adaptive-learning task is allowed to call
        ``strategy.update_params(...)`` directly; when ``False`` the task
        reports suggestions via logs only (dry-run mode, safer default).
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
        for _bk, _bv in (bracket_config or {}).items():
            if _bv is not None:
                engine_config[_bk] = _bv

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
                flush_interval=_PAPER_FLUSH_INTERVAL_SECONDS,
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
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")
        raise  # Must re-raise CancelledError for asyncio bookkeeping

    except Exception:
        final_status = "error"
        log.exception("runs.paper_engine_error")
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")

    finally:
        # Belt-and-suspenders teardown — idempotent no-op if the except
        # branches already ran; still safe to call on the happy path when no
        # side tasks were ever spawned.
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )

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

        # Sprint 50 Cycle 4: if the engine set its own stop event due to a
        # HALT circuit-breaker response, write an audit row BEFORE the status
        # transition so the forensic trail is complete even if status write fails.
        if engine is not None and getattr(engine, "auto_stop_reason", None) == _HALT_AUTO_STOP_REASON_VALUE:
            try:
                factory_audit = get_session_factory()
                async with factory_audit() as db_audit:
                    await record_audit_event(
                        db_audit,
                        event_type="circuit_breaker_halt_auto_stop",
                        resource_type="run",
                        resource_id=run_id_str,
                        request=None,  # system-initiated; no operator request context
                        payload={"trigger": "graduated_halt", "run_id": run_id_str},
                    )
                    await db_audit.commit()
            except Exception:
                log.warning(
                    "runs.paper_engine_halt_audit_failed",
                    run_id=run_id_str,
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

        # Auto-retry crashed paper runs (Sprint 48 D2): only on a genuine
        # error exit — operator stops and graceful shutdowns never reach
        # final_status == "error" (CancelledError re-raises above).  The
        # retry task is fire-and-forget; it sleeps the backoff first so this
        # finally block returns immediately.
        if final_status == "error":
            asyncio.create_task(
                _auto_retry_paper_run(
                    crashed_run_id=run_id_str,
                    auto_retry_attempt=auto_retry_attempt,
                    strategy_cls=strategy_cls,
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    symbols=symbols,
                    timeframe=timeframe,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    bracket_config=bracket_config,
                    enable_adaptive_learning=enable_adaptive_learning,
                    auto_apply_learning=auto_apply_learning,
                ),
                name=f"auto-retry-{run_id_str[:8]}",
            )


async def run_live_engine(
    *,
    run_id_str: str,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: str,
    trailing_stop_pct: float | None = None,
    bracket_config: dict[str, object] | None = None,
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
        for _bk, _bv in (bracket_config or {}).items():
            if _bv is not None:
                live_engine_config[_bk] = _bv

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
                flush_interval=_PAPER_FLUSH_INTERVAL_SECONDS,
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
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.live_engine_stop_error")
        raise  # Must re-raise CancelledError for asyncio bookkeeping

    except Exception:
        final_status = "error"
        log.exception("runs.live_engine_error")
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.live_engine_stop_error")

    finally:
        # Belt-and-suspenders teardown — idempotent no-op if the except
        # branches already ran; still safe on the happy path when no side
        # tasks were ever spawned.
        await _cancel_engine_side_tasks(
            auto_stop_task=auto_stop_task,
            learning_task=learning_task,
            learning_stop_event=learning_stop_event,
            flush_task=flush_task,
        )

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

        # Sprint 50 Cycle 4: audit HALT auto-stop before status transition
        # (mirrors paper engine pattern; availability > perfect auditability).
        if engine is not None and getattr(engine, "auto_stop_reason", None) == _HALT_AUTO_STOP_REASON_VALUE:
            try:
                factory_audit = get_session_factory()
                async with factory_audit() as db_audit:
                    await record_audit_event(
                        db_audit,
                        event_type="circuit_breaker_halt_auto_stop",
                        resource_type="run",
                        resource_id=run_id_str,
                        request=None,
                        payload={"trigger": "graduated_halt", "run_id": run_id_str},
                    )
                    await db_audit.commit()
            except Exception:
                log.warning(
                    "runs.live_engine_halt_audit_failed",
                    run_id=run_id_str,
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
# Backwards-compat aliases used by internal callers that still reference
# the underscore-prefixed names (e.g. incremental_flush_loop -> _flush_incremental).
# ---------------------------------------------------------------------------
_notify_trade_telegram = notify_trade_telegram
_flush_incremental = flush_incremental
_incremental_flush_loop = incremental_flush_loop
_auto_stop_after = auto_stop_after
_run_paper_engine = run_paper_engine
_run_live_engine = run_live_engine
