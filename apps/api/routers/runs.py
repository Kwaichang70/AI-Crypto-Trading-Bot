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
- Paper/Live run creation is stub-level: the run record is created with
  status="running", but the engine is not wired up until Sprint 2.
- Strategy parameter validation occurs at request time via ``parameter_schema()``.
- The ``config`` JSONB snapshot captures all run parameters at creation time
  so historical runs are fully self-contained even if strategy defaults change.
- Backtest metrics are written into ``config["backtest_metrics"]`` so they are
  available on ``GET /runs/{run_id}`` without a schema migration.
- LIVE mode requires passing the 3-layer safety gate:
  (1) ENABLE_LIVE_TRADING=true, (2) exchange API keys configured,
  (3) valid confirm_token matching LIVE_TRADING_CONFIRM_TOKEN.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import EquitySnapshotORM, RunORM, TradeORM
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

__all__ = ["router"]

router = APIRouter(prefix="/runs", tags=["runs"])

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Background task registry for paper/live trading engines
# ---------------------------------------------------------------------------
_RUN_TASKS: dict[str, asyncio.Task[None]] = {}

# ---------------------------------------------------------------------------
# Strategy registry — maps API names to strategy classes
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
            MACrossoverStrategy,
            RSIMeanReversionStrategy,
        )

        _STRATEGY_REGISTRY = {
            "ma_crossover": MACrossoverStrategy,
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "breakout": BreakoutStrategy,
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
        except Exception:  # noqa: BLE001 — best-effort; never fail a GET on bad stored data
            logger.warning(
                "runs.backtest_metrics_parse_error",
                run_id=str(run.id),
                exc_info=True,
            )

    return base


async def _run_paper_engine(
    *,
    run_id_str: str,
    strategy_cls: type,
    strategy_name: str,
    strategy_params: dict[str, Any],
    symbols: list[str],
    timeframe: TimeFrame,
    initial_capital: float,
) -> None:
    """
    Background coroutine that runs a paper trading engine for a single run.

    Creates all trading components, starts the StrategyEngine, and runs the
    live loop until stopped or errored. On exit, updates the run record in
    the database with the final status.

    This function uses its own database session (not the request session)
    because the POST handler's session is closed before this coroutine runs.

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

    try:
        settings = get_settings()
        capital = Decimal(str(initial_capital))

        # Extract exchange credentials
        api_key: str | None = None
        api_secret: str | None = None
        if settings.exchange_api_key is not None:
            api_key = settings.exchange_api_key.get_secret_value()
        if settings.exchange_api_secret is not None:
            api_secret = settings.exchange_api_secret.get_secret_value()

        # Instantiate components
        market_data = CCXTMarketDataService(
            exchange_id=settings.exchange_id,
            api_key=api_key,
            api_secret=api_secret,
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

        engine = StrategyEngine(
            strategies=[strategy_instance],
            execution_engine=execution,
            risk_manager=risk_manager,
            market_data=market_data,
            portfolio=portfolio,
            symbols=symbols,
            timeframe=timeframe,
            run_mode=RunMode.PAPER,
        )

        await engine.start(run_id_str)
        log.info("runs.paper_engine_running")
        await engine.run_live_loop()

    except asyncio.CancelledError:
        log.info("runs.paper_engine_cancelled")
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")
        raise  # Must re-raise CancelledError for asyncio bookkeeping

    except Exception:
        final_status = "error"
        log.exception("runs.paper_engine_error")
        if engine is not None:
            try:
                await engine.stop()
            except Exception:
                log.exception("runs.paper_engine_stop_error")

    finally:
        # Remove from task registry
        _RUN_TASKS.pop(run_id_str, None)

        # Update run status in DB using an isolated session
        try:
            factory = get_session_factory()
            async with factory() as db:
                try:
                    from sqlalchemy import select as sa_select
                    result = await db.execute(
                        sa_select(RunORM).where(RunORM.id == uuid.UUID(run_id_str))
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

    service = CCXTMarketDataService(
        exchange_id=settings.exchange_id,
        api_key=api_key,
        api_secret=api_secret,
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

    # --- Merge metrics into run.config JSONB ---
    metrics_response = _build_backtest_metrics(result)
    updated_config = dict(run_orm.config or {})
    updated_config["backtest_metrics"] = metrics_response.model_dump(mode="json")
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
# POST /api/v1/runs — start a new trading run
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
    #   Layer 1 — Environment: ENABLE_LIVE_TRADING must be True.
    #   Layer 2 — API Keys: EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be non-empty.
    #   Layer 3 — Confirmation Token: A runtime token provided in the request body
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
    # BACKTEST MODE — execute synchronously, persist results, finish run
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
            # Data fetch errors (400, 502) — mark run as error and re-raise
            error_time = datetime.now(tz=UTC)
            run_orm.status = "error"
            run_orm.stopped_at = error_time
            run_orm.updated_at = error_time
            await db.flush()
            raise

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
    # PAPER MODE — launch StrategyEngine as a background asyncio.Task
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
            ),
            name=f"paper-engine-{run_id}",
        )
        _RUN_TASKS[str(run_id)] = task
        log.info("runs.paper_engine_task_created", run_id=str(run_id))

    # ------------------------------------------------------------------
    # LIVE MODE — stub: record created with status="running"
    # The 3-layer LiveTradingGate is now fully enforced at the API level
    # (SEC-003 remediation). Sprint 2 must wire the StrategyEngine launch.
    # ------------------------------------------------------------------
    elif mode_value == "live":
        log.info(
            "runs.live_mode_stub",
            run_id=str(run_id),
            note="Live mode engine not yet wired — Sprint 2",
        )

    return _run_orm_to_detail_response(run_orm)


# ---------------------------------------------------------------------------
# GET /api/v1/runs — list all runs
# ---------------------------------------------------------------------------

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
) -> RunListResponse:
    """
    List all trading runs with pagination.

    Parameters
    ----------
    db:
        Injected async database session.
    offset:
        Number of records to skip.
    limit:
        Maximum records to return.

    Returns
    -------
    RunListResponse
        Paginated list of run records.
    """
    log = logger.bind(endpoint="list_runs", offset=offset, limit=limit)
    log.info("runs.list_requested")

    # Count total matching rows
    count_stmt = select(func.count()).select_from(RunORM)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Fetch the page
    page_stmt = (
        select(RunORM)
        .order_by(RunORM.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
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
# GET /api/v1/runs/{run_id} — get a single run
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
# DELETE /api/v1/runs/{run_id} — stop a running run
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
    if task is not None and not task.done():
        task.cancel()
        log.info("runs.paper_engine_task_cancelled", run_id=str(run_id))

    log.info("runs.stopped", run_id=str(run_id))
    return _run_orm_to_detail_response(run)


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
    for field in required_fields:
        if field not in params:
            errors.append(f"Required parameter missing: '{field}'")

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
