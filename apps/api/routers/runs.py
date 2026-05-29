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
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
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
from api.services.run_orchestrator import (
    _IncrementalFlushState,
    _LEARNING_INSTANCES,
    _RUN_ENGINES,
    _RUN_TASKS,
    _normalize_exchange_secret,
    auto_stop_after as _auto_stop_after,
    flush_incremental as _flush_incremental,
    incremental_flush_loop as _incremental_flush_loop,
    notify_trade_telegram as _notify_trade_telegram,
    run_live_engine as _run_live_engine,
    run_paper_engine as _run_paper_engine,
)
from api.services.run_persistence import (
    build_backtest_metrics as _build_backtest_metrics,
    persist_backtest_results as _persist_backtest_results,
    persist_paper_results as _persist_paper_results,
    run_orm_to_detail_response as _run_orm_to_detail_response,
    run_orm_to_response as _run_orm_to_response,
)
from common.types import RunMode, TimeFrame
from trading.strategy_availability import get_availability, is_mode_allowed

__all__ = ["router", "recover_orphaned_runs"]

router = APIRouter(prefix="/runs", tags=["runs"])

logger = structlog.get_logger(__name__)


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
# ``_run_orm_to_response`` and ``_run_orm_to_detail_response`` live in
# ``api.services.run_persistence`` since Sprint 40 Stap 2a; the top-level
# import block re-exports them under their original underscore-prefixed
# names for backwards compatibility.


# ---------------------------------------------------------------------------
# Helper: normalize exchange secret for CCXT compatibility
# ---------------------------------------------------------------------------
# ``_normalize_exchange_secret`` lives in ``api.services.run_orchestrator``
# since Sprint 40 Stap 2b; the top-level import block re-exports it under
# the original underscore-prefixed name for backwards compatibility.


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
# Helpers: build BacktestMetricsResponse + persist backtest results
# ---------------------------------------------------------------------------
# ``_build_backtest_metrics`` and ``_persist_backtest_results`` live in
# ``api.services.run_persistence`` since Sprint 40 Stap 2a; the top-level
# import block re-exports them under their original underscore-prefixed
# names for backwards compatibility.


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
    request: Request,
    x_live_confirm_token: Annotated[str | None, Header()] = None,
) -> RunDetailResponse:
    """
    Start a new trading run.

    Parameters
    ----------
    body:
        Run configuration from the request body.
    db:
        Injected async database session.
    x_live_confirm_token:
        Live-mode confirmation token supplied via the ``X-Live-Confirm-Token``
        header (SEC-004, Sprint 41).  Prefer this over ``body.confirm_token``
        — header transport keeps the secret out of request-body logs that some
        APM/proxy stacks capture.  Body-field still accepted as deprecated
        fallback until all clients migrate.

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

    # ------------------------------------------------------------------
    # Strategy-availability lockdown (Sprint 51 Cycle 2, IMPL-S51C2-101).
    # A demoted strategy may only run in backtest mode.  Single source of
    # truth: trading.strategy_availability.is_mode_allowed.  Fail-closed for
    # unlisted strategies (backtest-only).  strategy_name is already
    # normalized (lower + "-"->"_") above, matching the availability keyspace.
    # ------------------------------------------------------------------
    if not is_mode_allowed(strategy_name, body.mode):
        availability = get_availability(strategy_name)
        sorted_modes = sorted(m.value for m in availability.allowed_modes)
        status_value = availability.status.value
        demotion_reason = availability.demotion_reason
        log.warning(
            "runs.strategy_mode_not_allowed",
            strategy_name=strategy_name,
            mode=str(body.mode),
            status=status_value,
            allowed_modes=sorted_modes,
        )
        # detail is a plain STRING (not a dict) to avoid the UI
        # "[object Object]" envelope risk.  All context is embedded inline.
        # str(body.mode) is runtime-safe: body.mode is a plain str at runtime
        # (use_enum_values=True) but typed RunMode for mypy; .value would crash.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Strategy {strategy_name!r} is not available in "
                f"{str(body.mode)!r} mode (status={status_value}). "
                f"Allowed: {sorted_modes}. {demotion_reason}".strip()
            ),
        )

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
        # SEC-004: prefer the X-Live-Confirm-Token header — body-field is a
        # deprecated fallback so existing clients keep working until they
        # migrate.  Log a warning when the fallback is hit so migration
        # progress is observable.
        resolved_confirm_token = x_live_confirm_token
        if resolved_confirm_token is None and body.confirm_token:
            log.warning(
                "runs.live_confirm_token_body_fallback",
                reason="SEC-004 deprecated path; migrate clients to X-Live-Confirm-Token header",
            )
            resolved_confirm_token = body.confirm_token
        gate_result = gate.check_gate(
            settings=settings,
            confirm_token=resolved_confirm_token or "",
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

    # AR-006 (Sprint 45): concurrency cap.  Each active paper/live engine
    # holds DB-pool slots (incremental flush every 30 s) plus CCXT rate-
    # limit budget; an unbounded number would exhaust the pool and starve
    # health-check probes.  Backtest mode runs synchronously inside this
    # handler so it does not affect the running cap — only paper/live count.
    if body.mode in ("paper", "live"):
        from api.config import get_settings as _get_settings

        _settings = _get_settings()
        _active_count = sum(
            1 for t in _RUN_TASKS.values() if not t.done()
        )
        if _active_count >= _settings.max_concurrent_runs:
            log.warning(
                "runs.concurrency_cap_hit",
                active=_active_count,
                cap=_settings.max_concurrent_runs,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Concurrent run cap reached "
                    f"({_active_count}/{_settings.max_concurrent_runs}).  "
                    f"Stop an existing run before starting a new one, "
                    f"or raise MAX_CONCURRENT_RUNS in settings."
                ),
            )

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

    # SEC-002: persistent audit record for a passing live-trading gate.
    # Called after run_id is known so the audit row references the actual
    # run.  record_audit_event swallows DB errors so failure to audit never
    # blocks the functional request.
    if body.mode == "live":
        from api.services.audit_log import record_audit_event

        await record_audit_event(
            db,
            event_type="live_trading_enabled",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={
                "symbols": body.symbols,
                "timeframe": str(body.timeframe),
            },
        )

    db.add(run_orm)
    await db.flush()  # Assign the PK within the transaction without committing

    log.info(
        "runs.created",
        run_id=str(run_id),
        mode=mode_value,
        strategy=strategy_name,
    )

    # CR-007 (M7): warn if caller supplied a seed for paper/live — it is silently
    # ignored because BacktestRunner is only constructed for backtest mode.
    if body.seed is not None and not is_backtest:
        log.warning(
            "runs.seed_ignored_for_non_backtest_mode",
            seed=body.seed,
            mode=mode_value,
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
                seed=body.seed,
            )
            # M7 (Sprint 49 INF-9): persist the resolved seed (auto-generated or
            # caller-supplied) into the config dict.  Seed mutation here is captured
            # by _persist_backtest_results() below, which rebuilds config from
            # run_orm.config (including this mutation) and writes the complete
            # merged block back to the DB.
            config_snapshot["seed"] = runner.seed

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

# M5 (Sprint 49): allowed sort column names for GET /api/v1/runs.
# Only top-level RunORM columns are permitted — JSONB-resident fields (psr,
# sharpe_ratio) are excluded because they have no index and produce fragile
# mypy types under strict mode.  Frontend sorts JSONB fields client-side
# within the 50-row page.  See M5 producer report §3 for full rationale.
_VALID_SORT_BY: frozenset[str] = frozenset({"created_at", "n_closed_trades"})
_VALID_SORT_ORDERS: frozenset[str] = frozenset({"asc", "desc"})


@router.get(
    "",
    response_model=RunListResponse,
    summary="List all trading runs",
    description=(
        "Returns a paginated list of all runs with optional server-side filtering "
        "and sorting.  Default order is creation time descending.  Sortable by "
        "created_at or n_closed_trades (top-level columns only)."
    ),
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
    min_closed_trades: Annotated[
        int | None,
        Query(
            ge=0,
            description=(
                "Filter: only return runs with n_closed_trades >= this value. "
                "NULL runs (paper/live/pre-M3 backtests) are excluded when this "
                "param is supplied."
            ),
        ),
    ] = None,
    sort_by: Annotated[
        str | None,
        Query(
            description=(
                "Column to sort by. Allowed values: created_at, n_closed_trades. "
                "Defaults to created_at. Values not in the allowed set return HTTP 422."
            ),
        ),
    ] = None,
    sort_order: Annotated[
        str,
        Query(
            description="Sort direction: 'desc' (default) or 'asc'.",
        ),
    ] = "desc",
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
    min_closed_trades:
        Optional minimum n_closed_trades threshold. Runs with NULL
        n_closed_trades (paper/live/pre-M3) are excluded when supplied.
    sort_by:
        Column to sort by. Allowed: created_at, n_closed_trades.
        Invalid values raise HTTP 422.
    sort_order:
        Sort direction. Allowed: asc, desc. Invalid values raise HTTP 422.

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
        min_closed_trades=min_closed_trades,
        sort_by=sort_by,
        sort_order=sort_order,
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
    if sort_by is not None and sort_by not in _VALID_SORT_BY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid sort_by '{sort_by}'. "
                f"Must be one of: {sorted(_VALID_SORT_BY)}"
            ),
        )
    if sort_order not in _VALID_SORT_ORDERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sort_order '{sort_order}'. Must be 'asc' or 'desc'.",
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

    # min_closed_trades filter — NULL rows are excluded implicitly by >= comparison.
    # Runs where n_closed_trades IS NULL (paper/live/pre-M3) will not satisfy
    # n_closed_trades >= N and are therefore excluded when the param is set.
    if min_closed_trades is not None:
        filters.append(RunORM.n_closed_trades >= min_closed_trades)

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

    # Dynamic ORDER BY — default is created_at DESC (existing behaviour).
    # n_closed_trades DESC puts NULL last in PostgreSQL (NULLs sort after all
    # values in DESC order); we use NULLS LAST to be explicit and portable.
    # Type-safe: order_by is applied inside each branch to avoid mypy strict
    # complaints about reassigning a variable with incompatible column types.
    effective_sort = sort_by or "created_at"
    if effective_sort == "n_closed_trades":
        if sort_order == "asc":
            page_stmt = page_stmt.order_by(RunORM.n_closed_trades.asc().nulls_last())
        else:
            page_stmt = page_stmt.order_by(RunORM.n_closed_trades.desc().nulls_last())
    else:  # created_at (default)
        if sort_order == "asc":
            page_stmt = page_stmt.order_by(RunORM.created_at.asc())
        else:
            page_stmt = page_stmt.order_by(RunORM.created_at.desc())
    page_stmt = page_stmt.offset(offset).limit(limit)
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
# POST /api/v1/runs/{run_id}/emergency-stop  -- SEC-006 (Sprint 45)
# ---------------------------------------------------------------------------

@router.post(
    "/{run_id}/emergency-stop",
    status_code=status.HTTP_200_OK,
    response_model=RunDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        409: {"model": ErrorResponse, "description": "Run already in terminal state"},
    },
    summary="Emergency-stop a running trading run",
    description=(
        "Hard-stop a running paper/live engine, bypassing the regular "
        "rate-limit ceiling.  Every call is recorded in the audit_events "
        "table with event_type='emergency_stop' so post-incident review "
        "can isolate operator interventions.  Functionally equivalent to "
        "DELETE /runs/{id} but always available — use this when the "
        "API key bucket is throttled by the same incident you are trying "
        "to halt (e.g. a stuck client retrying DELETE)."
    ),
)
async def emergency_stop_run(
    run_id: uuid.UUID,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    reason: Annotated[str | None, Header(alias="X-Emergency-Reason")] = None,
) -> RunDetailResponse:
    """Hard-stop a run with persistent audit trail.

    Body identical to DELETE /runs/{id}: transitions status to 'stopped',
    cancels the engine task, removes from _RUN_ENGINES / _LEARNING_INSTANCES.

    Additional SEC-006 behaviour:
      * One ``audit_events`` row with ``event_type='emergency_stop'`` is
        written BEFORE the cancel sequence so the trail survives even if
        a subsequent step fails.
      * Optional ``X-Emergency-Reason`` header is captured in the audit
        payload so operators can leave a one-line incident note.
    """
    log = logger.bind(endpoint="emergency_stop_run", run_id=str(run_id))
    log.warning("runs.emergency_stop_requested", reason=reason)

    stmt = select(RunORM).where(RunORM.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.status != "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot emergency-stop run {run_id}: status is '{run.status}'. "
                f"Only 'running' runs can be emergency-stopped."
            ),
        )

    # SEC-002 + SEC-006: persist audit BEFORE state mutation so the trail
    # survives even if engine teardown fails mid-cancel.
    from api.services.audit_log import record_audit_event

    await record_audit_event(
        db,
        event_type="emergency_stop",
        resource_type="run",
        resource_id=str(run_id),
        request=request,
        payload={
            "run_mode": run.run_mode,
            "strategy": (run.config or {}).get("strategy_name"),
            "reason": reason,
        },
    )

    now = datetime.now(tz=UTC)
    run.status = "stopped"
    run.stopped_at = now
    run.updated_at = now

    await db.flush()

    # Cancel the background task — same teardown as DELETE /runs/{id}
    task = _RUN_TASKS.pop(str(run_id), None)
    _RUN_ENGINES.pop(str(run_id), None)
    _LEARNING_INSTANCES.pop(str(run_id), None)
    if task is not None and not task.done():
        task.cancel()
        log.warning("runs.emergency_engine_task_cancelled")

    log.warning("runs.emergency_stopped", reason=reason)
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
# GET /api/v1/runs/{run_id}/promotion-eligibility
# Sprint 50 Cycle 5 Sub-scope A
# ---------------------------------------------------------------------------

@router.get(
    "/{run_id}/promotion-eligibility",
    summary="Check whether a paper run is eligible for promotion to live",
    description=(
        "Returns a data-volume eligibility report for the given paper run.  "
        "Criteria: closed_trade_count >= MIN_PAPER_TRADES_FOR_PROMOTION and "
        "runtime_days >= MIN_PAPER_RUNTIME_DAYS.  Performance metrics "
        "(Sharpe, drawdown) are never gate criteria -- the operator decides "
        "performance acceptability."
    ),
)
async def get_promotion_eligibility(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Return promotion gate eligibility for a paper run."""
    from api.config import get_settings
    from api.services.promotion_gate import evaluate_paper_run_eligibility

    log = logger.bind(endpoint="get_promotion_eligibility", run_id=str(run_id))
    settings = get_settings()

    result_row = await db.execute(
        select(RunORM).where(RunORM.id == run_id)
    )
    run_orm: RunORM | None = result_row.scalar_one_or_none()

    if run_orm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found.",
        )

    eligibility = await evaluate_paper_run_eligibility(
        db=db,
        run_orm=run_orm,
        min_trades=settings.min_paper_trades_for_promotion,
        min_runtime_days=settings.min_paper_runtime_days,
    )

    log.info(
        "runs.promotion_eligibility_checked",
        eligible=eligibility.eligible,
        trade_count=eligibility.trade_count,
        runtime_days=round(eligibility.runtime_days, 1),
    )

    return {
        "run_id": str(run_id),
        "eligible": eligibility.eligible,
        "trade_count": eligibility.trade_count,
        "runtime_days": round(eligibility.runtime_days, 1),
        "min_trades_required": settings.min_paper_trades_for_promotion,
        "min_runtime_days_required": settings.min_paper_runtime_days,
        "reasons": eligibility.reasons,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/runs/{run_id}/promote-to-live
# Sprint 50 Cycle 5 Sub-scope A
# ---------------------------------------------------------------------------

@router.post(
    "/{run_id}/promote-to-live",
    status_code=status.HTTP_201_CREATED,
    summary="Promote a stopped paper run to a new live run",
    description=(
        "Creates a new live run with the same strategy configuration as the "
        "given paper run, setting promoted_from_run_id to the source paper run. "
        "Requires: (1) paper run is stopped, (2) data-volume gate passes "
        "(trade_count + runtime), (3) the full 3-layer live-trading safety gate "
        "(env flag + API keys + X-Live-Confirm-Token header). "
        "An audit row is written before any state mutation."
    ),
    responses={
        400: {"description": "Paper run not eligible (gate criteria not met)"},
        403: {"description": "Live trading safety gate failed"},
        404: {"description": "Source paper run not found"},
        422: {"description": "Strategy not available for live trading (demoted)"},
    },
)
async def promote_to_live(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: Request,
    x_live_confirm_token: Annotated[str | None, Header()] = None,
) -> RunDetailResponse:
    """Promote a stopped paper run to a new live run.

    Parameters
    ----------
    run_id:
        UUID of the source paper run to promote.
    db:
        Injected async database session.
    x_live_confirm_token:
        Live-mode confirmation token via X-Live-Confirm-Token header
        (SEC-004 mandatory -- no body fallback for promotion endpoint).

    Returns
    -------
    RunDetailResponse
        The newly created live run record.

    Raises
    ------
    HTTPException 404:
        Source paper run not found.
    HTTPException 400:
        Promotion gate criteria not met (data-volume insufficient).
    HTTPException 403:
        Live trading safety gate failed.
    """
    from api.config import get_settings
    from api.services.audit_log import record_audit_event
    from api.services.promotion_gate import evaluate_paper_run_eligibility
    from trading.safety import LiveTradingGate

    log = logger.bind(endpoint="promote_to_live", source_run_id=str(run_id))
    settings = get_settings()

    # Step 1: Fetch source paper run
    row = await db.execute(select(RunORM).where(RunORM.id == run_id))
    source_run: RunORM | None = row.scalar_one_or_none()

    if source_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper run {run_id} not found.",
        )

    # Strategy-availability lockdown (Sprint 51 Cycle 2, IMPL-S51C2-103).
    # OUTER guard, orthogonal to the per-run evidence gate: a demoted strategy
    # must never reach live via the promotion path even if its paper-run
    # evidence would otherwise satisfy evaluate_paper_run_eligibility.
    # strategy_name is read from the source run's immutable config snapshot and
    # normalized (SEC-001) to match create_run's availability keyspace.
    promotion_strategy_name = (
        str(source_run.config.get("strategy_name", "")) if source_run.config else ""
    ).lower().replace("-", "_")
    if not is_mode_allowed(promotion_strategy_name, RunMode.LIVE):
        availability = get_availability(promotion_strategy_name)
        log.warning(
            "runs.promotion_strategy_mode_not_allowed",
            source_run_id=str(run_id),
            strategy_name=promotion_strategy_name,
            status=availability.status.value,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Strategy {promotion_strategy_name!r} is not available for "
                f"live trading (status={availability.status.value}) and cannot "
                f"be promoted to live. {availability.demotion_reason}".strip()
            ),
        )

    # Step 2: Promotion gate (data-volume only)
    eligibility = await evaluate_paper_run_eligibility(
        db=db,
        run_orm=source_run,
        min_trades=settings.min_paper_trades_for_promotion,
        min_runtime_days=settings.min_paper_runtime_days,
    )

    if not eligibility.eligible:
        log.warning(
            "runs.promotion_gate_failed",
            reasons=eligibility.reasons,
            trade_count=eligibility.trade_count,
            runtime_days=round(eligibility.runtime_days, 1),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Paper run {run_id} is not eligible for promotion. "
                f"trade_count={eligibility.trade_count} "
                f"(min={settings.min_paper_trades_for_promotion}), "
                f"runtime={eligibility.runtime_days:.1f}d "
                f"(min={settings.min_paper_runtime_days:.1f}d). "
                f"Reasons: {', '.join(eligibility.reasons)}"
            ),
        )

    # Step 3: 3-layer live trading gate (SEC-004 -- header only, no body fallback)
    gate = LiveTradingGate()
    gate_result = gate.check_gate(
        settings=settings,
        confirm_token=x_live_confirm_token or "",
    )

    if not gate_result.passed:
        failed_layers = [layer.name for layer in gate_result.layers if not layer.passed]
        log.warning("runs.promotion_live_gate_failed", failures=gate_result.failures)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Live trading gate check failed. "
                f"Failed layers: {', '.join(failed_layers)}."
            ),
        )

    # Concurrency cap check (AR-006)
    active_count = sum(1 for t in _RUN_TASKS.values() if not t.done())
    if active_count >= settings.max_concurrent_runs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Concurrent run cap reached ({active_count}/{settings.max_concurrent_runs}). "
                "Stop an existing run before promoting."
            ),
        )

    # Step 4: Write audit row BEFORE creating the live run (SEC-002 + SAVEPOINT pattern).
    # async with db.begin_nested() creates a SAVEPOINT so the audit insert is independently
    # durable -- even if the outer transaction rolls back, the audit trail is preserved.
    new_run_id = uuid.uuid4()
    async with db.begin_nested():
        await record_audit_event(
            db,
            event_type="paper_promoted_to_live",
            resource_type="run",
            resource_id=str(new_run_id),
            request=request,
            payload={
                "source_paper_run_id": str(run_id),
                "strategy_name": source_run.config.get("strategy_name"),
                "symbols": source_run.config.get("symbols"),
                "timeframe": source_run.config.get("timeframe"),
            },
        )
    # SAVEPOINT released: audit row is now independently committed.
    # The outer transaction continues for RunORM creation below.

    # Step 5: Reconstruct config from source paper run, override mode.
    now = datetime.now(tz=UTC)
    promoted_config: dict[str, Any] = {
        **source_run.config,
        "mode": "live",
        "promoted_from_run_id": str(run_id),
    }
    # Remove backtest-specific keys that have no meaning for live mode.
    for _key in ("backtest_start", "backtest_end", "seed", "backtest_metrics"):
        promoted_config.pop(_key, None)

    strategy_name = source_run.config.get("strategy_name", "")
    strategy_cls = _get_strategy_registry().get(strategy_name)
    if strategy_cls is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Strategy '{strategy_name}' from source run is no longer registered.",
        )

    timeframe_val = TimeFrame(str(source_run.config.get("timeframe", "1h")))

    live_run_orm = RunORM(
        id=new_run_id,
        run_mode="live",
        status="running",
        config=promoted_config,
        started_at=now,
        created_at=now,
        updated_at=now,
        promoted_from_run_id=run_id,
    )
    db.add(live_run_orm)
    await db.flush()

    # Step 6: Commit FIRST -- the live engine must find the RunORM row on its
    # first DB read, so the row must be durable before the task starts.
    await db.commit()

    # Step 7: Launch live engine background task AFTER commit (CR5-003).
    task = asyncio.create_task(
        _run_live_engine(
            run_id_str=str(new_run_id),
            strategy_cls=strategy_cls,
            strategy_name=strategy_name,
            strategy_params=source_run.config.get("strategy_params", {}),
            symbols=source_run.config.get("symbols", []),
            timeframe=timeframe_val,
            initial_capital=source_run.config.get("initial_capital", "10000"),
            trailing_stop_pct=source_run.config.get("strategy_params", {}).get(
                "trailing_stop_pct"
            ),
            enable_adaptive_learning=False,
        ),
        name=f"live-engine-promoted-{new_run_id}",
    )
    _RUN_TASKS[str(new_run_id)] = task

    log.info(
        "runs.promoted_to_live",
        source_run_id=str(run_id),
        new_run_id=str(new_run_id),
        strategy=strategy_name,
    )

    return _run_orm_to_detail_response(live_run_orm)


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

                # Strategy-availability lockdown (Sprint 51 Cycle 2, IMPL-S51C2-102).
                # FAIL-CLOSED: a demoted strategy must NOT be auto-restarted in
                # paper/live.  Mark the orphan as error and skip (do not recreate).
                # Single source of truth shared with create_run.  Normalize the
                # strategy name (SEC-001) so the availability key matches
                # create_run's keyspace regardless of how it was stored.
                normalized_strategy_name = strategy_name.lower().replace("-", "_")
                # Validate orphan_mode before constructing RunMode (C1/SEC-007):
                # a corrupt run_mode column would otherwise raise an uncaught
                # ValueError; mirror the timeframe-validation skip pattern.
                try:
                    orphan_run_mode = RunMode(orphan_mode)
                except ValueError:
                    log.warning(
                        "recovery.orphan_skipped",
                        reason="invalid_run_mode",
                        run_id=orphan_id,
                        mode=orphan_mode,
                    )
                    await _mark_orphan_error(factory, orphan.id, log)
                    continue
                if not is_mode_allowed(normalized_strategy_name, orphan_run_mode):
                    log.warning(
                        "recovery.orphan_skipped",
                        run_id=orphan_id,
                        reason="strategy_mode_not_allowed",
                        strategy_name=normalized_strategy_name,
                        mode=orphan_mode,
                    )
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
