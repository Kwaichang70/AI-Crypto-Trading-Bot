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
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import String, cast, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from api.db.models import (
    AuditEventORM,
    EquitySnapshotORM,
    OrderORM,
    RunORM,
    TradeORM,
)
from api.db.session import get_db
from api.deps import require_admin
from api.schemas import (
    EntriesLatchClearResponse,
    ErrorResponse,
    ExitConfigWaivedResponse,
    ExitConfigWarningResponse,
    FlattenResultResponse,
    RunCreateRequest,
    RunDetailResponse,
    RunEmergencyStopResponse,
    RunListResponse,
    RunStopResponse,
    UnprotectedPositionResponse,
)
from api.services.run_orchestrator import (
    _IncrementalFlushState,
    _LEARNING_INSTANCES,
    _RUN_ENGINES,
    _RUN_TASKS,
    _normalize_exchange_secret,
    auto_stop_after as _auto_stop_after,
    build_live_ccxt_exchange as _build_live_ccxt_exchange,
    flush_incremental as _flush_incremental,
    incremental_flush_loop as _incremental_flush_loop,
    notify_trade_telegram as _notify_trade_telegram,
    run_live_engine as _run_live_engine,
    run_paper_engine as _run_paper_engine,
)
from api.services.run_persistence import (
    load_resume_snapshot as _load_resume_snapshot,
    persist_backtest_results as _persist_backtest_results,
    persist_paper_results as _persist_paper_results,
    run_orm_to_detail_response as _run_orm_to_detail_response,
    run_orm_to_response as _run_orm_to_response,
)
from common.types import OrderSide, RunMode, TimeFrame
from trading.exit_config import (
    ExitConfigError,
    strategy_requires_exit_manager,
    validate_run_exit_config,
)
from trading.recovery import ResumeRejected, check_fill_integrity, replay_sort_key
from trading.strategy_availability import get_availability, is_mode_allowed
from trading.strategy_engine import FlattenResult, build_synthetic_flatten_result

__all__ = ["router", "recover_orphaned_runs"]


def _flatten_result_to_response(result: FlattenResult) -> FlattenResultResponse:
    """WP1.7a: convert the engine-layer dataclass into the API schema."""
    return FlattenResultResponse.model_validate(result, from_attributes=True)

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
            MomentumBreakoutStrategy,
            RSIMeanReversionStrategy,
            SLTPReversionStrategy,
        )

        _STRATEGY_REGISTRY = {
            "ma_crossover": MACrossoverStrategy,
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "model_strategy": ModelStrategy,
            "dca_rsi_hybrid": DCARSIHybridStrategy,
            "grid_trading": GridTradingStrategy,
            "sl_tp_reversion": SLTPReversionStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
        }
    return _STRATEGY_REGISTRY


# Bracket-exit config keys consumed by the StrategyEngine.  All keys carry a
# ``bracket_`` prefix so they never collide with a strategy's own parameter
# names (e.g. dca_rsi_hybrid has its own ``take_profit_pct``; breakout has its
# own ``atr_period``).  Numeric values are coerced to float; the period to int;
# the mode stays a str.
_BRACKET_FLOAT_KEYS = (
    "bracket_stop_loss_pct",
    "bracket_take_profit_pct",
    "bracket_atr_sl_multiplier",
    "bracket_atr_tp_multiplier",
)


def _exit_config_error_detail(exc: ExitConfigError) -> dict[str, Any]:
    """WP1.3a (SY-13a-18): build the 422 ``detail`` envelope for any
    ``ExitConfigError``.  Shared by create/promote/resume so the wire
    format is identical everywhere."""
    detail: dict[str, Any] = {
        "code": exc.code,
        "errors": [
            {
                "field": issue.field,
                "reason": issue.reason,
                "value": issue.value,
                "min": issue.min,
                "max": issue.max,
                "message": issue.message,
            }
            for issue in exc.issues
        ],
        "warnings": [
            {"code": w.code, "field": w.field, "message": w.message}
            for w in exc.warnings
        ],
    }
    if exc.strategy is not None:
        detail["strategy"] = exc.strategy
    if exc.requires_one_of is not None:
        detail["requires_one_of"] = list(exc.requires_one_of)
    if exc.hint is not None:
        detail["hint"] = exc.hint
    return detail


def _extract_bracket_config(strategy_params: dict[str, Any]) -> dict[str, object]:
    """
    Pop ``bracket_*`` exit keys out of ``strategy_params`` in place.

    Returns a dict of the resolved engine-level bracket config (keys keep
    their ``bracket_`` prefix).  Blank values (``None`` / ``""``) are dropped
    so they never reach either the strategy schema validator or the
    StrategyEngine.  Mutates the input dict so the remaining params validate
    cleanly against the strategy schema (bracket settings are engine-level,
    not strategy params).  The ``bracket_`` prefix guarantees no collision
    with any strategy's native parameter names.

    WP1.3a (SY-13a-01/02): a value that is *not* cleanly coercible (a bool,
    a non-numeric string, a non-integral ``bracket_atr_period``) is no
    longer silently coerced (``float(True) == 1.0``) or allowed to raise an
    uncaught ``ValueError`` (a 500, R-05) here.  It is preserved verbatim so
    ``trading.exit_config.validate_run_exit_config`` -- called immediately
    after this, before the strategy-schema check -- can classify it
    (``invalid_type`` / ``not_a_number`` / ``not_finite``) and return a
    clean 422.  Every value this function DOES coerce (a valid numeric
    string, an integral float) keeps its exact historical return contract
    (``test_bracket_exit.py`` ``TestRunsBracketExtraction``).
    """
    bracket: dict[str, object] = {}
    for key in _BRACKET_FLOAT_KEYS:
        if key in strategy_params:
            raw = strategy_params.pop(key)
            if raw is not None and raw != "":
                if isinstance(raw, bool):
                    bracket[key] = raw
                else:
                    try:
                        bracket[key] = float(raw)
                    except (TypeError, ValueError, OverflowError):
                        # WP13a-S-R5-03 (security round 5, AC3): a Python
                        # ``int`` far outside float range (e.g. a
                        # ~4000-digit JSON integer) makes ``float(raw)``
                        # RAISE ``OverflowError`` rather than clamping to
                        # inf -- preserve the raw value verbatim, exactly
                        # like the TypeError/ValueError cases, so
                        # ``validate_run_exit_config`` (called right after
                        # this) classifies it as a clean 422, not a 500.
                        bracket[key] = raw
    if "bracket_atr_period" in strategy_params:
        raw_period = strategy_params.pop("bracket_atr_period")
        if raw_period is not None and raw_period != "":
            if isinstance(raw_period, bool):
                bracket["bracket_atr_period"] = raw_period
            else:
                try:
                    _period_float = float(raw_period)
                except (TypeError, ValueError, OverflowError):
                    # WP13a-S-R5-03 (security round 5, AC3): same
                    # OverflowError class as the float keys above.
                    bracket["bracket_atr_period"] = raw_period
                else:
                    bracket["bracket_atr_period"] = (
                        int(_period_float)
                        if _period_float.is_integer()
                        else raw_period
                    )
    if "bracket_mode" in strategy_params:
        raw_mode = strategy_params.pop("bracket_mode")
        if raw_mode is not None and raw_mode != "":
            bracket["bracket_mode"] = raw_mode if isinstance(raw_mode, bool) else str(raw_mode)
    return bracket


def _build_promoted_strategy_params(
    source_strategy_params: Mapping[str, Any] | None,
    trailing_stop_pct: object,
) -> dict[str, Any]:
    """WP13a-S-07 (security round 2, extracted for WP13a-C-R2-09 unit
    coverage): build the ``strategy_params`` dict for a promoted live run,
    mirroring ``create_run`` -- the NORMALISED ``trailing_stop_pct`` (from
    ``ExitConfigVerdict.config.trailing_stop_pct``) is written back, or
    popped when unset, instead of leaving the raw source value in place
    while the spawned engine itself runs with the normalised one.

    Always returns a FRESH dict copy -- ``source_strategy_params`` (which
    belongs to the source paper run, still attached to the DB session) is
    never mutated in place, and the returned dict is never the same object
    as the input.
    """
    promoted_strategy_params: dict[str, Any] = dict(source_strategy_params or {})
    if trailing_stop_pct is None:
        promoted_strategy_params.pop("trailing_stop_pct", None)
    else:
        promoted_strategy_params["trailing_stop_pct"] = trailing_stop_pct
    return promoted_strategy_params


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
    # left blank.  Peeked here (not yet deleted/coerced) -- the WP1.3a
    # validator below decides unset vs. invalid vs. valid; only THEN do we
    # mutate ``strategy_params`` (SY-13a-02).
    _raw_trailing_stop_pct: object = body.strategy_params.get("trailing_stop_pct")

    # Extract fixed/ATR bracket exit config (stop-loss + take-profit) from
    # strategy params BEFORE schema validation.  These are engine-level
    # settings consumed by the StrategyEngine, not strategy params, so they
    # are always stripped out (a blank UI field arrives as "" and is dropped).
    _bracket_config = _extract_bracket_config(body.strategy_params)

    # ------------------------------------------------------------------
    # WP1.3a (SY-13a-01/15): validate the exit config + allow_pyramiding
    # BEFORE the strategy-schema check, any audit row, db.add, or the
    # backtest bar fetch.  A failure here leaves no RunORM row and no audit
    # row (I6).
    # ------------------------------------------------------------------
    try:
        _verdict = validate_run_exit_config(
            strategy_cls,
            bracket=_bracket_config,
            trailing_stop_pct=_raw_trailing_stop_pct,
            mode=body.mode,
            allow_pyramiding=body.allow_pyramiding,
            timeframe=str(body.timeframe),
        )
    except ExitConfigError as exc:
        log.warning(
            "runs.exit_config_invalid",
            code=exc.code,
            reasons=[i.reason for i in exc.issues],
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_exit_config_error_detail(exc),
        ) from exc

    _bracket_config = dict(_verdict.config.bracket)
    _trailing_stop_pct: float | None = _verdict.config.trailing_stop_pct
    _allow_pyramiding: bool = _verdict.allow_pyramiding
    for _w in _verdict.warnings:
        log.info(
            "runs.exit_config_warning",
            code=_w.code,
            field=_w.field,
        )
    # An unset trailing_stop_pct is removed from strategy_params exactly
    # like a blank one always was; a valid, normalised value is written
    # back so a strategy that declares its own ``trailing_stop_pct`` field
    # (for the UI/schema range check) still sees the same clean value.
    if _trailing_stop_pct is None:
        body.strategy_params.pop("trailing_stop_pct", None)
    else:
        body.strategy_params["trailing_stop_pct"] = _trailing_stop_pct

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
        # WP1.7a (SY-08 extension): while the global kill switch is
        # latched, a new run of either mode would sit created but silently
        # inert (entries blocked from the very first bar via apply_latch,
        # I5) -- reject it outright instead. Backtests are unaffected (no
        # engine, no entries to block).
        from api.services import kill_switch as _kill_switch

        if _kill_switch.is_active():
            log.warning("runs.create_blocked_kill_switch_active", mode=body.mode)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "kill_switch_active"},
            )

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
    # Persist engine-level bracket-exit config separately from strategy_params
    # (it was popped out before validation) so paper/live recovery and
    # promotion can re-apply it.
    if _bracket_config:
        config_snapshot["bracket_config"] = _bracket_config
    # WP1.3a (SY-13a-08/I8): the RESOLVED bool is always persisted so a
    # run's pyramiding behaviour is visible and stays fixed for its lifetime.
    config_snapshot["allow_pyramiding"] = _allow_pyramiding
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
                bracket_config=_bracket_config,
                allow_pyramiding=_allow_pyramiding,
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
                bracket_config=_bracket_config,
                enable_adaptive_learning=body.enable_adaptive_learning,
                auto_apply_learning=body.auto_apply_learning,
                allow_pyramiding=_allow_pyramiding,
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
                bracket_config=_bracket_config,
                enable_adaptive_learning=body.enable_adaptive_learning,
                allow_pyramiding=_allow_pyramiding,
            ),
            name=f"live-engine-{run_id}",
        )
        _RUN_TASKS[str(run_id)] = task
        log.info("runs.live_engine_task_created", run_id=str(run_id))

    response = _run_orm_to_detail_response(run_orm)
    if _verdict.warnings:
        response = response.model_copy(
            update={
                "config_warnings": [
                    ExitConfigWarningResponse(code=w.code, field=w.field, message=w.message)
                    for w in _verdict.warnings
                ]
            }
        )
    return response


# ---------------------------------------------------------------------------
# GET /api/v1/runs  -- list all runs
# ---------------------------------------------------------------------------

_VALID_MODES: frozenset[str] = frozenset({"backtest", "paper", "live"})
_VALID_STATUSES: frozenset[str] = frozenset(
    {"running", "stopped", "error", "archived", "orphaned", "resuming"}
)

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
    response_model=RunStopResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        409: {
            "model": ErrorResponse,
            "description": (
                "Run is not in a stoppable state, its flatten did not "
                "complete, or flatten was requested against a non-running "
                "engine"
            ),
        },
        422: {
            "model": ErrorResponse,
            "description": "flatten is required for a running live run",
        },
    },
    summary="Stop a running trading run",
    description=(
        "Transitions a run from 'running' to 'stopped'. "
        "Returns 409 if the run is already stopped or errored. "
        "WP1.7a: '?flatten=true|false' is REQUIRED for a running LIVE run "
        "(422 'flatten_decision_required' with the held symbols if "
        "omitted); optional for paper (default false). "
        "'?flatten=true' sells every held symbol through the same capped "
        "process_signal path any other exit uses (never more than "
        "min(own, balance)) before the task is cancelled. An incomplete "
        "flatten leaves the run 'running' with entries latched and "
        "returns 409 'flatten_incomplete' with the partial result -- the "
        "operator may retry, or stop again with '?flatten=false'."
    ),
)
async def stop_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: Request,
    flatten: Annotated[bool | None, Query()] = None,
) -> RunStopResponse:
    """
    Stop a running trading run.

    Parameters
    ----------
    run_id:
        UUID of the run to stop.
    db:
        Injected async database session.
    request:
        WP1.8b (S2-03): passed through to the orphaned/resuming-stop audit
        row below so actor/IP/user-agent are captured (previously
        ``request=None``).
    flatten:
        WP1.7a (SY-05). Required for a running LIVE run; optional
        (default False) for paper, orphaned or resuming.

    Returns
    -------
    RunStopResponse
        The updated run record with status='stopped' (or, for an
        incomplete flatten, still 'running'), plus the flatten result.

    Raises
    ------
    HTTPException 404:
        When no run with the given ID exists.
    HTTPException 409:
        When the run is already in a terminal state (stopped/error), when
        flatten was requested against a non-running engine, or when the
        flatten did not complete.
    HTTPException 422:
        When a running live run omits the required flatten decision.
    """
    log = logger.bind(endpoint="stop_run", run_id=str(run_id))
    log.info("runs.stop_requested", flatten=flatten)

    # WP1.8a-round2 (C-01/S-01 item 4): lock the row before the status guard
    # so a concurrent resume's uncommitted orphaned->resuming->running
    # transition is waited on (not silently missed) rather than racing us.
    # WP1.8b (S2-01): a resume no longer holds this lock for the duration
    # of its exchange scan (only for its own two short CAS transactions),
    # so this SELECT ... FOR UPDATE now blocks for, at most, a few
    # milliseconds even while a resume's scan is in flight.
    stmt = select(RunORM).where(RunORM.id == run_id).with_for_update()
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        log.warning("runs.not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # WP1.8a: an orphaned run has no background task (O1) but is still a
    # live, unresolved run the operator must be able to close out without
    # first resuming it. WP1.8b (S2-01): 'resuming' is accepted too -- a
    # stop issued while a resume's scan is in flight (no lock held) must
    # win outright, not wait for the resume to finish.
    if run.status not in ("running", "orphaned", "resuming"):
        log.warning("runs.not_stoppable", current_status=run.status)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot stop run {run_id}: "
                f"current status is '{run.status}'. Only 'running', "
                f"'orphaned' or 'resuming' runs can be stopped."
            ),
        )

    # WP1.7a (SY-05): flatten only ever means something against a live,
    # running engine -- an orphaned/resuming run has no engine to flatten
    # through (flattening an orphaned run is explicitly out of scope for
    # 1.7a, per the synthesis spec).
    if flatten and run.status in ("orphaned", "resuming"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "flatten_requires_running_engine"},
        )

    engine = _RUN_ENGINES.get(str(run_id)) if run.status == "running" else None

    if run.status == "running" and run.run_mode == "live" and flatten is None:
        held_symbols: list[str] = []
        if engine is not None:
            for _sym in engine.symbols:
                _pos = engine.portfolio.get_position(_sym)
                if _pos is not None and not _pos.is_flat:
                    held_symbols.append(_sym)
        log.warning("runs.stop_flatten_decision_required", held_symbols=held_symbols)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "flatten_decision_required", "held_symbols": held_symbols},
        )

    do_flatten = bool(flatten) and run.status == "running"

    if do_flatten and engine is None:
        # Single-worker invariant broken (a 'running' row with no engine
        # in this process) -- cannot flatten what isn't here (open risk
        # WP17a-P-02, see producer report).
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "flatten_requires_running_engine"},
        )

    if not do_flatten:
        # ------------------------------------------------------------
        # Original stop flow (flatten not requested/false, or the run
        # was orphaned/resuming).
        # ------------------------------------------------------------
        was_orphaned_or_resuming = run.status in ("orphaned", "resuming")
        previous_status = run.status

        # WP1.7a round 2 (S-13): an explicit '?flatten=false' (or a
        # paper stop) on a running engine that still holds a position
        # must never strand it silently -- compute + log + audit exactly
        # like emergency-stop already does, BEFORE the status mutation
        # below so the "still running, still holding" snapshot is exact.
        unprotected: list[UnprotectedPositionResponse] = []
        if engine is not None:
            for _sym in engine.symbols:
                _pos = engine.portfolio.get_position(_sym)
                if _pos is not None and not _pos.is_flat:
                    unprotected.append(
                        UnprotectedPositionResponse(
                            symbol=_sym, qty=str(_pos.quantity), source="ledger"
                        )
                    )

        now = datetime.now(tz=UTC)
        run.status = "stopped"
        run.stopped_at = now
        run.updated_at = now

        await db.flush()

        from api.services.audit_log import record_audit_event

        # WP1.8a-round2 (S-08), extended WP1.8b for 'resuming': stopping an
        # orphaned OR resuming run may be closing out an unprotected position
        # (S8) -- make that loud and durable, not a silent, unaudited
        # transition like a normal stop.
        if was_orphaned_or_resuming:
            log.critical(
                "runs.orphan_stopped",
                run_id=str(run_id),
                run_mode=run.run_mode,
                previous_status=previous_status,
            )
            await record_audit_event(
                db,
                event_type="emergency_stop",
                resource_type="run",
                resource_id=str(run_id),
                request=request,
                payload={
                    "run_mode": run.run_mode,
                    "trigger": "stop_run_orphaned",
                    "previous_status": previous_status,
                },
            )

        if unprotected:
            log.critical(
                "runs.stopped_with_open_position",
                run_id=str(run_id),
                symbols=[p.symbol for p in unprotected],
            )
            await record_audit_event(
                db,
                event_type="run_flatten",
                resource_type="run",
                resource_id=str(run_id),
                request=request,
                payload={
                    "phase": "skipped",
                    "trigger": "stop_flatten_false",
                    "symbols": [p.symbol for p in unprotected],
                },
            )

        # Cancel the background task if one exists for this run (an orphaned
        # run has none -- O1 -- so this is a no-op in that case).
        task = _RUN_TASKS.pop(str(run_id), None)
        _RUN_ENGINES.pop(str(run_id), None)
        if task is not None and not task.done():
            task.cancel()
            log.info("runs.engine_task_cancelled", run_id=str(run_id))

        log.info("runs.stopped", run_id=str(run_id))
        base_response = _run_orm_to_detail_response(run)
        return RunStopResponse(
            **base_response.model_dump(), flatten=None, unprotected_positions=unprotected
        )

    # ------------------------------------------------------------------
    # WP1.7a flatten path (WP17-R-09): release this row's FOR UPDATE lock
    # BEFORE calling engine.flatten() -- it can take up to 30s of real
    # wall-clock time, which must never block a concurrent stop/resume/
    # kill-switch waiting on this same row.
    # ------------------------------------------------------------------
    await db.commit()

    from api.services.audit_log import record_audit_event

    # mypy: `do_flatten` being True at this point (the `not do_flatten`
    # branch above always returns) is exactly the condition the earlier
    # `if do_flatten and engine is None: raise` guard already enforced --
    # this assert only makes that cross-variable invariant visible to
    # the type checker, it can never actually fire at runtime.
    assert engine is not None
    engine.risk_manager.trigger_kill_switch("stop_in_progress")

    # WP1.7a round 2 (S-06/S-07): flatten() itself now bounds its own
    # _cycle_lock acquisition against timeout_s, but a hung process_signal
    # call *inside* an already-acquired attempt has no such bound -- the
    # shielded task + outer wait_for is the safety net for that, and for
    # any other exception flatten() did not already catch internally. A
    # shielded task is never itself cancelled by the wait_for timeout, so
    # an in-flight SELL keeps running to completion in the background
    # even if this request gives up waiting on it.
    flatten_task = asyncio.create_task(engine.flatten(reason="stop", timeout_s=30.0))
    try:
        flatten_result = await asyncio.wait_for(asyncio.shield(flatten_task), timeout=30.0 + 5.0)
    except TimeoutError:
        log.critical("runs.stop_flatten_timeout", run_id=str(run_id))
        flatten_result = build_synthetic_flatten_result(engine, reason="stop", cause="timeout_open")
    except Exception as exc:
        log.critical(
            "runs.stop_flatten_exception", run_id=str(run_id), error=str(exc), exc_info=True
        )
        flatten_result = build_synthetic_flatten_result(
            engine, reason="stop", cause="error", error=str(exc)
        )
    flatten_response = _flatten_result_to_response(flatten_result)

    # Re-acquire a short, fresh transaction to record the outcome.
    result2 = await db.execute(
        select(RunORM).where(RunORM.id == run_id).with_for_update()
    )
    run2 = result2.scalar_one_or_none()

    if run2 is None or run2.status != "running":
        # A concurrent kill-switch/emergency-stop/second stop already moved
        # this row while flatten was running -- surface the flatten result
        # without fighting whatever already happened to the row.
        log.warning(
            "runs.stop_flatten_race_lost",
            observed_status=run2.status if run2 is not None else None,
        )
        base_response = _run_orm_to_detail_response(run2 if run2 is not None else run)
        return RunStopResponse(**base_response.model_dump(), flatten=flatten_response)

    if not flatten_result.complete:
        now = datetime.now(tz=UTC)
        run2.entries_latch_reason = "flatten_incomplete"
        run2.entries_latched_at = now
        run2.updated_at = now
        await record_audit_event(
            db,
            event_type="run_flatten",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={
                "phase": "incomplete",
                "trigger": "stop",
                "outcome": flatten_result.outcome,
                "cause": [r.cause for r in flatten_result.symbols if r.cause is not None],
            },
        )
        # S-08: persist and commit FIRST. Only once that succeeds do we
        # touch the in-memory latch -- add the durable 'flatten_incomplete'
        # reason and release 'stop_in_progress' (the temporary latch this
        # handler itself added above); if the commit raises, the
        # exception propagates and 'stop_in_progress' is left in place
        # (fail-closed: the engine stays latched either way).
        await db.commit()
        engine.risk_manager.trigger_kill_switch("flatten_incomplete")
        engine.risk_manager.reset_kill_switch("stop_in_progress")
        log.critical(
            "runs.stop_flatten_incomplete",
            run_id=str(run_id),
            outcome=flatten_result.outcome,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "flatten_incomplete",
                "flatten": flatten_response.model_dump(by_alias=True),
            },
        )

    # Flatten complete -- proceed with the stop exactly like the
    # no-flatten path above (I8: never cancel while incomplete).
    now = datetime.now(tz=UTC)
    run2.status = "stopped"
    run2.stopped_at = now
    run2.updated_at = now
    if run2.entries_latch_reason == "flatten_incomplete":
        run2.entries_latch_reason = None
        run2.entries_latched_at = None
    await record_audit_event(
        db,
        event_type="run_flatten",
        resource_type="run",
        resource_id=str(run_id),
        request=request,
        payload={"phase": "completed", "trigger": "stop", "outcome": flatten_result.outcome},
    )
    await db.flush()

    task = _RUN_TASKS.pop(str(run_id), None)
    _RUN_ENGINES.pop(str(run_id), None)
    if task is not None and not task.done():
        task.cancel()
        log.info("runs.engine_task_cancelled", run_id=str(run_id))

    log.info("runs.stopped", run_id=str(run_id), flatten_outcome=flatten_result.outcome)
    base_response = _run_orm_to_detail_response(run2)
    return RunStopResponse(**base_response.model_dump(), flatten=flatten_response)


# ---------------------------------------------------------------------------
# POST /api/v1/runs/{run_id}/emergency-stop  -- SEC-006 (Sprint 45)
# ---------------------------------------------------------------------------

@router.post(
    "/{run_id}/emergency-stop",
    status_code=status.HTTP_200_OK,
    response_model=RunEmergencyStopResponse,
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
        "to halt (e.g. a stuck client retrying DELETE). "
        "WP1.7a: '?flatten=' is OPTIONAL and defaults to false -- this "
        "endpoint must NEVER refuse to stop (SY-07, never 422). It always "
        "stops regardless of the flatten outcome; when the run still "
        "holds crypto afterwards the response lists 'unprotected_positions' "
        "and 'runs.stopped_with_open_position' is logged at critical."
    ),
)
async def emergency_stop_run(
    run_id: uuid.UUID,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    reason: Annotated[str | None, Header(alias="X-Emergency-Reason")] = None,
    flatten: Annotated[bool, Query()] = False,
) -> RunEmergencyStopResponse:
    """Hard-stop a run with persistent audit trail.

    Body identical to DELETE /runs/{id}: transitions status to 'stopped',
    cancels the engine task, removes from _RUN_ENGINES / _LEARNING_INSTANCES.
    WP1.7a adds an optional, never-blocking flatten pass (SY-07) -- this
    endpoint always stops, even when flatten is incomplete or omitted.

    Additional SEC-006 behaviour:
      * One ``audit_events`` row with ``event_type='emergency_stop'`` is
        written BEFORE the cancel sequence so the trail survives even if
        a subsequent step fails.
      * Optional ``X-Emergency-Reason`` header is captured in the audit
        payload so operators can leave a one-line incident note.
    """
    log = logger.bind(endpoint="emergency_stop_run", run_id=str(run_id))
    log.warning("runs.emergency_stop_requested", reason=reason, flatten=flatten)

    # WP1.8a-round2 (C-01/S-01 item 4): lock the row before the status guard
    # so a concurrent resume's uncommitted transition is waited on.
    stmt = select(RunORM).where(RunORM.id == run_id).with_for_update()
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # WP1.8a-round2 (S-08): an orphaned run may still hold an unprotected
    # position -- emergency-stop must be able to close it out, not 409.
    # WP1.8b (S2-01): 'resuming' is accepted too -- see stop_run's
    # identical rationale.
    if run.status not in ("running", "orphaned", "resuming"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot emergency-stop run {run_id}: status is '{run.status}'. "
                f"Only 'running', 'orphaned' or 'resuming' runs can be "
                f"emergency-stopped."
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
            "flatten_requested": flatten,
        },
    )

    engine = _RUN_ENGINES.get(str(run_id)) if run.status == "running" else None
    flatten_result: FlattenResult | None = None

    if flatten and engine is not None:
        # WP17-R-09: release the row lock before the (up to 30s) flatten
        # call -- this transaction has already committed its audit row via
        # record_audit_event's own flush; commit here releases the FOR
        # UPDATE lock the rest of this request no longer needs to hold.
        await db.commit()
        engine.risk_manager.trigger_kill_switch("stop_in_progress")

        # WP1.7a round 2 (S-05): emergency-stop must ALWAYS stop. Run
        # flatten as a shielded task so a caller-side timeout can never
        # cancel an in-flight SELL, and bound the wait so a hung flatten
        # (or an exception it did not already catch internally) can never
        # prevent the stop below from happening.
        flatten_task = asyncio.create_task(
            engine.flatten(reason="emergency_stop", timeout_s=30.0)
        )
        try:
            flatten_result = await asyncio.wait_for(
                asyncio.shield(flatten_task), timeout=30.0 + 5.0
            )
        except TimeoutError:
            log.critical("runs.emergency_stop_flatten_timeout", run_id=str(run_id))
            flatten_result = build_synthetic_flatten_result(
                engine, reason="emergency_stop", cause="timeout_open"
            )
        except Exception as exc:
            log.critical(
                "runs.emergency_stop_flatten_exception",
                run_id=str(run_id),
                error=str(exc),
                exc_info=True,
            )
            flatten_result = build_synthetic_flatten_result(
                engine, reason="emergency_stop", cause="error", error=str(exc)
            )

        # mypy: every branch above (success, TimeoutError, any other
        # Exception) assigns a real FlattenResult -- this is never None
        # by the time control reaches here.
        assert flatten_result is not None

        result2 = await db.execute(
            select(RunORM).where(RunORM.id == run_id).with_for_update()
        )
        refreshed = result2.scalar_one_or_none()
        if refreshed is not None:
            run = refreshed

        await record_audit_event(
            db,
            event_type="run_flatten",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={
                "phase": "completed" if flatten_result.complete else "incomplete",
                "trigger": "emergency_stop",
                "outcome": flatten_result.outcome,
            },
        )
        if not flatten_result.complete:
            log.critical(
                "flatten.incomplete",
                run_id=str(run_id),
                outcome=flatten_result.outcome,
            )

    # SY-07: report any position still held so the operator knows the
    # exchange-side exposure this stop leaves behind. Best-effort from the
    # live in-process ledger only (source="ledger").
    unprotected: list[UnprotectedPositionResponse] = []
    exposure_unknown = False
    if engine is not None:
        for _sym in engine.symbols:
            _pos = engine.portfolio.get_position(_sym)
            if _pos is not None and not _pos.is_flat:
                unprotected.append(
                    UnprotectedPositionResponse(
                        symbol=_sym, qty=str(_pos.quantity), source="ledger"
                    )
                )
    elif run.run_mode == "live":
        # WP1.7a round 2 (S-14): a live run with NO in-process engine
        # (a truly orphaned live run, or the single-worker invariant
        # broken) -- exposure is not "empty", it is genuinely UNKNOWN.
        exposure_unknown = True
        log.critical("runs.stopped_with_unknown_exposure", run_id=str(run_id))
        await record_audit_event(
            db,
            event_type="emergency_stop",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={"trigger": "emergency_stop_exposure_unknown", "exposure_unknown": True},
        )

    if unprotected:
        log.critical(
            "runs.stopped_with_open_position",
            run_id=str(run_id),
            symbols=[p.symbol for p in unprotected],
        )

    # SY-07: emergency stop ALWAYS stops, regardless of the flatten
    # outcome above.
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

    log.warning(
        "runs.emergency_stopped",
        reason=reason,
        flatten_outcome=flatten_result.outcome if flatten_result is not None else None,
    )
    base_response = _run_orm_to_detail_response(run)
    return RunEmergencyStopResponse(
        **base_response.model_dump(),
        flatten=_flatten_result_to_response(flatten_result) if flatten_result is not None else None,
        unprotected_positions=unprotected,
        exposure_unknown=exposure_unknown,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/runs/{run_id}/entries-latch/clear  -- WP1.7a (SY-09)
# ---------------------------------------------------------------------------

class _ClearEntriesLatchRequest(BaseModel):
    """Body for POST /runs/{id}/entries-latch/clear."""

    reason: str = Field(min_length=3, max_length=500)

    @field_validator("reason")
    @classmethod
    def _sanitise(cls, v: str) -> str:
        # WP1.7a round 2 (S-17): sanitise identically to the global
        # kill-switch's clear reason and X-Emergency-Reason header.
        from api.services.audit_log import sanitise_reason

        return sanitise_reason(v)


@router.post(
    "/{run_id}/entries-latch/clear",
    status_code=status.HTTP_200_OK,
    response_model=EntriesLatchClearResponse,
    responses={
        401: {"description": "Missing or invalid X-Admin-Key"},
        403: {"description": "X-Admin-Key rejected"},
        404: {"description": "Run not found"},
        409: {"description": "Run is not latched"},
        422: {"description": "Live run missing X-Live-Confirm-Token"},
    },
    summary="Clear a run's persisted entries latch",
    description=(
        "Removes the 'flatten_incomplete' per-run latch (SY-02/SY-09) so "
        "entries can resume for THIS run. Requires the admin key plus a "
        "3-500 character reason; a live run also requires "
        "X-Live-Confirm-Token, because clearing re-enables BUYs. Never "
        "automatic (WP1.2 S9). Removes only 'flatten_incomplete' -- an "
        "active GLOBAL kill switch, if also latching this run's engine, "
        "is untouched (use POST /emergency/kill-switch/clear for that)."
    ),
    dependencies=[Depends(require_admin)],
)
async def clear_entries_latch(
    run_id: uuid.UUID,
    body: _ClearEntriesLatchRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: Request,
    x_live_confirm_token: Annotated[str | None, Header()] = None,
) -> EntriesLatchClearResponse:
    log = logger.bind(endpoint="clear_entries_latch", run_id=str(run_id))

    stmt = select(RunORM).where(RunORM.id == run_id).with_for_update()
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.run_mode == "live":
        from api.config import get_settings
        from trading.safety import LiveTradingGate

        gate = LiveTradingGate()
        gate_result = gate.check_gate(
            settings=get_settings(), confirm_token=x_live_confirm_token or ""
        )
        if not gate_result.passed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="X-Live-Confirm-Token is required to clear a live run's entries latch.",
            )

    if run.entries_latch_reason is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "not_latched"},
        )

    cleared_reason = run.entries_latch_reason
    now = datetime.now(tz=UTC)
    run.entries_latch_reason = None
    run.entries_latched_at = None
    run.updated_at = now

    # WP1.7a round 2 (S-10): write the audit row WITHOUT swallowing, then
    # commit. Only after that commit succeeds do we touch the in-process
    # engine's latch -- a failure here rolls back (via the session
    # dependency's own except-branch) and returns 503, leaving the run's
    # persisted latch (and the engine's own risk-manager reason) both
    # untouched.
    from api.services.audit_log import record_audit_event_strict

    try:
        await record_audit_event_strict(
            db,
            event_type="entries_latch_cleared",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={"reason": body.reason, "cleared_reason": cleared_reason},
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        log.critical("runs.entries_latch_clear_persist_failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "entries_latch_clear_not_persisted"},
        ) from exc

    # Reflect the clear on the in-process engine (if this run is currently
    # running) -- risk_manager.reset_kill_switch(reason) only removes the
    # ONE reason named; a global-latch reason held by the same engine (if
    # any) is left untouched, exactly matching the DB-level "only
    # flatten_incomplete" scope (SY-09).
    engine = _RUN_ENGINES.get(str(run_id))
    still_latched_by: list[str] = []
    if engine is not None:
        engine.risk_manager.reset_kill_switch(cleared_reason)
        still_latched_by = sorted(engine.risk_manager.kill_switch_reasons)

    log.warning("runs.entries_latch_cleared", cleared_reason=cleared_reason)
    return EntriesLatchClearResponse(
        run_id=str(run_id),
        cleared=cleared_reason,
        still_latched_by=still_latched_by,
    )


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

    # WP1.8a-round2 (C-05/S-02): the new 'orphaned'/'resuming' statuses must
    # be guarded too -- an orphaned live run can still hold an unprotected
    # position (S8); archiving it would silently remove it from the S8
    # repeater, boot recovery, and the resume endpoint with zero audit trail.
    if run.status not in ("stopped", "error"):
        log.warning("runs.archive_blocked_not_terminal", current_status=run.status)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot archive run {run_id}: status is '{run.status}'. "
                "Only 'stopped' or 'error' runs can be archived."
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
        422: {
            "description": (
                "Strategy not available for live trading (demoted), OR the source "
                "run's exit config is invalid (invalid_exit_config / "
                "exit_manager_required), OR the resolved allow_pyramiding is True "
                "for live (live_pyramiding_forbidden, WP1.3a)."
            )
        },
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

    # ------------------------------------------------------------------
    # WP1.3a (SY-13a-15/17): validate the SOURCE config for LIVE mode
    # (E11 included) right after the availability check, before the
    # promotion-gate evidence check, the 3-layer safety gate, and the audit
    # SAVEPOINT.  A legacy paper run predates this validator; a failure
    # here means "you can only promote what you validated" (SY-13a-09).
    # ------------------------------------------------------------------
    promotion_strategy_cls = _get_strategy_registry().get(promotion_strategy_name)
    if promotion_strategy_cls is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Strategy {promotion_strategy_name!r} from source run is no longer registered.",
        )
    _source_strategy_params: dict[str, Any] = source_run.config.get("strategy_params", {}) or {}
    try:
        _promote_verdict = validate_run_exit_config(
            promotion_strategy_cls,
            bracket=source_run.config.get("bracket_config") or {},
            trailing_stop_pct=_source_strategy_params.get("trailing_stop_pct"),
            mode=RunMode.LIVE,
            allow_pyramiding=source_run.config.get("allow_pyramiding"),
            timeframe=str(source_run.config.get("timeframe", "1h")),
        )
    except ExitConfigError as exc:
        log.warning(
            "runs.promote_exit_config_invalid",
            source_run_id=str(run_id),
            code=exc.code,
            reasons=[i.reason for i in exc.issues],
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_exit_config_error_detail(exc),
        ) from exc

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

    # WP1.7a (SY-08): while the global kill switch is latched, promotion
    # to a fresh live run is rejected outright -- same rationale as
    # create_run's own check.
    from api.services import kill_switch as _kill_switch

    if _kill_switch.is_active():
        log.warning("runs.promote_blocked_kill_switch_active", source_run_id=str(run_id))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "kill_switch_active"},
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

    # WP1.3a (I8): persist the NORMALISED bracket config and the RESOLVED
    # allow_pyramiding bool, exactly like create_run -- a promoted run's
    # persisted config is never the raw legacy source.
    if _promote_verdict.config.bracket:
        promoted_config["bracket_config"] = dict(_promote_verdict.config.bracket)
    else:
        promoted_config.pop("bracket_config", None)
    promoted_config["allow_pyramiding"] = _promote_verdict.allow_pyramiding
    # WP13a-S-07 (security round 2): mirror create_run -- persist the
    # NORMALISED trailing_stop_pct into strategy_params (or remove it when
    # unset), instead of leaving the raw source value in place while the
    # spawned engine itself gets the normalised one.  Extracted to
    # ``_build_promoted_strategy_params`` (WP13a-C-R2-09) for direct unit
    # coverage: it always returns a fresh dict, never mutating
    # ``source_run.config["strategy_params"]`` in place (it belongs to the
    # SOURCE paper run, still attached to this session).
    _promoted_strategy_params = _build_promoted_strategy_params(
        source_run.config.get("strategy_params"),
        _promote_verdict.config.trailing_stop_pct,
    )
    promoted_config["strategy_params"] = _promoted_strategy_params

    strategy_name = promotion_strategy_name
    strategy_cls = promotion_strategy_cls

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
            strategy_params=_promoted_strategy_params,
            symbols=source_run.config.get("symbols", []),
            timeframe=timeframe_val,
            initial_capital=source_run.config.get("initial_capital", "10000"),
            trailing_stop_pct=_promote_verdict.config.trailing_stop_pct,
            bracket_config=dict(_promote_verdict.config.bracket),
            enable_adaptive_learning=False,
            allow_pyramiding=_promote_verdict.allow_pyramiding,
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
# POST /api/v1/runs/{run_id}/resume
# WP1.8a (Verbeterplan v2 synthesis spec §5) -- resume an orphaned live run.
# ---------------------------------------------------------------------------


async def _kill_switch_triggered_after(db: AsyncSession, run: RunORM) -> bool:
    """True if a global kill-switch audit row postdates ``run.started_at`` (S4).

    Only the aggregate ``kill_switch`` event counts -- a per-run
    ``emergency_stop`` is an operator closing this one run out, not the
    global "something is on fire" signal S4 gates a normal resume on.
    """
    result = await db.execute(
        select(func.count())
        .select_from(AuditEventORM)
        .where(
            AuditEventORM.event_type == "kill_switch",
            AuditEventORM.timestamp > run.started_at,
        )
    )
    return (result.scalar() or 0) > 0


@router.post(
    "/{run_id}/resume",
    status_code=status.HTTP_200_OK,
    response_model=RunDetailResponse,
    responses={
        401: {"description": "Missing or invalid X-Admin-Key"},
        403: {"description": "X-Admin-Key rejected, or live trading gate check failed"},
        404: {"description": "Run not found"},
        409: {"description": "Run is not orphaned, or the resume was rejected"},
        422: {"description": "Strategy/config invalid, or invalid mode"},
    },
    summary="Resume an orphaned live run",
    description=(
        "Resumes an 'orphaned' live run under its EXISTING run_id (WP1.8a). "
        "Live-only -- paper runs resume automatically at boot and never reach "
        "this endpoint. Requires X-Admin-Key (WP1.8a-round2 S-07 -- same "
        "mechanism as /emergency/kill-switch) IN ADDITION TO the full 3-layer "
        "live-trading safety gate (env flag + API keys + X-Live-Confirm-Token "
        "header), even for mode=protective (U2: protective is token-gated and "
        "always manual). The compare-and-set orphaned->resuming transition "
        "ensures exactly one of two concurrent resume requests succeeds. "
        "WP1.8b: scans the exchange for every order placed under this run's "
        "clientOrderId prefix since it started, cancels any still open, "
        "polls each to a terminal state, and imports any order/fill missing "
        "from the DB before replaying. Any scan/cancel/trade-fetch failure, "
        "or a fill-history inconsistency, returns 409 with a machine-"
        "readable reason (e.g. 'fill_history_partial', 'fill_history_corrupt', "
        "'exchange_scan_incomplete') and reverts the run to 'orphaned' -- "
        "never overridable."
    ),
    dependencies=[Depends(require_admin)],
)
async def resume_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: Request,
    mode: Annotated[Literal["normal", "protective"], Query()] = "normal",
    x_live_confirm_token: Annotated[str | None, Header()] = None,
) -> RunDetailResponse:
    """Resume an orphaned live run in place (WP1.8a, reworked WP1.8a-round2
    and again for WP1.8b's S2-01 three-short-transaction redesign).

    Order of steps: 404 -> gate layers 1-3 (403) -> resolve/validate ALL
    config (strategy/symbols/timeframe/params/trailing/bracket/capital,
    422 on any failure -- C-03/S-03, before any DB mutation so a config
    problem never leaves a run stuck 'resuming') -> concurrency cap (503)
    -> **transaction (a)**: compare-and-set orphaned->resuming, commit
    IMMEDIATELY (0 rows -> 409; a real commit here, not a flush, is what
    releases the row lock before the scan -- WP18a-S2-01) -> S4
    kill-switch check #1 (no lock held) -> **transaction (b)**: build a
    real CCXT exchange (P-02) and call ``prepare_live_resume`` (the
    WP1.8b exchange scan/cancel/import -- also with no run-row lock held;
    every write it performs is its own short transaction against the
    orders/fills tables), guaranteed ``exchange.close()`` in a
    ``finally`` -> S4 kill-switch check #2 (closes the window the scan
    was running in) -> **transaction (c)**: audit ``run_resumed{...}``
    (S-09 enriched payload), conditional resuming->running (0 rows -> 409
    ``resume_state_lost``, reverting as appropriate), commit -> spawn the
    task with NO awaits between the commit and task-registration (C-01
    item 2).

    WP18a-S2-01: because transaction (a) commits before the scan starts,
    a kill-switch or stop/emergency-stop issued WHILE the scan is running
    sees the row as plain 'resuming' with NO lock held on it and can act
    immediately (kill-switch moves it to 'orphaned'; stop/emergency-stop
    move it straight to 'stopped') -- it no longer waits out the whole
    scan. Transaction (c)'s own conditional CAS then naturally detects
    that pre-emption (0 rows matched) and reverts to a 409 instead of
    silently overwriting a concurrent stop/kill-switch decision.

    Every exception raised between transaction (a) committing and
    transaction (c) committing -- other than an ``HTTPException`` this
    function itself raises after already performing its own explicit
    rollback+audit -- reverts the row back to 'orphaned' (a no-op if a
    concurrent kill-switch/stop already moved it elsewhere, C-04/S-06
    defense in depth) and writes a ``run_resume_rejected`` audit row
    before surfacing a 500 (no exception on this path may ever leave a
    run silently stuck at 'resuming').
    """
    from api.config import get_settings
    from api.services.audit_log import record_audit_event
    from api.services.run_recovery import prepare_live_resume
    from trading.safety import LiveTradingGate

    log = logger.bind(endpoint="resume_run", run_id=str(run_id), mode=mode)
    settings = get_settings()

    row = await db.execute(select(RunORM).where(RunORM.id == run_id))
    run: RunORM | None = row.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found.",
        )

    if run.run_mode != "live":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Run {run_id} is a {run.run_mode!r} run. Resume is live-only "
                "(S10) -- paper runs resume automatically at API boot."
            ),
        )

    # WP1.7a (SY-08): the global latch and this run's own persisted latch
    # are both orthogonal to (and cheaper than) the audit-row-derived
    # ``kill_switch_after_start`` check further down -- a protective resume
    # is exempt from ALL THREE (U2/S4: protective resumes exist precisely
    # to recover a run under an active/uncertain safety state; the
    # rebuilt engine still starts latched either way via ``apply_latch``,
    # so no entries can slip through).
    from api.services import kill_switch as _kill_switch

    if mode == "normal" and _kill_switch.is_active():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "kill_switch_active"},
        )
    if mode == "normal" and run.entries_latch_reason is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "entries_latched", "reason": run.entries_latch_reason},
        )

    # Gate layers 1-3 (env flag + API keys + token).  U2: mode=protective is
    # ALSO token-gated -- there is no lighter-weight path for either mode.
    gate = LiveTradingGate()
    gate_result = gate.check_gate(
        settings=settings,
        confirm_token=x_live_confirm_token or "",
    )
    if not gate_result.passed:
        failed_layers = [layer.name for layer in gate_result.layers if not layer.passed]
        log.warning("runs.resume_live_gate_failed", failures=gate_result.failures)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(f"Live trading gate check failed. Failed layers: {', '.join(failed_layers)}."),
        )

    # -----------------------------------------------------------------
    # WP1.8a-round2 (C-03/S-03): resolve and validate EVERY config value
    # the spawned engine will need, BEFORE the orphaned->resuming CAS.
    # A 422 here costs nothing -- the run stays 'orphaned', untouched.
    # Validating any of this AFTER the CAS (as round-1 did) risks leaving
    # a run stuck at 'resuming' forever if the spawn itself never happens.
    # -----------------------------------------------------------------
    orphan_config = dict(run.config or {})

    strategy_name = (str(orphan_config.get("strategy_name", "")).lower().replace("-", "_"))
    if not is_mode_allowed(strategy_name, RunMode.LIVE):
        availability = get_availability(strategy_name)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Strategy {strategy_name!r} is not available for live "
                f"trading (status={availability.status.value}) and cannot "
                f"be resumed. {availability.demotion_reason}".strip()
            ),
        )

    strategy_cls = _get_strategy_registry().get(strategy_name)
    if strategy_cls is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Strategy {strategy_name!r} is no longer registered.",
        )

    symbols: list[str] = orphan_config.get("symbols") or []
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Run {run_id} has no configured symbols; cannot resume.",
        )

    try:
        timeframe = TimeFrame(str(orphan_config.get("timeframe", "1h")))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_timeframe",
        ) from exc

    strategy_params: dict[str, Any] = orphan_config.get("strategy_params") or {}
    if not isinstance(strategy_params, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_strategy_params",
        )

    _bracket_config_raw: dict[str, object] = orphan_config.get("bracket_config") or {}
    if not isinstance(_bracket_config_raw, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_bracket_config",
        )
    _raw_trailing_for_validation = strategy_params.get("trailing_stop_pct")

    # ------------------------------------------------------------------
    # WP1.3a (SY-13a-16/SY-13a-15): replaces the old plain type-only checks.
    # Normal mode: full validation, 422 before the CAS (I6) -- the run
    # stays orphaned, untouched. Protective mode: "salvage with explicit
    # waiver" -- each component (bracket, trailing) is validated
    # INDEPENDENTLY; only the valid one(s) survive. Never raises. This is
    # the only route to a running engine that can reach
    # ``stop_run?flatten=true`` for a run whose persisted config has gone
    # stale/invalid.
    # ------------------------------------------------------------------
    exit_config_waived_payload: dict[str, Any] | None = None
    exit_manager_missing: bool | None = None
    if mode == "normal":
        try:
            _resume_verdict = validate_run_exit_config(
                strategy_cls,
                bracket=_bracket_config_raw,
                trailing_stop_pct=_raw_trailing_for_validation,
                mode=RunMode.LIVE,
                allow_pyramiding=orphan_config.get("allow_pyramiding"),
                timeframe=str(timeframe),
            )
        except ExitConfigError as exc:
            log.warning(
                "runs.resume_exit_config_invalid",
                code=exc.code,
                reasons=[i.reason for i in exc.issues],
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=_exit_config_error_detail(exc),
            ) from exc
        bracket_config: dict[str, object] = dict(_resume_verdict.config.bracket)
        trailing_pct: float | None = _resume_verdict.config.trailing_stop_pct
        resolved_allow_pyramiding: bool = _resume_verdict.allow_pyramiding
    else:  # protective
        from trading.exit_config import salvage_exit_config

        _salvaged_config, _salvage_issues = salvage_exit_config(
            bracket=_bracket_config_raw,
            trailing_stop_pct=_raw_trailing_for_validation,
        )
        bracket_config = dict(_salvaged_config.bracket)
        trailing_pct = _salvaged_config.trailing_stop_pct
        # Entries are already fully blocked in protective mode
        # (_drop_entry_signals) -- never persisted (SY-13a-16).
        resolved_allow_pyramiding = False
        exit_manager_missing = not _salvaged_config.has_downside_exit
        # WP13a-S-01 (security round 2): a BUY-only run resumed protectively
        # with NO exit config at all, or a TP-only one, produces ZERO
        # salvage issues (there is nothing invalid to drop -- the bracket
        # dict is simply empty, or has only a TP key that parses fine on
        # its own). Gating the waiver on "salvage produced issues" alone
        # therefore missed exactly the case SY-13a-16 exists for: the
        # engine's own require_exit_manager(...) still raises
        # exit_manager_required with no waiver granted, the background
        # task dies, the run goes to 'error', and flatten (which requires
        # a running engine) becomes unreachable -- the one in-platform
        # unwind route for this exact class of legacy run is lost. The
        # waiver must also cover "no issues, but the strategy requires an
        # exit and none survived salvage."
        _needs_waiver = bool(_salvage_issues) or (
            strategy_requires_exit_manager(strategy_cls) and exit_manager_missing
        )
        if _salvage_issues:
            exit_config_waived_payload = {
                "code": "invalid_exit_config",
                "errors": [
                    {
                        "field": i.field,
                        "reason": i.reason,
                        "value": i.value,
                        "min": i.min,
                        "max": i.max,
                        "message": i.message,
                    }
                    for i in _salvage_issues
                ],
            }
            log.warning(
                "resume.protective_exit_config_waived",
                errors=exit_config_waived_payload["errors"],
            )
        elif _needs_waiver:
            # No salvage issues (nothing was invalid to drop), but the
            # strategy requires an exit manager and none survived --
            # still an explicit waiver, with an empty errors[] list.
            exit_config_waived_payload = {
                "code": "exit_manager_required",
                "errors": [],
            }
            log.warning("resume.protective_exit_config_waived", errors=[])
        if exit_manager_missing:
            log.critical(
                "resume.protective_without_exit_manager",
                run_id=str(run_id),
            )

    initial_capital = str(orphan_config.get("initial_capital", "10000"))
    try:
        Decimal(initial_capital)
    except ArithmeticError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_initial_capital",
        ) from exc

    # Concurrency cap (AR-006, same as promote_to_live).
    active_count = sum(1 for t in _RUN_TASKS.values() if not t.done())
    if active_count >= settings.max_concurrent_runs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Concurrent run cap reached ({active_count}/{settings.max_concurrent_runs}). "
                "Stop an existing run before resuming."
            ),
        )

    # Transaction (a): compare-and-set orphaned -> resuming, committed
    # IMMEDIATELY (WP18a-S2-01) -- 0 rows updated means either the run was
    # never orphaned, or a concurrent resume request already won the race
    # -- both are a 409 (WP18-R-02). A real commit (not a flush) here is
    # what releases the row lock before the scan begins.
    #
    # WP18b-S-01: a fresh, random ``resume_attempt_id`` becomes this
    # attempt's FENCE, embedded in ``config`` by this very CAS.  Every
    # later mutation this attempt makes -- the final CAS, `_reject`, and
    # every per-order import transaction inside `scan_and_import` -- is
    # conditioned on `config->>'resume_attempt_id' == fence` in addition to
    # `status == 'resuming'`.  Without the fence, a resume that a kill
    # switch (or a second resume) has already cut off and recycled
    # (`resuming -> orphaned -> resuming` again, under a NEW attempt) would
    # match a plain `status == 'resuming'` WHERE clause even though the row
    # now belongs to a DIFFERENT attempt -- an ABA race that lets a
    # superseded resume both start its engine and double-import fills
    # alongside the attempt that actually owns the row (WP18b-S-01).
    #
    # The fence is NOT ``runs.updated_at`` -- migration 001's
    # ``BEFORE UPDATE`` trigger (``trigger_set_updated_at()``)
    # unconditionally overwrites that column with the database's own
    # ``now()`` on every UPDATE, so an application-supplied value can never
    # be compared back for equality after a round trip. ``config`` (JSONB)
    # has no such trigger, so a value written here survives verbatim.
    fence = str(uuid.uuid4())
    resuming_config = dict(orphan_config)
    resuming_config["resume_attempt_id"] = fence
    cas_result = await db.execute(
        update(RunORM)
        .where(RunORM.id == run_id, RunORM.status == "orphaned")
        .values(status="resuming", updated_at=datetime.now(tz=UTC), config=resuming_config)
    )
    if cas_result.rowcount == 0:  # type: ignore[attr-defined]
        await db.rollback()
        await db.refresh(run)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Run {run_id} is not orphaned (current status: "
                f"{run.status!r}); it cannot be resumed."
            ),
        )
    await db.commit()
    await db.refresh(run)

    async def _reject(reason: str) -> None:
        """Roll status back to orphaned and audit the rejection (S11).

        WP18b-S-02: rolls back FIRST -- any half-finished work left
        uncommitted on `db` by the caller (e.g. an order row flushed but
        its fills not yet added, before an exception hit) must never be
        silently committed by this function's own later `db.commit()`.
        Without the rollback, a per-order write that raised partway
        through could otherwise be persisted as a truncated, invalid
        order-with-no-fills row that then poisons every future resume
        attempt's own pre-scan integrity check (WP18b-S-02 PC evidence).

        A short transaction of its own, fenced (WP18b-S-01): the CAS is a
        safe no-op (0 rows) if a concurrent kill-switch/stop/emergency-stop
        already moved the row away from 'resuming', OR if a DIFFERENT
        resume attempt now owns 'resuming' under a new fence -- the audit
        row is still written either way so the rejection is never silently
        lost (S-03).
        """
        await db.rollback()
        await db.execute(
            update(RunORM)
            .where(
                RunORM.id == run_id,
                RunORM.status == "resuming",
                RunORM.config["resume_attempt_id"].astext == fence,
            )
            .values(status="orphaned", updated_at=datetime.now(tz=UTC))
        )
        await record_audit_event(
            db,
            event_type="run_resume_rejected",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={"reason": reason, "mode": mode},
        )
        await db.commit()

    kill_switch_after_start = False
    try:
        # S4 check #1: normal resume is blocked by a kill-switch pressed
        # after this run started; protective resume is explicitly exempt
        # (U2/S4). No row lock held here (transaction (a) already
        # committed) -- a plain read.
        if mode == "normal" and await _kill_switch_triggered_after(db, run):
            await _reject("kill_switch_after_start")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="kill_switch_after_start",
            )

        # Transaction (b): build a real CCXT exchange for the scan
        # (WP1.8b P-02) and run prepare_live_resume/scan_and_import with
        # NO run-row lock held -- this is the step that can take real
        # wall-clock time (cancel-then-poll up to 30s per open order).
        # ``exchange.close()`` runs in a finally so a scan failure never
        # leaks the connection (P-02: guaranteed cleanup).
        exchange = _build_live_ccxt_exchange(settings)
        try:
            snapshot = await prepare_live_resume(db, run, exchange, fence=fence)
        except ResumeRejected as exc:
            await _reject(exc.reason)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=exc.reason,
            ) from exc
        finally:
            try:
                await exchange.close()
            except Exception:
                log.warning("runs.resume_exchange_close_failed", exc_info=True)

        # S4 check #2 (WP1.8a-round2 C-01 item 1): the scan can take real
        # wall-clock time; re-check right before transaction (c) so a
        # kill-switch pressed WHILE it was running is not missed.
        kill_switch_after_start = await _kill_switch_triggered_after(db, run)
        if mode == "normal" and kill_switch_after_start:
            await _reject("kill_switch_after_start")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="kill_switch_after_start",
            )

        # WP1.8a-round2 (S-09): enrich the audit payload with everything an
        # operator reviewing the incident timeline would want, computed
        # from data already in hand (no extra DB round-trip). S2-02:
        # BUY-before-SELL tie-break on equal executed_at.
        fill_count = len(snapshot.fills)
        rebuilt_qty_by_symbol: dict[str, str] = {}
        for fill in sorted(snapshot.fills, key=replay_sort_key):
            held = Decimal(rebuilt_qty_by_symbol.get(fill.symbol, "0"))
            held = held + fill.quantity if fill.side == OrderSide.BUY else held - fill.quantity
            rebuilt_qty_by_symbol[fill.symbol] = str(held)
        elapsed_seconds = max((datetime.now(tz=UTC) - run.started_at).total_seconds(), 0.0)

        # Transaction (c): S4 re-check already done above; audit + the
        # conditional final CAS + commit, all in one short transaction --
        # no row lock is held any longer than this.
        _resumed_audit_payload: dict[str, Any] = {
            "mode": mode,
            "previous_status": "orphaned",
            "kill_switch_after_start": kill_switch_after_start,
            "fill_count": fill_count,
            "rebuilt_qty_by_symbol": rebuilt_qty_by_symbol,
            "peak_equity_hint": (
                str(snapshot.peak_equity_hint) if snapshot.peak_equity_hint is not None else None
            ),
            "elapsed_seconds": elapsed_seconds,
        }
        if exit_config_waived_payload is not None:
            # WP1.3a (SY-13a-16): protective salvage summary.
            _resumed_audit_payload["exit_config_waived"] = exit_config_waived_payload
        if mode == "protective":
            # WP13a-S-01 (security round 2): always record whether the
            # resumed engine has a downside exit, even in the (secondary
            # finding) case where exit_config_waived_payload is None
            # (normal mode never reaches this branch).
            _resumed_audit_payload["exit_manager_missing"] = exit_manager_missing
        await record_audit_event(
            db,
            event_type="run_resumed",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload=_resumed_audit_payload,
        )

        # WP1.8a-round2 (S-10): persist protective_mode onto the run's own
        # config so it survives process restarts and is visible on GET.
        updated_config = dict(orphan_config)
        updated_config["protective_mode"] = mode == "protective"
        # WP1.3a (I8): a NORMAL resume persists the normalised bracket
        # config and the resolved allow_pyramiding bool -- exactly like
        # create/promote.  A PROTECTIVE resume never persists the salvaged
        # (possibly reduced) config or the forced-False pyramiding value:
        # the waiver must never leak into a later normal resume, which
        # re-reads and re-validates the ORIGINAL persisted config
        # (SY-13a-16).
        if mode == "normal":
            updated_config["bracket_config"] = bracket_config
            updated_config["allow_pyramiding"] = resolved_allow_pyramiding

        final_now = datetime.now(tz=UTC)
        final_result = await db.execute(
            update(RunORM)
            .where(
                RunORM.id == run_id,
                RunORM.status == "resuming",
                RunORM.config["resume_attempt_id"].astext == fence,
            )
            .values(status="running", updated_at=final_now, config=updated_config)
        )
        if final_result.rowcount == 0:  # type: ignore[attr-defined]
            # WP18a-S2-01: this is no longer just defensive -- a
            # concurrent kill-switch (resuming->orphaned) or
            # stop/emergency-stop (resuming->stopped) issued WHILE the
            # scan above was running lands here as the expected outcome,
            # not an edge case. _reject's own CAS is a safe no-op in that
            # case (the row is already wherever the pre-emption left it).
            await _reject("resume_state_lost")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="resume_state_lost",
            )
        await db.commit()
    except asyncio.CancelledError:
        # WP18b-S-04: asyncio.CancelledError is a BaseException, not an
        # Exception -- it is NOT caught by the `except Exception:` branch
        # below, so without this handler a cancelled request (client
        # disconnect, worker shutdown) used to leave the row stuck at
        # 'resuming' with no audit row at all. The revert is fenced (S-01)
        # and runs in a FRESH session under `asyncio.shield` -- `db` itself
        # may be mid-teardown as part of the same cancellation, and a bare
        # `await` here (unshielded) could be cancelled again before the
        # revert completes.
        async def _shielded_cancelled_revert() -> None:
            from api.db.session import get_session_factory

            try:
                factory = get_session_factory()
                async with factory() as fresh_db:
                    await fresh_db.execute(
                        update(RunORM)
                        .where(
                            RunORM.id == run_id,
                            RunORM.status == "resuming",
                            RunORM.config["resume_attempt_id"].astext == fence,
                        )
                        .values(status="orphaned", updated_at=datetime.now(tz=UTC))
                    )
                    await record_audit_event(
                        fresh_db,
                        event_type="run_resume_rejected",
                        resource_type="run",
                        resource_id=str(run_id),
                        request=request,
                        payload={"reason": "resume_cancelled", "mode": mode},
                    )
                    await fresh_db.commit()
            except Exception:
                log.exception("runs.resume_cancelled_revert_failed")

        await asyncio.shield(_shielded_cancelled_revert())
        raise
    except HTTPException:
        # Already handled: the branch that raised it already performed its
        # own explicit rollback-to-orphaned + audit row above.
        raise
    except Exception:
        # WP1.8a-round2 (C-04/S-06): any OTHER exception on this path (a
        # DB error, an unexpected bug in payload construction, etc.) must
        # never leave the row silently stuck at 'resuming' -- revert and
        # audit here too, exactly like the explicit rejection paths above.
        log.exception("runs.resume_unexpected_error")
        try:
            await _reject("resume_internal_error")
        except Exception:
            log.exception("runs.resume_reject_after_error_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="resume_internal_error",
        ) from None

    # -----------------------------------------------------------------
    # WP1.8a-round2 (C-01 item 2): the commit above is the single
    # linearisation point a concurrent stop_run/emergency_stop_run/
    # kill-switch waits on (they all take their own row lock).  Everything
    # from here down uses ONLY values already resolved above (no
    # ``await db.refresh(run)``) and the two statements that matter --
    # ``asyncio.create_task`` and the ``_RUN_TASKS`` registration -- are
    # consecutive with no ``await`` between them and the commit, so no
    # concurrent request can observe status='running' with no task
    # registered yet (the race round-1 shipped with).
    # -----------------------------------------------------------------
    run.status = "running"
    run.updated_at = final_now
    run.config = updated_config

    try:
        task = asyncio.create_task(
            _run_live_engine(
                run_id_str=str(run_id),
                strategy_cls=strategy_cls,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                symbols=symbols,
                timeframe=timeframe,
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_pct,
                bracket_config=bracket_config,
                enable_adaptive_learning=False,
                resume=snapshot,
                protective_mode=(mode == "protective"),
                elapsed_seconds=elapsed_seconds,
                entries_latch_reason=run.entries_latch_reason,
                allow_pyramiding=resolved_allow_pyramiding,
                exit_config_waived=exit_config_waived_payload is not None,
            ),
            name=f"live-engine-resumed-{run_id}",
        )
        _RUN_TASKS[str(run_id)] = task
    except Exception as exc:
        # WP1.8a-round2 (C-03/S-03): the row already committed 'running' --
        # a failure constructing/spawning the task (e.g. a bad kwarg) must
        # not leave a 'running' row with no task ever backing it.  Revert
        # using the SAME session (already committed, free to start a new
        # transaction) since this is a fresh failure, not part of the
        # 'resuming' rollback above.
        log.exception("runs.resume_spawn_failed")
        await db.execute(
            update(RunORM)
            .where(RunORM.id == run_id, RunORM.status == "running")
            .values(status="orphaned", updated_at=datetime.now(tz=UTC))
        )
        await record_audit_event(
            db,
            event_type="run_resume_rejected",
            resource_type="run",
            resource_id=str(run_id),
            request=request,
            payload={"reason": "spawn_failed", "mode": mode},
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="spawn_failed",
        ) from exc

    log.info("runs.resumed", run_id=str(run_id), mode=mode)
    response = _run_orm_to_detail_response(run)
    if mode == "protective":
        response = response.model_copy(
            update={
                "exit_config_waived": (
                    ExitConfigWaivedResponse(**exit_config_waived_payload)
                    if exit_config_waived_payload is not None
                    else None
                ),
                "exit_manager_missing": exit_manager_missing,
            }
        )
    return response


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
# Startup helper: recover orphaned paper/live runs (WP1.8a, supersedes
# Sprint 24's copy-to-a-new-run-id recovery chain)
# ---------------------------------------------------------------------------
#
# WP1.8a changes boot-time recovery from "copy to a new run_id" (Sprint 24)
# to two mode-specific strategies (synthesis spec S5/S6):
#
# - LIVE: a 'running' (hard-kill -- the task never reached its own
#   `finally` block to write 'orphaned') or already-'orphaned' (graceful
#   shutdown, see run_orchestrator.py) run is left/transitioned to
#   'orphaned' with NO engine task started (O1) -- only a successful
#   POST /runs/{id}/resume starts one.
# - PAPER: a 'running' or 'orphaned' run is rebuilt IN PLACE under the SAME
#   run_id from persisted fill history (`PortfolioAccounting.from_fills`)
#   and its engine task restarts immediately.  `config["resume_count"]`
#   counts every attempt; a 4th (i.e. count > 3) marks the run 'error'
#   instead of retrying again, and a fill/order integrity failure
#   (`check_fill_integrity`, O10/R-06 -- e.g. the 30s flush window lost a
#   fill) also marks 'error'.
#
# The Sprint 24 `recovered_from_run_id IS NULL` filter is dropped entirely
# (S6) -- both modes now key off `status`, not lineage.


_MAX_PAPER_RESUME_COUNT = 3


async def _mark_run_error(
    factory: Any,
    run_id: uuid.UUID,
    log: Any,
    *,
    reason: str,
) -> None:
    """Mark a non-terminal ('running' or 'orphaned') run as 'error'.

    Used by both the live and paper boot-recovery paths when a run cannot
    be safely orphaned/rebuilt (unknown strategy, exhausted resume budget,
    corrupt fill history, ...).  Idempotent -- a no-op if the run has
    already left the non-terminal set.
    """
    try:
        async with factory() as session:
            result = await session.execute(select(RunORM).where(RunORM.id == run_id))
            stale = result.scalar_one_or_none()
            if stale is not None and stale.status in ("running", "orphaned"):
                now = datetime.now(tz=UTC)
                stale.status = "error"
                stale.stopped_at = now
                stale.updated_at = now
                await session.commit()
                log.warning("recovery.run_marked_error", run_id=str(run_id), reason=reason)
    except Exception:
        log.exception("recovery.mark_error_failed", run_id=str(run_id))


# Backwards-compat alias -- Sprint 24 tests / callers referenced this name.
_mark_orphan_error = _mark_run_error


async def _orphan_live_run(factory: Any, run_id: uuid.UUID, log: Any) -> bool:  # noqa: ANN401
    """Transition a live run to 'orphaned' with an audit row (S5).

    Conditional on the row still being 'running' -- if it is already
    'orphaned' (a prior graceful shutdown that nobody has resumed yet),
    this is a no-op and returns False so the caller does not spam a
    duplicate audit row on every subsequent boot.

    Returns
    -------
    bool
        True if this call performed the running -> orphaned transition.
    """
    from api.services.audit_log import record_audit_event

    try:
        async with factory() as session:
            result = await session.execute(select(RunORM).where(RunORM.id == run_id))
            run = result.scalar_one_or_none()
            # WP1.8a-round2 (S-04/C-06): a hard kill mid-resume leaves the
            # row 'resuming', not 'running' -- it never reached the final
            # CAS.  That row is just as orphaned (no task, no exchange
            # session) as a plain 'running' row killed outright, so it
            # must be covered here too, not just by the boot query.
            if run is None or run.status not in ("running", "resuming"):
                return False
            previous_status = run.status
            now = datetime.now(tz=UTC)
            run.status = "orphaned"
            run.updated_at = now
            await record_audit_event(
                session,
                event_type="run_orphaned",
                resource_type="run",
                resource_id=str(run_id),
                request=None,
                payload={
                    "trigger": "boot",
                    "run_mode": "live",
                    "previous_status": previous_status,
                },
            )
            await session.commit()
            return True
    except Exception:
        log.exception("recovery.orphan_live_run_failed", run_id=str(run_id))
        return False


async def recover_orphaned_runs() -> int:
    """Boot-time orphan handling for paper/live runs (WP1.8a S5/S6).

    Queries every run with ``run_mode IN ('paper', 'live')`` and
    ``status IN ('running', 'orphaned', 'resuming')`` -- 'running' catches
    a hard kill (the engine task never reached its own teardown),
    'orphaned' catches a graceful shutdown nobody has resumed yet, and
    'resuming' (WP1.8a-round2 S-04/C-06) catches a hard kill mid-resume
    that never reached its own final CAS.

    - Live runs are left/transitioned to 'orphaned' with no task started
      (O1); an operator must call ``POST /runs/{id}/resume``.
    - Paper runs are rebuilt in place under the SAME run_id from persisted
      fills and their task restarts immediately, subject to a bounded
      resume-count budget and a fill/order integrity check.

    Each run is processed in its own try/except so one bad row never blocks
    the rest; the whole function is wrapped so a DB error at startup never
    crashes the API process.

    Returns
    -------
    int
        Number of paper runs successfully rebuilt and restarted.  Live
        orphaning is not counted here (no task is started for it -- O1).
    """
    from api.db.models import RunORM
    from api.db.session import get_session_factory

    log = logger.bind(component="recovery")
    recovered_count = 0

    try:
        factory = get_session_factory()

        async with factory() as session:
            result = await session.execute(
                select(RunORM).where(
                    # WP1.8a-round2 (S-04/C-06): 'resuming' is included so a
                    # hard kill mid-resume (never reached the final CAS) is
                    # picked back up at the next boot instead of sitting
                    # invisibly stuck forever.
                    RunORM.status.in_(["running", "orphaned", "resuming"]),
                    RunORM.run_mode.in_(["paper", "live"]),
                )
            )
            candidates = list(result.scalars().all())

        if not candidates:
            log.debug("recovery.no_orphans_found")
            return 0

        log.info("recovery.orphans_found", count=len(candidates))

        for candidate in candidates:
            run_id = candidate.id
            run_id_str = str(run_id)
            run_config = dict(candidate.config or {})

            # -------------------------------------------------------------
            # LIVE: orphan it (or leave it orphaned) and move on -- never
            # start a task (O1).
            # -------------------------------------------------------------
            if candidate.run_mode == "live":
                try:
                    transitioned = await _orphan_live_run(factory, run_id, log)
                    if transitioned:
                        log.error(
                            "recovery.run_orphaned",
                            run_id=run_id_str,
                            strategy=run_config.get("strategy_name"),
                        )
                    else:
                        log.debug("recovery.live_already_orphaned", run_id=run_id_str)
                except Exception:
                    log.exception("recovery.run_failed", run_id=run_id_str)
                # WP1.3a (SY-13a-19): live boot recovery NEVER validates or
                # crashes (orphan-only) -- but a config the operator would
                # otherwise have to discover the hard way, at resume time,
                # is worth surfacing now.  Best-effort: never let this
                # affect the orphan transition above.
                try:
                    _live_strategy_name = str(
                        run_config.get("strategy_name", "")
                    ).lower().replace("-", "_")
                    _live_strategy_cls = _get_strategy_registry().get(_live_strategy_name)
                    if _live_strategy_cls is not None:
                        validate_run_exit_config(
                            _live_strategy_cls,
                            bracket=run_config.get("bracket_config") or {},
                            trailing_stop_pct=(
                                run_config.get("strategy_params") or {}
                            ).get("trailing_stop_pct"),
                            mode=RunMode.LIVE,
                            allow_pyramiding=run_config.get("allow_pyramiding"),
                            timeframe=str(run_config.get("timeframe", "1h")),
                        )
                except ExitConfigError as exc:
                    log.error(
                        "recovery.live_exit_config_invalid",
                        run_id=run_id_str,
                        code=exc.code,
                        reasons=[i.reason for i in exc.issues],
                    )
                except Exception:
                    log.debug("recovery.live_exit_config_check_failed", run_id=run_id_str)
                continue

            # -------------------------------------------------------------
            # PAPER: rebuild in place under the same run_id.
            # -------------------------------------------------------------
            try:
                strategy_name: str | None = run_config.get("strategy_name")
                symbols: list[str] = run_config.get("symbols") or []
                timeframe_str: str = run_config.get("timeframe", "1h")
                initial_capital: str = run_config.get("initial_capital", "10000")
                strategy_params: dict[str, Any] = run_config.get("strategy_params") or {}
                bracket_config: dict[str, object] = run_config.get("bracket_config") or {}

                if not strategy_name:
                    await _mark_run_error(factory, run_id, log, reason="missing_strategy_name")
                    continue

                registry = _get_strategy_registry()
                strategy_cls = registry.get(strategy_name)
                if strategy_cls is None:
                    await _mark_run_error(factory, run_id, log, reason="unknown_strategy")
                    continue

                if not symbols:
                    await _mark_run_error(factory, run_id, log, reason="empty_symbols")
                    continue

                try:
                    timeframe = TimeFrame(timeframe_str)
                except ValueError:
                    await _mark_run_error(factory, run_id, log, reason="invalid_timeframe")
                    continue

                normalized_strategy_name = strategy_name.lower().replace("-", "_")
                if not is_mode_allowed(normalized_strategy_name, RunMode.PAPER):
                    await _mark_run_error(factory, run_id, log, reason="strategy_mode_not_allowed")
                    continue

                resume_count = int(run_config.get("resume_count", 0)) + 1
                if resume_count > _MAX_PAPER_RESUME_COUNT:
                    await _mark_run_error(factory, run_id, log, reason="resume_count_exceeded")
                    continue

                # WP1.8a-round2 (S-06/C-04): a snapshot-load failure (a
                # corrupt row that cannot even be parsed back into the pure
                # Order/Fill models -- see run_persistence.load_resume_snapshot)
                # must fail this candidate closed, exactly like a
                # check_fill_integrity rejection below, never propagate up
                # into the outer ``except Exception`` (which would log it
                # but leave the row silently stuck at running/orphaned).
                try:
                    async with factory() as session:
                        snapshot = await _load_resume_snapshot(session, candidate)
                except ResumeRejected as exc:
                    await _mark_run_error(factory, run_id, log, reason=exc.reason)
                    continue
                except Exception:
                    log.exception("recovery.snapshot_load_failed", run_id=run_id_str)
                    await _mark_run_error(factory, run_id, log, reason="snapshot_load_failed")
                    continue
                try:
                    check_fill_integrity(snapshot.fills, snapshot.orders, symbols=set(symbols))
                except ResumeRejected as exc:
                    await _mark_run_error(factory, run_id, log, reason=exc.reason)
                    continue

                # WP1.3a (SY-13a-19, fixes A-10/R-11): validate the exit
                # config BEFORE the running commit below -- previously the
                # trailing_stop_pct float() parse ran AFTER the commit, so a
                # non-numeric legacy value left a ghost 'running' row with
                # no task (the commit had already happened; the exception
                # was only logged by the outer except-block).  A failure
                # here marks the run 'error' and moves on to the next
                # candidate, never touching 'running'.
                try:
                    _recovery_verdict = validate_run_exit_config(
                        strategy_cls,
                        bracket=bracket_config,
                        trailing_stop_pct=strategy_params.get("trailing_stop_pct"),
                        mode=RunMode.PAPER,
                        allow_pyramiding=run_config.get("allow_pyramiding"),
                        timeframe=timeframe_str,
                    )
                except ExitConfigError as exc:
                    log.warning(
                        "runs.recovery_exit_config_invalid",
                        run_id=run_id_str,
                        code=exc.code,
                        reasons=[i.reason for i in exc.issues],
                    )
                    await _mark_run_error(factory, run_id, log, reason=exc.code)
                    continue
                bracket_config = dict(_recovery_verdict.config.bracket)
                trailing_pct: float | None = _recovery_verdict.config.trailing_stop_pct
                resolved_allow_pyramiding: bool = _recovery_verdict.allow_pyramiding

                async with factory() as session:
                    result2 = await session.execute(select(RunORM).where(RunORM.id == run_id))
                    stale = result2.scalar_one_or_none()
                    # WP1.8a-round2 (S-04/C-06): also accept 'resuming' here --
                    # a paper row can only reach this rebuild path via the
                    # boot query above, which now includes it too.
                    if stale is None or stale.status not in ("running", "orphaned", "resuming"):
                        log.debug("recovery.orphan_already_handled", run_id=run_id_str)
                        continue

                    now = datetime.now(tz=UTC)
                    updated_config = dict(stale.config or {})
                    updated_config["resume_count"] = resume_count
                    # WP1.3a (I8): a missing legacy allow_pyramiding is
                    # resolved and persisted (SY-13a-19).
                    updated_config["bracket_config"] = bracket_config
                    updated_config["allow_pyramiding"] = resolved_allow_pyramiding
                    stale.config = updated_config
                    stale.status = "running"
                    stale.updated_at = now
                    started_at = stale.started_at
                    # WP1.7a (SY-02): carry the persisted per-run latch
                    # (only ever 'flatten_incomplete') across the rebuild
                    # -- captured before commit expires the ORM instance.
                    entries_latch_reason = stale.entries_latch_reason
                    await session.commit()

                elapsed_seconds = max((datetime.now(tz=UTC) - started_at).total_seconds(), 0.0)

                task = asyncio.create_task(
                    _run_paper_engine(
                        run_id_str=run_id_str,
                        strategy_cls=strategy_cls,
                        strategy_name=strategy_name,
                        strategy_params=strategy_params,
                        symbols=symbols,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        trailing_stop_pct=trailing_pct,
                        bracket_config=bracket_config,
                        auto_retry_attempt=int(run_config.get("auto_retry_attempt", 0)),
                        resume=snapshot,
                        elapsed_seconds=elapsed_seconds,
                        entries_latch_reason=entries_latch_reason,
                        allow_pyramiding=resolved_allow_pyramiding,
                    ),
                    name=f"recovery-paper-{run_id_str[:8]}",
                )
                _RUN_TASKS[run_id_str] = task

                log.info(
                    "recovery.paper_resumed",
                    run_id=run_id_str,
                    resume_count=resume_count,
                )
                recovered_count += 1

            except Exception:
                log.exception("recovery.run_failed", run_id=run_id_str)
                # Continue to the next candidate -- one bad run must not
                # block the others.

    except Exception:
        log.exception("recovery.fatal_error")

    return recovered_count
