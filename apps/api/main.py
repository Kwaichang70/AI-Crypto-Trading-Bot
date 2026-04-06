"""
apps/api/main.py
----------------
FastAPI application entry point.

Responsibilities
----------------
- Bootstrap structured logging on startup via ``common.logging.configure_logging``
- Initialise and tear down the async database engine via lifespan
- Mount CORS middleware with origins from settings
- Mount request timing + logging middleware (X-Process-Time header, structured log)
- Register all API routers under /api/v1/ with API key authentication
- Expose the /health endpoint (always available, no auth)
- Expose the /api/v1/metrics endpoint (always available, no auth)
- Expose the /api/v1/signals/current endpoint (always available, no auth — public market data)
- Start/stop RetrainingService when ml_auto_retrain=True (Sprint 23)
- Recover orphaned paper/live runs left running after a crash/restart (Sprint 24)
- Create TelegramNotifier when TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set
- Start CoinGeckoClient, FREDClient, WhaleAlertClient on startup (Sprint 37)

The ``lifespan`` context manager is the recommended FastAPI pattern for
startup/shutdown logic (replaces deprecated on_event handlers).

Authentication
--------------
When ``require_api_auth=True`` in settings, all /api/v1/* endpoints
(except /health, /api/v1/metrics, and /api/v1/signals/current) require a
valid API key via the ``X-API-Key`` header or ``?api_key=`` query parameter.
See ``api.auth`` for implementation details.
"""

from __future__ import annotations

import platform
import time
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Fix aiodns DNS failure on Windows.
# aiodns (pulled in by CCXT) uses c-ares which cannot contact Windows DNS
# servers.  Patching aiohttp's DefaultResolver to ThreadedResolver restores
# working async DNS on Windows while keeping aiodns installed for Linux.
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    try:
        import aiohttp.resolver as _resolver
        from aiohttp.resolver import ThreadedResolver

        _resolver.DefaultResolver = ThreadedResolver
    except ImportError:
        pass
from typing import Any, AsyncIterator, cast

import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.config import get_settings
from api.prometheus import setup_prometheus
from api.rate_limit import setup_rate_limiting
from common.logging import configure_logging
from common.metrics import metrics

__all__ = ["app", "create_app", "get_telegram_notifier"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level RetrainingService instance (None when ml_auto_retrain=False)
# ---------------------------------------------------------------------------
_retraining_service: Any = None

# ---------------------------------------------------------------------------
# Module-level FearGreedClient instance (Sprint 32)
# ---------------------------------------------------------------------------
_fgi_client: Any = None

# ---------------------------------------------------------------------------
# Module-level CoinGeckoClient instance (Sprint 37)
# ---------------------------------------------------------------------------
_coingecko_client: Any = None

# ---------------------------------------------------------------------------
# Module-level FREDClient instance (Sprint 37)
# ---------------------------------------------------------------------------
_fred_client: Any = None

# ---------------------------------------------------------------------------
# Module-level WhaleAlertClient instance (Sprint 37)
# ---------------------------------------------------------------------------
_whale_alert_client: Any = None

# ---------------------------------------------------------------------------
# Module-level equity pruning task
# ---------------------------------------------------------------------------
_equity_prune_task: Any = None

# ---------------------------------------------------------------------------
# Module-level TelegramNotifier instance (None when not configured)
# ---------------------------------------------------------------------------
_telegram_notifier: Any = None


def get_telegram_notifier() -> Any:
    """Return the module-level TelegramNotifier, or None when not configured.

    Used by routers that want to fire-and-forget trade/alert notifications
    without importing the notifier class directly.
    """
    return _telegram_notifier


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Startup (before yield)
    ----------------------
    1. Configure structured logging with service-level context fields
    2. Log application boot parameters
    3. Initialise SQLAlchemy async engine and connection pool
    4. Start RetrainingService if ml_auto_retrain=True (Sprint 23)
    5. Recover orphaned paper/live runs (Sprint 24)
    6. Start Fear & Greed Index client (Sprint 32)
    7. Telegram notifier (optional — requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
    8. Start CoinGecko client (Sprint 37 — always-on, no key required)
    9. Start FRED client (Sprint 37 — requires FRED_API_KEY)
    10. Start Whale Alert client (Sprint 37 — requires WHALE_ALERT_API_KEY)

    Shutdown (after yield)
    ----------------------
    1. Cancel all active paper/live trading engine tasks
    2. Stop RetrainingService if running
    3. Close all external signal clients (FGI, CoinGecko, FRED, Whale Alert)
    4. Close database engine (dispose connection pool)
    5. Log clean shutdown
    """
    global _retraining_service, _telegram_notifier
    global _coingecko_client, _fred_client, _whale_alert_client

    settings = get_settings()

    # 1. Configure structured logging with service-level context
    configure_logging(
        log_level=settings.log_level,
        json_output=not settings.debug,
        service_name=settings.app_name,
        environment="development" if settings.debug else "production",
    )

    log = structlog.get_logger(__name__)
    log.info(
        "api.starting",
        app=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        exchange=settings.exchange_id,
        live_trading_enabled=settings.enable_live_trading,
        api_auth_enabled=settings.require_api_auth,
        ml_auto_retrain=settings.ml_auto_retrain,
    )

    # Store boot timestamp on app state so /health can report uptime.
    app.state.started_at = time.monotonic()

    # ------------------------------------------------------------------
    # 3. Initialise async database engine and connection pool
    # ------------------------------------------------------------------
    from api.db.session import get_engine, get_session_factory

    engine = get_engine()
    app.state.db_engine = engine
    log.info(
        "db.engine_initialised",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )

    # ------------------------------------------------------------------
    # 3b. Recover orphaned paper/live runs (Sprint 24)
    # ------------------------------------------------------------------
    try:
        from api.routers.runs import recover_orphaned_runs

        recovered_count = await recover_orphaned_runs()
        if recovered_count > 0:
            log.info("recovery.completed", recovered_count=recovered_count)
        else:
            log.info("recovery.none_found")
    except Exception:
        log.exception("recovery.startup_error")

    # ------------------------------------------------------------------
    # 4. Start RetrainingService if ml_auto_retrain=True (Sprint 23)
    # ------------------------------------------------------------------
    if settings.ml_auto_retrain:
        from api.routers.ml import set_retraining_service
        from api.services.retraining import RetrainingService

        session_factory = get_session_factory()
        _retraining_service = RetrainingService(
            db_session_factory=session_factory,
            model_dir="models/",
            check_interval_seconds=settings.ml_retrain_interval_minutes * 60,
            min_trades_for_retrain=settings.ml_min_trades_for_retrain,
            min_accuracy_threshold=settings.ml_min_accuracy_threshold,
            max_model_versions=settings.ml_max_model_versions,
            exchange_id=settings.exchange_id,
        )
        set_retraining_service(_retraining_service)
        await _retraining_service.start()
        log.info(
            "retraining_service.wired",
            interval_minutes=settings.ml_retrain_interval_minutes,
            min_trades=settings.ml_min_trades_for_retrain,
        )
    else:
        log.info("retraining_service.disabled", reason="ML_AUTO_RETRAIN=false")

    # ------------------------------------------------------------------
    # 5. Start Fear & Greed Index client (Sprint 32)
    # ------------------------------------------------------------------
    try:
        from data.sentiment import FearGreedClient, set_global_client as _set_fgi

        _fgi_client_instance = FearGreedClient()
        _set_fgi(_fgi_client_instance)
        # Warm up the cache on startup (best-effort; never block startup)
        try:
            await _fgi_client_instance.get_latest()
            log.info("fgi_client.warmed_up")
        except Exception:
            log.warning("fgi_client.warmup_failed", exc_info=True)

        globals()["_fgi_client"] = _fgi_client_instance
        log.info("fgi_client.started")
    except ImportError:
        log.info("fgi_client.skipped", reason="aiohttp not installed")
    except Exception:
        log.warning("fgi_client.startup_error", exc_info=True)

    # ------------------------------------------------------------------
    # 6. Equity snapshot pruning (daily background task)
    # ------------------------------------------------------------------
    import asyncio as _asyncio

    global _equity_prune_task  # noqa: PLW0603

    async def _equity_prune_loop() -> None:
        """Delete equity snapshots older than retention period, once per day."""
        from datetime import UTC as _UTC, datetime as _dt, timedelta as _td

        from api.db.models import EquitySnapshotORM
        from api.db.session import get_session_factory as _get_sf
        from sqlalchemy import delete as _sa_delete

        _sf = _get_sf()

        # CR-008: 60s startup delay to let connection pool settle
        try:
            await _asyncio.sleep(60)
        except _asyncio.CancelledError:
            return

        while True:
            try:
                cutoff = _dt.now(_UTC) - _td(
                    days=settings.equity_snapshot_retention_days
                )
                async with _sf() as session:
                    del_result = await session.execute(
                        _sa_delete(EquitySnapshotORM).where(
                            EquitySnapshotORM.timestamp < cutoff
                        )
                    )
                    deleted: int = del_result.rowcount  # type: ignore[attr-defined]
                    await session.commit()
                    if deleted > 0:
                        log.info(
                            "equity_prune.completed",
                            deleted=deleted,
                            cutoff=str(cutoff),
                            retention_days=settings.equity_snapshot_retention_days,
                        )
            except _asyncio.CancelledError:
                break
            except Exception:
                log.warning("equity_prune.error", exc_info=True)

            try:
                await _asyncio.sleep(86400)
            except _asyncio.CancelledError:
                break

    _equity_prune_task = _asyncio.create_task(
        _equity_prune_loop(), name="equity_prune"
    )
    log.info(
        "equity_prune.scheduled",
        retention_days=settings.equity_snapshot_retention_days,
    )

    # ------------------------------------------------------------------
    # 7. Telegram notifier (optional)
    # ------------------------------------------------------------------
    if settings.telegram_bot_token and settings.telegram_chat_id:
        try:
            from trading.telegram import TelegramNotifier

            _telegram_notifier = TelegramNotifier(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
            )
            log.info("telegram.configured", chat_id=settings.telegram_chat_id)
        except ImportError:
            log.warning("telegram.skipped", reason="trading.telegram not importable")
        except Exception:
            log.warning("telegram.startup_error", exc_info=True)
    else:
        log.info("telegram.disabled", reason="TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")

    # ------------------------------------------------------------------
    # 8. CoinGecko client — always-on, no API key required (Sprint 37)
    # ------------------------------------------------------------------
    try:
        from data.market_signals import CoinGeckoClient, set_global_client as _set_cg

        _cg_instance = CoinGeckoClient()
        _set_cg(_cg_instance)
        # Warm up cache on startup (best-effort; never block startup)
        try:
            await _cg_instance.get_latest()
            log.info("coingecko_client.warmed_up")
        except Exception:
            log.warning("coingecko_client.warmup_failed", exc_info=True)

        _coingecko_client = _cg_instance
        log.info("coingecko_client.started")
    except ImportError:
        log.info("coingecko_client.skipped", reason="aiohttp not installed")
    except Exception:
        log.warning("coingecko_client.startup_error", exc_info=True)

    # ------------------------------------------------------------------
    # 9. FRED macro client — requires FRED_API_KEY (Sprint 37)
    # ------------------------------------------------------------------
    if settings.fred_api_key:
        try:
            from data.macro_data import FREDClient, set_global_client as _set_fred

            _fred_instance = FREDClient(api_key=settings.fred_api_key)
            _set_fred(_fred_instance)
            # Warm up cache on startup (best-effort; FRED data rarely changes)
            try:
                await _fred_instance.get_latest()
                log.info("fred_client.warmed_up")
            except Exception:
                log.warning("fred_client.warmup_failed", exc_info=True)

            _fred_client = _fred_instance
            log.info("fred_client.started")
        except ImportError:
            log.info("fred_client.skipped", reason="aiohttp not installed")
        except Exception:
            log.warning("fred_client.startup_error", exc_info=True)
    else:
        log.info("fred_client.disabled", reason="FRED_API_KEY not set")

    # ------------------------------------------------------------------
    # 10. Whale Alert client — requires WHALE_ALERT_API_KEY (Sprint 37)
    # ------------------------------------------------------------------
    if settings.whale_alert_api_key:
        try:
            from data.whale_tracker import WhaleAlertClient, set_global_client as _set_whale

            _whale_instance = WhaleAlertClient(api_key=settings.whale_alert_api_key)
            _set_whale(_whale_instance)
            # Warm up cache on startup (best-effort)
            try:
                await _whale_instance.get_latest()
                log.info("whale_alert_client.warmed_up")
            except Exception:
                log.warning("whale_alert_client.warmup_failed", exc_info=True)

            _whale_alert_client = _whale_instance
            log.info("whale_alert_client.started")
        except ImportError:
            log.info("whale_alert_client.skipped", reason="aiohttp not installed")
        except Exception:
            log.warning("whale_alert_client.startup_error", exc_info=True)
    else:
        log.info("whale_alert_client.disabled", reason="WHALE_ALERT_API_KEY not set")

    # ------------------------------------------------------------------
    # Future: Redis connection init goes here
    # ------------------------------------------------------------------

    yield

    # ------------------------------------------------------------------
    # Shutdown sequence
    # ------------------------------------------------------------------
    log.info("api.shutting_down")

    # Cancel all active paper/live trading engine tasks
    import asyncio

    from api.routers.runs import _RUN_TASKS

    active_tasks = [t for t in _RUN_TASKS.values() if not t.done()]
    if active_tasks:
        log.info("api.cancelling_engine_tasks", count=len(active_tasks))
        for task in active_tasks:
            task.cancel()
        done, pending = await asyncio.wait(active_tasks, timeout=30.0)
        if pending:
            log.warning(
                "api.engine_tasks_timeout",
                pending_count=len(pending),
            )

    # Stop RetrainingService (Sprint 23)
    if _retraining_service is not None:
        await _retraining_service.stop()
        _retraining_service = None

    # Stop equity snapshot pruning task
    if _equity_prune_task is not None and not _equity_prune_task.done():
        _equity_prune_task.cancel()
        await asyncio.gather(_equity_prune_task, return_exceptions=True)

    # Close FearGreedClient session (Sprint 32)
    if _fgi_client is not None:
        try:
            await _fgi_client.close()
        except Exception:
            pass

    # Close CoinGeckoClient session (Sprint 37)
    if _coingecko_client is not None:
        try:
            await _coingecko_client.close()
        except Exception:
            pass

    # Close FREDClient session (Sprint 37)
    if _fred_client is not None:
        try:
            await _fred_client.close()
        except Exception:
            pass

    # Close WhaleAlertClient session (Sprint 37)
    if _whale_alert_client is not None:
        try:
            await _whale_alert_client.close()
        except Exception:
            pass

    # Close TelegramNotifier session
    if _telegram_notifier is not None:
        try:
            await _telegram_notifier.close()
        except Exception:
            pass

    # Close all pooled database connections gracefully
    from api.db.session import dispose_engine

    await dispose_engine()

    log.info("api.stopped")


# ---------------------------------------------------------------------------
# Request timing + logging middleware
# ---------------------------------------------------------------------------

# Paths that are polled frequently by health checkers and Prometheus scrapers.
# Log at DEBUG to avoid drowning production logs with probe noise.
_PROBE_PATHS: frozenset[str] = frozenset({"/health", "/metrics", "/api/v1/metrics"})


async def _request_timing_middleware(request: Request, call_next: Any) -> Response:
    """
    Add X-Process-Time header and emit a structured HTTP request log entry.
    """
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Process-Time"] = str(elapsed_ms)

    path = request.url.path
    log_fn = logger.debug if path in _PROBE_PATHS else logger.info
    log_fn(
        "http.request",
        method=request.method,
        path=path,
        status_code=response.status_code,
        duration_ms=elapsed_ms,
    )

    return cast(Response, response)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI Crypto Trading Bot API  -- "
            "backtesting, paper trading, and live trading via CCXT"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "Authorization", "X-API-Key", "X-Requested-With"],
        expose_headers=["X-Process-Time"],
    )

    application.middleware("http")(_request_timing_middleware)
    setup_rate_limiting(application)
    _register_routes(application)
    setup_prometheus(application)

    return application


def _register_routes(application: FastAPI) -> None:
    """Mount all API routers onto the application."""

    @application.get(
        "/health",
        tags=["observability"],
        summary="Service health check",
        response_class=JSONResponse,
    )
    async def health() -> dict[str, Any]:
        from datetime import UTC, datetime as dt

        uptime_seconds = round(time.monotonic() - application.state.started_at, 2)
        return {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "version": get_settings().app_version,
            "timestamp": dt.now(tz=UTC).isoformat(),
        }

    @application.get(
        "/api/v1/metrics",
        tags=["observability"],
        summary="In-memory metrics snapshot",
        response_class=JSONResponse,
    )
    async def get_metrics() -> dict[str, Any]:
        from datetime import UTC, datetime as dt

        return {
            "metrics": metrics.get_all(),
            "timestamp": dt.now(tz=UTC).isoformat(),
        }

    from api.routers import circuit_breaker, learning, ml, optimize, orders, portfolio, runs, signals, strategies

    _V1 = "/api/v1"

    application.include_router(
        runs.router,
        prefix=_V1,
        tags=["runs"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        circuit_breaker.router,
        prefix=_V1,
        tags=["circuit-breaker"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        orders.router,
        prefix=_V1,
        tags=["orders"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        portfolio.router,
        prefix=_V1,
        tags=["portfolio"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        portfolio.summary_router,
        prefix=_V1,
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        strategies.router,
        prefix=_V1,
        tags=["strategies"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        ml.router,
        prefix=_V1,
        tags=["ml"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        optimize.router,
        prefix=_V1,
        tags=["optimize"],
        dependencies=[Depends(require_api_key)],
    )

    application.include_router(
        learning.router,
        tags=["learning"],
        dependencies=[Depends(require_api_key)],
    )

    # Signals router — unauthenticated (public read-only market data cache)
    application.include_router(
        signals.router,
        prefix=_V1,
    )


# ---------------------------------------------------------------------------
# Module-level app instance  -- used by uvicorn and test client
# ---------------------------------------------------------------------------
app = create_app()
