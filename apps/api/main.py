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

The ``lifespan`` context manager is the recommended FastAPI pattern for
startup/shutdown logic (replaces deprecated on_event handlers).

Authentication
--------------
When ``require_api_auth=True`` in settings, all /api/v1/* endpoints
(except /health and /api/v1/metrics) require a valid API key via the
``X-API-Key`` header or ``?api_key=`` query parameter.
See ``api.auth`` for implementation details.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
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

__all__ = ["app", "create_app"]

logger = structlog.get_logger(__name__)


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
    4. (Future) Initialise Redis connection
    5. (Future) Warm up CCXT exchange connection

    Shutdown (after yield)
    ----------------------
    1. (Future) Gracefully stop any running trading loops
    2. (Future) Flush pending fills / position state to database
    3. Close database engine (dispose connection pool)
    4. (Future) Close Redis connection
    5. Log clean shutdown
    """
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
    )

    # Store boot timestamp on app state so /health can report uptime.
    app.state.started_at = time.monotonic()

    # ------------------------------------------------------------------
    # 3. Initialise async database engine and connection pool
    # ------------------------------------------------------------------
    from api.db.session import get_engine

    engine = get_engine()
    app.state.db_engine = engine
    log.info(
        "db.engine_initialised",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )

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

    # Close all pooled database connections gracefully
    from api.db.session import dispose_engine

    await dispose_engine()

    # Future: await app.state.redis.aclose()

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

    Measures wall-clock time from request receipt to response completion.
    Emits a structured ``http.request`` log entry after each response,
    including the HTTP method, path, status code, and duration in milliseconds.

    Health-check and metrics probe paths are logged at DEBUG level to
    prevent high-frequency polling from flooding production log streams.

    Parameters
    ----------
    request:
        Incoming HTTP request.
    call_next:
        Next ASGI middleware / handler callable.

    Returns
    -------
    Response
        The response with the X-Process-Time header added.
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
    """
    Application factory.

    Separating app creation from module-level instantiation makes the
    application testable — test fixtures call ``create_app()`` and get
    an isolated instance without side effects.

    Returns
    -------
    FastAPI:
        Fully configured application ready for ``uvicorn``.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI Crypto Trading Bot API — "
            "backtesting, paper trading, and live trading via CCXT"
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware — order matters: added last runs outermost (first to execute)
    # ------------------------------------------------------------------

    # 1. CORS — must be outermost to handle preflight OPTIONS requests before
    #    auth/timing middleware process them.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "Authorization", "X-API-Key", "X-Requested-With"],
        expose_headers=["X-Process-Time"],
    )

    # 2. Request timing + logging — wraps every request/response cycle
    application.middleware("http")(_request_timing_middleware)

    # 3. Rate limiting — per-IP tiered limits (SEC-S2-001)
    #    Registered after timing middleware, so it runs outermost (Starlette
    #    reverse order). Rate-limited 429s are rejected cheaply without
    #    timing overhead; the rate_limit.exceeded log event provides
    #    security observability instead.
    setup_rate_limiting(application)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    _register_routes(application)

    # ------------------------------------------------------------------
    # Prometheus scrape endpoint — controlled by PROMETHEUS_ENABLED
    # ------------------------------------------------------------------
    setup_prometheus(application)

    return application


def _register_routes(application: FastAPI) -> None:
    """
    Mount all API routers onto the application.

    All versioned endpoints live under the ``/api/v1`` prefix.
    The /health endpoint is mounted directly (no versioning — it is used
    by Docker health checks and load balancers that should not need version
    awareness).

    Authentication
    ^^^^^^^^^^^^^^
    - ``/health`` and ``/api/v1/metrics`` are **always public** (no auth).
    - All other ``/api/v1/*`` routers use the ``require_api_key`` dependency
      which enforces API key validation when ``require_api_auth=True``.
    """
    # ------------------------------------------------------------------
    # Observability — no version prefix, no auth
    # ------------------------------------------------------------------
    @application.get(
        "/health",
        tags=["observability"],
        summary="Service health check",
        response_class=JSONResponse,
    )
    async def health() -> dict[str, Any]:
        """
        Return service health status.

        Used by Docker health checks, load balancers, and monitoring.
        Responds 200 OK when the service is ready to accept requests.
        Always public — no authentication required.

        Future: include database connectivity check, Redis ping, and
        exchange reachability in the response body.
        """
        from datetime import UTC, datetime as dt

        uptime_seconds = round(time.monotonic() - application.state.started_at, 2)
        return {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "version": get_settings().app_version,
            "timestamp": dt.now(tz=UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # In-memory metrics snapshot — always public (for monitoring/Prometheus)
    # ------------------------------------------------------------------
    @application.get(
        "/api/v1/metrics",
        tags=["observability"],
        summary="In-memory metrics snapshot",
        response_class=JSONResponse,
    )
    async def get_metrics() -> dict[str, Any]:
        """
        Return a point-in-time snapshot of all in-memory trading metrics.

        Includes:

        - **counters** — bars processed, signals generated, orders submitted,
          fills executed (with optional per-strategy / per-side label breakdowns).
        - **gauges** — portfolio equity, drawdown percentage, active positions.
        - **histograms** — bar processing duration summary statistics.

        All values reflect the state since last process restart.
        Always public — no authentication required (monitoring endpoint).

        Sprint 2 plan: complement or replace with a Prometheus ``/metrics``
        scrape endpoint for Grafana integration.
        """
        from datetime import UTC, datetime as dt

        return {
            "metrics": metrics.get_all(),
            "timestamp": dt.now(tz=UTC).isoformat(),
        }

    # ------------------------------------------------------------------
    # API v1 routers — protected by API key auth when enabled
    # ------------------------------------------------------------------
    from api.routers import orders, portfolio, runs, strategies

    _V1 = "/api/v1"

    # Run management
    application.include_router(
        runs.router,
        prefix=_V1,
        tags=["runs"],
        dependencies=[Depends(require_api_key)],
    )

    # Order and fill queries
    application.include_router(
        orders.router,
        prefix=_V1,
        tags=["orders"],
        dependencies=[Depends(require_api_key)],
    )

    # Portfolio: summary, equity curve, trades, positions
    application.include_router(
        portfolio.router,
        prefix=_V1,
        tags=["portfolio"],
        dependencies=[Depends(require_api_key)],
    )

    # Strategy discovery
    application.include_router(
        strategies.router,
        prefix=_V1,
        tags=["strategies"],
        dependencies=[Depends(require_api_key)],
    )


# ---------------------------------------------------------------------------
# Module-level app instance — used by uvicorn and test client
# ---------------------------------------------------------------------------
app = create_app()
