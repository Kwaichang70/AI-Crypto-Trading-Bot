"""
apps/api/main.py
----------------
FastAPI application entry point.

Responsibilities
----------------
- Bootstrap structlog on startup
- Initialise and tear down the async database engine
- Mount CORS middleware with origins from settings
- Register API routers (to be added as routes are implemented)
- Expose the /health endpoint

The ``lifespan`` context manager is the recommended FastAPI pattern for
startup/shutdown logic (replaces deprecated on_event handlers).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from common.config import configure_structlog

__all__ = ["app", "create_app"]

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """
    Application lifespan manager.

    Startup (before yield)
    ----------------------
    1. Configure structlog with settings-driven log level and format
    2. Log application boot parameters
    3. (Future) Initialise SQLAlchemy async engine and connection pool
    4. (Future) Initialise Redis connection
    5. (Future) Warm up CCXT exchange connection

    Shutdown (after yield)
    ----------------------
    1. (Future) Gracefully stop any running trading loops
    2. (Future) Flush pending fills / position state to database
    3. (Future) Close database engine
    4. (Future) Close Redis connection
    5. Log clean shutdown
    """
    settings = get_settings()

    # 1. Configure structured logging
    configure_structlog(
        log_level=settings.log_level,
        json_logs=not settings.debug,
    )

    log = structlog.get_logger(__name__)
    log.info(
        "api.starting",
        app=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        exchange=settings.exchange_id,
        live_trading_enabled=settings.enable_live_trading,
    )

    # Store boot timestamp on app state so /health can report uptime.
    app.state.started_at = time.monotonic()

    # ------------------------------------------------------------------
    # Future: database engine init goes here
    # engine = create_async_engine(settings.database_url, ...)
    # app.state.db_engine = engine
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Future: Redis connection init goes here
    # ------------------------------------------------------------------

    yield

    # ------------------------------------------------------------------
    # Shutdown sequence
    # ------------------------------------------------------------------
    log.info("api.shutting_down")

    # Future: await app.state.db_engine.dispose()
    # Future: await app.state.redis.aclose()


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
    # Middleware
    # ------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    _register_routes(application)

    return application


def _register_routes(application: FastAPI) -> None:
    """
    Mount all API routers onto the application.

    Routers are added here as they are implemented. This function is the
    single place to see the complete URL surface of the API.
    """
    # Health check — always available, no auth required
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

    # Future routers — uncomment as implemented:
    # from api.routers import runs, metrics
    # application.include_router(runs.router, prefix="/runs", tags=["runs"])
    # application.include_router(metrics.router, prefix="/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Module-level app instance — used by uvicorn and test client
# ---------------------------------------------------------------------------
app = create_app()
