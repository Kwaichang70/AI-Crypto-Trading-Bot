"""
apps/api/db/session.py
----------------------
Async SQLAlchemy engine and session factory.

Architecture decisions:
- ``create_async_engine`` with ``asyncpg`` dialect for non-blocking I/O.
- NullPool is used for Alembic migrations (synchronous context) while the
  application uses the standard async pool.
- Session factory is configured with ``expire_on_commit=False`` so that
  ORM objects remain usable after ``await session.commit()`` without
  triggering an implicit SELECT. This is the correct pattern for FastAPI
  request-response handlers where the session is closed after the response
  is sent.
- ``get_db()`` is an async generator dependency compatible with FastAPI's
  ``Depends()`` injection. It ensures the session is always closed even
  when an exception propagates through the handler.

Connection pool sizing:
  pool_size + max_overflow = maximum concurrent DB connections.
  Settings are read from AppConfig so they can be tuned per environment
  without code changes.

  Development defaults (from config.py):
    pool_size=10, max_overflow=20  → up to 30 concurrent connections.

  For production, tune based on PostgreSQL's max_connections setting:
    max_connections = pool_size + max_overflow + alembic_headroom (~5).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.config import get_settings

__all__ = [
    "get_engine",
    "get_session_factory",
    "get_db",
    "create_engine_from_url",
    "dispose_engine",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine factory — separated for testability
# ---------------------------------------------------------------------------

def create_engine_from_url(
    database_url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    echo: bool = False,
    **kwargs: Any,
) -> AsyncEngine:
    """
    Construct an AsyncEngine from a DSN string.

    Separated from module-level singleton creation so test fixtures can
    supply a test-database URL without mutating global state.

    Parameters
    ----------
    database_url:
        Must use the ``postgresql+asyncpg://`` scheme.
    pool_size:
        Number of persistent connections in the pool.
    max_overflow:
        Additional connections allowed when the pool is exhausted.
        Total max connections = pool_size + max_overflow.
    pool_timeout:
        Seconds to wait for a free connection before raising ``TimeoutError``.
    echo:
        If True, SQL statements are logged. Enable only in debug mode.
    **kwargs:
        Forwarded to ``create_async_engine``.

    Returns
    -------
    AsyncEngine
        Ready to use; no connections are opened until first use.
    """
    return create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_pre_ping=True,      # Validate connection health before use
        pool_recycle=3600,       # Recycle connections every hour (avoids stale connections)
        echo=echo,
        **kwargs,
    )


def _build_engine() -> AsyncEngine:
    """
    Build the module-level engine from application settings.

    Called once at import time. Settings are read via ``get_settings()``
    which uses lru_cache, so the .env file is parsed exactly once.
    """
    settings = get_settings()
    _engine = create_engine_from_url(
        settings.database_url.get_secret_value(),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        echo=settings.debug,
    )
    logger.debug(
        "db.engine_created",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    return _engine


# ---------------------------------------------------------------------------
# Lazy singletons — created on first access, not at import time
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Return the global async engine, creating it on first call.

    Lazy initialization avoids import-time crashes when DATABASE_URL is
    not configured (e.g., during test collection or Alembic offline mode).
    """
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return the global session factory, creating it on first call.

    Bound to the engine from ``get_engine()``. Configured with
    ``expire_on_commit=False`` so ORM objects remain live after commit
    (correct pattern for async FastAPI handlers).
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,   # Explicit flush control — avoids surprising implicit SELECTs
            autocommit=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    Usage in a router:
    ------------------
    .. code-block:: python

        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from api.db.session import get_db

        @router.get("/runs")
        async def list_runs(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(RunORM))
            return result.scalars().all()

    The session is:
    - Yielded to the handler (and any downstream Depends).
    - Committed if the handler returns without raising.
    - Rolled back if an exception propagates.
    - Closed by the async context manager (no explicit close needed).

    Note: Transaction management (commit/rollback) is the responsibility
    of the service layer. This dependency only provides a clean session.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        # No explicit close() — the async context manager handles it (S1-07)


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

async def dispose_engine() -> None:
    """
    Gracefully close all pooled connections.

    Call from the FastAPI lifespan shutdown handler:

    .. code-block:: python

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            await dispose_engine()

    This ensures clean shutdown — no connection leaks to PostgreSQL.
    """
    engine = get_engine()
    await engine.dispose()
    logger.info("db.engine_disposed")
