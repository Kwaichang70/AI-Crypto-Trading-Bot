"""
apps/api/db/__init__.py
-----------------------
Public surface of the database package.

Import from here rather than reaching into submodules:

    from api.db import Base, engine, AsyncSessionFactory, get_db

Re-exports
----------
Base
    SQLAlchemy ``DeclarativeBase`` — import this in Alembic ``env.py``
    for autogenerate to discover all ORM models.
engine
    Singleton ``AsyncEngine`` bound to the PostgreSQL DSN from settings.
AsyncSessionFactory
    ``async_sessionmaker`` — use directly when a FastAPI Depends() is
    not appropriate (e.g., background tasks, startup jobs).
get_db
    Async generator FastAPI dependency for session-per-request pattern.
dispose_engine
    Coroutine to gracefully drain the connection pool on shutdown.

All ORM model classes are also re-exported so callers can do:

    from api.db import RunORM, OrderORM
"""

from __future__ import annotations

# Import all models first so that the declarative base's metadata is
# populated before Alembic env.py imports ``Base``.
from api.db.models import (  # noqa: F401 — side-effect import registers models
    Base,
    EquitySnapshotORM,
    FillORM,
    OrderORM,
    RunORM,
    SignalORM,
    TradeORM,
)
from api.db.session import (
    AsyncSessionFactory,
    dispose_engine,
    engine,
    get_db,
)

__all__ = [
    # Base & metadata
    "Base",
    # Engine & session
    "engine",
    "AsyncSessionFactory",
    "get_db",
    "dispose_engine",
    # ORM models
    "RunORM",
    "OrderORM",
    "FillORM",
    "TradeORM",
    "EquitySnapshotORM",
    "SignalORM",
]
