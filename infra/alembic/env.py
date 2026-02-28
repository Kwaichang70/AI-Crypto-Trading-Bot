"""
infra/alembic/env.py
--------------------
Alembic migration environment for the AI Crypto Trading Bot.

This module is invoked by Alembic for every migration command. It:

1. Reads the database URL from the application's Settings object (not from
   alembic.ini) so credentials are never stored in version control.
2. Configures the SQLAlchemy metadata from the ORM models so autogenerate
   can detect schema drift.
3. Supports BOTH sync (offline) and async (online) migration modes.

Async migration pattern:
    Alembic's default runner is synchronous. For async SQLAlchemy engines
    we use ``run_sync`` inside an async context. This is the officially
    recommended approach from the Alembic 1.11+ docs.

    See: https://alembic.sqlalchemy.org/en/latest/cookbook.html
         #using-asyncio-with-alembic

Running migrations:
    # From the infra/ directory:
    alembic upgrade head
    alembic downgrade base
    alembic revision --autogenerate -m "describe_the_change"
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable from migration scripts.
# alembic.ini sets prepend_sys_path = ../.. which resolves to the repo root,
# but we add it explicitly here as a safety net.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import application settings for the database URL.
# This MUST happen after path setup so that `api` is importable.
# ---------------------------------------------------------------------------
from api.config import get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Import Base metadata.
# Importing api.db triggers model registration on Base.metadata.
# All ORM models defined in api.db.models are discovered automatically.
# ---------------------------------------------------------------------------
from api.db import Base  # noqa: E402 — models are registered as a side-effect

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to values in alembic.ini
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers defined in [loggers]/[handlers]/[formatters]
# sections of alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Metadata for autogenerate
# ---------------------------------------------------------------------------
#: The metadata object autogenerate inspects when comparing DB state to models.
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Database URL — injected from application settings at runtime.
# ---------------------------------------------------------------------------

def _get_database_url() -> str:
    """
    Return the async DSN for migrations.

    Reads from the application Settings (which reads from env vars / .env).
    Falls back to the ``sqlalchemy.url`` key in alembic.ini if Settings
    cannot be initialised (e.g., during CI with a test database).
    """
    try:
        settings = get_settings()
        url = settings.database_url
        return url.get_secret_value() if hasattr(url, "get_secret_value") else str(url)
    except Exception:
        # Fallback: read from alembic.ini [alembic] sqlalchemy.url
        url = config.get_main_option("sqlalchemy.url")
        if not url:
            raise RuntimeError(
                "Database URL not configured. "
                "Set DATABASE_URL or POSTGRES_* environment variables, "
                "or set sqlalchemy.url in alembic.ini."
            )
        return url


# ---------------------------------------------------------------------------
# Offline migrations (--sql mode — generates SQL without connecting)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    In this mode, Alembic generates SQL statements without connecting to
    the database. The output can be piped to psql or reviewed before apply.

    This is useful in production deployments where the migration runner
    does not have direct database access and generates a SQL script that
    is reviewed by a DBA before application.

    Usage:
        alembic upgrade head --sql > migration.sql
        psql $DATABASE_URL < migration.sql
    """
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schema comparison options for accurate autogenerate
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (connects to database and applies changes)
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    """
    Execute migrations against an active database connection.

    This function is called from within an async context via
    ``conn.run_sync(do_run_migrations)``.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect column type changes (e.g., String -> Text)
        compare_type=True,
        # Detect server default changes
        compare_server_default=True,
        # Render item-level changes for JSONB / complex columns
        include_schemas=True,
        # Transaction per migration — each revision is its own transaction.
        # This is the safe default: a failed migration is fully rolled back.
        transaction_per_migration=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine and run migrations in an async context.

    Uses NullPool to avoid holding a persistent connection during the
    potentially long migration process. Each Alembic command gets a
    fresh connection and releases it immediately on completion.
    """
    url = _get_database_url()
    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
        # Disable echo during migrations to keep output clean
        echo=False,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Entry point for online migration mode.

    Detects whether an event loop is already running (e.g., in a test
    harness using pytest-asyncio) and uses the appropriate execution
    strategy.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an async context (e.g., pytest-asyncio).
            # Schedule the coroutine as a task on the existing loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, run_async_migrations()
                )
                future.result()
        else:
            loop.run_until_complete(run_async_migrations())
    except RuntimeError:
        # No event loop exists — create a new one.
        asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Alembic entry point — called by the Alembic CLI
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
