"""
tests/migrations/test_019_idempotency_keys.py
--------------------------------------------------
Real-Postgres coverage for migration 019 (WP7.0): the ``idempotency_keys``
table -- ST-30..35 in ``reports/vp2-wp7.0/synthesis-spec.md``.

This test MUST run against a real PostgreSQL instance -- not mocked, not
SQLite (a plain ``CHECK``/FK-violation test still needs real constraint
enforcement to be meaningful). It reads the scratch-DB DSN from
``MIGRATION_TEST_DATABASE_URL`` (``postgresql+asyncpg://user:pass@host:port/dbname``);
when unset the whole module SKIPS.

See ``tests/migrations/test_018_kill_switch_latch.py`` for the shared
setup pattern this file follows.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- this test needs a real "
        "Postgres instance and is skipped in environments without one. "
        "See test_018_kill_switch_latch.py's module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(database_url: str) -> Any:
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    return cfg


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


async def _table_exists(dsn: str, table_name: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return bool(await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table_name))
    finally:
        await conn.close()


async def _index_exists(dsn: str, index_name: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return bool(
            await conn.fetchval(
                "SELECT to_regclass($1) IS NOT NULL", f"public.{index_name}"
            )
        )
    finally:
        await conn.close()


async def _insert_run(dsn: str) -> uuid.UUID:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        run_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'paper', 'running', '{}'::jsonb, now(), now(), now())
            """,
            run_id,
        )
        return run_id
    finally:
        await conn.close()


async def _insert_idempotency_key(
    dsn: str,
    *,
    claimed_run_id: uuid.UUID | None,
    run_id: uuid.UUID | None = None,
    status: str = "in_progress",
) -> uuid.UUID:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        key = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO idempotency_keys
                (key, endpoint, request_fingerprint, status, claimed_run_id, run_id)
            VALUES ($1, 'POST /runs', 'fp', $2, $3, $4)
            """,
            key,
            status,
            claimed_run_id,
            run_id,
        )
        return key
    finally:
        await conn.close()


async def _fetch_run_id_column(dsn: str, key: uuid.UUID) -> uuid.UUID | None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT run_id FROM idempotency_keys WHERE key = $1", key
        )
    finally:
        await conn.close()


async def _row_exists(dsn: str, key: uuid.UUID) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM idempotency_keys WHERE key = $1)", key
            )
        )
    finally:
        await conn.close()


async def _count_idempotency_keys(dsn: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return int(await conn.fetchval("SELECT count(*) FROM idempotency_keys"))
    finally:
        await conn.close()


async def _delete_run(dsn: str, run_id: uuid.UUID) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DELETE FROM runs WHERE id = $1", run_id)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# ST-30: upgrade -> downgrade -> upgrade round trip
# ---------------------------------------------------------------------------
def test_st30_upgrade_downgrade_upgrade_round_trip() -> None:
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by pytestmark skipif
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)

    command.upgrade(cfg, "head")
    assert asyncio.run(_table_exists(dsn, "idempotency_keys")) is True
    assert asyncio.run(_index_exists(dsn, "ix_idempotency_keys_updated_at")) is True

    command.downgrade(cfg, "-1")
    assert asyncio.run(_table_exists(dsn, "idempotency_keys")) is False

    command.upgrade(cfg, "head")
    assert asyncio.run(_table_exists(dsn, "idempotency_keys")) is True
    assert asyncio.run(_index_exists(dsn, "ix_idempotency_keys_updated_at")) is True


# ---------------------------------------------------------------------------
# ST-31: CHECK rejects a fourth status value
# ---------------------------------------------------------------------------
def test_st31_check_constraint_rejects_invalid_status() -> None:
    import asyncpg
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    async def _try_insert() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                INSERT INTO idempotency_keys
                    (key, endpoint, request_fingerprint, status, claimed_run_id)
                VALUES ($1, 'POST /runs', 'fp', 'done', $2)
                """,
                uuid.uuid4(),
                uuid.uuid4(),
            )
        finally:
            await conn.close()

    with pytest.raises(asyncpg.CheckViolationError):
        asyncio.run(_try_insert())


# ---------------------------------------------------------------------------
# ST-32: claimed_run_id has NO FK (nonexistent target succeeds); NULL fails
# ---------------------------------------------------------------------------
def test_st32_claimed_run_id_has_no_fk_but_is_not_null() -> None:
    import asyncpg
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    nonexistent = uuid.uuid4()
    key = asyncio.run(_insert_idempotency_key(dsn, claimed_run_id=nonexistent))
    assert asyncio.run(_row_exists(dsn, key)) is True

    async def _try_insert_null() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                INSERT INTO idempotency_keys
                    (key, endpoint, request_fingerprint, status, claimed_run_id)
                VALUES ($1, 'POST /runs', 'fp', 'in_progress', NULL)
                """,
                uuid.uuid4(),
            )
        finally:
            await conn.close()

    with pytest.raises(asyncpg.NotNullViolationError):
        asyncio.run(_try_insert_null())


# ---------------------------------------------------------------------------
# ST-33: run_id IS a real FK -- nonexistent target is rejected
# ---------------------------------------------------------------------------
def test_st33_run_id_fk_violation_on_nonexistent_run() -> None:
    import asyncpg
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        asyncio.run(
            _insert_idempotency_key(
                dsn,
                claimed_run_id=uuid.uuid4(),
                run_id=uuid.uuid4(),
                status="completed",
            )
        )


# ---------------------------------------------------------------------------
# ST-34: deleting the run SETs NULL on run_id, keeps the row
# ---------------------------------------------------------------------------
def test_st34_deleting_run_sets_run_id_null_keeps_row() -> None:
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    run_id = asyncio.run(_insert_run(dsn))
    key = asyncio.run(
        _insert_idempotency_key(
            dsn, claimed_run_id=run_id, run_id=run_id, status="completed"
        )
    )

    asyncio.run(_delete_run(dsn, run_id))

    assert asyncio.run(_row_exists(dsn, key)) is True
    assert asyncio.run(_fetch_run_id_column(dsn, key)) is None


# ---------------------------------------------------------------------------
# ST-35: CF-B7 teardown (tests/migrations/conftest.py DB-06) leaves the
# table empty for the NEXT test in this module.
# ---------------------------------------------------------------------------
def test_st35a_seed_a_row_for_teardown_check() -> None:
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    asyncio.run(_insert_idempotency_key(dsn, claimed_run_id=uuid.uuid4()))
    assert asyncio.run(_count_idempotency_keys(dsn)) >= 1


def test_st35b_teardown_left_the_table_empty() -> None:
    """Runs strictly after the previous test; the autouse
    ``_migrations_db_teardown`` fixture (tests/migrations/conftest.py,
    DB-06) must have deleted the row seeded there."""
    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    assert asyncio.run(_count_idempotency_keys(dsn)) == 0
