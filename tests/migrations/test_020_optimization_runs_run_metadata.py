"""
tests/migrations/test_020_optimization_runs_run_metadata.py
-------------------------------------------------------------
Coverage for migration 020 (WP2.1b, SY-21-08 / AG-12 / AC-14 / WP21-T-34):
the additive ``optimization_runs.run_metadata JSONB NULL`` column.

Two groups of tests:

1. Static / offline (always run, no database needed): the Alembic revision
   chain (019 -> 020, a single head whose ancestry contains 020, so the
   test keeps passing once later migrations land) read straight from the
   migration scripts, and the exact DDL the migration emits in offline
   (``--sql``) mode for upgrade and downgrade.

2. Real Postgres (SKIPPED unless ``MIGRATION_TEST_DATABASE_URL`` is set,
   same convention as ``test_019_idempotency_keys.py``): upgrade adds a
   nullable ``jsonb`` column with no default, pre-existing rows read back
   NULL (no backfill), JSON round-trips, downgrade removes the column, and
   re-upgrade restores it. The DSN format is
   ``postgresql+asyncpg://user:pass@host:port/dbname``; see
   ``test_018_kill_switch_latch.py`` for the scratch-DB setup.

The round trip downgrades to the explicit revision ``019`` (never a
relative ``-1``), so it stays correct when a later migration sits above 020.

Rollback caveat (documented by migration 020): an image older than 020
cannot start on a DB at 020 (unknown revision); rollback = stop api ->
downgrade to 019 with the new image -> deploy the older image.

Out of scope here: the ORM declaration ``OptimizationRunORM.run_metadata``
and the optimize router write/read (WP21-T-34).
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _REPO_ROOT / "infra" / "alembic"

requires_postgres = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- this test needs a real "
        "Postgres instance and is skipped in environments without one. "
        "See test_018_kill_switch_latch.py's module docstring for scratch-DB setup."
    ),
)


# ---------------------------------------------------------------------------
# Static / offline helpers (no database, no application settings)
# ---------------------------------------------------------------------------
def _script_directory() -> Any:
    """Build a ScriptDirectory straight from the versions folder.

    Deliberately does not go through ``env.py`` (which needs application
    settings); ``ScriptDirectory`` only parses the revision files.
    """
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(_ALEMBIC_DIR))


def _head_and_ancestry() -> tuple[str, set[str]]:
    """Return the single script head and the revisions reachable from it
    (the head itself included). Fails if there is more than one head."""
    script = _script_directory()
    heads = script.get_heads()
    assert len(heads) == 1
    head = heads[0]
    assert script.get_current_head() == head
    return head, {r.revision for r in script.walk_revisions(base="base", head=head)}


def _offline_sql(direction: str) -> str:
    """Run migration 020's ``upgrade``/``downgrade`` against a Postgres
    dialect in offline (as_sql) mode and return the emitted SQL."""
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    module = _script_directory().get_revision("020").module
    buf = io.StringIO()
    ctx = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": buf},
    )
    with Operations.context(ctx):
        getattr(module, direction)()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Static: revision chain
# ---------------------------------------------------------------------------
def test_revision_chain_019_to_020() -> None:
    script = _script_directory()
    rev = script.get_revision("020")
    assert rev is not None
    assert rev.revision == "020"
    assert rev.down_revision == "019"
    assert rev.branch_labels in (None, set(), ())
    assert rev.dependencies in (None, (), [])
    # 019 is the parent and still points at 018 (chain unbroken).
    parent = script.get_revision("019")
    assert parent.down_revision == "018"


def test_single_head_descends_from_020() -> None:
    """Exactly one head, and 020 is in that head's ancestry (or is the head).

    Deliberately not pinned to ``== ["020"]``: a later migration (021, ...)
    becomes the head without having to edit this test.
    """
    _head, ancestry = _head_and_ancestry()
    assert "020" in ancestry


def test_revision_file_name_matches_spec() -> None:
    expected = _ALEMBIC_DIR / "versions" / "020_optimization_runs_run_metadata.py"
    assert expected.is_file()


# ---------------------------------------------------------------------------
# Static: emitted DDL
# ---------------------------------------------------------------------------
def test_upgrade_ddl_adds_nullable_jsonb_without_default() -> None:
    sql = " ".join(_offline_sql("upgrade").split())
    assert "ALTER TABLE optimization_runs ADD COLUMN run_metadata JSONB" in sql
    # Additive and nullable: no NOT NULL, no DEFAULT, no backfill UPDATE.
    add_stmt = sql.split("ALTER TABLE optimization_runs ADD COLUMN", 1)[1]
    assert "NOT NULL" not in add_stmt.upper()
    assert "DEFAULT" not in add_stmt.upper()
    assert "UPDATE" not in sql.upper()
    assert "SET LOCAL lock_timeout = '5s'" in sql


def test_downgrade_ddl_drops_the_column() -> None:
    sql = " ".join(_offline_sql("downgrade").split())
    assert "ALTER TABLE optimization_runs DROP COLUMN run_metadata" in sql
    assert "DROP TABLE" not in sql.upper()


# ---------------------------------------------------------------------------
# Real-Postgres helpers
# ---------------------------------------------------------------------------
def _alembic_config(database_url: str) -> Any:
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_ALEMBIC_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    return cfg


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


async def _column_info(dsn: str) -> dict[str, Any] | None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'optimization_runs'
              AND column_name = 'run_metadata'
            """
        )
        return dict(row) if row is not None else None
    finally:
        await conn.close()


async def _insert_optimization_run(dsn: str, *, with_metadata: bool) -> uuid.UUID:
    """Insert one minimal optimization_runs row. When ``with_metadata`` is
    False the INSERT omits ``run_metadata`` entirely (legacy writer)."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        run_id = uuid.uuid4()
        if with_metadata:
            await conn.execute(
                """
                INSERT INTO optimization_runs
                    (id, strategy_name, symbols, timeframe, rank_by,
                     total_combinations, completed_combinations,
                     failed_combinations, elapsed_seconds, run_metadata)
                VALUES ($1, 'ma_crossover', '["BTC/USD"]'::jsonb, '1h',
                        'sharpe_ratio', 1, 1, 0, 0.5,
                        '{"cost_profile": {"id": "x"}, "engine_semantics": {}}'::jsonb)
                """,
                run_id,
            )
        else:
            await conn.execute(
                """
                INSERT INTO optimization_runs
                    (id, strategy_name, symbols, timeframe, rank_by,
                     total_combinations, completed_combinations,
                     failed_combinations, elapsed_seconds)
                VALUES ($1, 'ma_crossover', '["BTC/USD"]'::jsonb, '1h',
                        'sharpe_ratio', 1, 1, 0, 0.5)
                """,
                run_id,
            )
        return run_id
    finally:
        await conn.close()


async def _fetch_run_metadata_text(dsn: str, run_id: uuid.UUID) -> Any:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval(
            "SELECT run_metadata -> 'cost_profile' ->> 'id' FROM optimization_runs WHERE id = $1",
            run_id,
        )
    finally:
        await conn.close()


async def _run_metadata_is_sql_null(dsn: str, run_id: uuid.UUID) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return bool(
            await conn.fetchval(
                "SELECT run_metadata IS NULL FROM optimization_runs WHERE id = $1",
                run_id,
            )
        )
    finally:
        await conn.close()


async def _delete_optimization_run(dsn: str, run_id: uuid.UUID) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DELETE FROM optimization_runs WHERE id = $1", run_id)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Real Postgres: upgrade -> downgrade -> upgrade round trip
# ---------------------------------------------------------------------------
@requires_postgres
def test_round_trip_adds_and_removes_nullable_jsonb_column() -> None:
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by skipif
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)

    command.upgrade(cfg, "head")
    info = asyncio.run(_column_info(dsn))
    assert info is not None
    assert info["data_type"] == "jsonb"
    assert info["is_nullable"] == "YES"
    assert info["column_default"] is None

    command.downgrade(cfg, "019")
    assert asyncio.run(_column_info(dsn)) is None

    command.upgrade(cfg, "head")
    info = asyncio.run(_column_info(dsn))
    assert info is not None
    assert info["data_type"] == "jsonb"
    assert info["is_nullable"] == "YES"


@requires_postgres
def test_head_revision_applied_matches_script_head_and_includes_020() -> None:
    """After ``upgrade head`` the DB sits at the script head, and 020 is in
    that head's ancestry. Not pinned to ``"020"``, so a later migration does
    not require editing this test."""
    import asyncpg
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    async def _version() -> str:
        conn = await asyncpg.connect(dsn)
        try:
            return str(await conn.fetchval("SELECT version_num FROM alembic_version"))
        finally:
            await conn.close()

    head, ancestry = _head_and_ancestry()
    assert asyncio.run(_version()) == head
    assert "020" in ancestry


# ---------------------------------------------------------------------------
# Real Postgres: no backfill; legacy writers keep working; JSON round-trips
# ---------------------------------------------------------------------------
@requires_postgres
def test_preexisting_row_has_null_run_metadata_after_upgrade() -> None:
    """A row written while the schema is at 019 must read back NULL (legacy)
    after upgrading to 020: no default, no backfill."""
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "019")
    assert asyncio.run(_column_info(dsn)) is None

    run_id = asyncio.run(_insert_optimization_run(dsn, with_metadata=False))
    try:
        command.upgrade(cfg, "head")
        assert asyncio.run(_run_metadata_is_sql_null(dsn, run_id)) is True
    finally:
        asyncio.run(_delete_optimization_run(dsn, run_id))


@requires_postgres
def test_run_metadata_accepts_json_and_omitted_column_stays_null() -> None:
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    with_meta = asyncio.run(_insert_optimization_run(dsn, with_metadata=True))
    without_meta = asyncio.run(_insert_optimization_run(dsn, with_metadata=False))
    try:
        assert asyncio.run(_fetch_run_metadata_text(dsn, with_meta)) == "x"
        assert asyncio.run(_run_metadata_is_sql_null(dsn, without_meta)) is True
    finally:
        asyncio.run(_delete_optimization_run(dsn, with_meta))
        asyncio.run(_delete_optimization_run(dsn, without_meta))
