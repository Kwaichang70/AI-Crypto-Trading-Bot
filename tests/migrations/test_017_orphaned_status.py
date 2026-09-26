"""
tests/migrations/test_017_orphaned_status.py
------------------------------------------------
Real-Postgres round-trip test for migration 017 (WP1.8a): upgrade ->
downgrade -> upgrade, with seeded ``orphaned``/``resuming`` runs and
``run_orphaned``/``run_resumed``/``run_resume_rejected``/
``resume_orders_imported`` audit rows checked for relabel integrity.

This test MUST run against a real PostgreSQL instance -- it is not
mocked and does not use SQLite.  It reads the scratch-DB DSN from the
``MIGRATION_TEST_DATABASE_URL`` environment variable
(``postgresql+asyncpg://user:pass@host:port/dbname``, matching
``Settings.database_url``'s required scheme).  When that variable is
unset the test SKIPS (so the rest of the suite runs without a live
Postgres dependency) -- CI/the producer run MUST set it and report the
real output; skipping is only a local-dev fallback, never how this test
is meant to be exercised for the acceptance report.

Setup performed OUTSIDE this test (documented in the producer report):
    pg_ctlcluster 16 main start   # or: service postgresql start
    createuser wp18a_test --pwprompt   (or via psql)
    createdb wp18a_migration_test -O wp18a_test
    export MIGRATION_TEST_DATABASE_URL=postgresql+asyncpg://wp18a_test:...@localhost:5432/wp18a_migration_test
"""

from __future__ import annotations

import asyncio
import json
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
        "See the module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(database_url: str) -> Any:
    """Build an Alembic Config pointing at this repo's migration chain,
    with ``DATABASE_URL`` set so ``env.py``'s ``get_settings().database_url``
    resolves to the scratch DB."""
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    return cfg


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    """Strip the '+asyncpg' SQLAlchemy dialect suffix for a raw asyncpg.connect() DSN."""
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


async def _seed(dsn: str) -> dict[str, uuid.UUID]:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        orphaned_run_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'orphaned', '{}'::jsonb, now(), now(), now())
            """,
            orphaned_run_id,
        )

        resuming_run_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'resuming', '{}'::jsonb, now(), now(), now())
            """,
            resuming_run_id,
        )

        run_orphaned_audit_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO audit_events (id, actor, event_type, resource_type, resource_id, payload)
            VALUES ($1, 'system', 'run_orphaned', 'run', $2, '{"trigger": "boot"}'::jsonb)
            """,
            run_orphaned_audit_id,
            str(orphaned_run_id),
        )

        other_audit_ids: dict[str, uuid.UUID] = {}
        for event_type in ("run_resumed", "run_resume_rejected", "resume_orders_imported"):
            audit_id = uuid.uuid4()
            other_audit_ids[event_type] = audit_id
            await conn.execute(
                """
                INSERT INTO audit_events
                    (id, actor, event_type, resource_type, resource_id, payload)
                VALUES ($1, 'system', $2, 'run', $3, '{}'::jsonb)
                """,
                audit_id,
                event_type,
                str(orphaned_run_id),
            )

        return {
            "orphaned_run_id": orphaned_run_id,
            "resuming_run_id": resuming_run_id,
            "run_orphaned_audit_id": run_orphaned_audit_id,
            **other_audit_ids,
        }
    finally:
        await conn.close()


async def _fetch_run_status(dsn: str, run_id: uuid.UUID) -> str:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval("SELECT status FROM runs WHERE id = $1", run_id)
    finally:
        await conn.close()


async def _fetch_audit_event(dsn: str, audit_id: uuid.UUID) -> dict[str, Any]:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            "SELECT event_type, payload FROM audit_events WHERE id = $1", audit_id
        )
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {"event_type": row["event_type"], "payload": payload}
    finally:
        await conn.close()


async def _insert_orphaned_run(dsn: str) -> uuid.UUID:
    """Prove the widened constraint is back in force after the re-upgrade."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        run_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'orphaned', '{}'::jsonb, now(), now(), now())
            """,
            run_id,
        )
        return run_id
    finally:
        await conn.close()


def test_017_upgrade_downgrade_upgrade_round_trip() -> None:
    """Migration 017: upgrade -> downgrade -> upgrade against real Postgres.

    Verifies:
    - upgrade to head accepts 'orphaned'/'resuming' run rows and the four
      new audit event_type values (the widened CHECK constraints).
    - downgrade (-1) relabels 'orphaned'/'resuming' runs to 'error' and
      relabels the four new audit event types to 'emergency_stop', with
      the original event_type preserved in
      payload['original_event_type'] (forensic reversibility), THEN
      successfully narrows both CHECK constraints back (no leftover row
      violates the pre-017 constraint).
    - a second upgrade to head succeeds again (the widened constraints
      accept a fresh 'orphaned' row once more) -- a genuine round trip,
      not just one-way.
    """
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by pytestmark skipif
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)

    # --- 1. Upgrade to head (applies every migration 001..017 in order) ---
    command.upgrade(cfg, "head")

    ids = asyncio.run(_seed(dsn))

    # --- 2. Downgrade one revision: 017 -> 016 ---
    command.downgrade(cfg, "-1")

    orphaned_status = asyncio.run(_fetch_run_status(dsn, ids["orphaned_run_id"]))
    resuming_status = asyncio.run(_fetch_run_status(dsn, ids["resuming_run_id"]))
    assert orphaned_status == "error", (
        f"downgrade must relabel 'orphaned' runs to 'error', got {orphaned_status!r}"
    )
    assert resuming_status == "error", (
        f"downgrade must relabel 'resuming' runs to 'error', got {resuming_status!r}"
    )

    run_orphaned_after_downgrade = asyncio.run(
        _fetch_audit_event(dsn, ids["run_orphaned_audit_id"])
    )
    assert run_orphaned_after_downgrade["event_type"] == "emergency_stop"
    assert run_orphaned_after_downgrade["payload"]["original_event_type"] == "run_orphaned"

    for event_type in ("run_resumed", "run_resume_rejected", "resume_orders_imported"):
        relabelled = asyncio.run(_fetch_audit_event(dsn, ids[event_type]))
        assert relabelled["event_type"] == "emergency_stop", (
            f"{event_type} must be relabelled to 'emergency_stop' on downgrade"
        )
        assert relabelled["payload"]["original_event_type"] == event_type, (
            f"{event_type}'s original type must be preserved in the payload"
        )

    # --- 3. Upgrade again: 016 -> 017 -- must succeed a second time ---
    command.upgrade(cfg, "head")

    new_run_id = asyncio.run(_insert_orphaned_run(dsn))
    new_status = asyncio.run(_fetch_run_status(dsn, new_run_id))
    assert new_status == "orphaned", (
        "the widened ck_runs_status constraint must accept 'orphaned' again "
        "after the second upgrade -- a genuine round trip"
    )
