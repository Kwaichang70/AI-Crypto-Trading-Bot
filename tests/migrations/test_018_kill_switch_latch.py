"""
tests/migrations/test_018_kill_switch_latch.py
--------------------------------------------------
Real-Postgres round-trip test for migration 018 (WP1.7a): upgrade ->
downgrade -> upgrade, covering the new ``kill_switch_state`` singleton
table, the two ``runs.entries_latch_*`` columns, and the three new
``audit_events.event_type`` values.

This test MUST run against a real PostgreSQL instance -- not mocked, not
SQLite. It reads the scratch-DB DSN from ``MIGRATION_TEST_DATABASE_URL``
(``postgresql+asyncpg://user:pass@host:port/dbname``); when unset the
test SKIPS (local-dev fallback only -- CI/the producer run MUST set it).

Setup performed OUTSIDE this test (see the producer report):
    pg_ctlcluster 16 main start
    createuser wp17a_test --pwprompt
    createdb wp17a_migration_test -O wp17a_test
    export MIGRATION_TEST_DATABASE_URL=postgresql+asyncpg://wp17a_test:...@localhost:5432/wp17a_migration_test
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
        return await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table_name)
    finally:
        await conn.close()


async def _fetch_kill_switch_row(dsn: str) -> dict[str, Any] | None:
    import asyncpg

    if not await _table_exists(dsn, "kill_switch_state"):
        return None

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow("SELECT * FROM kill_switch_state WHERE id = 1")
        return dict(row) if row is not None else None
    finally:
        await conn.close()


async def _seed(dsn: str) -> dict[str, uuid.UUID]:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        latched_run_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO runs
                (id, run_mode, status, config, started_at, created_at, updated_at,
                 entries_latch_reason, entries_latched_at)
            VALUES ($1, 'live', 'running', '{}'::jsonb, now(), now(), now(),
                    'flatten_incomplete', now())
            """,
            latched_run_id,
        )

        audit_ids: dict[str, uuid.UUID] = {}
        for event_type in ("kill_switch_cleared", "run_flatten", "entries_latch_cleared"):
            audit_id = uuid.uuid4()
            audit_ids[event_type] = audit_id
            await conn.execute(
                """
                INSERT INTO audit_events
                    (id, actor, event_type, resource_type, resource_id, payload)
                VALUES ($1, 'system', $2, 'run', $3, '{}'::jsonb)
                """,
                audit_id,
                event_type,
                str(latched_run_id),
            )

        return {"latched_run_id": latched_run_id, **audit_ids}
    finally:
        await conn.close()


async def _fetch_run_latch(dsn: str, run_id: uuid.UUID) -> dict[str, Any]:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            "SELECT entries_latch_reason, entries_latched_at FROM runs WHERE id = $1",
            run_id,
        )
        return dict(row)
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


async def _insert_run_flatten_audit_row(dsn: str, run_id: uuid.UUID) -> uuid.UUID:
    """Prove the widened event_type constraint is back in force after re-upgrade."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        audit_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO audit_events (id, actor, event_type, resource_type, resource_id, payload)
            VALUES ($1, 'system', 'run_flatten', 'run', $2, '{}'::jsonb)
            """,
            audit_id,
            str(run_id),
        )
        return audit_id
    finally:
        await conn.close()


def test_018_upgrade_downgrade_upgrade_round_trip() -> None:
    """Migration 018: upgrade -> downgrade -> upgrade against real Postgres.

    Verifies:
    - upgrade to head creates ``kill_switch_state`` seeded with exactly
      one inactive row, adds the two nullable ``runs.entries_latch_*``
      columns, and widens ``ck_audit_events_event_type`` to accept the
      three new WP1.7a event types.
    - downgrade (-1) drops ``kill_switch_state`` and the two ``runs``
      columns, relabels the three new audit event types to
      'emergency_stop' with the original preserved under
      payload['original_event_type'], then narrows the CHECK constraint
      back -- no leftover row violates it.
    - a second upgrade to head succeeds again (genuine round trip, not
      one-way).
    """
    from alembic import command

    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by pytestmark skipif
    dsn = _asyncpg_dsn(database_url)
    cfg = _alembic_config(database_url)

    # --- 1. Upgrade to head (applies every migration 001..018 in order) ---
    command.upgrade(cfg, "head")

    seeded_row = asyncio.run(_fetch_kill_switch_row(dsn))
    assert seeded_row is not None, "018 must seed exactly one kill_switch_state row"
    assert seeded_row["active"] is False
    assert seeded_row["reason"] is None

    ids = asyncio.run(_seed(dsn))

    latch = asyncio.run(_fetch_run_latch(dsn, ids["latched_run_id"]))
    assert latch["entries_latch_reason"] == "flatten_incomplete"
    assert latch["entries_latched_at"] is not None

    # --- 2. Downgrade one revision: 018 -> 017 ---
    command.downgrade(cfg, "-1")

    kill_switch_row_after_downgrade = asyncio.run(_fetch_kill_switch_row(dsn))
    assert kill_switch_row_after_downgrade is None, (
        "downgrade must drop kill_switch_state entirely"
    )

    for event_type in ("kill_switch_cleared", "run_flatten", "entries_latch_cleared"):
        relabelled = asyncio.run(_fetch_audit_event(dsn, ids[event_type]))
        assert relabelled["event_type"] == "emergency_stop", (
            f"{event_type} must be relabelled to 'emergency_stop' on downgrade"
        )
        assert relabelled["payload"]["original_event_type"] == event_type

    # --- 3. Upgrade again: 017 -> 018 -- must succeed a second time ---
    command.upgrade(cfg, "head")

    reseeded_row = asyncio.run(_fetch_kill_switch_row(dsn))
    assert reseeded_row is not None, "the second upgrade must re-seed kill_switch_state"
    assert reseeded_row["active"] is False

    new_audit_id = asyncio.run(
        _insert_run_flatten_audit_row(dsn, ids["latched_run_id"])
    )
    new_event = asyncio.run(_fetch_audit_event(dsn, new_audit_id))
    assert new_event["event_type"] == "run_flatten", (
        "the widened ck_audit_events_event_type constraint must accept "
        "'run_flatten' again after the second upgrade -- a genuine round trip"
    )
