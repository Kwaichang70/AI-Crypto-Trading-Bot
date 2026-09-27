"""
tests/migrations/test_wp17b_lifespan_kill_switch.py
----------------------------------------------------
WP1.7b (CF-B6, S-R2-06 suggestion): real-Postgres test that exercises
``api.services.kill_switch.load()`` via the EXACT same code path
``apps/api/main.py``'s lifespan uses at boot (see ``main.py``'s section
"3a. WP1.7a (I4): load the persisted kill-switch latch BEFORE recovering
any orphaned run"):

    from api.services import kill_switch as _kill_switch
    try:
        from api.db.session import get_session_factory as _get_session_factory_ks
        _ks_factory = _get_session_factory_ks()
        async with _ks_factory() as _ks_db:
            await _kill_switch.load(_ks_db)
    except Exception:
        _kill_switch.mark_unknown()

Prior to this file, WP1.7a's own suite (``tests/unit/test_wp17a_*``) only
ever called ``kill_switch.load(db)`` against a mocked/fake ``AsyncSession``
-- never the real ``api.db.session`` module-level engine/session-factory
singletons that ``main.py`` actually builds and passes in, and never
against a real, migrated Postgres ``kill_switch_state`` row. Security
round 2 (WP17a-S-R2-06) flagged this gap; WP1.7a's final synthesis
(``reports/vp2-wp1.7/final-synthesis-1.7a.md`` CF-B6) carried it forward
here.

This MUST run against a real PostgreSQL instance -- SQLite cannot stand
in for asyncpg-specific connection-failure behaviour (case 3 below).
Gated the same way as every other file in this directory: reads
``MIGRATION_TEST_DATABASE_URL`` and SKIPS when unset.

Covers exactly the three CF-B6 scenarios:
    1. An unlatched ``kill_switch_state`` row -> ``is_active() is False``
       and ``is_loaded() is True``.
    2. A latched row -> ``is_active() is True``, ``is_loaded() is True``,
       and the persisted reason/since are reflected in the mirror.
    3. A failed DB read (session factory pointed at a nonexistent
       database, same credentials/host/port) -> ``load()``'s own internal
       ``except Exception`` fires, leaving the mirror unloaded --
       ``is_active() is True`` (fail-closed, I4) with reason
       ``latch_state_unknown`` and ``is_loaded() is False``.

Hygiene: every scenario rebuilds ``api.db.session``'s module-level
engine/session-factory singletons against its OWN dsn (mirroring the
``race_app`` fixture pattern in ``test_wp18a_resume_races.py``) using
``NullPool``, since each scenario runs inside its own ``asyncio.run()``
(hence its own event loop) and asyncpg connections cannot cross loops.
The engine is disposed at the end of each scenario's coroutine, and the
process-wide ``kill_switch`` mirror is reset to a clean, loaded,
un-latched state (:func:`reset_state_for_tests`) both before AND after
every test so no state leaks into another file in this suite. CF-B7's
shared ``tests/migrations/conftest.py`` teardown fixture independently
resets ``kill_switch_state`` in the DB itself after this file runs.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
import structlog

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- this test needs a real "
        "Postgres instance and is skipped in environments without one. "
        "See test_018_kill_switch_latch.py's module docstring for "
        "scratch-DB setup."
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


def _bogus_dsn(sqlalchemy_url: str) -> str:
    """Same scheme/user/pass/host/port, a database name that (almost
    certainly) does not exist -- used to force a genuine connection-time
    failure for scenario 3, without touching the real scratch DB at all."""
    parts = urlsplit(sqlalchemy_url)
    bogus_path = f"/wp17b_nonexistent_{uuid.uuid4().hex}"
    return urlunsplit((parts.scheme, parts.netloc, bogus_path, parts.query, parts.fragment))


async def _set_kill_switch_row(
    dsn: str, *, active: bool, reason: str | None = None
) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        if active:
            await conn.execute(
                """
                UPDATE kill_switch_state
                SET active = TRUE, reason = $1, activated_at = now(),
                    activated_by = 'wp17b-test', cleared_at = NULL,
                    cleared_by = NULL, clear_reason = NULL
                WHERE id = 1
                """,
                reason,
            )
        else:
            await conn.execute(
                """
                UPDATE kill_switch_state
                SET active = FALSE, reason = NULL, cleared_at = now(),
                    cleared_by = 'wp17b-test', clear_reason = 'test reset'
                WHERE id = 1
                """
            )
    finally:
        await conn.close()


async def _boot_load_kill_switch(database_url: str) -> tuple[bool, bool, str | None]:
    """Run the EXACT sequence ``main.py``'s lifespan (section 3a) runs,
    against ``database_url`` -- including its own outer ``except Exception:
    mark_unknown()`` fallback, in case connecting itself (rather than the
    query inside ``load()``) is what raises.

    Returns ``(is_active(), is_loaded(), current_state().reason)`` after
    the sequence completes.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    import api.db.session as session_module
    from api.services import kill_switch as _kill_switch

    # Force a clean, KNOWN starting point so this scenario's assertions
    # prove what `load()` itself did, not leftover state from a prior
    # scenario/module in the same process (mirrors the `race_app` fixture
    # pattern in test_wp18a_resume_races.py).
    _kill_switch.mark_unknown()
    assert _kill_switch.is_loaded() is False

    # The module-level engine/session-factory singletons in api.db.session
    # are lazy but cached forever once built -- rebuild against THIS
    # scenario's dsn, same as every other real-Postgres test in this
    # directory. NullPool: this coroutine's own asyncio.run() event loop
    # is torn down when it returns, and asyncpg connections cannot be
    # reused across loops.
    old_engine = session_module._engine
    if old_engine is not None:
        try:
            await old_engine.dispose()
        except Exception:
            # Best-effort only -- disposing a stale/never-connected engine
            # from a PRIOR scenario's event loop must never block this
            # one's own boot sequence.
            structlog.get_logger(__name__).warning(
                "wp17b_lifespan_test.old_engine_dispose_failed", exc_info=True
            )
    session_module._engine = create_async_engine(database_url, poolclass=NullPool)
    session_module._session_factory = None

    # --- verbatim main.py section 3a (apps/api/main.py:228-244) ---
    try:
        from api.db.session import get_session_factory as _get_session_factory_ks

        _ks_factory = _get_session_factory_ks()
        async with _ks_factory() as _ks_db:
            await _kill_switch.load(_ks_db)
    except Exception:
        _kill_switch.mark_unknown()
    # --- end verbatim main.py section 3a ---

    active = _kill_switch.is_active()
    loaded = _kill_switch.is_loaded()
    reason = _kill_switch.current_state().reason

    await session_module._engine.dispose()
    return active, loaded, reason


def _reset_kill_switch_mirror() -> None:
    from api.services import kill_switch as _kill_switch

    _kill_switch.reset_state_for_tests()


class TestWP17bLifespanKillSwitchLoad:
    """CF-B6: ``kill_switch.load()`` via main.py's own boot code path,
    against a real, migrated Postgres ``kill_switch_state`` row."""

    def setup_method(self) -> None:
        assert _MIGRATION_URL is not None  # guarded by pytestmark skipif

        # WP17b-S-06: this class's own tests directly UPDATE the
        # kill_switch_state row (_set_kill_switch_row) outside of the
        # shared CF-B7 teardown fixture -- apply the SAME guard here so a
        # mis-pointed MIGRATION_TEST_DATABASE_URL (equal to the real
        # DATABASE_URL, or a non-scratch-looking name) skips instead of
        # mutating a real safety-gate row.
        from tests.migrations.conftest import (
            ORIGINAL_DATABASE_URL,
            destructive_teardown_allowed,
        )

        allowed, reason = destructive_teardown_allowed(
            _MIGRATION_URL,
            original_database_url=ORIGINAL_DATABASE_URL,
            allow_destructive_env=os.environ.get("MIGRATION_TEST_ALLOW_DESTRUCTIVE"),
        )
        if not allowed:
            pytest.skip(f"WP17b-S-06 guard refused MIGRATION_TEST_DATABASE_URL: {reason}")

        cfg = _alembic_config(_MIGRATION_URL)
        from alembic import command

        command.upgrade(cfg, "head")

    def teardown_method(self) -> None:
        # Never leave the process-wide mirror latched/unknown for a LATER
        # test file in this same suite run.
        _reset_kill_switch_mirror()

    def test_unlatched_row_gives_not_active_and_loaded(self) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        asyncio.run(_set_kill_switch_row(dsn, active=False))

        active, loaded, reason = asyncio.run(_boot_load_kill_switch(_MIGRATION_URL))

        assert loaded is True, "a successful read must mark the mirror loaded"
        assert active is False, "an unlatched row must report is_active() False"
        assert reason is None

    def test_latched_row_gives_active_with_reason(self) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        asyncio.run(_set_kill_switch_row(dsn, active=True, reason="wp17b_cf_b6_test"))

        active, loaded, reason = asyncio.run(_boot_load_kill_switch(_MIGRATION_URL))

        assert loaded is True, "a successful read must mark the mirror loaded"
        assert active is True, "a latched row must report is_active() True"
        assert reason == "wp17b_cf_b6_test"

        # Leave the row unlatched for whatever runs after this test.
        asyncio.run(_set_kill_switch_row(dsn, active=False))

    def test_failed_db_read_gives_latch_state_unknown(self) -> None:
        bogus_url = _bogus_dsn(_MIGRATION_URL)  # type: ignore[arg-type]

        active, loaded, reason = asyncio.run(_boot_load_kill_switch(bogus_url))

        assert loaded is False, (
            "a failed read must leave the mirror UNLOADED (I4 fail-closed)"
        )
        assert active is True, (
            "an unloaded mirror must report is_active() True (fail-closed)"
        )
        assert reason == "latch_state_unknown"

        # Rebuild api.db.session's singletons back against the REAL
        # scratch DB so any test file collected after this one is not
        # left pointed at the bogus, nonexistent database.
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        import api.db.session as session_module

        async def _restore() -> None:
            if session_module._engine is not None:
                await session_module._engine.dispose()
            session_module._engine = create_async_engine(
                _MIGRATION_URL, poolclass=NullPool  # type: ignore[arg-type]
            )
            session_module._session_factory = None

        asyncio.run(_restore())
