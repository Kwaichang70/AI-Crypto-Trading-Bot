"""
tests/migrations/conftest.py
------------------------------
WP1.7b (CF-B7): shared teardown fixture for the whole ``tests/migrations``
suite, against the SAME real, scratch Postgres DB every file in this
directory shares for the duration of one ``pytest tests/migrations`` run
(see ``reports/vp2-wp1.7/critic-disposition-1.7a-acceptance.md``'s A3
disposition, which identified and carried forward this gap: five
pre-existing files -- ``test_017_orphaned_status.py``,
``test_wp14b_resume_scan.py``, ``test_wp18a_resume_races.py``,
``test_wp18b_scan_and_import.py`` and ``test_018_kill_switch_latch.py``
-- each ``INSERT`` into ``runs`` with no matching cleanup, and there was
no shared fixture that would do it for them).

This autouse, function-scoped fixture runs after EVERY test collected
under ``tests/migrations/`` and:
    - deletes every row from ``runs`` and ``audit_events`` (nulling
      ``runs``'s own two self-referencing FK columns first, since those
      are plain ``ForeignKey("runs.id")`` with no ``ondelete=CASCADE`` --
      every OTHER table with a ``runs.id`` FK, e.g. ``orders``/``trades``/
      ``equity_snapshots``, IS declared ``ondelete="CASCADE"`` and needs
      no help);
    - resets the ``kill_switch_state`` singleton row back to its
      migration-018-seeded default: ``active=false``, every other column
      ``NULL``.

Safe no-op (by design, per the CF-B7 spec):
    - when ``MIGRATION_TEST_DATABASE_URL`` is unset -- no real DB was
      ever touched by any test in this run, so there's nothing to clean;
    - when a table does not exist, e.g. right after a test's own
      upgrade/downgrade round trip left the scratch DB mid-downgrade
      (``test_018_kill_switch_latch.py`` drops ``kill_switch_state``
      entirely for one step of its round trip) -- each statement below is
      individually guarded with a ``to_regclass`` existence check;
    - when the scratch DB is temporarily unreachable at all (e.g. a
      connection-failure test in this same suite intentionally pointed
      ``api.db.session``'s OWN module-level engine singleton at a bogus,
      nonexistent database for one scenario) -- this fixture always opens
      its OWN, independent ``asyncpg`` connection straight from
      ``MIGRATION_TEST_DATABASE_URL``, never reusing
      ``api.db.session``'s (test-mutated) singletons, and swallows a
      connection failure rather than failing every other test in the
      run over it.

WP1.7b round 2 (WP17b-S-06, security round 1): this fixture is
destructive by design (bulk deletes + a safety-gate reset), which is
fine on a disposable scratch DB but becomes a safety-gate reset if an
operator ever points ``MIGRATION_TEST_DATABASE_URL`` at a real database.
:func:`destructive_teardown_allowed` is the pure (no I/O) gate checked
before any of the above runs:
    (a) refuses when ``MIGRATION_TEST_DATABASE_URL`` equals
        ``DATABASE_URL`` as it was AT CONFTEST IMPORT TIME (captured
        into :data:`ORIGINAL_DATABASE_URL` before any test file's own
        ``_alembic_config`` helper overwrites ``os.environ["DATABASE_URL"]``
        with the very same scratch DSN -- reading ``DATABASE_URL`` fresh
        at teardown time would make this check tautological);
    (b) refuses when the database NAME embedded in
        ``MIGRATION_TEST_DATABASE_URL`` does not look like a disposable
        scratch name (contains ``test``, ``scratch`` or ``tmp``,
        case-insensitive), unless ``MIGRATION_TEST_ALLOW_DESTRUCTIVE=1``
        is explicitly set.
A refusal is logged and the fixture is a no-op for that test -- it never
raises, so a misconfigured environment fails safe (no cleanup) rather
than failing the test suite.

This fixture never raises for any OTHER reason either. A cleanup
failure is logged and swallowed -- its job is post-test hygiene, not the
test's own assertions, and a pre-existing round-trip test
(upgrade/downgrade) must keep passing whether or not this fixture's own
cleanup happens to run against a mid-round-trip schema state.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from typing import Any
from urllib.parse import urlsplit

import pytest
import structlog

__all__ = [
    "ORIGINAL_DATABASE_URL",
    "destructive_teardown_allowed",
    "is_scratch_db_name",
]

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

#: WP17b-S-06: captured ONCE, at import time -- i.e. before pytest has
#: collected (let alone run) a single test file in this directory, and
#: therefore before any of them has had a chance to overwrite
#: ``os.environ["DATABASE_URL"]`` with the scratch DSN (every
#: real-Postgres test file's ``_alembic_config`` helper does exactly
#: that). This is what ``destructive_teardown_allowed``'s check (a)
#: compares ``MIGRATION_TEST_DATABASE_URL`` against.
ORIGINAL_DATABASE_URL = os.environ.get("DATABASE_URL")

#: WP17b-S-06: substrings (case-insensitive) that mark a database name as
#: a disposable scratch DB -- matches every naming convention already
#: used across this suite's own module docstrings/producer reports, e.g.
#: ``wp17a_migration_test``, ``sec17b_<ts>_<pid>``, ``wp17b_scratch_*``.
_SCRATCH_NAME_MARKERS = ("test", "scratch", "tmp")

logger = structlog.get_logger(__name__)


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


def _dsn_db_name(dsn: str) -> str:
    """Best-effort extraction of the database name (URL path component,
    leading slash stripped) from a SQLAlchemy- or asyncpg-style DSN.
    Pure string parsing -- no network, no DB driver needed."""
    return urlsplit(dsn).path.lstrip("/")


def is_scratch_db_name(dsn: str) -> bool:
    """WP17b-S-06: True when the database name embedded in ``dsn`` looks
    like a disposable scratch DB (contains ``test``, ``scratch`` or
    ``tmp``, case-insensitive). Pure function -- no I/O, safe to unit
    test with a bare string."""
    return any(marker in _dsn_db_name(dsn).lower() for marker in _SCRATCH_NAME_MARKERS)


def destructive_teardown_allowed(
    migration_url: str | None,
    *,
    original_database_url: str | None,
    allow_destructive_env: str | None,
) -> tuple[bool, str]:
    """WP17b-S-06: the pure (no I/O) decision function guarding every
    destructive operation in this file (and reused by
    ``test_wp17b_lifespan_kill_switch.py``, which also mutates
    ``kill_switch_state`` directly). Returns ``(allowed, reason)``:

    - ``migration_url`` falsy -> refused (nothing configured to clean).
    - ``migration_url == original_database_url`` -> refused: the
      "scratch" variable is pointed at the SAME database the app itself
      would connect to.
    - the DB name in ``migration_url`` does not look like a scratch name
      (:func:`is_scratch_db_name`) -> refused, UNLESS
      ``allow_destructive_env == "1"`` (``MIGRATION_TEST_ALLOW_DESTRUCTIVE=1``)
      is explicitly set.
    - otherwise -> allowed.
    """
    if not migration_url:
        return False, "MIGRATION_TEST_DATABASE_URL is unset"
    if original_database_url is not None and migration_url == original_database_url:
        return False, "MIGRATION_TEST_DATABASE_URL equals DATABASE_URL"
    if allow_destructive_env == "1":
        return True, "MIGRATION_TEST_ALLOW_DESTRUCTIVE=1 override"
    if not is_scratch_db_name(migration_url):
        return False, (
            "database name does not match a scratch pattern "
            "(test/scratch/tmp) -- set MIGRATION_TEST_ALLOW_DESTRUCTIVE=1 "
            "to override"
        )
    return True, "scratch DB name pattern matched"


async def _table_exists(conn: Any, table_name: str) -> bool:
    return bool(await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table_name))  # type: ignore[attr-defined]


async def _cleanup_migrations_db() -> None:
    if not _MIGRATION_URL:
        return

    allowed, reason = destructive_teardown_allowed(
        _MIGRATION_URL,
        original_database_url=ORIGINAL_DATABASE_URL,
        allow_destructive_env=os.environ.get("MIGRATION_TEST_ALLOW_DESTRUCTIVE"),
    )
    if not allowed:
        logger.warning("migrations_teardown.refused", reason=reason)
        return

    import asyncpg

    dsn = _asyncpg_dsn(_MIGRATION_URL)
    try:
        conn = await asyncpg.connect(dsn)
    except Exception:
        logger.warning("migrations_teardown.connect_failed", exc_info=True)
        return

    try:
        if await _table_exists(conn, "runs"):
            # Null the two self-referencing FK columns first (plain
            # ForeignKey("runs.id"), no CASCADE) so a bulk DELETE can
            # never trip over a row that references another row in the
            # SAME statement's deletion set.
            try:
                await conn.execute(
                    "UPDATE runs SET recovered_from_run_id = NULL, "
                    "promoted_from_run_id = NULL"
                )
            except Exception:
                logger.warning(
                    "migrations_teardown.null_self_fk_failed", exc_info=True
                )
            await conn.execute("DELETE FROM runs")

        if await _table_exists(conn, "audit_events"):
            await conn.execute("DELETE FROM audit_events")

        # WP7.0 (DB-06): idempotency_keys.run_id has a real FK to runs.id
        # (ON DELETE SET NULL) -- deleted AFTER runs above is fine either
        # way, but doing it explicitly here (rather than relying on the
        # FK's SET NULL side effect leaving stale rows behind) keeps this
        # suite's shared scratch DB fully empty between tests, exactly
        # like every other table in this fixture.
        if await _table_exists(conn, "idempotency_keys"):
            await conn.execute("DELETE FROM idempotency_keys")

        if await _table_exists(conn, "kill_switch_state"):
            await conn.execute(
                """
                UPDATE kill_switch_state
                SET active = FALSE, reason = NULL, activated_at = NULL,
                    activated_by = NULL, cleared_at = NULL, cleared_by = NULL,
                    clear_reason = NULL
                WHERE id = 1
                """
            )
    except Exception:
        # Post-test hygiene only -- never fail (or mask) the test itself.
        logger.warning("migrations_teardown.cleanup_failed", exc_info=True)
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
def _migrations_db_teardown() -> Generator[None, None, None]:
    """CF-B7: runs after every test in ``tests/migrations/`` (see module
    docstring). Yields first so the test itself runs unaffected, then
    cleans up the shared scratch DB (WP17b-S-06 guard permitting)."""
    yield
    asyncio.run(_cleanup_migrations_db())
