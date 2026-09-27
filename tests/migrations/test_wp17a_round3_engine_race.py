"""
tests/migrations/test_wp17a_round3_engine_race.py
-----------------------------------------------------
Real-Postgres regression for WP17a-S-R2-01 (round 3): a concurrent
kill-switch press and clear must never leave a running engine's own
latch state disagreeing with the persisted row / in-memory mirror.

Reproduces the two interleavings from
``reports/vp2-wp1.7/security-report-1.7a-r2.md``:

- **A7**: an external transaction holds ``FOR UPDATE`` on the
  ``kill_switch_state`` row; a :func:`kill_switch.clear` call blocks on
  that same row while it already holds ``kill_switch.py``'s own
  in-process ``asyncio.Lock``; a press (:func:`kill_switch.activate`)
  arrives concurrently and must queue behind that lock. Releasing the
  external Postgres lock lets ``clear()`` finish and release the
  asyncio lock; ``activate()`` then runs -- and must re-latch the
  engine atomically with the mirror.
- **R1**: an external transaction holds ``FOR UPDATE`` on a seeded
  ``runs`` row, standing in for the kill-switch ROUTER's own
  candidates-query lock wait (the thing that blocks a press BEFORE it
  ever calls :func:`kill_switch.activate`). While the simulated press
  is stuck there, a :func:`kill_switch.clear` call arrives and runs to
  completion, unlatching the engine. Only once the external lock is
  released does the press proceed to call :func:`kill_switch.activate`,
  which must re-latch the engine.

In both cases the invariant under test (per the security report) is::

    engine.kill_switch_active == kill_switch.is_active()

after every interleaving has fully settled -- never fail-open (engine
unlatched while the mirror/DB says latched), regardless of which call
happens to run last.

Round 4 (security round 3, WP17a-S-R3-01) hardens this file itself,
which round 3 shipped with two test-hygiene gaps:

- **No teardown**: both scenarios seeded ``kill_switch_state`` with
  ``reason='prior-press'`` and (R1) inserted a ``runs`` row, and left
  both behind on the shared migration DB -- running this file BEFORE
  ``test_018_kill_switch_latch.py`` in the same session made that
  file's own round-trip assertion fail (``'prior-press' is None``)
  because it reused the same singleton row. Both scenarios now run
  their full body inside ``try/finally``: the ``finally`` resets
  ``kill_switch_state`` to ``active=false, reason=NULL`` (and every
  other column :func:`_set_kill_switch_row`/:func:`activate`/
  :func:`clear` may have touched) and deletes any ``runs`` row this
  file inserted -- regardless of whether the scenario's own assertions
  passed or raised.
- **Never committed**: ``activate()``'s own persistence runs inside
  ``db.begin_nested()`` (a SAVEPOINT) but does not itself commit the
  OUTER session -- exiting ``async with session_factory() as db:``
  without an explicit ``await db.commit()`` rolls back whatever
  ``activate()`` just wrote, so the "DB row actually matches" half of
  the invariant was never genuinely exercised (only the in-memory
  mirror and the engine were ever checked). Both scenarios now commit
  explicitly after every ``activate()``/``clear()`` call and assert
  ``SELECT active FROM kill_switch_state`` (via a fresh raw connection)
  equals ``kill_switch.is_active()`` equals ``engine.kill_switch_active``.

This test MUST run against a real PostgreSQL instance -- not mocked, not
SQLite -- because the whole point is genuine row-lock contention forcing
the two coroutines to interleave in a specific, externally-controlled
order. It reads the scratch-DB DSN from ``MIGRATION_TEST_DATABASE_URL``
(``postgresql+asyncpg://user:pass@host:port/dbname``); when unset the
test SKIPS.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- this test needs a real "
        "Postgres instance and is skipped in environments without one."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_WAIT_GRACE_S = 0.3
_ASSERT_TIMEOUT_S = 5.0


def _ensure_head(database_url: str) -> None:
    from alembic import command
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    command.upgrade(cfg, "head")


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


class _FakeRiskManager:
    """A real (non-Mock) reason-set double -- see
    tests/unit/test_wp17a_round3_security.py for the rationale."""

    def __init__(self) -> None:
        self._reasons: set[str] = set()

    def trigger_kill_switch(self, reason: str) -> None:
        self._reasons.add(reason)

    def reset_kill_switch(self, reason: str) -> None:
        self._reasons.discard(reason)

    @property
    def kill_switch_active(self) -> bool:
        return bool(self._reasons)

    @property
    def kill_switch_reasons(self) -> frozenset[str]:
        return frozenset(self._reasons)


class _FakeEngine:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.risk_manager = _FakeRiskManager()

    # Convenience alias so the assertion can literally read
    # ``engine.kill_switch_active == kill_switch.is_active()`` per the
    # coordinator's required-test wording.
    @property
    def kill_switch_active(self) -> bool:
        return self.risk_manager.kill_switch_active


async def _set_kill_switch_row(dsn: str, *, active: bool, reason: str | None) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            UPDATE kill_switch_state
            SET active = $1, reason = $2, activated_at = now(), activated_by = 'seed',
                cleared_at = NULL, cleared_by = NULL, clear_reason = NULL
            WHERE id = 1
            """,
            active,
            reason,
        )
    finally:
        await conn.close()


async def _reset_kill_switch_row(dsn: str) -> None:
    """WP17a-S-R3-01 (round 4): unconditional teardown -- resets EVERY
    column either scenario (or activate()/clear() themselves) may have
    touched, back to migration 018's own seeded-inactive shape, so a
    later test (e.g. test_018_kill_switch_latch.py's own round-trip
    assertion) never observes this file's leftover state."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            UPDATE kill_switch_state
            SET active = false, reason = NULL, activated_at = NULL, activated_by = NULL,
                cleared_at = NULL, cleared_by = NULL, clear_reason = NULL
            WHERE id = 1
            """
        )
    finally:
        await conn.close()


async def _insert_run(dsn: str) -> uuid.UUID:
    import asyncpg

    run_id = uuid.uuid4()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'running', '{}'::jsonb, now(), now(), now())
            """,
            run_id,
        )
    finally:
        await conn.close()
    return run_id


async def _delete_run(dsn: str, run_id: uuid.UUID) -> None:
    """WP17a-S-R3-01 (round 4): delete the run row this file inserted --
    it has no relationship to any other suite's fixtures, so simple
    unconditional deletion is safe and sufficient."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DELETE FROM runs WHERE id = $1", run_id)
    finally:
        await conn.close()


async def _fetch_kill_switch_active(dsn: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow("SELECT active FROM kill_switch_state WHERE id = 1")
        assert row is not None
        return bool(row["active"])
    finally:
        await conn.close()


def test_a7_clear_blocked_on_kill_switch_row_then_press_arrives() -> None:
    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by pytestmark skipif
    _ensure_head(database_url)
    asyncio.run(_a7_scenario(database_url))


async def _a7_scenario(database_url: str) -> None:
    import asyncpg
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from api.services import kill_switch

    dsn = _asyncpg_dsn(database_url)
    kill_switch.reset_state_for_tests()
    engine = _FakeEngine("run-a7")

    try:
        # Seed: kill switch currently ACTIVE (a prior press), engine
        # latched to match -- the starting state before the race begins.
        await _set_kill_switch_row(dsn, active=True, reason="prior-press")
        kill_switch.latch_in_memory("prior-press")
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        assert engine.kill_switch_active == kill_switch.is_active() is True

        sa_engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(sa_engine, expire_on_commit=False)

        # An EXTERNAL connection holds FOR UPDATE on the kill_switch_state
        # row -- clear()'s own SELECT ... FOR UPDATE will genuinely block
        # on this at the Postgres level (A7).
        holder_conn = await asyncpg.connect(dsn)
        await holder_conn.execute("BEGIN")
        await holder_conn.fetchrow("SELECT * FROM kill_switch_state WHERE id = 1 FOR UPDATE")

        try:
            clear_started = asyncio.Event()

            async def _run_clear() -> None:
                async with session_factory() as db:
                    clear_started.set()
                    # kill_switch.clear() commits internally as part of
                    # its own critical section (before flipping the
                    # mirror) -- nothing extra to commit here.
                    await kill_switch.clear(
                        db,
                        reason="operator-clear",
                        actor="tester",
                        engines=lambda: [engine],
                    )

            clear_task = asyncio.create_task(_run_clear())
            await asyncio.wait_for(clear_started.wait(), timeout=_ASSERT_TIMEOUT_S)
            await asyncio.sleep(_LOCK_WAIT_GRACE_S)
            assert not clear_task.done(), (
                "clear() should still be blocked on the real Postgres row lock"
            )

            # The press arrives while clear() is stuck: it must queue
            # behind kill_switch.py's own asyncio lock (held by clear()).
            async def _run_activate() -> None:
                async with session_factory() as db:
                    await kill_switch.activate(
                        db,
                        reason="press-reason",
                        actor="tester",
                        engines=lambda: [engine],
                    )
                    # WP17a-S-R3-01 (round 4): activate()'s own persist
                    # step only runs inside a SAVEPOINT
                    # (db.begin_nested()) -- the OUTER session must be
                    # committed explicitly, or exiting this ``async
                    # with`` block rolls the write back.
                    await db.commit()

            activate_task = asyncio.create_task(_run_activate())
            await asyncio.sleep(_LOCK_WAIT_GRACE_S)
            assert not activate_task.done(), (
                "activate() must queue behind clear()'s lock hold"
            )

            # Release the external Postgres lock: clear() can now finish.
            await holder_conn.execute("COMMIT")
        finally:
            await holder_conn.close()

        await asyncio.wait_for(clear_task, timeout=_ASSERT_TIMEOUT_S)
        await asyncio.wait_for(activate_task, timeout=_ASSERT_TIMEOUT_S)

        # Whichever call's critical section ran last while holding the
        # lock -- here, activate(), since it was still queued when
        # clear() released -- the DB row, the mirror, AND the engine
        # must all agree (WP17a-S-R3-01: the DB row check is new).
        db_active = await _fetch_kill_switch_active(dsn)
        assert db_active == kill_switch.is_active() == engine.kill_switch_active
        assert kill_switch.is_active() is True
        assert engine.kill_switch_active is True
        assert db_active is True

        await sa_engine.dispose()
    finally:
        await _reset_kill_switch_row(dsn)
        kill_switch.reset_state_for_tests()


def test_r1_press_blocked_on_runs_row_then_clear_arrives() -> None:
    database_url = _MIGRATION_URL
    assert database_url is not None  # guarded by pytestmark skipif
    _ensure_head(database_url)
    asyncio.run(_r1_scenario(database_url))


async def _r1_scenario(database_url: str) -> None:
    import asyncpg
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from api.services import kill_switch

    dsn = _asyncpg_dsn(database_url)
    kill_switch.reset_state_for_tests()
    engine = _FakeEngine("run-r1")
    run_id: uuid.UUID | None = None

    try:
        # Seed: kill switch currently ACTIVE (a prior press), engine latched.
        await _set_kill_switch_row(dsn, active=True, reason="prior-press")
        kill_switch.latch_in_memory("prior-press")
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        assert engine.kill_switch_active == kill_switch.is_active() is True

        run_id = await _insert_run(dsn)

        sa_engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(sa_engine, expire_on_commit=False)

        # An EXTERNAL connection holds FOR UPDATE on the seeded runs row
        # -- standing in for the kill-switch router's own candidates
        # query, which blocks a press BEFORE it ever calls activate() (R1).
        holder_conn = await asyncpg.connect(dsn)
        await holder_conn.execute("BEGIN")
        await holder_conn.fetchrow("SELECT * FROM runs WHERE id = $1 FOR UPDATE", run_id)

        try:
            press_blocked = asyncio.Event()

            async def _run_press() -> None:
                # Genuinely await the SAME Postgres-level lock the
                # router's own candidates query would be stuck on, via a
                # SEPARATE raw connection (mirroring a different DB
                # session), THEN call activate() only once that resolves.
                conn = await asyncpg.connect(dsn)
                try:
                    press_blocked.set()
                    await conn.fetchrow("SELECT * FROM runs WHERE id = $1 FOR UPDATE", run_id)
                finally:
                    await conn.close()
                async with session_factory() as db:
                    await kill_switch.activate(
                        db,
                        reason="press-reason",
                        actor="tester",
                        engines=lambda: [engine],
                    )
                    # WP17a-S-R3-01 (round 4): see the A7 scenario's
                    # identical comment -- the outer session must be
                    # committed explicitly.
                    await db.commit()

            press_task = asyncio.create_task(_run_press())
            await asyncio.wait_for(press_blocked.wait(), timeout=_ASSERT_TIMEOUT_S)
            await asyncio.sleep(_LOCK_WAIT_GRACE_S)
            assert not press_task.done(), (
                "the press should still be blocked on the real runs-row lock"
            )

            # The clear arrives and runs to completion WHILE the press is
            # still stuck -- nothing about clear() touches the runs
            # table, so it is not blocked by the external lock above.
            async with session_factory() as db:
                await kill_switch.clear(
                    db,
                    reason="operator-clear",
                    actor="tester",
                    engines=lambda: [engine],
                )
            db_active_after_clear = await _fetch_kill_switch_active(dsn)
            assert db_active_after_clear == kill_switch.is_active() == engine.kill_switch_active
            assert kill_switch.is_active() is False
            assert engine.kill_switch_active is False

            # Release the runs-row lock: the press proceeds and must
            # re-latch the engine atomically with the mirror.
            await holder_conn.execute("COMMIT")
        finally:
            await holder_conn.close()

        await asyncio.wait_for(press_task, timeout=_ASSERT_TIMEOUT_S)

        db_active = await _fetch_kill_switch_active(dsn)
        assert db_active == kill_switch.is_active() == engine.kill_switch_active
        assert kill_switch.is_active() is True
        assert engine.kill_switch_active is True
        assert db_active is True

        await sa_engine.dispose()
    finally:
        await _reset_kill_switch_row(dsn)
        if run_id is not None:
            await _delete_run(dsn, run_id)
        kill_switch.reset_state_for_tests()
