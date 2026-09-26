"""
tests/migrations/test_wp18a_resume_races.py
---------------------------------------------
Real-Postgres concurrency tests for the WP1.8a-round2 resume races
(coordinator-mandated R2/R3, security review S-01/C-01).

These MUST run against a real PostgreSQL instance -- SQLite (used by the
rest of the hermetic suite via mocked/fake sessions) does not implement
real row-level locking (``SELECT ... FOR UPDATE``), so it cannot prove
the serialization property these tests exist to verify. Gated the same
way as ``test_017_orphaned_status.py``: reads ``MIGRATION_TEST_DATABASE_URL``
and SKIPS when unset.

R2 -- stop during resume
    A ``DELETE /runs/{id}`` (stop_run) issued while a resume is mid-flight
    (the row is locked at 'resuming' inside resume_run's own uncommitted
    transaction) must genuinely BLOCK on the real row lock -- not race
    past it and act on stale data -- and must only observe/mutate the row
    once resume_run's transaction has resolved (committed to 'running' or
    rolled back to 'orphaned').

R3 -- kill switch during a normal resume
    Same property for ``POST /emergency/kill-switch`` (WP1.8a-round2
    S-01 item 3: a single locking ``SELECT ... WHERE status IN
    ('running','orphaned') FOR UPDATE``).

Both tests use a real threading.Event-driven synchronization point
(patching ``run_recovery.scan_and_import``) so the "concurrent" request is
deterministically launched while resume_run's transaction is open and its
row lock is held, and use wall-clock timing to prove the concurrent
request was actually blocked on the database (not just fortuitously
ordered).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- these tests need a real "
        "Postgres instance (real row-level locking) and are skipped in "
        "environments without one. See test_017_orphaned_status.py's "
        "module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

ADMIN_KEY = "wp18a-race-test-admin-key-hex32x"  # noqa: S105 -- test fixture, not a real secret
CONFIRM_TOKEN = "wp18a-race-test-confirm-token"  # noqa: S105 -- test fixture, not a real secret


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


async def _seed_orphaned_live_run(dsn: str) -> uuid.UUID:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        run_id = uuid.uuid4()
        config = {
            "strategy_name": "grid_trading",
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
        }
        import json

        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', 'orphaned', $2::jsonb, now(), now(), now())
            """,
            run_id,
            json.dumps(config),
        )
        return run_id
    finally:
        await conn.close()


async def _fetch_run_status(dsn: str, run_id: uuid.UUID) -> str:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval("SELECT status FROM runs WHERE id = $1", run_id)
    finally:
        await conn.close()


@pytest.fixture()
def race_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real ``create_app()`` instance bound to the real scratch Postgres DB.

    Unlike the hermetic resume-endpoint tests (``test_wp18a_resume_endpoint.py``),
    ``get_db`` is NOT overridden here -- every request gets its own real
    ``AsyncSession`` from a real connection pool against
    ``MIGRATION_TEST_DATABASE_URL``, so Postgres's actual row-level locking
    is what these tests exercise.
    """
    assert _MIGRATION_URL is not None  # guarded by pytestmark skipif
    database_url = _MIGRATION_URL

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp18a-race-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp18a-race-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    from api.config import get_settings

    get_settings.cache_clear()

    # The module-level engine/session-factory singletons in api.db.session
    # are lazy but cached forever once built -- a prior test module in the
    # same process may have already built one against a different (or no)
    # DATABASE_URL. Force a rebuild against the scratch DB.
    import api.db.session as session_module

    session_module._engine = None
    session_module._session_factory = None

    from alembic import command

    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    import api.routers.runs as runs_module

    original_registry = runs_module._STRATEGY_REGISTRY
    runs_module._STRATEGY_REGISTRY = {"grid_trading": MagicMock()}
    runs_module._RUN_TASKS.clear()

    from api.main import create_app

    app = create_app()

    yield app

    runs_module._STRATEGY_REGISTRY = original_registry
    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()
    get_settings.cache_clear()


def _make_blocking_scan(scan_started: threading.Event, release_scan: threading.Event) -> Any:
    """A ``scan_and_import`` replacement that signals ``scan_started`` then
    polls ``release_scan`` (a plain ``threading.Event`` -- safe to ``.set()``
    from the test's main OS thread; the poll loop is what safely bridges it
    into the ASGI app's own asyncio event loop) before returning.

    While this coroutine is suspended, ``resume_run``'s transaction is still
    open and its ``orphaned -> resuming`` row lock is still held -- this is
    the window the concurrent request under test must genuinely block on.
    """
    from api.services.run_recovery import ImportReport

    async def _blocking_scan(db: Any, run: Any, exchange: Any) -> ImportReport:
        scan_started.set()
        while not release_scan.is_set():
            await asyncio.sleep(0.01)
        return ImportReport()

    return _blocking_scan


class TestR2StopDuringResume:
    def test_stop_run_blocks_until_resume_transaction_resolves(self, race_app: Any) -> None:
        """WP1.8a-round2 R2: DELETE /runs/{id} issued while a resume is
        mid-flight must block on the real row lock (stop_run's own
        ``.with_for_update()`` re-read, C-01/S-01 item 4) until resume_run's
        transaction commits, then correctly stop the now-'running' run --
        never act on the transient 'resuming' state, never lose the
        update."""
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        scan_started = threading.Event()
        release_scan = threading.Event()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _make_blocking_scan(scan_started, release_scan)

        resume_headers = {"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY}

        try:
            with (
                patch("api.routers.runs._run_live_engine", AsyncMock()),
                concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
            ):
                resume_future = pool.submit(
                    client.post, f"/api/v1/runs/{run_id}/resume", headers=resume_headers
                )

                # Wait for resume_run's transaction to actually be open and
                # mid-flight (the CAS to 'resuming' has flushed, the row
                # lock is held) before firing the concurrent stop.
                assert scan_started.wait(timeout=10.0), "resume never reached the scan stub"

                stop_started_at = time.monotonic()
                stop_future = pool.submit(client.delete, f"/api/v1/runs/{run_id}")

                # Give stop_run's own SELECT ... FOR UPDATE a real chance to
                # issue and block on the still-held row lock before we
                # release the resume side -- this is what makes the later
                # timing assertion meaningful (proves a genuine block, not
                # a lucky ordering).
                time.sleep(0.3)
                release_at = time.monotonic()
                release_scan.set()

                resume_resp = resume_future.result(timeout=10.0)
                stop_resp = stop_future.result(timeout=10.0)
                stop_completed_at = time.monotonic()
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert resume_resp.status_code == 200, resume_resp.text
        assert stop_resp.status_code == 200, stop_resp.text

        # The real proof this was a genuine Postgres-level block, not a
        # lucky thread-scheduling accident: stop_run's response could not
        # have completed until AFTER we released the resume side.
        assert stop_completed_at >= release_at, (
            "stop_run completed before the resume transaction released its "
            "row lock -- it did not actually block on the real lock"
        )
        assert (stop_completed_at - stop_started_at) >= 0.25, (
            "stop_run returned suspiciously fast for a request that should "
            "have been blocked behind an open transaction for ~0.3s"
        )

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "stopped", (
            "resume must complete (running) and then be stopped -- the "
            "row must never be observed/mutated while still 'resuming'"
        )


class TestR3KillSwitchDuringResume:
    def test_kill_switch_blocks_until_resume_transaction_resolves(self, race_app: Any) -> None:
        """WP1.8a-round2 R3: POST /emergency/kill-switch issued while a
        NORMAL resume is mid-flight must block on the real row lock
        (S-01 item 3's single ``SELECT ... WHERE status IN
        ('running','orphaned') FOR UPDATE``) until resume_run's
        transaction resolves, then correctly stop the now-'running' run.
        """
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        scan_started = threading.Event()
        release_scan = threading.Event()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _make_blocking_scan(scan_started, release_scan)

        resume_headers = {"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY}

        try:
            with (
                patch("api.routers.runs._run_live_engine", AsyncMock()),
                concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
            ):
                resume_future = pool.submit(
                    client.post, f"/api/v1/runs/{run_id}/resume", headers=resume_headers
                )

                assert scan_started.wait(timeout=10.0), "resume never reached the scan stub"

                kill_started_at = time.monotonic()
                kill_future = pool.submit(
                    client.post,
                    "/api/v1/emergency/kill-switch",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )

                time.sleep(0.3)
                release_at = time.monotonic()
                release_scan.set()

                resume_resp = resume_future.result(timeout=10.0)
                kill_resp = kill_future.result(timeout=10.0)
                kill_completed_at = time.monotonic()
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert resume_resp.status_code == 200, resume_resp.text
        assert kill_resp.status_code == 200, kill_resp.text

        assert kill_completed_at >= release_at, (
            "kill-switch completed before the resume transaction released "
            "its row lock -- it did not actually block on the real lock"
        )
        assert (kill_completed_at - kill_started_at) >= 0.25, (
            "kill-switch returned suspiciously fast for a request that "
            "should have been blocked behind an open transaction for ~0.3s"
        )

        body = kill_resp.json()
        assert str(run_id) in body["runs_stopped"], (
            "kill-switch must observe the run as 'running' (post-resume-"
            f"commit) and stop it, got: {body}"
        )

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "stopped"
