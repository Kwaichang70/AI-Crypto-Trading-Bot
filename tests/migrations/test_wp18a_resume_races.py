"""
tests/migrations/test_wp18a_resume_races.py
---------------------------------------------
Real-Postgres concurrency tests for the resume races (coordinator-mandated
R2/R3, security review S-01/C-01), REWRITTEN for WP1.8b's S2-01 binding
condition (three short transactions -- see ``apps.api.routers.runs
.resume_run``'s docstring and ``reports/vp2-wp1.8/security-report-1.8a-r2.md``
WP18a-S2-01).

These MUST run against a real PostgreSQL instance -- SQLite (used by the
rest of the hermetic suite via mocked/fake sessions) does not implement
real row-level locking (``SELECT ... FOR UPDATE``), so it cannot prove
the serialization property these tests exist to verify. Gated the same
way as ``test_017_orphaned_status.py``: reads ``MIGRATION_TEST_DATABASE_URL``
and SKIPS when unset.

WP1.8a-round2 shipped these two tests asserting the OPPOSITE property to
what they assert now: a stop/kill-switch issued during an in-flight
resume used to BLOCK on the resume's own row lock for the whole scan
duration (because the CAS to 'resuming' was never committed until the
scan finished). WP18a-S2-01 identified that as a latent regression --
once WP1.8b's real scan can take up to 30s per open order, that would
delay the GLOBAL kill switch by as much. The fix (three short
transactions -- commit orphaned->resuming immediately, scan with no lock
held, then a second short transaction for the final CAS) means a
stop/kill-switch issued while the scan is running now sees NO lock on the
row at all and acts immediately.

R2 -- stop during an in-flight resume
    A ``DELETE /runs/{id}`` (stop_run) issued while a resume's scan is
    running (status already committed to 'resuming', no lock held) must
    NOT block -- it wins the race outright, moving the row straight to
    'stopped'. The resume's own later final CAS (resuming->running) then
    finds 0 rows and returns 409 'resume_state_lost'.

R3 / S2-01 -- kill-switch latency during a 5s scan
    The mandatory 1.8b test: ``POST /emergency/kill-switch`` issued while
    a resume's scan is running (simulated as a genuine 5-second
    ``asyncio.sleep``, not event-gated -- this is real wall-clock latency,
    not a lock-contention proxy) must respond in under 1 second. The
    kill-switch moves the 'resuming' row to 'orphaned', so the resume's
    own final CAS again finds 0 rows and returns 409 'resume_state_lost'.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
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
    #
    # WP18b round 2: built with NullPool (no connection reuse across
    # checkouts) instead of the default pooled engine. This test file
    # deliberately drives concurrent requests from DIFFERENT OS threads
    # (ThreadPoolExecutor) each with their OWN asyncio event loop; asyncpg
    # connections are bound to the loop they were created on, and
    # WP18b-S-02's `_reject` now does a mid-request `rollback()` (checking
    # its connection back into the pool) before its own next `execute()`
    # (checking a connection back OUT) -- with a real pool, that in-between
    # window lets a DIFFERENT thread's concurrent checkout receive a
    # connection bound to THIS thread's loop, crashing with "Future
    # attached to a different loop". NullPool never reuses a connection
    # across a checkin/checkout boundary, so this can't happen -- a purely
    # test-harness fix, not a production behaviour change (a real
    # deployment runs one worker on one event loop).
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    import api.db.session as session_module

    session_module._engine = create_async_engine(database_url, poolclass=NullPool)
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

    async def _blocking_scan(
        db: Any, run: Any, exchange: Any, *, fence: Any = None
    ) -> ImportReport:
        scan_started.set()
        while not release_scan.is_set():
            await asyncio.sleep(0.01)
        return ImportReport()

    return _blocking_scan


def _make_sleeping_scan(scan_started: threading.Event, sleep_seconds: float) -> Any:
    """A ``scan_and_import`` replacement that signals ``scan_started`` then
    sleeps for ``sleep_seconds`` of REAL wall-clock time before returning
    (WP1.8b S2-01's mandatory latency test -- a genuine multi-second scan,
    not an event-gated proxy for one)."""
    from api.services.run_recovery import ImportReport

    async def _sleeping_scan(
        db: Any, run: Any, exchange: Any, *, fence: Any = None
    ) -> ImportReport:
        scan_started.set()
        await asyncio.sleep(sleep_seconds)
        return ImportReport()

    return _sleeping_scan


class TestR2StopDuringResume:
    def test_stop_run_does_not_block_and_wins_the_race(self, race_app: Any) -> None:
        """WP1.8b (S2-01): a DELETE /runs/{id} issued while a resume's scan
        is running (transaction (a) already committed -- status='resuming'
        with NO lock held) must NOT block -- it acts immediately, moving
        the row straight to 'stopped'. The resume's own later final CAS
        (resuming->running) then finds 0 rows and returns 409
        'resume_state_lost' instead of silently overwriting the stop."""
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

                # Wait for resume_run's transaction (a) to have committed
                # (status='resuming') and the scan to be in flight -- with
                # NO lock held on the row from this point on.
                assert scan_started.wait(timeout=10.0), "resume never reached the scan stub"

                stop_started_at = time.monotonic()
                stop_resp = client.delete(f"/api/v1/runs/{run_id}")
                stop_completed_at = time.monotonic()

                # Let the still-in-flight resume's scan finish and reach
                # its own final CAS.
                release_scan.set()
                resume_resp = resume_future.result(timeout=10.0)
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert stop_resp.status_code == 200, stop_resp.text
        assert (stop_completed_at - stop_started_at) < 1.0, (
            "stop_run should win the race immediately (no lock held during "
            f"an in-flight scan, S2-01) -- took {stop_completed_at - stop_started_at:.3f}s"
        )

        assert resume_resp.status_code == 409, resume_resp.text
        assert resume_resp.json()["detail"] == "resume_state_lost"

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "stopped", (
            "stop_run must win outright -- the resume's final CAS must not "
            "silently overwrite it"
        )


class TestR3KillSwitchLatencyDuringResumeScan:
    def test_kill_switch_latency_under_1s_during_5s_scan(self, race_app: Any) -> None:
        """WP1.8b S2-01 mandatory test: kill-switch latency stays under 1s
        while a resume's scan is a genuine 5-second operation. This is the
        regression test for WP18a-S2-01 (measured 2.82s with the OLD
        single-long-transaction design and a mere 3s scan -- unbounded
        with a real 30s-per-order cancel/poll). The kill switch moves the
        'resuming' row to 'orphaned'; the resume's own final CAS then
        finds 0 rows and returns 409 'resume_state_lost'."""
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        scan_started = threading.Event()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _make_sleeping_scan(scan_started, 5.0)

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
                kill_resp = client.post(
                    "/api/v1/emergency/kill-switch",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
                kill_completed_at = time.monotonic()

                resume_resp = resume_future.result(timeout=10.0)
        finally:
            run_recovery_module.scan_and_import = original_scan

        kill_latency = kill_completed_at - kill_started_at
        assert kill_resp.status_code == 200, kill_resp.text
        assert kill_latency < 1.0, (
            f"kill-switch latency was {kill_latency:.3f}s during a 5s scan "
            "-- S2-01 requires it stay under 1s regardless of scan duration"
        )

        # The resume's own S4 check #2 (a kill_switch audit row now
        # postdates run.started_at) fires before it would ever reach the
        # final CAS -- "kill_switch_after_start" is the more precise
        # reason and is asserted here; a "resume_state_lost" 409 from the
        # final CAS itself is exercised by TestR2StopDuringResume above
        # (stop_run writes no such audit row, so kill_switch_after_start
        # never fires there).
        assert resume_resp.status_code == 409, resume_resp.text
        assert resume_resp.json()["detail"] == "kill_switch_after_start"

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "orphaned", (
            "kill-switch must move the resuming row to orphaned (S2-01), "
            "not leave it stuck or let the resume overwrite it back to running"
        )


# ---------------------------------------------------------------------------
# WP1.8b round 2 (security re-audit WP18b-S-01/S-02/S-04): the fence,
# _reject's rollback-first ordering, and the CancelledError revert.
# ---------------------------------------------------------------------------


def _decimal(v: str) -> Any:
    from decimal import Decimal

    return Decimal(v)


class TestPD1AbaRaceFencedAgainstSupersededResume:
    def test_resume_cut_off_by_kill_switch_cannot_import_after_a_second_resume(
        self, race_app: Any
    ) -> None:
        """WP18b-S-01 PD1: A resumes; the kill switch cuts it off
        (resuming -> orphaned); B resumes and completes successfully. A's
        OWN scan (still in flight, holding its now-stale fence) must then
        be rejected with 409 'resume_state_lost' and must import NOTHING
        -- the DB's fill sum per order must equal exactly what B's
        legitimate scan imported (no double import), and at most one
        engine (B's) is ever spawned."""
        import uuid as uuid_mod

        from fastapi.testclient import TestClient

        from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        fake_exchange = FakeCCXTExchange(exchange_id="coinbase")
        fake_exchange.register_market("BTC/USD", base="BTC", quote="USD")
        fake_exchange.seed_flat_bars("BTC/USD", count=2, price=_decimal("50000"), timeframe="1h")
        order_client_id = f"{run_id}-{uuid_mod.uuid4().hex[:12]}"
        # `run.started_at` (seeded via SQL now()) is real wall-clock time,
        # far AFTER the fake exchange's synthetic bar epoch (2026-01-01) --
        # scan_and_import's since=started_at-5min window would otherwise
        # silently exclude this order. Anchor its timestamp to real "now"
        # so the scan actually finds it.
        fake_exchange.seed_exchange_order(
            client_order_id=order_client_id,
            symbol="BTC/USD",
            side="buy",
            amount=_decimal("0.01"),
            price=_decimal("50000"),
            status="closed",
            filled=_decimal("0.01"),
            timestamp_ms=int(time.time() * 1000),
            trades=[
                {
                    "id": "pd1-trade-1",
                    "price": 50000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.3, "currency": "USD"},
                    "takerOrMaker": "taker",
                }
            ],
        )

        import api.routers.runs as runs_module
        import api.services.run_recovery as run_recovery_module

        original_build_exchange = runs_module._build_live_ccxt_exchange
        original_scan = run_recovery_module.scan_and_import
        state: dict[str, Any] = {"first_fence": None}
        b_done = threading.Event()

        async def _pd1_scan(db: Any, run: Any, exchange: Any, *, fence: Any) -> Any:
            if state["first_fence"] is None:
                # This is A: remember its fence, then wait for B to fully
                # finish before (re)attempting the real scan under A's now
                # -stale fence.
                state["first_fence"] = fence
                while not b_done.is_set():
                    await asyncio.sleep(0.01)
                return await original_scan(db, run, exchange, fence=fence)
            # This is B: run the real scan immediately, then signal A.
            result = await original_scan(db, run, exchange, fence=fence)
            b_done.set()
            return result

        runs_module._build_live_ccxt_exchange = lambda settings: fake_exchange
        run_recovery_module.scan_and_import = _pd1_scan

        resume_headers = {"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY}

        try:
            with (
                patch("api.routers.runs._run_live_engine", AsyncMock()),
                concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
            ):
                a_future = pool.submit(
                    client.post,
                    f"/api/v1/runs/{run_id}/resume?mode=protective",
                    headers=resume_headers,
                )

                # Wait until A's own resume_run has committed transaction
                # (a) and reached the (now-blocked) scan.
                deadline = time.monotonic() + 10.0
                while state["first_fence"] is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert state["first_fence"] is not None, "A never reached the scan"

                kill_resp = client.post(
                    "/api/v1/emergency/kill-switch",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
                assert kill_resp.status_code == 200, kill_resp.text

                b_resp = client.post(
                    f"/api/v1/runs/{run_id}/resume?mode=protective",
                    headers=resume_headers,
                )

                a_resp = a_future.result(timeout=15.0)
        finally:
            runs_module._build_live_ccxt_exchange = original_build_exchange
            run_recovery_module.scan_and_import = original_scan

        assert b_resp.status_code == 200, b_resp.text
        assert a_resp.status_code == 409, a_resp.text
        assert a_resp.json()["detail"] == "resume_state_lost"

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "running", (
            "B's legitimate resume must be the one that ends up running"
        )

        # DB fill sum for the order equals exactly the exchange's own
        # filled quantity -- imported exactly once, by B, never by A.
        import asyncpg

        async def _fill_sum() -> Any:
            conn = await asyncpg.connect(dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT o.filled_quantity, COALESCE(SUM(f.quantity), 0) AS fill_sum "
                    "FROM orders o LEFT JOIN fills f ON f.order_id = o.id "
                    "WHERE o.client_order_id = $1 GROUP BY o.filled_quantity",
                    order_client_id,
                )
                return row
            finally:
                await conn.close()

        row = asyncio.run(_fill_sum())
        assert row is not None, "the order must have been imported exactly once (by B)"
        assert _decimal(str(row["fill_sum"])) == _decimal(str(row["filled_quantity"]))

        import api.routers.runs as runs_module2

        live_tasks = [
            t
            for rid, t in runs_module2._RUN_TASKS.items()
            if rid == str(run_id) and not t.done()
        ]
        assert len(live_tasks) <= 1


class TestS02RejectRollsBackBeforeReverting:
    def test_db_error_mid_import_rejects_with_audit_row_not_pending_rollback(
        self, race_app: Any
    ) -> None:
        """WP18b-S-02 PD2: a real DB error (a unique-constraint violation,
        simulated directly on resume_run's own session) mid-scan must
        leave the session in a state `_reject` can still use -- `_reject`
        rolling back FIRST is what makes its own audit INSERT succeed
        instead of raising PendingRollbackError and losing the audit row
        entirely (S-03)."""
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import

        async def _pd2_scan(db: Any, run: Any, exchange: Any, *, fence: Any) -> Any:
            import uuid as uuid_mod

            from sqlalchemy.exc import SQLAlchemyError

            from api.db.models import OrderORM

            dup_cid = f"{run.id}-{uuid_mod.uuid4().hex[:12]}"
            db.add(
                OrderORM(
                    id=uuid_mod.uuid4(),
                    client_order_id=dup_cid,
                    run_id=run.id,
                    symbol="BTC/USD",
                    side="buy",
                    order_type="market",
                    quantity=_decimal("0.01"),
                    status="filled",
                    filled_quantity=_decimal("0.01"),
                )
            )
            await db.flush()
            # A second row with the SAME client_order_id -> real unique
            # violation on the SAME session resume_run holds.
            db.add(
                OrderORM(
                    id=uuid_mod.uuid4(),
                    client_order_id=dup_cid,
                    run_id=run.id,
                    symbol="BTC/USD",
                    side="buy",
                    order_type="market",
                    quantity=_decimal("0.01"),
                    status="filled",
                    filled_quantity=_decimal("0.01"),
                )
            )
            try:
                await db.flush()
            except SQLAlchemyError as exc:
                from trading.recovery import ResumeRejected

                raise ResumeRejected("order_import_failed") from exc
            return None

        run_recovery_module.scan_and_import = _pd2_scan

        resume_headers = {"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY}
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                resp = client.post(
                    f"/api/v1/runs/{run_id}/resume",
                    headers=resume_headers,
                )
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"] == "order_import_failed"

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "orphaned"

        import asyncpg

        async def _fetch_audit_rows() -> list[Any]:
            conn = await asyncpg.connect(dsn)
            try:
                rows = await conn.fetch(
                    "SELECT event_type, payload FROM audit_events "
                    "WHERE resource_id = $1 ORDER BY timestamp",
                    str(run_id),
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        audit_rows = asyncio.run(_fetch_audit_rows())
        rejected_rows = [r for r in audit_rows if r["event_type"] == "run_resume_rejected"]
        assert rejected_rows, (
            "PD2: _reject's audit row must exist even after a DB error mid-import "
            f"-- got audit rows: {audit_rows!r}"
        )
        # WP18b-S-03: no stray committed run_resumed row survives a reject.
        resumed_rows = [r for r in audit_rows if r["event_type"] == "run_resumed"]
        assert resumed_rows == [], (
            f"S-03: a rejected resume must never leave a run_resumed audit row: {audit_rows!r}"
        )


class TestS04CancelledResumeReverts:
    def test_cancelled_resume_reverts_with_audit_row(self, race_app: Any) -> None:
        """WP18b-S-04: asyncio.CancelledError raised mid-scan must still
        revert 'resuming' -> 'orphaned' (fenced) and write a
        run_resume_rejected{reason: resume_cancelled} audit row -- the
        shielded revert runs (and is awaited to completion) in
        resume_run's own except-clause before the CancelledError is
        re-raised."""
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_orphaned_live_run(dsn))

        client = TestClient(race_app, raise_server_exceptions=False)

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import

        async def _cancelling_scan(db: Any, run: Any, exchange: Any, *, fence: Any) -> Any:
            raise asyncio.CancelledError()

        run_recovery_module.scan_and_import = _cancelling_scan

        resume_headers = {"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY}
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                try:
                    client.post(f"/api/v1/runs/{run_id}/resume", headers=resume_headers)
                except asyncio.CancelledError:
                    pass
        finally:
            run_recovery_module.scan_and_import = original_scan

        final_status = asyncio.run(_fetch_run_status(dsn, run_id))
        assert final_status == "orphaned"

        import asyncpg

        async def _fetch_audit_rows() -> list[Any]:
            conn = await asyncpg.connect(dsn)
            try:
                rows = await conn.fetch(
                    "SELECT event_type, payload FROM audit_events "
                    "WHERE resource_id = $1 ORDER BY timestamp",
                    str(run_id),
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        audit_rows = asyncio.run(_fetch_audit_rows())
        cancelled_rows = [
            r
            for r in audit_rows
            if r["event_type"] == "run_resume_rejected"
            and json.loads(r["payload"]).get("reason") == "resume_cancelled"
        ]
        assert cancelled_rows, f"expected a resume_cancelled audit row, got: {audit_rows!r}"


class TestS04RepeaterAlertsOnStuckResuming:
    def test_repeater_logs_critical_for_live_run_stuck_resuming(self) -> None:
        """WP18b-S-04 part 2: the orphan repeater's one-shot inner cycle
        logs critical for a live run that has sat 'resuming' longer than
        RESUMING_STUCK_ALERT_SECONDS."""
        from structlog.testing import capture_logs

        from api.services import run_recovery as run_recovery_module

        class _StaleRun:
            id = uuid.uuid4()
            status = "resuming"
            run_mode = "live"
            updated_at = datetime.now(tz=UTC) - timedelta(
                seconds=run_recovery_module.RESUMING_STUCK_ALERT_SECONDS + 5
            )

        class _FreshRun:
            id = uuid.uuid4()
            status = "resuming"
            run_mode = "live"
            updated_at = datetime.now(tz=UTC)

        class _FakeScalars:
            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def all(self) -> list[Any]:
                return self._items

        class _FakeResult:
            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def scalars(self) -> _FakeScalars:
                return _FakeScalars(self._items)

        class _FakeDb:
            def __init__(self, items: list[Any]) -> None:
                self._items = items

            async def execute(self, *args: Any, **kwargs: Any) -> _FakeResult:
                return _FakeResult(self._items)

            async def __aenter__(self) -> _FakeDb:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        fake_db = _FakeDb([_StaleRun(), _FreshRun()])

        async def _run_one_cycle() -> list[dict[str, Any]]:
            with capture_logs() as captured:

                def _factory() -> _FakeDb:
                    return fake_db

                import api.db.session as session_module

                original_get_factory = session_module.get_session_factory
                session_module.get_session_factory = lambda: _factory
                run_recovery_module_local = run_recovery_module
                try:
                    task = asyncio.ensure_future(
                        run_recovery_module_local.orphan_holding_repeater(interval_seconds=0.01)
                    )
                    await asyncio.sleep(0.05)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                finally:
                    session_module.get_session_factory = original_get_factory
            return list(captured)

        captured = asyncio.run(_run_one_cycle())
        stuck_events = [
            e
            for e in captured
            if e.get("event") == "recovery.resume_stuck" and e.get("run_id") == str(_StaleRun.id)
        ]
        assert stuck_events, f"expected a recovery.resume_stuck critical log, got: {captured!r}"
        fresh_events = [
            e for e in captured if e.get("run_id") == str(_FreshRun.id)
        ]
        assert not fresh_events, "a freshly-resuming run must not alert"
