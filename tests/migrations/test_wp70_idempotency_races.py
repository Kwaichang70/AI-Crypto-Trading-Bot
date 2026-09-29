"""
tests/migrations/test_wp70_idempotency_races.py
----------------------------------------------------
Real-Postgres concurrency tests for WP7.0 idempotency (ST-40..52,
reports/vp2-wp7.0/synthesis-spec.md §7). These MUST run against a real
PostgreSQL instance -- SQLite (used by the hermetic suite via
InMemoryIdempotencyStore) cannot prove real row-level locking / MVCC
visibility. Gated exactly like ``test_wp18a_resume_races.py``: reads
``MIGRATION_TEST_DATABASE_URL`` and SKIPS when unset.

Two test styles, matching the two kinds of races WP7.0 needs to prove:

- ST-40..45 (DB-T-07..12): STORE-level races. Two coroutines call
  ``PostgresIdempotencyStore`` methods directly via ``asyncio.gather`` in
  ONE event loop/process -- real network round trips to Postgres give
  genuine interleaving at each await point, and Postgres's own
  transaction semantics (not Python threading) resolve the race. No HTTP
  involved.
- ST-46..52 (B-T-14/15/17/18, DB-02, G-2, G-12): full end-to-end races
  through the real ``create_app()``, driven by ``httpx.AsyncClient`` +
  ``httpx.ASGITransport`` and ``asyncio.gather`` -- genuinely concurrent
  request coroutines interleaved on ONE event loop (``race_app`` fixture
  below). WP70-P-02 (disclosed in the producer report): this deliberately
  does NOT copy ``test_wp18a_resume_races.py``'s own
  ``ThreadPoolExecutor`` + ``TestClient`` pattern (multiple OS threads,
  each with its own event loop). That pattern was tried first here and
  reliably DEADLOCKED: this module's ``get_idempotency_store()`` /
  ``get_engine()`` singletons are module-global by design (SY-70-17: a
  real deployment runs one worker, one loop), and sharing that ONE
  ``AsyncEngine``/``NullPool`` across genuinely different OS threads each
  driving their own event loop hung on the second thread's very first
  claim-step connection under this test's specific same-endpoint,
  same-engine, concurrent-write shape (unlike ``test_wp18a_resume_races``'s
  own two tests, which race two DIFFERENT endpoints and never hit this).
  Driving both requests as coroutines on ONE loop sidesteps that hazard
  entirely and is also the more faithful model of SY-70-17's real
  topology. A same-loop ``_AsyncBarrier`` (below) forces genuine overlap
  for ST-46/47/49 where the spec calls for it; ST-48/50/51/52 do not need
  one.

WP7.0 round 2 (WP70-S-01, security-audit-specialist's adversarial S2):
``TestWP70S01AmbiguousCommitLeavesNoPhantomRun`` is not one of the
original ST-30..52 IDs -- it was added directly in response to the
security report's own S2 scenario (the server-side COMMIT for
``await db.commit()`` genuinely succeeds, then the driver/client side
raises anyway) and proves the fix: no durable ``running`` row survives
without an engine task, and a same-key replay reports ``status: error``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

_MIGRATION_URL = None  # set at import time below, after the skip check


def _read_migration_url() -> str | None:
    import os

    return os.environ.get("MIGRATION_TEST_DATABASE_URL")


_MIGRATION_URL = _read_migration_url()

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- these tests need a real "
        "Postgres instance (real row-level locking / MVCC) and are "
        "skipped in environments without one. See "
        "test_017_orphaned_status.py's module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRM_TOKEN = "wp70-race-confirm-token"  # noqa: S105 -- test fixture


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


def _alembic_config(database_url: str) -> Any:
    import os

    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    return cfg


# ---------------------------------------------------------------------------
# Direct asyncpg helpers
# ---------------------------------------------------------------------------
async def _connect(dsn: str) -> Any:
    import asyncpg

    return await asyncpg.connect(dsn)


async def _count_runs(dsn: str) -> int:
    conn = await _connect(dsn)
    try:
        return int(await conn.fetchval("SELECT count(*) FROM runs"))
    finally:
        await conn.close()


async def _run_ids(dsn: str) -> list[uuid.UUID]:
    conn = await _connect(dsn)
    try:
        rows = await conn.fetch("SELECT id FROM runs")
        return [r["id"] for r in rows]
    finally:
        await conn.close()


async def _count_audit_events(dsn: str, event_type: str) -> int:
    conn = await _connect(dsn)
    try:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM audit_events WHERE event_type = $1", event_type
            )
        )
    finally:
        await conn.close()


async def _fetch_idempotency_row(dsn: str, key: uuid.UUID) -> dict[str, Any] | None:
    conn = await _connect(dsn)
    try:
        row = await conn.fetchrow(
            "SELECT * FROM idempotency_keys WHERE key = $1", key
        )
        return dict(row) if row is not None else None
    finally:
        await conn.close()


async def _seed_idempotency_row(
    dsn: str,
    key: uuid.UUID,
    *,
    endpoint: str,
    fingerprint: str,
    status: str,
    claimed_run_id: uuid.UUID,
    run_id: uuid.UUID | None = None,
    updated_at: datetime | None = None,
    created_at: datetime | None = None,
) -> None:
    conn = await _connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO idempotency_keys
                (key, endpoint, request_fingerprint, status, claimed_run_id,
                 run_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            key,
            endpoint,
            fingerprint,
            status,
            claimed_run_id,
            run_id,
            created_at or datetime.now(UTC),
            updated_at or datetime.now(UTC),
        )
    finally:
        await conn.close()


async def _seed_run(dsn: str) -> uuid.UUID:
    conn = await _connect(dsn)
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


# ---------------------------------------------------------------------------
# ST-40..45: store-level races (PostgresIdempotencyStore against real PG,
# one event loop, asyncio.gather).
# ---------------------------------------------------------------------------
@pytest.fixture()
def pg_store(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real ``PostgresIdempotencyStore`` bound to the scratch DB, with
    migrations applied. No app, no HTTP -- direct store-level calls."""
    from alembic import command
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from api.services.idempotency import PostgresIdempotencyStore

    database_url = _MIGRATION_URL
    assert database_url is not None
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    from sqlalchemy.ext.asyncio import async_sessionmaker

    engine = create_async_engine(database_url, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, class_=__import__(
        "sqlalchemy.ext.asyncio", fromlist=["AsyncSession"]
    ).AsyncSession, expire_on_commit=False)

    store = PostgresIdempotencyStore(factory, stale_after_seconds=1.0)
    yield store

    asyncio.run(engine.dispose())


class TestST40ConcurrentInsertRace:
    def test_two_concurrent_claims_same_key_exactly_one_owned(self, pg_store: Any) -> None:
        key = uuid.uuid4()

        async def _race() -> tuple[Any, Any]:
            return await asyncio.gather(
                pg_store.claim(key=key, endpoint="e", fingerprint="fp"),
                pg_store.claim(key=key, endpoint="e", fingerprint="fp"),
                return_exceptions=True,
            )

        start = time.monotonic()
        results = asyncio.run(_race())
        elapsed = time.monotonic() - start

        from api.services.idempotency import Owned

        owned_count = sum(1 for r in results if isinstance(r, Owned))
        assert owned_count == 1, results
        # The loser must fail fast (409), not block for a long time.
        assert elapsed < 2.0, f"claim race took {elapsed:.3f}s"


class TestST41ConcurrentReclaimRace:
    def test_two_concurrent_reclaims_of_failed_row_exactly_one_hit(
        self, pg_store: Any
    ) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint="e",
                fingerprint="fp",
                status="failed",
                claimed_run_id=uuid.uuid4(),
            )
        )

        async def _race() -> tuple[Any, Any]:
            return await asyncio.gather(
                pg_store.claim(key=key, endpoint="e", fingerprint="fp"),
                pg_store.claim(key=key, endpoint="e", fingerprint="fp"),
                return_exceptions=True,
            )

        results = asyncio.run(_race())
        from api.services.idempotency import Owned

        owned_count = sum(1 for r in results if isinstance(r, Owned))
        assert owned_count == 1, results


class TestST42FencingRegression:
    def test_completion_fenced_on_stale_claimed_run_id_gets_ownership_lost(
        self, pg_store: Any
    ) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = uuid.uuid4()
        p1 = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint="e",
                fingerprint="fp",
                status="in_progress",
                claimed_run_id=p1,
                updated_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )

        from api.services.idempotency import Owned

        result = asyncio.run(pg_store.claim(key=key, endpoint="e", fingerprint="fp"))
        assert isinstance(result, Owned)
        p2 = result.claimed_run_id
        assert p2 != p1

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from api.services.idempotency import OwnershipLost

        async def _attempt_stale_complete() -> None:
            engine = create_async_engine(_MIGRATION_URL, poolclass=NullPool)
            factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
            try:
                async with factory() as session:
                    with pytest.raises(OwnershipLost):
                        await pg_store.complete(
                            session, key=key, claimed_run_id=p1, status_code=201
                        )
            finally:
                await engine.dispose()

        asyncio.run(_attempt_stale_complete())

        row = asyncio.run(_fetch_idempotency_row(dsn, key))
        assert row is not None
        assert row["claimed_run_id"] == p2, "P2's claim must be untouched by P1's fenced attempt"
        assert row["status"] == "in_progress"


class TestST43StaleReconciliation:
    def test_step_a_backfills_when_run_exists(self, pg_store: Any) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_run(dsn))
        key = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint="e",
                fingerprint="fp",
                status="in_progress",
                claimed_run_id=run_id,
                updated_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )
        from api.services.idempotency import Replay

        result = asyncio.run(pg_store.claim(key=key, endpoint="e", fingerprint="fp"))
        assert isinstance(result, Replay)
        assert result.run_id == run_id

    def test_step_b_reclaims_when_run_missing(self, pg_store: Any) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = uuid.uuid4()
        missing_run_id = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint="e",
                fingerprint="fp",
                status="in_progress",
                claimed_run_id=missing_run_id,
                updated_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )
        from api.services.idempotency import Owned

        result = asyncio.run(pg_store.claim(key=key, endpoint="e", fingerprint="fp"))
        assert isinstance(result, Owned)
        assert result.claimed_run_id != missing_run_id


class TestST44StalenessClockIsUpdatedAt:
    def test_old_created_at_fresh_updated_at_is_not_stale(self, pg_store: Any) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint="e",
                fingerprint="fp",
                status="in_progress",
                claimed_run_id=uuid.uuid4(),
                created_at=datetime.now(UTC) - timedelta(days=30),
                updated_at=datetime.now(UTC),
            )
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(pg_store.claim(key=key, endpoint="e", fingerprint="fp"))
        assert exc_info.value.status_code == 409


class TestST45PruneConcurrentWithClaim:
    def test_prune_does_not_interfere_with_a_concurrent_claim_on_another_key(
        self, pg_store: Any
    ) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        from api.services.idempotency import prune_expired_idempotency_keys

        # Seed 20 long-expired rows for the prune to sweep.
        for _ in range(20):
            asyncio.run(
                _seed_idempotency_row(
                    dsn,
                    uuid.uuid4(),
                    endpoint="e",
                    fingerprint="fp",
                    status="completed",
                    claimed_run_id=uuid.uuid4(),
                    updated_at=datetime.now(UTC) - timedelta(hours=48),
                )
            )

        fresh_key = uuid.uuid4()

        async def _both() -> tuple[Any, Any]:
            return await asyncio.gather(
                prune_expired_idempotency_keys(
                    pg_store._session_factory, ttl_hours=24, batch=5
                ),
                pg_store.claim(key=fresh_key, endpoint="e", fingerprint="fp"),
            )

        start = time.monotonic()
        deleted, claim_result = asyncio.run(_both())
        elapsed = time.monotonic() - start

        assert deleted == 20
        from api.services.idempotency import Owned

        assert isinstance(claim_result, Owned)
        assert elapsed < 2.0, f"prune+claim took {elapsed:.3f}s"

        row = asyncio.run(_fetch_idempotency_row(dsn, fresh_key))
        assert row is not None and row["status"] == "in_progress"


# ---------------------------------------------------------------------------
# ST-46..52: end-to-end HTTP races (real create_app(), NullPool, TestClient
# + ThreadPoolExecutor -- race_app fixture copied from
# test_wp18a_resume_races.py's own fixture of the same name/shape).
# ---------------------------------------------------------------------------
@pytest.fixture()
def race_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    assert _MIGRATION_URL is not None
    database_url = _MIGRATION_URL

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp70-race-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp70-race-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)

    from api.config import get_settings

    get_settings.cache_clear()

    from api.services import kill_switch as _kill_switch

    _kill_switch.reset_state_for_tests()

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    import api.db.session as session_module

    session_module._engine = create_async_engine(database_url, poolclass=NullPool)
    session_module._session_factory = None

    from api.services.idempotency import reset_idempotency_store_for_tests

    reset_idempotency_store_for_tests()

    from alembic import command

    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")

    import api.routers.runs as runs_module

    runs_module._RUN_TASKS.clear()

    from api.main import create_app

    app = create_app()

    yield app

    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()
    get_settings.cache_clear()
    reset_idempotency_store_for_tests()


def _paper_payload(key_marker: str) -> dict[str, Any]:
    return {
        "strategyName": "grid_trading",
        "strategyParams": {},
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "mode": "paper",
        "initialCapital": "10000.00",
    }


class _AsyncBarrier:
    """A same-loop async barrier (WP70-P-02): forces N concurrent
    coroutines to reach a point together before any of them proceeds.

    Deliberately NOT ``threading.Barrier`` + ``run_in_executor`` -- this
    module's own producer investigation found that a shared
    ``AsyncEngine``/``NullPool`` (the ``get_idempotency_store()``
    singleton, module-global by design for a real single-worker
    deployment, SY-70-17) used from GENUINELY DIFFERENT OS threads each
    running their OWN event loop (the ``ThreadPoolExecutor`` +
    ``TestClient`` pattern ``test_wp18a_resume_races.py`` uses for ITS
    two-DIFFERENT-endpoint races) deadlocks under this specific
    same-endpoint, same-engine, genuinely-concurrent-DB-write scenario.
    Driving both requests from coroutines on ONE event loop via
    ``httpx.ASGITransport`` + ``asyncio.gather`` avoids that cross-loop
    hazard entirely, and is a MORE faithful model of the real deployment
    topology anyway (SY-70-17: one worker, one loop, many concurrent
    request coroutines interleaved at await points -- never multiple
    OS threads each owning a separate loop).
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0
        self._event = asyncio.Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count >= self._n:
            self._event.set()
        await self._event.wait()


class TestST46TrueParallelCreateSameKey:
    def test_two_parallel_creates_same_key_exactly_one_run(self, race_app: Any) -> None:
        from api.services.idempotency import PostgresIdempotencyStore

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = str(uuid.uuid4())
        payload = _paper_payload(key)

        barrier = _AsyncBarrier(2)
        original_claim = PostgresIdempotencyStore.claim

        async def _barriered_claim(self: Any, **kwargs: Any) -> Any:
            await barrier.wait()
            return await original_claim(self, **kwargs)

        async def _run() -> tuple[Any, Any]:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with (
                    patch.object(PostgresIdempotencyStore, "claim", _barriered_claim),
                    patch("api.routers.runs._run_paper_engine", new=AsyncMock()),
                ):
                    r1c = client.post(
                        "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
                    )
                    r2c = client.post(
                        "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
                    )
                    return await asyncio.gather(r1c, r2c)

        resp1, resp2 = asyncio.run(_run())

        codes = sorted([resp1.status_code, resp2.status_code])
        assert codes in ([201, 201], [201, 409]), (
            resp1.status_code, resp1.text, resp2.status_code, resp2.text
        )

        if codes == [201, 201]:
            assert resp1.json()["id"] == resp2.json()["id"], "must never be two distinct ids"

        assert asyncio.run(_count_runs(dsn)) == 1


class TestST47TrueParallelPromoteSameKey:
    def test_two_parallel_promotes_same_source_same_key_exactly_one_live_run(
        self, race_app: Any
    ) -> None:
        from api.services.idempotency import PostgresIdempotencyStore

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]

        async def _seed_eligible_paper_run() -> uuid.UUID:
            conn = await _connect(dsn)
            try:
                run_id = uuid.uuid4()
                started = datetime.now(UTC) - timedelta(days=10)
                stopped = datetime.now(UTC)
                await conn.execute(
                    """
                    INSERT INTO runs
                        (id, run_mode, status, config, started_at, stopped_at,
                         created_at, updated_at)
                    VALUES ($1, 'paper', 'stopped', $2::jsonb, $3, $4, $3, $4)
                    """,
                    run_id,
                    (
                        '{"strategy_name": "grid_trading", "strategy_params": {}, '
                        '"symbols": ["BTC/USDT"], "timeframe": "1h", '
                        '"initial_capital": "10000.00", "allow_pyramiding": false}'
                    ),
                    started,
                    stopped,
                )
                return run_id
            finally:
                await conn.close()

        source_id = asyncio.run(_seed_eligible_paper_run())

        key = str(uuid.uuid4())
        barrier = _AsyncBarrier(2)
        original_claim = PostgresIdempotencyStore.claim

        async def _barriered_claim(self: Any, **kwargs: Any) -> Any:
            await barrier.wait()
            return await original_claim(self, **kwargs)

        from api.services.promotion_gate import PromotionEligibility

        eligible = AsyncMock(
            return_value=PromotionEligibility(eligible=True, trade_count=100, runtime_days=10.0)
        )

        async def _run() -> tuple[Any, Any]:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with (
                    patch.object(PostgresIdempotencyStore, "claim", _barriered_claim),
                    patch(
                        "api.services.promotion_gate.evaluate_paper_run_eligibility",
                        eligible,
                    ),
                    patch("api.routers.runs._run_live_engine", new=AsyncMock()),
                ):
                    headers = {
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    }
                    r1c = client.post(
                        f"/api/v1/runs/{source_id}/promote-to-live", headers=headers
                    )
                    r2c = client.post(
                        f"/api/v1/runs/{source_id}/promote-to-live", headers=headers
                    )
                    return await asyncio.gather(r1c, r2c)

        resp1, resp2 = asyncio.run(_run())

        codes = sorted([resp1.status_code, resp2.status_code])
        assert codes in ([201, 201], [201, 409]), (
            resp1.status_code, resp1.text, resp2.status_code, resp2.text
        )

        live_run_count = asyncio.run(_count_runs_matching(dsn, "live"))
        assert live_run_count == 1
        assert asyncio.run(_count_audit_events(dsn, "paper_promoted_to_live")) == 1


async def _count_runs_matching(dsn: str, run_mode: str) -> int:
    conn = await _connect(dsn)
    try:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM runs WHERE run_mode = $1", run_mode
            )
        )
    finally:
        await conn.close()


class TestST48ControlDifferentKeys:
    def test_two_parallel_creates_same_body_different_keys_two_runs(
        self, race_app: Any
    ) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        payload = _paper_payload("control")

        async def _run() -> tuple[Any, Any]:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                    r1c = client.post(
                        "/api/v1/runs",
                        json=payload,
                        headers={"Idempotency-Key": str(uuid.uuid4())},
                    )
                    r2c = client.post(
                        "/api/v1/runs",
                        json=payload,
                        headers={"Idempotency-Key": str(uuid.uuid4())},
                    )
                    return await asyncio.gather(r1c, r2c)

        resp1, resp2 = asyncio.run(_run())

        assert resp1.status_code == 201, resp1.text
        assert resp2.status_code == 201, resp2.text
        assert resp1.json()["id"] != resp2.json()["id"]
        assert asyncio.run(_count_runs(dsn)) == 2


class TestST49SlowOriginalVsStaleReclaim:
    def test_slow_original_loses_to_stale_reclaim(self, race_app: Any) -> None:
        """DB-02/G-2: R1 pauses right after its own claim (before
        completing); R2 (same key+body) reclaims the STALE claim and
        completes first. R1's own completion attempt is then fenced ->
        409, and its run row is rolled back (absent). Exactly 1 run
        (R2's) survives, and only one engine task is ever created."""
        from api.services.idempotency import PostgresIdempotencyStore, get_idempotency_store

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        key = str(uuid.uuid4())
        payload = _paper_payload(key)

        store = get_idempotency_store()
        store._stale_after_seconds = 1.0  # type: ignore[attr-defined]

        r1_claimed = asyncio.Event()
        release_r1 = asyncio.Event()
        original_claim = PostgresIdempotencyStore.claim
        call_count = {"n": 0}

        async def _r1_pauses_after_claim(self: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            is_first = call_count["n"] == 1
            result = await original_claim(self, **kwargs)
            if is_first:
                r1_claimed.set()
                await release_r1.wait()
                # Sleep past the 1s staleness window so R2's claim sees
                # R1's row as stale.
                await asyncio.sleep(1.5)
            return result

        async def _run() -> tuple[Any, Any]:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with (
                    patch.object(PostgresIdempotencyStore, "claim", _r1_pauses_after_claim),
                    patch("api.routers.runs._run_paper_engine", new=AsyncMock()),
                ):
                    r1_task = asyncio.ensure_future(
                        client.post(
                            "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
                        )
                    )
                    await asyncio.wait_for(r1_claimed.wait(), timeout=10.0)
                    await asyncio.sleep(0.2)
                    release_r1.set()
                    # R2 must wait long enough for R1's row to go stale
                    # (>1s) before its OWN claim() call reconciles it.
                    await asyncio.sleep(1.3)
                    r2_resp = await client.post(
                        "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
                    )
                    r1_resp = await asyncio.wait_for(r1_task, timeout=15.0)
                    return r1_resp, r2_resp

        r1_resp, r2_resp = asyncio.run(_run())

        assert r2_resp.status_code == 201, r2_resp.text
        assert r1_resp.status_code == 409, r1_resp.text
        assert r1_resp.json()["detail"]["code"] == "idempotency_in_progress"

        assert asyncio.run(_count_runs(dsn)) == 1
        run_ids = asyncio.run(_run_ids(dsn))
        assert uuid.UUID(r2_resp.json()["id"]) in run_ids

class TestST50RetryAfterCommitReplaysSameRow:
    def test_retry_after_server_commit_replays_same_row(self, race_app: Any) -> None:
        """B-T-15: from the server's perspective, a client that never sees
        the 201 (dropped connection) is indistinguishable from one that
        did -- the commit already happened. A same-key retry must replay
        the same run id, and the DB must hold exactly 1 row."""
        from fastapi.testclient import TestClient

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        client = TestClient(race_app, raise_server_exceptions=False)
        key = str(uuid.uuid4())
        payload = _paper_payload(key)

        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp1 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
            assert resp1.status_code == 201, resp1.text
            resp2 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
        assert resp2.status_code == 201, resp2.text
        assert resp2.json()["id"] == resp1.json()["id"]
        assert resp2.headers["Idempotent-Replay"] == "true"
        assert asyncio.run(_count_runs(dsn)) == 1


class TestST51CommitFailureNoDeadlock:
    def test_commit_failure_after_complete_ends_failed_retry_creates_one_run(
        self, race_app: Any
    ) -> None:
        """G-12: a commit failure injected right after complete() succeeds
        must not deadlock (the request finishes fast), the row ends
        'failed' (store.fail() ran in ITS OWN session, independent of the
        failed main transaction), and a retry creates exactly one run."""
        from fastapi.testclient import TestClient
        from sqlalchemy.ext.asyncio import AsyncSession

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        client = TestClient(race_app, raise_server_exceptions=False)
        key = str(uuid.uuid4())
        payload = _paper_payload(key)

        original_commit = AsyncSession.commit
        # NOTE: AsyncSession.commit() is patched at the CLASS level, which
        # also intercepts store.claim()'s OWN internal commits (its
        # dedicated sessions, SY-70-07) -- for a fresh key those complete
        # in exactly 1 commit (the winning INSERT) before create_run's
        # try block is even entered. The 2nd commit process-wide is
        # create_run's own ``await db.commit()`` (right after
        # store.complete()) -- that is the one G-12 wants to fail.
        state = {"count": 0}

        async def _commit_fails_on_second_call(self: AsyncSession) -> None:
            state["count"] += 1
            if state["count"] == 2:
                raise RuntimeError("WP7.0 ST-51: injected commit failure")
            return await original_commit(self)

        start = time.monotonic()
        with patch.object(AsyncSession, "commit", _commit_fails_on_second_call):
            resp1 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"request took {elapsed:.3f}s -- possible deadlock"
        assert resp1.status_code == 500, resp1.text

        row = asyncio.run(_fetch_idempotency_row(dsn, uuid.UUID(key)))
        assert row is not None
        assert row["status"] == "failed"

        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp2 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
        assert resp2.status_code == 201, resp2.text
        assert asyncio.run(_count_runs(dsn)) == 1


class TestST52SeededStaleReconciliationViaHttp:
    def test_stale_claimed_run_id_pointing_to_existing_run_replays_via_backfill(
        self, race_app: Any
    ) -> None:
        from fastapi.testclient import TestClient

        from api.services.idempotency import (
            CREATE_ENDPOINT,
            compute_fingerprint,
            get_idempotency_store,
        )

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        store = get_idempotency_store()
        store._stale_after_seconds = 1.0  # type: ignore[attr-defined]

        key = uuid.uuid4()
        payload = _paper_payload(str(key))
        body_snake = {
            "strategy_name": "grid_trading",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initial_capital": "10000.00",
            "backtest_start": None,
            "backtest_end": None,
            "seed": None,
            "allow_pyramiding": None,
            "enable_adaptive_learning": False,
            "auto_apply_learning": False,
        }
        fingerprint = compute_fingerprint(CREATE_ENDPOINT, {}, body_snake)

        existing_run_id = asyncio.run(_seed_run(dsn))
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint=CREATE_ENDPOINT,
                fingerprint=fingerprint,
                status="in_progress",
                claimed_run_id=existing_run_id,
                updated_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )

        client = TestClient(race_app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/runs", json=payload, headers={"Idempotency-Key": str(key)}
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["id"] == str(existing_run_id)
        assert resp.headers["Idempotent-Replay"] == "true"

    def test_stale_claimed_run_id_pointing_nowhere_reclaims_one_new_run(
        self, race_app: Any
    ) -> None:
        from fastapi.testclient import TestClient

        from api.services.idempotency import (
            CREATE_ENDPOINT,
            compute_fingerprint,
            get_idempotency_store,
        )

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        store = get_idempotency_store()
        store._stale_after_seconds = 1.0  # type: ignore[attr-defined]

        key = uuid.uuid4()
        payload = _paper_payload(str(key))
        body_snake = {
            "strategy_name": "grid_trading",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initial_capital": "10000.00",
            "backtest_start": None,
            "backtest_end": None,
            "seed": None,
            "allow_pyramiding": None,
            "enable_adaptive_learning": False,
            "auto_apply_learning": False,
        }
        fingerprint = compute_fingerprint(CREATE_ENDPOINT, {}, body_snake)

        missing_run_id = uuid.uuid4()
        asyncio.run(
            _seed_idempotency_row(
                dsn,
                key,
                endpoint=CREATE_ENDPOINT,
                fingerprint=fingerprint,
                status="in_progress",
                claimed_run_id=missing_run_id,
                updated_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )

        client = TestClient(race_app, raise_server_exceptions=False)
        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": str(key)}
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["id"] != str(missing_run_id)
        assert resp.headers["Idempotent-Replay"] == "false"
        assert asyncio.run(_count_runs(dsn)) == 1


# ---------------------------------------------------------------------------
# WP7.0 round 2 (WP70-S-01): ambiguous commit -- server-side COMMIT
# succeeds, then the driver/client side raises (mirrors the security
# report's own adversarial scenario S2). Not one of the original spec's
# ST-30..52 IDs; added directly in response to the security review.
# ---------------------------------------------------------------------------
class TestWP70S01AmbiguousCommitLeavesNoPhantomRun:
    def test_commit_succeeds_then_raises_no_phantom_running_row(
        self, race_app: Any
    ) -> None:
        """The server-side COMMIT for ``await db.commit()`` actually
        succeeds (the run row and the completed idempotency claim are
        both durably persisted, atomically), then the driver raises
        AFTER that -- e.g. a socket reset while reading the server's own
        success reply. Before the WP70-S-01 fix this left a durable
        ``running`` row with no engine task and no way to tell it apart
        from a healthy run. After the fix: the row is flipped to
        ``error`` in a dedicated session, no engine task is ever created,
        and a same-key replay reports ``status: error``, not
        ``running``."""
        from fastapi.testclient import TestClient
        from sqlalchemy.ext.asyncio import AsyncSession

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        client = TestClient(race_app, raise_server_exceptions=False)
        key = str(uuid.uuid4())
        payload = _paper_payload(key)

        original_commit = AsyncSession.commit
        # NOTE: see ST-51's identical comment -- AsyncSession.commit() is
        # patched at the CLASS level, which also intercepts store.claim()'s
        # OWN internal commits. For a fresh key those take exactly 1
        # commit (the winning INSERT) before create_run's try block is
        # even entered; the 2nd commit process-wide is create_run's own
        # ``await db.commit()`` -- the one this test targets. Unlike
        # ST-51, THIS call actually SUCCEEDS (the real commit runs and
        # completes) before the injected failure fires, faithfully
        # reproducing "the server received and applied the COMMIT, then
        # the client/driver raised anyway".
        state = {"count": 0}

        async def _commit_succeeds_then_raises(self: AsyncSession) -> None:
            state["count"] += 1
            if state["count"] == 2:
                await original_commit(self)
                raise RuntimeError(
                    "WP70-S-01 test: simulated post-commit driver failure "
                    "(the server-side COMMIT already succeeded)"
                )
            return await original_commit(self)

        with patch.object(AsyncSession, "commit", _commit_succeeds_then_raises):
            resp1 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
        assert resp1.status_code == 500, resp1.text

        # Exactly 1 run exists (the commit really did persist it), and it
        # must be 'error', never a durable 'running' row with no engine.
        run_ids = asyncio.run(_run_ids(dsn))
        assert len(run_ids) == 1
        run_id = run_ids[0]

        async def _fetch_run_status(rid: uuid.UUID) -> tuple[str, uuid.UUID | None]:
            conn = await _connect(dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT status FROM runs WHERE id = $1", rid
                )
                return (row["status"], rid) if row is not None else (None, rid)  # type: ignore[return-value]
            finally:
                await conn.close()

        status, _ = asyncio.run(_fetch_run_status(run_id))
        assert status == "error", (
            f"an ambiguous-commit row must be flipped to 'error', got {status!r}"
        )

        # No engine task was ever created for this run: the failure fires
        # inside `await db.commit()` itself, strictly BEFORE the
        # paper/live task-spawn code (SY-70-12), so this is really just a
        # sanity check that the test's own premise (failure before spawn)
        # holds and nothing downstream silently created one.
        import api.routers.runs as runs_module

        assert str(run_id) not in runs_module._RUN_TASKS

        # A same-key replay must report the run's CURRENT ('error') state,
        # never a stale/cached 'running'.
        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp2 = client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
        assert resp2.status_code == 201, resp2.text
        assert resp2.headers["Idempotent-Replay"] == "true"
        assert resp2.json()["id"] == str(run_id)
        assert resp2.json()["status"] == "error"

        # Still exactly 1 run overall -- replay never creates a second one.
        assert asyncio.run(_count_runs(dsn)) == 1


# ---------------------------------------------------------------------------
# WP7.0 round 3 (WP70-S-R2-02): asyncio.shield() must protect the
# ambiguous-commit recovery UPDATE from LEVEL-TRIGGERED cancellation (an
# anyio CancelScope, which keeps re-raising Cancelled at every checkpoint
# until the scope itself exits -- unlike a single edge-triggered
# asyncio.CancelledError). This directly exercises the exact
# ``await asyncio.shield(_shielded_recovery())`` pattern now used in both
# create_run's and promote_to_live's ``except BaseException`` blocks,
# against a real seeded 'running' row on real Postgres.
# ---------------------------------------------------------------------------
class TestWP70SR202ShieldSurvivesLevelTriggeredCancellation:
    def test_shielded_mark_run_error_completes_despite_level_triggered_cancellation(
        self, race_app: Any
    ) -> None:
        import anyio

        from api.services.idempotency import mark_run_error_after_ambiguous_commit

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_run(dsn))  # seeded with status='running'

        async def _fetch_status(rid: uuid.UUID) -> str | None:
            conn = await _connect(dsn)
            try:
                row = await conn.fetchrow("SELECT status FROM runs WHERE id = $1", rid)
                return row["status"] if row is not None else None
            finally:
                await conn.close()

        async def _scenario() -> None:
            from api.db.session import get_session_factory

            async def _shielded_recovery() -> None:
                await mark_run_error_after_ambiguous_commit(
                    get_session_factory(), run_id
                )

            # A cancelled anyio CancelScope re-raises Cancelled at EVERY
            # checkpoint for as long as it stays open -- the level-
            # triggered adversarial model the security report used.
            with anyio.CancelScope() as scope:
                scope.cancel()
                try:
                    await asyncio.shield(_shielded_recovery())
                except asyncio.CancelledError:
                    # Expected: the caller's OWN await on the shield is
                    # re-cancelled inside the still-open scope. The inner
                    # shielded task is unaffected and keeps running
                    # to completion independently -- that is exactly
                    # what this test proves below.
                    pass

            # Give the shielded inner task a chance to actually finish
            # (it is not tracked by the CancelScope above, so it was
            # never cancelled -- this is a plain, uncancelled wait for
            # it to complete, mirroring production where the request's
            # own task eventually finishes regardless of how the
            # cancelled caller's await resolved).
            await asyncio.sleep(0.5)

        asyncio.run(_scenario())

        final_status = asyncio.run(_fetch_status(run_id))
        assert final_status == "error", (
            "the shielded recovery UPDATE must complete even though the "
            f"calling scope was under level-triggered cancellation; got status={final_status!r}"
        )


# ---------------------------------------------------------------------------
# WP7.0 round 4 (WP70-S-R3-02): the security round-3 probe showed that
# TestWP70SR202 above (a "rollback does no I/O" scenario -- the session
# never touched the DB, so ``db.rollback()`` was a pure local no-op) missed
# a more realistic case: when the *rollback itself* needs to talk to the
# database (an open, dirty SQLAlchemy transaction), an UNSHIELDED
# ``await db.rollback()`` placed BEFORE ``asyncio.shield(...)`` is entered
# gets re-cancelled by a level-triggered scope before the shield -- and
# therefore the rest of recovery -- is ever reached. This test forces the
# session into exactly that state (one real statement executed on it, so
# SQLAlchemy believes a transaction is open and rollback() must send a
# real ROLLBACK over the wire) and exercises the FIXED ordering now used
# in both create_run's and promote_to_live's ``except BaseException``
# blocks: the rollback lives INSIDE the shielded coroutine, not before it.
# ---------------------------------------------------------------------------
class TestWP70SR302RollbackNeedingIOSurvivesLevelTriggeredCancellation:
    def test_rollback_inside_shield_completes_despite_level_triggered_cancellation(
        self, race_app: Any
    ) -> None:
        import anyio
        from sqlalchemy import text as _sa_text

        from api.services.idempotency import mark_run_error_after_ambiguous_commit

        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        run_id = asyncio.run(_seed_run(dsn))  # seeded with status='running'

        async def _fetch_status(rid: uuid.UUID) -> str | None:
            conn = await _connect(dsn)
            try:
                row = await conn.fetchrow("SELECT status FROM runs WHERE id = $1", rid)
                return row["status"] if row is not None else None
            finally:
                await conn.close()

        async def _scenario() -> None:
            from api.db.session import get_session_factory

            factory = get_session_factory()
            db = factory()

            # Force the session into a genuinely "dirty, transaction-open"
            # state: SQLAlchemy's AsyncSession autobegins on first use, so
            # this SELECT means db.rollback() below is NOT a local no-op --
            # it must send a real ROLLBACK to Postgres over the connection.
            await db.execute(_sa_text("SELECT 1"))

            # This mirrors the FIXED runs.py ordering exactly: the rollback
            # is the first statement INSIDE the shielded coroutine, wrapped
            # in its own try/except so a failure there can never block the
            # rest of recovery (WP70-S-R3-02's remediation).
            async def _shielded_recovery() -> None:
                try:
                    await db.rollback()
                except Exception as rollback_exc:
                    # Matches runs.py's own "never let a rollback failure
                    # block recovery" contract (WP70-S-R3-02); logging
                    # (not a bare pass) also keeps this out of ruff's S110.
                    print(f"test rollback_failed_in_recovery: {rollback_exc!r}")
                await mark_run_error_after_ambiguous_commit(
                    get_session_factory(), run_id
                )

            # Level-triggered: the scope is already cancelled before the
            # shielded call is even awaited, and stays cancelled throughout
            # -- the same adversarial model TestWP70SR202 uses above.
            with anyio.CancelScope() as scope:
                scope.cancel()
                try:
                    await asyncio.shield(_shielded_recovery())
                except asyncio.CancelledError:
                    # Expected: only the caller's OWN await on the shield
                    # is re-cancelled. The inner shielded task (rollback
                    # AND the mark-error UPDATE together) is unaffected.
                    pass

            # Let the shielded inner task finish; it was never itself
            # cancelled, so this is a plain uncancelled wait.
            await asyncio.sleep(0.5)
            await db.close()

        asyncio.run(_scenario())

        final_status = asyncio.run(_fetch_status(run_id))
        assert final_status == "error", (
            "with the rollback moved INSIDE the shield (WP70-S-R3-02), "
            "recovery must complete even when the rollback itself needs "
            f"real DB I/O and the caller is under level-triggered cancellation; "
            f"got status={final_status!r}"
        )
