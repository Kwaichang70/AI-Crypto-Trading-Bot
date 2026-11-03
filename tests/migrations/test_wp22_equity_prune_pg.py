"""
tests/migrations/test_wp22_equity_prune_pg.py
---------------------------------------------
WP2.2 P0 (SY-22-10, WP22-T-12): real-Postgres proof of the equity-snapshot
retention policy implemented by ``api.services.retention``:

- backtest run created now, snapshots stamped 2 years ago -> 0 rows deleted;
- backtest run with ``created_at = now - 91d`` -> all its rows deleted
  (even snapshots whose simulated stamp is recent);
- paper / live: stamps older than 90d deleted, younger kept;
- paper / live run created long ago (> 90d) but with recent stamps -> KEPT
  (the backtest rule must never reach paper / live runs);
- the return value equals the number of rows deleted, and the single
  ``equity_prune.completed`` event carries the per-mode counts;
- concurrent pruners neither deadlock nor double count;
- the per-batch ``SET LOCAL enable_bitmapscan = off`` never leaks: on a
  ``pool_size=1`` engine ``SHOW enable_bitmapscan`` is ``on`` on the reused
  connection after a prune, on both the commit and the rollback path.

Gated exactly like ``test_wp70_idempotency_races.py``: reads
``MIGRATION_TEST_DATABASE_URL`` and SKIPS when unset. The shared
``tests/migrations/conftest.py`` autouse teardown deletes ``runs`` (and,
via ON DELETE CASCADE, ``equity_snapshots``) after every test.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- these tests need a real "
        "Postgres instance and are skipped in environments without one. "
        "See test_017_orphaned_status.py's module docstring for scratch-DB setup."
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


@pytest.fixture(autouse=True)
def _migrated() -> None:
    from alembic import command

    assert _MIGRATION_URL is not None
    command.upgrade(_alembic_config(_MIGRATION_URL), "head")


async def _insert_run(conn: Any, mode: str, created_at: datetime) -> uuid.UUID:
    run_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
        VALUES ($1, $2, 'stopped', '{}'::jsonb, $3, $3, $3)
        """,
        run_id,
        mode,
        created_at,
    )
    return run_id


async def _insert_snaps(conn: Any, run_id: uuid.UUID, stamps: list[datetime]) -> None:
    await conn.executemany(
        """
        INSERT INTO equity_snapshots
            (run_id, equity, cash, unrealised_pnl, realised_pnl,
             drawdown_pct, bar_index, timestamp)
        VALUES ($1, 1000, 1000, 0, 0, 0, $2, $3)
        """,
        [(run_id, i, ts) for i, ts in enumerate(stamps)],
    )


async def _count(conn: Any, run_id: uuid.UUID) -> int:
    return int(
        await conn.fetchval(
            "SELECT count(*) FROM equity_snapshots WHERE run_id = $1", run_id
        )
    )


async def _prune(retention_days: int = 90, now: datetime | None = None) -> int:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from api.services.retention import prune_equity_snapshots

    assert _MIGRATION_URL is not None
    engine = create_async_engine(_MIGRATION_URL, poolclass=NullPool)
    try:
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        return await prune_equity_snapshots(
            factory, retention_days=retention_days, now=now
        )
    finally:
        await engine.dispose()


async def _connect() -> Any:
    import asyncpg

    assert _MIGRATION_URL is not None
    return await asyncpg.connect(_asyncpg_dsn(_MIGRATION_URL))


def test_backtest_created_now_with_old_stamps_is_kept() -> None:
    async def scenario() -> None:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            rid = await _insert_run(conn, "backtest", now)
            await _insert_snaps(conn, rid, [now - timedelta(days=730 + i) for i in range(5)])
            assert await _prune() == 0
            assert await _count(conn, rid) == 5
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_backtest_older_than_cutoff_is_deleted_entirely() -> None:
    async def scenario() -> None:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            old = await _insert_run(conn, "backtest", now - timedelta(days=91))
            # Mix of ancient and recent simulated stamps: all must go.
            await _insert_snaps(
                conn, old, [now - timedelta(days=800), now - timedelta(days=1), now]
            )
            young = await _insert_run(conn, "backtest", now - timedelta(days=89))
            await _insert_snaps(conn, young, [now - timedelta(days=900)] * 1)
            assert await _prune() == 3
            assert await _count(conn, old) == 0
            assert await _count(conn, young) == 1
        finally:
            await conn.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["paper", "live"])
def test_paper_and_live_pruned_by_snapshot_timestamp(mode: str) -> None:
    async def scenario() -> None:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            # Run created recently: only the stamp decides for paper/live.
            rid = await _insert_run(conn, mode, now - timedelta(days=1))
            await _insert_snaps(
                conn,
                rid,
                [
                    now - timedelta(days=120),
                    now - timedelta(days=91),
                    now - timedelta(days=89),
                    now - timedelta(days=10),
                ],
            )
            assert await _prune() == 2
            assert await _count(conn, rid) == 2
            remaining = await conn.fetch(
                "SELECT timestamp FROM equity_snapshots WHERE run_id = $1", rid
            )
            assert all(r["timestamp"] > now - timedelta(days=90) for r in remaining)
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_batching_and_return_value_across_modes() -> None:
    async def scenario() -> None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from api.services.retention import prune_equity_snapshots

        assert _MIGRATION_URL is not None
        now = datetime.now(UTC)
        conn = await _connect()
        engine = create_async_engine(_MIGRATION_URL, poolclass=NullPool)
        try:
            bt = await _insert_run(conn, "backtest", now - timedelta(days=200))
            await _insert_snaps(conn, bt, [now - timedelta(days=i) for i in range(7)])
            pp = await _insert_run(conn, "paper", now - timedelta(days=200))
            await _insert_snaps(conn, pp, [now - timedelta(days=100 + i) for i in range(5)])
            keep = await _insert_run(conn, "live", now)
            await _insert_snaps(conn, keep, [now - timedelta(days=i) for i in range(3)])
            factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
            # batch=3 forces multiple transactions per mode group.
            total = await prune_equity_snapshots(factory, retention_days=90, now=now, batch=3)
            assert total == 12
            assert await _count(conn, bt) == 0
            assert await _count(conn, pp) == 0
            assert await _count(conn, keep) == 3
            assert await _prune() == 0  # idempotent
        finally:
            await engine.dispose()
            await conn.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["paper", "live"])
def test_paper_and_live_old_created_at_with_recent_stamps_are_kept(mode: str) -> None:
    """WP22-P0V2-02 (CR-02): the backtest rule is keyed on runs.created_at, so a
    paper/live run created > 90d ago MUST NOT lose rows whose stamps are recent.
    Kills the mutant 'drop the run_mode filter in the backtest statement'."""

    async def scenario() -> None:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            old_run = await _insert_run(conn, mode, now - timedelta(days=400))
            recent = [now - timedelta(days=30), now - timedelta(days=5), now - timedelta(hours=1)]
            await _insert_snaps(conn, old_run, recent)
            assert await _prune() == 0
            assert await _count(conn, old_run) == 3

            # Same run also holds one genuinely old stamp: only that row goes.
            mixed = await _insert_run(conn, mode, now - timedelta(days=400))
            await _insert_snaps(conn, mixed, [now - timedelta(days=200), now - timedelta(days=1)])
            assert await _prune() == 1
            assert await _count(conn, mixed) == 1
            assert await _count(conn, old_run) == 3
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_old_paper_live_and_backtest_runs_side_by_side() -> None:
    """Old backtest is wiped, the equally old paper/live runs keep their recent rows."""

    async def scenario() -> None:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            bt = await _insert_run(conn, "backtest", now - timedelta(days=300))
            await _insert_snaps(conn, bt, [now - timedelta(days=i) for i in range(4)])
            kept: list[uuid.UUID] = []
            for mode in ("paper", "live"):
                rid = await _insert_run(conn, mode, now - timedelta(days=300))
                await _insert_snaps(conn, rid, [now - timedelta(days=i) for i in range(3)])
                kept.append(rid)
            assert await _prune() == 4
            assert await _count(conn, bt) == 0
            assert [await _count(conn, r) for r in kept] == [3, 3]
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_completed_event_carries_per_mode_counts() -> None:
    """WP22-P0V2-04 (CR-05 / SEC-02): one event, paper/live/backtest split."""

    async def scenario() -> list[Any]:
        now = datetime.now(UTC)
        conn = await _connect()
        try:
            for mode, n in (("paper", 3), ("live", 4), ("backtest", 5)):
                rid = await _insert_run(conn, mode, now - timedelta(days=200))
                await _insert_snaps(conn, rid, [now - timedelta(days=100 + i) for i in range(n)])
            with capture_logs() as logs:
                assert await _prune(now=now) == 12
            return list(logs)
        finally:
            await conn.close()

    logs = asyncio.run(scenario())
    assert [e["event"] for e in logs] == ["equity_prune.completed"]
    ev = logs[0]
    counts = (ev["deleted"], ev["deleted_paper"], ev["deleted_live"], ev["deleted_backtest"])
    assert counts == (12, 3, 4, 5)
    assert ev["retention_days"] == 90
    assert "cutoff" in ev


def test_concurrent_pruners_do_not_deadlock_or_double_count() -> None:
    """Three pruners racing (small batch, so they genuinely interleave): no error,
    and the returned counts add up to exactly the number of rows removed."""

    async def scenario() -> None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from api.services.retention import prune_equity_snapshots

        assert _MIGRATION_URL is not None
        now = datetime.now(UTC)
        conn = await _connect()
        engines = [create_async_engine(_MIGRATION_URL, poolclass=NullPool) for _ in range(3)]
        try:
            expected = 0
            for mode, n, age in (
                ("backtest", 900, 200),
                ("paper", 700, 150),
                ("live", 500, 150),
            ):
                rid = await _insert_run(conn, mode, now - timedelta(days=age))
                stamps = [now - timedelta(days=age - i / 100) for i in range(n)]
                await _insert_snaps(conn, rid, stamps)
                expected += n
            factories = [
                async_sessionmaker(bind=e, class_=AsyncSession, expire_on_commit=False)
                for e in engines
            ]
            results = await asyncio.gather(
                *(
                    prune_equity_snapshots(f, retention_days=90, now=now, batch=37)
                    for f in factories
                )
            )
            assert sum(results) == expected
            assert await conn.fetchval("SELECT count(*) FROM equity_snapshots") == 0
        finally:
            for e in engines:
                await e.dispose()
            await conn.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# WP22-P0-R2-CR-02: SET LOCAL must not leak through the connection pool
# ---------------------------------------------------------------------------
async def _pooled_scenario(*, fail_in_backtest_pass: bool) -> tuple[int, str, str, int, int]:
    """Prune on a ``pool_size=1`` engine, so the next session MUST reuse the same
    backend connection.

    Returns ``(pid_before, setting_before, setting_after, pid_after, deleted)``;
    ``deleted`` is -1 on the failing (rollback) path.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from api.services import retention

    assert _MIGRATION_URL is not None
    now = datetime.now(UTC)
    conn = await _connect()
    engine = create_async_engine(_MIGRATION_URL, pool_size=1, max_overflow=0)
    try:
        rid = await _insert_run(conn, "paper", now - timedelta(days=200))
        await _insert_snaps(conn, rid, [now - timedelta(days=100 + i) for i in range(4)])
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        async def probe() -> tuple[int, str]:
            async with factory() as s:
                pid = (await s.execute(text("SELECT pg_backend_pid()"))).scalar_one()
                setting = (await s.execute(text("SHOW enable_bitmapscan"))).scalar_one()
                return int(pid), str(setting)

        pid_before, before = await probe()
        deleted = -1
        if fail_in_backtest_pass:
            # The paper/live batches commit; the backtest batch then dies INSIDE its
            # transaction, after the SET LOCAL pin: the rollback path.
            with patch.object(
                retention, "build_backtest_prune_stmt", lambda *_a, **_k: text("SELECT 1/0")
            ):
                with pytest.raises(Exception, match="division by zero"):
                    await retention.prune_equity_snapshots(factory, retention_days=90, now=now)
        else:
            deleted = await retention.prune_equity_snapshots(factory, retention_days=90, now=now)
        pid_after, after = await probe()
        return pid_before, before, after, pid_after, deleted
    finally:
        await engine.dispose()
        await conn.close()


def test_set_local_pin_does_not_leak_to_the_reused_pooled_connection_commit_path() -> None:
    pid_before, before, after, pid_after, deleted = asyncio.run(
        _pooled_scenario(fail_in_backtest_pass=False)
    )
    assert deleted == 4  # the prune really ran (and pinned) before the probe
    assert pid_before == pid_after, "pool_size=1 must hand back the same backend"
    assert before == "on"
    assert after == "on", "SET LOCAL leaked: use SET LOCAL, never plain SET"


def test_set_local_pin_does_not_leak_to_the_reused_pooled_connection_rollback_path() -> None:
    pid_before, before, after, pid_after, _ = asyncio.run(
        _pooled_scenario(fail_in_backtest_pass=True)
    )
    assert pid_before == pid_after
    assert before == "on"
    assert after == "on"
