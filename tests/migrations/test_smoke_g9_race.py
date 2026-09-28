"""
tests/migrations/test_smoke_g9_race.py
------------------------------------------
WP-SMOKE fix F-1 (SMK-SEC-01) / SMK-T-37: real-Postgres proof that the
dialect-gated ``pg_advisory_xact_lock`` added to the G-9 smoke-live
exclusivity check in ``apps/api/routers/runs.py`` actually serialises two
concurrent ``POST /api/v1/runs`` smoke-live creates with DIFFERENT
Idempotency-Keys -- the exact TOCTOU shape SQLite/AsyncMock hermetic
tests (SMK-T-36, ``tests/integration/test_smoke_guardrails_api.py``)
cannot prove, because they never exercise real row-level locking / MVCC
visibility.

Gated exactly like ``tests/migrations/test_wp70_idempotency_races.py``'s
own ``pytestmark``: reads ``MIGRATION_TEST_DATABASE_URL`` and SKIPS when
unset. Reuses that module's fixtures/helpers by COPY (same convention
that module itself documents -- its own ``race_app`` is "copied from
``test_wp18a_resume_races.py``'s own fixture of the same name/shape";
every ``tests/migrations/*races*.py`` file in this repo self-contains
its own copy rather than cross-importing, so pytest fixture resolution
and ruff/mypy per-file analysis both stay simple), per the remediation
assignment (``reports/vp2-smoke/final-synthesis-smoke.md`` section 5,
F-1's SMK-T-37).

Race mechanics
---------------
Without the fix, the G-9 conflict SELECT and the eventual RunORM commit
have no serialisation between them: two concurrent live smoke creates
with different keys can both see zero conflicting rows and both commit,
producing two live runs (a genuine double real-money BUY). With the fix,
``pg_advisory_xact_lock`` is taken as the first statement inside the
smoke-live G-9 block, held for the rest of the transaction (released by
the eventual commit or rollback) -- so the second request blocks on the
lock BEFORE it ever runs its own conflict SELECT, and by the time it
acquires the lock (after the first request's transaction has ended) it
observes the first request's now-committed row and takes the 409 branch.

To force genuine overlap deterministically (WP70-P-02's own technique,
reused here), ``api.services.audit_log.record_audit_event`` -- called on
every successful live create, AFTER the G-9 block -- is wrapped so both
requests await a 2-party :class:`_AsyncBarrier` under
``asyncio.wait_for(..., 2.0)``, swallowing ``TimeoutError``, then call the
original. Under the fix, the second request can never even reach this
barriered call until the first request's transaction (and thus its lock)
has already ended, so only one request is ever waiting on the barrier at
a time (it times out after 2s and proceeds) -- proving the requests are
NOT running the audit-log/commit tail concurrently. Without the fix, both
requests would reach the barrier together (satisfying it immediately)
and both would commit, demonstrating the bug.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest


def _read_migration_url() -> str | None:
    import os

    return os.environ.get("MIGRATION_TEST_DATABASE_URL")


_MIGRATION_URL = _read_migration_url()

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- SMK-T-37 needs a real "
        "Postgres instance (real advisory-lock / MVCC semantics) and is "
        "skipped in environments without one. See "
        "test_wp70_idempotency_races.py's module docstring for scratch-DB "
        "setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRM_TOKEN = "wp-smoke-race-confirm-token"  # noqa: S105 -- test fixture, not a real secret


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


async def _connect(dsn: str) -> Any:
    import asyncpg

    return await asyncpg.connect(dsn)


async def _count_runs(dsn: str) -> int:
    conn = await _connect(dsn)
    try:
        return int(await conn.fetchval("SELECT count(*) FROM runs"))
    finally:
        await conn.close()


async def _fetch_idempotency_row(dsn: str, key: uuid.UUID) -> dict[str, Any] | None:
    conn = await _connect(dsn)
    try:
        row = await conn.fetchrow("SELECT * FROM idempotency_keys WHERE key = $1", key)
        return dict(row) if row is not None else None
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# race_app: copied verbatim (name/shape) from
# tests/migrations/test_wp70_idempotency_races.py, which itself copied it
# from test_wp18a_resume_races.py -- see this module's docstring.
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
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp-smoke-race-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp-smoke-race-secret")
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


class _AsyncBarrier:
    """A same-loop async barrier (WP70-P-02, copied here): forces N
    concurrent coroutines to reach a point together before any of them
    proceeds. See ``test_wp70_idempotency_races.py``'s own copy for the
    full rationale (avoids a cross-OS-thread deadlock hazard that a
    ``ThreadPoolExecutor`` + ``TestClient`` pattern hit in that module's
    own investigation)."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0
        self._event = asyncio.Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count >= self._n:
            self._event.set()
        await self._event.wait()


def _smoke_live_payload() -> dict[str, Any]:
    """The runbook body (synthesis-spec.md section 10, Run A)."""
    return {
        "strategyName": "smoke_roundtrip",
        "mode": "live",
        "symbols": ["XRP/EUR"],
        "timeframe": "5m",
        "initialCapital": "65",
        "allowPyramiding": False,
        "enableAdaptiveLearning": False,
        "strategyParams": {
            "notional_quote": 9.0,
            "hold_bars": 1,
            "exit_retry_bars": 4,
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.05,
        },
    }


class TestSMKT37AdvisoryLockRealPostgres:
    def test_two_concurrent_different_key_live_creates_exactly_one_run(
        self, race_app: Any
    ) -> None:
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        payload = _smoke_live_payload()
        key1 = str(uuid.uuid4())
        key2 = str(uuid.uuid4())

        barrier = _AsyncBarrier(2)

        from api.services import audit_log as audit_log_module

        original_record_audit_event = audit_log_module.record_audit_event

        async def _barriered_record_audit_event(db: Any, **kwargs: Any) -> Any:
            try:
                await asyncio.wait_for(barrier.wait(), timeout=2.0)
            except TimeoutError:
                pass
            return await original_record_audit_event(db, **kwargs)

        async def _run() -> tuple[Any, Any]:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with (
                    patch(
                        "api.services.audit_log.record_audit_event",
                        _barriered_record_audit_event,
                    ),
                    patch("api.routers.runs._run_live_engine", new=AsyncMock()),
                ):
                    r1c = client.post(
                        "/api/v1/runs",
                        json=payload,
                        headers={
                            "Idempotency-Key": key1,
                            "X-Live-Confirm-Token": CONFIRM_TOKEN,
                        },
                    )
                    r2c = client.post(
                        "/api/v1/runs",
                        json=payload,
                        headers={
                            "Idempotency-Key": key2,
                            "X-Live-Confirm-Token": CONFIRM_TOKEN,
                        },
                    )
                    return await asyncio.gather(r1c, r2c)

        resp1, resp2 = asyncio.run(_run())

        codes = sorted([resp1.status_code, resp2.status_code])
        assert codes == [201, 409], (
            resp1.status_code,
            resp1.text,
            resp2.status_code,
            resp2.text,
        )

        if resp1.status_code == 201:
            winner_resp, loser_resp, loser_key = resp1, resp2, key2
        else:
            winner_resp, loser_resp, loser_key = resp2, resp1, key1

        winner_id = winner_resp.json()["id"]
        loser_detail = loser_resp.json()["detail"]
        assert loser_detail["code"] == "smoke_requires_exclusive_live"
        assert winner_id in loser_detail["conflicting_run_ids"]

        # Exactly one live run row survived -- no phantom double BUY.
        assert asyncio.run(_count_runs(dsn)) == 1

        # The loser's idempotency key was released via store.fail() before
        # the 409 was raised (F-1's ordering is unchanged by the lock).
        loser_row = asyncio.run(_fetch_idempotency_row(dsn, uuid.UUID(loser_key)))
        assert loser_row is not None
        assert loser_row["status"] == "failed"

        # C-3 (unaffected by F-1): a same-key replay of the WINNER still
        # replays cleanly, never re-taking the lock or 409-ing against
        # itself (SMK-T-36(b) proves this hermetically; this closes the
        # loop against a real Postgres too).
        async def _replay() -> httpx.Response:
            transport = httpx.ASGITransport(app=race_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                    winner_key = key1 if resp1.status_code == 201 else key2
                    return await client.post(
                        "/api/v1/runs",
                        json=payload,
                        headers={
                            "Idempotency-Key": winner_key,
                            "X-Live-Confirm-Token": CONFIRM_TOKEN,
                        },
                    )

        replay_resp = asyncio.run(_replay())
        assert replay_resp.status_code in (200, 201), replay_resp.text
        assert replay_resp.headers.get("Idempotent-Replay") == "true"
        assert replay_resp.json()["id"] == winner_id
