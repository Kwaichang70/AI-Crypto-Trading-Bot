"""
tests/integration/test_wp18a_resume_endpoint.py
---------------------------------------------------
Integration tests for ``POST /api/v1/runs/{run_id}/resume`` (WP1.8a,
Verbeterplan v2 synthesis spec §5-§6).

Hermetic -- no real PostgreSQL.  The DB layer is a small dispatcher-based
fake session (``_DispatchSession``) that inspects each SQLAlchemy
statement's target table + statement type to decide what to return,
rather than a brittle ordered ``side_effect`` list -- this keeps the
tests robust to incidental extra SELECT calls (e.g. the WP1.8b scan would
add more) without needing to hand-count every call site.

Mandatory test list (synthesis spec §6):
- 404
- 403 for a missing or wrong header, and for a token supplied only in the body
- 409 when the run is not orphaned
- two concurrent resumes -> exactly one 409
- kill switch after start -> normal resume 409, protective resume 200
- the 1.8a stub -> 409
- the happy path with an injected scanner
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.selectable import Select

import api.routers.runs as runs_module
from api.config import get_settings
from api.services.run_recovery import ImportReport

CONFIRM_TOKEN = "wp18a-test-confirm-token"  # noqa: S105 -- test fixture, not a real secret
ADMIN_KEY = "wp18a-test-admin-key-hex32-0123456789abcdef"  # noqa: S105 -- test fixture
RUN_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# App fixture: dev mode + a full 3-layer live-trading gate that PASSES.
# ---------------------------------------------------------------------------


@pytest.fixture()
def resume_app(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp18a-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp18a-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)
    # WP1.8a-round2 (S-07): resume is admin-key-gated, same mechanism as
    # /emergency/kill-switch.
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    from api.main import create_app

    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_run_tasks() -> Generator[None, None, None]:
    runs_module._RUN_TASKS.clear()
    yield
    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()


@pytest.fixture(autouse=True)
def _reset_strategy_registry() -> Generator[None, None, None]:
    original = runs_module._STRATEGY_REGISTRY
    runs_module._STRATEGY_REGISTRY = {"grid_trading": MagicMock()}
    yield
    runs_module._STRATEGY_REGISTRY = original


# ---------------------------------------------------------------------------
# Dispatcher-based fake session
# ---------------------------------------------------------------------------


def _make_run_row(status: str = "orphaned", *, started_at: datetime | None = None) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=RUN_ID,
        run_mode="live",
        status=status,
        config={
            "strategy_name": "grid_trading",
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
        },
        started_at=started_at or datetime(2026, 1, 1, tzinfo=UTC),
        stopped_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_closed_trades=None,
        metrics_v2_backfilled=False,
    )


class _DispatchSession:
    """Fake AsyncSession dispatching execute() by (table, statement type).

    ``runs_update_rowcounts`` is an ORDERED queue consumed one entry per
    UPDATE-on-runs statement (CAS-to-resuming, the reject rollback, and the
    final CAS-to-running all draw from it in call order); once exhausted,
    the last value is reused.  SELECT-on-runs always returns ``run_row``;
    SELECT-on-audit_events always returns ``kill_switch_count``; every
    other SELECT (orders/fills/equity_snapshots, from load_resume_snapshot)
    returns an empty/None result -- the 1.8a tests never need real fill
    history since the exchange-scan stub rejects before it would matter,
    and the happy-path test injects a scanner that doesn't need one either.
    """

    def __init__(
        self,
        run_row: Any,
        *,
        runs_update_rowcounts: list[int] | None = None,
        kill_switch_count: int = 0,
    ) -> None:
        self.run_row = run_row
        self._runs_update_rowcounts = list(runs_update_rowcounts or [1])
        self._kill_switch_count = kill_switch_count
        self.add = MagicMock()
        self.commit = AsyncMock()
        self.flush = AsyncMock()
        self.rollback = AsyncMock()

    async def refresh(self, obj: Any) -> None:
        return None

    async def execute(self, stmt: Any, *args: Any, **kwargs: Any) -> Any:
        result = MagicMock()
        if isinstance(stmt, Update):
            table_name = stmt.table.name
            if table_name == "runs":
                rowcount = self._runs_update_rowcounts.pop(0) if self._runs_update_rowcounts else 1
                result.rowcount = rowcount
                if rowcount and self.run_row is not None:
                    # Reflect the CAS outcome so a subsequent SELECT-on-runs
                    # (there is none today, but future-proof) sees it.
                    new_status = stmt.compile().params.get("status")
                    if new_status:
                        self.run_row.status = new_status
                return result
            return result

        if isinstance(stmt, Select):
            froms = stmt.get_final_froms()
            table_name = froms[0].name if froms else None
            if table_name == "runs":
                result.scalar_one_or_none.return_value = self.run_row
                return result
            if table_name == "audit_events":
                result.scalar.return_value = self._kill_switch_count
                return result
            # orders / fills / equity_snapshots (load_resume_snapshot)
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = None
            return result

        return result


def _client(app: Any, session: _DispatchSession) -> TestClient:
    from api.db.session import get_db

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeNotFound:
    def test_unknown_run_id_returns_404(self, resume_app: Any) -> None:
        session = _DispatchSession(run_row=None)
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{uuid.uuid4()}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 404


class TestResumeGate:
    def test_missing_token_header_returns_403(self, resume_app: Any) -> None:
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(f"/api/v1/runs/{RUN_ID}/resume", headers={"X-Admin-Key": ADMIN_KEY})
        assert resp.status_code == 403

    def test_wrong_token_header_returns_403(self, resume_app: Any) -> None:
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": "not-the-right-token", "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 403

    def test_token_only_in_body_is_ignored_returns_403(self, resume_app: Any) -> None:
        """The endpoint has no body model that reads a confirm_token -- a
        token supplied only in the JSON body is never seen (SEC-004
        header-only policy), so the request is gated exactly as if no
        token were supplied at all."""
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            json={"confirm_token": CONFIRM_TOKEN},
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 403

    def test_protective_mode_is_also_token_gated(self, resume_app: Any) -> None:
        """U2: mode=protective is token-gated and always manual -- no lighter
        path exists for it."""
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume?mode=protective", headers={"X-Admin-Key": ADMIN_KEY}
        )
        assert resp.status_code == 403


class TestResumeAdminKey:
    """WP1.8a-round2 (S-07): resume additionally requires X-Admin-Key --
    same mechanism as /emergency/kill-switch -- ON TOP OF the existing
    3-layer live-trading gate. Enforced via ``dependencies=[Depends(
    require_admin)]``, resolved before the endpoint body runs, so these
    fire even with a perfectly valid X-Live-Confirm-Token present."""

    def test_missing_admin_key_returns_401(self, resume_app: Any) -> None:
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN},
        )
        assert resp.status_code == 401

    def test_wrong_admin_key_returns_403(self, resume_app: Any) -> None:
        session = _DispatchSession(run_row=_make_run_row())
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={
                "X-Live-Confirm-Token": CONFIRM_TOKEN,
                "X-Admin-Key": "definitely-wrong-admin-key",
            },
        )
        assert resp.status_code == 403


class TestResumeNotOrphaned:
    def test_running_run_returns_409(self, resume_app: Any) -> None:
        session = _DispatchSession(
            run_row=_make_run_row(status="running"), runs_update_rowcounts=[0]
        )
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 409


class TestResumeConcurrency:
    def test_two_concurrent_resumes_exactly_one_409(self, resume_app: Any) -> None:
        """Sequential-with-shared-session proxy for a true race: the first
        request's CAS-to-resuming AND final CAS-to-running both succeed
        (rowcount=1 twice); the second request's own CAS-to-resuming then
        sees the row already 'running' (not 'orphaned') and gets rowcount=0
        -- exactly one request lands on 200, the other on 409 (WP18-R-02)."""
        run_row = _make_run_row(status="orphaned")
        session = _DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1, 0])
        client = _client(resume_app, session)

        scanner_calls = {"n": 0}

        async def _fake_scan_and_import(db: Any, run: Any, exchange: Any) -> ImportReport:
            scanner_calls["n"] += 1
            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _fake_scan_and_import
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                first = client.post(
                    f"/api/v1/runs/{RUN_ID}/resume",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
                )
                second = client.post(
                    f"/api/v1/runs/{RUN_ID}/resume",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
                )
        finally:
            run_recovery_module.scan_and_import = original_scan

        statuses = sorted([first.status_code, second.status_code])
        assert statuses == [200, 409], (
            first.status_code,
            first.text,
            second.status_code,
            second.text,
        )


class TestResumeKillSwitch:
    def test_normal_resume_rejected_after_kill_switch(self, resume_app: Any) -> None:
        session = _DispatchSession(
            run_row=_make_run_row(status="orphaned"),
            runs_update_rowcounts=[1, 1],
            kill_switch_count=1,
        )
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "kill_switch_after_start"

    def test_protective_resume_allowed_after_kill_switch(self, resume_app: Any) -> None:
        run_row = _make_run_row(status="orphaned")
        session = _DispatchSession(
            run_row=run_row,
            runs_update_rowcounts=[1, 1],
            kill_switch_count=1,
        )
        client = _client(resume_app, session)

        async def _fake_scan_and_import(db: Any, run: Any, exchange: Any) -> ImportReport:
            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _fake_scan_and_import
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                resp = client.post(
                    f"/api/v1/runs/{RUN_ID}/resume?mode=protective",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
                )
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert resp.status_code == 200, resp.text


class TestResumeStubAndHappyPath:
    def test_1_8a_stub_returns_409(self, resume_app: Any) -> None:
        """No scanner injected -- the production scan_and_import stub always
        raises, so every resume in 1.8a is fail-closed 409."""
        session = _DispatchSession(
            run_row=_make_run_row(status="orphaned"), runs_update_rowcounts=[1, 1]
        )
        client = _client(resume_app, session)

        resp = client.post(
            f"/api/v1/runs/{RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "exchange_scan_not_implemented"

    def test_happy_path_with_injected_scanner_returns_200_and_spawns_task(
        self, resume_app: Any
    ) -> None:
        run_row = _make_run_row(status="orphaned")
        session = _DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
        client = _client(resume_app, session)

        async def _fake_scan_and_import(db: Any, run: Any, exchange: Any) -> ImportReport:
            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _fake_scan_and_import
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                resp = client.post(
                    f"/api/v1/runs/{RUN_ID}/resume",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
                )
        finally:
            run_recovery_module.scan_and_import = original_scan

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "running"
        assert str(RUN_ID) in runs_module._RUN_TASKS
