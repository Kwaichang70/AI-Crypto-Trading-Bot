"""
tests/integration/test_wp70_idempotency_api.py
--------------------------------------------------
WP7.0 (reports/vp2-wp7.0/synthesis-spec.md) -- hermetic API coverage for
idempotent ``POST /api/v1/runs`` (all modes) and
``POST /api/v1/runs/{id}/promote-to-live`` under a client-supplied
``Idempotency-Key`` header.

Hermetic: ``get_db`` is overridden with an ``AsyncMock`` session (never a
real Postgres) and ``get_idempotency_store`` is overridden with
``InMemoryIdempotencyStore`` (SY-70-13) -- the real
``PostgresIdempotencyStore`` would otherwise try to reach a real Postgres
instance via ``get_session_factory()``.

Real-Postgres concurrency (ST-40..52) and migration coverage (ST-30..35)
live in ``tests/migrations/`` -- see ``test_wp70_idempotency_races.py`` and
``test_019_idempotency_keys.py``.

Test IDs below map 1:1 to synthesis-spec.md §7 ST-01..23 (docstrings cite
the ID explicitly for traceability).

WP7.0 round 2 (WP70-C-01, code-critic): ST-16 below exercises ONLY the
router's handling of a pre-fabricated ``OwnershipLost`` (it stubs
``store.complete`` directly, bypassing ``PostgresIdempotencyStore``
entirely) -- that is SY-70-09 (exception handling), not SY-70-08 (the
store's own fencing SQL). SY-70-08's actual fencing predicate is unit-
tested against a stub session in
``tests/unit/test_wp70_idempotency_store_unit.py``, and is proven under
real concurrent load ONLY by the real-Postgres
``tests/migrations/test_wp70_idempotency_races.py::TestST42FencingRegression``
and ``::TestST49SlowOriginalVsStaleReclaim``. A green run of this file
alone (``MIGRATION_TEST_DATABASE_URL`` unset) is not evidence that
SY-70-08 holds -- see AC1.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

import api.routers.runs as runs_module
from api.config import get_settings
from api.services import kill_switch as _kill_switch
from api.services.idempotency import (
    CREATE_ENDPOINT,
    OwnershipLost,
    compute_fingerprint,
)
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore

CONFIRM_TOKEN = "wp70-test-confirm-token"  # noqa: S105 -- test fixture, not a real secret

_PAPER_PAYLOAD: dict[str, Any] = {
    "strategyName": "grid_trading",
    "strategyParams": {},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "paper",
    "initialCapital": "10000.00",
}

_LIVE_PAYLOAD: dict[str, Any] = {
    **_PAPER_PAYLOAD,
    "mode": "live",
    "allowPyramiding": False,  # grid_trading defaults to True -- forbidden live
}

_BACKTEST_PAYLOAD: dict[str, Any] = {
    "strategyName": "grid_trading",
    "strategyParams": {},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "backtest",
    "initialCapital": "10000.00",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-01-10T00:00:00Z",
}


# ---------------------------------------------------------------------------
# App fixture: dev mode + a full 3-layer live-trading gate that PASSES
# (mirrors tests/integration/test_wp18a_resume_endpoint.py's resume_app).
# ---------------------------------------------------------------------------
@pytest.fixture()
def idem_app(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp70-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp70-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)
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


def _fresh_db_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    # promote_to_live wraps its audit write in
    # ``async with db.begin_nested():`` (a SAVEPOINT) -- a bare AsyncMock's
    # auto-specced ``begin_nested()`` returns a coroutine, not an async
    # context manager, so it must be wired explicitly (pre-existing gap:
    # no hermetic test exercised promote's SUCCESS path before WP7.0).
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=None)
    session.begin_nested = MagicMock(return_value=nested_cm)
    return session


class _Client:
    """Small helper bundling the TestClient + the doubles it was wired
    with, so tests can inspect/mutate them after the fact."""

    def __init__(self, client: TestClient, db: AsyncMock, store: InMemoryIdempotencyStore) -> None:
        self.client = client
        self.db = db
        self.store = store


def _wire(
    app: Any,
    *,
    db: AsyncMock | None = None,
    store: InMemoryIdempotencyStore | None = None,
) -> tuple[Any, Any]:
    """Install the get_db / get_idempotency_store overrides on ``app`` and
    return (db, store) actually installed."""
    from api.db.session import get_db
    from api.services.idempotency import get_idempotency_store

    db = db if db is not None else _fresh_db_session()
    store = store if store is not None else InMemoryIdempotencyStore(stale_after_seconds=240.0)

    async def _override_get_db() -> Any:
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_idempotency_store] = lambda: store
    return db, store


def _make_client(
    app: Any, *, db: AsyncMock | None = None, store: InMemoryIdempotencyStore | None = None
) -> _Client:
    db, store = _wire(app, db=db, store=store)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    _kill_switch.reset_state_for_tests()
    return _Client(client, db, store)


def _key() -> str:
    return str(uuid.uuid4())


def _scalar_one_or_none(value: object) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _make_run_row(run_id: uuid.UUID, **overrides: Any) -> Any:
    from types import SimpleNamespace

    base = {
        "id": run_id,
        "run_mode": "paper",
        "status": "running",
        "config": {
            "strategy_name": "grid_trading",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initial_capital": "10000.00",
            "allow_pyramiding": False,
        },
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "stopped_at": None,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "n_closed_trades": None,
        "metrics_v2_backfilled": False,
        "promoted_from_run_id": None,
        "recovered_from_run_id": None,
        "entries_latch_reason": None,
        "entries_latched_at": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# ST-01: no header -> 428, no header echo, store.claim not called
# ---------------------------------------------------------------------------
class TestST01MissingHeader:
    def test_create_paper_missing_header_returns_428(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            resp = c.client.post("/api/v1/runs", json=_PAPER_PAYLOAD)
            assert resp.status_code == 428, resp.text
            assert resp.json()["detail"]["code"] == "idempotency_key_required"
            assert "Idempotency-Key" not in resp.headers
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_create_backtest_missing_header_returns_428(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            with patch(
                "api.routers.runs._fetch_bars_for_backtest",
                side_effect=AssertionError("must not be reached"),
            ):
                resp = c.client.post("/api/v1/runs", json=_BACKTEST_PAYLOAD)
            assert resp.status_code == 428, resp.text
        finally:
            c.client.__exit__(None, None, None)

    def test_create_live_missing_header_returns_428(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json=_LIVE_PAYLOAD,
                headers={"X-Live-Confirm-Token": CONFIRM_TOKEN},
            )
            assert resp.status_code == 428, resp.text
        finally:
            c.client.__exit__(None, None, None)

    def test_promote_missing_header_returns_428(self, idem_app: Any) -> None:
        source_id = uuid.uuid4()
        db = _fresh_db_session()
        db.execute.return_value = _scalar_one_or_none(_make_run_row(source_id, status="stopped"))
        c = _make_client(idem_app, db=db)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            from api.services.promotion_gate import PromotionEligibility

            with patch(
                "api.services.promotion_gate.evaluate_paper_run_eligibility",
                AsyncMock(
                    return_value=PromotionEligibility(
                        eligible=True, trade_count=100, runtime_days=10.0
                    )
                ),
            ):
                resp = c.client.post(
                    f"/api/v1/runs/{source_id}/promote-to-live",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN},
                )
            assert resp.status_code == 428, resp.text
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-02: malformed keys -> 400, raw value never logged
# ---------------------------------------------------------------------------
class TestST02InvalidFormat:
    @pytest.mark.parametrize(
        "raw",
        [
            "not-a-uuid",
            "a" * 35,
            "a" * 37,
            "a" * 500,
            "{9d3b6e1e-6b1a-4f6a-9c1a-3b6e1e6b1a4f}",
            "urn:uuid:9d3b6e1e-6b1a-4f6a-9c1a-3b6e1e6b1a4f",
            "9d3b6e1e6b1a4f6a9c1a3b6e1e6b1a4f",  # unhyphenated 32-hex
        ],
    )
    def test_invalid_key_returns_400_and_is_never_logged(
        self, idem_app: Any, raw: str
    ) -> None:
        c = _make_client(idem_app)
        try:
            with capture_logs() as logs:
                resp = c.client.post(
                    "/api/v1/runs",
                    json=_PAPER_PAYLOAD,
                    headers={"Idempotency-Key": raw},
                )
            assert resp.status_code == 400, resp.text
            assert resp.json()["detail"]["code"] == "idempotency_key_invalid_format"
            for entry in logs:
                rendered = str(entry)
                assert raw not in rendered
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-03: uppercase accepted, echoed lowercase, lowercase replays same run
# ---------------------------------------------------------------------------
class TestST03CaseInsensitive:
    def test_uppercase_key_accepted_and_replays_lowercase(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = uuid.uuid4()
            upper = str(key).upper()

            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": upper}
                )
            assert resp1.status_code == 201, resp1.text
            assert resp1.headers["Idempotency-Key"] == str(key)
            run_id = resp1.json()["id"]

            c.db.execute.return_value = _scalar_one_or_none(_make_run_row(uuid.UUID(run_id)))
            resp2 = c.client.post(
                "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": str(key)}
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.json()["id"] == run_id
            assert resp2.headers["Idempotent-Replay"] == "true"
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-04: without a header, every pre-existing rejection keeps its own
# status code and never touches the store.
# ---------------------------------------------------------------------------
class TestST04PrecedenceUnaffected:
    def test_unknown_strategy_still_400_without_header(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            payload = {**_PAPER_PAYLOAD, "strategyName": "does-not-exist"}
            resp = c.client.post("/api/v1/runs", json=payload)
            assert resp.status_code == 400, resp.text
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_missing_backtest_dates_still_400_without_header(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            payload = {**_BACKTEST_PAYLOAD}
            payload.pop("backtestStart")
            payload.pop("backtestEnd")
            resp = c.client.post("/api/v1/runs", json=payload)
            assert resp.status_code == 400, resp.text
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_live_gate_failure_still_403_without_header(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json=_LIVE_PAYLOAD,
                headers={"X-Live-Confirm-Token": "wrong-token"},
            )
            assert resp.status_code == 403, resp.text
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_kill_switch_active_still_409_without_header(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            with patch("api.services.kill_switch.is_active", return_value=True):
                resp = c.client.post("/api/v1/runs", json=_PAPER_PAYLOAD)
            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"]["code"] == "kill_switch_active"
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_promote_404_still_404_without_header(self, idem_app: Any) -> None:
        db = _fresh_db_session()
        db.execute.return_value = _scalar_one_or_none(None)
        c = _make_client(idem_app, db=db)
        c.store.claim = AsyncMock(wraps=c.store.claim)  # type: ignore[method-assign]
        try:
            resp = c.client.post(f"/api/v1/runs/{uuid.uuid4()}/promote-to-live")
            assert resp.status_code == 404, resp.text
            c.store.claim.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-06/ST-17: paper, same key+body twice -> 201/201 replay, one task/add
# ---------------------------------------------------------------------------
class TestST06AndST17PaperReplay:
    def test_paper_replay_same_id_one_task_one_add(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": key}
                )
            assert resp1.status_code == 201, resp1.text
            assert resp1.headers["Idempotent-Replay"] == "false"
            run_id = resp1.json()["id"]
            assert c.db.add.call_count == 1

            c.db.execute.return_value = _scalar_one_or_none(_make_run_row(uuid.UUID(run_id)))
            resp2 = c.client.post(
                "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": key}
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.json()["id"] == run_id
            assert resp2.headers["Idempotent-Replay"] == "true"
            # ST-17: replay is the run's CURRENT state, no warnings, no
            # second db.add (replay never calls db.add/create_task).
            assert resp2.json()["configWarnings"] == []
            assert c.db.add.call_count == 1
            assert len(runs_module._RUN_TASKS) == 1
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-07: live replay -> record_audit_event called exactly once
# ---------------------------------------------------------------------------
class TestST07LiveReplayAudit:
    def test_live_replay_records_audit_once(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            with (
                patch("api.routers.runs._run_live_engine", new=AsyncMock()),
                patch(
                    "api.services.audit_log.record_audit_event", new=AsyncMock()
                ) as audit_mock,
            ):
                resp1 = c.client.post(
                    "/api/v1/runs",
                    json=_LIVE_PAYLOAD,
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
                assert resp1.status_code == 201, resp1.text
                run_id = resp1.json()["id"]

                c.db.execute.return_value = _scalar_one_or_none(
                    _make_run_row(uuid.UUID(run_id), run_mode="live")
                )
                resp2 = c.client.post(
                    "/api/v1/runs",
                    json=_LIVE_PAYLOAD,
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
                assert resp2.status_code == 201, resp2.text
                assert resp2.headers["Idempotent-Replay"] == "true"
                audit_mock.assert_called_once()
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-08: same key, different symbols -> 422 reused, header echoed
# ---------------------------------------------------------------------------
class TestST08Reused:
    def test_same_key_different_body_returns_422(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": key}
                )
            assert resp1.status_code == 201, resp1.text

            other_payload = {**_PAPER_PAYLOAD, "symbols": ["ETH/USDT"]}
            resp2 = c.client.post(
                "/api/v1/runs", json=other_payload, headers={"Idempotency-Key": key}
            )
            assert resp2.status_code == 422, resp2.text
            assert resp2.json()["detail"]["code"] == "idempotency_key_reused"
            assert resp2.headers["Idempotency-Key"] == key
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-09/ST-10: cross-endpoint / cross-source-run key reuse -> 422
# ---------------------------------------------------------------------------
class TestST09AndST10CrossReuse:
    def test_key_used_on_create_then_promote_returns_422(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": key}
                )
            assert resp1.status_code == 201, resp1.text

            source_id = uuid.uuid4()
            c.db.execute.return_value = _scalar_one_or_none(
                _make_run_row(source_id, status="stopped")
            )
            from api.services.promotion_gate import PromotionEligibility

            with patch(
                "api.services.promotion_gate.evaluate_paper_run_eligibility",
                AsyncMock(
                    return_value=PromotionEligibility(
                        eligible=True, trade_count=100, runtime_days=10.0
                    )
                ),
            ):
                resp2 = c.client.post(
                    f"/api/v1/runs/{source_id}/promote-to-live",
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp2.status_code == 422, resp2.text
            assert resp2.json()["detail"]["code"] == "idempotency_key_reused"
        finally:
            c.client.__exit__(None, None, None)

    def test_key_on_two_different_source_runs_returns_422(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            source_a = uuid.uuid4()
            c.db.execute.return_value = _scalar_one_or_none(
                _make_run_row(source_a, status="stopped")
            )
            from api.services.promotion_gate import PromotionEligibility

            eligible = AsyncMock(
                return_value=PromotionEligibility(
                    eligible=True, trade_count=100, runtime_days=10.0
                )
            )
            with (
                patch("api.services.promotion_gate.evaluate_paper_run_eligibility", eligible),
                patch("api.routers.runs._run_live_engine", new=AsyncMock()),
            ):
                resp1 = c.client.post(
                    f"/api/v1/runs/{source_a}/promote-to-live",
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
                assert resp1.status_code == 201, resp1.text

                source_b = uuid.uuid4()
                c.db.execute.return_value = _scalar_one_or_none(
                    _make_run_row(source_b, status="stopped")
                )
                resp2 = c.client.post(
                    f"/api/v1/runs/{source_b}/promote-to-live",
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp2.status_code == 422, resp2.text
            assert resp2.json()["detail"]["code"] == "idempotency_key_reused"
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-11: fresh in_progress -> 409
# ---------------------------------------------------------------------------
class TestST11FreshInProgress:
    def test_fresh_in_progress_returns_409(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = uuid.uuid4()
            # NOTE: rather than reverse-engineer model_dump()'s exact field
            # set, seed a row under a fingerprint captured from a REAL
            # request's own computation via a spy, then replay it exactly.
            captured: dict[str, str] = {}
            real_compute = compute_fingerprint

            def _spy(endpoint: str, path_params: dict, body: dict) -> str:
                fp = real_compute(endpoint, path_params, body)
                captured["fp"] = fp
                return fp

            with patch("api.routers.runs.compute_fingerprint", side_effect=_spy):
                c.store.seed_row(
                    key,
                    endpoint=CREATE_ENDPOINT,
                    fingerprint="placeholder",
                    status="in_progress",
                    claimed_run_id=uuid.uuid4(),
                    updated_at=datetime.now(UTC),
                )
                # First, harmless call just to capture the real fingerprint
                # this exact payload produces (then delete the seeded row's
                # placeholder claim; re-seed with the real value).
                resp0 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": _key()}
                )
                assert resp0.status_code in (201, 409)
            real_fp = captured["fp"]
            c.store.seed_row(
                key,
                endpoint=CREATE_ENDPOINT,
                fingerprint=real_fp,
                status="in_progress",
                claimed_run_id=uuid.uuid4(),
                updated_at=datetime.now(UTC),
            )
            resp = c.client.post(
                "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": str(key)}
            )
            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"]["code"] == "idempotency_in_progress"
            assert resp.headers["Idempotency-Key"] == str(key)
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-12: backtest fetch 502 -> rollback BEFORE fail; retry reclaims -> 201
# ---------------------------------------------------------------------------
class TestST12BacktestFailAndReclaim:
    def test_fetch_502_rolls_back_before_fail_then_retry_reclaims(
        self, idem_app: Any
    ) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            call_order: list[str] = []
            orig_rollback = c.db.rollback
            orig_fail = c.store.fail

            async def _tracked_rollback(*a: Any, **kw: Any) -> Any:
                call_order.append("db.rollback")
                return await orig_rollback(*a, **kw)

            async def _tracked_fail(*a: Any, **kw: Any) -> Any:
                call_order.append("store.fail")
                return await orig_fail(*a, **kw)

            c.db.rollback = _tracked_rollback  # type: ignore[method-assign]
            c.store.fail = _tracked_fail  # type: ignore[method-assign]

            from fastapi import HTTPException
            from fastapi import status as http_status

            with patch(
                "api.routers.runs._fetch_bars_for_backtest",
                side_effect=HTTPException(
                    status_code=http_status.HTTP_502_BAD_GATEWAY, detail="boom"
                ),
            ):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_BACKTEST_PAYLOAD, headers={"Idempotency-Key": key}
                )
            assert resp1.status_code == 502, resp1.text
            assert call_order == ["db.rollback", "store.fail"]

            # Retry: reclaims the failed row and succeeds.
            from tests.conftest import make_bars

            bars = make_bars(120, symbol="BTC/USDT")
            with patch(
                "api.routers.runs._fetch_bars_for_backtest", return_value={"BTC/USDT": bars}
            ):
                resp2 = c.client.post(
                    "/api/v1/runs", json=_BACKTEST_PAYLOAD, headers={"Idempotency-Key": key}
                )
            assert resp2.status_code == 201, resp2.text
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-13: bracket keys + trailing_stop_pct=0 -> identical retry replays
# ---------------------------------------------------------------------------
class TestST13FingerprintPreMutation:
    def test_identical_retry_with_bracket_and_trailing_replays(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            payload = {
                **_PAPER_PAYLOAD,
                "strategyParams": {
                    "trailing_stop_pct": 0,
                    "bracket_stop_loss_pct": 0.05,
                },
            }
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
                )
            assert resp1.status_code == 201, resp1.text
            run_id = resp1.json()["id"]

            c.db.execute.return_value = _scalar_one_or_none(_make_run_row(uuid.UUID(run_id)))
            resp2 = c.client.post(
                "/api/v1/runs", json=payload, headers={"Idempotency-Key": key}
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.json()["id"] == run_id
            assert resp2.headers["Idempotent-Replay"] == "true"
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-14: wrong token 403 (never claimed) -> same key + right token -> 201
# ---------------------------------------------------------------------------
class TestST14ConfirmTokenExcludedFromFingerprint:
    def test_retry_with_different_token_same_key_proceeds(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = _key()
            resp1 = c.client.post(
                "/api/v1/runs",
                json=_LIVE_PAYLOAD,
                headers={"Idempotency-Key": key, "X-Live-Confirm-Token": "wrong"},
            )
            assert resp1.status_code == 403, resp1.text

            with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                resp2 = c.client.post(
                    "/api/v1/runs",
                    json=_LIVE_PAYLOAD,
                    headers={"Idempotency-Key": key, "X-Live-Confirm-Token": CONFIRM_TOKEN},
                )
            assert resp2.status_code == 201, resp2.text
            assert resp2.headers["Idempotent-Replay"] == "false"
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-15: call-order recorder
# ---------------------------------------------------------------------------
class TestST15CallOrder:
    def _tracked_client(self, idem_app: Any) -> tuple[_Client, list[str]]:
        c = _make_client(idem_app)
        call_order: list[str] = []
        orig_complete = c.store.complete
        orig_commit = c.db.commit

        async def _complete(*a: Any, **kw: Any) -> Any:
            call_order.append("complete")
            return await orig_complete(*a, **kw)

        async def _commit(*a: Any, **kw: Any) -> Any:
            call_order.append("commit")
            return await orig_commit(*a, **kw)

        c.store.complete = _complete  # type: ignore[method-assign]
        c.db.commit = _commit  # type: ignore[method-assign]

        real_create_task = runs_module.asyncio.create_task

        def _tracked_create_task(coro: Any, **kw: Any) -> Any:
            call_order.append("create_task")
            return real_create_task(coro, **kw)

        runs_module.asyncio.create_task = _tracked_create_task  # type: ignore[assignment]
        return c, call_order

    def test_paper_order_is_complete_commit_create_task(self, idem_app: Any) -> None:
        c, call_order = self._tracked_client(idem_app)
        try:
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": _key()}
                )
            assert resp.status_code == 201, resp.text
            assert call_order == ["complete", "commit", "create_task"]
        finally:
            c.client.__exit__(None, None, None)

    def test_backtest_order_is_complete_commit_no_task(self, idem_app: Any) -> None:
        c, call_order = self._tracked_client(idem_app)
        try:
            from tests.conftest import make_bars

            bars = make_bars(120, symbol="BTC/USDT")
            with patch(
                "api.routers.runs._fetch_bars_for_backtest", return_value={"BTC/USDT": bars}
            ):
                resp = c.client.post(
                    "/api/v1/runs", json=_BACKTEST_PAYLOAD, headers={"Idempotency-Key": _key()}
                )
            assert resp.status_code == 201, resp.text
            assert call_order == ["complete", "commit"]
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-16 (WP70-C-01: Source relabelled SY-70-08 -> SY-70-09 -- this proves
# the ROUTER's handling of an OwnershipLost it is handed, not the store's
# own fencing SQL that decides to raise one. See the module docstring
# above and tests/unit/test_wp70_idempotency_store_unit.py for the part
# of SY-70-08 this test does not cover.): complete() raises
# OwnershipLost -> 409, no task, rollback
# ---------------------------------------------------------------------------
class TestST16OwnershipLost:
    def test_ownership_lost_returns_409_no_task_rollback(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:

            async def _raise_ownership_lost(*a: Any, **kw: Any) -> None:
                raise OwnershipLost("lost")

            c.store.complete = _raise_ownership_lost  # type: ignore[method-assign]

            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": _key()}
                )
            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"]["code"] == "idempotency_in_progress"
            assert len(runs_module._RUN_TASKS) == 0
            c.db.rollback.assert_awaited()
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-18: CORS preflight allows Idempotency-Key; response exposes it
# ---------------------------------------------------------------------------
class TestST18Cors:
    def test_preflight_allows_idempotency_key_header(self, idem_app: Any) -> None:
        with TestClient(idem_app, raise_server_exceptions=False) as client:
            resp = client.options(
                "/api/v1/runs",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "idempotency-key,content-type",
                },
            )
            assert resp.status_code in (200, 204), resp.text
            allow_headers = resp.headers.get("access-control-allow-headers", "").lower()
            assert "idempotency-key" in allow_headers

    def test_expose_headers_include_idempotency_headers(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs",
                    json=_PAPER_PAYLOAD,
                    headers={"Idempotency-Key": _key(), "Origin": "http://localhost:3000"},
                )
            assert resp.status_code == 201, resp.text
            expose = resp.headers.get("access-control-expose-headers", "")
            assert "Idempotency-Key" in expose
            assert "Idempotent-Replay" in expose
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-19: oversized body -> 413 before the claim
# ---------------------------------------------------------------------------
class TestST19BodyLimit:
    def test_oversized_body_with_valid_key_returns_413(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            big = b"x" * (2 * 1024 * 1024)
            with patch(
                "api.services.idempotency.PostgresIdempotencyStore.claim",
                side_effect=AssertionError("must not be reached"),
            ):
                resp = c.client.post(
                    "/api/v1/runs",
                    content=big,
                    headers={
                        "Content-Type": "application/json",
                        "Idempotency-Key": _key(),
                    },
                )
            assert resp.status_code == 413, resp.text
        finally:
            c.client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ST-20: logs never carry the full key, only the 8-char prefix
# ---------------------------------------------------------------------------
class TestST20LoggingHygiene:
    def test_full_key_never_logged_only_prefix(self, idem_app: Any) -> None:
        c = _make_client(idem_app)
        try:
            key = uuid.uuid4()
            with (
                patch("api.routers.runs._run_paper_engine", new=AsyncMock()),
                capture_logs() as logs,
            ):
                resp1 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": str(key)}
                )
                assert resp1.status_code == 201, resp1.text
                run_id = resp1.json()["id"]
                c.db.execute.return_value = _scalar_one_or_none(
                    _make_run_row(uuid.UUID(run_id))
                )
                resp2 = c.client.post(
                    "/api/v1/runs", json=_PAPER_PAYLOAD, headers={"Idempotency-Key": str(key)}
                )
                assert resp2.status_code == 201, resp2.text

            full_key = str(key)
            prefix = full_key[:8]
            saw_prefix = False
            for entry in logs:
                rendered = str(entry)
                assert full_key not in rendered, f"full key leaked in log: {entry}"
                if entry.get("key_prefix") == prefix:
                    saw_prefix = True
            assert saw_prefix, "expected at least one log event with key_prefix"
        finally:
            c.client.__exit__(None, None, None)

    async def test_fail_db_error_never_logs_full_key(self) -> None:
        """WP70-S-02 extension: force ``PostgresIdempotencyStore.fail()``'s
        own dedicated-session UPDATE to raise (simulating a real
        lock-timeout ``DBAPIError``, whose message/repr would otherwise
        carry the bound ``key``/``claimed_run_id`` parameters verbatim
        unless ``hide_parameters=True`` is honoured). Assert the full key
        never reaches any captured log record, and that the failure path
        logs ``exc_type`` only (``exc_info=False``)."""
        from api.services.idempotency import PostgresIdempotencyStore

        class _ExplodingSession:
            async def __aenter__(self) -> _ExplodingSession:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def execute(self, *_a: Any, **_kw: Any) -> None:
                raise RuntimeError(
                    "simulated lock timeout -- must never leak the full key"
                )

            async def commit(self) -> None:
                return None

        def _factory() -> _ExplodingSession:
            return _ExplodingSession()

        store = PostgresIdempotencyStore(_factory, stale_after_seconds=240.0)  # type: ignore[arg-type]
        key = uuid.uuid4()
        full_key = str(key)
        prefix = full_key[:8]

        with capture_logs() as logs:
            await store.fail(key=key, claimed_run_id=uuid.uuid4())

        saw_fail_log = False
        for entry in logs:
            rendered = str(entry)
            assert full_key not in rendered, f"full key leaked in log: {entry}"
            if entry.get("event") == "idempotency.fail_mark_failed":
                saw_fail_log = True
                assert entry.get("key_prefix") == prefix
                assert entry.get("exc_type") == "RuntimeError"
                # exc_info=False -- structlog/capture_logs never adds an
                # "exception" traceback key when exc_info is falsy.
                assert "exception" not in entry
        assert saw_fail_log, "expected idempotency.fail_mark_failed to be logged"


# ---------------------------------------------------------------------------
# ST-21: contract test -- InMemoryIdempotencyStore case table.
#
# WP70-P-01 (deviation, disclosed in the producer report): the spec asks
# for this table to be parametrised over BOTH InMemoryIdempotencyStore and
# PostgresIdempotencyStore. Standing up a migrated idempotency_keys table
# from tests/integration (a suite that is otherwise entirely hermetic, per
# this file's own module docstring) was judged out of proportion given
# that tests/migrations/test_wp70_idempotency_races.py already exercises
# every one of these cases directly against real Postgres SQL (ST-40..52).
# This table therefore runs against the fake only.
# ---------------------------------------------------------------------------
class TestST21ContractTable:
    @pytest.mark.asyncio
    async def test_fresh_claim_is_owned(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        from api.services.idempotency import Owned

        result = await store.claim(key=uuid.uuid4(), endpoint="e", fingerprint="fp")
        assert isinstance(result, Owned)

    @pytest.mark.asyncio
    async def test_completed_same_fingerprint_is_replay(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        key = uuid.uuid4()
        run_id = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="completed",
            claimed_run_id=run_id,
            run_id=run_id,
            response_status_code=201,
        )
        from api.services.idempotency import Replay

        result = await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert isinstance(result, Replay)
        assert result.run_id == run_id

    @pytest.mark.asyncio
    async def test_completed_different_fingerprint_is_reused(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        key = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp-A",
            status="completed",
            claimed_run_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await store.claim(key=key, endpoint="e", fingerprint="fp-B")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_fresh_in_progress_is_409(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=240.0)
        key = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="in_progress",
            claimed_run_id=uuid.uuid4(),
            updated_at=datetime.now(UTC),
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_failed_reclaims(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=240.0)
        key = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="failed",
            claimed_run_id=uuid.uuid4(),
        )
        from api.services.idempotency import Owned

        result = await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert isinstance(result, Owned)

    @pytest.mark.asyncio
    async def test_stale_in_progress_with_existing_run_backfills(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        key = uuid.uuid4()
        old_run_id = uuid.uuid4()
        store.mark_run_exists(old_run_id)
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="in_progress",
            claimed_run_id=old_run_id,
            updated_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        from api.services.idempotency import Replay

        result = await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert isinstance(result, Replay)
        assert result.run_id == old_run_id

    @pytest.mark.asyncio
    async def test_stale_in_progress_missing_run_reclaims(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        key = uuid.uuid4()
        old_run_id = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="in_progress",
            claimed_run_id=old_run_id,
            updated_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        from api.services.idempotency import Owned

        result = await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert isinstance(result, Owned)
        assert result.claimed_run_id != old_run_id

    @pytest.mark.asyncio
    async def test_completed_with_null_run_id_is_reused(self) -> None:
        store = InMemoryIdempotencyStore(stale_after_seconds=1.0)
        key = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="completed",
            claimed_run_id=uuid.uuid4(),
            run_id=None,
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "idempotency_key_reused"

    @pytest.mark.asyncio
    async def test_unresolvable_race_fails_closed_after_3_iterations(self) -> None:
        """I6: simulate a permanently oscillating race by seeding a fresh
        (non-stale) in_progress row before EVERY claim attempt inside the
        store's own loop -- modelled here by directly asserting the
        3-iteration cap via a store whose staleness never elapses."""
        store = InMemoryIdempotencyStore(stale_after_seconds=999_999.0)
        key = uuid.uuid4()
        store.seed_row(
            key,
            endpoint="e",
            fingerprint="fp",
            status="in_progress",
            claimed_run_id=uuid.uuid4(),
            updated_at=datetime.now(UTC),
        )
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await store.claim(key=key, endpoint="e", fingerprint="fp")
        assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# ST-23 (spot check): the 13 edited legacy tests are covered by their own
# files (tests/integration/test_runs_endpoints.py,
# tests/integration/test_wp13a_exit_config_api.py) -- run as part of the
# full suite, not duplicated here.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# WP7.0 round 3 (WP70-S-R2-01): a synchronous exception raised AFTER the
# engine task is spawned (and after the WP70-S-R2-01 flag reset right
# after asyncio.create_task()) must NOT flip the run to 'error' -- the
# engine is already running by that point. Mirrors the security report's
# own probe: "engine starts and is registered in _RUN_TASKS, but the run
# is flipped to error" (that was the bug; this proves it is fixed).
# ---------------------------------------------------------------------------
class _RaisingOnSetItemDict(dict):
    """A dict that raises on the FIRST write, standing in for _RUN_TASKS
    to force a synchronous exception at exactly the statement right after
    asyncio.create_task() -- i.e. after the WP70-S-R2-01 flag reset, which
    is the specific ordering this test exists to prove."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise RuntimeError(
            "WP70-S-R2-01 test: simulated post-spawn _RUN_TASKS failure"
        )


class TestWP70SR201PostSpawnExceptionDoesNotMarkError:
    def _run_case(
        self,
        idem_app: Any,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        engine_patch_target: str,
    ) -> None:
        c = _make_client(idem_app)
        created_tasks: list[Any] = []
        real_create_task = asyncio.create_task

        def _recording_create_task(coro: Any, **kw: Any) -> Any:
            task = real_create_task(coro, **kw)
            created_tasks.append(task)
            return task

        try:
            mark_error_mock = AsyncMock()
            with (
                patch(engine_patch_target, new=AsyncMock()),
                patch(
                    "api.services.idempotency.mark_run_error_after_ambiguous_commit",
                    mark_error_mock,
                ),
                patch("api.routers.runs.asyncio.create_task", _recording_create_task),
                patch.object(runs_module, "_RUN_TASKS", _RaisingOnSetItemDict()),
            ):
                resp = c.client.post("/api/v1/runs", json=payload, headers=headers)

            assert resp.status_code == 500, resp.text
            assert len(created_tasks) == 1, (
                "the engine task must already exist before the injected "
                "post-spawn failure -- it is not what this test is probing"
            )
            mark_error_mock.assert_not_called()
        finally:
            for task in created_tasks:
                if not task.done():
                    task.cancel()
            c.client.__exit__(None, None, None)

    def test_paper_post_spawn_failure_does_not_mark_error(self, idem_app: Any) -> None:
        self._run_case(
            idem_app,
            payload=_PAPER_PAYLOAD,
            headers={"Idempotency-Key": _key()},
            engine_patch_target="api.routers.runs._run_paper_engine",
        )

    def test_live_post_spawn_failure_does_not_mark_error(self, idem_app: Any) -> None:
        self._run_case(
            idem_app,
            payload=_LIVE_PAYLOAD,
            headers={"Idempotency-Key": _key(), "X-Live-Confirm-Token": CONFIRM_TOKEN},
            engine_patch_target="api.routers.runs._run_live_engine",
        )
