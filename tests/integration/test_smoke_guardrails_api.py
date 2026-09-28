"""
tests/integration/test_smoke_guardrails_api.py
--------------------------------------------------
WP-SMOKE (reports/vp2-smoke/synthesis-spec.md section 9) -- API-layer
coverage for the ``smoke_roundtrip`` guardrails wired by Producer B into
``apps/api/routers/runs.py``, ``apps/api/routers/strategies.py`` and
``apps/api/services/run_orchestrator.py``.

Test IDs below map to synthesis-spec.md section 9's merged test list:

- SMK-T-25: ``POST /runs`` -- every G-1..G-8 guardrail rejection (422
  ``smoke_guardrail_violation``), asserting the correct ``reason`` and that
  no RunORM row / audit row / idempotency claim is ever created.
- SMK-T-26: ``POST /runs`` -- G-9 live exclusivity (409
  ``smoke_requires_exclusive_live``, the claim marked failed) and the
  Replay branch is NOT subject to G-9 (a retried create with the same
  Idempotency-Key replays cleanly).
- SMK-T-27: ``POST /runs/{id}/resume`` -- G-10 (``mode=normal`` -> 422
  ``smoke_resume_protective_only``, run stays ``orphaned``);
  ``mode=protective`` is unaffected by the guard.
- SMK-T-28: ``POST /runs/{id}/promote-to-live`` -- G-11 (422
  ``smoke_promotion_forbidden``).
- SMK-T-29: ``GET /strategies`` excludes ``smoke_roundtrip`` by default,
  ``?include_diagnostic=true`` includes it, ``total == len(strategies)``
  in both cases, and ``GET /strategies/smoke_roundtrip/schema`` returns 200.
- SMK-T-30: ``POST /runs`` -- paper mode with capital 10,000 and
  ``notional_quote=9`` is accepted (no guard rejection). Backtest-mode
  guard acceptance at the same capital/notional is Producer A's SMK-T-06
  (calls ``validate_smoke_run`` directly with ``mode="backtest"``); see
  the producer report's Deviations section for why a full synchronous
  backtest execution is not re-derived here.
- SMK-T-31: ``run_live_engine`` / ``run_paper_engine`` (G-13) -- a
  violating config raises before any exchange/market-data build and the
  run ends ``error``; a protective-mode resume never calls the guard.
- SMK-T-32: paper boot recovery (G-12) -- a ``smoke_roundtrip`` paper
  orphan is marked ``error`` with reason ``smoke_not_auto_recoverable``,
  never rebuilt.
- SMK-T-33: ``DELETE /runs/{id}`` stop-while-holding is unmodified for
  smoke_roundtrip -- no flag on a live running run holding a position ->
  422 ``flatten_decision_required`` with the held symbols; ``?flatten=true``
  flattens and stops.

Hermetic throughout: no real PostgreSQL. ``get_db`` is overridden with an
``AsyncMock``/dispatcher fake session; ``get_idempotency_store`` is
overridden with ``InMemoryIdempotencyStore`` (SY-70-13) for every test that
exercises ``create_run``/``promote_to_live``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.selectable import Select

import api.routers.runs as runs_module
from api.config import get_settings
from api.db.models import RunORM
from api.services import kill_switch as _kill_switch
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore

CONFIRM_TOKEN = "wp-smoke-test-confirm-token"  # noqa: S105 -- test fixture, not a real secret
ADMIN_KEY = "wp-smoke-test-admin-key-hex32-0123456789abcdef"

_VALID_LIVE_STRATEGY_PARAMS: dict[str, Any] = {
    "notional_quote": 9.0,
    "hold_bars": 1,
    "exit_retry_bars": 4,
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.05,
}

_VALID_LIVE_BODY: dict[str, Any] = {
    "strategyName": "smoke_roundtrip",
    "strategyParams": dict(_VALID_LIVE_STRATEGY_PARAMS),
    "symbols": ["XRP/EUR"],
    "timeframe": "5m",
    "mode": "live",
    "initialCapital": "65.00",
    "allowPyramiding": False,
    "enableAdaptiveLearning": False,
}


def _smoke_body(
    *,
    mode: str = "live",
    body_overrides: dict[str, Any] | None = None,
    params_overrides: dict[str, Any] | None = None,
    drop_params: list[str] | None = None,
) -> dict[str, Any]:
    """Build a valid smoke_roundtrip request body, then apply overrides.

    Starts from ``_VALID_LIVE_BODY`` (which passes every G-1..G-8 rule) so
    each test only needs to name the ONE thing it is mutating.
    """
    body = {**_VALID_LIVE_BODY, "mode": mode}
    params = dict(_VALID_LIVE_STRATEGY_PARAMS)
    if drop_params:
        for key in drop_params:
            params.pop(key, None)
    if params_overrides:
        params.update(params_overrides)
    body["strategyParams"] = params
    if body_overrides:
        body.update(body_overrides)
    return body


# ---------------------------------------------------------------------------
# App fixture: dev mode + a full 3-layer live-trading gate that PASSES.
# Mirrors tests/integration/test_wp70_idempotency_api.py's idem_app.
# ---------------------------------------------------------------------------
@pytest.fixture()
def smoke_app(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp-smoke-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp-smoke-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)
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


# ---------------------------------------------------------------------------
# Shared helpers (create_run / promote_to_live: AsyncMock db + in-memory store)
# ---------------------------------------------------------------------------
def _fresh_db_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=None)
    session.begin_nested = MagicMock(return_value=nested_cm)
    return session


class _Client:
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


def _scalars_all(values: list[Any]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    return result


def _make_run_row(run_id: uuid.UUID, **overrides: Any) -> Any:
    from types import SimpleNamespace

    base = {
        "id": run_id,
        "run_mode": "live",
        "status": "running",
        "config": {
            "strategy_name": "smoke_roundtrip",
            "strategy_params": dict(_VALID_LIVE_STRATEGY_PARAMS),
            "symbols": ["XRP/EUR"],
            "timeframe": "5m",
            "mode": "live",
            "initial_capital": "65.00",
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


# ===========================================================================
# SMK-T-25: G-1..G-8 guardrail rejections
# ===========================================================================
_GUARD_CASES: list[tuple[str, str, dict[str, Any], dict[str, Any], list[str], str]] = [
    # (case_id, mode, body_overrides, params_overrides/drop, expected_reason)
    (
        "capital_out_of_smoke_range",
        "live",
        {"initialCapital": "66.01"},
        {},
        [],
        "capital_out_of_smoke_range",
    ),
    (
        "too_many_symbols",
        "live",
        {"symbols": ["XRP/EUR", "LTC/EUR"]},
        {},
        [],
        "too_many_symbols",
    ),
    (
        "quote_not_allowed",
        "live",
        {"symbols": ["XRP/USD"]},
        {},
        [],
        "quote_not_allowed",
    ),
    (
        "timeframe_not_allowed",
        "live",
        {"timeframe": "15m"},
        {},
        [],
        "timeframe_not_allowed",
    ),
    (
        "param_out_of_range_notional_high",
        "live",
        {},
        {"notional_quote": 9.51},
        [],
        "param_out_of_range",
    ),
    (
        "notional_exceeds_risk_ceiling",
        "paper",
        {"initialCapital": "10.00"},
        {},
        [],
        "notional_exceeds_risk_ceiling",
    ),
    (
        "stop_loss_required",
        "live",
        {},
        {},
        ["bracket_stop_loss_pct"],
        "stop_loss_required",
    ),
    (
        "stop_loss_out_of_range_low",
        "live",
        {},
        {"bracket_stop_loss_pct": 0.02},
        [],
        "stop_loss_out_of_range",
    ),
    (
        "stop_loss_out_of_range_high",
        "live",
        {},
        {"bracket_stop_loss_pct": 0.09},
        [],
        "stop_loss_out_of_range",
    ),
    (
        "take_profit_not_allowed",
        "live",
        {},
        {"bracket_take_profit_pct": 0.02},
        [],
        "take_profit_not_allowed",
    ),
    (
        "trailing_not_allowed",
        "live",
        {},
        {"trailing_stop_pct": 0.02},
        [],
        "trailing_not_allowed",
    ),
    (
        "bracket_mode_must_be_fixed",
        "live",
        {},
        {"bracket_mode": "atr"},
        [],
        "bracket_mode_must_be_fixed",
    ),
    (
        "pyramiding_not_allowed",
        "live",
        {"allowPyramiding": True},
        {},
        [],
        "pyramiding_not_allowed",
    ),
    (
        "adaptive_learning_not_allowed",
        "live",
        {"enableAdaptiveLearning": True},
        {},
        [],
        "adaptive_learning_not_allowed",
    ),
]


class TestSMKT25GuardViolations:
    @pytest.mark.parametrize(
        "case_id,mode,body_overrides,params_overrides,drop_params,expected_reason",
        _GUARD_CASES,
        ids=[c[0] for c in _GUARD_CASES],
    )
    def test_guard_rejection_returns_422_with_reason(
        self,
        smoke_app: Any,
        case_id: str,
        mode: str,
        body_overrides: dict[str, Any],
        params_overrides: dict[str, Any],
        drop_params: list[str],
        expected_reason: str,
    ) -> None:
        c = _make_client(smoke_app)
        try:
            key = _key()
            body = _smoke_body(
                mode=mode,
                body_overrides=body_overrides,
                params_overrides=params_overrides,
                drop_params=drop_params,
            )
            resp = c.client.post(
                "/api/v1/runs",
                json=body,
                headers={
                    "Idempotency-Key": key,
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 422, resp.text
            detail = resp.json()["detail"]
            assert detail["code"] == "smoke_guardrail_violation"
            reasons = {e["reason"] for e in detail["errors"]}
            assert expected_reason in reasons, (case_id, reasons)

            # I2 (WP7.0): a G-1..G-8 rejection never touches the
            # idempotency store, the DB, or the audit log.
            assert c.store.row(uuid.UUID(key)) is None
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_guard_no_op_for_non_smoke_strategy(self, smoke_app: Any) -> None:
        """Sanity check: a non-smoke strategy is never touched by the
        guard, even with a config that would violate every G-1..G-8 rule
        for smoke_roundtrip (e.g. two symbols)."""
        c = _make_client(smoke_app)
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json={
                    "strategyName": "ma_crossover",
                    "strategyParams": {},
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "timeframe": "1h",
                    "mode": "backtest",
                    "initialCapital": "10000.00",
                    "backtestStart": "2024-01-01T00:00:00Z",
                    "backtestEnd": "2024-01-02T00:00:00Z",
                },
                headers={"Idempotency-Key": _key()},
            )
            assert resp.status_code != 422 or (
                resp.json().get("detail", {}) or {}
            ).get("code") != "smoke_guardrail_violation"
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# WP-SMOKE fix F-2 (SMK-SEC-03) / SMK-T-38b: raw-JSON NaN in
# strategyParams must return 422 smoke_guardrail_violation, never a 500.
# ===========================================================================
class TestF2NonFiniteStrategyParamsAPI:
    def _post_raw(self, c: Any, body: dict[str, Any], nan_placeholder: str) -> Any:
        raw = json.dumps(body).replace(f'"{nan_placeholder}"', "NaN")
        return c.client.post(
            "/api/v1/runs",
            content=raw,
            headers={
                "Idempotency-Key": _key(),
                "X-Live-Confirm-Token": CONFIRM_TOKEN,
                "Content-Type": "application/json",
            },
        )

    def test_smk_t_38b_notional_quote_nan_returns_422(self, smoke_app: Any) -> None:
        c = _make_client(smoke_app)
        try:
            body = _smoke_body(mode="live", params_overrides={"notional_quote": "__NAN__"})
            resp = self._post_raw(c, body, "__NAN__")
            assert resp.status_code == 422, resp.text
            detail = resp.json()["detail"]
            assert detail["code"] == "smoke_guardrail_violation"
            reasons = {e["reason"] for e in detail["errors"]}
            assert "param_out_of_range" in reasons
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_smk_t_38b_bracket_stop_loss_pct_nan_returns_422(self, smoke_app: Any) -> None:
        c = _make_client(smoke_app)
        try:
            body = _smoke_body(
                mode="live", params_overrides={"bracket_stop_loss_pct": "__NAN__"}
            )
            resp = self._post_raw(c, body, "__NAN__")
            assert resp.status_code == 422, resp.text
            detail = resp.json()["detail"]
            assert detail["code"] == "smoke_guardrail_violation"
            reasons = {e["reason"] for e in detail["errors"]}
            assert "stop_loss_out_of_range" in reasons
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# WP-SMOKE fix F-2 (SMK-SEC-03) / SMK-T-39: non-finite initialCapital is a
# clean Pydantic 422, never a 500 -- for both a smoke and a non-smoke
# strategy. "65" (a normal value) still passes.
# ===========================================================================
class TestF2NonFiniteInitialCapitalAPI:
    @pytest.mark.parametrize("bad_capital", ["NaN", "sNaN", "Infinity", "-Infinity"])
    def test_smk_t_39_smoke_strategy_rejects_non_finite_capital(
        self, smoke_app: Any, bad_capital: str
    ) -> None:
        c = _make_client(smoke_app)
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live", body_overrides={"initialCapital": bad_capital}),
                headers={
                    "Idempotency-Key": _key(),
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 422, resp.text
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    @pytest.mark.parametrize("bad_capital", ["NaN", "sNaN", "Infinity", "-Infinity"])
    def test_smk_t_39_nonsmoke_strategy_rejects_non_finite_capital(
        self, smoke_app: Any, bad_capital: str
    ) -> None:
        c = _make_client(smoke_app)
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json={
                    "strategyName": "ma_crossover",
                    "strategyParams": {},
                    "symbols": ["BTC/USDT"],
                    "timeframe": "1h",
                    "mode": "backtest",
                    "initialCapital": bad_capital,
                    "backtestStart": "2024-01-01T00:00:00Z",
                    "backtestEnd": "2024-01-02T00:00:00Z",
                },
                headers={"Idempotency-Key": _key()},
            )
            assert resp.status_code == 422, resp.text
        finally:
            c.client.__exit__(None, None, None)

    def test_smk_t_39_valid_capital_still_passes(self, smoke_app: Any) -> None:
        db = _fresh_db_session()
        db.execute.return_value = _scalars_all([])
        c = _make_client(smoke_app, db=db)
        try:
            with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs",
                    json=_smoke_body(mode="live", body_overrides={"initialCapital": "65"}),
                    headers={
                        "Idempotency-Key": _key(),
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp.status_code == 201, resp.text
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# WP-SMOKE fix F-3 (SMK-SEC-05) / SMK-T-40 (API case): a 50-unknown-key
# request body stays under 4 KB.
# ===========================================================================
class TestF3BoundedUnknownParamReflectionAPI:
    def test_smk_t_40_fifty_unknown_keys_response_under_4kb(self, smoke_app: Any) -> None:
        c = _make_client(smoke_app)
        try:
            params = {f"unknown_{i}": i for i in range(50)}
            resp = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live", params_overrides=params),
                headers={
                    "Idempotency-Key": _key(),
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 422, resp.text
            assert len(resp.content) < 4096
            detail = resp.json()["detail"]
            unknown_issues = [e for e in detail["errors"] if e["reason"] == "unknown_param"]
            assert len(unknown_issues) == 11
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# SMK-T-26: G-9 live exclusivity + Replay-branch exemption
# ===========================================================================
class TestSMKT26Exclusivity:
    def test_conflicting_live_run_returns_409_and_fails_claim(self, smoke_app: Any) -> None:
        db = _fresh_db_session()
        conflicting_id = uuid.uuid4()
        db.execute.return_value = _scalars_all([conflicting_id])
        c = _make_client(smoke_app, db=db)
        try:
            key = _key()
            resp = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live"),
                headers={
                    "Idempotency-Key": key,
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 409, resp.text
            detail = resp.json()["detail"]
            assert detail["code"] == "smoke_requires_exclusive_live"
            assert str(conflicting_id) in detail["conflicting_run_ids"]

            # The claim was made (Owned) then explicitly marked failed
            # (store.fail) BEFORE raising -- a future retry with a fresh
            # key is never blocked by a stale in_progress claim, and no
            # RunORM row was ever added.
            row = c.store.row(uuid.UUID(key))
            assert row is not None
            assert row.status == "failed"
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)

    def test_replay_same_key_not_blocked_by_exclusivity(self, smoke_app: Any) -> None:
        """C-3: a network retry of the SAME smoke create (same
        Idempotency-Key) must replay the original run, never 409 against
        its own prior attempt -- even though, at replay time, that very
        run is itself a live run in 'running' state."""
        db = _fresh_db_session()
        # First call: no conflicting live runs yet.
        db.execute.return_value = _scalars_all([])
        c = _make_client(smoke_app, db=db)
        try:
            key = _key()
            with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs",
                    json=_smoke_body(mode="live"),
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp1.status_code == 201, resp1.text
            run_id = resp1.json()["id"]

            # Second call, same key+body: the Replay branch fires BEFORE
            # G-9 ever runs, even though this run itself is now 'running'.
            c.db.execute.return_value = _scalar_one_or_none(
                _make_run_row(uuid.UUID(run_id), status="running")
            )
            resp2 = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live"),
                headers={
                    "Idempotency-Key": key,
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.json()["id"] == run_id
            assert resp2.headers["Idempotent-Replay"] == "true"
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# WP-SMOKE fix F-1 (SMK-SEC-01) / SMK-T-36: dialect-gated
# pg_advisory_xact_lock in the G-9 exclusivity block. Hermetic ordering
# and dialect-gating coverage only -- the real race is exercised against
# a real Postgres by SMK-T-37 (tests/migrations/test_smoke_g9_race.py).
# ===========================================================================
class _FakeDialect:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBind:
    def __init__(self, dialect_name: str) -> None:
        self.dialect = _FakeDialect(dialect_name)


class _LockRecordingSession:
    """Fake AsyncSession reporting a PostgreSQL dialect (so F-1's
    ``_is_postgresql`` gate fires) and recording every ``execute()``
    statement, in order, as one of ``"lock"`` / ``"conflict_select"`` /
    ``"replay_select"`` / ``"other"`` -- so SMK-T-36 can assert exactly
    which statements run and in what order, without a real database."""

    def __init__(
        self,
        *,
        conflicting_ids: list[uuid.UUID] | None = None,
        replay_row: Any = None,
        dialect_name: str = "postgresql",
    ) -> None:
        self.statements: list[str] = []
        self._conflicting_ids = conflicting_ids or []
        self._replay_row = replay_row
        self.add = MagicMock()
        self.add_all = MagicMock()
        self.commit = AsyncMock()
        self.flush = AsyncMock()
        self.rollback = AsyncMock()
        self.bind = _FakeBind(dialect_name)

    async def refresh(self, obj: Any) -> None:
        return None

    async def execute(self, stmt: Any, *args: Any, **kwargs: Any) -> Any:
        result = MagicMock()
        if "pg_advisory_xact_lock" in str(stmt):
            self.statements.append("lock")
            return result
        if isinstance(stmt, Select):
            froms = stmt.get_final_froms()
            table_name = froms[0].name if froms else None
            if table_name == "runs":
                col_names = [c.name for c in stmt.selected_columns]
                if col_names == ["id"]:
                    self.statements.append("conflict_select")
                    result.scalars.return_value.all.return_value = self._conflicting_ids
                    return result
                self.statements.append("replay_select")
                result.scalar_one_or_none.return_value = self._replay_row
                return result
        self.statements.append("other")
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        return result


class TestF1AdvisoryLockSMKT36:
    def test_a_lock_runs_exactly_once_before_conflict_select(self, smoke_app: Any) -> None:
        session = _LockRecordingSession(conflicting_ids=[])
        c = _make_client(smoke_app, db=session)
        try:
            with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs",
                    json=_smoke_body(mode="live"),
                    headers={
                        "Idempotency-Key": _key(),
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp.status_code == 201, resp.text
            assert session.statements.count("lock") == 1
            assert session.statements.index("lock") < session.statements.index("conflict_select")
        finally:
            c.client.__exit__(None, None, None)

    def test_b_same_key_replay_takes_no_lock(self, smoke_app: Any) -> None:
        session = _LockRecordingSession(conflicting_ids=[])
        c = _make_client(smoke_app, db=session)
        try:
            key = _key()
            with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
                resp1 = c.client.post(
                    "/api/v1/runs",
                    json=_smoke_body(mode="live"),
                    headers={
                        "Idempotency-Key": key,
                        "X-Live-Confirm-Token": CONFIRM_TOKEN,
                    },
                )
            assert resp1.status_code == 201, resp1.text
            run_id = resp1.json()["id"]
            assert "lock" in session.statements

            session.statements.clear()
            session._replay_row = _make_run_row(uuid.UUID(run_id), status="running")
            resp2 = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live"),
                headers={
                    "Idempotency-Key": key,
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.headers["Idempotent-Replay"] == "true"
            assert "lock" not in session.statements
        finally:
            c.client.__exit__(None, None, None)

    def test_c_smoke_paper_and_nonsmoke_live_take_no_lock(self, smoke_app: Any) -> None:
        _NONSMOKE_LIVE_BODY: dict[str, Any] = {
            "strategyName": "grid_trading",
            "strategyParams": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "live",
            "initialCapital": "10000.00",
            "allowPyramiding": False,
        }
        cases = [
            (_smoke_body(mode="paper"), False),
            (_NONSMOKE_LIVE_BODY, True),
        ]
        for body, needs_token in cases:
            session = _LockRecordingSession(conflicting_ids=[])
            c = _make_client(smoke_app, db=session)
            try:
                headers = {"Idempotency-Key": _key()}
                if needs_token:
                    headers["X-Live-Confirm-Token"] = CONFIRM_TOKEN
                with (
                    patch("api.routers.runs._run_live_engine", new=AsyncMock()),
                    patch("api.routers.runs._run_paper_engine", new=AsyncMock()),
                ):
                    resp = c.client.post("/api/v1/runs", json=body, headers=headers)
                assert resp.status_code == 201, resp.text
                assert "lock" not in session.statements
            finally:
                c.client.__exit__(None, None, None)

    def test_d_default_asyncmock_session_no_lock_smk_t_26_still_holds(
        self, smoke_app: Any
    ) -> None:
        db = _fresh_db_session()
        conflicting_id = uuid.uuid4()
        db.execute.return_value = _scalars_all([conflicting_id])
        c = _make_client(smoke_app, db=db)
        try:
            resp = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live"),
                headers={
                    "Idempotency-Key": _key(),
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 409, resp.text
            # db.bind.dialect.name is a plain Mock on an AsyncMock session,
            # never == "postgresql" -- exactly ONE db.execute call (the
            # conflict SELECT), no separate advisory-lock statement.
            assert db.execute.await_count == 1
            detail = resp.json()["detail"]
            assert detail["code"] == "smoke_requires_exclusive_live"
        finally:
            c.client.__exit__(None, None, None)

    def test_e_lock_taken_on_409_path_store_fail_before_raise(self, smoke_app: Any) -> None:
        conflicting_id = uuid.uuid4()
        session = _LockRecordingSession(conflicting_ids=[conflicting_id])
        c = _make_client(smoke_app, db=session)
        try:
            key = _key()
            resp = c.client.post(
                "/api/v1/runs",
                json=_smoke_body(mode="live"),
                headers={
                    "Idempotency-Key": key,
                    "X-Live-Confirm-Token": CONFIRM_TOKEN,
                },
            )
            assert resp.status_code == 409, resp.text
            assert session.statements == ["lock", "conflict_select"]
            session.add.assert_not_called()
            row = c.store.row(uuid.UUID(key))
            assert row is not None
            assert row.status == "failed"
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# SMK-T-27: G-10 resume protective-only
# ===========================================================================
def _make_resume_run_row(
    status: str = "orphaned", *, strategy_name: str = "smoke_roundtrip"
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=_RESUME_RUN_ID,
        run_mode="live",
        status=status,
        config={
            "strategy_name": strategy_name,
            "symbols": ["XRP/EUR"],
            "timeframe": "5m",
            "initial_capital": "65.00",
            "strategy_params": dict(_VALID_LIVE_STRATEGY_PARAMS),
        },
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        stopped_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_closed_trades=None,
        metrics_v2_backfilled=False,
        entries_latch_reason=None,
        entries_latched_at=None,
    )


_RESUME_RUN_ID = uuid.uuid4()


class _DispatchSession:
    """Minimal fake AsyncSession dispatching execute() by (table, stmt
    type) -- mirrors tests/integration/test_wp18a_resume_endpoint.py's
    helper of the same name, duplicated here for file self-containment."""

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
                rowcount = (
                    self._runs_update_rowcounts.pop(0) if self._runs_update_rowcounts else 1
                )
                result.rowcount = rowcount
                if rowcount and self.run_row is not None:
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
            result.scalars.return_value.all.return_value = []
            result.scalar.return_value = None
            return result

        return result


def _resume_client(app: Any, session: _DispatchSession) -> TestClient:
    from api.db.session import get_db

    async def _override_get_db() -> Any:
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app, raise_server_exceptions=False)


class TestSMKT27ResumeProtectiveOnly:
    def test_normal_mode_returns_422_before_cas(self, smoke_app: Any) -> None:
        run_row = _make_resume_run_row(status="orphaned")
        session = _DispatchSession(run_row=run_row)
        client = _resume_client(smoke_app, session)

        resp = client.post(
            f"/api/v1/runs/{_RESUME_RUN_ID}/resume",
            headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["code"] == "smoke_resume_protective_only"
        # The run stays 'orphaned', untouched -- rejected BEFORE the CAS.
        assert run_row.status == "orphaned"

    def test_protective_mode_is_unaffected_by_guard(self, smoke_app: Any) -> None:
        """G-10 only gates mode=normal -- a protective resume of a
        smoke_roundtrip run must reach exactly as far as any other
        strategy's protective resume would (the guard is not re-run)."""
        run_row = _make_resume_run_row(status="orphaned")
        session = _DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
        client = _resume_client(smoke_app, session)

        async def _fake_scan_and_import(
            db: Any, run: Any, exchange: Any, *, fence: Any = None
        ) -> Any:
            from api.services.run_recovery import ImportReport

            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        original_scan = run_recovery_module.scan_and_import
        run_recovery_module.scan_and_import = _fake_scan_and_import
        try:
            with patch("api.routers.runs._run_live_engine", AsyncMock()):
                resp = client.post(
                    f"/api/v1/runs/{_RESUME_RUN_ID}/resume?mode=protective",
                    headers={"X-Live-Confirm-Token": CONFIRM_TOKEN, "X-Admin-Key": ADMIN_KEY},
                )
        finally:
            run_recovery_module.scan_and_import = original_scan

        # 200 (not the 422 smoke_resume_protective_only G-10 would return
        # for mode=normal) proves the guard was not re-run for this
        # protective resume -- it proceeded all the way through the CAS,
        # the (stubbed) exchange scan, and the spawned engine.
        assert resp.status_code == 200, resp.text


# ===========================================================================
# SMK-T-28: G-11 promotion ban
# ===========================================================================
class TestSMKT28PromotionForbidden:
    def test_promote_smoke_run_returns_422(self, smoke_app: Any) -> None:
        source_id = uuid.uuid4()
        db = _fresh_db_session()
        db.execute.return_value = _scalar_one_or_none(
            _make_run_row(source_id, run_mode="paper", status="stopped")
        )
        c = _make_client(smoke_app, db=db)
        try:
            resp = c.client.post(f"/api/v1/runs/{source_id}/promote-to-live")
            assert resp.status_code == 422, resp.text
            assert resp.json()["detail"]["code"] == "smoke_promotion_forbidden"
            c.db.add.assert_not_called()
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# SMK-T-29: GET /strategies include_diagnostic filter + schema endpoint
# ===========================================================================
class TestSMKT29StrategyListing:
    def test_default_list_excludes_smoke_roundtrip(self, client_dev: TestClient) -> None:
        resp = client_dev.get("/api/v1/strategies")
        assert resp.status_code == 200
        data = resp.json()
        names = {s["name"] for s in data["strategies"]}
        assert "smoke_roundtrip" not in names
        assert data["total"] == len(data["strategies"])

    def test_include_diagnostic_true_includes_smoke_roundtrip(
        self, client_dev: TestClient
    ) -> None:
        resp = client_dev.get("/api/v1/strategies", params={"include_diagnostic": "true"})
        assert resp.status_code == 200
        data = resp.json()
        names = {s["name"] for s in data["strategies"]}
        assert "smoke_roundtrip" in names
        assert data["total"] == len(data["strategies"])

    def test_schema_endpoint_returns_200_for_smoke_roundtrip(
        self, client_dev: TestClient
    ) -> None:
        resp = client_dev.get("/api/v1/strategies/smoke_roundtrip/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "smoke_roundtrip"
        assert data["status"] == "diagnostic"


# ===========================================================================
# SMK-T-30: paper mode accepted at capital 10,000 / notional 9
# ===========================================================================
class TestSMKT30PaperAccepted:
    def test_paper_capital_10000_notional_9_accepted(self, smoke_app: Any) -> None:
        c = _make_client(smoke_app)
        try:
            with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
                resp = c.client.post(
                    "/api/v1/runs",
                    json=_smoke_body(
                        mode="paper", body_overrides={"initialCapital": "10000.00"}
                    ),
                    headers={"Idempotency-Key": _key()},
                )
            assert resp.status_code == 201, resp.text
            detail = (
                resp.json().get("detail") if isinstance(resp.json(), dict) else None
            )
            assert detail is None or (
                isinstance(detail, dict) and detail.get("code") != "smoke_guardrail_violation"
            )
        finally:
            c.client.__exit__(None, None, None)


# ===========================================================================
# SMK-T-31: G-13 defence in depth in run_live_engine / run_paper_engine
# ===========================================================================
def _make_orchestrator_db_session_factory(
    run_status: str = "running",
) -> tuple[MagicMock, MagicMock]:
    mock_run = MagicMock()
    mock_run.status = run_status

    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = mock_run

    db_mock = MagicMock()
    db_mock.execute = AsyncMock(return_value=mock_execute_result)
    db_mock.commit = AsyncMock()
    db_mock.rollback = AsyncMock()

    async_ctx = MagicMock()

    async def _aenter(self: Any = None) -> MagicMock:
        return db_mock

    async def _aexit(self: Any = None, *args: Any) -> bool:
        return False

    async_ctx.__aenter__ = _aenter
    async_ctx.__aexit__ = _aexit

    factory_fn = MagicMock(return_value=async_ctx)
    get_session_factory_mock = MagicMock(return_value=factory_fn)

    return get_session_factory_mock, mock_run


class TestSMKT31OrchestratorDefenceInDepth:
    @pytest.mark.asyncio
    async def test_run_live_engine_rejects_violating_config_before_exchange_build(
        self,
    ) -> None:
        from api.services.run_orchestrator import run_live_engine
        from common.types import TimeFrame
        from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

        get_sf_mock, mock_run = _make_orchestrator_db_session_factory(run_status="running")

        with (
            patch(
                "api.config.get_settings",
                side_effect=AssertionError("must not reach settings/exchange build"),
            ),
            patch("api.db.session.get_session_factory", get_sf_mock),
        ):
            await run_live_engine(
                run_id_str=str(uuid.uuid4()),
                strategy_cls=SmokeRoundtripStrategy,
                strategy_name="smoke_roundtrip",
                strategy_params={"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4},
                symbols=["XRP/EUR"],
                timeframe=TimeFrame.FIVE_MINUTES,
                initial_capital="65.00",
                bracket_config={},  # missing SL -> G-7 stop_loss_required
            )

        assert mock_run.status == "error"

    @pytest.mark.asyncio
    async def test_run_live_engine_protective_resume_skips_guard(self) -> None:
        from api.services.run_orchestrator import run_live_engine
        from common.types import TimeFrame
        from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

        get_sf_mock, mock_run = _make_orchestrator_db_session_factory(run_status="running")
        guard_mock = MagicMock(
            side_effect=AssertionError("validate_smoke_run must not run in protective mode")
        )

        with (
            patch("api.services.run_orchestrator.validate_smoke_run", guard_mock),
            patch(
                "api.config.get_settings",
                side_effect=RuntimeError("stop-after-guard-skip-point"),
            ),
            patch("api.db.session.get_session_factory", get_sf_mock),
        ):
            await run_live_engine(
                run_id_str=str(uuid.uuid4()),
                strategy_cls=SmokeRoundtripStrategy,
                strategy_name="smoke_roundtrip",
                strategy_params={"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4},
                symbols=["XRP/EUR"],
                timeframe=TimeFrame.FIVE_MINUTES,
                initial_capital="65.00",
                bracket_config={},  # would violate G-7, but the guard must be skipped
                protective_mode=True,
            )

        guard_mock.assert_not_called()
        # Proceeded past the guard-skip point straight into get_settings(),
        # which raised RuntimeError -- caught by the generic except, status
        # still ends 'error' (for an unrelated reason), proving the guard
        # itself was never consulted.
        assert mock_run.status == "error"

    @pytest.mark.asyncio
    async def test_run_paper_engine_rejects_violating_config_and_skips_auto_retry(
        self,
    ) -> None:
        from api.services.run_orchestrator import run_paper_engine
        from common.types import TimeFrame
        from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

        get_sf_mock, mock_run = _make_orchestrator_db_session_factory(run_status="running")
        auto_retry_mock = MagicMock(
            side_effect=AssertionError("a deterministic SmokeGuardError must never auto-retry")
        )

        with (
            patch(
                "api.config.get_settings",
                side_effect=AssertionError("must not reach settings/exchange build"),
            ),
            patch("api.db.session.get_session_factory", get_sf_mock),
            patch("api.services.run_orchestrator._auto_retry_paper_run", auto_retry_mock),
        ):
            await run_paper_engine(
                run_id_str=str(uuid.uuid4()),
                strategy_cls=SmokeRoundtripStrategy,
                strategy_name="smoke_roundtrip",
                strategy_params={"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4},
                symbols=["XRP/EUR"],
                timeframe=TimeFrame.FIVE_MINUTES,
                initial_capital="10000.00",
                bracket_config={},  # missing SL -> G-7 stop_loss_required
            )

        assert mock_run.status == "error"


# ===========================================================================
# WP-SMOKE fix F-5 (SMK-SEC-04) / SMK-T-42: a smoke_roundtrip paper run
# never auto-retries after ANY crash (not just a SmokeGuardError) --
# non-smoke behaviour is unchanged (control case).
# ===========================================================================
class _CrashingStrategy:
    """A strategy stand-in whose construction always raises -- simulates
    an engine-construction-time crash unrelated to the smoke guard, so
    the resulting exception is a plain RuntimeError, not SmokeGuardError/
    ExitConfigError."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SMK-T-42: simulated engine-construction crash")


class TestF5NoSmokePaperAutoRetry:
    @pytest.mark.asyncio
    async def test_smoke_paper_crash_never_auto_retries(self) -> None:
        from api.services.run_orchestrator import run_paper_engine
        from common.types import TimeFrame

        get_sf_mock, mock_run = _make_orchestrator_db_session_factory(run_status="running")
        auto_retry_mock = MagicMock(
            side_effect=AssertionError("smoke_roundtrip must never auto-retry after a crash")
        )

        with (
            patch("api.db.session.get_session_factory", get_sf_mock),
            patch("api.services.run_orchestrator._auto_retry_paper_run", auto_retry_mock),
        ):
            await run_paper_engine(
                run_id_str=str(uuid.uuid4()),
                strategy_cls=_CrashingStrategy,
                strategy_name="smoke_roundtrip",
                strategy_params={"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4},
                symbols=["XRP/EUR"],
                timeframe=TimeFrame.FIVE_MINUTES,
                initial_capital="65.00",
                bracket_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
            )

        assert mock_run.status == "error"
        auto_retry_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonsmoke_paper_crash_still_auto_retries(self) -> None:
        """Control: the SAME crash, for a non-smoke strategy, still
        schedules the retry -- proves F-5 changed nothing outside
        SMOKE_STRATEGY_NAMES."""
        from api.services.run_orchestrator import run_paper_engine
        from common.types import TimeFrame

        get_sf_mock, mock_run = _make_orchestrator_db_session_factory(run_status="running")
        auto_retry_mock = AsyncMock()

        with (
            patch("api.db.session.get_session_factory", get_sf_mock),
            patch("api.services.run_orchestrator._auto_retry_paper_run", auto_retry_mock),
        ):
            await run_paper_engine(
                run_id_str=str(uuid.uuid4()),
                strategy_cls=_CrashingStrategy,
                strategy_name="ma_crossover",
                strategy_params={},
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.FIVE_MINUTES,
                initial_capital="10000.00",
                bracket_config=None,
            )

        assert mock_run.status == "error"
        auto_retry_mock.assert_called_once()


# ===========================================================================
# SMK-T-32: G-12 paper boot recovery never rebuilds smoke_roundtrip
# ===========================================================================
def _make_orphan(
    run_mode: str = "paper",
    config: dict | None = None,
    status: str = "running",
    strategy_name: str = "smoke_roundtrip",
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=uuid.uuid4(),
        run_mode=run_mode,
        status=status,
        config=config
        or {
            "strategy_name": strategy_name,
            "symbols": ["XRP/EUR"],
            "timeframe": "5m",
            "initial_capital": "65.00",
            "strategy_params": dict(_VALID_LIVE_STRATEGY_PARAMS),
            "mode": run_mode,
        },
        started_at=datetime.now(UTC),
        stopped_at=None,
        updated_at=datetime.now(UTC),
        entries_latch_reason=None,
        entries_latched_at=None,
    )


def _make_session_for_select(orphans: list) -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock()
    result.scalars.return_value.all.return_value = orphans
    session.execute = AsyncMock(return_value=result)
    return session


def _make_session_for_write(orphan: Any) -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock()
    result.scalar_one_or_none.return_value = orphan
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


def _build_factory(*sessions: AsyncMock) -> MagicMock:
    contexts = []
    for s in sessions:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=s)
        ctx.__aexit__ = AsyncMock(return_value=False)
        contexts.append(ctx)

    call_state = {"i": 0}

    def _factory() -> Any:
        i = call_state["i"]
        ctx = contexts[i] if i < len(contexts) else contexts[-1]
        call_state["i"] = i + 1
        return ctx

    return MagicMock(side_effect=_factory)


class TestSMKT32PaperBootRecoveryNeverRebuildsSmoke:
    @pytest.mark.asyncio
    async def test_smoke_paper_orphan_marked_error_and_not_rebuilt(self) -> None:
        from api.routers.runs import recover_orphaned_runs
        from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

        orphan = _make_orphan(run_mode="paper", status="running")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch(
                "api.routers.runs._get_strategy_registry",
                return_value={"smoke_roundtrip": SmokeRoundtripStrategy},
            ),
            patch("api.routers.runs._run_paper_engine", new=AsyncMock()) as run_paper,
            patch("api.routers.runs._load_resume_snapshot", new=AsyncMock()) as load_snap,
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert orphan.status == "error"
        assert runs_module._RUN_TASKS == {}
        # Never even reached the snapshot-load / rebuild step.
        load_snap.assert_not_called()
        run_paper.assert_not_called()


# ===========================================================================
# SMK-T-33: DELETE /runs/{id} stop-while-holding is unmodified for smoke
# ===========================================================================
def _make_stop_request() -> MagicMock:
    from types import SimpleNamespace as _SN

    req = MagicMock()
    req.headers = {}
    req.client = _SN(host="127.0.0.1")
    return req


def _make_stop_run(*, status: str = "running") -> MagicMock:
    run = MagicMock(spec=RunORM)
    run.id = uuid.uuid4()
    run.run_mode = "live"
    run.status = status
    run.config = {"strategy_name": "smoke_roundtrip"}
    run.entries_latch_reason = None
    run.entries_latched_at = None
    run.started_at = datetime.now(tz=UTC)
    run.stopped_at = None
    run.created_at = datetime.now(tz=UTC)
    run.updated_at = datetime.now(tz=UTC)
    run.n_closed_trades = None
    run.metrics_v2_backfilled = False
    run.recovered_from_run_id = None
    run.promoted_from_run_id = None
    return run


def _make_stop_engine(*, symbols: list[str], held: dict[str, Decimal]) -> MagicMock:
    engine = MagicMock()
    engine.symbols = symbols
    engine.risk_manager = MagicMock()
    engine.risk_manager.trigger_kill_switch = MagicMock()
    engine.risk_manager.kill_switch_reasons = frozenset()

    def _get_position(symbol: str) -> MagicMock | None:
        qty = held.get(symbol)
        if qty is None:
            return None
        pos = MagicMock()
        pos.quantity = qty
        pos.is_flat = qty <= Decimal("0")
        return pos

    engine.portfolio = MagicMock()
    engine.portfolio.get_position = MagicMock(side_effect=_get_position)
    return engine


def _stop_db_with_single_result(run: MagicMock | None) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = run
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    return db


class TestSMKT33StopWhileHolding:
    @pytest.mark.asyncio
    async def test_live_running_smoke_run_holding_without_flatten_returns_422(self) -> None:
        from fastapi import HTTPException

        from api.routers.runs import stop_run

        run = _make_stop_run(status="running")
        db = _stop_db_with_single_result(run)
        engine = _make_stop_engine(symbols=["XRP/EUR"], held={"XRP/EUR": Decimal("9.5")})

        with patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}):
            with pytest.raises(HTTPException) as exc_info:
                await stop_run(run.id, db, _make_stop_request(), flatten=None)

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "flatten_decision_required"
        assert exc_info.value.detail["held_symbols"] == ["XRP/EUR"]

    @pytest.mark.asyncio
    async def test_live_running_smoke_run_flatten_true_stops(self) -> None:
        from api.routers.runs import stop_run

        run = _make_stop_run(status="running")
        db = _stop_db_with_single_result(run)
        engine = _make_stop_engine(symbols=["XRP/EUR"], held={"XRP/EUR": Decimal("0")})
        from trading.strategy_engine import FlattenResult

        complete_result = FlattenResult(
            run_id=str(run.id),
            outcome="flattened",
            complete=True,
            symbols=[],
        )
        engine.flatten = AsyncMock(return_value=complete_result)
        task = MagicMock()
        task.done.return_value = False

        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run.id): engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run.id): task}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            response = await stop_run(run.id, db, _make_stop_request(), flatten=True)

        assert run.status == "stopped"
        assert response.flatten is not None
        assert response.flatten.outcome == "flattened"
        task.cancel.assert_called_once()
