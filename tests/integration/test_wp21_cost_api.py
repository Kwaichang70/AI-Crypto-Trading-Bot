"""
tests/integration/test_wp21_cost_api.py
-----------------------------------------
WP2.1a API coverage (hermetic: no Postgres, no network, no exchange).

    WP21-T-20  create backtest / paper / live persist ``cost_profile`` + ``engine_semantics``
               (``backtest_circuit_breaker`` only on backtest; live ``fee_source``
               ``exchange_fills``); the ``run_*_engine`` spies receive the SAME fingerprint;
               ``persist_backtest_results`` logs an error (no raise) on a fingerprint mismatch
    WP21-T-21  promote overwrites the inherited profile with the DEFAULT and validates at
               ``policy_cost_model(DEFAULT)``
    WP21-T-22  resume (live) and boot recovery (paper) of a LEGACY run with a fixed SL of
               0.0068 still resume, the engine gets ``legacy_runtime_v0`` and no cost key is
               ever written
    WP21-T-27  request validation: stable 422 codes, backtest + ``legacy_backtest_v0`` -> 201
    WP21-T-28  the floor at the default (0.0130 / 0.0131) and SY-21-06 (API pass => engine pass)
    WP21-T-33  ``OptimizeResponse.costProfile`` on POST, request id rules as backtest
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

import api.routers.runs as runs_module
import trading.costs as costs_module
from api.config import get_settings
from api.services import kill_switch as _kill_switch
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.conftest import make_bars
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile

CONFIRM_TOKEN = "wp21-test-confirm-token"  # noqa: S105 -- test fixture, not a real secret
_SYMBOL = "BTC/USD"
_BARS = {_SYMBOL: make_bars(120, symbol=_SYMBOL)}


# ---------------------------------------------------------------------------
# App / client scaffolding (3-layer live gate PASSES, like test_wp70_idempotency_api)
# ---------------------------------------------------------------------------


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> Generator[Any, None, None]:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "wp21-key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "wp21-secret")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", CONFIRM_TOKEN)
    get_settings.cache_clear()
    from api.main import create_app

    yield create_app()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_run_tasks() -> Generator[None, None, None]:
    runs_module._RUN_TASKS.clear()
    yield
    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()


def _db() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    nested = AsyncMock()
    nested.__aenter__ = AsyncMock(return_value=None)
    nested.__aexit__ = AsyncMock(return_value=None)
    session.begin_nested = MagicMock(return_value=nested)
    return session


class _C:
    def __init__(self, app: Any) -> None:
        from api.db.session import get_db
        from api.services.idempotency import get_idempotency_store

        self.db = _db()
        self.store = InMemoryIdempotencyStore(stale_after_seconds=240.0)

        async def _override_get_db() -> Any:
            yield self.db

        app.dependency_overrides[get_db] = _override_get_db
        app.dependency_overrides[get_idempotency_store] = lambda: self.store
        self.client = TestClient(app, raise_server_exceptions=False)
        self.client.__enter__()
        _kill_switch.reset_state_for_tests()

    def close(self) -> None:
        self.client.__exit__(None, None, None)

    def created_run(self) -> Any:
        """The ``RunORM`` handed to ``db.add`` by the create / promote handler."""
        for call in self.db.add.call_args_list:
            if type(call.args[0]).__name__ == "RunORM":
                return call.args[0]
        raise AssertionError("no RunORM was added")


@pytest.fixture()
def c(app: Any) -> Generator[_C, None, None]:
    client = _C(app)
    yield client
    client.close()


_BACKTEST: dict[str, Any] = {
    "strategyName": "momentum_breakout",
    "strategyParams": {
        "lookback": 5,
        "trend_sma_period": 5,
        "position_size": 100.0,
        "bracket_mode": "fixed",
        "bracket_stop_loss_pct": 0.05,
        "bracket_take_profit_pct": 0.1,
    },
    "symbols": [_SYMBOL],
    "timeframe": "1h",
    "mode": "backtest",
    "initialCapital": "10000.00",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-06-01T00:00:00Z",
}
_PAPER: dict[str, Any] = {
    "strategyName": "grid_trading",
    "strategyParams": {},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "paper",
    "initialCapital": "10000.00",
}
_LIVE: dict[str, Any] = {**_PAPER, "mode": "live", "allowPyramiding": False}


def _hdr(**extra: str) -> dict[str, str]:
    return {"Idempotency-Key": str(uuid.uuid4()), **extra}


def _post_backtest(c: _C, **overrides: Any) -> Any:
    body = {**_BACKTEST, **overrides}
    with patch("api.routers.runs._fetch_bars_for_backtest", return_value=_BARS):
        return c.client.post("/api/v1/runs", json=body, headers=_hdr())


def _post_never_fetching(c: _C, body: dict[str, Any], **hdr: str) -> Any:
    with patch(
        "api.routers.runs._fetch_bars_for_backtest",
        side_effect=AssertionError("bar fetch must not be reached"),
    ):
        return c.client.post("/api/v1/runs", json=body, headers=_hdr(**hdr))


def _with_params(**params: Any) -> dict[str, Any]:
    return {"strategyParams": {**_BACKTEST["strategyParams"], **params}}


# ---------------------------------------------------------------------------
# WP21-T-20 -- persistence on create
# ---------------------------------------------------------------------------


class TestT20CreatePersistsCostProfile:
    def test_backtest_persists_default_profile_and_engine_semantics(self, c: _C) -> None:
        resp = _post_backtest(c)
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["cost_profile"] == DEFAULT_COST_PROFILE.to_metadata(fee_source="model")
        assert cfg["engine_semantics"] == {
            "schema": 1,
            "cost_model": "wp21-1",
            "exit_semantics": "wp13b-1",
            "daily_risk": "wp15-1",
            "backtest_circuit_breaker": False,
        }
        body = resp.json()
        assert body["costProfile"]["id"] == "coinbase_advanced_1"
        assert body["costProfile"]["takerFeeBps"] == "120"
        assert body["costProfile"]["roundTripBps"] == "260"
        assert body["costProfile"]["fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()
        assert (body["legacyCosts"], body["costProfileInvalid"], body["legacyExitSemantics"]) == (
            False,
            False,
            False,
        )

    def test_backtest_runs_under_the_persisted_profile(self, c: _C) -> None:
        resp = _post_backtest(c, costProfile="coinbase_advanced_d6")
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["cost_profile"]["id"] == "coinbase_advanced_d6"
        # the persisted metrics were produced under that profile: fees at 60 bps taker
        assert resp.json()["costProfile"]["id"] == "coinbase_advanced_d6"

    def test_explicit_profile_is_persisted_and_legacy_id_is_labelled_legacy(self, c: _C) -> None:
        resp = _post_backtest(c, costProfile="legacy_backtest_v0")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["costProfile"]["id"] == "legacy_backtest_v0"
        assert body["legacyCosts"] is True  # SY-21-10: a legacy_* registry id is legacy

    def test_paper_persists_profile_and_the_engine_spy_gets_the_same_fingerprint(
        self, c: _C
    ) -> None:
        spy = AsyncMock()
        with patch("api.routers.runs._run_paper_engine", new=spy):
            resp = c.client.post("/api/v1/runs", json=_PAPER, headers=_hdr())
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["cost_profile"]["fee_source"] == "model"
        assert "backtest_circuit_breaker" not in cfg["engine_semantics"]
        assert cfg["engine_semantics"]["exit_semantics"] == "wp13b-1"
        profile = spy.call_args.kwargs["cost_profile"]
        assert profile.fingerprint() == cfg["cost_profile"]["fingerprint"]
        assert profile is DEFAULT_COST_PROFILE

    def test_paper_with_an_explicit_live_equivalent_profile(self, c: _C) -> None:
        spy = AsyncMock()
        with patch("api.routers.runs._run_paper_engine", new=spy):
            resp = c.client.post(
                "/api/v1/runs",
                json={**_PAPER, "costProfile": "coinbase_advanced_d6"},
                headers=_hdr(),
            )
        assert resp.status_code == 201, resp.text
        assert c.created_run().config["cost_profile"]["id"] == "coinbase_advanced_d6"
        assert spy.call_args.kwargs["cost_profile"].id == "coinbase_advanced_d6"

    def test_live_persists_the_default_estimate_with_exchange_fills(self, c: _C) -> None:
        spy = AsyncMock()
        with patch("api.routers.runs._run_live_engine", new=spy):
            resp = c.client.post(
                "/api/v1/runs",
                json=_LIVE,
                headers=_hdr(**{"X-Live-Confirm-Token": CONFIRM_TOKEN}),
            )
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["cost_profile"]["id"] == "coinbase_advanced_1"
        assert cfg["cost_profile"]["fee_source"] == "exchange_fills"
        assert "backtest_circuit_breaker" not in cfg["engine_semantics"]
        assert (
            spy.call_args.kwargs["cost_profile"].fingerprint() == cfg["cost_profile"]["fingerprint"]
        )
        assert resp.json()["costProfile"]["feeSource"] == "exchange_fills"

    def test_persist_backtest_results_logs_an_error_on_a_fingerprint_mismatch(self) -> None:
        import asyncio

        from api.services.run_persistence import persist_backtest_results

        db = _db()
        run_orm = SimpleNamespace(config={"cost_profile": DEFAULT_COST_PROFILE.to_metadata()})
        result = MagicMock()
        result.trades = []
        result.equity_curve = []
        result.total_trades = 0
        result.total_return_pct = 0.0
        result.sharpe_ratio = 0.0
        result.cost_profile_fingerprint = "f" * 16
        with (
            patch("api.services.run_persistence.build_backtest_metrics") as metrics,
            patch("sqlalchemy.orm.attributes.flag_modified"),
            capture_logs() as logs,
        ):
            metrics.return_value.model_dump.return_value = {}
            asyncio.run(
                persist_backtest_results(
                    db, uuid.uuid4(), run_orm, result, structlog.get_logger("wp21-test")
                )
            )
        errors = [e for e in logs if e["event"] == "runs.cost_profile_fingerprint_mismatch"]
        assert len(errors) == 1  # logged, did not raise
        assert errors[0]["config_fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()
        assert errors[0]["result_fingerprint"] == "f" * 16


# ---------------------------------------------------------------------------
# WP21-T-27 -- request validation
# ---------------------------------------------------------------------------


class TestT27RequestValidation:
    def test_unknown_id_is_422_with_a_stable_code_and_no_echo(self, c: _C) -> None:
        resp = _post_never_fetching(c, {**_BACKTEST, "costProfile": "SECRET-nope"})
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": "unknown_cost_profile"}
        assert "SECRET-nope" not in resp.text
        c.db.add.assert_not_called()

    @pytest.mark.parametrize(
        "profile_id",
        [
            "coinbase_stress_t80",
            "frictionless_diagnostic",
            "legacy_backtest_v0",
            "legacy_runtime_v0",
        ],
    )
    def test_paper_rejects_non_live_equivalent_ids(self, c: _C, profile_id: str) -> None:
        resp = c.client.post(
            "/api/v1/runs", json={**_PAPER, "costProfile": profile_id}, headers=_hdr()
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": "cost_profile_not_live_equivalent"}
        c.db.add.assert_not_called()

    @pytest.mark.parametrize("profile_id", ["coinbase_advanced_1", "coinbase_advanced_d6", "nope"])
    def test_live_accepts_no_cost_profile(self, c: _C, profile_id: str) -> None:
        resp = c.client.post(
            "/api/v1/runs",
            json={**_LIVE, "costProfile": profile_id},
            headers=_hdr(**{"X-Live-Confirm-Token": CONFIRM_TOKEN}),
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": "cost_profile_not_allowed_for_live"}
        c.db.add.assert_not_called()

    def test_backtest_accepts_the_legacy_profile(self, c: _C) -> None:
        assert _post_backtest(c, costProfile="legacy_backtest_v0").status_code == 201

    def test_an_overlong_id_is_a_pydantic_422(self, c: _C) -> None:
        resp = _post_never_fetching(c, {**_BACKTEST, "costProfile": "x" * 65})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# WP21-T-28 -- the floor at the default + SY-21-06
# ---------------------------------------------------------------------------


class TestT28FloorAtTheDefault:
    def _body(self, sl: float, **extra: Any) -> dict[str, Any]:
        return {
            **_BACKTEST,
            **_with_params(
                bracket_mode="fixed", bracket_stop_loss_pct=sl, bracket_take_profit_pct=0.2
            ),
            **extra,
        }

    def test_sl_at_the_default_side_cost_is_rejected(self, c: _C) -> None:
        resp = _post_never_fetching(c, self._body(0.0130))
        assert resp.status_code == 422, resp.text
        errors = resp.json()["detail"]["errors"]
        assert any(e["reason"] == "stop_inside_entry_cost" for e in errors)
        c.db.add.assert_not_called()

    def test_sl_just_above_the_default_side_cost_is_created_with_a_warning(self, c: _C) -> None:
        resp = _post_backtest(c, **self._body(0.0131))
        assert resp.status_code == 201, resp.text
        assert any(w["code"] == "sl_below_round_trip_cost" for w in resp.json()["configWarnings"])

    def test_the_old_0066_case_is_now_rejected(self, c: _C) -> None:
        # HEAD accepted 0.0066 (side cost 0.0065); the default side cost is 0.0130 now
        assert _post_never_fetching(c, self._body(0.0066)).status_code == 422

    def test_api_pass_implies_engine_pass_when_the_default_moves_below_the_run_profile(
        self, c: _C, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SY-21-06: default switched to D6, a backtest requests coinbase_advanced_1 with SL
        0.010.  The API validates at the STRICTER of the two (0.0130) -> a clean 422, NOT a
        ValueError from inside the runner (a 500)."""
        monkeypatch.setattr(
            costs_module, "DEFAULT_COST_PROFILE", get_cost_profile("coinbase_advanced_d6")
        )
        resp = _post_never_fetching(c, self._body(0.010, costProfile="coinbase_advanced_1"))
        assert resp.status_code == 422, resp.text
        assert any(e["reason"] == "stop_inside_entry_cost" for e in resp.json()["detail"]["errors"])
        # control: the same SL under the D6 profile itself is fine at a D6 default
        ok = _post_backtest(c, **self._body(0.010, costProfile="coinbase_advanced_d6"))
        assert ok.status_code == 201, ok.text

    def test_a_legacy_backtest_profile_is_still_judged_at_the_stricter_default(self, c: _C) -> None:
        # SL 0.005 passes the legacy profile's own 0.0020 floor but not the policy basis
        resp = _post_never_fetching(c, self._body(0.005, costProfile="legacy_backtest_v0"))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# WP21-T-21 -- promote
# ---------------------------------------------------------------------------


def _paper_source(source_id: uuid.UUID, **config_overrides: Any) -> Any:
    config: dict[str, Any] = {
        "strategy_name": "grid_trading",
        "strategy_params": {},
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "mode": "paper",
        "initial_capital": "10000.00",
        "allow_pyramiding": False,
        "bracket_config": {
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.05,
            "bracket_take_profit_pct": 0.1,
        },
    }
    config.update(config_overrides)
    return SimpleNamespace(
        id=source_id,
        run_mode="paper",
        status="stopped",
        config=config,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        stopped_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_closed_trades=None,
        metrics_v2_backfilled=False,
        promoted_from_run_id=None,
        recovered_from_run_id=None,
        entries_latch_reason=None,
        entries_latched_at=None,
    )


def _promote(c: _C, source: Any) -> tuple[Any, AsyncMock]:
    from api.services.promotion_gate import PromotionEligibility

    result = MagicMock()
    result.scalar_one_or_none.return_value = source
    c.db.execute.return_value = result
    live_spy = AsyncMock()
    with (
        patch(
            "api.services.promotion_gate.evaluate_paper_run_eligibility",
            AsyncMock(
                return_value=PromotionEligibility(eligible=True, trade_count=100, runtime_days=10.0)
            ),
        ),
        patch("api.routers.runs._run_live_engine", new=live_spy),
    ):
        resp = c.client.post(
            f"/api/v1/runs/{source.id}/promote-to-live",
            headers=_hdr(**{"X-Live-Confirm-Token": CONFIRM_TOKEN}),
        )
    return resp, live_spy


class TestT21Promote:
    def test_promoting_a_d6_paper_run_overwrites_the_profile_with_the_default(self, c: _C) -> None:
        source = _paper_source(
            uuid.uuid4(),
            cost_profile=get_cost_profile("coinbase_advanced_d6").to_metadata(),
            engine_semantics={"schema": 1, "cost_model": "old", "exit_semantics": "old"},
        )
        resp, live_spy = _promote(c, source)
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["cost_profile"] == DEFAULT_COST_PROFILE.to_metadata(fee_source="exchange_fills")
        assert cfg["engine_semantics"]["cost_model"] == "wp21-1"
        assert "backtest_circuit_breaker" not in cfg["engine_semantics"]
        assert live_spy.call_args.kwargs["cost_profile"] is DEFAULT_COST_PROFILE
        # the SOURCE run's own config is untouched
        assert source.config["cost_profile"]["id"] == "coinbase_advanced_d6"

    def test_promoting_a_legacy_paper_run_stamps_the_default_too(self, c: _C) -> None:
        source = _paper_source(uuid.uuid4())  # no cost_profile key at all
        resp, live_spy = _promote(c, source)
        assert resp.status_code == 201, resp.text
        assert c.created_run().config["cost_profile"]["id"] == "coinbase_advanced_1"
        assert live_spy.call_args.kwargs["cost_profile"] is DEFAULT_COST_PROFILE

    def test_promote_validates_at_the_default_policy_basis(self, c: _C) -> None:
        # a persisted fixed SL of 0.0100 resumes fine (own profile) but cannot be PROMOTED
        source = _paper_source(
            uuid.uuid4(),
            bracket_config={
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.010,
                "bracket_take_profit_pct": 0.1,
            },
        )
        resp, live_spy = _promote(c, source)
        assert resp.status_code == 422, resp.text
        assert any(e["reason"] == "stop_inside_entry_cost" for e in resp.json()["detail"]["errors"])
        live_spy.assert_not_called()


# ---------------------------------------------------------------------------
# WP21-T-22 -- resume / boot recovery of a legacy run
# ---------------------------------------------------------------------------

_LEGACY_BRACKET = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.0068,
    "bracket_take_profit_pct": 0.05,
}


class TestT22LegacyRunsStayResumable:
    def test_live_resume_of_a_legacy_run_uses_legacy_runtime_and_writes_no_cost_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.integration import test_wp18a_resume_endpoint as resume_mod

        monkeypatch.setenv("REQUIRE_API_AUTH", "false")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
        monkeypatch.setenv("EXCHANGE_API_KEY", "k")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
        monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", resume_mod.CONFIRM_TOKEN)
        monkeypatch.setenv("ADMIN_API_KEY", resume_mod.ADMIN_KEY)
        get_settings.cache_clear()
        from api.main import create_app

        app = create_app()
        monkeypatch.setattr(runs_module, "_STRATEGY_REGISTRY", {"grid_trading": MagicMock()})
        run_row = resume_mod._make_run_row(status="orphaned")
        run_row.config["bracket_config"] = dict(_LEGACY_BRACKET)
        original_config = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in run_row.config.items()
        }
        session = resume_mod._DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
        client = resume_mod._client(app, session)

        async def _scan(db: Any, run: Any, exchange: Any, *, fence: Any = None) -> Any:
            from api.services.run_recovery import ImportReport

            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        monkeypatch.setattr(run_recovery_module, "scan_and_import", _scan)
        spy = AsyncMock()
        try:
            with patch("api.routers.runs._run_live_engine", spy):
                resp = client.post(
                    f"/api/v1/runs/{resume_mod.RUN_ID}/resume",
                    headers={
                        "X-Live-Confirm-Token": resume_mod.CONFIRM_TOKEN,
                        "X-Admin-Key": resume_mod.ADMIN_KEY,
                    },
                )
        finally:
            get_settings.cache_clear()
            runs_module._RUN_TASKS.clear()
        assert resp.status_code == 200, resp.text
        profile = spy.call_args.kwargs["cost_profile"]
        assert profile.id == "legacy_runtime_v0" and profile.source == "legacy_inferred"
        # resume never writes the cost keys (and judged SL 0.0068 at the run's OWN 0.0065 floor)
        assert "cost_profile" not in run_row.config and "engine_semantics" not in run_row.config
        # (the resume re-writes only the NORMALISED bracket config + bookkeeping keys)
        for key, value in original_config.items():
            if key not in ("bracket_config", "allow_pyramiding"):
                assert run_row.config[key] == value

    async def test_paper_boot_recovery_of_a_legacy_run_uses_legacy_runtime(self) -> None:
        from tests.unit import test_sprint24_run_recovery as rec

        config = {
            "strategy_name": "grid_trading",
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
            "bracket_config": dict(_LEGACY_BRACKET),
            "mode": "paper",
        }
        before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
        orphan = rec._make_orphan(run_mode="paper", status="running", config=config)
        select_session = rec._make_session_for_select([orphan])
        write_session = rec._make_session_for_write(orphan)
        factory = rec._build_factory(select_session, write_session, write_session)
        spy = AsyncMock()
        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            rec._patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=rec._EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", spy),
        ):
            recovered = await runs_module.recover_orphaned_runs()
        runs_module._RUN_TASKS.clear()
        assert recovered == 1
        profile = spy.call_args.kwargs["cost_profile"]
        assert profile.id == "legacy_runtime_v0" and profile.legacy is True
        assert "cost_profile" not in orphan.config and "engine_semantics" not in orphan.config
        for key, value in before.items():
            if key not in ("bracket_config", "allow_pyramiding"):
                assert orphan.config[key] == value

    async def test_paper_boot_recovery_of_a_persisted_run_uses_its_own_profile(self) -> None:
        from tests.unit import test_sprint24_run_recovery as rec

        d6 = get_cost_profile("coinbase_advanced_d6")
        config = {
            "strategy_name": "grid_trading",
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
            "bracket_config": {
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.05,
                "bracket_take_profit_pct": 0.1,
            },
            "mode": "paper",
            "cost_profile": d6.to_metadata(),
        }
        orphan = rec._make_orphan(run_mode="paper", status="running", config=config)
        factory = rec._build_factory(
            rec._make_session_for_select([orphan]),
            rec._make_session_for_write(orphan),
            rec._make_session_for_write(orphan),
        )
        spy = AsyncMock()
        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            rec._patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=rec._EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", spy),
        ):
            await runs_module.recover_orphaned_runs()
        runs_module._RUN_TASKS.clear()
        assert spy.call_args.kwargs["cost_profile"] == d6


# ---------------------------------------------------------------------------
# WP21-T-33 -- optimize
# ---------------------------------------------------------------------------

_OPT_BARS = {_SYMBOL: make_bars(300, symbol=_SYMBOL)}
_OPT_BODY: dict[str, Any] = {
    "strategyName": "ma_crossover",
    "paramGrid": {"fast_period": [5, 10], "slow_period": [20, 30]},
    "symbols": [_SYMBOL],
    "timeframe": "1h",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-06-01T00:00:00Z",
    "topN": 3,
    "maxCombinations": 10,
}


class TestT33Optimize:
    def _post(self, c: _C, **overrides: Any) -> Any:
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest", new=AsyncMock(return_value=_OPT_BARS)
        ):
            return c.client.post("/api/v1/optimize", json={**_OPT_BODY, **overrides})

    def test_response_carries_the_default_cost_profile(self, c: _C) -> None:
        resp = self._post(c)
        assert resp.status_code == 200, resp.text
        cp = resp.json()["costProfile"]
        assert (
            cp["id"] == "coinbase_advanced_1"
            and cp["fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()
        )

    def test_any_registry_id_is_accepted_like_a_backtest(self, c: _C) -> None:
        for pid in ("coinbase_stress_t80", "frictionless_diagnostic", "legacy_backtest_v0"):
            resp = self._post(c, costProfile=pid)
            assert resp.status_code == 200, (pid, resp.text)
            assert resp.json()["costProfile"]["id"] == pid

    def test_unknown_id_is_a_stable_422(self, c: _C) -> None:
        resp = self._post(c, costProfile="nope")
        assert resp.status_code == 422
        assert resp.json()["detail"] == {"code": "unknown_cost_profile"}

    def test_the_profile_reaches_every_combination_and_changes_the_ranking_inputs(
        self, c: _C
    ) -> None:
        free = self._post(c, costProfile="frictionless_diagnostic").json()["entries"]
        dear = self._post(c, costProfile="coinbase_stress_t100").json()["entries"]
        free_fees = {round(e["metrics"]["total_fees_paid"], 6) for e in free}
        assert free_fees == {0.0}
        assert any(e["metrics"]["total_fees_paid"] > 0 for e in dear)
