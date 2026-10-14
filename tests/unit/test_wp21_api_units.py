"""
tests/unit/test_wp21_api_units.py
-----------------------------------
WP2.1a unit coverage of the API layer around the cost model.

    WP21-T-23  auto-retry COPIES the crashed run's resolved profile (legacy stays legacy)
    WP21-T-24  orchestrator: the paper / live engines get the run's profile; the live risk
               limit (max_drawdown_pct == 0.20) is preserved
    WP21-T-25  ``run_orm_to_response`` cost fields (legacy / registry-legacy / malformed /
               no-metrics early-return path / ``legacyExitSemantics``)
    WP21-T-26  one eligibility helper: legacy, stress, diagnostic, override and invalid are never
               eligible; ``/portfolio`` best/worst excludes them and reports
               ``legacyCostRunsExcluded``
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.services.run_orchestrator as orchestrator
from api.routers.runs import _RUN_TASKS
from api.schemas import RunResponse, leaderboard_eligible
from api.services.run_orchestrator import _auto_retry_paper_run
from api.services.run_persistence import run_orm_to_detail_response, run_orm_to_response
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit.test_auto_retry import _mock_session_factory, _retry_kwargs
from tests.unit.test_live_engine_wiring import (
    _PT_CCXT,
    _PT_FLUSH,
    _PT_FLUSH_LOOP,
    _PT_LIVE_ENGINE,
    _PT_MDS,
    _PT_PORTFOLIO,
    _PT_RISK_MGR,
    _PT_SESSION_FACTORY,
    _PT_SETTINGS,
    _PT_STRAT_ENGINE,
    _make_ccxt_module_mock,
    _make_db_session_factory,
    _make_settings_mock,
    _make_strategy_cls_mock,
    _make_strategy_engine_mock,
    _risk_mgr_cls,
)
from tests.unit.test_wp18a_shutdown_orphan import (
    _FakeEngine,
    _make_run_row,
    _make_settings,
    _session_factory_for,
)
from trading.costs import (
    DEFAULT_COST_PROFILE,
    CostProfile,
    get_cost_profile,
    is_legacy_cost_config,
    resolve_run_cost_profile,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# WP21-T-23 -- auto-retry copies the resolved profile
# ---------------------------------------------------------------------------


async def _retry(
    crashed_config: dict[str, Any], profile: CostProfile
) -> tuple[Any, dict[str, Any]]:
    crashed = MagicMock()
    crashed.id = uuid.uuid4()
    crashed.status = "error"
    crashed.entries_latch_reason = None
    crashed.config = crashed_config
    factory, db = _mock_session_factory(crashed)
    captured: dict[str, Any] = {}

    async def _fake_engine(**kwargs: Any) -> None:
        captured.update(kwargs)

    with (
        patch("asyncio.sleep", new=AsyncMock()),
        patch("api.db.session.get_session_factory", return_value=factory),
        patch("api.services.run_orchestrator.run_paper_engine", new=_fake_engine),
    ):
        await _auto_retry_paper_run(
            **_retry_kwargs(crashed_run_id=str(crashed.id), cost_profile=profile)
        )
    await asyncio.sleep(0)
    _RUN_TASKS.pop(captured.get("run_id_str", ""), None)
    return db.add.call_args.args[0], captured


class TestT23AutoRetryCopies:
    async def test_a_legacy_run_stays_legacy_with_inferred_metadata(self) -> None:
        legacy = resolve_run_cost_profile({}, mode="paper").profile  # the engine's profile
        new_run, captured = await _retry({"strategy_name": "grid_trading"}, legacy)
        meta = new_run.config["cost_profile"]
        assert meta == legacy.to_metadata()
        assert (meta["id"], meta["source"], meta["legacy"]) == (
            "legacy_runtime_v0",
            "legacy_inferred",
            True,
        )
        assert is_legacy_cost_config(new_run.config) is True
        assert captured["cost_profile"] is legacy  # the relaunched engine gets the SAME profile

    async def test_a_persisted_profile_is_copied_unchanged(self) -> None:
        d6 = get_cost_profile("coinbase_advanced_d6")
        crashed_config = {
            "strategy_name": "grid_trading",
            "cost_profile": d6.to_metadata(),
            "engine_semantics": {"schema": 1, "cost_model": "wp21-1"},
        }
        new_run, captured = await _retry(crashed_config, d6)
        assert new_run.config["cost_profile"] == d6.to_metadata()
        assert new_run.config["engine_semantics"] == {"schema": 1, "cost_model": "wp21-1"}
        assert captured["cost_profile"] == d6

    async def test_a_retry_never_reprices_under_todays_default(self) -> None:
        legacy = get_cost_profile("legacy_runtime_v0")
        new_run, captured = await _retry({}, legacy)
        assert new_run.config["cost_profile"]["id"] != DEFAULT_COST_PROFILE.id
        assert captured["cost_profile"].id == "legacy_runtime_v0"


# ---------------------------------------------------------------------------
# WP21-T-24 -- orchestrator wiring
# ---------------------------------------------------------------------------


class _RecordingEngine(_FakeEngine):
    instances: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        _RecordingEngine.instances.append(dict(kwargs))


class TestT24Orchestrator:
    async def test_paper_engines_get_the_runs_profile(self) -> None:
        profile = get_cost_profile("coinbase_advanced_d6")
        _RecordingEngine.instances.clear()
        paper_cls = MagicMock()
        run_row = _make_run_row(status="running")
        with (
            patch("api.db.session.get_session_factory", return_value=_session_factory_for(run_row)),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("trading.engines.paper.PaperExecutionEngine", paper_cls),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.risk_manager.DefaultRiskManager", _risk_mgr_cls()),
            patch("trading.strategy_engine.StrategyEngine", _RecordingEngine),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
            patch.object(orchestrator, "_flush_incremental", AsyncMock()),
            patch.object(orchestrator, "_incremental_flush_loop", AsyncMock()),
            patch.object(orchestrator, "_auto_stop_after", AsyncMock()),
            patch.object(orchestrator, "_auto_retry_paper_run", AsyncMock()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await orchestrator.run_paper_engine(
                    run_id_str=str(run_row.id),
                    strategy_cls=MagicMock,
                    strategy_name="grid_trading",
                    strategy_params={},
                    symbols=["BTC/USD"],
                    timeframe=TimeFrame.ONE_HOUR,
                    initial_capital="10000",
                    cost_profile=profile,
                )
        assert paper_cls.call_args.kwargs["cost_profile"] is profile
        assert _RecordingEngine.instances[0]["cost_profile"] is profile

    async def test_live_engines_get_the_runs_profile_and_keep_the_live_drawdown_limit(self) -> None:
        profile = get_cost_profile("coinbase_advanced_d6")
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock()
        live_engine_cls = MagicMock()
        live_engine_cls.return_value = MagicMock()
        risk_cls = MagicMock()
        risk_cls.return_value = MagicMock(
            params=__import__("trading.risk", fromlist=["x"]).RiskParameters()
        )
        run_id = "cccc0000-0000-0000-0000-000000000021"
        _RUN_TASKS.pop(run_id, None)
        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, live_engine_cls),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, risk_cls),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await orchestrator.run_live_engine(
                run_id_str=run_id,
                strategy_cls=_make_strategy_cls_mock(),
                strategy_name="ma_crossover",
                strategy_params={},
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital="10000",
                cost_profile=profile,
            )
        assert live_engine_cls.call_args.kwargs["cost_profile"] is profile
        assert se_cls_mock.call_args.kwargs["cost_profile"] is profile
        # the cost refactor left the live RISK limits alone (R-02: 20% drawdown)
        assert risk_cls.call_args.kwargs["params"].max_drawdown_pct == 0.20

    def test_cost_profile_is_a_required_keyword_of_both_start_sites(self) -> None:
        import inspect

        for fn in (orchestrator.run_paper_engine, orchestrator.run_live_engine):
            param = inspect.signature(fn).parameters["cost_profile"]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# WP21-T-25 -- run_orm_to_response
# ---------------------------------------------------------------------------


def _run(config: dict[str, Any], *, mode: str = "backtest", n_trades: int | None = None) -> Any:
    return SimpleNamespace(
        id=uuid.UUID("aaaaaaaa-0000-0000-0000-000000000021"),
        run_mode=mode,
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_trades,
    )


_SEM = {"schema": 1, "cost_model": "wp21-1", "exit_semantics": "wp13b-1"}


class TestT25RunOrmToResponse:
    def test_a_legacy_run_gets_the_inferred_profile_and_flags(self) -> None:
        resp = run_orm_to_response(_run({"strategy_name": "x"}))
        assert resp.legacy_costs is True and resp.cost_profile_invalid is False
        assert resp.legacy_exit_semantics is True
        assert resp.cost_profile is not None
        assert (resp.cost_profile.id, resp.cost_profile.source) == (
            "legacy_backtest_v0",
            "legacy_inferred",
        )
        assert resp.cost_profile.round_trip_bps == "40"
        # paper/live legacy runs infer the RUNTIME legacy profile
        assert run_orm_to_response(_run({}, mode="paper")).cost_profile.id == "legacy_runtime_v0"  # type: ignore[union-attr]

    def test_a_registry_legacy_id_is_flagged_legacy(self) -> None:
        cfg = {
            "cost_profile": get_cost_profile("legacy_backtest_v0").to_metadata(),
            "engine_semantics": _SEM,
        }
        resp = run_orm_to_response(_run(cfg))
        assert resp.legacy_costs is True
        assert resp.cost_profile_invalid is False and resp.legacy_exit_semantics is False
        assert resp.cost_profile is not None and resp.cost_profile.legacy is True

    def test_a_current_run_is_not_legacy(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(), "engine_semantics": _SEM}
        resp = run_orm_to_response(_run(cfg))
        assert (resp.legacy_costs, resp.cost_profile_invalid, resp.legacy_exit_semantics) == (
            False,
            False,
            False,
        )
        assert resp.cost_profile is not None
        assert resp.cost_profile.id == "coinbase_advanced_1"
        assert resp.cost_profile.live_equivalent is True
        assert resp.cost_profile.fee_source == "model"

    @pytest.mark.parametrize("bad", [{"junk": 1}, "text", 7, [], {"id": "coinbase_advanced_1"}])
    def test_a_malformed_profile_is_legacy_and_invalid_and_never_raises(self, bad: object) -> None:
        resp = run_orm_to_response(_run({"cost_profile": bad}))
        assert resp.legacy_costs is True and resp.cost_profile_invalid is True
        assert resp.cost_profile is not None and resp.cost_profile.source == "legacy_inferred"
        detail = run_orm_to_detail_response(_run({"cost_profile": bad}))
        assert detail.cost_profile_invalid is True

    def test_tampered_numbers_are_invalid_too(self) -> None:
        meta = DEFAULT_COST_PROFILE.to_metadata()
        meta["taker_fee_bps"] = "1"  # fingerprint no longer matches
        assert run_orm_to_response(_run({"cost_profile": meta})).cost_profile_invalid is True

    def test_the_flags_are_set_on_the_no_metrics_early_return_path(self) -> None:
        # no backtest_metrics at all -> run_orm_to_response returns before the JSONB step
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(fee_source="exchange_fills")}
        resp = run_orm_to_response(_run(cfg, mode="live"))
        assert resp.confidence_flag is None and resp.psr is None
        assert resp.cost_profile is not None and resp.cost_profile.fee_source == "exchange_fills"
        assert resp.legacy_exit_semantics is True  # engine_semantics absent

    def test_the_detail_response_carries_the_same_fields(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(), "engine_semantics": _SEM}
        detail = run_orm_to_detail_response(_run(cfg))
        assert detail.cost_profile is not None and detail.cost_profile.id == "coinbase_advanced_1"
        assert detail.legacy_costs is False

    def test_wire_names_are_camel_case(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(), "engine_semantics": _SEM}
        body = run_orm_to_response(_run(cfg)).model_dump(by_alias=True, mode="json")
        assert {"costProfile", "legacyCosts", "costProfileInvalid", "legacyExitSemantics"} <= set(
            body
        )
        assert {"makerFeeBps", "takerFeeBps", "slippageBps", "roundTripBps", "feeSource"} <= set(
            body["costProfile"]
        )


# ---------------------------------------------------------------------------
# WP21-T-26 -- eligibility
# ---------------------------------------------------------------------------


def _cfg(profile: CostProfile | None, **extra: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(extra)
    if profile is not None:
        cfg["cost_profile"] = profile.to_metadata()
    return cfg


class TestT26Eligibility:
    def test_a_live_equivalent_registry_profile_is_eligible(self) -> None:
        for pid in ("coinbase_advanced_1", "coinbase_advanced_d6"):
            assert leaderboard_eligible(50, "high", _cfg(get_cost_profile(pid))) is True

    @pytest.mark.parametrize(
        "pid",
        [
            "coinbase_stress_t80",
            "coinbase_stress_t100",
            "frictionless_diagnostic",
            "legacy_backtest_v0",
            "legacy_runtime_v0",
        ],
    )
    def test_registry_non_live_equivalent_ids_are_never_eligible(self, pid: str) -> None:
        assert leaderboard_eligible(500, "high", _cfg(get_cost_profile(pid))) is False

    def test_absent_override_and_invalid_are_never_eligible(self) -> None:
        assert leaderboard_eligible(500, "high", {}) is False  # legacy: no profile key
        assert leaderboard_eligible(500, "high", None) is False
        override = get_cost_profile("coinbase_advanced_d6").with_overrides(
            taker_fee_bps=__import__("decimal").Decimal(55)
        )
        assert leaderboard_eligible(500, "high", _cfg(override)) is False
        assert leaderboard_eligible(500, "high", {"cost_profile": {"junk": 1}}) is False

    def test_the_trade_and_confidence_thresholds_still_apply(self) -> None:
        good = _cfg(DEFAULT_COST_PROFILE)
        assert leaderboard_eligible(10, "medium", good) is True
        assert leaderboard_eligible(9, "high", good) is False
        assert leaderboard_eligible(50, "low", good) is False
        assert leaderboard_eligible(50, None, good) is False
        assert leaderboard_eligible(None, "high", good) is False
        assert leaderboard_eligible(True, "high", good) is False  # a bool is not a trade count

    def test_run_response_uses_the_same_helper(self) -> None:
        for profile, expected in (
            (DEFAULT_COST_PROFILE, True),
            (get_cost_profile("coinbase_stress_t80"), False),
            (get_cost_profile("legacy_backtest_v0"), False),
            (None, False),
        ):
            cfg = _cfg(profile, backtest_metrics={"confidence_flag": "high", "psr": 0.97})
            ns = _run(cfg, n_trades=40)
            assert run_orm_to_response(ns).leaderboard_eligible is expected
            direct = RunResponse.model_validate(ns)  # the model validator, no JSONB step
            assert direct.leaderboard_eligible is False  # flag unknown there -> never True

    def test_both_call_sites_use_the_one_helper(self) -> None:
        import inspect

        from api.routers import portfolio as portfolio_router

        src = inspect.getsource(portfolio_router.get_aggregate_portfolio)
        assert "leaderboard_eligible(" in src
        assert "LEADERBOARD_MIN_CLOSED_TRADES" not in src  # no second copy of the rule


class TestT26PortfolioAggregate:
    async def test_best_worst_exclude_non_eligible_cost_models_and_count_them(self) -> None:
        from api.routers.portfolio import get_aggregate_portfolio

        def _metrics(ret: float) -> dict[str, Any]:
            return {
                "backtest_metrics": {
                    "total_trades": 40,
                    "confidence_flag": "high",
                    "total_return_pct": ret,
                }
            }

        configs = [
            {"initial_capital": "1000", **_cfg(DEFAULT_COST_PROFILE), **_metrics(0.10)},
            {
                "initial_capital": "1000",
                **_cfg(get_cost_profile("coinbase_advanced_d6")),
                **_metrics(0.20),
            },
            # excluded by cost model despite 40 trades / high confidence:
            {
                "initial_capital": "1000",
                **_cfg(get_cost_profile("frictionless_diagnostic")),
                **_metrics(9.0),
            },
            {
                "initial_capital": "1000",
                **_cfg(get_cost_profile("coinbase_stress_t80")),
                **_metrics(8.0),
            },
            {"initial_capital": "1000", **_metrics(7.0)},  # legacy: no profile
            # excluded by thresholds only (NOT counted as a cost exclusion):
            {
                "initial_capital": "1000",
                **_cfg(DEFAULT_COST_PROFILE),
                "backtest_metrics": {
                    "total_trades": 3,
                    "confidence_flag": "high",
                    "total_return_pct": 5.0,
                },
            },
        ]
        db = AsyncMock()
        status_result = MagicMock()
        status_result.all.return_value = []
        trade_result = MagicMock()
        trade_result.one.return_value = SimpleNamespace(total=0, wins=0, losses=0, pnl=0, fees=0)
        cfg_result = MagicMock()
        cfg_result.all.return_value = [(c,) for c in configs]
        db.execute = AsyncMock(side_effect=[status_result, trade_result, cfg_result])
        resp = await get_aggregate_portfolio(db=db)
        assert resp.eligible_runs_count == 2
        assert resp.best_run_return_pct == pytest.approx(0.20)
        assert resp.worst_run_return_pct == pytest.approx(0.10)
        assert resp.legacy_cost_runs_excluded == 3
