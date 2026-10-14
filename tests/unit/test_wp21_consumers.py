"""
tests/unit/test_wp21_consumers.py
-----------------------------------
WP2.1a: every cost CONSUMER reads the one profile.

    WP21-T-15  resolution parity: backtest, paper, live (estimate), optimizer, ML gate and both
               research scripts resolve the SAME default fingerprint without a request
    WP21-T-17  optimizer: ONE profile object reaches every combination; the result carries it;
               the grid is validated at ``policy_cost_model``
    WP21-T-18  ML gate passes the DEFAULT profile and records it in ``ModelVersionORM.extra``
    WP21-T-19  research scripts: default / ``--cost-profile`` / numeric overrides / header
    WP21-T-29  DS-1 precheck ``--cost-profile`` lists a persisted run with SL 0.0068 as
               "would fail at promote" (read-only)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import uuid
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import trading.backtest as backtest_module
from common.types import TimeFrame
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.unit.test_sprint50_cycle6_oos_backtest import TestTrainWfAggregation
from trading.backtest import BacktestRunner
from trading.costs import (
    DEFAULT_COST_PROFILE,
    CostProfileError,
    default_cost_profile,
    get_cost_profile,
    new_run_cost_profile,
)
from trading.costs_cli import add_cost_profile_args, cost_header, cost_profile_from_args
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine
from trading.exit_config import ExitConfigError
from trading.optimizer import ParameterOptimizer
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategy_engine import StrategyEngine

_REPO = Path(__file__).resolve().parents[2]
_FP = DEFAULT_COST_PROFILE.fingerprint()


def _load_script(name: str) -> Any:
    path = _REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wp21_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# WP21-T-17 -- optimizer
# ---------------------------------------------------------------------------


def _optimizer(**kwargs: Any) -> ParameterOptimizer:
    return ParameterOptimizer(
        strategy_cls=MACrossoverStrategy,
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
        **kwargs,
    )


class TestT17Optimizer:
    def test_default_profile_is_the_default(self) -> None:
        assert _optimizer()._cost_profile is DEFAULT_COST_PROFILE

    async def test_one_profile_object_reaches_every_combination_and_the_result(self) -> None:
        from tests.conftest import make_bars

        profile = get_cost_profile("coinbase_stress_t80")
        seen: list[Any] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw.get("cost_profile"))
            original_init(self, *args, **kw)

        bars = {"BTC/USDT": make_bars(300, symbol="BTC/USDT")}
        with patch.object(BacktestRunner, "__init__", _spy):
            result = await _optimizer(cost_profile=profile).run(bars)
        assert len(seen) == 4  # 2 x 2 combinations
        assert all(p is profile for p in seen)
        assert result.cost_profile is profile

    def test_the_grid_is_validated_at_the_policy_basis_of_the_profile(self) -> None:
        # SL 0.010 is above the legacy backtest profile's OWN floor (0.0020) but below the
        # stricter policy basis (the default's 0.0130): the optimizer rejects it up front,
        # exactly like the router (SY-21-06), instead of failing inside a combination.
        class _Strat(MACrossoverStrategy):
            pass

        with pytest.raises(ExitConfigError):
            ParameterOptimizer(
                strategy_cls=_Strat,
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                param_grid={
                    "fast_period": [5],
                    "slow_period": [20],
                    "bracket_mode": ["fixed"],
                    "bracket_stop_loss_pct": [0.010],
                },
                cost_profile=get_cost_profile("legacy_backtest_v0"),
            )

    def test_the_old_bps_kwargs_are_gone(self) -> None:
        for kw in ("maker_fee_bps", "taker_fee_bps", "slippage_bps"):
            with pytest.raises(TypeError):
                _optimizer(**{kw: 5})


# ---------------------------------------------------------------------------
# WP21-T-18 -- ML gate
# ---------------------------------------------------------------------------


class TestT18MlGate:
    def test_the_walk_forward_gate_backtests_under_the_default_profile(self) -> None:
        seen: list[Any] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw.get("cost_profile"))
            original_init(self, *args, **kw)

        with patch.object(BacktestRunner, "__init__", _spy):
            TestTrainWfAggregation()._run(
                fold_sharpes=[1.2, 0.8, 1.6], fold_trades=[10, 12, 8], fold_dds=[0.1, 0.1, 0.1]
            )
        assert len(seen) == 3
        assert all(p is DEFAULT_COST_PROFILE for p in seen)

    async def test_the_profile_is_recorded_in_the_model_versions_extra(self) -> None:
        from api.routers import ml

        db = AsyncMock()
        db.add = MagicMock()
        result = {
            "status": "completed",
            "model_path": "models/x.joblib",
            "bars_fetched": 400,
            "metrics": {"accuracy": 0.5},
            "walk_forward_status": "ok",
            "walk_forward_oos_skill_score": 1.0,
        }
        with patch.object(ml, "_train_model_with_wf_sync", return_value=result):
            await ml.train_model(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
                bars=400,
                n_estimators=10,
                horizon=1,
                threshold=0.01,
                num_wf_folds=3,
                db=db,
            )
        version = next(
            c.args[0]
            for c in db.add.call_args_list
            if type(c.args[0]).__name__ == "ModelVersionORM"
        )
        assert version.extra["cost_profile"] == DEFAULT_COST_PROFILE.to_metadata()
        assert version.extra["walk_forward"]["status"] == "ok"  # existing keys are untouched
        assert version.extra["accuracy"] == 0.5


# ---------------------------------------------------------------------------
# WP21-T-19 -- research scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["research_momentum_brackets", "validate_sl_tp_reversion"])
class TestT19Scripts:
    def test_default_is_the_default_profile(self, script: str) -> None:
        mod = _load_script(script)
        _args, profile = mod.parse_args([])
        assert profile is DEFAULT_COST_PROFILE
        assert mod._COST_PROFILE is DEFAULT_COST_PROFILE  # the module-level default

    def test_cost_profile_flag_selects_a_registry_id(self, script: str) -> None:
        mod = _load_script(script)
        _args, profile = mod.parse_args(["--cost-profile", "coinbase_stress_t80"])
        assert profile is get_cost_profile("coinbase_stress_t80")

    def test_numeric_override_makes_a_custom_override_profile(self, script: str) -> None:
        mod = _load_script(script)
        _args, profile = mod.parse_args(["--taker-bps", "90"])
        assert (profile.id, profile.source, profile.base_id) == (
            "custom",
            "override",
            DEFAULT_COST_PROFILE.id,
        )
        assert profile.taker_fee_bps == Decimal(90)
        assert profile.maker_fee_bps == DEFAULT_COST_PROFILE.maker_fee_bps
        assert profile.live_equivalent is False

    def test_overrides_combine_with_a_registry_base(self, script: str) -> None:
        mod = _load_script(script)
        _args, profile = mod.parse_args(
            ["--cost-profile", "coinbase_advanced_d6", "--slippage-bps", "25", "--maker-bps", "30"]
        )
        assert profile.base_id == "coinbase_advanced_d6"
        assert (profile.slippage_bps, profile.maker_fee_bps, profile.taker_fee_bps) == (
            Decimal(25),
            Decimal(30),
            Decimal(60),
        )

    def test_invalid_input_is_a_usage_error_not_a_traceback(self, script: str) -> None:
        mod = _load_script(script)
        for argv in (
            ["--cost-profile", "nope"],
            ["--taker-bps", "abc"],
            ["--taker-bps", "NaN"],
            ["--taker-bps", "1000"],  # above the CostProfile bound -> CostProfileError
            ["--maker-bps", "500", "--cost-profile", "coinbase_advanced_d6"],
        ):
            with pytest.raises(SystemExit) as exc:
                mod.parse_args(argv)
            assert exc.value.code == 2

    def test_the_runner_gets_the_script_profile(self, script: str) -> None:
        mod = _load_script(script)
        assert "cost_profile=_COST_PROFILE" in (_REPO / "scripts" / f"{script}.py").read_text()
        assert mod._COST_PROFILE is default_cost_profile()


def test_t19_the_header_names_id_fingerprint_source_and_rates() -> None:
    header = cost_header(DEFAULT_COST_PROFILE)
    assert f"id={DEFAULT_COST_PROFILE.id}" in header
    assert f"fingerprint={_FP}" in header
    assert "source=registry" in header and "taker=120bps" in header and "slippage=10bps" in header
    custom = DEFAULT_COST_PROFILE.with_overrides(taker_fee_bps=Decimal(90))
    assert "id=custom" in cost_header(custom) and "base=coinbase_advanced_1" in cost_header(custom)


def test_t19_cost_profile_from_args_without_flags_is_the_default() -> None:
    parser = argparse.ArgumentParser()
    add_cost_profile_args(parser)
    assert cost_profile_from_args(parser.parse_args([])) is DEFAULT_COST_PROFILE
    with pytest.raises(CostProfileError):
        cost_profile_from_args(parser.parse_args(["--taker-bps", "300"]))


# ---------------------------------------------------------------------------
# WP21-T-15 -- resolution parity
# ---------------------------------------------------------------------------


class TestT15ResolutionParity:
    def test_every_consumer_resolves_the_same_default_without_a_request(self) -> None:
        strat = ScriptedSequenceStrategy("t15", {"script": {}})
        runner = BacktestRunner(
            strategies=[strat], symbols=["BTC/USDT"], timeframe=TimeFrame.ONE_HOUR, seed=1
        )
        paper = PaperExecutionEngine(run_id="t15", risk_manager=MagicMock())
        engine = StrategyEngine(
            strategies=[strat],
            execution_engine=MagicMock(),
            risk_manager=MagicMock(),
            market_data=MagicMock(),
            portfolio=MagicMock(),
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            run_mode=__import__("common.types", fromlist=["RunMode"]).RunMode.BACKTEST,
        )
        optimizer = _optimizer()
        fingerprints = {
            "backtest": runner.cost_profile.fingerprint(),
            "paper_engine": paper.cost_profile.fingerprint(),
            "strategy_engine": engine.cost_profile.fingerprint(),
            "optimizer": optimizer._cost_profile.fingerprint(),
            "new_backtest": new_run_cost_profile(None, mode="backtest").fingerprint(),
            "new_optimize": new_run_cost_profile(None, mode="optimize").fingerprint(),
            "new_paper": new_run_cost_profile(None, mode="paper").fingerprint(),
            "new_live": new_run_cost_profile(None, mode="live").fingerprint(),
            "promote": default_cost_profile().fingerprint(),
            "script_a": _load_script("research_momentum_brackets").parse_args([])[1].fingerprint(),
            "script_b": _load_script("validate_sl_tp_reversion").parse_args([])[1].fingerprint(),
            "ml_gate": DEFAULT_COST_PROFILE.fingerprint(),
        }
        assert set(fingerprints.values()) == {_FP}, fingerprints

    def test_the_live_engine_holds_the_estimate_it_was_given(self) -> None:
        exchange = MagicMock()
        exchange.id = "coinbase"
        live = LiveExecutionEngine(
            run_id="t15l",
            risk_manager=MagicMock(),
            exchange=exchange,
            enable_live_trading=True,
            cost_profile=DEFAULT_COST_PROFILE,
        )
        assert live.cost_profile is DEFAULT_COST_PROFILE
        assert (
            LiveExecutionEngine(
                run_id="t15m", risk_manager=MagicMock(), exchange=exchange
            ).cost_profile
            is None
        )  # no profile -> no drift checks

    def test_the_ml_gate_source_passes_the_default(self) -> None:
        # (the behavioural proof is TestT18MlGate; this pins that no literal bps remain)
        src = (_REPO / "apps/api/routers/ml.py").read_text()
        assert "cost_profile=default_cost_profile()" in src
        assert "taker_fee_bps" not in src and "maker_fee_bps" not in src

    def test_backtest_module_has_no_bps_kwargs_left(self) -> None:
        import inspect

        params = inspect.signature(backtest_module.BacktestRunner.__init__).parameters
        assert "cost_profile" in params
        assert not {"slippage_bps", "maker_fee_bps", "taker_fee_bps"} & set(params)


# ---------------------------------------------------------------------------
# WP21-T-29 -- DS-1 precheck
# ---------------------------------------------------------------------------


class TestT29Precheck:
    def _check(
        self, config: dict[str, Any], profile_id: str | None, mode: str = "live"
    ) -> dict[str, Any]:
        mod = _load_script("wp13a_exit_config_precheck")
        with patch(
            "api.routers.runs._get_strategy_registry",
            return_value={"grid_trading": MagicMock()},
        ):
            return mod._check_one_run(
                uuid.uuid4(), mode, "orphaned", config, promote_profile_id=profile_id
            )

    @staticmethod
    def _cfg(sl: float, **extra: Any) -> dict[str, Any]:
        return {
            "strategy_name": "grid_trading",
            "strategy_params": {},
            "timeframe": "1h",
            "allow_pyramiding": False,
            "bracket_config": {
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": sl,
                "bracket_take_profit_pct": 0.1,
            },
            **extra,
        }

    def test_sl_0068_resumes_but_would_fail_at_promote_under_the_default(self) -> None:
        entry = self._check(self._cfg(0.0068), "coinbase_advanced_1")
        assert entry["valid"] is True  # resume basis: the run's own (legacy) 0.0065 floor
        assert entry["would_fail_at_promote"] is True
        assert entry["promote_codes"] == ["stop_inside_entry_cost"]

    def test_without_the_flag_nothing_is_listed(self) -> None:
        entry = self._check(self._cfg(0.0068), None)
        assert entry["valid"] is True and entry["would_fail_at_promote"] is False

    def test_a_wide_stop_is_clean(self) -> None:
        entry = self._check(self._cfg(0.05), "coinbase_advanced_1")
        assert entry["valid"] is True and entry["would_fail_at_promote"] is False

    def test_a_run_below_its_own_floor_is_still_invalid(self) -> None:
        entry = self._check(self._cfg(0.006), None)  # <= the legacy side cost 0.0065
        assert entry["valid"] is False

    def test_a_persisted_default_run_is_judged_at_its_own_profile(self) -> None:
        cfg = self._cfg(0.0068, cost_profile=DEFAULT_COST_PROFILE.to_metadata())
        entry = self._check(cfg, None)
        assert entry["valid"] is False  # its own (post-2.1a) floor is 0.0130

    def test_unknown_profile_id_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod = _load_script("wp13a_exit_config_precheck")
        with patch.object(sys, "argv", ["precheck", "--cost-profile", "nope"]):
            assert mod.main() == 2
        assert "unknown --cost-profile" in capsys.readouterr().err

    async def test_the_run_is_read_only_and_reports_the_promote_blockers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mod = _load_script("wp13a_exit_config_precheck")
        run = SimpleNamespace(
            id=uuid.uuid4(), run_mode="live", status="orphaned", config=self._cfg(0.0068)
        )
        result = MagicMock()
        result.scalars.return_value.all.return_value = [run]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.connection = AsyncMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with (
            patch("api.db.session.get_session_factory", return_value=MagicMock(return_value=ctx)),
            patch(
                "api.routers.runs._get_strategy_registry",
                return_value={"grid_trading": MagicMock()},
            ),
        ):
            results = await mod._run(None, "coinbase_advanced_1")
        assert mod._print_report(results, as_json=False) == 0  # informational: exit code 0
        out = capsys.readouterr().out
        assert "would_fail_at_promote" in out and "1 would fail at promote" in out
        # read-only: only SELECTs, no writes of any kind
        session.add.assert_not_called()
        session.commit.assert_not_called()
        assert run.config == self._cfg(0.0068)
