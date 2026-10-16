"""
tests/unit/test_wp21c_breaker_default.py
------------------------------------------
WP2.1c (SY-21-13, closes CF-26): the circuit breaker is ON by default in backtests.

    WP21-T-36
        * ``BacktestRunner`` and ``ParameterOptimizer`` default ``enable_circuit_breaker=True``
          (one constant, ``BACKTEST_CIRCUIT_BREAKER_DEFAULT``); the optimizer forwards it to
          every combination's runner (GAP-05)
        * ``engine_semantics.backtest_circuit_breaker`` is ``True`` on backtest runs and on
          optimization ``run_metadata`` and ABSENT on paper / live (SY-21-11)
        * a stored ``False`` (2.1a / 2.1b rows) is not misread: legacy detection, leaderboard
          eligibility and the response fields do not look at the key
        * the explicit ``False`` path is byte-identical to the 2.1a golden (T-32); the default
          is not
        * every caller is covered: the ML gate, both research scripts (``--no-circuit-breaker``)
        * WP21C-D-12: ``create_run`` stamps the same constant it hands to the runner, and both
          research scripts initialise ``_ENABLE_CB`` from it (no second literal ``True``)

The API half (a created backtest / optimization persists ``True``) is in
``tests/integration/test_wp21_cost_api.py`` and ``test_wp21b_optimize_api.py``.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import trading.backtest as backtest_module
import trading.optimizer as optimizer_module
from api.services.cost_profiles import (
    cost_response_fields,
    engine_semantics,
    stamp_cost_config,
)
from common.types import TimeFrame
from tests.conftest import make_bars
from tests.integration.test_wp16_cb_backtest import (
    GOLDEN_REFERENCE_DIGEST,
    _digest,
    _run,
    _runner,
)
from tests.unit.test_sprint50_cycle6_oos_backtest import TestTrainWfAggregation
from trading.backtest import BacktestRunner
from trading.costs import (
    DEFAULT_COST_PROFILE,
    get_cost_profile,
    is_legacy_cost_config,
)
from trading.costs_cli import (
    add_circuit_breaker_arg,
    circuit_breaker_from_args,
    circuit_breaker_header,
)
from trading.optimizer import ParameterOptimizer
from trading.safety import BACKTEST_CIRCUIT_BREAKER_DEFAULT
from trading.strategies.ma_crossover import MACrossoverStrategy

_REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> Any:
    path = _REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wp21c_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _optimizer(**kwargs: Any) -> ParameterOptimizer:
    return ParameterOptimizer(
        strategy_cls=MACrossoverStrategy,
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The defaults
# ---------------------------------------------------------------------------


class TestDefaultsAreOn:
    def test_the_one_constant_is_true(self) -> None:
        assert BACKTEST_CIRCUIT_BREAKER_DEFAULT is True

    def test_the_signature_defaults_are_literally_true(self) -> None:
        for cls in (BacktestRunner, ParameterOptimizer):
            param = inspect.signature(cls.__init__).parameters["enable_circuit_breaker"]
            assert param.default is True, cls.__name__
            assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_the_default_runner_builds_a_breaker(self) -> None:
        runner, _bars = _runner("reduce_daily", enable=None)
        assert runner.circuit_breaker_enabled is True
        engine, *_ = runner._build_engine("t36-default")
        assert engine.circuit_breaker is not None

    def test_an_explicit_false_builds_none(self) -> None:
        runner, _bars = _runner("reduce_daily", enable=False)
        assert runner.circuit_breaker_enabled is False
        engine, *_ = runner._build_engine("t36-off")
        assert engine.circuit_breaker is None

    def test_the_default_optimizer_is_on_and_the_flag_is_readable(self) -> None:
        assert _optimizer().circuit_breaker_enabled is True
        assert _optimizer(enable_circuit_breaker=False).circuit_breaker_enabled is False

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, True),
            ({"enable_circuit_breaker": True}, True),
            ({"enable_circuit_breaker": False}, False),
        ],
    )
    async def test_the_optimizer_forwards_the_flag_to_every_combination(
        self, kwargs: dict[str, Any], expected: bool
    ) -> None:
        seen: list[bool] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw["enable_circuit_breaker"])  # always passed explicitly
            original_init(self, *args, **kw)

        bars = {"BTC/USDT": make_bars(300, symbol="BTC/USDT")}
        with patch.object(BacktestRunner, "__init__", _spy):
            await _optimizer(**kwargs).run(bars)
        assert seen == [expected] * 4  # 2 x 2 combinations

    async def test_the_walk_forward_optimizer_forwards_it_too(self) -> None:
        seen: list[bool] = []
        original_init = optimizer_module.BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw["enable_circuit_breaker"])
            original_init(self, *args, **kw)

        bars = {"BTC/USDT": make_bars(700, symbol="BTC/USDT")}
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            param_grid={"fast_period": [5], "slow_period": [20]},
            walk_forward=True,
            walk_forward_folds=2,
            enable_circuit_breaker=False,
        )
        with patch.object(BacktestRunner, "__init__", _spy):
            await opt.run(bars)
        assert seen and set(seen) == {False}


# ---------------------------------------------------------------------------
# engine_semantics.backtest_circuit_breaker (SY-21-11)
# ---------------------------------------------------------------------------


class TestEngineSemantics:
    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_backtest_and_optimize_persist_true(self, mode: str) -> None:
        assert engine_semantics(mode)["backtest_circuit_breaker"] is True

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_paper_and_live_do_not_carry_the_field(self, mode: str) -> None:
        assert "backtest_circuit_breaker" not in engine_semantics(mode)

    def test_the_stamp_is_the_value_the_default_runner_runs_with(self) -> None:
        stamped = engine_semantics("backtest")["backtest_circuit_breaker"]
        runner, _ = _runner("halt", enable=None)
        assert stamped is runner.circuit_breaker_enabled is _optimizer().circuit_breaker_enabled

    def test_stamp_cost_config_writes_true_for_both_run_kinds(self) -> None:
        for mode in ("backtest", "optimize"):
            config: dict[str, Any] = {}
            stamp_cost_config(config, DEFAULT_COST_PROFILE, mode=mode)
            assert config["engine_semantics"]["backtest_circuit_breaker"] is True
        paper: dict[str, Any] = {}
        stamp_cost_config(paper, DEFAULT_COST_PROFILE, mode="paper")
        assert "backtest_circuit_breaker" not in paper["engine_semantics"]

    def test_an_explicit_false_can_still_be_stamped(self) -> None:
        config: dict[str, Any] = {}
        stamp_cost_config(
            config, DEFAULT_COST_PROFILE, mode="optimize", backtest_circuit_breaker=False
        )
        assert config["engine_semantics"]["backtest_circuit_breaker"] is False

    def test_nothing_but_the_flag_changed_in_the_semantics_block(self) -> None:
        assert engine_semantics("backtest") == {
            "schema": 1,
            "cost_model": "wp21-1",
            "exit_semantics": "wp13b-1",
            "daily_risk": "wp15-1",
            "backtest_circuit_breaker": True,
        }


class TestStoredFalseIsNotMisread:
    """Rows stamped ``false`` by 2.1a / 2.1b keep it, and nothing reads it as legacy."""

    @staticmethod
    def _config(flag: object | None) -> dict[str, Any]:
        semantics: dict[str, Any] = dict(engine_semantics("backtest"))
        if flag is None:
            semantics.pop("backtest_circuit_breaker")
        else:
            semantics["backtest_circuit_breaker"] = flag
        return {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(), "engine_semantics": semantics}

    @pytest.mark.parametrize("flag", [False, True, None, "yes", 1])
    def test_the_flag_never_changes_legacy_detection_or_eligibility(self, flag: object) -> None:
        cfg = self._config(flag)
        assert is_legacy_cost_config(cfg) is False
        fields = cost_response_fields(cfg, "backtest")
        reference = cost_response_fields(self._config(True), "backtest")
        assert fields == reference
        assert fields["legacy_costs"] is False
        assert fields["legacy_exit_semantics"] is False
        assert fields["live_equivalent_costs"] is True

    def test_stamping_a_new_run_does_not_touch_a_stored_config(self) -> None:
        stored = self._config(False)
        snapshot = {
            "cost_profile": dict(stored["cost_profile"]),
            "engine_semantics": dict(stored["engine_semantics"]),
        }
        # nothing but ``stamp_cost_config`` writes the block, and it only ever writes a NEW dict
        fresh: dict[str, Any] = {}
        stamp_cost_config(fresh, DEFAULT_COST_PROFILE, mode="backtest")
        assert stored == snapshot
        assert stored["engine_semantics"]["backtest_circuit_breaker"] is False


# ---------------------------------------------------------------------------
# The explicit False path is the 2.1a golden; the default is not
# ---------------------------------------------------------------------------


class TestExplicitFalseIsByteIdentical:
    async def test_explicit_false_reproduces_the_21a_t32_golden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = await _run("reduce_daily", enable=False, monkeypatch=monkeypatch)
        assert out["breakers"] == [] and out["transitions"] == []
        assert _digest(out) == GOLDEN_REFERENCE_DIGEST

    async def test_the_default_is_the_breaker_on_path_and_differs_from_the_golden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default = await _run("reduce_daily", enable=None, monkeypatch=monkeypatch)
        on = await _run("reduce_daily", enable=True, monkeypatch=monkeypatch)
        assert default["transitions"] and default["transitions"] == on["transitions"]
        assert _digest(default) == _digest(on)
        assert _digest(default) != GOLDEN_REFERENCE_DIGEST  # the breaker halved the REDUCE-band BUY


# ---------------------------------------------------------------------------
# Callers
# ---------------------------------------------------------------------------


class TestEveryCallerIsCovered:
    def test_the_ml_gate_runs_its_backtests_with_the_breaker(self) -> None:
        seen: list[bool] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw["enable_circuit_breaker"])
            original_init(self, *args, **kw)

        with patch.object(BacktestRunner, "__init__", _spy):
            TestTrainWfAggregation()._run(
                fold_sharpes=[1.2, 0.8, 1.6], fold_trades=[10, 12, 8], fold_dds=[0.1, 0.1, 0.1]
            )
        assert seen == [True] * 3

    def test_the_api_backtest_and_optimize_paths_pass_the_constant(self) -> None:
        for rel in (
            "apps/api/routers/runs.py",
            "apps/api/routers/optimize.py",
            "apps/api/routers/ml.py",
        ):
            text = (_REPO / rel).read_text()
            assert "enable_circuit_breaker=BACKTEST_CIRCUIT_BREAKER_DEFAULT" in text, rel
        # there is NO request field for it (SY-21-13: no API toggle)
        schemas = (_REPO / "apps/api/schemas.py").read_text()
        optimize = (_REPO / "apps/api/routers/optimize.py").read_text()
        for text in (schemas, optimize):
            assert "enableCircuitBreaker" not in text
            assert "enable_circuit_breaker: " not in text

    @pytest.mark.parametrize("script", ["research_momentum_brackets", "validate_sl_tp_reversion"])
    def test_the_research_scripts_default_on_and_take_no_circuit_breaker(self, script: str) -> None:
        mod = _load_script(script)
        args, _profile = mod.parse_args([])
        assert args.no_circuit_breaker is False
        assert circuit_breaker_from_args(args) is True
        assert mod._ENABLE_CB is True  # the module-level default
        # WP21C-D-12: ... which is the one constant, not a second literal
        assert mod._ENABLE_CB is BACKTEST_CIRCUIT_BREAKER_DEFAULT
        off, _ = mod.parse_args(["--no-circuit-breaker"])
        assert circuit_breaker_from_args(off) is False
        assert (
            "enable_circuit_breaker=_ENABLE_CB" in (_REPO / "scripts" / f"{script}.py").read_text()
        )

    @pytest.mark.parametrize("script", ["research_momentum_brackets", "validate_sl_tp_reversion"])
    def test_the_scripts_initialise_the_flag_from_the_constant_not_a_literal(
        self, script: str
    ) -> None:
        """WP21C-D-12 (SEC-09): ``_ENABLE_CB: bool = BACKTEST_CIRCUIT_BREAKER_DEFAULT``."""
        tree = ast.parse((_REPO / "scripts" / f"{script}.py").read_text())
        inits = [
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_ENABLE_CB"
        ]
        assert len(inits) == 1
        value = inits[0].value
        assert isinstance(value, ast.Name) and value.id == "BACKTEST_CIRCUIT_BREAKER_DEFAULT"
        mod = _load_script(script)
        assert mod.BACKTEST_CIRCUIT_BREAKER_DEFAULT is BACKTEST_CIRCUIT_BREAKER_DEFAULT

    def test_create_run_stamps_the_constant_it_hands_to_the_runner(self) -> None:
        """WP21C-D-12 (SEC-09): the stamp (``runs.py``, before the runner) names the constant.

        The behavioural twin is in ``tests/integration/test_wp21_cost_api.py``.  Every
        ``stamp_cost_config`` call that can stamp a backtest (anything but the literal
        ``mode="live"`` of the promote path) must pass ``backtest_circuit_breaker=`` explicitly.
        """
        tree = ast.parse((_REPO / "apps/api/routers/runs.py").read_text())
        checked = 0
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "stamp_cost_config"
            ):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            mode = kwargs["mode"]
            if isinstance(mode, ast.Constant) and mode.value == "live":
                assert "backtest_circuit_breaker" not in kwargs  # promote: live carries no key
                continue
            flag = kwargs.get("backtest_circuit_breaker")
            assert isinstance(flag, ast.Name), "create_run relies on the stamp's own default"
            assert flag.id == "BACKTEST_CIRCUIT_BREAKER_DEFAULT"
            checked += 1
        assert checked == 1  # exactly the create_run stamp

    def test_the_cli_helpers(self) -> None:
        parser = argparse.ArgumentParser()
        add_circuit_breaker_arg(parser)
        assert circuit_breaker_from_args(parser.parse_args([])) is True
        assert circuit_breaker_from_args(parser.parse_args(["--no-circuit-breaker"])) is False
        assert circuit_breaker_from_args(argparse.Namespace()) is True  # no flag defined at all
        assert circuit_breaker_header(True) == "circuit_breaker=on"
        assert circuit_breaker_header(False) == "circuit_breaker=off"

    def test_the_default_profile_and_registry_are_untouched(self) -> None:
        assert DEFAULT_COST_PROFILE is get_cost_profile("coinbase_advanced_1")
        assert backtest_module.BACKTEST_CIRCUIT_BREAKER_DEFAULT is BACKTEST_CIRCUIT_BREAKER_DEFAULT
