"""
tests/unit/test_wp21a_exit_basis.py
------------------------------------
WP2.1a fix round (WP21A-CR-01 = WP21A-SEC-01): every exit-config parse uses the cost
basis of the run it belongs to, never a module default.

    WP21-T-39  ``resolve_run_exit_check`` takes a REQUIRED ``cost`` and parses with it:
               a legacy run (side cost 0.0065) with a fixed SL/TP in (0.0065, 0.0130]
               keeps its live intra-bar exit check; the same bracket at the default
               basis (0.0130) does not (so the basis is demonstrably honoured)
    WP21-T-40  AST pin: every production call of the exit-config validators passes a
               ``cost=`` keyword whose value is not the literal ``None``
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

from common.types import RunMode, TimeFrame
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile
from trading.exit_config import ExitCheckResolution, ExitCostModel, resolve_run_exit_check

_REPO = Path(__file__).resolve().parents[2]

_LEGACY = ExitCostModel.from_cost_profile(get_cost_profile("legacy_runtime_v0"))
_DEFAULT = ExitCostModel.from_cost_profile(DEFAULT_COST_PROFILE)

# Fixed brackets inside the band (0.0065, 0.0130]: valid at the legacy floor, below the
# default floor.  The second case adds a (valid) trailing stop, which does not rescue a
# bracket whose parse fails.
_SL_0068: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.0068,
    "bracket_take_profit_pct": 0.05,
}
_BAND_CASES = [
    pytest.param(_SL_0068, None, id="sl_0.0068"),
    pytest.param(
        {
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.008,
            "bracket_take_profit_pct": 0.03,
        },
        0.05,
        id="sl_0.008_plus_trailing",
    ),
    pytest.param(
        {
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.03,
            "bracket_take_profit_pct": 0.01,
        },
        None,
        id="tp_0.01",
    ),
]


def _resolve(
    bracket: dict[str, object], trailing: float | None, cost: ExitCostModel
) -> ExitCheckResolution:
    return resolve_run_exit_check(
        300,
        mode=RunMode.LIVE,
        timeframe=TimeFrame.ONE_HOUR,
        strategy_name="momentum_breakout",
        bracket=bracket,
        trailing_stop_pct=trailing,
        cost=cost,
    )


class TestT39ResolverUsesTheGivenBasis:
    @pytest.mark.parametrize(("bracket", "trailing"), _BAND_CASES)
    def test_the_legacy_basis_keeps_the_intrabar_check(
        self, bracket: dict[str, object], trailing: float | None
    ) -> None:
        got = _resolve(bracket, trailing, _LEGACY)
        assert (got.interval_s, got.reason) == (300, "enabled")

    @pytest.mark.parametrize(("bracket", "trailing"), _BAND_CASES)
    def test_the_default_basis_rejects_the_same_bracket(
        self, bracket: dict[str, object], trailing: float | None
    ) -> None:
        got = _resolve(bracket, trailing, _DEFAULT)
        assert (got.interval_s, got.reason) == (None, "no_protective_exit")

    def test_the_basis_is_required(self) -> None:
        with pytest.raises(TypeError):
            resolve_run_exit_check(  # type: ignore[call-arg]
                300,
                mode=RunMode.LIVE,
                timeframe=TimeFrame.ONE_HOUR,
                strategy_name="momentum_breakout",
                bracket=_SL_0068,
                trailing_stop_pct=None,
            )


_VALIDATORS = frozenset(
    {
        "parse_exit_config",
        "validate_run_exit_config",
        "salvage_exit_config",
        "validate_param_grid_exit_config",
        "resolve_run_exit_check",
        "_intrabar_exit_checks_effective",
    }
)
_MIN_SITES = 19  # the WP2.1a-fix call-site audit; a drop means the scan itself broke
_SKIP_DIRS = frozenset({"node_modules", ".venv", "venv", "__pycache__", ".git"})


def _production_files() -> list[Path]:
    out: list[Path] = []
    for root in ("apps", "packages", "scripts"):
        for dirpath, dirnames, filenames in os.walk(_REPO / root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            out.extend(Path(dirpath) / f for f in filenames if f.endswith(".py"))
    return sorted(out)


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class TestT40EveryValidatorCallPassesItsBasis:
    def test_every_production_call_passes_a_non_none_cost(self) -> None:
        calls = 0
        offenders: list[str] = []
        for path in _production_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and _callee(node) in _VALIDATORS):
                    continue
                calls += 1
                where = f"{path.relative_to(_REPO)}:{node.lineno}:{_callee(node)}"
                costs: list[Any] = [kw.value for kw in node.keywords if kw.arg == "cost"]
                if not costs:
                    offenders.append(f"{where}: no cost= keyword")
                elif isinstance(costs[0], ast.Constant) and costs[0].value is None:
                    offenders.append(f"{where}: cost=None")
        assert calls >= _MIN_SITES, f"scan found only {calls} call sites (expected >= {_MIN_SITES})"
        assert offenders == []
