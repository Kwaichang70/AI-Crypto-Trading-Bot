"""
tests/unit/test_wp15a_contracts.py
------------------------------------
WP1.5a -- AST contract tests (WP15-T-18).

These pin structural promises that a behavioural test could miss:

* no wall-clock read on the risk-day path (determinism, AC-08);
* the paper engine's risk-state branch never falls back to its own legacy
  ledger helpers (the source of the C5 peak / drawdown error);
* R-02 stays narrow: only the live construction site uses the 20% limit.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from trading.engines.paper import PaperExecutionEngine
from trading.portfolio import PortfolioAccounting

_REPO = Path(__file__).resolve().parents[2]
_ORCHESTRATOR = _REPO / "apps" / "api" / "services" / "run_orchestrator.py"

#: Calls that read the wall clock.
_WALL_CLOCK_ATTRS = {"now", "utcnow", "today", "time", "monotonic", "time_ns"}


def _function(owner: type, name: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(owner, name))))
    fn = tree.body[0]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    return fn  # type: ignore[return-value]


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


@pytest.mark.parametrize(
    "method",
    [
        "_now",
        "_ensure_daily_pnl_date",
        "advance_clock",
        "risk_snapshot",
        "get_daily_pnl",
        "reset_daily_pnl",
        "_realised_today",
        "_record_nav",
        "_seed_daily_state",
    ],
)
def test_wp15_t18_no_wall_clock_read_on_the_risk_day_path(method: str) -> None:
    fn = _function(PortfolioAccounting, method)
    offenders = _called_names(fn) & _WALL_CLOCK_ATTRS
    assert not offenders, f"PortfolioAccounting.{method} reads the wall clock via {offenders}"


def test_wp15_t18_the_single_clock_seam_is_the_module_function() -> None:
    import trading.portfolio as module

    src = inspect.getsource(module._utc_now)
    assert "datetime.now" in src  # the one place the wall clock is read
    # ... and `_now` reaches it only through the seam (looked up at call time)
    assert "_utc_now" in inspect.getsource(PortfolioAccounting._now)


def test_wp15_t18_paper_source_branch_never_uses_the_legacy_ledger_helpers() -> None:
    fn = _function(PaperExecutionEngine, "_risk_gate_inputs")
    source_branch: ast.If | None = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            comp = node.test
            if (
                isinstance(comp.left, ast.Name)
                and comp.left.id == "source"
                and isinstance(comp.ops[0], ast.IsNot)
            ):
                source_branch = node
                break
    assert source_branch is not None, "expected an `if source is not None:` branch"

    legacy = {"_get_peak_equity", "_get_daily_pnl", "_get_current_equity"}
    in_branch = _called_names(ast.Module(body=source_branch.body, type_ignores=[]))
    assert not (in_branch & legacy), in_branch & legacy
    assert "risk_snapshot" in in_branch and "gate_daily_pnl" in in_branch
    # ... while the fall-back path outside the branch does use them (sanity)
    assert legacy <= _called_names(fn)


def test_wp15_t18_process_signal_takes_one_risk_read_through_the_helper() -> None:
    fn = _function(PaperExecutionEngine, "process_signal")
    calls = _called_names(fn)
    assert "_risk_gate_inputs" in calls
    assert not (calls & {"_get_peak_equity", "_get_daily_pnl", "_get_current_equity"})


def _orchestrator_function(name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(_ORCHESTRATOR.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_wp15_t18_r02_live_engine_uses_live_parameters_and_paper_does_not() -> None:
    live = _orchestrator_function("run_live_engine")
    paper = _orchestrator_function("run_paper_engine")
    assert "live_risk_parameters" in _called_names(live)
    assert "live_risk_parameters" not in _called_names(paper)

    def _risk_manager_ctor(fn: ast.AsyncFunctionDef) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "DefaultRiskManager"
        ]

    live_ctors = _risk_manager_ctor(live)
    assert len(live_ctors) == 1, "the single live risk-manager construction site"
    assert [kw.arg for kw in live_ctors[0].keywords] == ["run_id", "params"]
    paper_ctors = _risk_manager_ctor(paper)
    assert len(paper_ctors) == 1
    assert [kw.arg for kw in paper_ctors[0].keywords] == ["run_id"]  # default parameters
