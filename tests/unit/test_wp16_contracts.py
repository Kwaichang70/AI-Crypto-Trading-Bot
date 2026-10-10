"""
tests/unit/test_wp16_contracts.py
-----------------------------------
WP1.6 -- AST / contract pins.

    WP16-T-25  structural pins: no get_summary / legacy CB API / wipe / stop in the
               bar path; nothing sets ``_auto_stop_reason``; no wall clock in
               ``evaluate``; the SELL sizing branch ignores the multiplier; no
               production caller of the legacy CB API; one construction helper
    WP16-T-26  ``PortfolioSummary`` keys == the runtime keys, and every literal key
               read from a ``get_summary()`` result in ``strategy_engine.py`` is declared
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from decimal import Decimal
from pathlib import Path

import pytest

from tests.unit._wp15_support import Clock, D, portfolio_with_position
from trading.execution import BaseExecutionEngine
from trading.portfolio import PortfolioAccounting, PortfolioSummary
from trading.safety import (
    CircuitBreaker,
    CircuitBreakerInputs,
    CircuitBreakerThresholds,
    _require_finite_decimal,
)
from trading.strategy_engine import StrategyEngine

_REPO = Path(__file__).resolve().parents[2]
_PRODUCTION_ROOTS = (_REPO / "packages", _REPO / "apps")
_WALL_CLOCK = {"now", "utcnow", "today", "time", "monotonic", "time_ns"}


def _production_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        files.extend(
            p for p in root.rglob("*.py") if "tests" not in p.parts and "alembic" not in p.parts
        )
    return sorted(files)


def _fn(owner: object, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(owner, name))))
    fn = tree.body[0]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    return fn


def _calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _receiver_text(call: ast.Call) -> str:
    f = call.func
    return ast.unparse(f.value) if isinstance(f, ast.Attribute) else ""


# ---------------------------------------------------------------------------
# T-25
# ---------------------------------------------------------------------------


def test_wp16_t25_process_bar_has_no_legacy_breaker_path_wipe_or_stop() -> None:
    fn = _fn(StrategyEngine, "_process_bar")
    names = {_call_name(c) for c in _calls(fn)}
    # Spec deviation D-2: the display-only metrics gauges (step 8, `try:` block that
    # calls `_mc.gauge`) still read `get_summary()` -- the spec's own site list keeps
    # that call (:1429).  The pin is that NO decision path (everything outside that
    # block) reads it, and that the block uses only declared keys (T-26).
    metrics_tries = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Try) and any(_call_name(c) == "gauge" for c in _calls(n))
    ]
    assert len(metrics_tries) == 1
    inside = {id(c) for c in _calls(metrics_tries[0])}
    outside_summary = [
        c for c in _calls(fn) if _call_name(c) == "get_summary" and id(c) not in inside
    ]
    assert outside_summary == []
    assert "check_graduated" not in names
    assert "get_position_size_multiplier" not in names
    # `.check(` on the circuit breaker (bracket / trailing managers have their own,
    # unrelated `.check`)
    for c in _calls(fn):
        if _call_name(c) == "check":
            assert "circuit_breaker" not in _receiver_text(c)
    # the "wipe": `bar_signals = []`
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "bar_signals":
                    assert not (isinstance(node.value, ast.List) and not node.value.elts)
    # the auto-stop: `_stop_event.set()`
    for c in _calls(fn):
        assert not (_call_name(c) == "set" and "_stop_event" in _receiver_text(c))
    assert "evaluate" in names and "risk_snapshot" in names


def test_wp16_t25_nothing_in_production_assigns_auto_stop_reason_except_none_in_init() -> None:
    offenders: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            elif isinstance(node, ast.AugAssign):
                targets, value = [node.target], node.value
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr == "_auto_stop_reason":
                    is_none = isinstance(value, ast.Constant) and value.value is None
                    if not (is_none and path.name == "strategy_engine.py"):
                        offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert offenders == []


_EVALUATE_PATH = [
    CircuitBreaker.evaluate,
    CircuitBreaker._emit_events,
    CircuitBreaker._active_conditions,
    CircuitBreaker._multiplier_for,
    _require_finite_decimal,
    CircuitBreakerThresholds.__post_init__,
    CircuitBreakerThresholds.from_risk_parameters,
    CircuitBreakerThresholds.as_str_dict,
    CircuitBreakerInputs.as_str_dict,
]


@pytest.mark.parametrize("fn", _EVALUATE_PATH, ids=lambda f: f.__qualname__)
def test_wp16_t25_no_wall_clock_read_on_the_evaluation_path(fn: object) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))  # type: ignore[arg-type]
    offenders = {_call_name(c) for c in _calls(tree)} & _WALL_CLOCK
    assert not offenders, offenders
    src = inspect.getsource(fn)  # type: ignore[arg-type]
    assert "datetime.now" not in src and "date.today" not in src and "time.time" not in src


def test_wp16_t25_sell_branch_of_the_resolver_never_references_the_multiplier() -> None:
    fn = _fn(BaseExecutionEngine, "_resolve_order_quantity")
    buy_if = next(
        n for n in fn.body
        if isinstance(n, ast.If) and "OrderSide.BUY" in ast.unparse(n.test)
    )
    buy_src = ast.unparse(buy_if)
    assert "entry_size_multiplier" in buy_src
    idx = fn.body.index(buy_if)
    sell_part = ast.Module(body=fn.body[idx + 1 :], type_ignores=[])
    assert "entry_size_multiplier" not in ast.unparse(sell_part)


def test_wp16_t25_no_production_caller_of_the_legacy_breaker_api() -> None:
    offenders: list[str] = []
    for path in _production_files():
        if path.name == "safety.py" and path.parent.name == "trading":
            continue
        tree = ast.parse(path.read_text())
        for c in _calls(tree):
            name, recv = _call_name(c), _receiver_text(c)
            legacy = (
                name in {"check_graduated", "get_position_size_multiplier", "trip",
                         "check_daily_limit_reset", "_trip_internal"}
                or (name == "reset" and ("breaker" in recv or recv.endswith("cb")))
                or (name == "check" and ("breaker" in recv or recv.endswith("cb")))
            )
            if legacy:
                offenders.append(f"{path.relative_to(_REPO)}:{c.lineno} {recv}.{name}")
    assert offenders == []


def test_wp16_t25_circuit_breakers_are_built_only_through_the_helper() -> None:
    constructors = {"CircuitBreaker", "CircuitBreakerConfig"}
    offenders: list[str] = []
    paths = [p for p in _production_files() if "apps" in p.parts] + [
        _REPO / "packages" / "trading" / "backtest.py"
    ]
    for path in paths:
        for c in _calls(ast.parse(path.read_text())):
            if _call_name(c) in constructors:
                offenders.append(f"{path.relative_to(_REPO)}:{c.lineno}")
    assert offenders == []
    orch = (_REPO / "apps/api/services/run_orchestrator.py").read_text()
    assert orch.count("build_run_circuit_breaker(run_id_str, risk_manager.params)") == 2
    assert "build_run_circuit_breaker(run_id, risk_manager.params)" in (
        _REPO / "packages/trading/backtest.py"
    ).read_text()


# ---------------------------------------------------------------------------
# T-26
# ---------------------------------------------------------------------------

_SUMMARY_TYPES = {"str": str, "float": float, "int": int}


def test_wp16_t26_typed_dict_keys_equal_runtime_keys_and_value_types_match() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    summary = p.get_summary()
    assert set(summary) == set(PortfolioSummary.__required_keys__)
    assert PortfolioSummary.__optional_keys__ == frozenset()
    hints = typing.get_type_hints(PortfolioSummary)
    for key, value in summary.items():
        expected = _SUMMARY_TYPES[hints[key].__name__]
        assert isinstance(value, expected), (key, type(value), hints[key])


def test_wp16_t26_get_summary_is_annotated_with_the_typed_dict() -> None:
    assert typing.get_type_hints(PortfolioAccounting.get_summary)["return"] is PortfolioSummary


def unknown_summary_keys(source: str) -> set[str]:
    """Literal keys READ from a name bound directly from ``.get_summary()`` and not declared."""
    declared = set(PortfolioSummary.__required_keys__)
    tree = ast.parse(source)
    bound: set[str] = set()
    for node in ast.walk(tree):
        value: ast.expr | None = None
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get_summary"
        ):
            bound.add(target.id)
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in bound
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            found.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in bound
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
    return found - declared


def test_wp16_t26_every_literal_summary_key_in_the_engine_is_declared() -> None:
    src = (_REPO / "packages/trading/strategy_engine.py").read_text()
    assert unknown_summary_keys(src) == set()
    # the reads that exist must really be seen (the checker is not vacuous)
    assert "_summary[\"current_equity\"]" in src


def test_wp16_t26_mutation_check_the_head_style_typo_is_detected() -> None:
    mutated = textwrap.dedent(
        """
        def f(self):
            equity_summary = self._portfolio.get_summary()
            a = equity_summary.get("realised_pnl", 0.0)
            b = equity_summary["max_drawdown"]
            c = equity_summary["current_equity"]
        """
    )
    assert unknown_summary_keys(mutated) == {"realised_pnl", "max_drawdown"}


def test_wp16_t26_nothing_else_in_production_reads_undeclared_summary_keys() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _production_files():
        text = path.read_text()
        if "get_summary()" not in text:
            continue
        bad = unknown_summary_keys(text)
        if bad:
            offenders[str(path.relative_to(_REPO))] = bad
    assert offenders == {}
    assert Decimal(1)  # keep the Decimal import meaningful for future key additions
