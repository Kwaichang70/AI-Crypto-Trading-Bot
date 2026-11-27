"""WP2.5 / P0: AST pin of the sampler leaf ``packages/trading/equity_sampling.py``.

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-02, WP25-T-05). Same style as
``tests/unit/test_wp23_alignment.py`` (exact import whitelist, banned names, mutated-source
self-test). The runner part of the pins (WP25-T-23, P2) is at the end of this module.

The leaf calls ``float(...)`` in exactly one function (``simple_returns``, the HEAD return
formula); everywhere else a float literal, a ``float(...)`` call or ``**`` is a problem
(``float`` in annotations is allowed).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LEAF = _REPO / "packages" / "trading" / "equity_sampling.py"

# Exact import whitelist of the leaf: any new import forces an edit here.
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "bisect",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "typing",
    }
)
_BANNED_NAMES = frozenset(
    {
        "environ", "getenv", "putenv", "now", "utcnow", "today", "time", "monotonic",
        "perf_counter", "open", "print", "input", "metrics", "increment", "getLogger",
        "logging", "structlog", "random", "uuid4", "__import__", "import_module", "eval",
        "exec", "read_text", "write_text", "urlopen", "os", "sleep", "math",
    }
)  # fmt: skip
_FORBIDDEN_PREFIXES = ("trading", "common", "data", "apps")
_FLOAT_FUNCTION = "simple_returns"  # the only function allowed to name ``float``


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str | None]:
    owner: dict[ast.AST, str | None] = {}

    def visit(node: ast.AST, current: str | None) -> None:
        owner[node] = current
        nxt = node.name if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) else current
        for child in ast.iter_child_nodes(node):
            visit(child, nxt)

    visit(tree, None)
    return owner


def leaf_problems(src: str) -> list[str]:
    """Purity checker for the leaf source; returns ``"<category>: <detail>"`` problems."""
    tree = ast.parse(src)
    owner = _enclosing_functions(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"import: {a.name}" for a in node.names if a.name not in _ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or (node.module or "") not in _ALLOWED_IMPORTS:
                out.append(f"import: {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Constant) and type(node.value) is float:
            out.append("float: literal")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and owner.get(node) != _FLOAT_FUNCTION
        ):
            out.append("float: name")  # annotations (``-> list[float]``) are fine
        elif isinstance(node, ast.Pow):
            out.append("float: operator")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    out += [f"banned: {name}" for name in sorted(used & _BANNED_NAMES)]
    return out


class TestLeafAstPin:
    def test_imports_are_an_exact_whitelist(self) -> None:  # WP25-T-05
        src = _LEAF.read_text()
        assert not [p for p in leaf_problems(src) if p.startswith("import:")]
        tree = ast.parse(src)
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert imported, "leaf imports nothing?"
        assert imported <= _ALLOWED_IMPORTS
        assert all(m.split(".")[0] in sys.stdlib_module_names for m in _ALLOWED_IMPORTS)
        assert not [m for m in imported if m and m.split(".")[0] in _FORBIDDEN_PREFIXES]

    def test_no_env_clock_logging_io_or_metrics(self) -> None:
        assert not [p for p in leaf_problems(_LEAF.read_text()) if p.startswith("banned:")]

    def test_float_only_in_the_return_formula(self) -> None:
        src = _LEAF.read_text()
        assert not [p for p in leaf_problems(src) if p.startswith("float:")]
        calls = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"
        ]
        assert calls, "simple_returns must convert to float"

    def test_no_wall_clock_or_global_state_calls(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        calls = {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        assert not calls & {"now", "utcnow", "today", "time", "getLogger", "open", "print"}
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global | ast.Nonlocal)]

    def test_all_exports_exist(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)} | {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign | ast.AnnAssign)
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            if isinstance(t, ast.Name)
        }
        exported: list[str] = []
        for n in tree.body:
            if (
                isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "__all__"
            ):
                assert isinstance(n.value, ast.List)
                exported = [ast.literal_eval(e) for e in n.value.elts]
        assert exported
        assert set(exported) <= defined | {"SamplingMode"}

    @pytest.mark.parametrize(
        ("addition", "expected"),
        [
            ("import pathlib", "import: pathlib"),
            ("from random import shuffle", "import: random"),
            ("import urllib.request", "import: urllib.request"),
            ("from os import environ", "import: os"),
            ("import logging", "import: logging"),
            ("import structlog", "import: structlog"),
            ("from time import time", "import: time"),
            ("import math", "import: math"),
            ("from trading import metrics", "import: trading"),
            ("from common import types", "import: common"),
            ("from data import bar_alignment", "import: data"),
            ("from . import sibling", "import: ."),
            ("_x = datetime.now()", "banned: now"),
            ("_y = open('f')", "banned: open"),
            ("_p = print", "banned: print"),
            ("_z = 0.5", "float: literal"),
            ("_w = 2 ** 3", "float: operator"),
            ("_v = float(3)", "float: name"),
        ],
    )
    def test_pin_fails_on_mutated_source(self, addition: str, expected: str) -> None:
        src = _LEAF.read_text()
        assert leaf_problems(src) == []  # the real leaf is clean ...
        problems = leaf_problems(src + f"\n{addition}\n")  # ... and the pin can fail
        assert expected in problems, problems

    def test_float_inside_another_function_is_flagged(self) -> None:
        src = _LEAF.read_text() + "\n\ndef other() -> float:\n    return float(1)\n"
        assert "float: name" in leaf_problems(src)


# ---------------------------------------------------------------------------
# WP25-T-23 (P2): runner / call-site pins
# ---------------------------------------------------------------------------

_RUNNER = _REPO / "packages" / "trading" / "backtest.py"
_SCAN_ROOTS = ("packages", "apps", "scripts")
_SKIP_FILES = frozenset(
    {
        "packages/trading/equity_sampling.py",  # the leaf (defines simple_returns)
        "packages/trading/metrics.py",  # the formula owner
        "packages/trading/__init__.py",  # re-export only
    }
)
# (file, enclosing function) pairs that may call ``compute_returns_from_equity``:
# the CF-26 freeze helper (SY-25-17) and the backfill LEGACY path (SY-25-18, exempted by name).
_ALL_POINTS_ALLOW_LIST = frozenset(
    {
        ("scripts/wp21_rebaseline.py", "_pre_wp25_sharpe"),
        ("scripts/backfill_metrics_v2.py", "_recompute_psr_from_snapshots"),
    }
)
_RUNNER_FORBIDDEN_NAMES = frozenset(
    {
        "compute_returns_from_equity",
        "compute_sharpe",
        "compute_sortino",
        "compute_psr",
        "return_moments",
        "simple_returns",
        "compute_max_drawdown_duration",
    }
)


def _aliases(tree: ast.AST) -> dict[str, str]:
    """``{local alias: imported name}`` for every aliased ``from x import y as z`` (CR-02)."""
    return {
        a.asname: a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if a.asname
    }


def _call_name(node: ast.Call, aliases: dict[str, str] | None = None) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    name = str(getattr(func, "id", ""))
    return (aliases or {}).get(name, name)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def all_points_calls(src: str) -> list[str]:
    """Names of the functions that call ``compute_returns_from_equity`` (``<module>`` at top)."""
    tree = ast.parse(src)
    owner = _enclosing_functions(tree)
    aliases = _aliases(tree)
    return sorted(
        owner[n] or "<module>"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _call_name(n, aliases) == "compute_returns_from_equity"
    )


def sampler_mode_kwargs(src: str) -> list[str]:
    """``mode=`` keyword arguments passed to ``sample_equity_per_bar`` (QR-03)."""
    tree = ast.parse(src)
    aliases = _aliases(tree)
    return [
        "mode"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _call_name(n, aliases) == "sample_equity_per_bar"
        for kw in n.keywords
        if kw.arg == "mode" or kw.arg is None  # ``**kwargs`` could smuggle it in
    ]


def runner_problems(src: str) -> list[str]:
    """Metric-routing checks on the runner source; ``"<category>: <detail>"`` problems."""
    tree = ast.parse(src)
    parents = _parents(tree)
    out: list[str] = []
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    aliases = _aliases(tree)
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    out += [f"forbidden: {name}" for name in sorted((used | imported) & _RUNNER_FORBIDDEN_NAMES)]
    # no inline moments: ``x ** 3`` / ``x ** 4`` (the HEAD skew/kurtosis block)
    out += [
        "inline moment: **"
        for n in ast.walk(tree)
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.Pow)
        and isinstance(n.right, ast.Constant)
        and n.right.value in (3, 4)
    ]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if sum(_call_name(c, aliases) == "compute_return_metrics" for c in calls) < 2:
        out.append("missing: compute_return_metrics (full run and OOS window)")
    if not any(_call_name(c, aliases) == "sample_equity_per_bar" for c in calls):
        out.append("missing: sample_equity_per_bar")
    # ``oos_offset`` (the engine's POINT offset) may only be tested ``is None`` or logged
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and n.id == "oos_offset" and isinstance(n.ctx, ast.Load)):
            continue
        parent = parents[n]
        is_none_check = (
            isinstance(parent, ast.Compare)
            and parent.left is n
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], ast.Is)
            and isinstance(parent.comparators[0], ast.Constant)
            and parent.comparators[0].value is None
        )
        grand = parents.get(parent)
        in_info_log = (
            isinstance(parent, ast.Call)
            and _call_name(parent, aliases) == "int"
            and isinstance(grand, ast.keyword)
            and grand.arg == "oos_equity_curve_offset"
            and isinstance(parents.get(grand), ast.Call)
            and _call_name(parents[grand], aliases) == "info"  # type: ignore[arg-type]
        )
        if not (is_none_check or in_info_log):
            out.append(
                f"oos_offset: used outside the is-None check and the info log (line {n.lineno})"
            )
    # a direct ``engine_summary["oos_equity_curve_offset"]`` read bypasses the name check
    out += [
        f"oos_offset: subscript read (line {n.lineno})"
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.Constant)
        and n.slice.value == "oos_equity_curve_offset"
    ]
    return out


def _production_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            rel = path.relative_to(_REPO).as_posix()
            if rel in _SKIP_FILES or "/tests/" in f"/{rel}" or "/.venv/" in rel:
                continue
            files.append(path)
    return files


class TestRunnerAstPins:
    def test_all_points_returns_only_at_the_allow_list(self) -> None:  # WP25-T-23
        found: set[tuple[str, str]] = set()
        for path in _production_files():
            rel = path.relative_to(_REPO).as_posix()
            found |= {(rel, fn) for fn in all_points_calls(path.read_text())}
        assert found == _ALL_POINTS_ALLOW_LIST

    @pytest.mark.parametrize(
        "name", ["backtest.py", "optimizer.py", "walk_forward.py"]
    )
    def test_no_all_points_call_in_the_runner_optimizer_or_walk_forward(self, name: str) -> None:
        src = (_REPO / "packages" / "trading" / name).read_text()
        assert all_points_calls(src) == []

    def test_no_apps_module_calls_the_all_points_formula(self) -> None:
        for path in (_REPO / "apps").rglob("*.py"):
            if "/tests/" in path.as_posix():
                continue
            assert all_points_calls(path.read_text()) == [], path

    def test_no_production_call_site_passes_mode_to_the_sampler(self) -> None:  # QR-03
        offenders = [
            path.relative_to(_REPO).as_posix()
            for path in _production_files()
            if sampler_mode_kwargs(path.read_text())
        ]
        assert offenders == []
        for name in ("backtest.py", "optimizer.py", "walk_forward.py"):
            src = (_REPO / "packages" / "trading" / name).read_text()
            assert sampler_mode_kwargs(src) == [], name
        # the runner really calls the sampler (the scan above is not vacuous)
        assert "sample_equity_per_bar(" in _RUNNER.read_text()

    def test_runner_routes_every_return_metric_through_one_call(self) -> None:
        assert runner_problems(_RUNNER.read_text()) == []

    @pytest.mark.parametrize(
        ("addition", "expected"),
        [
            (
                "def _f(c):\n    return compute_returns_from_equity(c)",
                "forbidden: compute_returns_from_equity",
            ),
            ("def _f(r):\n    return compute_sharpe(r, 1.0)", "forbidden: compute_sharpe"),
            ("def _f(r):\n    return compute_psr(1.0, 5)", "forbidden: compute_psr"),
            ("def _f(r):\n    return sum(r) ** 3", "inline moment: **"),
            ("def _f(r):\n    return sum(r) ** 4", "inline moment: **"),
            ("def _f(oos_offset):\n    return oos_offset + 1", "oos_offset: used outside"),
            ("def _f(oos_offset):\n    return oos_offset > 3", "oos_offset: used outside"),
            (  # CR-02: aliased imports must not evade the pins
                "from trading.metrics import compute_sharpe as _cs\n\n"
                "def _f(r):\n    return _cs(r, 1.0)",
                "forbidden: compute_sharpe",
            ),
            (
                "from trading.metrics import compute_returns_from_equity as _cr\n\n"
                "def _f(c):\n    return _cr(c)",
                "forbidden: compute_returns_from_equity",
            ),
            (
                "def _f(s):\n    return s['oos_equity_curve_offset']",
                "oos_offset: subscript read",
            ),
        ],
    )
    def test_runner_pin_fails_on_mutated_source(self, addition: str, expected: str) -> None:
        src = _RUNNER.read_text()
        assert runner_problems(src) == []
        problems = runner_problems(src + f"\n\n{addition}\n")
        assert any(p.startswith(expected) for p in problems), problems

    def test_mode_kwarg_pin_fails_on_mutated_source(self) -> None:
        src = _RUNNER.read_text()
        assert sampler_mode_kwargs(src) == []
        assert sampler_mode_kwargs(src + "\n_x = sample_equity_per_bar([], mode='present_steps')\n")
        assert sampler_mode_kwargs(src + "\n_x = sample_equity_per_bar([], **{})\n")
        aliased = (
            "\nfrom trading.equity_sampling import sample_equity_per_bar as _s\n"
            "_x = _s([], mode='present_steps')\n"
        )
        assert sampler_mode_kwargs(src + aliased)  # CR-02

    def test_all_points_pin_fails_on_a_new_call_site(self) -> None:
        assert all_points_calls("def g(c):\n    return compute_returns_from_equity(c)\n") == ["g"]
        assert all_points_calls("x = compute_returns_from_equity([])\n") == ["<module>"]
        aliased = (
            "from m import compute_returns_from_equity as _cr\n\ndef g(c):\n    return _cr(c)\n"
        )
        assert all_points_calls(aliased) == ["g"]  # CR-02
