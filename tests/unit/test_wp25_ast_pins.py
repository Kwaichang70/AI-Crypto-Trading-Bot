"""WP2.5 / P0: AST pin of the sampler leaf ``packages/trading/equity_sampling.py``.

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-02, WP25-T-05). Same style as
``tests/unit/test_wp23_alignment.py`` (exact import whitelist, banned names, mutated-source
self-test). The runner part of the pins (WP25-T-23) is added in P2.

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
