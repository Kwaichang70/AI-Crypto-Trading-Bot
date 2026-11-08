"""WP2.4 / P0: AST pin of the point-in-time leaf ``packages/data/point_in_time.py``.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-01, SY-24-03, WP24-T-07 AST part,
WP24-T-11). Same style as ``tests/unit/test_wp25_ast_pins.py`` (exact import whitelist, banned
names, mutated-source self-test). The engine part of the pins (WP24-T-17, P1) is added to this
module by a later commit.

The ingestion helpers (``available_at_estimated`` / ``available_at_vintage``) have exactly one
in-leaf caller: the JSONL lag floor in ``parse_jsonl_lines`` (QR-09). ``asof`` and
``coverage_ppm`` stay fully pinned: no lag parameter, no reference to a helper or lag constant.

The leaf is float-free (no float literal, no ``float(...)`` call, no ``/``, no ``**``) because it
hashes and compares Decimal values and integer milliseconds only.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LEAF = _REPO / "packages" / "data" / "point_in_time.py"

# Exact import whitelist of the leaf: any new import forces an edit here.
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "bisect",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "json",
        "types",
        "typing",
    }
)
_BANNED_NAMES = frozenset(
    {
        "environ", "getenv", "putenv", "now", "utcnow", "today", "time", "monotonic",
        "perf_counter", "open", "print", "input", "metrics", "increment", "getLogger",
        "logging", "structlog", "random", "uuid4", "__import__", "import_module", "eval",
        "exec", "read_text", "write_text", "urlopen", "os", "sleep", "math", "float",
        "logger", "warnings",
    }
)  # fmt: skip
_FORBIDDEN_PREFIXES = ("trading", "common", "data", "apps")
# WP24-T-07 / SY-24-03: the join never sees a lag and never calls the ingestion helper.
_JOIN_METHODS = ("asof", "coverage_ppm")
_COVERAGE_TAIL = '*, back_ms: int, max_age_ms: int) -> int:\n        """Parts per million'
_INGESTION_ONLY = frozenset({"available_at_estimated", "available_at_vintage"})
# QR-09 (E9): the ONLY in-leaf caller of the ingestion helpers is the JSONL lag-floor check.
_INGESTION_EXEMPT = "parse_jsonl_lines"
_EXPECTED_INGESTION_SITES = [
    "parse_jsonl_lines -> available_at_estimated",
    "parse_jsonl_lines -> available_at_vintage",
]
# CR-07: exact join signatures and no lag constant in the join.
_JOIN_SIGNATURES = {
    "asof": ["self", "decision_ms", "max_age_ms"],
    "coverage_ppm": ["self", "decision_ms", "back_ms", "max_age_ms"],
}
_LAG_CONSTANTS = frozenset({"_FGI_LAG_MS", "_FRED_BUSINESS_DAYS", "_FRED_SLACK_MS"})


def leaf_problems(src: str) -> list[str]:
    """Purity checker for the leaf source; returns ``"<category>: <detail>"`` problems."""
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"import: {a.name}" for a in node.names if a.name not in _ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or (node.module or "") not in _ALLOWED_IMPORTS:
                out.append(f"import: {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Constant) and type(node.value) is float:
            out.append("float: literal")
        elif isinstance(node, ast.Div):
            out.append("float: operator /")
        elif isinstance(node, ast.Pow):
            out.append("float: operator **")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            out.append("state: global/nonlocal")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    out += [f"banned: {name}" for name in sorted(used & _BANNED_NAMES)]
    return out


def _join_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in _JOIN_METHODS]


def lag_problems(src: str) -> list[str]:
    """SY-24-03: no lag parameter on the join, and no ingestion helper reachable from it."""
    tree = ast.parse(src)
    out: list[str] = []
    joins = _join_functions(tree)
    found = sorted(f.name for f in joins)
    if found != sorted(_JOIN_METHODS):
        out.append(f"missing: join methods {found}")
    for fn in joins:
        args = fn.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        if args.vararg:
            names.append(args.vararg.arg)
        if args.kwarg:
            names.append(args.kwarg.arg)
        out += [f"lag: {fn.name} parameter {n}" for n in names if "lag" in n.lower()]
        if args.kwarg or args.vararg:
            out.append(f"lag: {fn.name} takes *args/**kwargs")
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
        }
        out += [f"lag: {fn.name} references {n}" for n in sorted(used & _INGESTION_ONLY)]
        if names != _JOIN_SIGNATURES.get(fn.name):
            out.append(f"lag: {fn.name} signature {names}")
        out += [f"lag: {fn.name} references {n}" for n in sorted(used & _LAG_CONSTANTS)]
    return out


def ingestion_call_sites(src: str) -> list[str]:
    """Functions of the leaf that call an ingestion-only helper (there must be none)."""
    tree = ast.parse(src)
    sites: list[str] = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)):
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
            name = getattr(call.func, "id", getattr(call.func, "attr", ""))
            if name in _INGESTION_ONLY:
                sites.append(f"{fn.name} -> {name}")
    return sorted(sites)


def ingestion_references(src: str) -> list[str]:
    """Every scope (function name or ``<module>``) that loads an ingestion-only name.

    Catches aliases (``_est = available_at_estimated``) that a call-site scan cannot see.
    """
    tree = ast.parse(src)
    owner: dict[ast.AST, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            for sub in ast.walk(fn):
                if sub is not fn:
                    owner.setdefault(sub, fn.name)  # ast.walk is breadth-first: outermost wins
    scopes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Load):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in _INGESTION_ONLY:
                scopes.add(owner.get(node, "<module>"))
    return sorted(scopes)


class TestLeafAstPin:
    def test_imports_are_an_exact_whitelist(self) -> None:  # WP24-T-11
        src = _LEAF.read_text()
        assert not [p for p in leaf_problems(src) if p.startswith("import:")]
        tree = ast.parse(src)
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert imported, "leaf imports nothing?"
        assert imported <= _ALLOWED_IMPORTS
        assert {
            "bisect",
            "dataclasses",
            "datetime",
            "decimal",
            "hashlib",
            "json",
            "typing",
        } <= imported
        assert all(m.split(".")[0] in sys.stdlib_module_names for m in _ALLOWED_IMPORTS)
        assert not [m for m in imported if m and m.split(".")[0] in _FORBIDDEN_PREFIXES]

    def test_no_env_clock_logging_io_random_or_metrics(self) -> None:
        assert not [p for p in leaf_problems(_LEAF.read_text()) if p.startswith("banned:")]

    def test_float_free_and_no_global_state(self) -> None:
        problems = leaf_problems(_LEAF.read_text())
        assert not [p for p in problems if p.startswith(("float:", "state:"))]

    def test_no_wall_clock_or_io_calls(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        calls = {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        assert not calls & {"now", "utcnow", "today", "time", "getLogger", "open", "print"}

    def test_leaf_is_clean(self) -> None:
        assert leaf_problems(_LEAF.read_text()) == []

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
        assert set(exported) <= defined
        # the spec's binding API (SY-24-01) is exported
        assert {
            "PIT_VERSION", "PitError", "Basis", "Observation", "PitSeries", "to_ms",
            "available_at_estimated", "series_digest", "parse_jsonl_lines", "SERIES_RULES",
            "FIELD_MAP", "BACK_7D_MS", "MIN_COVERAGE_PPM",
        } <= set(exported)  # fmt: skip

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
            ("import decimal as d, sqlite3", "import: sqlite3"),
            ("from trading import metrics", "import: trading"),
            ("from common import types", "import: common"),
            ("from data import bar_alignment", "import: data"),
            ("from data.bar_alignment import iso_from_ms", "import: data.bar_alignment"),
            ("from apps.api import main", "import: apps.api"),
            ("from . import sibling", "import: ."),
            ("_x = datetime.now()", "banned: now"),
            ("_x = datetime.utcnow()", "banned: utcnow"),
            ("_t = time.time()", "banned: time"),
            ("_y = open('f')", "banned: open"),
            ("_p = print", "banned: print"),
            ("_e = os.environ", "banned: environ"),
            ("_r = random.random()", "banned: random"),
            ("_l = logging.getLogger('x')", "banned: getLogger"),
            ("_z = 0.5", "float: literal"),
            ("_w = 2 ** 3", "float: operator **"),
            ("_d = 6 / 3", "float: operator /"),
            ("_v = float(3)", "banned: float"),
            ("def _g() -> None:\n    global _G\n    _G = 1", "state: global/nonlocal"),
        ],
    )
    def test_pin_fails_on_mutated_source(self, addition: str, expected: str) -> None:
        src = _LEAF.read_text()
        assert leaf_problems(src) == []  # the real leaf is clean ...
        problems = leaf_problems(src + f"\n{addition}\n")  # ... and the pin can fail
        assert expected in problems, problems


class TestNoLagInTheJoin:
    """WP24-T-07 (AST part): the lag is applied at ingestion only (SY-24-03)."""

    def test_asof_and_coverage_take_no_lag_and_never_reach_the_ingestion_helpers(self) -> None:
        src = _LEAF.read_text()
        assert lag_problems(src) == []
        assert ingestion_call_sites(src) == _EXPECTED_INGESTION_SITES
        assert ingestion_references(src) == [_INGESTION_EXEMPT]

    def test_the_ingestion_helpers_exist_and_have_one_ingestion_caller(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert _INGESTION_ONLY <= defs

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            (
                "def asof(self, decision_ms: int, *, max_age_ms: int)",
                "def asof(self, decision_ms: int, *, max_age_ms: int, lag_ms: int = 0)",
                "lag: asof parameter lag_ms",
            ),
            (
                _COVERAGE_TAIL,
                _COVERAGE_TAIL.replace("max_age_ms: int)", "max_age_ms: int, lag_ms: int = 0)"),
                "lag: coverage_ppm parameter lag_ms",
            ),
            (
                "def asof(self, decision_ms: int, *, max_age_ms: int)",
                "def asof(self, decision_ms: int, *, max_age_ms: int, **extra: int)",
                "lag: asof takes *args/**kwargs",
            ),
            (
                "        total = len(decision_ms)\n",
                "        total = len(decision_ms)\n        available_at_estimated('x', 0)\n",
                "lag: coverage_ppm references available_at_estimated",
            ),
            (
                "        visible = bisect_right(self._avail, decision_ms)\n",
                "        visible = bisect_right(self._avail, available_at_vintage(decision_ms))\n",
                "lag: asof references available_at_vintage",
            ),
            (
                "def asof(self, decision_ms: int, *, max_age_ms: int)",
                "def asof(self, decision_ms: int, *, max_age_ms: int, offset_ms: int = 0)",
                "lag: asof signature ['self', 'decision_ms', 'max_age_ms', 'offset_ms']",
            ),
            (
                "        visible = bisect_right(self._avail, decision_ms)\n",
                "        visible = bisect_right(self._avail, decision_ms - _FGI_LAG_MS)\n",
                "lag: asof references _FGI_LAG_MS",
            ),
        ],
    )
    def test_lag_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _LEAF.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert lag_problems(src) == []
        assert expected in lag_problems(src.replace(old, new)), lag_problems(src.replace(old, new))

    def test_ingestion_call_pin_fails_on_a_new_call_site(self) -> None:
        src = _LEAF.read_text()
        assert ingestion_call_sites(src) == _EXPECTED_INGESTION_SITES
        mutated = (
            src + "\n\ndef _sneaky(x: int) -> int:\n    return available_at_estimated('x', x)\n"
        )
        assert ingestion_call_sites(mutated) == sorted(
            [*_EXPECTED_INGESTION_SITES, "_sneaky -> available_at_estimated"]
        )
        method = (
            src
            + "\n\nclass _K:\n    def m(self) -> int:\n        return M.available_at_vintage(0)\n"
        )
        assert ingestion_call_sites(method) == sorted(
            [*_EXPECTED_INGESTION_SITES, "m -> available_at_vintage"]
        )

    @pytest.mark.parametrize(
        ("addition", "scope"),
        [
            ("_est = available_at_estimated", "<module>"),
            ("def _h() -> object:\n    return available_at_vintage", "_h"),
        ],
    )
    def test_alias_of_an_ingestion_helper_is_caught(self, addition: str, scope: str) -> None:
        src = _LEAF.read_text()
        assert ingestion_references(src) == [_INGESTION_EXEMPT]
        assert ingestion_references(src + f"\n{addition}\n") == sorted([_INGESTION_EXEMPT, scope])

    def test_lag_pin_notices_a_missing_join_method(self) -> None:
        src = _LEAF.read_text().replace("def coverage_ppm(", "def coverage_count(")
        assert any(p.startswith("missing:") for p in lag_problems(src))
