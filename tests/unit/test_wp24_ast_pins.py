"""WP2.4 / P0: AST pin of the point-in-time leaf ``packages/data/point_in_time.py``.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-01, SY-24-03, WP24-T-07 AST part,
WP24-T-11). Same style as ``tests/unit/test_wp25_ast_pins.py`` (exact import whitelist, banned
names, mutated-source self-test). The engine part of the pins (WP24-T-17, P1) is the second half
of this module (``TestEngineAstPin`` and the helpers above it).

The ingestion helpers (``available_at_estimated`` / ``available_at_vintage``) have exactly one
in-leaf caller: the JSONL lag floor in ``parse_jsonl_lines`` (QR-09). ``asof`` and
``coverage_ppm`` stay fully pinned: no lag parameter, no reference to a helper or lag constant.

The leaf is float-free (no float literal, no ``float(...)`` call, no ``/``, no ``**``) because it
hashes and compares Decimal values and integer milliseconds only.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

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
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in _INGESTION_ONLY:
            scopes.add(owner.get(node, "<module>"))  # ``from m import helper as _h``
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


# ===========================================================================
# WP24-T-17 (P1): the engine part -- C10 stays closed by construction
# ===========================================================================

_ENGINE = _REPO / "packages" / "trading" / "strategy_engine.py"
_SIGNALS = _REPO / "packages" / "trading" / "mtf_signals.py"
_PROVIDER_MODULES = frozenset(
    {"data.sentiment", "data.market_signals", "data.macro_data", "data.whale_tracker"}
)
_PROVIDER_LEAVES = frozenset({"sentiment", "market_signals", "macro_data", "whale_tracker"})
_SIGNAL_BANNED = frozenset(
    {
        "now", "utcnow", "today", "time", "monotonic", "perf_counter", "getenv", "environ",
        "putenv", "open", "print", "input", "getLogger", "logging", "structlog", "logger",
        "random", "uuid4", "os", "sleep", "exc_log_fields", "available_at_estimated",
        "available_at_vintage", "get_global_client",
    }
)  # fmt: skip
_LIVE_SCOPE = ["LiveSignals", "values_at"]


def _enclosing_names(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    names: list[str] = []
    cur: ast.AST | None = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(cur.name)
        cur = parents.get(cur)
    return list(reversed(names))


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _provider_refs(tree: ast.AST) -> list[ast.AST]:
    """Every node that imports or names one of the four provider modules / their getters."""
    refs: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in _PROVIDER_MODULES or (
                mod == "data" and any(a.name in _PROVIDER_LEAVES for a in node.names)
            ):
                refs.append(node)
        elif isinstance(node, ast.Import):
            if any(a.name in _PROVIDER_MODULES for a in node.names):
                refs.append(node)
        elif isinstance(node, ast.Attribute) and node.attr in _PROVIDER_LEAVES | {
            "get_global_client"
        }:
            refs.append(node)
        elif isinstance(node, ast.Name) and node.id in {"get_global_client"}:
            refs.append(node)
    return refs


_ENGINE_BANNED = ("data.live_context", "data.live_context_policy", "api.services.context_poller")
_ENGINE_BANNED_IDENTS = frozenset(
    {"live_context", "live_context_policy", "context_poller", "LIVE_BOOK"}
)


def engine_import_problems(src: str) -> list[str]:
    """The engine module reaches no provider client (no import, attribute or getter) and no live
    context module, book or poller (CF-24-01 T-14 (4), T-23); it takes ``LIVE_CONSUMERS`` from
    ``trading.mtf_signals`` (SY-01-21j)."""
    tree = ast.parse(src)
    out = [f"provider: line {n.lineno}" for n in _provider_refs(tree)]
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.ImportFrom):
            mods = [node.module or "", *[f"{node.module}.{a.name}" for a in node.names]]
        elif isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        if any(m == b or m.startswith(b + ".") for m in mods for b in _ENGINE_BANNED):
            out.append(f"live-context import: line {node.lineno}")
        ident = getattr(node, "id", None) or getattr(node, "attr", None)
        if ident in _ENGINE_BANNED_IDENTS:
            out.append(f"live-context name {ident}: line {node.lineno}")
    return out


def live_scope_problems(src: str) -> list[str]:
    """CF-24-01 (PC-5, SY-01-11), inverted: ``mtf_signals.py`` references no provider module,
    client or getter anywhere, ``LiveSignals.values_at`` included."""
    tree = ast.parse(src)
    parents = _parent_map(tree)
    return [
        f"provider reference: {'.'.join(_enclosing_names(n, parents)) or '<module>'}"
        for n in _provider_refs(tree)
    ]


def signal_purity_problems(src: str) -> list[str]:
    """Everything in mtf_signals.py outside ``LiveSignals`` is pure: no wall, env, I/O, logging."""
    tree = ast.parse(src)
    out: list[str] = []
    for top in tree.body:
        if isinstance(top, ast.ClassDef) and top.name == "LiveSignals":
            continue
        used = {n.id for n in ast.walk(top) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(top) if isinstance(n, ast.Attribute)
        }
        out += [f"banned: {name}" for name in sorted(used & _SIGNAL_BANNED)]
    return out


def signal_logging_problems(src: str) -> list[str]:
    """The whole module (the moved live block included) logs nothing new (SY-24-18)."""
    tree = ast.parse(src)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    out = [f"logging: {n}" for n in sorted(used & {"logging", "structlog", "getLogger", "print",
                                                    "exc_log_fields", "logger", "environ",
                                                    "getenv", "os"})]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names]
            mod = getattr(node, "module", None) or ""
            if {mod, *names} & {"logging", "structlog", "os", "common.logging"}:
                out.append(f"import: {mod or names}")
    return out


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n
        for n in klass.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name
    )


def _calls(fn: ast.AST, attr: str) -> list[ast.Call]:
    return [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and ((isinstance(c.func, ast.Attribute) and c.func.attr == attr)
             or (isinstance(c.func, ast.Name) and c.func.id == attr))
    ]


def swallow_problems(src: str) -> list[str]:
    """CR-01 / SEC-07 / RISK-09: swallow-all handlers live ONLY in the moved live block.

    Reports every ``except`` handler (``except*`` included) of any type outside
    ``LiveSignals.values_at``, any shape other than exactly one
    ``except Exception: return _LIVE_READ_FAILED`` inside it (CF-24-01, SY-01-21d), and any use
    of ``suppress`` anywhere.
    """
    tree = ast.parse(src)
    parents = _parent_map(tree)
    out: list[str] = []
    live: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            scope = _enclosing_names(node, parents)
            if scope == _LIVE_SCOPE:
                live.append(node)
            else:
                out.append(
                    f"except handler outside LiveSignals.values_at: {'.'.join(scope) or '<module>'}"
                )
        elif (isinstance(node, ast.Name) and node.id == "suppress") or (
            isinstance(node, ast.Attribute) and node.attr == "suppress"
        ):
            out.append(f"suppress: {'.'.join(_enclosing_names(node, parents)) or '<module>'}")
    malformed = [
        h for h in live
        if not (
            h.type is not None
            and ast.unparse(h.type) == "Exception"
            and h.name is None
            and len(h.body) == 1
            and ast.unparse(h.body[0]) == "return _LIVE_READ_FAILED"
        )
    ]
    if len(live) != 1 or malformed:
        out.append(f"live swallow handlers: count={len(live)} malformed={len(malformed)}")
    return out


#: sha256 of the source segment of the WHOLE ``LiveSignals`` ClassDef, from its ``class`` line to
#: its last line (CF-2401-57 / C12-C13): an ``async def``, a decorator, a default argument, a
#: stored lambda or a docstring edit changes it.  Changing the class means changing this on purpose.
_LIVE_BODY_SHA256 = "482fb405695b5a6284aaebee0a69856de816b09c33f94fa3e8b30fc5017be0bf"
_LIVE_BODY_TEXT = (
    "try:\n    now = WALL_CLOCK.now()\n"
    "    return PitSignals(LIVE_BOOK.view(to_ms(now)), _LIVE_FIELDS).values_at(now)\n"
    "except Exception:\n    return _LIVE_READ_FAILED"
)
#: PC-7: the only names the body may reference (``now`` / ``self`` / ``decision_at`` are locals).
_LIVE_BODY_NAMES = frozenset(
    {"WALL_CLOCK", "LIVE_BOOK", "to_ms", "PitSignals", "_LIVE_FIELDS", "_LIVE_READ_FAILED"}
)
_LIVE_BODY_ATTRS = frozenset({"now", "view", "values_at"})


def live_class_segment(src: str) -> str:
    tree = ast.parse(src)
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LiveSignals"), None
    )
    return (ast.get_source_segment(src, node) or "") if node else ""


def live_body_problems(src: str, *, check_hash: bool = True) -> list[str]:
    """CF-24-01 (SY-01-21a, PC-7): ``LiveSignals`` is ``__slots__ = ()`` plus ONE synchronous
    ``values_at`` whose body is exactly the SY-01-21a text, referencing only the allowlist."""
    tree = ast.parse(src)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LiveSignals"]
    if len(classes) != 1:
        return [f"LiveSignals class definitions: {len(classes)}"]
    cls = classes[0]
    out: list[str] = []
    if [ast.unparse(d) for d in cls.decorator_list] != ["final"] or cls.bases or cls.keywords:
        out.append("class header: expected bare @final")
    stmts = [
        n for n in cls.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str))
    ]  # fmt: skip
    if (
        len(stmts) != 2
        or ast.unparse(stmts[0]) != "__slots__ = ()"
        or type(stmts[1]) is not ast.FunctionDef
        or stmts[1].name != "values_at"
    ):
        out.append("class body is not exactly __slots__ = () and a synchronous values_at")
    else:
        fn = stmts[1]
        if fn.decorator_list or ast.unparse(fn.args) != "self, decision_at: datetime | None=None":
            out.append("values_at: decorators, defaults or signature changed")
        if [ast.unparse(s) for s in fn.body] != [_LIVE_BODY_TEXT]:
            out.append("values_at: body is not the SY-01-21a text")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        local = {"self", "decision_at", "now", "Exception", "datetime", "SignalValues"}
        allowed = _LIVE_BODY_NAMES | local
        out += [f"values_at: name {n}" for n in sorted(names - allowed)]
        out += [f"values_at: attribute {a}" for a in sorted(attrs - _LIVE_BODY_ATTRS)]
    digest = hashlib.sha256(live_class_segment(src).encode()).hexdigest()
    if check_hash and digest != _LIVE_BODY_SHA256:
        out.append("LiveSignals source hash")
    return out


def pit_join_problems(src: str) -> list[str]:
    """CR-02 / CF-SYN-02: ``PitSignals.values_at`` takes no lag and reaches no ingestion helper."""
    tree = ast.parse(src)
    klass = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PitSignals"), None
    )
    fn = next(
        (n for n in (klass.body if klass else [])
         if isinstance(n, ast.FunctionDef) and n.name == "values_at"),
        None,
    )
    if fn is None:
        return ["missing: PitSignals.values_at"]
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    out: list[str] = []
    if args.vararg or args.kwarg:
        out.append("lag: values_at takes *args/**kwargs")
    out += [f"lag: values_at parameter {n}" for n in names if "lag" in n.lower()]
    if names != ["self", "decision_at"]:
        out.append(f"lag: values_at signature {names}")
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    }
    banned = _LAG_CONSTANTS | _INGESTION_ONLY
    out += [f"lag: values_at references {n}" for n in sorted(used & banned)]
    return out


def engine_wiring_problems(src: str) -> list[str]:
    """``close_i`` is the decision time; the swap body is a single source call; guards hold."""
    tree = ast.parse(src)
    out: list[str] = []

    run = _method(tree, "StrategyEngine", "run_backtest")
    pb = _calls(run, "_process_bar")
    if len(pb) != 1:
        out.append(f"run_backtest _process_bar calls: {len(pb)}")
    else:
        kws = {k.arg: k.value for k in pb[0].keywords}
        value = kws.get("decision_at")
        if not (isinstance(value, ast.Name) and value.id == "close_i"):
            out.append("run_backtest does not pass the close_i name as decision_at")
        if [ast.unparse(a) for a in pb[0].args] != ["current_bars", "history_by_symbol"]:
            out.append("run_backtest _process_bar positional args changed")
    # decision_at is only ever the close_i NAME (no recomputation in the call)
    # CR-03: EVERY binding of ``close_i`` counts (Assign, AugAssign, AnnAssign, NamedExpr,
    # for/with targets, unpacking): exactly one, the clock-advance ``Assign``.
    close_binds = [
        n for n in ast.walk(run)
        if isinstance(n, ast.Name) and n.id == "close_i" and isinstance(n.ctx, ast.Store)
    ]
    if len(close_binds) != 1:
        out.append(f"close_i assignments in run_backtest: {len(close_binds)}")

    for cls_fn in ("_process_bar", "_build_mtf_context"):
        fn = _method(tree, "StrategyEngine", cls_fn)
        kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)}
        d = kwonly.get("decision_at")
        if not (isinstance(d, ast.Constant) and d.value is None):
            out.append(f"{cls_fn}: decision_at must be a keyword-only argument defaulting to None")

    process = _method(tree, "StrategyEngine", "_process_bar")
    builds = _calls(process, "_build_mtf_context")
    if len(builds) != 1 or [ast.unparse(k.value) for k in builds[0].keywords
                            if k.arg == "decision_at"] != ["decision_at"]:
        out.append("_process_bar does not forward decision_at to _build_mtf_context")

    build = _method(tree, "StrategyEngine", "_build_mtf_context")
    swallow = [
        n for n in ast.walk(build)
        if isinstance(n, ast.ExceptHandler) and ast.unparse(n.type or ast.Constant("")) in
        {"Exception", "BaseException", ""}
    ]  # the HTF branch keeps its narrow ``except ValueError``; the swallow-all moved out
    if swallow or any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(build)):
        out.append("_build_mtf_context still contains try/import (the live block must be moved)")
    swaps = [
        c for c in _calls(build, "values_at")
        if ast.unparse(c.func) == "self._mtf_signals.values_at"
        and [ast.unparse(a) for a in c.args] == ["decision_at"]
    ]
    if len(swaps) != 1:
        out.append("_build_mtf_context does not call self._mtf_signals.values_at(decision_at)")

    # the paper/live path never passes a decision time (the wall path is untouched)
    stamped = [
        c for c in _calls(tree, "_process_bar") if any(k.arg == "decision_at" for k in c.keywords)
    ]
    if len(stamped) != 1:
        out.append(f"_process_bar calls with decision_at: {len(stamped)} (expected 1)")

    init = _method(tree, "StrategyEngine", "__init__")
    text = ast.unparse(init)
    for needle in (
        "isinstance(mtf_signals, BacktestSignalSource)",
        "mtf_signals is not LIVE_SIGNALS",
        "isinstance(mtf_signals, LiveSignals)",
        "BacktestSignalSource not in type(mtf_signals).__mro__",
    ):
        if needle not in text:
            out.append(f"__init__ guard missing: {needle}")
    return out


class TestEngineAstPin:
    def test_the_engine_imports_no_provider_module(self) -> None:  # WP24-T-17
        assert engine_import_problems(_ENGINE.read_text()) == []

    def test_only_live_signals_values_at_references_the_providers(self) -> None:
        assert live_scope_problems(_SIGNALS.read_text()) == []

    def test_everything_but_the_live_class_is_pure_and_nothing_logs(self) -> None:
        assert signal_purity_problems(_SIGNALS.read_text()) == []
        assert signal_logging_problems(_SIGNALS.read_text()) == []

    def test_pit_signals_has_no_wall_clock_and_no_lag(self) -> None:
        tree = ast.parse(_SIGNALS.read_text())
        pit = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PitSignals")
        used = {n.id for n in ast.walk(pit) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(pit) if isinstance(n, ast.Attribute)
        }
        assert not used & {"now", "utcnow", "today", "time", "available_at_estimated",
                           "available_at_vintage"}
        assert not _calls(pit, "now")
        for fn in (n for n in ast.walk(pit) if isinstance(n, ast.FunctionDef)):
            params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
            assert not [p for p in params if "lag" in p.lower()], (fn.name, params)

    def test_run_backtest_passes_close_i_as_the_decision_time(self) -> None:
        assert engine_wiring_problems(_ENGINE.read_text()) == []

    def test_the_wp22_clock_lines_are_textually_unchanged(self) -> None:
        text = _ENGINE.read_text()
        block = (
            "                close_i = max(b.timestamp for b in current_bars.values()) + "
            "timedelta(\n"
            "                    seconds=_TIMEFRAME_SECONDS[self._timeframe]\n"
            "                )\n"
            "                bar_clock.advance(close_i)\n"
            "                _advance_clock = getattr(self._portfolio, \"advance_clock\", None)\n"
            "                if callable(_advance_clock):\n"
            "                    _advance_clock(close_i)\n"
        )
        assert text.count(block) == 1

    # -- the pins can fail ---------------------------------------------------

    @pytest.mark.parametrize(
        "addition",
        [
            "from data.sentiment import get_global_client as _g",
            "from data import market_signals",
            "import data.macro_data",
            "_x = whale_tracker.get_global_client",
            "from data.live_context import LIVE_BOOK",  # CF-24-01 T-14 (4)
            "import data.live_context",
            "from data import live_context",
            "from data.live_context_policy import FLOOR_S",
            "from api.services.context_poller import ContextPoller",
            "_b = LIVE_BOOK",
        ],
    )
    def test_engine_import_pin_fails_on_mutated_source(self, addition: str) -> None:
        src = _ENGINE.read_text()
        assert engine_import_problems(src) == []
        assert engine_import_problems(src + f"\n{addition}\n")

    def test_live_scope_pin_fails_when_a_provider_is_read_anywhere(self) -> None:
        src = _SIGNALS.read_text()
        assert live_scope_problems(src) == []
        leaked = src + "\n\ndef _leak() -> object:\n    from data.sentiment import get_global_client\n    return get_global_client()\n"  # noqa: E501
        assert any("provider reference: _leak" in p for p in live_scope_problems(leaked))
        old = "            now = WALL_CLOCK.now()\n"  # the old home of the provider reads
        assert src.count(old) == 1
        probe = "            from data.sentiment import get_global_client as _g\n"
        in_live = src.replace(old, probe + old)
        assert "provider reference: LiveSignals.values_at" in live_scope_problems(in_live)
        in_pit = src.replace(
            "        decision_ms = to_ms(decision_at)",
            "        from data.sentiment import get_global_client as _g\n"
            "        decision_ms = to_ms(decision_at)",
        )
        assert in_pit != src and live_scope_problems(in_pit)

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            ("        decision_ms = to_ms(decision_at)",
             "        decision_ms = to_ms(datetime.now())",
             "banned: now"),
            ("        decision_ms = to_ms(decision_at)",
             "        decision_ms = to_ms(decision_at) - available_at_estimated('x', 0)",
             "banned: available_at_estimated"),
            ("        decision_ms = to_ms(decision_at)",
             "        _ = os.environ",
             "banned: environ"),
        ],
    )
    def test_purity_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _SIGNALS.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert expected in signal_purity_problems(src.replace(old, new))

    def test_logging_pin_fails_on_a_logger(self) -> None:
        src = _SIGNALS.read_text()
        assert signal_logging_problems(src + "\nimport structlog\n_log = structlog.get_logger()\n")
        assert signal_logging_problems(src + "\n_x = print\n")

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            ("await self._process_bar(current_bars, history_by_symbol, decision_at=close_i)",
             "await self._process_bar(current_bars, history_by_symbol)",
             "run_backtest does not pass the close_i name as decision_at"),
            ("await self._process_bar(current_bars, history_by_symbol, decision_at=close_i)",
             "await self._process_bar(current_bars, history_by_symbol, decision_at=datetime.now())",
             "run_backtest does not pass the close_i name as decision_at"),
            ("await self._process_bar(current_bars, history_by_symbol, decision_at=close_i)",
             "await self._process_bar(current_bars, history_by_symbol, "
             "decision_at=bar_clock.now())",
             "run_backtest does not pass the close_i name as decision_at"),
            ("        mtf_context = self._build_mtf_context(bar_timestamp, "
             "decision_at=decision_at)",
             "        mtf_context = self._build_mtf_context(bar_timestamp)",
             "_process_bar does not forward decision_at to _build_mtf_context"),
            ("        values = self._mtf_signals.values_at(decision_at)\n",
             "        values = self._mtf_signals.values_at(None)\n",
             "_build_mtf_context does not call self._mtf_signals.values_at(decision_at)"),
            ("        elif mtf_signals is not None and mtf_signals is not LIVE_SIGNALS:",
             "        elif mtf_signals is not None and not isinstance(mtf_signals, LiveSignals):",
             "__init__ guard missing: mtf_signals is not LIVE_SIGNALS"),
        ],
    )
    def test_wiring_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _ENGINE.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert engine_wiring_problems(src) == []
        assert expected in engine_wiring_problems(src.replace(old, new))

    def test_wiring_pin_fails_when_the_live_block_comes_back_into_the_engine(self) -> None:
        src = _ENGINE.read_text()
        old = "        values = self._mtf_signals.values_at(decision_at)"
        new = (
            "        try:\n            pass\n        except Exception:\n            pass\n" + old
        )
        assert src.count(old) == 1
        assert any("still contains try/import" in p for p in
                   engine_wiring_problems(src.replace(old, new)))


    # -- WP2.4 P1 r2 (E1): the swallow-all pin ---------------------------------

    def test_swallow_alls_are_confined_to_the_live_block(self) -> None:
        assert swallow_problems(_SIGNALS.read_text()) == []

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            # (i) the critic's surviving mutant: a swallow around the int()/float() conversion
            ("            out[name] = int(value) if name in _INT_FIELDS else float(value)\n",
             "            try:\n"
             "                out[name] = int(value) if name in _INT_FIELDS else float(value)\n"
             "            except Exception:\n"
             "                pass\n",
             "except handler outside LiveSignals.values_at: PitSignals.values_at"),
            # (ii) a narrow handler in PitSignals
            ("        return SignalValues(**out)  # type: ignore[arg-type]\n",
             "        try:\n"
             "            return SignalValues(**out)  # type: ignore[arg-type]\n"
             "        except ValueError:\n"
             "            return SignalValues()\n",
             "except handler outside LiveSignals.values_at: PitSignals.values_at"),
            # (iii) a module-level helper
            ("\n\n@final\nclass LiveSignals:",
             "\n\ndef _swallow() -> None:\n    try:\n        pass\n    except Exception:\n"
             "        pass\n\n\n@final\nclass LiveSignals:",
             "except handler outside LiveSignals.values_at: _swallow"),
            # (iv) the one live handler narrowed (security M6) / back to the HEAD `pass` shape
            ("        except Exception:\n            return _LIVE_READ_FAILED",
             "        except ValueError:\n            return _LIVE_READ_FAILED",
             "live swallow handlers: count=1 malformed=1"),
            ("        except Exception:\n            return _LIVE_READ_FAILED",
             "        except Exception:\n            pass",
             "live swallow handlers: count=1 malformed=1"),
            ("        except Exception:\n            return _LIVE_READ_FAILED",
             "        except Exception:\n            return SignalValues()",
             "live swallow handlers: count=1 malformed=1"),
            # (v) a 2nd handler inside LiveSignals.values_at
            ("            return _LIVE_READ_FAILED\n",
             "            return _LIVE_READ_FAILED\n"
             "        try:\n            pass\n        except Exception:\n"
             "            return _LIVE_READ_FAILED\n",
             "live swallow handlers: count=2 malformed=0"),
            # (vii) the handler removed
            ("        except Exception:\n            return _LIVE_READ_FAILED",
             "        finally:\n            pass",
             "live swallow handlers: count=0 malformed=0"),
            # (vi) contextlib.suppress
            ("        decision_ms = to_ms(decision_at)  # plain int",
             "        from contextlib import suppress\n\n"
             "        with suppress(Exception):\n            pass\n"
             "        decision_ms = to_ms(decision_at)  # plain int",
             "suppress: PitSignals.values_at"),
        ],
    )
    def test_swallow_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _SIGNALS.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert swallow_problems(src) == []
        problems = swallow_problems(src.replace(old, new))
        assert expected in problems, problems

    # -- CF-24-01 P4a (T-15, T-22, PC-7, CF-2401-57): the LiveSignals text ----------------

    def test_the_live_class_is_the_sy_01_21a_text_and_its_hash_is_pinned(self) -> None:
        src = _SIGNALS.read_text()
        assert live_body_problems(src) == []
        assert hashlib.sha256(live_class_segment(src).encode()).hexdigest() == _LIVE_BODY_SHA256
        assert live_class_segment(src).startswith("class LiveSignals:")  # from the `class` line on

    @pytest.mark.parametrize(
        ("old", "new", "structural"),
        [
            ("    def values_at(self, decision_at: datetime | None = None)",
             "    async def values_at(self, decision_at: datetime | None = None)", True),  # C12
            ("    def values_at(self, decision_at: datetime | None = None)",
             "    @staticmethod\n    def values_at(self, decision_at: datetime | None = None)",
             True),
            ("decision_at: datetime | None = None) -> SignalValues:\n        try:",
             "decision_at: datetime | None = LIVE_BOOK.reset()) -> SignalValues:\n"
             "        try:", True),
            ("            now = WALL_CLOCK.now()\n",
             "            now = (lambda: WALL_CLOCK.now())()\n", True),  # C13
            ("            now = WALL_CLOCK.now()\n",
             "            _f = lambda: LIVE_BOOK.publish('s', [], 0)\n"
             "            now = WALL_CLOCK.now()\n",
             True),
            ("            now = WALL_CLOCK.now()\n",
             "            def _f() -> None: ...\n            now = WALL_CLOCK.now()\n", True),
            # CF-2401-66: the reader passes wall-clock, never a decision time
            ("LIVE_BOOK.view(to_ms(now))", "LIVE_BOOK.view(to_ms(decision_at))", True),
            ("LIVE_BOOK.view(to_ms(now))", "LIVE_BOOK.snapshot()", True),
            ("LIVE_BOOK.view(to_ms(now))", "LIVE_BOOK.heartbeat(to_ms(now))", True),
            ("_LIVE_FIELDS).values_at(now)", "_LIVE_FIELDS).values_at(decision_at)", True),
            ("        except Exception:\n            return _LIVE_READ_FAILED",
             "        except Exception:\n            return SignalValues()", True),
            ("    __slots__ = ()\n", "    __slots__ = ()\n    x = 1\n", True),
            ("    __slots__ = ()\n", "    __slots__ = ('x',)\n", True),
            ("\n\nLIVE_SIGNALS: Final",
             "    def other(self) -> None: ...\n\n\nLIVE_SIGNALS: Final", True),
            ("@final\nclass LiveSignals:", "class LiveSignals:", True),
            # a docstring edit: only the hash notices
            ("    guard checks it by identity.", "    guard checks it by identity!", False),
        ],
    )
    def test_live_body_pin_fails_on_mutated_source(
        self, old: str, new: str, structural: bool
    ) -> None:
        src = _SIGNALS.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        mutated = src.replace(old, new)
        assert live_body_problems(mutated) != []
        assert (live_body_problems(mutated, check_hash=False) != []) is structural

    # -- WP2.4 P1 r2 (E2): the lag pin on PitSignals.values_at ------------------

    def test_pit_values_at_takes_no_lag_and_never_reaches_the_ingestion_helpers(self) -> None:
        assert pit_join_problems(_SIGNALS.read_text()) == []
        assert ingestion_references(_SIGNALS.read_text()) == []

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            ("    def values_at(self, decision_at: datetime | None) -> SignalValues:\n"
             "        if decision_at is None:",
             "    def values_at(self, decision_at: datetime | None, offset_ms: int = 0) "
             "-> SignalValues:\n        if decision_at is None:",
             "lag: values_at signature ['self', 'decision_at', 'offset_ms']"),
            ("        decision_ms = to_ms(decision_at)  # plain int",
             "        decision_ms = to_ms(decision_at) - _FGI_LAG_MS  # plain int",
             "lag: values_at references _FGI_LAG_MS"),
        ],
    )
    def test_pit_join_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _SIGNALS.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert pit_join_problems(src) == []
        problems = pit_join_problems(src.replace(old, new))
        assert expected in problems, problems

    def test_pit_join_pin_reports_a_missing_method_instead_of_raising(self) -> None:
        src = _SIGNALS.read_text()
        old = "class PitSignals(BacktestSignalSource):"
        assert src.count(old) == 1
        assert pit_join_problems(src.replace(old, "class PitSignalsX(BacktestSignalSource):")) == [
            "missing: PitSignals.values_at"
        ]

    def test_an_ingestion_alias_in_the_signals_module_is_caught(self) -> None:
        src = _SIGNALS.read_text()
        assert ingestion_references(src) == []
        assert ingestion_references(src + "\n_est = available_at_estimated\n") == ["<module>"]

    # -- WP2.4 P1 r2 (E3): every close_i binding is counted --------------------

    @pytest.mark.parametrize(
        "stmt",
        [
            "close_i += timedelta(0)",
            "close_i: datetime = close_i + timedelta(0)",
            "(close_i := close_i + timedelta(0))",
        ],
    )
    def test_close_i_pin_fails_on_every_rebinding_form(self, stmt: str) -> None:
        src = _ENGINE.read_text()
        call = "await self._process_bar(current_bars, history_by_symbol, decision_at=close_i)"
        old = "                " + call
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert engine_wiring_problems(src) == []
        problems = engine_wiring_problems(src.replace(old, f"                {stmt}\n{old}"))
        assert "close_i assignments in run_backtest: 2" in problems, problems

    # -- WP2.4 P1 r2 (E4): each BACKTEST guard clause is pinned ----------------

    @pytest.mark.parametrize(
        ("old", "new", "needle"),
        [
            ("                isinstance(mtf_signals, LiveSignals)\n                or ",
             "                ",
             "isinstance(mtf_signals, LiveSignals)"),
            ("                or BacktestSignalSource not in type(mtf_signals).__mro__\n",
             "",
             "BacktestSignalSource not in type(mtf_signals).__mro__"),
            ("                or not isinstance(mtf_signals, BacktestSignalSource)\n",
             "",
             "isinstance(mtf_signals, BacktestSignalSource)"),
        ],
    )
    def test_init_guard_pin_fails_when_a_backtest_clause_is_removed(
        self, old: str, new: str, needle: str
    ) -> None:
        src = _ENGINE.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert engine_wiring_problems(src) == []
        problems = engine_wiring_problems(src.replace(old, new))
        assert f"__init__ guard missing: {needle}" in problems, problems


# ===========================================================================
# WP2.4 P2 carry-forwards (p1-synthesis.md section 6)
#   CF-P1-02  PitSignals.select joins the lag / ingestion / no-swallow pins
#   CF-P1-04  production name-reference allowlist for the live and backtest sources
# ===========================================================================

_SELECT_SIGNATURE = ["self", "decision_times", "fields"]


def pit_select_problems(src: str) -> list[str]:
    """CF-P1-02: ``PitSignals.select`` takes no lag, reaches no ingestion helper or lag constant.

    Same shape as :func:`pit_join_problems` (which keeps pinning ``values_at`` unchanged). The
    "no ``except`` anywhere but the moved live block" half is :func:`swallow_problems` (E1): a
    handler in ``select`` is reported there as ``except handler outside LiveSignals.values_at``.
    """
    tree = ast.parse(src)
    klass = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PitSignals"), None
    )
    fn = next(
        (n for n in (klass.body if klass else [])
         if isinstance(n, ast.FunctionDef) and n.name == "select"),
        None,
    )
    if fn is None:
        return ["missing: PitSignals.select"]
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    out: list[str] = []
    if args.vararg or args.kwarg:
        out.append("lag: select takes *args/**kwargs")
    out += [f"lag: select parameter {n}" for n in names if "lag" in n.lower()]
    if names != _SELECT_SIGNATURE:
        out.append(f"lag: select signature {names}")
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    } | {n.name for n in ast.walk(fn) if isinstance(n, ast.alias)}
    banned = _LAG_CONSTANTS | _INGESTION_ONLY
    out += [f"lag: select references {n}" for n in sorted(used & banned)]
    return out


class TestCfP102SelectPin:
    def test_select_takes_no_lag_and_never_reaches_the_ingestion_helpers(self) -> None:
        src = _SIGNALS.read_text()
        assert pit_select_problems(src) == []
        assert pit_join_problems(src) == []  # values_at stays pinned as before
        assert ingestion_references(src) == []
        assert swallow_problems(src) == []

    def test_the_expected_ingestion_sites_are_unchanged(self) -> None:
        assert _EXPECTED_INGESTION_SITES == [
            "parse_jsonl_lines -> available_at_estimated",
            "parse_jsonl_lines -> available_at_vintage",
        ]
        assert ingestion_call_sites(_LEAF.read_text()) == _EXPECTED_INGESTION_SITES

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            ("        self, decision_times: Iterable[datetime], fields: Iterable[str]\n"
             "    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "        self, decision_times: Iterable[datetime], fields: Iterable[str],"
             " offset_ms: int = 0\n    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "lag: select signature ['self', 'decision_times', 'fields', 'offset_ms']"),
            ("        self, decision_times: Iterable[datetime], fields: Iterable[str]\n"
             "    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "        self, decision_times: Iterable[datetime], fields: Iterable[str],"
             " *, lag_ms: int = 0\n    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "lag: select parameter lag_ms"),
            ("        self, decision_times: Iterable[datetime], fields: Iterable[str]\n"
             "    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "        self, fields: Iterable[str], decision_times: Iterable[datetime]\n"
             "    ) -> tuple[BacktestSignalSource, MtfContextReport]:",
             "lag: select signature ['self', 'fields', 'decision_times']"),
            ("        decisions = tuple(to_ms(moment) for moment in decision_times)",
             "        decisions = tuple(to_ms(moment) - _FGI_LAG_MS for moment in decision_times)",
             "lag: select references _FGI_LAG_MS"),
            ("        decisions = tuple(to_ms(moment) for moment in decision_times)",
             "        decisions = (available_at_estimated('x', 0),)",
             "lag: select references available_at_estimated"),
        ],
    )
    def test_select_pin_fails_on_mutated_source(self, old: str, new: str, expected: str) -> None:
        src = _SIGNALS.read_text()
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert pit_select_problems(src) == []
        problems = pit_select_problems(src.replace(old, new))
        assert expected in problems, problems

    def test_select_pin_reports_a_missing_method_instead_of_raising(self) -> None:
        src = _SIGNALS.read_text()
        old = "    def select(\n"
        assert src.count(old) == 1
        assert pit_select_problems(src.replace(old, "    def choose(\n")) == [
            "missing: PitSignals.select"
        ]

    def test_an_ingestion_alias_inside_select_is_caught_by_the_reference_scan(self) -> None:
        src = _SIGNALS.read_text()
        old = "        decisions = tuple(to_ms(moment) for moment in decision_times)"
        assert src.count(old) == 1
        mutated = src.replace(old, "        _est = available_at_estimated\n" + old)
        assert ingestion_references(mutated) == ["select"]

    def test_an_import_alias_inside_select_is_caught(self) -> None:
        # CR-06: ``from m import helper as _v; _v(0)`` has no Name/Attribute node for the helper
        src = _SIGNALS.read_text()
        old = "        decisions = tuple(to_ms(moment) for moment in decision_times)"
        assert src.count(old) == 1
        mutated = src.replace(
            old,
            "        from data.point_in_time import available_at_vintage as _v\n"
            "        _v(0)\n" + old,
        )
        assert "lag: select references available_at_vintage" in pit_select_problems(mutated)
        assert ingestion_references(mutated) == ["select"]
        assert pit_select_problems(src) == [] and ingestion_references(src) == []

    def test_an_except_handler_in_select_is_caught_by_the_swallow_pin(self) -> None:
        src = _SIGNALS.read_text()
        old = (
            "        decisions = tuple(to_ms(moment) for moment in decision_times)"
            "  # naive -> PitError\n"
        )
        assert src.count(old) == 1, "mutation anchor drifted: update this test"
        assert swallow_problems(src) == []
        mutated = src.replace(
            old,
            "        try:\n            decisions = tuple(to_ms(m) for m in decision_times)\n"
            "        except PitError:\n            decisions = ()\n",
        )
        assert "except handler outside LiveSignals.values_at: PitSignals.select" in (
            swallow_problems(mutated)
        )

    def test_coverage_code_added_to_the_module_is_covered_by_the_same_swallow_pin(self) -> None:
        src = _SIGNALS.read_text()
        anchor = "\n\ndef check_backtest_source("
        assert src.count(anchor) == 1
        extra = (
            "\n\ndef _coverage_helper(x: int) -> int:\n    try:\n        return x\n"
            "    except Exception:\n        return 0"
        )
        assert "except handler outside LiveSignals.values_at: _coverage_helper" in (
            swallow_problems(src.replace(anchor, extra + anchor))
        )


# --- CF-P1-04: who may name the live / backtest sources ---------------------

_MTF = "packages/trading/mtf_signals.py"
_ENGINE_REL = "packages/trading/strategy_engine.py"
_RUNNER_REL = "packages/trading/backtest.py"
_RUNNER = _REPO / "packages" / "trading" / "backtest.py"
_OPTIMIZER_REL = "packages/trading/optimizer.py"
_STRATEGY_REL = "packages/trading/strategy.py"
_LIVE_NAMES = frozenset({"LIVE_SIGNALS", "LiveSignals"})
_SOURCE_NAME = "BacktestSignalSource"
#: LIVE_SIGNALS / LiveSignals: the module that defines them and the engine that guards them.
_LIVE_ALLOWED = frozenset({_MTF, _ENGINE_REL})
#: BacktestSignalSource: those two plus the runner / optimizer sites that take the source parameter.
_SOURCE_ALLOWED = _LIVE_ALLOWED | {_RUNNER_REL, _OPTIMIZER_REL}
#: Who may import the module at all (strategy.py only for ``ALL_MTF_SIGNAL_FIELDS``).
#: WP2.4 P3: ``apps/api/schemas.py`` imports the two VERSION constants only.  The names are
#: pinned per file by ``_PINNED_IMPORT_NAMES`` (``pinned_import_problems`` and its tests); the
#: name bans above still apply to it: it may not mention LIVE_SIGNALS / LiveSignals /
#: BacktestSignalSource.
_API_SCHEMAS_REL = "apps/api/schemas.py"
_MODULE_IMPORTERS = _SOURCE_ALLOWED | {_STRATEGY_REL, _API_SCHEMAS_REL}
#: Every ``apps/`` importer of the module, with the EXACT names it may take (CR-01 / SEC-05).
_PINNED_IMPORT_NAMES: Final[Mapping[str, frozenset[str]]] = {
    _API_SCHEMAS_REL: frozenset({"KNOWN_MTF_CONTEXT_VERSIONS", "MTF_CONTEXT_VERSION"}),
}
#: Identifiers a pinned file may not use: they are how the module is reached without an import
#: statement (``trading.mtf_signals.X``, ``__import__``, ``importlib``, ``sys.modules``).
_DYNAMIC_IMPORT_IDENTS: Final = frozenset(
    {"importlib", "import_module", "__import__", "modules", "mtf_signals"}
)
_ALLOWED_SUBCLASSES = frozenset({"UnavailableSignals", "PitSignals"})
_META_CALLS = frozenset({"new_class", "ABCMeta"})


def production_files() -> list[tuple[str, Path]]:
    """Every production ``.py`` under ``packages/`` and ``apps/`` (tests and caches excluded)."""
    out: list[tuple[str, Path]] = []
    for root in ("packages", "apps"):
        for path in sorted((_REPO / root).rglob("*.py")):
            rel = path.relative_to(_REPO)
            parts = rel.parts
            if any(p in {"tests", "__pycache__", ".venv", "node_modules"} for p in parts):
                continue
            if path.name.startswith("test_") or path.name == "conftest.py":
                continue
            out.append((rel.as_posix(), path))
    return out


def _mentions(node: ast.AST, name: str) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id == name)
        or (isinstance(n, ast.Attribute) and n.attr == name)
        or (isinstance(n, ast.Constant) and n.value == name)
        for n in ast.walk(node)
    )


def _is_module_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        mod = (node.module or "").split(".")[-1]
        return mod == "mtf_signals" or any(a.name == "mtf_signals" for a in node.names)
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[-1] == "mtf_signals" for a in node.names)
    return False


def name_reference_problems(rel: str, src: str) -> list[str]:
    """CF-P1-04 / SEC-R2-03: the name allowlist, the importer allowlist and the construction ban."""
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name.split(".")[-1]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        if name in _LIVE_NAMES and rel not in _LIVE_ALLOWED:
            out.append(f"{rel}: {name} referenced outside {sorted(_LIVE_ALLOWED)}")
        elif name == _SOURCE_NAME and rel not in _SOURCE_ALLOWED:
            out.append(f"{rel}: {name} referenced outside {sorted(_SOURCE_ALLOWED)}")

        if _is_module_import(node):
            if rel not in _MODULE_IMPORTERS:
                allowed = sorted(_MODULE_IMPORTERS)
                out.append(f"{rel}: imports trading.mtf_signals outside {allowed}")
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                out.append(f"{rel}: star import of mtf_signals")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.split(".")[-1] == "mtf_signals"
            and "." in node.value
            and rel not in _MODULE_IMPORTERS
        ):
            out.append(f"{rel}: dynamic reference to the mtf_signals module")

        if isinstance(node, ast.ClassDef):
            heads = [*node.bases, *(k.value for k in node.keywords)]
            if any(_mentions(h, _SOURCE_NAME) for h in heads):
                if rel != _MTF:
                    out.append(f"{rel}: class {node.name} derives from {_SOURCE_NAME}")
                elif node.name not in _ALLOWED_SUBCLASSES:
                    out.append(f"{rel}: unexpected subclass {node.name} of {_SOURCE_NAME}")
            if node.name == _SOURCE_NAME and rel == _MTF:
                if [ast.unparse(b) for b in node.bases] != ["abc.ABC"] or node.keywords:
                    out.append(f"{rel}: {_SOURCE_NAME} must be a plain abc.ABC subclass")
                hooks = {
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                } & {"__subclasshook__", "__instancecheck__", "__subclasscheck__", "register"}
                out += [f"{rel}: {_SOURCE_NAME} defines {h}" for h in sorted(hooks)]

        if isinstance(node, ast.Call) and rel in _SOURCE_ALLOWED:
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if fname == "register":
                out.append(f"{rel}: .register( call in a file that names the sources")
            if fname in _META_CALLS:
                out.append(f"{rel}: {fname}( in a file that names the sources")
            if isinstance(func, ast.Name) and func.id == "type" and len(node.args) == 3:
                out.append(f"{rel}: three-argument type( in a file that names the sources")
    return out


def pinned_import_problems(src: str, names: frozenset[str]) -> list[str]:
    """CR-01 / SEC-05: ``src`` imports exactly ``names`` from ``trading.mtf_signals``, nothing else.

    Exactly one import statement, a plain absolute ``from trading.mtf_signals import ...`` with
    exactly ``names`` (no alias, star, duplicate, drop or extra), and no other way to reach the
    module: no ``importlib`` / ``import_module`` / ``__import__`` / ``modules`` / ``mtf_signals``
    identifier and no string constant naming the module.  Pure, so it also runs on hostile text.
    """
    tree = ast.parse(src)
    out: list[str] = []
    imports = [n for n in ast.walk(tree) if _is_module_import(n)]
    if len(imports) != 1:
        out.append(f"{len(imports)} import statements of trading.mtf_signals, expected 1")
    for node in imports:
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "trading.mtf_signals"
            and node.level == 0
        ):
            out.append("not a plain absolute 'from trading.mtf_signals import ...'")
            continue
        if sorted(a.name for a in node.names) != sorted(names):
            out.append(f"imported names {[a.name for a in node.names]} != pin {sorted(names)}")
        if any(a.asname is not None for a in node.names):
            out.append("aliased import")
    for node in ast.walk(tree):
        ident: str | None = None
        if isinstance(node, ast.Name):
            ident = node.id
        elif isinstance(node, ast.Attribute):
            ident = node.attr
        elif isinstance(node, ast.alias):
            ident = node.name.split(".")[0]
        if ident in _DYNAMIC_IMPORT_IDENTS:
            out.append(f"dynamic-import or module-object reference: {ident}")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value == "mtf_signals" or node.value.endswith(".mtf_signals"))
        ):
            out.append(f"string reference to the module: {node.value!r}")
    return out


_PIN_LINE = "from trading.mtf_signals import KNOWN_MTF_CONTEXT_VERSIONS, MTF_CONTEXT_VERSION\n"


class TestCfP104NameReferenceAllowlist:
    def test_the_production_tree_passes(self) -> None:
        files = production_files()
        assert len(files) > 50  # the scan really sees the tree
        assert _MTF in {rel for rel, _ in files}
        problems = [
            p for rel, path in files for p in name_reference_problems(rel, path.read_text())
        ]
        assert problems == []

    def test_the_allowlists_are_tight_not_stale(self) -> None:
        live: set[str] = set()
        source: set[str] = set()
        for rel, path in production_files():
            tree = ast.parse(path.read_text())
            for name in (_SOURCE_NAME, *_LIVE_NAMES):
                if _mentions(tree, name) or any(
                    isinstance(n, ast.alias) and n.name == name for n in ast.walk(tree)
                ):
                    (live if name in _LIVE_NAMES else source).add(rel)
        assert live == set(_LIVE_ALLOWED)
        assert source == set(_SOURCE_ALLOWED)

    def test_only_the_expected_files_import_the_module(self) -> None:
        importers = {
            rel
            for rel, path in production_files()
            if any(_is_module_import(n) for n in ast.walk(ast.parse(path.read_text())))
        }
        assert importers == set(_MODULE_IMPORTERS) - {_MTF}  # the module never imports itself

    def test_every_apps_importer_has_a_name_pin(self) -> None:
        # No ``apps/`` file can join the importer allowlist without a per-name pin (CF-P3-09).
        assert {f for f in _MODULE_IMPORTERS if f.startswith("apps/")} == set(_PINNED_IMPORT_NAMES)

    def test_pinned_files_import_exactly_their_names(self) -> None:
        for rel, names in _PINNED_IMPORT_NAMES.items():
            assert pinned_import_problems((_REPO / rel).read_text(), names) == [], rel

    def test_the_pin_helper_accepts_the_exact_line(self) -> None:
        assert pinned_import_problems(_PIN_LINE, _PINNED_IMPORT_NAMES[_API_SCHEMAS_REL]) == []

    @pytest.mark.parametrize(
        "src",
        [
            _PIN_LINE.replace("VERSION\n", "VERSION, PitSignals\n"),
            _PIN_LINE.replace("VERSION\n", "VERSION, select_mtf_signals\n"),
            _PIN_LINE + "import trading.mtf_signals as _m\nx = getattr(_m, 'LIVE_' + 'SIGNALS')\n",
            _PIN_LINE + "import importlib\nm = importlib.import_module('trading.mtf_signals')\n",
            _PIN_LINE.replace("MTF_CONTEXT_VERSION\n", "MTF_CONTEXT_VERSION as V\n"),
            _PIN_LINE + "from trading.mtf_signals import UNAVAILABLE\n",
            "from trading.mtf_signals import *\n",
            "from trading import mtf_signals\n",
            "from .mtf_signals import KNOWN_MTF_CONTEXT_VERSIONS, MTF_CONTEXT_VERSION\n",
            _PIN_LINE.replace("VERSION\n", "VERSION, MTF_CONTEXT_VERSION\n"),
            "from trading.mtf_signals import KNOWN_MTF_CONTEXT_VERSIONS\n",
            _PIN_LINE + "import trading\nx = trading.mtf_signals.PitSignals\n",
            _PIN_LINE + "m = __import__('trading.' + 'mtf_signals')\n",
            _PIN_LINE + "import sys\nm = sys.modules['trading.mtf' + '_signals']\n",
            _PIN_LINE
            + "import importlib\nm = importlib.import_module('trading.' + 'mtf_signals')\n",
        ],
        ids=[
            "extra-PitSignals",
            "extra-select",
            "module-alias-getattr",
            "importlib-string",
            "name-alias",
            "second-ImportFrom",
            "star",
            "from-trading-import-module",
            "relative",
            "duplicate",
            "dropped",
            "ARB-1-attribute",
            "ARB-2-dunder-import",
            "ARB-3-sys-modules",
            "ARB-4-import-module-split",
        ],
    )
    def test_the_pin_helper_catches_widening(self, src: str) -> None:
        assert pinned_import_problems(src, _PINNED_IMPORT_NAMES[_API_SCHEMAS_REL]) != []

    def test_the_two_real_subclasses_are_the_only_ones(self) -> None:
        tree = ast.parse(_SIGNALS.read_text())
        subs = sorted(
            n.name
            for n in tree.body
            if isinstance(n, ast.ClassDef) and any(_mentions(b, _SOURCE_NAME) for b in n.bases)
        )
        assert subs == sorted(_ALLOWED_SUBCLASSES)

    @pytest.mark.parametrize(
        ("rel", "src", "needle"),
        [
            ("packages/trading/other.py", "x = LIVE_SIGNALS\n", "LIVE_SIGNALS referenced outside"),
            ("packages/trading/other.py", "from trading.mtf_signals import LiveSignals\n",
             "LiveSignals referenced outside"),
            ("apps/api/routers/runs.py", "import trading.mtf_signals as m\ny = m.LIVE_SIGNALS\n",
             "imports trading.mtf_signals outside"),
            ("packages/trading/backtest.py", "x = LIVE_SIGNALS\n",
             "LIVE_SIGNALS referenced outside"),
            ("packages/trading/optimizer.py", "def f(s: LiveSignals) -> None: ...\n",
             "LiveSignals referenced outside"),
            ("packages/trading/other.py", "y = getattr(m, 'BacktestSignalSource')\n",
             "BacktestSignalSource referenced outside"),
            ("packages/trading/other.py", "from trading import mtf_signals\n",
             "imports trading.mtf_signals outside"),
            ("packages/trading/other.py", "from trading.mtf_signals import *\n",
             "star import of mtf_signals"),
            ("packages/trading/other.py", "m = importlib.import_module('trading.mtf_signals')\n",
             "dynamic reference to the mtf_signals module"),
            ("packages/trading/other.py",
             "class Evil(BacktestSignalSource):\n    pass\n",
             "class Evil derives from BacktestSignalSource"),
            ("packages/trading/backtest.py",
             "class Evil(BacktestSignalSource):\n    pass\n",
             "class Evil derives from BacktestSignalSource"),
            ("packages/trading/backtest.py",
             "class Evil(abc.ABC, metaclass=BacktestSignalSource):\n    pass\n",
             "class Evil derives from BacktestSignalSource"),
            ("packages/trading/mtf_signals.py",
             "class Evil(BacktestSignalSource):\n    pass\n",
             "unexpected subclass Evil of BacktestSignalSource"),
            ("packages/trading/backtest.py",
             "BacktestSignalSource.register(Evil)\n",
             ".register( call in a file that names the sources"),
            ("packages/trading/strategy_engine.py",
             "_b = BacktestSignalSource\n_b.register(Evil)\n",
             ".register( call in a file that names the sources"),
            ("packages/trading/backtest.py",
             "Evil = type('Evil', (BacktestSignalSource,), {})\n",
             "three-argument type( in a file that names the sources"),
            ("packages/trading/optimizer.py",
             "Evil = abc.ABCMeta('Evil', (BacktestSignalSource,), {})\n",
             "ABCMeta( in a file that names the sources"),
            ("packages/trading/optimizer.py",
             "Evil = types.new_class('Evil', (BacktestSignalSource,))\n",
             "new_class( in a file that names the sources"),
        ],
    )
    def test_the_pin_fails_on_every_route(self, rel: str, src: str, needle: str) -> None:
        problems = name_reference_problems(rel, src)
        assert any(needle in p for p in problems), problems

    def test_the_pin_fails_on_a_mutated_real_file(self) -> None:
        src = _RUNNER.read_text()
        assert name_reference_problems(_RUNNER_REL, src) == []
        assert name_reference_problems(
            _RUNNER_REL, src + "\nclass _Evil(BacktestSignalSource):\n    pass\n"
        )
        sig = _SIGNALS.read_text()
        assert name_reference_problems(_MTF, sig) == []
        assert name_reference_problems(
            _MTF, sig.replace("class BacktestSignalSource(abc.ABC):",
                              "class BacktestSignalSource(abc.ABC, metaclass=abc.ABCMeta):")
        )
        hooked = sig.replace(
            "    mode: ClassVar[str]\n\n    @abc.abstractmethod",
            "    mode: ClassVar[str]\n\n    @classmethod\n"
            "    def __subclasshook__(cls, other: type) -> bool:\n        return True\n\n"
            "    @abc.abstractmethod",
        )
        assert hooked != sig
        assert any("defines __subclasshook__" in p for p in name_reference_problems(_MTF, hooked))

    def test_the_pin_is_blind_to_tests_by_design(self) -> None:
        # tests may subclass the source (the P1 guard tests do): they are not production files
        assert all("/tests/" not in rel and not Path(rel).name.startswith("test_")
                   for rel, _ in production_files())
