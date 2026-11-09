"""
tests/unit/test_cf2401_ast_pins.py
-----------------------------------
CF-24-01 AST pins. P2 part (T-09, SY-01-22b): the provider fetch layer boundaries. P3/P4a extend
this file. Every pin is a function over SOURCE TEXT with a mutated-source self-test, so a pin
that stops detecting anything fails here instead of passing vacuously.
"""

from __future__ import annotations

import ast
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_DATA = Path(__file__).resolve().parents[2] / "packages" / "data"
_STDLIB = frozenset(sys.stdlib_module_names)
_P2_FILES = (
    "provider_http",
    "provider_bands",
    "sentiment",
    "market_signals",
    "macro_data",
    "whale_tracker",
)
_LOGGING_NAMES = frozenset(
    {"logging", "structlog", "logger", "getLogger", "exc_log_fields", "print"}
)

_EXPECTED_BANDS = {
    "FGI_MIN": 0,
    "FGI_MAX": 100,
    "DOMINANCE_MIN_EXCLUSIVE": Decimal("0"),
    "DOMINANCE_MAX": Decimal("100"),
    "T10Y2Y_MIN": Decimal("-10"),
    "T10Y2Y_MAX": Decimal("10"),
    "FEDFUNDS_MIN": Decimal("-5"),
    "FEDFUNDS_MAX": Decimal("30"),
    "CHANGE_24H_MIN": Decimal("-100"),
    "CHANGE_24H_MAX": Decimal("1000"),
    "WHALE_ABS_MAX": Decimal("1e12"),
    "WHALE_PAGE_LIMIT": 100,
    "FRED_PAGE_LIMIT": 5,
    "FGI_PAGE_LIMIT": 30,
    "COINGECKO_PROVIDER_STALE_MS": 5_400_000,
    "FGI_LABEL_BOUNDS": (
        (24, "Extreme Fear"),
        (44, "Fear"),
        (55, "Neutral"),
        (75, "Greed"),
        (100, "Extreme Greed"),
    ),
}


def _src(name: str) -> str:
    return (_DATA / f"{name}.py").read_text(encoding="utf-8")


def _inject(source: str, line: str) -> str:
    """Insert ``line`` right after ``from __future__`` so the mutated module still compiles."""
    marker = "from __future__ import annotations\n"
    assert marker in source
    return source.replace(marker, marker + line, 1)


def _roots(tree: ast.AST) -> list[str]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots.append(("." * node.level) + (node.module or "").split(".")[0])
    return roots


# ---------------------------------------------------------------------------
# Pin 1: provider_http.py = stdlib + aiohttp, no logging name
# ---------------------------------------------------------------------------
def http_problems(source: str) -> list[str]:
    tree = ast.parse(source)
    found = [f"import:{r}" for r in _roots(tree) if r not in _STDLIB and r != "aiohttp"]
    for node in ast.walk(tree):
        names = {
            ast.Name: lambda n: n.id,
            ast.Attribute: lambda n: n.attr,
            ast.alias: lambda n: n.name.split(".")[-1],
            ast.arg: lambda n: n.arg,
        }
        for kind, get in names.items():
            if isinstance(node, kind) and get(node) in _LOGGING_NAMES:
                found.append(f"name:{get(node)}")
        segments: list[str] = []
        if isinstance(node, ast.Import):
            segments = [seg for a in node.names for seg in a.name.split(".")]
        elif isinstance(node, ast.ImportFrom):
            segments = (node.module or "").split(".")
            segments += [seg for a in node.names for seg in a.name.split(".")]
        found += [f"import-segment:{seg}" for seg in segments if seg in _LOGGING_NAMES]
    return found


def test_provider_http_is_stdlib_plus_aiohttp_and_log_free() -> None:
    assert http_problems(_src("provider_http")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "import structlog\n",
        "import logging\n",
        "from common.logging import exc_log_fields\n",
        "import requests\n",
        "from data import provider_bands\n",
        "x = logger.info\n",
        "print('k')\n",
        "import logging.handlers\n",
        "from logging import handlers\n",
        "import logging.config as lc\n",
    ],
)
def test_http_pin_catches_mutations(mutation: str) -> None:
    assert http_problems(_inject(_src("provider_http"), mutation)) != []


# ---------------------------------------------------------------------------
# Pin 2: provider_bands.py = stdlib only; band values pinned (single risk sign-off block)
# ---------------------------------------------------------------------------
def bands_problems(source: str) -> list[str]:
    found = [f"import:{r}" for r in _roots(ast.parse(source)) if r not in _STDLIB]
    namespace: dict[str, object] = {"__name__": "bands_under_test"}
    exec(compile(source, "provider_bands.py", "exec"), namespace)  # noqa: S102  # stdlib-only module
    for name, expected in _EXPECTED_BANDS.items():
        got = namespace.get(name)
        if got != expected or type(got) is not type(expected):
            found.append(f"band:{name}")
    return found


def test_provider_bands_is_stdlib_only_with_pinned_values() -> None:
    assert bands_problems(_src("provider_bands")) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('Decimal("1e12")', 'Decimal("1e13")'),
        ("FGI_MAX: Final = 100", "FGI_MAX: Final = 101"),
        ("= 5_400_000", "= 5_400_001"),
        ('Decimal("-10")', 'Decimal("-11")'),
        ("WHALE_PAGE_LIMIT: Final = 100", "WHALE_PAGE_LIMIT: Final = 99"),
        ('(55, "Neutral")', '(54, "Neutral")'),
    ],
)
def test_bands_pin_catches_a_changed_value(old: str, new: str) -> None:
    source = _src("provider_bands")
    assert old in source
    assert bands_problems(source.replace(old, new, 1)) != []


def test_bands_pin_catches_a_third_party_import() -> None:
    for line in ("import aiohttp\n", "from data import provider_http\n", "import structlog\n"):
        assert bands_problems(_inject(_src("provider_bands"), line)) != []


# ---------------------------------------------------------------------------
# Pin 3: nothing under packages/data imports trading or apps (WP23-P2-ARCH-10)
# ---------------------------------------------------------------------------
_FORBIDDEN_ROOTS = frozenset({"trading", "apps", "api"})


def boundary_problems(source: str) -> list[str]:
    return [r for r in _roots(ast.parse(source)) if r in _FORBIDDEN_ROOTS]


def test_no_data_module_imports_trading_or_apps() -> None:
    files = sorted(_DATA.rglob("*.py"))
    assert len(files) > 10  # the walk is not vacuous
    offenders = {str(f.relative_to(_DATA)): boundary_problems(f.read_text("utf-8")) for f in files}
    assert {k: v for k, v in offenders.items() if v} == {}


@pytest.mark.parametrize(
    "line",
    [
        "import trading\n",
        "from trading.models import Signal\n",
        "import apps.api.main\n",
        "from api import deps\n",
    ],
)
def test_boundary_pin_catches_mutations(line: str) -> None:
    assert boundary_problems(_inject(_src("provider_bands"), line)) != []


# ---------------------------------------------------------------------------
# Pin 4: no TLS weakening or ambient proxy anywhere in packages/data
# ---------------------------------------------------------------------------
_SPLAT_CALLEES = frozenset({"get", "request", "ClientSession", "TCPConnector"})
_TLS_NAMES = frozenset(
    {
        "CERT_NONE",
        "CERT_OPTIONAL",
        "VerifyMode",
        "check_hostname",
        "verify_mode",
        "_create_unverified_context",
        "_trust_env",
        "SSLContext",
    }
)
_BANNED_KEYWORDS = frozenset({"ssl_context", "verify_ssl", "fingerprint", "connector"})


def _is_literal(node: ast.expr, value: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def transport_problems(source: str) -> list[str]:
    """Allow-list: TLS verification stays at its default and the environment stays ignored.

    Only a literal ``ssl=True`` and a literal ``trust_env=False`` pass. Everything else that can
    weaken verification, inject a context or connector, or re-enable proxy env is a problem.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [f"import:{a.name}" for a in node.names if a.name.split(".")[0] == "ssl"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ssl":
            found.append("import-from:ssl")
        elif isinstance(node, ast.keyword):
            if node.arg == "ssl" and not _is_literal(node.value, True):
                found.append("keyword:ssl")
            elif node.arg == "trust_env" and not _is_literal(node.value, False):
                found.append("keyword:trust_env")
            elif node.arg in _BANNED_KEYWORDS:  # keyword arguments only: a dict key is not one
                found.append(f"keyword:{node.arg}")
        elif isinstance(node, ast.Call):
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if name in _SPLAT_CALLEES and any(kw.arg is None for kw in node.keywords):
                found.append(f"splat:{name}")
        if isinstance(node, ast.Attribute) and (
            node.attr in _TLS_NAMES or node.attr == "trust_env"
        ):
            found.append(f"attr:{node.attr}")
        elif isinstance(node, ast.Name) and (node.id in _TLS_NAMES or node.id == "ssl"):
            found.append(f"name:{node.id}")
        elif isinstance(node, ast.alias) and (
            node.name in _TLS_NAMES or (node.asname or "") in _TLS_NAMES
        ):
            found.append(f"alias:{node.name}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value in _TLS_NAMES or node.value in {"trust_env", "ssl"})
        ):
            found.append(f"string:{node.value}")  # exact match only: prose never trips it
    return found


def test_no_ssl_off_verify_ssl_or_trust_env_in_data() -> None:
    files = sorted(_DATA.rglob("*.py"))
    assert len(files) > 10  # the walk is not vacuous
    offenders = {str(f.relative_to(_DATA)): transport_problems(f.read_text("utf-8")) for f in files}
    assert {k: v for k, v in offenders.items() if v} == {}


_TRANSPORT_BYPASSES = [
    # the shipped mutations
    "s = aiohttp.ClientSession(trust_env=True)\n",
    "r = s.get(u, ssl=False)\n",
    "r = s.get(u, verify_ssl=True)\n",
    "c = aiohttp.TCPConnector(verify_ssl=False)\n",
    "ctx.verify_mode = ssl.CERT_NONE\n",
    "ctx = ssl._create_unverified_context()\n",
    "s.trust_env = True\n",
    # SEC-P2-03: the 16 bypass forms
    "r = s.get(u, ssl=not True)\n",
    "OFF = False\nr = s.get(u, ssl=OFF)\n",
    "r = s.get(u, ssl=bool(0))\n",
    "r = s.get(u, **{'ssl': False})\n",
    "ctx = ssl.SSLContext()\n",
    "ctx = ssl.SSLContext(ssl.PROTOCOL_TLS)\n",
    "ctx.check_hostname = False\n",
    "ctx.verify_mode = 0\n",
    "x = getattr(ssl, 'CERT_' + 'NONE')\n",
    "x = ssl.VerifyMode(0)\n",
    "from ssl import CERT_NONE as N\n",
    "s = aiohttp.ClientSession(trust_env=1)\n",
    "s = aiohttp.ClientSession(trust_env=not False)\n",
    "s = aiohttp.ClientSession(**{'trust_env': True})\n",
    "setattr(s, 'trust_env', True)\n",
    "s._trust_env = True\n",
    # an injected connector, and ssl=None (only a literal True is allowed)
    "s = aiohttp.ClientSession(connector=c)\n",
    "r = s.get(u, ssl=None)\n",
]
_TRANSPORT_CLEAN = (
    "s = aiohttp.ClientSession(trust_env=False)\n"
    "r = s.get(u, ssl=True)\n"
    'log.debug("e", **fields)\n'
    '"""A docstring that mentions trust_env and ssl in prose."""\n'
)


def test_transport_pin_has_25_bypass_forms() -> None:
    assert len(_TRANSPORT_BYPASSES) == 25


@pytest.mark.parametrize("line", _TRANSPORT_BYPASSES)
def test_transport_pin_catches_mutations(line: str) -> None:
    assert transport_problems(line) != []


def test_transport_pin_clean_control_is_clean() -> None:
    assert transport_problems(_TRANSPORT_CLEAN) == []


# ---------------------------------------------------------------------------
# Pin 5: values come from text, never Decimal(<float>)
# ---------------------------------------------------------------------------
def _is_decimal_callee(func: ast.expr) -> bool:
    return (isinstance(func, ast.Name) and func.id == "Decimal") or (
        isinstance(func, ast.Attribute) and func.attr == "Decimal"
    )


def _float_tainted(node: ast.AST) -> bool:
    is_float_const = isinstance(node, ast.Constant) and isinstance(node.value, float)
    is_float_call = isinstance(node, ast.Call) and getattr(node.func, "id", None) in {
        "float",
        "round",
    }
    return (
        is_float_const
        or is_float_call
        or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div))
    )


def decimal_float_problems(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _is_decimal_callee(node.func):
            operands = [*node.args, *(kw.value for kw in node.keywords)]
            if any(_float_tainted(inner) for op in operands for inner in ast.walk(op)):
                found.append("Decimal(float)")
        if isinstance(node, ast.Attribute) and node.attr == "from_float":
            found.append("from_float")
        if isinstance(node, ast.keyword) and node.arg == "parse_float":
            if not (isinstance(node.value, ast.Name) and node.value.id == "Decimal"):
                found.append("parse_float!=Decimal")
    return found


def test_p2_modules_never_build_a_decimal_from_a_float() -> None:
    offenders = {n: decimal_float_problems(_src(n)) for n in _P2_FILES}
    assert {k: v for k, v in offenders.items() if v} == {}
    assert "parse_float=Decimal" in _src("provider_http")  # the pin has something to protect


@pytest.mark.parametrize(
    "line",
    [
        "x = Decimal(0.1)\n",
        "x = Decimal(float(s))\n",
        "json.loads(b, parse_float=float)\n",
        "x = decimal.Decimal(float(1))\n",
        "x = Decimal.from_float(v)\n",
        "x = Decimal(v * 1.5)\n",
        "x = Decimal(round(f, 2))\n",
        "x = Decimal(a / b)\n",
    ],
)
def test_decimal_pin_catches_mutations(line: str) -> None:
    assert decimal_float_problems(line) != []


def test_decimal_pin_clean_control_is_clean() -> None:
    clean = (
        "x = Decimal(str(v))\n"
        "f = Decimal(str(self._min_value_usd))\n"
        "j = json.loads(b, parse_float=Decimal)\n"
    )
    assert decimal_float_problems(clean) == []


# ---------------------------------------------------------------------------
# Pin 6: the fetch sites never format the caught exception (aiohttp text embeds the request URL)
# ---------------------------------------------------------------------------
_LOG_SITE_FUNCS = frozenset(
    {
        "fetch_validated",
        "_fetch_observation",
        "adopt_validated",
        "_aggregate_validated",
        "_validate_history",
        "_validate_global",
        "_validate_observation",
    }
)
_CLIENT_FILES = ("sentiment", "market_signals", "macro_data", "whale_tracker")


def _scan_log_sites(node: ast.AST, handlers: frozenset[str], found: list[str]) -> None:
    if isinstance(node, ast.ExceptHandler) and node.name:
        handlers = handlers | {node.name}
    if isinstance(node, ast.keyword) and node.arg == "exc_info":
        found.append("exc_info")
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in {"str", "repr"}:
        if any(isinstance(a, ast.Name) and a.id in handlers for a in node.args):
            found.append(f"{node.func.id}(exc)")  # type: ignore[attr-defined]
    if isinstance(node, ast.FormattedValue):
        if isinstance(node.value, ast.Name) and node.value.id in handlers:
            found.append("fstring(exc)")
    for child in ast.iter_child_nodes(node):
        _scan_log_sites(child, handlers, found)


def log_site_problems(source: str, *, whole_module: bool = False) -> list[str]:
    """``str``/``repr``/f-string of a caught exception, or ``exc_info``, at a fetch site."""
    tree = ast.parse(source)
    found: list[str] = []
    if whole_module:
        _scan_log_sites(tree, frozenset(), found)
        return found
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in _LOG_SITE_FUNCS
        ):
            _scan_log_sites(node, frozenset(), found)
    return found


def test_fetch_sites_never_format_the_caught_exception() -> None:
    assert log_site_problems(_src("provider_http"), whole_module=True) == []
    for name in _CLIENT_FILES:
        assert log_site_problems(_src(name)) == [], name
    # the pin has sites to protect in every client
    for name in _CLIENT_FILES:
        assert "fetch_validated" in _src(name) or "_fetch_observation" in _src(name)


@pytest.mark.parametrize(
    ("module", "old", "new", "kind"),
    [
        (
            "sentiment",
            "**exc_log_fields(exc))\n            return Err(",
            "error=str(exc))\n            return Err(",
            "str(exc)",
        ),
        (
            "market_signals",
            "**exc_log_fields(exc))\n            return Err(",
            "exc_info=exc)\n            return Err(",
            "exc_info",
        ),
        (
            "macro_data",
            "**exc_log_fields(exc))\n            return Err(",
            'error=f"{exc}")\n            return Err(',
            "fstring(exc)",
        ),
        (
            "whale_tracker",
            "**exc_log_fields(exc))\n            return Err(",
            "error=repr(exc))\n            return Err(",
            "repr(exc)",
        ),
    ],
)
def test_log_site_pin_catches_mutated_sources(module: str, old: str, new: str, kind: str) -> None:
    source = _src(module)
    assert source.count(old) == 1, module
    assert kind in log_site_problems(source.replace(old, new))


def test_log_site_pin_catches_a_mutated_provider_http() -> None:
    source = _inject(_src("provider_http"), "")
    mutated = source.replace(
        'return Err("timeout", error_type=type(exc).__name__)',
        'return Err("timeout", error_type=str(exc))',
    )
    assert mutated != source and "str(exc)" in log_site_problems(mutated, whole_module=True)


# ===========================================================================
# P3 (T-14, T-14b, N-2; append-only: the P2 pins and _EXPECTED_BANDS above are untouched)
# ===========================================================================
import types  # noqa: E402

_ROOT = _DATA.parents[1]
_P3_CLOCK_IO_NAMES = frozenset(
    {
        "now",
        "time",
        "monotonic",
        "getenv",
        "environ",
        "open",
        "logging",
        "structlog",
        "asyncio",
        "aiohttp",
        "utcnow",
        "today",
        "print",
        "exec",
        "eval",
        "__import__",
    }
)
# SEC-P3-05 / CR-P3-03: exact import allow-lists (full name, level 0); the policy has no Decimal
_P3_IMPORTS = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "decimal",
        "types",
        "typing",
        "data.point_in_time",
    }
)
# P3-B1 / CF-2401-30: the converters read none of the never-served fields; CF-2401-31: no
# CoinGecko `updated_at` future-reject; AD-13: never `realtime_start`.
_P3_NEVER_READ = (
    *("market_cap_change_24h", "total_volume_change_24h", "fed_funds_rate", "fedfunds"),
    *("realtime_start", "updated_at"),
)
_BOOK_API = {"view", "heartbeat", "publish", "snapshot", "freshness", "reset"}
_REGISTRY_API = {"started", "stopped", "counts", "reset"}


def _const(tree: ast.AST, name: str) -> object:
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value:
                return ast.literal_eval(node.value)
    return None


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _public(tree: ast.AST, cls: str) -> set[str]:
    """Public methods (sync or async), nested classes and class-level assignment targets such as
    ``expect = None`` or ``expect, other = None, None`` (CF-2401-61: unpack targets and nested
    ``ClassDef`` count; method bodies are never entered, so locals are not names)."""
    names: set[str] = set()
    for f in _class(tree, cls).body:
        if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(f.name)
        elif isinstance(f, ast.Assign | ast.AnnAssign):
            targets = [f.target] if isinstance(f, ast.AnnAssign) else f.targets
            names |= {
                n.id
                for t in targets
                for n in ast.walk(t)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
    return {n for n in names if not n.startswith("_")}


def context_problems(source: str, *, leaf: bool) -> list[str]:
    """Imports, clock/IO/logging names and never-served tokens (both modules); API (the book)."""
    tree = ast.parse(source)
    found: list[str] = []
    allowed = _P3_IMPORTS if leaf else _P3_IMPORTS - {"decimal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"import:{a.name}" for a in node.names if a.name not in allowed]
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "") not in allowed:
                found.append(f"import-from:{node.module}")
        names: list[str] = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.arg | ast.keyword):
            names = [(node.arg or "")]
        elif isinstance(node, ast.alias):
            names = node.name.split(".")
        elif isinstance(node, ast.ImportFrom | ast.Import):
            names = (getattr(node, "module", None) or "").split(".")
        found += [f"name:{n}" for n in names if n in _P3_CLOCK_IO_NAMES]
        texts = [*names, *([node.value] if isinstance(node, ast.Constant) else [])]
        texts += [node.name] if isinstance(node, ast.FunctionDef | ast.ClassDef) else []
        found += [
            f"never-read:{tok}"
            for t in texts
            if isinstance(t, str)
            for tok in _P3_NEVER_READ
            if tok in t.lower()
        ]
    if leaf:
        if _public(tree, "LiveContextBook") != _BOOK_API:
            found.append("book-api")
        if _public(tree, "ConsumerRegistry") != _REGISTRY_API:
            found.append("registry-api")
        slots = ast.literal_eval(
            next(
                n.value
                for n in _class(tree, "LiveContextBook").body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__slots__"
            )
        )
        expected = {
            "CLOCK_REGRESSION_TOLERANCE_MS": 300_000,
            "BUCKET_MS": 3_600_000,
            "CONSUMER_MODES": ("paper", "live"),
        }
        found += [f"const:{k}" for k, v in expected.items() if _const(tree, k) != v]
        if set(slots) != {"_high_water_ms", "_series", "regressed_reads"}:
            found.append("slots")
        finals = {  # LIVE_BOOK / LIVE_CONSUMERS must be annotated ``Final``
            n.target.id
            for n in tree.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
            and getattr(n.annotation, "id", "") == "Final"
        }  # fmt: skip
        found += [f"final:{k}" for k in ("LIVE_BOOK", "LIVE_CONSUMERS") if k not in finals]
    return found


def test_live_context_is_a_pure_leaf_with_the_pinned_api() -> None:
    assert context_problems(_src("live_context"), leaf=True) == []


def _uncalled(stmt: str) -> str:  # a probe wrapped so that nothing runs even if a pin regressed
    return "def _probe() -> None:\n    " + stmt.replace("\n", "\n    ") + "\n"


_LEAF_PROBES = (  # CR-P3-03 / SEC-P3-05: all of these passed the r1 deny-list
    "import random|import socket|import subprocess|import urllib.request|import http.client|"
    "datetime.utcnow()|datetime.today()|print(1)|exec('1')|eval('1')|__import__('os')|"
    "from .data.point_in_time import Observation"
).split("|")


@pytest.mark.parametrize(
    "mutation",
    [
        "import logging\n",
        "import asyncio\n",
        "import time\n",
        "import os\nx = os.environ\n",
        "import aiohttp\n",
        "from data import provider_bands\n",
        "from data.live_context_policy import FLOOR_S\n",
        "from trading.mtf_signals import LIVE_BOOK\n",
        "x = clock.now()\n",
        "x = f(monotonic)\n",
        "f = open\n",
        "x = snap.fed_funds_rate\n",
        "x = snap.market_cap_change_24h\n",
        "d = {'total_volume_change_24h': 1}\n",
        "def f(fedfunds): return fedfunds\n",
        "x = row.realtime_start\n",
        "x = body.updated_at\n",
        *(_uncalled(c) for c in _LEAF_PROBES),
    ],
)
def test_leaf_pin_catches_mutations(mutation: str) -> None:
    assert context_problems(_inject(_src("live_context"), mutation), leaf=True) != []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "    def snapshot(self)",
            "    def expect(self, ids: object) -> None: ...\n\n    def snapshot(self)",
        ),
        ("    def heartbeat(self", "    def _heartbeat(self"),
        ("    def counts(self)", "    def count(self)"),
        (
            "    def stopped(self",
            "    def stopped(self, token: int, extra: int = 0) -> None: ...\n\n    def kill(self",
        ),
        (
            "CLOCK_REGRESSION_TOLERANCE_MS: Final = 300_000",
            "CLOCK_REGRESSION_TOLERANCE_MS: Final = 300_001",
        ),
        ('("paper", "live")', '("paper", "live", "backtest")'),
        ("BUCKET_MS: Final = 3_600_000", "BUCKET_MS: Final = 3_600_001"),
        (
            '__slots__ = ("_high_water_ms", "_series", "regressed_reads")',
            '__slots__ = ("_high_water_ms", "_series")',
        ),
        (  # CR-P3-04b: extra slot, class attribute, async method, un-Final singleton
            '"_series", "regressed_reads")',
            '"_series", "regressed_reads", "expected")',
        ),
        ("    def snapshot(self)", "    expect = None\n\n    def snapshot(self)"),
        (
            "    def snapshot(self)",
            "    expect = staticmethod(lambda ids: None)\n\n    def snapshot(self)",
        ),
        (
            "    def snapshot(self)",
            "    async def poll(self) -> None: ...\n\n    def snapshot(self)",
        ),
        ("    def snapshot(self)", "    expect: int = 0\n\n    def snapshot(self)"),
        ("    def snapshot(self)", "    expect, other = None, None\n\n    def snapshot(self)"),
        ("    def snapshot(self)", "    class expect: ...\n\n    def snapshot(self)"),
        ("LIVE_BOOK: Final = ", "LIVE_BOOK = "),
        ("LIVE_BOOK: Final = ", "LIVE_BOOK: object = "),
        ("LIVE_CONSUMERS: Final = ", "LIVE_CONSUMERS = "),
    ],
)
def test_leaf_pin_catches_an_api_or_constant_mutation(old: str, new: str) -> None:
    source = _src("live_context")
    assert source.count(old) == 1, old
    assert context_problems(source.replace(old, new), leaf=True) != []


def _load(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102  # pure constants module
    finally:
        del sys.modules[name]
    return module


_EXPECTED_POLICY = {
    "FLOOR_S": 60, "CYCLE_BUDGET_S": 30, "ATTEMPT_BOUND_S": 10, "MAX_RETRIES": 2,
    "BACKOFF_CAP_S": 60, "FAILED_CYCLES_BEFORE_DOUBLING": 3, "RETRY_AFTER_MAX_S": 3_600,
    "AUTH_FAILED_WAIT_S": 21_600, "JITTER_PCT": 10, "BUCKET_MS": 3_600_000,
    "BUCKET_JITTER_MAX_S": 300, "BUDGET_WINDOW_MS": 3_600_000,
}  # fmt: skip
_EXPECTED_CADENCE = {  # series -> (interval_s, budget_per_hour, requests_per_cycle, bucketed)
    "alternative_me:fgi": (3_600, 4, 1, False),
    "coingecko:btc_dominance": (600, 12, 1, True),
    "fred:T10Y2Y": (21_600, 6, 2, False),
    "whale:net_flow_1h": (900, 8, 1, True),
}


def policy_problems(source: str) -> list[str]:
    """Same import/name rules as the book, plus the pinned cadence block and its invariants."""
    found = context_problems(source, leaf=False)
    if found:  # never execute a source that already failed the leaf rules (SEC-P3-05)
        return found
    mod = _load("policy_under_test", source)
    found += [f"const:{k}" for k, v in _EXPECTED_POLICY.items() if getattr(mod, k, None) != v]
    if not 0 < mod.HEARTBEAT_TICK_S <= 60:
        found.append("heartbeat")
    table = {
        s: (c.interval_s, c.budget_per_hour, c.requests_per_cycle, c.bucketed)
        for s, c in mod.CADENCE.items()
    }
    if table != _EXPECTED_CADENCE:
        found.append("cadence")
    from data.point_in_time import SERIES_RULES  # the caps the invariants are computed from

    if set(mod.CADENCE) != set(SERIES_RULES):
        found.append("series")
    for sid, row in mod.CADENCE.items():
        if sid not in SERIES_RULES:
            continue
        if not mod.FLOOR_S <= row.interval_s <= SERIES_RULES[sid].max_age_ms // 4000:
            found.append(f"interval:{sid}")  # >= the 60 s floor and <= cap/4
        if row.budget_per_hour < row.requests_per_cycle * (mod.MAX_RETRIES + 1):
            found.append(f"budget:{sid}")
    return found


def test_live_context_policy_is_pure_and_its_cadence_block_is_pinned() -> None:
    assert policy_problems(_src("live_context_policy")) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("FLOOR_S: Final = 60", "FLOOR_S: Final = 59"),
        ("CYCLE_BUDGET_S: Final = 30", "CYCLE_BUDGET_S: Final = 31"),
        ("ATTEMPT_BOUND_S: Final = 10", "ATTEMPT_BOUND_S: Final = 11"),
        ("HEARTBEAT_TICK_S: Final = 60", "HEARTBEAT_TICK_S: Final = 61"),
        ("BUCKET_JITTER_MAX_S: Final = 300", "BUCKET_JITTER_MAX_S: Final = 301"),
        ("BUCKET_MS: Final = 3_600_000", "BUCKET_MS: Final = 3_600_001"),
        ("Cadence(3_600, 4)", "Cadence(30, 4)"),  # below the 60 s floor
        ("Cadence(600, 12, bucketed=True)", "Cadence(6_000, 12, bucketed=True)"),  # > cap/4
        ("Cadence(900, 8, bucketed=True)", "Cadence(900, 2, bucketed=True)"),  # < one full cycle
        ("Cadence(21_600, 6, requests_per_cycle=2)", "Cadence(21_600, 5, requests_per_cycle=2)"),
        ('"fred:T10Y2Y": Cadence', '"fred:T10Y2Z": Cadence'),
        ("import logging\n", "import logging\n"),
    ],
)
def test_policy_pin_catches_mutations(old: str, new: str) -> None:
    source = _src("live_context_policy")
    if old == "import logging\n":  # an import mutation instead of a literal one
        source = _inject(source, "import logging\n")
    else:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    assert policy_problems(source) != []


@pytest.mark.parametrize("probe", ["import random", "datetime.utcnow()", "import decimal"])
def test_policy_pin_catches_leaf_probes_and_never_executes_the_source(
    probe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "_load", lambda *a: pytest.fail("executed"))  # SEC-P3-05
    assert policy_problems(_inject(_src("live_context_policy"), _uncalled(probe))) != []


def test_policy_pin_catches_an_import_of_the_book() -> None:
    mutated = _inject(_src("live_context_policy"), "from data.live_context import BUCKET_MS as B\n")
    assert policy_problems(mutated) != []


def bucket_problems(book_source: str, policy_source: str) -> list[str]:
    """N-2: the two modules may not import each other, so the 1 h literal is compared here."""
    values = [_const(ast.parse(s), "BUCKET_MS") for s in (book_source, policy_source)]
    return [] if values[0] == values[1] == 3_600_000 else [f"bucket:{values}"]


def test_the_two_bucket_literals_are_equal() -> None:
    from data import live_context, live_context_policy

    assert live_context.BUCKET_MS == live_context_policy.BUCKET_MS == 3_600_000
    book, policy = _src("live_context"), _src("live_context_policy")
    assert bucket_problems(book, policy) == []
    mutated = "BUCKET_MS: Final = 3_600_001"
    assert bucket_problems(book.replace("BUCKET_MS: Final = 3_600_000", mutated), policy) != []
    assert bucket_problems(book, policy.replace("BUCKET_MS: Final = 3_600_000", mutated)) != []
    assert bucket_problems(book, policy.replace("BUCKET_MS: Final", "BUCKET_MS2: Final")) != []


def test_new_modules_are_walked_by_the_p2_pins_and_need_no_waiver() -> None:
    """P3-B3: pins 3, 4 and 5 cover both modules with no waiver (security confirms in review)."""
    walked = {f.stem for f in _DATA.rglob("*.py")}
    for name in ("live_context", "live_context_policy"):
        assert name in walked
        assert boundary_problems(_src(name)) == []
        assert transport_problems(_src(name)) == []
        assert decimal_float_problems(_src(name)) == []


# ---------------------------------------------------------------------------
# T-14b: writer pins for LIVE_BOOK / LIVE_CONSUMERS over packages/ and apps/ (SY-01-21f, ER2-02)
# ---------------------------------------------------------------------------
_LEAF = "packages/data/live_context.py"
_MTF = "packages/trading/mtf_signals.py"
_ENGINE = "packages/trading/strategy_engine.py"
_POLLER = "apps/api/services/context_poller.py"
_RECORDER = "apps/api/services/context_recorder.py"
_ALERTS = "apps/api/services/context_alerts.py"
_HEALTH = "apps/api/routers/health.py"
_WRITER_RULES: dict[tuple[str, str], frozenset[str]] = {
    ("LIVE_BOOK", "publish"): frozenset({_POLLER, _RECORDER}),
    ("LIVE_BOOK", "heartbeat"): frozenset({_POLLER}),
    ("LIVE_BOOK", "view"): frozenset({_MTF}),  # and only inside LiveSignals.values_at
    ("LIVE_BOOK", "freshness"): frozenset({_POLLER, _HEALTH}),
    ("LIVE_BOOK", "snapshot"): frozenset({_POLLER, _HEALTH}),
    ("LIVE_BOOK", "regressed_reads"): frozenset({_POLLER}),
    ("LIVE_BOOK", "reset"): frozenset(),  # tests/ only
    ("LIVE_CONSUMERS", "started"): frozenset({_ENGINE}),
    ("LIVE_CONSUMERS", "stopped"): frozenset({_ENGINE}),
    ("LIVE_CONSUMERS", "counts"): frozenset({_POLLER, _ALERTS}),
    ("LIVE_CONSUMERS", "reset"): frozenset(),  # tests/ only
}
_SINGLETONS = frozenset(s for s, _ in _WRITER_RULES)
_CLASSES = frozenset({"LiveContextBook", "ConsumerRegistry"})
_PROTECTED = _SINGLETONS | _CLASSES | {"data.live_context"}  # string constants (SEC-P3-02)
_VIEW_SCOPE = ("LiveSignals", "values_at")
_MODULES = {"data.live_context": "live_context", "trading.mtf_signals": "mtf_signals"}


def _singleton(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _SINGLETONS:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _SINGLETONS:
        return node.attr
    return None


def _modref(node: ast.AST, names: set[str]) -> bool:
    """A name bound to the live_context / mtf_signals module object, or ``<x>.live_context``."""
    return (isinstance(node, ast.Name) and node.id in names) or (
        isinstance(node, ast.Attribute) and node.attr in _MODULES.values()
    )


def writer_problems(source: str, rel: str) -> list[str]:
    """Default-deny: any reference to a singleton outside the listed files, any method outside
    its row, any non-Load use of a singleton attribute, an aliased or star import, a bare
    (non-attribute) use of a singleton or of the two modules, a class reference, or a protected
    name given as a string constant, is a problem."""
    if rel == _LEAF:
        return []
    tree = ast.parse(source)
    found: list[str] = []
    mods: set[str] = set()  # names bound to the two module objects
    exported: set[int] = set()  # the MTF re-export list is the only place the names may be strings
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.asname for a in n.names if a.name in _MODULES and a.asname}
        elif isinstance(n, ast.ImportFrom):
            if n.module in ("data", "trading"):
                mods |= {a.asname or a.name for a in n.names if a.name in _MODULES.values()}
            found += [f"import:{a.name}" for a in n.names if a.name in _CLASSES]
        elif rel == _MTF and isinstance(n, ast.Assign):
            if any(getattr(t, "id", "") == "__all__" for t in n.targets):
                exported |= {id(c) for c in ast.walk(n.value)}
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}  # SEC-P3-10
    stack: list[tuple[ast.AST, tuple[str, ...], bool]] = [(tree, (), False)]
    while stack:
        node, scope, is_base = stack.pop()
        if isinstance(node, ast.Attribute) and (base := _singleton(node.value)):
            allowed = _WRITER_RULES.get((base, node.attr), frozenset())
            if rel not in allowed:
                found.append(f"{base}.{node.attr}")
            elif (base, node.attr) == ("LIVE_BOOK", "view") and scope != _VIEW_SCOPE:
                found.append("LIVE_BOOK.view:scope")
            if not isinstance(node.ctx, ast.Load):  # SEC-P3-03: no Store / AugAssign / Del
                found.append(f"store:{base}.{node.attr}")
            if is_base or (node.attr != "regressed_reads" and id(node) not in called):
                found.append(f"bound:{base}.{node.attr}")  # SEC-P3-10: __self__ / passed out
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "__dict__"
            and _modref(node.value, mods)
        ):
            found.append("dunder-dict")
        if _modref(node, mods) and not is_base:
            found.append("bare-module")
        if (getattr(node, "id", None) or getattr(node, "attr", None)) in _CLASSES:
            found.append("class-ref")
        if isinstance(node, ast.Constant) and node.value in _PROTECTED and id(node) not in exported:
            found.append(f"string:{node.value}")
        if rel == _MTF and getattr(node, "id", "") == "__all__" and isinstance(node.ctx, ast.Load):
            found.append("all-load")  # SEC-P3-12 (N8): the export list is never read back
        if _singleton(node) and not is_base:  # SEC-P3-01: no exemption for the anchor file
            found.append(f"bare:{_singleton(node)}")
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names) and (node.module or "").endswith(
                ("live_context", "mtf_signals")
            ):
                found.append("star-import")
            for a in node.names:
                files = {
                    _LEAF,
                    _MTF,
                    *(f for (s, _), fs in _WRITER_RULES.items() if s == a.name for f in fs),
                }
                if a.name in _SINGLETONS and rel not in files:
                    found.append(f"import:{a.name}")
                if a.name in _SINGLETONS and a.asname not in (None, a.name):
                    found.append(f"import-alias:{a.name}")
        inner = (
            (*scope, node.name)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            else scope
        )
        stack += [
            (c, inner, isinstance(node, ast.Attribute) and c is node.value)
            for c in ast.iter_child_nodes(node)
        ]
    return found


def test_only_the_listed_files_touch_the_live_singletons() -> None:
    files = [*sorted((_ROOT / "packages").rglob("*.py")), *sorted((_ROOT / "apps").rglob("*.py"))]
    assert len(files) > 100 and _ROOT / _LEAF in files  # the walk is not vacuous
    offenders = {}
    for f in files:
        rel = f.relative_to(_ROOT).as_posix()
        if problems := writer_problems(f.read_text("utf-8"), rel):
            offenders[rel] = problems
    assert offenders == {}


_IMP = "from data.live_context import LIVE_BOOK\n"  # one name per control: imports are default-deny
_IMC = "from data.live_context import LIVE_CONSUMERS\n"
_IMK = "from data.live_context import LiveContextBook\n"
_RUNS = "apps/api/routers/runs.py"
_VIEW_OK = "class LiveSignals:\n    def values_at(self):\n        return LIVE_BOOK.view(1)\n"


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        (
            _POLLER,
            _IMP
            + "LIVE_BOOK.publish('s', [], 0)\nLIVE_BOOK.heartbeat(1)\nLIVE_BOOK.freshness(1)\n",
        ),
        (_POLLER, _IMP + _IMC + "x = LIVE_BOOK.regressed_reads\nLIVE_CONSUMERS.counts()\n"),
        (_RECORDER, _IMP + "LIVE_BOOK.publish('s', [], 0)\n"),
        (_HEALTH, _IMP + "LIVE_BOOK.freshness(1)\nLIVE_BOOK.snapshot()\n"),
        (_ALERTS, _IMC + "LIVE_CONSUMERS.counts()\n"),
        (_ENGINE, _IMC + "t = LIVE_CONSUMERS.started('live')\nLIVE_CONSUMERS.stopped(t)\n"),
        (_MTF, _IMP + _VIEW_OK),
        (_POLLER, "import data.live_context as lc\nlc.LIVE_BOOK.publish('s', [], 0)\n"),
        (  # the P4a-shaped re-export: import alias nodes and ``__all__`` only
            _MTF,
            "from data.live_context import LIVE_BOOK, LIVE_CONSUMERS as LIVE_CONSUMERS\n"
            "__all__ = ['LIVE_CONSUMERS', 'LIVE_SIGNALS']\n" + _VIEW_OK,
        ),
        (  # the engine takes the registry from mtf_signals (module alias, parameter of that name)
            _ENGINE,
            "import trading.mtf_signals as mtf_signals_module\n"
            "from trading.mtf_signals import LIVE_CONSUMERS\n"
            "class E:\n    def __init__(self, mtf_signals=None):\n        self.m = mtf_signals\n"
            "        self.s = mtf_signals_module.STAGE\n"
            "    def run(self):\n        t = LIVE_CONSUMERS.started('live')\n"
            "        LIVE_CONSUMERS.stopped(t)\n",
        ),
    ],
)
def test_writer_pin_clean_controls(rel: str, source: str) -> None:
    assert writer_problems(source, rel) == []


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        (_RUNS, _IMP),
        (_RUNS, "import data.live_context as lc\nlc.LIVE_BOOK.publish('s', [], 0)\n"),
        (_HEALTH, _IMP + "LIVE_BOOK.view(1)\n"),
        (_HEALTH, _IMP + "LIVE_BOOK.heartbeat(1)\n"),
        (_HEALTH, _IMP + "LIVE_BOOK.publish('s', [], 0)\n"),
        (
            _HEALTH,
            _IMC + "LIVE_CONSUMERS.counts()\n",
        ),  # ER2-02: health references no LIVE_CONSUMERS name
        (_POLLER, _IMP + "LIVE_BOOK.view(1)\n"),
        (_POLLER, _IMP + "LIVE_BOOK.reset()\n"),
        (_POLLER, _IMC + "LIVE_CONSUMERS.reset()\n"),
        (_POLLER, _IMC + "LIVE_CONSUMERS.started('live')\n"),
        (_ENGINE, _IMC + "LIVE_CONSUMERS.counts()\n"),
        (_ENGINE, _IMP + "LIVE_BOOK.publish('s', [], 0)\n"),
        (_MTF, _IMP + "def other():\n    return LIVE_BOOK.view(1)\n"),
        (
            _MTF,
            _IMP + "class LiveSignals:\n    def other(self):\n        return LIVE_BOOK.view(1)\n",
        ),
        (_MTF, _IMP + "LIVE_BOOK.heartbeat(1)\n"),
        (_POLLER, _IMP + "b = LIVE_BOOK\nb.publish('s', [], 0)\n"),
        (_POLLER, _IMP + "getattr(LIVE_BOOK, 'reset')()\n"),
        (_POLLER, "from data.live_context import LIVE_BOOK as B\nB.publish('s', [], 0)\n"),
        (_POLLER, "from data.live_context import *\n"),
        (_POLLER, _IMP + "LIVE_BOOK.expect(['x'])\n"),
        (_MTF, _IMP + "_b = LIVE_BOOK\n_b.publish('s', [], 0)\n"),
        (_MTF, _IMP + "getattr(LIVE_BOOK, 'reset')()\n"),
        (_MTF, _IMP + "def other():\n    b = LIVE_BOOK\n    return b.view(1)\n"),
        (_MTF, _IMP + "def f(**k): k['b'].heartbeat(1)\nf(**{'b': LIVE_BOOK})\n"),
        (_MTF, _IMP + "setattr(LIVE_BOOK, 'regressed_reads', 0)\n"),
        (_MTF, _IMC + "_c = LIVE_CONSUMERS\n_c.reset()\n"),
        (_MTF, _IMP + "LIVE_BOOK._series = {}\n"),
        (_POLLER, _IMP + "LIVE_BOOK.regressed_reads = 0\n"),
        (_POLLER, _IMP + "LIVE_BOOK.regressed_reads += 1\n"),
        (_POLLER, _IMP + "del LIVE_BOOK.regressed_reads\n"),
        (_RUNS, "import data.live_context as lc\ngetattr(lc, 'LIVE_BOOK').reset()\n"),
        (_RUNS, "import data.live_context as lc\nvars(lc)['LIVE_BOOK'].reset()\n"),
        (_RUNS, "import data.live_context as lc\nlc.__dict__['x'].started('live')\n"),
        (_RUNS, "import data.live_context as lc\nuse(lc)\n"),
        (_RUNS, _IMK),  # a module object may only be a base
        (_RUNS, "import importlib\nimportlib.import_module('data.live_context')\n"),
        (_RUNS, "import sys\ngetattr(sys.modules['data.live_context'], 'LIVE_BOOK').reset()\n"),
        (_ENGINE, "import trading.mtf_signals as m\ngetattr(m, 'LIVE_BOOK').publish('s', [], 0)\n"),
        (
            _RUNS,
            _IMK + "LiveContextBook.view = lambda s, n: {}\n",
        ),  # K1: the classes are denied too
        (_RUNS, "from data import live_context as lc\nlc.ConsumerRegistry.counts = lambda s: {}\n"),
        (_MTF, _IMP + "def f():\n    " + _VIEW_OK.replace("\n", "\n    ")),  # C11: full scope path
        (_HEALTH, _IMP + "LIVE_BOOK.snapshot.__self__.publish('s', [], 0)\n"),  # N1
        (_POLLER, _IMP + "f = LIVE_BOOK.publish\nf.__self__._high_water_ms = 0\n"),  # N3
        (_RECORDER, _IMP + "sink(LIVE_BOOK.publish)\n"),  # N7: a passed-out bound method
        (_MTF, _IMP + "__all__ = ['LIVE_BOOK']\nglobals()[__all__[0]].reset()\n"),  # N8
    ],
)
def test_writer_pin_catches_mutations(rel: str, source: str) -> None:
    assert writer_problems(source, rel) != []


# T-14b: the root conftest keeps the autouse isolation fixture (reset both, before and after).
def conftest_problems(source: str) -> list[str]:
    fixtures = [
        f
        for f in ast.walk(ast.parse(source))
        if isinstance(f, ast.FunctionDef)
        and any("autouse" in ast.dump(d) and "fixture" in ast.dump(d) for d in f.decorator_list)
    ]
    for fixture in fixtures:
        body = fixture.body
        marks = [
            i
            for i, s in enumerate(body)
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield)
        ]
        resets = [
            (i, getattr(s.value.func.value, "id", ""))  # type: ignore[attr-defined]
            for i, s in enumerate(body)
            if isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "attr", "") == "reset"
        ]
        if len(marks) == 1 and sorted(
            ("before" if i < marks[0] else "after", n) for i, n in resets
        ) == sorted((w, n) for w in ("before", "after") for n in ("LIVE_BOOK", "LIVE_CONSUMERS")):
            return []
    return ["no autouse fixture resets LIVE_BOOK and LIVE_CONSUMERS before and after the test"]


def test_the_root_conftest_resets_both_singletons_around_every_test() -> None:
    source = (_ROOT / "tests" / "conftest.py").read_text("utf-8")
    assert conftest_problems(source) == []
    tail = "    yield\n    LIVE_BOOK.reset()\n    LIVE_CONSUMERS.reset()\n"
    assert source.count(tail) == 1
    mutations = [
        source.replace(tail, "    yield\n"),  # no reset after
        source.replace(tail, "    yield\n    LIVE_BOOK.reset()\n"),  # consumers not reset after
        source.replace("    LIVE_CONSUMERS.reset()\n    yield", "    yield"),  # not reset before
        source.replace("def _cf2401_live_context_isolated", "def _renamed").replace(
            "@pytest.fixture(autouse=True)\ndef _renamed", "@pytest.fixture\ndef _renamed"
        ),  # not autouse
    ]
    assert all(conftest_problems(m) != [] for m in mutations)


# --- P4a (T-14 (4), T-19, T-21, T-22 ARC-14, CF-2401-57/61/67): engine pins -- append-only ---
_ENGINE_FILE = _ROOT / _ENGINE
_STAGE_VALUE = (
    "mtf_signals_module.LIVE_CONTEXT_STAGE if self._mtf_signals is LIVE_SIGNALS else None"
)
_UNTOUCHED = frozenset(
    {"_stop_event", "_auto_stop_reason", "evaluate", "_circuit_breaker", "_risk_manager"}
)  # T-19: staleness never reaches the stop paths or the breaker inputs
_RELEASE = "if consumer_token is not None:\n    LIVE_CONSUMERS.stopped(consumer_token)"


def _engine_method(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    classes = (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    klass = next(n for n in classes if n.name == "StrategyEngine")
    fns = (n for n in klass.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))
    return next(n for n in fns if n.name == name)


def _calls_of(tree: ast.AST, dotted: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]


def engine_stage_problems(source: str) -> list[str]:
    """SY-01-21h / T-21 / T-19 / CF-2401-57: the stage fence and the consumer registration."""
    tree = ast.parse(source)
    init, build = _engine_method(tree, "__init__"), _engine_method(tree, "_build_mtf_context")
    reg = _engine_method(tree, "_register_live_consumer")
    loop = _engine_method(tree, "run_live_loop")
    out: list[str] = []
    stores = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "_live_stage" and isinstance(n.ctx, ast.Store)
    ]  # fmt: skip
    values = [
        ast.unparse(n.value) for n in ast.walk(init)
        if isinstance(n, ast.AnnAssign) and n.value and ast.unparse(n.target) == "self._live_stage"
    ]  # fmt: skip
    if len(stores) != 1 or values != [_STAGE_VALUE]:
        out.append("_live_stage is not set once, in __init__, behind the LIVE_SIGNALS identity")
    read = {n.id for n in ast.walk(build) if isinstance(n, ast.Name)}
    if read & {"LIVE_CONTEXT_STAGE", "mtf_signals_module"}:
        out.append("_build_mtf_context reads the stage constant instead of self._live_stage")
    guards = [
        n for n in ast.walk(build)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "self._live_stage == 'neutral_shadow'"
    ]  # fmt: skip
    names = [
        n for n in ast.walk(build) if isinstance(n, ast.Constant) and n.value == "neutral_shadow"
    ]
    if not (len(guards) == len(names) == 1 == len(_calls_of(guards[0], "SignalValues"))
            == len(_calls_of(build, "SignalValues"))):  # fmt: skip
        out.append("the neutral value is not built only under the self._live_stage guard")
    touched = {n.attr for m in (build, reg) for n in ast.walk(m) if isinstance(n, ast.Attribute)}
    out += [f"staleness path touches {n}" for n in sorted(touched & _UNTOUCHED)]  # T-19
    starts = _calls_of(tree, "LIVE_CONSUMERS.started")
    stops = _calls_of(tree, "LIVE_CONSUMERS.stopped")
    tries = [t for t in ast.walk(reg) if isinstance(t, ast.Try)]
    in_body = len(tries) == 1 and any(
        n is c for s in tries[0].body for n in ast.walk(s) for c in starts
    )
    handlers = tries[0].handlers if len(tries) == 1 else []
    swallowed = (
        len(handlers) == 1
        and ast.unparse(handlers[0].type or ast.Pass()) == "Exception"
        and not any(isinstance(n, ast.Raise) for n in ast.walk(handlers[0]))
    )
    if len(starts) != 1 or len(stops) != 1 or not in_body or not swallowed:
        out.append("LIVE_CONSUMERS.started/stopped: not one guarded start and one stop")
    loops = [n for n in loop.body if isinstance(n, ast.Try)]
    if len(loops) != 1:
        return [*out, "run_live_loop has no single try/finally"]
    if ast.unparse(loops[0].body[0]) != "consumer_token = self._register_live_consumer()":
        out.append("the registration is not the first statement of the loop's try")
    final = loops[0].finalbody  # CF-2401-57: exactly try: await stop-intrabar / finally: release
    inner = final[0] if len(final) == 1 and isinstance(final[0], ast.Try) else None
    if not (
        inner is not None
        and not inner.handlers
        and not inner.orelse
        and [ast.unparse(n) for n in inner.body] == ["await self._stop_intrabar_task()"]
        and [ast.unparse(n) for n in inner.finalbody] == [_RELEASE]
    ):
        out.append("stopped is not in a finally nested after the intra-bar stop")
    before = [ast.unparse(n) for n in loop.body[: loop.body.index(loops[0])]]
    if "self._start_intrabar_task()" not in before:
        out.append("the intra-bar task is not started before the registration")
    return out


def test_the_engine_stage_fence_and_registration_are_pinned() -> None:
    assert engine_stage_problems(_ENGINE_FILE.read_text("utf-8")) == []


_AWAIT, _IF = "await self._stop_intrabar_task()", "if consumer_token is not None:"
_REL = "LIVE_CONSUMERS.stopped(consumer_token)"  # fragments of the release in ``run_live_loop``
_NESTED_STOP = (
    f"            try:\n                {_AWAIT}\n            finally:\n"
    f"                {_IF}\n                    {_REL}\n"
)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("mtf_signals_module.LIVE_CONTEXT_STAGE if self._mtf_signals is LIVE_SIGNALS else None",
         "mtf_signals_module.LIVE_CONTEXT_STAGE"),  # the fence removed
        ("mtf_signals_module.LIVE_CONTEXT_STAGE if self._mtf_signals is LIVE_SIGNALS else None",
         "'fresh' if self._mtf_signals is LIVE_SIGNALS else None"),
        ("            if self._live_stage == \"neutral_shadow\":\n",
         "            if True:\n"),  # the guard is not the stage
        ("        values = self._mtf_signals.values_at(decision_at)\n",
         "        values = self._mtf_signals.values_at(decision_at)\n"
         "        if mtf_signals_module.LIVE_CONTEXT_STAGE == 'x':\n            pass\n"),
        ("        values = self._mtf_signals.values_at(decision_at)\n",
         "        values = self._mtf_signals.values_at(decision_at)\n"
         "        self._stop_event.set()\n"),
        ("            consumer_token = self._register_live_consumer()  # before the first bar\n",
         "            pass\n            consumer_token = self._register_live_consumer()\n"),
        ("                if consumer_token is not None:\n"
         "                    LIVE_CONSUMERS.stopped(consumer_token)\n",
         "                pass\n"),
        ("            return LIVE_CONSUMERS.started(self._run_mode.value)\n"
         "        except Exception as exc:\n",
         "            return LIVE_CONSUMERS.started(self._run_mode.value)\n"
         "        except Exception as exc:\n"
         "            raise\n"),
        ("        self._start_intrabar_task()\n        consumer_token", "        consumer_token"),
        (_NESTED_STOP, f"            {_AWAIT}\n            {_IF}\n                {_REL}\n"),  # E1
        (_NESTED_STOP, f"            {_IF}\n                {_REL}\n            {_AWAIT}\n"),  # E2
    ],
)  # fmt: skip
def test_the_engine_pin_catches_mutations(old: str, new: str) -> None:
    source = _ENGINE_FILE.read_text("utf-8")
    assert source.count(old) == 1, old
    assert engine_stage_problems(source.replace(old, new)) != []


_ARC14 = frozenset({"packages/trading/adaptive_learning.py"})  # report-only reader (SY-01-22c)


def arc14_problems(rel: str, source: str) -> list[str]:
    """ARC-14: no strategy-side module reads a provider singleton (``get_global_client``,
    ``cached_value``, ``*_from_cache``); only the report-only adaptive learning may."""
    if rel in _ARC14:
        return []
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names = [getattr(node, "id", None), getattr(node, "attr", None)]
        names += [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        found += [
            f"{rel}: {n}" for n in names
            if isinstance(n, str)
            and (n in ("get_global_client", "cached_value") or n.endswith("_from_cache"))
        ]  # fmt: skip
    return found


def test_only_the_report_only_adaptive_learning_reads_provider_singletons() -> None:
    files = sorted((_ROOT / "packages" / "trading").rglob("*.py"))
    assert len(files) > 30
    offenders = {
        f.relative_to(_ROOT).as_posix(): p
        for f in files
        if (p := arc14_problems(f.relative_to(_ROOT).as_posix(), f.read_text("utf-8")))
    }
    assert offenders == {}
    adaptive = _ROOT / next(iter(_ARC14))
    assert arc14_problems("x.py", adaptive.read_text("utf-8")) != []  # the allowlist is not stale


@pytest.mark.parametrize(
    "line",
    [
        "from data.sentiment import get_global_client\n",
        "x = client.cached_value\n",
        "y = c.btc_dominance_at_offset_from_cache(days_ago=7)\n",
        "getattr(c, 'cached_value')\n",
    ],
)
def test_the_arc14_pin_catches_mutations(line: str) -> None:
    source = (_ROOT / "packages" / "trading" / "mtf_signals.py").read_text("utf-8")
    assert arc14_problems("packages/trading/mtf_signals.py", source) == []
    assert arc14_problems("packages/trading/mtf_signals.py", source + "\n" + line) != []


def test_public_counts_unpack_targets_and_nested_classes_but_not_method_locals() -> None:
    source = (
        "class C:\n    def m(self):\n        a, b = 1, 2\n        class Inner: ...\n"
        "    x, (y, *z) = 1, (2, 3)\n    class N: ...\n    w: int = 0\n    _p = 1\n"
    )
    assert _public(ast.parse(source), "C") == {"m", "x", "y", "z", "N", "w"}


_LRF = "_LIVE_READ_FAILED: Final = SignalValues()"


def live_read_failed_problems(source: str) -> list[str]:
    """T-15 / SEC-P4A-01: the failure value is bound once, as the all-None ``SignalValues()``."""
    tree = ast.parse(source)
    binds = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name | ast.Attribute) and not isinstance(n.ctx, ast.Load)
        and (getattr(n, "id", None) or getattr(n, "attr", None)) == "_LIVE_READ_FAILED"
    ]  # fmt: skip
    exact = [n for n in tree.body if isinstance(n, ast.AnnAssign) and ast.unparse(n) == _LRF]
    return [] if len(binds) == 1 and len(exact) == 1 else ["_LIVE_READ_FAILED is not one binding"]


@pytest.mark.parametrize(
    "new",
    [
        "_LIVE_READ_FAILED: Final = SignalValues(fear_greed_index=50)",  # X1
        "_LIVE_READ_FAILED: Final = SignalValues(whale_net_flow=0.0)",  # X10
        "_LIVE_READ_FAILED = SignalValues()",  # no Final
        f"{_LRF}\n_LIVE_READ_FAILED = SignalValues(fed_funds_rate=1.0)",  # a second binding
    ],
)
def test_the_live_read_failed_pin_is_clean_on_the_real_file_and_catches_mutations(
    new: str,
) -> None:
    source = (_ROOT / _MTF).read_text("utf-8")
    assert source.count(_LRF) == 1 and live_read_failed_problems(source) == []
    assert live_read_failed_problems(source.replace(_LRF, new)) != []


_STAGE_NAMES = frozenset({"LIVE_CONTEXT_STAGE", "_live_stage"})
_STAGE_LITERAL = 'LIVE_CONTEXT_STAGE: Final = "neutral_shadow"'


def stage_source_problems(source: str, rel: str) -> list[str]:
    """SY-01-04/19/21h: the stage is one reviewed literal in ``mtf_signals.py``, read only by the
    engine; nothing else names it, writes it or spells it as a string (setattr / ``__dict__``)."""
    tree = ast.parse(source)
    out: list[str] = []
    in_all = {  # the ``__all__`` entry is the one allowed string spelling, in mtf_signals.py
        id(c) for a in tree.body if isinstance(a, ast.Assign)
        and any(getattr(t, "id", "") == "__all__" for t in a.targets) for c in ast.walk(a.value)
    }  # fmt: skip
    if rel == _MTF:
        binds = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Name | ast.Attribute) and not isinstance(n.ctx, ast.Load)
            and (getattr(n, "id", None) or getattr(n, "attr", None)) == "LIVE_CONTEXT_STAGE"
        ]  # fmt: skip
        lit = [
            n for n in tree.body
            if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "LIVE_CONTEXT_STAGE"
            and ast.unparse(n.annotation) == "Final"
            and isinstance(n.value, ast.Constant) and type(n.value.value) is str
        ]  # fmt: skip
        if len(binds) != 1 or len(lit) != 1:
            out.append("stage: not one Final string literal")
    for n in ast.walk(tree):
        ident = getattr(n, "id", None) or getattr(n, "attr", None)
        if ident in _STAGE_NAMES and rel not in (_MTF, _ENGINE):
            out.append(f"stage name outside its homes: {ident}")
        if (
            ident in _STAGE_NAMES and rel == _ENGINE and ident != "_live_stage"
            and not isinstance(n.ctx, ast.Load)  # type: ignore[attr-defined]
        ):  # fmt: skip
            out.append(f"stage store in the engine: {ident}")
        if isinstance(n, ast.Constant) and n.value in _STAGE_NAMES:
            if not (rel == _MTF and id(n) in in_all):
                out.append(f"stage name as a string: {n.value}")
    return out


def test_the_stage_is_one_reviewed_literal_read_only_by_the_engine() -> None:
    files = [*sorted((_ROOT / "packages").rglob("*.py")), *sorted((_ROOT / "apps").rglob("*.py"))]
    assert len(files) > 100  # the walk is not vacuous
    offenders = {}
    for f in files:
        rel = f.relative_to(_ROOT).as_posix()
        if problems := stage_source_problems(f.read_text("utf-8"), rel):
            offenders[rel] = problems
    assert offenders == {}


_STAGE_REGISTER = "            return LIVE_CONSUMERS.started(self._run_mode.value)\n"
_ORCHESTRATOR = "apps/api/services/run_orchestrator.py"


@pytest.mark.parametrize(
    ("rel", "anchor", "new"),
    [
        (_MTF, _STAGE_LITERAL,  # S3: read from a config file
         'from pathlib import Path\nLIVE_CONTEXT_STAGE: Final = Path("s").read_text()'),
        (_MTF, _STAGE_LITERAL, "import sys\nLIVE_CONTEXT_STAGE: Final = sys.argv[-1]"),  # S4
        (_MTF, _STAGE_LITERAL,  # S6: not a literal
         'LIVE_CONTEXT_STAGE: Final = {"a": "neutral_shadow"}.get("a", "fresh")'),
        (_ENGINE, _STAGE_REGISTER,  # S7: the engine stores to the module constant
         "            mtf_signals_module.LIVE_CONTEXT_STAGE = 'fresh'\n" + _STAGE_REGISTER),
        (_ENGINE, _STAGE_REGISTER,  # S10: the instance stage forced past the fence
         "            object.__setattr__(self, '_live_stage', 'fresh')\n" + _STAGE_REGISTER),
        (_ORCHESTRATOR, None, "engine._live_stage = 'fresh'\n"),  # S19
        (_ORCHESTRATOR, None, "setattr(engine, '_live_stage', None)\n"),  # S20
        ("apps/api/routers/runs.py", None,  # S21: a request value
         "import trading.mtf_signals as m\nm.LIVE_CONTEXT_STAGE = body.stage\n"),
    ],
)  # fmt: skip
def test_the_stage_pin_catches_mutations(rel: str, anchor: str | None, new: str) -> None:
    if anchor is None:
        source = new
    else:
        real = (_ROOT / rel).read_text("utf-8")
        assert real.count(anchor) == 1 and stage_source_problems(real, rel) == []
        source = real.replace(anchor, new)
    assert stage_source_problems(source, rel) != []


def test_a_read_only_import_of_the_stage_is_allowed() -> None:  # S22, CF-2401-81
    source = "from trading.mtf_signals import LIVE_CONTEXT_STAGE\n"
    assert stage_source_problems(source, "apps/api/routers/health.py") == []
