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
