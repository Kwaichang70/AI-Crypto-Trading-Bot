"""CF-97-15 T-9715-03: every market-data service built in production passes ``cache_ttl_seconds=0``.

An AST inventory over ``apps``, ``packages`` and ``scripts``: each ``CCXTMarketDataService(...)``
call and each ``super().__init__(...)`` of a ``BaseMarketDataService`` subclass must pass the
literal ``0``.  A missing keyword is a violation, because the constructor default is 60, which on a
1m poll serves the previous bar every other poll.  The inventory is exact: a new construction site
fails until it is reviewed.  Planted-offender controls prove the pin sees each violation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASES = {"BaseMarketDataService", "CCXTMarketDataService"}
_SITES = {
    ("apps/api/routers/runs.py", "_fetch_bars_for_backtest"),
    ("apps/api/services/run_orchestrator.py", "run_live_engine"),
    ("apps/api/services/run_orchestrator.py", "run_paper_engine"),
    ("packages/trading/backtest.py", "__init__"),
}


def _sites(src: str, rel: str) -> list[tuple[str, str, bool]]:
    """(file, enclosing function, passes the literal 0) for every construction site."""
    out: list[tuple[str, str, bool]] = []
    ctors = {"CCXTMarketDataService"}  # plus any ``import ... as`` alias (SE-9715-01)

    def is_super_init(c: ast.Call) -> bool:
        f = c.func
        return (
            isinstance(f, ast.Attribute)
            and f.attr == "__init__"
            and ast.unparse(f.value) == "super()"
        )

    def rec(node: ast.AST, func: str, bases: set[str], cls: str) -> None:
        for c in ast.iter_child_nodes(node):
            f = c.name if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef) else func
            sub = bases
            if isinstance(c, ast.ClassDef):
                names = {ast.unparse(b).split(".")[-1] for b in c.bases}
                sub = bases | ({c.name} if names & (_BASES | bases) else set())
            name = getattr(getattr(c, "func", None), "attr", None) or getattr(
                getattr(c, "func", None), "id", None
            )
            forwarder = (
                cls == "CCXTMarketDataService"
            )  # the implementation forwards its own parameter
            if isinstance(c, ast.Call) and (name in ctors or (
                    is_super_init(c) and cls in sub and not forwarder)):  # fmt: skip
                kw = {k.arg: k.value for k in c.keywords}.get("cache_ttl_seconds")
                ok = isinstance(kw, ast.Constant) and kw.value == 0 and type(kw.value) is int
                out.append((rel, f, ok))
            rec(c, f, sub, c.name if isinstance(c, ast.ClassDef) else cls)

    tree = ast.parse(src)
    ctors.update(a.asname for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                 for a in n.names if a.name == "CCXTMarketDataService" and a.asname)  # fmt: skip
    rec(tree, "<module>", set(), "")
    return out


def _py() -> Iterable[tuple[str, str]]:
    for root in ("apps", "packages", "scripts"):
        for path in sorted((_ROOT / root).rglob("*.py")):
            yield path.relative_to(_ROOT).as_posix(), path.read_text()


def test_every_production_service_passes_a_literal_zero_ttl_and_the_inventory_is_exact() -> None:
    found = [s for rel, src in _py() for s in _sites(src, rel)]
    assert {(r, f) for r, f, _ in found} == _SITES
    assert [s for s in found if not s[2]] == []


def test_the_pin_flags_planted_offenders() -> None:
    call = "def f():\n    CCXTMarketDataService(exchange_id='x'{})\n"
    flagged = [("x.py", "f", False)]
    assert _sites(call.format(""), "x.py") == flagged  # a missing keyword
    for bad in ("60", "ttl", "False"):
        assert _sites(call.format(f", cache_ttl_seconds={bad}"), "x.py") == flagged, bad
    assert _sites(call.format(", cache_ttl_seconds=0"), "x.py") == [("x.py", "f", True)]
    alias = "from m import CCXTMarketDataService as S\ndef f():\n    S(exchange_id='x')\n"
    assert _sites(alias, "x.py") == flagged  # an aliased import (SE-9715-01)
    sub = "class S(BaseMarketDataService):\n    def __init__(self):\n        super().__init__({})\n"
    assert _sites(sub.format("exchange_id='x'"), "x.py") == [("x.py", "__init__", False)]
    assert _sites(sub.format("cache_ttl_seconds=0"), "x.py") == [("x.py", "__init__", True)]
    assert _sites("class T:\n    def __init__(self):\n        super().__init__()\n", "x.py") == []
