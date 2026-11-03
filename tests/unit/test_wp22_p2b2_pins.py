"""WP22-P2B2 -- AST pin over the production ``._clock =`` write sites (P2B1-SEC-06).

Every attribute write to a ``_clock`` outside ``tests/`` must be one of the sites below.
Anything else (a new restore, a swap helper, a ``setattr(obj, "_clock", ...)``) bypasses
``attach_clock`` / ``_validated_clock`` and has to be reviewed against R1-R3 first.  The pin has a
positive control, so it cannot go vacuous.
"""

from __future__ import annotations

import ast

from tests.unit.test_wp22_ast_pins import _enclosing, _parents, _parse, _rel, _source_files

#: (file, enclosing scope) -> number of writes
_ALLOWED: dict[tuple[str, str], int] = {
    ("packages/trading/engines/paper.py", "PaperExecutionEngine.__init__"): 1,
    ("packages/trading/execution.py", "BaseExecutionEngine.attach_clock"): 1,
    ("packages/trading/portfolio.py", "PortfolioAccounting.__init__"): 1,
    ("packages/trading/strategy_engine.py", "StrategyEngine.__init__"): 1,
    ("packages/trading/strategy_engine.py", "StrategyEngine._restore_fallback_clock"): 2,
    ("packages/trading/strategy_engine.py", "StrategyEngine.run_backtest"): 1,
}


def _clock_writes(tree: ast.AST) -> list[tuple[str, int]]:
    parents = _parents(tree)
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Attribute) and t.attr == "_clock":
                out.append((".".join(_enclosing(node, parents)), node.lineno))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "object.__setattr__"}
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_clock"
        ):
            out.append((".".join(_enclosing(node, parents)) + "<setattr>", node.lineno))
    return out


def test_production_clock_writes_are_exactly_the_allowed_sites() -> None:
    found: dict[tuple[str, str], int] = {}
    for path in _source_files():
        for scope, _line in _clock_writes(_parse(path)):
            key = (_rel(path), scope)
            found[key] = found.get(key, 0) + 1
    assert found == _ALLOWED, found


def test_the_pin_flags_an_unlisted_write_and_a_setattr() -> None:
    tree = ast.parse(
        "class A:\n"
        "    def f(self, other):\n"
        "        other._clock = 1\n"
        "        setattr(other, '_clock', 2)\n"
    )
    assert [s for s, _ in _clock_writes(tree)] == ["A.f", "A.f<setattr>"]
