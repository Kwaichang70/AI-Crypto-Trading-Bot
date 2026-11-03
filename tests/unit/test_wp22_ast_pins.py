"""WP22-T-07 -- AST pins for the bar-time clock (WP2.2 commit P2b).

Static guarantees that no behavioural test can give: where the wall clock may
still be read, where the ambient clock may be entered or read, who may inject a
clock, and that the paper/live guards use IDENTITY.  Every pin has a positive
control (a snippet it must flag), so a pin cannot silently go vacuous.

    R4   exactly one ``with use_clock(<Name>)``, in ``StrategyEngine.run_backtest``,
         after the BACKTEST/RUNNING checks, directly preceded by an explicit
         ``if not isinstance(<Name>, BarClock): raise TypeError``; no ``yield`` in
         ``run_backtest``; no manual ``__enter__`` / ``enter_context``; ``_AMBIENT``
         untouched outside ``trading/clock.py``
    R5   no task / thread spawn in ``run_backtest`` or ``backtest.py``; the breaker
         alert listener is wired only in the orchestrator's paper/live functions
    R7   zero ``clock=`` on the engine/portfolio classes in ``apps/``; only
         ``backtest.py`` injects, and not via ``PaperExecutionEngine(clock=)``;
         ``attach_clock`` is called only from ``strategy_engine.py``
    R9   ``utc_now`` only as a pydantic ``default_factory``; every datetime
         ``default_factory`` in ``models.py`` is ``utc_now``
    R12  ``ambient_clock()`` only as an ``is`` / ``is not`` operand at tripwire sites
    R2/R1  identity guards, never ``isinstance(.., WallClock)`` / ``type(x) is``
    T-07 explicit (function, call) allowlist for wall-clock reads
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "packages" / "trading"
_SCAN_ROOTS = (_REPO / "packages", _REPO / "apps")
_CLOCK_PY = _PKG / "clock.py"

_STAMP_MODULES = (
    "portfolio.py",
    "engines/paper.py",
    "execution.py",
    "backtest.py",
    "strategy_engine.py",
    "models.py",
    "bracket_exit.py",
    "trailing_stop.py",
    "trade_journal.py",
    "risk.py",
    "risk_manager.py",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source_files() -> Iterator[Path]:
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            yield path


def _rel(path: Path) -> str:
    return path.relative_to(_REPO).as_posix()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    names: list[str] = []
    cur: ast.AST | None = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(cur.name)
        cur = parents.get(cur)
    return list(reversed(names))


def _func_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _find_function(
    tree: ast.Module, cls: str | None, name: str
) -> ast.AsyncFunctionDef | ast.FunctionDef:
    scope: list[ast.stmt] = tree.body
    if cls is not None:
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        scope = klass.body
    for node in scope:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls}.{name} not found")


# ---------------------------------------------------------------------------
# T-07 -- wall-clock read allowlist
# ---------------------------------------------------------------------------

_WALL_ATTRS = {"now", "utcnow", "today", "time", "time_ns", "perf_counter"}
_WALL_RECEIVERS = {"datetime", "date", "time", "dt"}


def _wall_reads(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(function-path, call, line)`` of every direct wall-clock read.

    ``self._clock.now()`` / ``bar_clock.now()`` are NOT wall reads: only calls on a
    ``datetime`` / ``date`` / ``time`` receiver (or a bare imported name) count.
    """
    parents = _parents(tree)
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        label: str | None = None
        if isinstance(func, ast.Attribute) and func.attr in _WALL_ATTRS:
            recv = func.value
            if isinstance(recv, ast.Name) and recv.id in _WALL_RECEIVERS:
                label = f"{recv.id}.{func.attr}"
            elif (
                isinstance(recv, ast.Attribute)
                and recv.attr in _WALL_RECEIVERS
                and isinstance(recv.value, ast.Name)
                and recv.value.id in {"datetime", "dt"}
            ):
                label = f"{recv.value.id}.{recv.attr}.{func.attr}"
        elif isinstance(func, ast.Name) and func.id in _WALL_ATTRS - {"now", "today"}:
            label = func.id
        if label is not None:
            found.append((".".join(_enclosing(node, parents)) or "<module>", label, node.lineno))
    return found


# (module, function path, call): the ONLY direct wall reads left in the stamp modules.
_WALL_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset(
    {
        # (portfolio.py's ``_utc_now`` seam delegates to ``trading.clock._wall_now``
        # since WP22-P2B1-03, so it is no longer a direct wall read)
        # live-only quote-age log field (never on the backtest path)
        ("strategy_engine.py", "StrategyEngine._quote_context", "datetime.now"),
    }
)


class TestWallClockAllowlist:
    def test_no_unlisted_wall_read_in_the_stamp_modules(self) -> None:
        unexpected: list[str] = []
        used: set[tuple[str, str, str]] = set()
        for rel in _STAMP_MODULES:
            for func, call, line in _wall_reads(_parse(_PKG / rel)):
                key = (rel, func, call)
                if key in _WALL_ALLOWLIST:
                    used.add(key)
                else:
                    unexpected.append(f"{rel}:{line} {func} -> {call}")
        assert not unexpected, unexpected
        assert used == _WALL_ALLOWLIST, f"stale allowlist entries: {_WALL_ALLOWLIST - used}"

    def test_safety_evaluate_reads_no_wall_clock(self) -> None:
        tree = _parse(_PKG / "safety.py")
        evaluate = _find_function(tree, "CircuitBreaker", "evaluate")
        assert _wall_reads(ast.Module(body=[evaluate], type_ignores=[])) == []

    def test_control_the_detector_flags_each_form(self) -> None:
        tree = ast.parse(
            "import time\n"
            "from datetime import datetime, UTC\n"
            "import datetime as dt\n"
            "def a():\n"
            "    return datetime.now(tz=UTC), datetime.utcnow(), time.time(), dt.datetime.now()\n"
            "def b(clock):\n"
            "    return clock.now()\n"
        )
        labels = sorted(call for _f, call, _l in _wall_reads(tree))
        assert labels == ["datetime.now", "datetime.utcnow", "dt.datetime.now", "time.time"]


# ---------------------------------------------------------------------------
# R9 -- models.py default factories and utc_now usage
# ---------------------------------------------------------------------------


class TestUtcNowOnlyAsDefaultFactory:
    def test_every_datetime_default_factory_in_models_is_utc_now(self) -> None:
        tree = _parse(_PKG / "models.py")
        checked = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call)):
                continue
            if "datetime" not in ast.unparse(node.annotation):
                continue
            for kw in node.value.keywords:
                if kw.arg == "default_factory":
                    assert isinstance(kw.value, ast.Name) and kw.value.id == "utc_now", ast.unparse(
                        node
                    )
                    checked += 1
        # Signal.generated_at, Order x2, Fill, Position x2, SkippedTrade, RiskCheckResult
        assert checked == 8

    def test_utc_now_is_referenced_only_as_a_default_factory(self) -> None:
        offenders: list[str] = []
        total = 0
        for path in _source_files():
            if path == _CLOCK_PY:
                continue
            tree = _parse(path)
            parents = _parents(tree)
            for node in ast.walk(tree):
                is_ref = (isinstance(node, ast.Name) and node.id == "utc_now") or (
                    isinstance(node, ast.Attribute) and node.attr == "utc_now"
                )
                if not is_ref:
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.keyword) and parent.arg == "default_factory":
                    total += 1
                    continue
                offenders.append(f"{_rel(path)}:{node.lineno}")
        assert not offenders, offenders
        assert total == 8

    def test_ambient_internals_are_private_to_clock_py(self) -> None:
        offenders = []
        for path in _source_files():
            if path == _CLOCK_PY:
                continue
            for node in ast.walk(_parse(path)):
                if (isinstance(node, ast.Name) and node.id == "_AMBIENT") or (
                    isinstance(node, ast.Attribute) and node.attr == "_AMBIENT"
                ):
                    offenders.append(f"{_rel(path)}:{node.lineno}")
        assert not offenders, offenders


# ---------------------------------------------------------------------------
# R4 -- use_clock placement
# ---------------------------------------------------------------------------


def _use_clock_calls(tree: ast.Module) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _func_name(n) == "use_clock"]


class TestUseClockPlacement:
    def test_exactly_one_with_use_clock_in_run_backtest(self) -> None:
        withs: list[tuple[Path, ast.With, list[str]]] = []
        for path in _source_files():
            if path == _CLOCK_PY:
                continue
            tree = _parse(path)
            parents = _parents(tree)
            for call in _use_clock_calls(tree):
                parent = parents.get(call)
                grand = parents.get(parent) if parent is not None else None
                if isinstance(parent, ast.withitem) and isinstance(grand, ast.With | ast.AsyncWith):
                    withs.append((path, grand, _enclosing(call, parents)))  # type: ignore[arg-type]
                else:  # manual __enter__, enter_context(use_clock(..)), a stored manager
                    pytest.fail(f"use_clock used outside a plain with: {_rel(path)}:{call.lineno}")
        assert len(withs) == 1, [f"{_rel(p)}:{w.lineno}" for p, w, _ in withs]
        path, with_node, scope = withs[0]
        assert path == _PKG / "strategy_engine.py"
        assert scope == ["StrategyEngine", "run_backtest"]
        assert isinstance(with_node, ast.With)  # not async with
        arg = with_node.items[0].context_expr
        assert (
            isinstance(arg, ast.Call) and len(arg.args) == 1 and isinstance(arg.args[0], ast.Name)
        )

    def test_with_is_after_the_mode_and_state_checks_and_after_an_explicit_raise(self) -> None:
        tree = _parse(_PKG / "strategy_engine.py")
        fn = _find_function(tree, "StrategyEngine", "run_backtest")
        idx = next(i for i, s in enumerate(fn.body) if isinstance(s, ast.With))
        with_node = fn.body[idx]
        assert isinstance(with_node, ast.With)
        name = with_node.items[0].context_expr.args[0].id  # type: ignore[attr-defined]

        guards = [ast.unparse(s.test) for s in fn.body[:idx] if isinstance(s, ast.If)]
        assert any("_run_mode" in g and "RunMode.BACKTEST" in g for g in guards), guards
        assert any("_state" in g and "EngineState.RUNNING" in g for g in guards), guards

        prev = fn.body[idx - 1]
        # explicit raise, never ``assert`` (stripped by ``python -O``)
        assert isinstance(prev, ast.If), ast.unparse(prev)
        assert ast.unparse(prev.test) == f"not isinstance({name}, BarClock)"
        assert len(prev.body) == 1 and isinstance(prev.body[0], ast.Raise)
        raised = prev.body[0].exc
        assert isinstance(raised, ast.Call) and _func_name(raised) == "TypeError"
        assert not any(isinstance(n, ast.Assert) for n in ast.walk(prev))

    def test_run_backtest_is_not_a_generator(self) -> None:
        fn = _find_function(_parse(_PKG / "strategy_engine.py"), "StrategyEngine", "run_backtest")
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.Yield | ast.YieldFrom)]

    def test_control_manual_enter_and_enter_context_are_flagged(self) -> None:
        for snippet in (
            "def f(c):\n    cm = use_clock(c)\n    cm.__enter__()\n",
            "def f(c, stack):\n    stack.enter_context(use_clock(c))\n",
        ):
            tree = ast.parse(snippet)
            parents = _parents(tree)
            call = _use_clock_calls(tree)[0]
            parent = parents.get(call)
            assert not isinstance(parent, ast.withitem)


# ---------------------------------------------------------------------------
# R5 -- no spawn inside the backtest path; listener wiring
# ---------------------------------------------------------------------------

_SPAWNERS = {
    "create_task",
    "ensure_future",
    "TaskGroup",
    "to_thread",
    "run_in_executor",
    "Thread",
    "submit",
    "gather",
}


def _spawns(node: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and _func_name(n) in _SPAWNERS:
            out.append(f"{_func_name(n)}@{n.lineno}")
    return out


class TestNoSpawnAndListenerWiring:
    def test_run_backtest_spawns_nothing(self) -> None:
        fn = _find_function(_parse(_PKG / "strategy_engine.py"), "StrategyEngine", "run_backtest")
        assert _spawns(fn) == []

    def test_backtest_module_spawns_nothing(self) -> None:
        assert _spawns(_parse(_PKG / "backtest.py")) == []

    def test_control_spawn_detector(self) -> None:
        assert _spawns(ast.parse("import asyncio\nasyncio.create_task(f())\n"))

    def test_breaker_alert_listener_wired_only_in_paper_and_live_entrypoints(self) -> None:
        names = {"make_circuit_breaker_alert_listener", "add_transition_listener"}
        sites: list[tuple[str, list[str]]] = []
        for path in _source_files():
            tree = _parse(path)
            parents = _parents(tree)
            for node in ast.walk(tree):
                ref = (isinstance(node, ast.Name) and node.id in names) or (
                    isinstance(node, ast.Attribute) and node.attr in names
                )
                if not ref:
                    continue
                scope = _enclosing(node, parents)
                if isinstance(parents.get(node), ast.alias | ast.Constant | ast.Assign | ast.List):
                    continue
                if scope and scope[-1] in names:  # the definition itself
                    continue
                sites.append((_rel(path), scope))
        assert sites, "no wiring site found: the pin would be vacuous"
        for rel, scope in sites:
            assert rel == "apps/api/services/run_orchestrator.py", (rel, scope)
            assert scope[-1] in {"run_paper_engine", "run_live_engine"}, (rel, scope)


# ---------------------------------------------------------------------------
# R7 / R3 -- who may inject a clock
# ---------------------------------------------------------------------------

_CLOCK_TAKERS = {
    "StrategyEngine",
    "PaperExecutionEngine",
    "LiveExecutionEngine",
    "PortfolioAccounting",
    "from_fills",
}


def _clock_kw_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _func_name(n) in _CLOCK_TAKERS
        and any(kw.arg == "clock" for kw in n.keywords)
    ]


class TestInjectionSites:
    def test_apps_never_pass_a_clock(self) -> None:
        offenders = [
            f"{_rel(p)}:{c.lineno}"
            for p in (_REPO / "apps").rglob("*.py")
            if "__pycache__" not in p.parts
            for c in _clock_kw_calls(_parse(p))
        ]
        assert offenders == []

    def test_only_the_backtest_runner_injects_in_packages(self) -> None:
        sites: dict[str, list[str]] = {}
        for path in (_REPO / "packages").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for call in _clock_kw_calls(_parse(path)):
                sites.setdefault(_rel(path), []).append(str(_func_name(call)))
        # the runner injects into the portfolio and the StrategyEngine ONLY
        # (``PortfolioAccounting.from_fills`` builds ``cls(...)``, which is not a call site here)
        assert sites == {
            "packages/trading/backtest.py": ["PortfolioAccounting", "StrategyEngine"]
        }, sites

    def test_runner_does_not_use_the_paper_engine_clock_kwarg(self) -> None:
        tree = _parse(_PKG / "backtest.py")
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and _func_name(call) == "PaperExecutionEngine":
                assert all(kw.arg != "clock" for kw in call.keywords)

    def test_attach_clock_is_called_only_from_strategy_engine(self) -> None:
        sites: list[str] = []
        for path in _source_files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                is_call = _func_name(node) == "attach_clock"
                is_getattr = _func_name(node) == "getattr" and any(
                    isinstance(a, ast.Constant) and a.value == "attach_clock" for a in node.args
                )
                if is_call or is_getattr:
                    sites.append(f"{_rel(path)}:{node.lineno}")
        assert sites, "no attach_clock site found: the pin would be vacuous"
        assert {s.rsplit(":", 1)[0] for s in sites} == {"packages/trading/strategy_engine.py"}

    def test_exec_clock_attach_runs_only_in_backtest_mode(self) -> None:
        tree = _parse(_PKG / "strategy_engine.py")
        parents = _parents(tree)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_attach_exec_clock"
        ]
        assert sorted(_enclosing(c, parents)[-1] for c in calls) == ["__init__", "run_backtest"]
        init_call = next(c for c in calls if _enclosing(c, parents)[-1] == "__init__")
        guard = parents[parents[init_call]]  # Expr -> If
        assert isinstance(guard, ast.If) and "RunMode.BACKTEST" in ast.unparse(guard.test)


# ---------------------------------------------------------------------------
# R12 -- ambient_clock() only as an identity operand at tripwire sites
# ---------------------------------------------------------------------------

_TRIPWIRE_SCOPES = {
    ("packages/trading/strategy_engine.py", "_wall_clock_tripwire"),
    ("packages/trading/engines/live.py", "on_start"),
    ("apps/api/services/run_orchestrator.py", "run_paper_engine"),
    ("apps/api/services/run_orchestrator.py", "run_live_engine"),
}


class TestAmbientClockReads:
    def test_ambient_clock_only_as_is_operand_at_tripwire_sites(self) -> None:
        seen: set[tuple[str, str]] = set()
        for path in _source_files():
            if path == _CLOCK_PY:
                continue
            tree = _parse(path)
            parents = _parents(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and _func_name(node) == "ambient_clock"):
                    continue
                parent = parents.get(node)
                where = f"{_rel(path)}:{node.lineno}"
                assert isinstance(parent, ast.Compare), where
                assert len(parent.ops) == 1 and isinstance(parent.ops[0], ast.Is | ast.IsNot), where
                assert parent.left is node and ast.unparse(parent.comparators[0]) == "WALL_CLOCK", (
                    where
                )
                scope = _enclosing(node, parents)[-1]
                assert (_rel(path), scope) in _TRIPWIRE_SCOPES, where
                seen.add((_rel(path), scope))
        assert seen == _TRIPWIRE_SCOPES

    def test_control_assigned_or_passed_ambient_clock_is_not_a_compare_operand(self) -> None:
        for snippet in (
            "x = ambient_clock()\n",
            "f(ambient_clock())\n",
            "return_ = type(ambient_clock())\n",
        ):
            tree = ast.parse(snippet)
            parents = _parents(tree)
            call = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and _func_name(n) == "ambient_clock"
            )
            assert not isinstance(parents.get(call), ast.Compare)


# ---------------------------------------------------------------------------
# R1 / R2 -- identity guards, constructor shapes
# ---------------------------------------------------------------------------


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    return {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]}


class TestGuardShapes:
    def test_clock_ctor_parameter_exists_only_on_paper_engine_and_strategy_engine(self) -> None:
        execution = _parse(_PKG / "execution.py")
        live = _parse(_PKG / "engines" / "live.py")
        paper = _parse(_PKG / "engines" / "paper.py")
        strategy = _parse(_PKG / "strategy_engine.py")
        assert "clock" not in _params(_find_function(execution, "BaseExecutionEngine", "__init__"))
        assert "clock" not in _params(_find_function(live, "LiveExecutionEngine", "__init__"))
        assert "clock" in _params(_find_function(paper, "PaperExecutionEngine", "__init__"))
        assert "clock" in _params(_find_function(strategy, "StrategyEngine", "__init__"))

    def test_base_engine_has_a_class_level_wall_clock_default(self) -> None:
        tree = _parse(_PKG / "execution.py")
        klass = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseExecutionEngine"
        )
        assigns = [
            n
            for n in klass.body
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "_clock"
        ]
        assert len(assigns) == 1
        assert ast.unparse(assigns[0].annotation) == "Clock"
        assert ast.unparse(assigns[0].value) == "WALL_CLOCK"  # type: ignore[arg-type]

    def test_live_attach_clock_is_an_identity_check(self) -> None:
        fn = _find_function(
            _parse(_PKG / "engines" / "live.py"), "LiveExecutionEngine", "attach_clock"
        )
        compares = [n for n in ast.walk(fn) if isinstance(n, ast.Compare)]
        assert [ast.unparse(c) for c in compares] == ["clock is not WALL_CLOCK"]
        assert not [
            n for n in ast.walk(fn) if isinstance(n, ast.Call) and _func_name(n) == "isinstance"
        ]

    @pytest.mark.parametrize(
        "rel",
        [
            "strategy_engine.py",
            "engines/live.py",
            "execution.py",
            "../../apps/api/services/run_orchestrator.py",
        ],
    )
    def test_guard_modules_import_the_singleton_by_name_and_never_test_wallclock_type(
        self, rel: str
    ) -> None:
        tree = _parse(_PKG / rel)
        imported = {
            alias.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module == "trading.clock"
            for alias in n.names
        }
        assert "WALL_CLOCK" in imported
        assert "WallClock" not in imported
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                assert n.id != "WallClock", f"{rel}:{n.lineno}"

    def test_no_type_identity_check_on_clocks_anywhere(self) -> None:
        offenders = []
        for path in _source_files():
            tree = _parse(path)
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Compare)
                    and isinstance(n.left, ast.Call)
                    and _func_name(n.left) == "type"
                    and any(
                        isinstance(c, ast.Name) and c.id in {"BarClock", "WallClock"}
                        for c in n.comparators
                    )
                ):
                    offenders.append(f"{_rel(path)}:{n.lineno}")
        assert offenders == []

    def test_strategy_engine_guard_is_identity_for_paper_live_and_isinstance_for_backtest(
        self,
    ) -> None:
        fn = _find_function(_parse(_PKG / "strategy_engine.py"), "StrategyEngine", "__init__")
        text = ast.unparse(fn)
        assert "isinstance(clock, BarClock)" in text
        assert "clock is not WALL_CLOCK" in text


# ---------------------------------------------------------------------------
# Amendment A -- one close_i object for both clocks
# ---------------------------------------------------------------------------


class TestSingleCloseValue:
    def test_run_backtest_passes_the_same_name_to_both_advances(self) -> None:
        fn = _find_function(_parse(_PKG / "strategy_engine.py"), "StrategyEngine", "run_backtest")
        assigns = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "close_i" for t in n.targets)
        ]
        assert len(assigns) == 1
        advance = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and _func_name(n) in {"advance", "_advance_clock"}
        ]
        assert sorted(str(_func_name(c)) for c in advance) == ["_advance_clock", "advance"]
        for call in advance:
            assert [ast.unparse(a) for a in call.args] == ["close_i"], ast.unparse(call)
        # both calls are fed before any mark / fill of the bar
        text = ast.unparse(fn)
        assert text.index("bar_clock.advance(close_i)") < text.index("_update_engine_prices")

    def test_run_backtest_does_not_recompute_the_close(self) -> None:
        fn = _find_function(_parse(_PKG / "strategy_engine.py"), "StrategyEngine", "run_backtest")
        maxes = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _func_name(n) == "max"]
        # one for ``close_i``, one for the (no-clock) BarClock seed
        assert len(maxes) == 2


# ---------------------------------------------------------------------------
# R13 -- Python 3.14 thread_inherit_context gate
# ---------------------------------------------------------------------------


def test_python_314_thread_inherit_context_needs_a_rereview() -> None:
    assert not getattr(sys.flags, "thread_inherit_context", False), (
        "thread_inherit_context is enabled: threads now inherit the caller's context, so a "
        "BarClock set by use_clock in run_backtest could leak into worker threads.  Re-review "
        "R5 / WP22-P1-SEC-08 / WP22-P1-R2-SEC-07 (no spawn in run_backtest, "
        "test_other_thread_sees_wall_time) before adopting Python 3.14."
    )
