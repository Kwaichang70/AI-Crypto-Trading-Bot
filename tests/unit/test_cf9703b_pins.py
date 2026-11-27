"""CF-97-03 P7b1: AST and text pins of the egress budget, the spawn hook and the two riders.

Every pin is a function over source text, so it runs on the real file and on planted offenders.
Groups follow ``reports/vp2-cf2703/p7b-size-ruling.md`` section 5 (pins 1, 2a-e, 4, 6, 7, 10-12,
14, 15b/c).  The P7a pin file is not edited: its helpers are re-stated here where needed.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RUNS, _ORCH = "apps/api/routers/runs.py", "apps/api/services/run_orchestrator.py"
_EB, _SH, _ADM = (
    f"apps/api/services/{n}.py" for n in ("egress_budget", "spawn_hooks", "admission")
)
_BANNED = {
    "reserve_new_run",
    "history_check",
    "reserve_history",
    "reserve_history_blocking",
    "suspend_history",
}


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text()


def _py(*roots: str) -> Iterable[tuple[str, str]]:
    for root in roots:
        for path in sorted((_ROOT / root).rglob("*.py")):
            yield path.relative_to(_ROOT).as_posix(), path.read_text()


def _u(node: ast.AST) -> str:
    return ast.unparse(node)


def _code(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _fn(src: str, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    found = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == name
    ]
    assert len(found) == 1, name
    return found[0]


def _blocks(fn: ast.AST) -> Iterable[list[ast.stmt]]:
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            if isinstance(block := getattr(n, field, None), list) and block:
                yield block


def _calls(node: ast.AST) -> list[str]:
    return [_u(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)]


# --------------------------------------------------------------------------------------------
# pin 1 (D1): one hook statement directly before every `_RUN_TASKS[...] = task`, six sites
# --------------------------------------------------------------------------------------------
_SITES = [
    (_RUNS, "create_run", 2), (_RUNS, "promote_to_live", 1), (_RUNS, "resume_run", 1),
    (_RUNS, "recover_orphaned_runs", 1), (_ORCH, "_auto_retry_paper_run", 1),
]  # fmt: skip


def _hook_flaws(src: str, func: str, expected: int) -> list[str]:
    bad, assigns = [], 0
    for block in _blocks(_fn(src, func)):
        for i, stmt in enumerate(block):
            if not (isinstance(stmt, ast.Assign) and _u(stmt.targets[0]).startswith("_RUN_TASKS[")):
                continue
            assigns += 1
            prev = _u(block[i - 1]) if i else ""
            args = block[i - 1].value.args if i and isinstance(block[i - 1], ast.Expr) else []  # type: ignore[attr-defined]
            if not prev.startswith("spawn_hooks.on_run_spawned("):
                bad.append("no hook statement directly before the registration")
            elif (
                len(args) != 5
                or _u(args[4]) != "task"
                or not (isinstance(args[1], ast.Constant) and args[1].value in ("paper", "live"))
            ):
                bad.append("the hook is not (run_id, mode, symbols, timeframe, task)")
    return bad + ([] if assigns == expected else [f"{assigns} registrations, expected {expected}"])


@pytest.mark.parametrize(("rel", "func", "n"), _SITES)
def test_pin1_every_spawn_site_has_its_hook_directly_before_the_registration(
    rel: str, func: str, n: int
) -> None:
    assert _hook_flaws(_read(rel), func, n) == []


def test_pin1_flags_planted_offenders() -> None:
    ok = _code(
        "def f(task):",
        "    spawn_hooks.on_run_spawned(rid, 'paper', syms, tf, task)",
        "    _RUN_TASKS[rid] = task",
    )
    assert _hook_flaws(ok, "f", 1) == []
    assert _hook_flaws(
        ok.replace("    spawn_hooks.on_run_spawned(rid, 'paper', syms, tf, task)\n", ""), "f", 1
    )
    after = _code(
        "def f(task):",
        "    _RUN_TASKS[rid] = task",
        "    spawn_hooks.on_run_spawned(rid, 'paper', syms, tf, task)",
    )
    assert _hook_flaws(after, "f", 1)
    between = ok.replace("    _RUN_TASKS", "    log.info('x')\n    _RUN_TASKS")
    assert _hook_flaws(between, "f", 1)
    assert _hook_flaws(ok.replace(", task)\n", ", other)\n", 1), "f", 1)


# --------------------------------------------------------------------------------------------
# pin 2 (D3 / RK-9703-57): the auto-retry site and the hook body
# --------------------------------------------------------------------------------------------
def _names(src: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)
    }


def _imports(src: str) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom):
            out |= {a.name for a in n.names} | {(n.module or "").split(".")[-1]}
        elif isinstance(n, ast.Import):
            out |= {a.name.split(".")[-1] for a in n.names}
    return out


def test_pin2_run_orchestrator_names_only_the_spawn_hook() -> None:
    src = _read(_ORCH)
    assert "spawn_hooks" in _imports(src) and "egress_budget" not in _imports(src)
    assert (
        "egress_budget" not in src
        and sum(c == "spawn_hooks.on_run_spawned" for c in _calls(ast.parse(src))) == 1
    )
    assert "spawn_hooks.on_run_spawned" in _calls(_fn(src, "_auto_retry_paper_run"))
    for rel, text in _py("packages/trading"):  # no engine module names a P7b gate
        assert not (_names(text) & _BANNED), rel
    assert not (_names(src) & _BANNED)


_HOOK_CALLS = {
    "egress_budget.register", "egress_budget.unregister", "task.add_done_callback", "_log.error",
    "str", "list", "on_run_ended",
}  # fmt: skip


def _hook_body_flaws(src: str) -> list[str]:
    tree, bad = ast.parse(src), []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)):
        if isinstance(fn, ast.AsyncFunctionDef) or any(
            isinstance(n, ast.Await) for n in ast.walk(fn)
        ):
            bad.append(f"{fn.name}: async or await")
        bad += [f"{fn.name}: call {c}" for c in _calls(fn) if c not in _HOOK_CALLS]
    return bad


def test_pin2_the_hook_module_has_no_await_and_only_the_allowed_calls() -> None:
    assert _hook_body_flaws(_read(_SH)) == []
    base = _code(
        "def on_run_spawned(task):",
        "    egress_budget.register(1)",
        "    task.add_done_callback(lambda _t: on_run_ended(1))",
    )
    assert _hook_body_flaws(base) == []
    assert _hook_body_flaws(base + "    requests.get('x')\n")
    assert _hook_body_flaws(base.replace("def ", "async def ", 1))
    assert _hook_body_flaws(base + "    await x()\n")
    assert "egress_budget" in _imports(
        _code("from api.services import egress_budget")
    )  # planted (a)


# --------------------------------------------------------------------------------------------
# pin 4 (C2): promote uses one timeframe value, no default
# --------------------------------------------------------------------------------------------
def _promote_flaws(src: str) -> list[str]:
    fn = _fn(src, "promote_to_live")
    text = _u(fn)
    reserve = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and _u(c.func) == "egress_budget.reserve_new_run"
    ]
    hook = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and _u(c.func) == "spawn_hooks.on_run_spawned"
    ]
    bad = [] if len(reserve) == 1 and len(hook) == 1 else ["one reserve and one hook call expected"]
    if not bad and (
        _u(reserve[0].args[3]) != _u(hook[0].args[3])
        or _u(reserve[0].args[2]) != _u(hook[0].args[2])
    ):
        bad.append("the check and the hook use different names")
    if re.search(r"_egress_timeframe\s*=\s*[^\n]*get\([^)]*,", text):
        bad.append("the timeframe has a default")
    return bad


def test_pin4_promote_computes_symbols_and_timeframe_once_without_a_default() -> None:
    src = _read(_RUNS)
    assert (
        _promote_flaws(src) == []
        and '_egress_timeframe = source_run.config.get("timeframe")' in src
    )
    ok = _code(
        "def promote_to_live():",
        "    _egress_timeframe = cfg.get('timeframe')",
        "    egress_budget.reserve_new_run(k, 'live', _egress_symbols, _egress_timeframe)",
        "    spawn_hooks.on_run_spawned(rid, 'live', _egress_symbols, _egress_timeframe, task)",
    )
    ok = ok.replace("def promote_to_live", "async def promote_to_live")
    assert _promote_flaws(ok) == []
    assert _promote_flaws(ok.replace("get('timeframe')", "get('timeframe', '1h')"))
    assert _promote_flaws(
        ok.replace("_egress_symbols, _egress_timeframe, task", "_egress_symbols, tf_val, task")
    )


# --------------------------------------------------------------------------------------------
# pin 6 (D2 / F2): the fetch-start check is the first statement of the fetch `try`
# --------------------------------------------------------------------------------------------
def _fetch_flaws(src: str) -> list[str]:
    fn = _fn(src, "_fetch_bars_for_backtest")
    tries = [n for n in fn.body if isinstance(n, ast.Try)]
    if len(tries) != 1:
        return ["one top-level try expected"]
    first = _u(tries[0].body[0])
    bad = [] if first.startswith("egress_budget.history_check(") else ["history_check is not first"]
    handlers = [_u(h.type) for h in tries[0].handlers]
    if not handlers or handlers[0] != "egress_budget.HistoryBudgetExhausted":
        bad.append("the exhaustion is not mapped before every other handler")
    return bad


def test_pin6_history_check_runs_before_the_cache_branch() -> None:
    assert _fetch_flaws(_read(_RUNS)) == []
    ok = _code(
        "def _fetch_bars_for_backtest():",
        "    try:",
        "        egress_budget.history_check(x)",
        "        if cached:",
        "            return 1",
        "    except egress_budget.HistoryBudgetExhausted:",
        "        raise",
        "    except MarketDataError:",
        "        raise",
    )
    assert _fetch_flaws(ok) == []
    late = ok.replace(
        "        egress_budget.history_check(x)\n        if cached:",
        "        if cached:\n            return 2\n        egress_budget.history_check(x)\n"
        "        if cached:",
    )
    assert _fetch_flaws(late)
    assert _fetch_flaws(
        ok.replace("    except egress_budget.HistoryBudgetExhausted:\n        raise\n", "")
    )


# --------------------------------------------------------------------------------------------
# pin 7 (D4 / C3): the processor is installed directly after configure_logging, once
# --------------------------------------------------------------------------------------------
def _install_flaws(src: str) -> list[str]:
    body = _fn(src, "lifespan").body
    at = [
        i
        for i, s in enumerate(body)
        if isinstance(s, ast.Expr) and _u(s.value).startswith("configure_logging(")
    ]
    if len(at) != 1:
        return ["one configure_logging statement expected"]
    nxt = _u(body[at[0] + 1]) if at[0] + 1 < len(body) else ""
    return (
        []
        if nxt == "egress_budget.install_log_processor()"
        else ["the install is not the next statement"]
    )


def test_pin7_install_follows_configure_logging_and_no_other_call_exists() -> None:
    assert _install_flaws(_read("apps/api/main.py")) == []
    sites = [rel for rel, s in _py("apps") if "configure_logging" in _calls(ast.parse(s))]
    assert sites == ["apps/api/main.py"]
    ok = _code(
        "async def lifespan():",
        "    configure_logging(a)",
        "    egress_budget.install_log_processor()",
        "    x()",
    )
    assert _install_flaws(ok) == []
    assert _install_flaws(
        ok.replace(
            "    configure_logging(a)\n    egress_budget.install_log_processor()\n",
            "    egress_budget.install_log_processor()\n    configure_logging(a)\n",
        )
    )
    assert _install_flaws(ok.replace("    egress_budget.install_log_processor()\n", "    y()\n"))


# --------------------------------------------------------------------------------------------
# pin 8 (C4, static half): the processor has no log call and no lock
# --------------------------------------------------------------------------------------------
def _processor_flaws(src: str) -> list[str]:
    fn = _fn(src, "live_429_processor")
    bad = [c for c in _calls(fn) if c.startswith(("_log", "log", "logger", "structlog"))]
    bad += [f"with {_u(w)}" for w in ast.walk(fn) if isinstance(w, ast.With | ast.AsyncWith)]
    bad += [c for c in _calls(fn) if c.endswith("acquire")]
    return bad


def test_pin8_the_processor_has_no_logger_call_lock_or_with() -> None:
    assert _processor_flaws(_read(_EB)) == []
    ok = "def live_429_processor(a, b, c):\n    return c\n"
    assert _processor_flaws(ok) == []
    assert _processor_flaws(ok.replace("    return c", "    _log.warning('x')\n    return c"))
    assert _processor_flaws(
        ok.replace("    return c", "    with _lock:\n        pass\n    return c")
    )
    assert _processor_flaws(ok.replace("    return c", "    _lock.acquire()\n    return c"))


# --------------------------------------------------------------------------------------------
# pin 10: module discipline
# --------------------------------------------------------------------------------------------
def _lock_log_flaws(src: str) -> list[str]:
    bad = []
    for w in (n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.With)):
        if any("_lock" in _u(i.context_expr) for i in w.items):
            bad += [c for c in _calls(w) if c.startswith(("_log", "logger", "log."))]
    return bad


def test_pin10_no_logging_inside_any_lock_block() -> None:
    assert _lock_log_flaws(_read(_EB)) == []
    assert _lock_log_flaws("def f():\n    with _lock:\n        _log.warning('x')\n")
    assert (
        _lock_log_flaws("def f():\n    with _lock:\n        x = 1\n    _log.warning('x')\n") == []
    )


_LOCK_FREE = {"_LIVE_RATE_LIMITED_UNTIL"}  # lock-free by design (TA-9703-12): one atomic store
_MUTATORS = {"pop", "popitem", "clear", "update", "setdefault"}


def _containers(src: str) -> set[str]:
    """Every module-level dict that is not a ``Final`` constant."""
    out: set[str] = set()
    for n in ast.parse(src).body:
        target = (
            n.target
            if isinstance(n, ast.AnnAssign)
            else n.targets[0]
            if isinstance(n, ast.Assign)
            else None
        )
        final = isinstance(n, ast.AnnAssign) and "Final" in _u(n.annotation)
        if (
            isinstance(target, ast.Name)
            and isinstance(getattr(n, "value", None), ast.Dict)
            and not final
        ):
            out.add(target.id)
    return out


def _unlocked_writes(src: str, names: set[str]) -> list[str]:
    """Every mutation of ``names`` (store, del, pop, clear, update, setdefault) is locked."""
    bad: list[str] = []

    def visit(node: ast.AST, locked: bool, fn: str) -> None:
        if isinstance(node, ast.FunctionDef):
            fn = node.name
        if isinstance(node, ast.With) and any("_lock" in _u(i.context_expr) for i in node.items):
            locked = True
        sub = isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del)
        call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
        )
        target = node.value if sub else node.func.value if call else None  # type: ignore[attr-defined]
        if target is not None and _u(target) in names and not locked and fn != "_reset_for_tests":
            bad.append(f"{fn}: unlocked write to {_u(target)}")
        for child in ast.iter_child_nodes(node):
            visit(child, locked, fn)

    visit(ast.parse(src), False, "<module>")
    return bad


def test_pin10_every_write_to_the_budget_state_is_under_the_lock() -> None:
    src = _read(_EB)
    locked = {"_weights", "_reserved", "_owner", "_next_slot", "history_suspended_until"}
    assert _containers(src) == locked | _LOCK_FREE  # a new container forces a review of this pin
    assert _unlocked_writes(src, locked) == []
    # the named exception is real: the lock-free dict IS written outside the lock (by the processor)
    assert {w.split(":")[0] for w in _unlocked_writes(src, _LOCK_FREE)} == {"live_429_processor"}
    for name in locked:  # each container, each mutation form, planted
        for stmt in (
            f"{name}['a'] = 1",
            f"{name}.pop('a')",
            f"{name}.popitem()",
            f"{name}.clear()",
            f"del {name}['a']",
            f"{name}.update(a=1)",
            f"{name}.setdefault('a', 1)",
        ):
            assert _unlocked_writes(_code("def f():", f"    {stmt}"), {name}), stmt
            assert (
                _unlocked_writes(_code("def f():", "    with _lock:", f"        {stmt}"), {name})
                == []
            )


def test_pin10_the_exhaustion_is_a_plain_exception_with_a_fixed_message() -> None:
    cls = next(
        n
        for n in ast.parse(_read(_EB)).body
        if isinstance(n, ast.ClassDef) and n.name == "HistoryBudgetExhausted"
    )
    assert [_u(b) for b in cls.bases] == ["Exception"]
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef))
    assert [a.arg for a in init.args.args] == [
        "self"
    ] and "super().__init__('history fetch budget exhausted')" in _u(init)


def _importers(name: str) -> set[str]:
    return {
        rel
        for rel, s in _py("apps", "packages")
        if not rel.endswith(f"/{name}.py") and name in _imports(s)
    }


def test_pin10_importer_lists_and_call_sites_are_exact() -> None:
    r, s = "apps/api/routers/", "apps/api/services/"
    assert _importers("egress_budget") == {
        r + "runs.py",
        r + "ml.py",
        s + "retraining.py",
        _SH,
        "apps/api/main.py",
    }
    assert _importers("spawn_hooks") == {r + "runs.py", _ORCH}
    sites = {
        (rel, fn.name)
        for rel, text in _py("apps", "packages")
        for fn in ast.walk(ast.parse(text))
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and _u(c.func) == "egress_budget.reserve_history_blocking"
    }
    assert sites == {
        (r + "ml.py", "_train_model_sync"),
        (r + "ml.py", "_train_model_with_wf_sync"),
        (s + "retraining.py", "_fetch_ohlcv_sync"),
    }


def test_pin10_settings_gain_no_egress_field() -> None:
    from api.config import Settings

    assert not [
        f
        for f in Settings.model_fields
        if re.search(r"egress|budget|cooldown|history_rate|spawn", f)
    ]


# --------------------------------------------------------------------------------------------
# pin 11 (static half): the protection handlers name no budget gate
# --------------------------------------------------------------------------------------------
def test_pin11_protection_handlers_name_no_egress_gate() -> None:
    for rel in (
        "apps/api/routers/emergency.py",
        "apps/api/routers/health.py",
        "apps/api/services/kill_switch.py",
    ):
        assert "egress_budget" not in _read(rel) and "spawn_hooks" not in _read(rel), rel
    runs = _read(_RUNS)
    for name in ("stop_run", "resume_run"):
        assert "egress_budget" not in _u(_fn(runs, name)), name


# --------------------------------------------------------------------------------------------
# pin 12: the constants
# --------------------------------------------------------------------------------------------
def test_pin12_constants_are_exact_and_final() -> None:
    from api.services import egress_budget as eb

    assert (
        eb.CAPS == {"kraken": 20.0, "binance": 100.0, "coinbase": 50.0} and eb.UNKNOWN_CAP == 20.0
    )
    assert (
        eb.HISTORY_RATE == {"kraken": 0.2, "coinbase": 3.0, "binance": 5.0}
        and eb.UNKNOWN_RATE == 0.5
    )
    assert (eb.RATE_FLOOR, eb.MIN_SPACING_S, eb.MAX_WAIT_S) == (0.05, 0.1, 5.0)
    assert (eb.COOLDOWN_S, eb.DEFAULT_COOLDOWN_S) == ((30.0, 600.0), 60.0)
    assert (eb.LIVE_SUSPEND_S, eb.DDOS_SUSPEND_S, eb.RESERVATION_TTL_S) == (60.0, 600.0, 120.0)
    final = {
        t.target.id for t in ast.parse(_read(_EB)).body
        if isinstance(t, ast.AnnAssign) and "Final" in _u(t.annotation)
    }  # fmt: skip
    assert final >= {
        "CAPS",
        "UNKNOWN_CAP",
        "HISTORY_RATE",
        "UNKNOWN_RATE",
        "MIN_SPACING_S",
        "RATE_FLOOR",
        "MAX_WAIT_S",
        "COOLDOWN_S",
        "DEFAULT_COOLDOWN_S",
        "LIVE_SUSPEND_S",
        "DDOS_SUSPEND_S",
        "RESERVATION_TTL_S",
    }


# --------------------------------------------------------------------------------------------
# pin 14 / 15: the riders (CF-97-26, CF-97-31(b), CF-97-31(c))
# --------------------------------------------------------------------------------------------
def _rider_flaws(adm: str, ret: str) -> list[str]:
    bad = []
    lease_cls = next(
        c for c in ast.parse(adm).body if isinstance(c, ast.ClassDef) and c.name == "Lease"
    )
    release = next(
        f for f in lease_cls.body if isinstance(f, ast.FunctionDef) and f.name == "release"
    )
    if "_IN_RESEARCH.reset(self.token)" not in _u(release):
        bad.append("release does not reset the token")
    if "_IN_RESEARCH" in _u(_fn(adm, "_free")):
        bad.append("_free touches the research flag from a callback")
    fin = [
        t
        for t in ast.walk(_fn(ret, "_check_and_retrain"))
        if isinstance(t, ast.Try) and t.finalbody
    ]
    if not (
        len(fin) == 1
        and [_u(s) for s in fin[0].finalbody] == ["lease.release()"]
        and "asyncio.shield(job)" in _u(fin[0])
    ):
        bad.append("the scheduler does not release in a finally around the job await")
    body = [_u(s) for s in _fn(adm, "try_acquire").body]
    gauge = next((i for i, s in enumerate(body) if "metrics.gauge(" in s), None)
    slot = next((i for i, s in enumerate(body) if s.startswith("_held = ")), None)
    if gauge is None or slot is None or gauge > slot:
        bad.append("the gauge does not precede the slot assignment")
    return bad


def test_pin14_the_riders_hold_and_the_planted_offenders_are_flagged() -> None:
    adm, ret = _read(_ADM), _read("apps/api/services/retraining.py")
    assert _rider_flaws(adm, ret) == []
    assert _rider_flaws(adm.replace("_IN_RESEARCH.reset(self.token)", "pass"), ret)  # M19
    assert _rider_flaws(
        adm.replace(
            "            _held = None\n",
            "            _held = None\n            _IN_RESEARCH.set(False)\n",
        ),
        ret,
    )
    assert _rider_flaws(
        adm,
        ret.replace(
            "            finally:\n                lease.release()",
            "            finally:\n                pass",
        ),
    )  # M20
    gauge = '    metrics.gauge("admission_slot_held", 1.0)'
    moved = adm.replace(
        gauge + "  # CF-97-31: before the slot, so a failure leaks none\n", ""
    ).replace("    return lease\n", gauge + "\n    return lease\n", 1)
    assert _rider_flaws(moved, ret)


_LOG_METHODS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "log",
    "fatal",
}


def _listener_flaws(src: str) -> list[str]:
    """CF-97-31(b): every pool listener is one ``with suppress``; no log call, by call TARGET."""
    bad, found = [], 0
    for fn in ast.walk(ast.parse(src)):
        if not (
            isinstance(fn, ast.FunctionDef)
            and any("listens_for" in _u(d) for d in fn.decorator_list)
        ):
            continue
        found += 1
        if not (
            len(fn.body) == 1
            and isinstance(fn.body[0], ast.With)
            and _u(fn.body[0].items[0].context_expr) == "contextlib.suppress(Exception)"
        ):
            bad.append(f"{fn.name}: not one suppress block")
            continue
        for c in ast.walk(fn.body[0]):
            if (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr in _LOG_METHODS
            ):
                bad.append(f"{fn.name}: log call {_u(c.func)}")
            if isinstance(c, ast.Await | ast.Raise):
                bad.append(f"{fn.name}: await or raise")
    return bad if found >= 2 else [*bad, "fewer than two listeners"]


def test_pin15b_the_listener_pin_detects_a_log_call_by_target() -> None:
    assert _listener_flaws(_read(_ADM)) == []
    ok = _code(
        "def install():", "    @event.listens_for(P, 'checkout')", "    def _out(d, r, p):",
        "        with contextlib.suppress(Exception):", "            r.info['x'] = 1",
        "    @event.listens_for(P, 'checkin')", "    def _in(d, r):",
        "        with contextlib.suppress(Exception):", "            r.info.pop('x', None)",
    )  # fmt: skip
    assert _listener_flaws(ok) == []
    for plant in (
        "_log.warning('x')",
        "structlog.get_logger().info('x')",
        "get_log().error('x')",
        "logging.getLogger('a').warning('x')",
    ):
        assert _listener_flaws(ok.replace("r.info['x'] = 1", plant)), plant


# --------------------------------------------------------------------------------------------
# pin 9 (D5, static half): the cooldown starts before the log line
# --------------------------------------------------------------------------------------------
def _cooldown_order_flaws(src: str) -> list[str]:
    handler = next(
        h
        for t in ast.walk(_fn(src, "_fetch_bars_for_backtest"))
        if isinstance(t, ast.Try)
        for h in t.handlers
        if _u(h.type or ast.Constant(0)) == "MarketDataError"
    )
    calls = [
        i for i, st in enumerate(handler.body) for c in _calls(st) if c.endswith("suspend_history")
    ]
    logs = [i for i, st in enumerate(handler.body) for c in _calls(st) if c.startswith("log.")]
    return (
        []
        if calls and logs and calls[0] < logs[0]
        else ["suspend_history does not precede the log call"]
    )


def test_pin9_suspend_history_runs_before_the_log_line() -> None:
    assert _cooldown_order_flaws(_read(_RUNS)) == []
    ok = _code(
        "def _fetch_bars_for_backtest():", "    try:", "        pass",
        "    except MarketDataError as exc:",
        "        egress_budget.suspend_history(x)", "        log.error('e')", "        raise",
    )  # fmt: skip
    assert _cooldown_order_flaws(ok) == []
    assert _cooldown_order_flaws(
        ok.replace(
            "        egress_budget.suspend_history(x)\n        log.error('e')",
            "        log.error('e')\n        egress_budget.suspend_history(x)",
        )
    )
