"""CF-97-03 P7a (T-m): AST and text pins of the research admission slot, with planted offenders.

The pins read source only (no import of the app).  Each pin is a function over source text, so the
same function runs on the real files and on a planted offender that it must flag.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ADM = "apps/api/services/admission.py"
_SE = "SE-9703-29/30"


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text()


def _py(*roots: str) -> Iterable[tuple[str, str]]:
    for root in roots:
        for path in sorted((_ROOT / root).rglob("*.py")):
            yield path.relative_to(_ROOT).as_posix(), path.read_text()


def _walk(node: ast.AST, func: str = "<module>") -> Iterable[tuple[ast.AST, str]]:
    for c in ast.iter_child_nodes(node):
        f = c.name if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef) else func
        yield c, f
        yield from _walk(c, f)


def _fn(src: str, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    found = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == name
    ]
    assert len(found) == 1, name
    return found[0]


def _code(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _u(node: ast.AST) -> str:
    return ast.unparse(node)


def _awaits(nodes: Iterable[ast.AST]) -> bool:
    return any(isinstance(x, ast.Await) for n in nodes for x in ast.walk(n))


def _has_acquire(node: ast.AST) -> bool:
    return any(
        isinstance(c, ast.Call) and _u(c.func).endswith("try_acquire") for c in ast.walk(node)
    )


# --------------------------------------------------------------------------------------------
# the acquire shapes (C3, with the D2 restatement for the create site)
# --------------------------------------------------------------------------------------------
def _create_shape(src: str, func: str) -> list[str]:
    body, bad = _fn(src, func).body, []
    spots = [i for i, n in enumerate(body) if isinstance(n, ast.If) and _has_acquire(n)]
    if len(spots) != 1:
        return ["exactly one If holds the acquire"]
    node, nxt = body[spots[0]], (body[spots[0] + 1 : spots[0] + 2] or [None])[0]
    assert isinstance(node, ast.If)
    if not isinstance(nxt, ast.Try):
        return ["the If is not directly followed by the existing Try"]
    if [_u(s) for s in nxt.finalbody] != ["lease.release()"]:
        bad.append("lease.release() is not the only finalbody statement")
    inner = node.body[0] if len(node.body) == 1 else None
    if not (isinstance(inner, ast.Try) and len(inner.body) == 1 and _has_acquire(inner.body[0])):
        return [*bad, "the acquire is not the sole statement of an inner try"]
    if _awaits(inner.body) or _awaits(node.orelse):
        bad.append("an await lies next to the acquire")
    if len(inner.handlers) != 1 or _u(inner.handlers[0].type) != "admission.AdmissionRefused":  # type: ignore[arg-type]
        return [*bad, "the only handler must be except AdmissionRefused"]
    h = inner.handlers[0].body
    if not (len(h) == 2 and _u(h[0]).startswith("await store.fail(") and _u(h[1]) == "raise"):
        bad.append("the handler is not `await store.fail(...)` then a bare raise")
    if [_u(s) for s in node.orelse] != ["lease = admission.NO_LEASE"]:
        bad.append("the else does not bind NO_LEASE")
    return bad


def _try_shape(src: str, func: str) -> list[str]:
    """optimize: ``lease = try_acquire(...)`` directly followed by the Try that releases it."""
    body = _fn(src, func).body
    spots = [i for i, n in enumerate(body) if isinstance(n, ast.Assign) and _has_acquire(n)]
    if len(spots) != 1 or len(body) <= spots[0] + 1:
        return ["exactly one acquire statement, followed by something"]
    nxt = body[spots[0] + 1]
    if not isinstance(nxt, ast.Try):
        return ["the acquire is not directly followed by a Try"]
    return (
        [] if [_u(s) for s in nxt.finalbody] == ["lease.release()"] else ["no `finally: release`"]
    )


def _blocks(fn: ast.AST) -> Iterable[list[ast.stmt]]:
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            if (
                isinstance(block := getattr(n, field, None), list)
                and block
                and isinstance(block[0], ast.stmt)
            ):
                yield block


def _thread_shape(src: str, func: str) -> list[str]:
    """ML and retraining: in one block, the acquire is bound to its future with no await between."""
    pairs = []
    for block in _blocks(_fn(src, func)):
        for i, stmt in enumerate(block):
            for j in range(i + 1, len(block)):
                bound = isinstance(block[j], ast.Expr) and _u(block[j].value).startswith(
                    "lease.bind_thread("
                )  # type: ignore[attr-defined]
                if _has_acquire(stmt) and bound:
                    pairs.append(block[i : j + 1])
    if len(pairs) != 1:
        return ["the acquire is not bound to a future in the block that holds it"]
    return ["an await lies between the acquire and the bind"] if _awaits(pairs[0]) else []


_SITES = {
    ("apps/api/routers/runs.py", "create_run"): _create_shape,
    ("apps/api/routers/optimize.py", "run_optimization"): _try_shape,
    ("apps/api/routers/ml.py", "train_model"): _thread_shape,
    ("apps/api/routers/ml.py", "manual_retrain"): _thread_shape,
    ("apps/api/services/retraining.py", "_check_and_retrain"): _thread_shape,
}
_CREATE_OK = """
async def create_run(store):
    if is_backtest:
        try:
            lease = admission.try_acquire("backtest")
        except admission.AdmissionRefused:
            await store.fail(key=k)
            raise
    else:
        lease = admission.NO_LEASE
    try:
        pass
    finally:
        lease.release()
"""


@pytest.mark.parametrize(("rel", "func"), list(_SITES))
def test_m_every_acquire_has_its_pinned_shape(rel: str, func: str) -> None:
    assert _SITES[(rel, func)](_read(rel), func) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace(
            "    try:\n        pass", "    await asyncio.sleep(0)\n    try:\n        pass"
        ),
        lambda s: s.replace("            raise\n", "            return\n"),
        lambda s: s.replace(
            "            raise\n",
            "            raise\n        except Exception:\n            pass\n",
        ),
        lambda s: s.replace(
            '            lease = admission.try_acquire("backtest")\n',
            '            lease = admission.try_acquire("backtest")\n            await x()\n',
        ),
        lambda s: s.replace("        lease.release()", "        pass"),
        lambda s: s.replace("        lease = admission.NO_LEASE", "        lease = None"),
    ],
    ids=[
        "await_before_try",
        "no_raise",
        "second_handler",
        "await_after_acquire",
        "no_finally",
        "no_null_lease",
    ],
)
def test_m_the_create_pin_flags_planted_offenders(mutate: object) -> None:
    assert _create_shape(_CREATE_OK, "create_run") == []
    assert _create_shape(mutate(_CREATE_OK), "create_run")  # type: ignore[operator]


def test_m_the_try_and_thread_pins_flag_planted_offenders() -> None:
    ok = _code(
        "async def f():", "    lease = admission.try_acquire('optimize')", "    try:",
        "        pass", "    finally:", "        lease.release()",
    )  # fmt: skip
    assert _try_shape(ok, "f") == []
    assert _try_shape(ok.replace("lease.release()", "pass"), "f")
    assert _try_shape(
        ok.replace("    try:", "    await x()\n    try:"), "f"
    )  # an await before the try
    assert _try_shape(
        "async def f():\n    lease = admission.try_acquire('o')\n    await x()\n", "f"
    )
    good = _code(
        "async def f():", "    lease = admission.try_acquire('ml')", "    task = make()",
        "    lease.bind_thread(task)",
    )  # fmt: skip
    assert _thread_shape(good, "f") == []
    assert _thread_shape(good.replace("    task = make()", "    task = await make()"), "f")
    assert _thread_shape(good.replace("    lease.bind_thread(task)\n", ""), "f")  # unbound acquire


# --------------------------------------------------------------------------------------------
# D4: AdmissionRefused is an HTTPException, so no acquire may sit where a broad handler sees it
# --------------------------------------------------------------------------------------------
_BROAD = {"Exception", "BaseException", "HTTPException"}


def _broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return bool({_u(k).split(".")[-1] for k in kinds} & _BROAD)


def _acquires_under_a_broad_handler(src: str) -> list[int]:
    out: list[int] = []

    def visit(node: ast.AST, bad: bool) -> None:
        if isinstance(node, ast.Call) and _u(node.func).endswith("try_acquire") and bad:
            out.append(node.lineno)
        if isinstance(node, ast.Try):
            covered = bad or any(_broad(h) for h in node.handlers)
            for c in node.body:
                visit(c, covered)
            for h in node.handlers:
                for c in h.body:
                    visit(c, bad or _broad(h))
            for c in [*node.orelse, *node.finalbody]:
                visit(c, bad)
        else:
            for c in ast.iter_child_nodes(node):
                visit(c, bad)

    visit(ast.parse(src), False)
    return out


@pytest.mark.parametrize("rel", sorted({r for r, _ in _SITES}))
def test_m_no_acquire_lies_under_a_broad_handler(rel: str) -> None:
    assert _acquires_under_a_broad_handler(_read(rel)) == []


def _try_acquire_in(kind: str, handler: str) -> str:
    if kind == "handler":
        return _code("def f():", "    try:", "        g()", f"    except {handler}:",
                     "        lease = admission.try_acquire('x')")  # fmt: skip
    return _code("def f():", "    try:", "        lease = admission.try_acquire('x')",
                 f"    except {handler}:", "        pass")  # fmt: skip


@pytest.mark.parametrize(
    ("kind", "handler"),
    [("body", "Exception"), ("body", "(ValueError, HTTPException)"), ("handler", "Exception"),
     ("body", "")],
    ids=["except_exception", "except_http", "inside_the_handler", "bare_except"],
)  # fmt: skip
def test_m_the_broad_handler_pin_flags_planted_offenders(kind: str, handler: str) -> None:
    assert _acquires_under_a_broad_handler(_try_acquire_in(kind, handler))
    assert (
        _acquires_under_a_broad_handler(_try_acquire_in("body", "admission.AdmissionRefused")) == []
    )


# --------------------------------------------------------------------------------------------
# call sites, importers and the never-gated paths
# --------------------------------------------------------------------------------------------
def _call_sites(rel: str, src: str) -> set[tuple[str, str]]:
    return {
        (rel, f)
        for c, f in _walk(ast.parse(src))
        if isinstance(c, ast.Call) and _u(c.func).endswith("try_acquire")
    }


def _imports_admission(src: str) -> bool:
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and (
            (n.module or "").endswith("admission") or any(a.name == "admission" for a in n.names)
        ):
            return True
        if isinstance(n, ast.Import) and any(a.name.endswith("admission") for a in n.names):
            return True
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "services.admission" in n.value
        ):
            return True
    return False


_NEVER_GATED = (  # SE-9703-09 as amended: nothing here may import or call the slot
    "apps/api/routers/emergency.py", "apps/api/routers/health.py",
    "apps/api/routers/circuit_breaker.py", "apps/api/services/kill_switch.py",
    "packages/trading/strategy_engine.py", "packages/data/services/ccxt_market_data.py",
)  # fmt: skip


def test_m_the_acquire_call_sites_and_importers_are_exactly_the_five_and_four() -> None:
    sites, importers = set(), set()
    for rel, src in _py("apps", "packages"):
        if rel == _ADM:
            continue
        sites |= _call_sites(rel, src)
        if _imports_admission(src):
            importers.add(rel)
    assert sites == set(_SITES), "a new acquire site needs a review (SY-9703R-04)"
    assert importers == {r for r, _ in _SITES}


def test_m_the_never_gated_paths_never_name_the_slot() -> None:
    for rel in _NEVER_GATED:
        assert (_ROOT / rel).exists(), rel
        assert "admission" not in _read(rel), rel
    orchestrator = _read("apps/api/services/run_orchestrator.py")
    assert "admission" not in orchestrator and not _call_sites("x", orchestrator)
    assert not any(
        rel.startswith("packages/") and _imports_admission(s) for rel, s in _py("packages")
    )


def test_m_the_site_and_import_pins_flag_planted_offenders() -> None:
    live = "async def run_live_engine():\n    lease = admission.try_acquire('backtest')\n"
    assert _call_sites("apps/api/services/run_orchestrator.py", live) == {
        ("apps/api/services/run_orchestrator.py", "run_live_engine")
    }
    for planted in (
        "from api.services import admission\n", "from api.services.admission import try_acquire\n",
        "import api.services.admission\n",
        "import importlib\nimportlib.import_module('api.services.admission')\n",
    ):  # fmt: skip
        assert _imports_admission(planted), planted
    assert not _imports_admission("from api.services import idempotency\n")


# --------------------------------------------------------------------------------------------
# egress: every non-live history/exchange fetch site is inside a lease; one slot (SE-9703-29/30)
# --------------------------------------------------------------------------------------------
_EGRESS = {
    "_fetch_bars_for_backtest",
    "fetch_history_bars",
    "fetch_ohlcv_range",
    "fetch_ohlcv_range_ex",
    "fetch_ohlcv",
}
_EGRESS_SITES = {
    ("apps/api/routers/ml.py", "_train_model_sync", "fetch_ohlcv"),
    ("apps/api/routers/ml.py", "_train_model_with_wf_sync", "fetch_ohlcv"),
    ("apps/api/routers/optimize.py", "run_optimization", "_fetch_bars_for_backtest"),
    ("apps/api/routers/runs.py", "_fetch_bars_for_backtest", "fetch_history_bars"),
    ("apps/api/routers/runs.py", "_fetch_bars_for_backtest", "fetch_ohlcv_range"),
    ("apps/api/routers/runs.py", "create_run", "_fetch_bars_for_backtest"),
    ("apps/api/services/ohlcv_history.py", "_bypass", "fetch_ohlcv_range_ex"),
    ("apps/api/services/ohlcv_history.py", "fetch_range", "fetch_ohlcv_range_ex"),
    ("apps/api/services/retraining.py", "_fetch_ohlcv_sync", "fetch_ohlcv"),
}
# who may NAME the entry points that reach an egress site (each is reachable only after an acquire)
_ENTRIES = {
    "_train_model_with_wf_sync": {("apps/api/routers/ml.py", "train_model")},
    "_fetch_ohlcv_sync": {("apps/api/services/retraining.py", "_do_retrain")},
    "fetch_history_bars": {("apps/api/routers/runs.py", "_fetch_bars_for_backtest")},
    "_do_retrain": {
        ("apps/api/services/retraining.py", "_check_and_retrain"),
        ("apps/api/services/retraining.py", "manual_retrain"),
    },
    "manual_retrain": {("apps/api/routers/ml.py", "manual_retrain")},
}


def _egress(
    rel: str, src: str
) -> tuple[set[tuple[str, str, str]], dict[str, set[tuple[str, str]]]]:
    sites: set[tuple[str, str, str]] = set()
    named: dict[str, set[tuple[str, str]]] = {}
    for c, f in _walk(ast.parse(src)):
        name = getattr(c, "attr", None) or getattr(c, "id", None)
        if (
            isinstance(c, ast.Call)
            and (cn := getattr(c.func, "attr", None) or getattr(c.func, "id", None)) in _EGRESS
        ):
            sites.add((rel, f, cn))
        if isinstance(c, ast.Name | ast.Attribute) and name in _ENTRIES:
            named.setdefault(name, set()).add((rel, f))
    return sites, named


def test_m_the_egress_allow_list_is_exhaustive_and_reachable_only_after_an_acquire() -> None:
    sites: set[tuple[str, str, str]] = set()
    named: dict[str, set[tuple[str, str]]] = {}
    for rel, src in _py("apps"):
        found, refs = _egress(rel, src)
        sites |= found
        for k, v in refs.items():
            named.setdefault(k, set()).update(v)
    assert sites == _EGRESS_SITES, f"an egress site outside the allow-list needs a lease ({_SE})"
    assert named == _ENTRIES


def _in_lease_try(src: str, func: str, callee: str) -> bool:
    """The call to ``callee`` lies inside a Try whose finalbody is ``lease.release()``."""
    covered: list[bool] = []

    def visit(node: ast.AST, inside: bool) -> None:
        if isinstance(node, ast.Call) and _u(node.func).split(".")[-1] == callee:
            covered.append(inside)
        leased = isinstance(node, ast.Try) and [_u(s) for s in node.finalbody] == [
            "lease.release()"
        ]
        for c in ast.iter_child_nodes(node):
            visit(c, inside or (leased and c in node.body))  # type: ignore[attr-defined]

    visit(_fn(src, func), False)
    return bool(covered) and all(covered)


@pytest.mark.parametrize(
    ("rel", "func"),
    [
        ("apps/api/routers/runs.py", "create_run"),
        ("apps/api/routers/optimize.py", "run_optimization"),
    ],
)
def test_m_a_history_fetch_of_a_route_runs_inside_its_lease(rel: str, func: str) -> None:
    assert _in_lease_try(_read(rel), func, "_fetch_bars_for_backtest")


def test_m_the_egress_pins_flag_planted_offenders() -> None:
    route = "async def planted_route():\n    bars = await _fetch_bars_for_backtest(a, b)\n"
    assert _egress("apps/api/routers/x.py", route)[0] == {
        ("apps/api/routers/x.py", "planted_route", "_fetch_bars_for_backtest")
    }
    outside = _code(
        "async def f():", "    lease = admission.try_acquire('o')", "    try:", "        pass",
        "    finally:", "        lease.release()", "    await _fetch_bars_for_backtest(a)",
    )  # fmt: skip
    assert not _in_lease_try(outside, "f", "_fetch_bars_for_backtest")
    inside = (
        "async def f():\n    lease = admission.try_acquire('o')\n    try:\n"
        "        await _fetch_bars_for_backtest(a)\n    finally:\n        lease.release()\n"
    )
    assert _in_lease_try(inside, "f", "_fetch_bars_for_backtest")
    assert _egress("x.py", "def g():\n    _do_retrain()\n")[1] == {"_do_retrain": {("x.py", "g")}}


def _slot_names(src: str) -> set[str]:
    out: set[str] = set()
    for n in ast.parse(src).body:
        targets = (
            [n.target]
            if isinstance(n, ast.AnnAssign)
            else n.targets
            if isinstance(n, ast.Assign)
            else []
        )
        out |= {
            t.id
            for t in targets
            if isinstance(t, ast.Name) and re.search(r"held|slot|permit|capacity|sem", t.id, re.I)
        }
    return out


def _refuse_test(src: str) -> set[str]:
    first = next(n for n in _fn(src, "try_acquire").body if isinstance(n, ast.If))
    return {n.id for n in ast.walk(first.test) if isinstance(n, ast.Name)}


def test_m_the_slot_count_is_one() -> None:
    msg = (
        f"a second slot must first re-introduce single-flight under {_SE}, or prove no shared fetch"
    )
    src = _read(_ADM)
    assert _slot_names(src) == {"_held"}, msg
    assert _refuse_test(src) <= {"_held", "_leaked"}, msg
    assert _slot_names("_held = None\n_held2 = None\n") == {"_held", "_held2"}
    assert _slot_names("_slots = [None, None]\n") == {"_slots"}
    assert _refuse_test(
        "def try_acquire(k):\n    if _held is not None and len(_active) >= 2:\n        pass\n"
    ) - {"_held"}


# --------------------------------------------------------------------------------------------
# the module itself: no asyncio primitive at import, one reset, no Settings field
# --------------------------------------------------------------------------------------------
_PRIMITIVES = {
    "Lock",
    "Semaphore",
    "BoundedSemaphore",
    "Event",
    "Condition",
    "Queue",
    "Future",
    "get_event_loop",
    "get_running_loop",
    "create_task",
    "ensure_future",
}


def _import_time_primitives(src: str) -> list[str]:
    out = []
    for stmt in ast.parse(src).body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        out += [
            _u(n) for n in ast.walk(stmt) if isinstance(n, ast.Attribute) and n.attr in _PRIMITIVES
        ]
    return out


def _listener_flaws(src: str) -> list[str]:
    """Every ``event.listens_for`` function is one ``with contextlib.suppress(Exception)``."""
    bad, found = [], 0
    for fn in ast.walk(ast.parse(src)):
        if not (
            isinstance(fn, ast.FunctionDef)
            and any("listens_for" in _u(d) for d in fn.decorator_list)
        ):
            continue
        found += 1
        body = fn.body
        if not (
            len(body) == 1
            and isinstance(body[0], ast.With)
            and _u(body[0].items[0].context_expr) == "contextlib.suppress(Exception)"
        ):
            bad.append(f"{fn.name}: the body is not one `with contextlib.suppress(Exception)`")
            continue
        for n in ast.walk(body[0]):
            logged = isinstance(n, ast.Call) and _u(n.func).startswith(("_log.", "logger.", "log."))
            if isinstance(n, ast.Await | ast.Raise) or logged:
                bad.append(f"{fn.name}: an await, raise or log call inside")
    return bad if found >= 2 else [*bad, "fewer than two listeners found"]


_LISTENERS = _code(
    "def install():", "    @event.listens_for(Pool, 'checkout')", "    def _out(d, r, p):",
    "        with contextlib.suppress(Exception):", "            r.info['x'] = 1",
    "    @event.listens_for(Pool, 'checkin')", "    def _in(d, r):",
    "        with contextlib.suppress(Exception):", "            r.info.pop('x', None)",
)  # fmt: skip


def test_m_the_pool_listeners_cannot_raise_into_a_checkout() -> None:
    assert _listener_flaws(_read(_ADM)) == [] and _listener_flaws(_LISTENERS) == []
    wrapped = "        with contextlib.suppress(Exception):\n            r.info['x'] = 1"
    plants = (
        (wrapped, "        r.info['x'] = 1"),
        ("r.info['x'] = 1", "raise RuntimeError"),
        ("r.info['x'] = 1", "await r.go()"),
        ("r.info['x'] = 1", "_log.warning('x')"),
    )
    for old, new in plants:
        assert _listener_flaws(_LISTENERS.replace(old, new, 1)), new


def test_m_no_asyncio_primitive_exists_at_import_and_there_is_one_reset() -> None:
    src = _read(_ADM)
    assert _import_time_primitives(src) == []
    assert (
        len(
            [
                n
                for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_reset_for_tests"
            ]
        )
        == 1
    )
    assert _import_time_primitives("import asyncio\n_lock = asyncio.Lock()\n") == ["asyncio.Lock"]
    assert _import_time_primitives("import asyncio\nQ = [asyncio.Queue()]\n") == ["asyncio.Queue"]
    assert _import_time_primitives("import asyncio\ndef f():\n    return asyncio.Lock()\n") == []


def test_m_settings_gain_no_admission_field() -> None:
    from api.config import Settings

    assert not [
        f
        for f in Settings.model_fields
        if re.search(r"admission|hold_long|retry_after|research|slot", f)
    ]
