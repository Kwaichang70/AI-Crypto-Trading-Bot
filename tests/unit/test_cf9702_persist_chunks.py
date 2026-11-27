"""CF-97-02 (T-9702-01, 04, 05): backtest persistence is built, added, flushed, yielded in chunks.

No wall time and no Postgres: a recording session and spies on the two name bindings of
``_yield_to_loop`` (``run_persistence`` and ``backtest``).  The engine's own binding is left alone:
``test_cf2697_yield.py`` patches that one and must not see the new yields.  Every AST pin has a
planted-offender control.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import uuid
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import structlog

from api.db.models import RunORM
from api.services import run_persistence as rp
from common.types import TimeFrame
from tests._cf2697_fingerprint import make_bars
from trading import backtest as bt
from trading.backtest import BacktestRunner
from trading.strategies import MACrossoverStrategy

_ROOT = Path(__file__).resolve().parents[2]
_RP = (_ROOT / "apps/api/services/run_persistence.py").read_text()
_BT = (_ROOT / "packages/trading/backtest.py").read_text()
_RANK = {"TradeORM": 0, "EquitySnapshotORM": 1, "OrderORM": 2, "FillORM": 3}
_RANK["PositionSnapshotORM"] = 4
_SMALL = 7


def _runner() -> BacktestRunner:
    strategy = MACrossoverStrategy(strategy_id="c", params={"fast_period": 5, "slow_period": 20})
    return BacktestRunner([strategy], ["BTC/USDT"], TimeFrame.ONE_MINUTE, seed=7)


@pytest.fixture(scope="module")
def run_result() -> tuple[Any, Any, Any]:
    runner = _runner()
    result = asyncio.run(runner.run({"BTC/USDT": make_bars("BTC/USDT", 1500, 7)}))
    ex, pf = runner.last_execution_engine, runner.last_portfolio
    assert min(len(result.trades), len(ex.get_all_orders()), len(ex.get_all_fills())) > 2 * _SMALL
    return result, ex, pf


class _Rec:
    def __init__(self, events: list[tuple[Any, ...]], keep: list[Any] | None = None) -> None:
        self.events, self.keep = events, keep

    def add_all(self, items: Any) -> None:
        items = list(items)
        self.events.append(("add", type(items[0]).__name__, len(items)))
        if self.keep is not None:
            self.keep.extend(items)

    async def flush(self) -> None:
        self.events.append(("flush",))


async def _persist(
    run_result: Any,
    mp: pytest.MonkeyPatch,
    chunk: int | None = None,
    keep: list[Any] | None = None,
    on_yield: Callable[[], Any] = bool,
) -> list[Any]:
    events: list[Any] = []

    async def spy() -> None:
        events.append(("yield",))
        on_yield()

    mp.setattr(rp, "_yield_to_loop", spy)
    if chunk is not None:
        mp.setattr(rp, "_PERSIST_CHUNK_ROWS", chunk)
    result, ex, pf = run_result
    run_id, log = uuid.UUID(int=1), structlog.get_logger("t")
    sink, run_orm = _Rec(events, keep), RunORM(id=run_id, config={})
    await rp.persist_backtest_results(sink, run_id, run_orm, result, log, ex, pf)  # type: ignore[arg-type]
    return events


def _plan_problems(events: list[Any], c: int, totals: dict[str, int]) -> list[str]:
    out, rows, flushes, sizes, rank, pending = [], 0, 0, {}, -1, 0
    for i, e in enumerate(events):
        if e[0] == "add":
            rows += e[2]
            pending += e[2]
            sizes.setdefault(e[1], []).append(e[2])
            out += [f"{e[1]} after a later table"] * (_RANK[e[1]] < rank)
            rank = max(rank, _RANK[e[1]])
        elif e[0] == "flush":
            flushes += 1
            pending = 0
            out += ["a flush is not followed by a yield"] * (events[i + 1 : i + 2] != [("yield",)])
        else:
            out += [f"{rows} rows, {flushes} flushes between yields"] * (rows > c or flushes > 1)
            out += [f"{pending} rows added but not flushed at a yield"] * bool(pending)
            rows = flushes = 0
    for cls, n in totals.items():
        want = [c] * (n // c) + [n % c] * bool(n % c)
        out += [f"{cls}: chunks {sizes.get(cls)}"] * (sizes.get(cls, []) != want)
    return out


@pytest.mark.parametrize("chunk", [None, _SMALL])
async def test_work_between_yields_is_bounded_and_ordered(
    run_result: Any, monkeypatch: pytest.MonkeyPatch, chunk: int | None
) -> None:
    result, ex, _ = run_result
    totals = {"TradeORM": len(result.trades), "EquitySnapshotORM": len(result.equity_curve),
              "OrderORM": len(ex.get_all_orders()), "FillORM": len(ex.get_all_fills())}  # fmt: skip
    events = await _persist(run_result, monkeypatch, chunk)
    assert _plan_problems(events, chunk or rp._PERSIST_CHUNK_ROWS, totals) == []
    assert events[-1] == ("add", "PositionSnapshotORM", 1)  # unflushed here: the status flush's
    bad = [("add", "FillORM", 1), ("flush",), ("yield",), ("add", "OrderORM", 1)]
    assert _plan_problems(bad, 1, {})  # control: fills before a later order
    assert _plan_problems([("add", "TradeORM", 9), ("flush",), ("yield",)], 7, {"TradeORM": 9})
    assert _plan_problems([("add", "TradeORM", 7), ("yield",)], 7, {"TradeORM": 7})  # unflushed


async def test_post_loop_steps_are_separated_by_yields(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def spy() -> None:
        seen.append("Y")

    def tag(owner: Any, attr: str, name: str) -> None:
        original = getattr(owner, attr)
        monkeypatch.setattr(owner, attr, lambda *a, **k: (seen.append(name), original(*a, **k))[1])

    monkeypatch.setattr(bt, "_yield_to_loop", spy)
    tag(BacktestRunner, "_build_equity_curve", "curve")
    for attr, name in (("sample_equity_per_bar", "sample"), ("compute_return_metrics", "metrics")):
        tag(bt, attr, name)
    tag(bt, "compute_calmar", "calmar")
    await _runner().run({"BTC/USDT": make_bars("BTC/USDT", 300, 7)})
    assert seen == ["Y", "curve", "Y", "sample", "Y", "metrics", "calmar", "Y"]


async def _max_live(run_result: Any, mp: pytest.MonkeyPatch, keep: bool) -> int:
    refs: list[Any] = []
    original, live = rp.EquitySnapshotORM, []

    def spy_orm(**kw: Any) -> Any:
        refs.append(weakref.ref(obj := original(**kw)))
        return obj

    mp.setattr(rp, "EquitySnapshotORM", spy_orm)
    count = lambda: live.append(sum(r() is not None for r in refs))  # noqa: E731
    await _persist(run_result, mp, _SMALL, [] if keep else None, count)
    return max(live)


async def test_no_equity_object_is_alive_at_a_yield(
    run_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _max_live(run_result, monkeypatch, keep=False) == 0  # released before the yield
    monkeypatch.undo()
    assert await _max_live(run_result, monkeypatch, keep=True) > _SMALL  # control


# --- T-9702-04: AST pins, each with planted-offender controls ------------------------------------

_CTOR = {"TradeORM": "259e9db19d535e17", "EquitySnapshotORM": "03071f6135487ed7",
         "OrderORM": "6fec35595270e82b", "FillORM": "1ffc1e7350a335a8"}  # fmt: skip
# ^ sha256 prefixes of ast.dump of the four constructor calls in persist_backtest_results at b9b83de
_LOOP = """for lo in range(0, len(items), _PERSIST_CHUNK_ROWS):
    part = items[lo:lo + _PERSIST_CHUNK_ROWS]
    chunk = [make(lo + i, item) for i, item in enumerate(part)]
    db.add_all(chunk)
    await db.flush()
    del chunk
    await _yield_to_loop()"""
_HELPER = "_flush_in_chunks"
_BANNED = {"commit", "begin", "begin_nested", "rollback", "to_thread", "run_in_executor"}


def _binding_problems(src: str) -> list[str]:
    """``_yield_to_loop`` is bound by name from the engine module; no gc, no thread offload."""
    tree, out = ast.parse(src), []
    froms = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    imports = [n for n in froms if n.module == "trading.strategy_engine"]
    imports = [
        n for n in imports if [a.name for a in n.names if not a.asname].count("_yield_to_loop")
    ]
    out += [] if len(imports) == 1 else ["no name-bound import"]
    for n in ast.walk(tree):
        out += ["attribute access"] * (isinstance(n, ast.Attribute) and n.attr == "_yield_to_loop")
        out += ["gc"] * (isinstance(n, ast.Name) and n.id == "gc")
        out += ["offload"] * (
            isinstance(n, ast.Attribute) and n.attr in {"to_thread", "run_in_executor"}
        )
    return out


def _persist_problems(src: str) -> list[str]:
    tree, out = ast.parse(src), []
    fns = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef)}
    helper, persist = fns["_flush_in_chunks"], fns["persist_backtest_results"]
    anns = [n for n in tree.body if isinstance(n, ast.AnnAssign)]
    consts = [ast.unparse(n) for n in anns if ast.unparse(n.target) == "_PERSIST_CHUNK_ROWS"]
    out += [] if consts == ["_PERSIST_CHUNK_ROWS: Final = 1000"] else ["chunk constant"]
    for f in (helper, persist):
        out += [f"{f.name}: try" for n in ast.walk(f) if isinstance(n, ast.Try)]
        out += [f"{f.name}: {n.func.attr}" for n in ast.walk(f) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in _BANNED]  # fmt: skip
    out += (
        []
        if [ast.unparse(n) for n in ast.walk(helper) if isinstance(n, ast.For)] == [_LOOP]
        else ["chunk loop"]
    )
    calls = sorted(
        (n for n in ast.walk(persist) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    names = [ast.unparse(c.func) for c in calls]
    used = [
        tuple(ast.unparse(a) for a in c.args[1:]) for c in calls if ast.unparse(c.func) == _HELPER
    ]
    want = [("result.trades", "_trade_row"), ("result.equity_curve", "_equity_row"),
            ("orders", "_order_row"), ("fills", "_fill_row")]  # fmt: skip
    out += [] if used == want else ["the helper is not called for trades, equity, orders, fills"]
    out += ["a direct flush or a second add_all"] * (
        "db.flush" in names or names.count("db.add_all") != 1
    )
    ctors = {c.func.id: hashlib.sha256(ast.dump(c).encode()).hexdigest()[:16] for c in calls
             if isinstance(c.func, ast.Name) and c.func.id in _CTOR}  # fmt: skip
    return out + ["a constructor changed"] * (ctors != _CTOR)


def _s2_problems(src: str) -> list[str]:
    run = next(
        f
        for f in ast.walk(ast.parse(src))
        if isinstance(f, ast.AsyncFunctionDef) and f.name == "run"
    )
    body = run.body

    def label(stmt: ast.stmt) -> str:
        return (
            "try-finally " + ast.unparse(stmt.finalbody[0])
            if isinstance(stmt, ast.Try)
            else ast.unparse(stmt)[:48]
        )

    got = [
        label(body[i - 1]) for i, s in enumerate(body) if ast.unparse(s) == "await _yield_to_loop()"
    ]
    want = ["try-finally await engine.stop()", "equity_curve = self._build_equity_curve(raw_equi",
            "sample = sample_equity_per_bar(equity_curve, sta",
            "calmar = compute_calmar(cagr, max_dd)"]  # fmt: skip
    return [] if got == want else [f"S2 yields follow {got}"]


_ORD, _FIL = (
    "    await _flush_in_chunks(db, orders, _order_row)\n",
    "    await _flush_in_chunks(db, fills, _fill_row)\n",
)
_CRV, _YLD = (
    "        equity_curve = self._build_equity_curve(raw_equity_curve)\n",
    "        await _yield_to_loop()\n",
)
_DEL = "        del chunk\n"
_TRY = "    try:\n    " + _ORD + "    except BaseException:\n        pass\n"
_ADD = "        db.add_all(chunk)\n"
_PLANTS_RP = {
    "no yield after a flush": (_DEL + _YLD, _DEL), "chunk 2000": ("Final = 1000", "Final = 2000"),
    "a commit in persist": (_FIL, _FIL + "    await db.commit()\n"),
    "a direct flush": ("    # --- Persist position snapshots ---\n", "    await db.flush()\n"),
    "fills before orders": (_ORD + _FIL, _FIL + _ORD),
    "a try around a table": (_ORD, _TRY),
    "a changed constructor": ('Decimal("0"),  # MVP: per-bar', 'Decimal("1"),  # MVP: per-bar'),
    "a restarted bar_index": ("make(lo + i, item)", "make(i, item)"),
    "all chunks kept": (_DEL, "        kept.append(chunk)\n"),
    "a thread offload": (_ADD, "        await asyncio.to_thread(db.add_all, chunk)\n"),
}  # fmt: skip
_PLANTS_BT = {
    "S2 yield removed": ("max_dd)\n" + _YLD, "max_dd)\n"),
    "yield before the curve": (_CRV + _YLD, _YLD + _CRV),
    "engine's binding called": (_CRV + _YLD, _CRV + _YLD.replace("await ", "await se.")),
}  # fmt: skip


def test_the_pins_hold_on_the_real_source_and_flag_planted_offenders() -> None:
    assert _persist_problems(_RP) == [] and _s2_problems(_BT) == []
    assert _binding_problems(_RP) == [] and _binding_problems(_BT) == []
    for name, (old, new) in _PLANTS_RP.items():
        assert _RP.count(old) == 1, name
        planted = _RP.replace(old, new)
        assert _persist_problems(planted) or _binding_problems(planted), name
    for name, (old, new) in _PLANTS_BT.items():
        assert _BT.count(old) == 1, name
        planted = _BT.replace(old, new)
        assert _s2_problems(planted) or _binding_problems(planted), name
    assert _binding_problems(_BT.replace("    _yield_to_loop,\n", ""))
    assert _binding_problems("import gc\ngc.collect()\n") and _binding_problems("x.to_thread(f)\n")
