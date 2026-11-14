"""CF-26-97 P2: the lazy fold cursor of ``PortfolioAccounting.get_max_drawdown``.

T-97P2-01 (seeded fuzz against the frozen base body), -02 (visit counts, no timing), -03 (purity),
-04 (writer/reader closure), -05 (no Decimal context change), -08 (no offload reaches a shared
portfolio).  Every AST pin has a planted-offender control.
"""

from __future__ import annotations

import ast
import copy
import random
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, Context, Decimal, localcontext
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from common.types import OrderSide
from tests._cf2697_reference import reference_get_max_drawdown, reference_max_dd_decimal
from trading.models import Fill
from trading.portfolio import PortfolioAccounting

_ROOT = Path(__file__).resolve().parents[2]
_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_CONTEXTS = (
    Context(prec=28),
    Context(prec=5, rounding=ROUND_DOWN),
    Context(prec=50, rounding=ROUND_CEILING),
)
_SPECIAL = ("NaN", "sNaN", "Infinity", "-Infinity", "0", "-0", "0E-10", "-5", "1E-20", "1E+25")
_TIES = ("100", "1E+2", "1.0E+2", "100.00", "99.5", "1.005E+2")


def _portfolio() -> PortfolioAccounting:
    ticks = iter(range(10**6))
    return PortfolioAccounting(
        "cf", Decimal("10000"), clock=lambda: _T0 + timedelta(seconds=next(ticks))
    )


def _fill(side: OrderSide, qty: str, price: str) -> Fill:
    return Fill(
        order_id=uuid4(), symbol="BTC/USDT", side=side, quantity=Decimal(qty), price=Decimal(price),
        fee=Decimal(1), fee_currency="USDT", is_maker=False, executed_at=_T0,
    )  # fmt: skip


def _outcome(fn: Any, p: PortfolioAccounting) -> tuple[Any, ...]:
    try:
        return ("ok", fn(p).hex())
    except Exception as exc:
        return ("exc", type(exc), exc.args)


def _assert_same(p: PortfolioAccounting) -> None:
    got, want = (
        _outcome(PortfolioAccounting.get_max_drawdown, p),
        _outcome(reference_get_max_drawdown, p),
    )
    assert got == want
    if want[0] == "ok" and len(p._equity_curve) >= 2:
        assert str(p._mdd_max) == str(reference_max_dd_decimal(p))


def _value(rnd: random.Random, prev: Decimal) -> Decimal:
    prev = prev if prev.is_finite() else Decimal(100)
    kind = rnd.choice(
        ("walk", "walk", "walk", "flat", "up", "down", "special", "tie", "digits", "tiny")
    )
    return {
        "walk": abs(prev * rnd.randint(9700, 10300) / 10000), "flat": prev, "up": prev + 1,
        "down": prev - 1, "special": Decimal(rnd.choice(_SPECIAL)),
        "tie": Decimal(rnd.choice(_TIES)),
        "tiny": Decimal(rnd.randint(1, 9)).scaleb(-18),
        "digits": Decimal("1234567890123456789012345678.9012") * rnd.randint(1, 3),
    }[kind]  # fmt: skip


def _step(rnd: random.Random, p: PortfolioAccounting, i: int) -> None:
    curve, op = p._equity_curve, rnd.choice("aaaaaaabcdeffg")
    point = (_T0 + timedelta(minutes=i), _value(rnd, curve[-1][1] if curve else Decimal(100)))
    if op == "b":  # replace the list object
        p._equity_curve = [point] + [
            (point[0], _value(rnd, point[1])) for _ in range(rnd.randint(0, 6))
        ]
    elif op == "c" and curve:  # truncate in place
        del curve[rnd.randint(0, len(curve) - 1) :]
    elif op == "d":
        curve.clear()
    elif op == "e" and curve:  # replace the last point
        curve[-1] = point
    elif op == "g" and p._mdd_n:  # a new list that still holds the last folded point object
        p._equity_curve = [(_T0, _value(rnd, Decimal(100))) for _ in range(p._mdd_n - 1)]
        p._equity_curve.append(p._mdd_last)  # type: ignore[arg-type]
    else:
        curve.extend([point] if op == "a" else [point, (point[0], _value(rnd, point[1]))])


@pytest.mark.parametrize("context", _CONTEXTS, ids=["p28", "p5-down", "p50-ceiling"])
def test_fuzz_equals_the_frozen_base_body(context: Context) -> None:
    rnd = random.Random(97)  # noqa: S311
    with localcontext(context):
        for _ in range(100):
            p = _portfolio()
            for i in range(40):
                _step(rnd, p, i)
                _assert_same(p)
                _assert_same(p)  # a repeated call must repeat the result (or the exception)


def test_from_fills_peak_hint_and_foreign_peak_equal_the_frozen_body() -> None:
    fills = [_fill(OrderSide.BUY, "1", "100"), _fill(OrderSide.SELL, "0.5", "120")]
    hinted = PortfolioAccounting.from_fills(
        "cf", Decimal(10000), fills, peak_equity_hint=Decimal(50000)
    )
    foreign = _portfolio()
    foreign._peak_equity = Decimal(20000)
    for p in (hinted, foreign):
        assert p._peak_equity != p._equity_curve[0][1]
        for price in ("100", "90", "130", "70", "80"):
            p.update_position(_fill(OrderSide.BUY, "0.1", price), Decimal(price))
            p.update_market_prices({"BTC/USDT": Decimal(price)})
            _assert_same(p)


# --- T-97P2-02: visit counts, never time ----------------------------------------------------


class _CountingList(list[Any]):
    visits = 0

    def __getitem__(self, index: Any) -> Any:
        self.visits += len(range(*index.indices(len(self)))) if isinstance(index, slice) else 1
        return super().__getitem__(index)

    def __iter__(self) -> Any:
        for item in super().__iter__():
            self.visits += 1
            yield item


def _visits(steps: int) -> int:
    p = _portfolio()
    p._equity_curve = counting = _CountingList(p._equity_curve)
    for _ in range(steps):
        p.update_market_prices({})
        p.get_summary()
    return counting.visits


def test_visits_grow_linearly_with_the_number_of_bars() -> None:
    v2000, v4000 = _visits(2000), _visits(4000)
    assert v4000 <= 2.2 * v2000
    assert v4000 <= 3 * 4000


# --- T-97P2-03: reads are observationally pure ----------------------------------------------


def test_reads_change_nothing_but_the_cursor() -> None:
    p = _portfolio()
    p.update_position(_fill(OrderSide.BUY, "1", "100"), Decimal(100))
    for price in ("100", "90", "120"):
        p.update_market_prices({"BTC/USDT": Decimal(price)})
    rest = lambda: {k: v for k, v in vars(p).items() if not k.startswith("_mdd_")}  # noqa: E731
    before, first = copy.deepcopy(rest()), p.get_summary()
    for read in (
        p.risk_snapshot,
        p.get_daily_pnl,
        p.get_peak_equity,
        p.get_summary,
        p.get_max_drawdown,
    ):
        read()
    assert p.get_summary() == first
    assert rest() == before


# --- AST pins: T-97P2-04 (cursor and curve writers), -05 (no Decimal context), -08 (offloads) --


def _py(*roots: str) -> Iterable[tuple[str, str]]:
    for root in roots:
        for path in sorted((_ROOT / root).rglob("*.py")):
            yield path.relative_to(_ROOT).as_posix(), path.read_text()


def _walk(src: str) -> Iterable[tuple[ast.AST, str]]:
    """Every node with the name of its innermost enclosing function."""

    def rec(node: ast.AST, func: str) -> Iterable[tuple[ast.AST, str]]:
        for c in ast.iter_child_nodes(node):
            f = c.name if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef) else func
            yield c, f
            yield from rec(c, f)

    return rec(ast.parse(src), "<module>")


_PF = "packages/trading/portfolio.py"
_NAMES = {"getcontext", "setcontext", "DefaultContext", "BasicContext", "ExtendedContext"}
_CTX_ATTRS = {"prec", "rounding", "Emax", "Emin", "capitals", "clamp", "traps", "flags"}
_OK_CURVE = {
    (_PF, "__init__", "_equity_curve", "Store"), (_PF, "from_fills", "_equity_curve", "Store"),
    (_PF, "_apply_fill", "_equity_curve", ".append"),
    (_PF, "update_market_prices", "_equity_curve", ".append"),
}  # fmt: skip
_OK_CURSOR = {
    (_PF, "__init__", "Store"),
    (_PF, "get_max_drawdown", "Load"),
    (_PF, "get_max_drawdown", "Store"),
}


def _problems(src: str, rel: str) -> set[tuple[str, ...]]:
    """Uses of the cursor, the curve and the Decimal-context names that are not allowed."""
    bad: set[tuple[str, ...]] = set()
    for c, func in _walk(src):
        name = getattr(c, "attr", None) or getattr(c, "id", None) or getattr(c, "name", None) or ""
        how = type(getattr(c, "ctx", None)).__name__
        if isinstance(c, ast.Attribute | ast.Name | ast.alias):
            if name.startswith("_mdd_") and (rel, func, how) not in _OK_CURSOR:
                bad.add((rel, func, name, how))
            if name in _NAMES or (
                name in _CTX_ATTRS and how != "Load" and isinstance(c, ast.Attribute)
            ):
                bad.add((rel, func, name, how))
        if isinstance(c, ast.Call) and (c.args or c.keywords):  # CR-97P2-01 / QR-97P2-04
            if (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) == "localcontext":
                bad.add((rel, func, "localcontext(...)", "Call"))
        base: Any = None
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute):
            base, how = c.func.value, "." + c.func.attr
        elif isinstance(c, ast.Subscript):
            base, how = c.value, "Load" if how == "Load" else "[]="
        elif isinstance(c, ast.AugAssign):
            base, how = c.target, "+="
        if isinstance(c, ast.Attribute) and c.attr == "_equity_curve":
            base = c
        if getattr(base, "attr", None) == "_equity_curve" and how != "Load":
            if (rel, func, "_equity_curve", how) not in _OK_CURVE:
                bad.add((rel, func, "_equity_curve", how))
    return bad


def test_pinned_names_have_only_their_known_users() -> None:
    assert {u for rel, src in _py("packages", "apps") for u in _problems(src, rel)} == set()


def test_pins_flag_planted_offenders() -> None:
    curve = "def f(p):\n    p._equity_curve += [1]\n    p._equity_curve[0] = 1\n"
    curve += "    p._equity_curve.sort()\n"
    assert len(_problems(curve, "x.py")) == 4
    assert _problems("def g(s):\n    return s._mdd_n\n", "x.py") and _problems(
        "def g(s):\n    s._mdd_n = 1\n", _PF
    )
    assert _problems("def _apply_fill(s):\n    s._equity_curve.append(1)\n", _PF) == set()
    assert _problems("from decimal import getcontext\ngetcontext().prec = 5\n", "x.py")
    assert _problems("import decimal\ndecimal.setcontext(c)\n", "x.py")
    assert _problems("with localcontext() as c:\n    c.prec = 5\n", "x.py")
    for s in ("localcontext(prec=3)", "localcontext(Context(prec=3))", "decimal.localcontext(c)"):
        assert _problems(f"with {s}:\n    pass\n", "x.py"), s
    assert _problems("CTX = Context(prec=3)\n", "x.py") == set()
    assert (
        _problems("from decimal import localcontext\nwith localcontext():\n    pass\n", "x.py")
        == set()
    )


_OFFLOADS = {"to_thread", "run_in_executor", "submit", "Thread"}
_PORTFOLIO_NAMES = {"PortfolioAccounting", "get_max_drawdown", "get_summary", "risk_snapshot"}
_RUN_LOCAL = {("apps/api/routers/ml.py", "train_model")}  # run-local portfolios on a private loop
_KNOWN_SITES = _RUN_LOCAL | {
    ("apps/api/services/ohlcv_history.py", "_cached"),
    ("apps/api/services/ohlcv_history.py", "fetch_range"),
    ("apps/api/services/ohlcv_store.py", "_off_loop"),
    ("apps/api/services/retraining.py", "_do_retrain"),
    ("packages/common/events.py", "publish"),
    ("packages/trading/strategies/model_strategy.py", "_load_model_with_timeout"),
}


def _offload_sites(src: str, rel: str) -> tuple[set[tuple[str, str]], bool]:
    sites, names = set(), set()
    for c, func in _walk(src):
        names.add(getattr(c, "id", None) or getattr(c, "attr", None))
        if isinstance(c, ast.Call):
            callee = getattr(c.func, "attr", None) or getattr(c.func, "id", "")
            if callee in _OFFLOADS or any(k.arg == "offload" for k in c.keywords):
                sites.add((rel, func))
    return sites, bool(names & _PORTFOLIO_NAMES)


def test_no_offload_reaches_a_shared_portfolio() -> None:
    sites: set[tuple[str, str]] = set()
    for rel, src in _py("packages", "apps"):
        found, touches = _offload_sites(src, rel)
        sites |= found
        assert not (found and touches) or found <= _RUN_LOCAL, rel
    assert sites == _KNOWN_SITES  # a new offload site needs a review of what it can reach


def test_offload_pin_flags_planted_offenders() -> None:
    bad = "async def f(p):\n    await asyncio.to_thread(p.get_summary)\n"
    assert _offload_sites(bad, "x.py") == ({("x.py", "f")}, True)
    assert _offload_sites("def g():\n    PortfolioAccounting\n", "x.py") == (set(), True)
    assert _offload_sites("async def h():\n    await run(offload=to_thread)\n", "x.py")[0]


_ROUTES = {"get", "post", "put", "patch", "delete", "api_route", "websocket"}


def _off_loop(src: str) -> bool:
    # RK-97P2-03: events.publish runs a sync subscriber via to_thread and FastAPI runs a `def`
    # route in a worker thread; neither is a site the offload inventory above can see
    for c, _ in _walk(src):
        if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "subscribe":
            return True
        if isinstance(c, ast.FunctionDef) and any(
            getattr(getattr(d, "func", None), "attr", None) in _ROUTES for d in c.decorator_list
        ):
            return True
    return False


def test_no_event_subscriber_or_sync_route_runs_off_the_loop() -> None:
    assert [rel for rel, src in _py("packages", "apps") if _off_loop(src)] == []
    assert _off_loop("def f(bus):\n    bus.subscribe(E, h)\n")
    assert _off_loop("@router.get('/x')\ndef g():\n    pass\n")
    assert not _off_loop("@router.get('/x')\nasync def g():\n    pass\n")
