"""CF-26-97 P3: the paper engine's open-order index (``_OrderRegistry``).

T-97P3-01 (registry semantics), -02 (seeded property test against the frozen base filter), -03
(multi-LIMIT ``check_resting_orders``), -04 (``_TERMINAL``), -05 (live and base engines keep a
dict), -06 (read counts, no timing), -07 (AST pins with controls and the resume path), and the
A-97-10 evidence that ``on_stop`` and ``StrategyEngine.stop`` cancel in the base-filter order.
"""

from __future__ import annotations

import ast
import copy
import inspect
import pickle
import random
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import structlog
from structlog.testing import capture_logs

from common.types import OrderSide, OrderStatus, OrderType, TimeFrame
from tests._cf2697_reference_orders import reference_get_open_orders
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import _TERMINAL, PaperExecutionEngine, _OrderRegistry
from trading.execution import BaseExecutionEngine
from trading.models import Fill, Order
from trading.portfolio import PortfolioAccounting
from trading.risk_manager import DefaultRiskManager
from trading.strategies import MACrossoverStrategy
from trading.strategy_engine import RunMode, StrategyEngine

_ROOT = Path(__file__).resolve().parents[2]
_ST = list(OrderStatus)
_OPEN, _DONE = OrderStatus.OPEN, OrderStatus.FILLED


def _o(status: OrderStatus = _OPEN) -> Any:
    return SimpleNamespace(status=status)


def _limit(n: int, run: str, symbol: str, price: str) -> Order:
    return Order(
        order_id=UUID(int=n), client_order_id=f"{run}-{n}", run_id=run, symbol=symbol,
        side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal(1), price=Decimal(price),
    )  # fmt: skip


def _same(reg: _OrderRegistry) -> None:
    got, want = reg.open_orders(), reference_get_open_orders(SimpleNamespace(_orders=reg))
    assert len(got) == len(want) and all(a is b for a, b in zip(got, want, strict=True))


@pytest.fixture(autouse=True)
def _unfiltered_logs() -> Iterable[None]:
    previous = structlog.get_config()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))
    yield
    structlog.configure(**previous)


# --- T-97P3-01: registry semantics -----------------------------------------------------------


def test_delete_reinsert_moves_to_the_end_and_overwrite_keeps_the_position() -> None:
    reg, (a, b, c) = _OrderRegistry(), (_o(), _o(), _o())
    reg.update({1: a, 2: b, 3: c})  # type: ignore[dict-item]
    reg[1] = _o()  # overwrite: position kept
    assert list(reg._open) == [1, 2, 3]
    del reg[1]
    reg[1] = a
    assert list(reg._open) == [2, 3, 1]
    _same(reg)


def test_revival_restores_the_dict_order_position() -> None:
    reg = _OrderRegistry()
    for k in (1, 2, 3):
        reg[k] = _o()  # type: ignore[index]
    reg[2] = _o(_DONE)  # type: ignore[index]
    reg[4] = _o()  # type: ignore[index]
    reg[2] = _o()  # type: ignore[index]  # a stale terminal -> OPEN direct write
    assert list(reg._open) == [1, 2, 3, 4]
    _same(reg)


def test_every_mutator_keeps_the_index_in_step() -> None:
    reg: Any = _OrderRegistry({1: _o(), 2: _o(_DONE), 3: _o()})
    steps = (
        lambda: reg.pop(1),
        lambda: reg.setdefault(5, _o()),
        lambda: reg.update({6: _o(), 3: _o(_DONE)}),
        lambda: reg.popitem(),
        lambda: reg.__ior__({7: _o()}),
        lambda: reg.pop(99, None),
        lambda: reg.clear(),
    )
    for step in steps:
        step()
        _same(reg)
        assert set(reg._open) <= set(reg)
    assert not reg and not reg._open


def test_a_plain_dict_or_a_subclass_falls_back_to_the_base_filter() -> None:
    engine = PaperExecutionEngine("r", MagicMock())
    plain_order = _limit(1, "r", "S", "1")
    engine._orders = {plain_order.order_id: plain_order}
    assert engine.get_open_orders() == [plain_order]

    class _Skips(_OrderRegistry):  # a subclass that never updates the index
        __setitem__ = dict.__setitem__  # type: ignore[assignment]

    engine._orders = skips = _Skips()
    skips[plain_order.order_id] = plain_order
    assert engine.get_open_orders() == [plain_order]  # an isinstance check would return []


def test_deepcopy_and_pickle_keep_content_order_and_index() -> None:
    reg: Any = _OrderRegistry()
    for k, status in ((1, _OPEN), (2, _DONE), (3, _OPEN)):
        reg[k] = _o(status)
    for clone in (copy.deepcopy(reg), pickle.loads(pickle.dumps(reg))):  # noqa: S301
        assert type(clone) is _OrderRegistry and list(clone) == [1, 2, 3]
        assert list(clone._open) == [1, 3]
        _same(clone)


def test_an_in_place_open_to_terminal_mutation_is_dropped_at_read() -> None:
    reg: Any = _OrderRegistry()
    first = reg[1] = _o()
    reg[2] = _o()
    first.status = _DONE
    assert len(reg.open_orders()) == 1
    _same(reg)


def test_a_value_without_status_raises_like_the_base_body() -> None:
    reg: Any = _OrderRegistry()
    reg[1] = _o()
    reg[2] = object()  # accepted at write time, like a plain dict
    with pytest.raises(AttributeError) as got:
        reg.open_orders()
    with pytest.raises(AttributeError) as want:
        reference_get_open_orders(SimpleNamespace(_orders=reg))
    assert got.value.args == want.value.args


# --- T-97P3-02: seeded property test against the frozen base body -----------------------------


def test_random_operations_match_the_frozen_base_filter() -> None:
    rnd = random.Random(97)  # noqa: S311
    for _ in range(300):
        reg: Any = _OrderRegistry()
        for _ in range(60):
            k, op = rnd.randint(0, 11), rnd.choice("sssssoodpqcuxyz")
            if (
                op in "so"
            ):  # new key or overwrite, any status (a terminal -> open write is a revival)
                reg[k] = _o(rnd.choice(_ST))
            elif op == "d" and k in reg:
                del reg[k]
            elif op == "p":
                reg.pop(k, None)
            elif op == "q" and reg:
                reg.popitem()
            elif op == "c" and rnd.random() < 0.2:
                reg.clear()
            elif op == "u":
                reg.update({k: _o(rnd.choice(_ST)), (k + 1) % 12: _o(rnd.choice(_ST))})
            elif op == "x":
                reg.setdefault(k, _o(rnd.choice(_ST)))
            elif op == "y":
                reg |= {k: _o(rnd.choice(_ST))}
            elif op == "z" and k in reg:
                reg[k].status = _DONE  # in place, open -> terminal
            _same(reg)


# --- T-97P3-03 / -04 / -05 / -06 --------------------------------------------------------------


async def _limit_scenario(*, plain: bool) -> tuple[Any, ...]:
    engine = PaperExecutionEngine("r", DefaultRiskManager(run_id="r"))
    if plain:
        engine._orders = {}
    steps = []
    for sym, px in (("A", "100"), ("B", "50")):
        engine.set_last_price(sym, Decimal(px))
    plan = "A 99,B 49,A 98,A 97,B 48"  # BUY LIMITs, symbol and price
    for n, (sym, px) in enumerate((x.split() for x in plan.split(",")), 1):
        await engine.submit_order(_limit(n, "r", sym, px))
    await engine.cancel_order(UUID(int=5))
    for sym, px in (("A", "100"), ("A", "96"), ("B", "47"), ("B", "60")):
        engine.set_last_price(sym, Decimal(px))
        filled = await engine.check_resting_orders(sym, Decimal(px))
        steps.append(
            (
                [o.order_id for o in filled],
                [o.order_id for o in engine.get_open_orders()],
                engine.cash,
            )
        )
    fills = [(f.order_id, f.price, f.quantity, f.fee) for f in engine.get_all_fills()]
    return (
        steps,
        fills,
        engine.cash,
        engine.positions.keys(),
        [o.status for o in engine.get_all_orders()],
    )


async def test_multi_limit_check_resting_orders_equals_a_plain_dict_store() -> None:
    registry, plain = await _limit_scenario(plain=False), await _limit_scenario(plain=True)
    assert list(registry[3]) == list(plain[3])
    assert registry[:3] + registry[4:] == plain[:3] + plain[4:]
    assert [f[1] for f in registry[1]] == [Decimal(x) for x in (99, 98, 97, 49)]  # submission order


def test_terminal_states_equal_the_base_literal() -> None:
    assert _TERMINAL == {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }


def test_live_and_base_engines_keep_a_plain_dict() -> None:
    class _Fake(BaseExecutionEngine):
        pass

    _Fake.__abstractmethods__ = frozenset()  # the abstract methods are never called here

    live = LiveExecutionEngine(
        run_id="r", risk_manager=MagicMock(), exchange=MagicMock(), enable_live_trading=True
    )
    assert type(live._orders) is dict and type(_Fake("r")._orders) is dict
    assert type(PaperExecutionEngine("r", MagicMock())._orders) is _OrderRegistry


class _Counting:
    reads = 0

    def __init__(self, status: OrderStatus) -> None:
        self._status = status

    @property
    def status(self) -> OrderStatus:
        _Counting.reads += 1
        return self._status


def _reads_in_get_open_orders(terminal: int) -> int:
    engine = PaperExecutionEngine("r", MagicMock())
    for i in range(terminal):  # each is placed open and then transitioned, as in a run
        engine._orders[UUID(int=i)] = _Counting(_OPEN)  # type: ignore[assignment]
        engine._orders[UUID(int=i)] = _Counting(_DONE)  # type: ignore[assignment]
    for i in (terminal, terminal + 1):
        engine._orders[UUID(int=i)] = _Counting(_OPEN)  # type: ignore[assignment]
    _Counting.reads = 0
    assert len(inspect.unwrap(PaperExecutionEngine.get_open_orders)(engine)) == 2  # not the plugin
    return _Counting.reads


def test_reads_do_not_grow_with_the_number_of_terminal_orders() -> None:
    assert _reads_in_get_open_orders(100) == _reads_in_get_open_orders(10_000)


# --- T-97P3-07: AST pins with controls -------------------------------------------------------


def _py(*roots: str) -> Iterable[tuple[str, str]]:
    for root in roots:
        for path in sorted((_ROOT / root).rglob("*.py")):
            yield path.relative_to(_ROOT).as_posix(), path.read_text()


def _bindings(src: str, rel: str) -> set[tuple[str, str, str]]:
    """(file, function, what) for every ``.status`` store and every ``._orders`` binding."""
    out: set[tuple[str, str, str]] = set()

    def rec(node: ast.AST, func: str) -> None:
        for c in ast.iter_child_nodes(node):
            f = c.name if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef) else func
            if (
                isinstance(c, ast.Attribute)
                and c.attr in ("status", "_orders")
                and not isinstance(c.ctx, ast.Load)
            ):
                out.add((rel, f, c.attr))
            rec(c, f)

    rec(ast.parse(src), "<module>")
    return out


_ORDERS_OK = {("packages/trading/execution.py", "__init__", "_orders"),
              ("packages/trading/engines/paper.py", "__init__", "_orders")}  # fmt: skip


def test_status_is_never_assigned_and_orders_is_bound_only_in_the_two_constructors() -> None:
    found = {b for rel, src in _py("packages", "apps") for b in _bindings(src, rel)}
    assert {b for b in found if b[2] == "status" and b[0].startswith("packages/trading/")} == set()
    assert {b for b in found if b[2] == "_orders"} == _ORDERS_OK  # every other write is a subscript


def test_binding_pin_flags_planted_offenders() -> None:
    assert _bindings("def f(o):\n    o.status = 1\n", "x.py") == {("x.py", "f", "status")}
    assert _bindings("def g(s):\n    s._orders = {}\n", "x.py") == {("x.py", "g", "_orders")}
    assert _bindings("def h(s):\n    s._orders[1] = 2\n    return s.status\n", "x.py") == set()


# --- A-97-10: PAPER-mode stop order and the resume path -------------------------------------


async def _resting_stack(run_id: str) -> tuple[StrategyEngine, PaperExecutionEngine]:
    class _MarketData:
        async def connect(self) -> None: ...
        async def close(self) -> None: ...

    rm = DefaultRiskManager(run_id=run_id)
    execution = PaperExecutionEngine(run_id, rm)
    engine = StrategyEngine(
        strategies=[MACrossoverStrategy(strategy_id="s", params={})], execution_engine=execution,
        risk_manager=rm, market_data=_MarketData(), symbols=["A", "B"],
        portfolio=PortfolioAccounting(run_id, Decimal(10000)), timeframe=TimeFrame.ONE_MINUTE,
        run_mode=RunMode.PAPER,
    )  # fmt: skip
    await engine.start(run_id)
    for sym in ("A", "B"):
        execution.set_last_price(sym, Decimal(100))
    for n, (sym, px) in enumerate(
        (("A", "99"), ("B", "98"), ("A", "97"), ("B", "96"), ("A", "95")), 1
    ):
        await execution.submit_order(_limit(n, run_id, sym, px))
    await execution.cancel_order(UUID(int=3))  # a cancelled order sits between the resting ones
    return engine, execution


async def test_paper_stop_and_on_stop_cancel_in_the_base_filter_order() -> None:
    engine, execution = await _resting_stack("stop")
    expected = [str(o.order_id) for o in reference_get_open_orders(execution)]
    assert expected == [str(o.order_id) for o in execution.get_open_orders()] and len(expected) == 4
    with capture_logs() as logs:
        await engine.stop()
    assert [
        e["order_id"] for e in logs if e["event"] == "engine.order_canceled_on_stop"
    ] == expected
    assert [e["order_id"] for e in logs if e["event"] == "paper.order_canceled"] == expected

    _, execution = await _resting_stack("on-stop")
    expected = [str(o.order_id) for o in reference_get_open_orders(execution)]
    with capture_logs() as logs:
        await execution.on_stop()
    assert [e["order_id"] for e in logs if e["event"] == "paper.order_canceled"] == expected


def test_a_resumed_paper_engine_still_has_the_registry() -> None:
    engine = PaperExecutionEngine(
        "resume", DefaultRiskManager(run_id="resume"), initial_cash=Decimal(1000)
    )
    fill = Fill(
        order_id=UUID(int=1), symbol="A", side=OrderSide.BUY, quantity=Decimal(1),
        price=Decimal(10), fee=Decimal(0), fee_currency="USDT", is_maker=False,
    )  # fmt: skip
    engine.restore_from_fills(
        [fill]
    )  # the orchestrator's resume step (run_orchestrator.py:1146-1160)
    assert type(engine._orders) is _OrderRegistry and engine.get_open_orders() == []
