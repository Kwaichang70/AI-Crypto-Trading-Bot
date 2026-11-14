"""CF-26-97 P1: the cooperative-yield seam (SY-97-05/06, T-97P1-01..03, 05, 07, 08).

No timing assertion anywhere: the yield is tested by counting, by sibling ticks (round-robin
scheduling is deterministic) and by AST pins that each carry a planted-offender control.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from common.types import TimeFrame
from tests._cf2697_fingerprint import make_bars
from trading import strategy_engine
from trading.backtest import BacktestRunner
from trading.clock import ambient_clock
from trading.strategies import MACrossoverStrategy
from trading.strategy_engine import EngineState

_ROOT = Path(__file__).resolve().parents[2]
_N = 300  # one symbol: 300 loop iterations, warmup (50) included


def _runner() -> BacktestRunner:
    strategy = MACrossoverStrategy(strategy_id="y", params={"fast_period": 5, "slow_period": 20})
    return BacktestRunner([strategy], ["BTC/USDT"], TimeFrame.ONE_MINUTE, seed=7)


@pytest.fixture(autouse=True)
def _unfiltered_logs() -> Iterator[None]:
    previous = structlog.get_config()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))
    yield
    structlog.configure(**previous)


def _spy_engine(monkeypatch: pytest.MonkeyPatch, runner: BacktestRunner) -> list[Any]:
    """Record the (engine, portfolio, risk, exec) tuple the runner builds."""
    built: list[Any] = []
    original = runner._build_engine

    def build(*args: Any, **kwargs: Any) -> Any:
        built.append(original(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(runner, "_build_engine", build)
    return built


async def test_yield_count_equals_loop_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def counted() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(strategy_engine, "_yield_to_loop", counted)
    await _runner().run({"BTC/USDT": make_bars("BTC/USDT", _N, 7)})
    assert calls == _N


async def test_early_stop_at_step_s_counts_s_plus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, calls, s = _runner(), 0, 150
    built = _spy_engine(monkeypatch, runner)

    async def counted() -> None:
        nonlocal calls
        calls += 1
        if calls == s + 1:
            built[0][0]._stop_event.set()

    monkeypatch.setattr(strategy_engine, "_yield_to_loop", counted)
    await runner.run({"BTC/USDT": make_bars("BTC/USDT", _N, 7)})
    assert calls == s + 1


async def test_sibling_task_ticks_at_every_step() -> None:
    done, ticks = False, 0

    async def sibling() -> None:
        nonlocal ticks
        while not done:
            await asyncio.sleep(0)
            ticks += 1

    task = asyncio.create_task(sibling())
    await _runner().run({"BTC/USDT": make_bars("BTC/USDT", _N, 7)})
    done = True
    await task
    assert ticks >= _N - 1


# --- T-97P1-03: AST pins, each with a planted-offender control --------------------------------


def _loops(src: str, func: str, target: str) -> list[ast.For]:
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == func]
    return [
        n
        for f in fns
        for n in ast.walk(f)
        if isinstance(n, ast.For) and ast.unparse(n.target) == target
    ]


def _loop_problems(src: str, func: str, target: str, first: str) -> list[str]:
    loops = _loops(src, func, target)
    if len(loops) != 1:
        return [f"{func}: {len(loops)} loops over {target}"]
    stmt = ast.unparse(loops[0].body[0])
    return [] if stmt == first else [f"{func}: first statement is {stmt!r}"]


def _helper_problems(src: str) -> list[str]:
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.AsyncFunctionDef)]
    helper = [f for f in fns if f.name == "_yield_to_loop"]
    if len(helper) != 1:
        return ["no module-level async _yield_to_loop"]
    body = [
        s
        for s in helper[0].body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return (
        []
        if [ast.unparse(s) for s in body] == ["await asyncio.sleep(0)"]
        else ["helper body differs"]
    )


def _modulo_problems(src: str, func: str) -> list[str]:
    fns = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == func
    ]
    mods = [
        n
        for f in fns
        for n in ast.walk(f)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
    ]
    return [ast.unparse(m) for m in mods if "bar_idx" in ast.unparse(m)]


_HEAD = "async def f(n):\n    for bar_idx in range(n):\n"
_GOOD = _HEAD + "        await _yield_to_loop()\n        if stop:\n            break\n"
_LATE = _HEAD + "        if stop:\n            break\n        await _yield_to_loop()\n"
_COND = _HEAD + "        if bar_idx % 2:\n            await _yield_to_loop()\n"


def test_engine_yield_is_the_first_statement_of_the_bar_loop() -> None:
    src = (_ROOT / "packages/trading/strategy_engine.py").read_text()
    assert _loop_problems(src, "run_backtest", "bar_idx", "await _yield_to_loop()") == []
    assert src.count("await _yield_to_loop()") == 1
    assert _helper_problems(src) == []
    assert _modulo_problems(src, "run_backtest") == []


def test_engine_yield_pins_flag_planted_offenders() -> None:
    first = "await _yield_to_loop()"
    assert _loop_problems(_GOOD, "f", "bar_idx", first) == []
    assert _loop_problems(_LATE, "f", "bar_idx", first)
    assert _loop_problems(_COND, "f", "bar_idx", first)
    assert _modulo_problems(_COND, "f") and not _modulo_problems(_GOOD, "f")
    assert _helper_problems("async def _yield_to_loop():\n    await asyncio.sleep(0)\n") == []
    assert _helper_problems("async def _yield_to_loop():\n    x = 1\n    await asyncio.sleep(0)\n")
    assert _helper_problems("def _yield_to_loop():\n    await asyncio.sleep(0)\n")


def test_optimizer_loops_start_with_a_sleep_zero() -> None:
    src = (_ROOT / "packages/trading/optimizer.py").read_text()
    first = "await asyncio.sleep(0)"
    for func, target in (
        ("_run_in_sample", "(idx, combo_values)"),
        ("_run_walk_forward", "(idx, combo_values)"),
        ("_run_walk_forward", "fold"),
    ):
        assert _loop_problems(src, func, target, first) == [], (func, target)
    late = "async def g(c):\n    for fold in c:\n        x = 1\n        await asyncio.sleep(0)\n"
    assert _loop_problems(
        late.replace("g", "_run_walk_forward"), "_run_walk_forward", "fold", first
    )


# --- T-97P1-05: no global-RNG draw in packages/trading and packages/data ----------------------

_ALLOWED_DRAWS = {
    ("packages/trading/backtest.py", "__init__", "random.randint"),
    ("packages/trading/backtest.py", "run", "random.seed"),
    ("packages/trading/backtest.py", "run", "np.random.seed"),
    ("packages/trading/ccxt_retry.py", "ccxt_retry", "random.uniform"),
    ("packages/data/services/ccxt_market_data.py", "_backoff", "random.uniform"),
}
_CONSTRUCTORS = {"random.Random", "np.random.default_rng", "numpy.random.default_rng"}
_RNG_PREFIXES = ("random.", "np.random.", "numpy.random.")


def _draw_sites(src: str, rel: str) -> set[tuple[str, str, str]]:
    sites: set[tuple[str, str, str]] = set()

    def walk(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else func
            if isinstance(child, ast.Call):
                callee = ast.unparse(child.func)
                if callee.startswith(_RNG_PREFIXES) and callee not in _CONSTRUCTORS:
                    sites.add((rel, func, callee))
            if isinstance(child, ast.ImportFrom) and child.module in ("random", "numpy.random"):
                sites.update(
                    (rel, func, f"from {child.module} import {a.name}") for a in child.names
                )
            walk(child, name)

    walk(ast.parse(src), "<module>")
    return sites


def test_no_global_rng_draw_outside_the_allowlist() -> None:
    found: set[tuple[str, str, str]] = set()
    for pkg in ("packages/trading", "packages/data"):
        for path in sorted((_ROOT / pkg).rglob("*.py")):
            found |= _draw_sites(path.read_text(), path.relative_to(_ROOT).as_posix())
    assert found == _ALLOWED_DRAWS


def test_draw_scanner_flags_planted_offenders() -> None:
    planted = (
        "import random\nimport numpy as np\ndef f():\n    random.random()\n    np.random.rand(3)\n"
    )
    assert {c for _, _, c in _draw_sites(planted, "x.py")} == {"random.random", "np.random.rand"}
    assert _draw_sites("from random import choice\n", "x.py") == {
        ("x.py", "<module>", "from random import choice")
    }
    assert _draw_sites("import random\nr = random.Random(1)\n", "x.py") == set()


# --- T-97P1-07 / 08: cancellation and a stop() that never suspends ---------------------------


async def test_cancellation_at_a_yield_stops_the_engine_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, calls, k = _runner(), 0, 100
    built = _spy_engine(monkeypatch, runner)
    exec_clock: list[Any] = []

    async def cancelling() -> None:
        nonlocal calls
        calls += 1
        exec_clock.append(built[0][3]._clock)
        if calls == k:
            raise asyncio.CancelledError

    monkeypatch.setattr(strategy_engine, "_yield_to_loop", cancelling)
    ambient_before = ambient_clock()
    with capture_logs() as logs, pytest.raises(asyncio.CancelledError):
        await runner.run({"BTC/USDT": make_bars("BTC/USDT", _N, 7)})
    engine, _, _, exec_engine = built[0]
    assert engine._state == EngineState.STOPPED
    assert [e["event"] for e in logs].count("engine.stopped") == 1
    assert engine._backtest_running is False
    assert ambient_clock() is ambient_before
    assert exec_engine._clock is exec_clock[0]
    assert runner.last_portfolio is None and runner.last_execution_engine is None


async def test_backtest_stop_never_suspends(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    built = _spy_engine(monkeypatch, runner)
    await runner.run({"BTC/USDT": make_bars("BTC/USDT", _N, 7)})
    with pytest.raises(StopIteration):
        engine = built[0][0]
        engine.stop().send(None)
    fresh = BacktestRunner(
        [MACrossoverStrategy(strategy_id="s", params={})], ["BTC/USDT"], TimeFrame.ONE_MINUTE
    )
    engine = fresh._build_engine("bt-000000000009")[0]
    await engine.start("bt-000000000009")
    coro = engine.stop()
    with pytest.raises(StopIteration):
        coro.send(None)
    assert engine._state == EngineState.STOPPED
