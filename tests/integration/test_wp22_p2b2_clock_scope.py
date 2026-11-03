"""WP22-P2B2 -- the run-scoped fallback clock and the ``run_backtest`` re-entrancy guard.

Folds the non-blocking P2b.1 review items:

    P2B1-CR-01 / SEC-05  the fallback is installed only after the attach succeeded (inside the
                         ``with``), so a rejecting execution engine or a raise before the ``with``
                         leaves no clock behind
    P2B1-CR-02 / SEC-06  the restore targets the execution engine captured at entry, and only
                         ``if execution._clock is bar_clock``
    P2B1-CR-03           cancellation restores the engine, the execution engine and the ambient
                         clock
    P2B1-SEC-07          a second concurrent ``run_backtest`` on one engine raises
                         ``RuntimeError`` with a type-name-only message (R11)

Guards are explicit raises, never ``assert`` (``python -O`` safe).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.types import RunMode, TimeFrame
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.test_wp22_bar_clock import (
    _FLAT_72,
    LEGACY,
    POST_WARMUP_OPEN,
    SYM,
    _live_engine,
    _strategy,
)
from trading.backtest import BacktestRunner
from trading.clock import WALL_CLOCK, BarClock, ambient_clock
from trading.engines.paper import PaperExecutionEngine
from trading.risk_manager import DefaultRiskManager
from trading.strategy_engine import EngineState, StrategyEngine

_BARS = {SYM: scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)}


def _direct(tag: str) -> tuple[StrategyEngine, PaperExecutionEngine]:
    """A BACKTEST engine built without a clock (fallback path) over a paper execution engine."""
    runner = BacktestRunner(
        strategies=[_strategy({0: "buy", 6: "sell"})],
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        seed=7,
        enable_circuit_breaker=False,
        cost_profile=LEGACY,
    )
    engine, _portfolio, _rm, exec_engine = runner._build_engine(tag)
    assert isinstance(exec_engine, PaperExecutionEngine)
    return engine, exec_engine


def _backtest_engine_over_live() -> tuple[StrategyEngine, Any]:
    """A BACKTEST engine, no clock, over a LiveExecutionEngine (which rejects a BarClock)."""
    strategy = MagicMock()
    strategy.strategy_id = "s"
    strategy.min_bars_required = 20
    strategy.htf_timeframes = None
    live = _live_engine()
    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=live,
        risk_manager=MagicMock(),
        market_data=AsyncMock(),
        portfolio=MagicMock(),
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
    )
    # ``start()`` would call ``LiveExecutionEngine.on_start`` (exchange); the state is the only
    # thing ``run_backtest`` reads, so the test sets it directly.
    engine._state = EngineState.RUNNING
    return engine, live


def _all_wall(engine: StrategyEngine, exec_engine: Any) -> bool:
    return (
        engine._clock is WALL_CLOCK
        and exec_engine._clock is WALL_CLOCK
        and ambient_clock() is WALL_CLOCK
    )


class TestInstallOnlyAfterAttachSucceeds:
    """P2B1-CR-01 / SEC-05."""

    async def test_two_runs_over_a_live_exec_engine_both_raise_and_leave_the_wall_clock(
        self,
    ) -> None:
        engine, live = _backtest_engine_over_live()
        for _attempt in range(2):
            with pytest.raises(ValueError):
                await engine.run_backtest(_BARS)
            # the first call must not leave a BarClock that turns the second into an
            # "injected clock" run (which skipped the attach and ran the loop)
            assert engine._clock is WALL_CLOCK
            assert live._clock is WALL_CLOCK
            assert ambient_clock() is WALL_CLOCK
            assert engine._backtest_running is False

    async def test_a_raise_before_the_with_leaves_no_clock_behind(self) -> None:
        engine, exec_engine = _direct("log-raise")
        await engine.start("log-raise")
        try:
            log = MagicMock()
            log.info.side_effect = OSError("log sink down")
            with patch.object(engine, "_log", log):
                with pytest.raises(OSError, match="log sink down"):
                    await engine.run_backtest(_BARS)
            assert _all_wall(engine, exec_engine)
            assert engine._backtest_running is False
            # ... and the engine is still usable
            await engine.run_backtest(_BARS)
            assert _all_wall(engine, exec_engine)
        finally:
            await engine.stop()


class TestRestoreTargetsTheCapturedEngine:
    """P2B1-CR-02 / SEC-06."""

    async def test_swapping_the_execution_engine_mid_run_never_writes_into_the_new_one(
        self,
    ) -> None:
        engine, exec_engine = _direct("swap")
        other_clock = BarClock(POST_WARMUP_OPEN)
        other = PaperExecutionEngine("other", DefaultRiskManager("other"))
        other.attach_clock(other_clock)
        swapped = False

        async def _swap(*_args: Any, **_kw: Any) -> None:
            nonlocal swapped
            if not swapped:
                swapped = True
                engine._execution_engine = other

        await engine.start("swap")
        try:
            with patch.object(engine, "_process_bar", new=_swap):
                await engine.run_backtest(_BARS)
        finally:
            engine._execution_engine = exec_engine
            await engine.stop()
        assert swapped
        assert other._clock is other_clock  # never written by the restore
        assert exec_engine._clock is WALL_CLOCK  # the captured engine is restored
        assert engine._clock is WALL_CLOCK and ambient_clock() is WALL_CLOCK

    async def test_an_exec_clock_replaced_mid_run_is_not_overwritten(self) -> None:
        engine, exec_engine = _direct("replaced")
        replacement = BarClock(POST_WARMUP_OPEN)
        done = False

        async def _replace(*_args: Any, **_kw: Any) -> None:
            nonlocal done
            if not done:
                done = True
                exec_engine.attach_clock(replacement)

        await engine.start("replaced")
        try:
            with patch.object(engine, "_process_bar", new=_replace):
                await engine.run_backtest(_BARS)
        finally:
            await engine.stop()
        assert done
        assert exec_engine._clock is replacement  # only what the scope installed is undone
        assert engine._clock is WALL_CLOCK and ambient_clock() is WALL_CLOCK


class TestCancelAndReentrancy:
    """P2B1-CR-03 and P2B1-SEC-07."""

    async def test_cancel_mid_run_restores_every_clock(self) -> None:
        engine, exec_engine = _direct("cancel")
        started, gate = asyncio.Event(), asyncio.Event()
        seen: list[Any] = []

        async def _block(*_args: Any, **_kw: Any) -> None:
            seen.append(ambient_clock())  # the run task's own context
            started.set()
            await gate.wait()

        await engine.start("cancel")
        try:
            with patch.object(engine, "_process_bar", new=_block):
                after: list[Any] = []

                async def _run_and_probe() -> None:
                    try:
                        await engine.run_backtest(_BARS)
                    finally:
                        after.append(ambient_clock())  # the run task's own context

                task = asyncio.ensure_future(_run_and_probe())
                await asyncio.wait_for(started.wait(), timeout=10)
                # non-vacuous: all three really are the bar clock mid-run
                assert isinstance(engine._clock, BarClock)
                assert exec_engine._clock is engine._clock
                assert seen == [engine._clock]
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert after == [WALL_CLOCK]
        finally:
            await engine.stop()
        assert _all_wall(engine, exec_engine)
        assert engine._backtest_running is False

    async def test_a_concurrent_second_run_is_rejected_and_the_first_is_untouched(self) -> None:
        engine, exec_engine = _direct("reenter")
        started, gate = asyncio.Event(), asyncio.Event()
        seen: list[Any] = []

        async def _block(*_args: Any, **_kw: Any) -> None:
            seen.append(ambient_clock())  # the first run's own context
            started.set()
            await gate.wait()

        await engine.start("reenter")
        try:
            with patch.object(engine, "_process_bar", new=_block):
                first = asyncio.ensure_future(engine.run_backtest(_BARS))
                await asyncio.wait_for(started.wait(), timeout=10)
                running_clock = engine._clock
                assert isinstance(running_clock, BarClock)
                # bounded: with no guard the second call would block on the gate forever
                with pytest.raises(RuntimeError) as err:
                    await asyncio.wait_for(engine.run_backtest(_BARS), timeout=5)
                # R11: the type name only, no clock repr / stamp / run id
                assert str(err.value) == "run_backtest() is already running on this StrategyEngine"
                # the rejected call must not restore (or replace) the running call's clocks
                assert engine._clock is running_clock
                assert exec_engine._clock is running_clock
                assert seen == [running_clock]
                assert engine._backtest_running is True
                gate.set()
                await asyncio.wait_for(first, timeout=30)
            assert _all_wall(engine, exec_engine)
            assert engine._backtest_running is False
            # the guard is released: a later sequential run is accepted
            await engine.run_backtest(_BARS)
            assert _all_wall(engine, exec_engine)
        finally:
            gate.set()
            await engine.stop()
