"""
tests/unit/test_strategy_engine.py
------------------------------------
Unit tests for the StrategyEngine lifecycle.

Module under test
-----------------
packages/trading/strategy_engine.py

Test coverage
-------------
- Initial state is IDLE
- start() transitions IDLE -> STARTING -> RUNNING (mocked dependencies)
- start() on a non-IDLE engine raises RuntimeError
- stop() transitions RUNNING -> STOPPING -> STOPPED
- stop() called from IDLE is a silent no-op (logs warning, no exception)
- stop() called from ERROR state succeeds
- Double start() raises RuntimeError on the second call
- Properties: run_id, run_mode, bar_count, portfolio before/after start
- get_status() structure and required keys
- get_status() includes portfolio_summary only in RUNNING/STOPPED states
- run_backtest() guard: requires RUNNING + BACKTEST mode
- run_live_loop() guard: requires RUNNING + PAPER or LIVE mode
- start() failure (on_start raises) transitions engine to ERROR state
- market_data.connect() called in PAPER/LIVE mode, not in BACKTEST mode
- strategy.on_start() called during engine start

Async note
----------
pyproject.toml sets asyncio_mode = "auto", so async test functions are
discovered and run automatically without @pytest.mark.asyncio decorators.

StrEnum note
------------
EngineState uses StrEnum with auto() which produces lowercase string values
(e.g. EngineState.RUNNING.value == "running"). Error messages from
strategy_engine.py embed .value directly, so match patterns use lowercase.
The guard messages in run_backtest() / run_live_loop() spell out "RUNNING"
and "BACKTEST" as literals in their f-strings — those use uppercase and
match correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.types import RunMode, TimeFrame
from trading.strategy_engine import EngineState, StrategyEngine


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    run_mode: RunMode = RunMode.BACKTEST,
    symbols: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[StrategyEngine, dict[str, Any]]:
    """
    Create a StrategyEngine with fully mocked dependencies.

    Returns the engine and a mocks dictionary keyed by dependency name so
    each test can make targeted assertions without coupling to internals.

    Parameters
    ----------
    run_mode:
        RunMode to configure the engine with.
    symbols:
        List of symbols.  Defaults to ["BTC/USDT"].
    config:
        Optional engine config overrides.

    Returns
    -------
    tuple[StrategyEngine, dict[str, Any]]
        (engine, mocks_dict)
    """
    strategy = MagicMock()
    strategy.strategy_id = "test_strategy"
    strategy.on_start = MagicMock(return_value=None)
    strategy.on_stop = MagicMock(return_value=None)

    execution = AsyncMock()
    execution.on_start = AsyncMock(return_value=None)
    execution.on_stop = AsyncMock(return_value=None)
    # get_open_orders is synchronous per BaseExecutionEngine
    execution.get_open_orders = MagicMock(return_value=[])
    execution.cancel_order = AsyncMock(return_value=None)

    market_data = AsyncMock()
    market_data.connect = AsyncMock(return_value=None)
    market_data.close = AsyncMock(return_value=None)

    risk_manager = MagicMock()
    risk_manager.kill_switch_active = False
    risk_manager.tick_cooldown = MagicMock(return_value=None)

    portfolio = MagicMock()
    portfolio.get_summary = MagicMock(return_value={
        "current_equity": "10000",
        "total_trades": 0,
    })

    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=symbols or ["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=run_mode,
        config=config,
    )

    mocks: dict[str, Any] = {
        "strategy": strategy,
        "execution": execution,
        "market_data": market_data,
        "risk_manager": risk_manager,
        "portfolio": portfolio,
    }
    return engine, mocks


# ===========================================================================
# State machine lifecycle
# ===========================================================================


class TestStrategyEngineLifecycle:
    """Tests for EngineState transitions through start() and stop()."""

    def test_initial_state_is_idle(self) -> None:
        """A freshly constructed StrategyEngine must start in IDLE state."""
        engine, _ = _make_engine()
        assert engine.state == EngineState.IDLE

    async def test_start_transitions_to_running(self) -> None:
        """
        Calling start() on an IDLE engine must end in RUNNING state.

        Intermediate STARTING state is transient and not externally visible
        once the coroutine returns.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        assert engine.state == EngineState.RUNNING

    async def test_start_on_running_raises_runtime_error(self) -> None:
        """
        Calling start() on an already-RUNNING engine must raise RuntimeError.

        The error message embeds the current state value via StrEnum.auto(),
        which produces lowercase strings — so the match uses "running".
        """
        engine, _ = _make_engine()
        await engine.start("run-001")

        with pytest.raises(RuntimeError, match="running"):
            await engine.start("run-002")

    async def test_start_on_non_idle_raises_runtime_error(self) -> None:
        """
        start() raises RuntimeError for any non-IDLE state, not just RUNNING.

        After stop() the engine is in STOPPED state; a second start() attempt
        must raise.  StrEnum.auto() gives "stopped" (lowercase).
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        assert engine.state == EngineState.STOPPED

        with pytest.raises(RuntimeError, match="stopped"):
            await engine.start("run-002")

    async def test_stop_transitions_to_stopped(self) -> None:
        """
        Calling stop() on a RUNNING engine must end in STOPPED state.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        assert engine.state == EngineState.STOPPED

    async def test_stop_on_idle_does_not_raise(self) -> None:
        """
        stop() called from IDLE silently returns — it does NOT raise.

        The StrategyEngine logs a warning for the invalid-state case but
        is designed to be safe to call defensively in teardown paths.
        """
        engine, _ = _make_engine()
        await engine.stop()
        assert engine.state == EngineState.IDLE

    async def test_double_start_raises_on_second_call(self) -> None:
        """
        Two consecutive start() calls raise RuntimeError on the second call.

        This is a focused regression guard on the double-start path.
        """
        engine, _ = _make_engine()
        await engine.start("run-first")

        with pytest.raises(RuntimeError):
            await engine.start("run-second")

    async def test_start_calls_execution_on_start(self) -> None:
        """execution_engine.on_start() must be awaited exactly once during start()."""
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].on_start.assert_awaited_once()

    async def test_start_calls_strategy_on_start(self) -> None:
        """Each injected strategy's on_start() must be called with the run_id."""
        engine, mocks = _make_engine()
        await engine.start("run-abc")
        mocks["strategy"].on_start.assert_called_once_with("run-abc")

    async def test_start_backtest_does_not_connect_market_data(self) -> None:
        """
        In BACKTEST mode, market_data.connect() must NOT be called.

        BACKTEST runs operate on pre-fetched bars and need no live connection.
        """
        engine, mocks = _make_engine(run_mode=RunMode.BACKTEST)
        await engine.start("run-bt")
        mocks["market_data"].connect.assert_not_awaited()

    async def test_start_paper_connects_market_data(self) -> None:
        """In PAPER mode, market_data.connect() must be awaited once during start()."""
        engine, mocks = _make_engine(run_mode=RunMode.PAPER)
        await engine.start("run-paper")
        mocks["market_data"].connect.assert_awaited_once()

    async def test_start_live_connects_market_data(self) -> None:
        """In LIVE mode, market_data.connect() must be awaited once during start()."""
        engine, mocks = _make_engine(run_mode=RunMode.LIVE)
        await engine.start("run-live")
        mocks["market_data"].connect.assert_awaited_once()

    async def test_double_stop_does_not_raise(self) -> None:
        """Calling stop() on an already-STOPPED engine silently returns."""
        engine, _ = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        assert engine.state == EngineState.STOPPED
        await engine.stop()
        assert engine.state == EngineState.STOPPED


# ===========================================================================
# Properties
# ===========================================================================


class TestStrategyEngineProperties:
    """Tests for the public property accessors."""

    def test_run_id_is_none_before_start(self) -> None:
        """run_id must be None before start() is called."""
        engine, _ = _make_engine()
        assert engine.run_id is None

    async def test_run_id_set_after_start(self) -> None:
        """run_id must equal the value passed to start() after it completes."""
        engine, _ = _make_engine()
        await engine.start("run-xyz-789")
        assert engine.run_id == "run-xyz-789"

    def test_run_mode_reflects_construction(self) -> None:
        """run_mode must always return the RunMode supplied at construction."""
        engine_bt, _ = _make_engine(run_mode=RunMode.BACKTEST)
        engine_paper, _ = _make_engine(run_mode=RunMode.PAPER)
        engine_live, _ = _make_engine(run_mode=RunMode.LIVE)

        assert engine_bt.run_mode == RunMode.BACKTEST
        assert engine_paper.run_mode == RunMode.PAPER
        assert engine_live.run_mode == RunMode.LIVE

    def test_bar_count_starts_at_zero(self) -> None:
        """bar_count must be 0 on a freshly constructed engine."""
        engine, _ = _make_engine()
        assert engine.bar_count == 0

    async def test_bar_count_still_zero_after_start_before_bars(self) -> None:
        """
        bar_count must remain 0 immediately after start() and before any
        bars are processed.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        assert engine.bar_count == 0

    def test_portfolio_property_returns_injected_portfolio(self) -> None:
        """
        engine.portfolio must return the exact PortfolioAccounting instance
        that was injected at construction time.
        """
        engine, mocks = _make_engine()
        assert engine.portfolio is mocks["portfolio"]

    def test_state_property_returns_engine_state_enum(self) -> None:
        """state property must return an EngineState instance."""
        engine, _ = _make_engine()
        assert isinstance(engine.state, EngineState)


# ===========================================================================
# get_status
# ===========================================================================


class TestStrategyEngineStatus:
    """Tests for the get_status() method output structure and content."""

    _REQUIRED_KEYS = {
        "state",
        "run_id",
        "run_mode",
        "timeframe",
        "symbols",
        "strategies",
        "bar_count",
        "total_signals",
        "total_orders",
        "total_fills",
    }

    def test_get_status_returns_dict_with_required_keys(self) -> None:
        """
        get_status() must return a dict containing all documented keys.

        These keys are part of the public API contract and must always
        be present regardless of engine state.
        """
        engine, _ = _make_engine()
        status = engine.get_status()
        missing = self._REQUIRED_KEYS - status.keys()
        assert not missing, f"get_status() is missing keys: {missing}"

    def test_status_state_is_idle_string_before_start(self) -> None:
        """
        The 'state' key in get_status() must be the string value 'idle'
        when the engine has not been started.

        EngineState is a StrEnum so .value == str(member) in lowercase.
        """
        engine, _ = _make_engine()
        status = engine.get_status()
        assert status["state"] == EngineState.IDLE.value

    async def test_status_reflects_state_after_start(self) -> None:
        """
        After start(), get_status()['state'] must reflect 'running'.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        status = engine.get_status()
        assert status["state"] == EngineState.RUNNING.value

    async def test_status_run_id_set_after_start(self) -> None:
        """get_status()['run_id'] must equal the run_id passed to start()."""
        engine, _ = _make_engine()
        await engine.start("run-status-test")
        status = engine.get_status()
        assert status["run_id"] == "run-status-test"

    def test_status_run_id_none_before_start(self) -> None:
        """get_status()['run_id'] must be None before start() is called."""
        engine, _ = _make_engine()
        status = engine.get_status()
        assert status["run_id"] is None

    def test_status_symbols_reflects_construction(self) -> None:
        """get_status()['symbols'] must list the symbols supplied at construction."""
        engine, _ = _make_engine(symbols=["ETH/USDT", "BTC/USDT"])
        status = engine.get_status()
        assert status["symbols"] == ["ETH/USDT", "BTC/USDT"]

    def test_status_no_portfolio_summary_when_idle(self) -> None:
        """
        get_status() must NOT include 'portfolio_summary' when state is IDLE.

        The portfolio_summary is gated to RUNNING and STOPPED states only.
        """
        engine, _ = _make_engine()
        status = engine.get_status()
        assert "portfolio_summary" not in status

    async def test_status_includes_portfolio_summary_when_running(self) -> None:
        """
        get_status() must include 'portfolio_summary' when state is RUNNING.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        status = engine.get_status()
        assert "portfolio_summary" in status

    async def test_status_includes_portfolio_summary_when_stopped(self) -> None:
        """
        get_status() must include 'portfolio_summary' when state is STOPPED.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        status = engine.get_status()
        assert "portfolio_summary" in status

    async def test_status_no_portfolio_summary_when_error(self) -> None:
        """
        get_status() must NOT include 'portfolio_summary' when state is ERROR.

        ERROR is not in the set {RUNNING, STOPPED} that gates summary inclusion.
        """
        engine, mocks = _make_engine()
        mocks["execution"].on_start.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await engine.start("run-err")

        assert engine.state == EngineState.ERROR
        status = engine.get_status()
        assert "portfolio_summary" not in status

    def test_status_strategies_lists_strategy_ids(self) -> None:
        """
        get_status()['strategies'] must be a list of strategy_id strings.
        """
        engine, mocks = _make_engine()
        status = engine.get_status()
        assert status["strategies"] == ["test_strategy"]

    async def test_status_bar_count_zero_before_processing(self) -> None:
        """
        get_status()['bar_count'] must be 0 immediately after start()
        before any bars are processed.
        """
        engine, _ = _make_engine()
        await engine.start("run-001")
        status = engine.get_status()
        assert status["bar_count"] == 0

    def test_status_values_have_correct_types(self) -> None:
        """get_status() values must be the documented types for serialization."""
        engine, _ = _make_engine()
        status = engine.get_status()
        assert isinstance(status["state"], str)
        assert isinstance(status["run_mode"], str)
        assert isinstance(status["symbols"], list)
        assert isinstance(status["bar_count"], int)


# ===========================================================================
# Guards: run_backtest and run_live_loop
# ===========================================================================


class TestStrategyEngineGuards:
    """Tests for method guards that enforce state and mode preconditions."""

    async def test_run_backtest_before_start_raises_runtime_error(self) -> None:
        """
        run_backtest() called before start() must raise RuntimeError.

        The engine is in IDLE state; the error message contains the literal
        string "RUNNING" (the guard spells it out, not via .value).
        """
        engine, _ = _make_engine(run_mode=RunMode.BACKTEST)
        # Engine not started — state is IDLE

        with pytest.raises(RuntimeError, match="RUNNING"):
            await engine.run_backtest(bars_by_symbol={"BTC/USDT": []})

    async def test_run_backtest_in_paper_mode_raises_runtime_error(self) -> None:
        """
        run_backtest() called on a PAPER-mode engine must raise RuntimeError
        indicating the mode mismatch.

        The guard message spells out "RunMode.BACKTEST" as a literal.
        """
        engine, _ = _make_engine(run_mode=RunMode.PAPER)
        await engine.start("run-paper")

        with pytest.raises(RuntimeError, match="BACKTEST"):
            await engine.run_backtest(bars_by_symbol={"BTC/USDT": []})

    async def test_run_backtest_in_live_mode_raises_runtime_error(self) -> None:
        """
        run_backtest() on a LIVE-mode engine must raise RuntimeError.
        """
        engine, _ = _make_engine(run_mode=RunMode.LIVE)
        await engine.start("run-live")

        with pytest.raises(RuntimeError, match="BACKTEST"):
            await engine.run_backtest(bars_by_symbol={"BTC/USDT": []})

    async def test_run_live_loop_before_start_raises_runtime_error(self) -> None:
        """
        run_live_loop() called before start() must raise RuntimeError.

        The guard message contains the literal string "RUNNING".
        """
        engine, _ = _make_engine(run_mode=RunMode.PAPER)

        with pytest.raises(RuntimeError, match="RUNNING"):
            await engine.run_live_loop()

    async def test_run_live_loop_in_backtest_mode_raises_runtime_error(self) -> None:
        """
        run_live_loop() on a BACKTEST-mode engine must raise RuntimeError
        indicating that PAPER or LIVE mode is required.

        The guard message spells out "PAPER" as a literal.
        """
        engine, _ = _make_engine(run_mode=RunMode.BACKTEST)
        await engine.start("run-bt")

        with pytest.raises(RuntimeError, match="PAPER"):
            await engine.run_live_loop()

    async def test_run_backtest_raises_value_error_on_missing_symbol_bars(self) -> None:
        """
        run_backtest() raises ValueError when bars_by_symbol does not include
        all configured symbols.
        """
        engine, _ = _make_engine(run_mode=RunMode.BACKTEST, symbols=["BTC/USDT"])
        await engine.start("run-bt")

        # Passing an empty dict — BTC/USDT bars are missing
        with pytest.raises(ValueError, match="BTC/USDT"):
            await engine.run_backtest(bars_by_symbol={})


# ===========================================================================
# Error handling
# ===========================================================================


class TestStrategyEngineErrorHandling:
    """Tests for failure paths during start() and stop()."""

    async def test_start_failure_transitions_to_error_state(self) -> None:
        """
        When execution_engine.on_start() raises, the engine must transition
        to ERROR state and re-raise the exception.

        This ensures callers can distinguish a failed start from a clean stop.
        """
        engine, mocks = _make_engine()
        mocks["execution"].on_start.side_effect = RuntimeError("exchange connection failed")

        with pytest.raises(RuntimeError, match="exchange connection failed"):
            await engine.start("run-fail")

        assert engine.state == EngineState.ERROR

    async def test_strategy_on_start_failure_transitions_to_error_state(self) -> None:
        """
        When a strategy's on_start() raises, the engine must transition
        to ERROR state and re-raise the exception.
        """
        engine, mocks = _make_engine()
        mocks["strategy"].on_start.side_effect = ValueError("bad strategy config")

        with pytest.raises(ValueError, match="bad strategy config"):
            await engine.start("run-strategy-fail")

        assert engine.state == EngineState.ERROR

    async def test_stop_from_error_state_succeeds(self) -> None:
        """
        stop() must succeed when the engine is in ERROR state.

        This is critical for cleanup after a failed start — teardown paths
        should not themselves raise.
        """
        engine, mocks = _make_engine()
        mocks["execution"].on_start.side_effect = RuntimeError("startup failure")

        with pytest.raises(RuntimeError):
            await engine.start("run-fail")

        assert engine.state == EngineState.ERROR

        # Must not raise
        await engine.stop()
        assert engine.state == EngineState.STOPPED

    async def test_stop_calls_execution_on_stop(self) -> None:
        """execution_engine.on_stop() must be awaited during stop()."""
        engine, mocks = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        mocks["execution"].on_stop.assert_awaited_once()

    async def test_stop_calls_strategy_on_stop(self) -> None:
        """Each strategy's on_stop() must be called during stop()."""
        engine, mocks = _make_engine()
        await engine.start("run-001")
        await engine.stop()
        mocks["strategy"].on_stop.assert_called_once()

    async def test_stop_backtest_does_not_close_market_data(self) -> None:
        """
        In BACKTEST mode, market_data.close() must NOT be called during stop().

        BACKTEST mode never opened a market data connection, so there is
        nothing to close.
        """
        engine, mocks = _make_engine(run_mode=RunMode.BACKTEST)
        await engine.start("run-bt")
        await engine.stop()
        mocks["market_data"].close.assert_not_awaited()

    async def test_stop_paper_closes_market_data(self) -> None:
        """In PAPER mode, market_data.close() must be awaited during stop()."""
        engine, mocks = _make_engine(run_mode=RunMode.PAPER)
        await engine.start("run-paper")
        await engine.stop()
        mocks["market_data"].close.assert_awaited_once()

    async def test_stop_live_closes_market_data(self) -> None:
        """In LIVE mode, market_data.close() must be awaited during stop()."""
        engine, mocks = _make_engine(run_mode=RunMode.LIVE)
        await engine.start("run-live")
        await engine.stop()
        mocks["market_data"].close.assert_awaited_once()


# ===========================================================================
# Constructor validation
# ===========================================================================


class TestStrategyEngineConstructorValidation:
    """Tests for __init__ guards that reject invalid construction arguments."""

    def test_empty_strategies_list_raises_value_error(self) -> None:
        """
        Constructing a StrategyEngine with an empty strategies list must
        raise ValueError immediately.
        """
        with pytest.raises(ValueError, match="strategy"):
            StrategyEngine(
                strategies=[],
                execution_engine=AsyncMock(),
                risk_manager=MagicMock(),
                market_data=AsyncMock(),
                portfolio=MagicMock(),
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                run_mode=RunMode.BACKTEST,
            )

    def test_empty_symbols_list_raises_value_error(self) -> None:
        """
        Constructing a StrategyEngine with an empty symbols list must
        raise ValueError immediately.
        """
        strategy = MagicMock()
        strategy.strategy_id = "test"

        with pytest.raises(ValueError, match="symbol"):
            StrategyEngine(
                strategies=[strategy],
                execution_engine=AsyncMock(),
                risk_manager=MagicMock(),
                market_data=AsyncMock(),
                portfolio=MagicMock(),
                symbols=[],
                timeframe=TimeFrame.ONE_HOUR,
                run_mode=RunMode.BACKTEST,
            )

    def test_repr_includes_state_and_mode(self) -> None:
        """
        __repr__ must include at least the state and run_mode values so that
        logs and debugger output are informative.
        """
        engine, _ = _make_engine(run_mode=RunMode.BACKTEST)
        representation = repr(engine)
        assert "idle" in representation.lower() or "IDLE" in representation
        assert "backtest" in representation.lower() or "BACKTEST" in representation
