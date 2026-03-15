"""
tests/unit/test_multi_timeframe.py
------------------------------------
Unit tests for the Sprint 28 Multi-Timeframe Analysis feature.

Modules under test
------------------
packages/common/models.py         -- MultiTimeframeContext dataclass
packages/trading/strategy.py     -- BaseStrategy.on_bar() mtf_context param,
                                     BaseStrategy.htf_timeframes property
packages/trading/strategy_engine.py -- _build_mtf_context(), _call_strategy_on_bar()
                                        with mtf_context, run_backtest() with htf_bars
packages/trading/backtest.py     -- BacktestRunner with htf_timeframes + htf_bars params

Test coverage
-------------
TestMultiTimeframeContext (3 tests)
- Default construction has empty htf_bars dict
- Frozen -- cannot mutate after creation
- Construction with data preserves structure

TestBuildMtfContext (6 tests)
- Returns None when _htf_bars is None (no HTF configured)
- Returns context with all bars when all are within timeframe
- Filters out bars with look-ahead bias (bar.timestamp + duration > current_timestamp)
- Includes bar whose period exactly completes at current_timestamp (boundary)
- Excludes bar that opens at current_timestamp (not yet complete)
- Multi-symbol filtering works independently per symbol

TestCallStrategyOnBarWithMtf (3 tests)
- mtf_context=None is passed when no HTF data
- mtf_context is passed through to strategy.on_bar()
- Multiple symbols all receive the same mtf_context object

TestRunBacktestWithHtfBars (3 tests)
- run_backtest without htf_bars -- works as before, mtf_context=None passed to strategy
- run_backtest with htf_bars -- stores and passes through to _build_mtf_context
- BacktestRunner.run() passes htf_bars to engine.run_backtest()

TestStrategyHtfTimeframes (2 tests)
- Default htf_timeframes returns empty list
- Subclass can override htf_timeframes

Async note
----------
pyproject.toml sets asyncio_mode = "auto"; no @pytest.mark.asyncio needed.

Design notes
------------
- _make_engine() and _make_bar() mirror the pattern in test_trailing_stop.py exactly.
- For _build_mtf_context tests, the engine._htf_bars attribute is set directly after
  construction, bypassing run_backtest(), to isolate the method under test.
- Look-ahead bias tests use concrete timestamps:
    Primary bar at 2024-01-01 04:00 UTC (hour 4).
    4h HTF bar at 2024-01-01 00:00 UTC -- complete at 04:00, INCLUDED at primary 04:00.
    4h HTF bar at 2024-01-01 04:00 UTC -- complete at 08:00, EXCLUDED at primary 04:00.
  The boundary condition: bar.timestamp.timestamp() + tf_duration <= current.timestamp()
  means a bar opening at T=0 with duration 14400 s is complete at T=14400, and is
  included if current_timestamp >= T=14400 (i.e. bar close <= current bar open).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, TimeFrame
from trading.strategy import BaseStrategy
from trading.models import Signal
from trading.strategy_engine import StrategyEngine


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_bar(
    *,
    symbol: str = "BTC/USD",
    close: str | Decimal = "50000",
    timestamp: datetime | None = None,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
) -> OHLCVBar:
    """Construct a minimal OHLCVBar satisfying all OHLCV constraints."""
    ts = timestamp or datetime(2024, 1, 1, tzinfo=UTC)
    close_d = Decimal(str(close))
    high_d = (close_d * Decimal("1.01")).quantize(Decimal("0.01"))
    low_d = (close_d * Decimal("0.99")).quantize(Decimal("0.01"))
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=close_d,
        high=high_d,
        low=low_d,
        close=close_d,
        volume=Decimal("100"),
    )


def _make_engine(
    *,
    symbols: list[str] | None = None,
    run_mode: RunMode = RunMode.BACKTEST,
    config: dict[str, Any] | None = None,
) -> tuple[StrategyEngine, dict[str, Any]]:
    """
    Create a StrategyEngine with fully mocked dependencies.

    Mirrors the pattern from test_trailing_stop.py and
    test_strategy_engine_bar_loop.py.

    Returns the engine (not yet started) and a mocks dict.
    Callers must call ``await engine.start(run_id)`` before bar-loop methods.
    """
    strategy = MagicMock()
    strategy.strategy_id = "test_strategy"
    strategy.min_bars_required = 20
    strategy.on_start = MagicMock(return_value=None)
    strategy.on_stop = MagicMock(return_value=None)
    strategy.on_bar = MagicMock(return_value=[])

    execution = AsyncMock()
    execution.on_start = AsyncMock(return_value=None)
    execution.on_stop = AsyncMock(return_value=None)
    execution.get_open_orders = MagicMock(return_value=[])
    execution.cancel_order = AsyncMock(return_value=None)
    execution.check_resting_orders = None  # suppress AsyncMock warnings

    market_data = AsyncMock()
    market_data.connect = AsyncMock(return_value=None)
    market_data.close = AsyncMock(return_value=None)

    risk_manager = MagicMock()
    risk_manager.kill_switch_active = False
    risk_manager.tick_cooldown = MagicMock(return_value=None)
    risk_manager.update_after_fill = MagicMock(return_value=None)

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
        symbols=symbols or ["BTC/USD"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=run_mode,
        config=config or {},
    )

    mocks: dict[str, Any] = {
        "strategy": strategy,
        "execution": execution,
        "market_data": market_data,
        "risk_manager": risk_manager,
        "portfolio": portfolio,
    }
    return engine, mocks


# ---------------------------------------------------------------------------
# Minimal concrete BaseStrategy subclass for htf_timeframes tests
# ---------------------------------------------------------------------------


class _PassthroughStrategy(BaseStrategy):
    """
    Concrete strategy that records every on_bar call for assertion.

    The received_mtf_contexts list collects the mtf_context kwarg from
    every on_bar invocation, enabling assertions about what the engine
    passes through.
    """

    def __init__(self, strategy_id: str = "passthrough") -> None:
        super().__init__(strategy_id=strategy_id)
        self.received_mtf_contexts: list[MultiTimeframeContext | None] = []

    def on_bar(
        self,
        bars: Any,
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        self.received_mtf_contexts.append(mtf_context)
        return []


class _MtfAwareStrategy(_PassthroughStrategy):
    """Strategy that declares 4h as a required higher timeframe."""

    @property
    def htf_timeframes(self) -> list[str]:
        return ["4h"]


# ===========================================================================
# TestMultiTimeframeContext
# ===========================================================================


class TestMultiTimeframeContext:
    """
    Unit tests for the MultiTimeframeContext dataclass defined in
    packages/common/models.py.
    """

    def test_default_construction_has_empty_htf_bars(self) -> None:
        """
        MultiTimeframeContext() must construct without arguments and
        initialise htf_bars to an empty dict via the default_factory.
        """
        ctx = MultiTimeframeContext()

        assert isinstance(ctx.htf_bars, dict)
        assert len(ctx.htf_bars) == 0

    def test_frozen_raises_on_mutation(self) -> None:
        """
        As a frozen dataclass, any attempt to assign a new value to
        htf_bars after construction must raise FrozenInstanceError
        (or AttributeError depending on Python version).
        """
        ctx = MultiTimeframeContext()

        with pytest.raises((AttributeError, TypeError)):
            ctx.htf_bars = {"4h": {}}  # type: ignore[misc]

    def test_construction_with_data_preserves_structure(self) -> None:
        """
        Constructing MultiTimeframeContext with a populated htf_bars dict
        must store the data exactly as provided, preserving both the
        outer timeframe key and the inner symbol-to-bar-list mapping.
        """
        bar = _make_bar(timeframe=TimeFrame.FOUR_HOURS)
        htf_data: dict[str, dict[str, list[OHLCVBar]]] = {
            "4h": {"BTC/USD": [bar]},
        }

        ctx = MultiTimeframeContext(htf_bars=htf_data)

        assert "4h" in ctx.htf_bars
        assert "BTC/USD" in ctx.htf_bars["4h"]
        assert ctx.htf_bars["4h"]["BTC/USD"] == [bar]


# ===========================================================================
# TestBuildMtfContext
# ===========================================================================


class TestBuildMtfContext:
    """
    Unit tests for StrategyEngine._build_mtf_context(timestamp).

    Each test sets engine._htf_bars directly to isolate the method from
    run_backtest(), then calls _build_mtf_context() with a specific
    timestamp and inspects the returned MultiTimeframeContext (or None).

    Look-ahead bias boundary
    -------------------------
    The filter condition is:
        bar.timestamp.timestamp() + tf_duration_seconds <= current_timestamp.timestamp()

    For a 4h bar (14400 s):
    - Bar at 2024-01-01 00:00 UTC: complete at 04:00. current=04:00 -> INCLUDED.
    - Bar at 2024-01-01 04:00 UTC: complete at 08:00. current=04:00 -> EXCLUDED.
    """

    def test_returns_none_when_htf_bars_is_none(self) -> None:
        """
        When _htf_bars is None (no higher timeframes configured), the method
        must return None rather than an empty MultiTimeframeContext.

        Returning None lets calling code take a fast-path and avoids
        constructing an unused context object on every bar.
        """
        engine, _ = _make_engine()
        # Default state: _htf_bars is None (not set via run_backtest)
        assert engine._htf_bars is None  # type: ignore[attr-defined]

        # Ensure no FGI global client leaks from other tests (Sprint 32)
        import data.sentiment as _sent_mod
        _prev = _sent_mod._global_client
        _sent_mod._global_client = None
        try:
            result = engine._build_mtf_context(datetime(2024, 1, 1, 4, tzinfo=UTC))
        finally:
            _sent_mod._global_client = _prev

        assert result is None

    def test_returns_context_with_all_bars_when_no_bias(self) -> None:
        """
        When all provided 4h bars are before the current primary bar's
        timestamp, they must all appear in the returned context.

        Both bars open at hours 0 and 4, current = hour 12 -- both
        4h periods complete well before hour 12.
        """
        engine, _ = _make_engine()

        bar_00 = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC),
        )
        bar_04 = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 4, tzinfo=UTC),
        )

        engine._htf_bars = {"4h": {"BTC/USD": [bar_00, bar_04]}}  # type: ignore[attr-defined]

        current_ts = datetime(2024, 1, 1, 12, tzinfo=UTC)
        ctx = engine._build_mtf_context(current_ts)

        assert ctx is not None
        assert "4h" in ctx.htf_bars
        assert len(ctx.htf_bars["4h"]["BTC/USD"]) == 2

    def test_filters_bar_with_look_ahead_bias(self) -> None:
        """
        A 4h bar that opens at the current primary bar timestamp is NOT
        yet complete (completes at current + 4h) and must be excluded.

        current_timestamp = 2024-01-01 04:00 UTC
        Bar opens at 04:00 -- complete at 08:00 -- look-ahead if included now.
        """
        engine, _ = _make_engine()

        # This bar opens at 04:00, period ends at 08:00 -- future data
        bar_future = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 4, tzinfo=UTC),
        )

        engine._htf_bars = {"4h": {"BTC/USD": [bar_future]}}  # type: ignore[attr-defined]

        current_ts = datetime(2024, 1, 1, 4, tzinfo=UTC)
        ctx = engine._build_mtf_context(current_ts)

        assert ctx is not None
        assert ctx.htf_bars["4h"]["BTC/USD"] == []

    def test_includes_bar_whose_period_exactly_completes_at_current_timestamp(
        self,
    ) -> None:
        """
        A 4h bar that opens at 00:00 UTC is complete at exactly 04:00 UTC.
        When the current primary bar is also at 04:00 UTC, this bar's period
        has just finished and it MUST be included (boundary-inclusive).

        Boundary condition: bar.timestamp.timestamp() + 14400 <= current.timestamp()
        00:00 + 14400 s = 04:00 -- equal to current -> INCLUDED.
        """
        engine, _ = _make_engine()

        bar_boundary = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC),
        )

        engine._htf_bars = {"4h": {"BTC/USD": [bar_boundary]}}  # type: ignore[attr-defined]

        # current_timestamp is exactly when the 4h bar at 00:00 completes
        current_ts = datetime(2024, 1, 1, 4, tzinfo=UTC)
        ctx = engine._build_mtf_context(current_ts)

        assert ctx is not None
        assert len(ctx.htf_bars["4h"]["BTC/USD"]) == 1
        assert ctx.htf_bars["4h"]["BTC/USD"][0] == bar_boundary

    def test_excludes_bar_opening_at_current_timestamp(self) -> None:
        """
        Complements the boundary-inclusive test: a bar that OPENS at the
        current timestamp (not closes) must be EXCLUDED.

        current = 2024-01-01 04:00.
        Bar at 04:00 closes at 08:00 -- not yet complete at bar=04:00.
        Bar at 00:00 closes at 04:00 -- complete, included.

        After filtering we must have exactly the 00:00 bar.
        """
        engine, _ = _make_engine()

        bar_complete = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC),
        )
        bar_incomplete = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 4, tzinfo=UTC),
        )

        engine._htf_bars = {  # type: ignore[attr-defined]
            "4h": {"BTC/USD": [bar_complete, bar_incomplete]}
        }

        current_ts = datetime(2024, 1, 1, 4, tzinfo=UTC)
        ctx = engine._build_mtf_context(current_ts)

        assert ctx is not None
        included = ctx.htf_bars["4h"]["BTC/USD"]
        assert len(included) == 1
        assert included[0] == bar_complete

    def test_multi_symbol_filtering_is_independent_per_symbol(self) -> None:
        """
        When two symbols have different numbers of completed 4h bars at the
        current timestamp, each symbol's filter must be applied independently.

        BTC/USD has a bar at 00:00 (complete at 04:00) -> INCLUDED.
        ETH/USD has a bar at 04:00 (complete at 08:00) -> EXCLUDED.
        At current_timestamp = 04:00: BTC gets 1 bar, ETH gets 0 bars.
        """
        engine, _ = _make_engine(symbols=["BTC/USD", "ETH/USD"])

        btc_bar = _make_bar(
            symbol="BTC/USD",
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC),
        )
        eth_bar = _make_bar(
            symbol="ETH/USD",
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 4, tzinfo=UTC),
        )

        engine._htf_bars = {  # type: ignore[attr-defined]
            "4h": {
                "BTC/USD": [btc_bar],
                "ETH/USD": [eth_bar],
            }
        }

        current_ts = datetime(2024, 1, 1, 4, tzinfo=UTC)
        ctx = engine._build_mtf_context(current_ts)

        assert ctx is not None
        assert len(ctx.htf_bars["4h"]["BTC/USD"]) == 1
        assert ctx.htf_bars["4h"]["ETH/USD"] == []


# ===========================================================================
# TestCallStrategyOnBarWithMtf
# ===========================================================================


class TestCallStrategyOnBarWithMtf:
    """
    Unit tests verifying that _call_strategy_on_bar correctly passes
    mtf_context through to strategy.on_bar().
    """

    async def test_none_mtf_context_is_passed_when_no_htf(self) -> None:
        """
        When mtf_context is None (no HTF bars configured), _call_strategy_on_bar
        must pass mtf_context=None to strategy.on_bar().

        Verified by inspecting the keyword arguments of the on_bar mock call.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        bar = _make_bar()
        engine._call_strategy_on_bar(  # type: ignore[attr-defined]
            mocks["strategy"],
            {"BTC/USD": [bar]},
            mtf_context=None,
        )

        mocks["strategy"].on_bar.assert_called_once()
        _, kwargs = mocks["strategy"].on_bar.call_args
        assert kwargs.get("mtf_context") is None

    async def test_mtf_context_is_passed_through_to_strategy_on_bar(
        self,
    ) -> None:
        """
        When a non-None MultiTimeframeContext is supplied to
        _call_strategy_on_bar, it must be forwarded verbatim to
        strategy.on_bar() as the mtf_context keyword argument.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        bar = _make_bar(
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC),
        )
        ctx = MultiTimeframeContext(htf_bars={"4h": {"BTC/USD": [bar]}})

        engine._call_strategy_on_bar(  # type: ignore[attr-defined]
            mocks["strategy"],
            {"BTC/USD": [_make_bar()]},
            mtf_context=ctx,
        )

        mocks["strategy"].on_bar.assert_called_once()
        _, kwargs = mocks["strategy"].on_bar.call_args
        assert kwargs.get("mtf_context") is ctx

    async def test_all_symbols_receive_same_mtf_context_object(self) -> None:
        """
        With two symbols, _call_strategy_on_bar must call strategy.on_bar
        for each symbol and pass the same mtf_context object to every call.

        The context must be identical (same object identity) for all symbols
        in a single bar, since context represents the global HTF state.
        """
        engine, mocks = _make_engine(symbols=["BTC/USD", "ETH/USD"])
        await engine.start("run-001")

        ctx = MultiTimeframeContext(htf_bars={"4h": {}})
        bar_btc = _make_bar(symbol="BTC/USD")
        bar_eth = _make_bar(symbol="ETH/USD")

        engine._call_strategy_on_bar(  # type: ignore[attr-defined]
            mocks["strategy"],
            {"BTC/USD": [bar_btc], "ETH/USD": [bar_eth]},
            mtf_context=ctx,
        )

        # on_bar called once per symbol (2 total)
        assert mocks["strategy"].on_bar.call_count == 2
        for single_call in mocks["strategy"].on_bar.call_args_list:
            _, kwargs = single_call
            assert kwargs.get("mtf_context") is ctx


# ===========================================================================
# TestRunBacktestWithHtfBars
# ===========================================================================


class TestRunBacktestWithHtfBars:
    """
    Integration tests verifying that run_backtest() and BacktestRunner.run()
    correctly propagate htf_bars through the pipeline.
    """

    def _make_minimal_bars(
        self,
        symbol: str = "BTC/USD",
        count: int = 60,
        timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    ) -> list[OHLCVBar]:
        """
        Generate a minimal chronological bar series for backtest tests.

        Default count=60 exceeds the default warmup_bars=50, ensuring
        that at least one bar reaches the strategy.
        """
        base = datetime(2024, 1, 1, tzinfo=UTC)
        bars = []
        for i in range(count):
            ts = base + timedelta(hours=i)
            bars.append(_make_bar(symbol=symbol, timestamp=ts, timeframe=timeframe))
        return bars

    async def test_run_backtest_without_htf_bars_passes_none_to_strategy(
        self,
    ) -> None:
        """
        When run_backtest() is called without htf_bars, the engine's
        _htf_bars is None and _build_mtf_context returns None. As a result,
        strategy.on_bar() must receive mtf_context=None on every bar call.
        """
        engine, mocks = _make_engine()
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        await engine.start("run-001")

        bars = self._make_minimal_bars()
        # No htf_bars argument -- default None
        # Ensure no FGI global client leaks from other tests (Sprint 32)
        import data.sentiment as _sent_mod
        _prev = _sent_mod._global_client
        _sent_mod._global_client = None
        try:
            await engine.run_backtest({"BTC/USD": bars})
        finally:
            _sent_mod._global_client = _prev

        assert mocks["strategy"].on_bar.call_count >= 1
        for single_call in mocks["strategy"].on_bar.call_args_list:
            _, kwargs = single_call
            assert kwargs.get("mtf_context") is None

    async def test_run_backtest_with_htf_bars_stores_and_uses_them(
        self,
    ) -> None:
        """
        When run_backtest() is called with htf_bars, the engine must store
        the data in _htf_bars and pass a non-None MultiTimeframeContext to
        strategy.on_bar() for bars after the warmup period.

        We supply a single 4h bar that is completed well before the first
        post-warmup primary bar, so it must appear in the context.
        """
        engine, mocks = _make_engine()
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        await engine.start("run-001")

        primary_bars = self._make_minimal_bars(count=60)

        # A 4h bar at 1970-01-01 -- trivially complete before any 2024 bar
        htf_bar = _make_bar(
            symbol="BTC/USD",
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(1970, 1, 1, tzinfo=UTC),
        )
        htf_bars: dict[str, dict[str, list[OHLCVBar]]] = {
            "4h": {"BTC/USD": [htf_bar]}
        }

        await engine.run_backtest({"BTC/USD": primary_bars}, htf_bars=htf_bars)

        # _htf_bars must be stored on the engine after run_backtest
        assert engine._htf_bars is not None  # type: ignore[attr-defined]
        assert "4h" in engine._htf_bars  # type: ignore[attr-defined]

        # At least one on_bar call must have received a non-None context
        assert mocks["strategy"].on_bar.call_count >= 1
        non_none_contexts = [
            call.kwargs.get("mtf_context")
            for call in mocks["strategy"].on_bar.call_args_list
            if call.kwargs.get("mtf_context") is not None
        ]
        assert len(non_none_contexts) >= 1
        # And the context must contain the htf bar
        first_ctx = non_none_contexts[0]
        assert isinstance(first_ctx, MultiTimeframeContext)
        assert "4h" in first_ctx.htf_bars

    async def test_backtest_runner_passes_htf_bars_to_engine(self) -> None:
        """
        BacktestRunner.run() must pass the htf_bars argument through to
        StrategyEngine.run_backtest() so that multi-timeframe strategies
        receive their higher-timeframe context.

        Verified by patching StrategyEngine.run_backtest and asserting it
        receives the htf_bars kwarg.
        """
        from unittest.mock import patch, AsyncMock as _AsyncMock
        from trading.backtest import BacktestRunner
        from trading.strategy import BaseStrategy

        # Minimal concrete strategy
        strategy = _PassthroughStrategy("bt_test")

        runner = BacktestRunner(
            strategies=[strategy],
            symbols=["BTC/USD"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
        )

        base = datetime(2024, 1, 1, tzinfo=UTC)
        primary_bars = [
            _make_bar(
                symbol="BTC/USD",
                timestamp=base + timedelta(hours=i),
            )
            for i in range(60)
        ]
        htf_bar = _make_bar(
            symbol="BTC/USD",
            timeframe=TimeFrame.FOUR_HOURS,
            timestamp=datetime(1970, 1, 1, tzinfo=UTC),
        )
        htf_bars: dict[str, dict[str, list[OHLCVBar]]] = {
            "4h": {"BTC/USD": [htf_bar]}
        }

        # Capture the htf_bars kwarg passed to run_backtest
        captured: dict[str, Any] = {}
        original_run_backtest = StrategyEngine.run_backtest

        async def _capturing_run_backtest(
            self_engine: StrategyEngine,
            bars_by_symbol: dict[str, list[OHLCVBar]],
            htf_bars: dict[str, dict[str, list[OHLCVBar]]] | None = None,
        ) -> dict[str, Any]:
            captured["htf_bars"] = htf_bars
            return await original_run_backtest(
                self_engine, bars_by_symbol, htf_bars=htf_bars
            )

        with patch.object(
            StrategyEngine, "run_backtest", _capturing_run_backtest
        ):
            await runner.run(
                {"BTC/USD": primary_bars},
                htf_bars=htf_bars,
            )

        assert "htf_bars" in captured
        assert captured["htf_bars"] is htf_bars


# ===========================================================================
# TestStrategyHtfTimeframes
# ===========================================================================


class TestStrategyHtfTimeframes:
    """
    Tests for BaseStrategy.htf_timeframes property.

    The base class returns an empty list by default. Concrete subclasses
    can override it to declare which higher timeframes they require.
    """

    def test_default_htf_timeframes_returns_empty_list(self) -> None:
        """
        BaseStrategy.htf_timeframes must return an empty list for strategies
        that do not override it, indicating they need no higher-timeframe data.
        """
        strategy = _PassthroughStrategy("default_tf")

        result = strategy.htf_timeframes

        assert result == []
        assert isinstance(result, list)

    def test_subclass_can_override_htf_timeframes(self) -> None:
        """
        A concrete subclass that overrides htf_timeframes must return the
        declared higher timeframe strings.

        _MtfAwareStrategy declares ["4h"]; the property must return that list.
        """
        strategy = _MtfAwareStrategy("mtf_aware")

        result = strategy.htf_timeframes

        assert result == ["4h"]
        assert "4h" in result
