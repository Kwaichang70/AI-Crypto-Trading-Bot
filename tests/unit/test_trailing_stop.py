"""
tests/unit/test_trailing_stop.py
---------------------------------
Unit tests for TrailingStopManager and its integration with StrategyEngine.

Modules under test
------------------
packages/trading/trailing_stop.py
packages/trading/strategy_engine.py  (integration tests only)

Test coverage
-------------
TestTrailingStopManagerInit (3 tests)
- Valid trailing_stop_pct values are accepted
- Invalid trailing_stop_pct values raise ValueError
- Custom strategy_id is stored correctly

TestTrailingStopCheck (15 tests)
- No position (None) returns None and clears tracking
- Flat position (qty=0) returns None and clears tracking
- Rising price updates peak without triggering
- Stable price does not trigger
- Price drop within threshold does not trigger
- Price drop exactly at threshold triggers SELL
- Price drop below threshold triggers SELL
- Emitted Signal has correct direction, target_position, and confidence
- Emitted Signal metadata contains trigger/peak_price/stop_price/current_price
- Peak tracks correctly through multi-bar price sequence
- Peak resets after position is closed (tracking cleared, fresh start)
- Multiple symbols tracked independently
- Trigger clears peak tracking for that symbol
- Very small pct (0.5%) with realistic BTC prices
- Very large pct (50%) only triggers on massive drops

TestTrailingStopReset (2 tests)
- reset() clears all tracked symbols
- After reset, new peaks start fresh

TestTrailingStopInStrategyEngine (10 tests)
- Engine with trailing_stop_pct=None creates no TrailingStopManager
- Engine with trailing_stop_pct=0.03 creates a TrailingStopManager
- Trailing stop SELL signal is submitted to execution engine
- Trailing stop fill is routed to portfolio.update_position
- Trailing stop fill triggers _record_trade_if_closed
- Trailing stop fill is routed to risk manager
- Trailing stop error is logged but does not crash bar loop
- Trailing stop is checked for all symbols each bar
- No trailing stop signal when position is flat
- Trailing stop works alongside normal strategy signals in same bar

Async note
----------
pyproject.toml sets asyncio_mode = "auto"; no @pytest.mark.asyncio needed.

Design notes
------------
Position is a Pydantic BaseModel; construct it directly with field kwargs.
The _make_engine() factory mirrors the pattern in test_strategy_engine_bar_loop.py:
returns (engine, mocks) where mocks is a dict of injected dependencies.
Tests that do not exercise check_resting_orders set it to None to suppress
AsyncMock auto-creation warnings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from common.models import OHLCVBar
from common.types import OrderSide, OrderType, RunMode, SignalDirection, TimeFrame
from trading.models import Fill, Order, Position, Signal
from trading.strategy_engine import StrategyEngine
from trading.trailing_stop import TrailingStopManager


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_position(
    *,
    symbol: str = "BTC/USD",
    run_id: str = "test-run",
    quantity: str = "0.01",
    average_entry_price: str = "50000",
    current_price: str = "50000",
) -> Position:
    """Construct a real Position Pydantic model with sensible defaults."""
    return Position(
        symbol=symbol,
        run_id=run_id,
        quantity=Decimal(quantity),
        average_entry_price=Decimal(average_entry_price),
        current_price=Decimal(current_price),
    )


def _flat_position(symbol: str = "BTC/USD") -> Position:
    """Construct a flat (qty=0) position."""
    return _make_position(symbol=symbol, quantity="0")


def _make_bar(
    *,
    symbol: str = "BTC/USD",
    close: str | Decimal = "50000",
    timestamp: datetime | None = None,
) -> OHLCVBar:
    """Construct a minimal OHLCVBar satisfying all OHLCV constraints."""
    ts = timestamp or datetime(2024, 1, 1, tzinfo=UTC)
    close_d = Decimal(str(close))
    high_d = (close_d * Decimal("1.01")).quantize(Decimal("0.01"))
    low_d = (close_d * Decimal("0.99")).quantize(Decimal("0.01"))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=ts,
        open=close_d,
        high=high_d,
        low=low_d,
        close=close_d,
        volume=Decimal("100"),
    )


def _make_fill(
    *,
    symbol: str = "BTC/USD",
    side: OrderSide = OrderSide.SELL,
    quantity: str = "0.01",
    price: str = "50000",
) -> Fill:
    """Construct a Fill for fill-routing assertions."""
    qty = Decimal(quantity)
    prc = Decimal(price)
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=qty,
        price=prc,
        fee=qty * prc * Decimal("0.001"),
        fee_currency="USD",
    )


def _make_engine(
    *,
    symbols: list[str] | None = None,
    run_mode: RunMode = RunMode.BACKTEST,
    trailing_stop_pct: float | None = None,
) -> tuple[StrategyEngine, dict[str, Any]]:
    """
    Create a StrategyEngine with fully mocked dependencies.

    Mirrors the pattern from test_strategy_engine_bar_loop.py.
    Pass trailing_stop_pct to inject it via the engine config.

    Returns the engine (not yet started) and a mocks dict.
    Callers must call ``await engine.start(run_id)`` before bar-loop methods.
    """
    strategy = MagicMock()
    strategy.strategy_id = "test_strategy"
    strategy.min_bars_required = 20
    strategy.on_start = MagicMock(return_value=None)
    strategy.on_stop = MagicMock(return_value=None)

    execution = AsyncMock()
    execution.on_start = AsyncMock(return_value=None)
    execution.on_stop = AsyncMock(return_value=None)
    execution.get_open_orders = MagicMock(return_value=[])
    execution.cancel_order = AsyncMock(return_value=None)

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

    config: dict[str, Any] = {}
    if trailing_stop_pct is not None:
        config["trailing_stop_pct"] = trailing_stop_pct

    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=symbols or ["BTC/USD"],
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
# TestTrailingStopManagerInit
# ===========================================================================


class TestTrailingStopManagerInit:
    """
    Tests for TrailingStopManager constructor validation and attribute storage.
    """

    def test_valid_pct_values_accepted(self) -> None:
        """
        Constructor must accept trailing_stop_pct values in (0, 0.50].

        Exercises the lower-open boundary (0.005), a typical value (0.03),
        and the upper-closed boundary (0.50).
        """
        for pct in (0.005, 0.03, 0.50):
            mgr = TrailingStopManager(trailing_stop_pct=pct)
            assert mgr is not None, f"Expected valid construction for pct={pct}"

    @pytest.mark.parametrize("bad_pct", [0, 0.004, -0.01, 0.51, 1.0])
    def test_invalid_pct_raises_value_error(self, bad_pct: float) -> None:
        """
        Constructor must raise ValueError for trailing_stop_pct values
        outside [0.005, 0.50]: below 0.005, negative, or greater than 0.50.
        """
        with pytest.raises(ValueError):
            TrailingStopManager(trailing_stop_pct=bad_pct)

    def test_custom_strategy_id_stored(self) -> None:
        """
        When a custom strategy_id is supplied, it must be stored on the
        instance and used in emitted signals.
        """
        mgr = TrailingStopManager(
            trailing_stop_pct=0.03,
            strategy_id="my_trailing_stop",
        )
        assert mgr.strategy_id == "my_trailing_stop"


# ===========================================================================
# TestTrailingStopCheck
# ===========================================================================


class TestTrailingStopCheck:
    """
    Tests for TrailingStopManager.check() — the core signal-emission logic.

    Each test constructs a fresh manager to guarantee isolation.
    """

    def test_no_position_returns_none_and_clears_tracking(self) -> None:
        """
        check() with position=None must return None and must not retain any
        tracking state for the symbol.

        Simulates: symbol was previously tracked, then position closes.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        # Pre-seed tracking by processing a bar with an open position
        pos = _make_position()
        mgr.check("BTC/USD", Decimal("50000"), pos)
        assert "BTC/USD" in mgr.peak_prices

        # Now close the position — check with None
        result = mgr.check("BTC/USD", Decimal("49000"), None)

        assert result is None
        assert "BTC/USD" not in mgr.peak_prices

    def test_flat_position_returns_none_and_clears_tracking(self) -> None:
        """
        check() with a flat position (qty=0) must return None and clear
        tracking for that symbol.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        # Pre-seed tracking
        pos = _make_position()
        mgr.check("BTC/USD", Decimal("50000"), pos)

        flat = _flat_position()
        result = mgr.check("BTC/USD", Decimal("49000"), flat)

        assert result is None
        assert "BTC/USD" not in mgr.peak_prices

    def test_rising_price_updates_peak_returns_none(self) -> None:
        """
        When price rises above the current peak, the peak must be updated
        and check() must return None (no trigger).
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        result = mgr.check("BTC/USD", Decimal("52000"), pos)

        assert result is None
        assert mgr.peak_prices["BTC/USD"] == Decimal("52000")

    def test_stable_price_does_not_trigger(self) -> None:
        """
        When price is equal to the peak, check() must return None.

        Equal price cannot be below the trailing-stop threshold.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        result = mgr.check("BTC/USD", Decimal("50000"), pos)

        assert result is None

    def test_price_drop_within_threshold_does_not_trigger(self) -> None:
        """
        A price drop smaller than trailing_stop_pct below the peak must
        NOT emit a signal.

        Peak=50000, pct=0.03 → stop=48500. Price=49000 is above stop.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        result = mgr.check("BTC/USD", Decimal("49000"), pos)

        assert result is None

    def test_price_drop_exactly_at_threshold_triggers_sell(self) -> None:
        """
        When current_price equals exactly peak * (1 - pct), a SELL signal
        must be emitted.

        Peak=50000, pct=0.03 → stop=48500. Price=48500 is at the boundary.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        result = mgr.check("BTC/USD", Decimal("48500"), pos)

        assert result is not None
        assert result.direction == SignalDirection.SELL

    def test_price_drop_below_threshold_triggers_sell(self) -> None:
        """
        When current_price falls below peak * (1 - pct), a SELL signal
        must be emitted.

        Peak=50000, pct=0.03 → stop=48500. Price=45000 is below stop.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        result = mgr.check("BTC/USD", Decimal("45000"), pos)

        assert result is not None
        assert result.direction == SignalDirection.SELL

    def test_sell_signal_has_correct_fields(self) -> None:
        """
        The emitted SELL signal must have:
        - direction = SELL
        - target_position = 0 (close entirely)
        - confidence = 1.0
        - symbol matching the tracked symbol
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        signal = mgr.check("BTC/USD", Decimal("45000"), pos)

        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.target_position == Decimal("0")
        assert signal.confidence == 1.0
        assert signal.symbol == "BTC/USD"

    def test_sell_signal_metadata_contains_key_fields(self) -> None:
        """
        The emitted SELL signal's metadata dict must contain:
        - "trigger" key
        - "peak_price" matching the recorded peak
        - "stop_price" = peak * (1 - pct)
        - "current_price" matching the current price passed in
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()
        peak = Decimal("50000")
        current = Decimal("45000")
        expected_stop = peak * (Decimal("1") - Decimal("0.03"))

        mgr.check("BTC/USD", peak, pos)
        signal = mgr.check("BTC/USD", current, pos)

        assert signal is not None
        meta = signal.metadata
        assert "trigger" in meta
        assert Decimal(str(meta["peak_price"])) == peak
        assert Decimal(str(meta["stop_price"])) == expected_stop
        assert Decimal(str(meta["current_price"])) == current

    def test_peak_tracks_through_multi_bar_sequence(self) -> None:
        """
        The peak must update to the highest close seen so far.

        Price sequence: 100 → 110 → 108 → 112 → 110
        Peak at each step:   100  110   110   112   112

        108 is within 3% of 110 (stop=106.7) so no trigger.
        110 is within 3% of 112 (stop=108.64) so no trigger.
        Final peak should be 112.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        prices = [
            Decimal("100"),
            Decimal("110"),
            Decimal("108"),  # within threshold (110 * 0.97 = 106.7)
            Decimal("112"),
            Decimal("110"),  # within threshold (112 * 0.97 = 108.64)
        ]
        for p in prices:
            result = mgr.check("BTC/USD", p, pos)
            assert result is None, f"Unexpected trigger at price {p}"

        assert mgr.peak_prices["BTC/USD"] == Decimal("112")

        # Now drop far enough to trigger: 112 * 0.97 = 108.64, so 108 triggers
        signal = mgr.check("BTC/USD", Decimal("108"), pos)
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_peak_resets_after_position_closed_then_new_position(self) -> None:
        """
        After a position closes (clearing tracking), a new position must
        start fresh peak tracking from the first bar price.

        Sequence:
        1. Open position, track price=50000 (peak=50000)
        2. Close position (None) → tracking cleared
        3. Open new position, price=40000 (peak resets to 40000)
        4. Price drops to 38000 — should NOT trigger (38000 > 40000 * 0.97 = 38800)
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        # First position lifecycle
        mgr.check("BTC/USD", Decimal("50000"), pos)
        assert mgr.peak_prices["BTC/USD"] == Decimal("50000")

        # Close position
        mgr.check("BTC/USD", Decimal("49000"), None)
        assert "BTC/USD" not in mgr.peak_prices

        # New position at lower price level
        mgr.check("BTC/USD", Decimal("40000"), pos)
        assert mgr.peak_prices["BTC/USD"] == Decimal("40000")

        # Price drops but is still within 3% of new peak (40000 * 0.97 = 38800)
        result = mgr.check("BTC/USD", Decimal("38900"), pos)
        assert result is None

    def test_multiple_symbols_tracked_independently(self) -> None:
        """
        Two symbols must be tracked with independent peak values.

        BTC/USD peak at 50000, ETH/USD peak at 3000.
        Dropping BTC/USD below threshold must not affect ETH/USD tracking.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        btc_pos = _make_position(symbol="BTC/USD")
        eth_pos = _make_position(symbol="ETH/USD")

        mgr.check("BTC/USD", Decimal("50000"), btc_pos)
        mgr.check("ETH/USD", Decimal("3000"), eth_pos)

        # Drop BTC/USD below threshold to trigger
        btc_signal = mgr.check("BTC/USD", Decimal("45000"), btc_pos)

        # ETH/USD tracking must remain intact
        assert btc_signal is not None
        assert "ETH/USD" in mgr.peak_prices
        assert mgr.peak_prices["ETH/USD"] == Decimal("3000")

    def test_trigger_adds_symbol_to_pending_stops(self) -> None:
        """
        After a trailing stop triggers, the symbol must be added to
        _pending_stop_symbols to prevent duplicate signals before the
        position is actually closed (fill latency safety — CR-002).
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        signal = mgr.check("BTC/USD", Decimal("45000"), pos)

        assert signal is not None
        assert "BTC/USD" in mgr.pending_stop_symbols

    def test_pending_stop_suppresses_duplicate_signal(self) -> None:
        """
        After a trailing stop triggers and before the position closes,
        subsequent check() calls for that symbol must return None to
        prevent duplicate SELL signals (CR-002).
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        first_signal = mgr.check("BTC/USD", Decimal("45000"), pos)
        assert first_signal is not None

        # Position still open (fill hasn't closed it yet) — next check must return None
        second_signal = mgr.check("BTC/USD", Decimal("44000"), pos)
        assert second_signal is None

    def test_pending_stop_cleared_when_position_closes(self) -> None:
        """
        When the position becomes flat (None or qty=0), the pending stop
        must be cleared so re-entry can be tracked fresh (CR-002).
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        mgr.check("BTC/USD", Decimal("45000"), pos)
        assert "BTC/USD" in mgr.pending_stop_symbols

        # Position closed
        mgr.check("BTC/USD", Decimal("44000"), None)
        assert "BTC/USD" not in mgr.pending_stop_symbols

    def test_trigger_clears_peak_tracking_for_that_symbol(self) -> None:
        """
        After a trigger fires, the peak for that symbol must be cleared.

        This prevents the manager from re-triggering on every subsequent bar
        until the position is actually flat and the engine re-registers it.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        signal = mgr.check("BTC/USD", Decimal("45000"), pos)

        assert signal is not None
        assert "BTC/USD" not in mgr.peak_prices

    def test_very_small_pct_with_realistic_btc_prices(self) -> None:
        """
        With trailing_stop_pct=0.005 (0.5%) and BTC-scale prices,
        the stop price arithmetic must be precise.

        Peak=60000, stop=60000 * 0.995 = 59700.
        Price=59701 must NOT trigger; price=59699 MUST trigger.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.005)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("60000"), pos)

        no_trigger = mgr.check("BTC/USD", Decimal("59701"), pos)
        assert no_trigger is None, "59701 is above stop price 59700"

        # Reset peak for clean second assertion
        mgr.reset()
        mgr.check("BTC/USD", Decimal("60000"), pos)
        trigger = mgr.check("BTC/USD", Decimal("59699"), pos)
        assert trigger is not None, "59699 is below stop price 59700"
        assert trigger.direction == SignalDirection.SELL

    def test_very_large_pct_only_triggers_on_massive_drops(self) -> None:
        """
        With trailing_stop_pct=0.50 (50%), a 30% drop must NOT trigger
        but a 51% drop MUST trigger.

        Peak=10000, stop=5000.
        Price=7000 (30% drop) → no trigger.
        Price=4900 (51% drop) → SELL signal.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.50)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("10000"), pos)
        no_trigger = mgr.check("BTC/USD", Decimal("7000"), pos)
        assert no_trigger is None

        mgr.reset()
        mgr.check("BTC/USD", Decimal("10000"), pos)
        trigger = mgr.check("BTC/USD", Decimal("4900"), pos)
        assert trigger is not None
        assert trigger.direction == SignalDirection.SELL


# ===========================================================================
# TestTrailingStopReset
# ===========================================================================


class TestTrailingStopReset:
    """
    Tests for TrailingStopManager.reset() — complete state wipe.
    """

    def test_reset_clears_all_tracked_symbols(self) -> None:
        """
        reset() must clear the peak_prices dict for all symbols that have
        been tracked, leaving it empty.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        btc_pos = _make_position(symbol="BTC/USD")
        eth_pos = _make_position(symbol="ETH/USD")

        mgr.check("BTC/USD", Decimal("50000"), btc_pos)
        mgr.check("ETH/USD", Decimal("3000"), eth_pos)
        assert len(mgr.peak_prices) == 2

        mgr.reset()

        assert mgr.peak_prices == {}

    def test_reset_clears_pending_stop_symbols(self) -> None:
        """
        reset() must also clear the pending_stop_symbols set (CR-002).
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        mgr.check("BTC/USD", Decimal("45000"), pos)
        assert len(mgr.pending_stop_symbols) == 1

        mgr.reset()

        assert mgr.pending_stop_symbols == set()

    def test_after_reset_new_peaks_start_fresh(self) -> None:
        """
        After reset(), the first check() for a symbol must initialise its
        peak from the price passed in, not the pre-reset value.

        Peak before reset: 50000. After reset + new bar at 30000, peak
        must be 30000, not 50000.
        """
        mgr = TrailingStopManager(trailing_stop_pct=0.03)
        pos = _make_position()

        mgr.check("BTC/USD", Decimal("50000"), pos)
        mgr.reset()

        mgr.check("BTC/USD", Decimal("30000"), pos)
        assert mgr.peak_prices["BTC/USD"] == Decimal("30000")


# ===========================================================================
# TestTrailingStopInStrategyEngine
# ===========================================================================


class TestTrailingStopInStrategyEngine:
    """
    Integration tests for the trailing stop hook inside StrategyEngine._process_bar().

    These tests verify that:
    1. A TrailingStopManager is (or is not) created based on config.
    2. The trailing stop SELL signal is processed through the full signal pipeline.
    3. Fills from trailing stop orders are routed to portfolio, trade recording,
       and risk manager correctly.
    4. Errors in trailing stop processing are caught and don't crash the bar loop.
    """

    async def test_engine_without_trailing_stop_pct_has_no_manager(self) -> None:
        """
        When trailing_stop_pct is absent from config, the engine must not
        create a TrailingStopManager instance.
        """
        engine, _ = _make_engine(trailing_stop_pct=None)
        await engine.start("run-001")

        assert not hasattr(engine, "_trailing_stop") or \
               engine._trailing_stop is None  # type: ignore[attr-defined]

    async def test_engine_with_invalid_trailing_stop_pct_disables_gracefully(self) -> None:
        """
        When trailing_stop_pct has an invalid value (e.g. 0 or negative),
        the engine must log a warning and disable trailing stop instead of
        crashing during construction (CR-001).
        """
        engine, _ = _make_engine(trailing_stop_pct=0.0)
        await engine.start("run-001")

        assert engine._trailing_stop is None  # type: ignore[attr-defined]

    async def test_engine_with_trailing_stop_pct_creates_manager(self) -> None:
        """
        When trailing_stop_pct=0.03 is in config, the engine must instantiate
        a TrailingStopManager and store it.
        """
        engine, _ = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")

        assert engine._trailing_stop is not None  # type: ignore[attr-defined]
        assert isinstance(engine._trailing_stop, TrailingStopManager)  # type: ignore[attr-defined]

    async def test_trailing_stop_sell_signal_submitted_to_execution(self) -> None:
        """
        When the TrailingStopManager emits a SELL signal for a symbol,
        the signal must be submitted to execution_engine.process_signal.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        # Place an open position so trailing stop has something to track
        open_pos = _make_position(symbol="BTC/USD")
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        # Simulate a high-water mark bar followed by a large drop
        ts_high = datetime(2024, 1, 1, 0, tzinfo=UTC)
        ts_low = datetime(2024, 1, 1, 1, tzinfo=UTC)

        bar_high = _make_bar(symbol="BTC/USD", close="50000", timestamp=ts_high)
        await engine._process_bar({"BTC/USD": bar_high}, {"BTC/USD": [bar_high]})

        # Reset process_signal call count before the trigger bar
        mocks["execution"].process_signal.reset_mock()

        # Price drops 10% below peak — well past 3% threshold
        bar_drop = _make_bar(symbol="BTC/USD", close="45000", timestamp=ts_low)
        await engine._process_bar({"BTC/USD": bar_drop}, {"BTC/USD": [bar_high, bar_drop]})

        # process_signal must have been called at least once with a SELL signal
        assert mocks["execution"].process_signal.await_count >= 1
        all_signals = [
            call.args[0]
            for call in mocks["execution"].process_signal.await_args_list
        ]
        sell_signals = [s for s in all_signals if s.direction == SignalDirection.SELL]
        assert len(sell_signals) >= 1, "Expected at least one SELL signal from trailing stop"

    async def test_trailing_stop_fill_routed_to_portfolio(self) -> None:
        """
        A fill generated from a trailing stop SELL order must be routed to
        portfolio.update_position with the bar's close price.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        open_pos = _make_position(symbol="BTC/USD")
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        sell_fill = _make_fill(symbol="BTC/USD", side=OrderSide.SELL, price="45000")
        sell_order = Order(
            client_order_id=f"run-001-{uuid4().hex[:12]}",
            run_id="run-001",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        # Build up peak then trigger
        bar_high = _make_bar(symbol="BTC/USD", close="50000")
        await engine._process_bar({"BTC/USD": bar_high}, {"BTC/USD": [bar_high]})
        mocks["portfolio"].update_position.reset_mock()

        bar_drop = _make_bar(symbol="BTC/USD", close="45000")
        await engine._process_bar({"BTC/USD": bar_drop}, {"BTC/USD": [bar_high, bar_drop]})

        # update_position must have been called with the fill and the close price
        mocks["portfolio"].update_position.assert_called()
        call_args_list = mocks["portfolio"].update_position.call_args_list
        fill_args = [c.args[0] for c in call_args_list]
        assert sell_fill in fill_args

    async def test_trailing_stop_fill_triggers_record_trade(self) -> None:
        """
        A trailing stop fill that closes an open position must invoke
        _record_trade_if_closed so the trade is persisted.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        open_pos = _make_position(
            symbol="BTC/USD",
            quantity="0.01",
            average_entry_price="50000",
        )
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        sell_fill = _make_fill(symbol="BTC/USD", side=OrderSide.SELL, price="45000")
        sell_order = Order(
            client_order_id=f"run-001-{uuid4().hex[:12]}",
            run_id="run-001",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        bar_high = _make_bar(symbol="BTC/USD", close="50000")
        await engine._process_bar({"BTC/USD": bar_high}, {"BTC/USD": [bar_high]})
        mocks["portfolio"].record_trade.reset_mock()

        bar_drop = _make_bar(symbol="BTC/USD", close="45000")
        await engine._process_bar({"BTC/USD": bar_drop}, {"BTC/USD": [bar_high, bar_drop]})

        mocks["portfolio"].record_trade.assert_called()

    async def test_trailing_stop_fill_routed_to_risk_manager(self) -> None:
        """
        A trailing stop fill must pass through _route_fill_to_risk_manager.

        When the fill closes the position (flat after fill), the risk manager's
        update_after_fill must be called.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        # Position is flat AFTER the sell fill (simulating full close)
        flat_pos = _flat_position()
        flat_pos_with_pnl = Position(
            symbol="BTC/USD",
            run_id="run-001",
            quantity=Decimal("0"),
            average_entry_price=Decimal("50000"),
            current_price=Decimal("45000"),
            realised_pnl=Decimal("-50"),
        )

        open_pos = _make_position(symbol="BTC/USD")
        # get_position call sequence:
        # bar 1: trailing stop check (no trigger, no fills) = 1 call
        # bar 2: trailing stop check (triggers), pre_fill_pos, _route_fill (flat)
        mocks["portfolio"].get_position = MagicMock(
            side_effect=[
                open_pos,                                  # bar 1: trailing check
                open_pos, open_pos, flat_pos_with_pnl,     # bar 2: trailing check, pre-fill, risk route
            ]
        )

        sell_fill = _make_fill(symbol="BTC/USD", side=OrderSide.SELL, price="45000")
        sell_order = Order(
            client_order_id=f"run-001-{uuid4().hex[:12]}",
            run_id="run-001",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        bar_high = _make_bar(symbol="BTC/USD", close="50000")
        await engine._process_bar({"BTC/USD": bar_high}, {"BTC/USD": [bar_high]})
        mocks["risk_manager"].update_after_fill.reset_mock()

        bar_drop = _make_bar(symbol="BTC/USD", close="45000")
        await engine._process_bar({"BTC/USD": bar_drop}, {"BTC/USD": [bar_high, bar_drop]})

        mocks["risk_manager"].update_after_fill.assert_called()

    async def test_trailing_stop_error_logged_does_not_crash_bar_loop(self) -> None:
        """
        If the TrailingStopManager.check() raises an unexpected exception,
        the bar loop must catch it, log it, and continue.

        bar_count must still increment — the bar is fully processed despite
        the trailing stop failure.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        # Inject failure into the trailing stop manager
        engine._trailing_stop.check = MagicMock(  # type: ignore[union-attr]
            side_effect=RuntimeError("trailing stop exploded")
        )
        mocks["portfolio"].get_position = MagicMock(return_value=_make_position())

        bar = _make_bar(symbol="BTC/USD", close="50000")
        # Must not raise
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

        assert engine.bar_count == 1

    async def test_trailing_stop_checked_for_all_symbols_each_bar(self) -> None:
        """
        With two symbols, the TrailingStopManager.check() must be called
        once per symbol per bar.
        """
        engine, mocks = _make_engine(
            symbols=["BTC/USD", "ETH/USD"],
            trailing_stop_pct=0.03,
        )
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["portfolio"].get_position = MagicMock(return_value=_make_position())

        engine._trailing_stop.check = MagicMock(return_value=None)  # type: ignore[union-attr]

        bar_btc = _make_bar(symbol="BTC/USD", close="50000")
        bar_eth = _make_bar(symbol="ETH/USD", close="3000")
        await engine._process_bar(
            {"BTC/USD": bar_btc, "ETH/USD": bar_eth},
            {"BTC/USD": [bar_btc], "ETH/USD": [bar_eth]},
        )

        check_calls = engine._trailing_stop.check.call_args_list  # type: ignore[union-attr]
        called_symbols = {c.kwargs.get("symbol", c.args[0] if c.args else None) for c in check_calls}
        assert called_symbols == {"BTC/USD", "ETH/USD"}

    async def test_no_trailing_stop_signal_when_position_is_flat(self) -> None:
        """
        When the portfolio returns a flat position, the trailing stop must
        not emit a signal and must not call process_signal with a SELL.
        """
        engine, mocks = _make_engine(trailing_stop_pct=0.03)
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        # Always flat position
        mocks["portfolio"].get_position = MagicMock(return_value=_flat_position())

        bar = _make_bar(symbol="BTC/USD", close="50000")
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

        # process_signal must not have been called with any SELL signal
        if mocks["execution"].process_signal.await_count > 0:
            all_signals = [
                c.args[0]
                for c in mocks["execution"].process_signal.await_args_list
            ]
            sell_signals = [s for s in all_signals if s.direction == SignalDirection.SELL]
            assert len(sell_signals) == 0, (
                "process_signal must not be called with SELL when position is flat"
            )

    async def test_trailing_stop_works_alongside_normal_strategy_signals(
        self,
    ) -> None:
        """
        In the same bar, both a normal strategy BUY signal and a trailing stop
        SELL signal (for a different scenario) must be processed.

        This test verifies that trailing stop checks do not interfere with
        normal strategy signal processing — both paths execute independently.

        We simulate: strategy emits BUY for ETH, trailing stop fires SELL for BTC.
        """
        engine, mocks = _make_engine(
            symbols=["BTC/USD", "ETH/USD"],
            trailing_stop_pct=0.03,
        )
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        # Strategy emits a BUY for ETH (unrelated to the trailing stop)
        eth_buy_signal = Signal(
            strategy_id="test_strategy",
            symbol="ETH/USD",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        mocks["strategy"].on_bar = MagicMock(return_value=[eth_buy_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        # BTC has an open position that will trigger the trailing stop
        btc_pos = _make_position(symbol="BTC/USD")
        eth_pos = _make_position(symbol="ETH/USD")

        def _get_pos(symbol: str) -> Position | None:
            return {"BTC/USD": btc_pos, "ETH/USD": eth_pos}.get(symbol)

        mocks["portfolio"].get_position = MagicMock(side_effect=_get_pos)

        # Seed the BTC peak
        bar_btc_high = _make_bar(symbol="BTC/USD", close="50000")
        bar_eth_high = _make_bar(symbol="ETH/USD", close="3000")
        await engine._process_bar(
            {"BTC/USD": bar_btc_high, "ETH/USD": bar_eth_high},
            {"BTC/USD": [bar_btc_high], "ETH/USD": [bar_eth_high]},
        )
        mocks["execution"].process_signal.reset_mock()

        # Now drop BTC below threshold — trailing stop fires SELL for BTC
        # Strategy still fires BUY for ETH (on_bar returns eth_buy_signal again)
        bar_btc_drop = _make_bar(symbol="BTC/USD", close="45000")
        bar_eth_2 = _make_bar(symbol="ETH/USD", close="3100")
        await engine._process_bar(
            {"BTC/USD": bar_btc_drop, "ETH/USD": bar_eth_2},
            {
                "BTC/USD": [bar_btc_high, bar_btc_drop],
                "ETH/USD": [bar_eth_high, bar_eth_2],
            },
        )

        # process_signal must have been called at least once (for ETH BUY from strategy,
        # and possibly once more for BTC SELL from trailing stop)
        assert mocks["execution"].process_signal.await_count >= 1

        all_signals = [
            c.args[0]
            for c in mocks["execution"].process_signal.await_args_list
        ]
        buy_signals = [s for s in all_signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for s in all_signals if s.direction == SignalDirection.SELL]

        assert len(buy_signals) >= 1, "Expected ETH BUY from normal strategy"
        assert len(sell_signals) >= 1, "Expected BTC SELL from trailing stop"
