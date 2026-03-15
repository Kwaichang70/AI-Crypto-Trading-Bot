"""
tests/unit/test_trade_journal.py
---------------------------------
Unit tests for Sprint 32 trade journal components.

Modules under test
------------------
packages/trading/trade_journal.py

  TradeExcursionTracker  -- bar-by-bar MAE/MFE tracking
  ExitReasonDetector     -- stateless exit reason classifier
  TradeSkipLogger        -- records trades not taken

Coverage groups (18 tests)
---------------------------
TestExcursionTrackerBasic    (4) -- open/bar/close lifecycle, edge cases
TestExcursionTrackerLong     (5) -- long-side MFE/MAE arithmetic
TestExcursionTrackerEdgeCases(4) -- invalid entry prices, double-open, passthrough
TestExitReasonDetector       (4) -- trailing_stop, metadata, default, constant
TestTradeSkipLogger          (4) -- log_skip, summary, count, clear
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from trading.trade_journal import ExitReasonDetector, TradeExcursionTracker, TradeSkipLogger


# ===========================================================================
# Helpers
# ===========================================================================


def _tracker() -> TradeExcursionTracker:
    return TradeExcursionTracker()


def _make_bar(high: float, low: float, close: float | None = None) -> tuple[Decimal, Decimal, Decimal]:
    """Return (high, low, close) as Decimals."""
    return Decimal(str(high)), Decimal(str(low)), Decimal(str(close if close is not None else (high + low) / 2))


# ===========================================================================
# TestExcursionTrackerBasic
# ===========================================================================


class TestExcursionTrackerBasic:
    """Lifecycle tests: open → on_bar → close."""

    def test_open_bar_close_lifecycle(self) -> None:
        """Open, send 3 bars, close. Verify returned tuple structure."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))

        h, l, c = _make_bar(102, 99, 101)
        tracker.on_bar("BTC/USDT", h, l, c)
        h, l, c = _make_bar(105, 98, 103)
        tracker.on_bar("BTC/USDT", h, l, c)
        h, l, c = _make_bar(103, 97, 100)
        tracker.on_bar("BTC/USDT", h, l, c)

        result = tracker.on_position_close("BTC/USDT")

        assert result is not None
        mae_pct, mfe_pct, regime, context = result
        assert isinstance(mae_pct, float)
        assert isinstance(mfe_pct, float)
        # Best high was 105 on entry 100 => MFE = (105-100)/100 = 0.05
        assert pytest.approx(mfe_pct, rel=1e-4) == 0.05
        # Worst low was 97 on entry 100 => MAE = (100-97)/100 = 0.03
        assert pytest.approx(mae_pct, rel=1e-4) == 0.03
        # No regime or context provided
        assert regime is None
        assert context is None

    def test_close_untracked_returns_none(self) -> None:
        """on_position_close for a symbol with no state returns None."""
        tracker = _tracker()
        result = tracker.on_position_close("ETH/USDT")
        assert result is None

    def test_on_bar_untracked_is_noop(self) -> None:
        """on_bar for an untracked symbol must not raise."""
        tracker = _tracker()
        # No exception should be raised
        tracker.on_bar("ETH/USDT", Decimal("110"), Decimal("90"), Decimal("100"))

    def test_clear_removes_all(self) -> None:
        """After clear(), both previously tracked symbols return None on close."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("50000"))
        tracker.on_position_open("ETH/USDT", Decimal("3000"))

        tracker.clear()

        assert tracker.on_position_close("BTC/USDT") is None
        assert tracker.on_position_close("ETH/USDT") is None
        assert tracker.tracked_symbols == []


# ===========================================================================
# TestExcursionTrackerLong
# ===========================================================================


class TestExcursionTrackerLong:
    """Arithmetic tests for long-position MAE/MFE calculations."""

    def test_rising_mfe(self) -> None:
        """Entry=100, bar high=110 => MFE=0.10, MAE=0.0 (low==entry)."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        tracker.on_bar("BTC/USDT", Decimal("110"), Decimal("100"), Decimal("105"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        assert pytest.approx(mfe_pct, rel=1e-5) == 0.10
        assert mae_pct == 0.0

    def test_falling_mae(self) -> None:
        """Entry=100, bar low=90 => MAE=0.10, MFE=0.0 (high==entry)."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        tracker.on_bar("BTC/USDT", Decimal("100"), Decimal("90"), Decimal("95"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        assert pytest.approx(mae_pct, rel=1e-5) == 0.10
        assert mfe_pct == 0.0

    def test_mixed_excursion(self) -> None:
        """Entry=100, bar1 high=105/low=95, bar2 high=112/low=92. MFE=0.12, MAE=0.08."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        tracker.on_bar("BTC/USDT", Decimal("105"), Decimal("95"), Decimal("100"))
        tracker.on_bar("BTC/USDT", Decimal("112"), Decimal("92"), Decimal("100"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        # MFE = (112 - 100) / 100 = 0.12
        assert pytest.approx(mfe_pct, rel=1e-5) == 0.12
        # MAE = (100 - 92) / 100 = 0.08
        assert pytest.approx(mae_pct, rel=1e-5) == 0.08

    def test_immediate_reversal(self) -> None:
        """Entry=100, bar1 high=100 low=80 => MAE=0.20, MFE=0.0."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        tracker.on_bar("BTC/USDT", Decimal("100"), Decimal("80"), Decimal("85"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        assert pytest.approx(mae_pct, rel=1e-5) == 0.20
        # high == entry => favorable_pct = 0 => MFE stays at 0
        assert mfe_pct == 0.0

    def test_break_even(self) -> None:
        """Entry=100, bars high=100 low=100 => MAE=0.0, MFE=0.0."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        for _ in range(3):
            tracker.on_bar("BTC/USDT", Decimal("100"), Decimal("100"), Decimal("100"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        assert mae_pct == 0.0
        assert mfe_pct == 0.0


# ===========================================================================
# TestExcursionTrackerEdgeCases
# ===========================================================================


class TestExcursionTrackerEdgeCases:
    """Edge-case tests for invalid inputs and optional metadata passthrough."""

    def test_zero_entry_price_rejected(self) -> None:
        """on_position_open with entry_price=0 must not register tracking."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("0"))
        assert "BTC/USDT" not in tracker.tracked_symbols
        # close must return None since nothing was registered
        assert tracker.on_position_close("BTC/USDT") is None

    def test_double_open_overwrites(self) -> None:
        """A second on_position_open for the same symbol replaces the first."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("100"))
        tracker.on_position_open("BTC/USDT", Decimal("200"))  # overwrite

        # Bar high=220 => MFE based on entry=200, not 100
        tracker.on_bar("BTC/USDT", Decimal("220"), Decimal("195"), Decimal("210"))

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        mae_pct, mfe_pct, _, _ = result
        # MFE = (220 - 200) / 200 = 0.10
        assert pytest.approx(mfe_pct, rel=1e-5) == 0.10

    def test_regime_and_context_passthrough(self) -> None:
        """regime_at_entry and signal_context are returned unchanged on close."""
        tracker = _tracker()
        ctx = {"rsi": 28, "volume_ratio": 1.5}
        tracker.on_position_open(
            "BTC/USDT",
            Decimal("50000"),
            regime_at_entry="RISK_ON",
            signal_context=ctx,
        )

        result = tracker.on_position_close("BTC/USDT")
        assert result is not None
        _, _, regime, context = result
        assert regime == "RISK_ON"
        assert context == ctx

    def test_negative_entry_price_rejected(self) -> None:
        """on_position_open with a negative entry_price must not register tracking."""
        tracker = _tracker()
        tracker.on_position_open("BTC/USDT", Decimal("-50"))
        assert "BTC/USDT" not in tracker.tracked_symbols


# ===========================================================================
# TestExitReasonDetector
# ===========================================================================


class TestExitReasonDetector:
    """Tests for ExitReasonDetector.detect() static method."""

    def test_trailing_stop_by_strategy_id(self) -> None:
        """strategy_id containing 'trailing_stop' returns 'trailing_stop'."""
        result = ExitReasonDetector.detect("trailing_stop")
        assert result == "trailing_stop"

    def test_trailing_stop_by_strategy_id_substring(self) -> None:
        """strategy_id with 'trailing_stop' as substring returns 'trailing_stop'."""
        result = ExitReasonDetector.detect("my_trailing_stop_manager")
        assert result == "trailing_stop"

    def test_default_signal_exit(self) -> None:
        """strategy_id='rsi', no metadata => default 'signal_exit'."""
        result = ExitReasonDetector.detect("rsi", signal_metadata=None)
        assert result == "signal_exit"

    def test_valid_reasons_constant(self) -> None:
        """VALID_REASONS must contain all six expected exit reason strings."""
        expected = {
            "trailing_stop",
            "signal_exit",
            "stop_loss",
            "take_profit",
            "regime_change",
            "manual",
        }
        assert set(ExitReasonDetector.VALID_REASONS) == expected

    def test_explicit_exit_reason_in_metadata(self) -> None:
        """signal_metadata with exit_reason key takes precedence."""
        result = ExitReasonDetector.detect("rsi", signal_metadata={"exit_reason": "take_profit"})
        assert result == "take_profit"

    def test_stop_loss_inferred_from_metadata(self) -> None:
        """signal_metadata with stop_loss key returns 'stop_loss'."""
        result = ExitReasonDetector.detect("rsi", signal_metadata={"stop_loss": True})
        assert result == "stop_loss"

    def test_regime_change_inferred_from_metadata(self) -> None:
        """signal_metadata with regime_change key returns 'regime_change'."""
        result = ExitReasonDetector.detect("rsi", signal_metadata={"regime_change": True})
        assert result == "regime_change"


# ===========================================================================
# TestTradeSkipLogger
# ===========================================================================


class TestTradeSkipLogger:
    """Tests for TradeSkipLogger."""

    def test_log_skip_basic(self) -> None:
        """log_skip() records one item with correct symbol and skip_reason."""
        skip_logger = TradeSkipLogger(run_id="run-001")
        skip_logger.log_skip(
            symbol="BTC/USDT",
            skip_reason="max_position_size",
            signal_context={"rsi": 28.4},
            hypothetical_entry_price=Decimal("42000"),
        )

        all_skips = skip_logger.get_all_skipped()
        assert len(all_skips) == 1
        item = all_skips[0]
        assert item.symbol == "BTC/USDT"
        assert item.skip_reason == "max_position_size"
        assert item.run_id == "run-001"
        assert item.hypothetical_entry_price == Decimal("42000")

    def test_skip_summary(self) -> None:
        """get_skip_summary() aggregates counts by skip_reason."""
        skip_logger = TradeSkipLogger(run_id="run-002")
        skip_logger.log_skip("BTC/USDT", "regime_risk_off")
        skip_logger.log_skip("ETH/USDT", "regime_risk_off")
        skip_logger.log_skip("BTC/USDT", "circuit_breaker")

        summary = skip_logger.get_skip_summary()
        assert summary == {"regime_risk_off": 2, "circuit_breaker": 1}

    def test_skip_count_property(self) -> None:
        """skip_count reflects the total number of log_skip() calls."""
        skip_logger = TradeSkipLogger()
        for i in range(5):
            skip_logger.log_skip(f"TOKEN{i}/USDT", "test_reason")

        assert skip_logger.skip_count == 5

    def test_clear(self) -> None:
        """clear() removes all recorded skip events."""
        skip_logger = TradeSkipLogger(run_id="run-003")
        skip_logger.log_skip("BTC/USDT", "concentration_cap")
        skip_logger.log_skip("ETH/USDT", "daily_limit")
        assert skip_logger.skip_count == 2

        skip_logger.clear()

        assert skip_logger.skip_count == 0
        assert skip_logger.get_all_skipped() == []
