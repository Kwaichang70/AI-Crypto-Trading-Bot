"""
tests/unit/test_sl_tp_reversion.py
-----------------------------------
Unit tests for SLTPReversionStrategy (packages/trading/strategies/sl_tp_reversion.py).

Verifies the dip-buying entry, the uptrend trend filter, the BUY-only
contract (no SELL ever — exits are delegated to the bracket manager), and
parameter validation.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from common.models import OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.strategies.sl_tp_reversion import SLTPReversionStrategy


def _bar(close: float, ts: datetime, symbol: str = "BTC/USD") -> OHLCVBar:
    c = Decimal(str(close))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=ts,
        open=c,
        high=c * Decimal("1.005"),
        low=c * Decimal("0.995"),
        close=c,
        volume=Decimal("100"),
    )


def _series(closes: list[float], symbol: str = "BTC/USD") -> list[OHLCVBar]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [_bar(c, base + timedelta(hours=i), symbol) for i, c in enumerate(closes)]


def _strategy(**params: object) -> SLTPReversionStrategy:
    return SLTPReversionStrategy(strategy_id="sltp-test", params=params)


class TestWarmup:
    def test_insufficient_bars_returns_empty(self) -> None:
        strat = _strategy(rsi_period=2, trend_sma_period=200)
        assert strat.on_bar(_series([100.0] * 10)) == []


class TestEntry:
    def test_oversold_dip_in_uptrend_triggers_buy(self) -> None:
        # Long rising trend (close > SMA200), then a sharp 2-bar dip pushes
        # RSI-2 below the entry threshold.
        strat = _strategy(rsi_period=2, entry_threshold=15.0, trend_sma_period=200)
        closes = [100.0 + i * 0.5 for i in range(220)]  # steady uptrend
        # Final dip: drop the last two bars while still above SMA200.
        closes[-2] = closes[-3] - 5.0
        closes[-1] = closes[-2] - 5.0
        signals = strat.on_bar(_series(closes))
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == SignalDirection.BUY
        assert sig.target_position > Decimal("0")
        assert 0.0 < sig.confidence <= 1.0

    def test_no_buy_when_rsi_above_threshold(self) -> None:
        # Steady rising trend -> RSI-2 high -> no entry.
        strat = _strategy(rsi_period=2, entry_threshold=10.0, trend_sma_period=200)
        closes = [100.0 + i * 0.5 for i in range(220)]
        assert strat.on_bar(_series(closes)) == []

    def test_trend_filter_blocks_dip_below_sma(self) -> None:
        # A downtrend dip: RSI-2 is low but close <= SMA200 -> filtered out.
        strat = _strategy(rsi_period=2, entry_threshold=30.0, trend_sma_period=200)
        closes = [300.0 - i * 0.5 for i in range(220)]  # steady downtrend
        assert strat.on_bar(_series(closes)) == []

    def test_dip_without_trend_filter_buys(self) -> None:
        # Disable the trend filter -> a dip buys regardless of trend.
        strat = _strategy(rsi_period=2, entry_threshold=30.0, trend_sma_period=None)
        closes = [300.0 - i * 0.5 for i in range(60)]
        closes[-1] = closes[-2] - 10.0  # extra dip
        signals = strat.on_bar(_series(closes))
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY


class TestNeverSells:
    def test_emits_no_sell_across_a_full_decline(self) -> None:
        # Feed a long monotonic decline; the strategy must NEVER emit SELL
        # (exits are the bracket manager's job).
        strat = _strategy(rsi_period=2, entry_threshold=10.0, trend_sma_period=None)
        closes = [200.0 - i for i in range(120)]
        bars = _series(closes)
        for end in range(strat.min_bars_required, len(bars)):
            sigs = strat.on_bar(bars[: end + 1])
            assert all(s.direction != SignalDirection.SELL for s in sigs)


class TestParams:
    def test_schema_exposes_bracket_and_entry_params(self) -> None:
        schema = SLTPReversionStrategy.parameter_schema()
        props = schema["properties"]
        for key in (
            "rsi_period", "entry_threshold", "trend_sma_period", "position_size",
            "bracket_stop_loss_pct", "bracket_take_profit_pct", "bracket_mode",
            "bracket_atr_sl_multiplier", "bracket_atr_tp_multiplier", "bracket_atr_period",
        ):
            assert key in props
        assert props["bracket_mode"]["enum"] == ["fixed", "atr"]

    def test_invalid_entry_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            _strategy(entry_threshold=99.0)  # > le=49

    def test_invalid_rsi_period_rejected(self) -> None:
        with pytest.raises(ValueError):
            _strategy(rsi_period=1)  # < ge=2
