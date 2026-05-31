"""
tests/unit/test_momentum_breakout.py
--------------------------------------
Unit tests for MomentumBreakoutStrategy
(packages/trading/strategies/momentum_breakout.py).

Verifies the BUY-only Donchian breakout entry, the uptrend trend filter, the
new-breakout event semantics (fires once per fresh breakout, not on every bar
above the level), the never-SELL contract (exits delegated to brackets), the
validated ATR-bracket schema defaults, and parameter validation.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from common.models import OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy


def _bar(close: float, high: float, ts: datetime, symbol: str = "BTC/USD") -> OHLCVBar:
    c, h = Decimal(str(close)), Decimal(str(high))
    return OHLCVBar(
        symbol=symbol, timeframe=TimeFrame.ONE_DAY, timestamp=ts,
        open=c, high=h, low=c * Decimal("0.99"), close=c, volume=Decimal("100"),
    )


def _series(rows: list[tuple[float, float]], symbol: str = "BTC/USD") -> list[OHLCVBar]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [_bar(c, h, base + timedelta(days=i), symbol) for i, (c, h) in enumerate(rows)]


def _strategy(**params: object) -> MomentumBreakoutStrategy:
    return MomentumBreakoutStrategy(strategy_id="mb-test", params=params)


class TestWarmup:
    def test_insufficient_bars_returns_empty(self) -> None:
        strat = _strategy(lookback=40, trend_sma_period=100)
        assert strat.on_bar(_series([(100.0, 101.0)] * 10)) == []


class TestEntry:
    def test_new_breakout_in_uptrend_buys(self) -> None:
        strat = _strategy(lookback=10, trend_sma_period=20, position_size=2000.0)
        # 30 rising bars (uptrend), flat range, then a fresh breakout high.
        rows = [(100.0 + i, 101.0 + i) for i in range(30)]  # close=high-1, rising
        # Make the last bar a NEW high clearly above the prior 10-bar high.
        rows.append((200.0, 200.0))  # big breakout close above prior highs
        signals = strat.on_bar(_series(rows))
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].target_position == Decimal("2000")

    def test_no_signal_when_not_a_new_breakout(self) -> None:
        # Already broke out on the prior bar -> current bar is not a NEW event.
        strat = _strategy(lookback=5, trend_sma_period=10, position_size=2000.0)
        rows = [(100.0, 101.0) for _ in range(20)]
        rows.append((150.0, 150.0))  # breakout bar
        rows.append((151.0, 151.0))  # still above, but not a new event
        signals = strat.on_bar(_series(rows))
        assert signals == []

    def test_trend_filter_blocks_breakout_below_sma(self) -> None:
        # A breakout to a new high but price is below the long SMA (downtrend).
        strat = _strategy(lookback=5, trend_sma_period=20, position_size=2000.0)
        rows = [(300.0 - i * 5, 301.0 - i * 5) for i in range(25)]  # steady decline
        # Local new high vs prior 5 bars but still far below SMA20.
        last_close = rows[-1][0] + 6.0
        rows.append((last_close, last_close))
        assert strat.on_bar(_series(rows)) == []

    def test_breakout_without_trend_filter(self) -> None:
        strat = _strategy(lookback=5, trend_sma_period=None, position_size=2000.0)
        rows = [(100.0, 101.0) for _ in range(15)]
        rows.append((130.0, 130.0))  # fresh breakout, no trend gate
        signals = strat.on_bar(_series(rows))
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY


class TestNeverSells:
    def test_emits_no_sell_across_a_decline(self) -> None:
        strat = _strategy(lookback=10, trend_sma_period=20)
        rows = [(200.0 - i, 201.0 - i) for i in range(120)]
        bars = _series(rows)
        for end in range(strat.min_bars_required, len(bars)):
            sigs = strat.on_bar(bars[: end + 1])
            assert all(s.direction != SignalDirection.SELL for s in sigs)


class TestParams:
    def test_schema_has_validated_atr_bracket_defaults(self) -> None:
        schema = MomentumBreakoutStrategy.parameter_schema()
        props = schema["properties"]
        assert props["lookback"]["default"] == 40
        assert props["trend_sma_period"]["default"] == 100
        assert props["bracket_mode"]["default"] == "atr"
        assert props["bracket_mode"]["enum"] == ["fixed", "atr"]
        assert props["bracket_atr_sl_multiplier"]["default"] == 1.5
        assert props["bracket_atr_tp_multiplier"]["default"] == 3.0

    def test_min_bars_excludes_atr_period(self) -> None:
        # min_bars is driven by entry indicators only (lookback / trend SMA),
        # not the engine-level ATR bracket period.
        strat = _strategy(lookback=40, trend_sma_period=100)
        assert strat.min_bars_required == max(40, 100) + 3

    def test_invalid_lookback_rejected(self) -> None:
        with pytest.raises(ValueError):
            _strategy(lookback=1)  # < ge=2

    def test_invalid_atr_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError):
            _strategy(bracket_atr_sl_multiplier=50.0)  # > le=20
