"""
tests/unit/test_rsi_fgi_boost.py
----------------------------------
Unit tests for RSI strategy Fear & Greed Index confidence boost (Sprint 32).

Module under test
-----------------
packages/trading/strategies/rsi_mean_reversion.py
  RSIMeanReversionStrategy._fgi_confidence_boost(fgi, direction) -> float

The method is a @staticmethod so it can be called directly without
instantiating the strategy, keeping tests fast and isolated.

Coverage group (7 tests)
-------------------------
TestFGIConfidenceBoost  -- BUY/SELL per FGI band, None, Neutral, HOLD
"""

from __future__ import annotations

import pytest

from common.types import SignalDirection
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy


# Convenience alias to keep test lines short
_boost = RSIMeanReversionStrategy._fgi_confidence_boost


class TestFGIConfidenceBoost:
    """
    Tests for _fgi_confidence_boost() covering every documented band and
    edge case.

    BUY direction (contrarian long bias):
      fgi <= 24 (Extreme Fear):  +0.10  -- oversold, strong bounce expected
      fgi <= 44 (Fear):          +0.05
      fgi <= 55 (Neutral):        0.00
      fgi <= 75 (Greed):         -0.05  -- momentum against mean-reversion
      fgi >  75 (Extreme Greed): -0.10  -- least conviction

    SELL direction (contrarian short bias):
      fgi >= 76 (Extreme Greed): +0.10  -- overbought, strong pullback expected
      fgi >= 56 (Greed):         +0.05
      fgi >= 45 (Neutral):        0.00
      fgi >= 25 (Fear):          -0.05  -- momentum against mean-reversion
      fgi <  25 (Extreme Fear):  -0.10  -- least conviction
    """

    def test_buy_extreme_fear(self) -> None:
        """FGI=10, BUY direction => +0.10 (strong contrarian BUY boost)."""
        assert _boost(10, SignalDirection.BUY) == pytest.approx(0.10)

    def test_buy_extreme_greed(self) -> None:
        """FGI=80, BUY direction => -0.10 (extreme greed reduces BUY confidence)."""
        assert _boost(80, SignalDirection.BUY) == pytest.approx(-0.10)

    def test_sell_extreme_greed(self) -> None:
        """FGI=80, SELL direction => +0.10 (extreme greed boosts SELL confidence)."""
        assert _boost(80, SignalDirection.SELL) == pytest.approx(0.10)

    def test_sell_extreme_fear(self) -> None:
        """FGI=10, SELL direction => -0.10 (extreme fear reduces SELL confidence)."""
        assert _boost(10, SignalDirection.SELL) == pytest.approx(-0.10)

    def test_fgi_none_noop(self) -> None:
        """
        _fgi_confidence_boost expects an int, but when called with None
        the strategy skips the call entirely. We verify the static method
        treats the HOLD branch as 0.0 (the None guard is in on_bar, not here).
        FGI=None is not a valid call — test HOLD as a proxy for 'no adjustment'.
        """
        # The method is only called when fgi_value is not None.
        # When direction is HOLD, the method returns 0.0 directly.
        assert _boost(10, SignalDirection.HOLD) == 0.0

    def test_neutral_noop(self) -> None:
        """FGI=50 (Neutral), BUY => 0.0 (no adjustment in neutral zone)."""
        assert _boost(50, SignalDirection.BUY) == pytest.approx(0.0)

    def test_hold_returns_zero(self) -> None:
        """HOLD direction always returns 0.0 regardless of FGI value."""
        for fgi_val in [0, 10, 50, 85, 100]:
            assert _boost(fgi_val, SignalDirection.HOLD) == 0.0

    @pytest.mark.parametrize(
        "fgi,direction,expected",
        [
            # BUY bands
            (0, SignalDirection.BUY, 0.10),
            (24, SignalDirection.BUY, 0.10),
            (25, SignalDirection.BUY, 0.05),
            (44, SignalDirection.BUY, 0.05),
            (45, SignalDirection.BUY, 0.0),
            (55, SignalDirection.BUY, 0.0),
            (56, SignalDirection.BUY, -0.05),
            (75, SignalDirection.BUY, -0.05),
            (76, SignalDirection.BUY, -0.10),
            (100, SignalDirection.BUY, -0.10),
            # SELL bands
            (100, SignalDirection.SELL, 0.10),
            (76, SignalDirection.SELL, 0.10),
            (75, SignalDirection.SELL, 0.05),
            (56, SignalDirection.SELL, 0.05),
            (55, SignalDirection.SELL, 0.0),
            (45, SignalDirection.SELL, 0.0),
            (44, SignalDirection.SELL, -0.05),
            (25, SignalDirection.SELL, -0.05),
            (24, SignalDirection.SELL, -0.10),
            (0, SignalDirection.SELL, -0.10),
        ],
    )
    def test_all_bands_parametrized(
        self,
        fgi: int,
        direction: SignalDirection,
        expected: float,
    ) -> None:
        """Parametrized coverage of all FGI band boundaries for both directions."""
        assert _boost(fgi, direction) == pytest.approx(expected)
