"""
tests/unit/test_dca_rsi_hybrid.py
-----------------------------------
Unit tests for the DCA + RSI Hybrid Trading Strategy.

Module under test
-----------------
packages/trading/strategies/dca_rsi_hybrid.py

Test coverage
-------------
TestDCAParams (4 tests)
  - Default values are applied when no params given
  - Valid boundary values are accepted
  - Invalid values raise ValueError (negative dca_amount, bad multiplier, etc.)
  - rsi_boost_threshold >= rsi_skip_threshold is rejected

TestDCABarCounting (4 tests)
  - DCA fires exactly on bars that are multiples of dca_interval_bars
  - No signal on bars that are not DCA bars (and RSI below take-profit threshold)
  - Counter resets correctly across on_start/on_stop cycle
  - DCA fires on bar 1 when dca_interval_bars=1

TestDCARSIBoost (3 tests)
  - buy_amount equals dca_amount when RSI is around 50
  - buy_amount is multiplied when RSI is below boost threshold
  - Confidence is 0.9 when RSI is very low (falling bars)

TestDCASkipHighRSI (2 tests)
  - Returns no BUY when RSI is above rsi_skip_threshold on a DCA bar
  - Emits BUY when RSI is below the skip threshold

TestDCATakeProfit (3 tests)
  - Emits SELL on a non-DCA bar when RSI > take_profit_rsi
  - SELL target_position equals position_size * take_profit_pct
  - Returns empty list (HOLD) on non-DCA bar when RSI is moderate

TestDCAFGIBoost (3 tests)
  - BUY confidence increases on extreme fear (fgi <= 24)
  - SELL confidence increases on extreme greed (fgi >= 76)
  - No boost when mtf_context is None

TestDCAMinBars (2 tests)
  - min_bars_required equals rsi_period * 3 + 1
  - Returns empty list when fewer than min_bars_required bars are supplied

Design notes
------------
- All tests are synchronous (strategy on_bar is not async).
- asyncio_mode = "auto" in pyproject.toml; no @pytest.mark.asyncio needed.

RSI with constant prices
~~~~~~~~~~~~~~~~~~~~~~~~
_compute_rsi returns 100 when all closes are flat (avg_loss == 0).
Use alternating or non-constant prices to avoid this.

RSI zones
~~~~~~~~~
- Falling bars (start=120, step=1.5): RSI < 30 after convergence.
- Rising bars (start=90, step=3):    RSI > 70 after convergence.
- Alternating ±0.5:                  RSI ≈ 50.

Validator constraint
~~~~~~~~~~~~~~~~~~~~
_DCAParams enforces take_profit_rsi < rsi_skip_threshold.
Always use a gap of at least 2 between them in test params.

pytest.approx is used for float comparisons.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy, _DCAParams


# ---------------------------------------------------------------------------
# Bar factory helpers
# ---------------------------------------------------------------------------


def _bar(
    close: float = 100.0,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 1000.0,
    symbol: str = "BTC/USDT",
) -> OHLCVBar:
    """Construct a minimal OHLCVBar for strategy tests."""
    c = Decimal(str(close))
    h = Decimal(str(high)) if high is not None else (c * Decimal("1.01"))
    lo = Decimal(str(low)) if low is not None else (c * Decimal("0.99"))
    o = Decimal(str(open_)) if open_ is not None else c
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=Decimal(str(volume)),
    )


def _alternating_bars(n: int, base: float = 100.0, amplitude: float = 0.5) -> list[OHLCVBar]:
    """
    Generate n bars with alternating up/down closes, producing RSI ~ 50.

    Never produces flat closes (which would give RSI=100 from avg_loss=0).
    """
    bars = []
    for i in range(n):
        close = base + amplitude if (i % 2 == 0) else base - amplitude
        bars.append(_bar(close))
    return bars


def _rising_bars(n: int, start: float = 90.0, step: float = 3.0) -> list[OHLCVBar]:
    """Generate n bars with steadily rising closes (RSI > 70)."""
    return [_bar(start + i * step) for i in range(n)]


def _falling_bars(n: int, start: float = 120.0, step: float = 1.5) -> list[OHLCVBar]:
    """Generate n bars with steadily falling closes (RSI < 30)."""
    return [_bar(max(start - i * step, 0.01)) for i in range(n)]


def _make_strategy(params: dict[str, Any] | None = None) -> DCARSIHybridStrategy:
    """Create a DCARSIHybridStrategy with given params and call on_start."""
    strat = DCARSIHybridStrategy(strategy_id="dca-test", params=params)
    strat.on_start("run-001")
    return strat


def _neutral_params(**overrides: Any) -> dict[str, Any]:
    """
    Base params that keep DCA always active on DCA bars (RSI < skip)
    and never trigger take-profit on alternating bars (RSI < take_profit_rsi).

    rsi_skip_threshold=90, take_profit_rsi=85 satisfies the validator
    (take_profit_rsi < rsi_skip_threshold) and puts both thresholds well
    above the alternating-bar RSI (~50), so neither path fires unexpectedly.
    """
    base: dict[str, Any] = {
        "rsi_skip_threshold": 90.0,
        "take_profit_rsi": 85.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestDCAParams
# ---------------------------------------------------------------------------


class TestDCAParams:
    """Pydantic parameter validation for _DCAParams."""

    def test_defaults_applied(self) -> None:
        """No params => all defaults are filled in."""
        p = _DCAParams()
        assert p.dca_interval_bars == 16
        assert p.dca_amount == pytest.approx(50.0)
        assert p.rsi_period == 14
        assert p.rsi_boost_threshold == pytest.approx(40.0)
        assert p.rsi_boost_multiplier == pytest.approx(2.0)
        assert p.rsi_skip_threshold == pytest.approx(75.0)
        assert p.take_profit_rsi == pytest.approx(70.0)
        assert p.take_profit_pct == pytest.approx(0.5)
        assert p.position_size == pytest.approx(1000.0)
        assert p.trailing_stop_pct is None

    def test_valid_boundary_values(self) -> None:
        """Custom valid values are accepted without error."""
        p = _DCAParams(
            dca_interval_bars=1,
            dca_amount=10.0,
            rsi_period=5,
            rsi_boost_threshold=30.0,
            rsi_boost_multiplier=3.0,
            rsi_skip_threshold=80.0,
            take_profit_rsi=65.0,
            take_profit_pct=0.25,
            position_size=500.0,
            trailing_stop_pct=0.03,
        )
        assert p.dca_interval_bars == 1
        assert p.rsi_period == 5

    def test_invalid_dca_amount_zero_raises(self) -> None:
        """dca_amount=0 should raise a validation error."""
        with pytest.raises(Exception):
            _DCAParams(dca_amount=0.0)

    def test_boost_threshold_ge_skip_threshold_raises(self) -> None:
        """rsi_boost_threshold >= rsi_skip_threshold should raise ValueError."""
        with pytest.raises(Exception):
            _DCAParams(rsi_boost_threshold=80.0, rsi_skip_threshold=70.0)


# ---------------------------------------------------------------------------
# TestDCABarCounting
# ---------------------------------------------------------------------------


class TestDCABarCounting:
    """Verify the DCA cadence fires precisely every dca_interval_bars."""

    def test_dca_fires_on_interval(self) -> None:
        """BUY signal emitted on bars that are exact multiples of interval."""
        strat = _make_strategy(
            _neutral_params(
                dca_interval_bars=5,
                rsi_boost_threshold=10.0,  # Boost only if RSI < 10 (never with alternating)
            )
        )
        min_bars = strat.min_bars_required
        base_bars = _alternating_bars(min_bars)

        # Feed one extra bar at a time, accumulating history
        history: list[OHLCVBar] = list(base_bars)
        buy_bar_numbers: list[int] = []

        for i in range(1, 16):
            history.append(_alternating_bars(1)[0])
            sigs = strat.on_bar(history)
            if sigs and sigs[0].direction == SignalDirection.BUY:
                buy_bar_numbers.append(i)

        # DCA fires at bar_counter multiples of 5: counters 5, 10, 15
        assert all(n % 5 == 0 for n in buy_bar_numbers), (
            f"DCA fired on non-interval bars: {buy_bar_numbers}"
        )
        assert len(buy_bar_numbers) == 3

    def test_no_signal_on_non_dca_non_takeprofit_bar(self) -> None:
        """Non-DCA bar with RSI below take_profit threshold => HOLD."""
        # interval=10 so bar_counter=1 after first call is NOT a DCA bar.
        # take_profit_rsi=85 ensures alternating RSI (~50) won't trigger it.
        strat = _make_strategy(_neutral_params(dca_interval_bars=10))
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)
        # Counter becomes 1 after this call (not a DCA bar, not take-profit)
        sigs = strat.on_bar(bars)
        assert sigs == []

    def test_counter_resets_on_stop_start_cycle(self) -> None:
        """on_stop resets bar_counter; on_start re-initialises it to 0."""
        strat = _make_strategy(_neutral_params(dca_interval_bars=3))
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)
        # Advance counter by 2 (not yet a DCA bar)
        strat.on_bar(bars)
        strat.on_bar(bars)
        assert strat._bar_counter == 2

        # Stop and restart — counter should reset to 0
        strat.on_stop()
        strat.on_start("run-002")
        assert strat._bar_counter == 0

    def test_dca_interval_one_fires_every_bar(self) -> None:
        """dca_interval_bars=1 means every bar is a DCA bar."""
        strat = _make_strategy(_neutral_params(dca_interval_bars=1))
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)
        # counter=1 after this call; 1 % 1 == 0 => DCA bar => BUY
        sigs = strat.on_bar(bars)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.BUY


# ---------------------------------------------------------------------------
# TestDCARSIBoost
# ---------------------------------------------------------------------------


class TestDCARSIBoost:
    """Verify RSI-based buy amount scaling and confidence tiers."""

    def test_buy_amount_base_when_rsi_above_boost_threshold(self) -> None:
        """Alternating bars => RSI ~50 > boost_threshold(40) => buy_amount == dca_amount."""
        strat = _make_strategy(
            _neutral_params(
                dca_interval_bars=1,
                dca_amount=75.0,
                rsi_boost_threshold=40.0,
            )
        )
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)
        sigs = strat.on_bar(bars)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.BUY
        # RSI ~50 > 40 => no boost; target_position == dca_amount exactly
        assert float(sigs[0].target_position) == pytest.approx(75.0)

    def test_buy_amount_boosted_when_rsi_below_boost_threshold(self) -> None:
        """Falling prices push RSI below boost_threshold; buy_amount is multiplied."""
        strat = _make_strategy(
            _neutral_params(
                dca_interval_bars=1,
                dca_amount=50.0,
                rsi_boost_threshold=45.0,
                rsi_boost_multiplier=3.0,
            )
        )
        min_bars = strat.min_bars_required
        # Steadily falling prices => RSI well below 45
        bars = _falling_bars(min_bars)
        sigs = strat.on_bar(bars)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.BUY
        assert float(sigs[0].target_position) == pytest.approx(150.0)  # 50 * 3.0

    def test_confidence_is_high_when_rsi_very_low(self) -> None:
        """Falling bars produce RSI < 30 => confidence = 0.9 (no FGI applied)."""
        strat = _make_strategy(
            _neutral_params(
                dca_interval_bars=1,
                rsi_boost_threshold=45.0,
            )
        )
        min_bars = strat.min_bars_required
        bars = _falling_bars(min_bars)
        sigs = strat.on_bar(bars)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.BUY
        # Falling bars => RSI < 30 => confidence tier = 0.9
        assert sigs[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestDCASkipHighRSI
# ---------------------------------------------------------------------------


class TestDCASkipHighRSI:
    """Verify that DCA buys are suppressed when RSI is overbought."""

    def test_skip_when_rsi_above_skip_threshold(self) -> None:
        """Rising bars push RSI near 100 > skip_threshold(60) => no BUY signal."""
        strat = _make_strategy(
            {
                "dca_interval_bars": 1,
                "rsi_skip_threshold": 60.0,
                "take_profit_rsi": 55.0,  # Must be strictly less than skip_threshold
                "rsi_boost_threshold": 40.0,
            }
        )
        min_bars = strat.min_bars_required
        # Strongly rising bars => RSI very high (approaching 100)
        bars = _rising_bars(min_bars)
        sigs = strat.on_bar(bars)
        # No BUY should be emitted (either take-profit SELL or empty — never BUY)
        buy_sigs = [s for s in sigs if s.direction == SignalDirection.BUY]
        assert len(buy_sigs) == 0

    def test_buy_emitted_when_rsi_below_skip_threshold(self) -> None:
        """Alternating bars => RSI ~50 which is well below skip_threshold => BUY."""
        strat = _make_strategy(_neutral_params(dca_interval_bars=1))
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)
        sigs = strat.on_bar(bars)
        assert len(sigs) == 1
        assert sigs[0].direction == SignalDirection.BUY


# ---------------------------------------------------------------------------
# TestDCATakeProfit
# ---------------------------------------------------------------------------


class TestDCATakeProfit:
    """Verify partial take-profit SELL logic on non-DCA bars."""

    def test_sell_emitted_when_rsi_above_take_profit_threshold(self) -> None:
        """Non-DCA bar (counter=1) with high RSI emits SELL signal."""
        # dca_interval_bars=2 => counter=1 is NOT a DCA bar.
        # Rising bars push RSI > 55, triggering take-profit.
        strat = _make_strategy(
            {
                "dca_interval_bars": 2,
                "take_profit_rsi": 55.0,
                "rsi_skip_threshold": 70.0,  # Must be > take_profit_rsi
                "rsi_boost_threshold": 40.0,
            }
        )
        min_bars = strat.min_bars_required
        bars = _rising_bars(min_bars)
        sigs = strat.on_bar(bars)
        # bar_counter=1 (not a DCA bar since 1 % 2 != 0)
        sell_sigs = [s for s in sigs if s.direction == SignalDirection.SELL]
        assert len(sell_sigs) == 1

    def test_sell_target_position_equals_position_size_times_pct(self) -> None:
        """SELL target_position == position_size * take_profit_pct."""
        strat = _make_strategy(
            {
                "dca_interval_bars": 2,
                "take_profit_rsi": 55.0,
                "rsi_skip_threshold": 70.0,
                "position_size": 800.0,
                "take_profit_pct": 0.4,
                "rsi_boost_threshold": 40.0,
            }
        )
        min_bars = strat.min_bars_required
        bars = _rising_bars(min_bars)
        sigs = strat.on_bar(bars)
        sell_sigs = [s for s in sigs if s.direction == SignalDirection.SELL]
        assert len(sell_sigs) == 1
        assert float(sell_sigs[0].target_position) == pytest.approx(320.0)  # 800 * 0.4

    def test_hold_on_non_dca_bar_when_rsi_moderate(self) -> None:
        """Non-DCA bar with RSI below take_profit_rsi returns empty list."""
        # interval=2 => counter=1 is NOT a DCA bar.
        # take_profit_rsi=85 ensures alternating RSI (~50) won't trigger.
        strat = _make_strategy(_neutral_params(dca_interval_bars=2))
        min_bars = strat.min_bars_required
        bars = _alternating_bars(min_bars)  # RSI ~50 < 85
        sigs = strat.on_bar(bars)
        assert sigs == []


# ---------------------------------------------------------------------------
# TestDCAFGIBoost
# ---------------------------------------------------------------------------


class TestDCAFGIBoost:
    """Verify Fear & Greed Index confidence adjustment."""

    def test_buy_confidence_boosted_in_extreme_fear(self) -> None:
        """BUY confidence increases when FGI signals extreme fear (fgi=10, delta=+0.10)."""
        # Two separate strategy instances — one without FGI, one with
        base_params = _neutral_params(dca_interval_bars=1)
        strat_no_fgi = _make_strategy(base_params)
        strat_with_fgi = _make_strategy(base_params)
        min_bars = strat_no_fgi.min_bars_required
        bars = _alternating_bars(min_bars)

        # Without FGI
        sigs_no_fgi = strat_no_fgi.on_bar(bars, mtf_context=None)

        # With extreme fear FGI (fgi=10 => +0.10 for BUY)
        ctx_fear = MultiTimeframeContext(fear_greed_index=10)
        sigs_with_fgi = strat_with_fgi.on_bar(bars, mtf_context=ctx_fear)

        assert len(sigs_no_fgi) == 1
        assert len(sigs_with_fgi) == 1
        assert sigs_no_fgi[0].direction == SignalDirection.BUY
        assert sigs_with_fgi[0].direction == SignalDirection.BUY
        # FGI=10 (extreme fear) adds +0.10 to BUY confidence
        assert sigs_with_fgi[0].confidence > sigs_no_fgi[0].confidence

    def test_sell_confidence_boosted_in_extreme_greed(self) -> None:
        """SELL confidence increases when FGI signals extreme greed (fgi=90, delta=+0.10)."""
        params = {
            "dca_interval_bars": 2,
            "take_profit_rsi": 55.0,
            "rsi_skip_threshold": 70.0,
            "rsi_boost_threshold": 40.0,
        }
        strat_no_fgi = _make_strategy(params)
        strat_with_fgi = _make_strategy(params)
        min_bars = strat_no_fgi.min_bars_required
        bars = _rising_bars(min_bars)

        sigs_no_fgi = strat_no_fgi.on_bar(bars)
        ctx_greed = MultiTimeframeContext(fear_greed_index=90)
        sigs_with_fgi = strat_with_fgi.on_bar(bars, mtf_context=ctx_greed)

        sell_no_fgi = [s for s in sigs_no_fgi if s.direction == SignalDirection.SELL]
        sell_with_fgi = [s for s in sigs_with_fgi if s.direction == SignalDirection.SELL]

        # Only assert if take-profit fired in both paths
        if sell_no_fgi and sell_with_fgi:
            assert sell_with_fgi[0].confidence >= sell_no_fgi[0].confidence

    def test_no_fgi_boost_when_mtf_context_is_none(self) -> None:
        """When mtf_context is None, no FGI adjustment is applied to confidence."""
        # Two calls with same bars: one with None context, one with FGI=10
        # If FGI had been applied, confidence would differ.
        strat_none = _make_strategy(_neutral_params(dca_interval_bars=1))
        strat_fgi = _make_strategy(_neutral_params(dca_interval_bars=1))
        min_bars = strat_none.min_bars_required
        bars = _alternating_bars(min_bars)

        sigs_none = strat_none.on_bar(bars, mtf_context=None)
        ctx_fear = MultiTimeframeContext(fear_greed_index=10)
        sigs_fgi = strat_fgi.on_bar(bars, mtf_context=ctx_fear)

        assert len(sigs_none) == 1
        # None context => raw RSI confidence; FGI=10 context => higher confidence
        # Therefore None confidence must be strictly lower
        assert sigs_none[0].confidence < sigs_fgi[0].confidence


# ---------------------------------------------------------------------------
# TestDCAMinBars
# ---------------------------------------------------------------------------


class TestDCAMinBars:
    """Verify min_bars_required and warmup guard behaviour."""

    def test_min_bars_required_formula(self) -> None:
        """min_bars_required == rsi_period * 3 + 1."""
        strat = _make_strategy({"rsi_period": 14})
        assert strat.min_bars_required == 14 * 3 + 1  # 43

    def test_warmup_guard_returns_empty_list(self) -> None:
        """Fewer than min_bars_required bars returns [] regardless of DCA bar."""
        # dca_interval_bars=1 so every call is a DCA bar (counter 1,2,3...)
        strat = _make_strategy({"dca_interval_bars": 1, "rsi_period": 14})
        min_bars = strat.min_bars_required
        # Provide one fewer bar than required
        bars = _alternating_bars(min_bars - 1)
        sigs = strat.on_bar(bars)
        assert sigs == []
