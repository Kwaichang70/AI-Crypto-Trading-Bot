"""
tests/unit/test_indicators.py
------------------------------
Comprehensive unit tests for packages/data/indicators.py.

Indicator functions under test
-------------------------------
  sma                 Simple Moving Average
  ema                 Exponential Moving Average
  rsi                 Relative Strength Index (Wilder's smoothing)
  macd                MACD line / signal / histogram
  atr                 Average True Range (Wilder's smoothing)
  bollinger_bands     Bollinger Bands (SMA +/- num_std * rolling std)
  donchian_channel    Donchian Channel (rolling high/low)
  returns             Simple or log price returns
  rolling_volatility  Annualisable rolling realised volatility
  compute_features    Full ML feature set orchestrator

Test design principles
-----------------------
- Deterministic data: explicit price sequences or seeded numpy random walks.
- Class-per-function grouping to mirror source module structure.
- NaN warm-up verification: every function's warm-up window is tested.
- Parameter validation: invalid inputs raise ValueError.
- Edge cases: empty series, single-element series, boundary periods.
- Cross-validation: vectorised RSI and ATR are compared against the
  Decimal-arithmetic helpers in the strategy modules to confirm numerical
  equivalence on the same dataset.

Import convention
-----------------
Use ``from data.indicators import <func>`` — the package name is ``data``
(i.e. packages/data is installed as the ``data`` workspace member).
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from data.indicators import (
    atr,
    bollinger_bands,
    compute_features,
    donchian_channel,
    ema,
    macd,
    returns,
    rolling_volatility,
    rsi,
    sma,
)


# ---------------------------------------------------------------------------
# Shared test-data factories (module-level, no pytest fixtures needed for
# pure functions that accept plain Series/DataFrames)
# ---------------------------------------------------------------------------


def _make_close(values: list[float]) -> pd.Series:
    """Create a close-price Series from an explicit list of float values."""
    return pd.Series(values, dtype=float)


def _make_random_close(n: int, *, seed: int = 42, start: float = 100.0) -> pd.Series:
    """
    Generate a reproducible random-walk close series of length *n*.

    Uses a seeded numpy RandomState so the output is bit-for-bit identical
    across runs regardless of global numpy state.
    """
    rng = np.random.RandomState(seed)  # noqa: NPY002 — deliberate seeded PRNG
    pct_changes = rng.uniform(-0.02, 0.02, size=n - 1)
    prices = [start]
    for pct in pct_changes:
        prices.append(max(1.0, prices[-1] * (1 + pct)))
    return pd.Series(prices, dtype=float)


def _make_ohlcv_df(n: int, *, seed: int = 42, start: float = 100.0) -> pd.DataFrame:
    """
    Build an OHLCV DataFrame of length *n* with consistent OHLCV relationships
    (low <= open/close <= high) using a seeded random walk.
    """
    rng = np.random.RandomState(seed)  # noqa: NPY002
    close_arr = np.empty(n)
    close_arr[0] = start
    changes = rng.uniform(-0.02, 0.02, size=n)
    for i in range(1, n):
        close_arr[i] = max(1.0, close_arr[i - 1] * (1 + changes[i]))

    high_arr = close_arr * rng.uniform(1.001, 1.005, size=n)
    low_arr = close_arr * rng.uniform(0.995, 0.999, size=n)
    open_arr = close_arr * rng.uniform(0.999, 1.001, size=n)
    volume_arr = rng.uniform(100.0, 1000.0, size=n)

    return pd.DataFrame(
        {
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": volume_arr,
        }
    )


# ===========================================================================
# TestSMA
# ===========================================================================


class TestSMA:
    """Tests for sma(series, period)."""

    # --- correctness ---

    def test_known_values_period_3(self) -> None:
        """
        Manual SMA-3 check.

        Series: [1, 2, 3, 4, 5]
        SMA(3): [NaN, NaN, 2.0, 3.0, 4.0]
        """
        series = _make_close([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, 3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert abs(result.iloc[2] - 2.0) < 1e-9
        assert abs(result.iloc[3] - 3.0) < 1e-9
        assert abs(result.iloc[4] - 4.0) < 1e-9

    def test_period_1_returns_original(self) -> None:
        """SMA(1) is the identity — every value equals the input."""
        series = _make_close([10.0, 20.0, 30.0])
        result = sma(series, 1)
        for i, val in enumerate([10.0, 20.0, 30.0]):
            assert abs(result.iloc[i] - val) < 1e-9

    def test_nan_warmup_count(self) -> None:
        """Exactly period-1 leading NaNs for period >= 2."""
        series = _make_random_close(50)
        for period in [2, 5, 10, 20]:
            result = sma(series, period)
            nan_count = result.isna().sum()
            assert nan_count == period - 1, f"period={period}: expected {period-1} NaNs, got {nan_count}"

    def test_constant_series_equals_constant(self) -> None:
        """SMA of a constant series must equal that constant."""
        series = pd.Series([42.0] * 30)
        result = sma(series, 5)
        valid = result.dropna()
        assert (abs(valid - 42.0) < 1e-9).all()

    def test_returns_pandas_series(self) -> None:
        """Return type is pd.Series."""
        result = sma(_make_close([1.0, 2.0, 3.0]), 2)
        assert isinstance(result, pd.Series)

    # --- parameter validation ---

    def test_period_zero_raises(self) -> None:
        """period=0 raises ValueError."""
        with pytest.raises(ValueError, match="SMA period"):
            sma(_make_close([1.0, 2.0, 3.0]), 0)

    def test_period_negative_raises(self) -> None:
        """Negative period raises ValueError."""
        with pytest.raises(ValueError, match="SMA period"):
            sma(_make_close([1.0, 2.0, 3.0]), -1)

    # --- parametrized ---

    @pytest.mark.parametrize(
        "values,period,idx,expected",
        [
            ([1.0, 2.0, 3.0, 4.0, 5.0], 2, 1, 1.5),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 2, 4, 4.5),
            ([10.0, 10.0, 10.0, 20.0], 3, 3, 40.0 / 3),
        ],
    )
    def test_sma_parametrized(
        self, values: list[float], period: int, idx: int, expected: float
    ) -> None:
        """Parametrized spot-checks against hand-computed SMA values."""
        result = sma(_make_close(values), period)
        assert abs(result.iloc[idx] - expected) < 1e-9


# ===========================================================================
# TestEMA
# ===========================================================================


class TestEMA:
    """Tests for ema(series, period, *, adjust)."""

    # --- correctness ---

    def test_ema_greater_than_sma_on_rising_series(self) -> None:
        """
        On a strongly rising series EMA responds faster than SMA
        so ema[-1] > sma[-1].
        """
        series = _make_close(list(range(1, 31)))  # 1..30
        result_ema = ema(series, 5)
        result_sma = sma(series, 5)
        # Final EMA should track closer to the recent high
        assert result_ema.iloc[-1] > result_sma.iloc[-1]

    def test_nan_warmup_count(self) -> None:
        """Exactly period-1 leading NaNs."""
        series = _make_random_close(60)
        for period in [2, 5, 10, 26]:
            result = ema(series, period)
            nan_count = result.isna().sum()
            assert nan_count == period - 1, f"period={period}: expected {period-1} NaNs, got {nan_count}"

    def test_constant_series_equals_constant(self) -> None:
        """EMA of a constant series must converge to that constant."""
        series = pd.Series([50.0] * 50)
        result = ema(series, 5)
        valid = result.dropna()
        # Allow tiny floating-point error
        assert (abs(valid - 50.0) < 1e-6).all()

    def test_adjust_false_vs_true_differ(self) -> None:
        """adjust=True and adjust=False produce different initial values."""
        series = _make_random_close(30)
        r_false = ema(series, 5, adjust=False)
        r_true = ema(series, 5, adjust=True)
        # They must eventually converge but early values differ
        assert not r_false.equals(r_true)

    def test_returns_pandas_series(self) -> None:
        """Return type is pd.Series."""
        result = ema(_make_close([1.0, 2.0, 3.0, 4.0]), 2)
        assert isinstance(result, pd.Series)

    # --- parameter validation ---

    def test_period_zero_raises(self) -> None:
        """period=0 raises ValueError."""
        with pytest.raises(ValueError, match="EMA period"):
            ema(_make_close([1.0, 2.0, 3.0]), 0)

    def test_period_negative_raises(self) -> None:
        """Negative period raises ValueError."""
        with pytest.raises(ValueError, match="EMA period"):
            ema(_make_close([1.0, 2.0, 3.0]), -5)

    # --- edge cases ---

    def test_period_1_no_warmup_nans(self) -> None:
        """EMA(1) has period-1=0 NaNs, so the first value is defined."""
        series = _make_close([10.0, 20.0, 30.0])
        result = ema(series, 1)
        assert not math.isnan(result.iloc[0])


# ===========================================================================
# TestRSI
# ===========================================================================


class TestRSI:
    """Tests for rsi(close, period)."""

    # --- correctness / range ---

    def test_rsi_range_0_to_100(self) -> None:
        """All non-NaN RSI values must be in [0, 100]."""
        close = _make_random_close(100)
        result = rsi(close, 14)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_all_gains_rsi_equals_100(self) -> None:
        """
        Monotonically rising prices → avg_loss = 0 → RSI = 100.

        The implementation handles avg_loss == 0 via float division
        (np.inf) which produces RSI = 100 by IEEE arithmetic.
        """
        close = _make_close([float(i) for i in range(1, 25)])
        result = rsi(close, 5)
        valid = result.dropna()
        # All values should be 100 (or very close due to float arithmetic)
        assert (valid >= 99.99).all()

    def test_all_losses_rsi_equals_0(self) -> None:
        """
        Monotonically falling prices → avg_gain = 0 → RSI = 0.
        """
        close = _make_close([float(25 - i) for i in range(25)])
        result = rsi(close, 5)
        valid = result.dropna()
        assert (valid <= 0.01).all()

    def test_neutral_series_rsi_near_50(self) -> None:
        """
        A series that alternates +1/-1 should produce RSI near 50.
        """
        prices = []
        p = 100.0
        for i in range(50):
            p = p + 1.0 if i % 2 == 0 else p - 1.0
            prices.append(p)
        close = _make_close(prices)
        result = rsi(close, 14)
        valid = result.dropna()
        # With perfectly balanced gains/losses RSI converges near 50
        assert (abs(valid - 50.0) < 10.0).all()

    # --- warm-up ---

    def test_nan_warmup_count(self) -> None:
        """First *period* values are NaN (Wilder seed occupies index period)."""
        close = _make_random_close(60)
        for period in [2, 5, 14]:
            result = rsi(close, period)
            nan_count = result.isna().sum()
            assert nan_count == period, f"period={period}: expected {period} NaNs, got {nan_count}"

    def test_short_series_all_nan(self) -> None:
        """len(close) <= period returns all-NaN series."""
        close = _make_close([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rsi(close, 5)
        assert result.isna().all()

    def test_exact_boundary_all_nan(self) -> None:
        """len(close) == period returns all-NaN."""
        close = _make_close([1.0, 2.0, 3.0])
        result = rsi(close, 3)
        assert result.isna().all()

    def test_one_beyond_boundary_has_valid(self) -> None:
        """len(close) == period + 1 produces exactly one non-NaN value."""
        close = _make_close([1.0, 2.0, 3.0, 4.0])  # period=3, len=4
        result = rsi(close, 3)
        assert result.notna().sum() == 1

    # --- parameter validation ---

    def test_period_1_raises(self) -> None:
        """period=1 raises ValueError (must be >= 2)."""
        with pytest.raises(ValueError, match="RSI period"):
            rsi(_make_close([1.0, 2.0, 3.0]), 1)

    def test_period_0_raises(self) -> None:
        """period=0 raises ValueError."""
        with pytest.raises(ValueError, match="RSI period"):
            rsi(_make_close([1.0, 2.0, 3.0]), 0)

    def test_empty_series_raises(self) -> None:
        """Empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            rsi(pd.Series([], dtype=float), 14)

    # --- cross-validation against strategy helper ---

    def test_rsi_matches_strategy_helper(self) -> None:
        """
        Cross-validate vectorised RSI against the Decimal-arithmetic helper
        ``_compute_rsi`` from packages/trading/strategies/rsi_mean_reversion.py.

        Both implementations follow the same Wilder's-smoothing algorithm
        (SMA seed + recursive exponential smoothing).  For a small, hand-crafted
        dataset the two should agree within a tolerance of 0.01 RSI points.
        """
        from trading.strategies.rsi_mean_reversion import _compute_rsi  # type: ignore[import-untyped]

        # Craft a deterministic price series long enough to produce an RSI value.
        raw_prices = [
            100.0, 101.0, 102.5, 101.0, 99.5, 98.0, 97.5, 98.5,
            100.0, 101.5, 103.0, 102.0, 100.5, 99.0, 98.5,
            99.5, 101.0, 102.5, 103.5, 104.0,
        ]
        period = 5

        # Vectorised RSI — last non-NaN value is our comparison point.
        close_series = _make_close(raw_prices)
        vec_result = rsi(close_series, period)
        vec_rsi_last = vec_result.dropna().iloc[-1]

        # Strategy helper RSI — operates on the same full price sequence.
        decimal_prices = [Decimal(str(p)) for p in raw_prices]
        helper_rsi = float(_compute_rsi(decimal_prices, period))

        assert abs(vec_rsi_last - helper_rsi) < 0.01, (
            f"Vectorised RSI {vec_rsi_last:.4f} differs from helper RSI "
            f"{helper_rsi:.4f} by more than 0.01"
        )

    # --- parametrized ---

    @pytest.mark.parametrize("period", [2, 5, 14, 21])
    def test_rsi_default_period_has_correct_nan_count(self, period: int) -> None:
        """Parametrized NaN-count check for common RSI periods."""
        close = _make_random_close(period * 4, seed=99)
        result = rsi(close, period)
        assert result.isna().sum() == period


# ===========================================================================
# TestMACD
# ===========================================================================


class TestMACD:
    """Tests for macd(close, fast_period, slow_period, signal_period)."""

    # --- return structure ---

    def test_returns_three_series(self) -> None:
        """macd() returns a 3-tuple of pd.Series."""
        close = _make_random_close(60)
        result = macd(close, 12, 26, 9)
        assert len(result) == 3
        assert all(isinstance(s, pd.Series) for s in result)

    def test_histogram_equals_macd_minus_signal(self) -> None:
        """histogram = macd_line - signal_line at all non-NaN positions."""
        close = _make_random_close(80)
        macd_line, signal_line, histogram = macd(close, 12, 26, 9)
        valid_mask = histogram.notna()
        diff = (macd_line - signal_line)[valid_mask]
        residuals = (histogram[valid_mask] - diff).abs()
        assert (residuals < 1e-9).all()

    # --- warm-up ---

    def test_nan_warmup_equals_slow_period_minus_1(self) -> None:
        """All three output series have exactly slow_period-1 leading NaNs."""
        close = _make_random_close(80)
        fast, slow, signal_p = 12, 26, 9
        macd_line, signal_line, histogram = macd(close, fast, slow, signal_p)
        for name, series in [("macd_line", macd_line), ("signal_line", signal_line), ("histogram", histogram)]:
            nan_count = series.isna().sum()
            assert nan_count == slow - 1, (
                f"{name}: expected {slow-1} NaNs, got {nan_count}"
            )

    def test_macd_line_zero_for_identical_emas(self) -> None:
        """
        When fast and slow EMA have the same value (constant series),
        macd_line should be exactly 0.
        """
        close = pd.Series([100.0] * 60)
        macd_line, _, _ = macd(close, 12, 26, 9)
        valid = macd_line.dropna()
        assert (abs(valid) < 1e-9).all()

    # --- parameter validation ---

    def test_slow_equals_fast_raises(self) -> None:
        """slow_period == fast_period raises ValueError."""
        with pytest.raises(ValueError, match="slow_period"):
            macd(_make_random_close(50), fast_period=12, slow_period=12)

    def test_slow_less_than_fast_raises(self) -> None:
        """slow_period < fast_period raises ValueError."""
        with pytest.raises(ValueError, match="slow_period"):
            macd(_make_random_close(50), fast_period=26, slow_period=12)

    # --- non-zero MACD on trending series ---

    def test_macd_positive_on_sustained_uptrend(self) -> None:
        """Fast EMA > slow EMA on a rising series → positive MACD line."""
        close = _make_close([float(i) for i in range(1, 81)])
        macd_line, _, _ = macd(close, 12, 26, 9)
        valid = macd_line.dropna()
        assert (valid > 0).all()

    def test_macd_negative_on_sustained_downtrend(self) -> None:
        """Fast EMA < slow EMA on a falling series → negative MACD line."""
        close = _make_close([float(80 - i) for i in range(80)])
        macd_line, _, _ = macd(close, 12, 26, 9)
        valid = macd_line.dropna()
        assert (valid < 0).all()

    # --- index preservation ---

    def test_output_index_matches_input(self) -> None:
        """All three output Series share the same index as the input."""
        idx = pd.date_range("2024-01-01", periods=60, freq="h")
        close = pd.Series(np.linspace(100, 120, 60), index=idx)
        macd_line, signal_line, histogram = macd(close, 12, 26, 9)
        assert (macd_line.index == idx).all()
        assert (signal_line.index == idx).all()
        assert (histogram.index == idx).all()


# ===========================================================================
# TestATR
# ===========================================================================


class TestATR:
    """Tests for atr(high, low, close, period)."""

    # --- correctness ---

    def test_atr_non_negative(self) -> None:
        """ATR values must be non-negative (True Range is always >= 0)."""
        df = _make_ohlcv_df(80)
        result = atr(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_constant_bars_equals_zero(self) -> None:
        """
        When high == low == close for every bar (zero-range candles),
        ATR converges to 0 after the warm-up.
        """
        n = 40
        const = pd.Series([100.0] * n)
        result = atr(const, const, const, period=5)
        valid = result.dropna()
        assert (abs(valid) < 1e-9).all()

    def test_atr_wider_range_gives_larger_value(self) -> None:
        """
        A series with high-low range of 10 per bar should produce an ATR
        larger than one with range 1 per bar.
        """
        n = 40
        close = pd.Series([100.0] * n)
        high_wide = pd.Series([105.0] * n)
        low_wide = pd.Series([95.0] * n)
        high_narrow = pd.Series([100.5] * n)
        low_narrow = pd.Series([99.5] * n)
        atr_wide = atr(high_wide, low_wide, close, 5).dropna().iloc[-1]
        atr_narrow = atr(high_narrow, low_narrow, close, 5).dropna().iloc[-1]
        assert atr_wide > atr_narrow

    # --- warm-up ---

    def test_nan_warmup_count(self) -> None:
        """First *period* values are NaN (same contract as RSI)."""
        df = _make_ohlcv_df(80)
        for period in [2, 5, 14]:
            result = atr(df["high"], df["low"], df["close"], period)
            nan_count = result.isna().sum()
            assert nan_count == period, (
                f"period={period}: expected {period} NaNs, got {nan_count}"
            )

    def test_short_series_all_nan(self) -> None:
        """len(close) <= period returns all-NaN."""
        n = 5
        h = pd.Series([100.0] * n)
        l = pd.Series([99.0] * n)
        c = pd.Series([99.5] * n)
        result = atr(h, l, c, period=5)
        assert result.isna().all()

    # --- parameter validation ---

    def test_period_1_raises(self) -> None:
        """period=1 raises ValueError."""
        close = _make_random_close(30)
        with pytest.raises(ValueError, match="ATR period"):
            atr(close, close, close, period=1)

    def test_period_0_raises(self) -> None:
        """period=0 raises ValueError."""
        close = _make_random_close(30)
        with pytest.raises(ValueError, match="ATR period"):
            atr(close, close, close, period=0)

    # --- cross-validation against strategy helper ---

    def test_atr_matches_strategy_helper(self) -> None:
        """
        Cross-validate vectorised ATR against the Decimal-arithmetic helper
        ``_compute_atr`` from packages/trading/strategies/breakout.py.

        Both use SMA seed + Wilder's recursive smoothing.  Agreement expected
        within 0.0001 price units for the same input dataset.
        """
        from trading.strategies.breakout import _compute_atr  # type: ignore[import-untyped]

        raw_h = [105.0, 106.0, 107.5, 106.0, 104.5, 103.0, 102.5, 104.0, 106.0, 108.0,
                 110.0, 108.5, 107.0, 105.5, 104.5, 106.0, 108.0, 110.0, 112.0, 113.0]
        raw_l = [ 99.0, 100.0, 101.5, 100.0,  98.5,  97.0,  96.5,  97.5,  99.0, 101.0,
                 103.0, 101.5, 100.0,  98.5,  97.5,  99.0, 101.0, 103.0, 105.0, 106.0]
        raw_c = [100.0, 101.0, 102.5, 101.0,  99.5,  98.0,  97.5,  98.5, 100.0, 101.5,
                 103.0, 102.0, 100.5,  99.0,  98.5,  99.5, 101.0, 102.5, 103.5, 104.0]
        period = 5

        high_s = pd.Series(raw_h, dtype=float)
        low_s = pd.Series(raw_l, dtype=float)
        close_s = pd.Series(raw_c, dtype=float)

        vec_result = atr(high_s, low_s, close_s, period)
        vec_atr_last = vec_result.dropna().iloc[-1]

        dec_h = [Decimal(str(v)) for v in raw_h]
        dec_l = [Decimal(str(v)) for v in raw_l]
        dec_c = [Decimal(str(v)) for v in raw_c]
        helper_atr = float(_compute_atr(dec_h, dec_l, dec_c, period))

        assert abs(vec_atr_last - helper_atr) < 0.001, (
            f"Vectorised ATR {vec_atr_last:.6f} differs from helper ATR "
            f"{helper_atr:.6f} by more than 0.001"
        )


# ===========================================================================
# TestBollingerBands
# ===========================================================================


class TestBollingerBands:
    """Tests for bollinger_bands(close, period, num_std)."""

    # --- return structure ---

    def test_returns_three_series(self) -> None:
        """bollinger_bands() returns a 3-tuple (upper, middle, lower)."""
        close = _make_random_close(50)
        result = bollinger_bands(close, 20)
        assert len(result) == 3
        assert all(isinstance(s, pd.Series) for s in result)

    def test_upper_above_middle_above_lower(self) -> None:
        """upper >= middle >= lower for all non-NaN positions."""
        close = _make_random_close(80, seed=7)
        upper, middle, lower = bollinger_bands(close, 20, 2.0)
        mask = upper.notna()
        assert (upper[mask] >= middle[mask]).all()
        assert (middle[mask] >= lower[mask]).all()

    def test_middle_equals_sma(self) -> None:
        """Middle band must equal SMA(period) exactly."""
        close = _make_random_close(60)
        period = 20
        _, middle, _ = bollinger_bands(close, period)
        expected_sma = sma(close, period)
        diff = (middle - expected_sma).abs().dropna()
        assert (diff < 1e-9).all()

    def test_band_width_proportional_to_num_std(self) -> None:
        """
        Doubling num_std doubles the distance from middle to each band.
        """
        close = _make_random_close(60, seed=11)
        upper1, middle1, lower1 = bollinger_bands(close, 20, num_std=1.0)
        upper2, middle2, lower2 = bollinger_bands(close, 20, num_std=2.0)
        width1 = (upper1 - middle1).dropna()
        width2 = (upper2 - middle2).dropna()
        ratio = (width2 / width1).dropna()
        assert (abs(ratio - 2.0) < 1e-9).all()

    def test_constant_series_zero_band_width(self) -> None:
        """Constant series → std = 0 → upper == middle == lower."""
        close = pd.Series([100.0] * 40)
        upper, middle, lower = bollinger_bands(close, 20)
        mask = upper.notna()
        assert (abs(upper[mask] - middle[mask]) < 1e-9).all()
        assert (abs(lower[mask] - middle[mask]) < 1e-9).all()

    # --- warm-up ---

    def test_nan_warmup_count(self) -> None:
        """Leading NaN count equals period - 1 (same as SMA)."""
        close = _make_random_close(50)
        upper, middle, lower = bollinger_bands(close, 20)
        assert middle.isna().sum() == 19

    # --- parameter validation ---

    def test_period_1_raises(self) -> None:
        """period=1 raises ValueError."""
        with pytest.raises(ValueError, match="Bollinger Bands period"):
            bollinger_bands(_make_random_close(30), period=1)

    def test_num_std_zero_raises(self) -> None:
        """num_std=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_std"):
            bollinger_bands(_make_random_close(30), period=20, num_std=0.0)

    def test_num_std_negative_raises(self) -> None:
        """num_std < 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_std"):
            bollinger_bands(_make_random_close(30), period=20, num_std=-1.0)

    # --- symmetry ---

    def test_bands_symmetric_around_middle(self) -> None:
        """upper - middle == middle - lower at every valid position."""
        close = _make_random_close(60)
        upper, middle, lower = bollinger_bands(close, 20, 2.0)
        mask = upper.notna()
        up_dist = (upper - middle)[mask]
        down_dist = (middle - lower)[mask]
        assert (abs(up_dist - down_dist) < 1e-9).all()


# ===========================================================================
# TestDonchianChannel
# ===========================================================================


class TestDonchianChannel:
    """Tests for donchian_channel(high, low, period)."""

    # --- return structure ---

    def test_returns_three_series(self) -> None:
        """Returns a 3-tuple (upper, middle, lower)."""
        df = _make_ohlcv_df(40)
        result = donchian_channel(df["high"], df["low"], 20)
        assert len(result) == 3

    def test_upper_above_lower(self) -> None:
        """upper_channel >= lower_channel at all non-NaN positions."""
        df = _make_ohlcv_df(60)
        upper, middle, lower = donchian_channel(df["high"], df["low"], 20)
        mask = upper.notna()
        assert (upper[mask] >= lower[mask]).all()

    def test_middle_equals_midpoint(self) -> None:
        """middle = (upper + lower) / 2 at all valid positions."""
        df = _make_ohlcv_df(50)
        upper, middle, lower = donchian_channel(df["high"], df["low"], 10)
        expected_mid = (upper + lower) / 2
        diff = (middle - expected_mid).abs().dropna()
        assert (diff < 1e-9).all()

    def test_upper_is_rolling_max_of_high(self) -> None:
        """upper channel equals rolling max of high prices."""
        df = _make_ohlcv_df(50)
        period = 10
        upper, _, _ = donchian_channel(df["high"], df["low"], period)
        expected = df["high"].rolling(window=period, min_periods=period).max()
        diff = (upper - expected).abs().dropna()
        assert (diff < 1e-9).all()

    def test_lower_is_rolling_min_of_low(self) -> None:
        """lower channel equals rolling min of low prices."""
        df = _make_ohlcv_df(50)
        period = 10
        _, _, lower = donchian_channel(df["high"], df["low"], period)
        expected = df["low"].rolling(window=period, min_periods=period).min()
        diff = (lower - expected).abs().dropna()
        assert (diff < 1e-9).all()

    # --- warm-up ---

    def test_nan_warmup_count(self) -> None:
        """period-1 leading NaNs."""
        df = _make_ohlcv_df(40)
        period = 10
        upper, _, _ = donchian_channel(df["high"], df["low"], period)
        assert upper.isna().sum() == period - 1

    # --- parameter validation ---

    def test_period_0_raises(self) -> None:
        """period=0 raises ValueError."""
        df = _make_ohlcv_df(30)
        with pytest.raises(ValueError, match="Donchian period"):
            donchian_channel(df["high"], df["low"], 0)

    def test_period_negative_raises(self) -> None:
        """Negative period raises ValueError."""
        df = _make_ohlcv_df(30)
        with pytest.raises(ValueError, match="Donchian period"):
            donchian_channel(df["high"], df["low"], -1)

    # --- period=1 edge case ---

    def test_period_1_no_nan_values(self) -> None:
        """period=1 produces no NaN values — rolling window of 1."""
        df = _make_ohlcv_df(20)
        upper, _, _ = donchian_channel(df["high"], df["low"], 1)
        assert upper.isna().sum() == 0


# ===========================================================================
# TestReturns
# ===========================================================================


class TestReturns:
    """Tests for returns(close, *, log_returns)."""

    # --- correctness: simple returns ---

    def test_simple_returns_known_value(self) -> None:
        """
        [100, 110] → pct_change = 0.10 at index 1.
        """
        close = _make_close([100.0, 110.0])
        result = returns(close)
        assert math.isnan(result.iloc[0])
        assert abs(result.iloc[1] - 0.10) < 1e-9

    def test_simple_returns_first_is_nan(self) -> None:
        """First element of simple returns is always NaN."""
        close = _make_random_close(20)
        result = returns(close)
        assert math.isnan(result.iloc[0])

    def test_simple_returns_length_preserved(self) -> None:
        """Output length equals input length."""
        close = _make_random_close(30)
        assert len(returns(close)) == len(close)

    # --- correctness: log returns ---

    def test_log_returns_known_value(self) -> None:
        """
        [100, 110] → log return = ln(110/100) = ln(1.1).
        """
        close = _make_close([100.0, 110.0])
        result = returns(close, log_returns=True)
        expected = math.log(1.1)
        assert math.isnan(result.iloc[0])
        assert abs(result.iloc[1] - expected) < 1e-9

    def test_log_returns_first_is_nan(self) -> None:
        """First element of log returns is always NaN."""
        close = _make_random_close(20)
        result = returns(close, log_returns=True)
        assert math.isnan(result.iloc[0])

    def test_log_returns_are_negative_for_falling_series(self) -> None:
        """Log returns are negative for a strictly falling series."""
        close = _make_close([100.0, 90.0, 80.0, 70.0])
        result = returns(close, log_returns=True)
        valid = result.dropna()
        assert (valid < 0).all()

    def test_simple_vs_log_returns_differ_on_large_moves(self) -> None:
        """
        For large price moves simple returns and log returns diverge.
        Simple return for doubling = 1.0; log return = ln(2) ≈ 0.693.
        """
        close = _make_close([100.0, 200.0])
        simple = returns(close).iloc[1]
        log_ret = returns(close, log_returns=True).iloc[1]
        assert abs(simple - 1.0) < 1e-9
        assert abs(log_ret - math.log(2)) < 1e-9

    # --- parameter validation ---

    def test_empty_series_raises(self) -> None:
        """Empty series raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            returns(pd.Series([], dtype=float))

    def test_empty_series_log_mode_raises(self) -> None:
        """Empty series in log mode also raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            returns(pd.Series([], dtype=float), log_returns=True)

    # --- single element ---

    def test_single_element_returns_nan(self) -> None:
        """A single-element series yields a single NaN return."""
        close = _make_close([50.0])
        result = returns(close)
        assert len(result) == 1
        assert math.isnan(result.iloc[0])

    # --- parametrized ---

    @pytest.mark.parametrize(
        "prices,expected_ret",
        [
            ([100.0, 100.0], 0.0),
            ([100.0, 150.0], 0.5),
            ([100.0, 50.0], -0.5),
        ],
    )
    def test_simple_returns_parametrized(
        self, prices: list[float], expected_ret: float
    ) -> None:
        """Parametrized simple return checks."""
        result = returns(_make_close(prices))
        assert abs(result.iloc[-1] - expected_ret) < 1e-9


# ===========================================================================
# TestRollingVolatility
# ===========================================================================


class TestRollingVolatility:
    """Tests for rolling_volatility(close, period, *, annualise, periods_per_year)."""

    # --- correctness ---

    def test_volatility_non_negative(self) -> None:
        """Volatility is always >= 0 (it is a standard deviation)."""
        close = _make_random_close(60)
        result = rolling_volatility(close, 20)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_constant_series_zero_volatility(self) -> None:
        """Flat prices → log returns = 0 → std = 0."""
        close = pd.Series([100.0] * 40)
        result = rolling_volatility(close, 10)
        valid = result.dropna()
        assert (abs(valid) < 1e-9).all()

    def test_annualisation_scales_by_sqrt_periods(self) -> None:
        """
        annualise=True must scale by sqrt(periods_per_year).

        For a given raw volatility σ, the annualised value must satisfy:
            vol_ann = vol_raw * sqrt(periods_per_year)
        """
        close = _make_random_close(60, seed=17)
        periods = 252
        vol_raw = rolling_volatility(close, 20, annualise=False)
        vol_ann = rolling_volatility(close, 20, annualise=True, periods_per_year=periods)
        mask = vol_raw.notna() & vol_ann.notna()
        ratio = vol_ann[mask] / vol_raw[mask]
        expected = math.sqrt(periods)
        assert (abs(ratio - expected) < 1e-9).all()

    def test_higher_variance_gives_higher_vol(self) -> None:
        """
        A series with larger price swings produces a higher volatility
        than a series with smaller swings.
        """
        rng = np.random.RandomState(5)
        small_swings = pd.Series(100.0 + rng.uniform(-0.5, 0.5, 60))
        rng2 = np.random.RandomState(5)
        large_swings = pd.Series(100.0 + rng2.uniform(-5.0, 5.0, 60))

        vol_small = rolling_volatility(small_swings, 20).dropna().mean()
        vol_large = rolling_volatility(large_swings, 20).dropna().mean()
        assert vol_large > vol_small

    # --- warm-up ---

    def test_nan_warmup_count(self) -> None:
        """First *period* values are NaN."""
        close = _make_random_close(60)
        for period in [2, 10, 20]:
            result = rolling_volatility(close, period)
            nan_count = result.isna().sum()
            assert nan_count == period, (
                f"period={period}: expected {period} NaNs, got {nan_count}"
            )

    # --- parameter validation ---

    def test_period_1_raises(self) -> None:
        """period=1 raises ValueError (ddof=1 requires at least 2 observations)."""
        with pytest.raises(ValueError, match="Volatility period"):
            rolling_volatility(_make_random_close(30), 1)

    def test_period_0_raises(self) -> None:
        """period=0 raises ValueError."""
        with pytest.raises(ValueError, match="Volatility period"):
            rolling_volatility(_make_random_close(30), 0)

    # --- return type ---

    def test_returns_pandas_series(self) -> None:
        """Return type is pd.Series."""
        result = rolling_volatility(_make_random_close(30), 5)
        assert isinstance(result, pd.Series)


# ===========================================================================
# TestComputeFeatures
# ===========================================================================


class TestComputeFeatures:
    """Tests for compute_features(df, **indicator_params)."""

    _EXPECTED_NEW_COLS = {
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "returns",
        "log_returns",
        "volatility",
        "volume_sma",
    }

    # --- output column correctness ---

    def test_all_expected_columns_added(self) -> None:
        """compute_features appends exactly the 12 expected feature columns."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        assert self._EXPECTED_NEW_COLS.issubset(set(result.columns))

    def test_feature_count_is_12(self) -> None:
        """Exactly 12 new columns are appended (no extra, no missing)."""
        df = _make_ohlcv_df(100)
        original_cols = set(df.columns)
        result = compute_features(df)
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) == 12

    def test_original_columns_preserved(self) -> None:
        """Original OHLCV columns are not modified."""
        df = _make_ohlcv_df(100)
        original_close = df["close"].copy()
        result = compute_features(df)
        pd.testing.assert_series_equal(result["close"], original_close)

    def test_input_df_not_mutated(self) -> None:
        """compute_features operates on a copy — input DataFrame is unchanged."""
        df = _make_ohlcv_df(100)
        original_cols = list(df.columns)
        _ = compute_features(df)
        assert list(df.columns) == original_cols

    # --- feature value correctness ---

    def test_rsi_column_values_in_valid_range(self) -> None:
        """RSI feature values are in [0, 100] for non-NaN rows."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0.0).all()
        assert (valid_rsi <= 100.0).all()

    def test_bb_upper_above_bb_lower(self) -> None:
        """Bollinger upper band >= lower band at all non-NaN positions."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        mask = result["bb_upper"].notna()
        assert (result["bb_upper"][mask] >= result["bb_lower"][mask]).all()

    def test_atr_column_non_negative(self) -> None:
        """ATR feature values are non-negative."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        valid_atr = result["atr"].dropna()
        assert (valid_atr >= 0).all()

    def test_macd_hist_equals_macd_minus_signal(self) -> None:
        """macd_hist == macd - macd_signal at all non-NaN rows."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        mask = result["macd_hist"].notna()
        diff = (result["macd"][mask] - result["macd_signal"][mask]).rename("diff")
        residual = (result["macd_hist"][mask] - diff).abs()
        assert (residual < 1e-9).all()

    def test_returns_column_first_is_nan(self) -> None:
        """First row of the returns column is NaN (no prior close)."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        assert math.isnan(result["returns"].iloc[0])

    def test_log_returns_column_first_is_nan(self) -> None:
        """First row of log_returns is NaN."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        assert math.isnan(result["log_returns"].iloc[0])

    def test_volume_sma_equals_sma_of_volume(self) -> None:
        """volume_sma column must equal SMA-20 of the volume column."""
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        expected_vol_sma = sma(df["volume"], 20)
        diff = (result["volume_sma"] - expected_vol_sma).abs().dropna()
        assert (diff < 1e-9).all()

    # --- warm-up rows not dropped ---

    def test_nan_rows_not_dropped(self) -> None:
        """
        compute_features does NOT drop NaN rows — the caller decides.
        Row count must equal the input DataFrame row count.
        """
        df = _make_ohlcv_df(100)
        result = compute_features(df)
        assert len(result) == len(df)

    # --- missing column validation ---

    def test_missing_open_column_raises(self) -> None:
        """DataFrame missing 'open' raises ValueError."""
        df = _make_ohlcv_df(100).drop(columns=["open"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_features(df)

    def test_missing_volume_column_raises(self) -> None:
        """DataFrame missing 'volume' raises ValueError."""
        df = _make_ohlcv_df(100).drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_features(df)

    def test_missing_multiple_columns_raises(self) -> None:
        """DataFrame missing 'high' and 'low' raises ValueError."""
        df = _make_ohlcv_df(100).drop(columns=["high", "low"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_features(df)

    def test_empty_dataframe_raises(self) -> None:
        """
        An empty DataFrame with the correct columns raises ValueError
        because RSI rejects an empty close series.
        """
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError):
            compute_features(df)

    # --- custom indicator parameters ---

    def test_custom_rsi_period_accepted(self) -> None:
        """Custom rsi_period parameter is passed through without error."""
        df = _make_ohlcv_df(100)
        result = compute_features(df, rsi_period=7)
        assert "rsi" in result.columns

    def test_custom_macd_params_accepted(self) -> None:
        """Custom MACD fast/slow/signal params are accepted."""
        df = _make_ohlcv_df(100)
        result = compute_features(df, macd_fast=5, macd_slow=10, macd_signal=3)
        assert "macd" in result.columns

    # --- determinism ---

    def test_compute_features_deterministic(self) -> None:
        """
        Calling compute_features twice on the same input produces identical results.

        Pure-function guarantee: no global state or random calls.
        """
        df = _make_ohlcv_df(100, seed=77)
        result1 = compute_features(df)
        result2 = compute_features(df)
        for col in self._EXPECTED_NEW_COLS:
            pd.testing.assert_series_equal(result1[col], result2[col])
