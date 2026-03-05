"""
tests/unit/test_ml_features.py
-------------------------------
Unit tests for the shared ML feature builder.

Module under test
-----------------
packages/data/ml_features.py

Test coverage
-------------
1. TestFeatureSchema         -- FEATURE_NAMES constant shape and contents
2. TestBuildFeatureVector    -- build_feature_vector_from_bars(): length, guard clauses,
                               RSI normalisation, volatility ddof=0, SMA ratio fallback
3. TestBuildFeatureMatrix    -- build_feature_matrix(): columns, missing-columns guard,
                               empty-DataFrame guard
4. TestNumericalParity       -- bars path vs DataFrame path agree to within 1e-9 for
                               identically constructed data

Design notes
------------
- All tests are synchronous (no async path in ml_features.py).
- pytest.approx is used throughout for float comparisons.
- The bar factory (_make_bars) generates strictly monotone-incrementing closes to
  produce a well-behaved, non-degenerate dataset.  A minimum of 110 bars is used for
  tests that exercise all warmup windows (max window = 100 bars for SMA-100).
- OHLCVBar high/low are set to close +/- 1.0 so that the high_low_range feature
  equals 2.0 / close on the last bar (used in the vector-length test only; exact
  value is not asserted because the matrix path may see NaN in warmup rows).
- For the SMA ratio fallback test, a single-bar sequence is supplied.  With only 1
  close, _sma_float(closes, 10) returns 0.0, so sma_ratio_10_50 must fall back to 1.0.
- For the parity test, the same numeric data is presented as both an OHLCVBar sequence
  (bars path) and as a plain OHLCV DataFrame (matrix path).  The last row of the
  matrix (after NaN warmup rows) is compared against the vector built from the same bars.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from data.ml_features import (
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_vector_from_bars,
)

# ---------------------------------------------------------------------------
# Bar factory helper
# ---------------------------------------------------------------------------


def _make_bars(
    n: int,
    base_close: float = 100.0,
    increment: float = 0.5,
) -> list[OHLCVBar]:
    """Create n synthetic OHLCVBar instances with incrementing closes.

    Closes range from base_close to base_close + (n-1) * increment.
    High = close + 1.0, Low = close - 1.0 so the OHLCV consistency constraint
    (low <= open/close <= high) is satisfied and high_low_range is computable.
    Volume increases linearly for variety.
    """
    bars: list[OHLCVBar] = []
    for i in range(n):
        close = base_close + i * increment
        bars.append(
            OHLCVBar(
                symbol="BTC/USDT",
                timeframe=TimeFrame.ONE_HOUR,
                timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=Decimal(str(close - 0.1)),
                high=Decimal(str(close + 1.0)),
                low=Decimal(str(close - 1.0)),
                close=Decimal(str(close)),
                volume=Decimal(str(1000.0 + i * 10)),
            )
        )
    return bars


def _bars_to_df(bars: list[OHLCVBar]) -> pd.DataFrame:
    """Convert an OHLCVBar list to a plain OHLCV DataFrame (float columns)."""
    return pd.DataFrame(
        [
            {
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
            }
            for b in bars
        ]
    )


# ---------------------------------------------------------------------------
# Class 1: TestFeatureSchema
# ---------------------------------------------------------------------------


class TestFeatureSchema:
    """Verify the public FEATURE_NAMES constant is correct and complete."""

    def test_feature_names_length(self) -> None:
        """FEATURE_NAMES must contain exactly 10 entries.

        The feature schema is defined as a 10-element vector; any drift from
        this count would break downstream model loading (shape mismatch).
        """
        assert len(FEATURE_NAMES) == 10, (
            f"FEATURE_NAMES must have 10 entries, got {len(FEATURE_NAMES)}"
        )

    def test_feature_names_contents(self) -> None:
        """FEATURE_NAMES must contain all 10 expected column names in order.

        The ordering is part of the public API: consumers that index by position
        rely on the schema never changing silently.
        """
        expected = [
            "log_return_1",
            "log_return_5",
            "log_return_10",
            "volatility_10",
            "volatility_20",
            "rsi_14",
            "sma_ratio_10_50",
            "sma_ratio_20_100",
            "volume_ratio_10",
            "high_low_range",
        ]
        assert FEATURE_NAMES == expected, (
            f"FEATURE_NAMES order/content mismatch.\n"
            f"Expected: {expected}\n"
            f"Got:      {FEATURE_NAMES}"
        )


# ---------------------------------------------------------------------------
# Class 2: TestBuildFeatureVector
# ---------------------------------------------------------------------------


class TestBuildFeatureVector:
    """Verify build_feature_vector_from_bars() output shape, guard clauses,
    and numerical properties."""

    def test_feature_vector_length(self) -> None:
        """build_feature_vector_from_bars returns exactly 10 elements.

        A minimum of 110 bars is supplied so all warmup windows are satisfied
        (SMA-100 requires 100 bars; we add 10 extra for headroom).  The result
        must always be a list of exactly 10 floats regardless of input size.
        """
        bars = _make_bars(110)
        vector = build_feature_vector_from_bars(bars)

        assert len(vector) == 10, (
            f"Expected feature vector of length 10, got {len(vector)}"
        )

    def test_feature_vector_empty_bars_raises(self) -> None:
        """An empty bar list must raise ValueError immediately.

        The function requires at least one bar to compute any feature.  Raising
        ValueError (not returning a zero vector) is the correct contract so
        callers can detect bad input at the boundary.
        """
        with pytest.raises(ValueError, match="bars must not be empty"):
            build_feature_vector_from_bars([])

    def test_rsi_normalised(self) -> None:
        """The rsi_14 feature (index 5) must be in [0.0, 1.0].

        Wilder RSI is in [0, 100].  The feature builder divides by 100.0 to
        normalise to [0, 1].  We test with 110 bars (above the 15-bar warmup
        needed for RSI-14) so the raw value is not the 50.0 neutral fallback.
        """
        bars = _make_bars(110)
        vector = build_feature_vector_from_bars(bars)

        rsi_normalised = vector[5]
        assert 0.0 <= rsi_normalised <= 1.0, (
            f"rsi_14 feature must be in [0, 1], got {rsi_normalised}"
        )

    def test_rsi_below_one(self) -> None:
        """The rsi_14 feature must be strictly below 1.0 for normal trending data.

        100/100 = 1.0 only when there are zero losses (all gains).  Our
        incrementing bars have perfectly monotone closes: every bar is an up bar,
        so avg_loss == 0 and RSI == 100.0, normalised to 1.0.  We therefore
        verify the value is exactly 1.0 for a fully trending sequence.
        """
        bars = _make_bars(110, increment=1.0)  # strictly increasing
        vector = build_feature_vector_from_bars(bars)

        rsi_normalised = vector[5]
        # Strictly increasing closes => all gains, avg_loss=0 => RSI=100 => normalised=1.0
        assert rsi_normalised == pytest.approx(1.0, abs=1e-9), (
            f"Fully trending (all-up) bars should produce rsi_14=1.0, got {rsi_normalised}"
        )

    def test_volatility_ddof_zero(self) -> None:
        """volatility_10 uses population std (ddof=0), not sample std (ddof=1).

        We compute the expected population std manually and verify the feature
        matches it.  If ddof=1 were used instead, the result would differ by
        a factor of sqrt(9/10) for a 10-element window.
        """
        # Use 110 bars so the volatility_10 window is fully populated
        bars = _make_bars(110)

        closes = [float(b.close) for b in bars]
        log_closes = [math.log(c) for c in closes]
        # log_returns from the last 11 log_closes (to get 10 returns)
        log_returns_last_11 = [
            log_closes[i] - log_closes[i - 1]
            for i in range(len(log_closes) - 10, len(log_closes))
        ]

        # Population std (ddof=0)
        mean_lr = sum(log_returns_last_11) / 10
        variance = sum((x - mean_lr) ** 2 for x in log_returns_last_11) / 10
        expected_vol_10 = math.sqrt(variance)

        vector = build_feature_vector_from_bars(bars)
        vol_10 = vector[3]  # index 3 = volatility_10

        assert vol_10 == pytest.approx(expected_vol_10, rel=1e-9), (
            f"volatility_10 must use ddof=0 (population std).\n"
            f"Expected: {expected_vol_10}\n"
            f"Got:      {vol_10}"
        )

    def test_sma_ratio_fallback(self) -> None:
        """When SMA denominators are 0.0 (insufficient data), ratios fall back to 1.0.

        With a single bar, _sma_float(closes, 10) = 0.0 and _sma_float(closes, 50) = 0.0,
        so sma_ratio_10_50 and sma_ratio_20_100 must both equal 1.0.  Returning 1.0
        (neutral ratio) is the correct contract: the model sees "no difference" rather
        than NaN or division-by-zero error.
        """
        bars = _make_bars(1)  # single bar — all SMA windows return 0.0
        vector = build_feature_vector_from_bars(bars)

        sma_ratio_10_50 = vector[6]
        sma_ratio_20_100 = vector[7]

        assert sma_ratio_10_50 == pytest.approx(1.0), (
            f"sma_ratio_10_50 must fall back to 1.0 when SMA(50)=0, got {sma_ratio_10_50}"
        )
        assert sma_ratio_20_100 == pytest.approx(1.0), (
            f"sma_ratio_20_100 must fall back to 1.0 when SMA(100)=0, got {sma_ratio_20_100}"
        )

    def test_feature_vector_all_finite(self) -> None:
        """All 10 features must be finite (not NaN or Inf) for a well-populated sequence.

        With 110 bars all warmup windows are fully satisfied; no feature should
        return a degenerate NaN or Inf value.
        """
        bars = _make_bars(110)
        vector = build_feature_vector_from_bars(bars)

        for i, (name, value) in enumerate(zip(FEATURE_NAMES, vector)):
            assert math.isfinite(value), (
                f"Feature {i} ({name}) is not finite: {value}"
            )


# ---------------------------------------------------------------------------
# Class 3: TestBuildFeatureMatrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """Verify build_feature_matrix() output shape and guard clauses."""

    def test_feature_matrix_columns(self) -> None:
        """build_feature_matrix returns a DataFrame with exactly 10 named columns.

        The column names must exactly match FEATURE_NAMES in order.  Downstream
        ML code indexes by column name, so any mismatch breaks training/inference.
        """
        bars = _make_bars(110)
        df = _bars_to_df(bars)
        result = build_feature_matrix(df)

        assert list(result.columns) == FEATURE_NAMES, (
            f"Feature matrix columns mismatch.\n"
            f"Expected: {FEATURE_NAMES}\n"
            f"Got:      {list(result.columns)}"
        )
        assert result.shape[1] == 10, (
            f"Feature matrix must have 10 columns, got {result.shape[1]}"
        )

    def test_feature_matrix_row_count(self) -> None:
        """build_feature_matrix returns the same number of rows as the input DataFrame.

        Warmup rows are present but contain NaN values; they are not dropped.
        The caller (ModelTrainer.prepare_dataset) handles NaN removal with the
        valid_mask filter.
        """
        n = 110
        bars = _make_bars(n)
        df = _bars_to_df(bars)
        result = build_feature_matrix(df)

        assert len(result) == n, (
            f"Feature matrix must have {n} rows (same as input), got {len(result)}"
        )

    def test_feature_matrix_missing_columns_raises(self) -> None:
        """A DataFrame missing any required column must raise ValueError.

        Required columns are: close, volume, high, low.  Omitting any one of
        them must raise ValueError with a message identifying the missing column.
        """
        df = pd.DataFrame({"close": [100.0], "high": [101.0], "low": [99.0]})
        # 'volume' column is absent

        with pytest.raises(ValueError, match="missing required columns"):
            build_feature_matrix(df)

    def test_feature_matrix_empty_raises(self) -> None:
        """An empty DataFrame must raise ValueError.

        Passing an empty DataFrame is a programming error at the call site; the
        function must raise rather than returning an empty feature matrix.
        """
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        with pytest.raises(ValueError, match="must not be empty"):
            build_feature_matrix(df)

    def test_feature_matrix_warmup_rows_are_nan(self) -> None:
        """The first warmup rows (index 0) must contain NaN for the log_return_1 feature.

        The log_return_1 feature requires at least 2 bars (diff(1)).  The first
        row must therefore be NaN because there is no predecessor bar.
        """
        bars = _make_bars(10)
        df = _bars_to_df(bars)
        result = build_feature_matrix(df)

        assert pd.isna(result["log_return_1"].iloc[0]), (
            "First row of log_return_1 must be NaN (no predecessor bar for diff(1))"
        )

    def test_feature_matrix_rsi_normalised(self) -> None:
        """rsi_14 column values in the non-NaN rows must be in [0.0, 1.0].

        After the 14-bar RSI warmup period, all valid rsi_14 values must be in
        [0, 1].  We verify the max and min of non-NaN entries.
        """
        bars = _make_bars(110)
        df = _bars_to_df(bars)
        result = build_feature_matrix(df)

        rsi_col = result["rsi_14"].dropna()
        assert rsi_col.max() <= 1.0 + 1e-12, (
            f"rsi_14 must be <= 1.0, got max={rsi_col.max()}"
        )
        assert rsi_col.min() >= 0.0 - 1e-12, (
            f"rsi_14 must be >= 0.0, got min={rsi_col.min()}"
        )


# ---------------------------------------------------------------------------
# Class 4: TestNumericalParity
# ---------------------------------------------------------------------------


class TestNumericalParity:
    """Verify that the bars path and the DataFrame path are numerically identical.

    Both build_feature_vector_from_bars (pure-Python) and build_feature_matrix
    (vectorised Pandas/NumPy) must produce the same numbers for the same input
    data.  The last fully-warmed-up row of the matrix is compared against the
    vector computed from the same bars, with tolerance 1e-9 (relative).
    """

    def test_numerical_parity(self) -> None:
        """bars path and DataFrame path produce identical values (within 1e-9).

        We use 110 bars so all windows (max: SMA-100) are fully populated.
        The comparison is done on the last row of the feature matrix (index -1),
        which corresponds to the same bar used by build_feature_vector_from_bars.

        Note: the SMA-ratio features may still be NaN in the matrix for early
        rows, but the last row of a 110-bar sequence is fully warmed up for all
        features.
        """
        bars = _make_bars(110)
        df = _bars_to_df(bars)

        # bars path — uses pure-Python helpers
        vector = build_feature_vector_from_bars(bars)

        # DataFrame path — uses vectorised Pandas/NumPy
        matrix = build_feature_matrix(df)
        last_row = matrix.iloc[-1]

        for i, name in enumerate(FEATURE_NAMES):
            vec_val = vector[i]
            mat_val = float(last_row[name])

            if math.isnan(mat_val):
                # If the matrix produces NaN for this feature at the last row,
                # the bars path should also return 0.0 (warmup fallback).
                # This should not happen for 110 bars, but we guard defensively.
                assert vec_val == pytest.approx(0.0, abs=1e-12), (
                    f"Feature {name}: matrix returned NaN but vector returned {vec_val!r}"
                )
            else:
                assert vec_val == pytest.approx(mat_val, rel=1e-9, abs=1e-12), (
                    f"Parity failure for feature {name} (index {i}):\n"
                    f"  bars path:      {vec_val}\n"
                    f"  DataFrame path: {mat_val}\n"
                    f"  diff:           {abs(vec_val - mat_val)}"
                )

    def test_numerical_parity_different_seed(self) -> None:
        """Parity holds for a different base_close value (non-trivial RSI / volatility).

        Using base_close=200.0 and increment=0.3 exercises different SMA ratios
        and RSI values than the default 100.0/0.5 setup.
        """
        bars = _make_bars(110, base_close=200.0, increment=0.3)
        df = _bars_to_df(bars)

        vector = build_feature_vector_from_bars(bars)
        matrix = build_feature_matrix(df)
        last_row = matrix.iloc[-1]

        for i, name in enumerate(FEATURE_NAMES):
            vec_val = vector[i]
            mat_val = float(last_row[name])

            if math.isnan(mat_val):
                assert vec_val == pytest.approx(0.0, abs=1e-12), (
                    f"Feature {name}: matrix NaN but vector={vec_val!r}"
                )
            else:
                assert vec_val == pytest.approx(mat_val, rel=1e-9, abs=1e-12), (
                    f"Parity failure for feature {name}: "
                    f"bars={vec_val}, df={mat_val}"
                )
