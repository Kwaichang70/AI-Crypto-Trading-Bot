"""
packages/data/ml_features.py
------------------------------
Shared ML feature builder for the AI Crypto Trading Bot.

Feature schema (10 elements, fixed order)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Index  Column name           Description
    -----  --------------------  ----------------------------------------
    0      log_return_1          1-bar log return
    1      log_return_5          5-bar cumulative log return
    2      log_return_10         10-bar cumulative log return
    3      volatility_10         10-bar rolling population std of log returns
    4      volatility_20         20-bar rolling population std of log returns
    5      rsi_14                14-bar Wilder RSI normalised to [0, 1]
    6      sma_ratio_10_50       SMA(10) / SMA(50) close ratio
    7      sma_ratio_20_100      SMA(20) / SMA(100) close ratio
    8      volume_ratio_10       current volume / SMA(volume, 10)
    9      high_low_range        (high - low) / close of the current bar

Design notes
~~~~~~~~~~~~
- ``build_feature_vector_from_bars`` uses pure-Python helpers identical to
  the original ``ModelStrategy._build_feature_vector`` implementation.
- ``build_feature_matrix`` uses vectorised Pandas/NumPy operations.
- Both paths use ddof=0 (population std) for volatility features.
- RSI uses Wilder SMA-seeded smoothing normalised to [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import structlog

from common.models import OHLCVBar

__all__ = [
    "FEATURE_NAMES",
    "build_feature_vector_from_bars",
    "build_feature_matrix",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public feature name registry
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
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

# ---------------------------------------------------------------------------
# Pure-Python helpers (bars path) — identical to model_strategy.py originals
# ---------------------------------------------------------------------------


def _safe_log(value: float) -> float:
    """Return natural log, returning 0.0 for non-positive values."""
    if value <= 0.0:
        return 0.0
    return math.log(value)


def _sma_float(values: Sequence[float], period: int) -> float:
    """Simple moving average over the last *period* values. Returns 0.0 if insufficient data."""
    if len(values) < period:
        return 0.0
    window = values[-period:]
    return sum(window) / period


def _wilder_rsi(closes: Sequence[float], period: int = 14) -> float:
    """Compute Wilder RSI. Returns 50.0 (neutral) if insufficient data."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(0.0, d) for d in deltas[:period]]
    losses = [max(0.0, -d) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for delta in deltas[period:]:
        gain = max(0.0, delta)
        loss = max(0.0, -delta)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _rolling_std_population(values: Sequence[float], window: int) -> float:
    """Population standard deviation (ddof=0) over the last *window* values."""
    if len(values) < window:
        return 0.0
    subset = list(values[-window:])
    mean = sum(subset) / window
    variance = sum((x - mean) ** 2 for x in subset) / window
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Public API — bars path (runtime / strategy use)
# ---------------------------------------------------------------------------


def build_feature_vector_from_bars(bars: Sequence[OHLCVBar]) -> list[float]:
    """Build a 10-element feature vector from an OHLCVBar sequence.

    This is the runtime path used by ModelStrategy. Numerically identical
    to the original ModelStrategy._build_feature_vector implementation.
    """
    if not bars:
        raise ValueError("bars must not be empty")

    closes = [float(bar.close) for bar in bars]
    volumes = [float(bar.volume) for bar in bars]
    current_bar = bars[-1]

    log_closes = [_safe_log(float(bar.close)) for bar in bars]
    log_return_1 = log_closes[-1] - log_closes[-2] if len(log_closes) >= 2 else 0.0
    log_return_5 = log_closes[-1] - log_closes[-6] if len(log_closes) >= 6 else 0.0
    log_return_10 = log_closes[-1] - log_closes[-11] if len(log_closes) >= 11 else 0.0

    log_returns = [
        log_closes[i] - log_closes[i - 1]
        for i in range(1, len(log_closes))
    ]
    volatility_10 = _rolling_std_population(log_returns, 10)
    volatility_20 = _rolling_std_population(log_returns, 20)

    rsi_14 = _wilder_rsi(closes, period=14) / 100.0

    sma_10 = _sma_float(closes, 10)
    sma_20 = _sma_float(closes, 20)
    sma_50 = _sma_float(closes, 50)
    sma_100 = _sma_float(closes, 100)

    sma_ratio_10_50 = (sma_10 / sma_50) if sma_50 != 0.0 else 1.0
    sma_ratio_20_100 = (sma_20 / sma_100) if sma_100 != 0.0 else 1.0

    sma_vol_10 = _sma_float(volumes, 10)
    volume_ratio_10 = (
        float(current_bar.volume) / sma_vol_10
        if sma_vol_10 > 0.0
        else 1.0
    )

    close_f = float(current_bar.close)
    high_low_range = (
        (float(current_bar.high) - float(current_bar.low)) / close_f
        if close_f > 0.0
        else 0.0
    )

    return [
        log_return_1,
        log_return_5,
        log_return_10,
        volatility_10,
        volatility_20,
        rsi_14,
        sma_ratio_10_50,
        sma_ratio_20_100,
        volume_ratio_10,
        high_low_range,
    ]


# ---------------------------------------------------------------------------
# Public API — DataFrame path (ML training pipeline)
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from an OHLCV DataFrame.

    Uses vectorised operations. Numerically consistent with
    build_feature_vector_from_bars: volatility uses ddof=0 (population std),
    RSI uses Wilder SMA-seeded recursion normalised to [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Required columns: close, volume, high, low.

    Returns
    -------
    pd.DataFrame
        10 feature columns. NaN in warmup rows.
    """
    required_cols = {"close", "volume", "high", "low"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame must not be empty")

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Log returns — guard against non-positive closes
    log_close = np.log(close.where(close > 0.0, np.nan))
    log_return_1 = log_close.diff(1)
    log_return_5 = log_close.diff(5)
    log_return_10 = log_close.diff(10)

    # Volatility: population std (ddof=0) — matches ModelStrategy
    log_ret_1bar = log_close.diff(1)
    volatility_10 = log_ret_1bar.rolling(window=10, min_periods=10).std(ddof=0)
    volatility_20 = log_ret_1bar.rolling(window=20, min_periods=20).std(ddof=0)

    # Wilder RSI normalised to [0, 1]
    rsi_14_raw = _wilder_rsi_vectorized(close, period=14)
    rsi_14 = rsi_14_raw / 100.0

    # SMA ratios with 1.0 fallback
    sma_10 = close.rolling(window=10, min_periods=10).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    sma_50 = close.rolling(window=50, min_periods=50).mean()
    sma_100 = close.rolling(window=100, min_periods=100).mean()

    sma_ratio_10_50 = _safe_ratio(sma_10, sma_50, fallback=1.0)
    sma_ratio_20_100 = _safe_ratio(sma_20, sma_100, fallback=1.0)

    # Volume ratio with 1.0 fallback
    sma_vol_10 = volume.rolling(window=10, min_periods=10).mean()
    volume_ratio_10 = _safe_ratio(volume, sma_vol_10, fallback=1.0)

    # High-low range
    high_low_range = (high - low) / close.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "log_return_1": log_return_1,
            "log_return_5": log_return_5,
            "log_return_10": log_return_10,
            "volatility_10": volatility_10,
            "volatility_20": volatility_20,
            "rsi_14": rsi_14,
            "sma_ratio_10_50": sma_ratio_10_50,
            "sma_ratio_20_100": sma_ratio_20_100,
            "volume_ratio_10": volume_ratio_10,
            "high_low_range": high_low_range,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Internal helpers for the vectorized path
# ---------------------------------------------------------------------------


def _wilder_rsi_vectorized(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised Wilder RSI matching _wilder_rsi exactly."""
    n = len(close)
    if n < period + 1:
        return pd.Series(np.nan, index=close.index)

    close_arr = close.to_numpy(dtype=float, copy=True)
    deltas = np.empty(n, dtype=float)
    deltas[0] = np.nan
    deltas[1:] = np.diff(close_arr)

    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    avg_gain[period] = gains[1: period + 1].mean()
    avg_loss[period] = losses[1: period + 1].mean()

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_arr = 100.0 - (100.0 / (1.0 + rs))

    result = pd.Series(rsi_arr, index=close.index)
    result.iloc[:period] = np.nan
    return result


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    fallback: float = 1.0,
) -> pd.Series:
    """Element-wise division with fallback for zero/NaN denominators."""
    denom_bad = denominator.isna() | (denominator == 0.0)
    ratio: Any = numerator / denominator.replace(0.0, np.nan)
    fallback_mask = denom_bad & numerator.notna()
    ratio = ratio.where(~fallback_mask, other=fallback)
    return ratio
