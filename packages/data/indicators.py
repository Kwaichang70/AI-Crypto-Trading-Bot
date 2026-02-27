"""
packages/data/indicators.py
----------------------------
Technical indicator functions for strategy signal generation and ML features.

Design rules
------------
- All functions are pure (no side effects, no global state).
- All functions accept pandas Series or NumPy arrays and return the same.
- NaN is used for the warm-up period (e.g. RSI returns NaN for the first
  ``period`` values).
- Functions raise ``ValueError`` for invalid parameters; callers must handle.
- This module is the only place indicator logic is implemented — strategies
  import from here, never reimplement their own indicator calculations.

All implementations here are stubs that raise ``NotImplementedError``.
Full implementations land in the subsequent sprint.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

__all__ = [
    "rsi",
    "macd",
    "atr",
    "ema",
    "sma",
    "bollinger_bands",
    "donchian_channel",
    "returns",
    "rolling_volatility",
    "compute_features",
]


def rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Uses Wilder's smoothed average (EMA with alpha = 1/period).

    Parameters
    ----------
    close:
        Series of closing prices. Must be non-empty and contain finite values.
    period:
        Look-back window. Typically 14. Must be >= 2.

    Returns
    -------
    pd.Series:
        RSI values in [0, 100]. NaN for the first ``period`` elements.

    Raises
    ------
    ValueError:
        If ``period < 2`` or ``close`` is empty.
    """
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}")
    if close.empty:
        raise ValueError("close series must not be empty")
    raise NotImplementedError("RSI implementation pending sprint 2")


def ema(
    series: pd.Series,
    period: int,
    *,
    adjust: bool = False,
) -> pd.Series:
    """
    Exponential Moving Average.

    Parameters
    ----------
    series:
        Input price series.
    period:
        Span for the EMA calculation. Alpha = 2 / (period + 1).
    adjust:
        When True, uses adjusted EMA (pandas default). When False,
        uses the recursive Wilder-style EMA (standard for RSI/ATR).

    Returns
    -------
    pd.Series:
        EMA values. NaN for the first ``period - 1`` elements.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")
    raise NotImplementedError("EMA implementation pending sprint 2")


def sma(
    series: pd.Series,
    period: int,
) -> pd.Series:
    """
    Simple Moving Average.

    Parameters
    ----------
    series:
        Input price series.
    period:
        Rolling window size. Must be >= 1.

    Returns
    -------
    pd.Series:
        SMA values. NaN for the first ``period - 1`` elements.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}")
    raise NotImplementedError("SMA implementation pending sprint 2")


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    close:
        Series of closing prices.
    fast_period:
        Fast EMA period. Default 12.
    slow_period:
        Slow EMA period. Default 26. Must be > ``fast_period``.
    signal_period:
        Signal line EMA period. Default 9.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]:
        ``(macd_line, signal_line, histogram)``
        - ``macd_line`` = fast_ema - slow_ema
        - ``signal_line`` = EMA(macd_line, signal_period)
        - ``histogram`` = macd_line - signal_line

    Raises
    ------
    ValueError:
        If ``slow_period <= fast_period``.
    """
    if slow_period <= fast_period:
        raise ValueError(
            f"slow_period ({slow_period}) must be > fast_period ({fast_period})"
        )
    raise NotImplementedError("MACD implementation pending sprint 2")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR) using Wilder's smoothing.

    True Range is: max(high - low, |high - prev_close|, |low - prev_close|)

    Parameters
    ----------
    high:
        Series of high prices.
    low:
        Series of low prices.
    close:
        Series of close prices.
    period:
        ATR smoothing period. Default 14.

    Returns
    -------
    pd.Series:
        ATR values in price units. NaN for the first ``period`` elements.
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}")
    raise NotImplementedError("ATR implementation pending sprint 2")


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Parameters
    ----------
    close:
        Series of closing prices.
    period:
        SMA look-back window.
    num_std:
        Number of standard deviations for band width.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]:
        ``(upper_band, middle_band, lower_band)``
    """
    if period < 2:
        raise ValueError(f"Bollinger Bands period must be >= 2, got {period}")
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")
    raise NotImplementedError("Bollinger Bands implementation pending sprint 2")


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channel (price breakout channel).

    Parameters
    ----------
    high:
        Series of high prices.
    low:
        Series of low prices.
    period:
        Look-back window for the rolling max/min.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]:
        ``(upper_channel, middle_channel, lower_channel)``
        - ``upper_channel`` = rolling max of high
        - ``lower_channel`` = rolling min of low
        - ``middle_channel`` = midpoint
    """
    if period < 1:
        raise ValueError(f"Donchian period must be >= 1, got {period}")
    raise NotImplementedError("Donchian Channel implementation pending sprint 2")


def returns(
    close: pd.Series,
    *,
    log_returns: bool = False,
) -> pd.Series:
    """
    Compute price returns.

    Parameters
    ----------
    close:
        Series of closing prices.
    log_returns:
        When True, compute log returns: ln(p_t / p_{t-1}).
        When False, compute simple returns: (p_t - p_{t-1}) / p_{t-1}.

    Returns
    -------
    pd.Series:
        Return series. First element is NaN.
    """
    raise NotImplementedError("Returns implementation pending sprint 2")


def rolling_volatility(
    close: pd.Series,
    period: int = 20,
    *,
    annualise: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rolling realised volatility (standard deviation of log returns).

    Parameters
    ----------
    close:
        Series of closing prices.
    period:
        Rolling window for standard deviation.
    annualise:
        When True, multiply by ``sqrt(periods_per_year)``.
    periods_per_year:
        Used when ``annualise=True``. Default 252 (daily bars).
        For hourly bars use 8760; for 5-minute bars use 105120.

    Returns
    -------
    pd.Series:
        Volatility values. NaN for the first ``period`` elements.
    """
    if period < 2:
        raise ValueError(f"Volatility period must be >= 2, got {period}")
    raise NotImplementedError("Rolling volatility implementation pending sprint 2")


def compute_features(
    df: pd.DataFrame,
    *,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
    bb_period: int = 20,
    vol_period: int = 20,
) -> pd.DataFrame:
    """
    Compute the full ML feature set for a DataFrame of OHLCV bars.

    Expected input columns: ``open``, ``high``, ``low``, ``close``, ``volume``.

    Output columns appended to the input DataFrame:
    - ``rsi``
    - ``macd``, ``macd_signal``, ``macd_hist``
    - ``atr``
    - ``bb_upper``, ``bb_mid``, ``bb_lower``
    - ``returns``
    - ``log_returns``
    - ``volatility``
    - ``volume_sma`` (20-bar SMA of volume)

    Parameters
    ----------
    df:
        OHLCV DataFrame with a DatetimeIndex (UTC).
    rsi_period, macd_fast, macd_slow, macd_signal, atr_period, bb_period, vol_period:
        Indicator parameters.

    Returns
    -------
    pd.DataFrame:
        Original DataFrame with feature columns appended.
        Rows with NaN feature values (warm-up period) are NOT dropped —
        the caller decides whether to drop or forward-fill.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    raise NotImplementedError("Feature computation implementation pending sprint 2")
