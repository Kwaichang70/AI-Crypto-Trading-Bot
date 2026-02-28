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

Vectorized implementations use pandas and NumPy for efficient batch
computation. RSI and ATR use SMA-seeded Wilder's smoothing to match
the canonical formula used by TradingView, TA-Lib, and the inline
strategy helpers in ``packages/trading/strategies/``.
"""

from __future__ import annotations

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
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing with SMA seed (matches _compute_rsi in strategies).
    # Step 1: SMA of first `period` deltas as the initial avg_gain / avg_loss.
    # Step 2: Recursive Wilder: avg = (prev_avg * (period-1) + current) / period.
    n = len(close)
    avg_gain_arr = np.full(n, np.nan)
    avg_loss_arr = np.full(n, np.nan)

    # SMA seed at index `period` (deltas start at index 1, so first period deltas are [1..period])
    avg_gain_arr[period] = gain.iloc[1 : period + 1].mean()
    avg_loss_arr[period] = loss.iloc[1 : period + 1].mean()

    # Wilder recursion forward
    for i in range(period + 1, n):
        avg_gain_arr[i] = (avg_gain_arr[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss_arr[i] = (avg_loss_arr[i - 1] * (period - 1) + loss.iloc[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain_arr / avg_loss_arr
        result_arr = 100.0 - (100.0 / (1.0 + rs))

    result = pd.Series(result_arr, index=close.index)
    # NaN for warmup; all-gains (avg_loss=0) → rs=inf → RSI=100 (correct via float math)
    result.iloc[:period] = np.nan
    return result


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
    result = series.ewm(span=period, adjust=adjust).mean()
    result.iloc[:period - 1] = np.nan
    return result


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
    return series.rolling(window=period, min_periods=period).mean()


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
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    # Mask warmup: slow EMA needs slow_period bars to stabilise
    macd_line.iloc[: slow_period - 1] = np.nan
    signal_line.iloc[: slow_period - 1] = np.nan
    histogram.iloc[: slow_period - 1] = np.nan
    return macd_line, signal_line, histogram


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
    if period < 2:
        raise ValueError(f"ATR period must be >= 2, got {period}")
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing with SMA seed (matches _compute_atr in strategies).
    # TR values start at index 1 (index 0 has no prev_close).
    n = len(close)
    atr_arr = np.full(n, np.nan)

    # SMA seed: average of first `period` TR values (indices 1..period)
    atr_arr[period] = true_range.iloc[1 : period + 1].mean()

    # Wilder recursion forward
    for i in range(period + 1, n):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + true_range.iloc[i]) / period

    return pd.Series(atr_arr, index=close.index)


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
    middle = sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


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
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


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

    Raises
    ------
    ValueError:
        If ``close`` is empty.
    """
    if close.empty:
        raise ValueError("close series must not be empty")
    if log_returns:
        return np.log(close / close.shift(1))
    return close.pct_change()


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
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window=period, min_periods=period).std(ddof=1)
    if annualise:
        vol = vol * np.sqrt(periods_per_year)
    vol.iloc[:period] = np.nan
    return vol


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
    result = df.copy()
    result["rsi"] = rsi(df["close"], period=rsi_period)
    macd_line, macd_sig, macd_h = macd(df["close"], macd_fast, macd_slow, macd_signal)
    result["macd"] = macd_line
    result["macd_signal"] = macd_sig
    result["macd_hist"] = macd_h
    result["atr"] = atr(df["high"], df["low"], df["close"], period=atr_period)
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"], period=bb_period)
    result["bb_upper"] = bb_upper
    result["bb_mid"] = bb_mid
    result["bb_lower"] = bb_lower
    result["returns"] = returns(df["close"])
    result["log_returns"] = returns(df["close"], log_returns=True)
    result["volatility"] = rolling_volatility(df["close"], period=vol_period)
    result["volume_sma"] = sma(df["volume"], 20)
    return result
