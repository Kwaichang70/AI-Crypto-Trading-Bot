"""
packages/data/ml_features.py
------------------------------
Shared ML feature builder for the AI Crypto Trading Bot.

Feature schemas
~~~~~~~~~~~~~~~

v1 (10 elements -- legacy)
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

v2 (14 elements -- QT-007 Sprint 46)
    The 10 v1 features above, PLUS four context-aware features:
    10     htf_trend             +1/0/-1 from highest-resolution HTF SMA cross
    11     fgi_level_norm        FGI / 100, clamped to [0, 1]; 0.5 if missing
    12     fgi_delta_7d          (FGI now - FGI 7d ago) / 100; 0.0 if missing
    13     btc_dom_delta_7d      (BTC dom now - BTC dom 7d ago) / 50; 0.0 if missing

API surface
~~~~~~~~~~~

* ``build_feature_vector_from_bars(bars)`` -- v1 inference path (pure Python).
* ``build_feature_matrix(df)`` -- v1 training path (vectorised pandas).
* ``build_extended_feature_vector_from_bars(bars, *, mtf_context, symbol)``
  -- v2 inference path; accepts optional MultiTimeframeContext.
* ``build_extended_feature_matrix(df, *, htf_close, fgi_series, btc_dom_series)``
  -- v2 training path; accepts optional aligned time-series inputs.
* ``feature_names_for_schema(version)`` -- returns the column-name list for
  the requested schema version (1 or 2).

Design notes
~~~~~~~~~~~~
- ``build_feature_vector_from_bars`` uses pure-Python helpers identical to
  the original ``ModelStrategy._build_feature_vector`` implementation.
- ``build_feature_matrix`` uses vectorised Pandas/NumPy operations.
- Both paths use ddof=0 (population std) for volatility features.
- RSI uses Wilder SMA-seeded smoothing normalised to [0, 1].
- v2 builders gracefully degrade: missing optional inputs fall back to
  neutral defaults so the function always returns the full 14-element
  vector / 14-column DataFrame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import structlog

from common.models import MultiTimeframeContext, OHLCVBar

__all__ = [
    "CURRENT_FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION_V1",
    "FEATURE_SCHEMA_VERSION_V2",
    "build_extended_feature_matrix",
    "build_extended_feature_vector_from_bars",
    "build_feature_matrix",
    "build_feature_vector_from_bars",
    "feature_names_for_schema",
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


# ===========================================================================
# QT-007 (Sprint 46) -- v2 extended feature schema (14 elements)
# ===========================================================================
#
# v2 schema adds four context-aware features on top of the 10 technical
# features from v1.  All four sources are OPTIONAL -- when a signal is
# missing the corresponding feature uses a neutral default so the model
# treats the absence as "no information" rather than as a spurious signal.
#
#     Position  Name                Default if missing
#     --------  ------------------- ------------------
#     10        htf_trend           0.0   (no HTF data)
#     11        fgi_level_norm      0.5   (FGI is missing -> neutral)
#     12        fgi_delta_7d        0.0   (no 7d-ago datum)
#     13        btc_dom_delta_7d    0.0   (no 7d-ago datum)
#
# v2 builders gracefully degrade when MultiTimeframeContext is None -- they
# return the same 14-element vector with all four context features at
# their neutral defaults.

FEATURE_SCHEMA_VERSION_V1 = 1
FEATURE_SCHEMA_VERSION_V2 = 2
CURRENT_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION_V2

_EXTENDED_FEATURE_NAMES_V2: list[str] = FEATURE_NAMES + [
    "htf_trend",
    "fgi_level_norm",
    "fgi_delta_7d",
    "btc_dom_delta_7d",
]


def feature_names_for_schema(version: int) -> list[str]:
    """Return the feature-name list for a given schema version."""
    if version == FEATURE_SCHEMA_VERSION_V1:
        return list(FEATURE_NAMES)
    if version == FEATURE_SCHEMA_VERSION_V2:
        return list(_EXTENDED_FEATURE_NAMES_V2)
    raise ValueError(f"Unknown feature schema version: {version}")


# ---------------------------------------------------------------------------
# HTF trend helper -- pure function, no MTF dependency
# ---------------------------------------------------------------------------


# Tier order used when MultiTimeframeContext provides multiple HTFs.  The
# higher-resolution HTF is preferred because it adapts faster -- daily/
# weekly are kept as fallbacks for strategies that only resample to those.
_HTF_TIER_ORDER: tuple[str, ...] = ("4h", "1d", "1w")


def _htf_trend_signal(htf_bars: Sequence[OHLCVBar], sma_period: int = 20) -> float:
    """Return signed trend from HTF bars: +1.0 above SMA, -1.0 below, 0.0 if
    insufficient bars (fewer than ``sma_period`` HTF bars available).

    The signal is intentionally coarse -- strategies that need a continuous
    trend score should compute it from the bars directly.  This helper
    exists to provide a stable, low-cardinality feature that a tree-based
    model can split on without overfitting noise.
    """
    if len(htf_bars) < sma_period:
        return 0.0
    closes = [float(bar.close) for bar in htf_bars[-sma_period:]]
    sma = sum(closes) / sma_period
    last_close = float(htf_bars[-1].close)
    if last_close > sma:
        return 1.0
    if last_close < sma:
        return -1.0
    return 0.0


def _resolve_htf_trend_from_context(
    mtf_context: MultiTimeframeContext | None,
    symbol: str,
) -> float:
    """Find the best-available HTF bar series for *symbol* and compute the
    trend signal.  Walks ``_HTF_TIER_ORDER`` in preference order and returns
    on the first tier that has bars for *symbol*.  Returns 0.0 if no HTF
    data is present in the context.

    Defensive ``getattr`` access supports duck-typed objects passed by
    tests (e.g., SimpleNamespace) -- the type annotation is the strict
    contract but the body tolerates loose shapes.
    """
    if mtf_context is None:
        return 0.0
    htf_bars = getattr(mtf_context, "htf_bars", None)
    if not isinstance(htf_bars, dict) or not htf_bars:
        return 0.0
    for tier in _HTF_TIER_ORDER:
        tier_bars = htf_bars.get(tier)
        if not isinstance(tier_bars, dict):
            continue
        bars_for_symbol = tier_bars.get(symbol)
        if isinstance(bars_for_symbol, list) and bars_for_symbol:
            return _htf_trend_signal(bars_for_symbol)
    return 0.0


# ---------------------------------------------------------------------------
# Public API -- v2 bars path
# ---------------------------------------------------------------------------


# BTC dominance delta divisor: dominance historically ranges in [40, 70]%, so
# week-over-week deltas rarely exceed +/- 10 percentage points.  Dividing by
# 50 keeps the normalised feature comfortably inside [-1, 1] for normal
# market behaviour while still expressing extreme regime shifts (a +/- 50pp
# move would be a once-in-a-decade event and would saturate to +/- 1).
_BTC_DOM_DELTA_DIVISOR: float = 50.0


def build_extended_feature_vector_from_bars(
    bars: Sequence[OHLCVBar],
    *,
    mtf_context: MultiTimeframeContext | None = None,
    symbol: str | None = None,
) -> list[float]:
    """Build a 14-element v2 feature vector.

    The first 10 elements are byte-identical to
    :func:`build_feature_vector_from_bars`.  The four additional features
    are extracted from ``mtf_context``:

    * ``htf_trend`` -- signed SMA-cross signal on the highest-resolution
      HTF available for ``symbol``.  Returns 0.0 when no HTF data is present.
    * ``fgi_level_norm`` -- ``mtf_context.fear_greed_index / 100``, clamped
      to ``[0, 1]``; 0.5 (neutral) when None.
    * ``fgi_delta_7d`` -- normalised ``(fgi - fgi_7d_ago) / 100``; 0.0 when
      either value is None.
    * ``btc_dom_delta_7d`` -- normalised ``(btc_dom - btc_dom_7d_ago) / 50``;
      0.0 when either value is None.

    Parameters
    ----------
    bars:
        Same as v1 -- the primary-timeframe OHLCV series.
    mtf_context:
        Optional :class:`MultiTimeframeContext`.  When None all four
        context features fall back to their neutral defaults so the
        function still returns a valid 14-element vector.
    symbol:
        Symbol key used to look up HTF bars in ``mtf_context.htf_bars``
        for the ``htf_trend`` feature ONLY.  FGI and BTC-dominance
        features are context-level (market-wide) and are not affected
        by this parameter.  Defaults to ``bars[-1].symbol`` when None.

    Returns
    -------
    list[float]:
        14-element feature vector in the order defined by
        :data:`_EXTENDED_FEATURE_NAMES_V2`.
    """
    base_features = build_feature_vector_from_bars(bars)
    resolved_symbol = symbol if symbol is not None else bars[-1].symbol

    htf_trend = _resolve_htf_trend_from_context(mtf_context, resolved_symbol)

    if mtf_context is None:
        fgi_level = 0.5
        fgi_delta = 0.0
        btc_dom_delta = 0.0
    else:
        fgi_current = getattr(mtf_context, "fear_greed_index", None)
        fgi_7d_ago = getattr(mtf_context, "fear_greed_index_7d_ago", None)
        btc_dom_current = getattr(mtf_context, "btc_dominance", None)
        btc_dom_7d_ago = getattr(mtf_context, "btc_dominance_7d_ago", None)

        if fgi_current is not None:
            # Clamp to [0, 1] -- alternative.me has historically returned
            # values slightly outside [0, 100] due to rounding glitches;
            # corrupted model inputs are worse than a clamped one.
            fgi_level = max(0.0, min(1.0, float(fgi_current) / 100.0))
        else:
            fgi_level = 0.5

        if fgi_current is not None and fgi_7d_ago is not None:
            fgi_delta = (float(fgi_current) - float(fgi_7d_ago)) / 100.0
        else:
            fgi_delta = 0.0

        if btc_dom_current is not None and btc_dom_7d_ago is not None:
            btc_dom_delta = (
                (float(btc_dom_current) - float(btc_dom_7d_ago))
                / _BTC_DOM_DELTA_DIVISOR
            )
        else:
            btc_dom_delta = 0.0

    return base_features + [htf_trend, fgi_level, fgi_delta, btc_dom_delta]


def build_extended_feature_matrix(
    df: pd.DataFrame,
    *,
    htf_close: pd.Series | None = None,
    htf_sma_period: int = 20,
    fgi_series: pd.Series | None = None,
    btc_dom_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a 14-column v2 feature matrix from an OHLCV DataFrame.

    First 10 columns are byte-identical to :func:`build_feature_matrix`.
    The four additional columns are derived from optional aligned
    time-series inputs:

    * ``htf_trend`` -- ``htf_close`` forward-filled to ``df.index`` and
      compared against its rolling SMA(``htf_sma_period``).  Above -> +1,
      below -> -1, equal/insufficient -> 0.  Returns 0.0 column when
      ``htf_close`` is None.
    * ``fgi_level_norm`` -- ``fgi_series`` ffilled to ``df.index``,
      divided by 100, clamped to ``[0, 1]``.  Returns 0.5 column when
      ``fgi_series`` is None.
    * ``fgi_delta_7d`` -- ``(fgi_aligned - fgi_aligned.shifted_by_7d) / 100``.
      Returns 0.0 column when ``fgi_series`` is None.
    * ``btc_dom_delta_7d`` -- ``(btc_dom_aligned - btc_dom_aligned.shifted_by_7d) / 50``.
      Returns 0.0 column when ``btc_dom_series`` is None.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must have ``close``, ``volume``, ``high``, ``low``.
        If any of ``htf_close``, ``fgi_series``, ``btc_dom_series`` is
        provided, ``df.index`` MUST be a monotonically increasing
        ``DatetimeIndex`` (required for time-based shift / ffill alignment).
        When all optional inputs are None, any index is accepted (matches
        the v1 contract).
    htf_close:
        Optional HTF close series with a DatetimeIndex.  Granularity can
        be coarser than ``df`` (e.g., 4h closes against 1h primary).
        Callers with a full HTF OHLCV DataFrame should pass ``htf_df["close"]``.
        Forward-filled onto ``df.index`` before SMA computation so each
        row gets the most recently observed HTF close.
    htf_sma_period:
        Rolling-mean period applied to the aligned HTF close series.
        Default 20 matches :func:`_htf_trend_signal` in the bars path.
    fgi_series:
        Optional FGI value series ([0, 100]) with a DatetimeIndex.  Daily
        granularity typical (alternative.me publishes once per day).
    btc_dom_series:
        Optional BTC dominance percentage series ([0, 100]) with a
        DatetimeIndex.

    Returns
    -------
    pd.DataFrame:
        14 columns named per :data:`_EXTENDED_FEATURE_NAMES_V2`.  Warmup
        rows of the v1 columns remain NaN as in v1.  The v2 columns are
        fully populated (neutral defaults backfill any gaps).

    Raises
    ------
    ValueError:
        If ``df`` is empty, missing required columns, or any optional
        series is supplied with a non-DatetimeIndex or non-monotonic
        ``df`` index.
    TypeError:
        If ``df.index`` and any optional series have incompatible timezone
        awareness (pandas raises this from ``reindex``).
    """
    base = build_feature_matrix(df)   # 10 columns + same row count + NaN warmup

    # Index-validity guards: time-shift operations need a sorted DatetimeIndex.
    needs_datetime_index = any(
        s is not None for s in (htf_close, fgi_series, btc_dom_series)
    )
    if needs_datetime_index:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "build_extended_feature_matrix requires a DatetimeIndex on df "
                "when htf_close / fgi_series / btc_dom_series is supplied"
            )
        if not df.index.is_monotonic_increasing:
            # shift(freq=...) silently misbehaves on non-monotonic indexes
            # in older pandas; fail loudly so the caller fixes the input.
            raise ValueError(
                "build_extended_feature_matrix requires a monotonically "
                "increasing DatetimeIndex when htf_close / fgi_series / "
                "btc_dom_series is supplied"
            )

    # -- htf_trend -------------------------------------------------------
    if htf_close is not None:
        htf_aligned = htf_close.reindex(df.index, method="ffill").astype(float)
        sma = htf_aligned.rolling(
            window=htf_sma_period, min_periods=htf_sma_period,
        ).mean()
        # NaN comparisons are False in NumPy, so warmup rows (sma=NaN ->
        # diff=NaN) naturally produce 0.0 from both legs of np.where.
        diff = htf_aligned - sma
        htf_trend = pd.Series(
            np.where(diff > 0.0, 1.0, np.where(diff < 0.0, -1.0, 0.0)),
            index=df.index,
            dtype=float,
        )
    else:
        htf_trend = pd.Series(0.0, index=df.index, dtype=float)

    # -- fgi_level_norm + fgi_delta_7d ----------------------------------
    if fgi_series is not None:
        fgi_aligned = fgi_series.reindex(df.index, method="ffill").astype(float)
        fgi_level = (fgi_aligned / 100.0).clip(lower=0.0, upper=1.0)
        fgi_level = fgi_level.fillna(0.5)
        # shift(freq=7d) moves each FGI timestamp forward by 7 days in
        # the index, so the value originally at t0 is now accessible at
        # t0+7d.  reindex+ffill then propagates each shifted daily value
        # across all intra-day hourly bars, giving us: "what was FGI 7
        # days before this bar?"
        fgi_shifted = fgi_series.shift(freq=pd.Timedelta(days=7))
        fgi_7d_ago = fgi_shifted.reindex(df.index, method="ffill").astype(float)
        # 0.0 = "no observed change" -- covers pre-history rows and
        # genuine FGI data gaps; the model learns to treat 0 as unknown.
        fgi_delta = ((fgi_aligned - fgi_7d_ago) / 100.0).fillna(0.0)
    else:
        fgi_level = pd.Series(0.5, index=df.index, dtype=float)
        fgi_delta = pd.Series(0.0, index=df.index, dtype=float)

    # -- btc_dom_delta_7d -----------------------------------------------
    if btc_dom_series is not None:
        dom_aligned = btc_dom_series.reindex(df.index, method="ffill").astype(float)
        dom_shifted = btc_dom_series.shift(freq=pd.Timedelta(days=7))
        dom_7d_ago = dom_shifted.reindex(df.index, method="ffill").astype(float)
        btc_dom_delta = (
            (dom_aligned - dom_7d_ago) / _BTC_DOM_DELTA_DIVISOR
        ).fillna(0.0)
    else:
        btc_dom_delta = pd.Series(0.0, index=df.index, dtype=float)

    extended = base.copy()
    extended["htf_trend"] = htf_trend
    extended["fgi_level_norm"] = fgi_level
    extended["fgi_delta_7d"] = fgi_delta
    extended["btc_dom_delta_7d"] = btc_dom_delta

    return extended
