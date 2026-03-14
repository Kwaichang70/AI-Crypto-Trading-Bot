"""
packages/trading/strategies/breakout.py
-----------------------------------------
Donchian Channel Breakout Strategy with ATR-based confidence scaling.

Generates BUY signals when the current close exceeds the highest high of
the previous *lookback_period* bars (upside breakout), and SELL signals
when the current close falls below the lowest low (downside breakout).

Donchian Channel
~~~~~~~~~~~~~~~~
    upper_band = max(high[i] for i in range(-lookback, -1))  # exclude current bar
    lower_band = min(low[i]  for i in range(-lookback, -1))

Breakout detection uses the *previous* bars' range (excluding the current
bar) so there is no look-ahead bias:

- **Upside breakout**: close > upper_band  (new high)
- **Downside breakout**: close < lower_band (new low)

ATR calculation (Average True Range)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uses Wilder's smoothing:

    TR[i] = max(
        high[i] - low[i],
        abs(high[i] - close[i-1]),
        abs(low[i]  - close[i-1]),
    )

    ATR_1 = SMA(TR[1..period])
    ATR_i = (ATR_{i-1} * (period - 1) + TR_i) / period    (for i > period)

Confidence scoring
~~~~~~~~~~~~~~~~~~
Confidence is inversely related to ATR relative to price -- the idea is
that breakouts in low-volatility environments are more likely to sustain
(compression breakout thesis).  In high-ATR environments, mean-reversion
is more likely, so confidence is reduced.

    atr_pct = ATR / close
    raw_conf = 1.0 - min(1.0, atr_pct * atr_multiplier * 10)
    confidence = max(0.1, raw_conf)

Parameters
----------
lookback_period : int
    Donchian channel look-back window. Default 20.
position_size : float
    Target notional position in quote currency. Default 1000.0.
atr_period : int
    ATR smoothing period. Default 14.
atr_multiplier : float
    Scaling factor for ATR-based confidence dampening. Default 1.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["BreakoutStrategy"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------

class _BreakoutParams(BaseModel):
    """Pydantic model for Breakout strategy parameter validation."""

    lookback_period: int = Field(
        default=20, ge=2, le=500, description="Donchian channel look-back window"
    )
    position_size: float = Field(
        default=1000.0, gt=0.0, le=1_000_000.0, description="Quote-currency notional"
    )
    atr_period: int = Field(
        default=14, ge=2, le=500, description="ATR smoothing period"
    )
    atr_multiplier: float = Field(
        default=1.5, gt=0.0, le=10.0, description="ATR confidence scaling factor"
    )
    trailing_stop_pct: float | None = Field(
        default=None, ge=0.005, le=0.50,
        description="Trailing stop-loss percentage (e.g. 0.03 = 3%). None to disable."
    )


# ---------------------------------------------------------------------------
# Helper: Average True Range (Wilder's smoothing)
# ---------------------------------------------------------------------------

def _compute_atr(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int,
) -> Decimal:
    """
    Compute the Average True Range using Wilder's smoothing.

    Parameters
    ----------
    highs : Sequence[Decimal]
        High prices, oldest first.
    lows : Sequence[Decimal]
        Low prices, oldest first.
    closes : Sequence[Decimal]
        Close prices, oldest first.
    period : int
        ATR smoothing period.

    Returns
    -------
    Decimal
        ATR value rounded to 8 decimal places.

    Raises
    ------
    ValueError
        If fewer than ``period + 1`` data points are provided.

    Notes
    -----
    True Range requires the previous close, so we need ``period + 1``
    bars to produce *period* TR values for the initial SMA, plus any
    additional bars are smoothed exponentially.
    """
    n = len(highs)
    if n < period + 1:
        raise ValueError(f"Need >= {period + 1} data points, got {n}")
    if not (len(lows) == n and len(closes) == n):
        raise ValueError("highs, lows, and closes must have equal length")

    # Compute True Range series (starts from index 1)
    tr_values: list[Decimal] = []
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hc, lc))

    # Initial ATR: SMA of first *period* TRs
    period_dec = Decimal(period)
    atr = sum(tr_values[:period]) / period_dec

    # Wilder's smoothing for remaining TRs
    period_minus_one = period_dec - Decimal(1)
    for i in range(period, len(tr_values)):
        atr = (atr * period_minus_one + tr_values[i]) / period_dec

    return atr.quantize(Decimal("1E-8"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Helper: Donchian Channel (upper/lower band)
# ---------------------------------------------------------------------------

def _donchian_channel(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    period: int,
) -> tuple[Decimal, Decimal]:
    """
    Compute the Donchian channel bands over the given window.

    Parameters
    ----------
    highs : Sequence[Decimal]
        High prices for the look-back window.
    lows : Sequence[Decimal]
        Low prices for the look-back window.
    period : int
        Number of bars to look back.

    Returns
    -------
    tuple[Decimal, Decimal]
        (upper_band, lower_band) -- highest high and lowest low.

    Raises
    ------
    ValueError
        If fewer than *period* data points are provided.
    """
    if len(highs) < period or len(lows) < period:
        raise ValueError(f"Need >= {period} data points for Donchian channel")

    window_highs = highs[-period:]
    window_lows = lows[-period:]
    return max(window_highs), min(window_lows)


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class BreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout with ATR confidence scaling.

    Emits a BUY signal when the current close exceeds the highest high of
    the previous *lookback_period* bars.  Emits a SELL signal when the
    current close falls below the lowest low.  Confidence is scaled
    inversely with ATR -- breakouts in low-volatility environments receive
    higher confidence.

    Attributes
    ----------
    metadata : StrategyMetadata
        Class-level strategy metadata for registry introspection.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Breakout",
        version="1.0.0",
        description="Donchian channel breakout with ATR-scaled confidence",
        author="trading-engine-architect",
        tags=["breakout", "donchian", "trend", "volatility"],
    )

    # ------------------------------------------------------------------ #
    # Parameter validation
    # ------------------------------------------------------------------ #

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and coerce parameters via the Pydantic schema.

        Parameters
        ----------
        params : dict[str, Any]
            Raw parameter dict from the caller.

        Returns
        -------
        dict[str, Any]
            Validated parameter dict with defaults filled in.

        Raises
        ------
        ValueError
            On any constraint violation.
        """
        validated = _BreakoutParams(**params)
        return validated.model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """
        Return JSON Schema for accepted parameters.

        Returns
        -------
        dict[str, Any]
            JSON Schema derived from the Pydantic params model.
        """
        return _BreakoutParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        """Max of Donchian window and ATR convergence (3x period)."""
        lookback = int(self._params["lookback_period"])
        atr_p = int(self._params["atr_period"])
        return max(lookback + 1, atr_p * 3 + 1)

    # ------------------------------------------------------------------ #
    # Core signal generation
    # ------------------------------------------------------------------ #

    def on_bar(self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None) -> list[Signal]:
        """
        Process a batch of OHLCV bars and detect channel breakouts.

        The method requires ``max(lookback_period + 1, atr_period * 3 + 1)``
        bars. The Donchian channel needs ``lookback_period`` previous bars
        plus the current bar. The ATR needs ``~3 * atr_period`` bars for
        Wilder's exponential smoothing to converge beyond the initial SMA
        seed.

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            OHLCV bars ordered oldest-first.  The last element is the
            current (most recent) bar.

        Returns
        -------
        list[Signal]
            A single-element list on breakout, or an empty list on HOLD.
        """
        lookback_period: int = self._params["lookback_period"]
        position_size: float = self._params["position_size"]
        atr_period: int = self._params["atr_period"]
        atr_multiplier: float = self._params["atr_multiplier"]

        # Need enough bars for both:
        # 1. Donchian channel: lookback_period + 1 (lookback previous + current)
        # 2. ATR with Wilder convergence: atr_period * 3 + 1 (~3x for smoothing)
        _ATR_WARMUP_MULTIPLIER = 3
        min_bars = max(lookback_period + 1, atr_period * _ATR_WARMUP_MULTIPLIER + 1)
        if len(bars) < min_bars:
            self._log.debug(
                "breakout.warmup",
                bars_available=len(bars),
                bars_required=min_bars,
            )
            return []

        current_bar = bars[-1]

        # --- Donchian Channel (exclude current bar) ---
        # Use the *previous* lookback_period bars for the channel
        prev_bars = bars[-(lookback_period + 1):-1]
        prev_highs: list[Decimal] = [b.high for b in prev_bars]
        prev_lows: list[Decimal] = [b.low for b in prev_bars]
        upper_band, lower_band = _donchian_channel(prev_highs, prev_lows, lookback_period)

        # --- ATR calculation ---
        # Use all available bars for ATR so Wilder's smoothing has maximum
        # history to converge. min_bars guarantees at least 3*atr_period+1 bars.
        atr_bars_needed = min(len(bars), atr_period * _ATR_WARMUP_MULTIPLIER + 1)
        atr_slice = bars[-atr_bars_needed:]
        atr_highs: list[Decimal] = [b.high for b in atr_slice]
        atr_lows: list[Decimal] = [b.low for b in atr_slice]
        atr_closes: list[Decimal] = [b.close for b in atr_slice]
        atr = _compute_atr(atr_highs, atr_lows, atr_closes, atr_period)

        # --- Breakout detection ---
        direction = SignalDirection.HOLD

        if current_bar.close > upper_band:
            direction = SignalDirection.BUY
        elif current_bar.close < lower_band:
            direction = SignalDirection.SELL

        if direction == SignalDirection.HOLD:
            self._log.debug(
                "breakout.hold",
                close=str(current_bar.close),
                upper_band=str(upper_band),
                lower_band=str(lower_band),
                atr=str(atr),
            )
            return []

        # --- Confidence: inverse ATR scaling ---
        # Lower volatility relative to price = higher confidence
        if current_bar.close > Decimal(0):
            atr_pct = float(atr / current_bar.close)
        else:
            atr_pct = 0.0

        raw_confidence = 1.0 - min(1.0, atr_pct * atr_multiplier * 10.0)
        confidence = max(0.1, raw_confidence)

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=current_bar.symbol,
            direction=direction,
            target_position=Decimal(str(position_size)),
            confidence=round(confidence, 4),
            metadata={
                "upper_band": str(upper_band),
                "lower_band": str(lower_band),
                "atr": str(atr),
                "atr_pct": round(atr_pct, 6),
                "lookback_period": lookback_period,
                "atr_period": atr_period,
                "atr_multiplier": atr_multiplier,
                "close": str(current_bar.close),
            },
        )

        self._log.info(
            "breakout.signal",
            direction=direction.value,
            confidence=signal.confidence,
            close=str(current_bar.close),
            upper_band=str(upper_band),
            lower_band=str(lower_band),
            atr=str(atr),
            symbol=current_bar.symbol,
        )

        return [signal]
