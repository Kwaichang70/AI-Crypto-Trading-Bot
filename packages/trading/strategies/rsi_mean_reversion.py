"""
packages/trading/strategies/rsi_mean_reversion.py
---------------------------------------------------
RSI Mean-Reversion Strategy.

Generates BUY signals when RSI crosses below the oversold threshold
(anticipating a bounce) and SELL signals when RSI crosses above the
overbought threshold (anticipating a pullback).

RSI calculation
~~~~~~~~~~~~~~~
Uses Wilder's exponential smoothing method:

1. Compute price changes:  delta[i] = close[i] - close[i-1]
2. Separate gains (positive deltas) and losses (absolute negative deltas).
3. First average gain/loss = SMA of first *period* values.
4. Subsequent values use exponential smoothing:
       avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
       avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
5. RS = avg_gain / avg_loss
6. RSI = 100 - (100 / (1 + RS))

Crossover detection
~~~~~~~~~~~~~~~~~~~
To avoid repeated signals on consecutive bars that stay in extreme zones,
the strategy only fires on the *transition* into the zone:

- BUY:  rsi_prev >= oversold  AND  rsi_curr < oversold
- SELL: rsi_prev <= overbought AND  rsi_curr > overbought

Confidence scoring
~~~~~~~~~~~~~~~~~~
Deeper penetration into the extreme zone yields higher confidence:

    BUY:  confidence = min(1.0, (oversold - rsi) / oversold)
    SELL: confidence = min(1.0, (rsi - overbought) / (100 - overbought))

Both clamped to [0.1, 1.0].

Parameters
----------
rsi_period : int
    RSI look-back period. Default 14.
oversold : float
    RSI level below which a BUY signal is generated. Default 30.
overbought : float
    RSI level above which a SELL signal is generated. Default 70.
position_size : float
    Target notional position in quote currency. Default 1000.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field, model_validator

from common.models import OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["RSIMeanReversionStrategy"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------

class _RSIParams(BaseModel):
    """Pydantic model for RSI Mean Reversion parameter validation."""

    rsi_period: int = Field(default=14, ge=2, le=500, description="RSI look-back period")
    oversold: float = Field(default=30.0, ge=1.0, le=49.0, description="Oversold threshold")
    overbought: float = Field(default=70.0, ge=51.0, le=99.0, description="Overbought threshold")
    position_size: float = Field(
        default=1000.0, gt=0.0, le=1_000_000.0, description="Quote-currency notional"
    )
    trailing_stop_pct: float | None = Field(
        default=None, ge=0.005, le=0.50,
        description="Trailing stop-loss percentage (e.g. 0.03 = 3%). None to disable."
    )

    @model_validator(mode="after")
    def oversold_lt_overbought(self) -> _RSIParams:
        """Ensure oversold < overbought with a reasonable gap."""
        if self.oversold >= self.overbought:
            raise ValueError(
                f"oversold ({self.oversold}) must be < overbought ({self.overbought})"
            )
        return self


# ---------------------------------------------------------------------------
# Helper: RSI calculation using Wilder's smoothing
# ---------------------------------------------------------------------------

def _compute_rsi(closes: Sequence[Decimal], period: int) -> Decimal:
    """
    Compute the Relative Strength Index using Wilder's smoothing.

    Parameters
    ----------
    closes : Sequence[Decimal]
        Close prices, oldest first.  Must contain at least ``period + 1``
        elements (we need *period* deltas, which requires period + 1 prices).
    period : int
        RSI look-back period.

    Returns
    -------
    Decimal
        RSI value in [0, 100], rounded to 4 decimal places.

    Raises
    ------
    ValueError
        If fewer than ``period + 1`` closes are provided.
    """
    min_required = period + 1
    if len(closes) < min_required:
        raise ValueError(f"Need >= {min_required} closes, got {len(closes)}")

    # Step 1: Compute deltas
    deltas: list[Decimal] = [
        closes[i] - closes[i - 1] for i in range(1, len(closes))
    ]

    # Step 2: Separate gains and losses
    gains: list[Decimal] = [max(d, Decimal(0)) for d in deltas]
    losses: list[Decimal] = [abs(min(d, Decimal(0))) for d in deltas]

    # Step 3: Initial average (SMA of first *period* values)
    avg_gain = sum(gains[:period]) / Decimal(period)
    avg_loss = sum(losses[:period]) / Decimal(period)

    # Step 4: Wilder's exponential smoothing for remaining values
    period_dec = Decimal(period)
    period_minus_one = period_dec - Decimal(1)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * period_minus_one + gains[i]) / period_dec
        avg_loss = (avg_loss * period_minus_one + losses[i]) / period_dec

    # Step 5/6: RS and RSI
    hundred = Decimal(100)
    if avg_loss == Decimal(0):
        # All gains, no losses -- RSI = 100
        return hundred

    rs = avg_gain / avg_loss
    rsi = hundred - (hundred / (Decimal(1) + rs))

    return rsi.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion -- counter-trend strategy.

    Generates BUY signals when RSI crosses below the oversold level,
    anticipating a price bounce.  Generates SELL signals when RSI crosses
    above the overbought level, anticipating a pullback.  Returns an
    empty signal list when RSI is within the neutral zone or when
    insufficient data is available.

    Attributes
    ----------
    metadata : StrategyMetadata
        Class-level strategy metadata for registry introspection.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="RSI Mean Reversion",
        version="1.0.0",
        description="RSI-based mean-reversion strategy with Wilder's smoothing",
        author="trading-engine-architect",
        tags=["mean-reversion", "rsi", "oscillator"],
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
            On any constraint violation (e.g. oversold >= overbought).
        """
        validated = _RSIParams(**params)
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
        return _RSIParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        """Wilder convergence (3x period) + 2 for previous-bar detection."""
        return int(self._params["rsi_period"]) * 3 + 2

    # ------------------------------------------------------------------ #
    # Core signal generation
    # ------------------------------------------------------------------ #

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        """
        Process a batch of OHLCV bars and detect RSI zone crossovers.

        The method requires at least ``rsi_period * 3 + 2`` bars to allow
        Wilder's exponential smoothing to converge beyond the initial SMA
        seed. With fewer bars, the RSI is numerically inaccurate. The extra
        bar beyond the convergence window enables computation of the
        "previous" RSI for crossover detection.

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            OHLCV bars ordered oldest-first.  The last element is the
            current (most recent) bar.

        Returns
        -------
        list[Signal]
            A single-element list on zone crossover, or empty list on HOLD.
        """
        rsi_period: int = self._params["rsi_period"]
        oversold: float = self._params["oversold"]
        overbought: float = self._params["overbought"]
        position_size: float = self._params["position_size"]

        # Wilder's smoothing requires ~3x the period for convergence.
        # We need enough bars for the smoothing to wash out the SMA seed,
        # plus 1 extra for the "previous RSI" (shifted back one bar).
        _WILDER_WARMUP_MULTIPLIER = 3
        min_bars = rsi_period * _WILDER_WARMUP_MULTIPLIER + 2
        if len(bars) < min_bars:
            self._log.debug(
                "rsi_mean_reversion.warmup",
                bars_available=len(bars),
                bars_required=min_bars,
            )
            return []

        # Extract the close prices we need -- use the full available tail
        # so that Wilder's smoothing iterates over the maximum history.
        # For current RSI: last (min_bars) closes
        # For previous RSI: drop the last close from that window
        needed = min_bars
        tail_bars = bars[-needed:]
        closes: list[Decimal] = [bar.close for bar in tail_bars]

        # Current RSI: all closes
        rsi_curr = _compute_rsi(closes, rsi_period)

        # Previous RSI: drop the last close
        rsi_prev = _compute_rsi(closes[:-1], rsi_period)

        rsi_curr_float = float(rsi_curr)
        rsi_prev_float = float(rsi_prev)

        # Detect zone crossover
        direction = SignalDirection.HOLD
        confidence = 0.5  # default, overridden below

        if rsi_prev_float >= oversold and rsi_curr_float < oversold:
            # RSI just crossed below oversold -- anticipate bounce
            direction = SignalDirection.BUY
            # Deeper into oversold = higher confidence
            depth = (oversold - rsi_curr_float) / oversold if oversold > 0 else 0.0
            confidence = min(1.0, max(0.1, depth))

        elif rsi_prev_float <= overbought and rsi_curr_float > overbought:
            # RSI just crossed above overbought -- anticipate pullback
            direction = SignalDirection.SELL
            ceiling = 100.0 - overbought
            depth = (rsi_curr_float - overbought) / ceiling if ceiling > 0 else 0.0
            confidence = min(1.0, max(0.1, depth))

        if direction == SignalDirection.HOLD:
            self._log.debug(
                "rsi_mean_reversion.hold",
                rsi_curr=rsi_curr_float,
                rsi_prev=rsi_prev_float,
            )
            return []

        current_bar = bars[-1]

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=current_bar.symbol,
            direction=direction,
            target_position=Decimal(str(position_size)),
            confidence=round(confidence, 4),
            metadata={
                "rsi": rsi_curr_float,
                "rsi_prev": rsi_prev_float,
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
                "close": str(current_bar.close),
            },
        )

        self._log.info(
            "rsi_mean_reversion.signal",
            direction=direction.value,
            confidence=signal.confidence,
            rsi=rsi_curr_float,
            symbol=current_bar.symbol,
        )

        return [signal]
