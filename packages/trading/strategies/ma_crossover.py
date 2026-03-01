"""
packages/trading/strategies/ma_crossover.py
--------------------------------------------
Dual Moving Average Crossover Strategy.

A classic trend-following approach that generates BUY signals when the fast
simple moving average (SMA) crosses above the slow SMA (golden cross), and
SELL signals when the fast SMA crosses below (death cross).

Indicator maths
~~~~~~~~~~~~~~~
SMA_n = sum(close[i] for i in range(n)) / n

A crossover is detected by comparing the relative position of fast vs. slow
SMA on the current bar against the previous bar:

- **Golden cross**: fast_prev <= slow_prev AND fast_curr > slow_curr
- **Death cross**:  fast_prev >= slow_prev AND fast_curr < slow_curr

Confidence scoring
~~~~~~~~~~~~~~~~~~
Confidence is derived from the percentage spread between the two moving
averages at the moment of crossover, clamped to [0.1, 1.0].  A wider spread
suggests stronger momentum behind the signal.

    spread_pct = abs(fast - slow) / slow
    confidence = min(1.0, max(0.1, spread_pct * 20))

The ``* 20`` scaling factor maps a 5 % spread to full confidence.

Parameters
----------
fast_period : int
    Look-back window for the fast SMA. Default 10.
slow_period : int
    Look-back window for the slow SMA. Default 50.  Must be > fast_period.
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

__all__ = ["MACrossoverStrategy"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------

class _MACrossoverParams(BaseModel):
    """Pydantic model for MA Crossover parameter validation."""

    fast_period: int = Field(default=10, ge=2, le=500, description="Fast SMA window")
    slow_period: int = Field(default=50, ge=3, le=2000, description="Slow SMA window")
    position_size: float = Field(
        default=1000.0, gt=0.0, le=1_000_000.0, description="Quote-currency notional"
    )

    @model_validator(mode="after")
    def fast_lt_slow(self) -> _MACrossoverParams:
        """Ensure the fast period is strictly less than the slow period."""
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be < slow_period ({self.slow_period})"
            )
        return self


# ---------------------------------------------------------------------------
# Helper: Simple Moving Average over Decimal sequence
# ---------------------------------------------------------------------------

def _sma(values: Sequence[Decimal], period: int) -> Decimal:
    """
    Compute the Simple Moving Average of the last *period* values.

    Parameters
    ----------
    values : Sequence[Decimal]
        Price series (at least *period* elements long).
    period : int
        Number of data points to average.

    Returns
    -------
    Decimal
        The SMA value, rounded to 8 decimal places.

    Raises
    ------
    ValueError
        If fewer values than *period* are provided.
    """
    if len(values) < period:
        raise ValueError(f"Need >= {period} values, got {len(values)}")
    window = values[-period:]
    total = sum(window, Decimal(0))
    return (total / Decimal(period)).quantize(Decimal("1E-8"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class MACrossoverStrategy(BaseStrategy):
    """
    Dual SMA Crossover -- trend-following strategy.

    Emits a BUY signal on a golden cross (fast SMA crosses above slow SMA)
    and a SELL signal on a death cross (fast SMA crosses below slow SMA).
    Returns an empty signal list when there is no crossover or when
    insufficient data is available for warm-up.

    Attributes
    ----------
    metadata : StrategyMetadata
        Class-level strategy metadata for registry introspection.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="MA Crossover",
        version="1.0.0",
        description="Dual SMA crossover trend-following strategy",
        author="trading-engine-architect",
        tags=["trend", "moving-average", "crossover"],
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
            On any constraint violation (e.g. fast_period >= slow_period).
        """
        validated = _MACrossoverParams(**params)
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
        return _MACrossoverParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        """Slow SMA window + 1 bar for previous-bar crossover detection."""
        return int(self._params["slow_period"]) + 1

    # ------------------------------------------------------------------ #
    # Core signal generation
    # ------------------------------------------------------------------ #

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        """
        Process a batch of OHLCV bars and detect SMA crossovers.

        The method requires at least ``slow_period + 1`` bars so that both
        the current and previous SMA values can be computed.  If fewer bars
        are available, an empty list is returned (warm-up phase).

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            OHLCV bars ordered oldest-first.  The last element is the
            current (most recent) bar.

        Returns
        -------
        list[Signal]
            A single-element list on crossover, or an empty list on HOLD.
        """
        fast_period: int = self._params["fast_period"]
        slow_period: int = self._params["slow_period"]
        position_size: float = self._params["position_size"]

        min_bars = slow_period + 1
        if len(bars) < min_bars:
            self._log.debug(
                "ma_crossover.warmup",
                bars_available=len(bars),
                bars_required=min_bars,
            )
            return []

        # Extract close prices as Decimal sequence -- only the tail we need
        # We need slow_period + 1 bars to compute current + previous SMA
        needed = slow_period + 1
        closes: list[Decimal] = [bar.close for bar in bars[-needed:]]

        # Current SMAs (using the last N closes)
        fast_curr = _sma(closes, fast_period)
        slow_curr = _sma(closes, slow_period)

        # Previous SMAs (shift window back by one bar)
        closes_prev = closes[:-1]  # drop the last bar
        fast_prev = _sma(closes_prev, fast_period)
        slow_prev = _sma(closes_prev, slow_period)

        # Detect crossover
        direction = SignalDirection.HOLD

        if fast_prev <= slow_prev and fast_curr > slow_curr:
            # Golden cross -- fast crossed above slow
            direction = SignalDirection.BUY
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            # Death cross -- fast crossed below slow
            direction = SignalDirection.SELL

        if direction == SignalDirection.HOLD:
            self._log.debug(
                "ma_crossover.hold",
                fast_sma=str(fast_curr),
                slow_sma=str(slow_curr),
            )
            return []

        # Confidence: based on spread between the two SMAs
        if slow_curr > Decimal(0):
            spread_pct = float(abs(fast_curr - slow_curr) / slow_curr)
        else:
            spread_pct = 0.0
        confidence = min(1.0, max(0.1, spread_pct * 20.0))

        current_bar = bars[-1]

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=current_bar.symbol,
            direction=direction,
            target_position=Decimal(str(position_size)),
            confidence=round(confidence, 4),
            metadata={
                "fast_sma": str(fast_curr),
                "slow_sma": str(slow_curr),
                "fast_sma_prev": str(fast_prev),
                "slow_sma_prev": str(slow_prev),
                "spread_pct": round(spread_pct, 6),
                "fast_period": fast_period,
                "slow_period": slow_period,
                "close": str(current_bar.close),
            },
        )

        self._log.info(
            "ma_crossover.signal",
            direction=direction.value,
            confidence=signal.confidence,
            fast_sma=str(fast_curr),
            slow_sma=str(slow_curr),
            symbol=current_bar.symbol,
        )

        return [signal]
