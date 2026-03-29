"""
packages/trading/strategies/grid_trading.py
--------------------------------------------
Grid Trading Strategy.

Divides price space into equally-spaced grid levels above and below a
reference price.  The engine accumulates positions on downward moves
(BUY at lower grid levels) and takes profit on upward moves (SELL at higher
grid levels), capturing the oscillation within the expected price range.

Design
------
- On the first bar the current close becomes the ``reference_price``.
- Grid levels are indexed as integers relative to the reference:
    - index -1  => reference * (1 - 1 * grid_size_pct)   # first buy level
    - index -2  => reference * (1 - 2 * grid_size_pct)   # second buy level
    - index +1  => reference * (1 + 1 * grid_size_pct)   # first sell level
- Each grid level can trigger at most once in each direction until the
  opposite side fires (reset mechanic prevents runaway one-sided fills).
- RSI (optional) can gate BUY signals: ``min_rsi_buy`` suppresses buys when
  RSI is too low (panic) or a ``max_rsi_sell`` suppresses sells when RSI is
  too high.  Default values (0 / 100) disable both filters.

Parameters
----------
grid_size_pct : float
    Fractional distance between adjacent grid levels.  0.01 = 1%.
    Range (0, 0.50].  Default 0.01.
num_grids : int
    Number of grid levels above and below the reference price.
    Range [1, 50].  Default 5.
position_size : float
    Quote-currency amount per grid order.
    Range (0, 1_000_000].  Default 100.0.
rsi_period : int
    RSI look-back period used by the optional RSI filters.
    Range [2, 500].  Default 14.
min_rsi_buy : float
    RSI must be >= this value to allow a grid BUY.
    0 disables the filter (default).  Range [0, 100].
max_rsi_sell : float
    RSI must be <= this value to allow a grid SELL.
    100 disables the filter (default).  Range [0, 100].
trailing_stop_pct : float | None
    Trailing stop-loss percentage (e.g. 0.03 = 3 %).  None to disable.

Confidence scoring
------------------
Deeper grid levels yield higher confidence -- larger deviations from the
reference price carry more mean-reversion conviction::

    confidence = min(1.0, abs(grid_idx) * 0.20)   clamped to [0.10, 1.0]

RSI calculation
~~~~~~~~~~~~~~~
Delegates to ``_compute_rsi`` from ``rsi_mean_reversion`` (Wilder's
exponential smoothing method) to avoid code duplication.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field, model_validator

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategies.rsi_mean_reversion import _compute_rsi

__all__ = ["GridTradingStrategy"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------


class _GridParams(BaseModel):
    """Pydantic model for Grid Trading Strategy parameter validation."""

    grid_size_pct: float = Field(
        default=0.01,
        gt=0.0,
        le=0.50,
        description="Fractional distance between grid levels (e.g. 0.01 = 1%)",
    )
    num_grids: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of grid levels above and below the reference price",
    )
    position_size: float = Field(
        default=100.0,
        gt=0.0,
        le=1_000_000.0,
        description="Quote-currency amount per grid order",
    )
    rsi_period: int = Field(
        default=14,
        ge=2,
        le=500,
        description="RSI look-back period (used by optional RSI filters)",
    )
    min_rsi_buy: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="RSI must be >= this to allow grid BUY (0 = disabled)",
    )
    max_rsi_sell: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="RSI must be <= this to allow grid SELL (100 = disabled)",
    )
    trailing_stop_pct: float | None = Field(
        default=None,
        ge=0.005,
        le=0.50,
        description="Trailing stop-loss percentage (e.g. 0.03 = 3 %). None to disable.",
    )

    @model_validator(mode="after")
    def _rsi_filters_coherent(self) -> _GridParams:
        """Ensure RSI filter bounds don't conflict."""
        if self.min_rsi_buy > self.max_rsi_sell:
            raise ValueError(
                f"min_rsi_buy ({self.min_rsi_buy}) must be <= "
                f"max_rsi_sell ({self.max_rsi_sell})"
            )
        return self


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy -- systematic buy-low / sell-high oscillation.

    Tracks a reference price established on the first bar and divides price
    space into ``num_grids`` levels in each direction separated by
    ``grid_size_pct``.  BUY signals fire when price touches a lower grid
    level for the first time; SELL signals fire at upper grid levels.

    Attributes
    ----------
    metadata : StrategyMetadata
        Class-level strategy metadata for registry introspection.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Grid Trading",
        version="1.0.0",
        description=(
            "Systematic grid-based accumulation and profit-taking: "
            "buys at lower price levels, sells at upper levels"
        ),
        author="python-backend-specialist",
        tags=["grid", "range", "mean-reversion", "accumulation"],
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self, run_id: str) -> None:
        """Initialise per-symbol grid state for a new run."""
        super().on_start(run_id)
        # reference_price: first bar's close price per symbol
        self._reference_prices: dict[str, Decimal] = {}
        # grids_hit: set of (symbol, grid_idx) tuples that have triggered
        self._grids_hit: dict[str, set[int]] = {}
        self._log.info("grid_trading.started", run_id=run_id)

    def on_stop(self) -> None:
        """Reset per-symbol state on run stop."""
        self._reference_prices = {}
        self._grids_hit = {}
        super().on_stop()

    # ------------------------------------------------------------------
    # Parameter schema
    # ------------------------------------------------------------------

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce parameters via the Pydantic schema."""
        validated = _GridParams(**params)
        return validated.model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for accepted parameters."""
        return _GridParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        """Wilder convergence (3x period) + 1 for RSI computation."""
        return int(self._params["rsi_period"]) * 3 + 1

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        """
        Process a batch of OHLCV bars and emit grid buy/sell signals.

        Logic per bar
        -------------
        1. Establish reference price on first bar.
        2. Compute RSI if RSI filters are active.
        3. Calculate which grid level the current close corresponds to.
        4. If the level has not fired before, emit a BUY (negative index)
           or SELL (positive index) signal.
        5. Mark the level as triggered.
        6. Grid reset: when a BUY fires, clear all previously hit sell
           levels, and vice versa (prevents one-sided saturation).

        Parameters
        ----------
        bars:
            OHLCV bars ordered oldest-first; last element is current bar.
        mtf_context:
            Unused by this strategy (accepted for interface compatibility).

        Returns
        -------
        list[Signal]
            Zero or one signal per call.
        """
        grid_size_pct: float = self._params["grid_size_pct"]
        num_grids: int = self._params["num_grids"]
        position_size: float = self._params["position_size"]
        rsi_period: int = self._params["rsi_period"]
        min_rsi_buy: float = self._params["min_rsi_buy"]
        max_rsi_sell: float = self._params["max_rsi_sell"]

        current_bar = bars[-1]
        symbol = current_bar.symbol
        close = current_bar.close

        # Warm-up guard: need enough bars for RSI (even if filters disabled,
        # compute RSI consistently to keep the interface predictable)
        min_bars = rsi_period * 3 + 1
        if len(bars) < min_bars:
            self._log.debug(
                "grid_trading.warmup",
                symbol=symbol,
                bars_available=len(bars),
                bars_required=min_bars,
            )
            return []

        # Establish reference price on first bar after warmup
        if symbol not in self._reference_prices:
            self._reference_prices[symbol] = close
            self._grids_hit[symbol] = set()
            self._log.info(
                "grid_trading.reference_set",
                symbol=symbol,
                reference_price=str(close),
            )
            return []

        ref_price = self._reference_prices[symbol]
        grids_hit = self._grids_hit[symbol]

        # Compute RSI (always, for logging and optional filters)
        tail_closes: list[Decimal] = [bar.close for bar in bars[-min_bars:]]
        rsi_curr = _compute_rsi(tail_closes, rsi_period)
        rsi_float = float(rsi_curr)

        # Determine the grid index for the current close price.
        # pct_change is signed: negative = below reference, positive = above.
        ref_float = float(ref_price)
        if ref_float <= 0.0:
            return []
        pct_change = (float(close) - ref_float) / ref_float

        # Floor division maps each interval to its index:
        #   pct_change in [-0.02, -0.01) => grid_idx = -2
        #   pct_change in [-0.01,  0.00) => grid_idx = -1
        #   pct_change in [ 0.00,  0.01) => grid_idx =  0  (no signal)
        #   pct_change in [ 0.01,  0.02) => grid_idx = +1
        grid_idx = math.floor(pct_change / grid_size_pct)

        # index 0 means price is inside the reference interval -- no signal
        if grid_idx == 0:
            return []

        # Clamp to configured range
        if abs(grid_idx) > num_grids:
            return []

        # Already fired for this level
        if grid_idx in grids_hit:
            return []

        # Confidence: deeper grid = more conviction; clamped to [0.10, 1.0]
        confidence = min(1.0, max(0.1, abs(grid_idx) * 0.20))

        if grid_idx < 0:
            # ------------------------------------------------------------------
            # BUY signal: price dropped to a lower grid level
            # ------------------------------------------------------------------

            # Optional RSI filter: block buys when RSI is too low (panic)
            if min_rsi_buy > 0.0 and rsi_float < min_rsi_buy:
                self._log.debug(
                    "grid_trading.buy_rsi_blocked",
                    symbol=symbol,
                    rsi=rsi_float,
                    min_rsi_buy=min_rsi_buy,
                    grid_idx=grid_idx,
                )
                return []

            # Mark this buy level as triggered; clear all sell levels (reset)
            grids_hit.add(grid_idx)
            sell_levels = {idx for idx in grids_hit if idx > 0}
            grids_hit -= sell_levels

            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.BUY,
                target_position=Decimal(str(round(position_size, 8))),
                confidence=round(confidence, 4),
                metadata={
                    "grid_idx": grid_idx,
                    "reference_price": str(ref_price),
                    "close": str(close),
                    "pct_change": round(pct_change, 6),
                    "rsi": rsi_float,
                    "grid_size_pct": grid_size_pct,
                },
            )

            self._log.info(
                "grid_trading.buy",
                symbol=symbol,
                grid_idx=grid_idx,
                confidence=signal.confidence,
                rsi=rsi_float,
                close=str(close),
            )

            return [signal]

        else:
            # ------------------------------------------------------------------
            # SELL signal: price rose to an upper grid level
            # ------------------------------------------------------------------

            # Optional RSI filter: block sells when RSI is too high (momentum)
            if max_rsi_sell < 100.0 and rsi_float > max_rsi_sell:
                self._log.debug(
                    "grid_trading.sell_rsi_blocked",
                    symbol=symbol,
                    rsi=rsi_float,
                    max_rsi_sell=max_rsi_sell,
                    grid_idx=grid_idx,
                )
                return []

            # Mark this sell level as triggered; clear all buy levels (reset)
            grids_hit.add(grid_idx)
            buy_levels = {idx for idx in grids_hit if idx < 0}
            grids_hit -= buy_levels

            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.SELL,
                target_position=Decimal(str(round(position_size, 8))),
                confidence=round(confidence, 4),
                metadata={
                    "grid_idx": grid_idx,
                    "reference_price": str(ref_price),
                    "close": str(close),
                    "pct_change": round(pct_change, 6),
                    "rsi": rsi_float,
                    "grid_size_pct": grid_size_pct,
                },
            )

            self._log.info(
                "grid_trading.sell",
                symbol=symbol,
                grid_idx=grid_idx,
                confidence=signal.confidence,
                rsi=rsi_float,
                close=str(close),
            )

            return [signal]
