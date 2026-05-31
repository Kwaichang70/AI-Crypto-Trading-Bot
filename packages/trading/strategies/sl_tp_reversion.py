"""
packages/trading/strategies/sl_tp_reversion.py
-----------------------------------------------
SL/TP Reversion Strategy — a dip-buying mean-reversion entry designed to be
paired with the engine-level bracket exit (fixed/ATR stop-loss + take-profit).

Design philosophy
~~~~~~~~~~~~~~~~~~
Unlike the other strategies, this one deliberately emits **only BUY signals**.
It has *no internal sell logic* — every exit is delegated to the
``BracketExitManager`` (configured per run via ``stop_loss_pct`` /
``take_profit_pct`` or the ATR multiples).  This makes the strategy a clean
testbed for "trade with a stop-loss and a take-profit": the entry decides
*when* to be in the market, the bracket decides *how* to get out.

Entry logic (Connors RSI-2 style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Compute a short-period RSI (default period 2).
2. Optional trend filter: only buy dips while the market is in an uptrend,
   i.e. ``close > SMA(trend_sma_period)`` (default 200).  Set
   ``trend_sma_period`` to null to disable the filter.
3. BUY when ``RSI < entry_threshold`` (default 10) and the filter passes.

The strategy holds at most one entry intent per bar and never sells; the
bracket manager closes the position when price hits the stop or target.

Confidence
~~~~~~~~~~
Deeper oversold readings yield higher confidence::

    confidence = clamp((entry_threshold - rsi) / entry_threshold, 0.1, 1.0)
"""
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar, Literal

import structlog
from pydantic import BaseModel, Field

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading._schema_utils import normalise_nullable_json_schema
from trading.models import Signal
from trading.strategies.rsi_mean_reversion import _compute_rsi
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["SLTPReversionStrategy"]

logger = structlog.get_logger(__name__)

# RSI warm-up: Wilder smoothing needs ~3x the period to wash out the SMA seed.
_WILDER_WARMUP_MULTIPLIER = 3


class _SLTPReversionParams(BaseModel):
    """Pydantic parameter schema for the SL/TP reversion strategy.

    The bracket-exit fields (``stop_loss_pct`` … ``atr_period``) are declared
    here purely so the dashboard renders inputs for them.  They are consumed
    at the engine level (stripped from strategy params before validation);
    the strategy itself ignores them.
    """

    # --- entry parameters ---
    rsi_period: int = Field(default=2, ge=2, le=50, description="Short RSI period")
    entry_threshold: float = Field(
        default=10.0, ge=1.0, le=49.0,
        description="Buy when RSI falls below this oversold level",
    )
    trend_sma_period: int | None = Field(
        default=200, ge=2, le=1000,
        description="Only buy dips while close > SMA(this). Null disables the filter.",
    )
    position_size: float = Field(
        default=1000.0, gt=0.0, le=1_000_000.0,
        description="Target notional position in quote currency",
    )

    # --- engine-level bracket exit (rendered for the UI; consumed by engine) ---
    # All bracket fields carry a ``bracket_`` prefix so they are namespaced
    # away from any strategy's native params and routed to the engine-level
    # BracketExitManager (stripped from strategy_params before validation).
    bracket_stop_loss_pct: float | None = Field(
        default=None, ge=0.001, le=0.95,
        description="Fixed stop-loss as a fraction of entry (e.g. 0.02 = 2%).",
    )
    bracket_take_profit_pct: float | None = Field(
        default=None, ge=0.001, le=0.95,
        description="Fixed take-profit as a fraction of entry (e.g. 0.04 = 4%).",
    )
    bracket_mode: Literal["fixed", "atr"] = Field(
        default="fixed",
        description="Bracket basis: 'fixed' percentages or 'atr' multiples.",
    )
    bracket_atr_sl_multiplier: float | None = Field(
        default=None, ge=0.1, le=20.0,
        description="ATR-mode stop distance (x ATR below entry).",
    )
    bracket_atr_tp_multiplier: float | None = Field(
        default=None, ge=0.1, le=20.0,
        description="ATR-mode take-profit distance (x ATR above entry).",
    )
    bracket_atr_period: int = Field(
        default=14, ge=2, le=500,
        description="ATR look-back used for ATR-mode brackets.",
    )


class SLTPReversionStrategy(BaseStrategy):
    """
    Dip-buying mean-reversion entry with exits delegated to bracket SL/TP.

    Emits BUY signals only; the engine's BracketExitManager closes positions
    at the configured stop-loss / take-profit levels.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="SL/TP Reversion",
        version="1.0.0",
        description=(
            "RSI-2 dip-buying entry designed for fixed/ATR stop-loss + "
            "take-profit bracket exits"
        ),
        author="trading-engine-architect",
        tags=["mean-reversion", "rsi", "stop-loss", "take-profit", "bracket"],
    )

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return _SLTPReversionParams(**params).model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return normalise_nullable_json_schema(_SLTPReversionParams.model_json_schema())

    @property
    def min_bars_required(self) -> int:
        # The strategy itself uses only RSI + the optional trend SMA.  ATR
        # warmup (when ATR brackets are configured) is the engine's concern,
        # not the strategy's, so it is intentionally excluded here.
        rsi_warmup = int(self._params["rsi_period"]) * _WILDER_WARMUP_MULTIPLIER + 2
        trend = self._params.get("trend_sma_period") or 0
        return max(rsi_warmup, int(trend)) + 1

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        rsi_period: int = self._params["rsi_period"]
        entry_threshold: float = self._params["entry_threshold"]
        trend_sma_period: int | None = self._params.get("trend_sma_period")
        position_size: float = self._params["position_size"]

        if len(bars) < self.min_bars_required:
            self._log.debug(
                "sl_tp_reversion.warmup",
                bars_available=len(bars),
                bars_required=self.min_bars_required,
            )
            return []

        closes: list[Decimal] = [bar.close for bar in bars]
        current_bar = bars[-1]

        # Short-period RSI over a converged window.
        rsi_window = rsi_period * _WILDER_WARMUP_MULTIPLIER + 2
        rsi_value = float(_compute_rsi(closes[-rsi_window:], rsi_period))

        # Optional trend filter: only buy dips inside an uptrend.
        if trend_sma_period is not None:
            sma = sum(closes[-trend_sma_period:]) / Decimal(trend_sma_period)
            if current_bar.close <= sma:
                self._log.debug(
                    "sl_tp_reversion.trend_filter_block",
                    close=str(current_bar.close),
                    sma=str(sma),
                    rsi=rsi_value,
                )
                return []

        if rsi_value >= entry_threshold:
            return []

        # Oversold dip in an uptrend -> BUY.  No SELL ever (bracket exits).
        depth = (entry_threshold - rsi_value) / entry_threshold if entry_threshold > 0 else 0.0
        confidence = min(1.0, max(0.1, depth))

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=current_bar.symbol,
            direction=SignalDirection.BUY,
            target_position=Decimal(str(position_size)),
            confidence=round(confidence, 4),
            metadata={
                "rsi": rsi_value,
                "rsi_period": rsi_period,
                "entry_threshold": entry_threshold,
                "trend_sma_period": trend_sma_period,
                "close": str(current_bar.close),
            },
        )

        self._log.info(
            "sl_tp_reversion.signal",
            direction="BUY",
            confidence=signal.confidence,
            rsi=rsi_value,
            symbol=current_bar.symbol,
        )
        return [signal]
