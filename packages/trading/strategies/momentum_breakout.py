"""
packages/trading/strategies/momentum_breakout.py
-------------------------------------------------
Momentum Breakout strategy — a BUY-only trend entry designed for ATR bracket
exits (fixed/ATR stop-loss + take-profit handled by the engine).

Thesis
~~~~~~
Crypto exhibits time-series momentum: sustained breakouts tend to continue.
This strategy buys when price makes a *new* `lookback`-bar high while in an
uptrend (close above a long SMA), and delegates the exit entirely to the
engine's bracket manager.  Paired with an **asymmetric** ATR bracket (a tight
stop and a wider target) the position cuts losers quickly while letting the
fat-tailed trend winners run — the structure that gives positive expectancy.

Validation (Sprint 51 Cycle 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default config (lookback 40, trend SMA 100, SL 1.5xATR, TP 3.0xATR, daily)
was selected by walk-forward out-of-sample search on BTC/ETH and confirmed to
remain profitable on TWO fully held-out symbol universes never used in the
search (SOL/BNB/XRP, then ADA/AVAX/LINK/DOGE/LTC/DOT/ATOM/TRX) — median OOS
profit factor 1.2-2.0 net of fees + slippage across 13 assets, including the
2022 bear market.  Short-lookback variants overfit and were rejected.

This strategy emits **BUY only** — it never sells.  Always pair it with a
bracket config (the schema defaults provide the validated ATR bracket).
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
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["MomentumBreakoutStrategy"]

logger = structlog.get_logger(__name__)


class _MomentumBreakoutParams(BaseModel):
    """Parameter schema.  Bracket fields render in the UI and are consumed at
    the engine level (stripped before strategy validation); the defaults
    encode the walk-forward-validated ATR bracket."""

    lookback: int = Field(
        default=40, ge=2, le=400,
        description="Breakout lookback: buy on a new high over this many bars.",
    )
    trend_sma_period: int | None = Field(
        default=100, ge=2, le=1000,
        description="Only buy breakouts while close > SMA(this). Null disables.",
    )
    position_size: float = Field(
        default=2000.0, gt=0.0, le=1_000_000.0,
        description="Target notional position in quote currency.",
    )

    # --- engine-level bracket exit (validated defaults: asymmetric ATR) ---
    bracket_mode: Literal["fixed", "atr"] = Field(
        default="atr",
        description="Bracket basis: 'atr' multiples (recommended) or 'fixed' %.",
    )
    bracket_atr_sl_multiplier: float | None = Field(
        default=1.5, ge=0.1, le=20.0,
        description="ATR-mode stop distance (x ATR below entry).",
    )
    bracket_atr_tp_multiplier: float | None = Field(
        default=3.0, ge=0.1, le=20.0,
        description="ATR-mode take-profit distance (x ATR above entry). Keep > SL.",
    )
    bracket_atr_period: int = Field(
        default=14, ge=2, le=500,
        description="ATR look-back used for ATR-mode brackets.",
    )
    bracket_stop_loss_pct: float | None = Field(
        default=None, ge=0.001, le=0.95,
        description="Fixed-mode stop-loss fraction of entry (used if mode=fixed).",
    )
    bracket_take_profit_pct: float | None = Field(
        default=None, ge=0.001, le=0.95,
        description="Fixed-mode take-profit fraction of entry (used if mode=fixed).",
    )


class MomentumBreakoutStrategy(BaseStrategy):
    """BUY-only Donchian-style breakout entry; exits delegated to brackets."""

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Momentum Breakout",
        version="1.0.0",
        description=(
            "BUY-only trend breakout (new N-bar high in an uptrend) for "
            "asymmetric ATR bracket exits"
        ),
        author="quant-strategy-analyst",
        tags=["momentum", "trend", "breakout", "stop-loss", "take-profit", "bracket"],
    )

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return _MomentumBreakoutParams(**params).model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return normalise_nullable_json_schema(_MomentumBreakoutParams.model_json_schema())

    @property
    def min_bars_required(self) -> int:
        lookback = int(self._params["lookback"])
        trend = self._params.get("trend_sma_period") or 0
        return max(lookback, int(trend)) + 3

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        if len(bars) < self.min_bars_required:
            self._log.debug(
                "momentum_breakout.warmup",
                bars_available=len(bars),
                bars_required=self.min_bars_required,
            )
            return []

        lookback: int = self._params["lookback"]
        trend_sma_period: int | None = self._params.get("trend_sma_period")
        position_size: float = self._params["position_size"]

        highs = [b.high for b in bars]
        close = bars[-1].close
        prev_close = bars[-2].close

        # Highest high of the prior `lookback` bars, for the current and the
        # previous bar.  A *new* breakout is one where this bar closes above
        # its prior-high but the previous bar did not (event, not a level).
        prior_high = max(highs[-(lookback + 1):-1])
        prior_high_prev = max(highs[-(lookback + 2):-2])
        new_breakout = close > prior_high and prev_close <= prior_high_prev
        if not new_breakout:
            return []

        # Trend filter: only buy breakouts while above the long SMA.
        if trend_sma_period is not None:
            sma = sum(b.close for b in bars[-trend_sma_period:]) / Decimal(trend_sma_period)
            if close <= sma:
                self._log.debug(
                    "momentum_breakout.trend_filter_block",
                    close=str(close), sma=str(sma),
                )
                return []

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=bars[-1].symbol,
            direction=SignalDirection.BUY,
            target_position=Decimal(str(position_size)),
            confidence=1.0,
            metadata={
                "trigger": "momentum_breakout",
                "lookback": lookback,
                "prior_high": str(prior_high),
                "close": str(close),
            },
        )
        self._log.info(
            "momentum_breakout.signal",
            direction="BUY", symbol=bars[-1].symbol,
            close=str(close), prior_high=str(prior_high),
        )
        return [signal]
