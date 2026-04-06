"""
packages/trading/strategies/dca_rsi_hybrid.py
----------------------------------------------
DCA + RSI Hybrid Strategy.

Combines systematic Dollar Cost Averaging with RSI-based dip buying and
profit taking.  Unlike pure RSI strategies that wait for extreme RSI levels,
this strategy generates BUY signals at a regular cadence (every
``dca_interval_bars`` bars) and supplements them with RSI intelligence:

- **RSI Boost:** Increase buy size when RSI is low (dip buying).
- **RSI Skip:** Suppress the periodic DCA buy when RSI is very high.
- **Profit Taking:** Emit a partial SELL when RSI is overbought AND the
  strategy is within a bar that is *not* a DCA bar, giving the engine a
  chance to capture gains without waiting for the next DCA cycle.
- **FGI Integration:** Apply the same Fear & Greed Index confidence
  adjustment as the RSI mean-reversion strategy.
- **CoinGecko Integration:** BTC dominance and market cap momentum signals.
- **FRED Macro Integration:** Yield curve regime adjustment.
- **Whale Alert Integration:** On-chain flow accumulation/distribution signal.

Parameters
----------
dca_interval_bars : int
    Buy every N bars (e.g. 16 bars on 15 m = every 4 hours).  Default 16.
dca_amount : float
    Base DCA amount in quote currency per buy.  Default 50.0.
rsi_period : int
    RSI calculation period.  Default 14.
rsi_boost_threshold : float
    RSI below this value multiplies the buy amount.  Default 40.0.
rsi_boost_multiplier : float
    Multiplier applied to ``dca_amount`` when RSI < ``rsi_boost_threshold``.
    Default 2.0.
rsi_skip_threshold : float
    RSI above this value suppresses the periodic DCA buy.  Default 75.0.
take_profit_rsi : float
    Trigger a partial SELL when RSI crosses above this level.  Default 70.0.
take_profit_pct : float
    Fraction of ``position_size`` to sell on take-profit.  Must be in
    (0, 1].  Default 0.5 (50 %).
position_size : float
    Maximum total notional position in quote currency.  Acts as the
    reference size for take-profit quantity calculation.  Default 1000.0.
trailing_stop_pct : float | None
    Trailing stop-loss percentage (e.g. 0.03 = 3 %).  None to disable.

RSI calculation
~~~~~~~~~~~~~~~
Delegates to ``_compute_rsi`` from ``rsi_mean_reversion`` (Wilder's
exponential smoothing method) to avoid code duplication.

Confidence scoring
~~~~~~~~~~~~~~~~~~
DCA BUY confidence scales with RSI depth below 50::

    rsi < 30  => 0.9
    rsi < 40  => 0.7
    rsi < 50  => 0.5
    rsi >= 50 => 0.3

Take-profit SELL confidence scales with how far RSI is above
``take_profit_rsi``::

    confidence = (rsi - take_profit_rsi) / (100 - take_profit_rsi)

Both clamped to [0.1, 1.0] before signal boosts.
"""

from __future__ import annotations

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

__all__ = ["DCARSIHybridStrategy"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------


class _DCAParams(BaseModel):
    """Pydantic model for DCA + RSI Hybrid parameter validation."""

    dca_interval_bars: int = Field(
        default=16,
        ge=1,
        le=10_000,
        description="Buy every N bars",
    )
    dca_amount: float = Field(
        default=50.0,
        gt=0.0,
        le=1_000_000.0,
        description="Base DCA amount in quote currency per buy",
    )
    rsi_period: int = Field(
        default=14,
        ge=2,
        le=500,
        description="RSI look-back period",
    )
    rsi_boost_threshold: float = Field(
        default=40.0,
        ge=1.0,
        le=99.0,
        description="RSI below this = increase buy size",
    )
    rsi_boost_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiply dca_amount when RSI < rsi_boost_threshold",
    )
    rsi_skip_threshold: float = Field(
        default=75.0,
        ge=1.0,
        le=99.0,
        description="RSI above this = skip DCA buy",
    )
    take_profit_rsi: float = Field(
        default=70.0,
        ge=1.0,
        le=99.0,
        description="Take partial profit when RSI above this level",
    )
    take_profit_pct: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of position_size to sell on take-profit",
    )
    position_size: float = Field(
        default=1000.0,
        gt=0.0,
        le=1_000_000.0,
        description="Max total notional position in quote currency",
    )
    trailing_stop_pct: float | None = Field(
        default=None,
        ge=0.005,
        le=0.50,
        description="Trailing stop-loss percentage (e.g. 0.03 = 3 %). None to disable.",
    )

    @model_validator(mode="after")
    def _thresholds_coherent(self) -> _DCAParams:
        """Ensure boost < skip thresholds with a reasonable gap."""
        if self.rsi_boost_threshold >= self.rsi_skip_threshold:
            raise ValueError(
                f"rsi_boost_threshold ({self.rsi_boost_threshold}) must be "
                f"< rsi_skip_threshold ({self.rsi_skip_threshold})"
            )
        if self.take_profit_rsi >= self.rsi_skip_threshold:
            raise ValueError(
                f"take_profit_rsi ({self.take_profit_rsi}) must be "
                f"< rsi_skip_threshold ({self.rsi_skip_threshold})"
            )
        return self


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class DCARSIHybridStrategy(BaseStrategy):
    """
    DCA + RSI Hybrid -- systematic accumulation with RSI intelligence.

    Generates BUY signals at a regular cadence (every ``dca_interval_bars``
    bars) while using RSI to:
    - Increase buy size on dips (RSI < ``rsi_boost_threshold``).
    - Skip buys when the market is overbought (RSI > ``rsi_skip_threshold``).
    - Take partial profit on non-DCA bars when RSI is elevated
      (RSI > ``take_profit_rsi``).

    Attributes
    ----------
    metadata : StrategyMetadata
        Class-level strategy metadata for registry introspection.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="DCA + RSI Hybrid",
        version="1.0.0",
        description=(
            "Systematic DCA accumulation with RSI dip-buying boost, "
            "overbought skip, and partial take-profit"
        ),
        author="python-backend-specialist",
        tags=["dca", "rsi", "hybrid", "accumulation"],
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self, run_id: str) -> None:
        """Initialise bar counter and reset state for a new run."""
        super().on_start(run_id)
        self._bar_counter: int = 0
        self._log.info("dca_rsi_hybrid.started", run_id=run_id)

    def on_stop(self) -> None:
        """Reset stateful counters on run stop."""
        self._bar_counter = 0
        super().on_stop()

    # ------------------------------------------------------------------
    # Fear & Greed Index confidence modifier (mirrors RSI strategy)
    # ------------------------------------------------------------------

    @staticmethod
    def _fgi_confidence_boost(fgi: int, direction: SignalDirection) -> float:
        """
        Return a confidence delta based on the Fear & Greed Index.

        DCA buys in extreme fear = stronger accumulation signal.
        Take-profit sells in extreme greed = more conviction.

        Parameters
        ----------
        fgi : int
            Fear & Greed Index value in [0, 100].
        direction : SignalDirection
            The signal direction (BUY or SELL).

        Returns
        -------
        float
            Confidence delta in [-0.10, +0.10].
        """
        if direction == SignalDirection.BUY:
            if fgi <= 24:
                return 0.10   # Extreme fear: strong accumulation opportunity
            elif fgi <= 44:
                return 0.05
            elif fgi <= 55:
                return 0.0
            elif fgi <= 75:
                return -0.05  # Greed: less conviction for DCA buy
            else:
                return -0.10  # Extreme greed: least conviction for BUY

        elif direction == SignalDirection.SELL:
            if fgi >= 76:
                return 0.10   # Extreme greed: strong take-profit opportunity
            elif fgi >= 56:
                return 0.05
            elif fgi >= 45:
                return 0.0
            elif fgi >= 25:
                return -0.05  # Fear: less conviction for take-profit
            else:
                return -0.10  # Extreme fear: least conviction for SELL

        return 0.0

    # ------------------------------------------------------------------
    # CoinGecko BTC dominance confidence modifier
    # ------------------------------------------------------------------

    @staticmethod
    def _btc_dominance_boost(dominance: float | None, direction: SignalDirection) -> float:
        """
        Return a confidence delta based on BTC dominance from CoinGecko.

        High BTC dominance (>55%) signals risk-off for altcoins — whales
        are rotating into BTC and out of alts.  Low dominance (<45%)
        signals alt-season — altcoins outperform BTC in these regimes.

        Note: This boost is only meaningful for non-BTC trading pairs.
        For BTC itself the signal is directionally neutral.

        Parameters
        ----------
        dominance : float | None
            BTC market cap dominance percentage in [0, 100], or None if
            not available.
        direction : SignalDirection
            The signal direction (BUY or SELL).

        Returns
        -------
        float
            Confidence delta in [-0.05, +0.05].  Returns 0.0 if
            ``dominance`` is None (signal not available).
        """
        if dominance is None:
            return 0.0
        if direction == SignalDirection.BUY:
            if dominance < 45.0:
                return 0.05    # Alt rally regime: accumulation favourable
            if dominance > 55.0:
                return -0.05   # BTC dominance surge: risk-off for alts
        elif direction == SignalDirection.SELL:
            if dominance < 45.0:
                return -0.05   # Alt rally: less conviction for take-profit
            if dominance > 55.0:
                return 0.05    # Risk-off: more conviction for profit-taking
        return 0.0

    # ------------------------------------------------------------------
    # FRED yield curve macro confidence modifier
    # ------------------------------------------------------------------

    @staticmethod
    def _macro_boost(yield_spread: float | None, direction: SignalDirection) -> float:
        """
        Return a confidence delta based on the 10Y-2Y yield curve spread.

        The yield curve spread is a well-established macro indicator:
        - Deep inversion (< -0.5%) historically precedes recession and
          broad risk-asset drawdowns including crypto.
        - Healthy positive spread (> 1.0%) indicates a growth-favouring
          macro environment where risk assets tend to perform well.

        Parameters
        ----------
        yield_spread : float | None
            10-Year minus 2-Year Treasury yield spread in percent.
            Negative = inverted curve.  None if not available.
        direction : SignalDirection
            The signal direction (BUY or SELL).

        Returns
        -------
        float
            Confidence delta in [-0.05, +0.05].  Returns 0.0 if
            ``yield_spread`` is None (signal not available).
        """
        if yield_spread is None:
            return 0.0
        if direction == SignalDirection.BUY:
            if yield_spread < -0.5:
                return -0.05  # Deep inversion: recession risk, reduce BUY conviction
            if yield_spread > 1.0:
                return 0.05   # Healthy curve: risk-on macro, increase BUY conviction
        elif direction == SignalDirection.SELL:
            if yield_spread < -0.5:
                return 0.05   # Recession signal: more conviction for taking profit
            if yield_spread > 1.0:
                return -0.05  # Healthy macro: less urgency to take profit
        return 0.0

    # ------------------------------------------------------------------
    # Whale Alert on-chain flow confidence modifier
    # ------------------------------------------------------------------

    @staticmethod
    def _whale_flow_boost(net_flow: float | None, direction: SignalDirection) -> float:
        """
        Return a confidence delta based on Whale Alert net exchange flow.

        Large on-chain flows indicate institutional activity:
        - Heavy outflow from exchanges (negative net_flow) → whales moving
          assets to cold storage.  Historically bullish (accumulation).
        - Heavy inflow to exchanges (positive net_flow) → whales depositing
          to sell.  Historically bearish (distribution / sell pressure).

        Parameters
        ----------
        net_flow : float | None
            Net USD flow: positive = to exchanges, negative = from exchanges.
            None if not available.
        direction : SignalDirection
            The signal direction (BUY or SELL).

        Returns
        -------
        float
            Confidence delta in [-0.05, +0.05].  Returns 0.0 if
            ``net_flow`` is None (signal not available).
        """
        if net_flow is None:
            return 0.0
        if direction == SignalDirection.BUY:
            if net_flow < -5_000_000:
                return 0.05    # Heavy outflow: whale accumulation signal
            if net_flow > 5_000_000:
                return -0.05   # Heavy inflow: distribution / sell pressure
        elif direction == SignalDirection.SELL:
            if net_flow < -5_000_000:
                return -0.05   # Accumulation: less conviction for selling
            if net_flow > 5_000_000:
                return 0.05    # Distribution: more conviction for taking profit
        return 0.0

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce parameters via the Pydantic schema."""
        validated = _DCAParams(**params)
        return validated.model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for accepted parameters."""
        return _DCAParams.model_json_schema()

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
        Process a batch of OHLCV bars and emit DCA or take-profit signals.

        Called on every bar. The bar counter is incremented unconditionally
        so that the DCA cadence is stable regardless of warm-up skip.

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            OHLCV bars ordered oldest-first. The last element is current.
        mtf_context : MultiTimeframeContext | None
            Optional higher-timeframe context including Fear & Greed Index,
            CoinGecko dominance, FRED yield curve, and Whale Alert flow.

        Returns
        -------
        list[Signal]
            A single-element list (BUY or SELL) or empty list (HOLD).
        """
        rsi_period: int = self._params["rsi_period"]
        dca_interval_bars: int = self._params["dca_interval_bars"]
        dca_amount: float = self._params["dca_amount"]
        rsi_boost_threshold: float = self._params["rsi_boost_threshold"]
        rsi_boost_multiplier: float = self._params["rsi_boost_multiplier"]
        rsi_skip_threshold: float = self._params["rsi_skip_threshold"]
        take_profit_rsi: float = self._params["take_profit_rsi"]
        take_profit_pct: float = self._params["take_profit_pct"]
        position_size: float = self._params["position_size"]

        # Increment bar counter BEFORE the warm-up guard so the cadence is
        # preserved even during the warm-up period.
        self._bar_counter += 1
        is_dca_bar = (self._bar_counter % dca_interval_bars) == 0

        # Warm-up guard: Wilder's smoothing needs rsi_period * 3 + 1 bars.
        min_bars = rsi_period * 3 + 1
        if len(bars) < min_bars:
            self._log.debug(
                "dca_rsi_hybrid.warmup",
                bars_available=len(bars),
                bars_required=min_bars,
            )
            return []

        # Compute current RSI from the last min_bars closes.
        tail_closes: list[Decimal] = [bar.close for bar in bars[-min_bars:]]
        rsi_curr = _compute_rsi(tail_closes, rsi_period)
        rsi_float = float(rsi_curr)

        current_bar = bars[-1]

        # Extract external signal values once; used for both BUY and SELL paths.
        fgi_value: int | None = None
        btc_dominance: float | None = None
        yield_curve_spread: float | None = None
        whale_net_flow: float | None = None
        if mtf_context is not None:
            if mtf_context.fear_greed_index is not None:
                fgi_value = mtf_context.fear_greed_index
            btc_dominance = mtf_context.btc_dominance
            yield_curve_spread = mtf_context.yield_curve_spread
            whale_net_flow = mtf_context.whale_net_flow

        # ------------------------------------------------------------------
        # Path A: DCA bar — consider buying
        # ------------------------------------------------------------------
        if is_dca_bar:
            # Skip the DCA buy when RSI is too high (overbought market).
            if rsi_float > rsi_skip_threshold:
                self._log.debug(
                    "dca_rsi_hybrid.dca_skipped",
                    reason="rsi_above_skip_threshold",
                    rsi=rsi_float,
                    skip_threshold=rsi_skip_threshold,
                    bar_counter=self._bar_counter,
                )
                return []

            # Determine buy amount: boost on dips, base amount otherwise.
            buy_amount = dca_amount
            if rsi_float < rsi_boost_threshold:
                buy_amount = dca_amount * rsi_boost_multiplier

            # Confidence scales with RSI depth.
            if rsi_float < 30.0:
                confidence = 0.9
            elif rsi_float < 40.0:
                confidence = 0.7
            elif rsi_float < 50.0:
                confidence = 0.5
            else:
                confidence = 0.3

            # Apply Fear & Greed Index boost.
            if fgi_value is not None:
                confidence += DCARSIHybridStrategy._fgi_confidence_boost(
                    fgi_value, SignalDirection.BUY
                )

            # Apply BTC dominance boost.
            confidence += DCARSIHybridStrategy._btc_dominance_boost(
                btc_dominance, SignalDirection.BUY
            )

            # Apply yield curve macro boost.
            confidence += DCARSIHybridStrategy._macro_boost(
                yield_curve_spread, SignalDirection.BUY
            )

            # Apply whale flow boost.
            confidence += DCARSIHybridStrategy._whale_flow_boost(
                whale_net_flow, SignalDirection.BUY
            )

            confidence = min(1.0, max(0.1, confidence))

            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=current_bar.symbol,
                direction=SignalDirection.BUY,
                target_position=Decimal(str(round(buy_amount, 8))),
                confidence=round(confidence, 4),
                metadata={
                    "rsi": rsi_float,
                    "rsi_period": rsi_period,
                    "dca_amount": dca_amount,
                    "buy_amount": buy_amount,
                    "bar_counter": self._bar_counter,
                    "is_dca_bar": True,
                    "fear_greed_index": fgi_value,
                    "btc_dominance": btc_dominance,
                    "yield_curve_spread": yield_curve_spread,
                    "whale_net_flow": whale_net_flow,
                },
            )

            self._log.info(
                "dca_rsi_hybrid.buy",
                confidence=signal.confidence,
                rsi=rsi_float,
                buy_amount=buy_amount,
                bar_counter=self._bar_counter,
                symbol=current_bar.symbol,
            )

            return [signal]

        # ------------------------------------------------------------------
        # Path B: Non-DCA bar — consider taking profit
        # ------------------------------------------------------------------
        if rsi_float > take_profit_rsi:
            # Confidence proportional to how far RSI is above the threshold.
            ceiling = 100.0 - take_profit_rsi
            depth = (rsi_float - take_profit_rsi) / ceiling if ceiling > 0 else 0.0
            confidence = min(1.0, max(0.1, depth))

            # Apply Fear & Greed Index boost.
            if fgi_value is not None:
                confidence += DCARSIHybridStrategy._fgi_confidence_boost(
                    fgi_value, SignalDirection.SELL
                )

            # Apply BTC dominance boost.
            confidence += DCARSIHybridStrategy._btc_dominance_boost(
                btc_dominance, SignalDirection.SELL
            )

            # Apply yield curve macro boost.
            confidence += DCARSIHybridStrategy._macro_boost(
                yield_curve_spread, SignalDirection.SELL
            )

            # Apply whale flow boost.
            confidence += DCARSIHybridStrategy._whale_flow_boost(
                whale_net_flow, SignalDirection.SELL
            )

            confidence = min(1.0, max(0.1, confidence))

            sell_amount = position_size * take_profit_pct

            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=current_bar.symbol,
                direction=SignalDirection.SELL,
                target_position=Decimal(str(round(sell_amount, 8))),
                confidence=round(confidence, 4),
                metadata={
                    "rsi": rsi_float,
                    "rsi_period": rsi_period,
                    "take_profit_rsi": take_profit_rsi,
                    "sell_amount": sell_amount,
                    "bar_counter": self._bar_counter,
                    "is_dca_bar": False,
                    "fear_greed_index": fgi_value,
                    "btc_dominance": btc_dominance,
                    "yield_curve_spread": yield_curve_spread,
                    "whale_net_flow": whale_net_flow,
                },
            )

            self._log.info(
                "dca_rsi_hybrid.take_profit",
                confidence=signal.confidence,
                rsi=rsi_float,
                sell_amount=sell_amount,
                bar_counter=self._bar_counter,
                symbol=current_bar.symbol,
            )

            return [signal]

        # ------------------------------------------------------------------
        # HOLD: not a DCA bar and RSI not triggering take-profit
        # ------------------------------------------------------------------
        self._log.debug(
            "dca_rsi_hybrid.hold",
            rsi=rsi_float,
            bar_counter=self._bar_counter,
        )
        return []
