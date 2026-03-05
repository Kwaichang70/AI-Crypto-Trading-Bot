"""
packages/trading/strategies/model_strategy.py
-----------------------------------------------
ML Model-Based Trading Strategy (Sprint 2 placeholder).

This strategy demonstrates the integration pattern for scikit-learn (or
compatible) models within the pluggable strategy framework.  It loads a
serialised model from disk, builds a feature vector from recent OHLCV bars,
and converts the model's prediction into a trading Signal with confidence.

Feature schema (built by ``_build_feature_vector``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The feature vector for each bar tick contains:

    Index  Feature               Description
    -----  --------------------  ------------------------------------
    0      log_return_1          1-bar log return
    1      log_return_5          5-bar cumulative log return
    2      log_return_10         10-bar cumulative log return
    3      volatility_10         10-bar rolling std of log returns
    4      volatility_20         20-bar rolling std of log returns
    5      rsi_14                14-bar Wilder RSI
    6      sma_ratio_10_50       SMA(10) / SMA(50) ratio
    7      sma_ratio_20_100      SMA(20) / SMA(100) ratio
    8      volume_ratio_10       current volume / SMA(volume, 10)
    9      high_low_range        (high - low) / close of current bar

Model contract
~~~~~~~~~~~~~~
- ``model.predict(X)``          -> array of class labels (0=SELL, 1=HOLD, 2=BUY)
- ``model.predict_proba(X)``    -> array of shape (n, 3) with class probabilities
- The model is loaded via ``joblib.load(model_path)`` in ``on_start()``.

MVP behaviour
~~~~~~~~~~~~~
When no model file is found, the strategy logs a warning on every bar and
returns HOLD.  This allows the run to proceed in paper/backtest mode so
that the pipeline integration can be tested end-to-end before a model is
trained.

Parameters
----------
model_path : str
    Filesystem path to a serialised model (joblib or pickle).
feature_window : int
    Number of historical bars required for feature extraction.
    Default 100 (must be >= 100 to cover SMA(100)).
prediction_threshold : float
    Minimum predicted-class probability to emit a signal.
    Below this threshold, the strategy returns HOLD.  Default 0.60.
position_size : float
    Target notional position in quote currency.  Default 1000.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field, field_validator

from common.models import OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["ModelStrategy"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Class label mapping — matches scikit-learn LabelEncoder convention
# ---------------------------------------------------------------------------
_LABEL_SELL = 0
_LABEL_HOLD = 1
_LABEL_BUY = 2

_LABEL_TO_DIRECTION: dict[int, SignalDirection] = {
    _LABEL_SELL: SignalDirection.SELL,
    _LABEL_HOLD: SignalDirection.HOLD,
    _LABEL_BUY: SignalDirection.BUY,
}

# ---------------------------------------------------------------------------
# Parameter schema (Pydantic validation)
# ---------------------------------------------------------------------------


class _ModelStrategyParams(BaseModel):
    """Pydantic model for ModelStrategy parameter validation."""

    model_path: str = Field(
        default="",
        description="Path to serialised model file (joblib/pickle). Empty = no model.",
    )
    feature_window: int = Field(
        default=100,
        ge=100,
        le=5000,
        description="Number of bars used for feature extraction (>= 100 for SMA-100)",
    )
    prediction_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum class probability to emit a non-HOLD signal",
    )
    position_size: float = Field(
        default=1000.0,
        gt=0.0,
        le=1_000_000.0,
        description="Target notional position in quote currency",
    )

    @field_validator("model_path")
    @classmethod
    def strip_path(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class ModelStrategy(BaseStrategy):
    """
    ML model-based trading strategy (placeholder for Sprint 2).

    Designed to load a trained scikit-learn model and use its predictions
    for signal generation.  The model receives a feature vector built from
    recent OHLCV bars and outputs a BUY/SELL/HOLD prediction with confidence.

    Parameters
    ----------
    model_path : str
        Path to serialised model (joblib/pickle).
    feature_window : int
        Number of bars for feature extraction.
    prediction_threshold : float
        Minimum confidence to emit signal.
    position_size : float
        Target position in quote currency.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="ML Model Strategy",
        version="0.1.0",
        description="ML model-based trading strategy (placeholder for Sprint 2)",
        author="trading-engine-architect",
        tags=["ml", "model", "scikit-learn", "placeholder"],
    )

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id=strategy_id, params=params)
        self._model: Any = None
        self._model_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Parameter validation
    # ------------------------------------------------------------------ #

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce parameters via the Pydantic schema."""
        validated = _ModelStrategyParams(**params)
        return validated.model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for accepted parameters."""
        return _ModelStrategyParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        """Feature window determines minimum history needed."""
        return int(self._params.get("feature_window", 100))

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def on_start(self, run_id: str) -> None:
        """
        Load the ML model from disk.

        If the model file is not found or model_path is empty, logs a
        warning and continues in placeholder mode (always returns HOLD).
        """
        super().on_start(run_id)

        model_path_str: str = self._params.get("model_path", "")
        if not model_path_str:
            self._log.warning(
                "model_strategy.no_model_path",
                msg="No model_path configured. Strategy will return HOLD on every bar. "
                    "Train and serialise a model, then set model_path to enable predictions.",
            )
            return

        model_path = Path(model_path_str)
        if not model_path.exists():
            self._log.warning(
                "model_strategy.model_not_found",
                model_path=str(model_path),
                msg="Model file not found. Strategy will return HOLD on every bar.",
            )
            return

        try:
            import joblib

            self._model = joblib.load(model_path)
            self._model_loaded = True
            self._log.info(
                "model_strategy.model_loaded",
                model_path=str(model_path),
                model_type=type(self._model).__name__,
            )
        except ImportError:
            self._log.error(
                "model_strategy.joblib_not_installed",
                msg="joblib is required to load models. Install with: pip install joblib",
            )
        except Exception as exc:
            self._log.error(
                "model_strategy.model_load_failed",
                model_path=str(model_path),
                error=str(exc),
                error_type=type(exc).__name__,
            )

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    def _build_feature_vector(self, bars: Sequence[OHLCVBar]) -> list[float]:
        """Build a 10-element feature vector from OHLCV bars.

        Delegates to data.ml_features.build_feature_vector_from_bars for
        single-source feature schema consistency between training and inference.
        See that module for the canonical feature schema documentation.
        """
        from data.ml_features import build_feature_vector_from_bars

        return build_feature_vector_from_bars(bars)

    # ------------------------------------------------------------------ #
    # Core signal generation
    # ------------------------------------------------------------------ #

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        """
        Process OHLCV bars through the ML model to generate signals.

        If no model is loaded, returns an empty list (equivalent to HOLD).
        Otherwise, builds a feature vector, runs prediction, and emits a
        Signal if the predicted-class probability exceeds the configured
        threshold.

        Parameters
        ----------
        bars : Sequence[OHLCVBar]
            OHLCV bars ordered oldest-first.

        Returns
        -------
        list[Signal]
            Zero or one Signal.
        """
        feature_window: int = self._params["feature_window"]
        prediction_threshold: float = self._params["prediction_threshold"]
        position_size: float = self._params["position_size"]

        # Warm-up check
        if len(bars) < feature_window:
            self._log.debug(
                "model_strategy.warmup",
                bars_available=len(bars),
                bars_required=feature_window,
            )
            return []

        # Placeholder mode: no model loaded
        if not self._model_loaded or self._model is None:
            self._log.debug(
                "model_strategy.no_model",
                msg="No ML model loaded. Returning HOLD.",
            )
            return []

        # Build feature vector
        feature_vector = self._build_feature_vector(bars)

        # Predict — wrap in 2D array for scikit-learn API
        try:
            import numpy as np

            X = np.array([feature_vector])  # shape (1, 10)

            # Get predicted class label
            prediction = int(self._model.predict(X)[0])

            # Get class probabilities for confidence scoring
            if hasattr(self._model, "predict_proba"):
                probas = self._model.predict_proba(X)[0]  # shape (3,)
                confidence = float(probas[prediction])
            else:
                # Model does not support predict_proba — use fixed confidence
                confidence = 0.5
                self._log.debug(
                    "model_strategy.no_predict_proba",
                    msg="Model lacks predict_proba. Using fixed confidence=0.5.",
                )
        except Exception as exc:
            self._log.error(
                "model_strategy.prediction_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

        # Map label to direction
        direction = _LABEL_TO_DIRECTION.get(prediction, SignalDirection.HOLD)

        # Apply threshold filter
        if direction == SignalDirection.HOLD:
            self._log.debug(
                "model_strategy.hold",
                prediction=prediction,
                confidence=round(confidence, 4),
            )
            return []

        if confidence < prediction_threshold:
            self._log.debug(
                "model_strategy.below_threshold",
                direction=direction.value,
                confidence=round(confidence, 4),
                threshold=prediction_threshold,
            )
            return []

        current_bar = bars[-1]

        signal = Signal(
            strategy_id=self._strategy_id,
            symbol=current_bar.symbol,
            direction=direction,
            target_position=Decimal(str(position_size)),
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            metadata={
                "model_type": type(self._model).__name__,
                "prediction_label": prediction,
                "prediction_confidence": round(confidence, 6),
                "prediction_threshold": prediction_threshold,
                "features": {
                    "log_return_1": round(feature_vector[0], 6),
                    "log_return_5": round(feature_vector[1], 6),
                    "log_return_10": round(feature_vector[2], 6),
                    "volatility_10": round(feature_vector[3], 6),
                    "volatility_20": round(feature_vector[4], 6),
                    "rsi_14": round(feature_vector[5], 6),
                    "sma_ratio_10_50": round(feature_vector[6], 6),
                    "sma_ratio_20_100": round(feature_vector[7], 6),
                    "volume_ratio_10": round(feature_vector[8], 6),
                    "high_low_range": round(feature_vector[9], 6),
                },
                "close": str(current_bar.close),
            },
        )

        self._log.info(
            "model_strategy.signal",
            direction=direction.value,
            confidence=signal.confidence,
            prediction_label=prediction,
            symbol=current_bar.symbol,
        )

        return [signal]

    def on_stop(self) -> None:
        """Release the model reference on shutdown."""
        if self._model_loaded:
            self._log.info(
                "model_strategy.model_released",
                model_type=type(self._model).__name__ if self._model else "None",
            )
        self._model = None
        self._model_loaded = False
        super().on_stop()
