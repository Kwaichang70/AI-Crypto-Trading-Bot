"""
packages/trading/strategies/model_strategy.py
-----------------------------------------------
ML Model-Based Trading Strategy (Sprint 2 placeholder, Sprint 23 hot-swap).

This strategy demonstrates the integration pattern for scikit-learn (or
compatible) models within the pluggable strategy framework.  It loads a
serialised model from disk, builds a feature vector from recent OHLCV bars,
and converts the model's prediction into a trading Signal with confidence.

Sprint 23 — Hot-swap mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ModelStrategy reads the `models/{safe_symbol}_active.json` sidecar file on
every on_bar() call. When the sidecar's version_id differs from the currently
loaded model's version_id, the model is reloaded from the path in the sidecar.
This enables RetrainingService to swap in retrained models without restarting
the strategy. If the sidecar does not exist, the strategy continues with the
originally loaded model (backward compatible).

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
model_dir : str
    Directory where the active sidecar JSON is written by RetrainingService.
    Default "models/". Used to construct the sidecar path for hot-swap.
"""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field, field_validator

from common.models import MultiTimeframeContext, OHLCVBar
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
    model_dir: str = Field(
        default="models/",
        description=(
            "Directory for the active-version sidecar JSON. "
            "Must match RetrainingService.model_dir for hot-swap to work."
        ),
    )
    trailing_stop_pct: float | None = Field(
        default=None,
        ge=0.005,
        le=0.50,
        description="Trailing stop-loss percentage (e.g. 0.03 = 3%). None to disable.",
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
    ML model-based trading strategy with hot-swap model reloading (Sprint 23).

    Designed to load a trained scikit-learn model and use its predictions
    for signal generation.  The model receives a feature vector built from
    recent OHLCV bars and outputs a BUY/SELL/HOLD prediction with confidence.

    Hot-swap
    --------
    On each on_bar() call, the strategy checks the
    `models/{safe_symbol}_active.json` sidecar file written by RetrainingService.
    If the sidecar's version_id differs from the currently loaded version, the
    model is reloaded transparently without interrupting the strategy lifecycle.

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
    model_dir : str
        Directory where sidecar JSON is located.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="ML Model Strategy",
        version="0.2.0",
        description=(
            "ML model-based trading strategy with automatic hot-swap reloading "
            "when RetrainingService publishes a new model version."
        ),
        author="trading-engine-architect",
        tags=["ml", "model", "scikit-learn", "adaptive"],
    )

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id=strategy_id, params=params)
        self._model: Any = None
        self._model_loaded: bool = False
        # Sprint 23: hot-swap tracking
        self._active_version_id: str | None = None
        self._sidecar_path: Path | None = None
        # Default path; overwritten in on_start() with the configured model_dir param.
        # Initialized here so _check_model_version() never raises AttributeError
        # if called before on_start() (e.g., in unit tests or atypical lifecycle).
        self._model_dir_path: Path = Path("models/")

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
        Load the ML model from disk and compute the sidecar path.

        If the model file is not found or model_path is empty, logs a
        warning and continues in placeholder mode (always returns HOLD).
        """
        super().on_start(run_id)

        # Sprint 23: compute sidecar path from model_dir + symbol
        # symbol is not known at on_start time (it is per-bar); we set the
        # sidecar_path lazily on first on_bar() call when we have bars[-1].symbol.
        # Store model_dir for deferred sidecar path construction.
        self._model_dir_path = Path(self._params.get("model_dir", "models/"))

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
    # Sprint 23: hot-swap check
    # ------------------------------------------------------------------ #

    def _load_model_with_timeout(self, path: str, timeout: float = 5.0) -> Any:
        """Load a model file with a timeout to prevent I/O hangs.

        Uses a thread pool to enforce the timeout since joblib.load is
        synchronous blocking I/O.
        """
        import joblib

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(joblib.load, path)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)
            raise
        finally:
            executor.shutdown(wait=False)

    def _check_model_version(self, symbol: str) -> None:
        """Check sidecar JSON for a new model version and hot-swap if changed.

        Reads the `models/{safe_symbol}_active.json` sidecar file written by
        RetrainingService. If the sidecar's version_id differs from the
        currently loaded version_id, the model is reloaded from the path in
        the sidecar. File reads are <1ms and do not require DB access.

        Parameters
        ----------
        symbol : str
            Trading pair for this bar (used to locate the sidecar file).
        """
        # Lazily construct the sidecar path on first call
        if self._sidecar_path is None:
            safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
            self._sidecar_path = self._model_dir_path / f"{safe_symbol}_active.json"

        if not self._sidecar_path.exists():
            return

        try:
            data: dict[str, Any] = json.loads(
                self._sidecar_path.read_text(encoding="utf-8")
            )
            sidecar_version_id: str | None = data.get("version_id")
            if sidecar_version_id is None or sidecar_version_id == self._active_version_id:
                return

            # Version changed — reload model
            new_model_path: str = data["model_path"]
            new_model = self._load_model_with_timeout(new_model_path)
            self._model = new_model
            self._model_loaded = True
            self._active_version_id = sidecar_version_id

            self._log.info(
                "model_strategy.hot_swap",
                version_id=sidecar_version_id,
                model_path=new_model_path,
                model_type=type(new_model).__name__,
            )
        except Exception as exc:
            self._log.warning(
                "model_strategy.hot_swap_failed",
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

    def on_bar(self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None) -> list[Signal]:
        """
        Process OHLCV bars through the ML model to generate signals.

        Sprint 23: calls _check_model_version() at the top to hot-swap
        the model if RetrainingService has published a new version.

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

        # Sprint 23: hot-swap check (before warmup — no bars needed for file read)
        if bars:
            self._check_model_version(symbol=bars[-1].symbol)

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
                "active_version_id": self._active_version_id,
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
            active_version_id=self._active_version_id,
        )

        return [signal]

    def on_stop(self) -> None:
        """Release the model reference on shutdown."""
        if self._model_loaded:
            self._log.info(
                "model_strategy.model_released",
                model_type=type(self._model).__name__ if self._model else "None",
                active_version_id=self._active_version_id,
            )
        self._model = None
        self._model_loaded = False
        self._active_version_id = None
        self._sidecar_path = None
        super().on_stop()
