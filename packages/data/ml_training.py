"""
packages/data/ml_training.py
------------------------------
ML model training pipeline for ModelStrategy.

Trains a RandomForestClassifier using the canonical 10-element feature schema.
Label encoding: SELL=0, HOLD=1, BUY=2 (matches ModelStrategy._LABEL_* constants).

Requires scikit-learn>=1.5 and joblib>=1.4, imported lazily inside methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

__all__ = ["ModelTrainer"]

logger = structlog.get_logger(__name__)

LABEL_SELL: int = 0
LABEL_HOLD: int = 1
LABEL_BUY: int = 2

_FEATURE_COLS: list[str] = [
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

_MIN_SAMPLES: int = 50
_MIN_STRATIFY_SAMPLES_PER_CLASS: int = 10


class ModelTrainer:
    """Train a RandomForestClassifier for ModelStrategy signal prediction.

    Parameters
    ----------
    model_dir : str
        Directory where models are saved/loaded. Created on save_model().
    """

    def __init__(self, model_dir: str = "models/") -> None:
        self._model_dir = Path(model_dir)
        self._model: Any = None
        self._log = logger.bind(component="model_trainer")

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and label vector y from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
        horizon : int
            Number of bars to look ahead for labeling.
        threshold : float
            Minimum absolute return for BUY/SELL labels.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (X, y) where X has shape (n_samples, 10) and y has shape (n_samples,).
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {threshold}")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        from data.ml_features import build_feature_matrix

        features_df: pd.DataFrame = build_feature_matrix(df)

        future_close: pd.Series = df["close"].shift(-horizon)
        future_return: pd.Series = (future_close - df["close"]) / df["close"]

        labels: pd.Series = pd.Series(LABEL_HOLD, index=df.index, dtype=int)
        labels.loc[future_return > threshold] = LABEL_BUY
        labels.loc[future_return < -threshold] = LABEL_SELL

        valid_mask: pd.Series = (
            features_df[_FEATURE_COLS].notna().all(axis=1)
            & future_return.notna()
        )

        X: np.ndarray = features_df.loc[valid_mask, _FEATURE_COLS].values
        y: np.ndarray = labels[valid_mask].values

        self._log.info(
            "training.dataset_prepared",
            total_bars=len(df),
            valid_samples=len(X),
            buy_count=int((y == LABEL_BUY).sum()),
            hold_count=int((y == LABEL_HOLD).sum()),
            sell_count=int((y == LABEL_SELL).sum()),
        )

        if len(X) < _MIN_SAMPLES:
            raise ValueError(
                f"Dataset has only {len(X)} valid samples after NaN removal. "
                f"Need at least {_MIN_SAMPLES}. Provide more data or reduce horizon."
            )

        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train a RandomForestClassifier on the prepared dataset.

        Returns
        -------
        dict[str, Any]
            Training metrics: accuracy, train/test samples, feature importances,
            classification report.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got X={len(X)}, y={len(y)}")
        if len(X) < _MIN_SAMPLES:
            raise ValueError(f"Need at least {_MIN_SAMPLES} samples, got {len(X)}")

        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
        use_stratify = (
            len(unique_classes) > 1
            and min_class_count >= _MIN_STRATIFY_SAMPLES_PER_CLASS
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if use_stratify else None,
        )

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._model.fit(X_train, y_train)

        y_pred: np.ndarray = self._model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        importances: dict[str, float] = dict(
            zip(_FEATURE_COLS, [float(v) for v in self._model.feature_importances_])
        )

        report: dict[str, Any] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0,
        )
        report.pop("accuracy", None)

        metrics: dict[str, Any] = {
            "accuracy": round(accuracy, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importances": importances,
            "classification_report": report,
        }

        self._log.info(
            "training.completed",
            accuracy=metrics["accuracy"],
            train_samples=metrics["train_samples"],
            test_samples=metrics["test_samples"],
        )

        return metrics

    def save_model(self, symbol: str) -> Path:
        """Serialise the trained model to disk using joblib."""
        if self._model is None:
            raise ValueError("No trained model. Call train() first.")

        import joblib

        self._model_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
        model_path = self._model_dir / f"{safe_symbol}_model.joblib"

        joblib.dump(self._model, model_path)
        self._log.info("training.model_saved", path=str(model_path))
        return model_path

    def load_model(self, symbol: str) -> Any:
        """Load a previously saved model from disk."""
        import joblib

        safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
        model_path = self._model_dir / f"{safe_symbol}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        self._model = joblib.load(model_path)
        self._log.info("training.model_loaded", path=str(model_path))
        return self._model
