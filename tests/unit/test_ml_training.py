"""
tests/unit/test_ml_training.py
--------------------------------
Unit tests for the ML model training pipeline.

Module under test
-----------------
packages/data/ml_training.py

Test coverage
-------------
1. TestPrepareDataset    -- prepare_dataset(): X shape (10 cols), y label set {0,1,2},
                           insufficient-data guard, horizon < 1 guard,
                           threshold <= 0 guard, missing-columns guard
2. TestTrain             -- train(): returns metrics dict with required keys,
                           accuracy is in [0.0, 1.0], metrics keys present
3. TestSaveLoad          -- save_model() + load_model() roundtrip: loaded model
                           can predict; save before train() raises ValueError
4. TestModelTrainerInit  -- __init__ defaults; custom model_dir is respected

Design notes
------------
- All tests are synchronous (no async code in ml_training.py).
- asyncio_mode = "auto" in pyproject.toml; no @pytest.mark.asyncio needed.
- scikit-learn and joblib are production dependencies; they are NOT mocked.
  We test the real RandomForestClassifier training path on synthetic data.
- np.random.seed(42) is set locally in _make_ohlcv_df to ensure deterministic
  synthetic data without polluting the global NumPy random state.
- pytest.approx is used for float comparisons.
- tmp_path (built-in pytest fixture) is used for save/load tests to avoid
  leaving artefacts on disk.
- The _make_ohlcv_df helper creates 500 bars of trending data, which is more
  than sufficient to satisfy the _MIN_SAMPLES=50 threshold even after NaN warmup
  removal (SMA-100 removes 100 warmup rows; 500 - 100 - horizon = 395 valid rows).
- For the insufficient-data test we use a 20-row DataFrame so that after
  warmup removal fewer than _MIN_SAMPLES=50 valid rows remain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.ml_training import LABEL_BUY, LABEL_HOLD, LABEL_SELL, ModelTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_REQUIRED_METRICS_KEYS = {"accuracy", "train_samples", "test_samples", "feature_importances"}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 500) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with trending data.

    Uses a local RandomState seeded with 42 for determinism without polluting
    the global NumPy random state.  The close series is a random walk with
    drift so that both BUY and SELL labels appear in the prepared dataset.
    """
    rng = np.random.RandomState(42)  # noqa: NPY002 — intentional for test determinism
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    # Ensure all closes are positive (required by build_feature_matrix log-return path)
    close = np.abs(close) + 1.0
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0.1, 0.5, n),
            "high": close + rng.uniform(0.5, 2.0, n),
            "low": close - rng.uniform(0.5, 2.0, n),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )


def _make_trainer(model_dir: str | Path = "models/") -> ModelTrainer:
    """Construct a ModelTrainer with the given model directory."""
    return ModelTrainer(model_dir=str(model_dir))


def _prepare_and_train(trainer: ModelTrainer, n: int = 500) -> dict:  # type: ignore[type-arg]
    """Convenience: prepare dataset + train on n synthetic bars, return metrics."""
    df = _make_ohlcv_df(n)
    X, y = trainer.prepare_dataset(df)
    return trainer.train(X, y)


# ---------------------------------------------------------------------------
# Class 1: TestPrepareDataset
# ---------------------------------------------------------------------------


class TestPrepareDataset:
    """Verify ModelTrainer.prepare_dataset() output shapes and guard clauses."""

    def test_prepare_dataset_shape(self) -> None:
        """X has exactly 10 feature columns; y has the same number of rows as X.

        The feature schema is fixed at 10 elements.  Any mismatch here would
        break model.fit() due to shape inconsistency.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)
        X, y = trainer.prepare_dataset(df)

        assert X.shape[1] == 10, (
            f"X must have 10 feature columns, got {X.shape[1]}"
        )
        assert X.shape[0] == y.shape[0], (
            f"X and y must have the same number of rows: X={X.shape[0]}, y={y.shape[0]}"
        )

    def test_prepare_dataset_labels(self) -> None:
        """All label values must be in {{LABEL_SELL, LABEL_HOLD, LABEL_BUY}} = {{0, 1, 2}}.

        Labels outside this set would cause the RandomForestClassifier to learn
        unexpected classes and break the ModelStrategy signal-routing logic that
        relies on these constants.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)
        _, y = trainer.prepare_dataset(df)

        unique_labels = set(y.tolist())
        valid_labels = {LABEL_SELL, LABEL_HOLD, LABEL_BUY}  # {0, 1, 2}

        assert unique_labels.issubset(valid_labels), (
            f"Labels must be a subset of {valid_labels}, got {unique_labels}"
        )

    def test_prepare_dataset_insufficient_data_raises(self) -> None:
        """Very short DataFrame raises ValueError when valid samples < _MIN_SAMPLES=50.

        A 20-row DataFrame with default horizon=5 produces at most 20 - 100 warmup
        rows of NaN from SMA-100, leaving zero valid samples — well below the
        50-sample minimum.  The function must raise rather than return a tiny dataset.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(20)

        with pytest.raises(ValueError, match="valid samples"):
            trainer.prepare_dataset(df)

    def test_prepare_dataset_invalid_horizon_raises(self) -> None:
        """horizon < 1 must raise ValueError.

        A zero or negative horizon is nonsensical (no lookahead) and the guard
        clause must fire before any computation.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)

        with pytest.raises(ValueError, match="horizon must be >= 1"):
            trainer.prepare_dataset(df, horizon=0)

    def test_prepare_dataset_horizon_negative_raises(self) -> None:
        """A negative horizon must also raise ValueError.

        The guard covers horizon < 1, which includes negative values.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)

        with pytest.raises(ValueError, match="horizon must be >= 1"):
            trainer.prepare_dataset(df, horizon=-3)

    def test_prepare_dataset_invalid_threshold_raises(self) -> None:
        """threshold <= 0 must raise ValueError.

        A non-positive threshold produces degenerate labeling (all bars would
        be labeled BUY or SELL, defeating the HOLD class).
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)

        with pytest.raises(ValueError, match="threshold must be > 0"):
            trainer.prepare_dataset(df, threshold=0.0)

    def test_prepare_dataset_negative_threshold_raises(self) -> None:
        """A negative threshold must also raise ValueError.

        Threshold must be strictly positive to be meaningful as a return band.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)

        with pytest.raises(ValueError, match="threshold must be > 0"):
            trainer.prepare_dataset(df, threshold=-0.01)

    def test_prepare_dataset_missing_columns_raises(self) -> None:
        """A DataFrame missing required columns raises ValueError.

        Required columns are: open, high, low, close, volume.  Omitting any
        one of them must raise ValueError before any computation begins.
        """
        trainer = _make_trainer()
        # Omit 'volume' column
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                # 'volume' is intentionally absent
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            trainer.prepare_dataset(df)


# ---------------------------------------------------------------------------
# Class 2: TestTrain
# ---------------------------------------------------------------------------


class TestTrain:
    """Verify ModelTrainer.train() return contract and accuracy bounds."""

    def test_train_returns_metrics(self) -> None:
        """train() returns a dict containing the required metric keys.

        Callers (e.g. training scripts, CI jobs) rely on specific keys being
        present.  Any missing key is a silent regression.
        """
        trainer = _make_trainer()
        metrics = _prepare_and_train(trainer)

        for key in _REQUIRED_METRICS_KEYS:
            assert key in metrics, (
                f"train() metrics dict is missing required key: {key!r}\n"
                f"Present keys: {list(metrics.keys())}"
            )

    def test_train_accuracy_range(self) -> None:
        """accuracy metric must be in [0.0, 1.0].

        accuracy_score always returns a value in this range; if it falls
        outside, something is wrong with the label mapping or the test
        set construction.
        """
        trainer = _make_trainer()
        metrics = _prepare_and_train(trainer)

        accuracy = metrics["accuracy"]
        assert 0.0 <= accuracy <= 1.0, (
            f"accuracy must be in [0.0, 1.0], got {accuracy}"
        )

    def test_train_sample_counts_positive(self) -> None:
        """train_samples and test_samples must both be positive integers.

        train_test_split with test_size=0.2 and >= 50 samples always produces
        non-empty splits.
        """
        trainer = _make_trainer()
        metrics = _prepare_and_train(trainer)

        assert metrics["train_samples"] > 0, (
            f"train_samples must be > 0, got {metrics['train_samples']}"
        )
        assert metrics["test_samples"] > 0, (
            f"test_samples must be > 0, got {metrics['test_samples']}"
        )

    def test_train_feature_importances_keys(self) -> None:
        """feature_importances must be a dict keyed by all 10 FEATURE_NAMES.

        The importance dict is consumed by analysis tooling to identify which
        features drive predictions.  Missing keys indicate a misalignment
        between the importance array and the feature name list.
        """
        from data.ml_features import FEATURE_NAMES

        trainer = _make_trainer()
        metrics = _prepare_and_train(trainer)

        importances = metrics["feature_importances"]
        assert isinstance(importances, dict), (
            f"feature_importances must be a dict, got {type(importances).__name__}"
        )
        assert set(importances.keys()) == set(FEATURE_NAMES), (
            f"feature_importances keys mismatch.\n"
            f"Expected: {sorted(FEATURE_NAMES)}\n"
            f"Got:      {sorted(importances.keys())}"
        )

    def test_train_feature_importances_sum_to_one(self) -> None:
        """Feature importances from RandomForest must sum to approximately 1.0.

        sklearn RandomForestClassifier always normalises feature importances
        to sum to 1.0.  Any drift indicates wrong extraction logic.
        """
        trainer = _make_trainer()
        metrics = _prepare_and_train(trainer)

        total_importance = sum(metrics["feature_importances"].values())
        assert total_importance == pytest.approx(1.0, abs=1e-9), (
            f"Feature importances must sum to 1.0, got {total_importance}"
        )

    def test_train_sample_counts_consistent(self) -> None:
        """train_samples + test_samples must equal the total valid sample count.

        prepare_dataset returns (X, y); len(X) == total valid samples.
        train_test_split with test_size=0.2 must partition these exactly.
        """
        trainer = _make_trainer()
        df = _make_ohlcv_df(500)
        X, y = trainer.prepare_dataset(df)
        total = len(X)

        metrics = trainer.train(X, y)

        assert metrics["train_samples"] + metrics["test_samples"] == total, (
            f"train_samples ({metrics['train_samples']}) + "
            f"test_samples ({metrics['test_samples']}) "
            f"must equal total samples ({total})"
        )


# ---------------------------------------------------------------------------
# Class 3: TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Verify save_model() / load_model() roundtrip and pre-train guard."""

    def test_save_without_train_raises(self, tmp_path: Path) -> None:
        """save_model() before train() must raise ValueError.

        Persisting an untrained model (self._model is None) is a programming
        error.  The guard must fire with a clear error message.
        """
        trainer = _make_trainer(model_dir=tmp_path)

        with pytest.raises(ValueError, match="No trained model"):
            trainer.save_model(_SYMBOL)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save then load produces a model object that can call predict().

        After loading, the model must accept a 2-D NumPy array with 10 columns
        and return an array of integer predictions in {0, 1, 2}.  This is the
        exact interface used by ModelStrategy.on_bar().
        """
        trainer = _make_trainer(model_dir=tmp_path)
        df = _make_ohlcv_df(500)
        X, y = trainer.prepare_dataset(df)
        trainer.train(X, y)

        model_path = trainer.save_model(_SYMBOL)

        # Verify the file exists on disk
        assert model_path.exists(), (
            f"Model file must exist after save_model(), path: {model_path}"
        )

        # Load and predict
        fresh_trainer = _make_trainer(model_dir=tmp_path)
        loaded_model = fresh_trainer.load_model(_SYMBOL)

        # Build a single test sample using the same feature schema
        sample = X[:1]  # shape (1, 10)
        predictions = loaded_model.predict(sample)

        assert len(predictions) == 1, (
            f"predict() on 1 sample must return 1 prediction, got {len(predictions)}"
        )
        assert predictions[0] in {LABEL_SELL, LABEL_HOLD, LABEL_BUY}, (
            f"Prediction must be in {{0, 1, 2}}, got {predictions[0]}"
        )

    def test_save_creates_correct_filename(self, tmp_path: Path) -> None:
        """save_model() must create a file named <safe_symbol>_model.joblib.

        The symbol 'BTC/USDT' must be sanitised to 'btc_usdt' and the file
        must be named 'btc_usdt_model.joblib' inside model_dir.
        """
        trainer = _make_trainer(model_dir=tmp_path)
        df = _make_ohlcv_df(500)
        X, y = trainer.prepare_dataset(df)
        trainer.train(X, y)

        model_path = trainer.save_model("BTC/USDT")

        assert model_path.name == "btc_usdt_model.joblib", (
            f"Model file must be named 'btc_usdt_model.joblib', got {model_path.name!r}"
        )
        assert model_path.parent == tmp_path, (
            f"Model file must be in model_dir ({tmp_path}), got {model_path.parent}"
        )

    def test_load_missing_model_raises(self, tmp_path: Path) -> None:
        """load_model() for a non-existent symbol must raise FileNotFoundError.

        When the expected .joblib file does not exist, the function must raise
        FileNotFoundError with the path included in the message.
        """
        trainer = _make_trainer(model_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            trainer.load_model("ETH/USDT")


# ---------------------------------------------------------------------------
# Class 4: TestModelTrainerInit
# ---------------------------------------------------------------------------


class TestModelTrainerInit:
    """Verify ModelTrainer.__init__ defaults and attribute initialisation."""

    def test_default_model_dir(self) -> None:
        """ModelTrainer() without arguments uses 'models/' as the default directory.

        The default model_dir is a relative path 'models/' converted to a
        pathlib.Path internally.
        """
        trainer = ModelTrainer()
        assert trainer._model_dir == Path("models/"), (
            f"Default model_dir must be Path('models/'), got {trainer._model_dir!r}"
        )

    def test_custom_model_dir(self, tmp_path: Path) -> None:
        """ModelTrainer(model_dir=...) stores the provided path as a Path object.

        The stored path must match the value passed as a string.
        """
        trainer = _make_trainer(model_dir=tmp_path)
        assert trainer._model_dir == tmp_path, (
            f"model_dir must be stored as {tmp_path}, got {trainer._model_dir!r}"
        )

    def test_model_initially_none(self) -> None:
        """Before train(), the internal _model attribute must be None.

        The save_model() guard relies on this being None to detect untrained
        state.  Any other initial value would cause the guard to malfunction.
        """
        trainer = ModelTrainer()
        assert trainer._model is None, (
            f"_model must be None before train(), got {trainer._model!r}"
        )
