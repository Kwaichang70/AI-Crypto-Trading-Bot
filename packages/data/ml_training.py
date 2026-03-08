"""
packages/data/ml_training.py
------------------------------
ML model training pipeline for ModelStrategy.

Trains a RandomForestClassifier using the canonical 10-element feature schema.
Label encoding: SELL=0, HOLD=1, BUY=2 (matches ModelStrategy._LABEL_* constants).

Requires scikit-learn>=1.5 and joblib>=1.4, imported lazily inside methods.

New in Sprint 23
----------------
- map_trade_to_label()          Converts a closed trade to a training label using
                                 PnL percentage with 3-tier (BUY/HOLD/SELL) scheme.
- floor_to_bar()                 Floors a timestamp to its bar-open boundary for
                                 feature-alignment (used by prepare_dataset_from_trades).
- prepare_dataset_from_trades()  Builds X, y from closed TradeORM dicts + OHLCV DataFrame.
- save_model() gains optional version_suffix parameter for versioned filenames.
- load_model() gains optional model_path parameter to load from an explicit path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

__all__ = [
    "ModelTrainer",
    "map_trade_to_label",
    "floor_to_bar",
    "LABEL_SELL",
    "LABEL_HOLD",
    "LABEL_BUY",
    "PNL_THRESHOLD_PCT",
    "BREAKEVEN_BAND_PCT",
]

logger = structlog.get_logger(__name__)

LABEL_SELL: int = 0
LABEL_HOLD: int = 1
LABEL_BUY: int = 2

# ---------------------------------------------------------------------------
# PnL label thresholds (Sprint 23 — quant agent QUANT-S23-PNL, Section 1.4)
# ---------------------------------------------------------------------------
PNL_THRESHOLD_PCT: float = 0.0015   # 0.15% — covers Coinbase round-trip fees
BREAKEVEN_BAND_PCT: float = 0.0005  # 0.05% — fee-dominated noise band

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

# ---------------------------------------------------------------------------
# Supported timeframe deltas for floor_to_bar
# ---------------------------------------------------------------------------
_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m":  timedelta(minutes=1),
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h":  timedelta(hours=1),
    "4h":  timedelta(hours=4),
    "1d":  timedelta(days=1),
}


# ---------------------------------------------------------------------------
# Public standalone helpers (Sprint 23)
# ---------------------------------------------------------------------------

def map_trade_to_label(
    side: str,
    realised_pnl: Decimal,
    entry_price: Decimal,
    quantity: Decimal,
    pnl_threshold: float = PNL_THRESHOLD_PCT,
    breakeven_band: float = BREAKEVEN_BAND_PCT,
) -> int:
    """Map a closed trade to a training label based on realized PnL.

    Uses a 3-tier labeling scheme that accounts for the side of the trade
    and the magnitude of the return relative to fee thresholds.

    Parameters
    ----------
    side : str
        Opening side of the trade: "buy" or "sell".
    realised_pnl : Decimal
        Net realised PnL after fees, in quote currency.
    entry_price : Decimal
        Volume-weighted average entry price.
    quantity : Decimal
        Total traded quantity in base asset.
    pnl_threshold : float
        Minimum absolute return percentage (default 0.15%) to classify
        as a correct directional signal. Covers Coinbase round-trip fees.
    breakeven_band : float
        Return band (default 0.05%) around zero that is classified as
        HOLD (fee-dominated noise).

    Returns
    -------
    int
        LABEL_BUY (2), LABEL_HOLD (1), or LABEL_SELL (0).
    """
    cost_basis = float(entry_price) * float(quantity)
    if cost_basis == 0.0:
        logger.warning(
            "map_trade_to_label.zero_cost_basis",
            entry_price=str(entry_price),
            quantity=str(quantity),
        )
        return LABEL_HOLD

    return_pct = float(realised_pnl) / cost_basis
    side_lower = side.lower()

    if side_lower == "buy":
        if return_pct > pnl_threshold:
            return LABEL_BUY
        elif return_pct < -breakeven_band:
            return LABEL_SELL
        else:
            return LABEL_HOLD
    else:
        # sell side: a large negative return_pct means profit on a short
        if return_pct < -pnl_threshold:
            return LABEL_SELL
        elif return_pct > breakeven_band:
            return LABEL_BUY
        else:
            return LABEL_HOLD


def floor_to_bar(dt: datetime, timeframe: str) -> datetime:
    """Floor a timestamp to its bar-open boundary.

    Parameters
    ----------
    dt : datetime
        The datetime to floor. Must be timezone-aware (UTC).
    timeframe : str
        Candle timeframe. Supported: 1m, 5m, 15m, 1h, 4h, 1d.

    Returns
    -------
    datetime
        The floored datetime at the bar-open boundary (UTC, timezone-aware).

    Raises
    ------
    ValueError
        If the timeframe is not supported.
    """
    delta = _TIMEFRAME_DELTAS.get(timeframe)
    if delta is None:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {list(_TIMEFRAME_DELTAS)}"
        )

    # Ensure UTC-aware for consistent arithmetic
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds_since_epoch = (dt - epoch).total_seconds()
    delta_seconds = delta.total_seconds()
    floored_seconds = (seconds_since_epoch // delta_seconds) * delta_seconds
    return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)


# ---------------------------------------------------------------------------
# ModelTrainer class
# ---------------------------------------------------------------------------


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

    def prepare_dataset_from_trades(
        self,
        trades: list[dict[str, Any]],
        ohlcv_df: pd.DataFrame,
        timeframe: str = "1h",
        context_bars: int = 120,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and label vector y from closed trade records.

        Uses PnL-labeled training data (Sprint 23 MVP). Each trade is mapped to
        the bar that triggered the signal (entry bar minus 1, accounting for the
        fact that the signal fires at bar N's close and fills during bar N+1).

        Parameters
        ----------
        trades : list[dict]
            Closed trade records. Each dict must contain:
                - side: str ("buy" or "sell")
                - realised_pnl: Decimal
                - entry_price: Decimal
                - quantity: Decimal
                - entry_at: datetime (UTC-aware)
        ohlcv_df : pd.DataFrame
            Full OHLCV history with columns: open, high, low, close, volume.
            Index must be a DatetimeIndex with UTC-aware timestamps.
        timeframe : str
            Candle timeframe for bar alignment (e.g. "1h").
        context_bars : int
            Minimum bars of look-back required before a signal bar (default 120).
            Must be >= 100 to cover SMA-100 feature warmup.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (X, y) where X has shape (n_valid_samples, 10) and
            y has shape (n_valid_samples,).

        Raises
        ------
        ValueError
            If fewer than 50 valid samples survive alignment + NaN filtering,
            or if the OHLCV DataFrame is missing required columns.
        """
        from data.ml_features import build_feature_matrix

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(ohlcv_df.columns)
        if missing:
            raise ValueError(f"ohlcv_df missing required columns: {missing}")
        if ohlcv_df.empty:
            raise ValueError("ohlcv_df must not be empty")
        if not trades:
            raise ValueError("trades list must not be empty")

        # Build feature matrix once over the full OHLCV DataFrame
        features_df: pd.DataFrame = build_feature_matrix(ohlcv_df)

        # Ensure the index is UTC-aware for alignment
        idx = features_df.index
        if hasattr(idx, "tz") and idx.tz is None:
            features_df.index = idx.tz_localize("UTC")

        bar_delta = _TIMEFRAME_DELTAS.get(timeframe)
        if bar_delta is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(_TIMEFRAME_DELTAS)}"
            )

        # Use pandas-native searchsorted with Timestamps (avoids asi8 ns/us compat issues)
        ts_index: pd.DatetimeIndex = pd.DatetimeIndex(features_df.index)

        X_rows: list[np.ndarray] = []
        y_vals: list[int] = []
        dropped_no_bar = 0
        dropped_nan = 0
        dropped_lookback = 0

        for trade in trades:
            entry_at: datetime = trade["entry_at"]
            side: str = trade["side"]
            realised_pnl: Decimal = Decimal(str(trade["realised_pnl"]))
            entry_price: Decimal = Decimal(str(trade["entry_price"]))
            quantity: Decimal = Decimal(str(trade["quantity"]))

            # Ensure timezone-aware
            if entry_at.tzinfo is None:
                entry_at = entry_at.replace(tzinfo=timezone.utc)

            # Signal bar = floor(entry_at) shifted back by 1 bar
            # The model saw bar N's close and submitted during bar N+1
            fill_bar_time = floor_to_bar(entry_at, timeframe)
            signal_bar_time = fill_bar_time - bar_delta

            # Find the signal bar using pandas-native searchsorted (avoids asi8 compat issues)
            signal_ts = pd.Timestamp(signal_bar_time)
            pos = int(ts_index.searchsorted(signal_ts, side="left"))

            # Search within ±3 bar positions for the closest bar
            found_pos: int | None = None
            bar_delta_seconds = bar_delta.total_seconds()
            for candidate_pos in range(max(0, pos - 3), min(len(ts_index), pos + 4)):
                candidate_ts = ts_index[candidate_pos]
                diff_seconds = abs((candidate_ts - signal_ts).total_seconds())
                diff_bars = diff_seconds / bar_delta_seconds
                if diff_bars <= 3.0:
                    found_pos = candidate_pos
                    break

            if found_pos is None:
                self._log.warning(
                    "training.trade_no_bar",
                    entry_at=entry_at.isoformat(),
                    signal_bar_time=signal_bar_time.isoformat(),
                )
                dropped_no_bar += 1
                continue

            # Enforce minimum look-back (context_bars before the signal bar)
            if found_pos < context_bars:
                self._log.debug(
                    "training.trade_insufficient_lookback",
                    found_pos=found_pos,
                    context_bars=context_bars,
                )
                dropped_lookback += 1
                continue

            # Extract feature row at the signal bar
            bar_label = features_df.index[found_pos]
            feature_row: pd.Series = features_df.loc[bar_label, _FEATURE_COLS]

            if feature_row.isna().any():
                dropped_nan += 1
                continue

            X_rows.append(feature_row.to_numpy(dtype=float))
            label = map_trade_to_label(
                side=side,
                realised_pnl=realised_pnl,
                entry_price=entry_price,
                quantity=quantity,
            )
            y_vals.append(label)

        n_valid = len(X_rows)
        self._log.info(
            "training.trade_dataset_prepared",
            input_trades=len(trades),
            valid_samples=n_valid,
            dropped_no_bar=dropped_no_bar,
            dropped_nan=dropped_nan,
            dropped_lookback=dropped_lookback,
            buy_count=y_vals.count(LABEL_BUY),
            hold_count=y_vals.count(LABEL_HOLD),
            sell_count=y_vals.count(LABEL_SELL),
        )

        if n_valid < _MIN_SAMPLES:
            raise ValueError(
                f"Only {n_valid} valid trade samples after alignment + NaN filtering. "
                f"Need at least {_MIN_SAMPLES}. Run more trades or reduce context_bars."
            )

        return np.array(X_rows, dtype=float), np.array(y_vals, dtype=int)

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

    def save_model(self, symbol: str, version_suffix: str = "") -> Path:
        """Serialise the trained model to disk using joblib.

        Parameters
        ----------
        symbol : str
            Trading pair used to build the filename base (e.g. "BTC/USD").
        version_suffix : str, optional
            When provided, the filename becomes
            ``{safe_symbol}_{version_suffix}_model.joblib`` instead of
            ``{safe_symbol}_model.joblib``. Existing callers pass no suffix
            and continue to work unchanged (backward compatible).

        Returns
        -------
        Path
            Absolute path to the saved model file.
        """
        if self._model is None:
            raise ValueError("No trained model. Call train() first.")

        import joblib

        self._model_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
        if version_suffix:
            filename = f"{safe_symbol}_{version_suffix}_model.joblib"
        else:
            filename = f"{safe_symbol}_model.joblib"
        model_path = self._model_dir / filename

        joblib.dump(self._model, model_path)
        self._log.info("training.model_saved", path=str(model_path))
        return model_path

    def load_model(self, symbol: str, model_path: str | None = None) -> Any:
        """Load a previously saved model from disk.

        Parameters
        ----------
        symbol : str
            Trading pair used to construct the default filename if
            ``model_path`` is not provided.
        model_path : str | None, optional
            When provided, loads from this exact path instead of the
            constructed default. Supports versioned model loading.

        Returns
        -------
        Any
            The loaded scikit-learn model object.

        Raises
        ------
        FileNotFoundError
            If no model file exists at the resolved path.
        """
        import joblib

        if model_path is not None:
            resolved_path = Path(model_path)
        else:
            safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
            resolved_path = self._model_dir / f"{safe_symbol}_model.joblib"

        if not resolved_path.exists():
            raise FileNotFoundError(f"No model found at {resolved_path}")

        self._model = joblib.load(resolved_path)
        self._log.info("training.model_loaded", path=str(resolved_path))
        return self._model
