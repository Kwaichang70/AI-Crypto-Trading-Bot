"""
apps/api/routers/ml.py
-----------------------
Machine learning model training endpoints.

POST /api/v1/ml/train  -- Train a RandomForestClassifier for a symbol.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, status

__all__ = ["router"]

router = APIRouter(prefix="/ml", tags=["ml"])
logger = structlog.get_logger(__name__)


@router.post(
    "/train",
    summary="Train ML model for a symbol",
    description=(
        "Fetches historical OHLCV data and trains a RandomForestClassifier "
        "for the specified symbol. The trained model is saved to the models/ "
        "directory."
    ),
)
async def train_model(
    symbol: str = Query(..., description="Trading pair, e.g. BTC/USDT"),
    exchange: str = Query(default="binance", description="CCXT exchange ID"),
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    bars: int = Query(default=2000, ge=200, le=10000, description="Number of bars"),
    n_estimators: int = Query(default=100, ge=10, le=500, description="Number of trees"),
    horizon: int = Query(default=5, ge=1, le=50, description="Prediction horizon in bars"),
    threshold: float = Query(default=0.01, ge=0.001, le=0.1, description="Return threshold"),
) -> dict[str, Any]:
    """Train a RandomForestClassifier and save it to disk."""
    log = logger.bind(
        endpoint="train_model", symbol=symbol, exchange=exchange, timeframe=timeframe,
    )
    log.info("ml.train_requested", bars=bars, n_estimators=n_estimators)

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            _train_model_sync,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            bars=bars,
            n_estimators=n_estimators,
            horizon=horizon,
            threshold=threshold,
        )
    except ImportError as exc:
        log.error("ml.missing_dependency", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing ML dependency: {exc}. Install scikit-learn and joblib.",
        ) from exc
    except ValueError as exc:
        log.warning("ml.train_validation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.error("ml.train_failed", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        ) from exc

    log.info("ml.train_completed", model_path=result.get("model_path"))
    return result


def _train_model_sync(
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    bars: int,
    n_estimators: int,
    horizon: int,
    threshold: float,
) -> dict[str, Any]:
    """Synchronous training pipeline (runs in thread pool)."""
    import ccxt
    import pandas as pd

    from data.ml_training import ModelTrainer

    # 1. Fetch candles via synchronous ccxt
    exchange_cls = getattr(ccxt, exchange, None)
    if exchange_cls is None:
        raise ValueError(f"Exchange '{exchange}' is not supported by ccxt.")

    exc_instance = exchange_cls({"enableRateLimit": True})
    try:
        raw: list[list[Any]] = exc_instance.fetch_ohlcv(
            symbol=symbol, timeframe=timeframe, limit=bars,
        )
    finally:
        try:
            exc_instance.close()
        except Exception:
            pass

    if not raw:
        raise ValueError(f"No OHLCV data returned for {symbol} on {exchange}.")

    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )

    # 2. Delegate to ModelTrainer
    trainer = ModelTrainer(model_dir="models/")
    X, y = trainer.prepare_dataset(df, horizon=horizon, threshold=threshold)
    metrics = trainer.train(X, y, n_estimators=n_estimators)
    model_path = trainer.save_model(symbol=symbol)

    return {
        "status": "completed",
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "model_path": str(model_path),
        "bars_fetched": len(raw),
        "training_samples": metrics["train_samples"],
        "test_samples": metrics["test_samples"],
        "metrics": metrics,
    }
