"""
apps/api/routers/ml.py
-----------------------
Machine learning model training and management endpoints.

Endpoints
---------
POST /api/v1/ml/train               -- Train using horizon-labeled OHLCV data (existing)
GET  /api/v1/ml/models              -- List model versions with optional filters
POST /api/v1/ml/retrain/{symbol}    -- Manual PnL-labeled retrain from trade history
PUT  /api/v1/ml/models/{model_id}/activate -- Rollback/promote a specific model version
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_db
from api.schemas import ModelVersionListResponse, ModelVersionResponse

__all__ = ["router"]

router = APIRouter(prefix="/ml", tags=["ml"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level retraining service reference (set by main.py lifespan)
# ---------------------------------------------------------------------------
# This is set from main.py after the service is instantiated.
# Using a module-level variable avoids threading the service through Depends().
_retraining_service: Any = None


def set_retraining_service(service: Any) -> None:
    """Called by main.py lifespan to wire the RetrainingService instance."""
    global _retraining_service
    _retraining_service = service


# ---------------------------------------------------------------------------
# Existing endpoint: POST /train (horizon-labeled, OHLCV-only)
# ---------------------------------------------------------------------------

@router.post(
    "/train",
    summary="Train ML model for a symbol",
    description=(
        "Fetches historical OHLCV data and trains a RandomForestClassifier "
        "for the specified symbol using horizon-based labels. The trained model "
        "is saved to the models/ directory."
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
    db: AsyncSession = Depends(get_db),
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

    # Persist model version to database
    from datetime import UTC, datetime as _dt
    from api.db.models import ModelVersionORM

    model_version = ModelVersionORM(
        symbol=symbol,
        timeframe=timeframe,
        trained_at=_dt.now(tz=UTC),
        accuracy=result.get("metrics", {}).get("accuracy", 0.0),
        n_trades_used=0,  # horizon-based training uses bars, not trades
        n_bars_used=result.get("bars_fetched", 0),
        label_method="future_return",
        model_path=result.get("model_path", ""),
        is_active=True,
        trigger="manual",
        extra=result.get("metrics"),
    )
    # Deactivate previous active model for this symbol+timeframe
    from sqlalchemy import update
    await db.execute(
        update(ModelVersionORM)
        .where(
            ModelVersionORM.symbol == symbol,
            ModelVersionORM.timeframe == timeframe,
            ModelVersionORM.is_active.is_(True),
        )
        .values(is_active=False)
    )
    db.add(model_version)
    await db.commit()

    log.info("ml.train_completed", model_path=result.get("model_path"), model_id=str(model_version.id))
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


# ---------------------------------------------------------------------------
# New endpoint: GET /models — list model versions
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=ModelVersionListResponse,
    summary="List ML model versions",
    description=(
        "Returns all trained model versions with optional filtering by symbol, "
        "timeframe, and active status. Results are ordered by trained_at descending."
    ),
)
async def list_model_versions(
    symbol: str | None = Query(default=None, description="Filter by trading pair, e.g. BTC/USD"),
    timeframe: str | None = Query(default=None, description="Filter by timeframe, e.g. 1h"),
    active_only: bool = Query(default=False, description="Return only currently active models"),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Records to skip"),
    db: AsyncSession = Depends(get_db),
) -> ModelVersionListResponse:
    """List ML model versions from the database."""
    from sqlalchemy import func, select

    from api.db.models import ModelVersionORM

    log = logger.bind(endpoint="list_model_versions")

    # Build base query
    base_query = select(ModelVersionORM)
    count_query = select(func.count()).select_from(ModelVersionORM)

    filters = []
    if symbol is not None:
        filters.append(ModelVersionORM.symbol == symbol)
    if timeframe is not None:
        filters.append(ModelVersionORM.timeframe == timeframe)
    if active_only:
        filters.append(ModelVersionORM.is_active.is_(True))

    if filters:
        base_query = base_query.where(*filters)
        count_query = count_query.where(*filters)

    # Total count
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Paginated results, newest first
    result = await db.execute(
        base_query
        .order_by(ModelVersionORM.trained_at.desc())
        .offset(offset)
        .limit(limit)
    )
    versions = list(result.scalars().all())

    log.info("ml.list_model_versions", total=total, returned=len(versions))

    return ModelVersionListResponse(
        items=[ModelVersionResponse.model_validate(v) for v in versions],
        total=total,
    )


# ---------------------------------------------------------------------------
# New endpoint: POST /retrain/{symbol} — manual PnL-labeled retrain
# ---------------------------------------------------------------------------

@router.post(
    "/retrain/{symbol:path}",
    summary="Manually trigger PnL-labeled model retraining",
    description=(
        "Triggers an immediate retraining cycle for the specified symbol using "
        "closed trade history (PnL-labeled). Requires ml_auto_retrain=True and "
        "an active model version in the database. Returns immediately — training "
        "runs in the background via the RetrainingService."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def manual_retrain(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe for OHLCV fetch"),
) -> dict[str, str]:
    """Trigger manual PnL-labeled retraining for a symbol."""
    log = logger.bind(endpoint="manual_retrain", symbol=symbol, timeframe=timeframe)

    if _retraining_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RetrainingService is not running. "
                "Set ML_AUTO_RETRAIN=true and restart the API to enable."
            ),
        )

    log.info("ml.manual_retrain_requested")

    # Fire-and-forget: not tracked for cancellation on shutdown (MVP scope).
    # _do_retrain catches all exceptions internally; the DB row is only written
    # after training succeeds, so mid-flight abandonment on shutdown is safe.
    asyncio.create_task(
        _retraining_service.manual_retrain(symbol=symbol, timeframe=timeframe),
        name=f"manual_retrain_{symbol}_{timeframe}",
    )

    return {
        "status": "accepted",
        "symbol": symbol,
        "timeframe": timeframe,
        "message": (
            f"Retraining scheduled for {symbol}/{timeframe}. "
            "Check logs for progress and GET /ml/models for the result."
        ),
    }


# ---------------------------------------------------------------------------
# New endpoint: PUT /models/{model_id}/activate — rollback/promote a version
# ---------------------------------------------------------------------------

@router.put(
    "/models/{model_id}/activate",
    response_model=ModelVersionResponse,
    summary="Activate a specific model version",
    description=(
        "Deactivates the current active model for the target (symbol, timeframe) pair "
        "and activates the specified version. Enables rollback to a previous model "
        "or promotion of a higher-accuracy historical version."
    ),
)
async def activate_model_version(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ModelVersionResponse:
    """Activate a specific model version by UUID, deactivating the current active one."""
    from sqlalchemy import select, update

    from api.db.models import ModelVersionORM
    from api.services.retraining import RetrainingService

    log = logger.bind(endpoint="activate_model_version", model_id=str(model_id))

    # Fetch the target version
    result = await db.execute(
        select(ModelVersionORM).where(ModelVersionORM.id == model_id)
    )
    target: ModelVersionORM | None = result.scalar_one_or_none()

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version {model_id} not found.",
        )

    if target.is_active:
        # Already active — return immediately without DB writes
        log.info("ml.model_already_active", model_id=str(model_id))
        return ModelVersionResponse.model_validate(target)

    symbol = target.symbol
    timeframe = target.timeframe

    # Deactivate all currently active models for this symbol+timeframe.
    # Both UPDATEs commit atomically when get_db's session exits on handler return.
    # Do NOT call db.begin() — get_db already manages the transaction lifecycle.
    await db.execute(
        update(ModelVersionORM)
        .where(
            ModelVersionORM.symbol == symbol,
            ModelVersionORM.timeframe == timeframe,
            ModelVersionORM.is_active.is_(True),
        )
        .values(is_active=False)
    )
    # Activate the target
    await db.execute(
        update(ModelVersionORM)
        .where(ModelVersionORM.id == model_id)
        .values(is_active=True)
    )

    await db.refresh(target)

    # Update the sidecar JSON so ModelStrategy hot-swaps immediately
    if _retraining_service is not None:
        _retraining_service._write_active_sidecar(
            symbol=symbol,
            version_id=str(target.id),
            model_path=target.model_path,
            accuracy=float(target.accuracy),
        )

    log.info(
        "ml.model_activated",
        model_id=str(model_id),
        symbol=symbol,
        timeframe=timeframe,
    )

    return ModelVersionResponse.model_validate(target)
