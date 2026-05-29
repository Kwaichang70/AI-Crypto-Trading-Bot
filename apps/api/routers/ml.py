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
                                            -- Now includes OOS Sharpe gate (Sprint 50 Cycle 5)
"""

from __future__ import annotations

import asyncio
import os
import statistics as _statistics_stdlib  # avoid shadowing trading._statistics
import tempfile
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_db
from api.schemas import ModelVersionListResponse, ModelVersionResponse

__all__ = ["router"]

router = APIRouter(prefix="/ml", tags=["ml"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level retraining service reference — DEPRECATED: mirror of
# container.services.retraining_service — sunset by Sprint 41.
# ---------------------------------------------------------------------------
# Set from main.py lifespan via set_retraining_service() during transitional
# period.  Endpoint handlers prefer _resolve_retraining_service(request) which
# reads from app.state.container first and falls back to this global.
_retraining_service: Any = None


def set_retraining_service(service: Any) -> None:
    """Called by main.py lifespan to wire the RetrainingService instance."""
    global _retraining_service
    _retraining_service = service


def _resolve_retraining_service(request: Request) -> Any | None:
    """Prefer container.services.retraining_service; fall back to module global.

    Sprint 40 Stap 1d migration: canonical source is the AppContainer, which
    main.py populates in the lifespan hook.  The module-level ``_retraining_service``
    mirror is retained as a fallback for early-boot and test contexts, and is
    removed in Sprint 41.
    """
    container = getattr(request.app.state, "container", None)
    if container is not None:
        svc = getattr(container.services, "retraining_service", None)
        if svc is not None:
            return svc
    return _retraining_service


# ---------------------------------------------------------------------------
# Existing endpoint: POST /train (horizon-labeled, OHLCV-only)
# ---------------------------------------------------------------------------

@router.post(
    "/train",
    summary="Train ML model for a symbol",
    description=(
        "Fetches historical OHLCV data and trains a RandomForestClassifier "
        "for the specified symbol using horizon-based labels. The trained model "
        "is saved to the models/ directory.  Walk-forward OOS Sharpe is computed "
        "across num_wf_folds folds and persisted on ModelVersionORM (Sprint 50 Cycle 5)."
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
    num_wf_folds: int = Query(
        default=5,
        ge=2,
        le=20,
        description=(
            "Number of walk-forward folds for OOS Sharpe computation. "
            "Default 5 (expanding mode). Higher values increase training time."
        ),
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Train a RandomForestClassifier and save it to disk."""
    from api.config import get_settings

    log = logger.bind(
        endpoint="train_model", symbol=symbol, exchange=exchange, timeframe=timeframe,
    )
    log.info("ml.train_requested", bars=bars, n_estimators=n_estimators, num_wf_folds=num_wf_folds)

    settings = get_settings()

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            _train_model_with_wf_sync,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            bars=bars,
            n_estimators=n_estimators,
            horizon=horizon,
            threshold=threshold,
            num_wf_folds=num_wf_folds,
            min_trades_per_fold=settings.min_trades_per_fold,
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
    await db.flush()

    # Persist walk-forward OOS metrics (Sprint 50 Cycle 5).
    # CR5v2-001: source existing_extra from the ORM row, not the result dict,
    # to avoid overwriting prior feature-importance metadata on retrain.
    existing_extra: dict[str, Any] = dict(model_version.extra or {})
    existing_extra["walk_forward"] = {
        "schema_version": 2,
        "metric_type": "directional_zscore_proxy",
        "status": result.get("walk_forward_status", "ok"),
        "num_folds": num_wf_folds,
        "fold_skill_scores_raw": result.get("walk_forward_fold_skill_scores", []),
        "fold_trade_counts": result.get("walk_forward_fold_trade_counts", []),
        "oos_skill_median_raw": result.get("walk_forward_oos_skill_score_raw"),
        "oos_skill_worst": result.get("walk_forward_oos_skill_score_worst"),
        "oos_skill_median_deflated": result.get("walk_forward_oos_skill_score"),
        "folds_below_threshold": result.get("walk_forward_folds_below_threshold", []),
    }
    model_version.extra = existing_extra
    # The typed column stores the DEFLATED MEDIAN directional z-score skill proxy --
    # canonical gate value. NOT a trading Sharpe (magnitude-blind).
    model_version.walk_forward_oos_skill_score = result.get("walk_forward_oos_skill_score")

    await db.commit()

    log.info(
        "ml.train_completed",
        model_path=result.get("model_path"),
        model_id=str(model_version.id),
        walk_forward_oos_skill_score=result.get("walk_forward_oos_skill_score"),
    )
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
    """Synchronous training pipeline without walk-forward (legacy / internal helper)."""
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


def _train_model_with_wf_sync(
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    bars: int,
    n_estimators: int,
    horizon: int,
    threshold: float,
    num_wf_folds: int = 5,
    min_trades_per_fold: int = 20,
) -> dict[str, Any]:
    """Synchronous training pipeline with walk-forward OOS Sharpe computation.

    Extends ``_train_model_sync`` to additionally run ``WalkForwardValidator``
    on the fetched OHLCV data.  Each fold trains a fresh ``ModelTrainer`` on
    the train bars and evaluates a simplified OOS accuracy-based Sharpe proxy
    using the fold's test bars.

    The DEFLATED MEDIAN OOS Sharpe (Bailey & Lopez de Prado) is the canonical
    gate value persisted to ``ModelVersionORM.walk_forward_oos_skill_score``.

    OOS skill score per fold is computed as: (2*accuracy - 1) * sqrt(n_test_samples).
    This is a directional z-score PROXY, NOT a trading Sharpe -- magnitude-blind:
    does not account for fees, slippage, or position sizing.
    A value > 0 means the model predicts better than random on the OOS window.

    Parameters
    ----------
    symbol, exchange, timeframe, bars, n_estimators, horizon, threshold:
        Passed through to the main training path (same as ``_train_model_sync``).
    num_wf_folds:
        Number of walk-forward folds.  Must be >= 2.
    min_trades_per_fold:
        Minimum OOS test samples per fold to consider skill score statistically
        meaningful.  Folds below this set walk_forward_status="insufficient_samples".
    """
    import math

    import ccxt
    import numpy as _np
    import pandas as pd

    from common.models import OHLCVBar
    from common.types import TimeFrame as _TimeFrame
    from data.ml_training import ModelTrainer
    from datetime import UTC, datetime as _dt
    from trading.walk_forward import WalkForwardValidator, deflated_sharpe_ratio

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

    # 2. Train full model on all data (production model)
    trainer = ModelTrainer(model_dir="models/")
    X, y = trainer.prepare_dataset(df, horizon=horizon, threshold=threshold)
    metrics = trainer.train(X, y, n_estimators=n_estimators)
    model_path = trainer.save_model(symbol=symbol)

    # 3. Convert raw OHLCV to OHLCVBar list for WalkForwardValidator
    tf_enum = _TimeFrame(timeframe)
    ohlcv_bars: list[OHLCVBar] = []
    for row in raw:
        ts_ms, o, h, lo, c, vol = row[0], row[1], row[2], row[3], row[4], row[5]
        ohlcv_bars.append(
            OHLCVBar(
                symbol=symbol,
                timeframe=tf_enum,
                timestamp=_dt.fromtimestamp(ts_ms / 1000.0, tz=UTC),
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=float(vol),
            )
        )

    bars_by_symbol: dict[str, list[OHLCVBar]] = {symbol: ohlcv_bars}

    # 4. Walk-forward: train per-fold model + compute OOS accuracy-based skill score proxy
    validator = WalkForwardValidator(num_folds=num_wf_folds, train_fraction=0.7, mode="expanding")
    try:
        folds = validator.split(bars_by_symbol)
    except ValueError:
        # Not enough bars for the requested folds -- fall back to no WF metrics
        folds = []

    fold_skill_scores: list[float] = []
    fold_trade_counts: list[int] = []  # used as OOS test-sample counts

    with tempfile.TemporaryDirectory(prefix="wf_fold_") as _tmpdir:
        for fold in folds:
            try:
                # Build DataFrame from fold train bars
                fold_train_bars = fold.train_bars.get(symbol, [])
                fold_df_rows = [
                    [
                        int(b.timestamp.timestamp() * 1000),
                        float(b.open), float(b.high), float(b.low),
                        float(b.close), float(b.volume),
                    ]
                    for b in fold_train_bars
                ]
                fold_df = pd.DataFrame(
                    fold_df_rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                fold_trainer = ModelTrainer(model_dir=_tmpdir)
                fold_X_train, fold_y_train = fold_trainer.prepare_dataset(
                    fold_df, horizon=horizon, threshold=threshold
                )
                fold_trainer.train(fold_X_train, fold_y_train, n_estimators=n_estimators)

                # Build OOS test DataFrame from fold test bars
                fold_test_bars = fold.test_bars.get(symbol, [])
                fold_test_df_rows = [
                    [
                        int(b.timestamp.timestamp() * 1000),
                        float(b.open), float(b.high), float(b.low),
                        float(b.close), float(b.volume),
                    ]
                    for b in fold_test_bars
                ]
                fold_test_df = pd.DataFrame(
                    fold_test_df_rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                try:
                    fold_X_test, fold_y_test = fold_trainer.prepare_dataset(
                        fold_test_df, horizon=horizon, threshold=threshold
                    )
                    fold_model = fold_trainer._model
                    if fold_model is not None and len(fold_X_test) > 0:
                        fold_y_pred = fold_model.predict(fold_X_test)
                        fold_acc = float(_np.mean(fold_y_pred == fold_y_test))
                        n_test = len(fold_y_test)
                        # Accuracy-based pseudo-Sharpe: (2*acc - 1) * sqrt(n)
                        fold_skill_score = (2.0 * fold_acc - 1.0) * math.sqrt(n_test)
                        fold_skill_scores.append(fold_skill_score)
                        fold_trade_counts.append(n_test)
                    else:
                        fold_skill_scores.append(0.0)
                        fold_trade_counts.append(0)
                except Exception as oos_exc:
                    logger.warning(
                        "ml.walk_forward_fold_oos_eval_failed",
                        fold_index=fold.index,
                        error=str(oos_exc),
                    )
                    fold_skill_scores.append(0.0)
                    fold_trade_counts.append(0)

            except Exception as exc:
                logger.warning(
                    "ml.walk_forward_fold_failed",
                    fold_index=fold.index,
                    error=str(exc),
                )
                fold_skill_scores.append(0.0)
                fold_trade_counts.append(0)
        # TemporaryDirectory.__exit__ deletes _tmpdir and all per-fold files.

    # 5. QT5-003: check for insufficient OOS samples in any fold
    folds_below_threshold: list[int] = [
        i for i, tc in enumerate(fold_trade_counts)
        if tc < min_trades_per_fold
    ]
    insufficient_samples = len(folds_below_threshold) > 0
    if insufficient_samples:
        actual_min_fold_trades = min(fold_trade_counts) if fold_trade_counts else 0
        logger.warning(
            "ml.walk_forward_insufficient_oos_samples",
            folds_below_threshold=folds_below_threshold,
            min_trades_per_fold=min_trades_per_fold,
            actual_min=actual_min_fold_trades,
        )

    # 6. QT5-001 + QT5-002: MEDIAN + Deflated Sharpe aggregation over skill scores
    #    NOTE: "Deflated Sharpe" here refers to the Bailey-Lopez de Prado (2014)
    #    deflation formula applied to the DIRECTIONAL Z-SCORE PROXY (2*acc-1)*sqrt(n),
    #    NOT to a realized trading Sharpe.  The result is still a skill proxy.
    oos_skill_median_raw: float | None = None
    oos_skill_worst: float | None = None
    oos_skill_median_deflated: float | None = None

    if len(fold_skill_scores) >= 2:
        oos_skill_median_raw = _statistics_stdlib.median(fold_skill_scores)
        oos_skill_worst = min(fold_skill_scores)
        skill_std = _statistics_stdlib.stdev(fold_skill_scores)
        oos_skill_median_deflated = deflated_sharpe_ratio(
            observed_sharpe=oos_skill_median_raw,
            sharpe_stddev=skill_std,
            num_trials=len(fold_skill_scores),
        )
    elif len(fold_skill_scores) == 1:
        oos_skill_median_raw = fold_skill_scores[0]
        oos_skill_worst = fold_skill_scores[0]
        oos_skill_median_deflated = fold_skill_scores[0]
    # else: no folds computed -- all WF metrics remain None

    # Quant review caveat (Cycle 5): emit warning so operator knows the gate
    # value is a classification-accuracy proxy, not a realized trading metric.
    logger.warning(
        "ml.walk_forward_skill_proxy",
        msg=(
            "OOS gate uses directional z-score proxy (2*acc-1)*sqrt(n), NOT a "
            "trading Sharpe. Magnitude-blind: does not account for fees/slippage/"
            "sizing. See cycle 5 quant review and sprint50-cycle5-quant-backlog.md."
        ),
        model_version_id=None,  # not yet persisted at this point; caller logs model_id
        oos_skill_score_deflated=oos_skill_median_deflated,
        oos_skill_score_raw=oos_skill_median_raw,
    )

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
        # Walk-forward OOS skill score metrics (Sprint 50 Cycle 5)
        # Key prefix uses "skill_score" not "sharpe" -- directional z-score proxy.
        "walk_forward_oos_skill_score": oos_skill_median_deflated,
        "walk_forward_oos_skill_score_raw": oos_skill_median_raw,
        "walk_forward_oos_skill_score_worst": oos_skill_worst,
        "walk_forward_fold_skill_scores": fold_skill_scores,
        "walk_forward_fold_trade_counts": fold_trade_counts,
        "walk_forward_status": "insufficient_samples" if insufficient_samples else "ok",
        "walk_forward_folds_below_threshold": folds_below_threshold,
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
    request: Request,
    timeframe: str = Query(default="1h", description="Candle timeframe for OHLCV fetch"),
) -> dict[str, str]:
    """Trigger manual PnL-labeled retraining for a symbol."""
    log = logger.bind(endpoint="manual_retrain", symbol=symbol, timeframe=timeframe)

    retraining_service = _resolve_retraining_service(request)
    if retraining_service is None:
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
        retraining_service.manual_retrain(symbol=symbol, timeframe=timeframe),
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
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ModelVersionResponse:
    """Activate a specific model version by UUID, deactivating the current active one.

    Sprint 50 Cycle 5: checks walk-forward OOS skill score gate before activation.
    The gate uses a directional z-score proxy (2*acc-1)*sqrt(n), NOT a trading Sharpe.
    If the model's OOS skill score (directional z-score proxy) is below
    ``min_oos_skill_score`` the endpoint returns HTTP 422 with ``oos_gate_failed``
    error.  Supply ``X-Override-OOS-Gate: <admin_key>`` to bypass the gate for
    exceptional cases; the bypass is always recorded as an audit event before state
    mutation.
    """
    import hmac

    from sqlalchemy import select, update

    from api.config import get_settings
    from api.db.models import ModelVersionORM
    from api.services.audit_log import record_audit_event
    from api.services.model_activation_gate import check_oos_eligibility

    log = logger.bind(endpoint="activate_model_version", model_id=str(model_id))
    settings = get_settings()

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
        # Already active -- return immediately without DB writes
        log.info("ml.model_already_active", model_id=str(model_id))
        return ModelVersionResponse.model_validate(target)

    symbol = target.symbol
    timeframe = target.timeframe

    # Sprint 50 Cycle 5: OOS gate check.
    # X-Override-OOS-Gate header is optional: absent = gate applies normally,
    # present+valid = gate bypassed (audit row written), present+invalid = gate applies
    # (no error revealed to caller -- same behaviour as absent to prevent oracle).
    x_override_oos_gate: str | None = request.headers.get("X-Override-OOS-Gate")
    oos_override_active = False

    if x_override_oos_gate is not None:
        admin_key = settings.admin_api_key.get_secret_value()
        if admin_key and hmac.compare_digest(
            x_override_oos_gate.encode(), admin_key.encode()
        ):
            oos_override_active = True
            log.warning(
                "ml.oos_gate_override_requested",
                model_id=str(model_id),
                actor="admin",
            )
        else:
            # Wrong key -- treat as non-override; gate still applies.
            log.warning(
                "ml.oos_gate_override_invalid_key",
                model_id=str(model_id),
            )

    oos_result = check_oos_eligibility(
        target,
        min_oos_skill_score=settings.min_oos_skill_score,
        min_worst_fold_skill_score=settings.min_worst_fold_skill_score,
        min_trades_per_fold=settings.min_trades_per_fold,
    )

    if not oos_result.eligible and not oos_override_active:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "oos_gate_failed",
                "model_id": str(model_id),
                "oos_skill_score": oos_result.oos_skill_score,
                "threshold": oos_result.threshold,
                "reason": oos_result.reason,
                "hint": (
                    "Retrain via POST /ml/train to compute updated OOS metrics, "
                    "lower MIN_OOS_SKILL_SCORE in settings, "
                    "or supply X-Override-OOS-Gate: <admin_key> to bypass. "
                    "Note: the gate uses a directional z-score proxy (2*acc-1)*sqrt(n), "
                    "NOT a trading Sharpe."
                ),
            },
        )

    if oos_override_active:
        # Audit the bypass BEFORE state mutation (SEC-002 + SAVEPOINT pattern).
        async with db.begin_nested():
            await record_audit_event(
                db,
                event_type="model_oos_gate_bypassed",
                resource_type="model_version",
                resource_id=str(target.id),
                request=request,
                payload={
                    "symbol": target.symbol,
                    "timeframe": target.timeframe,
                    "oos_skill_score": oos_result.oos_skill_score,
                    "threshold": oos_result.threshold,
                    "oos_eligible": oos_result.eligible,
                },
            )
        # SAVEPOINT released: audit row durable before is_active mutation.

    # Deactivate all currently active models for this symbol+timeframe.
    # Both UPDATEs commit atomically when get_db's session exits on handler return.
    # Do NOT call db.begin() -- get_db already manages the transaction lifecycle.
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

    # SEC-002: persist audit record for model activation (rollback / promotion
    # of a ModelStrategy version is a security-sensitive state transition).
    await record_audit_event(
        db,
        event_type="model_activated",
        resource_type="model_version",
        resource_id=str(target.id),
        request=request,
        payload={
            "symbol": symbol,
            "timeframe": timeframe,
            "accuracy": float(target.accuracy),
            "oos_skill_score": oos_result.oos_skill_score,
            "oos_warning": oos_result.warning or None,
        },
    )

    # Update the sidecar JSON so ModelStrategy hot-swaps immediately
    retraining_service = _resolve_retraining_service(request)
    if retraining_service is not None:
        retraining_service._write_active_sidecar(
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
        oos_skill_score=oos_result.oos_skill_score,
        oos_warning=oos_result.warning or None,
    )

    return ModelVersionResponse.model_validate(target)
