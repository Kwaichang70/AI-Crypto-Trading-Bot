"""
apps/api/services/retraining.py
---------------------------------
RetrainingService — background asyncio.Task that monitors trade counts
and triggers automatic ML model retraining.

Sprint 23: Adaptive Learning MVP.

Architecture
------------
- RetrainingService is instantiated in the FastAPI lifespan when
  settings.ml_auto_retrain is True.
- _poll_loop() sleeps for check_interval_seconds, then queries the DB for
  distinct (symbol, timeframe) pairs from active ModelVersionORM records.
- For each pair, _check_and_retrain() counts trades since last_retrain_at
  and triggers _do_retrain() if the threshold is met.
- _do_retrain() fetches OHLCV via CCXT in a thread, queries trades,
  calls ModelTrainer.prepare_dataset_from_trades() + train() in a thread,
  validates accuracy, saves the model with a versioned filename, writes
  a ModelVersionORM row, and atomically writes the sidecar JSON file.
- Version pruning removes the oldest .joblib files and DB rows beyond
  ml_max_model_versions per (symbol, timeframe) pair.

Manual retrain
--------------
manual_retrain(symbol, timeframe) is callable from the API endpoint
(POST /api/v1/ml/retrain/{symbol}) and executes _do_retrain directly.

Thread safety
--------------
The _last_retrain_at dict is modified only inside _poll_loop / _do_retrain,
which are called sequentially within the single asyncio Task. No locks needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

__all__ = ["RetrainingService"]

logger = structlog.get_logger(__name__)

# Sentinel: beginning of time — all trades qualify on first run
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class RetrainingService:
    """Background asyncio.Task that monitors trade counts and triggers retraining."""

    def __init__(
        self,
        db_session_factory: Callable[[], Any],
        model_dir: str = "models/",
        check_interval_seconds: int = 3600,
        min_trades_for_retrain: int = 50,
        min_accuracy_threshold: float = 0.38,
        max_model_versions: int = 5,
        exchange_id: str = "binance",
    ) -> None:
        self._db_session_factory = db_session_factory
        self._model_dir = Path(model_dir)
        self._check_interval_seconds = check_interval_seconds
        self._min_trades = min_trades_for_retrain
        self._min_accuracy = min_accuracy_threshold
        self._max_versions = max_model_versions
        self._exchange_id = exchange_id
        self._log = logger.bind(component="retraining_service")
        self._task: asyncio.Task[None] | None = None
        # In-memory watermark: keyed by "{symbol}::{timeframe}"
        self._last_retrain_at: dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Create the background polling asyncio.Task."""
        if self._task is not None and not self._task.done():
            self._log.warning("retraining_service.already_running")
            return
        self._task = asyncio.create_task(
            self._poll_loop(), name="retraining_service_poll"
        )
        self._log.info(
            "retraining_service.started",
            interval_seconds=self._check_interval_seconds,
            min_trades=self._min_trades,
        )

    async def stop(self) -> None:
        """Cancel the background polling task and await its completion."""
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._log.info("retraining_service.stopped")

    # ------------------------------------------------------------------ #
    # Poll loop
    # ------------------------------------------------------------------ #

    async def _poll_loop(self) -> None:
        """Main polling loop: sleep first, then check all (symbol, timeframe) pairs."""
        while True:
            try:
                await asyncio.sleep(self._check_interval_seconds)
            except asyncio.CancelledError:
                return

            self._log.debug("retraining_service.poll_tick")
            pairs = await self._get_active_symbol_timeframe_pairs()
            for symbol, timeframe in pairs:
                try:
                    await self._check_and_retrain(symbol, timeframe)
                except Exception as exc:
                    # Never abort the loop — log per-symbol errors and continue
                    self._log.error(
                        "retraining_service.check_failed",
                        symbol=symbol,
                        timeframe=timeframe,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

    # ------------------------------------------------------------------ #
    # Active symbol/timeframe discovery
    # ------------------------------------------------------------------ #

    async def _get_active_symbol_timeframe_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """Query DB for distinct (symbol, timeframe) pairs with is_active=True."""
        from sqlalchemy import select

        from api.db.models import ModelVersionORM

        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelVersionORM.symbol, ModelVersionORM.timeframe)
                .where(ModelVersionORM.is_active.is_(True))
                .distinct()
            )
            rows = result.all()
        return [(row[0], row[1]) for row in rows]

    # ------------------------------------------------------------------ #
    # Trade-count check
    # ------------------------------------------------------------------ #

    async def _check_and_retrain(self, symbol: str, timeframe: str) -> None:
        """Count trades since last retrain; trigger retraining if threshold met."""
        from sqlalchemy import func, select

        from api.db.models import TradeORM

        key = f"{symbol}::{timeframe}"
        last_at = self._last_retrain_at.get(key, _EPOCH)

        async with self._db_session_factory() as session:
            count_result = await session.execute(
                select(func.count())
                .select_from(TradeORM)
                .where(
                    TradeORM.strategy_id == "model_strategy",
                    TradeORM.symbol == symbol,
                    TradeORM.exit_at > last_at,
                )
            )
            trade_count = count_result.scalar_one()

        self._log.debug(
            "retraining_service.trade_count",
            symbol=symbol,
            timeframe=timeframe,
            count=trade_count,
            threshold=self._min_trades,
            last_retrain_at=last_at.isoformat(),
        )

        if trade_count >= self._min_trades:
            self._log.info(
                "retraining_service.threshold_met",
                symbol=symbol,
                timeframe=timeframe,
                trade_count=trade_count,
            )
            await self._do_retrain(symbol=symbol, timeframe=timeframe, trigger="auto")

    # ------------------------------------------------------------------ #
    # Retraining pipeline
    # ------------------------------------------------------------------ #

    async def _do_retrain(
        self,
        symbol: str,
        timeframe: str,
        trigger: str = "auto",
    ) -> None:
        """Execute the full retraining pipeline for one (symbol, timeframe) pair.

        Steps
        -----
        1. Fetch OHLCV via synchronous CCXT in asyncio.to_thread (1000 bars).
        2. Query all closed model_strategy trades for the symbol from DB.
        3. Run prepare_dataset_from_trades + train in asyncio.to_thread.
        4. Validate accuracy >= min_accuracy_threshold.
        5. Save versioned .joblib file.
        6. Write ModelVersionORM row (deactivates old, activates new in one txn).
        7. Prune old versions beyond max_model_versions.
        8. Write sidecar JSON atomically.
        """
        self._log.info(
            "retraining_service.retrain_start",
            symbol=symbol,
            timeframe=timeframe,
            trigger=trigger,
        )

        # 1. Fetch OHLCV in thread
        try:
            ohlcv_df = await asyncio.to_thread(
                self._fetch_ohlcv_sync,
                symbol=symbol,
                timeframe=timeframe,
                bars=1000,
            )
        except Exception as exc:
            self._log.error(
                "retraining_service.ohlcv_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            return

        # 2. Query trades from DB
        trade_dicts = await self._fetch_trade_dicts(symbol)
        if len(trade_dicts) < self._min_trades:
            self._log.warning(
                "retraining_service.insufficient_trades",
                symbol=symbol,
                count=len(trade_dicts),
                required=self._min_trades,
            )
            return

        # 3. Train in thread (scikit-learn is CPU-bound + synchronous)
        version_id = uuid.uuid4()
        version_suffix = version_id.hex[:8]

        try:
            metrics: dict[str, Any] = await asyncio.to_thread(
                self._train_sync,
                trade_dicts=trade_dicts,
                ohlcv_df=ohlcv_df,
                timeframe=timeframe,
                symbol=symbol,
                version_suffix=version_suffix,
            )
        except ValueError as exc:
            self._log.warning(
                "retraining_service.dataset_error",
                symbol=symbol,
                error=str(exc),
            )
            return
        except Exception as exc:
            self._log.error(
                "retraining_service.train_failed",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return

        accuracy: float = metrics["accuracy"]
        model_path: str = metrics["model_path"]
        n_trades: int = metrics["n_trades"]
        n_bars: int = metrics["n_bars"]

        # 4. Accuracy gate
        if accuracy < self._min_accuracy:
            self._log.warning(
                "retraining_service.accuracy_below_threshold",
                symbol=symbol,
                accuracy=accuracy,
                threshold=self._min_accuracy,
                model_path=model_path,
            )
            # Model saved to disk but not activated — useful for debugging
            return

        # 5. Register in DB (deactivate old, insert new, all in one transaction)
        mv = await self._register_model_version(
            version_id=version_id,
            symbol=symbol,
            timeframe=timeframe,
            accuracy=accuracy,
            n_trades_used=n_trades,
            n_bars_used=n_bars,
            model_path=model_path,
            trigger=trigger,
            extra=metrics.get("extra"),
        )

        # 6. Prune old versions
        await self._prune_old_versions(symbol=symbol, timeframe=timeframe)

        # 7. Write sidecar JSON atomically
        self._write_active_sidecar(
            symbol=symbol,
            version_id=str(mv.id),
            model_path=model_path,
            accuracy=accuracy,
        )

        # Update in-memory watermark
        key = f"{symbol}::{timeframe}"
        self._last_retrain_at[key] = datetime.now(tz=UTC)

        self._log.info(
            "retraining_service.retrain_complete",
            symbol=symbol,
            timeframe=timeframe,
            version_id=str(mv.id),
            accuracy=accuracy,
            n_trades=n_trades,
        )

    # ------------------------------------------------------------------ #
    # Synchronous helpers (run inside asyncio.to_thread)
    # ------------------------------------------------------------------ #

    def _fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
    ) -> Any:
        """Fetch OHLCV via synchronous CCXT. Returns a pandas DataFrame."""
        import ccxt
        import pandas as pd

        exchange_cls = getattr(ccxt, self._exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Exchange '{self._exchange_id}' not supported by ccxt")

        exc_instance = exchange_cls({"enableRateLimit": True})
        try:
            raw: list[list[Any]] = exc_instance.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=bars,
            )
        finally:
            try:
                exc_instance.close()
            except Exception:
                pass

        if not raw:
            raise ValueError(f"No OHLCV data returned for {symbol}")

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        return df

    def _train_sync(
        self,
        trade_dicts: list[dict[str, Any]],
        ohlcv_df: Any,
        timeframe: str,
        symbol: str,
        version_suffix: str,
    ) -> dict[str, Any]:
        """Synchronous training pipeline. Runs in asyncio.to_thread."""
        from data.ml_training import ModelTrainer

        trainer = ModelTrainer(model_dir=str(self._model_dir))
        X, y = trainer.prepare_dataset_from_trades(
            trades=trade_dicts,
            ohlcv_df=ohlcv_df,
            timeframe=timeframe,
        )
        metrics = trainer.train(X, y)
        model_path = trainer.save_model(symbol=symbol, version_suffix=version_suffix)

        return {
            "accuracy": metrics["accuracy"],
            "model_path": str(model_path),
            "n_trades": len(trade_dicts),
            "n_bars": len(ohlcv_df),
            "extra": {
                "feature_importances": metrics.get("feature_importances", {}),
                "classification_report": metrics.get("classification_report", {}),
                "train_samples": metrics.get("train_samples"),
                "test_samples": metrics.get("test_samples"),
            },
        }

    # ------------------------------------------------------------------ #
    # DB helpers
    # ------------------------------------------------------------------ #

    async def _fetch_trade_dicts(self, symbol: str) -> list[dict[str, Any]]:
        """Query all closed model_strategy trades for the symbol and return as dicts."""
        from sqlalchemy import select

        from api.db.models import TradeORM

        async with self._db_session_factory() as session:
            result = await session.execute(
                select(
                    TradeORM.side,
                    TradeORM.realised_pnl,
                    TradeORM.entry_price,
                    TradeORM.quantity,
                    TradeORM.entry_at,
                )
                .where(
                    TradeORM.strategy_id == "model_strategy",
                    TradeORM.symbol == symbol,
                )
                .order_by(TradeORM.entry_at)
            )
            rows = result.all()

        return [
            {
                "side": row[0],
                "realised_pnl": row[1],
                "entry_price": row[2],
                "quantity": row[3],
                "entry_at": row[4],
            }
            for row in rows
        ]

    async def _register_model_version(
        self,
        version_id: uuid.UUID,
        symbol: str,
        timeframe: str,
        accuracy: float,
        n_trades_used: int,
        n_bars_used: int,
        model_path: str,
        trigger: str,
        extra: dict[str, Any] | None,
    ) -> Any:
        """Deactivate existing active model, then INSERT new active model version."""
        from sqlalchemy import select, update

        from api.db.models import ModelVersionORM

        now = datetime.now(tz=UTC)

        async with self._db_session_factory() as session:
            async with session.begin():
                # Deactivate all currently active models for this symbol+timeframe
                await session.execute(
                    update(ModelVersionORM)
                    .where(
                        ModelVersionORM.symbol == symbol,
                        ModelVersionORM.timeframe == timeframe,
                        ModelVersionORM.is_active.is_(True),
                    )
                    .values(is_active=False)
                )

                # Insert new active version
                new_mv = ModelVersionORM(
                    id=version_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    trained_at=now,
                    accuracy=accuracy,
                    n_trades_used=n_trades_used,
                    n_bars_used=n_bars_used,
                    label_method="trade_outcome",
                    model_path=model_path,
                    is_active=True,
                    trigger=trigger,
                    extra=extra,
                )
                session.add(new_mv)

            # Refresh outside begin block to get server defaults
            await session.refresh(new_mv)

        return new_mv

    async def _prune_old_versions(self, symbol: str, timeframe: str) -> None:
        """Delete .joblib files and DB rows for versions beyond max_model_versions."""
        from sqlalchemy import delete, select

        from api.db.models import ModelVersionORM

        async with self._db_session_factory() as session:
            result = await session.execute(
                select(ModelVersionORM)
                .where(
                    ModelVersionORM.symbol == symbol,
                    ModelVersionORM.timeframe == timeframe,
                )
                .order_by(ModelVersionORM.trained_at.desc())
            )
            all_versions: list[ModelVersionORM] = list(result.scalars().all())

        if len(all_versions) <= self._max_versions:
            return

        to_prune = all_versions[self._max_versions:]
        for mv in to_prune:
            if mv.is_active:
                # Safety: never delete the active model
                continue
            # Delete .joblib file from disk
            model_file = Path(mv.model_path)
            if model_file.exists():
                try:
                    model_file.unlink()
                    self._log.info(
                        "retraining_service.model_file_deleted",
                        path=str(model_file),
                    )
                except OSError as exc:
                    self._log.warning(
                        "retraining_service.model_file_delete_failed",
                        path=str(model_file),
                        error=str(exc),
                    )
            # Delete DB row
            async with self._db_session_factory() as session:
                async with session.begin():
                    await session.execute(
                        delete(ModelVersionORM).where(ModelVersionORM.id == mv.id)
                    )
            self._log.info(
                "retraining_service.version_pruned",
                version_id=str(mv.id),
                symbol=symbol,
                timeframe=timeframe,
            )

    # ------------------------------------------------------------------ #
    # Sidecar JSON (hot-swap mechanism)
    # ------------------------------------------------------------------ #

    def _write_active_sidecar(
        self,
        symbol: str,
        version_id: str,
        model_path: str,
        accuracy: float,
    ) -> None:
        """Atomically write the active model sidecar JSON file.

        ModelStrategy reads this file on each on_bar() call. Atomic write
        (tmp file + os.replace) prevents a partial-read race condition.

        File: models/{safe_symbol}_active.json
        """
        self._model_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(" ", "_").lower()
        sidecar_path = self._model_dir / f"{safe_symbol}_active.json"
        tmp_path = self._model_dir / f"{safe_symbol}_active.json.tmp"

        payload = {
            "version_id": version_id,
            "model_path": model_path,
            "accuracy": accuracy,
            "trained_at": datetime.now(tz=UTC).isoformat(),
        }
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, sidecar_path)

        self._log.info(
            "retraining_service.sidecar_written",
            symbol=symbol,
            sidecar_path=str(sidecar_path),
            version_id=version_id,
        )

    # ------------------------------------------------------------------ #
    # Public API: manual retrain
    # ------------------------------------------------------------------ #

    async def manual_retrain(self, symbol: str, timeframe: str) -> None:
        """Trigger an immediate retraining cycle for the given symbol/timeframe.

        Called by POST /api/v1/ml/retrain/{symbol} API endpoint.
        Runs the full pipeline synchronously (from caller's perspective —
        still uses asyncio.to_thread internally for CPU-bound work).

        Raises
        ------
        ValueError
            If there are insufficient trades to build a training dataset.
        RuntimeError
            If OHLCV fetching or training fails.
        """
        await self._do_retrain(symbol=symbol, timeframe=timeframe, trigger="manual")
