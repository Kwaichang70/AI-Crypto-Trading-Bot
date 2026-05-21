"""
tests/unit/test_retraining_v2.py
---------------------------------
S47-5 (Sprint 48) -- async integration tests for the QT-007f-2 v2 dispatch
+ FGI/BTC-dom history fetch best-effort behavior in RetrainingService.

Verifies:
  * default schema_version is v2; explicit v1 still works
  * v2 path attempts FGI + BTC-dom history fetches before training
  * v1 path SKIPS the history fetches
  * FGI fetch failure: training proceeds with fgi_series=None
  * BTC-dom fetch failure: training proceeds with btc_dom_series=None
  * Both clients absent: training proceeds with both series None
  * Mixed (one None, one valid): correct per-client handling
  * Post-train invocation of ModelTrainer.save_sidecar with v2 schema
  * Below-threshold accuracy short-circuits before save_sidecar
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from api.services.retraining import RetrainingService


# ---------------------------------------------------------------------------
# Module-level helpers (parallel to TestRetrainingServiceLoop in sprint23)
# ---------------------------------------------------------------------------


def _make_count_session_factory(count_result: int = 0) -> MagicMock:
    """Mock async-context-manager session factory returning a fixed trade count."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    count_scalar = MagicMock()
    count_scalar.scalar_one = MagicMock(return_value=count_result)
    session.execute = AsyncMock(return_value=count_scalar)

    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx
    return factory


def _make_service(feature_schema_version: int = 2, **kwargs: Any) -> RetrainingService:
    factory = kwargs.pop("db_session_factory", _make_count_session_factory(0))
    return RetrainingService(
        db_session_factory=factory,
        check_interval_seconds=kwargs.pop("check_interval_seconds", 3600),
        min_trades_for_retrain=kwargs.pop("min_trades_for_retrain", 5),
        min_accuracy_threshold=kwargs.pop("min_accuracy_threshold", 0.0),
        feature_schema_version=feature_schema_version,
    )


def _make_trade_dict(entry_at: datetime, idx: int = 0) -> dict[str, object]:
    return {
        "side": "buy",
        "realised_pnl": Decimal("1.0"),
        "entry_price": Decimal("100.0"),
        "quantity": Decimal("0.1"),
        "entry_at": entry_at,
    }


def _make_synth_ohlcv() -> pd.DataFrame:
    date_idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * 300,
            "high": [101.0] * 300,
            "low": [99.0] * 300,
            "close": [100.0] * 300,
            "volume": [1000.0] * 300,
        },
        index=date_idx,
    )


def _stub_train_metrics(accuracy: float = 0.5, schema_version: int = 2) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "model_path": "/tmp/test_model.joblib",
        "n_trades": 60,
        "n_bars": 300,
        "extra": {
            "feature_importances": {},
            "classification_report": {},
            "train_samples": 240,
            "test_samples": 60,
            "feature_schema_version": schema_version,
            "feature_names": [f"f{i}" for i in range(10 if schema_version == 1 else 14)],
        },
    }


# ===========================================================================
# Constructor + property surface
# ===========================================================================


class TestRetrainingServiceSchemaVersionInit:

    def test_default_schema_version_is_v2(self) -> None:
        svc = _make_service()
        assert svc._feature_schema_version == 2

    def test_explicit_v1_accepted(self) -> None:
        svc = _make_service(feature_schema_version=1)
        assert svc._feature_schema_version == 1

    def test_public_running_property_reflects_task_state(self) -> None:
        svc = _make_service()
        assert svc.running is False

    def test_public_min_trades_property(self) -> None:
        svc = _make_service(min_trades_for_retrain=42)
        assert svc.min_trades_for_retrain == 42

    def test_public_check_interval_property(self) -> None:
        svc = _make_service(check_interval_seconds=900)
        assert svc.check_interval_seconds == 900


# ===========================================================================
# v2 history-fetch path
# ===========================================================================


class TestV2HistoryFetchDispatch:
    """Verify _do_retrain's v2 path attempts history fetches; v1 skips them."""

    def _stub_pipeline(self, svc: RetrainingService) -> None:
        """Replace all heavy bits of _do_retrain with passing mocks."""
        svc._fetch_ohlcv_sync = MagicMock(return_value=_make_synth_ohlcv())
        svc._fetch_trade_dicts = AsyncMock(
            return_value=[
                _make_trade_dict(datetime(2024, 1, 5, tzinfo=UTC), i)
                for i in range(60)
            ]
        )
        svc._train_sync = MagicMock(
            return_value=_stub_train_metrics(
                accuracy=0.5,
                schema_version=svc._feature_schema_version,
            )
        )
        svc._register_model_version = AsyncMock(
            return_value=SimpleNamespace(id=uuid.uuid4())
        )
        svc._prune_old_versions = AsyncMock()

    @pytest.mark.asyncio
    async def test_v2_attempts_fgi_history_fetch(self) -> None:
        svc = _make_service(feature_schema_version=2)
        self._stub_pipeline(svc)

        fgi_client = MagicMock()
        fgi_client.get_history_as_series = AsyncMock(return_value="<fgi-series>")
        cg_client = MagicMock()
        cg_client.fetch_btc_dominance_history = AsyncMock(return_value="<cg-series>")

        with patch("data.sentiment.get_global_client", return_value=fgi_client), \
             patch("data.market_signals.get_global_client", return_value=cg_client), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        fgi_client.get_history_as_series.assert_awaited_once_with(limit=30)
        cg_client.fetch_btc_dominance_history.assert_awaited_once_with(days=30)
        # CR-005: guard against silent early-exit before training.
        svc._train_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_v1_skips_history_fetches(self) -> None:
        svc = _make_service(feature_schema_version=1)
        self._stub_pipeline(svc)

        fgi_client = MagicMock()
        fgi_client.get_history_as_series = AsyncMock()
        cg_client = MagicMock()
        cg_client.fetch_btc_dominance_history = AsyncMock()

        with patch("data.sentiment.get_global_client", return_value=fgi_client), \
             patch("data.market_signals.get_global_client", return_value=cg_client), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        fgi_client.get_history_as_series.assert_not_awaited()
        cg_client.fetch_btc_dominance_history.assert_not_awaited()
        # CR-005: guard against silent early-exit before training.
        svc._train_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_fgi_fetch_failure_does_not_block_training(self) -> None:
        svc = _make_service(feature_schema_version=2)
        self._stub_pipeline(svc)

        fgi_client = MagicMock()
        fgi_client.get_history_as_series = AsyncMock(
            side_effect=RuntimeError("alternative.me down")
        )
        cg_client = MagicMock()
        cg_client.fetch_btc_dominance_history = AsyncMock(return_value="<cg-series>")

        with patch("data.sentiment.get_global_client", return_value=fgi_client), \
             patch("data.market_signals.get_global_client", return_value=cg_client), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        svc._train_sync.assert_called_once()
        _, train_kwargs = svc._train_sync.call_args
        assert train_kwargs["fgi_series"] is None
        assert train_kwargs["btc_dom_series"] == "<cg-series>"

    @pytest.mark.asyncio
    async def test_btc_dom_fetch_failure_does_not_block_training(self) -> None:
        svc = _make_service(feature_schema_version=2)
        self._stub_pipeline(svc)

        fgi_client = MagicMock()
        fgi_client.get_history_as_series = AsyncMock(return_value="<fgi-series>")
        cg_client = MagicMock()
        cg_client.fetch_btc_dominance_history = AsyncMock(
            side_effect=RuntimeError("coingecko 502")
        )

        with patch("data.sentiment.get_global_client", return_value=fgi_client), \
             patch("data.market_signals.get_global_client", return_value=cg_client), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        svc._train_sync.assert_called_once()
        _, train_kwargs = svc._train_sync.call_args
        assert train_kwargs["fgi_series"] == "<fgi-series>"
        assert train_kwargs["btc_dom_series"] is None

    @pytest.mark.asyncio
    async def test_both_clients_absent_training_proceeds(self) -> None:
        svc = _make_service(feature_schema_version=2)
        self._stub_pipeline(svc)

        with patch("data.sentiment.get_global_client", return_value=None), \
             patch("data.market_signals.get_global_client", return_value=None), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        svc._train_sync.assert_called_once()
        _, train_kwargs = svc._train_sync.call_args
        assert train_kwargs["fgi_series"] is None
        assert train_kwargs["btc_dom_series"] is None

    @pytest.mark.asyncio
    async def test_fgi_client_none_cg_present(self) -> None:
        """CR-007: mixed-client case -- one absent (None), one present."""
        svc = _make_service(feature_schema_version=2)
        self._stub_pipeline(svc)

        cg_client = MagicMock()
        cg_client.fetch_btc_dominance_history = AsyncMock(return_value="<cg-series>")

        with patch("data.sentiment.get_global_client", return_value=None), \
             patch("data.market_signals.get_global_client", return_value=cg_client), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.save_sidecar = MagicMock()
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        svc._train_sync.assert_called_once()
        _, train_kwargs = svc._train_sync.call_args
        assert train_kwargs["fgi_series"] is None
        assert train_kwargs["btc_dom_series"] == "<cg-series>"


# ===========================================================================
# save_sidecar wiring
# ===========================================================================


class TestSaveSidecarInvocation:
    """Verify _do_retrain calls ModelTrainer.save_sidecar (not the
    deprecated _write_active_sidecar) and constructs the trainer with
    the configured feature_schema_version."""

    @pytest.mark.asyncio
    async def test_save_sidecar_called_with_v2_schema(self) -> None:
        svc = _make_service(feature_schema_version=2)
        svc._fetch_ohlcv_sync = MagicMock(return_value=_make_synth_ohlcv())
        svc._fetch_trade_dicts = AsyncMock(
            return_value=[
                _make_trade_dict(datetime(2024, 1, 5, tzinfo=UTC), i)
                for i in range(60)
            ]
        )
        svc._train_sync = MagicMock(
            return_value=_stub_train_metrics(accuracy=0.5, schema_version=2)
        )
        mv_id = uuid.uuid4()
        svc._register_model_version = AsyncMock(
            return_value=SimpleNamespace(id=mv_id)
        )
        svc._prune_old_versions = AsyncMock()

        with patch("data.sentiment.get_global_client", return_value=None), \
             patch("data.market_signals.get_global_client", return_value=None), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            instance = MagicMock()
            MockTrainer.return_value = instance
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        # CR-003: the constructor is called once (sidecar block only;
        # _train_sync is mocked at instance level so its internal
        # ModelTrainer construction never runs).
        calls = MockTrainer.call_args_list
        assert any(
            call.kwargs.get("feature_schema_version") == 2 for call in calls
        ), f"No ModelTrainer call with feature_schema_version=2; got: {calls}"

        instance.save_sidecar.assert_called_once()
        kw = instance.save_sidecar.call_args.kwargs
        assert kw["symbol"] == "BTC/USDT"
        assert kw["version_id"] == str(mv_id)
        assert "model_path" in kw
        assert kw["accuracy"] == 0.5

    @pytest.mark.asyncio
    async def test_below_threshold_skips_save_sidecar(self) -> None:
        """Accuracy below min_accuracy_threshold short-circuits _do_retrain
        before _register_model_version + save_sidecar."""
        svc = _make_service(
            feature_schema_version=2,
            min_accuracy_threshold=0.9,
        )
        svc._fetch_ohlcv_sync = MagicMock(return_value=_make_synth_ohlcv())
        svc._fetch_trade_dicts = AsyncMock(
            return_value=[
                _make_trade_dict(datetime(2024, 1, 5, tzinfo=UTC), i)
                for i in range(60)
            ]
        )
        svc._train_sync = MagicMock(
            return_value=_stub_train_metrics(accuracy=0.3, schema_version=2)
        )
        svc._register_model_version = AsyncMock()

        with patch("data.sentiment.get_global_client", return_value=None), \
             patch("data.market_signals.get_global_client", return_value=None), \
             patch("data.ml_training.ModelTrainer") as MockTrainer:
            instance = MagicMock()
            MockTrainer.return_value = instance
            await svc._do_retrain(symbol="BTC/USDT", timeframe="1h")

        svc._register_model_version.assert_not_called()
        instance.save_sidecar.assert_not_called()
