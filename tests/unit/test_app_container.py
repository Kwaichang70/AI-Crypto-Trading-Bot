"""
tests/unit/test_app_container.py
---------------------------------
Unit tests for AppContainer, BackgroundTaskRegistry, and deps.py providers.

All tests are pure-Python; no database, network, or filesystem I/O.
asyncio_mode = "auto" (set in pyproject.toml) so async tests need no
explicit decorator.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from api.config import Settings
from api.container import AppContainer, BackgroundTaskRegistry, ServiceRegistry
from api.deps import (
    get_container,
    get_db_engine,
    get_retraining_service,
    get_run_registry,
    get_telegram_notifier,
)
from api.run_registry import RunRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    """Construct a minimal Settings instance without .env file parsing."""
    return Settings(
        database_url="postgresql+asyncpg://trading:test@localhost:5432/trading_bot"
    )


def _make_request(container: AppContainer | None = None) -> Any:
    """Build a minimal fake Request-like object with app.state.container."""
    app = FastAPI()
    if container is not None:
        app.state.container = container
    return SimpleNamespace(app=app)


# ---------------------------------------------------------------------------
# AC-001: for_testing() default construction
# ---------------------------------------------------------------------------


class TestForTestingDefaultConstruction:
    """AC-001 — for_testing() with default settings returns fully-None services."""

    def test_services_all_none(self) -> None:
        container = AppContainer.for_testing()
        assert container.services.retraining_service is None
        assert container.services.fgi_client is None
        assert container.services.coingecko_client is None
        assert container.services.fred_client is None
        assert container.services.whale_alert_client is None
        assert container.services.telegram_notifier is None

    def test_db_engine_none(self) -> None:
        container = AppContainer.for_testing()
        assert container.db_engine is None

    def test_run_registry_is_instance(self) -> None:
        container = AppContainer.for_testing()
        assert isinstance(container.run_registry, RunRegistry)

    def test_background_tasks_all_none(self) -> None:
        container = AppContainer.for_testing()
        assert container.background_tasks.equity_prune_task is None


# ---------------------------------------------------------------------------
# AC-002: for_testing() with custom settings
# ---------------------------------------------------------------------------


class TestForTestingCustomSettings:
    """AC-002 — for_testing(settings=custom) uses provided settings."""

    def test_custom_settings_stored(self) -> None:
        custom = _make_settings()
        container = AppContainer.for_testing(settings=custom)
        assert container.settings is custom

    def test_settings_identity_preserved(self) -> None:
        custom = _make_settings()
        container = AppContainer.for_testing(settings=custom)
        assert container.settings is custom


# ---------------------------------------------------------------------------
# AC-003: for_testing() with service overrides
# ---------------------------------------------------------------------------


class TestForTestingOverrides:
    """AC-003 — for_testing(**overrides) applies overrides."""

    def test_telegram_notifier_override(self) -> None:
        mock_notifier = MagicMock()
        container = AppContainer.for_testing(telegram_notifier=mock_notifier)
        assert container.services.telegram_notifier is mock_notifier

    def test_db_engine_override(self) -> None:
        mock_engine = MagicMock()
        container = AppContainer.for_testing(db_engine=mock_engine)
        assert container.db_engine is mock_engine

    def test_unknown_override_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="unknown override key"):
            AppContainer.for_testing(nonexistent_field=True)

    def test_background_tasks_override(self) -> None:
        custom_bg = BackgroundTaskRegistry()
        container = AppContainer.for_testing(background_tasks=custom_bg)
        assert container.background_tasks is custom_bg

    def test_run_registry_override(self) -> None:
        custom_reg = RunRegistry()
        container = AppContainer.for_testing(run_registry=custom_reg)
        assert container.run_registry is custom_reg


# ---------------------------------------------------------------------------
# AC-004: health() all-False on fresh container
# ---------------------------------------------------------------------------


class TestHealthAllFalse:
    """AC-004 — health() returns all-False on a fresh container."""

    def test_all_false(self) -> None:
        container = AppContainer.for_testing()
        snapshot = container.health()
        expected_keys = {
            "db_engine",
            "telegram_notifier",
            "retraining_service",
            "fgi_client",
            "coingecko_client",
            "fred_client",
            "whale_alert_client",
            "equity_prune_task",
        }
        assert set(snapshot.keys()) == expected_keys
        assert all(v is False for v in snapshot.values())


# ---------------------------------------------------------------------------
# AC-005: health() reflects populated services
# ---------------------------------------------------------------------------


class TestHealthReflectsPopulation:
    """AC-005 — health() returns True for populated fields."""

    def test_db_engine_true(self) -> None:
        container = AppContainer.for_testing(db_engine=MagicMock())
        assert container.health()["db_engine"] is True

    def test_telegram_notifier_true(self) -> None:
        container = AppContainer.for_testing(telegram_notifier=MagicMock())
        assert container.health()["telegram_notifier"] is True

    def test_fgi_client_true(self) -> None:
        container = AppContainer.for_testing(fgi_client=MagicMock())
        assert container.health()["fgi_client"] is True

    async def test_equity_prune_task_true(self) -> None:
        container = AppContainer.for_testing()
        task: asyncio.Task[None] = asyncio.ensure_future(asyncio.sleep(9999))
        container.background_tasks.equity_prune_task = task
        try:
            assert container.health()["equity_prune_task"] is True
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# AC-006: shutdown() on fresh container is no-op
# ---------------------------------------------------------------------------


class TestShutdownFreshNoOp:
    """AC-006 — shutdown() on a fresh container completes without error."""

    async def test_shutdown_fresh_no_error(self) -> None:
        container = AppContainer.for_testing()
        await container.shutdown()  # Must not raise


# ---------------------------------------------------------------------------
# AC-007: shutdown() cancels background_tasks then run_registry
# ---------------------------------------------------------------------------


class TestShutdownCancelOrder:
    """AC-007 — shutdown() cancels background tasks and run registry."""

    async def test_shutdown_cancels_equity_prune_task(self) -> None:
        container = AppContainer.for_testing()

        async def long_running() -> None:
            await asyncio.sleep(9999)

        task: asyncio.Task[None] = asyncio.ensure_future(long_running())
        container.background_tasks.equity_prune_task = task

        await container.shutdown()

        assert task.cancelled()

    async def test_shutdown_calls_run_registry_cancel_all(self) -> None:
        container = AppContainer.for_testing()
        container.run_registry.cancel_all = AsyncMock()  # type: ignore[method-assign]

        await container.shutdown()

        container.run_registry.cancel_all.assert_awaited_once()


# ---------------------------------------------------------------------------
# AC-008: shutdown() is idempotent
# ---------------------------------------------------------------------------


class TestShutdownIdempotent:
    """AC-008 — shutdown() can be called multiple times without error."""

    async def test_double_shutdown(self) -> None:
        container = AppContainer.for_testing()
        await container.shutdown()
        await container.shutdown()  # Must not raise


# ---------------------------------------------------------------------------
# BackgroundTaskRegistry.cancel_all
# ---------------------------------------------------------------------------


class TestBackgroundTaskRegistryCancelAll:
    """cancel_all() cancels a pending task and tolerates a finished task."""

    async def test_cancel_all_cancels_pending_task(self) -> None:
        registry = BackgroundTaskRegistry()

        async def blocker() -> None:
            await asyncio.sleep(9999)

        task: asyncio.Task[None] = asyncio.ensure_future(blocker())
        registry.equity_prune_task = task

        await registry.cancel_all(timeout=1.0)

        assert task.cancelled()

    async def test_cancel_all_tolerates_finished_task(self) -> None:
        registry = BackgroundTaskRegistry()

        async def noop() -> None:
            return

        task: asyncio.Task[None] = asyncio.ensure_future(noop())
        await asyncio.sleep(0)  # Let it complete
        registry.equity_prune_task = task

        await registry.cancel_all(timeout=1.0)  # Must not raise


# ---------------------------------------------------------------------------
# deps.py: get_container raises when not populated
# ---------------------------------------------------------------------------


class TestDepsGetContainerRaises:
    """get_container raises RuntimeError when container is absent."""

    def test_raises_runtime_error(self) -> None:
        request = _make_request(container=None)
        with pytest.raises(RuntimeError, match="AppContainer is not initialised"):
            get_container(request)


# ---------------------------------------------------------------------------
# deps.py: get_telegram_notifier
# ---------------------------------------------------------------------------


class TestDepsGetTelegramNotifier:
    """get_telegram_notifier returns None when notifier is None."""

    def test_returns_none_when_not_configured(self) -> None:
        container = AppContainer.for_testing()
        request = _make_request(container=container)
        assert get_telegram_notifier(request) is None

    def test_returns_notifier_when_configured(self) -> None:
        mock_notifier = MagicMock()
        container = AppContainer.for_testing(telegram_notifier=mock_notifier)
        request = _make_request(container=container)
        assert get_telegram_notifier(request) is mock_notifier

    def test_returns_none_when_container_absent(self) -> None:
        request = _make_request(container=None)
        assert get_telegram_notifier(request) is None


# ---------------------------------------------------------------------------
# deps.py: get_retraining_service
# ---------------------------------------------------------------------------


class TestDepsGetRetrainingService:
    """get_retraining_service returns None when service is None or container absent."""

    def test_returns_none_when_not_configured(self) -> None:
        container = AppContainer.for_testing()
        request = _make_request(container=container)
        assert get_retraining_service(request) is None

    def test_returns_none_when_container_absent(self) -> None:
        request = _make_request(container=None)
        assert get_retraining_service(request) is None


# ---------------------------------------------------------------------------
# deps.py: get_db_engine
# ---------------------------------------------------------------------------


class TestDepsGetDbEngine:
    """get_db_engine raises RuntimeError when db_engine is None."""

    def test_raises_when_engine_none(self) -> None:
        container = AppContainer.for_testing()
        request = _make_request(container=container)
        with pytest.raises(RuntimeError, match="Database engine is not initialised"):
            get_db_engine(request)

    def test_returns_engine_when_present(self) -> None:
        mock_engine = MagicMock()
        container = AppContainer.for_testing(db_engine=mock_engine)
        request = _make_request(container=container)
        assert get_db_engine(request) is mock_engine


# ---------------------------------------------------------------------------
# deps.py: get_run_registry
# ---------------------------------------------------------------------------


class TestDepsGetRunRegistry:
    """get_run_registry returns container.run_registry."""

    def test_returns_run_registry(self) -> None:
        container = AppContainer.for_testing()
        request = _make_request(container=container)
        registry = get_run_registry(request)
        assert registry is container.run_registry
        assert isinstance(registry, RunRegistry)
