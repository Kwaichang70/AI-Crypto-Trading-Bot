"""
apps/api/deps.py
----------------
FastAPI Depends providers for AppContainer-managed resources.

Each provider reads from ``request.app.state.container`` which is
populated by the main.py lifespan hook (Stap 1c).  Required providers
raise RuntimeError when the resource is unavailable so callers receive
a clear 500 rather than a confusing AttributeError deep in business
logic.  Optional providers return None so routes can degrade gracefully
(e.g. skip Telegram notifications when the notifier is not configured).

Usage in a router::

    from api.deps import get_run_registry, get_telegram_notifier

    @router.post("/runs")
    async def start_run(
        registry: RunRegistry = Depends(get_run_registry),
        notifier: Any | None = Depends(get_telegram_notifier),
    ) -> ...:
        ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request

from api.run_registry import RunRegistry

if TYPE_CHECKING:
    from api.container import AppContainer

logger = logging.getLogger(__name__)

__all__ = [
    "get_container",
    "get_db_engine",
    "get_retraining_service",
    "get_run_registry",
    "get_telegram_notifier",
]


def get_container(request: Request) -> "AppContainer":
    """Return the AppContainer from app state.

    Raises
    ------
    RuntimeError
        When ``app.state.container`` has not been set by the lifespan hook.
    """
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError(
            "AppContainer is not initialised. "
            "Ensure the lifespan hook has completed before serving requests."
        )
    return container


def get_run_registry(request: Request) -> RunRegistry:
    """Shortcut provider for ``container.run_registry``."""
    return get_container(request).run_registry


def get_telegram_notifier(request: Request) -> Any | None:
    """Return ``container.services.telegram_notifier`` (may be None)."""
    try:
        container = get_container(request)
    except RuntimeError:
        logger.debug("get_telegram_notifier: container unavailable, returning None")
        return None
    return container.services.telegram_notifier


def get_retraining_service(request: Request) -> Any | None:
    """Return ``container.services.retraining_service`` (may be None)."""
    try:
        container = get_container(request)
    except RuntimeError:
        logger.debug("get_retraining_service: container unavailable, returning None")
        return None
    return container.services.retraining_service


def get_db_engine(request: Request) -> Any:
    """Return ``container.db_engine``.

    Raises
    ------
    RuntimeError
        When the database engine has not been initialised.
    """
    container = get_container(request)
    if container.db_engine is None:
        raise RuntimeError(
            "Database engine is not initialised. "
            "Check that the lifespan hook completed successfully and the "
            "DATABASE_URL environment variable is correctly configured."
        )
    return container.db_engine
