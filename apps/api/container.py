"""
apps/api/container.py
---------------------
Canonical owner of application-lifetime state.

AppContainer holds references to every long-lived resource: the
SQLAlchemy async engine, service clients (FGI, CoinGecko, FRED,
Whale Alert, Telegram, RetrainingService), named background asyncio
tasks, and the RunRegistry that tracks active trading runs.

Design notes (AR-003-02, AR-003-05, AR-003-07):
  - Services and background tasks are split into distinct dataclasses so
    each concern is independently replaceable in tests.
  - AppContainer.for_testing() builds a fully-None container without any
    network or filesystem I/O — safe to call in any test.
  - startup() is a placeholder; main.py lifespan (Stap 1c) populates
    services after the container exists.
  - shutdown() is idempotent: safe to call multiple times.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, fields as _dc_fields
from typing import TYPE_CHECKING, Any

from api.run_registry import RunRegistry

if TYPE_CHECKING:
    from api.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "AppContainer",
    "BackgroundTaskRegistry",
    "ServiceRegistry",
]


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------


@dataclass
class ServiceRegistry:
    """Long-lived service references (clients, notifiers, background services).

    All fields default to None so that for_testing() can construct a
    fully-inert container without touching the network.
    """

    retraining_service: Any | None = None  # RetrainingService
    fgi_client: Any | None = None           # FearGreedClient
    coingecko_client: Any | None = None     # CoinGeckoClient
    fred_client: Any | None = None          # FREDClient
    whale_alert_client: Any | None = None   # WhaleAlertClient
    telegram_notifier: Any | None = None    # TelegramNotifier


# ---------------------------------------------------------------------------
# BackgroundTaskRegistry
# ---------------------------------------------------------------------------


@dataclass
class BackgroundTaskRegistry:
    """Handles for long-lived asyncio tasks with cancel-all-on-shutdown semantics.

    Named task slots make it easy to check liveness and cancel individual
    tasks from health probes or lifespan hooks without keeping a raw list.
    """

    equity_prune_task: asyncio.Task[Any] | None = None
    # Reserved for future named long-lived tasks.

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """Cancel every registered task and wait up to *timeout* seconds.

        Finished or already-cancelled tasks are silently skipped.
        """
        tasks: list[asyncio.Task[Any]] = []
        if self.equity_prune_task is not None and not self.equity_prune_task.done():
            self.equity_prune_task.cancel()
            tasks.append(self.equity_prune_task)

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                logger.warning(
                    "background_tasks.cancel_all: %d task(s) did not finish within %.1fs",
                    len(pending),
                    timeout,
                )


# ---------------------------------------------------------------------------
# AppContainer
# ---------------------------------------------------------------------------


@dataclass
class AppContainer:
    """Canonical owner of application-lifetime state.

    Instantiate via AppContainer.for_testing() in tests, or via the
    main.py lifespan hook in production (Stap 1c).

    Attributes
    ----------
    settings:
        Snapshot of application settings taken at startup.  Required.
    db_engine:
        SQLAlchemy AsyncEngine.  None until startup() completes.
    services:
        Long-lived service client references.
    background_tasks:
        Named asyncio.Task handles for long-lived background work.
    run_registry:
        Tracks active trading run tasks and engines.
    """

    settings: "Settings"
    db_engine: Any | None = None
    services: ServiceRegistry = field(default_factory=ServiceRegistry)
    background_tasks: BackgroundTaskRegistry = field(
        default_factory=BackgroundTaskRegistry
    )
    run_registry: RunRegistry = field(default_factory=RunRegistry)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_testing(
        cls,
        settings: "Settings | None" = None,
        **overrides: Any,
    ) -> "AppContainer":
        """Build a container without network I/O — all service clients default to None.

        Parameters
        ----------
        settings:
            Optional Settings instance.  When omitted, a fresh Settings()
            is constructed with an explicit database_url so no .env file
            is required in test environments.
        **overrides:
            Keyword arguments forwarded as attribute overrides on the
            returned container.  Supported: any top-level AppContainer
            field (``db_engine``, ``services``, ``background_tasks``,
            ``run_registry``) or any field on ``ServiceRegistry``
            (e.g. ``telegram_notifier=mock``).
        """
        if settings is None:
            from api.config import Settings as _Settings  # noqa: PLC0415

            settings = _Settings(
                database_url="postgresql+asyncpg://trading:test@localhost:5432/trading_bot"
            )

        container = cls(settings=settings)

        top_level_fields = {"db_engine", "services", "background_tasks", "run_registry"}
        service_fields = {f.name for f in _dc_fields(ServiceRegistry)}

        service_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in top_level_fields:
                setattr(container, key, value)
            elif key in service_fields:
                service_overrides[key] = value
            else:
                raise TypeError(
                    f"AppContainer.for_testing() received unknown override key: {key!r}"
                )

        if service_overrides:
            for key, value in service_overrides.items():
                setattr(container.services, key, value)

        return container

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Placeholder — main.py lifespan populates services in Stap 1c.

        This method exists so callers can rely on a uniform startup API.
        """

    async def shutdown(self) -> None:
        """Cancel all background tasks and cancel all active runs.

        Shutdown order:
          1. background_tasks.cancel_all() — prune loops and similar
          2. run_registry.cancel_all()     — active trading run tasks

        Service-level close() calls (DB engine dispose, HTTP session
        close) belong in the main.py lifespan for Stap 1c.

        This method is idempotent: safe to call multiple times.
        """
        # CancelledError is BaseException in Python 3.8+; it propagates through except Exception correctly.
        try:
            await self.background_tasks.cancel_all()
        except Exception:  # noqa: BLE001
            logger.exception("Error cancelling background tasks during shutdown")

        # CancelledError is BaseException in Python 3.8+; it propagates through except Exception correctly.
        try:
            await self.run_registry.cancel_all()
        except Exception:  # noqa: BLE001
            logger.exception("Error cancelling run registry during shutdown")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def health(self) -> dict[str, bool]:
        """Return readiness snapshot.

        Each key is True when the component has been populated (non-None).

        Keys: db_engine, telegram_notifier, retraining_service, fgi_client,
        coingecko_client, fred_client, whale_alert_client, equity_prune_task.
        """
        return {
            "db_engine": self.db_engine is not None,
            "telegram_notifier": self.services.telegram_notifier is not None,
            "retraining_service": self.services.retraining_service is not None,
            "fgi_client": self.services.fgi_client is not None,
            "coingecko_client": self.services.coingecko_client is not None,
            "fred_client": self.services.fred_client is not None,
            "whale_alert_client": self.services.whale_alert_client is not None,
            "equity_prune_task": self.background_tasks.equity_prune_task is not None,
        }
