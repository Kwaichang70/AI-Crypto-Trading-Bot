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

import hashlib
import hmac
import logging
from typing import TYPE_CHECKING, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from api.config import Settings, get_settings
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
    "require_admin",
]

# ---------------------------------------------------------------------------
# Admin key security scheme (appears in OpenAPI docs)
# ---------------------------------------------------------------------------
_admin_key_header = APIKeyHeader(
    name="X-Admin-Key",
    auto_error=False,
    description=(
        "Admin API key for privileged operations (e.g. global kill-switch).  "
        "Separate from X-API-Key to allow independent rotation.  "
        "Generate with: openssl rand -hex 32"
    ),
)


async def require_admin(
    request: Request,
    header_key: str | None = Depends(_admin_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency that enforces admin-level authentication.

    Validates the X-Admin-Key header against settings.admin_api_key using
    hmac.compare_digest for constant-time comparison (prevents timing attacks).

    Design notes (Cycle 3 scope decision)
    --------------------------------------
    Uses the "separator token" pattern (a distinct second shared secret) rather
    than HMAC-signed role headers (Cycle 2 SR-001 alternative).  The single-admin
    operator topology + Tailscale network boundary makes this simpler pattern
    sufficient.  HMAC-signed role headers are deferred to Cycle 4+ when per-user
    audit trails and role granularity become necessary.

    Rotation strategy
    -----------------
    Rotating ADMIN_API_KEY only invalidates the kill-switch token — the regular
    X-API-Key (api_key_hash) remains unaffected.  Procedure:
    1. Generate a new key: openssl rand -hex 32
    2. Update ADMIN_API_KEY in .env on production server
    3. Restart API (settings are cached via lru_cache — requires restart)
    4. Invalidate the old token from all operator runbooks

    Raises
    ------
    HTTPException 401:
        When the X-Admin-Key header is absent, or when admin_api_key is not
        configured (indistinguishable by design — reveals nothing to callers).
    HTTPException 403:
        When the X-Admin-Key header is present but does not match.
    """
    admin_key_secret = settings.admin_api_key.get_secret_value()

    # Guard: if admin_api_key is not configured, log a server-side warning
    # then return 401 (same as absent header).  Returning 503 would reveal
    # server misconfiguration to any unauthenticated caller; 401 reveals
    # nothing beyond "authentication is required."
    if not admin_key_secret:
        logger.warning(
            "deps.require_admin_not_configured: "
            "ADMIN_API_KEY is empty — all admin operations return 401"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Admin-Key header is required for this endpoint",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )

    # 401 when header is absent — distinguishable from 403 (wrong key)
    if header_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Admin-Key header is required for this endpoint",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )

    # Timing-safe comparison — always runs both paths to prevent oracle attacks
    submitted_bytes = header_key.encode("utf-8")
    expected_bytes = admin_key_secret.encode("utf-8")
    if not hmac.compare_digest(submitted_bytes, expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )


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
