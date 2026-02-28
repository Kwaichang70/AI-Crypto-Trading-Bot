"""
apps/api/auth.py
-----------------
API key authentication module for the AI Crypto Trading Bot API.

Supports two authentication methods:
1. ``X-API-Key`` header (preferred for programmatic clients)
2. ``?api_key=`` query parameter (for WebSocket/SSE compatibility)

Security design:
- Keys are stored as SHA-256 hashes in the configuration. The raw key
  never appears in config dumps, logs, or error responses.
- Hash comparison uses ``hmac.compare_digest`` for constant-time
  comparison, preventing timing side-channel attacks.
- Authentication failures are logged with structured context (source IP,
  method used) but never include the submitted key value.
- When ``require_api_auth`` is False (dev mode), all endpoints are open.

Generating a key hash for configuration::

    # Linux / macOS / Git Bash:
    echo -n "my-secret-api-key" | sha256sum | awk '{print $1}'

    # Python one-liner:
    python -c "import hashlib; print(hashlib.sha256(b'my-secret-api-key').hexdigest())"

    # Then set in .env:
    API_KEY_HASH=<the hex digest output>
    REQUIRE_API_AUTH=true
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

import structlog
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader

from api.config import Settings, get_settings
from api.rate_limit import check_auth_failure_rate

__all__ = ["require_api_key"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI security scheme (appears in OpenAPI docs when debug=True)
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # We handle missing keys ourselves for better error messages
    description=(
        "API key for authentication. "
        "Pass via this header or as a ?api_key= query parameter."
    ),
)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def _hash_key(raw_key: str) -> str:
    """
    Compute the SHA-256 hex digest of a raw API key.

    Parameters
    ----------
    raw_key:
        The plaintext API key submitted by the client.

    Returns
    -------
    str
        Lowercase hex-encoded SHA-256 hash of the key.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(submitted_key: str, expected_hash: str) -> bool:
    """
    Hash the submitted key and compare against the stored hash.

    Uses ``hmac.compare_digest`` for constant-time comparison to prevent
    timing side-channel attacks that could leak hash prefix information.

    Parameters
    ----------
    submitted_key:
        The raw API key submitted by the client (never stored or logged).
    expected_hash:
        The SHA-256 hex digest of the valid API key, from configuration.

    Returns
    -------
    bool
        True if the submitted key matches the expected hash; False otherwise.
    """
    submitted_hash = _hash_key(submitted_key)
    return hmac.compare_digest(submitted_hash, expected_hash)


# ---------------------------------------------------------------------------
# FastAPI dependency — wire this into protected routes
# ---------------------------------------------------------------------------

async def require_api_key(
    request: Request,
    header_key: Optional[str] = Depends(_api_key_header),
    query_key: Optional[str] = Query(
        default=None,
        alias="api_key",
        include_in_schema=False,  # Hide from OpenAPI to discourage URL-based auth
        description="API key via query param (WebSocket/SSE fallback only)",
    ),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """
    FastAPI dependency that enforces API key authentication.

    Extraction priority:
    1. ``X-API-Key`` header (preferred)
    2. ``?api_key=`` query parameter (fallback for WebSocket/SSE)

    When ``settings.require_api_auth`` is False, authentication is skipped
    entirely and the dependency returns None. This allows frictionless
    local development while enforcing security in production.

    Parameters
    ----------
    request:
        The incoming HTTP request (used for logging context).
    header_key:
        API key extracted from the ``X-API-Key`` header, or None.
    query_key:
        API key extracted from the ``?api_key=`` query parameter, or None.
    settings:
        Application settings (injected via FastAPI dependency).

    Returns
    -------
    Optional[str]
        A constant string ``"authenticated"`` on success, or None when
        auth is disabled. Never returns the raw key.

    Raises
    ------
    HTTPException 401:
        When authentication is required but the key is missing or invalid.
    HTTPException 500:
        When authentication is required but no API key hash is configured
        (server misconfiguration).
    RateLimitExceeded:
        When the client IP has exceeded the auth failure rate limit.
        Caught by the 429 handler registered in ``setup_rate_limiting``.
    """
    # Dev mode: auth disabled
    if not settings.require_api_auth:
        return None

    # Production mode: auth required — verify config is complete
    if not settings.api_key_hash:
        logger.error(
            "auth.misconfigured",
            detail="require_api_auth=True but api_key_hash is empty",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication is misconfigured. Contact the administrator.",
        )

    # Extract the key from header or query param (header takes priority)
    submitted_key = header_key or query_key

    # Determine authentication method for logging (never log the key itself)
    auth_method = "header" if header_key else ("query" if query_key else "none")
    client_host = request.client.host if request.client else "unknown"

    if not submitted_key:
        logger.warning(
            "auth.missing_key",
            client=client_host,
            path=request.url.path,
            method=request.method,
        )
        # Check auth failure rate limit after logging but before returning 401.
        # This prevents brute-force enumeration of valid endpoints.
        check_auth_failure_rate(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify the submitted key against the stored hash
    if not verify_api_key(submitted_key, settings.api_key_hash):
        logger.warning(
            "auth.invalid_key",
            client=client_host,
            path=request.url.path,
            method=request.method,
            auth_method=auth_method,
        )
        # Check auth failure rate limit after logging but before returning 401.
        # This is the critical path for brute-force API key guessing prevention.
        check_auth_failure_rate(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Success — log at debug level to avoid flooding production logs
    logger.debug(
        "auth.success",
        client=client_host,
        path=request.url.path,
        auth_method=auth_method,
    )

    # Return a sentinel value (never the raw key)
    return "authenticated"
