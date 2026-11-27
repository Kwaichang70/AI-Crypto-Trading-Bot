"""
apps/api/rate_limit.py
-----------------------
Per-IP rate limiting for the AI Crypto Trading Bot API.

Implements tiered rate limits using ``slowapi`` (built on ``limits``):

- **Auth failures:** 5/minute per IP — prevents brute-force API key guessing
- **Write endpoints:** 30/minute per IP (POST, PUT, PATCH, DELETE)
- **Read endpoints:** 120/minute per IP (GET)
- **Health/metrics:** exempt from rate limiting (monitoring probes)

Security design
---------------
- Rate limits are applied per client IP address.
- X-Forwarded-For is only consulted when ``trusted_proxy_count > 0`` is
  explicitly configured. The default (0) ignores XFF entirely and uses the
  direct TCP peer address, which is safe for deployments without a reverse
  proxy. Accepting XFF unconditionally would allow any client to forge an
  arbitrary source IP and bypass or exhaust other clients' quotas.

  WARNING: If you deploy behind a load balancer or reverse proxy, set
  ``TRUSTED_PROXY_COUNT`` to the number of trusted proxy hops (usually 1).
  Using the wrong value will either expose the spoofing vulnerability (too
  high) or misidentify all clients as the proxy IP (too low).

- Exceeding a limit returns HTTP 429 Too Many Requests with a JSON body
  and a ``Retry-After`` header indicating when the client may retry.
- All rate-limit violations are logged with structured context (client IP,
  path, method, limit hit) for security monitoring and incident response.
- Limits are configurable via environment variables; see ``api.config.Settings``.

Storage
-------
MVP uses an in-memory storage backend (no external dependencies). This means
rate limit state is per-process and resets on restart. For multi-replica
deployments, switch to Redis storage by setting ``REDIS_URL`` and updating
the ``_create_storage`` function.

Sprint 2 plan
--------------
- Redis-backed storage for multi-replica consistency
- Per-API-key rate limits (in addition to per-IP)
- Sliding window algorithm (currently fixed window via ``limits``)
- Expose rate strategy through a public wrapper to avoid ``_limiter``
  private attribute access (SEC-S2-001b)
- Exact Retry-After calculation from storage expiry (SEC-S2-001b)

References
----------
- slowapi: https://github.com/laurentS/slowapi
- limits: https://limits.readthedocs.io/
- SEC-S2-001: Sprint 2 P0 security audit finding

Dependency version notes (CR-RL-010)
-------------------------------------
``pyproject.toml`` should pin ``"slowapi>=0.1.9,<0.2"`` and add an explicit
``"limits>=3.0,<4"`` dependency. The private ``_limiter`` strategy attribute
access is sensitive to internal API changes across minor releases.
"""

from __future__ import annotations

import re
from typing import Any, cast

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from limits import parse as parse_limit  # CR-RL-009: module-level import
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.wrappers import Limit
from starlette.routing import Match
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from api.config import Settings, get_settings

__all__ = [
    "limiter",
    "get_client_ip",
    "setup_rate_limiting",
    "check_auth_failure_rate",
    "AUTH_FAILURE_LIMIT",
    "WRITE_LIMIT",
    "READ_LIMIT",
]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# IP extraction — proxy-aware with spoofing protection (CR-RL-002)
# ---------------------------------------------------------------------------

def get_client_ip(request: Request, trusted_proxy_count: int = 0) -> str:
    """
    Extract the real client IP address from the incoming request.

    By default (``trusted_proxy_count=0``) this function ignores the
    ``X-Forwarded-For`` header entirely and returns the direct TCP peer
    address. This is the safe default for direct-connection deployments.

    When ``trusted_proxy_count > 0``, the function reads the XFF list and
    returns the entry that the outermost trusted proxy would have appended —
    specifically the entry at index ``max(0, len(parts) - trusted_proxy_count)``
    (Nth-from-right). This is the correct approach for reverse-proxy
    deployments: the rightmost proxy appends the client IP it received,
    so "1 trusted proxy" means reading the RIGHTMOST entry (index len - 1);
    entries to its left are client-supplied and ignored.

    SECURITY NOTE: Never set ``trusted_proxy_count`` higher than the actual
    number of proxy hops you control. Clients can inject arbitrary entries at
    the *left* of the XFF list; only the proxy-appended entries on the *right*
    are trustworthy.

    Parameters
    ----------
    request:
        The incoming HTTP request.
    trusted_proxy_count:
        Number of trusted proxy hops between the internet and this service.
        Default 0 disables XFF parsing entirely (safe for direct connections).

    Returns
    -------
    str
        The client IP address string, or ``"unknown"`` if unavailable.
    """
    if trusted_proxy_count > 0:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            parts = [p.strip() for p in forwarded_for.split(",")]
            # Use the entry added by the outermost trusted proxy.
            # With N trusted proxies, use index len(parts) - N (clamped to 0).
            index = max(0, len(parts) - trusted_proxy_count)
            return parts[index]
    # Direct connection mode: use the TCP peer address unconditionally.
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Rate limit string helpers — read from settings at call time
# ---------------------------------------------------------------------------

def _get_auth_failure_limit() -> str:
    """Return the auth failure rate limit string from settings."""
    settings = get_settings()
    return settings.rate_limit_auth_failures


def _get_write_limit() -> str:
    """Return the write endpoint rate limit string from settings."""
    settings = get_settings()
    return settings.rate_limit_write


def _get_read_limit() -> str:
    """Return the read endpoint rate limit string from settings."""
    settings = get_settings()
    return settings.rate_limit_read


# ---------------------------------------------------------------------------
# Default limit string constants (used for decorator hints)
# ---------------------------------------------------------------------------

AUTH_FAILURE_LIMIT = "5/minute"
WRITE_LIMIT = "30/minute"
READ_LIMIT = "120/minute"


# ---------------------------------------------------------------------------
# Limiter instance — module-level singleton
# ---------------------------------------------------------------------------

limiter = Limiter(
    key_func=get_client_ip,
    default_limits=[],  # No blanket default; limits applied per-endpoint
    storage_uri="memory://",
    strategy="fixed-window",
)


# ---------------------------------------------------------------------------
# Paths exempt from rate limiting
# ---------------------------------------------------------------------------

_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/metrics",
    "/api/v1/metrics",
    # The following paths exist unless ENVIRONMENT=production (docs_url/redoc_url/
    # openapi_url are None there; CF-97-37). Exempt as a convenience for dev/test.
    "/docs",
    "/redoc",
    "/openapi.json",
})


# CF-97-53 (SE-9741-04): the emergency-stop exemption matches the one real route,
# POST /api/v1/runs/{uuid}/emergency-stop, with the uuid in the canonical dashed or
# the 32-hex form (the forms the API and the UI emit). It does not depend on the
# route lookup below, so a lookup fault can never throttle the emergency stop.
_HEX = "[0-9a-fA-F]"
_EMERGENCY_STOP_RE = re.compile(
    rf"/api/v1/runs/(?:{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}|{_HEX}{{32}})"
    r"/emergency-stop"
)

# Bucket for every path that matches no route (CF-97-53, SE-9741-03): varying the
# path no longer yields a fresh bucket. Real templates start with "/", so no clash.
_UNMATCHED = "_unmatched_"


def _route_template(request: Request) -> str:
    """Return the path template of the route that will serve ``request``.

    The middleware runs before routing, so it asks the router: the first route
    whose ``matches`` is FULL is the one the router dispatches to (a PARTIAL
    match is a method mismatch and is skipped, as the router does). Cost is one
    regex test per route, measured at 6-21 us on the 43-route app. The route list
    is read per request because routes are added after ``setup_rate_limiting``.
    """
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match is Match.FULL:
            path = getattr(route, "path", None)
            return path if isinstance(path, str) else _UNMATCHED
    return _UNMATCHED


# ---------------------------------------------------------------------------
# Limit wrapper — bridges limits.RateLimitItem to slowapi.wrappers.Limit
# ---------------------------------------------------------------------------

def _make_slowapi_limit(parsed_limit: Any) -> Limit:
    """Wrap a ``limits.RateLimitItem`` in a ``slowapi.wrappers.Limit``.

    ``RateLimitExceeded.__init__`` requires a ``Limit`` object (not a raw
    ``RateLimitItem``).  Passing the wrong type causes an ``AttributeError``
    on ``limit.error_message`` inside the constructor.
    """
    return Limit(
        limit=parsed_limit,
        key_func=get_client_ip,
        scope="",
        per_method=False,
        methods=None,
        error_message=None,
        exempt_when=None,
        cost=1,
        override_defaults=False,
    )


# ---------------------------------------------------------------------------
# 429 error handler — structured JSON response with Retry-After
# ---------------------------------------------------------------------------

def _rate_limit_exceeded_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """
    Custom handler for 429 Too Many Requests responses.

    Returns a structured JSON error body with:
    - ``error``: Human-readable error type
    - ``detail``: Description including the limit that was exceeded
    - ``retry_after``: Seconds until the client may retry

    Also sets the ``Retry-After`` HTTP header (RFC 7231 Section 7.1.3).

    All rate limit violations are logged at WARNING level with full
    structured context for security monitoring dashboards.

    Parameters
    ----------
    request:
        The rate-limited HTTP request.
    exc:
        The rate limit exception containing limit details.

    Returns
    -------
    JSONResponse
        A 429 response with structured error body and Retry-After header.
    """
    rate_exc = cast(RateLimitExceeded, exc)

    # Parse retry window from the exception's limit string
    retry_after = _parse_retry_after(str(rate_exc.detail))

    settings = get_settings()
    client_ip = get_client_ip(request, trusted_proxy_count=settings.trusted_proxy_count)

    logger.warning(
        "rate_limit.exceeded",
        client=client_ip,
        path=request.url.path,
        method=request.method,
        limit=str(rate_exc.detail),
        retry_after=retry_after,
    )

    return JSONResponse(
        status_code=HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {rate_exc.detail}",
            "retry_after": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
        },
    )


def _parse_retry_after(limit_detail: str) -> int:
    """
    Parse the retry-after duration in seconds from a rate limit detail string.

    The ``limits`` library uses strings like ``"5 per 1 minute"``. We extract
    the window duration to compute a sensible Retry-After value.

    NOTE (CR-RL-005): This function returns the **full window duration** as a
    conservative upper bound. The actual time until the window resets may be
    shorter depending on when in the window the limit was hit. Exact
    calculation (using storage expiry timestamps) is deferred to Sprint 2
    when Redis storage and sliding window algorithms are added (SEC-S2-001b).
    RFC 7231 Section 7.1.3 permits conservative over-estimates.

    Parameters
    ----------
    limit_detail:
        The string representation of the rate limit (e.g. "5 per 1 minute").

    Returns
    -------
    int
        Seconds the client should wait before retrying.
        Defaults to 60 if parsing fails.
    """
    detail_lower = limit_detail.lower()

    if "second" in detail_lower:
        return 1
    elif "minute" in detail_lower:
        return 60
    elif "hour" in detail_lower:
        return 3600
    elif "day" in detail_lower:
        return 86400
    else:
        return 60  # Safe default


# ---------------------------------------------------------------------------
# HTTP method-based rate limiting middleware
# ---------------------------------------------------------------------------

async def _rate_limit_middleware(request: Request, call_next: Any) -> Response:
    """
    Middleware that applies tiered rate limits based on HTTP method.

    Limit tiers:
    - ``GET``: read limit (default 120/minute)
    - ``POST``, ``PUT``, ``PATCH``, ``DELETE``: write limit (default 30/minute)
    - Exempt paths (health, metrics, docs): no limit applied

    This middleware runs after authentication but before the route handler.
    Auth-failure-specific rate limiting is handled separately in
    ``auth.require_api_key`` via the ``@limiter.limit`` decorator.

    Parameters
    ----------
    request:
        The incoming HTTP request.
    call_next:
        The next middleware or route handler in the chain.

    Returns
    -------
    Response
        The response from the downstream handler, or a 429 response
        if the rate limit is exceeded.
    """
    settings = get_settings()

    # Skip rate limiting if globally disabled
    if not settings.rate_limit_enabled:
        return cast(Response, await call_next(request))

    # Skip exempt paths
    path = request.url.path
    if path in _EXEMPT_PATHS:
        return cast(Response, await call_next(request))

    # SEC-006 (Sprint 45): exempt the emergency-stop endpoint from rate
    # limits.  Operators MUST be able to kill a runaway live run during
    # an incident even if the API key bucket is throttled by the same
    # incident's symptoms (e.g. a misconfigured client hammering the
    # /runs endpoint).  Availability of the kill switch is a security
    # property — the audit_events row records every emergency-stop so
    # abuse is detectable post-incident.
    if request.method.upper() == "POST" and _EMERGENCY_STOP_RE.fullmatch(path):
        return cast(Response, await call_next(request))

    # Sprint 50 Cycle 3: exempt the global kill-switch from rate limits.
    # Same rationale as /emergency-stop above — availability of the kill
    # switch is a security property; the audit_events row records every
    # invocation so abuse is detectable post-incident.
    # Exact match (not endswith) to prevent unintended path inheritance.
    # WP1.7a round 2 (S-11): POST-only -- GET /kill-switch is a plain
    # authenticated status read (X-API-Key, not admin) with no incident-
    # response urgency justifying a rate-limit bypass.
    if path == "/api/v1/emergency/kill-switch" and request.method.upper() == "POST":
        return cast(Response, await call_next(request))

    # Determine the appropriate limit tier
    method = request.method.upper()
    if method == "GET":
        limit_string = settings.rate_limit_read
    elif method in {"POST", "PUT", "PATCH", "DELETE"}:
        limit_string = settings.rate_limit_write
    else:
        # OPTIONS, HEAD, etc. — no rate limit
        return cast(Response, await call_next(request))

    # Apply the rate limit check using the limiter
    client_ip = get_client_ip(request, trusted_proxy_count=settings.trusted_proxy_count)

    try:
        # Use the limiter's internal strategy layer to check and record the hit.
        # The key is composed of: method + route template + client_ip. Paths that
        # match no route share one bucket (CF-97-53), so no path can mint a bucket.
        path_key = _route_template(request)
        rate_key = f"{method}:{path_key}:{client_ip}"

        # CR-RL-001: Use limiter._limiter (strategy layer), NOT limiter._storage
        # (storage backend). The hit() method exists only on strategy objects
        # (FixedWindowRateLimiter, etc.), not on storage backends (MemoryStorage).
        # Sprint 2: expose this through a public wrapper to avoid private access.
        parsed_limit = parse_limit(limit_string)
        strategy = limiter._limiter  # noqa: SLF001 — rate limiting strategy layer

        hit = strategy.hit(parsed_limit, rate_key)

        if not hit:
            # CR-RL-003: Wrap in a Limit object as RateLimitExceeded requires.
            raise RateLimitExceeded(_make_slowapi_limit(parsed_limit))

    except RateLimitExceeded as exc:
        return _rate_limit_exceeded_handler(request, exc)
    except Exception:
        # If rate limiting fails for any reason (storage error, parse error),
        # fail open — do not block legitimate requests due to limiter bugs.
        # Log the error for investigation.
        logger.error(
            "rate_limit.check_failed",
            client=client_ip,
            path=path,
            method=method,
            exc_info=True,
        )

    return cast(Response, await call_next(request))


# ---------------------------------------------------------------------------
# Auth failure rate limiting — called from auth.py
# ---------------------------------------------------------------------------

def check_auth_failure_rate(request: Request) -> None:
    """
    Check and enforce the auth-failure-specific rate limit.

    This function should be called from ``auth.require_api_key`` when
    an authentication attempt fails (missing key, invalid key). It
    uses a tighter rate limit (default 5/minute) to prevent brute-force
    API key guessing attacks.

    The rate limit key is: ``auth_failure:<client_ip>``

    Parameters
    ----------
    request:
        The HTTP request that failed authentication.

    Raises
    ------
    RateLimitExceeded:
        When the client has exceeded the auth failure rate limit.
        The caller should convert this to an HTTP 429 response.
    """
    settings = get_settings()

    if not settings.rate_limit_enabled:
        return

    client_ip = get_client_ip(request, trusted_proxy_count=settings.trusted_proxy_count)
    rate_key = f"auth_failure:{client_ip}"

    try:
        # CR-RL-001: Use limiter._limiter (strategy layer), NOT limiter._storage.
        # CR-RL-003: Raise RateLimitExceeded with the RateLimitItem, not a string.
        parsed_limit = parse_limit(settings.rate_limit_auth_failures)
        strategy = limiter._limiter  # noqa: SLF001 — rate limiting strategy layer

        hit = strategy.hit(parsed_limit, rate_key)

        if not hit:
            logger.warning(
                "rate_limit.auth_brute_force_blocked",
                client=client_ip,
                path=request.url.path,
                limit=settings.rate_limit_auth_failures,
            )
            raise RateLimitExceeded(_make_slowapi_limit(parsed_limit))

    except RateLimitExceeded:
        raise
    except Exception:
        # Fail open — log but don't block
        logger.error(
            "rate_limit.auth_check_failed",
            client=client_ip,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

def setup_rate_limiting(app: FastAPI) -> None:
    """
    Configure rate limiting on the FastAPI application.

    This function should be called from ``create_app()`` in ``main.py``.
    It performs the following setup:

    1. Validates all rate limit configuration strings at startup (CR-RL-006)
    2. Registers the custom 429 error handler
    3. Attaches the limiter to the app state (required by slowapi)
    4. Registers the rate-limit middleware for method-based tiering

    When ``RATE_LIMIT_ENABLED=false``, the middleware is still registered
    but short-circuits immediately (no performance overhead).

    Raises
    ------
    ValueError:
        If any rate limit string in settings cannot be parsed by ``limits``.
        This fails fast at startup and prevents silent fail-open due to
        misconfiguration (CR-RL-006).

    Parameters
    ----------
    app:
        The FastAPI application instance.
    """
    settings = get_settings()

    # CR-RL-006: Validate all rate limit strings at startup.
    # A misconfigured string (e.g. "5 per minuto") would silently pass
    # pydantic validation and only fail at the first incoming request,
    # causing the limiter to fail open. Validate eagerly here to surface
    # config errors during application startup.
    _validate_rate_limit_strings(settings)

    # Register the custom 429 error handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Attach the limiter to the app state (slowapi convention)
    app.state.limiter = limiter

    # Register the method-based rate limit middleware.
    # NOTE (CR-RL-004): Starlette middleware registered via app.middleware("http")
    # executes in reverse registration order — the last registered middleware
    # runs outermost (first). Registering rate limiting here means it runs
    # BEFORE the timing middleware, so blocked 429 responses are rejected
    # cheaply without incurring timing middleware overhead.
    app.middleware("http")(_rate_limit_middleware)

    logger.info(
        "rate_limit.configured",
        enabled=settings.rate_limit_enabled,
        auth_failures=settings.rate_limit_auth_failures,
        write=settings.rate_limit_write,
        read=settings.rate_limit_read,
        trusted_proxy_count=settings.trusted_proxy_count,
    )


def _validate_rate_limit_strings(settings: Settings) -> None:
    """
    Validate all rate limit configuration strings against the ``limits`` parser.

    Called once at application startup from ``setup_rate_limiting()``.

    Parameters
    ----------
    settings:
        The application settings instance.

    Raises
    ------
    ValueError:
        If any rate limit string cannot be parsed.
    """
    fields = {
        "rate_limit_auth_failures": settings.rate_limit_auth_failures,
        "rate_limit_write": settings.rate_limit_write,
        "rate_limit_read": settings.rate_limit_read,
    }
    for field_name, value in fields.items():
        try:
            parse_limit(value)
        except Exception as exc:
            raise ValueError(
                f"Invalid rate limit configuration for '{field_name}': '{value}'. "
                "Expected format: '<count>/<period>' e.g. '5/minute', '30/hour'. "
                f"Parse error: {exc}"
            ) from exc
