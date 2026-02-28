"""
packages/common/logging.py
--------------------------
Centralized structured logging configuration for the trading bot.

Entry point: call ``configure_logging()`` once in the FastAPI lifespan
handler (or any process entry point) before the first log statement.

Usage
-----
From application startup::

    from common.logging import configure_logging
    configure_logging(log_level="INFO", json_output=True, service_name="api")

From any module::

    from common.logging import get_logger
    log = get_logger(__name__)
    log.info("my.event", key="value")

Request-ID correlation
----------------------
Set a request ID at the beginning of an HTTP request and it will
automatically appear in every log entry emitted during that request::

    from common.logging import set_request_id
    set_request_id("req-abc123")

The request ID is stored in a ``contextvars.ContextVar`` and merged into
structlog context via ``structlog.contextvars.bind_contextvars``.  Because
structlog's ``merge_contextvars`` processor runs first in the processor
chain (configured in ``common.config.configure_structlog``), the field
appears automatically in all log entries emitted within the same asyncio
Task context — including log entries from third-party libraries that use
the stdlib ``logging`` bridge.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

from common.config import configure_structlog

__all__ = [
    "configure_logging",
    "get_logger",
    "set_request_id",
    "clear_request_id",
    "REQUEST_ID_CTX_VAR",
]

# ---------------------------------------------------------------------------
# Request-ID context variable
# ---------------------------------------------------------------------------

REQUEST_ID_CTX_VAR: ContextVar[str | None] = ContextVar(
    "request_id", default=None
)


def set_request_id(request_id: str | None = None) -> str:
    """
    Bind a request ID into the current async context.

    Stores the ID in a ``contextvars.ContextVar`` *and* binds it into
    structlog's contextvars store so it appears in every log entry
    emitted within this asyncio Task.

    Parameters
    ----------
    request_id:
        Explicit request ID string. If ``None``, a UUID4 hex is generated.

    Returns
    -------
    str
        The request ID that was set (generated or provided).
    """
    rid = request_id or uuid.uuid4().hex
    REQUEST_ID_CTX_VAR.set(rid)
    # Bind into structlog's contextvars so merge_contextvars picks it up
    # automatically on every log call within this async context.
    structlog.contextvars.bind_contextvars(request_id=rid)
    return rid


def clear_request_id() -> None:
    """Remove the request ID binding from the current async context."""
    REQUEST_ID_CTX_VAR.set(None)
    structlog.contextvars.unbind_contextvars("request_id")


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Module-level guard: configure_logging is idempotent after the first call.
_LOGGING_CONFIGURED: bool = False


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    service_name: str = "trading-bot",
    environment: str = "production",
) -> None:
    """
    Configure structured logging for the entire process.

    Delegates to ``common.config.configure_structlog`` for the structlog
    processor chain, then configures the Python stdlib ``logging`` root
    logger to forward through structlog so third-party libraries (uvicorn,
    SQLAlchemy, httpx) emit structured JSON entries alongside application
    logs.

    Service-level fields (``service`` and ``env``) are bound into
    structlog's contextvars so they appear in every log entry in this
    process without requiring explicit passing at each call site.

    Parameters
    ----------
    log_level:
        Minimum log level string, e.g. ``"INFO"`` or ``"DEBUG"``.
        Case-insensitive. Defaults to ``"INFO"``.
    json_output:
        When ``True``, renders logs as JSON (production / log-aggregator use).
        When ``False``, renders coloured console output (local development).
        Defaults to ``True``.
    service_name:
        Label identifying which service produced a log entry.  Added to
        every log record as the ``service`` field.  Defaults to
        ``"trading-bot"``.
    environment:
        Deployment environment tag, e.g. ``"production"``, ``"staging"``,
        ``"development"``.  Added to every log record as ``env``.
        Defaults to ``"production"``.

    Notes
    -----
    - Idempotent: subsequent calls after the first are silently ignored.
    - In tests, reset the guard with ``common.logging._LOGGING_CONFIGURED = False``
      before calling again, or construct a fresh process.
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603
    if _LOGGING_CONFIGURED:
        return

    # 1. Configure the structlog processor chain (JSON or coloured console).
    configure_structlog(log_level=log_level, json_logs=json_output)

    # 2. Bind process-wide context fields. These appear in every log entry
    #    because merge_contextvars is the first processor in the chain.
    structlog.contextvars.bind_contextvars(
        service=service_name,
        env=environment,
    )

    # 3. Configure stdlib root logger to forward through structlog.
    #    This captures uvicorn access logs, SQLAlchemy warnings, httpx
    #    debug output, and any other library that uses stdlib logging.
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    _configure_stdlib_bridge(numeric_level)

    # 4. Suppress noisy third-party loggers at WARNING unless in DEBUG mode.
    #    These libraries emit per-request noise at INFO that drowns signal.
    if numeric_level > logging.DEBUG:
        for noisy_logger in ("asyncio", "urllib3", "httpcore", "hpack"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def _configure_stdlib_bridge(numeric_level: int) -> None:
    """
    Install a stdlib logging handler that forwards records through structlog.

    Uses ``structlog.stdlib.ProcessorFormatter`` so that records emitted
    by third-party libraries pass through the same JSON / console renderer
    as native structlog calls, producing a single unified log stream.

    Parameters
    ----------
    numeric_level:
        The ``logging.*`` integer level constant (e.g. ``logging.INFO``).
    """
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ExtraAdder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any handlers that basicConfig may have previously installed
    # to avoid duplicate output in test environments.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_logger(name: str, **initial_values: Any) -> Any:
    """
    Return a structlog bound logger for ``name``.

    This is the primary way to obtain a logger in application code.  The
    returned object supports ``log.info(event, **kv)``, ``log.debug(...)``,
    ``log.warning(...)``, ``log.error(...)``, and ``log.exception(...)``.

    Parameters
    ----------
    name:
        Logger name — use ``__name__`` as the convention.
    **initial_values:
        Key-value pairs permanently bound to this logger instance.
        These appear in every log entry from this logger without being
        passed at each call site.

    Returns
    -------
    structlog BoundLogger
        Thread-safe bound logger instance.

    Examples
    --------
    ::

        log = get_logger(__name__, component="order_router")
        log.info("order.submitted", order_id="abc123", symbol="BTC/USDT")
        # -> {"event": "order.submitted", "logger": "...", "component":
        #     "order_router", "order_id": "abc123", "symbol": "BTC/USDT",
        #     "timestamp": "...", "service": "trading-bot", ...}
    """
    log = structlog.get_logger(name)
    if initial_values:
        log = log.bind(**initial_values)
    return log
