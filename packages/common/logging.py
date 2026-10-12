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

import contextlib
import logging
import os
import sys
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

from common import redaction as _redaction
from common.config import build_renderer, configure_structlog

__all__ = [
    "configure_logging",
    "exc_log_fields",
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
    *,
    register_env_secrets: bool = True,
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
    register_env_secrets:
        When ``True`` (default) credential-named process environment variables
        are registered for exact-match redaction (CF-36 fallback source for
        processes that never build ``Settings``).  ``Settings``-held secrets
        are registered separately by ``api.config.register_settings_secrets``.

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
    _configure_stdlib_bridge(numeric_level, json_output)

    # 4. Suppress noisy third-party loggers at WARNING unless in DEBUG mode.
    #    These libraries emit per-request noise at INFO that drowns signal.
    if numeric_level > logging.DEBUG:
        for noisy_logger in ("asyncio", "urllib3", "httpcore", "hpack"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # 5. CF36-SEC-07: ccxt's library logger emits request headers
    #    (X-MBX-APIKEY, Authorization, CB-ACCESS-*) at DEBUG by construction and
    #    has no production use.  Forced to WARNING regardless of LOG_LEVEL.
    logging.getLogger("ccxt").setLevel(logging.WARNING)

    # 6. CF36-SEC-09: uvicorn installs its own handlers (propagate=False) that
    #    bypass the bridge.  Absorb them so its records (including "Exception in
    #    ASGI application" tracebacks) are redacted like every other record.
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(uvicorn_logger)
        if uvicorn_logger == "uvicorn.access":
            if not lg.handlers and not lg.propagate:
                # uvicorn --no-access-log (Config.configure_logging) leaves handlers=[] and
                # propagate=False so the protocols' per-connection hasHandlers() is False.
                # Re-enabling propagation would switch access logging back on (CF36A-SEC-01).
                continue
            if not any(isinstance(f, _AccessQueryFilter) for f in lg.filters):
                lg.addFilter(_AccessQueryFilter())
        lg.handlers.clear()
        lg.propagate = True

    # 7. Exact-match registry, env fallback source.
    if register_env_secrets:
        _redaction.register_secrets_from_env()

    _LOGGING_CONFIGURED = True


class _AccessQueryFilter(logging.Filter):
    """CF36A-SEC-01: drop the query string from uvicorn access records.

    uvicorn logs the RAW query string (``?api%5Fkey=...``) while Starlette decodes
    parameter names, so name-based rules cannot be relied on.  Never raises.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
                path = args[2]
                if "?" in path:
                    record.args = (
                        *args[:2],
                        path.split("?", 1)[0] + "?[QUERY_REDACTED]",
                        *args[3:],
                    )
        except Exception:
            record.msg, record.args = "[REDACTION_FAILED]", None
        return True


_BRIDGE_FAILED_LINE = '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'


def _drop_color_message(
    _: object, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Drop uvicorn's ``color_message`` extra (ANSI duplicate of the message)."""
    event_dict.pop("color_message", None)
    return event_dict


class _SafeProcessorFormatter(structlog.stdlib.ProcessorFormatter):
    """ProcessorFormatter that can never raise or echo a record's message.

    ``ProcessorFormatter.format`` calls ``record.getMessage()`` BEFORE any
    processor, so a malformed third-party call (``"%d" % "x"``) raises inside
    ``format``; ``logging.Handler.handleError`` would then dump ``record.msg``
    and ``record.args`` raw to stderr (ARB-CF36-10).
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            return super().format(record)
        except Exception:
            _redaction._count_failure()
            return _BRIDGE_FAILED_LINE


class _SafeStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler whose ``handleError`` writes a constant line only."""

    def handleError(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            stream = sys.__stderr__
            if stream is not None:
                stream.write(_BRIDGE_FAILED_LINE + "\n")


def _configure_stdlib_bridge(numeric_level: int, json_output: bool = True) -> None:
    """
    Install a stdlib logging handler that forwards records through structlog.

    Uses ``structlog.stdlib.ProcessorFormatter`` so that records emitted
    by third-party libraries pass through the same JSON / console renderer
    and the same redaction tail as native structlog calls, producing a single
    unified, redacted log stream (CF36-SEC-06 repair: the old ``processors``
    ended in ``wrap_for_formatter``, which returns a tuple, not a string).

    Parameters
    ----------
    numeric_level:
        The ``logging.*`` integer level constant (e.g. ``logging.INFO``).
    json_output:
        ``True`` for JSON lines, ``False`` for the coloured console renderer.
    """
    formatter = _SafeProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ExtraAdder(),
            _drop_color_message,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            _redaction.redact_event_dict,
            _redaction.RedactingRenderer(build_renderer(json_output)),
        ],
    )

    handler = _SafeStreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any handlers that basicConfig may have previously installed
    # to avoid duplicate output in test environments.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)


# ---------------------------------------------------------------------------
# Exception fields for log calls (CF-36 layer 2 helper)
# ---------------------------------------------------------------------------


def exc_log_fields(exc: BaseException) -> dict[str, object]:
    """
    Safe structured fields describing ``exc``: type, HTTP status, code location.

    Use as ``log.warning("event", **exc_log_fields(exc))`` at every site that
    handles an outbound-I/O exception.  Never ``str(exc)`` / ``repr(exc)``:
    ccxt and aiohttp exception text embeds the request URL, query string
    (``signature=``, ``api_key=``) and response body.

    Returns
    -------
    dict
        ``error_type`` (class name), ``http_status`` (only when ``exc.status``
        is an int in 100..599, e.g. aiohttp ``ClientResponseError``) and
        ``error_at`` (``"<file>:<line>:<function>"`` of the innermost
        traceback frame; code identifiers only, never values).  Never raises.
    """
    fields: dict[str, object] = {"error_type": type(exc).__name__}
    with contextlib.suppress(Exception):
        status = getattr(exc, "status", None)
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
            fields["http_status"] = status
        tb = exc.__traceback__
        if tb is not None:
            while tb.tb_next is not None:
                tb = tb.tb_next
            code = tb.tb_frame.f_code
            fields["error_at"] = (
                f"{os.path.basename(code.co_filename)}:{tb.tb_lineno}:{code.co_name}"
            )
    return fields


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
