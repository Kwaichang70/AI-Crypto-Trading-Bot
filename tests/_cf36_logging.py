"""
tests/_cf36_logging.py
----------------------
Harness for the CF-36 layer-1 tests: runs the PRODUCTION logging configuration
(``common.logging.configure_logging``) and captures what really reaches stdout
(structlog path) and stderr (stdlib bridge).

Rules (synthesis-spec §8, binding)
----------------------------------
* ``structlog.testing.capture_logs`` REPLACES the processors, so it can never
  prove layer 1.  Layer-1 tests use :func:`production_logging`.
* Loggers are not cached (``cache_logger_on_first_use=False``): a cached logger
  binds ``sys.stdout`` at first use and would silently print to an old stream
  (RK-12).
* Every ABSENCE assertion must be paired with a PRESENCE assertion for the
  same event (event name plus a ``[REDACTED...]`` marker) -- use
  :meth:`Capture.assert_redacted`.
* All sentinels are fake.  ``SENTINEL`` is a distinctive prefix so that even a
  partial leak is detected.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import structlog

import common.logging as clog
from common import redaction

#: Distinctive prefix of every fake secret in the CF-36 tests.
SENTINEL = "CF36SENTINEL"


def b64url(text: str) -> str:
    """Unpadded URL-safe base64 of ``text`` (JWT segment encoding)."""
    return base64.urlsafe_b64encode(text.encode()).rstrip(b"=").decode()


def fake_jwt(header: str, payload: str, signature: str) -> str:
    """Assemble a JWT-shaped fake at runtime.

    The tests need JWT-shaped text to exercise the redaction rules.  Building it
    here instead of spelling out ``eyJ...`` literals keeps secret scanners from
    mistaking a fake for a committed credential.  Nothing is signed.
    """
    return ".".join((b64url(header), b64url(payload), b64url(signature)))


_LOGGER_NAMES = ("uvicorn", "uvicorn.error", "uvicorn.access", "ccxt", "cf36", "stdlib")


@contextmanager
def isolated_logging_state() -> Iterator[None]:
    """Snapshot and restore every piece of global logging / registry state."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_loggers = {
        n: (
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).propagate,
            logging.getLogger(n).level,
            list(logging.getLogger(n).filters),
        )
        for n in _LOGGER_NAMES
    }
    saved_structlog = structlog.get_config()
    saved_ctx = dict(structlog.contextvars.get_contextvars())
    saved_configured = clog._LOGGING_CONFIGURED
    saved_registry = redaction._get_registry()
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for n, (handlers, propagate, level, filters) in saved_loggers.items():
            lg = logging.getLogger(n)
            lg.handlers[:] = handlers
            lg.propagate = propagate
            lg.setLevel(level)
            lg.filters[:] = filters
        structlog.configure(**saved_structlog)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**saved_ctx)
        clog._LOGGING_CONFIGURED = saved_configured
        redaction._set_registry(saved_registry)


class Capture:
    """Accumulates captured stdout / stderr across ``readouterr`` calls."""

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._capsys = capsys
        self._out = ""
        self._err = ""
        self._out_cursor = 0
        self._err_cursor = 0

    def _pull(self) -> None:
        o, e = self._capsys.readouterr()
        self._out += o
        self._err += e

    @property
    def out(self) -> str:
        self._pull()
        return self._out

    @property
    def err(self) -> str:
        self._pull()
        return self._err

    @property
    def text(self) -> str:
        return self.out + self.err

    def lines(self) -> list[str]:
        return [ln for ln in self.text.splitlines() if ln.strip()]

    def lines_with(self, needle: str) -> list[str]:
        return [ln for ln in self.lines() if needle in ln]

    def assert_redacted(self, event: str, *secrets: str, floor: int = 1) -> None:
        """Presence of ``event`` + a redaction marker in the NEW output, absence of secrets.

        "New" = everything captured since the previous ``assert_redacted`` /
        ``assert_present`` call, so a marker from an earlier event can never
        satisfy a later one.  A console traceback spans several lines, hence the
        marker is looked for in the whole new chunk, not only on the event line.
        """
        new = self._new_chunk()
        assert sum(event in ln for ln in new.splitlines()) >= floor, (
            f"event {event!r} not emitted (vacuous test)"
        )
        assert "[REDACTED" in new, "no redaction marker in the output of this event"
        text = self.text
        for secret in secrets:
            assert secret not in text, "secret leaked into the log stream"
        assert SENTINEL not in text, "sentinel prefix leaked into the log stream"

    def assert_present(self, event: str) -> None:
        """The event was emitted (non-vacuity) and no sentinel reached the sinks."""
        new = self._new_chunk()
        assert event in new, f"event {event!r} not emitted (vacuous test)"
        assert SENTINEL not in self.text, "sentinel prefix leaked into the log stream"

    def _new_chunk(self) -> str:
        out, err = self.out, self.err
        chunk = out[self._out_cursor :] + err[self._err_cursor :]
        self._out_cursor, self._err_cursor = len(out), len(err)
        return chunk


ConfigureFn = Callable[..., Capture]


@pytest.fixture
def production_logging(capsys: pytest.CaptureFixture[str]) -> Iterator[ConfigureFn]:
    """Return ``configure(json_output=True, log_level="INFO", ...) -> Capture``.

    Runs the real ``configure_logging`` after resetting structlog and the
    idempotency guard; everything is restored on teardown.
    """
    with isolated_logging_state():
        cap = Capture(capsys)

        def configure(
            json_output: bool = True,
            log_level: str = "INFO",
            *,
            register_env_secrets: bool = False,
        ) -> Capture:
            structlog.reset_defaults()
            structlog.contextvars.clear_contextvars()
            clog._LOGGING_CONFIGURED = False
            redaction._reset_registry_for_tests()
            clog.configure_logging(
                log_level=log_level,
                json_output=json_output,
                register_env_secrets=register_env_secrets,
            )
            structlog.configure(cache_logger_on_first_use=False)
            cap._pull()  # drain anything printed during configuration
            cap._out = cap._err = ""
            cap._out_cursor = cap._err_cursor = 0
            return cap

        yield configure


def disable_layer1() -> None:
    """Remove the layer-1 processor and unwrap the renderer on the LIVE structlog config.

    cf36b "red-before-green" aid: after :func:`production_logging` has configured the
    production chain, this leaves ``format_exc_info`` and the plain renderer only, so
    whatever a log SITE emits reaches the sink verbatim.  A site test that passes in this
    mode proves the site fix itself, independent of the cf36a processor.
    """
    cfg = structlog.get_config()
    processors: list[Any] = []
    for p in cfg["processors"]:
        if p is redaction.redact_event_dict:
            continue
        processors.append(p.inner if isinstance(p, redaction.RedactingRenderer) else p)
    assert not any(p is redaction.redact_event_dict for p in processors)
    assert not any(isinstance(p, redaction.RedactingRenderer) for p in processors)
    structlog.configure(processors=processors)


def json_lines(text: str) -> list[dict[str, Any]]:
    import json

    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# Shared credential-name matrices (T-01p / T-01n / T-36)
# ---------------------------------------------------------------------------

#: an UNREGISTERED fake value: only the pattern rules can protect it
U = f"{SENTINEL}UnregVal7x9Q2k"

PARITY_NAMES = [
    "api_key",
    "apiKey",
    "ApiKey",
    "APIKey",
    "XApiKey",
    "X-API-KEY",
    "x_api_key",
    "exchange_api_key",
    "exchangeApiKey",
    "api_secret",
    "apiSecret",
    "secret",
    "secret_key",
    "secretKey",
    "access_key",
    "accessKey",
    "AccessKeyId",
    "client_secret",
    "private_key",
    "privateKey",
    "token",
    "botToken",
    "bot_token",
    "access_token",
    "accesstoken",
    "refresh_token",
    "password",
    "db_password",
    "dbPassword",
    "passwd",
    "pwd",
    "passphrase",
    "signature",
    "Signature",
    "sign",
    "sig",
    "CB-ACCESS-KEY",
    "CB-ACCESS-SIGN",
    "CB-ACCESS-PASSPHRASE",
    "OK-ACCESS-SIGN",
    "KC-API-SIGN",
    "KC-API-KEY",
    "X-MBX-APIKEY",
    "X-BAPI-SIGN",
    "ACCESS-SIGN",
    "X-CMC_PRO_API_KEY",
    "X-Api-Secret",
    "X-ADMIN-KEY",
    "x-live-confirm-token",
    "authorization",
    "Authorization",
    "proxy-authorization",
    "cookie",
    "Set-Cookie",
    "dsn",
    "credential",
    "credentials",
    "auth",
    "basic_auth",
    "admin_key",
    "master_key",
    "hmac_key",
    "session_token",
    "telegram_bot_token",
]

NEGATIVE_NAMES = [
    "idempotency_key",
    "cache_key",
    "sort_key",
    "primary_key",
    "public_key",
    "tokens",
    "max_tokens",
    "token_count",
    "design",
    "assign",
    "signal",
    "signals",
    "key_prefix",
    "monkey",
    "auth_method",
    "api_auth_enabled",
    "total_signals",
    "WWW-Authenticate",
    "keys",
]
