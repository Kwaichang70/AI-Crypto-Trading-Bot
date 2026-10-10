"""
tests/unit/test_wp16_telegram_log_hygiene.py
--------------------------------------------
WP16-SEC-06 (WP1.6 final synthesis, ruling FS-03): ``TelegramNotifier.send_message``
must never log the bot token.  The request URL ``.../bot<token>/sendMessage`` is
embedded in the text and repr of many aiohttp exceptions, so the failure log
carries the exception TYPE only.

    T-45a  every aiohttp exception class that embeds the URL when constructed
           for real: no token in the captured logs; ``error_type`` is logged
    T-45b  every Exception class exported by ``aiohttp`` (sweep), as a subclass
           whose str / repr / args are the token URL: no token in the logs
    T-45c  a non-aiohttp exception with the URL in its message, and a timeout
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from structlog.testing import capture_logs
from yarl import URL

from trading.telegram import TelegramNotifier

TOKEN = "123456789:AAFakeBotTokenValue_xyz-0"  # noqa: S105 - fake value, proves it is never logged
SECRET_PART = TOKEN.split(":", 1)[1]
TOKEN_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"


def _request_info() -> aiohttp.RequestInfo:
    url = URL(TOKEN_URL)
    return aiohttp.RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)


#: Constructed exactly as aiohttp raises them; each embeds the token at HEAD.
_REAL: dict[str, Callable[[], Exception]] = {
    "InvalidURL": lambda: aiohttp.InvalidURL(TOKEN_URL),
    "InvalidUrlClientError": lambda: aiohttp.InvalidUrlClientError(TOKEN_URL),
    "NonHttpUrlClientError": lambda: aiohttp.NonHttpUrlClientError(TOKEN_URL),
    "InvalidUrlRedirectClientError": lambda: aiohttp.InvalidUrlRedirectClientError(TOKEN_URL),
    "NonHttpUrlRedirectClientError": lambda: aiohttp.NonHttpUrlRedirectClientError(TOKEN_URL),
    "TooManyRedirects": lambda: aiohttp.TooManyRedirects(_request_info(), ()),
    "ClientResponseError": lambda: aiohttp.ClientResponseError(
        _request_info(), (), status=500, message="Internal Server Error"
    ),
    "ContentTypeError": lambda: aiohttp.ContentTypeError(_request_info(), ()),
    "WSServerHandshakeError": lambda: aiohttp.WSServerHandshakeError(_request_info(), ()),
}

_AIOHTTP_EXCEPTIONS = sorted(
    name
    for name in dir(aiohttp)
    if isinstance(getattr(aiohttp, name), type) and issubclass(getattr(aiohttp, name), Exception)
)


def _leaky(name: str) -> Exception:
    """An instance of ``aiohttp.<name>`` (subclass) whose str / repr / args are the URL."""
    base: type[Exception] = getattr(aiohttp, name)
    leaky = type(name, (base,), {"__str__": lambda _s: TOKEN_URL, "__repr__": lambda _s: TOKEN_URL})
    exc: Exception = base.__new__(leaky)
    exc.args = (TOKEN_URL,)
    return exc


async def _send_raising(exc: BaseException) -> tuple[bool, list[dict[str, Any]]]:
    notifier = TelegramNotifier(bot_token=TOKEN, chat_id="-100123")
    session = MagicMock()
    session.post = MagicMock(side_effect=exc)
    notifier._get_session = AsyncMock(return_value=session)  # type: ignore[method-assign]
    with capture_logs() as logs:
        ok = await notifier.send_message("HALT")
    return ok, logs


def _assert_no_token(ok: bool, logs: list[dict[str, Any]], exc: BaseException) -> None:
    blob = repr(logs)
    assert ok is False
    assert TOKEN not in blob and SECRET_PART not in blob, blob
    assert "api.telegram.org/bot" not in blob, blob
    errors = [e for e in logs if e["event"] == "telegram.send_error"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == type(exc).__name__
    assert "exc_info" not in errors[0] and "error" not in errors[0]


def test_wp16_t45_the_real_exceptions_do_embed_the_token() -> None:
    """Guard on the premise: if aiohttp stops embedding the URL, T-45a is vacuous."""
    assert len(_AIOHTTP_EXCEPTIONS) >= 9  # WP16-SEC-202: an empty sweep must fail, not skip
    for factory in _REAL.values():
        exc = factory()
        assert SECRET_PART in str(exc) or SECRET_PART in repr(exc)


@pytest.mark.parametrize("name", sorted(_REAL))
async def test_wp16_t45a_real_aiohttp_exception_never_logs_the_token(name: str) -> None:
    exc = _REAL[name]()
    ok, logs = await _send_raising(exc)
    _assert_no_token(ok, logs, exc)


@pytest.mark.parametrize("name", _AIOHTTP_EXCEPTIONS)
async def test_wp16_t45b_every_aiohttp_exception_type_never_logs_the_token(name: str) -> None:
    exc = _leaky(name)
    ok, logs = await _send_raising(exc)
    _assert_no_token(ok, logs, exc)


@pytest.mark.parametrize(
    "exc",
    [RuntimeError(f"POST {TOKEN_URL} failed"), TimeoutError()],
    ids=["runtime_error_with_url", "timeout"],
)
async def test_wp16_t45c_other_exceptions_never_log_the_token(exc: Exception) -> None:
    ok, logs = await _send_raising(exc)
    _assert_no_token(ok, logs, exc)
