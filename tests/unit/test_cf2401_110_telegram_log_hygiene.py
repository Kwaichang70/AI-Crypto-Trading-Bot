"""
tests/unit/test_cf2401_110_telegram_log_hygiene.py
--------------------------------------------------
CF-2401-110 (SEC-GD-04): no Telegram log line carries the chat id, bot token, request URL,
message text, Bot API response text or exception text, at any level.  Sites pinned:
``telegram.send_failed`` (``status`` int only, the body is never read), ``telegram.send_error``
(``error_type`` only) and, in ``api.main`` step 7, ``telegram.configured`` (no field) and
``telegram.startup_error`` (``exc_log_fields`` only).

Each test runs the PRODUCTION logging chain at DEBUG with layer 1 (the redactor) removed, so a
pass proves the SITE, not the redaction layer.  Every absence assertion is paired with a presence
assertion and a CLOSED field set.  Step 7 cannot run in a unit test without the poller (network),
so its real statements are compiled out of ``apps/api/main.py`` with ``ast`` and run on a copy of
the module globals.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import structlog
from pydantic import SecretStr

import api.main
from tests._cf36_logging import Capture, ConfigureFn, disable_layer1, json_lines
from trading.telegram import TelegramNotifier

# Synthetic sentinels, assembled at runtime so scanners do not match them.
CHAT = "-100" + "2401110110"
TOKEN = "2401110" + ":" + "AAcf110FakeTokenSentinel_xyz-0"
TOKEN_SECRET = TOKEN.split(":", 1)[1]
TOKEN_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
EXC_MARK, MSG_MARK, RSN_MARK = "CF110EXCSENTINEL", "CF110MSGSENTINEL", "CF110RSNSENTINEL"
EXC_TEXT = f"POST {TOKEN_URL} chat {CHAT} failed {EXC_MARK}"
MSG = f"HALT {MSG_MARK}"  # the message text is never logged either
# TOKEN_SECRET is a substring of TOKEN, so TOKEN needs no entry of its own.
LEAKS = (CHAT, "2401110110", TOKEN_SECRET, "api.telegram.org/bot", EXC_MARK, MSG_MARK, RSN_MARK)
_ERROR_AT = re.compile(r"^[\w.\-]+\.py:\d+:[\w<>]+$")


def _setup(production_logging: ConfigureFn) -> tuple[Capture, set[str]]:
    """DEBUG, layer 1 off; returns the capture and the envelope keys of a probe event."""
    cap = production_logging(json_output=True, log_level="DEBUG")
    disable_layer1()
    structlog.get_logger("cf2401.probe").debug("cf2401_110.probe")
    probes = _events(cap, "cf2401_110.probe")
    assert len(probes) == 1, "DEBUG probe not emitted: the chain is not at DEBUG"
    return cap, set(probes[0])


def _events(cap: Capture, name: str) -> list[dict[str, Any]]:
    return [e for e in json_lines(cap.out) if e.get("event") == name]


def _assert_clean(cap: Capture) -> None:
    text = cap.text
    assert [m for m in LEAKS if m in text] == [], (
        "a Telegram secret or response text reached the sink"
    )
    assert "Traceback" not in text and "exc_info" not in text and '"exception"' not in text


def _the_event(cap: Capture, envelope: set[str], name: str) -> tuple[dict[str, Any], set[str]]:
    evs = _events(cap, name)
    assert len(evs) == 1, f"{name!r} not emitted exactly once; got {cap.out[:400]}"
    assert _events(cap, f"{name}.trace") == [], "a DEBUG trace sibling would carry exception text"
    return evs[0], set(evs[0]) - envelope


class _Resp:
    """``aiohttp.ClientResponse`` stand-in WITHOUT ``text()`` / ``read()`` / ``json()``: a body
    read raises ``AttributeError``, ends as ``telegram.send_error`` and fails the test.  As on the
    real class, ``repr``, ``url``, ``real_url`` and ``request_info`` carry the token URL."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = f"Bad Request {RSN_MARK}"  # status-line text, reachable without a body read
        self.url = self.real_url = TOKEN_URL
        self.request_info = SimpleNamespace(method="POST", url=TOKEN_URL, real_url=TOKEN_URL)

    def __repr__(self) -> str:
        return f"<ClientResponse({TOKEN_URL}) [{self.status} {self.reason}]>"

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *_a: object) -> None:
        return None


def _notifier(post: Any) -> tuple[TelegramNotifier, MagicMock]:
    notifier = TelegramNotifier(bot_token=TOKEN, chat_id=CHAT)
    session = MagicMock()
    session.post = post
    notifier._get_session = AsyncMock(return_value=session)  # type: ignore[method-assign]
    return notifier, session


# --- packages/trading/telegram.py ---


@pytest.mark.parametrize("status", [400, 403, 429, 500])
async def test_send_failed_logs_status_int_only(
    production_logging: ConfigureFn, status: int
) -> None:
    cap, envelope = _setup(production_logging)
    notifier, _ = _notifier(MagicMock(return_value=_Resp(status)))
    assert await notifier.send_message(MSG) is False
    ev, fields = _the_event(cap, envelope, "telegram.send_failed")
    assert fields == {"status"}, fields
    assert type(ev["status"]) is int and ev["status"] == status
    _assert_clean(cap)


async def test_send_return_values_and_request_unchanged(production_logging: ConfigureFn) -> None:
    cap, _ = _setup(production_logging)
    post = MagicMock(return_value=_Resp(200))
    notifier, _ = _notifier(post)
    assert await notifier.send_message(MSG, parse_mode="HTML") is True
    assert not [e for e in json_lines(cap.out) if str(e.get("event", "")).startswith("telegram.")]
    args, kwargs = post.call_args
    assert args == (TOKEN_URL,)
    assert kwargs["json"] == {
        "chat_id": CHAT,
        "text": MSG,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    assert kwargs["timeout"].total == 10
    _assert_clean(cap)


@pytest.mark.parametrize(
    "exc",
    [aiohttp.InvalidURL(TOKEN_URL), RuntimeError(EXC_TEXT), TimeoutError(EXC_TEXT)],
    ids=["invalid_url", "runtime_error", "timeout"],
)
async def test_send_error_logs_type_only_at_debug(
    production_logging: ConfigureFn, exc: Exception
) -> None:
    cap, envelope = _setup(production_logging)
    notifier, _ = _notifier(MagicMock(side_effect=exc))
    assert await notifier.send_message(MSG) is False
    ev, fields = _the_event(cap, envelope, "telegram.send_error")
    assert fields == {"error_type"}, fields
    assert ev["error_type"] == type(exc).__name__
    _assert_clean(cap)


# --- apps/api/main.py step 7 (the real statements, compiled out of the source) ---


def _step7_block() -> tuple[list[ast.stmt], str]:
    path = Path(api.main.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    lifespans = [
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan"
    ]
    assert len(lifespans) == 1
    body = lifespans[0].body
    starts = [
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "telegram_token" for t in n.targets)
    ]
    assert len(starts) == 1, "step 7 moved: the harness must be updated, not skipped"
    block = body[starts[0] : starts[0] + 2]
    assert isinstance(block[1], ast.If)
    return block, str(path)


def _step7_runner() -> dict[str, Any]:
    block, filename = _step7_block()
    mod = ast.parse("async def _step7(settings, log, container):\n    global _telegram_notifier\n")
    fn = mod.body[0]
    assert isinstance(fn, ast.AsyncFunctionDef)
    fn.body.extend(block)
    ast.fix_missing_locations(mod)
    ns = dict(vars(api.main))  # a copy: the real module global is never written
    exec(compile(mod, filename, "exec"), ns)  # noqa: S102 - production source under test
    return ns


async def _run_step7(chat: str | None = CHAT) -> tuple[Any, Any]:
    ns = _step7_runner()
    settings = SimpleNamespace(telegram_bot_token=SecretStr(TOKEN), telegram_chat_id=chat)
    container = SimpleNamespace(services=SimpleNamespace(telegram_notifier=None))
    await ns["_step7"](settings, structlog.get_logger("api.main"), container)
    return container, ns["_telegram_notifier"]


async def test_step7_configured_logs_no_chat_id(
    production_logging: ConfigureFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap, envelope = _setup(production_logging)
    built: list[dict[str, Any]] = []

    def fake_notifier(**kwargs: Any) -> object:
        built.append(kwargs)
        return object()

    monkeypatch.setattr("trading.telegram.TelegramNotifier", fake_notifier)
    container, notifier = await _run_step7()
    assert built == [{"bot_token": TOKEN, "chat_id": CHAT}]
    assert container.services.telegram_notifier is notifier is not None
    _, fields = _the_event(cap, envelope, "telegram.configured")
    assert fields == set(), fields
    _assert_clean(cap)


async def test_step7_startup_error_logs_type_only(
    production_logging: ConfigureFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap, envelope = _setup(production_logging)

    def boom(**_kwargs: Any) -> object:
        raise RuntimeError(EXC_TEXT)

    monkeypatch.setattr("trading.telegram.TelegramNotifier", boom)
    container, notifier = await _run_step7()  # must not raise
    assert notifier is None and container.services.telegram_notifier is None
    ev, fields = _the_event(cap, envelope, "telegram.startup_error")
    assert fields <= {"error_type", "error_at", "http_status"}, fields
    assert ev["error_type"] == "RuntimeError"
    assert _ERROR_AT.match(ev["error_at"]), ev.get("error_at")
    assert _events(cap, "telegram.configured") == []
    _assert_clean(cap)
