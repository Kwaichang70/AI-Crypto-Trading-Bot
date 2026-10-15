"""
tests/integration/test_cf36_lifespan.py
---------------------------------------
CF-36 cf36a: the API lifespan registers every configured secret BEFORE the first
application log line (T-19) and hands the Telegram notifier a plain-str token (F-16).

T-12e (cf36a-fix): a REAL ``python -m uvicorn`` server whose lifespan calls
``configure_logging``: ``--no-access-log`` must stay silent (CF36A-SEC-01), an enabled
access log must not carry the query string, and a 60 KB pre-auth URL must not stall the
event loop (CF36A-SEC-02, R-07 ReDoS).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog
from fastapi.testclient import TestClient

import api
import common
import common.logging as clog
from api.config import get_settings
from common import redaction as R
from tests._cf36_logging import isolated_logging_state

S = "CF36SENTINEL"
TG_TOKEN = f"123456789:AA{S}TelegramSecret_0123456789-xyz"
ADMIN = f"{S}adminkey0123456789abcdef0123456789"
DB_PW = f"{S}-DbPw-0123456789"
REDIS_PW = f"{S}-RedisPw-0123456789"
FRED = f"{S}-fred-key-0123456789"


@pytest.fixture(autouse=True)
def _no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Settings`` reads ``.env`` / ``.env.local`` from the CWD: never let a developer or
    deploy checkout's real values into these tests (or into a failure diff). CF36B-SEC-02."""
    monkeypatch.chdir(tmp_path)


def _env(monkeypatch: pytest.MonkeyPatch, *, debug: bool) -> None:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DEBUG", "true" if debug else "false")
    monkeypatch.setenv("DATABASE_URL", f"postgresql+asyncpg://test:{DB_PW}@localhost:5432/test")
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TG_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    monkeypatch.setenv("REDIS_URL", f"redis://:{REDIS_PW}@localhost:6379/0")
    monkeypatch.setenv("FRED_API_KEY", FRED)


@pytest.mark.parametrize("debug", [False, True], ids=["json", "console"])
def test_t19_secrets_are_registered_before_api_starting(
    debug: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, debug=debug)
    notifier_kwargs: dict[str, Any] = {}

    def fake_notifier(**kwargs: Any) -> AsyncMock:
        notifier_kwargs.update(kwargs)
        return AsyncMock()

    monkeypatch.setattr("trading.telegram.TelegramNotifier", fake_notifier)
    with isolated_logging_state():
        structlog.reset_defaults()
        clog._LOGGING_CONFIGURED = False
        R._reset_registry_for_tests()
        get_settings.cache_clear()
        try:
            from api.main import create_app

            app = create_app()
            with TestClient(app, raise_server_exceptions=False):
                assert R.registered_secret_count() > 0
        finally:
            get_settings.cache_clear()
    captured = capsys.readouterr()
    out = captured.out
    lines = out.splitlines()
    first_registered = next(
        i for i, ln in enumerate(lines) if "log_redaction.secrets_registered" in ln
    )
    first_starting = next(i for i, ln in enumerate(lines) if "api.starting" in ln)
    assert first_registered < first_starting, (
        "secrets must be registered before the first app log line"
    )
    assert S not in out and S not in captured.err
    # F-16: the notifier received a plain str token (SecretStr unwrapped exactly once, in main.py)
    assert notifier_kwargs == {"bot_token": TG_TOKEN, "chat_id": "-100123"}
    assert type(notifier_kwargs["bot_token"]) is str


def test_t19_without_a_token_telegram_stays_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, debug=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    constructed: list[Any] = []
    monkeypatch.setattr("trading.telegram.TelegramNotifier", lambda **kw: constructed.append(kw))
    with isolated_logging_state():
        structlog.reset_defaults()
        clog._LOGGING_CONFIGURED = False
        get_settings.cache_clear()
        try:
            from api.main import create_app

            with TestClient(create_app(), raise_server_exceptions=False):
                pass
        finally:
            get_settings.cache_clear()
    assert constructed == []
    assert "telegram.disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# T-12e  real uvicorn server (cf36a-fix)
# ---------------------------------------------------------------------------

_APP_SRC = """
from contextlib import asynccontextmanager

from fastapi import FastAPI

import common.logging as clog


@asynccontextmanager
async def lifespan(app):
    clog.configure_logging(
        log_level="INFO",
        json_output=True,
        service_name="t12e",
        environment="test",
        register_env_secrets=False,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/{p:path}")
async def handler(p: str):
    return {"ok": True}
"""

INBOUND_KEY = f"{S}INBOUNDKEY0123456789"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _raw_get(port: int, path: str) -> tuple[float, str]:
    deadline = time.monotonic() + 30
    while True:
        try:
            conn = socket.create_connection(("127.0.0.1", port), timeout=30)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    with conn:
        t0 = time.perf_counter()
        conn.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while chunk := conn.recv(65536):
            data += chunk
        elapsed = time.perf_counter() - t0
    return elapsed, data.split(b"\r\n", 1)[0].decode()


@pytest.mark.parametrize("no_access_log", [True, False], ids=["no-access-log", "access-log"])
def test_t12e_real_uvicorn_access_log_and_redos(no_access_log: bool, tmp_path: Path) -> None:
    (tmp_path / "cf36_t12e_app.py").write_text(_APP_SRC)
    port = _free_port()
    pythonpath = os.pathsep.join(
        [
            str(tmp_path),
            str(Path(api.__file__).resolve().parent.parent),
            str(Path(common.__file__).resolve().parent.parent),
        ]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE", "COVERAGE"))}
    env["PYTHONPATH"] = pythonpath
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "cf36_t12e_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    try:
        import httptools  # noqa: F401
    except ImportError:  # h11 would answer 400 to a 60 KB request line
        cmd += ["--h11-max-incomplete-event-size", "131072"]
    if no_access_log:
        cmd.append("--no-access-log")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, st1 = _raw_get(
            port,
            f"/ping?api%5Fkey={INBOUND_KEY}&%61pi_key={INBOUND_KEY}"
            f"&apiSecret={INBOUND_KEY}&auth={INBOUND_KEY}",
        )
        big_elapsed, st2 = _raw_get(port, "/" + "-eyJ" * 15000)  # 60 KB, pre-auth
        _, st3 = _raw_get(port, "/ping")
    finally:
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
    for status in (st1, st2, st3):
        assert status.startswith("HTTP/1.1 200"), status
    text = out + err
    assert INBOUND_KEY not in text and S not in text, "an inbound credential reached the logs"
    access = []
    for ln in text.splitlines():
        if ln.startswith("{"):
            rec = json.loads(ln)
            if rec.get("logger") == "uvicorn.access":
                access.append(rec)
    if no_access_log:
        assert access == [], "configure_logging switched the access log back on"
    else:
        assert len(access) == 3
        assert "?[QUERY_REDACTED]" in access[0]["event"]
    # pre-fix: 1.3-1.9 s (quadratic R-07 on the access line); fixed: a few ms
    assert big_elapsed < 0.5, f"{big_elapsed * 1e3:.0f} ms"
