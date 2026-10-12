"""
tests/integration/test_cf36_ws_realio.py
----------------------------------------
CF-36 cf36b-fix, CF-A (CF36A-SEC-204) on a REAL ``python -m uvicorn`` server whose
lifespan calls ``configure_logging`` (as ``api.main`` does).

uvicorn writes ``'%s - "WebSocket %s" [accepted]|403|%d'`` to ``uvicorn.error`` for
every Upgrade request, from the protocol layer, before routing: an accepted route, a
path with NO WebSocket route (403) and a denial response (``%d``) are all logged with
the raw query string, with or without ``--no-access-log``.  The same server also
answers an HTTP request carrying the key, so both log paths are checked in one run.

Server output goes to a file in ``tmp_path`` (never a pipe, CF36A-SEC-210).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import api
import common
from tests._cf36_logging import SENTINEL as S

KEY = f"{S}WSREALKEY0123456789abcdef"
QUERY = f"api%5Fkey={KEY}&%61pi_key={KEY}&t={KEY}"

_APP_SRC = """
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from starlette.responses import Response

import common.logging as clog


@asynccontextmanager
async def lifespan(app):
    clog.configure_logging(
        log_level="INFO",
        json_output=True,
        service_name="cfa",
        environment="test",
        register_env_secrets=False,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.close()


@app.websocket("/deny")
async def deny(websocket: WebSocket):
    await websocket.send_denial_response(Response(status_code=401))


@app.get("/{p:path}")
async def handler(p: str):
    return {"ok": True}
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _connect(port: int) -> socket.socket:
    deadline = time.monotonic() + 30
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=30)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def _status(port: int, request: str) -> str:
    with _connect(port) as conn:
        conn.sendall(request.encode())
        data = b""
        while b"\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.split(b"\r\n", 1)[0].decode()


def _upgrade(port: int, path: str) -> str:
    return _status(
        port,
        f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
    )


def _get(port: int, path: str) -> str:
    return _status(port, f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")


@pytest.mark.parametrize("no_access_log", [True, False], ids=["no-access-log", "access-log"])
@pytest.mark.parametrize("ws_impl", ["websockets", "websockets-sansio", "wsproto"])
def test_cfa_real_uvicorn_ws_and_http_lines_carry_no_query(
    ws_impl: str, no_access_log: bool, tmp_path: Path
) -> None:
    pytest.importorskip("wsproto" if ws_impl == "wsproto" else "websockets")
    (tmp_path / "cf36_cfa_app.py").write_text(_APP_SRC)
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE", "COVERAGE"))}
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(tmp_path),
            str(Path(api.__file__).resolve().parent.parent),
            str(Path(common.__file__).resolve().parent.parent),
        ]
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "cf36_cfa_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--ws",
        ws_impl,
    ]
    if no_access_log:
        cmd.append("--no-access-log")
    log_path = tmp_path / "server.log"
    with log_path.open("w") as sink:
        proc = subprocess.Popen(  # noqa: S603
            cmd, cwd=tmp_path, env=env, stdout=sink, stderr=subprocess.STDOUT, text=True
        )
        try:
            st_ws = _upgrade(port, f"/ws?{QUERY}")
            st_noroute = _upgrade(port, f"/no/ws/route?{QUERY}")
            st_deny = _upgrade(port, f"/deny?{QUERY}")
            st_http = _get(port, f"/ping?{QUERY}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    assert st_ws.startswith("HTTP/1.1 101"), st_ws
    assert st_noroute.startswith("HTTP/1.1 403"), st_noroute
    assert st_deny.startswith("HTTP/1.1 401"), st_deny
    assert st_http.startswith("HTTP/1.1 200"), st_http

    text = log_path.read_text()
    assert KEY not in text and S not in text, "an inbound credential reached the logs"
    records = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    ws_events = sorted(
        r["event"]
        for r in records
        if r.get("logger") == "uvicorn.error" and '"WebSocket ' in r.get("event", "")
    )
    assert len(ws_events) == 3, ws_events
    for path, suffix in (("/ws", "[accepted]"), ("/no/ws/route", "403"), ("/deny", "401")):
        needle = f'"WebSocket {path}?[QUERY_REDACTED]" {suffix}'
        assert sum(e.endswith(needle) for e in ws_events) == 1, (needle, ws_events)
    access = [r for r in records if r.get("logger") == "uvicorn.access"]
    if no_access_log:
        assert access == [], "configure_logging switched the access log back on"
    else:
        assert len(access) == 1 and "/ping?[QUERY_REDACTED]" in access[0]["event"]


def test_cfa_real_api_app_route_less_upgrade_is_stripped(tmp_path: Path) -> None:
    """CF36B-SEC-01: the production ``api.main:app`` under the image flags has no WebSocket
    route, yet uvicorn logs the Upgrade request (403) with its raw query on ``uvicorn.error``.
    Hermetic: whitelisted environment, CWD without ``.env``, unreachable database."""
    port = _free_port()
    env = {
        k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ
    }
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(api.__file__).resolve().parent.parent),
                str(Path(common.__file__).resolve().parent.parent),
            ]
        ),
        DATABASE_URL=f"postgresql+asyncpg://u:{S}-DbPw-0123456789@127.0.0.1:1/x",
        ADMIN_API_KEY=f"{S}adminkey0123456789abcdef0123456789",
        REQUIRE_API_AUTH="true",
        DEBUG="false",
        LOG_LEVEL="INFO",
        PROMETHEUS_ENABLED="false",
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--no-access-log",
    ]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as sink:
        proc = subprocess.Popen(  # noqa: S603
            cmd, cwd=tmp_path, env=env, stdout=sink, stderr=subprocess.STDOUT, text=True
        )
        try:
            status = _upgrade(port, f"/health?{QUERY}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    assert status.startswith("HTTP/1.1 403"), status
    text = log_path.read_text()
    assert KEY not in text and S not in text, "an inbound credential reached the logs"
    assert (
        sum(
            '"WebSocket /health?[QUERY_REDACTED]" 403' in ln.replace('\\"', '"')
            for ln in text.splitlines()
        )
        == 1
    )
