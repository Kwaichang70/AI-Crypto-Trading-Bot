"""
tests/unit/test_cf36_ws.py
--------------------------
CF-36 cf36b-fix, CF-A (CF36A-SEC-204): uvicorn logs every WebSocket Upgrade request on
``uvicorn.error`` (INFO) as ``'%s - "WebSocket %s" [accepted]|403|%d'`` with the RAW
query string, also for paths without a WebSocket route and also under
``--no-access-log``.  ``?api%5Fkey=<K>`` authenticates on HTTP routes (Starlette decodes
the name), and no name-based rule sees it.  ``_WsQueryFilter`` strips the query.

The templates and argument shapes below are copied from uvicorn 0.41.0
(``protocols/websockets/{websockets_impl,websockets_sansio_impl,wsproto_impl}.py``):
``args = (client, path_with_query[, status])``, where ``client`` is a ``str`` for
``websockets`` and a ``(host, port)`` tuple for ``websockets-sansio`` and ``wsproto``.
The real-server proof is ``tests/integration/test_cf36_ws_realio.py``.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

import pytest

import common.logging as clog
from tests._cf36_logging import SENTINEL, ConfigureFn

S = SENTINEL
KEY = f"{S}WSKEY0123456789abcdef"
#: Encoded names (R-08 cannot see them) and a neutral name (no rule at all).
QUERY = f"api%5Fkey={KEY}&%61pi_key={KEY}&t={KEY}"

_TEMPLATES: list[tuple[str, tuple[Any, ...], str]] = [
    ('%s - "WebSocket %s" [accepted]', (), "[accepted]"),
    ('%s - "WebSocket %s" 403', (), "403"),
    ('%s - "WebSocket %s" %d', (401,), "401"),
]
_CLIENTS: list[Any] = ["127.0.0.1:50000", ("127.0.0.1", 50000)]


def _uvicorn_state() -> None:
    import uvicorn.config

    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)


def _ws_filters(name: str) -> int:
    return sum(isinstance(f, clog._WsQueryFilter) for f in logging.getLogger(name).filters)


@pytest.mark.parametrize("client", _CLIENTS, ids=["client-str", "client-tuple"])
@pytest.mark.parametrize(
    ("template", "extra", "suffix"), _TEMPLATES, ids=["accepted", "403", "status"]
)
@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
def test_cfa_ws_handshake_query_is_stripped(
    json_output: bool,
    template: str,
    extra: tuple[Any, ...],
    suffix: str,
    client: Any,
    production_logging: ConfigureFn,
) -> None:
    _uvicorn_state()
    cap = production_logging(json_output=json_output)
    logging.getLogger("uvicorn.error").info(template, client, f"/ws/x?{QUERY}", *extra)
    lines = cap.lines_with("WebSocket /ws/x?[QUERY_REDACTED]")  # presence ...
    assert len(lines) == 1 and lines[0].split("[QUERY_REDACTED]", 1)[1].find(suffix) >= 0
    assert KEY not in cap.text and S not in cap.text  # ... plus absence


@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
def test_cfa_premise_layer1_alone_does_not_strip_the_query(
    json_output: bool,
    production_logging: ConfigureFn,
) -> None:
    """Non-vacuity: without the filter the key reaches the sink, so the filter is the control."""
    _uvicorn_state()
    cap = production_logging(json_output=json_output)
    err = logging.getLogger("uvicorn.error")
    err.filters[:] = [f for f in err.filters if not isinstance(f, clog._WsQueryFilter)]
    err.info('%s - "WebSocket %s" 403', "127.0.0.1:50000", f"/ws?t={KEY}")
    assert KEY in cap.text


def test_cfa_filter_is_installed_once_and_only_on_uvicorn_error(
    production_logging: ConfigureFn,
) -> None:
    _uvicorn_state()
    production_logging(json_output=True)
    clog._LOGGING_CONFIGURED = False  # configure a second time (idempotent install)
    clog.configure_logging(log_level="INFO", json_output=True, register_env_secrets=False)
    assert _ws_filters("uvicorn.error") == 1
    assert _ws_filters("uvicorn") == 0 and _ws_filters("uvicorn.access") == 0
    assert _ws_filters("") == 0


def _record(msg: Any, args: Any) -> logging.LogRecord:
    rec = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, msg, None, None)
    rec.args = args  # set after construction: LogRecord unwraps a lone mapping argument
    return rec


@pytest.mark.parametrize(
    ("msg", "args"),
    [
        ("%s %s", ("127.0.0.1:1", f"/p?t={KEY}")),  # other uvicorn.error template
        ("Invalid HTTP request received.", None),
        ('%s - "WebSocket %s" 403', None),
        ('%s - "WebSocket %s" 403', ("127.0.0.1:1",)),
        ('%s - "WebSocket %s" 403', ("127.0.0.1:1", b"/ws?t=x")),
        ('%s - "WebSocket %s" 403', ("127.0.0.1:1", "/ws")),
        ('%s - "WebSocket %s" 403', {"a": 1}),
    ],
    ids=["other-template", "no-args", "ws-no-args", "ws-short", "ws-bytes", "ws-no-query", "dict"],
)
def test_cfa_non_matching_records_pass_unchanged(msg: Any, args: Any) -> None:
    rec = _record(msg, args)
    assert clog._WsQueryFilter().filter(rec) is True
    assert rec.msg == msg and rec.args == args


def test_cfa_query_is_cut_at_the_first_question_mark_and_status_is_kept() -> None:
    client = ("127.0.0.1", 50000)
    rec = _record('%s - "WebSocket %s" %d', (client, f"/ws?a=1?t={KEY}", 401))
    assert clog._WsQueryFilter().filter(rec) is True
    assert rec.args == (client, "/ws?[QUERY_REDACTED]", 401)
    assert rec.getMessage() == "('127.0.0.1', 50000) - \"WebSocket /ws?[QUERY_REDACTED]\" 401"


class _HostileArgs(tuple):  # type: ignore[type-arg]
    def __getitem__(self, i: Any) -> Any:
        raise RuntimeError(KEY)


def test_cfa_filter_never_raises_and_fails_closed() -> None:
    rec = _record('%s - "WebSocket %s" 403', _HostileArgs(("a", f"/ws?t={KEY}")))
    assert clog._WsQueryFilter().filter(rec) is True
    assert rec.msg == "[REDACTION_FAILED]" and rec.args is None
    assert KEY not in rec.getMessage()


def test_cfa_image_cmd_keeps_uvicorn_quiet() -> None:
    """The filters cover uvicorn's INFO lines only.  At ``--log-level debug|trace`` the
    ``websockets`` library logs ``< GET <path>?<raw query>`` and every request header on
    ``uvicorn.error``; ``--access-log`` re-enables the access log.  Pin the image CMD."""
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "infra" / "Dockerfile.api").read_text()
    match = re.search(r"^CMD (\[.*?\])\s*$", dockerfile.replace("\\\n", " "), re.M | re.S)
    assert match, "CMD exec form not found in infra/Dockerfile.api"
    cmd = json.loads(match.group(1))
    assert cmd[:3] == ["python", "-m", "uvicorn"], cmd
    assert "--no-access-log" in cmd
    assert not any(a.startswith(("--log-level", "--access-log", "--log-config")) for a in cmd), cmd
    compose = (root / "infra" / "docker-compose.yml").read_text()
    api_block = compose.split("\n  api:\n", 1)[1].split("\n  ui:\n", 1)[0]
    assert "\n    command:" not in api_block and "\n    entrypoint:" not in api_block
