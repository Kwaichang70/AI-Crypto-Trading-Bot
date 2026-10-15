"""
tests/integration/test_cf36_realio.py
-------------------------------------
CF-36 cf36a: REAL I/O against a local stub server (no external network).

    T-09  FRED / Whale Alert / ccxt binance / Telegram, through the production
          logging chain; run once with the layer-2 site fix and once with it
          reverted to ``str(exc)`` to prove layer 1 alone holds
    T-28  FRED and Whale Alert: ``error_type`` + ``http_status``, never the key
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import ccxt.async_support as ccxt_async
import pytest
import structlog
from aiohttp import web
from structlog.testing import capture_logs

import data.macro_data as macro
import data.whale_tracker as whale
from common import redaction as R
from data.market_data import MarketDataError
from tests._cf36_logging import SENTINEL as S
from tests._cf36_logging import ConfigureFn

FRED_KEY = f"{S}FREDKEY0123456789"
WHALE_KEY = f"{S}WHALEKEY0123456789"
SIG = f"{S}SIG0123456789abcdef"
TG_TOKEN = f"123456789:AA{S}TelegramSecret_0123456789-xyz"
TG_SECRET = TG_TOKEN.split(":", 1)[1]


@pytest.fixture
async def stub() -> AsyncIterator[SimpleNamespace]:
    state: dict[str, Any] = {"status": 400, "body": "bad request", "paths": []}

    async def handler(request: web.Request) -> web.Response:
        state["paths"].append(request.path_qs)
        body = state["body"](request) if callable(state["body"]) else state["body"]
        return web.Response(status=state["status"], text=body)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield SimpleNamespace(port=port, base=f"http://127.0.0.1:{port}", state=state)
    await runner.cleanup()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _point_clients_at(base: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macro, "_BASE_URL", f"{base}/fred/series/observations")
    monkeypatch.setattr(whale, "_ENDPOINT", f"{base}/whale/v1/transactions")


# ---------------------------------------------------------------------------
# T-28  FRED / Whale: structured fields, never the key (capture_logs = layer 2 alone)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 429, 500, 503])
async def test_t28_fred_and_whale_log_type_and_status_not_the_key(
    status: int, stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_clients_at(stub.base, monkeypatch)
    stub.state["status"] = status
    fred = macro.FREDClient(api_key=FRED_KEY)
    wh = whale.WhaleAlertClient(api_key=WHALE_KEY)
    try:
        with capture_logs() as logs:
            await fred.get_latest()
            assert await wh.get_latest() is None
    finally:
        await fred.close()
        await wh.close()
    assert any("/fred/" in p and f"api_key={FRED_KEY}" in p for p in stub.state["paths"]), (
        "premise: the key really is in the request URL"
    )
    blob = repr(logs)
    assert FRED_KEY not in blob and WHALE_KEY not in blob and S not in blob
    series = [e for e in logs if e["event"] == "fred_client.series_fetch_failed"]
    assert {e["series_id"] for e in series} == {"FEDFUNDS", "T10Y2Y"}
    (whale_ev,) = [e for e in logs if e["event"] == "whale_alert_client.fetch_failed"]
    for e in (*series, whale_ev):
        assert e["error_type"] == "ClientResponseError"
        assert e["http_status"] == status
        assert "error" not in e and "exc_info" not in e
        assert e["error_at"].count(":") == 2 and e["error_at"].split(":")[2] == "raise_for_status"


async def test_t28_connection_refused_has_a_type_and_no_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _point_clients_at(f"http://127.0.0.1:{_closed_port()}", monkeypatch)
    wh = whale.WhaleAlertClient(api_key=WHALE_KEY)
    try:
        with capture_logs() as logs:
            assert await wh.get_latest() is None
    finally:
        await wh.close()
    (ev,) = [e for e in logs if e["event"] == "whale_alert_client.fetch_failed"]
    assert ev["error_type"].startswith("ClientConnector")
    assert "http_status" not in ev and WHALE_KEY not in repr(logs)


# ---------------------------------------------------------------------------
# T-09  production chain, layer 2 on and layer 2 reverted
# ---------------------------------------------------------------------------


def _revert_layer2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the pre-fix ``error=str(exc)`` behaviour back at the F-10 sites."""
    monkeypatch.setattr(macro, "exc_log_fields", lambda exc: {"error": str(exc)})
    monkeypatch.setattr(whale, "exc_log_fields", lambda exc: {"error": str(exc)})


@pytest.mark.parametrize("layer2", ["fixed", "reverted"])
@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
@pytest.mark.parametrize("status", [400, 429, 503])
async def test_t09_fred_and_whale_through_the_production_chain(
    layer2: str,
    json_output: bool,
    status: int,
    stub: SimpleNamespace,
    production_logging: ConfigureFn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = production_logging(json_output=json_output)
    _point_clients_at(stub.base, monkeypatch)
    stub.state["status"] = status
    if layer2 == "reverted":
        _revert_layer2(monkeypatch)
    fred = macro.FREDClient(api_key=FRED_KEY)
    wh = whale.WhaleAlertClient(api_key=WHALE_KEY)
    try:
        await fred.get_latest()
        await wh.get_latest()
    finally:
        await fred.close()
        await wh.close()
    text = cap.text
    assert "fred_client.series_fetch_failed" in text and "whale_alert_client.fetch_failed" in text
    assert FRED_KEY not in text and WHALE_KEY not in text and S not in text
    if layer2 == "reverted":
        # the leaked URL is still in the event; only layer 1 stands between it and the sink
        assert text.count("api_key=[REDACTED]") >= 3
    else:
        assert "ClientResponseError" in text and str(status) in text


async def test_t09_reverted_layer2_leaks_without_layer1(
    stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red evidence in the suite: the reverted site + capture_logs (no layer 1) leaks."""
    _point_clients_at(stub.base, monkeypatch)
    _revert_layer2(monkeypatch)
    fred = macro.FREDClient(api_key=FRED_KEY)
    try:
        with capture_logs() as logs:
            await fred.get_latest()
    finally:
        await fred.close()
    assert FRED_KEY in repr(logs), "premise: the site fix is what stops the leak at layer 2"


@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
@pytest.mark.parametrize("status", [503, 400])
async def test_t09_real_ccxt_binance_fetch_through_the_production_chain(
    json_output: bool,
    status: int,
    stub: SimpleNamespace,
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=json_output)
    stub.state["status"] = status
    # plain-text body: a JSON body would be mapped by binance.handle_errors to a
    # URL-less exception before ccxt's generic status handler (which embeds the URL)
    stub.state["body"] = "upstream unavailable"
    exchange = ccxt_async.binance({"enableRateLimit": False})
    url = f"{stub.base}/api/v3/account?timestamp=1&recvWindow=5000&signature={SIG}"
    log = structlog.get_logger("cf36.realio")
    try:
        with pytest.raises(Exception) as info:
            await exchange.fetch(url, "GET", {"X-MBX-APIKEY": f"{S}APIKEYHEADER0123456789"}, None)
        exc = info.value
        assert SIG in str(exc), "premise: ccxt embeds the signed URL in the exception text"
        log.error("t09.ccxt.str", error=str(exc))
        try:
            raise MarketDataError(f"Network error fetching 'BTC/USDT': {exc}") from exc
        except MarketDataError:
            log.exception("t09.ccxt.chain")
    finally:
        await exchange.close()
    text = cap.text
    assert "t09.ccxt.str" in text and "t09.ccxt.chain" in text
    assert SIG not in text and S not in text
    assert "signature=[REDACTED]" in text


@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
async def test_t09_telegram_echo_server_body_is_redacted(
    json_output: bool,
    stub: SimpleNamespace,
    production_logging: ConfigureFn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading import telegram

    cap = production_logging(json_output=json_output)
    R.register_secret(TG_TOKEN, name="telegram_bot_token", kind="telegram")
    stub.state["status"] = 404
    stub.state["body"] = lambda req: f'{{"ok":false,"description":"Not Found: {req.path}"}}'
    # the module-level logger may be cached against an older config; use a fresh one
    monkeypatch.setattr(telegram, "logger", structlog.get_logger("trading.telegram"))
    notifier = telegram.TelegramNotifier(bot_token=TG_TOKEN, chat_id="-100123")
    notifier._url = f"{stub.base}/bot{TG_TOKEN}/sendMessage"
    try:
        assert await notifier.send_message("HALT") is False
    finally:
        await notifier.close()
    assert any(TG_TOKEN in p for p in stub.state["paths"]), (
        "premise: the token is in the request path"
    )
    text = cap.text
    assert "telegram.send_failed" in text
    assert TG_TOKEN not in text and TG_SECRET not in text and S not in text
    assert "[REDACTED" in text


async def test_t09_unregistered_token_is_still_caught_by_the_pattern_rules(
    stub: SimpleNamespace,
    production_logging: ConfigureFn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading import telegram

    cap = production_logging(json_output=True)  # nothing registered
    stub.state["status"] = 404
    stub.state["body"] = lambda req: f'{{"ok":false,"description":"Not Found: {req.path}"}}'
    monkeypatch.setattr(telegram, "logger", structlog.get_logger("trading.telegram"))
    notifier = telegram.TelegramNotifier(bot_token=TG_TOKEN, chat_id="-100123")
    notifier._url = f"{stub.base}/bot{TG_TOKEN}/sendMessage"
    try:
        await notifier.send_message("HALT")
    finally:
        await notifier.close()
    assert R.registered_secret_count() == 0
    assert "telegram.send_failed" in cap.text
    assert TG_SECRET not in cap.text and S not in cap.text
    assert f"bot{TG_TOKEN.split(':')[0]}:[REDACTED]" in cap.text
