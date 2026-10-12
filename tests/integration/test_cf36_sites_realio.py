"""
tests/integration/test_cf36_sites_realio.py
--------------------------------------
CF-36 cf36b: layer-2 site fixes against REAL ccxt exceptions and the real
live-engine stack (no external network).

    T-09 (layer 2)  ``CCXTMarketDataService`` against a local stub HTTP server:
                    ccxt builds its own exception text (URL, status, body) and the
                    service must log/raise NONE of it
    T-32            leak canary through ``StrategyEngine`` + ``LiveExecutionEngine``
                    + ``CCXTMarketDataService`` on ``FakeCCXTExchange``, including the
                    ``live.engine_start_failed`` -> ``engine.start_failed`` path
                    (synthesis-spec V-13) where layer 1 is the only control

Every site test runs with the cf36a processor DISABLED (``layer1_off``: proves the
site fix) and with the production chain (``layer1_on``).  All secrets are fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import ccxt.async_support as ccxt_async
import pytest
from aiohttp import web

from common.types import TimeFrame
from data.market_data import MarketDataError
from data.services.ccxt_market_data import CCXTMarketDataService
from tests._cf36_logging import SENTINEL as S
from tests._cf36_logging import Capture, ConfigureFn, disable_layer1, json_lines
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    build_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSignalStrategy

SIG = f"{S}SIG0123456789abcdef"
KEY = f"{S}APIKEY0123456789abcdef"
HOST = "api.cf36b-exchange.example"
LEAK_TEXT = (
    f"coinbase GET https://{HOST}/api/v3/order?timestamp=1&signature={SIG} "
    f'502 Bad Gateway {{"msg":"{S}-RESPONSE-BODY-0123"}}'
)
SYMBOL = "BTC/EUR"


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.fixture(params=["layer1_off", "layer1_on"])
def layer1(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _configure(production_logging: ConfigureFn, layer1: str, *, level: str = "INFO") -> Capture:
    cap = production_logging(json_output=True, log_level=level)
    if layer1 == "layer1_off":
        disable_layer1()
    return cap


def _events(cap: Capture, name: str) -> list[dict[str, Any]]:
    return [e for e in json_lines(cap.out) if e.get("event") == name]


# ---------------------------------------------------------------------------
# T-09 (layer 2): real ccxt exception text from a local stub server
# ---------------------------------------------------------------------------


@pytest.fixture
async def stub() -> AsyncIterator[SimpleNamespace]:
    state: dict[str, Any] = {"status": 503, "body": "", "paths": []}

    async def handler(request: web.Request) -> web.Response:
        state["paths"].append(request.path_qs)
        return web.Response(status=state["status"], text=state["body"])

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield SimpleNamespace(port=port, base=f"http://127.0.0.1:{port}", state=state)
    await runner.cleanup()


def _service_against(base: str) -> CCXTMarketDataService:
    service = CCXTMarketDataService(exchange_id="binance", cache_ttl_seconds=0)
    api = service._exchange.urls["api"]
    for name, value in list(api.items()):
        if isinstance(value, str):
            api[name] = f"{base}/api/v3"
    return service


@pytest.mark.parametrize("status", [503, 500])
async def test_t09_service_logs_and_raises_no_ccxt_text_for_a_real_http_error(
    status: int,
    stub: SimpleNamespace,
    production_logging: ConfigureFn,
    layer1: str,
) -> None:
    cap = _configure(production_logging, layer1)
    stub.state["status"] = status
    stub.state["body"] = f"upstream said https://up.example/x?signature={SIG}&api_key={KEY}"
    service = _service_against(stub.base)
    try:
        # premise: the raw ccxt exception really embeds the URL and the echoed body
        with pytest.raises(Exception) as raw:
            await service._exchange.fetch_ohlcv("BTC/USDT", "1h", None, 5)
        assert stub.base in str(raw.value) and SIG in str(raw.value)

        with pytest.raises(MarketDataError) as info:
            await service.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=5)
    finally:
        await service.close()
    err = info.value
    assert type(raw.value).__name__ in str(err), "the domain error names the class"
    assert err.__cause__ is None and err.__suppress_context__
    for fragment in (stub.base, SIG, KEY, S):
        assert fragment not in str(err)
    text = cap.text
    assert "exchange_connected" not in text  # non-vacuity: the service never connected
    for fragment in (stub.base, SIG, KEY, S):
        assert fragment not in text, "ccxt exception text reached the log sink"


async def test_t09_service_retry_log_for_a_real_request_timeout(
    stub: SimpleNamespace, production_logging: ConfigureFn, layer1: str
) -> None:
    """HTTP 504 -> ccxt ``RequestTimeout`` (a ``NetworkError``): the retry path
    (``network_error_retrying``, F-05) and the final domain error (F-09)."""
    cap = _configure(production_logging, layer1)
    stub.state["status"] = 504
    stub.state["body"] = f"gateway timeout https://up.example/x?signature={SIG}&api_key={KEY}"
    service = _service_against(stub.base)
    try:
        with pytest.raises(MarketDataError) as info:
            await service.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=5)
    finally:
        await service.close()
    retries = _events(cap, "network_error_retrying")
    assert len(retries) == 4, "one warning per attempt (non-vacuity)"
    assert all(e["error_type"] == "RequestTimeout" for e in retries)
    assert all("error" not in e for e in retries)
    assert str(info.value) == "Network error fetching 'BTC/USDT' (RequestTimeout)"
    for fragment in (stub.base, SIG, KEY, S):
        assert fragment not in cap.text and fragment not in str(info.value)


# ---------------------------------------------------------------------------
# T-32: leak canary through the real live stack
# ---------------------------------------------------------------------------


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("1000"))
    ex.seed_flat_bars(SYMBOL, count=100, price=Decimal("50000"), timeframe="1h")
    patch_exchange_factory(monkeypatch, ex)
    return ex


async def _stack(exchange: FakeCCXTExchange, run_id: str = "cf36b-canary") -> Any:
    strategy = ScriptedSignalStrategy(
        "cf36b-buy", {"direction": "buy", "call_index": 0, "target_notional": "100"}
    )
    return await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("1000"),
        run_id=run_id,
    )


def _fail(exc: Exception) -> Any:
    async def _raise(*_a: Any, **_k: Any) -> Any:
        raise exc

    return _raise


def _no_exception_text(text: str) -> None:
    for fragment in (SIG, HOST, S, "502 Bad Gateway"):
        assert fragment not in text, f"exception text reached the sink: {fragment!r}"


async def test_t32_canary_warmup_failure(
    exchange: FakeCCXTExchange, production_logging: ConfigureFn, layer1: str
) -> None:
    cap = _configure(production_logging, layer1)
    stack = await _stack(exchange)
    await stack.engine.start("cf36b-canary")
    exchange.fetch_ohlcv = _fail(ccxt_async.NetworkError(LEAK_TEXT))  # type: ignore[method-assign]

    await stack.engine._warmup_bar_windows()  # (e): must log type only and carry on

    (ev,) = _events(cap, "engine.warmup_fetch_failed")
    assert ev["error_type"] == "MarketDataError" and "exception" not in ev
    assert len(_events(cap, "network_error_retrying")) == 4
    _no_exception_text(cap.text)


async def test_t32_canary_per_bar_poll_failure(
    exchange: FakeCCXTExchange, production_logging: ConfigureFn, layer1: str
) -> None:
    cap = _configure(production_logging, layer1)
    stack = await _stack(exchange)
    await start_and_warmup(stack, "cf36b-canary")
    exchange.fetch_ohlcv = _fail(ccxt_async.ExchangeNotAvailable(LEAK_TEXT))  # type: ignore[method-assign]

    for _ in range(3):  # three consecutive bars with the exchange down
        await stack.engine._poll_and_process()

    evs = _events(cap, "engine.poll_fetch_failed")
    assert len(evs) == 3 and all(e["error_type"] == "MarketDataError" for e in evs)
    assert not any("exception" in e for e in evs)
    _no_exception_text(cap.text)


async def test_t32_canary_ticker_failure_inside_process_signal(
    exchange: FakeCCXTExchange, production_logging: ConfigureFn, layer1: str
) -> None:
    cap = _configure(production_logging, layer1)
    stack = await _stack(exchange)
    await start_and_warmup(stack, "cf36b-canary")
    exchange.queue_ticker_error(SYMBOL, ccxt_async.NetworkError(LEAK_TEXT), times=50)

    await step_bar(stack, SYMBOL, Decimal("50000"))

    (ev,) = _events(cap, "live.ticker_fetch_failed")
    assert ev["error_type"] == "NetworkError" and "error" not in ev
    assert _events(cap, "ccxt_retry.exhausted")
    assert exchange.order_log == []  # trading behaviour unchanged: no order
    _no_exception_text(cap.text)


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
async def test_t32_canary_engine_start_failed_catch_all_is_layer1_only(
    exchange: FakeCCXTExchange, production_logging: ConfigureFn, level: str
) -> None:
    """V-13: the live engine's raw ccxt exception re-raises out of
    ``live.engine_start_failed`` into ``engine.start_failed`` (a ``.exception``
    catch-all).  A site fix cannot help there; layer 1 must redact the credential
    (ARB-CF36-06), and the type-only ``live.engine_start_failed`` line is clean."""
    cap = _configure(production_logging, "layer1_on", level=level)
    stack = await _stack(exchange)

    async def failing_load(reload: bool = False) -> dict[str, Any]:
        raise ccxt_async.ExchangeNotAvailable(LEAK_TEXT)

    exchange.load_markets = failing_load  # type: ignore[method-assign]

    with pytest.raises(ccxt_async.ExchangeNotAvailable):
        await stack.engine.start("cf36b-canary")

    (site_ev,) = _events(cap, "live.engine_start_failed")
    assert site_ev["error_type"] == "ExchangeNotAvailable" and "error" not in site_ev
    catch_all = _events(cap, "engine.start_failed")
    assert catch_all, "the catch-all event must exist (non-vacuity)"
    assert "ExchangeNotAvailable" in catch_all[0]["exception"]
    assert "[REDACTED" in cap.text
    assert SIG not in cap.text


@pytest.mark.parametrize("layer1", ["layer1_off", "layer1_on"])
async def test_t32_canary_market_data_connect_failure_is_clean_even_without_layer1(
    exchange: FakeCCXTExchange, production_logging: ConfigureFn, layer1: str
) -> None:
    """When the market-data connect fails first, F-09 (fixed text, ``from None``)
    keeps the URL out of the ``engine.start_failed`` traceback on its own."""
    cap = _configure(production_logging, layer1)
    stack = await _stack(exchange)
    calls = {"n": 0}
    real_load = exchange.load_markets

    async def second_call_fails(reload: bool = False) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise ccxt_async.ExchangeNotAvailable(LEAK_TEXT)
        return await real_load(reload)

    exchange.load_markets = second_call_fails  # type: ignore[method-assign]

    with pytest.raises(MarketDataError):
        await stack.engine.start("cf36b-canary")

    (ev,) = _events(cap, "engine.start_failed")
    assert "MarketDataError" in ev["exception"] and "(ExchangeNotAvailable)" in ev["exception"]
    _no_exception_text(cap.text)
