"""
tests/unit/test_cf36_sites.py
------------------------------
CF-36 cf36b: the per-site log fixes (layer 2).  synthesis-spec section 6 / 8.

Every site test runs TWICE through the production logging chain
(``tests/_cf36_logging.production_logging``):

``layer1_off``
    ``redact_event_dict`` is removed and the renderer is unwrapped.  Whatever
    the site logs reaches the sink verbatim, so a pass proves the SITE FIX on
    its own, not "the processor caught it" (the red-before-green requirement).
``layer1_on``
    The full production chain.  Proves nothing regressed end to end.

Exception text used by every test embeds a signed URL, a host and a response
body (``LEAK_TEXT``).  A site passes only if NONE of it reaches the output.
Each leak assertion is paired with a presence assertion (event name and the
type field), so a vacuous capture cannot pass (RK-12).

Test ids follow synthesis-spec section 8: T-21 (F-01/F-02), T-22 (F-03/F-04),
T-23 (F-11), T-24 (F-05/F-09), T-25 ((e), F-07), T-26 (F-06), T-27 (F-08, F-12,
F-13, F-15).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import ccxt.async_support as ccxt_async
import pytest
import structlog

from common.types import OrderSide, SignalDirection, TimeFrame
from data.market_data import DataNotAvailableError, MarketDataError, RateLimitError
from tests._cf36_logging import SENTINEL as S
from tests._cf36_logging import Capture, ConfigureFn, disable_layer1, json_lines
from tests.unit import test_ccxt_market_data as _md
from tests.unit import test_live_execution as _live
from tests.unit import test_trailing_stop as _ts
from trading.ccxt_errors import translate_ccxt_error
from trading.ccxt_retry import ccxt_retry

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SIG = f"{S}SIG0123456789abcdef"
HOST = "api.cf36b-exchange.example"
URL = f"https://{HOST}/api/v3/order?symbol=BTCUSDT&timestamp=1&signature={SIG}"
BODY = f"{S}-RESPONSE-BODY-0123"
LEAK_TEXT = f'binance GET {URL} 502 Bad Gateway {{"msg":"{BODY}"}}'
#: anything of LEAK_TEXT that must never reach a log sink from a fixed site
LEAK_MARKERS = (SIG, HOST, BODY, S, "502 Bad Gateway", "binance GET")
_ERROR_AT = re.compile(r"^[\w.\-]+\.py:\d+:[\w<>]+$")

_SYMBOL = "BTC/USDT"


def leaked(text: str) -> list[str]:
    return [m for m in LEAK_MARKERS if m in text]


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant)


class SiteLog:
    """``configure(level=..., json_output=...) -> Capture`` plus the active layer-1 mode."""

    def __init__(self, mode: str, production_logging: ConfigureFn, mp: pytest.MonkeyPatch) -> None:
        self.mode = mode
        self._production_logging = production_logging
        self._mp = mp

    @property
    def layer1_on(self) -> bool:
        return self.mode == "layer1_on"

    def __call__(self, *, level: str = "INFO", json_output: bool = True) -> Capture:
        cap = self._production_logging(json_output=json_output, log_level=level)
        if not self.layer1_on:
            disable_layer1()
        # module-level loggers may be cached against an older configuration
        import api.routers.runs as runs_mod
        import api.services.run_recovery as recovery_mod
        import data.services.ccxt_market_data as md_mod
        import trading.ccxt_retry as retry_mod
        import trading.engines.live as live_mod
        import trading.strategy_engine as se_mod

        for mod in (runs_mod, recovery_mod, md_mod, retry_mod, live_mod, se_mod):
            self._mp.setattr(mod, "logger", structlog.get_logger("cf36b.site"))
        return cap


@pytest.fixture(params=["layer1_off", "layer1_on"])
def site_log(
    request: pytest.FixtureRequest,
    production_logging: ConfigureFn,
    monkeypatch: pytest.MonkeyPatch,
) -> SiteLog:
    return SiteLog(request.param, production_logging, monkeypatch)


def events(cap: Capture, name: str) -> list[dict[str, Any]]:
    return [e for e in json_lines(cap.out) if e.get("event") == name]


def assert_clean(cap: Capture) -> None:
    text = cap.text
    assert not leaked(text), f"exception text reached the sink: {leaked(text)}"


def assert_type_only(
    cap: Capture, name: str, error_type: str, *, floor: int = 1, status: int | None = None
) -> list[dict[str, Any]]:
    """Presence + shape of a fixed site event; absence of any exception text."""
    evs = events(cap, name)
    assert len(evs) >= floor, f"event {name!r} not emitted (vacuous test); got {cap.out[:500]}"
    assert_clean(cap)  # first: a failing run then names exactly what leaked
    for e in evs:
        assert e["error_type"] == error_type
        assert "error" not in e and "exception" not in e and "exc_info" not in e
        if status is not None:
            assert e["http_status"] == status
        assert _ERROR_AT.match(e["error_at"]), e.get("error_at")
    return evs


# ---------------------------------------------------------------------------
# helper unit tests
# ---------------------------------------------------------------------------


def test_bounded_scalar_contract() -> None:
    from common.logging import bounded_scalar

    assert bounded_scalar(7) == 7 and bounded_scalar(1.5) == 1.5
    assert bounded_scalar(True) == "bool"  # a bool is not a number here
    assert bounded_scalar(None) is None
    assert bounded_scalar(Decimal("1.25")) == "1.25"
    assert bounded_scalar("x" * 100) == "x" * 32
    assert len(str(bounded_scalar("y" * 100))) == 32
    assert bounded_scalar(Decimal("1." + "1" * 80)) == ("1." + "1" * 80)[:32]
    assert bounded_scalar({"info": URL}) == "dict"
    assert bounded_scalar([URL]) == "list"
    assert bounded_scalar(max_len=4, v="abcdefgh") == "abcd"


def test_bounded_scalar_never_returns_a_container_or_raises() -> None:
    from common.logging import bounded_scalar

    class Boom:
        def __str__(self) -> str:
            raise RuntimeError(LEAK_TEXT)

    assert bounded_scalar(Boom()) == "Boom"
    assert bounded_scalar(b"bytes") == "bytes"


def test_f02_translate_ccxt_error_fallback_has_the_type_only() -> None:
    msg = translate_ccxt_error(RuntimeError(LEAK_TEXT))
    assert msg == "Unexpected exchange error (RuntimeError)"
    assert not leaked(msg)
    # mapped classes keep their fixed texts
    assert translate_ccxt_error(ccxt_async.RateLimitExceeded(LEAK_TEXT)).startswith("Rate limited")
    assert translate_ccxt_error(ccxt_async.ExchangeError(LEAK_TEXT)).startswith("Exchange error")


def _all_ccxt_error_classes() -> list[type[Exception]]:
    return [
        c
        for c in vars(ccxt_async).values()
        if isinstance(c, type) and issubclass(c, Exception) and c.__module__ == "ccxt.base.errors"
    ]


def test_f02_translate_ccxt_error_is_fixed_text_for_every_ccxt_class() -> None:
    classes = _all_ccxt_error_classes()
    assert len(classes) >= 40, "sweep floor (vacuity guard)"
    for cls in classes:
        message = translate_ccxt_error(cls(LEAK_TEXT))
        assert message and not leaked(message), cls.__name__
    for exc in (ValueError(LEAK_TEXT), KeyError(LEAK_TEXT), OSError(LEAK_TEXT)):
        assert not leaked(translate_ccxt_error(exc))


# ---------------------------------------------------------------------------
# T-21  F-01 / F-02: ccxt_retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc_cls", "user_message_prefix"),
    [
        (ccxt_async.NetworkError, "Network error"),
        (ccxt_async.ExchangeError, "Exchange error"),
        (RuntimeError, "Unexpected exchange error (RuntimeError)"),
    ],
)
@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
async def test_t21_ccxt_retry_logs_type_only(
    site_log: SiteLog,
    exc_cls: type[Exception],
    user_message_prefix: str,
    json_output: bool,
) -> None:
    cap = site_log(json_output=json_output)

    async def fn() -> None:
        raise exc_cls(LEAK_TEXT)

    with pytest.raises(exc_cls):
        await ccxt_retry(
            fn, max_retries=1, base_delay=0.0, max_delay=0.0, operation="fetch_ticker(BTC/USDT)"
        )
    if not json_output:
        text = cap.text
        assert "ccxt_retry.retrying" in text and "ccxt_retry.exhausted" in text
        assert exc_cls.__name__ in text
        assert not leaked(text), f"exception text reached the sink: {leaked(text)}"
        return
    for name in ("ccxt_retry.retrying", "ccxt_retry.exhausted"):
        (ev,) = assert_type_only(cap, name, exc_cls.__name__)
        assert ev["user_message"].startswith(user_message_prefix)
        assert ev["operation"] == "fetch_ticker(BTC/USDT)"


# ---------------------------------------------------------------------------
# T-22  F-03 / F-04: live.ticker_fetch_failed, live.invalid_ticker_price
# ---------------------------------------------------------------------------


async def test_t22_ticker_fetch_failed_logs_type_only(site_log: SiteLog) -> None:
    cap = site_log()
    engine, _, ex = _live._make_engine()
    ex.fetch_ticker.side_effect = ccxt_async.NetworkError(LEAK_TEXT)

    orders = await engine.process_signal(_live._make_signal(direction=SignalDirection.BUY))

    assert orders == []  # trading behaviour unchanged
    (ev,) = assert_type_only(cap, "live.ticker_fetch_failed", "NetworkError")
    assert ev["symbol"] == _SYMBOL and ev["user_message"].startswith("Network error")
    # the three retry attempts of ccxt_retry are covered by the same sweep
    assert_type_only(cap, "ccxt_retry.exhausted", "NetworkError")


async def test_t22_invalid_ticker_price_logs_bounded_scalars_not_the_dict(
    site_log: SiteLog,
) -> None:
    cap = site_log()
    engine, _, _ex = _live._make_engine(
        exchange_kwargs={
            "fetch_ticker_response": {
                "last": 0,
                "bid": 0.0,
                "ask": None,
                "info": {"note": f"{S}-INFO-LEAK", "url": URL},
            }
        }
    )

    orders = await engine.process_signal(_live._make_signal(direction=SignalDirection.BUY))

    assert orders == []
    (ev,) = events(cap, "live.invalid_ticker_price")
    assert_clean(cap)
    assert "ticker" not in ev
    assert (ev["last"], ev["bid"], ev["ask"], ev["has_info"]) == (0, 0.0, None, True)
    assert ev["symbol"] == _SYMBOL


async def test_t22_invalid_ticker_price_bounds_hostile_scalar_values(site_log: SiteLog) -> None:
    cap = site_log()
    engine, _, _ex = _live._make_engine(
        exchange_kwargs={"fetch_ticker_response": {"last": "0", "bid": {"x": URL}, "ask": "9" * 80}}
    )
    await engine.process_signal(_live._make_signal(direction=SignalDirection.BUY))
    (ev,) = events(cap, "live.invalid_ticker_price")
    assert_clean(cap)
    assert ev["bid"] == "dict" and ev["ask"] == "9" * 32 and ev["has_info"] is False


# ---------------------------------------------------------------------------
# T-23  F-11: eight live.py sites
# ---------------------------------------------------------------------------


async def _drive_balance(engine: Any, ex: Any) -> None:
    ex.fetch_balance.side_effect = ccxt_async.ExchangeError(LEAK_TEXT)
    assert await engine._fetch_balance_cached(fresh=True) is None


async def _drive_lookup(engine: Any, ex: Any) -> None:
    ex.fetch_orders.side_effect = ccxt_async.NetworkError(LEAK_TEXT)
    assert await engine._lookup_by_cid(_SYMBOL, "cid-1", 1_700_000_000_000, OrderSide.BUY) == (
        "failed",
        None,
    )


async def _drive_rejected(engine: Any, ex: Any) -> None:
    ex.create_order.side_effect = ccxt_async.InsufficientFunds(LEAK_TEXT)
    await engine.submit_order(_live._make_market_order())


async def _drive_ambiguous(engine: Any, ex: Any) -> None:
    ex.create_order.side_effect = ccxt_async.RequestTimeout(LEAK_TEXT)
    await engine.submit_order(_live._make_market_order())


async def _drive_cancel(engine: Any, ex: Any) -> None:
    submitted = await engine.submit_order(_live._make_market_order())
    ex.cancel_order.side_effect = ccxt_async.ExchangeError(LEAK_TEXT)
    await engine.cancel_order(submitted.order_id)


async def _drive_fills(engine: Any, ex: Any) -> None:
    submitted = await engine.submit_order(_live._make_market_order())
    ex.fetch_order_trades.side_effect = ccxt_async.NetworkError(LEAK_TEXT)
    await engine.get_fills(submitted.order_id)


async def _drive_reconcile(engine: Any, ex: Any) -> None:
    submitted = await engine.submit_order(_live._make_market_order())
    ex.fetch_order.side_effect = ccxt_async.ExchangeError(LEAK_TEXT)
    await engine._reconcile_order_ex(submitted)


async def _drive_start(engine: Any, ex: Any) -> None:
    ex.load_markets.side_effect = ccxt_async.ExchangeNotAvailable(LEAK_TEXT)
    stub = MagicMock(
        get_position=MagicMock(return_value=None),
        get_open_positions=MagicMock(return_value=[]),
        get_daily_pnl=MagicMock(return_value=Decimal("0")),
        cash=Decimal("0"),
        initial_cash=Decimal("0"),
        current_equity=Decimal("0"),
        get_peak_equity=MagicMock(return_value=Decimal("0")),
    )
    engine.attach_position_source(stub, symbols=[_SYMBOL])
    with pytest.raises(ccxt_async.ExchangeNotAvailable):
        await engine.on_start()


async def _drive_close(engine: Any, ex: Any) -> None:
    ex.close.side_effect = ccxt_async.NetworkError(LEAK_TEXT)
    await engine.on_stop()


_F11_SITES: list[tuple[str, Callable[..., Any], str]] = [
    ("live.balance_fetch_failed", _drive_balance, "ExchangeError"),
    ("live.submit_lookup_failed", _drive_lookup, "NetworkError"),
    ("live.order_submission_rejected", _drive_rejected, "InsufficientFunds"),
    ("live.order_submission_ambiguous", _drive_ambiguous, "RequestTimeout"),
    ("live.cancel_failed", _drive_cancel, "ExchangeError"),
    ("live.fetch_fills_failed", _drive_fills, "NetworkError"),
    ("live.reconciliation_failed", _drive_reconcile, "ExchangeError"),
    ("live.engine_start_failed", _drive_start, "ExchangeNotAvailable"),
    ("live.exchange_close_failed", _drive_close, "NetworkError"),
]


@pytest.mark.parametrize(
    ("event", "driver", "error_type"), _F11_SITES, ids=[s[0] for s in _F11_SITES]
)
async def test_t23_live_site_logs_type_only(
    site_log: SiteLog, event: str, driver: Callable[..., Any], error_type: str
) -> None:
    cap = site_log()
    engine, _, ex = _live._make_engine()
    await driver(engine, ex)
    assert_type_only(cap, event, error_type)


# ---------------------------------------------------------------------------
# T-24  F-05 / F-09: CCXTMarketDataService
# ---------------------------------------------------------------------------

_TF = TimeFrame.ONE_HOUR


async def _fetch_service(exc: Exception) -> tuple[Any, Any]:
    service, mock_ex = _md._make_service(cache_ttl=0)
    await service.connect()
    mock_ex.fetch_ohlcv = AsyncMock(side_effect=exc)
    return service, mock_ex


_FETCH_CASES: list[tuple[Exception, type[Exception], str]] = [
    (
        ccxt_async.NetworkError(LEAK_TEXT),
        MarketDataError,
        "Network error fetching 'BTC/USDT' (NetworkError)",
    ),
    (
        ccxt_async.ExchangeNotAvailable(LEAK_TEXT),
        MarketDataError,
        "Exchange 'binance' is currently unavailable (ExchangeNotAvailable)",
    ),
    (
        ccxt_async.BadSymbol(LEAK_TEXT),
        DataNotAvailableError,
        "Symbol 'BTC/USDT' is not available on exchange 'binance' (BadSymbol)",
    ),
    (
        ccxt_async.BadRequest(LEAK_TEXT),
        DataNotAvailableError,
        "Bad request for 'BTC/USDT' / 1h (BadRequest)",
    ),
    (
        ccxt_async.ExchangeError(LEAK_TEXT),
        MarketDataError,
        "Exchange error fetching OHLCV for 'BTC/USDT' (ExchangeError)",
    ),
    (
        ccxt_async.RateLimitExceeded(LEAK_TEXT),
        RateLimitError,
        "Rate limit exceeded for 'BTC/USDT' after 3 retries",
    ),
]


@pytest.mark.parametrize(
    ("exc", "domain_cls", "message"), _FETCH_CASES, ids=[type(c[0]).__name__ for c in _FETCH_CASES]
)
async def test_t24_fetch_raises_fixed_text_from_none(
    site_log: SiteLog, exc: Exception, domain_cls: type[Exception], message: str
) -> None:
    cap = site_log()
    service, _ = await _fetch_service(exc)

    with pytest.raises(domain_cls) as info:
        await service.fetch_ohlcv(_SYMBOL, _TF, limit=10)

    err = info.value
    assert str(err) == message
    assert not leaked(str(err))
    assert err.__cause__ is None and err.__suppress_context__ is True  # `from None`
    assert not leaked(cap.text)


async def test_t24_network_error_retrying_logs_type_only(site_log: SiteLog) -> None:
    cap = site_log()
    service, _ = await _fetch_service(ccxt_async.NetworkError(LEAK_TEXT))
    with pytest.raises(MarketDataError):
        await service.fetch_ohlcv(_SYMBOL, _TF, limit=10)
    evs = assert_type_only(cap, "network_error_retrying", "NetworkError", floor=3)
    assert all(e["symbol"] == _SYMBOL and "wait_seconds" in e for e in evs)


_CONNECT_CASES: list[tuple[Exception, str]] = [
    (
        ccxt_async.AuthenticationError(LEAK_TEXT),
        "Authentication failed for exchange 'binance' (AuthenticationError)",
    ),
    (
        ccxt_async.NetworkError(LEAK_TEXT),
        "Network error loading markets for 'binance' (NetworkError)",
    ),
    (
        ccxt_async.ExchangeError(LEAK_TEXT),
        "Exchange error loading markets for 'binance' (ExchangeError)",
    ),
]


@pytest.mark.parametrize(
    ("exc", "message"), _CONNECT_CASES, ids=[type(c[0]).__name__ for c in _CONNECT_CASES]
)
async def test_t24_connect_raises_fixed_text_from_none(
    site_log: SiteLog, exc: Exception, message: str
) -> None:
    cap = site_log()
    service, mock_ex = _md._make_service()
    mock_ex.load_markets = AsyncMock(side_effect=exc)

    with pytest.raises(MarketDataError) as info:
        await service.connect()

    assert str(info.value) == message
    assert info.value.__cause__ is None and info.value.__suppress_context__ is True
    assert "connecting_to_exchange" in cap.text  # non-vacuity
    assert not leaked(cap.text)


async def test_t24_exchange_close_error_logs_type_only(site_log: SiteLog) -> None:
    cap = site_log()
    service, mock_ex = _md._make_service()
    mock_ex.close = AsyncMock(side_effect=ccxt_async.NetworkError(LEAK_TEXT))
    await service.close()
    assert_type_only(cap, "exchange_close_error", "NetworkError")


async def test_t24_raise_trace_is_absent_at_info(
    site_log: SiteLog,
) -> None:
    """At INFO the ``market_data.raise.trace`` event does not exist (level filter)."""
    cap = site_log(level="INFO")
    service, _ = await _fetch_service(ccxt_async.ExchangeError(LEAK_TEXT))
    with pytest.raises(MarketDataError):
        await service.fetch_ohlcv(_SYMBOL, _TF, limit=10)
    assert "market_data.raise.trace" not in cap.text
    assert not leaked(cap.text)


@pytest.mark.parametrize(
    ("op", "exc_cls"),
    [
        ("fetch_ohlcv", ccxt_async.ExchangeError),
        ("fetch_ohlcv", ccxt_async.BadRequest),
        ("fetch_ohlcv", ccxt_async.BadSymbol),
        ("fetch_ohlcv", ccxt_async.ExchangeNotAvailable),
        ("fetch_ohlcv", ccxt_async.RateLimitExceeded),
        ("fetch_ohlcv", ccxt_async.NetworkError),
        ("connect", ccxt_async.AuthenticationError),
        ("connect", ccxt_async.NetworkError),
        ("connect", ccxt_async.ExchangeError),
    ],
)
async def test_t24_trace_event_at_debug_carries_a_redacted_traceback(
    production_logging: ConfigureFn,
    monkeypatch: pytest.MonkeyPatch,
    op: str,
    exc_cls: type[Exception],
) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    import data.services.ccxt_market_data as md_mod

    monkeypatch.setattr(md_mod, "logger", structlog.get_logger("cf36b.site"))
    if op == "connect":
        service, mock_ex = _md._make_service()
        mock_ex.load_markets = AsyncMock(side_effect=exc_cls(LEAK_TEXT))
        with pytest.raises(MarketDataError):
            await service.connect()
    else:
        service, _ = await _fetch_service(exc_cls(LEAK_TEXT))
        with pytest.raises((MarketDataError, RateLimitError, DataNotAvailableError)):
            await service.fetch_ohlcv(_SYMBOL, _TF, limit=10)
    evs = events(cap, "market_data.raise.trace")
    assert len(evs) == 1, "the debug .trace event must exist exactly once (diagnosability)"
    assert evs[0]["op"] == op
    assert (evs[0].get("symbol") == _SYMBOL) is (op == "fetch_ohlcv")
    assert exc_cls.__name__ in evs[0]["exception"]
    assert "[REDACTED" in cap.text
    assert SIG not in cap.text  # the DEBUG traceback keeps non-credential text


# ---------------------------------------------------------------------------
# T-25  (e) F-07: engine.warmup_fetch_failed / warmup_unexpected_error
# ---------------------------------------------------------------------------


async def _started_engine(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    engine, mocks = _ts._make_engine(**kwargs)
    await engine.start("run-001")
    mocks["execution"].check_resting_orders = None
    mocks["strategy"].on_bar = MagicMock(return_value=[])
    return engine, mocks


@pytest.mark.parametrize(
    ("event", "exc"),
    [
        ("engine.warmup_fetch_failed", MarketDataError(LEAK_TEXT)),
        ("engine.warmup_unexpected_error", RuntimeError(LEAK_TEXT)),
    ],
)
@pytest.mark.parametrize("json_output", [True, False], ids=["json", "console"])
async def test_t25_warmup_failure_logs_type_only(
    site_log: SiteLog, event: str, exc: Exception, json_output: bool
) -> None:
    cap = site_log(json_output=json_output)
    engine, mocks = await _started_engine()
    mocks["market_data"].fetch_ohlcv = AsyncMock(side_effect=exc)

    await engine._warmup_bar_windows()  # must not raise

    if json_output:
        (ev,) = assert_type_only(cap, event, type(exc).__name__)
        assert ev["symbol"] == "BTC/USD"
    else:
        assert event in cap.text and type(exc).__name__ in cap.text
        assert "Traceback" not in cap.text
        assert not leaked(cap.text)


@pytest.mark.parametrize(
    ("event", "exc"),
    [
        ("engine.warmup_fetch_failed", MarketDataError(LEAK_TEXT)),
        ("engine.warmup_unexpected_error", RuntimeError(LEAK_TEXT)),
    ],
)
async def test_t25_warmup_trace_at_debug_is_redacted_by_layer1(
    production_logging: ConfigureFn, event: str, exc: Exception
) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    engine, mocks = await _started_engine()
    mocks["market_data"].fetch_ohlcv = AsyncMock(side_effect=exc)
    await engine._warmup_bar_windows()
    (ev,) = events(cap, f"{event}.trace")
    assert type(exc).__name__ in ev["exception"] and "Traceback" in ev["exception"]
    assert "[REDACTED" in cap.text
    assert SIG not in cap.text


async def test_t25_warmup_trace_is_absent_at_info(site_log: SiteLog) -> None:
    cap = site_log(level="INFO")
    engine, mocks = await _started_engine()
    mocks["market_data"].fetch_ohlcv = AsyncMock(side_effect=MarketDataError(LEAK_TEXT))
    await engine._warmup_bar_windows()
    assert ".trace" not in cap.text
    assert "engine.warmup_fetch_failed" in cap.text
    assert "Traceback" not in cap.text
    assert_clean(cap)


# ---------------------------------------------------------------------------
# T-26  F-06: engine.bracket_exit_error / engine.trailing_stop_error
# ---------------------------------------------------------------------------


async def test_t26_trailing_stop_error_logs_type_only(site_log: SiteLog) -> None:
    cap = site_log()
    engine, mocks = await _started_engine(trailing_stop_pct=0.03)
    engine._trailing_stop.check = MagicMock(side_effect=RuntimeError(LEAK_TEXT))
    mocks["portfolio"].get_position = MagicMock(return_value=_ts._make_position())

    bar = _ts._make_bar(symbol="BTC/USD", close="50000")
    await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})  # must not raise

    assert engine.bar_count == 1
    (ev,) = assert_type_only(cap, "engine.trailing_stop_error", "RuntimeError")
    assert ev["symbol"] == "BTC/USD"
    assert ev["error_at"].endswith(":_execute_mock_call") or _ERROR_AT.match(ev["error_at"])


async def test_t26_bracket_exit_error_logs_type_only(site_log: SiteLog) -> None:
    cap = site_log()
    engine, mocks = await _started_engine()
    bracket = MagicMock()
    bracket.requires_atr = False
    bracket.check = MagicMock(side_effect=RuntimeError(LEAK_TEXT))
    engine._bracket_exit = bracket
    mocks["portfolio"].get_position = MagicMock(return_value=_ts._make_position())

    bar = _ts._make_bar(symbol="BTC/USD", close="50000")
    await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

    assert engine.bar_count == 1
    (ev,) = assert_type_only(cap, "engine.bracket_exit_error", "RuntimeError")
    assert ev["symbol"] == "BTC/USD"


@pytest.mark.parametrize("which", ["bracket", "trailing"])
async def test_t26_trace_event_only_at_debug(production_logging: ConfigureFn, which: str) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    engine, mocks = await _started_engine(trailing_stop_pct=0.03 if which == "trailing" else None)
    if which == "trailing":
        engine._trailing_stop.check = MagicMock(side_effect=RuntimeError(LEAK_TEXT))
    else:
        bracket = MagicMock(requires_atr=False)
        bracket.check = MagicMock(side_effect=RuntimeError(LEAK_TEXT))
        engine._bracket_exit = bracket
    mocks["portfolio"].get_position = MagicMock(return_value=_ts._make_position())
    bar = _ts._make_bar(symbol="BTC/USD", close="50000")
    await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})
    name = f"engine.{which}_{'exit' if which == 'bracket' else 'stop'}_error"
    (ev,) = events(cap, f"{name}.trace")
    assert "RuntimeError" in ev["exception"]
    assert "[REDACTED" in cap.text
    assert SIG not in cap.text  # the DEBUG traceback keeps non-credential text


# ---------------------------------------------------------------------------
# T-27  F-08 (per-bar poll), F-12 (_log_exc), F-13 (runs.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "exc"),
    [
        ("engine.poll_fetch_failed", MarketDataError(LEAK_TEXT)),
        ("engine.poll_unexpected_error", RuntimeError(LEAK_TEXT)),
    ],
)
async def test_t27_poll_failure_logs_type_only_every_bar(
    site_log: SiteLog, event: str, exc: Exception
) -> None:
    cap = site_log()
    engine, mocks = await _started_engine()
    bar = _ts._make_bar(symbol="BTC/USD", close="50000")
    engine._bar_windows["BTC/USD"] = [bar]
    mocks["market_data"].get_latest_bar = AsyncMock(side_effect=exc)

    for _ in range(3):  # the per-bar path: every poll fails the same way
        await engine._poll_and_process()  # must not raise; falls back to window[-1]

    evs = assert_type_only(cap, event, type(exc).__name__, floor=3)
    assert all(e["symbol"] == "BTC/USD" for e in evs)
    assert "Traceback" not in cap.text


@pytest.mark.parametrize(
    ("event", "exc"),
    [
        ("engine.poll_fetch_failed", MarketDataError(LEAK_TEXT)),
        ("engine.poll_unexpected_error", RuntimeError(LEAK_TEXT)),
    ],
)
async def test_t27_poll_trace_only_at_debug(
    production_logging: ConfigureFn, event: str, exc: Exception
) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    engine, mocks = await _started_engine()
    mocks["market_data"].get_latest_bar = AsyncMock(side_effect=exc)
    await engine._poll_and_process()
    (ev,) = events(cap, f"{event}.trace")
    assert "Traceback" in ev["exception"] and type(exc).__name__ in ev["exception"]
    assert "[REDACTED" in cap.text
    assert SIG not in cap.text  # the DEBUG traceback keeps non-credential text


async def test_t27_log_exc_helper_logs_type_status_and_location_only(site_log: SiteLog) -> None:
    from api.services.run_recovery import _log_exc

    cap = site_log()
    log = structlog.get_logger("cf36b.recovery")

    def _raise() -> None:
        raise ccxt_async.ExchangeError(LEAK_TEXT)

    try:
        _raise()
    except ccxt_async.ExchangeError as exc:
        _log_exc(log, "recovery.scan_fetch_orders_failed", exc, symbol=_SYMBOL)

    (ev,) = assert_type_only(cap, "recovery.scan_fetch_orders_failed", "ExchangeError")
    assert ev["symbol"] == _SYMBOL and ev["error_at"].endswith(":_raise")


async def test_t27_runs_backtest_fetch_errors_log_type_only(
    site_log: SiteLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import api.config as api_config
    import data.services.ccxt_market_data as md_mod
    from api.routers import runs

    cap = site_log()
    monkeypatch.setattr(
        api_config,
        "get_settings",
        lambda: SimpleNamespace(
            exchange_api_key=None,
            exchange_api_secret=None,
            exchange_api_passphrase=None,
            exchange_id="binance",
        ),
    )

    class _Svc:
        def __init__(self, **_: Any) -> None:
            pass

        async def connect(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def fetch_ohlcv_range(self, **_: Any) -> list[Any]:
            raise self.exc

    cases = [
        ("runs.backtest_data_not_available", DataNotAvailableError(LEAK_TEXT), 400),
        ("runs.backtest_market_data_error", MarketDataError(LEAK_TEXT), 502),
    ]
    for event, exc, code in cases:
        _Svc.exc = exc  # type: ignore[attr-defined]
        monkeypatch.setattr(md_mod, "CCXTMarketDataService", _Svc)
        from datetime import UTC, datetime

        with pytest.raises(HTTPException) as info:
            await runs._fetch_bars_for_backtest(
                [_SYMBOL],
                _TF,
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                structlog.get_logger("cf36b.runs"),
            )
        assert info.value.status_code == code
        name_events = events(cap, event)
        assert len(name_events) == 1, f"{event} not emitted (vacuous test)"
        assert name_events[0]["error_type"] == type(exc).__name__
        assert "error" not in name_events[0]
    # the log lines carry no exception text; the HTTP `detail` is a separate sink (CF-37)
    assert_clean(cap)


async def test_t27_backtest_502_detail_has_no_exchange_text_end_to_end(
    site_log: SiteLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-09 + F-13: the real service wraps a leaky ccxt error; neither the log line nor the
    client-facing ``detail`` carries any of it."""
    from datetime import UTC, datetime

    from fastapi import HTTPException

    import api.config as api_config
    import data.services.ccxt_market_data as md_mod
    from api.routers import runs

    cap = site_log()
    monkeypatch.setattr(
        api_config,
        "get_settings",
        lambda: SimpleNamespace(
            exchange_api_key=None,
            exchange_api_secret=None,
            exchange_api_passphrase=None,
            exchange_id="binance",
        ),
    )
    real_cls = md_mod.CCXTMarketDataService

    class _Service(real_cls):  # type: ignore[valid-type, misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._exchange = MagicMock()
            self._exchange.load_markets = AsyncMock(side_effect=ccxt_async.NetworkError(LEAK_TEXT))
            self._exchange.close = AsyncMock(return_value=None)

    monkeypatch.setattr(md_mod, "CCXTMarketDataService", _Service)
    with pytest.raises(HTTPException) as info:
        await runs._fetch_bars_for_backtest(
            [_SYMBOL],
            _TF,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            structlog.get_logger("cf36b.runs"),
        )
    assert info.value.status_code == 502
    assert info.value.detail == (
        "Exchange error fetching historical data: "
        "Network error loading markets for 'binance' (NetworkError)"
    )
    assert not leaked(str(info.value.detail))
    (ev,) = events(cap, "runs.backtest_market_data_error")
    assert ev["error_type"] == "MarketDataError"
    assert_clean(cap)


async def test_t29_recovery_scan_load_markets_failure_logs_type_only(site_log: SiteLog) -> None:
    """The ``recovery.scan_*`` family goes through ``_log_exc`` (F-12); drive a real caller."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from api.services.run_recovery import scan_and_import
    from trading.recovery import ResumeRejected

    cap = site_log()
    run = SimpleNamespace(id=uuid4(), config={"symbols": [_SYMBOL]}, started_at=datetime.now(UTC))
    exchange = SimpleNamespace(
        load_markets=AsyncMock(side_effect=ccxt_async.ExchangeNotAvailable(LEAK_TEXT))
    )

    with pytest.raises(ResumeRejected):
        await scan_and_import(MagicMock(), run, exchange, fence="fence-1")  # type: ignore[arg-type]

    (ev,) = assert_type_only(cap, "recovery.scan_load_markets_failed", "ExchangeNotAvailable")
    assert "error" not in ev
