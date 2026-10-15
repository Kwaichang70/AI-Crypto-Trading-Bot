"""
tests/unit/test_cf36_logging_chain.py
-------------------------------------
CF-36 layer 1 (cf36a), end to end through the PRODUCTION logging configuration
(``common.logging.configure_logging``), capturing real stdout (structlog) and
stderr (stdlib bridge).  ``capture_logs`` is deliberately NOT used here: it
replaces the processors and so cannot prove layer 1.

    T-07  ccxt exception sweep (>= 40 classes) x 4 URL styles x 5 log forms x 2 renderers
    T-08  aiohttp exception sweep (>= 30 classes) + 9 real constructions
    T-10  three-link chained traceback, exc_info forms, stack_info
    T-11  stdlib bridge: one valid line per record, no tuple repr, no warning
    T-12  uvicorn absorption (+ T-12a..d: --no-access-log guard, query-strip filter)
    T-01p/T-01n  credential-name parity matrix across the native and stdlib paths
    T-13  fail-safe through the real chain      T-14  renderer fail-safe
    T-17  chain invariants (native + bridge)
    T-35  stdlib format / handleError fail-safe
    plus: ccxt logger pinned to WARNING, exc_log_fields, hit/failure counters
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import logging.config
import warnings
from collections.abc import Callable
from typing import Any

import aiohttp
import ccxt
import pytest
import structlog
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import common.logging as clog
from common import redaction as R
from common.logging import exc_log_fields
from data.market_data import MarketDataError
from tests._cf36_logging import (
    NEGATIVE_NAMES,
    PARITY_NAMES,
    Capture,
    ConfigureFn,
    U,
    b64url,
    json_lines,
)
from tests._cf36_logging import SENTINEL as S
from trading.ccxt_errors import translate_ccxt_error

KEY = f"{S}KEY0123456789abcdef"
SIG = f"{S}SIG0123456789abcdef"
# JWT / Telegram shapes are assembled at runtime so that secret scanners do not match
# the source text; the runtime values are what the tests exercise.
_JWT_HEAD = b64url('{"alg":"ES256"}')
_JWT_SIG = b64url("signature-sentinel")
JWT = f"{_JWT_HEAD}.ey" + f"JzdWIiOiJ{S}In0.{_JWT_SIG}"
_TG_ID = "1234" + "56789"
TG = f"{_TG_ID}:AA{S}TelegramSecret_0123456789-xyz"
TG_SECRET = TG.split(":", 1)[1]

#: 4 realistic URL / header styles (design §6 T-07)
URL_STYLES: dict[str, str] = {
    "binance": f"binance GET https://api.binance.com/api/v3/account?timestamp=1&recvWindow=5000&signature={SIG} 400 Bad Request",  # noqa: E501
    "htx": (
        "huobi GET https://api.huobi.pro/v1/account/accounts?"
        f"AccessKeyId={KEY}&SignatureMethod=HmacSHA256&SignatureVersion=2&Timestamp=2026-01-01T00%3A00%3A00&Signature={SIG} 403"  # noqa: E501
    ),
    "coinbase": (
        "coinbase GET https://api.coinbase.com/api/v3/brokerage/products/BTC-USD 401 "
        f"{{'headers': {{'Authorization': 'Bearer {JWT}', 'CB-ACCESS-KEY': '{KEY}'}}}}"
    ),
    "api_key_url": f"GET https://api.stlouisfed.org/fred/series/observations?series_id=X&api_key={KEY}&file_type=json",
}

_errors = importlib.import_module("ccxt.base.errors")
CCXT_CLASSES = sorted(
    n for n, c in vars(_errors).items() if inspect.isclass(c) and issubclass(c, _errors.BaseError)
)
AIOHTTP_CLASSES = sorted(
    n
    for n in dir(aiohttp)
    if isinstance(getattr(aiohttp, n), type) and issubclass(getattr(aiohttp, n), Exception)
)

RENDERERS = [True, False]
RENDERER_IDS = ["json", "console"]


def _log_exception_forms(cap: Capture, tag: str, exc: Exception, secrets: tuple[str, ...]) -> None:
    """Log ``exc`` in the 4 leak-prone forms and assert redaction for each."""
    log = structlog.get_logger("cf36.sweep")

    log.error(f"{tag}.str", error=str(exc))
    cap.assert_redacted(f"{tag}.str", *secrets)

    log.error(f"{tag}.obj", error=exc)
    cap.assert_redacted(f"{tag}.obj", *secrets)

    try:
        raise exc
    except Exception:
        log.error(f"{tag}.exc_info", exc_info=True)
    cap.assert_redacted(f"{tag}.exc_info", *secrets)

    try:
        try:
            raise exc
        except Exception as inner:
            raise MarketDataError(f"market data failed: {inner}") from inner
    except MarketDataError:
        log.exception(f"{tag}.chain")
    cap.assert_redacted(f"{tag}.chain", *secrets)


# ---------------------------------------------------------------------------
# T-07  ccxt sweep
# ---------------------------------------------------------------------------


def test_t07_sweep_floor() -> None:
    assert len(CCXT_CLASSES) >= 40, "ccxt error class discovery is vacuous"
    assert len(URL_STYLES) == 4


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
@pytest.mark.parametrize("cls_name", CCXT_CLASSES)
def test_t07_ccxt_exceptions_never_leak_in_any_log_form(
    cls_name: str, json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    cls = getattr(_errors, cls_name)
    for style, text in URL_STYLES.items():
        _log_exception_forms(cap, f"t07.{cls_name}.{style}", cls(text), (KEY, SIG, JWT))
        # the 5th form: the user-facing message (fixed text, or the [:200] fallback)
        ev = f"t07.{cls_name}.{style}.user"
        structlog.get_logger("cf36.sweep").error(ev, user_message=translate_ccxt_error(cls(text)))
        cap.assert_present(ev)


def test_t07_the_premise_holds_the_unredacted_text_does_contain_the_secrets() -> None:
    """If ccxt stopped embedding URLs, the sweep above would be vacuous."""
    for style, text in URL_STYLES.items():
        exc = ccxt.ExchangeNotAvailable(text)
        assert S in str(exc) and S in repr(exc), style


# ---------------------------------------------------------------------------
# T-08  aiohttp sweep
# ---------------------------------------------------------------------------


def _request_info(url: str) -> aiohttp.RequestInfo:
    u = URL(url)
    return aiohttp.RequestInfo(u, "GET", CIMultiDictProxy(CIMultiDict()), u)


_KEY_URL = (
    f"https://api.stlouisfed.org/fred/series/observations?series_id=X&api_key={KEY}&file_type=json"
)
_TG_URL = f"https://api.telegram.org/bot{TG}/sendMessage"

_REAL: dict[str, Callable[[], Exception]] = {
    "InvalidURL": lambda: aiohttp.InvalidURL(_KEY_URL),
    "InvalidUrlClientError": lambda: aiohttp.InvalidUrlClientError(_TG_URL),
    "NonHttpUrlClientError": lambda: aiohttp.NonHttpUrlClientError(_KEY_URL),
    "InvalidUrlRedirectClientError": lambda: aiohttp.InvalidUrlRedirectClientError(_KEY_URL),
    "NonHttpUrlRedirectClientError": lambda: aiohttp.NonHttpUrlRedirectClientError(_TG_URL),
    "TooManyRedirects": lambda: aiohttp.TooManyRedirects(_request_info(_KEY_URL), ()),
    "ClientResponseError": lambda: aiohttp.ClientResponseError(
        _request_info(_KEY_URL), (), status=400, message="Bad Request"
    ),
    "ContentTypeError": lambda: aiohttp.ContentTypeError(_request_info(_TG_URL), ()),
    "WSServerHandshakeError": lambda: aiohttp.WSServerHandshakeError(_request_info(_KEY_URL), ()),
}


def _leaky(name: str) -> Exception:
    base: type[Exception] = getattr(aiohttp, name)
    url = _KEY_URL

    def _s(_self: Any) -> str:
        return url

    leaky = type(name, (base,), {"__str__": _s, "__repr__": _s})
    exc: Exception = base.__new__(leaky)
    exc.args = (url,)
    return exc


def test_t08_sweep_floor_and_premise() -> None:
    assert len(AIOHTTP_CLASSES) >= 30, "aiohttp exception discovery is vacuous"
    assert len(_REAL) >= 9
    for name, factory in _REAL.items():
        exc = factory()
        assert S in str(exc) or S in repr(exc), f"{name} no longer embeds the URL"


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
@pytest.mark.parametrize("name", sorted(_REAL))
def test_t08_real_aiohttp_exceptions_never_leak(
    name: str, json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    _log_exception_forms(cap, f"t08.real.{name}", _REAL[name](), (KEY, TG, TG_SECRET))


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
@pytest.mark.parametrize("name", AIOHTTP_CLASSES)
def test_t08_every_aiohttp_exception_class_never_leaks(
    name: str, json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    _log_exception_forms(cap, f"t08.leaky.{name}", _leaky(name), (KEY,))


# ---------------------------------------------------------------------------
# T-10  tracebacks
# ---------------------------------------------------------------------------


def _three_link() -> BaseException:
    try:
        try:
            try:
                raise ValueError(f"link1 api_key={S}ONE0123456789")
            except ValueError as e1:
                raise RuntimeError(f"link2 signature={S}TWO0123456789") from e1
        except RuntimeError:
            raise KeyError(f"link3 https://h/p?token={S}THREE0123456789")  # noqa: B904 -- implicit context
    except KeyError as e3:
        return e3
    raise AssertionError("unreachable")


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t10_chained_traceback_is_redacted_in_every_link(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    log = structlog.get_logger("cf36.tb")
    exc = _three_link()

    secrets = (f"{S}ONE", f"{S}TWO", f"{S}THREE")
    try:
        raise exc
    except KeyError:
        log.exception("t10.exception")
        cap.assert_redacted("t10.exception", *secrets)
        log.error("t10.exc_info_true", exc_info=True)
        cap.assert_redacted("t10.exc_info_true", *secrets)
    log.error("t10.exc_info_obj", exc_info=exc)
    cap.assert_redacted("t10.exc_info_obj", *secrets)

    def where() -> None:
        log.error("t10.stack", stack_info=True)  # api_key=CF36SENTINELSTACK0123456789

    where()
    cap.assert_redacted("t10.stack", f"{S}STACK")


def test_t10_traceback_structure_is_preserved_in_the_json_field(
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=True)
    exc = _three_link()
    try:
        raise exc
    except KeyError:
        structlog.get_logger("cf36.tb").exception("t10.structure")
    (record,) = [r for r in json_lines(cap.out) if r["event"] == "t10.structure"]
    text = record["exception"]
    assert isinstance(text, str)
    assert text.count("Traceback (most recent call last):") >= 3
    assert "The above exception was the direct cause of the following exception:" in text
    assert "During handling of the above exception, another exception occurred:" in text
    assert (
        'File "' in text and "ValueError" in text and "RuntimeError" in text and "KeyError" in text
    )
    assert text.count("[REDACTED") >= 3
    assert "exc_info" not in record


# ---------------------------------------------------------------------------
# T-11  stdlib bridge
# ---------------------------------------------------------------------------


def test_t11_foreign_records_are_one_valid_redacted_json_line_each(
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logging.getLogger("ccxt.base.exchange").debug(
            "%s %s, Request: %s %s",
            "GET",
            f"https://api.binance.com/x?signature={SIG}",
            {"X-MBX-APIKEY": KEY, "Authorization": f"Bearer {JWT}"},
            None,
        )
        try:
            raise RuntimeError(f"Task failed https://h/p?api_key={KEY}")
        except RuntimeError:
            logging.getLogger("asyncio").error("Task exception was never retrieved", exc_info=True)
        logging.getLogger("sqlalchemy.engine").warning(
            "connect failed postgresql+asyncpg://u:%s@db/x", f"{S}DbPw0123456789"
        )
        logging.getLogger("x").info("%s", f"https://h/p?api_key={KEY}")
        logging.getLogger("stdlib").warning(
            "plain message", extra={"order_id": "abc", "token": KEY}
        )
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    err = cap.err
    assert S not in err and S not in cap.out
    records = json_lines(err)  # raises if ANY line is not valid JSON
    by_logger = {r["logger"]: r for r in records}
    assert "ccxt.base.exchange" not in by_logger, (
        "ccxt DEBUG must be suppressed (pinned to WARNING)"
    )
    for name in ("asyncio", "sqlalchemy.engine", "x", "stdlib"):
        assert name in by_logger, f"{name} record missing (vacuous)"
    for r in records:
        assert "_record" not in r and "exc_info" not in r and "_from_structlog" not in r
        assert not any(
            isinstance(v, (list, tuple)) and v and isinstance(v[0], tuple) for v in r.values()
        )
    assert "((" not in err
    asyncio_rec = by_logger["asyncio"]
    assert isinstance(asyncio_rec["exception"], str) and "[REDACTED" in asyncio_rec["exception"]
    assert "[REDACTED" in by_logger["sqlalchemy.engine"]["event"]
    assert "[REDACTED" in by_logger["x"]["event"]
    assert by_logger["stdlib"]["order_id"] == "abc"
    extra_credential = by_logger["stdlib"]["token"]  # R-10 on a sensitive extra key
    assert extra_credential == "[REDACTED]"
    assert {"level", "timestamp", "logger", "event"} <= set(asyncio_rec)


def test_t11_console_mode_bridge_lines_are_redacted_text(production_logging: ConfigureFn) -> None:
    cap = production_logging(json_output=False)
    try:
        raise RuntimeError(f"boom https://h/p?signature={SIG}")
    except RuntimeError:
        logging.getLogger("stdlib").error("failed", exc_info=True)
    assert "failed" in cap.err and "[REDACTED" in cap.err
    assert S not in cap.text and "((" not in cap.err and "_record" not in cap.err


# ---------------------------------------------------------------------------
# T-12  uvicorn absorption
# ---------------------------------------------------------------------------


def test_t12_uvicorn_records_flow_through_the_redacting_bridge(
    production_logging: ConfigureFn,
) -> None:
    import uvicorn.config

    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
    assert logging.getLogger("uvicorn").handlers, "premise: uvicorn installs its own handlers"
    assert (
        logging.getLogger("uvicorn.access").handlers
        and not logging.getLogger("uvicorn.access").propagate
    )
    cap = production_logging(json_output=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        assert lg.handlers == [] and lg.propagate is True, name
    assert (
        sum(
            isinstance(f, clog._AccessQueryFilter)
            for f in logging.getLogger("uvicorn.access").filters
        )
        == 1
    )

    try:
        raise RuntimeError(f"ASGI boom https://h/p?signature={SIG}")
    except RuntimeError:
        logging.getLogger("uvicorn.error").error("Exception in ASGI application\n", exc_info=True)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5000",
        "GET",
        f"/api/v1/stream?api_key={KEY}",
        "1.1",
        200,
        extra={"color_message": f"\x1b[32mGET /api/v1/stream?api_key={KEY}\x1b[0m"},
    )
    assert S not in cap.text
    records = json_lines(cap.err)
    names = {r["logger"] for r in records}
    assert {"uvicorn.error", "uvicorn.access"} <= names
    access = next(r for r in records if r["logger"] == "uvicorn.access")
    assert "?[QUERY_REDACTED]" in access["event"]
    assert "color_message" not in access and all("color_message" not in r for r in records)
    err_rec = next(r for r in records if r["logger"] == "uvicorn.error")
    assert "[REDACTED" in err_rec["exception"]


def test_t12a_no_access_log_state_is_left_untouched(production_logging: ConfigureFn) -> None:
    """CF36A-SEC-01: absorbing ``uvicorn.access`` must not switch the access log back on."""
    import uvicorn

    uvicorn.Config("x:y", access_log=False)  # applies LOGGING_CONFIG + the no-access-log branch
    access = logging.getLogger("uvicorn.access")
    assert not access.handlers and access.propagate is False, "premise: uvicorn's off state"
    production_logging(json_output=True)
    assert access.hasHandlers() is False, "the protocols' per-connection check must stay False"
    assert access.propagate is False and access.handlers == []
    assert not any(isinstance(f, clog._AccessQueryFilter) for f in access.filters)
    error = logging.getLogger("uvicorn.error")
    assert error.propagate is True and error.handlers == []
    assert logging.getLogger("uvicorn").propagate is True


def test_t12b_access_log_on_installs_exactly_one_filter(production_logging: ConfigureFn) -> None:
    import uvicorn

    uvicorn.Config("x:y", access_log=True)
    production_logging(json_output=True)
    access = logging.getLogger("uvicorn.access")
    assert access.propagate is True and access.handlers == []
    clog._LOGGING_CONFIGURED = False  # configure a second time (idempotent install)
    clog.configure_logging(log_level="INFO", json_output=True, register_env_secrets=False)
    assert access.propagate is True and access.handlers == []
    assert sum(isinstance(f, clog._AccessQueryFilter) for f in access.filters) == 1


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t12c_access_record_query_is_stripped(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    import uvicorn

    uvicorn.Config("x:y", access_log=True)
    cap = production_logging(json_output=json_output)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5000",
        "GET",
        f"/api/v1/stream?api%5Fkey={KEY}&%61pi_key={KEY}&apiSecret={KEY}&auth={KEY}",
        "1.1",
        200,
    )
    assert "GET /api/v1/stream?[QUERY_REDACTED]" in cap.text  # presence ...
    assert KEY not in cap.out and KEY not in cap.err  # ... plus absence
    assert S not in cap.text


def test_t12d_query_filter_is_fail_safe() -> None:
    class Boom(tuple):  # type: ignore[type-arg]
        def __getitem__(self, _i: Any) -> Any:
            raise RuntimeError(S)

    flt = clog._AccessQueryFilter()

    def record(args: Any) -> logging.LogRecord:
        rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "msg %s", (), None)
        rec.args = args
        return rec

    for args in (None, {"a": 1}, ("a", "b"), ("a", "b", 3, "d", 5)):
        rec = record(args)
        assert flt.filter(rec) is True
        assert rec.args == args and rec.msg == "msg %s"
    plain = record(("c", "GET", "/p", "1.1", 200))
    assert flt.filter(plain) is True and plain.args == ("c", "GET", "/p", "1.1", 200)
    hostile = record(Boom((1, 2, 3)))
    assert flt.filter(hostile) is True  # never raises
    assert hostile.msg == "[REDACTION_FAILED]" and hostile.args is None


def test_t12_uvicorn_loggers_without_a_prior_dictconfig_are_also_absorbed(
    production_logging: ConfigureFn,
) -> None:
    production_logging(json_output=True)
    assert logging.getLogger("uvicorn").propagate is True


# ---------------------------------------------------------------------------
# ccxt library logger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING"])
def test_ccxt_logger_is_pinned_to_warning_regardless_of_log_level(
    level: str, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=True, log_level=level)
    assert logging.getLogger("ccxt").level == logging.WARNING
    lg = logging.getLogger("ccxt.base.exchange")
    assert lg.isEnabledFor(logging.DEBUG) is False
    lg.debug("Request: %s", {"X-MBX-APIKEY": KEY})
    lg.warning("a warning still gets through: signature=%s", SIG)
    assert KEY not in cap.text and S not in cap.text
    assert [r for r in json_lines(cap.err) if r["logger"] == "ccxt.base.exchange"]


# ---------------------------------------------------------------------------
# T-13 / T-14  fail-safe through the real chain
# ---------------------------------------------------------------------------


class _BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError(f"{S} repr exploded")


class _Flaky:
    """repr works for the redactor (first call) and fails for the renderer."""

    calls = 0

    def __repr__(self) -> str:
        _Flaky.calls += 1
        if _Flaky.calls > 1:
            raise RuntimeError(S)
        return "flaky"


def _failure_count() -> int:
    from common.metrics import metrics

    return int(metrics.get_all()["counters"]["log_redaction_failures_total"])


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t13_failing_values_emit_one_safe_line_and_never_raise(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    log = structlog.get_logger("cf36.failsafe")
    before = _failure_count()

    log.error("t13.badrepr", bad=_BadRepr(), detail=f"api_key={KEY}")
    assert len(cap.lines_with("t13.badrepr")) == 1
    assert "[REDACTION_FAILED]" in cap.text and S not in cap.text
    assert _failure_count() > before

    cyc: dict[str, Any] = {}
    cyc["self"] = cyc
    log.error("t13.cyclic", c=cyc, detail=f"api_key={KEY}")  # must not raise
    assert S not in cap.text

    log.error("t13.huge", text="x" * 10_000_000 + f" api_key={KEY}")
    (huge,) = cap.lines_with("t13.huge")
    assert "TRUNCATED len=" in huge and len(huge) < 200_000 and S not in cap.text


def test_t13_a_failing_rule_still_yields_exactly_one_line(
    production_logging: ConfigureFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = production_logging(json_output=True)

    class Boom:
        def subn(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError(S)

    monkeypatch.setattr(R._RULES[7], "rx", Boom())
    structlog.get_logger("cf36.failsafe").error(
        "t13.rule", detail=f"x=1&api_key={KEY}", level_hint="keep"
    )
    (line,) = cap.lines_with("t13.rule")
    rec = json.loads(line)
    assert rec["detail"] == "[REDACTION_FAILED]" and rec["level_hint"] == "keep"
    assert rec["level"] == "error" and "timestamp" in rec
    assert S not in cap.text


def test_t14_renderer_failure_emits_the_constant_line(production_logging: ConfigureFn) -> None:
    cap = production_logging(json_output=True)
    _Flaky.calls = 0
    before = _failure_count()
    structlog.get_logger("cf36.failsafe").error("t14.flaky", obj=_Flaky())  # must not raise
    lines = cap.lines()
    assert lines == ['{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}']
    assert _failure_count() > before


def test_t14_wrapper_catches_any_inner_exception() -> None:
    def boom(*_a: Any) -> str:
        raise ValueError(S)

    out = R.RedactingRenderer(boom)(None, "info", {"event": "x"})
    assert out == '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'
    assert S not in out
    assert (
        R.RedactingRenderer(lambda *_a: b"bytes-pass-through")(None, "i", {})
        == b"bytes-pass-through"
    )


def test_t14_final_pass_redacts_a_registered_secret_the_walker_never_saw(
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=True)
    secret = f"{S}-final-pass-secret-0123456789"
    R.register_secret(secret, name="x")

    class LateSecret:
        """Renders the secret only inside the renderer (after the walker ran)."""

        n = 0

        def __repr__(self) -> str:
            LateSecret.n += 1
            return "clean" if LateSecret.n == 1 else f"late {secret}"

    structlog.get_logger("cf36.final").error("t14.late", obj=LateSecret())
    (line,) = cap.lines_with("t14.late")
    assert secret not in line and "[REDACTED]" in line


# ---------------------------------------------------------------------------
# T-17  chain invariants
# ---------------------------------------------------------------------------


def _names(procs: list[Any]) -> list[str]:
    return [getattr(p, "__name__", type(p).__name__) for p in procs]


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t17_chain_invariants_native_and_bridge(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    production_logging(json_output=json_output)
    native = list(structlog.get_config()["processors"])
    assert isinstance(native[-1], R.RedactingRenderer)
    assert native[-2] is R.redact_event_dict
    names = _names(native)
    assert names.index("ExceptionRenderer") < len(native) - 2
    assert names.index("StackInfoRenderer") < names.index("ExceptionRenderer")

    handler = logging.getLogger().handlers[0]
    procs = list(handler.formatter.processors)  # type: ignore[union-attr]
    assert isinstance(procs[-1], R.RedactingRenderer) and procs[-2] is R.redact_event_dict
    pnames = _names(procs)
    assert pnames.index("ExceptionRenderer") == len(procs) - 3
    assert pnames[0] == "remove_processors_meta"
    pre = _names(list(handler.formatter.foreign_pre_chain))  # type: ignore[union-attr]
    assert "ExtraAdder" in pre and "_drop_color_message" in pre and "TimeStamper" in pre

    for chain in (native, procs):
        inner = chain[-1].inner
        if json_output:
            assert isinstance(inner, structlog.processors.JSONRenderer)
        else:
            assert isinstance(inner, structlog.dev.ConsoleRenderer)
            assert getattr(inner, "_exception_formatter", structlog.dev.plain_traceback) is (
                structlog.dev.plain_traceback
            )
    for chain in (native, procs):
        assert not any(
            "rich" in repr(type(p)).lower() or "dict_tracebacks" in n.lower()
            for p, n in zip(chain, _names(chain), strict=True)
        )


def test_t17_wrapper_class_and_factory_are_unchanged(production_logging: ConfigureFn) -> None:
    production_logging(json_output=True)
    cfg = structlog.get_config()
    assert cfg["context_class"] is dict
    assert isinstance(cfg["logger_factory"], structlog.PrintLoggerFactory)


# ---------------------------------------------------------------------------
# T-35  stdlib fail-safe
# ---------------------------------------------------------------------------


def test_t35_malformed_third_party_call_emits_one_constant_line(
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=True)
    logging.getLogger("x.bad").info("%d", f"{S}-SECRET-ARG-123456")
    logging.getLogger("x.ok").info("after %s", "ok")
    text = cap.err
    assert "Logging error" not in text and "Traceback" not in text
    assert S not in text and S not in cap.out
    lines = [ln for ln in text.splitlines() if ln.strip()]
    failures = [ln for ln in lines if "[REDACTION_FAILED]" in ln]
    assert failures == [
        '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'
    ]
    assert any("after ok" in ln for ln in lines)


def test_t35_handle_error_writes_a_constant_line_and_never_msg_or_args(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from common.logging import _SafeStreamHandler

    class Broken:
        def write(self, _s: str) -> None:
            raise OSError(f"{S} broken pipe")

        def flush(self) -> None:
            raise OSError(S)

    handler = _SafeStreamHandler(stream=Broken())  # type: ignore[arg-type]
    record = logging.LogRecord(
        "x", logging.ERROR, __file__, 1, f"msg {S}-MSG %s", (f"{S}-ARG",), None
    )
    handler.emit(record)  # StreamHandler.emit -> write fails -> handleError
    captured = capfd.readouterr()
    assert S not in captured.out + captured.err
    assert "Logging error" not in captured.err and "Traceback" not in captured.err
    assert (
        captured.err.strip()
        == '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'
    )


def test_t35_handle_error_never_raises_even_without_a_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from common.logging import _SafeStreamHandler

    monkeypatch.setattr(sys, "__stderr__", None)
    _SafeStreamHandler().handleError(logging.LogRecord("x", 40, __file__, 1, "m", (), None))


# ---------------------------------------------------------------------------
# exc_log_fields and counters
# ---------------------------------------------------------------------------


def test_exc_log_fields_is_type_status_and_location_only() -> None:
    class Status(Exception):
        status = 429

    def raiser() -> None:
        raise Status(f"url with {S}")

    try:
        raiser()
    except Status as exc:
        fields = exc_log_fields(exc)
    assert fields["error_type"] == "Status" and fields["http_status"] == 429
    assert str(fields["error_at"]).startswith("test_cf36_logging_chain.py:")
    assert str(fields["error_at"]).endswith(":raiser")
    assert S not in repr(fields)

    assert exc_log_fields(ValueError("x")) == {
        "error_type": "ValueError"
    }  # no traceback -> no error_at

    class Odd(Exception):
        status = True

    class Out(Exception):
        status = 99

    class Str(Exception):
        status = "500"

    for cls in (Odd, Out, Str):
        assert "http_status" not in exc_log_fields(cls("x"))

    class Hostile(Exception):
        @property
        def status(self) -> int:
            raise RuntimeError(S)

    assert exc_log_fields(Hostile("x")) == {"error_type": "Hostile"}


def test_hit_and_failure_counters_are_pre_registered_and_count() -> None:
    from common.metrics import metrics

    counters = metrics.get_all()["counters"]
    assert "log_redaction_failures_total" in counters
    for i in range(1, 11):
        assert f'log_redaction_hits_total{{rule="R-{i:02d}"}}' in counters
    before = dict(counters)
    R.redact_text(f"https://h/p?api_key={KEY}")
    R.redact_text("postgres://u:pw@h/db")
    after = metrics.get_all()["counters"]
    assert (
        after['log_redaction_hits_total{rule="R-08"}']
        == before['log_redaction_hits_total{rule="R-08"}'] + 1
    )
    assert (
        after['log_redaction_hits_total{rule="R-02"}']
        == before['log_redaction_hits_total{rule="R-02"}'] + 1
    )
    R.register_secret(f"{S}-counter-secret-0123456789", name="x")
    R.redact_text(f"x {S}-counter-secret-0123456789")
    assert (
        metrics.get_all()["counters"]['log_redaction_hits_total{rule="R-09"}']
        > before['log_redaction_hits_total{rule="R-09"}']
    )
    R.redact_event_dict(None, "i", {"event": "e", "token": "abc"})
    assert (
        metrics.get_all()["counters"]['log_redaction_hits_total{rule="R-10"}']
        > before['log_redaction_hits_total{rule="R-10"}']
    )


def test_prometheus_bridge_still_renders_with_the_new_counters() -> None:
    from prometheus_client.core import CollectorRegistry

    from api.prometheus import TradingBotCollector

    reg = CollectorRegistry()
    reg.register(TradingBotCollector(0.0))
    R.redact_text(f"https://h/p?api_key={KEY}")  # a labelled hit
    names = {m.name for m in reg.collect()}
    assert "trading_log_redaction_hits" in names or "trading_log_redaction_hits_total" in names
    assert any(n.startswith("trading_log_redaction_failures") for n in names)


def test_r11_counter_is_pre_registered_and_counts() -> None:
    from common.metrics import metrics

    before = dict(metrics.get_all()["counters"])
    assert 'log_redaction_hits_total{rule="R-11"}' in before
    R.redact_text('{"apiKey": "' + "VAL" + '0123456789"}')
    assert (
        metrics.get_all()["counters"]['log_redaction_hits_total{rule="R-11"}']
        == before['log_redaction_hits_total{rule="R-11"}'] + 1
    )


def test_console_renderer_is_built_with_the_plain_traceback_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CF36A-CR-06(f): the pin is unobservable without ``rich``; record the kwarg."""
    from common.config import build_renderer

    seen: dict[str, Any] = {}

    def recorder(*_a: Any, **kw: Any) -> Callable[..., str]:
        seen.update(kw)
        return lambda *_x: ""

    monkeypatch.setattr(structlog.dev, "ConsoleRenderer", recorder)
    build_renderer(False)
    assert seen["exception_formatter"] is structlog.dev.plain_traceback


# ---------------------------------------------------------------------------
# T-01p / T-01n  credential-name parity matrix (CF36A-SEC-03)
# ---------------------------------------------------------------------------


def _forms(log: Any, std: logging.Logger, n: str) -> dict[str, Callable[[], None]]:
    d = {n: U}
    return {
        "native_kwarg": lambda: log.info("p.kwarg", **{n: U}),
        "native_headers_dict": lambda: log.info("p.dict", headers=d),
        "native_text_eq": lambda: log.info("p.eq", detail=f"{n}={U}"),
        "native_text_quoted": lambda: log.info("p.quoted", detail=f"{n}='{U}'"),
        "native_json": lambda: log.info("p.json", detail=json.dumps(d)),
        "native_repr": lambda: log.info("p.repr", detail=repr(d)),
        "native_text_colon": lambda: log.info("p.colon", detail=f"{n}: {U}"),
        "std_dict_arg": lambda: std.info("p.s1 %s", d),
        "std_extra": lambda: std.info("p.s2", extra={"hdrs": d}),
        "std_json": lambda: std.info("p.s3 %s", json.dumps(d)),
        "std_text_colon": lambda: std.info("p.s4 %s", f"{n}: {U}"),
        "std_exception_arg": lambda: std.error("p.s5", exc_info=ValueError(("x", d))),
    }


def _emit(cap: Capture, fn: Callable[[], None]) -> str:
    a, b = len(cap.out), len(cap.err)
    fn()
    return cap.out[a:] + cap.err[b:]


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t01p_parity_matrix_no_unregistered_credential_leaks(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    """64 credential names x 12 forms x both paths: 768 cells, 0 leaks, a marker in each."""
    cap = production_logging(json_output=json_output)
    log, std = structlog.get_logger("cf36.parity"), logging.getLogger("cf36.parity.std")
    assert len(PARITY_NAMES) == 64 and len(set(PARITY_NAMES)) == 64
    cells = 0
    leaks: list[str] = []
    unmarked: list[str] = []
    for n in PARITY_NAMES:
        forms = _forms(log, std, n)
        assert len(forms) == 12
        for form, fn in forms.items():
            new = _emit(cap, fn)
            cells += 1
            if U in new or S in new:
                leaks.append(f"{n}|{form}")
            elif "[REDACTED" not in new:
                unmarked.append(f"{n}|{form}")
    assert cells == 768
    assert leaks == [], f"{len(leaks)} leaking cells: {leaks[:12]}"
    assert unmarked == [], f"{len(unmarked)} cells without a redaction marker: {unmarked[:12]}"


_C2_NAMES = ["passPhrase", "PassPhrase", "apiPassPhrase", "passWord", "pass_phrase"]


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_c2_camelcase_pass_word_and_pass_phrase_parity(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    """CF36A-SEC-203: native (R-10) and stdlib (R-08/R-11) agree on split pass-words."""
    cap = production_logging(json_output=json_output)
    log, std = structlog.get_logger("cf36.parity"), logging.getLogger("cf36.parity.std")
    bad: list[str] = []
    for n in _C2_NAMES:
        assert R.is_sensitive_key(n), n
        for form, fn in _forms(log, std, n).items():
            new = _emit(cap, fn)
            if U in new or "[REDACTED" not in new:
                bad.append(f"{n}|{form}")
    assert bad == [], bad


@pytest.mark.parametrize("json_output", RENDERERS, ids=RENDERER_IDS)
def test_t01n_ordinary_names_are_not_over_redacted(
    json_output: bool, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=json_output)
    log, std = structlog.get_logger("cf36.parity"), logging.getLogger("cf36.parity.std")
    cells = 0
    for n in NEGATIVE_NAMES:
        forms = _forms(log, std, n)
        for form in (
            "native_kwarg",
            "native_headers_dict",
            "native_text_eq",
            "native_json",
            "std_dict_arg",
        ):
            new = _emit(cap, forms[form])
            cells += 1
            assert U in new, f"over-redacted: {n}|{form}"
        d = {n: U}
        out = R.redact_event_dict(None, "info", {"event": "e", "headers": d})
        assert out["headers"] is d, f"identity lost for {n}"
    assert cells == len(NEGATIVE_NAMES) * 5
