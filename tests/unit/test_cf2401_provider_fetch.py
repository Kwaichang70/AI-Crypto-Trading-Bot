"""
tests/unit/test_cf2401_provider_fetch.py
-----------------------------------------
CF-24-01 P2 (client fetch layer): T-04 .. T-09 of reports/vp2-cf2401 (synthesis-spec, amended by
spec-amendment-r1 section 7 and spec-erratum-r2 N-1/A-1).

No network: every test drives a fake aiohttp-shaped session. The AST pins live in
``test_cf2401_ast_pins.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, get_args
from unittest.mock import MagicMock

import aiohttp
import pytest

from data import provider_bands, provider_http
from data.macro_data import FREDClient, MacroSnapshot
from data.market_signals import CoinGeckoClient, CoinGeckoSnapshot
from data.point_in_time import SERIES_RULES
from data.provider_bands import COINGECKO_PROVIDER_STALE_MS
from data.provider_http import (
    MAX_BYTES_COINGECKO,
    MAX_BYTES_FGI,
    MAX_BYTES_FRED,
    MAX_BYTES_WHALE,
    Err,
    ErrReason,
    Ok,
    fetch_json,
)
from data.sentiment import FearGreedClient, FearGreedSnapshot
from data.whale_tracker import WhaleAlertClient, WhaleFlowSnapshot
from tests._cf36_logging import SENTINEL, ConfigureFn, disable_layer1, json_lines

KEY = f"{SENTINEL}KEY0123456789"
NOW_MS = 1_700_000_000_000
DAY0 = 19_700 * 86_400  # a midnight-UTC unix stamp
JSON_HDR = {"Content-Type": "application/json; charset=utf-8"}


# ---------------------------------------------------------------------------
# Fake aiohttp-shaped session
# ---------------------------------------------------------------------------
class _Content:
    """``read(n)`` honours ``n`` like aiohttp's StreamReader; ``short`` caps a read at 4096."""

    def __init__(self, body: bytes, short: bool = False) -> None:
        self._body, self._pos, self._short = body, 0, short
        self.served, self.max_n = 0, 0

    async def read(self, n: int = -1) -> bytes:
        self.max_n = max(self.max_n, n)
        take = len(self._body) - self._pos if n < 0 else n
        if self._short:
            take = min(take, 4096)  # short reads, like a socket
        chunk = self._body[self._pos : self._pos + take]
        self._pos += len(chunk)
        self.served += len(chunk)
        return chunk


class _Response:
    def __init__(
        self, status: int, body: bytes, headers: dict[str, str], short: bool = False
    ) -> None:
        self.status, self.headers, self.content = status, headers, _Content(body, short)
        self._body = body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(MagicMock(), (), status=self.status)

    async def json(self, content_type: object = None) -> Any:
        return json.loads(self._body)


class _Ctx:
    def __init__(self, item: _Response | BaseException) -> None:
        self.item = item

    async def __aenter__(self) -> _Response:
        if isinstance(self.item, BaseException):
            raise self.item
        return self.item

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _Session:
    closed = False

    def __init__(self, *items: _Response | BaseException) -> None:
        self.items, self.calls = list(items), []

    def get(self, url: str, **kwargs: Any) -> _Ctx:
        self.calls.append((url, kwargs))
        return _Ctx(self.items.pop(0) if len(self.items) > 1 else self.items[0])


def ok(
    body: object, status: int = 200, headers: dict[str, str] | None = None, short: bool = False
) -> _Response:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return _Response(status, raw, JSON_HDR | (headers or {}), short)


def fgi_rows(n: int = 30, value: object = "25", step: int = 1) -> list[dict[str, object]]:
    return [
        {"value": value, "value_classification": "x", "timestamp": str(DAY0 - i * step * 86_400)}
        for i in range(n)
    ]


def cg_body(btc: object = "57.2", **data: object) -> dict[str, object]:
    return {"data": {"market_cap_percentage": {"btc": btc}, **data}}


def fred_rows(*pairs: tuple[str, str]) -> dict[str, object]:
    return {"observations": [{"date": d, "value": v} for d, v in pairs]}


def whale_body(*txs: dict[str, object], result: str = "success") -> dict[str, object]:
    return {"result": result, "transactions": list(txs)}


def tx(amount: object, to: object = None, frm: object = None) -> dict[str, object]:
    return {"amount_usd": amount, "to": to, "from": frm}


EXCH = {"owner_type": "exchange", "owner": "binance"}
WALLET = {"owner_type": "unknown", "owner": "unknown"}


def fgi(*items: _Response | BaseException) -> FearGreedClient:
    client = FearGreedClient()
    client._session = _Session(*items)
    return client


def cg(*items: _Response | BaseException) -> CoinGeckoClient:
    client = CoinGeckoClient()
    client._session = _Session(*items)
    return client


def fred(*items: _Response | BaseException) -> FREDClient:
    return FREDClient(api_key=KEY, session=_Session(*items))


def whale(*items: _Response | BaseException) -> WhaleAlertClient:
    return WhaleAlertClient(api_key=KEY, session=_Session(*items))


def reason_of(result: object) -> tuple[str, str | None]:
    assert isinstance(result, Err), result
    return result.outcome, result.reason


NAMES = ("fgi", "cg", "fred", "whale")
FACTORY: dict[str, Callable[..., Any]] = {"fgi": fgi, "cg": cg, "fred": fred, "whale": whale}
KWARGS: dict[str, dict[str, Any]] = {
    "fgi": {},
    "cg": {"now_ms": NOW_MS},
    "fred": {},
    "whale": {"now_ms": NOW_MS},
}
EVENT_PREFIX = {
    "fgi": "fear_greed_client",
    "cg": "coingecko_client",
    "fred": "fred_client",
    "whale": "whale_alert_client",
}
CAP_OF = {
    "fgi": MAX_BYTES_FGI,
    "cg": MAX_BYTES_COINGECKO,
    "fred": MAX_BYTES_FRED,
    "whale": MAX_BYTES_WHALE,
}


def good_items(name: str) -> list[_Response]:
    """Responses that make ``fetch_validated`` return ``Ok`` (FRED needs two legs)."""
    return {
        "fgi": [ok({"data": fgi_rows()})],
        "cg": [ok(cg_body())],
        "fred": [ok(fred_rows(("2024-03-11", "0.5"))), ok(fred_rows(("2024-03-01", "5.33")))],
        "whale": [ok(whale_body())],
    }[name]


def invalid_items(name: str) -> list[_Response]:
    """Responses that reach the validator and are refused with a closed reason."""
    return {
        "fgi": [ok({"data": []})],
        "cg": [ok(cg_body(btc=None))],
        "fred": [ok({})],
        "whale": [ok(whale_body(result="error"))],
    }[name]


# ---------------------------------------------------------------------------
# T-04: validators reject with closed reasons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("nan", "bad_number"),
        ("inf", "bad_number"),
        ("1e400", "bad_number"),
        (True, "bad_type"),
        (False, "bad_type"),
        (1.5, "bad_number"),  # a JSON number arrives as Decimal("1.5"): not an integer
        (None, "bad_type"),
        ([25], "bad_type"),
        (101, "out_of_band"),
        (-1, "out_of_band"),
        ("99.9", "bad_number"),
    ],
)
async def test_fgi_value_vectors(value: object, reason: str) -> None:
    result = await fgi(ok({"data": fgi_rows(1, value)})).fetch_validated()
    assert reason_of(result) == ("invalid_payload", reason)


def test_a_python_float_is_never_accepted_as_a_value() -> None:
    from data.provider_bands import Reject, to_decimal

    with pytest.raises(Reject) as info:
        to_decimal(1.5)
    assert info.value.reason == "bad_type" and str(info.value) == "bad_type"


async def test_fgi_structure_vectors() -> None:
    row = fgi_rows(1)[0]
    cases = {
        "missing_key": {"data": [{"timestamp": row["timestamp"]}]},
        "bad_shape": {"data": []},
        "bad_timestamp": {"data": [{**row, "timestamp": str(DAY0 + 3600)}]},
        "bad_type": {"data": [{**row, "timestamp": True}]},
    }
    for reason, body in cases.items():
        assert reason_of(await fgi(ok(body)).fetch_validated()) == ("invalid_payload", reason)
    assert reason_of(await fgi(ok({"data": fgi_rows(31)})).fetch_validated())[1] == "bad_shape"
    dup = {"data": [row, row]}
    assert reason_of(await fgi(ok(dup)).fetch_validated())[1] == "bad_shape"
    assert reason_of(await fgi(ok([row])).fetch_validated())[1] == "bad_shape"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (cg_body(btc=None), "bad_type"),
        (cg_body(btc="nan"), "bad_number"),
        (cg_body(btc=True), "bad_type"),
        (cg_body(btc="0"), "out_of_band"),
        (cg_body(btc="100.01"), "out_of_band"),
        (cg_body(btc="-3"), "out_of_band"),
        (cg_body(market_cap_change_percentage_24h_usd="1000.5"), "out_of_band"),
        (cg_body(total_volume_change_percentage_24h="-100.1"), "out_of_band"),
        (cg_body(market_cap_change_percentage_24h_usd=None), "bad_type"),
        ({"data": {"market_cap_percentage": {"eth": "10"}}}, "missing_key"),
        ({"data": {"market_cap_percentage": [57]}}, "bad_shape"),
        ({"data": {}}, "missing_key"),
        ({"nope": 1}, "missing_key"),
        (cg_body(updated_at="soon"), "bad_type"),
    ],
)
async def test_coingecko_vectors(body: object, reason: str) -> None:
    result = await cg(ok(body)).fetch_validated(now_ms=NOW_MS)
    assert reason_of(result) == ("invalid_payload", reason)


async def test_coingecko_exponent_beyond_band_is_rejected_from_the_wire() -> None:
    # "1e400" as a JSON number token parses to a finite Decimal and is caught by the band.
    result = await cg(ok(b'{"data":{"market_cap_percentage":{"btc":1e400}}}')).fetch_validated()
    assert reason_of(result) == ("invalid_payload", "out_of_band")


async def test_coingecko_ok_values_are_exact_decimals_and_optional_fields_default() -> None:
    result = await cg(
        ok(b'{"data":{"market_cap_percentage":{"btc":57.123456789012}}}')
    ).fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Ok)
    snap = result.value
    assert snap.btc_dominance == 57.123456789012
    assert (snap.market_cap_change_24h, snap.total_volume_change_24h) == (0.0, 0.0)
    assert snap.timestamp == datetime.fromtimestamp(NOW_MS / 1000, tz=UTC)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (whale_body(result="error"), "provider_error"),
        ({"transactions": []}, "missing_key"),
        ({"result": "success"}, "missing_key"),
        ({"result": "success", "transactions": {}}, "bad_shape"),
        (whale_body(*[tx(2e6)] * 100), "truncated"),
        (whale_body(*[tx(2e6)] * 101), "truncated"),
        (whale_body(tx("nan")), "bad_number"),
        (whale_body(tx("1e400")), "bad_number"),
        (whale_body(tx(True)), "bad_type"),
        (whale_body(tx(-5)), "out_of_band"),
        (whale_body(tx(2e12)), "out_of_band"),
        (whale_body(tx("1000000000000.01")), "out_of_band"),  # just over the 1e12 amount band
        (whale_body({"to": EXCH}), "missing_key"),
        (whale_body(tx(2e6, to="binance")), "bad_shape"),
        (whale_body(*[tx(9e11, to=EXCH)] * 2), "out_of_band"),  # |net_flow| > 1e12
        ([], "bad_shape"),
    ],
)
async def test_whale_vectors(body: object, reason: str) -> None:
    result = await whale(ok(body)).fetch_validated(now_ms=NOW_MS)
    assert reason_of(result) == ("invalid_payload", reason)


async def test_whale_ok_path_is_exact_and_requests_the_one_hour_window() -> None:
    body = whale_body(
        tx(3_000_000.10, to=EXCH),
        tx(1_000_000.05, frm=EXCH),
        tx(5_000_000, to=WALLET, frm=WALLET),
        tx(10, to=EXCH),  # below min_value: skipped
    )
    client = whale(ok(body))
    result = await client.fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Ok)
    assert (result.value.net_flow, result.value.large_tx_count) == (2_000_000.05, 3)
    _url, kwargs = client._session.calls[0]
    assert kwargs["params"]["start"] == str(NOW_MS // 1000 - 3600)
    assert (kwargs["params"]["min_value"], kwargs["params"]["limit"]) == ("1000000", "100")
    full = whale(ok(whale_body(*[tx(2e6, to=EXCH)] * 99)))
    assert isinstance(await full.fetch_validated(now_ms=NOW_MS), Ok)


@pytest.mark.parametrize(
    ("stamp", "reason"),
    [
        (0, "bad_timestamp"),
        (-1, "bad_timestamp"),
        (provider_bands._UNIX_MAX, None),  # 2100-01-01T00:00:00Z, a midnight stamp
        (provider_bands._UNIX_MAX + 1, "bad_timestamp"),
    ],
)
async def test_fgi_timestamp_edges(stamp: int, reason: str | None) -> None:
    row = {**fgi_rows(1)[0], "timestamp": stamp}
    result = await fgi(ok({"data": [row]})).fetch_validated()
    if reason is None:
        assert isinstance(result, Ok)
    else:
        assert reason_of(result) == ("invalid_payload", reason)


@pytest.mark.parametrize(
    ("amount", "to", "frm", "net", "count"),
    [
        (0, EXCH, None, 0.0, 0),  # a zero amount is in band, and below the floor
        ("1000000000000", EXCH, None, 1e12, 1),  # exactly +1e12 is valid
        ("1000000000000", None, EXCH, -1e12, 1),  # exactly -1e12 is valid
        (1_000_000, EXCH, None, 1e6, 1),  # amount == min_value is counted
        (999_999, EXCH, None, 0.0, 0),
    ],
)
async def test_whale_band_and_floor_edges(
    amount: object, to: object, frm: object, net: float, count: int
) -> None:
    result = await whale(ok(whale_body(tx(amount, to=to, frm=frm)))).fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Ok)
    assert (result.value.net_flow, result.value.large_tx_count) == (net, count)


async def test_whale_floor_is_built_from_text_not_from_a_float() -> None:
    # Decimal(0.1) is 0.1000000000000000055...: it would drop an amount equal to the floor.
    client = WhaleAlertClient(api_key=KEY, session=_Session(ok(whale_body(tx("0.1", to=EXCH)))))
    client._min_value_usd = 0.1  # type: ignore[assignment]
    result = await client.fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Ok) and result.value.large_tx_count == 1


# ---------------------------------------------------------------------------
# T-05: size caps, parse hardening, Decimal from text
# ---------------------------------------------------------------------------
async def test_oversize_by_content_length_reads_nothing() -> None:
    resp = ok(fgi_rows(1), headers={"Content-Length": str(MAX_BYTES_FGI + 1)})
    result = await fetch_json(_Session(resp), "u", params=None, max_bytes=MAX_BYTES_FGI)
    assert reason_of(result) == ("oversize", "content_length")
    assert resp.content.served == 0


async def test_oversize_by_streamed_body_is_bounded() -> None:
    resp = ok(b" " * (MAX_BYTES_FGI * 40))  # no Content-Length: a chunked/lying server
    result = await fetch_json(_Session(resp), "u", params=None, max_bytes=MAX_BYTES_FGI)
    assert reason_of(result) == ("oversize", "stream")
    assert resp.content.served <= MAX_BYTES_FGI + 1  # never reads on past cap + 1
    assert resp.content.max_n <= provider_http._CHUNK  # and never asks for more than one chunk


def _padded(size: int) -> bytes:
    return b'{"a": 1}' + b" " * (size - 8)


_CAP = MAX_BYTES_FGI


@pytest.mark.parametrize(
    ("length", "size", "expected"),
    [
        # malformed or absurd Content-Length: refused before any read
        ("abc", 2, ("invalid_payload", "bad_length")),
        ("-1", 2, ("invalid_payload", "bad_length")),
        ("", 2, ("invalid_payload", "bad_length")),
        ("9" * 40, 2, ("oversize", "content_length")),
        # the cap edge, declared and streamed
        (str(_CAP), _CAP, None),
        (str(_CAP + 1), _CAP + 1, ("oversize", "content_length")),
        (None, _CAP, None),
        (None, _CAP + 1, ("oversize", "stream")),
    ],
)
async def test_size_cap_edges(
    length: str | None, size: int, expected: tuple[str, str] | None
) -> None:
    body = _padded(size) if size >= 8 else b"{}"
    resp = ok(body, headers=None if length is None else {"Content-Length": length})
    result = await fetch_json(_Session(resp), "u", params=None, max_bytes=_CAP)
    if expected is None:
        assert isinstance(result, Ok) and result.value == {"a": Decimal(1)}
    else:
        assert reason_of(result) == expected
        if expected[1] != "stream":
            assert resp.content.served == 0


def test_response_caps_are_pinned() -> None:
    assert MAX_BYTES_FGI == 65_536
    assert MAX_BYTES_COINGECKO == 262_144
    assert MAX_BYTES_FRED == 65_536
    assert MAX_BYTES_WHALE == 2_097_152


@pytest.mark.parametrize("name", NAMES)
async def test_each_client_passes_its_own_cap(name: str) -> None:
    resp = ok(b" " * (CAP_OF[name] + 1))  # one byte over this provider's cap, no Content-Length
    result = await FACTORY[name](resp).fetch_validated(**KWARGS[name])
    assert reason_of(result) == ("oversize", "stream")
    assert resp.content.served == CAP_OF[name] + 1


async def test_a_body_spanning_many_short_reads_is_assembled() -> None:
    padded = json.dumps({"data": fgi_rows(30)}).encode() + b" " * 20_000
    assert isinstance(await fgi(ok(padded, short=True)).fetch_validated(), Ok)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "gzip, identity", "GZIP"])
async def test_content_encoding_is_refused_before_any_read(encoding: str) -> None:
    resp = ok(b'{"a": 1}', headers={"Content-Encoding": encoding})
    result = await fetch_json(_Session(resp), "u", params=None, max_bytes=100)
    assert reason_of(result) == ("invalid_payload", "bad_encoding")
    assert resp.content.served == 0


@pytest.mark.parametrize("encoding", [None, "identity", "IDENTITY", " Identity "])
async def test_identity_or_absent_content_encoding_is_accepted(encoding: str | None) -> None:
    headers = None if encoding is None else {"Content-Encoding": encoding}
    result = await fetch_json(
        _Session(ok(b'{"a": 1}', headers=headers)), "u", params=None, max_bytes=100
    )
    assert isinstance(result, Ok)


@pytest.mark.parametrize(
    "body",
    [
        b"[" * 100_000 + b"]" * 100_000,  # recursion
        b'{"data":[{"value":NaN}]}',
        b'{"data":[{"value":Infinity}]}',
        b'{"data":[{"value":-Infinity}]}',
        b'{"a": ' + b"9" * 5000 + b"}",  # int digit limit
        b'{"a": 1e9999999999999999999999}',  # decimal.InvalidOperation (an ArithmeticError)
        b'{"a": 1e-9999999999999999999999}',
        b"\xff\xfe",
        b"",
        b"<html>",
    ],
)
async def test_hostile_bodies_are_invalid_not_crashes(body: bytes) -> None:
    result = await cg(ok(body)).fetch_validated(now_ms=NOW_MS)
    assert reason_of(result) == ("invalid_payload", "bad_json")


@pytest.mark.parametrize("token", ["1e9999999999999999999999", "1e-9999999999999999999999"])
async def test_a_decimal_overflow_returns_err_and_never_raises(token: str) -> None:
    body = b'{"a": ' + token.encode() + b"}"
    result = await fetch_json(_Session(ok(body)), "u", params=None, max_bytes=100)
    assert reason_of(result) == ("invalid_payload", "bad_json")


async def test_decimal_overflow_is_not_logged_as_an_unexpected_error(
    production_logging: ConfigureFn,
) -> None:
    cap = production_logging(json_output=True, log_level="INFO")
    overflow = b'{"data":{"market_cap_percentage":{"btc":1e9999999999999999999999}}}'
    assert reason_of(await cg(ok(overflow)).fetch_validated(now_ms=NOW_MS)) == (
        "invalid_payload",
        "bad_json",
    )
    funds = ok(b'{"observations":[{"date":"2024-03-01","value":1e9999999999999999999999}]}')
    assert isinstance(await fred(ok(fred_rows(("2024-03-11", "0.5"))), funds).fetch_validated(), Ok)
    assert "fetch_validated_error" not in cap.text
    # presence: a genuinely unexpected bug is still logged, so the absence above is not vacuous
    await cg(ValueError("x")).fetch_validated(now_ms=NOW_MS)
    assert cap.text.count("fetch_validated_error") == 1


async def test_json_floats_arrive_as_decimal_text() -> None:
    result = await fetch_json(
        _Session(ok(b'{"v": 0.1, "w": [1, 2.50]}')), "u", params=None, max_bytes=100
    )
    assert isinstance(result, Ok)
    assert result.value["v"] == Decimal("0.1") and str(result.value["w"][1]) == "2.50"


# ---------------------------------------------------------------------------
# Transport classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (301, "http_3xx"),
        (307, "http_3xx"),
        (401, "auth_failed"),
        (403, "auth_failed"),
        (404, "http_4xx"),
        (429, "http_429"),
        (500, "http_5xx"),
        (503, "http_5xx"),
    ],
)
async def test_status_classification(status: int, outcome: str) -> None:
    result = await cg(ok({}, status=status)).fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Err) and result.outcome == outcome


@pytest.mark.parametrize(
    ("header", "kept"),
    [
        ("7", "7"),
        ("999999", "999999"),
        ("Wed, 21 Oct 2015 07:28:00 GMT", "Wed, 21 Oct 2015 07:28:00 GMT"),
        ("1234567", None),
        ('{"k":"v"}', None),
        ("-1", None),
        ("1.5", None),
        ("\u0663", None),  # an Arabic-Indic digit: matches \d, never [0-9]
        ("Wed, 21 Oct 2015 07:28:00 UTC", None),
        ("Wednesday, 21-Oct-15 07:28:00 GMT", None),  # RFC 850
        ("", None),
        ("x" * 65, None),
        (None, None),  # header absent
    ],
)
async def test_retry_after_is_header_only_and_follows_the_grammar(
    header: str | None, kept: str | None
) -> None:
    headers = None if header is None else {"Retry-After": header}
    for status in (429, 503):
        result = await fetch_json(_Session(ok({}, status, headers)), "u", params=None, max_bytes=9)
        assert isinstance(result, Err) and result.retry_after == kept


def _proxy_error() -> aiohttp.ClientProxyConnectionError:
    return aiohttp.ClientProxyConnectionError.__new__(aiohttp.ClientProxyConnectionError)


@pytest.mark.parametrize(
    ("exc", "outcome"),
    [
        (TimeoutError(), "timeout"),
        (aiohttp.ServerDisconnectedError(), "connect_error"),
        (aiohttp.ClientConnectionError(f"https://h/?api_key={KEY}"), "connect_error"),
        (aiohttp.ClientPayloadError("x"), "invalid_payload"),
        (_proxy_error(), "egress_blocked"),
    ],
)
async def test_exception_classification(exc: BaseException, outcome: str) -> None:
    result = await cg(exc).fetch_validated(now_ms=NOW_MS)
    assert isinstance(result, Err) and result.outcome == outcome
    assert KEY not in repr(result)


async def test_content_type_rules() -> None:
    html = _Response(200, b"{}", {"Content-Type": "text/html"})
    assert reason_of(await cg(html).fetch_validated(now_ms=NOW_MS)) == (
        "invalid_payload",
        "bad_content_type",
    )
    plain = _Response(
        200, json.dumps({"data": fgi_rows(1)}).encode(), {"Content-Type": "text/plain"}
    )
    assert isinstance(await fgi(plain).fetch_validated(), Ok)  # FGI only


@pytest.mark.parametrize("name", ["cg", "fred", "whale"])
async def test_text_plain_is_refused_outside_fgi(name: str) -> None:
    body = {"cg": cg_body(), "fred": fred_rows(("2024-03-11", "0.5")), "whale": whale_body()}[name]
    plain = _Response(200, json.dumps(body).encode(), {"Content-Type": "text/plain"})
    result = await FACTORY[name](plain).fetch_validated(**KWARGS[name])
    assert reason_of(result) == ("invalid_payload", "bad_content_type")


@pytest.mark.parametrize("name", NAMES)
async def test_cancellation_propagates(name: str) -> None:
    with pytest.raises(asyncio.CancelledError):
        await FACTORY[name](asyncio.CancelledError()).fetch_validated(**KWARGS[name])


async def test_cancellation_on_the_second_fred_request_propagates() -> None:
    client = fred(ok(fred_rows(("2024-03-11", "0.5"))), asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await client.fetch_validated()
    assert len(client._session.calls) == 2


def test_outcomes_are_the_closed_set() -> None:
    assert set(provider_http.OUTCOMES) == {
        "http_3xx",
        "http_4xx",
        "http_429",
        "http_5xx",
        "timeout",
        "connect_error",
        "egress_blocked",
        "invalid_payload",
        "oversize",
        "auth_failed",
    }


TRANSPORT_REASONS = {
    "bad_status",
    "bad_content_type",
    "bad_encoding",
    "bad_length",
    "content_length",
    "stream",
    "payload",
    "bad_json",
    "internal_error",
}


def test_err_reasons_are_the_closed_transport_plus_validator_set() -> None:
    assert set(get_args(ErrReason)) == TRANSPORT_REASONS | set(get_args(provider_bands.Reason))
    assert len(get_args(ErrReason)) == len(set(get_args(ErrReason)))  # no duplicate literal


# ---------------------------------------------------------------------------
# T-06: side-effect-free fetch, adopt_validated is the only writer
# ---------------------------------------------------------------------------
def _stale_snapshots() -> dict[str, Any]:
    return {
        "fgi": FearGreedSnapshot(
            value=1, classification="Extreme Fear", timestamp=datetime(2020, 1, 1, tzinfo=UTC)
        ),
        "cg": CoinGeckoSnapshot(
            btc_dominance=1.0,
            market_cap_change_24h=0.0,
            total_volume_change_24h=0.0,
            timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        ),
        "fred": MacroSnapshot(timestamp=datetime(2020, 1, 1, tzinfo=UTC)),
        "whale": WhaleFlowSnapshot(
            net_flow=1.0, large_tx_count=0, timestamp=datetime(2020, 1, 1, tzinfo=UTC)
        ),
    }


def _fetch_cases() -> list[tuple[str, Any, Any, Any]]:
    good_fred = [ok(fred_rows(("2024-03-11", "-0.37"))), ok(fred_rows(("2024-03-01", "5.33")))]
    return [
        ("fgi", fgi(ok({"data": fgi_rows()})), fgi(ok({}, 500)), {}),
        ("cg", cg(ok(cg_body())), cg(ok(cg_body(btc=None))), {"now_ms": NOW_MS}),
        ("fred", fred(*good_fred), fred(ok({}, 429)), {}),
        (
            "whale",
            whale(ok(whale_body())),
            whale(ok(whale_body(result="error"))),
            {"now_ms": NOW_MS},
        ),
    ]


@pytest.mark.parametrize("which", [0, 1, 2, 3])
@pytest.mark.parametrize("good", [True, False])
async def test_fetch_never_touches_the_caches(which: int, good: bool) -> None:
    name, good_client, bad_client, kw = _fetch_cases()[which]
    client = good_client if good else bad_client
    sentinel = (_stale_snapshots()[name], 123.0)
    client._latest_cache = sentinel
    if name == "fgi":
        history = ([_stale_snapshots()["fgi"]], 123.0)
        client._history_cache = history
    result = await client.fetch_validated(**kw)
    assert isinstance(result, Ok if good else Err)
    assert client._latest_cache is sentinel
    if name == "fgi":
        assert client._history_cache is history
    if not good:  # never a cached snapshot, even though one is available
        assert result != Ok(sentinel[0])


def test_adopt_validated_sets_the_caches_once() -> None:
    snaps = _stale_snapshots()
    for name, client in (
        ("cg", CoinGeckoClient()),
        ("fred", FREDClient(api_key=KEY)),
        ("whale", WhaleAlertClient(api_key=KEY)),
    ):
        assert client._latest_cache is None
        client.adopt_validated(snaps[name], now_mono=7.5)
        assert client._latest_cache == (snaps[name], 7.5)


def test_fgi_adopt_validated_uses_list30_and_the_unchanged_tuple_shape() -> None:
    client = FearGreedClient()
    days = [
        FearGreedSnapshot(
            value=v, classification="x", timestamp=datetime.fromtimestamp(DAY0 + 86_400 * d, tz=UTC)
        )
        for d, v in ((0, 10), (2, 30), (1, 20))
    ]
    client.adopt_validated(days, now_mono=9.0)
    assert client._latest_cache == (days[1], 9.0)  # newest by timestamp, not by position
    assert client._history_cache is not None and client._history_cache[0] is days
    assert client._history_cache[1] == 9.0 and [s.value for s in days] == [10, 30, 20]
    assert client.cached_value == 30
    with pytest.raises(ValueError, match="fgi_adopt_empty"):
        client.adopt_validated([], now_mono=1.0)


# ---------------------------------------------------------------------------
# T-07: FRED limit=5, the "." row, FEDFUNDS banded and not served
# ---------------------------------------------------------------------------
async def test_fred_skips_the_sentinel_row_and_returns_value_with_date() -> None:
    client = fred(
        ok(fred_rows(("2024-03-12", "."), ("2024-03-11", "-0.37"), ("2024-03-08", "0.1"))),
        ok(fred_rows(("2024-03-01", "5.33"))),
    )
    result = await client.fetch_validated()
    assert isinstance(result, Ok)
    snap = result.value
    assert (snap.yield_curve_spread, snap.yield_curve_observed_on) == (-0.37, date(2024, 3, 11))
    assert snap.fed_funds_rate == 5.33
    calls = client._session.calls
    assert [c[1]["params"]["series_id"] for c in calls] == ["T10Y2Y", "FEDFUNDS"]
    assert all(c[1]["params"]["limit"] == "5" for c in calls)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (fred_rows(("2024-03-12", ".")), "no_observation"),
        ({"observations": []}, "no_observation"),
        (fred_rows(*[(f"2024-03-1{i}", "1") for i in range(6)]), "bad_shape"),
        (fred_rows(("2024-03-12", "1e400")), "bad_number"),
        (fred_rows(("2024-03-12", "nan")), "bad_number"),
        (fred_rows(("2024-03-12", "10.01")), "out_of_band"),
        (fred_rows(("2024-03-12", "-10.01")), "out_of_band"),
        (fred_rows(("2024-13-12", "1")), "bad_timestamp"),
        (fred_rows(("2024-W11-1", "1")), "bad_timestamp"),
        ({"observations": [{"date": "2024-03-12"}]}, "missing_key"),
        ({"observations": {}}, "bad_shape"),
        ({}, "missing_key"),
    ],
)
async def test_fred_t10y2y_vectors(body: object, reason: str) -> None:
    client = fred(ok(body))
    assert reason_of(await client.fetch_validated()) == ("invalid_payload", reason)
    assert len(client._session.calls) == 1  # a served-series failure short-circuits


async def test_fred_fedfunds_is_banded_and_never_blocks_the_served_series() -> None:
    overflow = b'{"observations":[{"date":"2024-03-01","value":1e9999999999999999999999}]}'
    bad_bodies = (
        fred_rows(("2024-03-01", "30.5")),
        fred_rows(("2024-03-01", "-5.1")),
        {},
        overflow,  # decimal.InvalidOperation must not kill the valid T10Y2Y
    )
    for bad in bad_bodies:
        result = await fred(ok(fred_rows(("2024-03-11", "0.5"))), ok(bad)).fetch_validated()
        assert isinstance(result, Ok) and result.value.fed_funds_rate is None
        assert result.value.yield_curve_spread == 0.5
        assert result.value.yield_curve_observed_on == date(2024, 3, 11)


async def test_fred_fedfunds_failure_is_logged_with_closed_codes_at_debug_only(
    production_logging: ConfigureFn,
) -> None:
    event = "fred_client.fetch_validated_fedfunds_err"
    overflow = b'{"observations":[{"date":"2024-03-01","value":1e9999999999999999999999}]}'
    for level in ("DEBUG", "INFO"):
        cap = production_logging(json_output=True, log_level=level)
        await fred(ok(fred_rows(("2024-03-11", "0.5"))), ok(overflow)).fetch_validated()
        assert KEY not in cap.text
        rows = [r for r in json_lines(cap.text) if r.get("event") == event]
        if level == "DEBUG":
            assert len(rows) == 1
            assert (rows[0]["outcome"], rows[0]["reason"]) == ("invalid_payload", "bad_json")
        else:
            assert rows == []


async def test_fred_takes_the_newest_row_even_when_the_rows_are_misordered() -> None:
    rows = fred_rows(("2024-03-08", "0.1"), ("2024-03-11", "-0.37"), ("2024-03-10", "0.2"))
    result = await fred(ok(rows), ok(fred_rows(("2024-03-01", "5.33")))).fetch_validated()
    assert isinstance(result, Ok)
    assert (result.value.yield_curve_spread, result.value.yield_curve_observed_on) == (
        -0.37,
        date(2024, 3, 11),
    )


# ---------------------------------------------------------------------------
# T-08: FGI get_history (CF-24-09 via SY-01-21e) and classification
# ---------------------------------------------------------------------------
def _legacy(n: int = 30, ascending: bool = False) -> _Response:
    rows = [{**r, "value": str(i), "timestamp": r["timestamp"]} for i, r in enumerate(fgi_rows(n))]
    return ok({"data": rows[::-1] if ascending else rows})


async def test_get_history_always_requests_30_and_serves_the_newest_slice() -> None:
    client = fgi(_legacy())
    seven = await client.get_history(limit=7)
    session = client._session
    assert len(session.calls) == 1 and session.calls[0][1]["params"]["limit"] == 30
    assert [s.value for s in seven] == list(range(7))  # newest first: value i is day0 - i
    assert client._history_cache is not None
    cached, stamp = client._history_cache
    assert len(cached) == 30 and isinstance(stamp, float)
    assert len(await client.get_history(limit=30)) == 30 and len(session.calls) == 1
    before = list(cached)
    assert len(await client.get_history(limit=3)) == 3 and cached == before  # no in-place sort


async def test_get_history_never_reorders_the_cached_list() -> None:
    client = fgi(_legacy(ascending=True))  # the provider sends oldest first
    fresh = await client.get_history(limit=7)
    assert client._history_cache is not None
    cached = client._history_cache[0]
    original = list(cached)
    stamps = [s.timestamp for s in cached]
    assert stamps == sorted(stamps)  # the cache keeps the provider (ascending) order
    served: list[list[FearGreedSnapshot]] = [fresh]
    served.append(await client.get_history(limit=30))  # TTL cache hit
    client._cache_ttl = 0.0
    client._session = _Session(ok({}, 500))
    served.append(await client.get_history(limit=5))  # stale fallback
    for result, size in zip(served, (7, 30, 5), strict=True):
        assert len(result) == min(size, 30)
        assert [s.timestamp for s in result] == sorted((s.timestamp for s in result), reverse=True)
        assert result is not cached
    assert client._history_cache[0] is cached and cached == original  # never sorted in place


async def test_get_history_reverse_order_and_stale_fallback() -> None:
    client = fgi(_legacy())
    assert len(await client.get_history(limit=30)) == 30
    assert len(await client.get_history(limit=7)) == 7
    client._cache_ttl = 0.0  # force a refetch that fails
    client._session = _Session(ok({}, 500))
    stale = await client.get_history(limit=5)
    assert [s.value for s in stale] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0, "Extreme Fear"),
        (24, "Extreme Fear"),
        (25, "Fear"),
        (44, "Fear"),
        (45, "Neutral"),
        (55, "Neutral"),
        (56, "Greed"),
        (75, "Greed"),
        (76, "Extreme Greed"),
        (100, "Extreme Greed"),
    ],
)
async def test_fgi_classification_is_derived_from_the_value(value: int, label: str) -> None:
    result = await fgi(ok({"data": fgi_rows(1, str(value))})).fetch_validated()
    assert isinstance(result, Ok) and result.value[0].classification == label  # provider said "x"


async def test_fgi_fetch_requests_limit_30_and_keeps_provider_order() -> None:
    client = fgi(ok({"data": fgi_rows(30)}))
    result = await client.fetch_validated()
    assert isinstance(result, Ok) and len(result.value) == 30
    assert client._session.calls[0][1]["params"]["limit"] == 30
    stamps = [s.timestamp for s in result.value]
    assert stamps == sorted(stamps, reverse=True)
    assert all(s.timestamp.hour == 0 and s.timestamp.minute == 0 for s in result.value)


# ---------------------------------------------------------------------------
# T-09: redirects, provider_stale, key hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.parametrize("name", NAMES)
async def test_no_provider_follows_redirects(name: str, status: int) -> None:
    client = FACTORY[name](ok({}, status))
    result = await client.fetch_validated(**KWARGS[name])
    assert isinstance(result, Err) and result.outcome == "http_3xx"
    calls = client._session.calls
    assert calls and all(c[1]["allow_redirects"] is False for c in calls)


async def test_fred_passes_allow_redirects_false_on_both_legs() -> None:
    good = fred(*good_items("fred"))
    await good.fetch_validated()
    assert [c[1]["allow_redirects"] for c in good._session.calls] == [False, False]


# ---------------------------------------------------------------------------
# Transport posture: runtime pins (the AST pins live in test_cf2401_ast_pins.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
async def test_a_default_session_is_verified_and_ignores_the_environment(name: str) -> None:
    client = {
        "fgi": FearGreedClient(),
        "cg": CoinGeckoClient(),
        "fred": FREDClient(api_key=KEY),
        "whale": WhaleAlertClient(api_key=KEY),
    }[name]
    session = await client._get_session()
    try:
        assert session.trust_env is False
        assert session.connector is not None and session.connector._ssl is True
    finally:
        await session.close()


@pytest.mark.parametrize("name", NAMES)
async def test_the_exact_transport_kwargs_reach_the_session(name: str) -> None:
    client = FACTORY[name](*good_items(name))
    assert isinstance(await client.fetch_validated(**KWARGS[name]), Ok)
    calls = client._session.calls
    assert len(calls) == (2 if name == "fred" else 1)
    for _url, kwargs in calls:
        assert set(kwargs) == {"params", "allow_redirects", "timeout", "headers", "auto_decompress"}
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == aiohttp.ClientTimeout(total=10, sock_connect=5, sock_read=10)
        assert kwargs["headers"] == {"Accept-Encoding": "identity"}
        assert kwargs["auto_decompress"] is False


def test_provider_stale_threshold_is_cap_over_four() -> None:
    assert COINGECKO_PROVIDER_STALE_MS == SERIES_RULES["coingecko:btc_dominance"].max_age_ms // 4


async def test_coingecko_updated_at_boundary() -> None:
    at = NOW_MS // 1000 - 5400  # exactly 5_400_000 ms old at NOW_MS
    body = cg_body(updated_at=at)
    assert isinstance(await cg(ok(body)).fetch_validated(now_ms=NOW_MS), Ok)
    late = await cg(ok(body)).fetch_validated(now_ms=NOW_MS + 1)
    assert reason_of(late) == ("invalid_payload", "provider_stale")
    assert isinstance(await cg(ok(cg_body())).fetch_validated(now_ms=NOW_MS), Ok)  # absent: valid


async def test_keys_never_reach_logs_outcomes_or_reprs(production_logging: ConfigureFn) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    disable_layer1()  # prove the sites on their own, not the redactor
    leaky = (
        aiohttp.ClientConnectionError(f"GET https://h/x?api_key={KEY} failed"),
        ValueError(f"https://h/x?api_key={KEY}"),  # an unexpected bug: exc_log_fields only
        TimeoutError(f"api_key={KEY}"),
    )
    results: list[object] = []
    for exc in leaky:
        results.append(await fred(exc).fetch_validated())
        results.append(await whale(exc).fetch_validated(now_ms=NOW_MS))
    for status in (302, 429, 500):
        results.append(await fred(ok({}, status, {"Retry-After": "7"})).fetch_validated())
    results.append(await whale(ok(b" " * 3_000_000)).fetch_validated(now_ms=NOW_MS))
    assert all(isinstance(r, Err) for r in results)
    assert KEY not in cap.text and KEY not in repr(results)
    assert "fetch_validated_error" in cap.text and "ValueError" in cap.text  # not a vacuous capture


@pytest.mark.parametrize("name", NAMES)
async def test_an_unexpected_exception_logs_only_its_class_name(
    name: str, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=True, log_level="INFO")
    disable_layer1()  # prove the site on its own, not the redactor
    leaky = ValueError(f"https://h/x?api_key={KEY}")
    result = await FACTORY[name](leaky).fetch_validated(**KWARGS[name])
    assert reason_of(result) == ("invalid_payload", "internal_error")
    event = f"{EVENT_PREFIX[name]}.fetch_validated_error"
    rows = [r for r in json_lines(cap.text) if r.get("event") == event]
    assert len(rows) == 1 and rows[0]["level"] == "warning"
    assert rows[0]["error_type"] == "ValueError"
    assert not {"error", "exc_info", "exception", "traceback", "stack"} & set(rows[0])
    assert KEY not in cap.text and "https://h/x" not in cap.text


@pytest.mark.parametrize("name", NAMES)
async def test_ok_err_and_invalid_paths_are_silent_at_info(
    name: str, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=True, log_level="INFO")
    assert isinstance(await FACTORY[name](*good_items(name)).fetch_validated(**KWARGS[name]), Ok)
    assert isinstance(await FACTORY[name](ok({}, 500)).fetch_validated(**KWARGS[name]), Err)
    invalid = await FACTORY[name](*invalid_items(name)).fetch_validated(**KWARGS[name])
    assert reason_of(invalid)[0] == "invalid_payload"
    assert cap.lines() == []
    debug = production_logging(json_output=True, log_level="DEBUG")  # presence: DEBUG sees them
    await FACTORY[name](*good_items(name)).fetch_validated(**KWARGS[name])
    assert f"{EVENT_PREFIX[name]}.fetch_validated_ok" in debug.text
