"""
packages/data/provider_http.py
-------------------------------
Transport for the live provider fetch layer (CF-24-01 P2, SY-01-10).

Transport only: status classification, size caps (Content-Length and a streamed read),
strict JSON parsing and closed outcome codes. Imports stdlib and ``aiohttp`` only and holds
no logger by design (AST-pinned): exception text never leaves this module, because aiohttp
messages embed the request URL, including a query ``api_key``. Callers log the closed
``Err`` fields. Bands and field validation live in ``data.provider_bands`` and the clients.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Generic, Literal, TypeVar, get_args

import aiohttp

__all__ = ["OUTCOMES", "Err", "ErrReason", "Ok", "Outcome", "fetch_json"]

Outcome = Literal[
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
]
OUTCOMES: Final[tuple[str, ...]] = get_args(Outcome)

#: Closed ``Err.reason`` codes. The last 11 repeat ``provider_bands.Reason`` verbatim (this module
#: may not import it, AST pin 1); ``test_cf2401_provider_fetch`` pins the exact union.
ErrReason = Literal[
    "bad_status",
    "bad_content_type",
    "bad_encoding",
    "bad_length",
    "content_length",
    "stream",
    "payload",
    "bad_json",
    "internal_error",
    "bad_type",
    "bad_number",
    "non_finite",
    "out_of_band",
    "missing_key",
    "bad_shape",
    "bad_timestamp",
    "truncated",
    "provider_stale",
    "provider_error",
    "no_observation",
]

#: response size caps in bytes (SEC-02)
MAX_BYTES_FGI: Final = 64 * 1024
MAX_BYTES_COINGECKO: Final = 256 * 1024
MAX_BYTES_FRED: Final = 64 * 1024
MAX_BYTES_WHALE: Final = 2 * 1024 * 1024

_TIMEOUT: Final = aiohttp.ClientTimeout(total=10, sock_connect=5, sock_read=10)
_CHUNK: Final = 8192
# Retry-After grammar (RFC 9110): delta-seconds or IMF-fixdate. ``[0-9]``, never ``\d`` (which
# matches non-ASCII digits).
_RETRY_AFTER_SECONDS: Final = re.compile(r"[0-9]{1,6}")
_RETRY_AFTER_DATE: Final = re.compile(
    r"[A-Z][a-z]{2}, [0-9]{2} [A-Z][a-z]{2} [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT"
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Err:
    """A failed fetch. Only closed codes, a class name and a bounded Retry-After header."""

    outcome: Outcome
    reason: ErrReason | None = None
    retry_after: str | None = None
    error_type: str | None = None


def _reject_constant(_token: str) -> object:
    raise ValueError("non_finite_token")


def _retry_after(headers: Any) -> str | None:  # noqa: ANN401
    value = headers.get("Retry-After")
    if isinstance(value, str) and (
        _RETRY_AFTER_SECONDS.fullmatch(value) or _RETRY_AFTER_DATE.fullmatch(value)
    ):
        return value
    return None


def _status_error(status: object, headers: Any) -> Err | None:  # noqa: ANN401
    if type(status) is not int or not 100 <= status <= 599:
        return Err("invalid_payload", "bad_status")
    if 200 <= status < 300:
        return None
    if 300 <= status < 400:
        return Err("http_3xx")
    if status in (401, 403):
        return Err("auth_failed")
    if status == 429:
        return Err("http_429", retry_after=_retry_after(headers))
    if status >= 500:
        return Err("http_5xx", retry_after=_retry_after(headers))
    if status >= 400:
        return Err("http_4xx")
    return Err("invalid_payload", "bad_status")


async def _read_capped(content: Any, max_bytes: int) -> bytes | None:  # noqa: ANN401
    """Read at most ``max_bytes`` + 1 bytes in small chunks; ``None`` when over the cap."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await content.read(min(_CHUNK, max_bytes + 1 - total))
        if not chunk:
            return b"".join(parts)
        total += len(chunk)
        if total > max_bytes:
            return None
        parts.append(chunk)


async def fetch_json(
    session: Any,  # noqa: ANN401 -- duck-typed aiohttp.ClientSession
    url: str,
    *,
    params: dict[str, Any] | None,
    max_bytes: int,
    allow_text_plain: bool = False,
) -> Ok[Any] | Err:
    """GET ``url`` and return the parsed JSON (floats as ``Decimal``) or a closed ``Err``."""
    try:
        # Redirects are never followed (aiohttp would follow https->http and cross-host hops), and
        # the body is never content-decoded: an identity encoding is requested and enforced, so the
        # size cap bounds the bytes that are parsed (no decompression bomb).
        async with session.get(
            url,
            params=params,
            allow_redirects=False,
            timeout=_TIMEOUT,
            headers={"Accept-Encoding": "identity"},
            auto_decompress=False,
        ) as response:
            failure = _status_error(response.status, response.headers)
            if failure is not None:
                return failure
            encoding = str(response.headers.get("Content-Encoding", "identity")).strip().lower()
            if encoding != "identity":
                return Err("invalid_payload", "bad_encoding")
            ctype = str(response.headers.get("Content-Type", "")).split(";")[0].strip().lower()
            is_json = ctype == "application/json" or ctype.endswith("+json")
            if not is_json and not (allow_text_plain and ctype == "text/plain"):
                return Err("invalid_payload", "bad_content_type")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                if not (isinstance(declared, str) and declared.isascii() and declared.isdigit()):
                    return Err("invalid_payload", "bad_length")
                if len(declared) > 18 or int(declared) > max_bytes:
                    return Err("oversize", "content_length")
            body = await _read_capped(response.content, max_bytes)
    except TimeoutError as exc:
        return Err("timeout", error_type=type(exc).__name__)
    except (aiohttp.ClientHttpProxyError, aiohttp.ClientProxyConnectionError) as exc:
        return Err("egress_blocked", error_type=type(exc).__name__)
    except aiohttp.ClientPayloadError as exc:
        return Err("invalid_payload", "payload", error_type=type(exc).__name__)
    except aiohttp.ClientError as exc:
        return Err("connect_error", error_type=type(exc).__name__)
    if body is None:
        return Err("oversize", "stream")
    try:
        return Ok(
            json.loads(body.decode("utf-8"), parse_float=Decimal, parse_constant=_reject_constant)
        )
    except (ValueError, RecursionError, ArithmeticError):  # incl. decimal.InvalidOperation
        return Err("invalid_payload", "bad_json")
