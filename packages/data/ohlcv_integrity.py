"""OHLCV integrity and hash leaf (WP2.6, SY-26-02/03/04/08/12): pure and stdlib-only.

Integer-millisecond grid arithmetic, candle validation with closed reason codes, the 12-decimal
half-up quantiser, the canonical number text, the per-series digest and the per-run ``data_hash``.
Standard library only: no I/O, wall clock, logging, env or float (``Decimal(float)`` is rejected as
``bad_type``). Everything hashed is an int or ASCII text. ``Decimal.normalize()`` is never used
(it rounds to the context precision). Not re-exported from ``data/__init__``.

What ``data_hash`` means (QA-08, CF-24-14): equal ``data_hash`` means the same set of
``(exchange, symbol, timeframe_s)`` series, each with byte-identical canonical bars (``open_ms``
and the 12-decimal OHLCV text, Decimal exponent included). ``series_digest`` accepts only bars
that are exactly ``canonicalise_candle`` output (``not_canonical`` otherwise), so equal digests
mean equal engine inputs. It identifies the canonical bars, not the raw exchange payload: raw
values that differ only beyond 12 decimals hash equal (``rounded_values`` counts them beside the
hash). Callers include every requested series, empty ones too (n = 0), so the symbol set is part
of the hash. With the same requested window, timeframe and bar-alignment version it implies an
equal ``presence_digest``; for one symbol alone it does not (presence covers the basket's
execution window). It is never merged with the PIT ``series_digest``. It does not cover the
window, costs, parameters, seed, engine or alignment version or MTF context, and it is no
data-quality, eligibility or performance signal.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from typing import Final

__all__ = [
    "CACHEABLE_TIMEFRAMES_S",
    "OHLCV_CODES",
    "OHLCV_DATA_VERSION",
    "SCALE_DECIMALS",
    "SETTLE_BARS",
    "Candle",
    "OhlcvDataError",
    "SeriesSummary",
    "canon_decimal",
    "canon_text",
    "canonicalise_candle",
    "ceil_tf",
    "check_closed",
    "check_identifiers",
    "closed_end_ms",
    "floor_tf",
    "is_on_grid",
    "is_storable",
    "run_digest",
    "series_digest",
    "settle_end_ms",
    "storable_end_ms",
    "summarise_series",
    "tf_ms",
]

OHLCV_DATA_VERSION: Final = "wp26-1"
SETTLE_BARS: Final = 1  # SY-26-08: a code constant, never env or API
SCALE_DECIMALS: Final = 12

CODE_OFF_GRID: Final = "off_grid"
CODE_FORMING: Final = "forming_or_future"
CODE_BAD_OHLC: Final = "bad_ohlc"
CODE_BAD_VOLUME: Final = "bad_volume"
CODE_OUT_OF_RANGE: Final = "out_of_range"
CODE_NOT_FINITE: Final = "not_finite"
CODE_BAD_IDENTIFIER: Final = "bad_identifier"
CODE_DUPLICATE_SERIES: Final = "duplicate_series"
CODE_BAD_TYPE: Final = "bad_type"
CODE_NOT_INCREASING: Final = "not_increasing"
CODE_BAD_DIGEST: Final = "bad_digest"
CODE_NOT_CANONICAL: Final = "not_canonical"
OHLCV_CODES: Final[tuple[str, ...]] = (
    CODE_OFF_GRID,
    CODE_FORMING,
    CODE_BAD_OHLC,
    CODE_BAD_VOLUME,
    CODE_OUT_OF_RANGE,
    CODE_NOT_FINITE,
    CODE_BAD_IDENTIFIER,
    CODE_DUPLICATE_SERIES,
    CODE_BAD_TYPE,
    CODE_NOT_INCREASING,
    CODE_BAD_DIGEST,
    CODE_NOT_CANONICAL,
)

# SY-26-12: TimeFrame minus 1w, in seconds (pinned to strategy_engine._TIMEFRAME_SECONDS by a test).
CACHEABLE_TIMEFRAMES_S: Final[tuple[int, ...]] = (60, 180, 300, 900, 1800, 3600, 14400, 86400)

_QUANTUM: Final = Decimal("1E-12")
_CTX: Final = Context(prec=60, rounding=ROUND_HALF_UP)  # private; the default context is never used
_PRICE_LIMIT: Final = Decimal("1E16")  # NUMERIC(28,12)
_VOLUME_LIMIT: Final = Decimal("1E18")  # NUMERIC(30,12)
# SY-26-23: the same rule as the DDL CHECK (^[a-z0-9_]+$ and ^[!-.0-{}~]+/[!-.0-{}~]+$). The class
# is 0x21..0x7E minus "/" and "|". fullmatch only: ``$`` would accept a trailing newline.
_SYMBOL_CHAR: Final = "[!-.0-{}~]"
_SYMBOL_RE: Final = re.compile(_SYMBOL_CHAR + "+/" + _SYMBOL_CHAR + "+")
_EXCHANGE_RE: Final = re.compile(r"[a-z0-9_]+")
_HEX: Final = frozenset("0123456789abcdef")
_NUM_TYPES: Final = (Decimal, str, int)


class OhlcvDataError(ValueError):
    """OHLCV check failed. The message is fixed text; never carries data (CF-36)."""

    def __init__(self, code: str) -> None:
        if code not in OHLCV_CODES:
            raise ValueError("unknown ohlcv code")
        super().__init__(f"ohlcv data check failed ({code})")
        self.code: str = code

    def __reduce__(self) -> tuple[type[OhlcvDataError], tuple[str]]:
        return (type(self), (self.code,))


@dataclass(frozen=True, slots=True)
class Candle:
    """A validated candle: canonical Decimals (SY-26-03). ``open_`` avoids the banned name."""

    open_ms: int
    open_: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    exchange: str
    symbol: str
    timeframe_s: int
    n: int
    digest: str


# ---------------------------------------------------------------------------
# grid arithmetic (integer ms, epoch anchor; every cacheable tf divides a day)
# ---------------------------------------------------------------------------
def _int(value: int) -> int:
    if type(value) is not int:
        raise OhlcvDataError(CODE_BAD_TYPE)
    return value


def tf_ms(timeframe_s: int) -> int:
    if type(timeframe_s) is not int or timeframe_s not in CACHEABLE_TIMEFRAMES_S:
        raise OhlcvDataError(CODE_BAD_IDENTIFIER)
    return timeframe_s * 1000


def floor_tf(ms: int, timeframe_s: int) -> int:
    step = tf_ms(timeframe_s)
    return _int(ms) - _int(ms) % step


def ceil_tf(ms: int, timeframe_s: int) -> int:
    floor = floor_tf(ms, timeframe_s)
    return floor if floor == ms else floor + tf_ms(timeframe_s)


def is_on_grid(ms: int, timeframe_s: int) -> bool:
    return _int(ms) % tf_ms(timeframe_s) == 0


def closed_end_ms(end_ms: int, now_ms: int, timeframe_s: int) -> int:
    """``min(end, floor_tf(now))``: the closed-bar clamp (exclusive end)."""
    return min(_int(end_ms), floor_tf(now_ms, timeframe_s))


def settle_end_ms(now_ms: int, timeframe_s: int) -> int:
    """``floor_tf(now) - SETTLE_BARS * tf``: nothing at or after it is stored (SY-26-08)."""
    return floor_tf(now_ms, timeframe_s) - SETTLE_BARS * tf_ms(timeframe_s)


def storable_end_ms(end_ms: int, now_ms: int, timeframe_s: int) -> int:
    return min(closed_end_ms(end_ms, now_ms, timeframe_s), settle_end_ms(now_ms, timeframe_s))


def is_storable(open_ms: int, now_ms: int, timeframe_s: int) -> bool:
    """A bar is storable iff ``open + tf <= floor_tf(now) - SETTLE_BARS * tf``."""
    return _int(open_ms) + tf_ms(timeframe_s) <= settle_end_ms(now_ms, timeframe_s)


def check_closed(open_ms: int, now_ms: int, timeframe_s: int) -> None:
    """Raise ``forming_or_future`` unless the bar is closed: ``open + tf <= floor_tf(now)``."""
    if _int(open_ms) + tf_ms(timeframe_s) > floor_tf(now_ms, timeframe_s):
        raise OhlcvDataError(CODE_FORMING)


# ---------------------------------------------------------------------------
# identifiers (SY-26-23; the DB CHECK accepts exactly this set)
# ---------------------------------------------------------------------------
def _valid_exchange(value: object) -> bool:
    return type(value) is str and len(value) <= 32 and _EXCHANGE_RE.fullmatch(value) is not None


def _valid_symbol(value: object) -> bool:
    return type(value) is str and len(value) <= 32 and _SYMBOL_RE.fullmatch(value) is not None


def check_identifiers(exchange: str, symbol: str, timeframe_s: int) -> None:
    ok = _valid_exchange(exchange) and _valid_symbol(symbol)
    if not ok:
        raise OhlcvDataError(CODE_BAD_IDENTIFIER)
    tf_ms(timeframe_s)


# ---------------------------------------------------------------------------
# numbers (SY-26-02 / SY-26-03)
# ---------------------------------------------------------------------------
def _decimal(value: Decimal | str | int) -> Decimal:
    if type(value) is bool or not isinstance(value, _NUM_TYPES):
        raise OhlcvDataError(CODE_BAD_TYPE)  # float, bool, None, ...
    try:
        number = value if isinstance(value, Decimal) else Decimal(value)
    except InvalidOperation:
        raise OhlcvDataError(CODE_BAD_TYPE) from None
    if not number.is_finite():
        raise OhlcvDataError(CODE_NOT_FINITE)
    return number


def _quantise(number: Decimal, limit: Decimal) -> Decimal:
    # copy_abs is exact; abs() would round to the context precision
    if number.copy_abs() >= limit:
        raise OhlcvDataError(CODE_OUT_OF_RANGE)
    quantised = number.quantize(_QUANTUM, context=_CTX)
    if quantised.copy_abs() >= limit:  # a value that rounds up to the limit
        raise OhlcvDataError(CODE_OUT_OF_RANGE)
    return quantised


def _text(quantised: Decimal) -> str:
    text = format(quantised, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def canon_text(value: Decimal | str | int) -> str:
    """SY-26-03: quantise to 12 dp half-up, fixed-point text, zeros stripped, ``-0`` -> ``0``."""
    return _text(_quantise(_decimal(value), _VOLUME_LIMIT))


def canon_decimal(value: Decimal | str | int) -> Decimal:
    """The only Decimal form handed to ``OHLCVBar`` on every path."""
    return Decimal(canon_text(value))


def _check_bar(o: Decimal, h: Decimal, lo: Decimal, cl: Decimal, vol: Decimal) -> None:
    """The ``OHLCVBar`` invariants on canonical values; shared by both entry points."""
    if not (lo > 0 and lo <= o <= h and lo <= cl <= h):
        raise OhlcvDataError(CODE_BAD_OHLC)
    if vol < 0:
        raise OhlcvDataError(CODE_BAD_VOLUME)


def canonicalise_candle(
    open_ms: int,
    open_: Decimal | str | int,
    high: Decimal | str | int,
    low: Decimal | str | int,
    close: Decimal | str | int,
    volume: Decimal | str | int,
    *,
    timeframe_s: int,
) -> tuple[Candle, int]:
    """Validate and canonicalise one raw row; returns ``(candle, rounded_values)``."""
    if not is_on_grid(open_ms, timeframe_s) or open_ms < 0:
        raise OhlcvDataError(CODE_OFF_GRID)
    raw = [_decimal(v) for v in (open_, high, low, close, volume)]
    quantised = [_quantise(r, _PRICE_LIMIT) for r in raw[:4]]
    quantised.append(_quantise(raw[4], _VOLUME_LIMIT))
    rounded = sum(1 for r, q in zip(raw, quantised, strict=True) if r != q)
    q_open, q_high, q_low, q_close, q_volume = quantised
    _check_bar(q_open, q_high, q_low, q_close, q_volume)
    candle = Candle(open_ms, *(Decimal(_text(q)) for q in quantised))
    return candle, rounded


# ---------------------------------------------------------------------------
# hashes (SY-26-04)
# ---------------------------------------------------------------------------
def _hashed_numbers(c: Candle) -> str:
    """The bar's number text. ``not_canonical`` unless every field is the exact Decimal that
    ``canonicalise_candle`` returns (value and exponent): equal digests mean equal engine inputs."""
    fields = (c.open_, c.high, c.low, c.close, c.volume)
    texts: list[str] = []
    for value in fields:
        if type(value) is not Decimal:
            raise OhlcvDataError(CODE_NOT_CANONICAL)
        text = canon_text(value)
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int) or exponent > 0 or format(value, "f") != text:
            raise OhlcvDataError(CODE_NOT_CANONICAL)
        texts.append(text)
    o, h, lo, cl, vol = fields
    _check_bar(o, h, lo, cl, vol)
    if h >= _PRICE_LIMIT:
        raise OhlcvDataError(CODE_OUT_OF_RANGE)
    return "|".join(texts)


def series_digest(exchange: str, symbol: str, timeframe_s: int, candles: Sequence[Candle]) -> str:
    """sha256 over the header and one line per bar, strictly increasing ``open_ms``. Every candle
    must be a ``canonicalise_candle`` output (``not_canonical`` otherwise)."""
    check_identifiers(exchange, symbol, timeframe_s)
    digest = hashlib.sha256(
        f"ohlcv-{OHLCV_DATA_VERSION}|series|{exchange}|{symbol}|{timeframe_s}\n".encode("ascii")
    )
    previous = -1
    for c in candles:
        if not is_on_grid(c.open_ms, timeframe_s) or c.open_ms < 0:
            raise OhlcvDataError(CODE_OFF_GRID)
        if c.open_ms <= previous:
            raise OhlcvDataError(CODE_NOT_INCREASING)
        previous = c.open_ms
        numbers = _hashed_numbers(c)
        digest.update(f"{c.open_ms}|{numbers}\n".encode("ascii"))
    return digest.hexdigest()


def summarise_series(
    exchange: str, symbol: str, timeframe_s: int, candles: Sequence[Candle]
) -> SeriesSummary:
    digest = series_digest(exchange, symbol, timeframe_s, candles)
    return SeriesSummary(exchange, symbol, timeframe_s, len(candles), digest)


def run_digest(series: Iterable[SeriesSummary]) -> str:
    """The per-run ``data_hash``. Lines sorted by the TUPLE (exchange, symbol, timeframe_s), not by
    line bytes (``BTC/EURC|`` would sort before ``BTC/EUR|``). The requested range is NOT hashed."""
    rows = list(series)
    for s in rows:
        check_identifiers(s.exchange, s.symbol, s.timeframe_s)
        if (
            type(s.n) is not int
            or s.n < 0
            or type(s.digest) is not str
            or len(s.digest) != 64
            or not set(s.digest) <= _HEX
        ):
            raise OhlcvDataError(CODE_BAD_DIGEST)
    rows.sort(key=lambda s: (s.exchange, s.symbol, s.timeframe_s))
    digest = hashlib.sha256(f"ohlcv-{OHLCV_DATA_VERSION}|run\n".encode("ascii"))
    previous: tuple[str, str, int] | None = None
    for s in rows:
        key = (s.exchange, s.symbol, s.timeframe_s)
        if key == previous:
            raise OhlcvDataError(CODE_DUPLICATE_SERIES)
        previous = key
        line = f"{s.exchange}|{s.symbol}|{s.timeframe_s}|{s.n}|{s.digest}\n"
        digest.update(line.encode("ascii"))
    return digest.hexdigest()
