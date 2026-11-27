"""
packages/data/provider_bands.py
--------------------------------
Plausibility bands and number grammar for the live provider fetch layer (CF-24-01 P2).

Stdlib only: no logging, env, clock or I/O. The band block below is the single place the
risk reviewer signs off (SY-01-10, SY-01-22b); it is value-pinned in
``tests/unit/test_cf2401_ast_pins.py``. Every rejection is a :class:`Reject` carrying a
closed reason code; the message is the code, never payload text.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Final, Literal

# ---------------------------------------------------------------------------
# BANDS (risk sign-off block)
# ---------------------------------------------------------------------------
FGI_MIN: Final = 0
FGI_MAX: Final = 100
DOMINANCE_MIN_EXCLUSIVE: Final = Decimal("0")  # 0 < v
DOMINANCE_MAX: Final = Decimal("100")  # v <= 100
T10Y2Y_MIN: Final = Decimal("-10")
T10Y2Y_MAX: Final = Decimal("10")
FEDFUNDS_MIN: Final = Decimal("-5")
FEDFUNDS_MAX: Final = Decimal("30")
CHANGE_24H_MIN: Final = Decimal("-100")
CHANGE_24H_MAX: Final = Decimal("1000")
WHALE_ABS_MAX: Final = Decimal("1e12")  # |net_flow| and each amount_usd
WHALE_PAGE_LIMIT: Final = 100  # count 0..limit; a full page is truncated
FRED_PAGE_LIMIT: Final = 5
FGI_PAGE_LIMIT: Final = 30
# cap/4 of SERIES_RULES["coingecko:btc_dominance"].max_age_ms (6 h); equality-tested.
COINGECKO_PROVIDER_STALE_MS: Final = 5_400_000
# (inclusive upper bound, label): the 5 fixed FGI classifications, derived from the value.
FGI_LABEL_BOUNDS: Final = (
    (24, "Extreme Fear"),
    (44, "Fear"),
    (55, "Neutral"),
    (75, "Greed"),
    (100, "Extreme Greed"),
)
# ---------------------------------------------------------------------------

Reason = Literal[
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

_NUMBER: Final = re.compile(r"-?[0-9]{1,18}(\.[0-9]{1,12})?")
_UNSIGNED: Final = re.compile(r"[0-9]{1,12}")
_ISO_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_UNIX_MAX: Final = 4_102_444_800  # 2100-01-01


class Reject(ValueError):
    """A validator refusal. The message is the closed reason code."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason)
        self.reason: Reason = reason


def mapping(obj: object) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise Reject("bad_shape")
    return obj


def sequence(obj: object) -> list[object]:
    if not isinstance(obj, list):
        raise Reject("bad_shape")
    return obj


def require(obj: object, key: str) -> object:
    body = mapping(obj)
    if key not in body:
        raise Reject("missing_key")
    return body[key]


def to_decimal(raw: object) -> Decimal:
    """Exact, finite Decimal from a JSON int, Decimal or ASCII numeric string (never a float)."""
    if isinstance(raw, bool):
        raise Reject("bad_type")
    if isinstance(raw, Decimal):
        value = raw
    elif isinstance(raw, int):
        value = Decimal(raw)
    elif isinstance(raw, str):
        if _NUMBER.fullmatch(raw) is None:
            raise Reject("bad_number")
        value = Decimal(raw)
    else:
        raise Reject("bad_type")
    if not value.is_finite():
        raise Reject("non_finite")
    return value


def in_band(
    value: Decimal, low: Decimal | int, high: Decimal | int, *, low_open: bool = False
) -> Decimal:
    if value < low or value > high or (low_open and value == low):
        raise Reject("out_of_band")
    return value


def unix_seconds(raw: object) -> int:
    if isinstance(raw, bool):
        raise Reject("bad_type")
    if isinstance(raw, str) and _UNSIGNED.fullmatch(raw) is not None:
        raw = int(raw)
    if not isinstance(raw, int):
        raise Reject("bad_type")
    if not 0 < raw <= _UNIX_MAX:
        raise Reject("bad_timestamp")
    return raw


def iso_date(raw: object) -> date:
    if not isinstance(raw, str) or _ISO_DATE.fullmatch(raw) is None:
        raise Reject("bad_timestamp")
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise Reject("bad_timestamp") from None


def fgi_label(value: int) -> str:
    for upper, label in FGI_LABEL_BOUNDS:
        if value <= upper:
            return label
    raise Reject("out_of_band")
