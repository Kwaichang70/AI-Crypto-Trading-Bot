"""Point-in-time context series (WP2.4, SY-24): a pure, stdlib-only leaf.

A decision at a bar's ``close_i`` may only see an observation whose ``available_ms`` is
at or before ``close_i`` (inclusive; one millisecond later is invisible). The lag between
"observed" and "usable" is applied exactly once, at ingestion, and stored in
``available_ms`` (SY-24-03). The join (:meth:`PitSeries.asof`, :meth:`PitSeries.coverage_ppm`)
has NO lag parameter and never calls :func:`available_at_estimated`.

Join rules (SY-24-04):

* Selection: the greatest ``observed_ms`` among the visible rows; ties are broken by the
  greatest ``available_ms`` (a visible revision wins).
* Staleness: ``decision - first_available(selected observed_ms) <= max_age_ms`` (inclusive).
  The age counts from the FIRST availability of that ``observed_ms``, so a revision never
  refreshes an old value.
* "N days ago" is ``asof(decision - N days)``: what was visible then, evaluated then.

Purity: standard library only; no logging, environment, wall clock, I/O or metrics.
Times are integer milliseconds since the epoch, values are ``Decimal`` and everything
hashed is text built from those (never a float). Errors are :class:`PitError` with a
fixed message and a closed code (CF-36: never data in a message).
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import ClassVar, Final, Literal, get_args

__all__ = [
    "BACK_7D_MS",
    "FIELD_MAP",
    "MIN_COVERAGE_PPM",
    "NOT_SERVED_FIELDS",
    "PIT_VERSION",
    "SERIES_RULES",
    "Basis",
    "FieldSpec",
    "Observation",
    "PitError",
    "PitSeries",
    "SeriesRule",
    "available_at_estimated",
    "available_at_vintage",
    "parse_jsonl_lines",
    "series_digest",
    "to_ms",
]

PIT_VERSION: Final = "wp24-1"

Basis = Literal["recorded", "estimated_lag", "vintage"]
_BASES: Final[tuple[str, ...]] = get_args(Basis)

_CODE_NAIVE: Final = "pit_naive_time"
_CODE_BEFORE: Final = "pit_available_before_observed"
_CODE_DUPLICATE: Final = "pit_duplicate"
_CODE_BAD_VALUE: Final = "pit_bad_value"
_CODE_UNKNOWN: Final = "pit_unknown_series"

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MS: Final = timedelta(milliseconds=1)
_HOUR_MS: Final = 3_600_000
_DAY_MS: Final = 86_400_000
_EPOCH_DAY_WEEKDAY: Final = 3  # 1970-01-01 was a Thursday (Monday == 0)
_VALUE_QUANTUM: Final = Decimal("1e-8")  # NUMERIC(24,8) in the Phase B store
_VALUE_LIMIT: Final = Decimal("1e16")  # |value| < 1e16 (24 digits, 8 of them decimals)

# ---------------------------------------------------------------------------
# SERIES RULES  (SY-24-04, UD-5 option (a) = 24h FGI lag)
#
# The ONLY place the per-series numbers live: code constants, never environment,
# never an API field. ``lag_rule`` names the ingestion rule (CF-23-18: explicit lag).
# ---------------------------------------------------------------------------
_FGI: Final = "alternative_me:fgi"
_T10Y2Y: Final = "fred:T10Y2Y"
_DOMINANCE: Final = "coingecko:btc_dominance"
_WHALE: Final = "whale:net_flow_1h"

_FGI_LAG_MS: Final = 24 * _HOUR_MS  # observed (day D 00:00 UTC) + 24h
_FGI_MAX_AGE_MS: Final = 48 * _HOUR_MS
_FRED_BUSINESS_DAYS: Final = 2  # fallback: observed + 2 business days (Mon-Fri) ...
_FRED_SLACK_MS: Final = _DAY_MS  # ... + 1 day; the vintage path adds the same 1 day
_FRED_MAX_AGE_MS: Final = 7 * _DAY_MS
_DOMINANCE_MAX_AGE_MS: Final = 6 * _HOUR_MS
_WHALE_MAX_AGE_MS: Final = 2 * _HOUR_MS
# Bases an ingested row of each series may carry (SY-24-04 table, column `available_at`).
_ALLOWED_BASES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        _FGI: frozenset({"estimated_lag", "recorded"}),
        _T10Y2Y: frozenset({"vintage", "estimated_lag"}),
        _DOMINANCE: frozenset({"recorded"}),
        _WHALE: frozenset({"recorded"}),
    }
)
# ---------------------------------------------------------------------------

BACK_7D_MS: Final = 7 * _DAY_MS
MIN_COVERAGE_PPM: Final = 990_000


@dataclass(frozen=True, slots=True)
class SeriesRule:
    """One served series: the fields it feeds, its ingestion lag rule and its staleness cap."""

    series_id: str
    fields: tuple[str, ...]
    lag_rule: str
    max_age_ms: int
    in_backtest: bool


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """A context field: the series that serves it and how far back it looks (0 = now)."""

    series_id: str
    back_ms: int


SERIES_RULES: Final[Mapping[str, SeriesRule]] = MappingProxyType(
    {
        _FGI: SeriesRule(
            _FGI,
            ("fear_greed_index", "fear_greed_index_7d_ago"),
            "observed+24h|recorded",
            _FGI_MAX_AGE_MS,
            True,
        ),
        _T10Y2Y: SeriesRule(
            _T10Y2Y,
            ("yield_curve_spread",),
            "vintage:realtime_start+1d|observed+2bd+1d",
            _FRED_MAX_AGE_MS,
            True,
        ),
        _DOMINANCE: SeriesRule(
            _DOMINANCE,
            ("btc_dominance", "btc_dominance_7d_ago"),
            "recorded",
            _DOMINANCE_MAX_AGE_MS,
            True,
        ),
        _WHALE: SeriesRule(_WHALE, ("whale_net_flow",), "recorded", _WHALE_MAX_AGE_MS, True),
    }
)

FIELD_MAP: Final[Mapping[str, FieldSpec]] = MappingProxyType(
    {
        "fear_greed_index": FieldSpec(_FGI, 0),
        "fear_greed_index_7d_ago": FieldSpec(_FGI, BACK_7D_MS),
        "yield_curve_spread": FieldSpec(_T10Y2Y, 0),
        "btc_dominance": FieldSpec(_DOMINANCE, 0),
        "btc_dominance_7d_ago": FieldSpec(_DOMINANCE, BACK_7D_MS),
        "whale_net_flow": FieldSpec(_WHALE, 0),
    }
)

# Context fields that no backtest serves (no consumer, SY-24-04 / CF-24-13): status ``not_served``.
NOT_SERVED_FIELDS: Final[frozenset[str]] = frozenset(
    {"fed_funds_rate", "market_cap_change_24h", "total_volume_change_24h"}
)


class PitError(ValueError):
    """Point-in-time check failed. The message is fixed text; never carries data (CF-36)."""

    CODES: ClassVar[tuple[str, ...]] = (
        _CODE_NAIVE,
        _CODE_BEFORE,
        _CODE_DUPLICATE,
        _CODE_BAD_VALUE,
        _CODE_UNKNOWN,
    )

    def __init__(self, code: str) -> None:
        if code not in PitError.CODES:
            raise ValueError("unknown point-in-time code")
        super().__init__(f"point-in-time check failed ({code})")
        self.code: str = code

    def __reduce__(self) -> tuple[type[PitError], tuple[str]]:
        return (type(self), (self.code,))


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------


def to_ms(dt: datetime) -> int:
    """Integer epoch milliseconds (floor) of an aware datetime. Naive -> ``pit_naive_time``."""
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise PitError(_CODE_NAIVE)
    return (dt - _EPOCH) // _MS


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_business_day(epoch_day: int) -> bool:
    return (epoch_day + _EPOCH_DAY_WEEKDAY) % 7 < 5


# ---------------------------------------------------------------------------
# ingestion-side availability (SY-24-03: NEVER called from the join)
# ---------------------------------------------------------------------------


def available_at_estimated(series_id: str, observed_ms: int) -> int:
    """Estimated ``available_ms`` for a series without a recorded stamp (ingestion only).

    * ``alternative_me:fgi``: ``observed + 24h`` (UD-5 (a)).
    * ``fred:T10Y2Y``: ``observed + 2 business days (Mon-Fri) + 1 day``. Holidays are not
      modelled: a wrong-early stamp is look-ahead, a wrong-late one only conservative, so
      the fallback counts one business day more than the first release needs.

    ``observed_ms`` must be day D 00:00 UTC (SY-24-04): any other instant, such as a local
    midnight east of UTC, raises ``pit_bad_value`` (QR-08), because it would stamp the value
    early. Recorded-only series (dominance, whale) and unknown ids raise
    ``pit_unknown_series``.
    """
    if not _is_int(observed_ms) or observed_ms % _DAY_MS != 0:  # QR-08: day D 00:00 UTC only
        raise PitError(_CODE_BAD_VALUE)
    if series_id == _FGI:
        return observed_ms + _FGI_LAG_MS
    if series_id == _T10Y2Y:
        day = observed_ms // _DAY_MS
        remaining = _FRED_BUSINESS_DAYS
        cursor = day
        while remaining > 0:
            cursor += 1
            if _is_business_day(cursor):
                remaining -= 1
        return observed_ms + (cursor - day) * _DAY_MS + _FRED_SLACK_MS
    raise PitError(_CODE_UNKNOWN)


def available_at_vintage(realtime_start_ms: int) -> int:
    """FRED/ALFRED vintage path: ``realtime_start`` (00:00 UTC) + 1 day (ingestion only).

    ``realtime_start_ms`` must be 00:00 UTC; anything else raises ``pit_bad_value`` (QR-08).
    """
    if not _is_int(realtime_start_ms) or realtime_start_ms % _DAY_MS != 0:  # QR-08
        raise PitError(_CODE_BAD_VALUE)
    return realtime_start_ms + _FRED_SLACK_MS


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


def _value_text(value: Decimal) -> str:
    """Canonical 8-decimal text of a validated value (``-0`` is written as ``0``)."""
    text = format(value.quantize(_VALUE_QUANTUM), "f")
    return text[1:] if value == 0 and text.startswith("-") else text


@dataclass(frozen=True, slots=True)
class Observation:
    """One observation. Validated at construction; ``value`` is a finite 8-decimal ``Decimal``."""

    observed_ms: int
    available_ms: int
    value: Decimal
    basis: Basis

    def __post_init__(self) -> None:
        if not _is_int(self.observed_ms) or not _is_int(self.available_ms):
            raise PitError(_CODE_BAD_VALUE)
        if self.available_ms < self.observed_ms:
            raise PitError(_CODE_BEFORE)
        if type(self.value) is not Decimal or not self.value.is_finite():
            raise PitError(_CODE_BAD_VALUE)
        if self.value.copy_abs() >= _VALUE_LIMIT:  # quiet: no context, no Overflow
            raise PitError(_CODE_BAD_VALUE)
        try:
            exact = self.value.quantize(_VALUE_QUANTUM) == self.value
        except InvalidOperation:
            exact = False
        if not exact:  # more than 8 decimals would not survive NUMERIC(24,8)
            raise PitError(_CODE_BAD_VALUE)
        if self.basis not in _BASES:
            raise PitError(_CODE_BAD_VALUE)


@dataclass(frozen=True, slots=True)
class PitSeries:
    """One series, validated and sorted at construction; the join is O(log n)."""

    series_id: str
    observations: tuple[Observation, ...]
    _avail: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _best: tuple[Observation, ...] = field(init=False, repr=False, compare=False)
    _first_avail: Mapping[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sid = self.series_id
        if type(sid) is not str or sid == "":
            raise PitError(_CODE_BAD_VALUE)
        if not (sid.isascii() and sid.isprintable()) or "|" in sid:  # digest line format
            raise PitError(_CODE_BAD_VALUE)
        rows = tuple(self.observations)
        if not all(type(o) is Observation for o in rows):
            raise PitError(_CODE_BAD_VALUE)
        ordered = tuple(sorted(rows, key=lambda o: (o.observed_ms, o.available_ms)))
        keys = {(o.observed_ms, o.available_ms) for o in ordered}
        if len(keys) != len(ordered):
            raise PitError(_CODE_DUPLICATE)
        if self.series_id == _FGI and any(o.value != o.value.to_integral_value() for o in ordered):
            raise PitError(_CODE_BAD_VALUE)  # the Fear & Greed index is an integer
        by_avail = sorted(ordered, key=lambda o: (o.available_ms, o.observed_ms))
        best: list[Observation] = []
        for obs in by_avail:
            if best and (best[-1].observed_ms, best[-1].available_ms) > (
                obs.observed_ms,
                obs.available_ms,
            ):
                best.append(best[-1])
            else:
                best.append(obs)
        first: dict[int, int] = {}
        for obs in by_avail:
            first.setdefault(obs.observed_ms, obs.available_ms)
        object.__setattr__(self, "observations", ordered)
        object.__setattr__(self, "_avail", tuple(o.available_ms for o in by_avail))
        object.__setattr__(self, "_best", tuple(best))
        object.__setattr__(self, "_first_avail", MappingProxyType(first))

    def asof(self, decision_ms: int, *, max_age_ms: int) -> Decimal | None:
        """The value visible at ``decision_ms`` (``available_ms <= decision_ms``), or None.

        None when nothing is visible yet or the selected value is older than ``max_age_ms``
        counted from its first availability (inclusive).
        """
        if not _is_int(decision_ms) or not _is_int(max_age_ms) or max_age_ms < 0:
            raise PitError(_CODE_BAD_VALUE)
        visible = bisect_right(self._avail, decision_ms)
        if visible == 0:
            return None
        chosen = self._best[visible - 1]
        if decision_ms - self._first_avail[chosen.observed_ms] > max_age_ms:
            return None
        return chosen.value

    def coverage_ppm(self, decision_ms: Sequence[int], *, back_ms: int, max_age_ms: int) -> int:
        """Parts per million (floor) of ``decision_ms`` with a value now AND ``back_ms`` earlier.

        ``back_ms == 0`` asks for the current value only. No decisions -> 0.
        """
        if not _is_int(back_ms) or back_ms < 0:
            raise PitError(_CODE_BAD_VALUE)
        total = len(decision_ms)
        if total == 0:
            return 0
        covered = 0
        for decision in decision_ms:
            if self.asof(decision, max_age_ms=max_age_ms) is None:
                continue
            if back_ms > 0 and self.asof(decision - back_ms, max_age_ms=max_age_ms) is None:
                continue
            covered += 1
        return covered * 1_000_000 // total


# ---------------------------------------------------------------------------
# digest and JSONL
# ---------------------------------------------------------------------------


def series_digest(series: Iterable[PitSeries]) -> str:
    """sha256 hex over the sorted ``series_id|observed_ms|available_ms|value_text|basis`` lines.

    Independent of input order. A series id may appear only once (``pit_duplicate``).
    """
    seen: set[str] = set()
    rows: list[tuple[str, int, int, str, str]] = []
    for item in series:
        if item.series_id in seen:
            raise PitError(_CODE_DUPLICATE)
        seen.add(item.series_id)
        rows.extend(
            (item.series_id, o.observed_ms, o.available_ms, _value_text(o.value), o.basis)
            for o in item.observations
        )
    rows.sort()
    digest = hashlib.sha256()
    for sid, observed, available, text, basis in rows:
        digest.update(f"{sid}|{observed}|{available}|{text}|{basis}\n".encode("ascii"))
    return digest.hexdigest()


_JSONL_KEYS: Final = frozenset({"series_id", "observed_at", "available_at", "value", "basis"})


def _parse_instant(raw: object) -> int:
    if type(raw) is not str:
        raise PitError(_CODE_BAD_VALUE)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise PitError(_CODE_BAD_VALUE) from None
    return to_ms(parsed)


def _parse_value(raw: object) -> Decimal:
    if type(raw) is Decimal:
        return raw
    if type(raw) is str:
        try:
            return Decimal(raw)
        except ArithmeticError:
            raise PitError(_CODE_BAD_VALUE) from None
    raise PitError(_CODE_BAD_VALUE)


def parse_jsonl_lines(lines: Iterable[str]) -> dict[str, PitSeries]:
    """Parse JSONL observations into series (pure: the caller opens the file).

    Each non-blank line is an object with exactly the keys ``series_id`` (a known id in
    :data:`SERIES_RULES`), ``observed_at`` and ``available_at`` (ISO-8601 with an offset),
    ``value`` (a decimal string or JSON number) and ``basis``. Any failure raises
    :class:`PitError`; the line content is never echoed.

    The ``basis`` must be allowed for the series (:data:`_ALLOWED_BASES`, SY-24-04), else
    ``pit_bad_value``. The stored ``available_at`` is validated against the SY-24-03 lag
    floor, never rewritten: ``estimated_lag`` rows must be at or after
    :func:`available_at_estimated`, ``vintage`` rows at or after :func:`available_at_vintage`
    of ``observed_at`` (both therefore need a 00:00 UTC ``observed_at``), and ``recorded``
    rows at or after ``observed_at``. A stamp below the floor raises
    ``pit_available_before_observed`` (QR-09).
    """
    grouped: dict[str, list[Observation]] = {}
    for line in lines:
        if type(line) is not str:
            raise PitError(_CODE_BAD_VALUE)
        if line.strip() == "":
            continue
        try:
            raw = json.loads(line, parse_float=Decimal, parse_int=Decimal)
        except (ValueError, RecursionError, ArithmeticError):
            raise PitError(_CODE_BAD_VALUE) from None
        if type(raw) is not dict or set(raw) != _JSONL_KEYS:
            raise PitError(_CODE_BAD_VALUE)
        sid = raw["series_id"]
        if type(sid) is not str:
            raise PitError(_CODE_BAD_VALUE)
        if sid not in SERIES_RULES:
            raise PitError(_CODE_UNKNOWN)
        basis = raw["basis"]
        if type(basis) is not str or basis not in _BASES:
            raise PitError(_CODE_BAD_VALUE)
        if basis not in _ALLOWED_BASES[sid]:
            raise PitError(_CODE_BAD_VALUE)
        obs = Observation(
            _parse_instant(raw["observed_at"]),
            _parse_instant(raw["available_at"]),
            _parse_value(raw["value"]),
            basis,  # type: ignore[arg-type]
        )
        # SY-24-03 lag floor: validate the stored stamp, never rewrite it.
        if basis == "estimated_lag":
            floor = available_at_estimated(sid, obs.observed_ms)
        elif basis == "vintage":
            floor = available_at_vintage(obs.observed_ms)  # realtime_start >= observed
        else:
            floor = obs.observed_ms  # recorded: a true sighting (SY-24-09)
        if obs.available_ms < floor:
            raise PitError(_CODE_BEFORE)
        grouped.setdefault(sid, []).append(obs)
    return {sid: PitSeries(sid, tuple(rows)) for sid, rows in grouped.items()}
