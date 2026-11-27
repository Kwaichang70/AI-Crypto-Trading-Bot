"""Live context book, converters and consumer registry (CF-24-01 P3; SY-01-07/14/17, SY-01-21a/f/j).

A pure leaf: standard library plus ``data.point_in_time`` only. No logging, clock, environment,
I/O or metrics. Every time is an integer ``*_ms`` handed in by the caller, so the module is a
function of its inputs. It is unwired until P4a/P5: nothing imports it yet.

* The converters emit point-in-time series only and read none of the three never-served fields
  (CF-24-13, CF-2401-30): they take no argument for them.
* :class:`LiveContextBook` is copy-on-write. ``view`` is the only read the engine may use; it
  carries the clock-regression guard (SY-01-21a). ``snapshot`` and ``freshness`` never mutate.
* Cadence, budgets and scheduling live in ``live_context_policy.py`` (ER2-05), not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from types import MappingProxyType
from typing import Final

from data.point_in_time import (
    BACK_7D_MS,
    SERIES_RULES,
    Observation,
    PitError,
    PitSeries,
    available_at_estimated,
)

CLOCK_REGRESSION_TOLERANCE_MS: Final = 300_000
BUCKET_MS: Final = 3_600_000  # recording bucket of the snapshot series (SY-01-07)
CONSUMER_MODES: Final = ("paper", "live")
SNAPSHOT_SERIES: Final = frozenset({"coingecko:btc_dominance", "whale:net_flow_1h"})
_FGI: Final = "alternative_me:fgi"
_T10Y2Y: Final = "fred:T10Y2Y"
_QUANTUM: Final = Decimal("1e-8")
_BAD: Final = "pit_bad_value"
_EMPTY: Final[Mapping[str, PitSeries]] = MappingProxyType({})
_CTX: Final = Context(prec=60, rounding=ROUND_HALF_UP)  # private: the ambient context is never read
_ZERO: Final = Decimal(0).quantize(_QUANTUM)


def _int(value: object) -> int:
    if type(value) is not int:
        raise PitError(_BAD)
    return value


def _value(raw: object) -> Decimal:
    """``Decimal(str(raw))`` quantised to 1e-8 HALF_UP under a private context; -0 -> 0.

    bool, str, None, a non-finite value and an int too long for ``str`` raise ``pit_bad_value``.
    """
    if type(raw) not in (int, float, Decimal):
        raise PitError(_BAD)
    try:
        result = Decimal(str(raw)).quantize(_QUANTUM, context=_CTX)
    except (InvalidOperation, ValueError):
        raise PitError(_BAD) from None
    if not result.is_finite():
        raise PitError(_BAD)
    return result if result else _ZERO


def _not_future(observed_ms: int, fetch_ms: int) -> None:
    """CF-2401-31: a day stamp or observation date after the fetch is implausible (5 min slack)."""
    if observed_ms > _int(fetch_ms) + CLOCK_REGRESSION_TOLERANCE_MS:
        raise PitError(_BAD)


def fgi_observations(
    points: Iterable[tuple[int, object]], fetch_ms: int
) -> tuple[Observation, ...]:
    """FGI rows ``(day D 00:00 UTC ms, value)`` -> ``estimated_lag`` (observed + 24h; SY-01-06)."""
    rows: list[Observation] = []
    for observed_ms, raw in points:
        available_ms = available_at_estimated(_FGI, observed_ms)  # rejects a non-midnight day
        _not_future(observed_ms, fetch_ms)
        rows.append(Observation(observed_ms, available_ms, _value(raw), "estimated_lag"))
    return tuple(rows)


def t10y2y_observation(observed_ms: int, raw: object, fetch_ms: int) -> Observation:
    """T10Y2Y from its OBSERVATION date (AD-13), basis ``estimated_lag``."""
    available_ms = available_at_estimated(_T10Y2Y, observed_ms)
    _not_future(observed_ms, fetch_ms)
    return Observation(observed_ms, available_ms, _value(raw), "estimated_lag")


def t10y2y_observations(
    points: Iterable[tuple[int, object]], fetch_ms: int
) -> tuple[Observation, ...]:
    """Every released T10Y2Y row of one page; one future-dated row rejects the whole page."""
    return tuple(t10y2y_observation(observed_ms, raw, fetch_ms) for observed_ms, raw in points)


def snapshot_observation(series_id: str, raw: object, fetch_ms: int) -> Observation:
    """Dominance / whale: observed = floor(fetch, 1h), available = fetch (``recorded``)."""
    if series_id not in SNAPSHOT_SERIES:
        raise PitError("pit_unknown_series")
    return Observation(_int(fetch_ms) - fetch_ms % BUCKET_MS, fetch_ms, _value(raw), "recorded")


@dataclass(frozen=True, slots=True)
class SeriesFreshness:
    rows: int
    fresh: bool  # asof(now) is not None
    age_ms: int | None  # now - first availability of the visible value; None = nothing visible


@dataclass(frozen=True, slots=True)
class BookFreshness:
    regressed: bool  # same comparison as ``view``; changes nothing (N-3)
    series: Mapping[str, SeriesFreshness]


class LiveContextBook:
    """Copy-on-write series store with a clock-regression high-water mark (SY-01-21a).

    Single event loop; no locking.
    """

    __slots__ = ("_high_water_ms", "_series", "regressed_reads")

    def __init__(self) -> None:
        self._series: Mapping[str, PitSeries] = _EMPTY
        self._high_water_ms: int = 0
        self.regressed_reads: int = 0

    def _regressed(self, now_ms: int) -> bool:
        return _int(now_ms) + CLOCK_REGRESSION_TOLERANCE_MS < self._high_water_ms

    def view(self, now_ms: int) -> Mapping[str, PitSeries]:
        """The engine read. Regressed clock -> empty mapping and ``regressed_reads += 1``."""
        if self._regressed(now_ms):
            self.regressed_reads += 1
            self._drop_future_recorded(now_ms)
            return _EMPTY
        self._high_water_ms = max(self._high_water_ms, now_ms)
        return self._series

    def heartbeat(self, now_ms: int) -> bool:
        """Poller tick: ``True`` when regressed (mark unchanged), else raises the mark."""
        if self._regressed(now_ms):
            self._drop_future_recorded(now_ms)
            return True
        self._high_water_ms = max(self._high_water_ms, now_ms)
        return False

    def publish(self, series_id: str, rows: Iterable[Observation], now_ms: int) -> int:
        """Merge validated rows; returns the number of rows applied (ER3-02).

        0 when nothing was applied (identical row, trimmed row, or a held row protected by ER3-02
        rules 4/5). A non-``Observation`` row, or a ``recorded`` row with ``available_ms > now_ms``,
        raises ``pit_bad_value`` and leaves the book unchanged. Applying a row raises the mark.
        """
        rule = SERIES_RULES.get(series_id)
        if rule is None:
            raise PitError("pit_unknown_series")
        cutoff = _int(now_ms) - BACK_7D_MS - rule.max_age_ms
        held = self._series.get(series_id)
        merged = {o.observed_ms: o for o in held.observations} if held else {}
        applied = 0
        for obs in rows:
            if type(obs) is not Observation:
                raise PitError(_BAD)
            if obs.basis == "recorded" and obs.available_ms > now_ms:
                raise PitError(_BAD)  # a recorded stamp is a past fetch time
            current = merged.get(obs.observed_ms)
            if obs.available_ms < cutoff or current == obs:
                continue
            if current is not None and (
                series_id in SNAPSHOT_SERIES or current.available_ms <= now_ms
            ):
                continue  # E-1: a held bucket row is never replaced
            merged[obs.observed_ms] = obs
            applied += 1
        if applied == 0:
            return 0
        kept = tuple(o for o in merged.values() if o.available_ms >= cutoff)
        updated = dict(self._series)
        updated[series_id] = PitSeries(series_id, kept)  # validates before any state changes
        self._series = MappingProxyType(updated)
        self._high_water_ms = max(self._high_water_ms, now_ms)
        return applied

    def _drop_future_recorded(self, now_ms: int) -> None:
        """Drop ``recorded`` observations with
        ``available_ms > now_ms + CLOCK_REGRESSION_TOLERANCE_MS`` (ER3-01). Called only on the
        regressed branches of ``view`` and ``heartbeat``; ``estimated_lag`` rows are never dropped.
        """
        limit = now_ms + CLOCK_REGRESSION_TOLERANCE_MS
        updated: dict[str, PitSeries] = {}
        for series_id, series in self._series.items():
            kept = tuple(
                o for o in series.observations if o.basis != "recorded" or o.available_ms <= limit
            )
            if kept:
                updated[series_id] = (
                    series if len(kept) == len(series.observations) else PitSeries(series_id, kept)
                )
        if len(updated) != len(self._series) or any(
            updated[k] is not self._series[k] for k in updated
        ):
            self._series = MappingProxyType(updated)

    def snapshot(self) -> Mapping[str, PitSeries]:
        """The current immutable mapping; no guard side effects."""
        return self._series

    def freshness(self, now_ms: int) -> BookFreshness:
        """Per-series state for health and metrics. Never mutates the book (SY-01-21f, N-3)."""
        out: dict[str, SeriesFreshness] = {}
        for series_id, series in self._series.items():
            visible = [o for o in series.observations if o.available_ms <= now_ms]
            newest = max(visible, key=lambda o: o.observed_ms) if visible else None
            fresh = series.asof(now_ms, max_age_ms=SERIES_RULES[series_id].max_age_ms) is not None
            age = None if newest is None else now_ms - newest.available_ms
            out[series_id] = SeriesFreshness(len(series.observations), fresh, age)
        return BookFreshness(self._regressed(now_ms), MappingProxyType(out))

    def reset(self) -> None:
        """Test isolation only (writer-pinned to ``tests/``)."""
        self._series = _EMPTY
        self._high_water_ms = 0
        self.regressed_reads = 0


class ConsumerRegistry:
    """Counts running consumer engines per mode (SY-01-21j). Single event loop; no locking."""

    __slots__ = ("_active", "_next")

    def __init__(self) -> None:
        self._active: dict[int, str] = {}
        self._next = 1

    def started(self, mode: str) -> int:
        if type(mode) is not str or mode not in CONSUMER_MODES:
            raise ValueError("unknown consumer mode")
        token = self._next
        self._next += 1
        self._active[token] = mode
        return token

    def stopped(self, token: int) -> None:
        """Idempotent; an unknown token is a no-op."""
        if type(token) is int:
            self._active.pop(token, None)

    def counts(self) -> Mapping[str, int]:
        modes = list(self._active.values())
        return MappingProxyType({m: modes.count(m) for m in CONSUMER_MODES})

    def reset(self) -> None:
        self._active.clear()


LIVE_BOOK: Final = LiveContextBook()
LIVE_CONSUMERS: Final = ConsumerRegistry()
