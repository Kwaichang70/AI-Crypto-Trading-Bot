"""Signal sources for the multi-timeframe context (WP2.4, SY-24-02).

``StrategyEngine._build_mtf_context`` used to read four process-global provider clients
(Fear & Greed, CoinGecko, FRED, Whale Alert) on every bar, in every mode. In a backtest
that injected the values cached at API boot (and a wall-clock "7 days ago") into every
historical bar: look-ahead (C10). The engine now asks a *signal source* instead:

* :data:`UNAVAILABLE` (backtest default): every field is ``None``; no client is touched.
* :class:`PitSignals` (backtest): values served as-of the bar's ``close_i`` from
  :mod:`data.point_in_time` series (``available_at <= close_i``, no lag parameter).
  :meth:`PitSignals.select` makes the per-run, per-field coverage decision (SY-24-05) and
  returns the source the run uses together with its :class:`MtfContextReport` (SY-24-07).
* :data:`LIVE_SIGNALS` (paper and live): a pure read of the live context book at the wall clock
  (CF-24-01); no provider client is touched.

:class:`LiveSignals` is deliberately NOT a :class:`BacktestSignalSource`. The engine guards
(``StrategyEngine.__init__``) keep a live source out of a backtest and the reverse.

This module emits nothing: the one swallow-all of ``LiveSignals.values_at`` returns all ``None``.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, ClassVar, Final, final

from data.live_context import LIVE_BOOK, LIVE_CONSUMERS
from data.point_in_time import (
    FIELD_MAP,
    MIN_COVERAGE_PPM,
    NOT_SERVED_FIELDS,
    SERIES_RULES,
    PitError,
    PitSeries,
    series_digest,
    to_ms,
)
from trading.clock import WALL_CLOCK

__all__ = [
    "ALL_MTF_SIGNAL_FIELDS",
    "KNOWN_MTF_CONTEXT_VERSIONS",
    "LIVE_CONSUMERS",
    "LIVE_CONTEXT_STAGE",
    "LIVE_SIGNALS",
    "MTF_CONTEXT_VERSION",
    "UNAVAILABLE",
    "BacktestSignalSource",
    "LiveSignals",
    "MtfContextReport",
    "PitSignals",
    "SeriesReport",
    "SignalValues",
    "UnavailableSignals",
    "check_backtest_source",
    "live_served_fields",
    "select_mtf_signals",
]

MTF_CONTEXT_VERSION: Final = "wp24-1"
KNOWN_MTF_CONTEXT_VERSIONS: Final[frozenset[str]] = frozenset({MTF_CONTEXT_VERSION})

#: The fields typed ``int`` in ``MultiTimeframeContext`` (every other signal field is ``float``).
_INT_FIELDS: Final[frozenset[str]] = frozenset({"fear_greed_index", "fear_greed_index_7d_ago"})
_BAD_DECISION_TIME: Final = "PitSignals.values_at requires a decision time"

#: The nine external signal fields of ``MultiTimeframeContext`` (SY-24-06: the fail-closed
#: default of ``BaseStrategy.consumes_mtf_signals``): the six served ones plus the three
#: ``not_served`` ones (SY-24-04).
ALL_MTF_SIGNAL_FIELDS: Final[frozenset[str]] = frozenset(FIELD_MAP) | NOT_SERVED_FIELDS

# Closed vocabularies of the run metadata (SY-24-07).
_FIELD_PIT: Final = "pit"
_FIELD_NO_SOURCE: Final = "no_source"
_FIELD_LOW_COVERAGE: Final = "insufficient_coverage"
_FIELD_NOT_SERVED: Final = "not_served"
_SOURCE_PIT: Final = "pit"
_SOURCE_UNAVAILABLE: Final = "unavailable"
_STATUS_NOT_USED: Final = "not_used"
_STATUS_UNAVAILABLE: Final = "unavailable"
_STATUS_PARTIAL: Final = "partial"
_STATUS_PIT: Final = "pit"
_REASON_NOT_USED: Final = "no_consumed_fields"
_REASON_NO_SOURCE: Final = "no_signal_source"
_REASON_LOW_COVERAGE: Final = "insufficient_coverage"
_REASON_NO_HISTORY: Final = "no_history"
_REASON_PARTIAL: Final = "partial_fields"
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)

#: Rollout stage of the live context (SY-01-04, UD-1 (a)): ``neutral_shadow`` serves consumers
#: ``SignalValues()`` and emits the shadow event; P9 flips it to ``fresh`` after G-E.
LIVE_CONTEXT_STAGE: Final = "neutral_shadow"
#: The six fields the book can serve; the three never-served ones stay ``None`` (CF-2401-30).
_LIVE_FIELDS: Final[frozenset[str]] = frozenset(FIELD_MAP)


@dataclass(frozen=True, slots=True)
class SignalValues:
    """The nine external signal fields of ``MultiTimeframeContext``; ``None`` = absent."""

    fear_greed_index: int | None = None
    btc_dominance: float | None = None
    market_cap_change_24h: float | None = None
    total_volume_change_24h: float | None = None
    fed_funds_rate: float | None = None
    yield_curve_spread: float | None = None
    whale_net_flow: float | None = None
    fear_greed_index_7d_ago: int | None = None
    btc_dominance_7d_ago: float | None = None

    def to_context_kwargs(self) -> dict[str, Any]:
        """Only the non-None fields (the HEAD ``ctx_kwargs`` rule), in the HEAD order.

        Absent fields stay at the frozen dataclass defaults of ``MultiTimeframeContext``.
        Values pass through unchanged: the typing (``int`` for the Fear & Greed fields,
        ``float`` for the rest) is applied by the source that builds the values.
        """
        ctx_kwargs: dict[str, Any] = {}
        if self.fear_greed_index is not None:
            ctx_kwargs["fear_greed_index"] = self.fear_greed_index
        if self.btc_dominance is not None:
            ctx_kwargs["btc_dominance"] = self.btc_dominance
        if self.market_cap_change_24h is not None:
            ctx_kwargs["market_cap_change_24h"] = self.market_cap_change_24h
        if self.total_volume_change_24h is not None:
            ctx_kwargs["total_volume_change_24h"] = self.total_volume_change_24h
        if self.fed_funds_rate is not None:
            ctx_kwargs["fed_funds_rate"] = self.fed_funds_rate
        if self.yield_curve_spread is not None:
            ctx_kwargs["yield_curve_spread"] = self.yield_curve_spread
        if self.whale_net_flow is not None:
            ctx_kwargs["whale_net_flow"] = self.whale_net_flow
        if self.fear_greed_index_7d_ago is not None:
            ctx_kwargs["fear_greed_index_7d_ago"] = self.fear_greed_index_7d_ago
        if self.btc_dominance_7d_ago is not None:
            ctx_kwargs["btc_dominance_7d_ago"] = self.btc_dominance_7d_ago
        return ctx_kwargs

    def restricted(self, names: Iterable[str]) -> SignalValues:
        """A copy that keeps only the fields in ``names``; every other field is ``None``."""
        keep = frozenset(names)
        kept = {f.name: getattr(self, f.name) for f in dataclass_fields(self) if f.name in keep}
        return SignalValues(**kept)


class BacktestSignalSource(abc.ABC):
    """Abstract marker base of the sources a BACKTEST engine accepts.

    The engine guard requires a real subclass; ``register()`` is not honoured.
    """

    mode: ClassVar[str]

    @abc.abstractmethod
    def values_at(self, decision_at: datetime | None) -> SignalValues:
        """The signal values a decision at ``decision_at`` (the bar's close) may see."""


@final
class UnavailableSignals(BacktestSignalSource):
    """No history: every field is ``None``, whatever the decision time."""

    mode: ClassVar[str] = "unavailable"

    def values_at(self, decision_at: datetime | None) -> SignalValues:
        return SignalValues()


UNAVAILABLE: Final = UnavailableSignals()


def _now_field_of(field_name: str) -> str | None:
    """The 'now' field fed by the same series as a back-looking (7d) field, else None."""
    spec = FIELD_MAP[field_name]
    if spec.back_ms == 0:
        return None
    for other, other_spec in FIELD_MAP.items():
        if other_spec.series_id == spec.series_id and other_spec.back_ms == 0:
            return other
    return None


def live_served_fields(declared: Iterable[object]) -> frozenset[str]:
    """Fields a paper/live run serves: the union of the strategies' declarations (a non-frozenset
    declares all nine, SY-24-06) that the book can serve, plus the 'now' partner of each 7d
    field."""
    consumed: set[str] = set()
    for item in declared:
        consumed |= item if isinstance(item, frozenset) else ALL_MTF_SIGNAL_FIELDS
    served = frozenset(consumed) & _LIVE_FIELDS
    return served | {p for p in map(_now_field_of, served) if p is not None}


def _iso(ms: int) -> str:
    """Integer epoch milliseconds -> ISO-8601 UTC text (no wall clock)."""
    return (_EPOCH + timedelta(milliseconds=ms)).isoformat()


@dataclass(frozen=True, slots=True)
class SeriesReport:
    """Per-series run facts: how much history the source holds and how well it covers."""

    series_id: str
    rows: int
    first_available_at: str | None
    last_available_at: str | None
    lag_rule: str
    max_age_s: int
    coverage_ppm: int


@dataclass(frozen=True, slots=True)
class MtfContextReport:
    """What a run's MTF context actually was (SY-24-07); recorded as ``config["mtf_context"]``.

    ``status`` is derived over the strategies' CONSUMED fields only; ``fields`` lists all
    nine context fields. ``ingest_cutoff`` stays ``None`` until the Phase B store exists.
    ``fields`` is each field's coverage verdict: for a consumed field ``pit`` means served;
    for an unconsumed field it means it passed coverage and is NOT served (unless it is the
    'now' partner of a served 7d field). ``source`` is ``pit`` iff a field is served, i.e.
    iff ``status`` is ``pit`` or ``partial``.
    Immutable; :meth:`to_metadata` returns a fresh JSON-safe dict (ints, str, bool, ISO only).
    """

    source: str
    status: str
    fields: tuple[tuple[str, str], ...]
    series: tuple[SeriesReport, ...]
    series_digest: str | None
    ingest_cutoff: str | None
    reason: str | None
    version: str = MTF_CONTEXT_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "status": self.status,
            "fields": dict(self.fields),
            "series": {
                s.series_id: {
                    "rows": s.rows,
                    "first_available_at": s.first_available_at,
                    "last_available_at": s.last_available_at,
                    "lag_rule": s.lag_rule,
                    "max_age_s": s.max_age_s,
                    "coverage_ppm": s.coverage_ppm,
                }
                for s in self.series
            },
            "series_digest": self.series_digest,
            "ingest_cutoff": self.ingest_cutoff,
            "reason": self.reason,
        }


def _no_source_fields() -> dict[str, str]:
    return {
        name: (_FIELD_NOT_SERVED if name in NOT_SERVED_FIELDS else _FIELD_NO_SOURCE)
        for name in sorted(ALL_MTF_SIGNAL_FIELDS)
    }


def _unavailable_report(consumed: frozenset[str], reason: str) -> MtfContextReport:
    """Report of a run that gets ``UNAVAILABLE`` (not used, or no source offered)."""
    return MtfContextReport(
        source=_SOURCE_UNAVAILABLE,
        status=_STATUS_UNAVAILABLE if consumed else _STATUS_NOT_USED,
        fields=tuple(_no_source_fields().items()),
        series=(),
        series_digest=None,
        ingest_cutoff=None,
        reason=reason,
    )


def _status_of(consumed: frozenset[str], statuses: Mapping[str, str]) -> str:
    """not_used / unavailable / partial / pit over the consumed fields only (SY-24-07)."""
    if not consumed:
        return _STATUS_NOT_USED
    served = [statuses[name] == _FIELD_PIT for name in sorted(consumed)]
    if not any(served):
        return _STATUS_UNAVAILABLE
    return _STATUS_PIT if all(served) else _STATUS_PARTIAL


@final
class PitSignals(BacktestSignalSource):
    """Point-in-time values from immutable :class:`PitSeries`; a pure function of its inputs.

    ``fields_on`` names the fields this source serves; every other field is ``None`` on
    every bar. A back-looking field (7d) may only be on together with its 'now' field
    (QR-05: "7d ON implies now ON").

    A field is served as ``asof(decision - FIELD_MAP[f].back_ms)`` with the series' own
    ``SERIES_RULES`` max age (QR-13). The lag is already inside each observation's
    ``available_ms`` (SY-24-03), so no lag is applied here. Run-level selection and the
    coverage report belong to P2.
    """

    mode: ClassVar[str] = "pit"

    __slots__ = ("_fields_on", "_series")

    def __init__(self, series: Mapping[str, PitSeries], fields_on: frozenset[str]) -> None:
        copied: dict[str, PitSeries] = {}
        for series_id, item in series.items():
            if type(item) is not PitSeries or item.series_id != series_id:
                raise PitError("pit_bad_value")
            if series_id not in SERIES_RULES:
                raise PitError("pit_unknown_series")
            copied[series_id] = item
        on = frozenset(fields_on)
        for name in on:
            if name not in FIELD_MAP:
                raise PitError("pit_unknown_series")
            partner = _now_field_of(name)
            if partner is not None and partner not in on:
                raise PitError("pit_bad_value")
        self._series: Mapping[str, PitSeries] = MappingProxyType(copied)
        self._fields_on: frozenset[str] = on

    @property
    def series(self) -> Mapping[str, PitSeries]:
        return self._series

    @property
    def fields_on(self) -> frozenset[str]:
        return self._fields_on

    def values_at(self, decision_at: datetime | None) -> SignalValues:
        if decision_at is None:
            raise TypeError(_BAD_DECISION_TIME)
        decision_ms = to_ms(decision_at)  # plain int; naive -> PitError (fails closed)
        out: dict[str, int | float | None] = {}
        for name in self._fields_on:
            spec = FIELD_MAP[name]
            series = self._series.get(spec.series_id)
            if series is None:
                continue
            value = series.asof(
                decision_ms - spec.back_ms, max_age_ms=SERIES_RULES[spec.series_id].max_age_ms
            )
            if value is None:
                continue
            out[name] = int(value) if name in _INT_FIELDS else float(value)
        return SignalValues(**out)  # type: ignore[arg-type]

    def select(
        self, decision_times: Iterable[datetime], fields: Iterable[str]
    ) -> tuple[BacktestSignalSource, MtfContextReport]:
        """The per-run source and its report (SY-24-05 / SY-24-07). Pure; raises on bad input.

        ``decision_times`` are the run's trading decisions ``close_k`` (k in [W, N)),
        ``fields`` the fields the run's strategies CONSUME. No consumed field -> ``UNAVAILABLE``
        (``not_used``, so a non-consumer's trades never depend on the store, G8). Otherwise a
        served field is ON iff its coverage over ALL decisions (the field's back leg included)
        is >= ``MIN_COVERAGE_PPM``; an OFF field is removed from the returned source, and so is an
        ON field the run does not consume (QR-08), except the 'now' partner a served 7d field
        needs.
        Zero decisions -> every coverage is 0 -> everything OFF.
        """
        consumed = frozenset(fields)
        if not consumed <= ALL_MTF_SIGNAL_FIELDS:
            raise PitError("pit_unknown_series")
        if not consumed:
            return UNAVAILABLE, _unavailable_report(consumed, _REASON_NOT_USED)
        decisions = tuple(to_ms(moment) for moment in decision_times)  # naive -> PitError
        statuses = _no_source_fields()
        ppm_of: dict[str, int] = {}
        on: set[str] = set()
        for name, spec in FIELD_MAP.items():
            history = self._series.get(spec.series_id)
            if name not in self._fields_on or history is None or not history.observations:
                ppm_of[name] = 0
                continue
            ppm = history.coverage_ppm(
                decisions,
                back_ms=spec.back_ms,
                max_age_ms=SERIES_RULES[spec.series_id].max_age_ms,
            )
            ppm_of[name] = ppm
            if ppm >= MIN_COVERAGE_PPM:
                on.add(name)
                statuses[name] = _FIELD_PIT
            else:
                statuses[name] = _FIELD_LOW_COVERAGE
        rows: list[SeriesReport] = []
        for series_id, rule in SERIES_RULES.items():
            if not rule.in_backtest:
                continue
            history = self._series.get(series_id)
            available = [] if history is None else [o.available_ms for o in history.observations]
            now_field = next(f for f in rule.fields if FIELD_MAP[f].back_ms == 0)
            rows.append(
                SeriesReport(
                    series_id=series_id,
                    rows=len(available),
                    first_available_at=_iso(min(available)) if available else None,
                    last_available_at=_iso(max(available)) if available else None,
                    lag_rule=rule.lag_rule,
                    max_age_s=rule.max_age_ms // 1000,
                    coverage_ppm=ppm_of[now_field],
                )
            )
        status = _status_of(consumed, statuses)
        # QR-08: serve only consumed ON fields, plus the 'now' partner a served 7d field
        # needs (PitSignals invariant). Generalised G8: trades depend on the store only
        # through series that feed consumed fields; ``source`` agrees with ``status``.
        served = {name for name in on if name in consumed}
        served |= {p for p in (_now_field_of(n) for n in tuple(served)) if p is not None}
        effective: BacktestSignalSource = (
            PitSignals(self._series, frozenset(served)) if served else UNAVAILABLE
        )
        if status == _STATUS_PIT:
            reason = None
        elif status == _STATUS_PARTIAL:
            reason = _REASON_PARTIAL
        elif _FIELD_LOW_COVERAGE in {statuses[n] for n in consumed}:
            reason = _REASON_LOW_COVERAGE
        else:
            reason = _REASON_NO_HISTORY
        report = MtfContextReport(
            source=_SOURCE_PIT if served else _SOURCE_UNAVAILABLE,
            status=status,
            fields=tuple(sorted(statuses.items())),
            series=tuple(sorted(rows, key=lambda row: row.series_id)),
            series_digest=series_digest(self._series.values()),
            ingest_cutoff=None,
            reason=reason,
        )
        return effective, report


def check_backtest_source(source: object) -> None:
    """Raise ``ValueError`` unless ``source`` is ``None`` or a real ``BacktestSignalSource``.

    The same rule as the BACKTEST engine guard (a ``LiveSignals``, a foreign object or a
    ``register()``-ed virtual subclass is refused). Type names only in the message (CF-36).
    The runner and the optimizer call it BEFORE any run id or engine exists (CF-P1-01).
    """
    if source is not None and (
        isinstance(source, LiveSignals)
        or not isinstance(source, BacktestSignalSource)
        or BacktestSignalSource not in type(source).__mro__
    ):
        raise ValueError(
            "mtf_signals must be a BacktestSignalSource or None, "
            f"got {type(source).__name__}"
        )


def select_mtf_signals(
    source: BacktestSignalSource | None,
    decision_times: Iterable[datetime],
    fields: Iterable[str],
) -> tuple[BacktestSignalSource, MtfContextReport]:
    """Run-level selection for any offered source (SY-24-05 / SY-24-07). Pure.

    Validates ``source`` first (so a foreign object fails even for a non-consumer), then:
    no consumed field -> ``UNAVAILABLE`` + ``not_used``; no source / ``UNAVAILABLE`` ->
    ``UNAVAILABLE`` + ``unavailable``; ``PitSignals`` -> :meth:`PitSignals.select`.
    A real third-party ``BacktestSignalSource`` subclass passes the constructor guard but is
    refused HERE with ``ValueError`` (type name only), at ``run()``, still before ``run_id``
    and before any combination: a report cannot describe an unknown source.
    """
    check_backtest_source(source)
    consumed = frozenset(fields)
    if not consumed <= ALL_MTF_SIGNAL_FIELDS:
        raise PitError("pit_unknown_series")
    if isinstance(source, PitSignals):
        return source.select(decision_times, consumed)
    if source is not None and not isinstance(source, UnavailableSignals):
        raise ValueError(f"unsupported mtf_signals source {type(source).__name__}")
    reason = _REASON_NO_SOURCE if consumed else _REASON_NOT_USED
    return UNAVAILABLE, _unavailable_report(consumed, reason)


_LIVE_READ_FAILED: Final = SignalValues()


@final
class LiveSignals:
    """Paper/live source: the live context book read through the backtest join (CF-24-01).

    NOT a :class:`BacktestSignalSource`. Use the :data:`LIVE_SIGNALS` singleton: the engine
    guard checks it by identity. ``decision_at`` is ignored: the decision time is the wall clock,
    read once.
    """

    __slots__ = ()

    def values_at(self, decision_at: datetime | None = None) -> SignalValues:
        try:
            now = WALL_CLOCK.now()
            return PitSignals(LIVE_BOOK.view(to_ms(now)), _LIVE_FIELDS).values_at(now)
        except Exception:
            return _LIVE_READ_FAILED


LIVE_SIGNALS: Final = LiveSignals()
