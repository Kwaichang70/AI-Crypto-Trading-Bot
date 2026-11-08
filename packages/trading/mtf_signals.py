# ruff: noqa: S110  (the four best-effort swallow-all reads are moved verbatim from HEAD)
"""Signal sources for the multi-timeframe context (WP2.4, SY-24-02).

``StrategyEngine._build_mtf_context`` used to read four process-global provider clients
(Fear & Greed, CoinGecko, FRED, Whale Alert) on every bar, in every mode. In a backtest
that injected the values cached at API boot (and a wall-clock "7 days ago") into every
historical bar: look-ahead (C10). The engine now asks a *signal source* instead:

* :data:`UNAVAILABLE` (backtest default): every field is ``None``; no client is touched.
* :class:`PitSignals` (backtest): values served as-of the bar's ``close_i`` from
  :mod:`data.point_in_time` series (``available_at <= close_i``, no lag parameter).
* :data:`LIVE_SIGNALS` (paper and live): the pre-WP2.4 code, moved verbatim.

:class:`LiveSignals` is deliberately NOT a :class:`BacktestSignalSource`. The engine guards
(``StrategyEngine.__init__``) keep a live source out of a backtest and the reverse.

Logging: this module logs nothing. The moved live code keeps its HEAD swallow-all.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar, Final, final

from data.point_in_time import FIELD_MAP, SERIES_RULES, PitError, PitSeries, to_ms

__all__ = [
    "KNOWN_MTF_CONTEXT_VERSIONS",
    "LIVE_SIGNALS",
    "MTF_CONTEXT_VERSION",
    "UNAVAILABLE",
    "BacktestSignalSource",
    "LiveSignals",
    "PitSignals",
    "SignalValues",
    "UnavailableSignals",
]

MTF_CONTEXT_VERSION: Final = "wp24-1"
KNOWN_MTF_CONTEXT_VERSIONS: Final[frozenset[str]] = frozenset({MTF_CONTEXT_VERSION})

#: The fields typed ``int`` in ``MultiTimeframeContext`` (every other signal field is ``float``).
_INT_FIELDS: Final[frozenset[str]] = frozenset({"fear_greed_index", "fear_greed_index_7d_ago"})
_BAD_DECISION_TIME: Final = "PitSignals.values_at requires a decision time"


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


@final
class LiveSignals:
    """Paper/live source: the pre-WP2.4 provider-singleton reads, moved verbatim.

    NOT a :class:`BacktestSignalSource`. Use the :data:`LIVE_SIGNALS` singleton: the engine
    guard checks it by identity.
    """

    __slots__ = ()

    def values_at(self, decision_at: datetime | None = None) -> SignalValues:
        # --- moved verbatim from StrategyEngine._build_mtf_context (HEAD bbe7142,
        # strategy_engine.py:2129-2193): the four best-effort singleton reads ---
        # Read cached FGI value (best-effort; never crash bar processing
        # if the client is unavailable or has not populated its cache yet).
        fgi_value: int | None = None
        fgi_value_7d_ago: int | None = None
        try:
            from data.sentiment import get_global_client as _get_fgi_client
            _fgi_client = _get_fgi_client()
            if _fgi_client is not None:
                fgi_value = _fgi_client.cached_value
                # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
                # Returns None when history has not been populated yet by the
                # background refresh task; v2 feature builder treats None as
                # "no observed change".
                fgi_value_7d_ago = _fgi_client.value_at_offset_from_cache(
                    days_ago=7,
                )
        except Exception:
            pass  # FGI is best-effort; never crash bar processing

        # CoinGecko market structure signals (best-effort)
        btc_dominance: float | None = None
        btc_dominance_7d_ago: float | None = None
        market_cap_change_24h: float | None = None
        total_volume_change_24h: float | None = None
        try:
            from data.market_signals import get_global_client as _get_cg_client
            _cg_client = _get_cg_client()
            if _cg_client is not None:
                _cg_snap = _cg_client.cached_value
                if _cg_snap is not None:
                    btc_dominance = _cg_snap.btc_dominance
                    market_cap_change_24h = _cg_snap.market_cap_change_24h
                    total_volume_change_24h = _cg_snap.total_volume_change_24h
                # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
                btc_dominance_7d_ago = (
                    _cg_client.btc_dominance_at_offset_from_cache(days_ago=7)
                )
        except Exception:
            pass  # CoinGecko is best-effort; never crash bar processing

        # FRED macro-economic signals (best-effort)
        fed_funds_rate: float | None = None
        yield_curve_spread: float | None = None
        try:
            from data.macro_data import get_global_client as _get_fred_client
            _fred_client = _get_fred_client()
            if _fred_client is not None:
                _fred_snap = _fred_client.cached_value
                if _fred_snap is not None:
                    fed_funds_rate = _fred_snap.fed_funds_rate
                    yield_curve_spread = _fred_snap.yield_curve_spread
        except Exception:
            pass  # FRED is best-effort; never crash bar processing

        # Whale Alert on-chain flow signals (best-effort)
        whale_net_flow: float | None = None
        try:
            from data.whale_tracker import get_global_client as _get_whale_client
            _whale_client = _get_whale_client()
            if _whale_client is not None:
                _whale_snap = _whale_client.cached_value
                if _whale_snap is not None:
                    whale_net_flow = _whale_snap.net_flow
        except Exception:
            pass  # Whale Alert is best-effort; never crash bar processing

        # --- end of the verbatim block; the HEAD ``ctx_kwargs`` non-None rule
        # (strategy_engine.py:2195-2216) is SignalValues.to_context_kwargs ---
        return SignalValues(
            fear_greed_index=fgi_value,
            btc_dominance=btc_dominance,
            market_cap_change_24h=market_cap_change_24h,
            total_volume_change_24h=total_volume_change_24h,
            fed_funds_rate=fed_funds_rate,
            yield_curve_spread=yield_curve_spread,
            whale_net_flow=whale_net_flow,
            fear_greed_index_7d_ago=fgi_value_7d_ago,
            btc_dominance_7d_ago=btc_dominance_7d_ago,
        )


LIVE_SIGNALS: Final = LiveSignals()
