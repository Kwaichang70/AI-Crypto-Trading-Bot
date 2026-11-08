# ruff: noqa: S311  (a seeded stdlib generator builds deterministic price paths)
"""WP2.4 / P1: the signal source of the multi-timeframe context and its engine wiring.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-02, SY-24-04, SY-24-16..18,
WP24-T-16, T-19, T-20) and the P0 carry-forward QR-13 (``reports/vp2-wp2.4/p0-quant-r1.md``).

* T-16  guards (``StrategyEngine.__init__``): type names only, identity for paper/live.
* T-19  engine-level point-in-time: dca and rsi on series S vs S' (S' differs only in
        observations that become available after ``close_k``) are identical through bar k.
* T-20  None audit: for every consumer ``mtf_context=None`` equals an all-None context with
        HTF present; an inert v2 model decides identically under any context.
* QR-13 a 7d field is served as ``asof(d - back_ms)`` with the series' own max age;
        decision times are plain ``int``; "7d ON implies now ON"; FGI is ``int``, rest ``float``.

The live-parity oracle (T-15) is ``tests/unit/test_wp24_live_parity.py``.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, TimeFrame
from data.point_in_time import (
    BACK_7D_MS,
    FIELD_MAP,
    SERIES_RULES,
    Observation,
    PitError,
    PitSeries,
    to_ms,
)
from tests.integration.fakes.bar_paths import scripted_path
from trading import backtest as backtest_mod
from trading.backtest import BacktestRunner
from trading.clock import BarClock
from trading.costs import get_cost_profile
from trading.models import Signal
from trading.mtf_signals import (
    KNOWN_MTF_CONTEXT_VERSIONS,
    LIVE_SIGNALS,
    MTF_CONTEXT_VERSION,
    UNAVAILABLE,
    BacktestSignalSource,
    LiveSignals,
    PitSignals,
    SignalValues,
    UnavailableSignals,
)
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy
from trading.strategies.model_strategy import ModelStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

SYM = "BTC/USDT"
H = timedelta(hours=1)
H_MS = 3_600_000
D_MS = 86_400_000
FIRST_OPEN = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)  # open of the first post-warm-up bar
WARMUP = 50
FGI = "alternative_me:fgi"
T10 = "fred:T10Y2Y"
DOM = "coingecko:btc_dominance"
WHALE = "whale:net_flow_1h"
FGI_FIELDS = frozenset({"fear_greed_index", "fear_greed_index_7d_ago"})
ALL_PIT_FIELDS = frozenset(FIELD_MAP)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _obs(observed_ms: int, available_ms: int, value: str | int, basis: str) -> Observation:
    return Observation(observed_ms, available_ms, Decimal(str(value)), basis)  # type: ignore[arg-type]


def _fgi_daily(first_day_ms: int, values: Sequence[int]) -> PitSeries:
    """FGI day D 00:00 UTC, available D + 24h (UD-5 (a), ingestion rule)."""
    return PitSeries(
        FGI,
        tuple(
            _obs(first_day_ms + k * D_MS, first_day_ms + (k + 1) * D_MS, v, "estimated_lag")
            for k, v in enumerate(values)
        ),
    )


def _ms(*args: int) -> int:
    return to_ms(datetime(*args, tzinfo=UTC))


def _closes(n: int = 300) -> list[float]:
    rnd = random.Random(5)
    return [round(100 + 12 * math.sin(i / 5) + rnd.uniform(-1, 1), 2) for i in range(n)]


def _bars(closes: Sequence[float] | None = None, symbol: str = SYM) -> list[OHLCVBar]:
    return scripted_path(closes or _closes(), first_open=FIRST_OPEN, symbol=symbol, warmup=WARMUP)


def _close_of(index: int) -> datetime:
    """Close time of the bar at ``index`` of the full list (warm-up included)."""
    return FIRST_OPEN + (index - WARMUP) * H + H


def _full_source(first_day_ms: int, days: int, flip_after_ms: int | None = None) -> PitSignals:
    """FGI + T10Y2Y + recorded dominance/whale series; values change after ``flip_after_ms``."""
    fgi = [10 + (k * 37) % 80 for k in range(days)]
    t10 = [Decimal(str(round(-0.5 + 0.1 * (k % 9), 2))) for k in range(days)]
    obs_t10 = tuple(
        Observation(
            first_day_ms + k * D_MS,
            first_day_ms + k * D_MS + 3 * D_MS,
            v,
            "estimated_lag",
        )
        for k, v in enumerate(t10)
    )
    hours = days * 24
    dom = tuple(
        Observation(
            first_day_ms + k * H_MS,
            first_day_ms + k * H_MS + 60_000,
            Decimal("50") + Decimal(k % 7),
            "recorded",
        )
        for k in range(hours)
    )
    whale = tuple(
        Observation(
            first_day_ms + k * H_MS,
            first_day_ms + k * H_MS + 60_000,
            Decimal(1000 * (k % 5 - 2)),
            "recorded",
        )
        for k in range(hours)
    )
    series = {
        FGI: _fgi_daily(first_day_ms, fgi),
        T10: PitSeries(T10, obs_t10),
        DOM: PitSeries(DOM, dom),
        WHALE: PitSeries(WHALE, whale),
    }
    if flip_after_ms is not None:
        series = {sid: _mutate_after(s, flip_after_ms) for sid, s in series.items()}
    return PitSignals(series, ALL_PIT_FIELDS)


def _mutate_after(series: PitSeries, cutoff_ms: int) -> PitSeries:
    """S': every observation available after ``cutoff_ms`` gets another value."""
    out: list[Observation] = []
    for o in series.observations:
        if o.available_ms > cutoff_ms:
            bump = Decimal(7) if series.series_id == FGI else Decimal("3.5")
            if series.series_id == FGI:
                value = Decimal((int(o.value) + 33) % 100)
            else:
                value = o.value + bump
            out.append(Observation(o.observed_ms, o.available_ms, value, o.basis))
        else:
            out.append(o)
    return PitSeries(series.series_id, tuple(out))


# ---------------------------------------------------------------------------
# SignalValues / UNAVAILABLE
# ---------------------------------------------------------------------------


class TestSignalValues:
    def test_all_none_by_default_and_exactly_the_nine_context_fields(self) -> None:
        sv = SignalValues()
        names = [f.name for f in dataclasses.fields(SignalValues)]
        assert len(names) == 9 and all(getattr(sv, n) is None for n in names)
        ctx_fields = set(MultiTimeframeContext.__dataclass_fields__) - {"htf_bars"}
        assert set(names) == ctx_fields
        assert sv.to_context_kwargs() == {}

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            SignalValues().fear_greed_index = 5  # type: ignore[misc]

    def test_only_non_none_fields_zero_is_kept_in_the_head_order(self) -> None:
        sv = SignalValues(
            fear_greed_index=0,
            btc_dominance=0.0,
            whale_net_flow=-1.5,
            fear_greed_index_7d_ago=0,
        )
        kw = sv.to_context_kwargs()
        assert list(kw) == ["fear_greed_index", "btc_dominance", "whale_net_flow",
                            "fear_greed_index_7d_ago"]
        assert kw == {"fear_greed_index": 0, "btc_dominance": 0.0, "whale_net_flow": -1.5,
                      "fear_greed_index_7d_ago": 0}
        # absent fields keep the MultiTimeframeContext defaults
        assert MultiTimeframeContext(**kw).fed_funds_rate is None

    def test_versions(self) -> None:
        assert MTF_CONTEXT_VERSION == "wp24-1"
        assert KNOWN_MTF_CONTEXT_VERSIONS == frozenset({"wp24-1"})


class TestUnavailable:
    def test_always_all_none_and_ignores_the_decision_time(self) -> None:
        assert isinstance(UNAVAILABLE, UnavailableSignals)
        assert isinstance(UNAVAILABLE, BacktestSignalSource)
        for at in (None, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1)):
            assert UNAVAILABLE.values_at(at) == SignalValues()
        assert UnavailableSignals.mode == "unavailable"

    def test_touches_no_provider_client(self) -> None:
        with ExitPatches() as seen:
            assert UNAVAILABLE.values_at(datetime(2026, 1, 1, tzinfo=UTC)) == SignalValues()
        assert seen == []


class ExitPatches:
    """Patch the four ``get_global_client`` getters with recorders (a call is a failure)."""

    def __enter__(self) -> list[str]:
        self.seen: list[str] = []
        self._patches = []
        for module in ("sentiment", "market_signals", "macro_data", "whale_tracker"):
            def _rec(module: str = module) -> None:
                self.seen.append(module)

            p = patch(f"data.{module}.get_global_client", side_effect=_rec)
            p.start()
            self._patches.append(p)
        return self.seen

    def __exit__(self, *exc: object) -> None:
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# PitSignals (construction + values_at; QR-13)
# ---------------------------------------------------------------------------


class TestPitSignalsServing:
    def _fgi_source(self) -> PitSignals:
        d0 = _ms(2026, 3, 5)
        series = PitSeries(
            FGI,
            (
                _obs(d0, d0 + D_MS, 20, "estimated_lag"),  # Mar 5 -> visible Mar 6 00:00
                _obs(d0 + 7 * D_MS, d0 + 8 * D_MS, 45, "estimated_lag"),  # Mar 12 -> Mar 13 00:00
            ),
        )
        return PitSignals({FGI: series}, FGI_FIELDS)

    def test_t05_style_now_and_seven_days_ago(self) -> None:
        # Mar 13 10:00: FGI(Mar 12)=45 is 10h old; seven days earlier (Mar 6 10:00) the
        # visible value was FGI(Mar 5)=20, 10h old.
        sv = self._fgi_source().values_at(datetime(2026, 3, 13, 10, tzinfo=UTC))
        assert (sv.fear_greed_index, sv.fear_greed_index_7d_ago) == (45, 20)

    def test_the_seven_day_leg_is_asof_at_the_earlier_time_not_nearest_snapshot(self) -> None:
        # Mon Mar 9 05:00: now=20 (Mar 5, avail Mar 6, 71h old -> stale) is None; this decision
        # is where the "nearest snapshot" and "ignore lag" mutants would answer 40 / 20.
        src = self._fgi_source()
        at = datetime(2026, 3, 13, 0, tzinfo=UTC)  # exactly available_ms of 45: inclusive
        sv = src.values_at(at)
        assert sv.fear_greed_index == 45
        assert sv.fear_greed_index_7d_ago == 20  # Mar 6 00:00 == avail of 20: inclusive
        assert src.values_at(at - timedelta(milliseconds=1)).fear_greed_index is None

    def test_the_seven_day_leg_applies_staleness_at_the_earlier_time(self) -> None:
        d0 = _ms(2026, 3, 3)
        # Mar 3 value (avail Mar 4 00:00); decision Mar 13 10:00 -> 7d earlier is Mar 6 10:00,
        # 58h after availability > 48h: the 7d value is None although "now" (Mar 12) is served.
        series = PitSeries(
            FGI,
            (_obs(d0, d0 + D_MS, 20, "estimated_lag"),
             _obs(d0 + 9 * D_MS, d0 + 10 * D_MS, 45, "estimated_lag")),
        )
        sv = PitSignals({FGI: series}, FGI_FIELDS).values_at(datetime(2026, 3, 13, 10, tzinfo=UTC))
        assert sv.fear_greed_index == 45
        assert sv.fear_greed_index_7d_ago is None

    def test_each_field_uses_its_own_series_max_age(self) -> None:
        # dominance cap is 6h; FGI cap is 48h: same age, different answer.
        t0 = _ms(2026, 3, 10, 0)
        dom = PitSeries(DOM, (_obs(t0, t0, "55.5", "recorded"),))
        fgi = PitSeries(FGI, (_obs(t0 - D_MS, t0, 30, "estimated_lag"),))
        src = PitSignals({DOM: dom, FGI: fgi}, frozenset({"btc_dominance", "fear_greed_index"}))
        at_6h = datetime(2026, 3, 10, 6, tzinfo=UTC)
        at_6h1 = at_6h + timedelta(milliseconds=1)
        assert SERIES_RULES[DOM].max_age_ms == 6 * H_MS
        assert src.values_at(at_6h).btc_dominance == 55.5  # inclusive
        assert src.values_at(at_6h1).btc_dominance is None
        assert src.values_at(at_6h1).fear_greed_index == 30

    def test_types_fgi_is_int_every_other_field_is_float(self) -> None:
        t0 = _ms(2026, 3, 1)
        series = {
            FGI: PitSeries(FGI, (_obs(t0 - 8 * D_MS, t0 - 7 * D_MS, 12, "estimated_lag"),
                                 _obs(t0, t0 + D_MS, 77, "estimated_lag"))),
            T10: PitSeries(T10, (_obs(t0, t0 + D_MS, "-0.25", "estimated_lag"),)),
            DOM: PitSeries(DOM, (_obs(t0 - 6 * D_MS + 1_800_000, t0 - 6 * D_MS + 1_800_000,
                                      "51.25", "recorded"),
                                 _obs(t0 + D_MS, t0 + D_MS, "52.5", "recorded"))),
            WHALE: PitSeries(WHALE, (_obs(t0 + D_MS, t0 + D_MS, "-1500", "recorded"),)),
        }
        # decision = Mar 2 01:00; dominance 7d earlier (Feb 23 01:00) needs a value <= 6h old
        sv = PitSignals(series, ALL_PIT_FIELDS).values_at(
            datetime(2026, 3, 1, tzinfo=UTC) + timedelta(days=1, hours=1)
        )
        assert type(sv.fear_greed_index) is int and sv.fear_greed_index == 77
        assert type(sv.fear_greed_index_7d_ago) is int and sv.fear_greed_index_7d_ago == 12
        assert type(sv.yield_curve_spread) is float and sv.yield_curve_spread == -0.25
        assert type(sv.btc_dominance) is float and sv.btc_dominance == 52.5
        assert type(sv.btc_dominance_7d_ago) is float
        assert type(sv.whale_net_flow) is float and sv.whale_net_flow == -1500.0
        # not served in a backtest (SY-24-04): no source, always None
        assert sv.fed_funds_rate is None
        assert sv.market_cap_change_24h is None and sv.total_volume_change_24h is None

    def test_a_field_that_is_off_is_none_on_every_bar(self) -> None:
        t0 = _ms(2026, 3, 1)
        series = {FGI: _fgi_daily(t0 - 10 * D_MS, list(range(30, 60)))}
        on = PitSignals(series, FGI_FIELDS).values_at(datetime(2026, 3, 5, tzinfo=UTC))
        off = PitSignals(series, frozenset()).values_at(datetime(2026, 3, 5, tzinfo=UTC))
        assert on.fear_greed_index is not None and off == SignalValues()

    def test_a_missing_series_is_none_not_an_error(self) -> None:
        sv = PitSignals({}, ALL_PIT_FIELDS).values_at(datetime(2026, 3, 5, tzinfo=UTC))
        assert sv == SignalValues()

    def test_the_decision_time_is_a_plain_int(self) -> None:
        seen: list[type] = []
        real = PitSeries.asof

        def spy(self: PitSeries, decision_ms: int, *, max_age_ms: int) -> Decimal | None:
            seen.append(type(decision_ms))
            return real(self, decision_ms, max_age_ms=max_age_ms)

        src = self._fgi_source()
        with patch.object(PitSeries, "asof", spy):
            for at in (datetime(2026, 3, 13, 10, tzinfo=UTC),
                       pd.Timestamp("2026-03-13 10:00", tz="UTC")):
                assert src.values_at(at).fear_greed_index == 45
        assert seen and set(seen) == {int}

    def test_decision_time_errors_fail_closed(self) -> None:
        src = self._fgi_source()
        with pytest.raises(TypeError, match="requires a decision time"):
            src.values_at(None)
        with pytest.raises(PitError) as err:
            src.values_at(datetime(2026, 3, 13, 10))
        assert err.value.code == "pit_naive_time"

    def test_deterministic_and_pure(self) -> None:
        src = self._fgi_source()
        at = datetime(2026, 3, 13, 10, tzinfo=UTC)
        assert src.values_at(at) == src.values_at(at)
        assert PitSignals.mode == "pit"


class TestPitSignalsConstruction:
    def test_seven_day_field_requires_its_now_field(self) -> None:
        series = {FGI: _fgi_daily(_ms(2026, 3, 1), [10, 20, 30])}
        for bad in (frozenset({"fear_greed_index_7d_ago"}),
                    frozenset({"btc_dominance_7d_ago", "fear_greed_index"})):
            with pytest.raises(PitError) as err:
                PitSignals(series, bad)
            assert err.value.code == "pit_bad_value"
        PitSignals(series, FGI_FIELDS)  # both on: fine
        PitSignals(series, frozenset({"fear_greed_index"}))  # now alone: fine

    def test_the_back_ms_of_the_seven_day_fields_is_seven_days(self) -> None:
        assert FIELD_MAP["fear_greed_index_7d_ago"].back_ms == BACK_7D_MS == 7 * D_MS

    def test_unknown_field_unknown_series_and_key_mismatch_are_rejected(self) -> None:
        good = _fgi_daily(_ms(2026, 3, 1), [10, 20])
        with pytest.raises(PitError) as e1:
            PitSignals({FGI: good}, frozenset({"fed_funds_rate"}))
        assert e1.value.code == "pit_unknown_series"
        with pytest.raises(PitError) as e2:
            PitSignals({T10: good}, frozenset())  # key != series_id
        assert e2.value.code == "pit_bad_value"
        other = PitSeries("not:served", ())
        with pytest.raises(PitError) as e3:
            PitSignals({"not:served": other}, frozenset())
        assert e3.value.code == "pit_unknown_series"
        with pytest.raises(PitError):
            PitSignals({FGI: object()}, frozenset())  # type: ignore[dict-item]

    def test_the_source_is_immutable_after_construction(self) -> None:
        d: dict[str, PitSeries] = {FGI: _fgi_daily(_ms(2026, 3, 1), [10, 20, 30])}
        on = {"fear_greed_index"}
        src = PitSignals(d, frozenset(on))
        d.clear()
        on.clear()
        assert FGI in src.series and src.fields_on == frozenset({"fear_greed_index"})
        with pytest.raises(TypeError):
            src.series[T10] = good_series()  # type: ignore[index]


def good_series() -> PitSeries:
    return PitSeries(T10, ())


class TestPitSignalsPrefixInvariance:
    def test_values_at_ignores_everything_that_becomes_available_later(self) -> None:
        rnd = random.Random(24)
        base = _ms(2026, 3, 1)
        for trial in range(200):
            rows = []
            for k in range(rnd.randint(3, 14)):
                observed = base + k * D_MS
                lag = rnd.choice([D_MS, 2 * D_MS, 3 * D_MS + rnd.randint(0, H_MS)])
                rows.append((observed, observed + lag, rnd.randint(0, 100)))
            cut = base + rnd.randint(2, 18) * D_MS + rnd.randint(0, D_MS)
            kept = [r for r in rows if r[1] <= cut]
            mutated = [(o, a, v if a <= cut else (v + 41) % 101) for o, a, v in rows]
            sv_a = PitSignals(
                {FGI: PitSeries(FGI, tuple(_obs(o, a, v, "estimated_lag") for o, a, v in rows))},
                FGI_FIELDS,
            ).values_at(_dt(cut))
            sv_b = PitSignals(
                {FGI: PitSeries(FGI, tuple(_obs(o, a, v, "estimated_lag") for o, a, v in mutated))},
                FGI_FIELDS,
            ).values_at(_dt(cut))
            sv_c = PitSignals(
                {FGI: PitSeries(FGI, tuple(_obs(o, a, v, "estimated_lag") for o, a, v in kept))},
                FGI_FIELDS,
            ).values_at(_dt(cut)) if kept else SignalValues()
            assert sv_a == sv_b == sv_c, trial


def _dt(ms: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)


# ---------------------------------------------------------------------------
# T-16 guards
# ---------------------------------------------------------------------------


class _Secret:
    def values_at(self, decision_at: datetime | None = None) -> SignalValues:
        return SignalValues()

    def __repr__(self) -> str:
        return "SECRET-REPR-DO-NOT-LEAK"


class _CustomBacktestSource(BacktestSignalSource):
    mode = "custom"

    def values_at(self, decision_at: datetime | None) -> SignalValues:
        return SignalValues(fear_greed_index=1)


def _engine(mode: RunMode, **kw: Any) -> StrategyEngine:
    strategy = MagicMock()
    strategy.strategy_id = "s"
    strategy.min_bars_required = 20
    strategy.htf_timeframes = None
    return StrategyEngine(
        strategies=[strategy],
        execution_engine=AsyncMock(),
        risk_manager=MagicMock(),
        market_data=AsyncMock(),
        portfolio=MagicMock(),
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=mode,
        **kw,
    )


def _pit() -> PitSignals:
    return PitSignals({FGI: _fgi_daily(_ms(2026, 3, 1), [10, 20])}, FGI_FIELDS)


def _foreign() -> list[tuple[str, Any]]:
    return [
        ("secret-duck", _Secret()),
        ("magicmock", MagicMock()),
        ("simplenamespace", SimpleNamespace(values_at=lambda at: SignalValues())),
        ("str", "live"),
        ("dict", {"mode": "pit"}),
        ("int", 0),
    ]


class TestT16Guards:
    def test_backtest_defaults_to_unavailable_and_accepts_every_backtest_source(self) -> None:
        assert _engine(RunMode.BACKTEST)._mtf_signals is UNAVAILABLE
        assert _engine(RunMode.BACKTEST, mtf_signals=UNAVAILABLE)._mtf_signals is UNAVAILABLE
        pit = _pit()
        assert _engine(RunMode.BACKTEST, mtf_signals=pit)._mtf_signals is pit
        custom = _CustomBacktestSource()
        assert _engine(RunMode.BACKTEST, mtf_signals=custom)._mtf_signals is custom

    @pytest.mark.parametrize("name", [n for n, _ in _foreign()])
    def test_backtest_rejects_foreign_objects_with_type_names_only(self, name: str) -> None:
        candidate = dict(_foreign())[name]
        with pytest.raises(ValueError, match="BACKTEST StrategyEngine accepts only a backtest "
                                             "signal source") as err:
            _engine(RunMode.BACKTEST, mtf_signals=candidate)
        assert type(candidate).__name__ in str(err.value)
        assert "SECRET-REPR" not in str(err.value)

    @pytest.mark.parametrize("live", [LIVE_SIGNALS, LiveSignals()], ids=["singleton", "fresh"])
    def test_backtest_rejects_the_live_source_even_the_singleton(self, live: LiveSignals) -> None:
        with pytest.raises(ValueError, match="LiveSignals"):
            _engine(RunMode.BACKTEST, mtf_signals=live)  # type: ignore[arg-type]

    # SEC-03 / RISK-05: the BACKTEST guard accepts a REAL subclass only (fail closed).

    @staticmethod
    def _unregister_all() -> None:
        BacktestSignalSource._abc_registry_clear()  # type: ignore[attr-defined]
        BacktestSignalSource._abc_caches_clear()  # type: ignore[attr-defined]

    def test_backtest_rejects_an_abc_registered_live_source(self) -> None:
        BacktestSignalSource.register(LiveSignals)
        try:
            assert isinstance(LIVE_SIGNALS, BacktestSignalSource)  # the vector is real
            for live in (LIVE_SIGNALS, LiveSignals()):
                with pytest.raises(ValueError, match="got LiveSignals"):
                    _engine(RunMode.BACKTEST, mtf_signals=live)
        finally:
            self._unregister_all()
        assert not isinstance(LIVE_SIGNALS, BacktestSignalSource)  # no leak

    def test_backtest_rejects_an_abc_registered_plain_class(self) -> None:
        class _Virtual:
            def values_at(self, decision_at: datetime | None) -> SignalValues:
                return SignalValues()

        BacktestSignalSource.register(_Virtual)
        try:
            assert isinstance(_Virtual(), BacktestSignalSource)  # the vector is real
            with pytest.raises(ValueError, match="got _Virtual"):
                _engine(RunMode.BACKTEST, mtf_signals=_Virtual())  # type: ignore[arg-type]
        finally:
            self._unregister_all()
        assert not isinstance(_Virtual(), BacktestSignalSource)  # no leak

    def test_backtest_rejects_a_spec_mock_of_a_backtest_source(self) -> None:
        mock = MagicMock(spec=PitSignals)
        assert isinstance(mock, BacktestSignalSource)  # the vector is real
        with pytest.raises(ValueError, match="got MagicMock"):
            _engine(RunMode.BACKTEST, mtf_signals=mock)

    def test_backtest_rejects_a_source_that_is_also_a_live_source(self) -> None:
        class _Both(LiveSignals, BacktestSignalSource):  # type: ignore[misc]
            mode = "both"

        assert BacktestSignalSource in type(_Both()).__mro__  # only the LiveSignals clause stops it
        with pytest.raises(ValueError, match="got _Both"):
            _engine(RunMode.BACKTEST, mtf_signals=_Both())  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    def test_paper_and_live_default_to_and_accept_only_the_live_singleton(
        self, mode: RunMode
    ) -> None:
        assert _engine(mode)._mtf_signals is LIVE_SIGNALS
        assert _engine(mode, mtf_signals=LIVE_SIGNALS)._mtf_signals is LIVE_SIGNALS

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    @pytest.mark.parametrize("name", ["unavailable", "pit", "custom-backtest", "fresh-live",
                                      *[n for n, _ in _foreign()]])
    def test_paper_and_live_reject_every_other_object(self, mode: RunMode, name: str) -> None:
        special: dict[str, Any] = {
            "unavailable": UNAVAILABLE,
            "pit": _pit(),
            "custom-backtest": _CustomBacktestSource(),
            "fresh-live": LiveSignals(),  # identity, not isinstance
        }
        candidate = special[name] if name in special else dict(_foreign())[name]
        with pytest.raises(ValueError, match=f"{mode.value} StrategyEngine accepts only the "
                                             "live signal source") as err:
            _engine(mode, mtf_signals=candidate)
        assert type(candidate).__name__ in str(err.value)
        assert "SECRET-REPR" not in str(err.value)

    def test_the_orchestrator_default_path_needs_no_edit(self) -> None:
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2] / "apps" / "api" / "services"
               / "run_orchestrator.py").read_text()
        assert "mtf_signals" not in src  # default path: no keyword, so it is not edited

    def test_the_engine_dispatches_to_its_source_with_decision_at(self) -> None:
        class _Spy(BacktestSignalSource):
            mode = "spy"

            def __init__(self) -> None:
                self.calls: list[datetime | None] = []

            def values_at(self, decision_at: datetime | None) -> SignalValues:
                self.calls.append(decision_at)
                return SignalValues(fear_greed_index=7, btc_dominance=51.0)

        spy = _Spy()
        engine = _engine(RunMode.BACKTEST, mtf_signals=spy)
        stamp = datetime(2026, 3, 10, 1, tzinfo=UTC)
        ctx = engine._build_mtf_context(datetime(2026, 3, 10, tzinfo=UTC), decision_at=stamp)
        assert spy.calls == [stamp] and spy.calls[0] is stamp
        assert ctx == MultiTimeframeContext(fear_greed_index=7, btc_dominance=51.0)
        engine._build_mtf_context(datetime(2026, 3, 10, tzinfo=UTC))  # direct callers: no stamp
        assert spy.calls[1] is None

    def test_unavailable_with_htf_present_gives_a_default_signal_context(self) -> None:
        engine = _engine(RunMode.BACKTEST)
        assert engine._build_mtf_context(datetime(2026, 3, 10, tzinfo=UTC)) is None
        engine._htf_bars = {}
        ctx = engine._build_mtf_context(datetime(2026, 3, 10, tzinfo=UTC))
        assert ctx == MultiTimeframeContext()


# ---------------------------------------------------------------------------
# engine-level harness
# ---------------------------------------------------------------------------


class _Recording:
    """Per-bar record of what the engine fed a strategy and what it answered."""

    def __init__(self) -> None:
        self.rows: list[tuple[datetime, MultiTimeframeContext | None, tuple[Any, ...]]] = []


def _record(strategy: BaseStrategy) -> _Recording:
    rec = _Recording()
    original = strategy.on_bar

    def on_bar(bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
               ) -> list[Signal]:
        signals = original(bars, mtf_context=mtf_context)
        rec.rows.append((
            bars[-1].timestamp + H,
            mtf_context,
            tuple((s.direction, s.confidence, s.target_position, s.symbol) for s in signals),
        ))
        return signals

    strategy.on_bar = on_bar  # type: ignore[method-assign]
    return rec


async def _run(
    cls: type[BaseStrategy],
    source: BacktestSignalSource | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, _Recording]:
    strategy = cls("s1", {})
    rec = _record(strategy)
    if source is not None:

        def build(**kw: Any) -> StrategyEngine:
            return StrategyEngine(mtf_signals=source, **kw)

        monkeypatch.setattr(backtest_mod, "StrategyEngine", build)
    else:
        monkeypatch.setattr(backtest_mod, "StrategyEngine", StrategyEngine)  # undo an earlier run
    runner = BacktestRunner(
        [strategy],
        [SYM],
        TimeFrame.ONE_HOUR,
        seed=7,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
    )
    result = await runner.run({SYM: _bars()})
    return result, rec


_CONSUMERS = [RSIMeanReversionStrategy, DCARSIHybridStrategy]


class TestT19EnginePointInTime:
    K = 150  # index (warm-up included) of the bar whose close is the cut

    def _sources(self) -> tuple[PitSignals, PitSignals, int]:
        close_k = to_ms(_close_of(self.K))
        first_day = to_ms(FIRST_OPEN) - 20 * D_MS
        s = _full_source(first_day, 40)
        s_prime = _full_source(first_day, 40, flip_after_ms=close_k)
        return s, s_prime, close_k

    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    async def test_s_and_s_prime_agree_through_bar_k_and_the_control_diverges_after(
        self, cls: type[BaseStrategy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s, s_prime, _close_k = self._sources()
        res_a, rec_a = await _run(cls, s, monkeypatch)
        res_b, rec_b = await _run(cls, s_prime, monkeypatch)
        cut = _close_of(self.K)

        before_a = [r for r in rec_a.rows if r[0] <= cut]
        before_b = [r for r in rec_b.rows if r[0] <= cut]
        assert len(before_a) > 50 and before_a == before_b  # contexts AND signals, bar by bar
        assert any(r[1] is not None and r[1].fear_greed_index is not None for r in before_a)

        curve_a = [(p.timestamp, p.equity) for p in res_a.equity_curve if p.timestamp <= cut]
        curve_b = [(p.timestamp, p.equity) for p in res_b.equity_curve if p.timestamp <= cut]
        assert len(curve_a) > 50 and curve_a == curve_b

        def closed(res: Any) -> list[tuple[Any, ...]]:
            # a round trip that closed by the cut is fully determined by pre-cut information
            # (open round trips are not compared: a later scale-in or exit would change them)
            return [(t.entry_at, t.entry_price, t.quantity, t.exit_at, t.exit_price,
                     t.realised_pnl, t.symbol, t.strategy_id)
                    for t in res.trades if t.exit_at <= cut]

        assert closed(res_a) and closed(res_a) == closed(res_b)

        # control: the series DO differ, so contexts after the cut differ (the test can fail)
        after_a = [r[1] for r in rec_a.rows if r[0] > cut]
        after_b = [r[1] for r in rec_b.rows if r[0] > cut]
        assert after_a != after_b

    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    async def test_a_pit_source_changes_the_outcome_versus_unavailable(
        self, cls: type[BaseStrategy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s, _, _ = self._sources()
        res_pit, _ = await _run(cls, s, monkeypatch)
        res_none, rec_none = await _run(cls, None, monkeypatch)
        assert all(r[1] is None for r in rec_none.rows)  # UNAVAILABLE: no context at all
        assert [t.realised_pnl for t in res_pit.trades] != [t.realised_pnl for t in res_none.trades]

    async def test_decision_at_is_the_object_that_drove_the_bar_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advanced: list[datetime] = []
        real_advance = BarClock.advance

        def advance(self: BarClock, to: datetime) -> None:
            advanced.append(to)
            real_advance(self, to)

        class _Spy(BacktestSignalSource):
            mode = "spy"

            def __init__(self) -> None:
                self.stamps: list[datetime | None] = []

            def values_at(self, decision_at: datetime | None) -> SignalValues:
                self.stamps.append(decision_at)
                return SignalValues()

        spy = _Spy()
        monkeypatch.setattr(BarClock, "advance", advance)
        await _run(RSIMeanReversionStrategy, spy, monkeypatch)
        assert spy.stamps and all(s is not None for s in spy.stamps)
        assert all(any(s is a for a in advanced) for s in spy.stamps)  # SAME object (CF-23-18)
        closes = [b.timestamp + H for b in _bars()]
        # one per processed bar (after the runner's warm-up), the bar CLOSE, consecutive
        assert len(spy.stamps) > 100 and spy.stamps == closes[len(closes) - len(spy.stamps):]


# ---------------------------------------------------------------------------
# T-20 None audit
# ---------------------------------------------------------------------------


def _all_none_ctx_with_htf() -> MultiTimeframeContext:
    return MultiTimeframeContext(htf_bars={"4h": {SYM: []}, "1d": {SYM: []}})


def _drive(
    strategy: BaseStrategy, ctx: MultiTimeframeContext | None, *, meta: bool = True
) -> list[tuple[Any, ...]]:
    bars = _bars()
    out: list[tuple[Any, ...]] = []
    strategy.on_start("t20")
    for i in range(20, len(bars) + 1):
        for s in strategy.on_bar(bars[:i], mtf_context=ctx):
            out.append((i, s.direction, s.confidence, s.target_position, s.symbol, s.strategy_id,
                        s.metadata if meta else None))
    return out


class TestT20NoneAudit:
    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    def test_consumer_none_equals_an_all_none_context_with_htf(
        self, cls: type[BaseStrategy]
    ) -> None:
        none_run = _drive(cls("a", {}), None)
        all_none = _drive(cls("a", {}), _all_none_ctx_with_htf())
        assert len(none_run) > 3 and none_run == all_none

    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    def test_control_a_populated_context_does_change_the_consumer(
        self, cls: type[BaseStrategy]
    ) -> None:
        populated = MultiTimeframeContext(fear_greed_index=8, btc_dominance=63.0,
                                          yield_curve_spread=-1.2, whale_net_flow=9e7)
        assert _drive(cls("a", {}), None) != _drive(cls("a", {}), populated)

    def test_regime_label_of_an_all_none_context_is_none(self) -> None:
        assert StrategyEngine._fgi_to_regime(_all_none_ctx_with_htf().fear_greed_index) is None

    def _v2_strategy(self, model: Any, monkeypatch: pytest.MonkeyPatch) -> ModelStrategy:
        strategy = ModelStrategy("m", {"feature_window": 100, "prediction_threshold": 0.0})
        strategy._model = model
        strategy._model_loaded = True
        strategy._active_schema_version = 2
        monkeypatch.setattr(strategy, "_check_model_version", lambda **_k: None)
        return strategy

    def test_v2_feature_vector_defaults_equal_between_none_and_all_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = self._v2_strategy(_InertModel(), monkeypatch)
        bars = _bars()[:120]
        vec_none = strategy._build_feature_vector(bars, mtf_context=None)
        vec_ctx = strategy._build_feature_vector(bars, mtf_context=_all_none_ctx_with_htf())
        assert vec_none == vec_ctx and len(vec_none) == 14
        assert vec_none[10:] == [0.0, 0.5, 0.0, 0.0]

    def test_an_inert_v2_model_decides_identically_under_any_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contexts = [
            None,
            _all_none_ctx_with_htf(),
            MultiTimeframeContext(fear_greed_index=5, fear_greed_index_7d_ago=95,
                                  btc_dominance=70.0, btc_dominance_7d_ago=40.0),
        ]
        outs = []
        for ctx in contexts:
            strategy = self._v2_strategy(_InertModel(), monkeypatch)
            outs.append(_drive(strategy, ctx, meta=False))  # metadata echoes the features
        assert len(outs[0]) > 3 and outs[0] == outs[1] == outs[2]

    def test_control_a_context_trained_model_does_react(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        populated = MultiTimeframeContext(fear_greed_index=5, fear_greed_index_7d_ago=95)
        a = _drive(self._v2_strategy(_ContextModel(), monkeypatch), None, meta=False)
        b = _drive(self._v2_strategy(_ContextModel(), monkeypatch), populated, meta=False)
        assert a != b


class _InertModel:
    """A v2 model whose importance on the four context features is 0.0 (QD-06)."""

    n_features_in_ = 14

    def predict(self, x: Any) -> Any:
        row = x[0]
        # decision from the first ten features only; deterministic, class 0/1/2
        return [int(abs(sum(row[:10])) * 1000) % 3]

    def predict_proba(self, x: Any) -> Any:
        return [[0.2, 0.3, 0.9]]


class _ContextModel(_InertModel):
    def predict(self, x: Any) -> Any:
        return [int(abs(sum(x[0][:10])) * 1000 + 777 * x[0][11]) % 3]  # reads fgi_level_norm

