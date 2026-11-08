"""WP2.4 / P2: run-level selection, the run report and strategy consumption.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-05, SY-24-06, SY-24-07, SY-24-17,
WP24-T-21, T-22, T-23, T-25, T-26) and the P1 carry-forward CF-P1-01
(``reports/vp2-wp2.4/p1-synthesis.md`` section 6).

* T-21  coverage floor per field, report types, ``last_mtf_context`` reset per run
* T-22  ``consumes_mtf_signals`` declared per registry strategy; ensemble union; base default
* T-23  status truth table; a non-consumer gets ``UNAVAILABLE`` and identical trades (G8)
* T-25  determinism: same seed and series -> byte-identical results and metadata
* T-26  a daily run with FGI at 24h lag sees FGI(D) at the close of bar D (QD T-02)
* CF-P1-01  a naive timestamp or a bad source fails BEFORE the run id exists
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import TimeFrame
from data.point_in_time import (
    FIELD_MAP,
    MIN_COVERAGE_PPM,
    NOT_SERVED_FIELDS,
    SERIES_RULES,
    Observation,
    PitError,
    PitSeries,
    to_ms,
)
from tests.integration.fakes.bar_paths import scripted_path
from trading import backtest as backtest_mod
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.models import Signal
from trading.mtf_signals import (
    ALL_MTF_SIGNAL_FIELDS,
    LIVE_SIGNALS,
    MTF_CONTEXT_VERSION,
    UNAVAILABLE,
    BacktestSignalSource,
    MtfContextReport,
    PitSignals,
    SignalValues,
    _now_field_of,
    check_backtest_source,
    select_mtf_signals,
)
from trading.strategies.breakout import BreakoutStrategy
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy
from trading.strategies.ensemble import EnsembleStrategy
from trading.strategies.grid_trading import GridTradingStrategy
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategies.model_strategy import ModelStrategy
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading.strategies.sl_tp_reversion import SLTPReversionStrategy
from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

SYM = "BTC/USDT"
H = timedelta(hours=1)
H_MS = 3_600_000
D_MS = 86_400_000
FIRST_OPEN = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)  # open of the first post-warm-up bar
FGI = "alternative_me:fgi"
T10 = "fred:T10Y2Y"
DOM = "coingecko:btc_dominance"
WHALE = "whale:net_flow_1h"
LEGACY_COST = get_cost_profile("legacy_backtest_v0")
BRACKET = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_tp_multiplier": 3.0,
    "bracket_atr_period": 14,
}


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _closes(n: int = 300) -> list[float]:
    # a deterministic zig-zag with trend legs: RSI crosses and momentum breakouts both happen
    return [
        round(100 + 14 * ((i // 25) % 2 * 2 - 1) * ((i % 25) / 25) + i * 0.05, 2)
        for i in range(n)
    ]


def _bars(
    n: int = 300, timeframe: TimeFrame = TimeFrame.ONE_HOUR, first_open: datetime = FIRST_OPEN
) -> list[OHLCVBar]:
    return scripted_path(_closes(n), first_open=first_open, symbol=SYM, timeframe=timeframe)


def _fgi_daily(first_day_ms: int, values: Sequence[int], lag_ms: int = D_MS) -> PitSeries:
    """FGI day D 00:00 UTC, available D + 24h (the ingestion rule, UD-5 (a))."""
    return PitSeries(
        FGI,
        tuple(
            Observation(
                first_day_ms + k * D_MS,
                first_day_ms + k * D_MS + lag_ms,
                Decimal(v),
                "estimated_lag",
            )
            for k, v in enumerate(values)
        ),
    )


def _hourly(series_id: str, first_ms: int, hours: int, value: int = 5) -> PitSeries:
    """Recorded hourly snapshots, each available one minute after its hour."""
    return PitSeries(
        series_id,
        tuple(
            Observation(
                first_ms + k * H_MS,
                first_ms + k * H_MS + 60_000,
                Decimal(value + k % 3),
                "recorded",
            )
            for k in range(hours)
        ),
    )


def _t10_daily(first_day_ms: int, days: int) -> PitSeries:
    return PitSeries(
        T10,
        tuple(
            Observation(
                first_day_ms + k * D_MS,
                first_day_ms + k * D_MS + 3 * D_MS,
                Decimal("0.5") + Decimal(k % 5) / 10,
                "estimated_lag",
            )
            for k in range(days)
        ),
    )


def _full(first_day_ms: int, days: int = 40) -> dict[str, PitSeries]:
    return {
        FGI: _fgi_daily(first_day_ms, [20 + (k * 7) % 60 for k in range(days)]),
        T10: _t10_daily(first_day_ms, days),
        DOM: _hourly(DOM, first_day_ms, days * 24, 50),
        WHALE: _hourly(WHALE, first_day_ms, days * 24, 1000),
    }


FIRST_DAY_MS = to_ms(FIRST_OPEN) - 20 * D_MS
ALL_FIELDS = frozenset(FIELD_MAP)


def _source(series: dict[str, PitSeries], fields: frozenset[str] | None = None) -> PitSignals:
    return PitSignals(series, ALL_FIELDS if fields is None else fields)


class _Recording:
    def __init__(self) -> None:
        self.contexts: list[MultiTimeframeContext | None] = []


def _record(strategy: BaseStrategy) -> _Recording:
    rec = _Recording()
    original = strategy.on_bar

    def on_bar(
        bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        rec.contexts.append(mtf_context)
        return original(bars, mtf_context=mtf_context)

    strategy.on_bar = on_bar  # type: ignore[method-assign]
    return rec


def _rsi_runner(source: BacktestSignalSource | None) -> tuple[BacktestRunner, _Recording]:
    strategy = RSIMeanReversionStrategy("s1", {})
    rec = _record(strategy)
    runner = BacktestRunner(
        [strategy], [SYM], TimeFrame.ONE_HOUR, seed=7, cost_profile=LEGACY_COST, mtf_signals=source
    )
    return runner, rec


class _YieldOnly(RSIMeanReversionStrategy):
    """Test-local (NOT in the registry): consumes only the yield-curve spread (E2c)."""

    consumes_mtf_signals = frozenset({"yield_curve_spread"})


class _Fgi7d(RSIMeanReversionStrategy):
    """Test-local (NOT in the registry): the rsi set plus the FGI 7d field (E2e)."""

    consumes_mtf_signals = RSIMeanReversionStrategy.consumes_mtf_signals | {
        "fear_greed_index_7d_ago"
    }


_VOLATILE_KEYS = frozenset({"run_id", "trade_id", "order_id", "fill_id"})


def _scrub(node: Any) -> Any:
    """Drop the per-run random identifiers; everything else must be reproducible."""
    if isinstance(node, dict):
        return {k: _scrub(v) for k, v in node.items() if k not in _VOLATILE_KEYS}
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    return node


def _decisions(runner: BacktestRunner, bars: list[OHLCVBar]) -> list[int]:
    """close_k in epoch ms for k in [W, N): the independent definition of the decision set."""
    w = runner._warmup_bars
    return [to_ms(b.timestamp + H) for b in bars[w:]]


def _oracle_ppm(
    series: PitSeries, decisions: Iterable[int], *, back_ms: int, max_age_ms: int
) -> int:
    """Brute-force reference of the coverage rule (no bisect, no shared helper)."""

    def value_at(t: int) -> bool:
        seen = [o for o in series.observations if o.available_ms <= t]
        if not seen:
            return False
        top = max(o.observed_ms for o in seen)
        first = min(o.available_ms for o in series.observations if o.observed_ms == top)
        return t - first <= max_age_ms

    ds = list(decisions)
    ok = sum(1 for d in ds if value_at(d) and (back_ms == 0 or value_at(d - back_ms)))
    return ok * 1_000_000 // len(ds) if ds else 0


# ---------------------------------------------------------------------------
# T-21: coverage floor per field, report types, reset per run
# ---------------------------------------------------------------------------


class TestT21CoverageAndReport:
    async def test_full_coverage_serves_every_consumed_field_and_the_report_says_pit(self) -> None:
        runner, rec = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        await runner.run({SYM: _bars()})
        report = runner.last_mtf_context
        assert isinstance(report, MtfContextReport)
        assert (report.source, report.status, report.reason) == ("pit", "pit", None)
        meta = report.to_metadata()
        assert meta["fields"]["fear_greed_index"] == "pit"
        assert meta["fields"]["fed_funds_rate"] == "not_served"
        got = [c for c in rec.contexts if c is not None]
        assert got and all(c.fear_greed_index is not None for c in got)
        assert all(c.btc_dominance is not None and c.whale_net_flow is not None for c in got)
        assert all(isinstance(c.fear_greed_index, int) for c in got)

    @pytest.mark.parametrize(("uncovered", "on"), [(2, True), (3, False)])
    async def test_990000_ppm_boundary_is_all_or_nothing_per_field(
        self, uncovered: int, on: bool
    ) -> None:
        # dominance stops early so exactly ``uncovered`` of the decisions lose their value
        bars = _bars()
        probe, _ = _rsi_runner(None)
        decisions = _decisions(probe, bars)
        rule = SERIES_RULES[DOM]
        chosen: PitSeries | None = None
        for stop in range(len(bars) + 60):
            cand = _hourly(DOM, FIRST_DAY_MS, 20 * 24 + stop)
            ppm = _oracle_ppm(cand, decisions, back_ms=0, max_age_ms=rule.max_age_ms)
            missing = len(decisions) - ppm_to_count(ppm, len(decisions))
            if missing == uncovered:
                chosen = cand
                break
        assert chosen is not None, "no stop hour gives the wanted number of uncovered decisions"
        series = _full(FIRST_DAY_MS)
        series[DOM] = chosen
        runner, rec = _rsi_runner(_source(series))
        await runner.run({SYM: bars})
        report = runner.last_mtf_context
        assert report is not None
        expected = _oracle_ppm(chosen, decisions, back_ms=0, max_age_ms=rule.max_age_ms)
        assert (expected >= MIN_COVERAGE_PPM) is on
        row = report.to_metadata()["series"][DOM]
        assert row["coverage_ppm"] == expected
        status = report.to_metadata()["fields"]["btc_dominance"]
        values = [c.btc_dominance for c in rec.contexts if c is not None]
        if on:
            assert status == "pit"
            # an ON field still yields None on its (<= 1 %) uncovered bars
            assert values.count(None) == uncovered and len(values) - values.count(None) > 0
        else:
            assert status == "insufficient_coverage"
            assert values and all(v is None for v in values)  # None on EVERY bar
            assert report.status == "partial"
            # the other consumed fields are unaffected
            assert all(c.fear_greed_index is not None for c in rec.contexts if c is not None)

    async def test_a_field_without_rows_is_no_source_and_a_half_covered_one_is_insufficient(
        self,
    ) -> None:
        series = _full(FIRST_DAY_MS)
        del series[WHALE]
        # FGI starts in the middle of the trading window: 50 % coverage (QD T-20)
        mid = to_ms(FIRST_OPEN) + 130 * H_MS
        series[FGI] = _fgi_daily(mid - mid % D_MS, [30, 40, 50, 60, 70, 80, 90, 55, 65, 75])
        runner, rec = _rsi_runner(_source(series))
        await runner.run({SYM: _bars()})
        report = runner.last_mtf_context
        assert report is not None
        fields = report.to_metadata()["fields"]
        assert fields["whale_net_flow"] == "no_source"
        assert fields["fear_greed_index"] == "insufficient_coverage"
        assert fields["fear_greed_index_7d_ago"] == "insufficient_coverage"
        assert report.status == "partial"
        assert all(c.fear_greed_index is None for c in rec.contexts if c is not None)
        assert all(c.whale_net_flow is None for c in rec.contexts if c is not None)

    def test_a_7d_field_needs_seven_extra_days_of_history(self) -> None:
        # the 'now' leg is covered from the first decision, the 7d leg is not
        dec = [to_ms(FIRST_OPEN) + k * H_MS for k in range(1, 100)]
        start = to_ms(FIRST_OPEN) - 3 * D_MS  # only 3 days before the first decision
        src = _source({FGI: _fgi_daily(start, list(range(30, 50)))})
        _, report = src.select([datetime.fromtimestamp(d / 1000, UTC) for d in dec], ALL_FIELDS)
        fields = report.to_metadata()["fields"]
        assert fields["fear_greed_index"] == "pit"
        assert fields["fear_greed_index_7d_ago"] == "insufficient_coverage"

    @pytest.mark.parametrize(
        ("inside", "ppm", "status"), [(99, 990_000, "pit"), (98, 980_000, "insufficient_coverage")]
    )
    def test_the_990000_floor_is_inclusive(self, inside: int, ppm: int, status: str) -> None:
        # CR-01: exactly 990000 ppm is ON (``>=``); one decision less is OFF
        src = _source({DOM: _hourly(DOM, to_ms(FIRST_OPEN), 200)})
        decisions = [FIRST_OPEN + k * H for k in range(1, inside + 1)] + [
            FIRST_OPEN + timedelta(days=200 + j) for j in range(100 - inside)
        ]
        assert len(decisions) == 100
        _, report = src.select(decisions, frozenset({"btc_dominance"}))
        meta = report.to_metadata()
        assert meta["series"][DOM]["coverage_ppm"] == ppm
        assert meta["fields"]["btc_dominance"] == status

    def test_a_field_outside_the_offered_fields_on_is_never_reported_pit(self) -> None:
        # CR-03: the source offers FGI only; a consumed dominance must stay no_source
        src = _source(_full(FIRST_DAY_MS), frozenset({"fear_greed_index"}))
        served, report = src.select(
            _decision_list(), frozenset({"fear_greed_index", "btc_dominance"})
        )
        fields = report.to_metadata()["fields"]
        assert fields["fear_greed_index"] == "pit"
        assert fields["btc_dominance"] == "no_source"
        assert isinstance(served, PitSignals) and served.fields_on == {"fear_greed_index"}
        assert report.status == "partial"

    def test_the_series_rows_carry_the_exact_facts(self) -> None:
        # CR-04: concrete SeriesReport values, one per series, absent series included
        day0 = to_ms(datetime(2026, 3, 8, tzinfo=UTC))
        series = {FGI: _fgi_daily(day0, [30, 40, 50]), DOM: _hourly(DOM, day0, 4)}
        _, report = _source(series).select(_decision_list(), frozenset({"fear_greed_index"}))
        rows = report.to_metadata()["series"]
        assert rows[FGI] == {
            "rows": 3,
            "first_available_at": "2026-03-09T00:00:00+00:00",
            "last_available_at": "2026-03-11T00:00:00+00:00",
            "lag_rule": "observed+24h|recorded",
            "max_age_s": 172_800,
            "coverage_ppm": 1_000_000,
        }
        assert rows[DOM] == {
            "rows": 4,
            "first_available_at": "2026-03-08T00:01:00+00:00",
            "last_available_at": "2026-03-08T03:01:00+00:00",
            "lag_rule": "recorded",
            "max_age_s": 21_600,
            "coverage_ppm": 0,
        }
        assert rows[T10]["rows"] == 0
        assert rows[T10]["first_available_at"] is None and rows[T10]["last_available_at"] is None
        assert rows[T10]["coverage_ppm"] == 0 and rows[T10]["max_age_s"] == 7 * 86_400
        assert rows[WHALE]["rows"] == 0 and rows[WHALE]["max_age_s"] == 7_200

    def test_a_series_with_no_rows_is_no_source_and_all_no_source_is_no_history(self) -> None:
        # CR-05: ``PitSeries(DOM, ())`` is "no_source", not insufficient_coverage
        _, report = _source({DOM: PitSeries(DOM, ())}).select(
            _decision_list(), frozenset({"btc_dominance"})
        )
        assert dict(report.fields)["btc_dominance"] == "no_source"
        assert (report.status, report.source, report.reason) == (
            "unavailable", "unavailable", "no_history",
        )  # fmt: skip

    async def test_the_runner_selects_at_the_decision_times_the_engine_processes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CR-02: the runner's decision generator == the ``decision_at`` of every processed bar
        chosen: list[list[datetime]] = []
        processed: list[datetime] = []
        real_select = backtest_mod.select_mtf_signals
        real_process = StrategyEngine._process_bar

        def select_spy(source: Any, decisions: Iterable[datetime], fields: Any) -> Any:
            items = list(decisions)
            chosen.append(items)
            return real_select(source, items, fields)

        async def process_spy(self: StrategyEngine, *args: Any, **kwargs: Any) -> Any:
            processed.append(kwargs["decision_at"])
            return await real_process(self, *args, **kwargs)

        monkeypatch.setattr(backtest_mod, "select_mtf_signals", select_spy)
        monkeypatch.setattr(StrategyEngine, "_process_bar", process_spy)
        runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        bars = _bars()
        await runner.run({SYM: bars})
        assert runner._warmup_bars > 0 and len(chosen) == 1
        assert chosen[0] == processed
        assert chosen[0] == [b.timestamp + H for b in bars[runner._warmup_bars :]]

    async def test_zero_trading_steps_is_off(self) -> None:
        _, report = _source(_full(FIRST_DAY_MS)).select([], ALL_FIELDS)
        assert report.status == "unavailable" and report.source == "unavailable"
        assert set(report.to_metadata()["fields"].values()) <= {
            "no_source",
            "insufficient_coverage",
            "not_served",
        }

    async def test_metadata_is_json_safe_and_complete(self) -> None:
        runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        await runner.run({SYM: _bars()})
        report = runner.last_mtf_context
        assert report is not None
        meta = report.to_metadata()
        assert list(meta) == [
            "version", "source", "status", "fields", "series", "series_digest",
            "ingest_cutoff", "reason",
        ]  # fmt: skip
        assert meta["version"] == MTF_CONTEXT_VERSION == "wp24-1"
        assert set(meta["fields"]) == set(ALL_MTF_SIGNAL_FIELDS) and len(meta["fields"]) == 9
        assert set(meta["series"]) == set(SERIES_RULES)
        assert meta["ingest_cutoff"] is None  # no store before Phase B

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, val in node.items():
                    assert isinstance(key, str)
                    walk(val)
            else:
                assert node is None or type(node) in (int, str, bool), type(node)

        walk(meta)
        assert json.loads(json.dumps(meta)) == meta
        for row in meta["series"].values():
            assert set(row) == {
                "rows", "first_available_at", "last_available_at", "lag_rule", "max_age_s",
                "coverage_ppm",
            }  # fmt: skip
            assert datetime.fromisoformat(row["first_available_at"]).tzinfo is not None
            assert datetime.fromisoformat(row["last_available_at"]).tzinfo is not None
        assert meta["series"][FGI]["max_age_s"] == 48 * 3600
        assert len(meta["series_digest"]) == 64
        # a fresh dict every call: mutating it never reaches the report
        meta["fields"]["fear_greed_index"] = "x"
        assert report.to_metadata()["fields"]["fear_greed_index"] == "pit"

    async def test_last_mtf_context_is_reset_on_every_run(self) -> None:
        runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        assert runner.last_mtf_context is None
        await runner.run({SYM: _bars()})
        assert runner.last_mtf_context is not None
        with pytest.raises(ValueError, match="Insufficient bars"):
            await runner.run({SYM: _bars(60)[:70]})  # fails after the reset, before selection
        assert runner.last_mtf_context is None
        await runner.run({SYM: _bars()})
        assert runner.last_mtf_context is not None

    async def test_backtest_result_has_no_mtf_field(self) -> None:
        runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        result = await runner.run({SYM: _bars()})
        assert not [f for f in vars(result) if "mtf" in f.lower()]
        assert not [f for f in type(result).model_fields if "mtf" in f.lower()]


def ppm_to_count(ppm: int, total: int) -> int:
    """Smallest covered count whose floor-ppm equals ``ppm`` (the oracle works in ppm)."""
    for covered in range(total + 1):
        if covered * 1_000_000 // total == ppm:
            return covered
    raise AssertionError("ppm not reachable")


# ---------------------------------------------------------------------------
# CF-P1-01: validate before the run id and the engine exist
# ---------------------------------------------------------------------------


class TestCfP101ValidateBeforeTheRun:
    class _Boom(Exception):
        pass

    def _no_engine(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        built: list[str] = []

        def forbidden(**kw: Any) -> StrategyEngine:
            built.append("engine")
            raise self._Boom

        monkeypatch.setattr(backtest_mod, "StrategyEngine", forbidden)
        return built

    async def test_a_naive_timestamp_fails_before_the_run_id_and_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = self._no_engine(monkeypatch)
        # OHLCVBar normalises a naive stamp to UTC, so a naive bar only exists via model_copy
        naive = [
            b.model_copy(update={"timestamp": b.timestamp.replace(tzinfo=None)}) for b in _bars()
        ]
        assert naive[0].timestamp.tzinfo is None
        runner, rec = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        with pytest.raises(PitError) as err:
            await runner.run({SYM: naive})
        assert str(err.value) == "point-in-time check failed (pit_naive_time)"
        assert built == [] and rec.contexts == []
        assert runner.last_execution_engine is None and runner.last_portfolio is None

    async def test_the_selection_runs_before_the_run_id_is_drawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CR-07: a naive bar must raise PitError even if the run id could not be drawn
        def no_uuid() -> Any:
            raise self._Boom

        monkeypatch.setattr(backtest_mod.uuid, "uuid4", no_uuid)
        naive = [
            b.model_copy(update={"timestamp": b.timestamp.replace(tzinfo=None)}) for b in _bars()
        ]
        runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        with pytest.raises(PitError) as err:
            await runner.run({SYM: naive})
        assert str(err.value) == "point-in-time check failed (pit_naive_time)"

    async def test_a_pit_error_on_a_naive_decision_time_is_raised_by_the_selection_itself(
        self,
    ) -> None:
        with pytest.raises(PitError) as err:
            _source(_full(FIRST_DAY_MS)).select([datetime(2026, 3, 10, 1)], ALL_FIELDS)
        assert str(err.value) == "point-in-time check failed (pit_naive_time)"

    @pytest.mark.parametrize(
        "bad", [LIVE_SIGNALS, object(), "pit", 3], ids=["live", "object", "str", "int"]
    )
    def test_a_bad_source_fails_at_construction_with_the_type_name_only(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="mtf_signals must be a BacktestSignalSource") as err:
            BacktestRunner(
                [MACrossoverStrategy("s", {})], [SYM], TimeFrame.ONE_HOUR, mtf_signals=bad
            )
        assert str(err.value).endswith(f"got {type(bad).__name__}")

    async def test_a_bad_source_set_after_construction_still_fails_before_the_run_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = self._no_engine(monkeypatch)
        runner, _ = _rsi_runner(None)
        runner._mtf_signals = LIVE_SIGNALS  # type: ignore[assignment]
        with pytest.raises(ValueError, match="LiveSignals"):
            await runner.run({SYM: _bars()})
        assert built == []

    async def test_a_foreign_source_fails_even_for_a_non_consumer(self) -> None:
        runner = BacktestRunner(
            [MACrossoverStrategy("s", {})], [SYM], TimeFrame.ONE_HOUR, cost_profile=LEGACY_COST
        )
        runner._mtf_signals = object()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="object"):
            await runner.run({SYM: _bars()})

    def test_a_registered_virtual_subclass_is_refused(self) -> None:
        class _Virtual:
            def values_at(self, decision_at: datetime | None) -> SignalValues:
                return SignalValues()

        BacktestSignalSource.register(_Virtual)
        with pytest.raises(ValueError, match="_Virtual"):
            check_backtest_source(_Virtual())

    def test_a_real_custom_subclass_is_valid_for_the_guard_but_not_selectable(self) -> None:
        class _Custom(BacktestSignalSource):
            mode = "custom"

            def values_at(self, decision_at: datetime | None) -> SignalValues:
                return SignalValues()

        check_backtest_source(_Custom())
        with pytest.raises(ValueError, match="unsupported mtf_signals source _Custom"):
            select_mtf_signals(_Custom(), [], {"fear_greed_index"})


# ---------------------------------------------------------------------------
# T-22: consumption declaration (SY-24-06)
# ---------------------------------------------------------------------------

_FOUR = frozenset({"fear_greed_index", "btc_dominance", "yield_curve_spread", "whale_net_flow"})
_MODEL = frozenset(
    {"fear_greed_index", "fear_greed_index_7d_ago", "btc_dominance", "btc_dominance_7d_ago"}
)
_EXPECTED: dict[type[BaseStrategy], frozenset[str]] = {
    RSIMeanReversionStrategy: _FOUR,
    DCARSIHybridStrategy: _FOUR,
    ModelStrategy: _MODEL,
    MACrossoverStrategy: frozenset(),
    BreakoutStrategy: frozenset(),
    GridTradingStrategy: frozenset(),
    SLTPReversionStrategy: frozenset(),
    MomentumBreakoutStrategy: frozenset(),
    SmokeRoundtripStrategy: frozenset(),
}


def _own_declaration(cls: type) -> ast.AnnAssign | None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    klass = tree.body[0]
    assert isinstance(klass, ast.ClassDef)
    for node in klass.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "consumes_mtf_signals"
        ):
            return node
    return None


class TestT22Consumption:
    @pytest.mark.parametrize("cls", list(_EXPECTED), ids=lambda c: c.__name__)
    def test_every_registry_strategy_declares_its_own_set_with_the_spec_value(
        self, cls: type[BaseStrategy]
    ) -> None:
        decl = _own_declaration(cls)
        assert decl is not None, f"{cls.__name__} must declare consumes_mtf_signals in its body"
        assert "ClassVar[frozenset[str]]" in ast.unparse(decl.annotation)
        assert cls.consumes_mtf_signals == _EXPECTED[cls]
        assert type(cls.consumes_mtf_signals) is frozenset
        assert cls.consumes_mtf_signals <= ALL_MTF_SIGNAL_FIELDS

    def test_the_declarations_cover_exactly_the_api_registry(self) -> None:
        from api.routers.runs import _get_strategy_registry

        assert set(_get_strategy_registry().values()) == set(_EXPECTED)

    def test_the_base_default_is_all_nine_fields(self) -> None:
        assert BaseStrategy.consumes_mtf_signals == ALL_MTF_SIGNAL_FIELDS
        assert len(ALL_MTF_SIGNAL_FIELDS) == 9
        assert ALL_MTF_SIGNAL_FIELDS == frozenset(FIELD_MAP) | NOT_SERVED_FIELDS
        names = set(MultiTimeframeContext.__dataclass_fields__) - {"htf_bars"}
        assert ALL_MTF_SIGNAL_FIELDS == names

    def test_an_undeclared_strategy_is_a_consumer_of_everything(self) -> None:
        class _New(BaseStrategy):
            def on_bar(
                self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
            ) -> list[Signal]:
                return []

        assert _New("n").mtf_signal_fields() == ALL_MTF_SIGNAL_FIELDS

    @pytest.mark.parametrize("cls", list(_EXPECTED), ids=lambda c: c.__name__)
    def test_every_consumed_7d_field_has_its_now_partner_in_the_set(
        self, cls: type[BaseStrategy]
    ) -> None:
        # QR-08 closure: a served 7d field drags its 'now' partner along (PitSignals invariant),
        # so no registry strategy may consume a 7d field without the partner
        for name in cls.consumes_mtf_signals:
            partner = _now_field_of(name) if name in FIELD_MAP else None
            assert partner is None or partner in cls.consumes_mtf_signals, (cls.__name__, name)

    def test_mtf_signal_fields_returns_the_class_value(self) -> None:
        assert RSIMeanReversionStrategy("r", {}).mtf_signal_fields() == _FOUR
        assert MACrossoverStrategy("m", {}).mtf_signal_fields() == frozenset()

    def test_every_declared_field_is_read_by_on_bar_and_nothing_else_is(self) -> None:
        """The declared sets match the ``mtf_context.<field>`` attribute reads in the source."""
        import trading.strategies as pkg

        names = ALL_MTF_SIGNAL_FIELDS
        for cls, declared in _EXPECTED.items():
            if cls is ModelStrategy:
                continue  # reads through the v2 feature builder (data.ml_features), checked below
            module = inspect.getmodule(cls)
            assert module is not None and module.__name__.startswith(pkg.__name__)
            tree = ast.parse(inspect.getsource(module))
            read = {
                n.attr
                for n in ast.walk(tree)
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "mtf_context"
                and n.attr in names
            }
            assert read == declared, (cls.__name__, read)
        import data.ml_features as feats

        tree = ast.parse(inspect.getsource(feats))
        fetched = {
            n.args[1].value
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
            and len(n.args) == 3
            and isinstance(n.args[0], ast.Name)
            and n.args[0].id == "mtf_context"
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value in names
        }
        assert fetched == _MODEL

    def test_the_ensemble_declares_the_union_of_its_subs(self) -> None:
        subs: list[BaseStrategy] = [MACrossoverStrategy("a", {}), RSIMeanReversionStrategy("b", {})]
        assert EnsembleStrategy("e", subs).mtf_signal_fields() == _FOUR
        none = [MACrossoverStrategy("a", {}), MomentumBreakoutStrategy("b", {})]
        assert EnsembleStrategy("e", none).mtf_signal_fields() == frozenset()
        mixed = [DCARSIHybridStrategy("d", {}), BreakoutStrategy("b", {})]
        assert EnsembleStrategy("e", mixed).mtf_signal_fields() == _FOUR
        assert EnsembleStrategy.consumes_mtf_signals == ALL_MTF_SIGNAL_FIELDS  # class: fail-closed

    async def test_the_runner_takes_the_union_over_its_strategies(self) -> None:
        runner = BacktestRunner(
            [MACrossoverStrategy("a", {}), RSIMeanReversionStrategy("b", {})],
            [SYM],
            TimeFrame.ONE_HOUR,
            seed=7,
            cost_profile=LEGACY_COST,
            mtf_signals=_source(_full(FIRST_DAY_MS)),
        )
        await runner.run({SYM: _bars()})
        report = runner.last_mtf_context
        assert report is not None and report.status == "pit"


# ---------------------------------------------------------------------------
# T-23: status truth table; a non-consumer never sees the store (G8)
# ---------------------------------------------------------------------------


def _decision_list(n: int = 30) -> list[datetime]:
    return [FIRST_OPEN + (k + 1) * H for k in range(n)]


class TestT23StatusTruthTable:
    def test_no_consumed_field_is_not_used_even_with_a_pit_source(self) -> None:
        pit = _source(_full(FIRST_DAY_MS))
        src, report = select_mtf_signals(pit, _decision_list(), frozenset())
        assert src is UNAVAILABLE
        assert (report.source, report.status, report.reason) == (
            "unavailable", "not_used", "no_consumed_fields",
        )  # fmt: skip
        assert report.series == () and report.series_digest is None

    def test_no_source_is_unavailable_for_a_consumer_and_not_used_for_a_non_consumer(self) -> None:
        for source in (None, UNAVAILABLE):
            src, report = select_mtf_signals(source, _decision_list(), _FOUR)
            assert src is UNAVAILABLE
            assert (report.source, report.status, report.reason) == (
                "unavailable", "unavailable", "no_signal_source",
            )  # fmt: skip
            assert set(dict(report.fields).values()) == {"no_source", "not_served"}
            src, report = select_mtf_signals(source, _decision_list(), frozenset())
            assert report.status == "not_used"

    def test_pit_partial_unavailable_over_the_consumed_fields_only(self) -> None:
        dec = _decision_list()
        series = _full(FIRST_DAY_MS)
        # every consumed field served -> pit (the 7d and not-served fields do not matter)
        _, rep = _source(series).select(dec, _FOUR)
        assert rep.status == "pit" and rep.reason is None
        # not_served fields among the consumed ones can never be pit -> partial
        _, rep = _source(series).select(dec, _FOUR | {"fed_funds_rate"})
        assert rep.status == "partial" and rep.reason == "partial_fields"
        # one consumed field missing -> partial
        no_whale = {k: v for k, v in series.items() if k != WHALE}
        _, rep = _source(no_whale).select(dec, _FOUR)
        assert rep.status == "partial"
        # an unconsumed field with no data does not matter
        _, rep = _source(no_whale).select(dec, frozenset({"fear_greed_index"}))
        assert rep.status == "pit"
        # no consumed field served -> unavailable
        _, rep = _source({}).select(dec, _FOUR)
        assert (rep.status, rep.source, rep.reason) == ("unavailable", "unavailable", "no_history")
        only_other = {WHALE: series[WHALE]}
        _, rep = _source(only_other).select(dec, frozenset({"fear_greed_index"}))
        assert rep.status == "unavailable"
        # a not-served field alone -> unavailable
        _, rep = _source(series).select(dec, frozenset({"fed_funds_rate"}))
        assert rep.status == "unavailable"

    def test_unavailable_because_of_coverage_has_the_coverage_reason(self) -> None:
        short = {FGI: _fgi_daily(to_ms(FIRST_OPEN) + 10 * D_MS, [30, 40, 50])}
        _, rep = _source(short).select(_decision_list(), frozenset({"fear_greed_index"}))
        assert (rep.status, rep.reason) == ("unavailable", "insufficient_coverage")
        assert dict(rep.fields)["fear_greed_index"] == "insufficient_coverage"

    def test_reason_ignores_unconsumed_fields(self) -> None:
        # QR-07: only a T10Y2Y series that starts mid-window; the ModelStrategy set never
        # consumes it, so its low coverage must not leak into the reason
        late = {T10: _t10_daily(to_ms(FIRST_OPEN) + 10 * D_MS, 20)}
        _, rep = _source(late).select(_decision_list(), _MODEL)
        assert dict(rep.fields)["yield_curve_spread"] == "insufficient_coverage"
        assert (rep.status, rep.reason) == ("unavailable", "no_history")
        # the mirror case: the rsi set consumes whale, whose only rows start too late
        low = {WHALE: _hourly(WHALE, to_ms(FIRST_OPEN) + 40 * H_MS, 10)}
        _, rep = _source(low).select(_decision_list(), _FOUR)
        assert (rep.status, rep.reason) == ("unavailable", "insufficient_coverage")

    @pytest.mark.parametrize(
        ("consumed", "served"),
        [
            (_FOUR, _FOUR),
            (_MODEL, _MODEL),
            (
                frozenset({"fear_greed_index_7d_ago"}),
                frozenset({"fear_greed_index_7d_ago", "fear_greed_index"}),
            ),
            (frozenset({"yield_curve_spread"}), frozenset({"yield_curve_spread"})),
        ],
        ids=["rsi", "model", "fgi_7d_only", "yield_only"],
    )
    def test_only_consumed_on_fields_are_served(
        self, consumed: frozenset[str], served: frozenset[str]
    ) -> None:
        # QR-08: the served set is consumed-and-ON plus the 'now' partner of a served 7d field
        src, rep = _source(_full(FIRST_DAY_MS)).select(_decision_list(), consumed)
        assert isinstance(src, PitSignals) and src.fields_on == served
        statuses = dict(rep.fields)
        assert {name for name in FIELD_MAP if statuses[name] == "pit"} == set(FIELD_MAP)
        assert len(FIELD_MAP) == 6 and rep.source == "pit" and rep.status == "pit"

    def test_source_agrees_with_status(self) -> None:
        # QR-08: ModelStrategy needs FGI and dominance; only T10 and whale have data, which the
        # run does not consume -> status unavailable, so the source must be UNAVAILABLE too
        series = _full(FIRST_DAY_MS)
        only_other = {T10: series[T10], WHALE: series[WHALE]}
        src, rep = _source(only_other).select(_decision_list(), _MODEL)
        assert (rep.status, rep.source, rep.reason) == ("unavailable", "unavailable", "no_history")
        assert src is UNAVAILABLE

    async def test_a_strategy_never_sees_a_series_it_does_not_consume(self) -> None:
        # QR-08 generalised G8: a yield-only consumer must not have its regime driven by FGI
        runs = []
        for series in (_full(FIRST_DAY_MS), {T10: _full(FIRST_DAY_MS)[T10]}):
            runner = BacktestRunner(
                [_YieldOnly("s1", {})],
                [SYM],
                TimeFrame.ONE_HOUR,
                seed=7,
                cost_profile=LEGACY_COST,
                mtf_signals=_source(series),
            )
            result = await runner.run({SYM: _bars()})
            trades = runner.last_portfolio.get_trade_history()  # type: ignore[union-attr]
            assert runner.last_mtf_context is not None
            assert runner.last_mtf_context.status == "pit"
            assert len(trades) >= 1, "the scenario must trade for the comparison to mean anything"
            assert all(t.regime_at_entry is None for t in trades)
            runs.append(
                (
                    [_scrub(t.model_dump()) for t in trades],
                    _scrub(result.model_dump()),
                )
            )
        assert runs[0] == runs[1]

    def test_the_returned_source_serves_only_the_on_fields(self) -> None:
        dec = _decision_list()
        series = _full(FIRST_DAY_MS)
        short = dict(series)
        short[WHALE] = _hourly(WHALE, to_ms(FIRST_OPEN) + 40 * H_MS, 10)
        src, rep = _source(short).select(dec, _FOUR)
        assert isinstance(src, PitSignals)
        assert "whale_net_flow" not in src.fields_on
        assert {"fear_greed_index", "btc_dominance", "yield_curve_spread"} <= src.fields_on
        assert src.values_at(dec[3]).whale_net_flow is None
        assert rep.source == "pit" and rep.status == "partial"

    def test_an_unknown_consumed_field_is_rejected(self) -> None:
        with pytest.raises(PitError) as err:
            _source(_full(FIRST_DAY_MS)).select(_decision_list(), {"not_a_field"})
        assert str(err.value) == "point-in-time check failed (pit_unknown_series)"

    def test_the_selection_is_pure_and_immutable(self) -> None:
        pit = _source(_full(FIRST_DAY_MS))
        dec = _decision_list()
        a_src, a_rep = pit.select(dec, _FOUR)
        _, b_rep = pit.select(iter(dec), _FOUR)
        assert a_rep == b_rep and a_rep is not b_rep
        assert a_src is not pit and pit.fields_on == ALL_FIELDS  # the offered source is untouched
        with pytest.raises(AttributeError):
            a_rep.status = "pit"  # type: ignore[misc]

    async def test_momentum_breakout_with_a_pit_source_is_not_used_and_trades_identically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def build(source: BacktestSignalSource | None, force: bool = False) -> BacktestRunner:
            return BacktestRunner(
                [MomentumBreakoutStrategy("m", {"lookback": 20, "trend_sma_period": None})],
                [SYM],
                TimeFrame.ONE_HOUR,
                seed=7,
                cost_profile=LEGACY_COST,
                bracket_config=BRACKET,
                mtf_signals=source,
            )

        engines: list[StrategyEngine] = []
        real = StrategyEngine

        def spy(**kw: Any) -> StrategyEngine:
            engine = real(**kw)
            engines.append(engine)
            return engine

        monkeypatch.setattr(backtest_mod, "StrategyEngine", spy)
        pit = _source(_full(FIRST_DAY_MS))
        base = build(None)
        base_result = await base.run({SYM: _bars()})
        with_pit = build(pit)
        pit_result = await with_pit.run({SYM: _bars()})

        assert with_pit.last_mtf_context is not None
        assert with_pit.last_mtf_context.status == "not_used"
        assert with_pit.last_mtf_context.source == "unavailable"
        assert base.last_mtf_context is not None and base.last_mtf_context.status == "not_used"
        assert engines[1]._mtf_signals is UNAVAILABLE  # receives UNAVAILABLE, not the PIT source
        trades_a = base.last_portfolio.get_trade_history()  # type: ignore[union-attr]
        trades_b = with_pit.last_portfolio.get_trade_history()  # type: ignore[union-attr]
        assert len(trades_a) >= 1, "the scenario must trade for the comparison to mean anything"
        dump_a = [_scrub(t.model_dump()) for t in trades_a]
        dump_b = [_scrub(t.model_dump()) for t in trades_b]
        assert dump_a == dump_b
        assert _scrub(base_result.model_dump()) == _scrub(pit_result.model_dump())
        assert all(t.regime_at_entry is None for t in trades_b)

        # control: had the PIT source reached this strategy, regime_at_entry WOULD differ
        def forced(**kw: Any) -> StrategyEngine:
            return real(mtf_signals=pit, **kw)

        monkeypatch.setattr(backtest_mod, "StrategyEngine", forced)
        control = build(None)
        await control.run({SYM: _bars()})
        trades_c = control.last_portfolio.get_trade_history()  # type: ignore[union-attr]
        assert any(t.regime_at_entry is not None for t in trades_c)

    async def test_no_source_is_byte_identical_to_the_pre_p2_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the engine constructor call carries no mtf_signals keyword at all without a source
        seen: list[dict[str, Any]] = []
        real = StrategyEngine

        def spy(**kw: Any) -> StrategyEngine:
            seen.append(kw)
            return real(**kw)

        monkeypatch.setattr(backtest_mod, "StrategyEngine", spy)
        runner, _ = _rsi_runner(None)
        await runner.run({SYM: _bars()})
        runner2, _ = _rsi_runner(UNAVAILABLE)
        await runner2.run({SYM: _bars()})
        assert all("mtf_signals" not in kw for kw in seen)
        runner3, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        await runner3.run({SYM: _bars()})
        assert isinstance(seen[-1]["mtf_signals"], PitSignals)


# ---------------------------------------------------------------------------
# T-25: determinism
# ---------------------------------------------------------------------------


class TestT25Determinism:
    async def test_same_seed_and_series_give_byte_identical_results_and_metadata(self) -> None:
        outs: list[tuple[str, str]] = []
        for _ in range(2):
            runner, _ = _rsi_runner(_source(_full(FIRST_DAY_MS)))
            result = await runner.run({SYM: _bars()})
            report = runner.last_mtf_context
            assert report is not None
            outs.append(
                (
                    json.dumps(_scrub(result.model_dump(mode="json")), sort_keys=True),
                    json.dumps(report.to_metadata(), sort_keys=True),
                )
            )
        assert outs[0] == outs[1]

    async def test_series_digest_is_recorded_and_tracks_the_data(self) -> None:
        digests = []
        for first_day in (FIRST_DAY_MS, FIRST_DAY_MS, FIRST_DAY_MS - D_MS):
            runner, _ = _rsi_runner(_source(_full(first_day)))
            await runner.run({SYM: _bars()})
            assert runner.last_mtf_context is not None
            digests.append(runner.last_mtf_context.series_digest)
        assert digests[0] == digests[1] and digests[0] != digests[2]
        assert digests[0] is not None and len(digests[0]) == 64

    async def test_the_digest_does_not_depend_on_the_selection(self) -> None:
        series = _full(FIRST_DAY_MS)
        _, full = _source(series).select(_decision_list(), _FOUR)
        _, narrow = _source(series, frozenset({"fear_greed_index"})).select(
            _decision_list(), _FOUR
        )
        assert full.series_digest == narrow.series_digest

    async def test_a_pit_run_differs_from_the_unavailable_run_for_a_consumer(self) -> None:
        a, rec_a = _rsi_runner(None)
        b, rec_b = _rsi_runner(_source(_full(FIRST_DAY_MS)))
        await a.run({SYM: _bars()})
        await b.run({SYM: _bars()})
        assert all(c is None for c in rec_a.contexts) or all(
            c is None or c.fear_greed_index is None for c in rec_a.contexts
        )
        assert any(c is not None and c.fear_greed_index is not None for c in rec_b.contexts)


# ---------------------------------------------------------------------------
# T-26: daily bars, FGI at the 24h ingestion lag (QD T-02 through the runner)
# ---------------------------------------------------------------------------


class TestT26DailyFgiThroughTheRunner:
    @pytest.mark.parametrize(
        ("cls", "sees_7d"), [(RSIMeanReversionStrategy, False), (_Fgi7d, True)], ids=["rsi", "7d"]
    )
    async def test_fgi_of_day_d_is_seen_at_the_close_of_bar_d(
        self, cls: type[RSIMeanReversionStrategy], sees_7d: bool
    ) -> None:
        first_open = datetime(2026, 1, 1, tzinfo=UTC)
        n = 120
        day0 = to_ms(first_open) - 60 * D_MS
        values = [(k * 13) % 97 for k in range(200)]
        series = {FGI: _fgi_daily(day0, values)}
        strategy = cls("s1", {})
        rec = _record(strategy)
        runner = BacktestRunner(
            [strategy],
            [SYM],
            TimeFrame.ONE_DAY,
            seed=7,
            cost_profile=LEGACY_COST,
            mtf_signals=_source(series, frozenset({"fear_greed_index", "fear_greed_index_7d_ago"})),
        )
        bars = scripted_path(
            _closes(n), first_open=first_open, symbol=SYM, timeframe=TimeFrame.ONE_DAY
        )
        await runner.run({SYM: bars})
        report = runner.last_mtf_context
        assert report is not None
        assert report.to_metadata()["fields"]["fear_greed_index"] == "pit"
        w = runner._warmup_bars
        assert len(rec.contexts) == len(bars) - w
        for offset, ctx in enumerate(rec.contexts):
            bar = bars[w + offset]
            assert ctx is not None
            # the bar OPENS at 00:00 of day D and CLOSES at 00:00 of D+1, exactly when FGI(D)
            # (observed D 00:00 + 24h) becomes available: FGI(D) is visible at its close.
            day_index = (to_ms(bar.timestamp) - day0) // D_MS
            assert ctx.fear_greed_index == values[day_index]
            # QR-08: the 7d field is served only to a strategy that consumes it
            assert ctx.fear_greed_index_7d_ago == (values[day_index - 7] if sees_7d else None)

    async def test_one_ms_earlier_the_value_is_not_yet_visible(self) -> None:
        day0 = to_ms(datetime(2026, 1, 1, tzinfo=UTC))
        pit = _source({FGI: _fgi_daily(day0, [10, 20, 30])}, frozenset({"fear_greed_index"}))
        close_of_day_1 = datetime(2026, 1, 3, tzinfo=UTC)  # day 1 (Jan 2) closes at Jan 3 00:00
        assert pit.values_at(close_of_day_1).fear_greed_index == 20
        assert pit.values_at(close_of_day_1 - timedelta(milliseconds=1)).fear_greed_index == 10
