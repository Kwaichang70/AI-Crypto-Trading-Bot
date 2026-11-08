"""WP2.4 / P2: the optimizer (in-sample and walk-forward) selects ONE MTF source per run.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-05, SY-24-07, SY-24-17, WP24-T-24,
T-25) and the P1 carry-forward CF-P1-01.

* T-24  one selection and one report per ``run``; inner runners skip coverage; the SAME
        immutable source reaches every combination and every walk-forward fold
* T-25  determinism: same seed and series -> identical entries and metadata
* No source passed -> the optimizer result equals the pre-P2 one.

Coverage for the optimizer is computed once over the whole aligned window and EVERY timeline
step counts as a decision (the warm-up depends on each combination's parameters): a superset of
the trading decisions, PIT-safe but NOT monotone, documented in ``ParameterOptimizer``
(QR-04: ``TestOptimizerVsRunnerCoverage`` pins the counterexample).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from data.bar_alignment import BarAlignmentError
from data.point_in_time import FIELD_MAP, Observation, PitError, PitSeries, to_ms
from tests.integration.fakes.bar_paths import scripted_path
from trading import optimizer as optimizer_mod
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.mtf_signals import (
    LIVE_SIGNALS,
    UNAVAILABLE,
    BacktestSignalSource,
    MtfContextReport,
    PitSignals,
)
from trading.optimizer import OptimizationResult, ParameterOptimizer
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy

SYM = "BTC/USDT"
TF = TimeFrame.ONE_HOUR
H = timedelta(hours=1)
H_MS = 3_600_000
D_MS = 86_400_000
FIRST_OPEN = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
FGI = "alternative_me:fgi"
DOM = "coingecko:btc_dominance"
T10 = "fred:T10Y2Y"
WHALE = "whale:net_flow_1h"
FIRST_DAY_MS = to_ms(FIRST_OPEN) - 30 * D_MS
N_BARS = 700
GRID = {"rsi_period": [8, 14], "oversold": [25.0, 30.0]}


def _closes(n: int) -> list[float]:
    return [
        round(100 + 14 * ((i // 25) % 2 * 2 - 1) * ((i % 25) / 25) + i * 0.02, 2) for i in range(n)
    ]


def _bars(n: int = N_BARS) -> dict[str, list[OHLCVBar]]:
    return {SYM: scripted_path(_closes(n), first_open=FIRST_OPEN, symbol=SYM, timeframe=TF)}


def _pit(first_day_ms: int = FIRST_DAY_MS, days: int = 60) -> PitSignals:
    fgi = PitSeries(
        FGI,
        tuple(
            Observation(
                first_day_ms + k * D_MS,
                first_day_ms + (k + 1) * D_MS,
                Decimal(15 + (k * 11) % 70),
                "estimated_lag",
            )
            for k in range(days)
        ),
    )
    dom = PitSeries(
        DOM,
        tuple(
            Observation(
                first_day_ms + k * H_MS,
                first_day_ms + k * H_MS + 60_000,
                Decimal(48 + k % 5),
                "recorded",
            )
            for k in range(days * 24)
        ),
    )
    return PitSignals({FGI: fgi, DOM: dom}, frozenset(FIELD_MAP))


class _Spy:
    """Records every inner ``BacktestRunner`` the optimizer builds, and every ``select`` call."""

    def __init__(self) -> None:
        self.runners: list[BacktestRunner] = []
        self.run_kwargs: list[dict[str, Any]] = []
        self.selects = 0


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    state = _Spy()

    class SpyRunner(BacktestRunner):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            state.runners.append(self)

        async def run(self, bars_by_symbol: Any, *args: Any, **kwargs: Any) -> Any:
            state.run_kwargs.append(dict(kwargs))
            return await super().run(bars_by_symbol, *args, **kwargs)

    monkeypatch.setattr(optimizer_mod, "BacktestRunner", SpyRunner)
    original = PitSignals.select

    def counted(self: PitSignals, decision_times: Any, fields: Any) -> Any:
        state.selects += 1
        return original(self, decision_times, fields)

    monkeypatch.setattr(PitSignals, "select", counted)
    return state


def _opt(
    source: BacktestSignalSource | None,
    *,
    strategy_cls: Any = RSIMeanReversionStrategy,
    walk_forward: bool = False,
    grid: dict[str, list[Any]] | None = None,
) -> ParameterOptimizer:
    return ParameterOptimizer(
        strategy_cls=strategy_cls,
        symbols=[SYM],
        timeframe=TF,
        param_grid=grid or GRID,
        rank_by="total_return_pct",
        walk_forward=walk_forward,
        walk_forward_folds=2,
        walk_forward_train_fraction=0.5,
        mtf_signals=source,
    )


def _entries(result: OptimizationResult) -> str:
    return json.dumps(
        [
            {"params": e.params, "metrics": e.metrics}
            for e in result.entries
        ],
        sort_keys=True,
        default=str,
    )


class TestT24OneSelectionPerRun:
    async def test_in_sample_selects_once_and_every_combination_shares_the_source(
        self, spy: _Spy
    ) -> None:
        opt = _opt(_pit())
        assert opt.last_mtf_context is None
        result = await opt.run(_bars())
        assert result.completed_combinations == 4 and result.failed_combinations == 0
        assert spy.selects == 1  # ONE selection for four combinations
        report = opt.last_mtf_context
        assert isinstance(report, MtfContextReport)
        # the fixture has FGI and dominance only: yield curve and whale are not served
        assert report.status == "partial" and report.source == "pit"
        fields = report.to_metadata()["fields"]
        assert fields["fear_greed_index"] == "pit" and fields["btc_dominance"] == "pit"
        assert fields["yield_curve_spread"] == "no_source" == fields["whale_net_flow"]
        assert len(spy.runners) == 4
        sources = {id(r._mtf_signals) for r in spy.runners}
        assert len(sources) == 1  # the same immutable object in every combination
        shared = spy.runners[0]._mtf_signals
        assert isinstance(shared, PitSignals) and shared is not opt._mtf_signals
        assert shared.fields_on == opt._mtf_selected.fields_on  # type: ignore[union-attr]
        # inner runners skip coverage: no selection, no report of their own
        assert all(kw.get("select_mtf_context") is False for kw in spy.run_kwargs)
        assert all(r.last_mtf_context is None for r in spy.runners)

    async def test_walk_forward_selects_once_for_the_in_sample_pass_and_every_fold(
        self, spy: _Spy
    ) -> None:
        opt = _opt(_pit(), walk_forward=True, grid={"rsi_period": [8, 14]})
        result = await opt.run(_bars())
        assert result.walk_forward is True and result.completed_combinations == 2
        assert spy.selects == 1
        # 2 combinations x (1 in-sample + 2 folds) inner runners, one shared source
        assert len(spy.runners) == 2 * (1 + 2)
        assert len({id(r._mtf_signals) for r in spy.runners}) == 1
        assert all(kw.get("select_mtf_context") is False for kw in spy.run_kwargs)
        assert opt.last_mtf_context is not None

    async def test_one_report_per_run_and_it_is_reset_at_the_next_run(self, spy: _Spy) -> None:
        opt = _opt(_pit())
        await opt.run(_bars())
        first = opt.last_mtf_context
        assert first is not None
        window = (FIRST_OPEN - timedelta(days=400), FIRST_OPEN + timedelta(days=400))
        with pytest.raises(BarAlignmentError):  # a run-level rejection, before the selection
            await opt.run(_bars(), requested_window=window)
        assert opt.last_mtf_context is None and opt._mtf_selected is None
        await opt.run(_bars())
        second = opt.last_mtf_context
        assert second is not None and second is not first and second == first
        assert spy.selects == 2

    async def test_the_selected_source_is_not_mutated_and_the_offered_one_is_untouched(
        self, spy: _Spy
    ) -> None:
        offered = _pit()
        before = (offered.fields_on, dict(offered.series))
        opt = _opt(offered)
        await opt.run(_bars())
        assert (offered.fields_on, dict(offered.series)) == before

    async def test_the_decisions_are_every_step_of_the_aligned_window(self, spy: _Spy) -> None:
        # FGI/dominance start 25 bars into the window: a runner (warm-up W >= 50) would see
        # >= 99 % coverage, the optimizer counts the warm-up region too -> OFF (documented)
        late = to_ms(FIRST_OPEN) - 50 * H_MS + 25 * H_MS
        late_day = late - late % D_MS
        opt = _opt(_pit(first_day_ms=late_day + D_MS, days=40))
        await opt.run(_bars())
        report = opt.last_mtf_context
        assert report is not None
        assert report.to_metadata()["fields"]["fear_greed_index"] == "insufficient_coverage"
        assert report.status == "unavailable"
        assert isinstance(opt._mtf_selected, (PitSignals, type(UNAVAILABLE)))

    async def test_the_decisions_are_the_bar_closes_of_every_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CR-02: warm-up 0, one decision per timeline step, at ``open + one timeframe``
        seen: list[list[datetime]] = []
        real = optimizer_mod.select_mtf_signals

        def select_spy(source: Any, decisions: Any, fields: Any) -> Any:
            items = list(decisions)
            seen.append(items)
            return real(source, items, fields)

        monkeypatch.setattr(optimizer_mod, "select_mtf_signals", select_spy)
        bars = _bars()
        await _opt(_pit()).run(bars)
        assert seen == [[b.timestamp + H for b in bars[SYM]]]

    async def test_a_non_consumer_class_is_not_used_and_gets_unavailable(self, spy: _Spy) -> None:
        opt = _opt(_pit(), strategy_cls=MACrossoverStrategy, grid={"fast_period": [5, 8]})
        base = _opt(None, strategy_cls=MACrossoverStrategy, grid={"fast_period": [5, 8]})
        res = await opt.run(_bars())
        ref = await base.run(_bars())
        assert opt.last_mtf_context is not None and opt.last_mtf_context.status == "not_used"
        assert all(r._mtf_signals is UNAVAILABLE for r in spy.runners)
        assert _entries(res) == _entries(ref)
        assert spy.selects == 1  # still ONE selection per run (here the quick not_used path)


class TestNoSourceEqualsHead:
    async def test_without_a_source_the_result_is_the_same_as_with_unavailable(
        self, spy: _Spy
    ) -> None:
        a = _opt(None)
        b = _opt(UNAVAILABLE)
        res_a = await a.run(_bars())
        res_b = await b.run(_bars())
        assert _entries(res_a) == _entries(res_b)
        assert a.last_mtf_context is not None and a.last_mtf_context.status == "unavailable"
        assert a.last_mtf_context.source == "unavailable"
        assert all(r._mtf_signals is UNAVAILABLE or r._mtf_signals is None for r in spy.runners)

    async def test_a_consumer_with_a_pit_source_differs_from_no_source(self) -> None:
        with_pit = await _opt(_pit()).run(_bars())
        without = await _opt(None).run(_bars())
        assert _entries(with_pit) != _entries(without)  # the effect is real, not a no-op


class TestFailuresAreRunLevelNotCombinationLevel:
    def test_a_bad_source_fails_at_construction_with_the_type_name_only(self) -> None:
        for bad in (LIVE_SIGNALS, object(), "pit"):
            with pytest.raises(ValueError, match="must be a BacktestSignalSource") as err:
                _opt(bad)  # type: ignore[arg-type]
            assert str(err.value).endswith(f"got {type(bad).__name__}")

    async def test_a_naive_timestamp_fails_the_run_before_any_combination(
        self, spy: _Spy
    ) -> None:
        naive = {
            SYM: [
                b.model_copy(update={"timestamp": b.timestamp.replace(tzinfo=None)})
                for b in _bars()[SYM]
            ]
        }
        opt = _opt(_pit())
        with pytest.raises(PitError) as err:  # NOT counted into failed_combinations
            await opt.run(naive)
        assert str(err.value) == "point-in-time check failed (pit_naive_time)"
        assert spy.runners == [] and opt.last_mtf_context is None

    async def test_a_source_replaced_by_a_foreign_object_fails_the_run(self, spy: _Spy) -> None:
        opt = _opt(None)
        opt._mtf_signals = object()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="object"):
            await opt.run(_bars())
        assert spy.runners == []


class TestT25Determinism:
    async def test_same_seed_and_series_give_identical_entries_and_metadata(self) -> None:
        outs = []
        for _ in range(2):
            opt = _opt(_pit())
            result = await opt.run(_bars())
            assert opt.last_mtf_context is not None
            outs.append((_entries(result), json.dumps(opt.last_mtf_context.to_metadata())))
        assert outs[0] == outs[1]
        assert json.loads(outs[0][1])["series_digest"] is not None


def _whale_counterexample() -> tuple[list[OHLCVBar], PitSignals]:
    """QR-04: 4 consecutive whale snapshots missing deep inside every trading window."""
    closes = [
        round(100 + 14 * ((i // 25) % 2 * 2 - 1) * ((i % 25) / 25) + i * 0.05, 2)
        for i in range(400)
    ]
    bars = scripted_path(closes, first_open=FIRST_OPEN, symbol=SYM, timeframe=TF)
    assert len(bars) == 450
    f0 = to_ms(bars[0].timestamp) - 20 * D_MS
    hours = 20 * 24 + 470
    hole_at = (to_ms(bars[0].timestamp) - f0) // H_MS + 300

    def hourly(series_id: str, base: int, skip: range = range(0)) -> PitSeries:
        return PitSeries(
            series_id,
            tuple(
                Observation(
                    f0 + k * H_MS, f0 + k * H_MS + 60_000, Decimal(base + k % 7), "recorded"
                )
                for k in range(hours)
                if k not in skip
            ),
        )

    series = {
        FGI: PitSeries(
            FGI,
            tuple(
                Observation(
                    f0 + k * D_MS,
                    f0 + k * D_MS + D_MS,
                    Decimal(10 + (k * 37) % 81),
                    "estimated_lag",
                )
                for k in range(40)
            ),
        ),
        T10: PitSeries(
            T10,
            tuple(
                Observation(
                    f0 + k * D_MS,
                    f0 + k * D_MS + 3 * D_MS,
                    Decimal("0.25") + Decimal(k % 9) / 20,
                    "estimated_lag",
                )
                for k in range(40)
            ),
        ),
        DOM: hourly(DOM, 50),
        WHALE: hourly(WHALE, 1000, range(hole_at, hole_at + 4)),
    }
    return bars, PitSignals(series, frozenset(FIELD_MAP))


class TestOptimizerVsRunnerCoverage:
    """QR-04: the optimizer's run-level ON/OFF can differ from a single runner's, either way."""

    async def test_the_optimizer_says_pit_where_the_long_warmup_runner_says_insufficient(
        self,
    ) -> None:
        bars, source = _whale_counterexample()
        cost = get_cost_profile("legacy_backtest_v0")
        opt = ParameterOptimizer(
            RSIMeanReversionStrategy, [SYM], TF, {"rsi_period": [7, 30]},
            cost_profile=cost, mtf_signals=source,
        )  # fmt: skip
        await opt.run({SYM: bars})
        report = opt.last_mtf_context
        assert report is not None
        meta = report.to_metadata()
        assert (meta["status"], meta["fields"]["whale_net_flow"]) == ("pit", "pit")
        assert meta["series"][WHALE]["coverage_ppm"] == 993_333

        expected = {
            7: (50, "pit", "pit", 992_500),
            30: (184, "partial", "insufficient_coverage", 988_721),
        }
        for period, (warmup, status, whale, ppm) in expected.items():
            runner = BacktestRunner(
                [RSIMeanReversionStrategy("s", {"rsi_period": period})], [SYM], TF,
                seed=42, cost_profile=cost, mtf_signals=source,
            )  # fmt: skip
            await runner.run({SYM: bars})
            assert runner._warmup_bars == warmup
            single = runner.last_mtf_context
            assert single is not None
            row = single.to_metadata()
            assert (row["status"], row["fields"]["whale_net_flow"]) == (status, whale)
            assert row["series"][WHALE]["coverage_ppm"] == ppm
