"""WP2.3 / P3: optimizer alignment-once and the time-based walk-forward split.

Binding spec: ``reports/vp2-wp2.3/synthesis-spec.md`` (SY-23-08, SY-23-17, SY-23-18, WP23-T-18,
T-19) plus ``p1-quant-r1.md`` QA-07 (fold windows are timeline ``[start, end)``).  Test ids are
``WP23-P3-NN``.
"""

# ruff: noqa: S311  (seeded stdlib generators build deterministic fixtures, not secrets)

from __future__ import annotations

import ast
import math
import random
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pytest
import structlog

from common.metrics import metrics
from common.models import OHLCVBar
from common.types import TimeFrame
from data import bar_alignment
from data.bar_alignment import BarAlignmentError
from trading.backtest import BacktestRunner
from trading.optimizer import ParameterOptimizer
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.walk_forward import WalkForwardValidator

A, B = "AAA/EUR", "BBB/EUR"
H = timedelta(hours=1)
T0 = datetime(2025, 1, 1, tzinfo=UTC)
Mode = Literal["expanding", "rolling"]
GRID = {"fast_period": [5, 8], "slow_period": [20]}


def _series(sym: str, n: int, seed: int) -> list[OHLCVBar]:
    rng = random.Random(seed)
    px = 100.0
    out: list[OHLCVBar] = []
    for k in range(n):
        o = px
        c = max(1.0, o * math.exp(rng.gauss(0.0001, 0.006)))
        px = c
        out.append(
            OHLCVBar(
                symbol=sym,
                timeframe=TimeFrame.ONE_HOUR,
                timestamp=T0 + k * H,
                open=Decimal(f"{o:.4f}"),
                high=Decimal(f"{max(o, c) * 1.001:.4f}"),
                low=Decimal(f"{min(o, c) * 0.999:.4f}"),
                close=Decimal(f"{c:.4f}"),
                volume=Decimal(10),
            )
        )
    return out


def _bars(n: int, missing_b: Collection[int] = ()) -> dict[str, list[OHLCVBar]]:
    return {
        A: _series(A, n, 1),
        B: [b for k, b in enumerate(_series(B, n, 2)) if k not in missing_b],
    }


def _optimizer(**kw: Any) -> ParameterOptimizer:
    return ParameterOptimizer(
        strategy_cls=MACrossoverStrategy,
        symbols=kw.pop("symbols", [A, B]),
        timeframe=TimeFrame.ONE_HOUR,
        param_grid=GRID,
        **kw,
    )


@pytest.fixture(autouse=True)
def _fresh_metrics() -> Any:
    metrics.reset()
    yield
    metrics.reset()


def _counters() -> dict[str, int]:
    return {k: v for k, v in metrics.get_all()["counters"].items() if "bar_alignment" in k}


@pytest.fixture
def run_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keyword arguments of every inner ``BacktestRunner.run`` call."""
    calls: list[dict[str, Any]] = []
    orig = BacktestRunner.run

    async def spy(self: BacktestRunner, bars: Any, *a: Any, **k: Any) -> Any:
        calls.append(k)
        return await orig(self, bars, *a, **k)

    monkeypatch.setattr(BacktestRunner, "run", spy)
    return calls


# ---------------------------------------------------------------------------
# WP23-P3-70..  optimizer: align and check ONCE (SY-23-18, T-19)
# ---------------------------------------------------------------------------


class TestOptimizerChecksOnce:
    async def test_p3_70_whole_run_rejection_is_before_any_combination(
        self, run_spy: list[dict[str, Any]]
    ) -> None:
        bars = _bars(600, {100, 101, 102, 103, 104})  # max gap 5 > 4 (0.83% missing passes)
        opt = _optimizer()
        with structlog.testing.capture_logs() as logs, pytest.raises(BarAlignmentError) as ei:
            await opt.run(bars)
        assert ei.value.code == "bar_alignment_gap_too_long"
        assert run_spy == []  # no combination ran
        assert not [e for e in logs if "combination_failed" in str(e["event"])]
        assert _counters() == {
            'bar_alignment_runs_total{outcome="rejected"}': 1,
            'bar_alignment_rejected_total{reason="bar_alignment_gap_too_long"}': 1,
            'bar_alignment_missing_bars_total{symbol="BBB/EUR",timeframe="1h"}': 5,
        }
        assert opt.last_alignment is not None

    async def test_p3_71_in_sample_records_once_and_inner_runners_do_not(
        self, run_spy: list[dict[str, Any]]
    ) -> None:
        opt = _optimizer()
        with structlog.testing.capture_logs() as logs:
            res = await opt.run(_bars(300, {100, 101}))
        assert res.completed_combinations == 2 and res.failed_combinations == 0
        assert [c.get("record_alignment") for c in run_spy] == [False, False]
        assert _counters() == {
            'bar_alignment_runs_total{outcome="ok_with_gaps"}': 1,
            'bar_alignment_missing_bars_total{symbol="BBB/EUR",timeframe="1h"}': 2,
        }
        assert sum(1 for e in logs if e["event"] == "backtest.bar_alignment") == 1
        assert sum(1 for e in logs if e["event"] == "backtest.bar_alignment_gaps") == 1

    async def test_p3_72_a_data_error_inside_a_combination_propagates_in_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: BacktestRunner, *_a: Any, **_k: Any) -> Any:
            raise BarAlignmentError("bar_alignment_not_trimmed")

        monkeypatch.setattr(BacktestRunner, "run", boom)
        with structlog.testing.capture_logs() as logs, pytest.raises(BarAlignmentError) as ei:
            await _optimizer().run(_bars(300))
        assert ei.value.code == "bar_alignment_not_trimmed"
        assert not [e for e in logs if "combination_failed" in str(e["event"])]

    async def test_p3_73_a_data_error_inside_a_combination_propagates_in_walk_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: BacktestRunner, *_a: Any, **_k: Any) -> Any:
            raise BarAlignmentError("bar_alignment_not_trimmed")

        monkeypatch.setattr(BacktestRunner, "run", boom)
        opt = _optimizer(walk_forward=True, walk_forward_folds=2, walk_forward_train_fraction=0.5)
        with pytest.raises(BarAlignmentError):
            await opt.run(_bars(800))

    @pytest.mark.parametrize("walk_forward", [False, True])
    async def test_p3_74_control_other_errors_still_count_as_failed_combinations(
        self, monkeypatch: pytest.MonkeyPatch, walk_forward: bool
    ) -> None:
        async def boom(self: BacktestRunner, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("combination blew up")

        monkeypatch.setattr(BacktestRunner, "run", boom)
        opt = _optimizer(
            walk_forward=walk_forward, walk_forward_folds=2, walk_forward_train_fraction=0.5
        )
        res = await opt.run(_bars(800))
        assert res.failed_combinations == 2 and res.completed_combinations == 0

    async def test_p3_75_single_symbol_is_report_only_without_a_window(self) -> None:
        bars = {A: _bars(600)[A]}
        bars[A] = [b for k, b in enumerate(bars[A]) if k not in {100, 101, 102, 103, 104}]
        opt = _optimizer(symbols=[A])
        res = await opt.run(bars)
        assert res.failed_combinations == 0
        assert _counters()['bar_alignment_runs_total{outcome="ok_with_gaps"}'] == 1

    async def test_p3_76_single_symbol_with_a_requested_window_is_enforced(self) -> None:
        bars = {A: [b for k, b in enumerate(_bars(600)[A]) if k not in {100, 101, 102, 103, 104}]}
        opt = _optimizer(symbols=[A])
        with pytest.raises(BarAlignmentError):
            await opt.run(bars, requested_window=(T0, T0 + 600 * H))

    async def test_p3_77_requested_window_trims_the_forming_bar(
        self, run_spy: list[dict[str, Any]]
    ) -> None:
        bars = {A: _bars(300)[A]}
        opt = _optimizer(symbols=[A])
        res = await opt.run(bars, requested_window=(T0, T0 + 299 * H))
        assert res.failed_combinations == 0
        rep = opt.last_alignment
        assert rep is not None and rep.per_symbol[0].trimmed_outside == 1


class TestOptimizerFoldCheck:
    def _wf(self) -> ParameterOptimizer:
        return _optimizer(walk_forward=True, walk_forward_folds=2, walk_forward_train_fraction=0.5)

    async def test_p3_80_a_bad_test_fold_is_rejected_before_any_combination(
        self, run_spy: list[dict[str, Any]]
    ) -> None:
        # whole run: 6/800 = 0.75% passes; fold 0 = [400, 600) allows max(1, 4) = 4
        bars = _bars(800, {410, 430, 450, 470, 490, 510})
        with pytest.raises(BarAlignmentError) as ei:
            await self._wf().run(bars)
        assert ei.value.code == "bar_alignment_fold_tolerance"
        assert [(v.symbol, v.fold, v.observed, v.limit) for v in ei.value.violations] == [
            (B, 0, 6, 4)
        ]
        assert run_spy == []
        assert _counters() == {
            'bar_alignment_runs_total{outcome="rejected"}': 1,
            'bar_alignment_rejected_total{reason="bar_alignment_fold_tolerance"}': 1,
            'bar_alignment_missing_bars_total{symbol="BBB/EUR",timeframe="1h"}': 6,
        }

    async def test_p3_81_control_a_tolerable_fold_runs_every_combination_once(
        self, run_spy: list[dict[str, Any]]
    ) -> None:
        bars = _bars(800, {410, 450, 490})  # 3 <= 4 in fold 0
        res = await self._wf().run(bars)
        assert res.completed_combinations == 2 and res.failed_combinations == 0
        assert len(run_spy) == 2 * (1 + 2)  # combinations x (in-sample + 2 test folds)
        assert all(c.get("record_alignment") is False for c in run_spy)
        assert _counters()['bar_alignment_runs_total{outcome="ok_with_gaps"}'] == 1

    @pytest.mark.parametrize(
        ("missing", "expected"),
        [
            (frozenset(), [(400, 600), (600, 800)]),
            # B misses the first test row: the fold window snaps FORWARD to the next full row
            (frozenset({400}), [(401, 600), (600, 800)]),
            # ... and the last row of fold 0: the window ends on the last full row before it
            (frozenset({599}), [(400, 599), (600, 800)]),
        ],
    )
    async def test_p3_82_fold_windows_are_timeline_start_end_exclusive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        missing: frozenset[int],
        expected: list[tuple[int, int]],
    ) -> None:
        seen: list[Any] = []
        real = bar_alignment.fold_check

        def spy(bars: Any, symbols: Any, tf_s: Any, folds: Any, **kw: Any) -> None:
            seen.append(list(folds))
            real(bars, symbols, tf_s, folds, **kw)

        monkeypatch.setattr("trading.optimizer.fold_check", spy)
        await self._wf().run(_bars(800, set(missing)))
        assert len(seen) == 1  # once per run, with every fold
        assert seen[0] == [(T0 + a * H, T0 + b * H) for a, b in expected]


# ---------------------------------------------------------------------------
# WP23-P3-90..  walk-forward split (SY-23-17, T-18)
# ---------------------------------------------------------------------------


def _legacy_split(
    n: int, num_folds: int, frac: float, mode: Mode
) -> list[tuple[int, int, int, int]] | None:
    """HEAD's index arithmetic: (train_start, train_end, test_start, test_end) per fold."""
    if n < num_folds * 2:
        return None
    initial = int(n * frac)
    remaining = n - initial
    if remaining < num_folds or remaining // num_folds == 0:
        return None
    size = remaining // num_folds
    out = []
    for i in range(num_folds):
        ts = initial + i * size
        te = n if i == num_folds - 1 else ts + size
        out.append((0 if mode == "expanding" else ts - initial, ts, ts, te))
    return out


class TestWalkForwardTimeSplit:
    @pytest.mark.parametrize(
        ("n", "folds", "frac", "mode"),
        [
            (n, f, fr, m)
            for n, f, fr, m in product(
                (20, 57, 100, 333), (2, 3, 4, 5), (0.5, 0.7, 0.8), ("expanding", "rolling")
            )
        ],
    )
    def test_p3_90_single_symbol_slices_equal_the_old_index_slices(
        self, n: int, folds: int, frac: float, mode: Mode
    ) -> None:
        bars = _series(A, n, 3)
        expected = _legacy_split(n, folds, frac, mode)
        v = WalkForwardValidator(
            num_folds=folds,
            train_fraction=frac,
            mode=mode,
        )
        if expected is None:
            with pytest.raises(ValueError):
                v.split({A: bars})
            return
        got = v.split({A: bars})
        assert len(got) == len(expected)
        for fold, (a, b, c, d) in zip(got, expected, strict=True):
            assert fold.train_bars[A] == bars[a:b]
            assert fold.test_bars[A] == bars[c:d]

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_p3_91_symbols_sharing_one_timeline_equal_the_index_slices(self, mode: Mode) -> None:
        data = {A: _series(A, 100, 4), B: _series(B, 100, 5)}
        got = WalkForwardValidator(mode=mode).split(data)
        for fold, (a, b, c, d) in zip(got, _legacy_split(100, 4, 0.7, mode) or [], strict=True):
            for s in (A, B):
                assert fold.train_bars[s] == data[s][a:b]
                assert fold.test_bars[s] == data[s][c:d]

    def test_p3_92_unequal_counts_give_identical_windows_equal_to_the_gap_free_ones(self) -> None:
        """T-18: gaps away from the fold edges do not move the boundaries."""
        free = WalkForwardValidator().split(_bars(400))
        gapped = WalkForwardValidator().split(_bars(400, {50, 120, 331}))
        assert len(_bars(400, {50, 120, 331})[B]) == 397
        for f_free, f_gap in zip(free, gapped, strict=True):
            for part_free, part_gap in (
                (f_free.train_bars, f_gap.train_bars),
                (f_free.test_bars, f_gap.test_bars),
            ):
                assert (
                    part_gap[A][0].timestamp
                    == part_gap[B][0].timestamp
                    == part_free[A][0].timestamp
                )
                assert (
                    part_gap[A][-1].timestamp
                    == part_gap[B][-1].timestamp
                    == part_free[A][-1].timestamp
                )

    def test_p3_93_expanding_edges_snap_to_full_rows(self) -> None:
        """B misses the test_start row (280) and the row before it (279)."""
        data = _bars(400, {279, 280})
        fold0 = WalkForwardValidator().split(data)[0]
        # train ends on the last FULL row before the boundary, test starts on the first one after
        assert (
            fold0.train_bars[A][-1].timestamp == T0 + 278 * H == fold0.train_bars[B][-1].timestamp
        )
        assert fold0.test_bars[A][0].timestamp == T0 + 281 * H == fold0.test_bars[B][0].timestamp

    def test_p3_94_rolling_train_start_snaps_to_a_full_row(self) -> None:
        """Rolling fold 1: train_start ordinal = 310 - 280 = 30; B misses rows 30 and 31."""
        data = _bars(400, {30, 31})
        folds = WalkForwardValidator(mode="rolling").split(data)
        assert (
            folds[1].train_bars[A][0].timestamp
            == T0 + 32 * H
            == folds[1].train_bars[B][0].timestamp
        )
        # fold 0's train_start (ordinal 0) is untouched: row 0 is full
        assert folds[0].train_bars[A][0].timestamp == T0

    def test_p3_95_every_slice_starts_and_ends_on_a_full_row(self) -> None:
        rng = random.Random(5)
        for trial in range(25):
            miss_b = set(rng.sample(range(1, 399), 8))
            data = _bars(400, miss_b)
            common = {b.timestamp for b in data[A]} & {b.timestamp for b in data[B]}
            modes: tuple[Mode, ...] = ("expanding", "rolling")
            for mode in modes:
                for fold in WalkForwardValidator(mode=mode).split(data):
                    for part in (fold.train_bars, fold.test_bars):
                        assert part[A][0].timestamp == part[B][0].timestamp
                        assert part[A][-1].timestamp == part[B][-1].timestamp
                        assert part[A][0].timestamp in common, (trial, mode)
                        assert part[A][-1].timestamp in common, (trial, mode)

    def test_p3_96_no_bare_assert_remains_in_the_module(self) -> None:
        src = Path(WalkForwardValidator.__module__.replace(".", "/") + ".py")
        path = next(
            p
            for p in (Path("packages") / src, Path(__file__).parents[2] / "packages" / src)
            if p.exists()
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert [n for n in ast.walk(tree) if isinstance(n, ast.Assert)] == []

    def test_p3_97_a_degenerate_fold_raises_an_explicit_value_error(self) -> None:
        """Rows common to both symbols only exist early on, so the test windows hold none."""
        data = {A: _series(A, 100, 6), B: _series(B, 100, 7)[:40]}
        with pytest.raises(ValueError, match="holds no full row"):
            WalkForwardValidator().split(data)

    def test_p3_98_too_few_bars_still_uses_the_union_count(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            WalkForwardValidator(num_folds=10).split({A: _series(A, 15, 8), B: _series(B, 12, 9)})


class TestOptimizerOutcomeAndStaleness:
    async def test_p3_100_a_split_value_error_still_records_one_outcome(self) -> None:
        """CR-03: the whole-run check passed, the fold split then failed (too few bars)."""
        opt = _optimizer(walk_forward=True, walk_forward_folds=10)
        with pytest.raises(ValueError, match="Need at least") as ei:
            await opt.run(_bars(15))
        assert not isinstance(ei.value, BarAlignmentError)
        assert _counters() == {'bar_alignment_runs_total{outcome="ok"}': 1}  # once, no "rejected"
        assert opt.last_alignment is not None

    async def test_p3_101_optimizer_resets_last_alignment_at_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CR-04: a later run that fails before aligning must not expose the earlier report."""
        opt = _optimizer()
        await opt.run(_bars(200))
        assert opt.last_alignment is not None
        monkeypatch.setattr("trading.optimizer._TIMEFRAME_SECONDS", {})
        with pytest.raises(ValueError, match="unsupported timeframe"):
            await opt.run(_bars(200))
        assert opt.last_alignment is None

    async def test_p3_102_last_alignment_docstring_states_the_warmup_zero_caveat(self) -> None:
        doc = ParameterOptimizer.last_alignment.__doc__ or ""
        assert "warmup 0" in doc and "presence_digest" in doc and "fingerprint" in doc
