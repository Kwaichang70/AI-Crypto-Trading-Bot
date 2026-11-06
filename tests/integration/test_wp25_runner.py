"""WP2.5 / P2: the behaviour switch, end to end through ``BacktestRunner``.

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-04..11, SY-25-17, SY-25-23,
WP25-T-11..T-20, T-24, T-25).  T-17 (invariants vs ``head_capture.json``) lives in
``tests/unit/test_wp25_metric_identity.py``; T-21 in ``test_wp23_runner.py`` /
``test_sprint50_cycle6_oos_backtest.py``; T-22 is ``test_wp22_bar_clock.py::TestT16...`` plus
``test_wp21c_rebaseline.py`` (both unedited); T-23 is ``tests/unit/test_wp25_ast_pins.py``.

The oracle below (``_oracle_sample``) is written independently of
``trading.equity_sampling``: last point per stamp, calendar grid, as-of carry.  The reference
values are the closed-form numbers of SY-25-21 / QD-12.
"""

# ruff: noqa: S311  (a seeded stdlib generator builds a deterministic price path)

from __future__ import annotations

import ast
import inspect
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from data.bar_alignment import AlignmentTolerance
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.unit import test_wp25_metric_identity as harness
from tests.unit.test_wp21c_rebaseline import rb
from trading import clock as clock_mod
from trading import optimizer as optimizer_mod
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.equity_sampling import EquitySamplingError, sample_equity_per_bar, simple_returns
from trading.metrics import (
    TIMEFRAME_PERIODS_PER_YEAR,
    BacktestResult,
    EquityCurvePoint,
    _derive_confidence_flag,
    _max_dd_duration,
    compute_cagr,
    compute_calmar,
    compute_max_drawdown_duration,
    compute_psr,
    compute_return_metrics,
    compute_returns_from_equity,
    compute_sharpe,
    return_moments,
)
from trading.optimizer import ParameterOptimizer
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategy_engine import StrategyEngine
from trading.walk_forward import WalkForwardValidator, deflated_sharpe_ratio

SYM = "BTC/USDT"
SYM2 = "ETH/USDT"
H = timedelta(hours=1)
D = timedelta(days=1)
FIRST_OPEN = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)  # open of the first post-warm-up bar
CAPITAL = Decimal("10000")
LENIENT = AlignmentTolerance(
    max_missing_ppm=1_000_000,
    warn_missing_ppm=1_000_000,
    max_gap_bars=10_000,
    warn_gap_bars=10_000,
    max_head_gap_bars=10_000,
    max_tail_gap_bars=10_000,
)
#: QD-12 ENG-1: 50 flat warm-up bars at 100, closes 100..110, then 29 bars flat at 110.
ENG1_CLOSES: list[int] = [100 + i for i in range(11)] + [110] * 29
ENG1_SCRIPT: dict[int, str] = {0: "buy", 10: "sell"}


# ---------------------------------------------------------------------------
# harness + independent oracle
# ---------------------------------------------------------------------------


async def _run(
    script: dict[int, str],
    closes: Sequence[int | str | Decimal],
    *,
    profile: str = "legacy_backtest_v0",
    symbols: tuple[str, ...] = (SYM,),
    warmup: int | None = None,
    oos: int | None = None,
    tolerance: AlignmentTolerance | None = None,
    bars: dict[str, list[OHLCVBar]] | None = None,
    breaker: bool = False,
) -> tuple[BacktestRunner, BacktestResult, dict[str, list[OHLCVBar]]]:
    runner = BacktestRunner(
        strategies=[
            ScriptedSequenceStrategy("wp25", {"script": script, "target_notional": "3000"})
        ],
        symbols=list(symbols),
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=CAPITAL,
        seed=7,
        allow_pyramiding=False,
        enable_circuit_breaker=breaker,
        cost_profile=get_cost_profile(profile),
        alignment_tolerance=tolerance,
    )
    if warmup is not None:
        runner._warmup_bars = warmup  # the only way to W != 50 (harness precedent)
    if bars is None:
        w = 50 if warmup is None else warmup
        bars = {
            s: scripted_path(closes, first_open=FIRST_OPEN, symbol=s, warmup=w) for s in symbols
        }
    result = await runner.run(bars, oos_start_index=oos)
    return runner, result, bars


def _oracle_sample(
    curve: Sequence[Any], start_at: datetime, tf: timedelta
) -> tuple[list[Decimal], int]:
    """Independent calendar as-of sample: ``(equity per slot, carried slots)``."""
    last: dict[datetime, Decimal] = {}
    for p in curve:
        last[p.timestamp] = p.equity  # later points of a stamp win
    stamps = sorted(s for s in last if s >= start_at)
    assert stamps and stamps[0] == start_at, "no base point"
    slots = (stamps[-1] - start_at) // tf
    equity: list[Decimal] = []
    carried = 0
    j = 0
    for k in range(slots + 1):
        slot = start_at + k * tf
        while j + 1 < len(stamps) and stamps[j + 1] <= slot:
            j += 1
        equity.append(last[stamps[j]])
        if k >= 1 and slot not in last:
            carried += 1
    return equity, carried


def _returns(equity: Sequence[Decimal]) -> list[float]:
    return [
        float((equity[i] - equity[i - 1]) / equity[i - 1]) if equity[i - 1] > 0 else 0.0
        for i in range(1, len(equity))
    ]


def _points(equities: Sequence[Decimal], t0: datetime, tf: timedelta) -> list[EquityCurvePoint]:
    return [
        EquityCurvePoint(timestamp=t0 + i * tf, equity=e, drawdown_pct=0.0)
        for i, e in enumerate(equities)
    ]


def _compound(rets: Sequence[str], times: int) -> list[Decimal]:
    eq = [CAPITAL]
    for _ in range(times):
        for r in rets:
            eq.append(eq[-1] * (Decimal(1) + Decimal(r)))
    return eq


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# WP25-T-11  REF-C through sampler + compute_return_metrics == REF-B
# ---------------------------------------------------------------------------


class TestT11RefC:
    REF_B_RETURNS = ("0.01", "0.01", "-0.015")

    def _curves(self) -> tuple[list[EquityCurvePoint], list[EquityCurvePoint]]:
        trading = _compound(self.REF_B_RETURNS, 120)  # 361 values, 360 returns
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        ref_b = _points(trading, t0, D)
        # REF-C: the seed plus 206 flat warm-up marks, then REF-B
        ref_c = _points([CAPITAL] * 206 + trading, t0, D)
        return ref_b, ref_c

    def test_ref_c_through_the_sampler_equals_ref_b_bit_for_bit(self) -> None:
        ref_b, ref_c = self._curves()
        t0 = ref_b[0].timestamp
        sb = sample_equity_per_bar(ref_b, start_at=t0, timeframe_s=86400)
        sc = sample_equity_per_bar(ref_c, start_at=t0 + 206 * D, timeframe_s=86400)
        assert sc.equity == sb.equity and sc.n_returns == sb.n_returns == 360
        mb = compute_return_metrics(sb.returns(), 365.25)
        mc = compute_return_metrics(sc.returns(), 365.25)
        assert mc == mb  # dataclass equality: every float bit for bit
        assert mb.sharpe == pytest.approx(2.699019883834, rel=1e-9)
        assert mb.psr == pytest.approx(0.994591818926, rel=1e-9)
        assert sb.equity[-1] == pytest.approx(Decimal("17761.407055941010"), rel=Decimal("1e-12"))

    def test_head_style_all_points_values_are_documented(self) -> None:
        """What HEAD computed for REF-C: n 566, deflated Sharpe, 565-day CAGR."""
        _, ref_c = self._curves()
        head = compute_return_metrics(simple_returns([p.equity for p in ref_c]), 365.25)
        assert head.n == 566
        assert head.sharpe == pytest.approx(2.145824653418, rel=1e-9)
        assert head.psr == pytest.approx(0.994877450250, rel=1e-9)
        final = ref_c[-1].equity
        assert compute_cagr(CAPITAL, final, 565) == pytest.approx(0.449696809730, rel=1e-9)
        assert compute_cagr(CAPITAL, final, 360.0) == pytest.approx(0.791082452901, rel=1e-9)


# ---------------------------------------------------------------------------
# WP25-T-12  ENG-1, three profiles (end to end)
# ---------------------------------------------------------------------------

ENG1 = [
    # profile, sharpe, HEAD max drawdown, final equity, CAGR (40 h span; None = not pinned)
    ("frictionless_diagnostic", 53.37490449241541, 0.0, "10150", 25.1236080375),
    ("legacy_backtest_v0", 51.618945300970225, 0.000325097355, "10143.7001125", None),
    ("coinbase_advanced_1", 34.347493637194724, 0.002115417608, "10109.0518", 9.7718169678),
]


class TestT12Eng1:
    @pytest.mark.parametrize(("profile", "sharpe", "mdd", "final", "cagr"), ENG1)
    async def test_eng1_values(
        self, profile: str, sharpe: float, mdd: float, final: str, cagr: float | None
    ) -> None:
        _, res, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES, profile=profile)
        assert res.n_observations == 40
        assert res.sharpe_ratio == pytest.approx(sharpe, rel=1e-9)
        # max_drawdown_pct stays on ALL points (SY-25-05): the HEAD value, intrabar mark included
        assert res.max_drawdown_pct == pytest.approx(mdd, abs=1e-12)  # table rounded to 12 dp
        assert res.final_equity == Decimal(final)
        if cagr is not None:
            assert res.cagr == pytest.approx(cagr, rel=1e-9)
        assert res.trading_start_at == datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
        assert res.trading_end_at == datetime(2026, 3, 11, 16, 0, tzinfo=UTC)
        assert res.metric_sampling == "calendar_asof" and res.carried_slots == 0
        assert res.metrics_version == "wp25-1" and res.periods_per_year == 8766.0
        eq, _ = _oracle_sample(res.equity_curve, res.trading_start_at, H)
        assert eq[0] == CAPITAL  # CAGR numerator base (SY-25-06)
        assert eq[-1] == res.final_equity

    async def test_cagr_uses_the_trading_window_not_the_bar_open_span(self) -> None:
        _, res, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES, profile="coinbase_advanced_1")
        assert res.trading_start_at is not None and res.trading_end_at is not None
        span = (res.trading_end_at - res.trading_start_at).total_seconds() / 86400
        assert span == pytest.approx(40 / 24)
        assert res.cagr == compute_cagr(CAPITAL, res.final_equity, span)
        # HEAD: integer floor of the bar-open span, warm-up included
        assert res.duration_days == 3
        assert res.cagr != compute_cagr(CAPITAL, res.final_equity, res.duration_days)
        assert res.calmar_ratio == compute_calmar(res.cagr, res.max_drawdown_pct)


# ---------------------------------------------------------------------------
# WP25-T-13  contract: base point, W = 0, OOS by time
# ---------------------------------------------------------------------------


class TestT13Contract:
    @pytest.mark.parametrize("warmup", [0, 1, 50, 120])
    async def test_trading_start_is_the_close_of_the_last_warmup_step(self, warmup: int) -> None:
        _, res, bars = await _run(ENG1_SCRIPT, ENG1_CLOSES, warmup=warmup)
        curve = res.equity_curve
        t0 = bars[SYM][0].timestamp
        assert curve[0].timestamp == t0
        assert res.trading_start_at == curve[warmup].timestamp
        if warmup == 0:
            assert res.trading_start_at == t0  # W = 0: the seed point
        else:
            assert res.trading_start_at == bars[SYM][warmup - 1].timestamp + H  # close_{W-1}
        n_bars = len(bars[SYM])
        assert res.n_observations == n_bars - warmup == res.total_bars  # gap-free
        assert res.trading_end_at == curve[-1].timestamp

    async def test_oos_by_time_is_gap_free_with_fills_inside_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        holder: dict[str, Any] = {}
        orig = StrategyEngine.run_backtest

        async def spy(self: StrategyEngine, *a: Any, **kw: Any) -> Any:
            holder["summary"] = await orig(self, *a, **kw)
            return holder["summary"]

        monkeypatch.setattr(StrategyEngine, "run_backtest", spy)
        closes = [100 + (i * 7) % 13 for i in range(60)]
        oos = 70  # bar ordinal; call index 20
        script = {0: "buy", 10: "sell", 22: "buy", 30: "sell", 40: "buy", 50: "sell"}
        _, res, bars = await _run(script, closes, oos=oos)
        n_total = len(bars[SYM])
        offset = holder["summary"]["oos_equity_curve_offset"]
        curve = res.equity_curve
        assert res.oos_start_at == curve[offset - 1].timestamp  # close_{oos-1}
        assert res.oos_start_at == bars[SYM][oos - 1].timestamp + H
        assert res.oos_n_returns == n_total - oos == 40  # gap-free
        assert res.n_observations == n_total - 50 == 60
        # fills INSIDE the window add curve points; HEAD counted them as extra returns
        head_n = len(curve[offset - 1 :]) - 1
        assert head_n > 40
        eq, _ = _oracle_sample(curve, res.oos_start_at, H)
        assert res.oos_sharpe == compute_sharpe(_returns(eq), 8766.0)

    async def test_oos_at_the_warmup_boundary_equals_the_full_window(self) -> None:
        closes = [100 + (i * 7) % 13 for i in range(60)]
        _, res, _ = await _run({0: "buy", 20: "sell"}, closes, oos=50)
        assert res.oos_start_at == res.trading_start_at
        assert res.oos_n_returns == res.n_observations
        assert res.oos_sharpe == res.sharpe_ratio


# ---------------------------------------------------------------------------
# WP25-T-14  multi-symbol all-absent slot
# ---------------------------------------------------------------------------


class TestT14AllAbsentSlot:
    async def test_calendar_asof_carries_the_slot_and_the_invariant_holds(self) -> None:
        closes = [100 + (i * 7) % 13 for i in range(60)]
        script = {0: "buy", 20: "sell", 30: "buy"}
        symbols = (SYM, SYM2)
        bars = {s: scripted_path(closes, first_open=FIRST_OPEN, symbol=s) for s in symbols}
        drop = 80  # a post-warm-up row that BOTH symbols miss: no timeline step at all
        bars = {s: [b for i, b in enumerate(v) if i != drop] for s, v in bars.items()}
        _, res, _ = await _run(script, closes, symbols=symbols, tolerance=LENIENT, bars=bars)
        assert res.metric_sampling == "calendar_asof"
        assert res.carried_slots == 1
        assert (
            res.total_bars == 109 - 50
        )  # timeline steps (110 bars, one row missing) minus warm-up
        assert res.n_observations == res.total_bars + res.carried_slots == 110 - 50
        assert res.trading_start_at is not None and res.trading_end_at is not None
        assert (res.trading_end_at - res.trading_start_at) / H == res.n_observations
        eq, carried = _oracle_sample(res.equity_curve, res.trading_start_at, H)
        assert carried == 1 and len(eq) - 1 == res.n_observations
        assert res.sharpe_ratio == compute_sharpe(_returns(eq), 8766.0)
        assert 0.0 in _returns(eq)  # the carried slot is a zero return


# ---------------------------------------------------------------------------
# WP25-T-15  off-grid single-symbol library run: present_steps + one warning
# ---------------------------------------------------------------------------


class TestT15OffGridFallback:
    async def test_completes_with_present_steps_and_one_numbers_only_warning(self) -> None:
        closes = [100 + (i * 7) % 13 for i in range(60)]
        bars = scripted_path(closes, first_open=FIRST_OPEN, symbol=SYM)
        shifted = bars[80].model_copy(
            update={"timestamp": bars[80].timestamp + timedelta(minutes=30)}
        )
        bars = [*bars[:80], shifted, *bars[81:]]
        runner = BacktestRunner(
            strategies=[
                ScriptedSequenceStrategy(
                    "wp25", {"script": {0: "buy", 20: "sell"}, "target_notional": "3000"}
                )
            ],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=CAPITAL,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
        )
        log = MagicMock()
        runner._log = log
        res = await runner.run({SYM: bars})
        assert res.metric_sampling == "present_steps" and res.carried_slots == 0
        assert res.n_observations == len(bars) - 50  # every step is a distinct stamp
        fallbacks = [
            c
            for c in log.warning.call_args_list
            if c.args[0] == "backtest.metric_sampling_fallback"
        ]
        assert len(fallbacks) == 1
        assert set(fallbacks[0].kwargs) == {"metric_sampling", "n_returns"}  # no exception text
        assert fallbacks[0].kwargs == {
            "metric_sampling": "present_steps",
            "n_returns": res.n_observations,
        }
        complete = next(c for c in log.info.call_args_list if c.args[0] == "backtest.complete")
        assert complete.kwargs["metric_sampling"] == "present_steps"
        assert complete.kwargs["n_returns"] == res.n_observations

    async def test_on_grid_run_logs_no_fallback_and_reports_the_counts(self) -> None:
        runner = BacktestRunner(
            strategies=[
                ScriptedSequenceStrategy("wp25", {"script": ENG1_SCRIPT, "target_notional": "3000"})
            ],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=CAPITAL,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
        )
        log = MagicMock()
        runner._log = log
        res = await runner.run({SYM: scripted_path(ENG1_CLOSES, first_open=FIRST_OPEN, symbol=SYM)})
        assert not [c for c in log.warning.call_args_list if "fallback" in c.args[0]]
        complete = next(c for c in log.info.call_args_list if c.args[0] == "backtest.complete")
        assert complete.kwargs["n_returns"] == res.n_observations == 40
        assert complete.kwargs["metric_sampling"] == "calendar_asof"


# ---------------------------------------------------------------------------
# WP25-T-16  stop inside warm-up; forged curve; fail-fast annualisation
# ---------------------------------------------------------------------------


class TestT16EdgeCases:
    async def test_stop_inside_warmup_persists_the_empty_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QR-09 (P0): n 0, Sharpe 0.0, PSR None, CAGR 0.0 and the mode fields still set."""
        orig = StrategyEngine._update_engine_prices
        seen = {"n": 0}

        def stop_after_ten(self: StrategyEngine, current_bars: Any) -> Any:
            seen["n"] += 1
            if seen["n"] == 10:
                self._stop_event.set()
            return orig(self, current_bars)

        monkeypatch.setattr(StrategyEngine, "_update_engine_prices", stop_after_ten)
        _, res, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES, oos=70)
        assert len(res.equity_curve) < 50  # stopped before warm-up ended
        assert res.n_observations == 0
        assert res.sharpe_ratio == 0.0 and res.sortino_ratio == 0.0
        assert res.psr is None and res.confidence_flag is None
        assert res.cagr == 0.0 and res.calmar_ratio == 0.0
        assert res.max_drawdown_duration_bars == 0
        assert res.metric_sampling == "calendar_asof" and res.carried_slots == 0
        assert res.metrics_version == "wp25-1" and res.periods_per_year == 8766.0
        assert res.trading_start_at is not None
        assert res.trading_end_at is None  # no trading window: never a reversed [start, end]
        assert res.oos_sharpe is None and res.oos_n_returns is None and res.oos_start_at is None

    async def test_forged_curve_without_the_base_stamp_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orig = BacktestRunner._build_equity_curve

        def forged(self: BacktestRunner, raw: Any) -> list[EquityCurvePoint]:
            pts = orig(self, raw)
            return [*pts[:50], *pts[51:]]  # drop the base point curve[W]

        monkeypatch.setattr(BacktestRunner, "_build_equity_curve", forged)
        with pytest.raises(
            EquitySamplingError, match=r"^equity sampling failed \(equity_sampling_no_base_point\)$"
        ):
            await _run(ENG1_SCRIPT, ENG1_CLOSES)

    async def test_annualisation_is_resolved_before_the_simulation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QR-10(e): an unsupported timeframe fails fast, the engine never runs."""
        import trading.backtest as backtest_mod

        def boom(_tf: TimeFrame) -> float:
            raise ValueError("unknown timeframe for annualisation")

        called = {"run": False}
        orig = StrategyEngine.run_backtest

        async def spy(self: StrategyEngine, *a: Any, **kw: Any) -> Any:
            called["run"] = True
            return await orig(self, *a, **kw)

        monkeypatch.setattr(backtest_mod, "_periods_per_year", boom)
        monkeypatch.setattr(StrategyEngine, "run_backtest", spy)
        with pytest.raises(ValueError, match="unknown timeframe for annualisation"):
            await _run(ENG1_SCRIPT, ENG1_CLOSES)
        assert called["run"] is False


# ---------------------------------------------------------------------------
# WP25-T-17 (values): the harness scenarios against the independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", harness.SCENARIOS, ids=[sc.name for sc in harness.SCENARIOS])
async def test_t17_harness_values_equal_the_oracle_on_the_head_curve(sc: harness.Scenario) -> None:
    """The new metric values of each of the 10 harness scenarios.

    The oracle samples the equity curve CAPTURED AT HEAD (01d393d, so independent of this
    run's own curve object) with ``start_at = curve[W].timestamp``; every allow-listed
    number must equal what the runner reports.
    """
    cap = harness._load_capture()[sc.name]["result"]
    w = 50 if sc.warmup_override is None else sc.warmup_override
    tf = harness._STEP[sc.timeframe]
    head_curve = [
        SimpleNamespace(timestamp=_utc(p["timestamp"]), equity=Decimal(p["equity"]))
        for p in cap["equity_curve"]
    ]
    start_at = head_curve[w].timestamp
    eq, carried = _oracle_sample(head_curve, start_at, tf)
    ppy = {TimeFrame.ONE_HOUR: 8766.0, TimeFrame.ONE_DAY: 365.25}[sc.timeframe]
    expected = compute_return_metrics(_returns(eq), ppy)

    runner = harness._runner(sc)
    res = await runner.run(harness._gen_bars(sc), oos_start_index=sc.oos_start_index)

    assert res.trading_start_at == start_at
    assert res.carried_slots == carried and res.metric_sampling == "calendar_asof"
    assert res.n_observations == expected.n == len(eq) - 1
    assert res.sharpe_ratio == expected.sharpe
    assert res.sortino_ratio == expected.sortino
    assert res.psr == expected.psr
    assert res.confidence_flag == _derive_confidence_flag(expected.psr, cap["total_trades"])
    assert res.max_drawdown_duration_bars == _max_dd_duration(eq)
    assert res.max_drawdown_pct == cap["max_drawdown_pct"]  # all points: the HEAD value
    span = max(1.0, (head_curve[-1].timestamp - start_at).total_seconds() / 86400)
    cagr = compute_cagr(eq[0], res.final_equity, span)
    assert eq[0] == CAPITAL and res.cagr == cagr
    assert res.calmar_ratio == compute_calmar(cagr, cap["max_drawdown_pct"])
    if sc.oos_start_index is None:
        assert res.oos_sharpe is None and res.oos_start_at is None
    else:
        assert res.oos_start_at == start_at + (sc.oos_start_index - w) * tf or sc.holes
        assert res.oos_start_at is not None
        oos_eq, _ = _oracle_sample(head_curve, res.oos_start_at, tf)
        oos = compute_return_metrics(_returns(oos_eq), ppy)
        assert res.oos_n_returns == oos.n and res.oos_sharpe == oos.sharpe


# ---------------------------------------------------------------------------
# WP25-T-18  duration in sample bars; max_drawdown_pct == max(point drawdown)
# ---------------------------------------------------------------------------


class TestT18DurationAndDrawdown:
    def test_adding_fill_points_leaves_the_sample_duration_unchanged(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        base = _points([CAPITAL, Decimal(11), Decimal(9), Decimal(8), Decimal(12)], t0, H)
        extra = [
            *base[:3],
            EquityCurvePoint(timestamp=base[2].timestamp, equity=Decimal(7), drawdown_pct=0.0),
            EquityCurvePoint(timestamp=base[2].timestamp, equity=Decimal(9), drawdown_pct=0.0),
            *base[3:],
        ]
        start = base[0].timestamp
        a = sample_equity_per_bar(base, start_at=start, timeframe_s=3600)
        b = sample_equity_per_bar(extra, start_at=start, timeframe_s=3600)
        assert _max_dd_duration(a.equity) == _max_dd_duration(b.equity)
        # the point-based count (HEAD) is NOT stable under fill points
        assert compute_max_drawdown_duration(base) != compute_max_drawdown_duration(extra)

    async def test_runner_duration_counts_bars_and_mdd_is_the_max_point_drawdown(self) -> None:
        closes = [100, 99, 98, 97, 96, 97, 98, 99, 100, 101] * 6
        script = {0: "buy", 3: "sell", 8: "buy", 14: "sell", 20: "buy", 26: "sell"}
        _, res, _ = await _run(script, closes, profile="coinbase_advanced_1")
        assert res.trading_start_at is not None
        eq, _ = _oracle_sample(res.equity_curve, res.trading_start_at, H)
        assert res.max_drawdown_duration_bars == _max_dd_duration(eq) > 0
        points_based = compute_max_drawdown_duration(res.equity_curve)
        assert len(res.equity_curve) > len(eq)  # fills add curve points
        assert points_based >= res.max_drawdown_duration_bars
        assert res.max_drawdown_pct == max(p.drawdown_pct for p in res.equity_curve)


# ---------------------------------------------------------------------------
# WP25-T-19  CAGR spans and W-neutrality
# ---------------------------------------------------------------------------


class TestT19CagrAndWarmupNeutrality:
    def test_cagr_spans(self) -> None:
        assert compute_cagr(CAPITAL, Decimal(10500), 30.0) == pytest.approx(
            0.811255495540, rel=1e-9
        )
        assert compute_cagr(CAPITAL, Decimal(13000), 730.0) == pytest.approx(
            0.140277875348, rel=1e-9
        )
        # HEAD's floored bar-open spans, for the record
        assert compute_cagr(CAPITAL, Decimal(10500), 29) == pytest.approx(0.848738853132, rel=1e-9)
        assert compute_cagr(CAPITAL, Decimal(13000), 935) == pytest.approx(0.107926693724, rel=1e-9)

    async def test_cagr_span_is_floored_at_one_day(self) -> None:  # CR-01
        _, res, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES[:16], profile="coinbase_advanced_1")
        assert res.trading_start_at is not None and res.trading_end_at is not None
        span = (res.trading_end_at - res.trading_start_at).total_seconds() / 86400
        assert span < 1.0  # 16 h here, so the floor must bite
        assert res.cagr == compute_cagr(CAPITAL, res.final_equity, 1.0)
        assert res.cagr != compute_cagr(CAPITAL, res.final_equity, span)
        assert res.calmar_ratio == compute_calmar(res.cagr, res.max_drawdown_pct)

    async def test_sub_day_giant_move_gives_inf_cagr_without_raising(self) -> None:  # CR-03
        closes = [100] * 3 + [5000] * 13  # 30 units bought at 100, marked at 5000 within 16 h
        _, res, _ = await _run({0: "buy", 14: "sell"}, closes, profile="coinbase_advanced_1")
        assert res.trading_start_at is not None and res.trading_end_at is not None
        assert (res.trading_end_at - res.trading_start_at).total_seconds() < 86400
        assert res.final_equity / CAPITAL > 7
        assert math.isinf(res.cagr)

    async def test_two_warmups_with_the_same_trading_path_give_identical_metrics(self) -> None:
        closes = [100 + (i * 7) % 13 for i in range(120)]
        script = {0: "buy", 20: "sell", 40: "buy", 70: "sell"}
        _, r50, _ = await _run(script, closes, warmup=50, profile="coinbase_advanced_1")
        _, r206, _ = await _run(script, closes, warmup=206, profile="coinbase_advanced_1")
        assert r50.trading_start_at == r206.trading_start_at == FIRST_OPEN
        for f in ("sharpe_ratio", "sortino_ratio", "cagr", "psr", "n_observations",
                  "calmar_ratio", "max_drawdown_duration_bars", "confidence_flag"):  # fmt: skip
            assert getattr(r50, f) == getattr(r206, f), f
        assert r50.final_equity == r206.final_equity and r50.n_observations == 120
        # ... where the all-points formula (HEAD) depends on W
        head50 = compute_sharpe(compute_returns_from_equity(r50.equity_curve), 8766.0)
        head206 = compute_sharpe(compute_returns_from_equity(r206.equity_curve), 8766.0)
        assert head50 != head206
        assert head206 < head50 < r50.sharpe_ratio


# ---------------------------------------------------------------------------
# WP25-T-20  PSR identity, including medium / high confidence (QR-09, CR-09)
# ---------------------------------------------------------------------------


def _drift_closes(trips: int) -> list[Decimal]:
    import random

    rng = random.Random(5)
    px = Decimal(100)
    out: list[Decimal] = []
    for _ in range(trips * 4 + 10):
        px = px * (Decimal(1) + Decimal(str(round(rng.gauss(0.0012, 0.004), 5))))
        out.append(px.quantize(Decimal("0.0001")))
    return out


class TestT20PsrIdentity:
    @staticmethod
    def _check_psr(res: BacktestResult) -> None:
        assert res.trading_start_at is not None and res.periods_per_year is not None
        eq, _ = _oracle_sample(res.equity_curve, res.trading_start_at, H)
        rets = _returns(eq)
        skew, kurt = return_moments(rets)
        expected = compute_psr(
            res.sharpe_ratio / math.sqrt(res.periods_per_year),
            res.n_observations,
            skew=skew,
            kurtosis=kurt,
        )
        assert res.psr == expected
        assert res.n_observations == len(rets)

    async def test_eng1(self) -> None:
        _, res, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES)
        assert res.psr is not None
        self._check_psr(res)

    @pytest.mark.parametrize(
        ("trips", "flag", "n", "psr", "head_n", "head_psr"),
        [
            (40, "medium", 170, 0.9824383645361262, 296, 0.9834432681168097),
            (60, "high", 250, 0.9985202217729746, 414, 0.9986889133963168),
        ],
    )
    async def test_runs_reaching_medium_and_high_confidence(
        self, trips: int, flag: str, n: int, psr: float, head_n: int, head_psr: float
    ) -> None:
        script: dict[int, str] = {}
        for k in range(trips):
            script[4 * k] = "buy"
            script[4 * k + 2] = "sell"
        _, res, _ = await _run(script, _drift_closes(trips), profile="frictionless_diagnostic")
        assert res.n_observations == n
        assert 0.95 <= (res.psr or 0.0) < 1.0  # not saturated: the value is discriminating
        assert res.psr == pytest.approx(psr, rel=1e-9)
        assert res.confidence_flag == flag
        self._check_psr(res)
        # the point-based estimator (HEAD) on the same curve: n inflated by warm-up + fills,
        # PSR nearly invariant (WP2.3 QD-05)
        head = compute_return_metrics(compute_returns_from_equity(res.equity_curve), 8766.0)
        assert head.n == head_n and head.psr == pytest.approx(head_psr, rel=1e-9)
        assert abs((head.psr or 0.0) - (res.psr or 0.0)) < 0.002
        assert head.sharpe < res.sharpe_ratio


# ---------------------------------------------------------------------------
# WP25-T-22  CF-26 freeze: the rebaseline helpers keep the pre-WP2.5 formulas
# ---------------------------------------------------------------------------


class TestT22RebaselineFreeze:
    """``test_wp22_bar_clock.py::TestT16RebaselineUnaffected`` (the 117236f digest) and
    ``test_wp21c_rebaseline.py`` are unedited and green; here the two helpers are pinned
    to the old all-points formulas on a fixture (SY-25-17)."""

    async def test_helpers_equal_the_all_points_formulas_and_differ_from_wp25(self) -> None:
        _, res, _ = await _run(
            {0: "buy", 20: "sell", 40: "buy", 70: "sell"},
            [100 + (i * 7) % 13 for i in range(120)],
            profile="coinbase_advanced_1",
        )
        old_sharpe = compute_sharpe(
            compute_returns_from_equity(res.equity_curve),
            TIMEFRAME_PERIODS_PER_YEAR[res.timeframe],
        )
        assert rb._pre_wp25_sharpe(res) == old_sharpe
        assert rb._pre_wp25_cagr(res) == compute_cagr(CAPITAL, res.final_equity, res.duration_days)
        # the new numbers are the WP2.5 ones, so the frozen helpers must not read them
        assert rb._pre_wp25_sharpe(res) != res.sharpe_ratio
        assert rb._pre_wp25_cagr(res) != res.cagr


# ---------------------------------------------------------------------------
# WP25-T-24  determinism
# ---------------------------------------------------------------------------


def _stable(res: BacktestResult) -> str:
    dumped = res.model_dump(mode="json")
    dumped.pop("run_id")
    for trade in dumped["trades"]:
        trade.pop("run_id", None)
        trade.pop("trade_id", None)
    return harness.canonical(dumped)


class TestT24Determinism:
    async def test_two_runs_are_byte_identical_and_never_read_the_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, a, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES, profile="coinbase_advanced_1", oos=70)

        def poisoned() -> datetime:
            raise AssertionError("the backtest path read the wall clock")

        monkeypatch.setattr(clock_mod, "_wall_now", poisoned)
        _, b, _ = await _run(ENG1_SCRIPT, ENG1_CLOSES, profile="coinbase_advanced_1", oos=70)
        assert _stable(a) == _stable(b)
        for f in ("trading_start_at", "trading_end_at", "periods_per_year", "metric_sampling",
                  "carried_slots", "oos_start_at", "metrics_version"):  # fmt: skip
            assert getattr(a, f) == getattr(b, f), f


# ---------------------------------------------------------------------------
# WP25-T-25  optimizer / walk-forward / CF-22-04
# ---------------------------------------------------------------------------


class TestT25OptimizerWalkForward:
    async def test_every_fold_counts_bars_after_warmup(self) -> None:
        sc = harness.SCENARIOS[0]
        bars = harness._gen_bars(sc)
        folds = WalkForwardValidator(num_folds=3, train_fraction=0.7, mode="expanding").split(bars)
        assert folds
        for fold in folds:
            test = fold.test_bars[sc.symbols[0]]
            runner = harness._runner(sc)
            res = await runner.run(fold.test_bars, record_alignment=False)
            assert res.n_observations == len(test) - 50  # len(test) - W
            assert res.metric_sampling == "calendar_asof"

    async def test_ranking_reads_the_new_sharpe(self) -> None:
        sc = harness.SCENARIOS[0]
        bars = harness._gen_bars(sc)
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[sc.symbols[0]],
            timeframe=TimeFrame.ONE_HOUR,
            param_grid={"fast_period": [5, 8], "slow_period": [20, 30]},
            cost_profile=get_cost_profile("legacy_backtest_v0"),
            enable_circuit_breaker=False,
        )
        result = await opt.run(bars)
        sharpes = [float(e.metrics["sharpe_ratio"] or 0.0) for e in result.entries]
        assert sharpes == sorted(sharpes, reverse=True) and len(sharpes) == 4
        for entry in result.entries:
            res = await opt._run_single_backtest(
                combo_params=dict(entry.params), strategy_id="check", bars_by_symbol=bars
            )
            assert entry.metrics["sharpe_ratio"] == res.sharpe_ratio
            assert res.n_observations == res.total_bars and res.metrics_version == "wp25-1"
            w = next(
                i for i, p in enumerate(res.equity_curve) if p.timestamp == res.trading_start_at
            )
            assert w >= 50  # the runner floors the warm-up at 50 (longer for slow_period 30)
            assert res.n_observations == len(bars[sc.symbols[0]]) - w

    async def test_deflated_sharpe_call_signature_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = inspect.signature(deflated_sharpe_ratio).parameters
        assert list(params)[:3] == ["observed_sharpe", "sharpe_stddev", "num_trials"]
        assert all(p.default is not inspect.Parameter.empty for p in list(params.values())[3:])
        calls: list[dict[str, Any]] = []

        def spy(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return deflated_sharpe_ratio(**kwargs)

        monkeypatch.setattr(optimizer_mod, "deflated_sharpe_ratio", spy)
        sc = harness.SCENARIOS[0]
        bars = harness._gen_bars(sc)
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[sc.symbols[0]],
            timeframe=TimeFrame.ONE_HOUR,
            param_grid={"fast_period": [5, 8], "slow_period": [20]},
            walk_forward=True,
            walk_forward_folds=2,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
            enable_circuit_breaker=False,
        )
        result = await opt.run(bars)
        assert result.walk_forward and len(calls) == 1
        assert set(calls[0]) == {"observed_sharpe", "sharpe_stddev", "num_trials"}
        src = ast.parse(Path(inspect.getsourcefile(optimizer_mod) or "").read_text())
        kws = {
            kw.arg
            for n in ast.walk(src)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "deflated_sharpe_ratio"
            for kw in n.keywords
        }
        assert kws == {"observed_sharpe", "sharpe_stddev", "num_trials"}


class TestCf2204ExposureIsTheIntervalUnion:
    """SY-25-23: no holding metric; ``bars_in_market`` is ``|union [entry_at, exit_at)| / tf``.

    Partial exits (several ``TradeResult`` rows with one ``entry_at``) and several symbols make
    the naive per-trade sum larger.  An open lot is counted up to and including the last bar,
    i.e. to ``trading_end_at + tf``.
    """

    @staticmethod
    def _union_bars(res: BacktestResult, tf: timedelta) -> tuple[int, int]:
        assert res.trading_end_at is not None
        iv = [(t.entry_at, t.exit_at) for t in res.trades]
        iv += [(p.opened_at, res.trading_end_at + tf) for p in res.open_positions_mtm]
        iv.sort()
        total = timedelta(0)
        cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s > cur_e:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        total += cur_e - cur_s
        return total // tf, sum((e - s) // tf for s, e in iv)

    @pytest.mark.parametrize("idx", [0, 1, 5])  # partial exits / open lot / multi-symbol
    async def test_bars_in_market_equals_the_union(self, idx: int) -> None:
        sc = harness.SCENARIOS[idx]
        runner = harness._runner(sc)
        res = await runner.run(harness._gen_bars(sc), oos_start_index=sc.oos_start_index)
        union, naive = self._union_bars(res, harness._STEP[sc.timeframe])
        assert res.bars_in_market == union
        if idx in (0, 5):
            assert naive > union  # the per-trade sum double counts


# ---------------------------------------------------------------------------
# field descriptions (QR-08) and the defaulted constructor
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_duration_description_says_steps_in_present_steps_mode(self) -> None:
        desc = BacktestResult.model_fields["max_drawdown_duration_bars"].description or ""
        assert "present_steps" in desc and "steps" in desc

    def test_stale_descriptions_are_gone(self) -> None:
        fields = BacktestResult.model_fields
        assert "len(equity_curve) - 1" not in (fields["n_observations"].description or "")
        assert "len(oos_equity_slice) - 1" not in (fields["oos_n_returns"].description or "")
        assert "all curve points" in (fields["max_drawdown_pct"].description or "")
        assert "trading window" in (fields["cagr"].description or "")

    def test_new_fields_are_defaulted(self) -> None:
        for name in ("metrics_version", "trading_start_at", "trading_end_at", "periods_per_year",
                     "metric_sampling", "carried_slots", "oos_start_at"):  # fmt: skip
            assert not BacktestResult.model_fields[name].is_required(), name
        assert BacktestResult.model_fields["metrics_version"].default == ""
        assert BacktestResult.model_fields["carried_slots"].default == 0
