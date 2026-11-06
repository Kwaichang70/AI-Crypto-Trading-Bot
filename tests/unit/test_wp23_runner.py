"""WP2.3 / P3: runner alignment, bookkeeping and telemetry (``packages/trading/backtest.py``).

Binding spec: ``reports/vp2-wp2.3/synthesis-spec.md`` (SY-23-05/07/11/15, I-1/I-5/I-6/I-8/I-10,
WP23-T-11, T-15, T-16, T-20, T-23, T-24) plus ``p2-arch-r1.md`` conditions 2-4 (ARCH-04/05/06) and
the P2 quant QA-06 expected values.  Test ids are ``WP23-P3-NN``.

The synthetic series is the quant's ``gen(1500, 20251)`` (3 symbols, 1h, seeded), so the expected
numbers are the ones the quant re-checks: ``total_bars`` 1450, S2 exposure 0.6352 and the OOS
Sharpe at index 1430 / 1449.
"""

# ruff: noqa: S311  (seeded stdlib generators build deterministic fixtures, not secrets)

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import cache
from typing import Any

import pytest
import structlog

from common.metrics import metrics
from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, SignalDirection, TimeFrame
from data.bar_alignment import (
    AlignmentTolerance,
    BarAlignmentError,
    align_bars,
    build_timeline,
)
from tests._cf36_logging import ConfigureFn
from trading import backtest as backtest_mod
from trading import clock as clock_mod
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.models import Signal
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import StrategyEngine

SYMS = ["AAA/EUR", "BBB/EUR", "CCC/EUR"]
H = timedelta(hours=1)
T0 = datetime(2025, 1, 1, tzinfo=UTC)
WARMUP = 50  # max(min_bars_required * 2, 50) for every strategy used here
LENIENT = AlignmentTolerance(
    max_missing_ppm=1_000_000,
    warn_missing_ppm=1_000_000,
    max_gap_bars=10_000,
    warn_gap_bars=10_000,
    max_head_gap_bars=10_000,
    max_tail_gap_bars=10_000,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@cache
def _gen(n: int, seed: int) -> dict[str, list[OHLCVBar]]:
    rng = random.Random(seed)
    px = dict.fromkeys(SYMS, 100.0)
    out: dict[str, list[OHLCVBar]] = {s: [] for s in SYMS}
    for k in range(n):
        f = rng.gauss(0.0002, 0.008)
        for s in SYMS:
            o = px[s]
            c = max(1.0, o * math.exp(f + rng.gauss(0.0001, 0.006)))
            hi = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
            lo = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
            px[s] = c
            out[s].append(
                OHLCVBar(
                    symbol=s,
                    timeframe=TimeFrame.ONE_HOUR,
                    timestamp=T0 + k * H,
                    open=Decimal(f"{o:.4f}"),
                    high=Decimal(f"{hi:.4f}"),
                    low=Decimal(f"{lo:.4f}"),
                    close=Decimal(f"{c:.4f}"),
                    volume=Decimal(10),
                )
            )
    return out


def _drop(
    full: dict[str, list[OHLCVBar]], missing: dict[str, set[int]]
) -> dict[str, list[OHLCVBar]]:
    return {s: [b for k, b in enumerate(full[s]) if k not in missing.get(s, set())] for s in full}


def _ma_runner(
    symbols: Sequence[str] = tuple(SYMS), tolerance: AlignmentTolerance | None = None
) -> BacktestRunner:
    return BacktestRunner(
        strategies=[
            MACrossoverStrategy(strategy_id="ma", params={"fast_period": 5, "slow_period": 20})
        ],
        symbols=list(symbols),
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config={
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.03,
            "bracket_take_profit_pct": 0.06,
        },
        trailing_stop_pct=0.04,
        allow_pyramiding=False,
        enable_circuit_breaker=True,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
        alignment_tolerance=tolerance,
    )


async def _run_ma(
    bars: dict[str, list[OHLCVBar]],
    *,
    symbols: Sequence[str] = tuple(SYMS),
    tolerance: AlignmentTolerance | None = None,
    oos: int | None = None,
    window: tuple[datetime, datetime] | None = None,
) -> tuple[BacktestRunner, Any]:
    runner = _ma_runner(symbols, tolerance)
    result = await runner.run(
        {s: bars[s] for s in symbols}, oos_start_index=oos, requested_window=window
    )
    return runner, result


class _BuyAtStrategy(BaseStrategy):
    """BUY ``bars[-1].symbol`` the first time its history is ``at`` bars long; never sells."""

    metadata = StrategyMetadata(name="wp23_buy_at", version="1.0.0", description="test double")
    requires_exit_manager = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._at = int(self._params.get("at", 60))
        self._done: set[str] = set()

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        sym = bars[-1].symbol
        if len(bars) < self._at or sym in self._done:
            return []
        self._done.add(sym)
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=sym,
                direction=SignalDirection.BUY,
                target_position=Decimal("1000"),
                confidence=1.0,
            )
        ]

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}


def _hold_runner(
    symbols: Sequence[str], tolerance: AlignmentTolerance | None = None, at: int = 60
) -> BacktestRunner:
    return BacktestRunner(
        strategies=[_BuyAtStrategy("buy_at", {"at": at})],
        symbols=list(symbols),
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        enable_circuit_breaker=False,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
        alignment_tolerance=tolerance,
    )


@pytest.fixture(autouse=True)
def _fresh_metrics() -> Any:
    metrics.reset()
    yield
    metrics.reset()


def _alignment_counters() -> dict[str, int]:
    return {k: v for k, v in metrics.get_all()["counters"].items() if "bar_alignment" in k}


# ---------------------------------------------------------------------------
# WP23-P3-01..04  runner bookkeeping rebinds (SY-23-11, T-15, T-16, QA-06)
# ---------------------------------------------------------------------------

S1_MISSING = {"BBB/EUR": {400, 401, 402}}


@cache
def _s2_missing() -> dict[str, set[int]]:
    rng = random.Random(99)
    return {
        "BBB/EUR": set(rng.sample(range(1, 1500), int(0.02 * 1500))),
        "CCC/EUR": set(rng.sample(range(1, 1500), int(0.05 * 1500))),
    }


class TestRunnerBookkeeping:
    async def test_p3_01_total_bars_is_timeline_minus_warmup_s1(self) -> None:
        """WP23-T-16: S1 total_bars 1450 (HEAD 1447); passes the DEFAULT tolerance (0.2%, gap 3)."""
        bars = _drop(_gen(1500, 20251), S1_MISSING)
        assert min(len(v) for v in bars.values()) == 1497  # HEAD: 1497 - 50 = 1447
        runner, res = await _run_ma(bars)
        assert res.total_bars == 1450
        rep = runner.last_alignment
        assert rep is not None and rep.steps == 1500 and not rep.violations
        assert res.exposure_pct == pytest.approx(res.bars_in_market / 1450)

    async def test_p3_02_s2_total_bars_and_exposure_denominator(self) -> None:
        """QA-06: S2 total_bars 1450 (HEAD 1375) and exposure 0.6352 (HEAD 0.6698)."""
        bars = _drop(_gen(1500, 20251), _s2_missing())
        assert min(len(v) for v in bars.values()) == 1425
        _, res = await _run_ma(bars, tolerance=LENIENT)
        assert res.total_bars == 1450
        assert round(res.exposure_pct, 4) == 0.6352
        assert res.bars_in_market == 921
        assert res.exposure_pct == pytest.approx(921 / 1450)
        assert all(0.0 <= v <= 1.0 for v in res.exposure_pct_per_symbol.values())

    @pytest.mark.parametrize(
        ("idx", "sharpe", "n_returns"),
        [(1430, -2.156, 85), (1449, -1.159, 62)],
    )
    async def test_p3_03_oos_index_between_min_len_and_timeline_is_accepted(
        self, idx: int, sharpe: float, n_returns: int
    ) -> None:
        """QA-06: an ordinal in [min(len), len(timeline)) was rejected (None) before."""
        rng = random.Random(99)
        miss = {
            "BBB/EUR": set(rng.sample(range(1, 1500), 30)),
            "CCC/EUR": set(rng.sample(range(1, 1500), 75)),
        }
        bars = _drop(_gen(1500, 20251), miss)
        assert min(len(v) for v in bars.values()) == 1425 <= idx  # HEAD: out of range -> None
        _, res = await _run_ma(bars, tolerance=LENIENT, oos=idx)
        assert res.oos_sharpe is not None
        assert res.oos_sharpe == pytest.approx(sharpe, abs=1e-3)
        assert res.oos_n_returns == n_returns

    async def test_p3_04_oos_index_at_timeline_length_is_still_out_of_range(self) -> None:
        bars = _drop(_gen(1500, 20251), S1_MISSING)
        _, res = await _run_ma(bars, oos=1500)  # len(timeline) == 1500
        assert res.oos_sharpe is None and res.oos_n_returns is None

    async def test_p3_05_mc_seed_23_ragged_tail_equals_the_trimmed_input(self) -> None:
        """P2 quant MC seed 23: C misses the LAST row; accepted after trimming, same result."""
        n = 1000
        full = _gen(n, 1000 + 23)
        rng = random.Random(23)
        rng.randrange(150, 600)  # the quant's outage start, consumed first
        miss = {
            "BBB/EUR": set(rng.sample(range(1, n), 20)),
            "CCC/EUR": set(rng.sample(range(1, n), 50)),
        }
        assert n - 1 in miss["CCC/EUR"]  # the fixture property this test is about
        raw = _drop(full, miss)
        lo = max(raw[s][0].timestamp for s in raw)
        hi = min(raw[s][-1].timestamp for s in raw)
        trimmed = {s: [b for b in raw[s] if lo <= b.timestamp <= hi] for s in raw}
        r_raw, res_raw = await _run_ma(raw, tolerance=LENIENT, oos=700)
        _, res_trim = await _run_ma(trimmed, tolerance=LENIENT, oos=700)
        rep = r_raw.last_alignment
        assert rep is not None
        assert {s.symbol: s.trimmed_trailing for s in rep.per_symbol}["AAA/EUR"] == 1
        for field in (
            "total_bars",
            "final_equity",
            "total_trades",
            "sharpe_ratio",
            "oos_sharpe",
            "max_drawdown_pct",
            "exposure_pct",
            "n_observations",
            "end_date",
        ):
            assert getattr(res_raw, field) == getattr(res_trim, field), field
        assert [(p.timestamp, p.equity) for p in res_raw.equity_curve] == [
            (p.timestamp, p.equity) for p in res_trim.equity_curve
        ]

    async def test_p3_06_terminal_mtm_and_date_span_follow_the_last_full_row(self) -> None:
        """T-15: B misses the last row (tail 1 passes); A's raw ``[-1]`` is not the terminal bar."""
        n = 200
        full = _gen(n, 5)
        bars = _drop({s: full[s] for s in ("AAA/EUR", "BBB/EUR")}, {"BBB/EUR": {n - 1}})
        assert bars["AAA/EUR"][-1].timestamp > bars["BBB/EUR"][-1].timestamp
        runner = _hold_runner(["AAA/EUR", "BBB/EUR"])
        res = await runner.run(bars)
        last_full = bars["BBB/EUR"][-1]
        assert res.end_date == last_full.timestamp
        assert res.start_date == bars["AAA/EUR"][0].timestamp
        assert res.duration_days == max(1, (res.end_date - res.start_date).days)
        mtm = {m.symbol: m for m in res.open_positions_mtm}
        assert set(mtm) == {"AAA/EUR", "BBB/EUR"}
        a_row = next(b for b in bars["AAA/EUR"] if b.timestamp == last_full.timestamp)
        assert mtm["AAA/EUR"].last_price == Decimal(str(a_row.close))
        assert mtm["AAA/EUR"].last_price != Decimal(str(bars["AAA/EUR"][-1].close))
        assert res.total_bars == (n - 1) - WARMUP

    async def test_p3_07_warmup_is_checked_on_the_timeline_not_on_raw_counts(self) -> None:
        """A has 100 raw bars, B has 100: HEAD's ``min(len) > 50`` passes; E holds only 45 rows."""
        full = _gen(155, 6)
        a = full["AAA/EUR"][:100]
        b = full["BBB/EUR"][55:155]
        assert len(a) == len(b) == 100 > WARMUP
        runner = _hold_runner(["AAA/EUR", "BBB/EUR"], tolerance=LENIENT)
        with pytest.raises(ValueError, match="Insufficient bars for warm-up") as ei:
            await runner.run({"AAA/EUR": a, "BBB/EUR": b})
        assert not isinstance(ei.value, BarAlignmentError)
        assert "45 steps" in str(ei.value)

    async def test_p3_08_a_held_symbol_counts_as_exposed_through_its_absent_steps(self) -> None:
        n = 160
        full = _gen(n, 7)
        two = {s: full[s] for s in ("AAA/EUR", "BBB/EUR")}
        # B is absent for three steps AFTER both entries (history 60 -> index 59).
        bars = _drop(two, {"BBB/EUR": {100, 101, 102}})
        runner = _hold_runner(["AAA/EUR", "BBB/EUR"], tolerance=LENIENT)
        res = await runner.run(bars)
        assert res.total_bars == n - WARMUP
        per = res.exposure_pct_per_symbol
        assert per["AAA/EUR"] == pytest.approx(per["BBB/EUR"])
        assert per["BBB/EUR"] > 0.9  # held to the end, absent steps included


# ---------------------------------------------------------------------------
# WP23-P3-10..  order, enforcement and rejection (SY-23-05/07, I-6, T-20)
# ---------------------------------------------------------------------------


def _violating_multi() -> dict[str, list[OHLCVBar]]:
    """C misses 5 consecutive rows of 600 (0.83%): max_gap 5 > 4 -> ``gap_too_long``."""
    return _drop(_gen(600, 11), {"CCC/EUR": {100, 101, 102, 103, 104}})


class TestEnforcementAndRejection:
    async def test_p3_10_rejection_happens_before_run_id_and_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> Any:
            raise AssertionError("must not be reached on a rejected alignment (I-6)")

        monkeypatch.setattr(BacktestRunner, "_build_engine", boom)
        monkeypatch.setattr("trading.backtest.uuid.uuid4", boom)
        runner = _ma_runner()
        with pytest.raises(BarAlignmentError) as ei:
            await runner.run(_violating_multi())
        assert ei.value.code == "bar_alignment_gap_too_long"
        assert str(ei.value) == "bar alignment check failed (bar_alignment_gap_too_long)"
        assert runner.last_alignment is not None  # the report survives a rejection
        assert runner.last_execution_engine is None and runner.last_portfolio is None

    async def test_p3_11_multi_symbol_off_grid_is_rejected_by_the_runner(self) -> None:
        """WP22-T-05 phase fixture: A on :00, B on :30 -> ``off_grid`` (never the engine guard)."""
        full = _gen(120, 12)
        shifted = [
            b.model_copy(update={"timestamp": b.timestamp + timedelta(minutes=30)})
            for b in full["BBB/EUR"]
        ]
        runner = _ma_runner(["AAA/EUR", "BBB/EUR"], LENIENT)
        with pytest.raises(BarAlignmentError) as ei:
            await runner.run({"AAA/EUR": full["AAA/EUR"], "BBB/EUR": shifted})
        assert ei.value.code == "bar_alignment_off_grid"

    async def test_p3_12_single_symbol_without_window_is_report_only(self) -> None:
        """SY-23-05 / T-10: a library single-symbol run violating the tolerance is reported."""
        bars = _drop(_gen(600, 13), {"CCC/EUR": {100, 101, 102, 103, 104}})
        runner, res = await _run_ma(bars, symbols=["CCC/EUR"])
        assert res.total_bars == 595 - WARMUP  # identity timeline, no trimming
        rep = runner.last_alignment
        assert rep is not None and [v.code for v in rep.violations] == [
            "bar_alignment_gap_too_long"
        ]
        counters = _alignment_counters()
        assert counters == {
            'bar_alignment_runs_total{outcome="ok_with_gaps"}': 1,
            'bar_alignment_missing_bars_total{symbol="CCC/EUR",timeframe="1h"}': 5,
        }

    async def test_p3_13_single_symbol_with_a_requested_window_is_enforced(self) -> None:
        bars = _drop(_gen(600, 13), {"CCC/EUR": {100, 101, 102, 103, 104}})
        window = (T0, T0 + 600 * H)
        runner = _ma_runner(["CCC/EUR"])
        with pytest.raises(BarAlignmentError) as ei:
            await runner.run({"CCC/EUR": bars["CCC/EUR"]}, requested_window=window)
        assert ei.value.code == "bar_alignment_gap_too_long"
        assert (
            _alignment_counters()[
                'bar_alignment_rejected_total{reason="bar_alignment_gap_too_long"}'
            ]
            == 1
        )

    async def test_p3_14_requested_window_trims_the_forming_bar(self) -> None:
        """SY-23-14: bars outside R are trimmed; the tail check then passes."""
        bars = _gen(200, 14)
        window = (T0, T0 + 199 * H)  # drops the last (forming) bar
        runner, res = await _run_ma(bars, symbols=["AAA/EUR"], window=window)
        rep = runner.last_alignment
        assert rep is not None and rep.per_symbol[0].trimmed_outside == 1
        assert res.total_bars == 199 - WARMUP
        assert res.end_date == T0 + 198 * H

    async def test_p3_15_duplicate_symbols_are_rejected_by_the_engine_arch_04(self) -> None:
        """ARCH-04: ``["A", "A"]`` made the engine (len) and the leaf (distinct) disagree."""
        runner = _ma_runner(["AAA/EUR", "AAA/EUR"])
        with pytest.raises(ValueError, match="duplicate symbols"):
            runner._build_engine("bt-x")


# ---------------------------------------------------------------------------
# WP23-P3-20..  telemetry (SY-23-15, I-8, I-10, T-20, T-23)
# ---------------------------------------------------------------------------


class TestTelemetry:
    async def test_p3_20_counters_have_no_run_id_and_one_label_key_set_per_name(self) -> None:
        bars = _drop(_gen(300, 20), {"BBB/EUR": {100, 101}, "CCC/EUR": {150}})
        await _run_ma(bars, tolerance=LENIENT)
        await _run_ma(_drop(_gen(300, 21), {"BBB/EUR": {120}}), tolerance=LENIENT)
        with pytest.raises(BarAlignmentError):
            await _ma_runner().run(_violating_multi())
        counters = _alignment_counters()
        assert counters, "vacuous: no bar_alignment counter recorded"
        assert not [k for k in counters if "run_id" in k]
        keysets: dict[str, set[frozenset[str]]] = {}
        for key in counters:
            name, _, rest = key.partition("{")
            labels = frozenset(p.split("=")[0] for p in rest.rstrip("}").split(",") if p)
            keysets.setdefault(name, set()).add(labels)
        assert keysets == {
            "bar_alignment_runs_total": {frozenset({"outcome"})},
            "bar_alignment_missing_bars_total": {frozenset({"symbol", "timeframe"})},
            "bar_alignment_rejected_total": {frozenset({"reason"})},
        }
        assert counters['bar_alignment_runs_total{outcome="ok_with_gaps"}'] == 2
        assert counters['bar_alignment_runs_total{outcome="rejected"}'] == 1
        assert counters['bar_alignment_rejected_total{reason="bar_alignment_gap_too_long"}'] == 1
        assert counters['bar_alignment_missing_bars_total{symbol="BBB/EUR",timeframe="1h"}'] == 3
        assert counters['bar_alignment_missing_bars_total{symbol="CCC/EUR",timeframe="1h"}'] == 6

    async def test_p3_21_a_clean_run_counts_ok_and_no_missing_bars(self) -> None:
        await _run_ma(_gen(200, 22))
        assert _alignment_counters() == {'bar_alignment_runs_total{outcome="ok"}': 1}

    async def test_p3_22_prometheus_export_uses_trading_prefix_and_consistent_labels(self) -> None:
        pytest.importorskip("prometheus_client")
        from apps.api.prometheus import TradingBotCollector

        await _run_ma(_drop(_gen(250, 23), {"BBB/EUR": {100}}), tolerance=LENIENT)
        with pytest.raises(BarAlignmentError):
            await _ma_runner().run(_violating_multi())
        fams = {f.name: f for f in TradingBotCollector(0.0).collect() if "bar_alignment" in f.name}
        assert set(fams) == {
            "trading_bar_alignment_runs",
            "trading_bar_alignment_missing_bars",
            "trading_bar_alignment_rejected",
        }
        labels = {n: {frozenset(s.labels) for s in f.samples} for n, f in fams.items()}
        assert labels == {
            "trading_bar_alignment_runs": {frozenset({"outcome"})},
            "trading_bar_alignment_missing_bars": {frozenset({"symbol", "timeframe"})},
            "trading_bar_alignment_rejected": {frozenset({"reason"})},
        }

    async def test_p3_23_events_carry_numbers_and_iso_stamps_only(self) -> None:
        bars = _drop(_gen(300, 24), {"BBB/EUR": {100, 101}})
        with structlog.testing.capture_logs() as logs:
            await _run_ma(bars, tolerance=LENIENT)
            with pytest.raises(BarAlignmentError):
                await _ma_runner().run(_violating_multi())
        by_event = {
            e["event"]: e for e in logs if str(e["event"]).startswith("backtest.bar_alignment")
        }
        assert set(by_event) == {
            "backtest.bar_alignment",
            "backtest.bar_alignment_gaps",
            "backtest.bar_alignment_rejected",
        }
        gaps = [e for e in logs if e["event"] == "backtest.bar_alignment_gaps"]
        first = next(e for e in gaps if e["symbol"] == "BBB/EUR")
        assert first["missing"] == 2 and first["max_gap_bars"] == 2
        assert first["first_gap_at"] == (T0 + 100 * H).isoformat()  # GapRun.start_iso (CR-05)
        assert first["gap_spans"] == [
            {"start": (T0 + 100 * H).isoformat(), "bars": 2, "kind": "interior"}
        ]
        rej = by_event["backtest.bar_alignment_rejected"]
        assert rej["code"] == "bar_alignment_gap_too_long"
        assert (rej["symbol"], rej["observed"], rej["limit"]) == ("CCC/EUR", 5, 4)
        assert one_event(logs, "backtest.bar_alignment") == 2  # one per run, none per bar
        for event in logs:
            for value in event.values():
                assert "bar alignment check failed" not in str(value)  # never str(exc)
                assert "Traceback" not in str(value)
            assert "exc_info" not in event

    async def test_p3_24_cf36_production_chain_has_no_exception_text(
        self, production_logging: ConfigureFn
    ) -> None:
        cap = production_logging(json_output=True)
        with pytest.raises(BarAlignmentError):
            await _ma_runner().run(_violating_multi())
        text = cap.text
        lines = cap.lines_with("backtest.bar_alignment_rejected")
        assert len(lines) == 1, "one rejection event (non-vacuity)"
        event = json.loads(lines[0])
        assert event["code"] == "bar_alignment_gap_too_long"
        assert "bar alignment check failed" not in text
        assert "Traceback" not in text

    async def test_p3_25_record_alignment_false_skips_telemetry_and_the_check(self) -> None:
        """SY-23-18: the optimizer's inner runners trim and build the timeline only."""
        bars = _violating_multi()
        runner = _ma_runner()
        with structlog.testing.capture_logs() as logs:
            res = await runner.run(bars, record_alignment=False)
        assert res.total_bars == 600 - WARMUP
        assert _alignment_counters() == {}
        assert not [e for e in logs if str(e["event"]).startswith("backtest.bar_alignment")]

    async def test_p3_26_record_alignment_false_still_rejects_structural_input(self) -> None:
        full = _gen(120, 12)
        shifted = [
            b.model_copy(update={"timestamp": b.timestamp + timedelta(minutes=30)})
            for b in full["BBB/EUR"]
        ]
        runner = _ma_runner(["AAA/EUR", "BBB/EUR"])
        with pytest.raises(BarAlignmentError) as ei:
            await runner.run(
                {"AAA/EUR": full["AAA/EUR"], "BBB/EUR": shifted}, record_alignment=False
            )
        assert ei.value.code == "bar_alignment_off_grid"
        assert _alignment_counters() == {}


def one_event(logs: list[Any], name: str) -> int:
    return sum(1 for e in logs if e["event"] == name)


# ---------------------------------------------------------------------------
# WP23-P3-30..  runner timeline == engine timeline (ARCH-05, p2-arch-r1 condition 2)
# ---------------------------------------------------------------------------


class TestRunnerEngineTimelineAgree:
    async def test_p3_30_engine_walks_exactly_the_runners_timeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        orig = StrategyEngine.run_backtest

        async def spy(self: StrategyEngine, bars: Any, *a: Any, **k: Any) -> Any:
            seen["bars"] = bars
            seen["kwargs"] = k
            seen["summary"] = await orig(self, bars, *a, **k)
            return seen["summary"]

        monkeypatch.setattr(StrategyEngine, "run_backtest", spy)
        n, oos = 400, 300
        raw = _drop(_gen(n, 30), {"CCC/EUR": {n - 1}, "BBB/EUR": {120, 121}})
        runner, res = await _run_ma(raw, tolerance=LENIENT, oos=oos)

        aligned = align_bars(raw, SYMS, 3600, tolerance=LENIENT, warmup_bars=WARMUP)
        engine_tl = build_timeline(seen["bars"], SYMS, 3600)
        assert engine_tl == aligned.timeline  # same steps, same present masks
        assert seen["bars"] == aligned.bars_by_symbol  # the ALIGNED dict reached the engine
        assert len(engine_tl) == n - 1
        assert res.total_bars == len(engine_tl) - WARMUP
        # OOS boundary: the equity curve is cut at the close of timeline step oos-1
        offset = seen["summary"]["oos_equity_curve_offset"]
        curve = runner.last_portfolio.get_equity_curve()  # type: ignore[union-attr]
        assert curve[offset - 1][0] == engine_tl[oos - 1].ts + H
        assert res.oos_sharpe is not None

    def test_p3_31_run_backtest_has_no_timeline_parameter(self) -> None:
        params = inspect.signature(StrategyEngine.run_backtest).parameters
        assert "timeline" not in params
        tree = ast.parse(inspect.getsource(backtest_mod))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "timeline"]
        assert calls == []


# ---------------------------------------------------------------------------
# WP23-P3-40..  I-5 by construction (ARCH-06)
# ---------------------------------------------------------------------------


class _CrossSymbolStrategy(BaseStrategy):
    """On A's bar stamped ``when`` emit a BUY for ``target`` (a symbol that may be absent)."""

    metadata = StrategyMetadata(name="wp23_cross", version="1.0.0", description="test double")
    requires_exit_manager = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._when: datetime = self._params["when"]
        self._target: str = self._params["target"]

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        if bars[-1].symbol != "AAA/EUR" or bars[-1].timestamp != self._when:
            return []
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=self._target,
                direction=SignalDirection.BUY,
                target_position=Decimal("1000"),
                confidence=1.0,
            )
        ]

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}


async def _cross_run(when: datetime) -> tuple[BacktestRunner, list[Any]]:
    full = _gen(200, 40)
    bars = _drop({s: full[s] for s in ("AAA/EUR", "BBB/EUR")}, {"BBB/EUR": {100}})
    runner = BacktestRunner(
        strategies=[_CrossSymbolStrategy("cross", {"when": when, "target": "BBB/EUR"})],
        symbols=["AAA/EUR", "BBB/EUR"],
        timeframe=TimeFrame.ONE_HOUR,
        seed=7,
        enable_circuit_breaker=False,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
        alignment_tolerance=LENIENT,
    )
    with structlog.testing.capture_logs() as logs:
        await runner.run(bars)
    return runner, logs


class TestAbsentSymbolSignals:
    async def test_p3_40_signal_for_an_absent_symbol_is_dropped_and_logged(self) -> None:
        runner, logs = await _cross_run(T0 + 100 * H)  # B has no bar at step 100
        assert runner.last_execution_engine.get_all_fills() == []  # type: ignore[union-attr]
        dropped = [e for e in logs if e["event"] == "engine.signal_dropped_symbol_absent"]
        assert len(dropped) == 1
        assert dropped[0]["symbol"] == "BBB/EUR" and dropped[0]["direction"] == "buy"
        assert dropped[0]["strategy_id"] == "cross"
        assert "exc_info" not in dropped[0]

    async def test_p3_41_control_the_same_signal_fills_when_the_symbol_is_present(self) -> None:
        runner, logs = await _cross_run(T0 + 110 * H)
        fills = runner.last_execution_engine.get_all_fills()  # type: ignore[union-attr]
        assert [f.symbol for f in fills] == ["BBB/EUR"]
        assert not [e for e in logs if e["event"] == "engine.signal_dropped_symbol_absent"]

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    async def test_p3_42_the_drop_is_backtest_only_paper_and_live_keep_the_signal(
        self, mode: RunMode
    ) -> None:
        """CR-01 / ARCH-06 scope: outside BACKTEST a signal for a symbol absent from
        ``current_bars`` is NOT dropped and no drop event is logged (CF-23-01 owns live)."""
        full = _gen(80, 42)
        a_bars = full["AAA/EUR"]
        when = a_bars[-1].timestamp
        runner = BacktestRunner(
            strategies=[_CrossSymbolStrategy("cross", {"when": when, "target": "BBB/EUR"})],
            symbols=["AAA/EUR", "BBB/EUR"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
        )
        engine, *_ = runner._build_engine("bt-cr01")
        engine._run_mode = mode  # the step under test only reads the mode
        seen: list[Signal] = []

        async def fake_process(signal: Signal) -> list[Any]:
            seen.append(signal)
            return []

        engine._execution_engine.process_signal = fake_process  # type: ignore[method-assign]
        engine._entry_protectable = lambda *_a, **_k: True  # type: ignore[method-assign]
        with structlog.testing.capture_logs() as logs:
            await engine._process_bar({"AAA/EUR": a_bars[-1]}, {"AAA/EUR": list(a_bars)})
        assert [sig.symbol for sig in seen] == ["BBB/EUR"], "the signal must reach the executor"
        assert not [e for e in logs if e["event"] == "engine.signal_dropped_symbol_absent"]

    async def test_p3_43_control_backtest_mode_drops_the_same_signal_directly(self) -> None:
        full = _gen(80, 42)
        a_bars = full["AAA/EUR"]
        runner = BacktestRunner(
            strategies=[
                _CrossSymbolStrategy("cross", {"when": a_bars[-1].timestamp, "target": "BBB/EUR"})
            ],
            symbols=["AAA/EUR", "BBB/EUR"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
        )
        engine, *_ = runner._build_engine("bt-cr01c")
        seen: list[Signal] = []

        async def fake_process(signal: Signal) -> list[Any]:
            seen.append(signal)
            return []

        engine._execution_engine.process_signal = fake_process  # type: ignore[method-assign]
        with structlog.testing.capture_logs() as logs:
            await engine._process_bar({"AAA/EUR": a_bars[-1]}, {"AAA/EUR": list(a_bars)})
        assert seen == []
        assert len([e for e in logs if e["event"] == "engine.signal_dropped_symbol_absent"]) == 1


class TestLastAlignmentIsNeverStale:
    async def test_p3_44_runner_resets_last_alignment_at_entry(self) -> None:
        """CR-04: after a successful run, a run that fails validation leaves ``None``."""
        runner = _ma_runner(["AAA/EUR", "BBB/EUR"])
        good = {s: _gen(120, 44)[s] for s in ("AAA/EUR", "BBB/EUR")}
        await runner.run(good)
        first = runner.last_alignment
        assert first is not None
        with pytest.raises(ValueError, match="No bar data"):
            await runner.run({"AAA/EUR": good["AAA/EUR"]})  # BBB missing -> _validate_bars
        assert runner.last_alignment is None

    async def test_p3_45_a_rejected_run_holds_its_own_report_not_the_previous_one(self) -> None:
        runner = _ma_runner()
        await runner.run(_gen(200, 45))
        ok_report = runner.last_alignment
        assert ok_report is not None and not ok_report.violations
        with pytest.raises(BarAlignmentError):
            await runner.run(_violating_multi())
        rep = runner.last_alignment
        assert rep is not None and rep is not ok_report
        assert [v.code for v in rep.violations] == ["bar_alignment_gap_too_long"]


# ---------------------------------------------------------------------------
# WP23-P3-50..  determinism (T-24) and multi-symbol gap-free identity (T-11)
# ---------------------------------------------------------------------------


class TestDeterminismAndIdentity:
    async def test_p3_50_permuted_dict_order_and_poisoned_wall_clock_give_identical_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> Any:
            raise AssertionError("wall clock read")

        monkeypatch.setattr(clock_mod, "_wall_now", boom)
        bars = _drop(_gen(400, 50), {"BBB/EUR": {150, 151}, "CCC/EUR": {200}})
        r1, a = await _run_ma(bars, tolerance=LENIENT, oos=300)
        permuted = {s: bars[s] for s in reversed(list(bars))}
        r2, b = await _run_ma(permuted, tolerance=LENIENT, oos=300)
        for field in (
            "final_equity",
            "total_trades",
            "sharpe_ratio",
            "oos_sharpe",
            "total_bars",
            "exposure_pct",
            "max_drawdown_pct",
        ):
            assert getattr(a, field) == getattr(b, field), field
        assert [(p.timestamp, p.equity) for p in a.equity_curve] == [
            (p.timestamp, p.equity) for p in b.equity_curve
        ]
        assert r1.last_alignment is not None and r2.last_alignment is not None
        assert r1.last_alignment.presence_digest == r2.last_alignment.presence_digest
        assert r1.last_alignment.to_canonical_json() == r2.last_alignment.to_canonical_json()

    async def test_p3_51_gap_free_multi_symbol_hands_the_same_lists_to_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T-11 / I-1: nothing is copied or trimmed when nothing is wrong."""
        seen: dict[str, Any] = {}
        orig = StrategyEngine.run_backtest

        async def spy(self: StrategyEngine, bars: Any, *a: Any, **k: Any) -> Any:
            seen["bars"] = bars
            return await orig(self, bars, *a, **k)

        monkeypatch.setattr(StrategyEngine, "run_backtest", spy)
        bars = _gen(200, 51)
        runner, res = await _run_ma(bars)
        assert all(seen["bars"][s] is bars[s] for s in SYMS)
        assert res.total_bars == 200 - WARMUP
        rep = runner.last_alignment
        assert rep is not None and not rep.violations and not rep.warnings
        assert _alignment_counters() == {'bar_alignment_runs_total{outcome="ok"}': 1}

    def test_p3_52_bar_count_mismatch_warning_is_retired(self) -> None:
        tree = ast.parse(inspect.getsource(backtest_mod))
        events = {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert "backtest.bar_count_mismatch" not in events
        assert "Series will be truncated to shortest length" not in events

    def test_p3_53_event_loop_is_not_needed_by_the_helpers(self) -> None:
        # the telemetry helpers are plain sync functions (no spawn on the backtest path)
        assert not asyncio.iscoroutinefunction(backtest_mod.record_bar_alignment)
        assert not asyncio.iscoroutinefunction(backtest_mod.record_bar_alignment_outcome)


# ---------------------------------------------------------------------------
# WP23-P3-60  T-17: an in-sample outage does not move the OOS window
# ---------------------------------------------------------------------------


class TestOosIsTimeExact:
    async def test_p3_60_in_sample_outage_leaves_the_oos_result_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        summaries: list[dict[str, Any]] = []
        orig = StrategyEngine.run_backtest

        async def spy(self: StrategyEngine, bars: Any, *a: Any, **k: Any) -> Any:
            summaries.append(await orig(self, bars, *a, **k))
            return summaries[-1]

        monkeypatch.setattr(StrategyEngine, "run_backtest", spy)
        n, idx = 1500, 1000
        full = _gen(n, 20251)
        outage = _drop(full, {"CCC/EUR": set(range(500, 548))})  # 48 bars, in-sample
        assert len(outage["CCC/EUR"]) == n - 48
        r_gap, gap = await _run_ma(outage, tolerance=LENIENT, oos=idx)
        _, ref = await _run_ma(full, oos=idx)
        assert gap.oos_sharpe is not None and ref.oos_sharpe is not None
        assert gap.oos_sharpe == pytest.approx(ref.oos_sharpe, abs=1e-4)
        assert gap.oos_n_returns == ref.oos_n_returns
        rep = r_gap.last_alignment
        assert rep is not None and rep.steps == n
        # boundary stamp = the close of timeline step idx-1; the other n-idx steps are OOS
        curve = r_gap.last_portfolio.get_equity_curve()  # type: ignore[union-attr]
        offset = summaries[0]["oos_equity_curve_offset"]
        assert curve[offset - 1][0] == T0 + (idx - 1) * H + H
        assert len({ts for ts, _ in curve[offset:]}) == n - idx
