"""WP2.5 / P1: formula consolidation in ``packages/trading/metrics.py``.

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-02, -06, -07, -08, -09; WP25-T-07,
T-08, T-10).  The reference values are the closed-form ones of SY-25-21 (quant QD-12, population
moments, ``rel=1e-9``).  P1 is behaviour-neutral; the identity harness (T-09) lives in
``test_wp25_metric_identity.py``.
"""

# ruff: noqa: S311  (seeded stdlib generators build deterministic fixtures, not secrets)

from __future__ import annotations

import ast
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from common.types import TimeFrame
from trading import metrics as metrics_mod
from trading.equity_sampling import simple_returns
from trading.metrics import (
    KNOWN_METRICS_VERSIONS,
    METRICS_VERSION,
    TIMEFRAME_PERIODS_PER_YEAR,
    EquityCurvePoint,
    ReturnMetrics,
    _max_dd_duration,
    compute_cagr,
    compute_psr,
    compute_return_metrics,
    compute_returns_from_equity,
    compute_sharpe,
    periods_per_year,
    return_moments,
)
from trading.strategy_engine import _TIMEFRAME_SECONDS

_REPO = Path(__file__).resolve().parents[2]
_BACKTEST = _REPO / "packages" / "trading" / "backtest.py"
_T0 = datetime(2025, 1, 1, tzinfo=UTC)
_REL = 1e-9
_SEED_CAPITAL = Decimal(10000)


def _equity(returns: list[str], initial: Decimal = _SEED_CAPITAL) -> list[Decimal]:
    """Equity path ``[initial, initial*(1+r0), ...]`` with exact Decimal compounding."""
    out = [initial]
    for r in returns:
        out.append(out[-1] * (Decimal(1) + Decimal(r)))
    return out


def _ref_a() -> list[Decimal]:
    """REF-A: 1h, 100 returns alternating +2% / -1%."""
    return _equity(["0.02", "-0.01"] * 50)


def _ref_b() -> list[Decimal]:
    """REF-B: 1d, [+1%, +1%, -1.5%] x 120."""
    return _equity(["0.01", "0.01", "-0.015"] * 120)


def _ref_d() -> list[Decimal]:
    """REF-D: 1d, [+0.5% x 29, -10%] x 12."""
    return _equity((["0.005"] * 29 + ["-0.10"]) * 12)


def _max_drawdown(values: list[Decimal]) -> float:
    curve = [
        EquityCurvePoint(timestamp=_T0 + timedelta(days=i), equity=v, drawdown_pct=0.0)
        for i, v in enumerate(values)
    ]
    return metrics_mod.compute_max_drawdown(curve)


def _check(m: ReturnMetrics, *, n: int, sharpe: float, sortino: float, psr: float) -> None:
    assert m.n == n
    assert m.sharpe == pytest.approx(sharpe, rel=_REL)
    assert m.sortino == pytest.approx(sortino, rel=_REL)
    assert m.psr == pytest.approx(psr, rel=_REL)


# ---------------------------------------------------------------------------
# WP25-T-07: strict annualisation
# ---------------------------------------------------------------------------


class TestT07PeriodsPerYear:
    def test_table_keys_are_every_timeframe(self) -> None:
        assert set(TIMEFRAME_PERIODS_PER_YEAR) == set(TimeFrame)

    @pytest.mark.parametrize("tf", list(TimeFrame))
    def test_value_is_365_25_days_over_timeframe_seconds(self, tf: TimeFrame) -> None:
        assert TIMEFRAME_PERIODS_PER_YEAR[tf] == 365.25 * 86400 / _TIMEFRAME_SECONDS[tf]
        assert periods_per_year(tf) == TIMEFRAME_PERIODS_PER_YEAR[tf]

    def test_unknown_timeframe_raises_fixed_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = dict(TIMEFRAME_PERIODS_PER_YEAR)
        del table[TimeFrame.ONE_HOUR]
        monkeypatch.setattr(metrics_mod, "TIMEFRAME_PERIODS_PER_YEAR", table)
        with pytest.raises(ValueError, match=r"^unknown timeframe for annualisation$") as ei:
            periods_per_year(TimeFrame.ONE_HOUR)
        assert ei.value.__cause__ is None
        assert ei.value.__suppress_context__ is True  # `from None` is present

    def test_backtest_has_no_silent_fallback(self) -> None:
        tree = ast.parse(_BACKTEST.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "TIMEFRAME_PERIODS_PER_YEAR" not in names | attrs
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "get":
                    seg = ast.unparse(node)
                    assert "365.25" not in seg
                    assert "PERIODS_PER_YEAR" not in seg
        imported = {
            a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module == "trading.metrics"
            for a in n.names
        }
        assert "periods_per_year" in imported
        assert "compute_return_metrics" in imported

    def test_version_markers(self) -> None:
        assert METRICS_VERSION == "wp25-1"
        assert frozenset({METRICS_VERSION}) == KNOWN_METRICS_VERSIONS


# ---------------------------------------------------------------------------
# WP25-T-08: reference values (SY-25-21, moments column)
# ---------------------------------------------------------------------------


class TestT08ReferenceValues:
    def test_ref_a_truth(self) -> None:
        eq = _ref_a()
        returns = simple_returns(eq)
        m = compute_return_metrics(returns, 8766.0)
        _check(m, n=100, sharpe=31.052536128310, sortino=65.872376608105, psr=0.999516575858)
        assert m.sharpe == pytest.approx(math.sqrt(0.99 * 8766) / 3, rel=1e-12)
        assert m.skew == pytest.approx(0.0, abs=1e-9)
        assert m.kurtosis == pytest.approx(1.0, rel=_REL)
        assert float(eq[-1]) == pytest.approx(16284.270878242956, rel=_REL)

    def test_ref_a_holes_calendar_asof(self) -> None:
        """Holes {10, 11, 40} carried forward on the 101-slot grid."""
        full = _ref_a()
        grid = [full[0]]
        for k in range(1, len(full)):
            grid.append(grid[-1] if k in (10, 11, 40) else full[k])
        m = compute_return_metrics(simple_returns(grid), 8766.0)
        _check(m, n=100, sharpe=31.734024926469, sortino=67.873857157875, psr=0.999632787550)
        assert grid[-1] == full[-1]
        assert float(grid[-1]) == pytest.approx(16284.270878242956, rel=_REL)
        # documenting assertion (CF-23-05 hand-over): normal moments, not sample moments
        normal = compute_psr(m.sharpe / math.sqrt(8766.0), m.n, skew=0.0, kurtosis=3.0)
        assert normal == pytest.approx(0.999480133000, rel=_REL)

    def test_ref_a_holes_present_steps(self) -> None:
        full = _ref_a()
        kept = [e for k, e in enumerate(full) if k not in (10, 11, 40)]
        m = compute_return_metrics(simple_returns(kept), 8766.0)
        _check(m, n=97, sharpe=32.273953452686, sortino=68.904695060463, psr=0.999617543652)
        normal = compute_psr(m.sharpe / math.sqrt(8766.0), m.n, skew=0.0, kurtosis=3.0)
        assert normal == pytest.approx(0.999483471599, rel=_REL)

    def test_ref_a_normal_moments_documenting(self) -> None:
        m = compute_return_metrics(simple_returns(_ref_a()), 8766.0)
        normal = compute_psr(m.sharpe / math.sqrt(8766.0), m.n, skew=0.0, kurtosis=3.0)
        assert normal == pytest.approx(0.999342829083, rel=_REL)

    def test_ref_b(self) -> None:
        eq = _ref_b()
        m = compute_return_metrics(simple_returns(eq), 365.25)
        _check(m, n=360, sharpe=2.699019883834, sortino=3.672900845011, psr=0.994591818926)
        assert m.skew == pytest.approx(-1 / math.sqrt(2), rel=_REL)
        assert m.kurtosis == pytest.approx(1.5, rel=_REL)
        assert float(eq[-1]) == pytest.approx(17761.407055941010, rel=_REL)
        assert _max_dd_duration(eq) == 2
        assert _max_drawdown(eq) == pytest.approx(0.015, rel=_REL)
        assert compute_cagr(eq[0], eq[-1], 360.0) == pytest.approx(0.791082452901, rel=_REL)

    def test_ref_d(self) -> None:
        eq = _ref_d()
        m = compute_return_metrics(simple_returns(eq), 365.25)
        _check(m, n=360, sharpe=1.518851456416, sortino=1.567988859176, psr=0.893977246831)
        assert m.skew == pytest.approx(-5.199469468957, rel=_REL)
        assert m.kurtosis == pytest.approx(28.034482758621, rel=_REL)
        assert float(eq[-1]) == pytest.approx(16021.368315442864, rel=_REL)
        assert _max_dd_duration(eq) == 22
        assert _max_drawdown(eq) == pytest.approx(0.10, rel=_REL)
        assert compute_cagr(eq[0], eq[-1], 360.0) == pytest.approx(0.613187347495, rel=_REL)
        normal = compute_psr(m.sharpe / math.sqrt(365.25), m.n, skew=0.0, kurtosis=3.0)
        assert normal == pytest.approx(0.933635694562, rel=_REL)

    def test_cagr_spans(self) -> None:
        assert compute_cagr(Decimal(10000), Decimal(10500), 30.0) == pytest.approx(
            0.811255495540, rel=_REL
        )
        assert compute_cagr(Decimal(10000), Decimal(13000), 730.0) == pytest.approx(
            0.140277875348, rel=_REL
        )

    def test_short_series_is_psr_less_but_still_counted(self) -> None:
        m = compute_return_metrics([0.01, -0.005, 0.002], 365.25)
        assert (m.n, m.psr) == (3, None)
        assert (m.skew, m.kurtosis) == (0.0, 3.0)  # fewer than 4 observations

    def test_empty_series(self) -> None:
        m = compute_return_metrics([], 8766.0)
        assert m == ReturnMetrics(n=0, sharpe=0.0, sortino=0.0, psr=None, skew=0.0, kurtosis=3.0)

    def test_non_positive_annualisation_uses_raw_sharpe(self) -> None:
        returns = [0.01, -0.004, 0.006, 0.002] * 10
        m = compute_return_metrics(returns, 0.0)
        assert m.sharpe == 0.0
        assert m.psr == compute_psr(0.0, 40, skew=m.skew, kurtosis=m.kurtosis)

    def test_matches_the_standalone_functions(self) -> None:
        rng = random.Random(5)
        returns = [rng.gauss(0.0004, 0.01) for _ in range(250)]
        m = compute_return_metrics(returns, 8766.0)
        skew, kurt = return_moments(returns)
        assert m.sharpe == compute_sharpe(returns, 8766.0)
        assert (m.skew, m.kurtosis) == (skew, kurt)
        assert m.psr == compute_psr(m.sharpe / math.sqrt(8766.0), 250, skew=skew, kurtosis=kurt)


# ---------------------------------------------------------------------------
# WP25-T-10: zero-std guard, CAGR overflow, one returns formula
# ---------------------------------------------------------------------------


def _head_cagr(initial: Decimal, final: Decimal, days: int) -> float:
    """The HEAD (01d393d) ``compute_cagr`` body, verbatim."""
    if initial <= Decimal("0") or days <= 0:
        return 0.0
    ratio = float(final / initial)
    if ratio <= 0:
        return -1.0
    exponent = 365.25 / days
    return float(ratio**exponent) - 1.0


class TestT10EdgeCases:
    def test_constant_series_sharpe_is_zero(self) -> None:
        assert compute_sharpe([0.01] * 50, periods_per_year=252.0) == 0.0
        assert compute_sharpe([-0.003] * 37, periods_per_year=8766.0) == 0.0
        assert compute_sharpe([0.0] * 20, periods_per_year=8766.0) == 0.0

    def test_guard_boundary_is_relative(self) -> None:
        mean, std = 0.01, 2e-9 * 0.01  # std/|mean| = 2e-9 > tolerance -> a real Sharpe
        returns = [mean - std, mean + std] * 10
        assert compute_sharpe(returns, 1.0) > 1e8
        m, s = 0.01, 5e-10 * 0.01  # sample std/|mean| ~5.1e-10: inside 1e-9, outside 1e-10
        assert compute_sharpe([m - s, m + s] * 10, 1.0) == 0.0
        assert metrics_mod._ZERO_STD_REL_TOL == 1e-9
        noisy = [mean - 1e-12 * mean, mean + 1e-12 * mean] * 10  # std/|mean| ~ 1e-12
        assert compute_sharpe(noisy, 1.0) == 0.0

    def test_return_moments_zero_std(self) -> None:
        assert return_moments([0.01] * 50) == (0.0, 3.0)
        assert return_moments([0.0] * 50) == (0.0, 3.0)
        assert return_moments([0.01, 0.02, 0.03]) == (0.0, 3.0)  # n < 4
        assert return_moments([0.01, -0.02, 0.03, -0.01]) != (0.0, 3.0)  # exactly n == 4
        neg = [-0.01 * (1 + 1e-12), -0.01 * (1 - 1e-12)] * 10  # |mean| 0.01, std/|mean| ~1e-12
        assert return_moments(neg) == (0.0, 3.0)

    def test_return_moments_values_are_population(self) -> None:
        skew, kurt = return_moments([0.02, -0.01] * 10)
        assert skew == pytest.approx(0.0, abs=1e-9)
        assert kurt == pytest.approx(1.0, rel=_REL)

    def test_cagr_overflow_returns_inf_without_raising(self) -> None:
        with pytest.raises(OverflowError):
            float(1000.0 ** (365.25 / 1.0))  # the HEAD failure mode
        assert compute_cagr(Decimal(10), Decimal(10000), 1.0) == math.inf
        assert compute_cagr(Decimal(10), Decimal(10000), 1) == math.inf

    @pytest.mark.parametrize("days", [1, 2, 7, 30, 90, 365, 366, 730, 1500])
    @pytest.mark.parametrize("final", ["12000", "10000", "9000", "10001.5", "250", "0", "-5"])
    def test_integer_day_calls_are_byte_identical_to_head(self, days: int, final: str) -> None:
        try:
            expected = _head_cagr(_SEED_CAPITAL, Decimal(final), days)
        except OverflowError:  # pragma: no cover - none of these overflow
            pytest.fail("HEAD overflowed on a fixture that must not")
        assert compute_cagr(_SEED_CAPITAL, Decimal(final), days) == expected
        assert compute_cagr(_SEED_CAPITAL, Decimal(final), float(days)) == expected

    def test_cagr_degenerate_inputs(self) -> None:
        assert compute_cagr(Decimal(0), Decimal(5), 10) == 0.0
        assert compute_cagr(Decimal(10), Decimal(5), 0) == 0.0
        assert compute_cagr(Decimal(10), Decimal(5), -2.5) == 0.0
        assert compute_cagr(Decimal(10), Decimal(-5), 10.5) == -1.0

    def test_returns_from_equity_delegates_to_simple_returns(self) -> None:
        rng = random.Random(11)
        for _ in range(25):
            n = rng.randint(0, 60)
            values = [Decimal(str(round(rng.uniform(-5, 20000), 4))) for _ in range(n)]
            curve = [
                EquityCurvePoint(
                    timestamp=_T0 + timedelta(hours=i), equity=v, drawdown_pct=0.0
                )
                for i, v in enumerate(values)
            ]
            assert compute_returns_from_equity(curve) == simple_returns([p.equity for p in curve])
        assert compute_returns_from_equity([]) == []

    def test_returns_from_equity_zero_base_is_zero_return(self) -> None:
        curve = [
            EquityCurvePoint(
                timestamp=_T0 + timedelta(hours=i), equity=Decimal(v), drawdown_pct=0.0
            )
            for i, v in enumerate(["0", "5", "10"])
        ]
        assert compute_returns_from_equity(curve) == [0.0, 1.0]

    def test_max_dd_duration_matches_the_curve_wrapper(self) -> None:
        rng = random.Random(3)
        for _ in range(25):
            values = [
                Decimal(str(round(rng.uniform(90, 110), 2))) for _ in range(rng.randint(0, 80))
            ]
            curve = [
                EquityCurvePoint(timestamp=_T0 + timedelta(hours=i), equity=v, drawdown_pct=0.0)
                for i, v in enumerate(values)
            ]
            assert metrics_mod.compute_max_drawdown_duration(curve) == _max_dd_duration(values)
        assert _max_dd_duration([Decimal(1)]) == 0
        assert _max_dd_duration([Decimal(2), Decimal(1), Decimal(1), Decimal(3)]) == 2
        assert _max_dd_duration([Decimal(2), Decimal(1), Decimal(1)]) == 2  # ends in drawdown
