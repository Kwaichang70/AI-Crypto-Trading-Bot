"""
tests/unit/test_sprint44_metrics.py
------------------------------------
Sprint 44 Observability & Metrics Uplift — coverage for the new
metric functions in :mod:`trading.metrics`.

Items
-----
QT-005   Ulcer Index, Omega Ratio, CVaR (Conditional VaR).
QT-006   Exposure-adjusted Sharpe.
QT-012   Buy-and-hold return, alpha/beta, information ratio.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading.metrics import (
    EquityCurvePoint,
    compute_alpha_beta,
    compute_buy_and_hold_return,
    compute_cvar,
    compute_exposure_adjusted_sharpe,
    compute_information_ratio,
    compute_omega_ratio,
    compute_ulcer_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curve(equities: list[float]) -> list[EquityCurvePoint]:
    """Build an equity curve from a list of equity values."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        EquityCurvePoint(
            timestamp=base + timedelta(hours=i),
            equity=Decimal(str(eq)),
            drawdown_pct=0.0,
        )
        for i, eq in enumerate(equities)
    ]


# ---------------------------------------------------------------------------
# QT-005a: Ulcer Index
# ---------------------------------------------------------------------------


class TestUlcerIndex:
    def test_empty_curve_returns_zero(self) -> None:
        assert compute_ulcer_index([]) == 0.0

    def test_single_point_returns_zero(self) -> None:
        assert compute_ulcer_index(_curve([10000.0])) == 0.0

    def test_monotonically_rising_curve_returns_zero(self) -> None:
        """No drawdowns → Ulcer = 0."""
        assert compute_ulcer_index(_curve([100, 110, 120, 130])) == 0.0

    def test_single_drawdown_yields_positive_ulcer(self) -> None:
        """100 → 90 → 100: 50% of bars have a 10% DD, the rest 0."""
        # peaks: 100, 100, 100  →  dd_pct: 0, 10, 0  →  rms = sqrt((0+100+0)/3) ≈ 5.77
        result = compute_ulcer_index(_curve([100, 90, 100]))
        assert result == pytest.approx(math.sqrt(100.0 / 3.0), rel=1e-4)

    def test_longer_drawdown_yields_higher_ulcer(self) -> None:
        """Two strategies with the same max-DD but different durations
        have different Ulcer scores — that is the whole point."""
        short_dd = compute_ulcer_index(_curve([100, 90, 100, 100]))
        long_dd = compute_ulcer_index(_curve([100, 90, 90, 90]))
        assert long_dd > short_dd

    def test_ulcer_is_non_negative(self) -> None:
        result = compute_ulcer_index(_curve([100, 50, 25, 12.5]))
        assert result >= 0.0


# ---------------------------------------------------------------------------
# QT-005b: Omega Ratio
# ---------------------------------------------------------------------------


class TestOmegaRatio:
    def test_empty_returns_zero(self) -> None:
        assert compute_omega_ratio([]) == 0.0

    def test_all_gains_yields_infinity(self) -> None:
        assert math.isinf(compute_omega_ratio([0.01, 0.02, 0.03]))

    def test_all_losses_yields_zero(self) -> None:
        """No gains above threshold → omega is 0 (downside-only)."""
        assert compute_omega_ratio([-0.01, -0.02, -0.03]) == 0.0

    def test_balanced_returns_yields_unity(self) -> None:
        """Equal-magnitude gains and losses → omega = 1."""
        assert compute_omega_ratio([0.02, -0.02]) == pytest.approx(1.0)

    def test_more_upside_than_downside_yields_greater_than_one(self) -> None:
        assert compute_omega_ratio([0.03, -0.01]) == pytest.approx(3.0)

    def test_threshold_shifts_baseline(self) -> None:
        """Returns of +5% and +1% with threshold 2% → gains=3, losses=1."""
        assert compute_omega_ratio([0.05, 0.01], threshold=0.02) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# QT-005c: Conditional VaR (Expected Shortfall)
# ---------------------------------------------------------------------------


class TestCVaR:
    def test_empty_returns_zero(self) -> None:
        assert compute_cvar([]) == 0.0

    def test_uniform_returns_yields_worst_observation(self) -> None:
        """With 20 obs and 95% confidence, tail size = 1 → CVaR = min."""
        returns = [0.01 * i for i in range(-10, 10)]  # -0.10 .. 0.09
        # Worst observation
        assert compute_cvar(returns, confidence=0.95) == pytest.approx(-0.10)

    def test_lower_confidence_includes_more_tail(self) -> None:
        """80% CVaR averages a deeper tail than 95% CVaR — should be
        less extreme (closer to zero) because mean of more obs."""
        returns = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.05, 0.10]
        cvar_95 = compute_cvar(returns, confidence=0.95)
        cvar_80 = compute_cvar(returns, confidence=0.80)
        assert cvar_95 <= cvar_80  # 95% is the deeper / worse tail

    def test_invalid_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            compute_cvar([0.01], confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            compute_cvar([0.01], confidence=1.0)

    def test_cvar_captures_left_tail_mass(self) -> None:
        """Two distributions with same stdev but different tail mass —
        CVaR must distinguish them."""
        thin_tail = [-0.02] * 5 + [0.02] * 5
        fat_tail = [-0.10] + [-0.01] * 4 + [0.02] * 5
        # 80% CVaR → tail_n = 2
        cvar_thin = compute_cvar(thin_tail, confidence=0.80)
        cvar_fat = compute_cvar(fat_tail, confidence=0.80)
        assert cvar_fat < cvar_thin  # fat-tail more punishing


# ---------------------------------------------------------------------------
# QT-006: Exposure-adjusted Sharpe
# ---------------------------------------------------------------------------


class TestExposureAdjustedSharpe:
    def test_full_exposure_unchanged(self) -> None:
        """Strategy in market 100% of bars: adjustment is identity."""
        assert compute_exposure_adjusted_sharpe(1.0, 1.0) == pytest.approx(1.0)

    def test_quarter_exposure_doubles_sharpe(self) -> None:
        """25% exposure: Sharpe / sqrt(0.25) = Sharpe / 0.5 = 2 * Sharpe."""
        assert compute_exposure_adjusted_sharpe(1.0, 0.25) == pytest.approx(2.0)

    def test_zero_exposure_returns_input(self) -> None:
        """Degenerate: no exposure → return raw Sharpe to avoid div-by-zero."""
        assert compute_exposure_adjusted_sharpe(1.5, 0.0) == 1.5

    def test_negative_exposure_treated_as_zero(self) -> None:
        assert compute_exposure_adjusted_sharpe(1.5, -0.1) == 1.5

    def test_negative_sharpe_amplified_correctly(self) -> None:
        """Adjustment must flip sign correctly for negative Sharpe."""
        assert compute_exposure_adjusted_sharpe(-1.0, 0.25) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# QT-012: Buy-and-hold benchmark
# ---------------------------------------------------------------------------


class TestBuyAndHoldReturn:
    def test_empty_input_returns_zero(self) -> None:
        assert compute_buy_and_hold_return([]) == 0.0

    def test_single_price_returns_zero(self) -> None:
        assert compute_buy_and_hold_return([100.0]) == 0.0

    def test_50_pct_increase(self) -> None:
        assert compute_buy_and_hold_return([100.0, 150.0]) == pytest.approx(0.5)

    def test_50_pct_decrease(self) -> None:
        assert compute_buy_and_hold_return([100.0, 50.0]) == pytest.approx(-0.5)

    def test_uses_only_first_and_last_prices(self) -> None:
        """Middle prices irrelevant — BHLD = (end-start)/start regardless."""
        result_a = compute_buy_and_hold_return([100.0, 200.0, 50.0, 110.0])
        result_b = compute_buy_and_hold_return([100.0, 110.0])
        assert result_a == pytest.approx(result_b)

    def test_zero_start_price_yields_zero(self) -> None:
        """Defensive: division by zero must not crash."""
        assert compute_buy_and_hold_return([0.0, 100.0]) == 0.0


# ---------------------------------------------------------------------------
# QT-012b: Alpha + Beta
# ---------------------------------------------------------------------------


class TestAlphaBeta:
    def test_mismatched_lengths_yields_zeros(self) -> None:
        assert compute_alpha_beta([0.01, 0.02], [0.01]) == (0.0, 0.0)

    def test_empty_inputs_yields_zeros(self) -> None:
        assert compute_alpha_beta([], []) == (0.0, 0.0)

    def test_perfect_tracking_yields_unit_beta(self) -> None:
        """Strategy == benchmark → beta = 1, alpha = 0."""
        bench = [0.01, -0.02, 0.03, -0.01]
        strat = bench
        alpha, beta = compute_alpha_beta(strat, bench)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.0)

    def test_double_amplitude_yields_beta_two(self) -> None:
        bench = [0.01, -0.02, 0.03, -0.01]
        strat = [r * 2.0 for r in bench]
        _alpha, beta = compute_alpha_beta(strat, bench)
        assert beta == pytest.approx(2.0)

    def test_constant_alpha_above_benchmark(self) -> None:
        """Strategy = benchmark + 0.005 per period → beta=1, alpha=0.005."""
        bench = [0.01, -0.02, 0.03, -0.01]
        strat = [r + 0.005 for r in bench]
        alpha, beta = compute_alpha_beta(strat, bench)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.005)

    def test_flat_benchmark_yields_zero_beta(self) -> None:
        """Var(benchmark) = 0 → beta = 0 by definition."""
        bench = [0.0, 0.0, 0.0]
        strat = [0.01, 0.02, 0.03]
        alpha, beta = compute_alpha_beta(strat, bench)
        assert beta == 0.0
        assert alpha == pytest.approx(0.02)  # mean of strat


# ---------------------------------------------------------------------------
# QT-012c: Information Ratio
# ---------------------------------------------------------------------------


class TestInformationRatio:
    def test_mismatched_lengths_yields_zero(self) -> None:
        assert compute_information_ratio([0.01], [0.01, 0.02]) == 0.0

    def test_empty_yields_zero(self) -> None:
        assert compute_information_ratio([], []) == 0.0

    def test_perfect_tracking_yields_zero(self) -> None:
        """Strategy == benchmark → active return is zero everywhere → IR = 0."""
        rs = [0.01, -0.02, 0.03]
        assert compute_information_ratio(rs, rs) == 0.0

    def test_truly_constant_active_return_yields_zero(self) -> None:
        """Active return identical (exactly zero variance) → IR is
        undefined; guard returns 0.0."""
        # Use float-friendly values to ensure exact-zero tracking error.
        bench = [0.0, 0.0, 0.0, 0.0]
        strat = [0.005, 0.005, 0.005, 0.005]
        assert compute_information_ratio(strat, bench) == 0.0

    def test_near_constant_outperformance_yields_large_ir(self) -> None:
        """Near-constant active return with float-level noise → variance
        is tiny but non-zero, so IR is finite but very large."""
        bench = [0.01, -0.02, 0.03, -0.01]
        strat = [r + 0.005 for r in bench]
        result = compute_information_ratio(strat, bench)
        # Tiny tracking error → result must be a large positive number,
        # not the zero-variance guard (which would mask a real signal).
        assert result > 1000.0

    def test_positive_ir_when_strategy_beats_benchmark_on_average(self) -> None:
        bench = [0.01, 0.01, 0.01, 0.01]
        strat = [0.02, 0.01, 0.02, 0.01]  # mean active = +0.005, some variance
        result = compute_information_ratio(strat, bench)
        assert result > 0.0
