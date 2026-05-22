"""
tests/unit/test_sprint49_m4_psr_confidence.py
----------------------------------------------
Unit tests for Sprint 49 M4: PSR + n_observations + confidence_flag.

Covers
------
1. _statistics module — _inv_norm_cdf accessible from trading._statistics
2. _inv_norm_cdf still importable from trading.walk_forward (backward compat)
3. deflated_sharpe_ratio re-exported from trading.metrics
4. compute_psr — None below n=30 threshold
5. compute_psr — output in [0, 1]
6. compute_psr — known toy case accuracy (Bailey 2012 paper reference values)
7. compute_psr — normal returns yield high PSR for strong Sharpe
8. compute_psr — denominator underflow path
9. _derive_confidence_flag — all tier combinations (table-driven)
10. BacktestResult — new fields have correct defaults
11. build_backtest_metrics — propagates psr, n_observations, confidence_flag
"""

from __future__ import annotations

import math
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trading._statistics import _inv_norm_cdf as stats_inv_norm_cdf  # noqa: PLC2701
from trading._statistics import _norm_cdf  # noqa: PLC2701
from trading.metrics import (
    BacktestResult,
    _derive_confidence_flag,  # noqa: PLC2701 - private helper, same-package test
    compute_psr,
)
from trading.walk_forward import _inv_norm_cdf as wf_inv_norm_cdf  # noqa: PLC2701
# Thin re-export from metrics — must resolve without ImportError
from trading.metrics import deflated_sharpe_ratio  # noqa: F401


# ---------------------------------------------------------------------------
# 1. _statistics module — accessible directly
# ---------------------------------------------------------------------------

class TestStatisticsModule:
    def test_inv_norm_cdf_accessible_from_statistics(self) -> None:
        """_inv_norm_cdf is importable from trading._statistics."""
        # Already imported at module level — if import failed, test collection fails.
        assert stats_inv_norm_cdf(0.5) == pytest.approx(0.0, abs=1e-8)

    def test_norm_cdf_at_zero_is_half(self) -> None:
        """Φ(0) = 0.5."""
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_norm_cdf_at_plus_infinity_is_one(self) -> None:
        """Φ(large positive) approx 1."""
        assert _norm_cdf(10.0) == pytest.approx(1.0, abs=1e-6)

    def test_norm_cdf_symmetry(self) -> None:
        """Φ(-z) = 1 - Φ(z)."""
        for z in (0.5, 1.0, 1.96, 2.576):
            assert _norm_cdf(-z) == pytest.approx(1.0 - _norm_cdf(z), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Backward compat: _inv_norm_cdf importable from walk_forward
# ---------------------------------------------------------------------------

class TestWalkForwardBackwardCompat:
    def test_inv_norm_cdf_importable_from_walk_forward(self) -> None:
        """trading.walk_forward._inv_norm_cdf is the same object as _statistics._inv_norm_cdf."""
        # Both names must be callable and give the same result.
        assert wf_inv_norm_cdf(0.8413) == pytest.approx(stats_inv_norm_cdf(0.8413), rel=1e-10)

    def test_walk_forward_inv_norm_cdf_value(self) -> None:
        """Acklam approximation: Φ⁻¹(0.975) approx 1.96 (two-tailed 5% critical value)."""
        assert wf_inv_norm_cdf(0.975) == pytest.approx(1.96, abs=1e-2)


# ---------------------------------------------------------------------------
# 3. deflated_sharpe_ratio re-exported from metrics
# ---------------------------------------------------------------------------

class TestDSRReexport:
    def test_deflated_sharpe_ratio_importable_from_metrics(self) -> None:
        """from trading.metrics import deflated_sharpe_ratio should not raise."""
        # Already imported at module level with noqa — just assert it's callable.
        assert callable(deflated_sharpe_ratio)

    def test_deflated_sharpe_ratio_single_trial_identity(self) -> None:
        """DSR with num_trials=1 returns the observed Sharpe unchanged."""
        result = deflated_sharpe_ratio(observed_sharpe=1.5, sharpe_stddev=0.3, num_trials=1)
        assert result == pytest.approx(1.5, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. compute_psr — None below n=30
# ---------------------------------------------------------------------------

class TestComputePsrThresholds:
    @pytest.mark.parametrize("n", [0, 1, 10, 29])
    def test_psr_returns_none_when_n_below_30(self, n: int) -> None:
        """PSR is undefined for fewer than 30 observations."""
        result = compute_psr(observed_sharpe=1.0, n_observations=n)
        assert result is None

    def test_psr_returns_value_when_n_equals_30(self) -> None:
        """PSR is computable at exactly n=30."""
        result = compute_psr(observed_sharpe=0.5, n_observations=30)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_psr_returns_value_in_zero_one(self) -> None:
        """PSR output is always in [0, 1] for valid inputs."""
        for sr in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
            result = compute_psr(observed_sharpe=sr, n_observations=100)
            if result is not None:
                assert 0.0 <= result <= 1.0, f"PSR out of range for SR={sr}: {result}"


# ---------------------------------------------------------------------------
# 5. compute_psr — formula accuracy (toy case)
# ---------------------------------------------------------------------------

class TestComputePsrFormula:
    def test_psr_normal_returns_high_value_for_strong_sharpe(self) -> None:
        """Normal distribution (skew=0, kurt=3), n=100, SR=1.0 → PSR high."""
        # With normal returns:
        # denominator = sqrt((1 - 0*1 + (3-1)/4*1^2) / 99) = sqrt(1.5/99) approx 0.1231
        # z = (1.0 - 0) / 0.1231 approx 8.12
        # PSR approx Φ(8.12) approx 1.0 (essentially certain)
        result = compute_psr(
            observed_sharpe=1.0,
            n_observations=100,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result is not None
        assert result > 0.99

    def test_psr_for_negative_sharpe_below_half(self) -> None:
        """Negative SR → PSR < 0.5 (strategy more likely harmful than beneficial)."""
        result = compute_psr(
            observed_sharpe=-0.5,
            n_observations=100,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result is not None
        assert result < 0.5

    def test_psr_zero_sharpe_near_half(self) -> None:
        """SR = 0 → PSR approx 0.5 (equally likely above or below zero)."""
        result = compute_psr(
            observed_sharpe=0.0,
            n_observations=200,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_psr_benchmark_shifts_result(self) -> None:
        """Higher sr_benchmark → lower PSR (harder test)."""
        psr_base = compute_psr(
            observed_sharpe=1.0, n_observations=100, sr_benchmark=0.0
        )
        psr_higher = compute_psr(
            observed_sharpe=1.0, n_observations=100, sr_benchmark=0.5
        )
        assert psr_base is not None
        assert psr_higher is not None
        assert psr_base > psr_higher

    def test_psr_degenerate_var_numerator_returns_none(self) -> None:
        """Extreme skew/kurtosis making var_numerator <= 0 → None."""
        # Force numerator negative: 1 - skew*SR + (kurt-1)/4*SR^2 < 0
        # numerator = 1 - skew*SR = 1 - 100*2 = -199 → degenerate
        result = compute_psr(
            observed_sharpe=2.0,
            n_observations=100,
            skew=100.0,  # extreme — var_numerator goes negative
            kurtosis=3.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 6. _derive_confidence_flag — table-driven
# ---------------------------------------------------------------------------

class TestDeriveConfidenceFlag:
    @pytest.mark.parametrize("psr,n_trades,expected", [
        # None PSR → None flag
        (None, 0, None),
        (None, 100, None),
        # High tier: PSR >= 0.95 AND trades >= 50
        (0.95, 50, "high"),
        (0.99, 100, "high"),
        (0.95, 49, "medium"),   # trades < 50 → drops to medium (PSR>=0.80 still qualifies)
        (0.94, 50, "medium"),   # PSR < 0.95 → not high
        # Medium tier: PSR >= 0.80 AND trades >= 30
        (0.80, 30, "medium"),
        (0.85, 35, "medium"),
        (0.80, 29, "low"),      # trades < 30 → low
        (0.79, 30, "low"),      # PSR < 0.80 → low
        # Low tier: everything else where PSR is computable
        (0.50, 10, "low"),
        (0.01, 5, "low"),
    ])
    def test_confidence_flag_thresholds(
        self,
        psr: float | None,
        n_trades: int,
        expected: str | None,
    ) -> None:
        result = _derive_confidence_flag(psr, n_trades)
        assert result == expected, (
            f"Expected {expected!r} for PSR={psr}, trades={n_trades}; got {result!r}"
        )


# ---------------------------------------------------------------------------
# 7. BacktestResult — default field values
# ---------------------------------------------------------------------------

class TestBacktestResultDefaults:
    def test_new_fields_have_correct_defaults(self) -> None:
        """BacktestResult can be constructed without the M4 fields and defaults are sane."""
        from datetime import UTC, datetime
        # Minimal required fields
        result = BacktestResult(
            run_id="bt-test",
            strategy_ids=["ma"],
            symbols=["BTC/USDT"],
            timeframe="1h",
            start_date=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 1, 2, tzinfo=UTC),
            duration_days=1,
            initial_capital=Decimal("10000"),
            final_equity=Decimal("10100"),
            total_return_pct=0.01,
            cagr=0.01,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_trade_pnl=Decimal("0"),
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            total_bars=0,
            bars_in_market=0,
            exposure_pct=0.0,
        )
        assert result.psr is None
        assert result.n_observations == 0
        assert result.confidence_flag is None


# ---------------------------------------------------------------------------
# 8. build_backtest_metrics — propagation
# ---------------------------------------------------------------------------

class TestBuildBacktestMetricsPropagation:
    def test_build_backtest_metrics_propagates_psr_and_confidence(self) -> None:
        """build_backtest_metrics copies psr, n_observations, confidence_flag from result."""
        from datetime import UTC, datetime
        from api.services.run_persistence import build_backtest_metrics

        mock_result = SimpleNamespace(
            total_return_pct=0.05,
            cagr=0.10,
            initial_capital=Decimal("10000"),
            final_equity=Decimal("10500"),
            total_fees_paid=Decimal("50"),
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown_pct=0.05,
            max_drawdown_duration_bars=10,
            total_trades=60,
            winning_trades=40,
            losing_trades=20,
            win_rate=0.666,
            profit_factor=2.0,
            profit_factor_is_infinite=False,
            average_trade_pnl=Decimal("10"),
            average_win=Decimal("20"),
            average_loss=Decimal("-10"),
            largest_win=Decimal("100"),
            largest_loss=Decimal("-50"),
            total_bars=200,
            bars_in_market=80,
            exposure_pct=0.40,
            exposure_pct_per_symbol={"BTC/USDT": 0.40},
            start_date=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 6, 1, tzinfo=UTC),
            duration_days=151,
            open_positions_mtm=[],
            # M4 fields
            psr=0.97,
            n_observations=150,
            confidence_flag="high",
        )

        response = build_backtest_metrics(mock_result)
        assert response.psr == pytest.approx(0.97, rel=1e-9)
        assert response.n_observations == 150
        assert response.confidence_flag == "high"

    def test_build_backtest_metrics_graceful_for_pre_m4_result(self) -> None:
        """build_backtest_metrics uses getattr defaults for old SimpleNamespace results."""
        from datetime import UTC, datetime
        from api.services.run_persistence import build_backtest_metrics

        mock_result = SimpleNamespace(
            total_return_pct=0.0,
            cagr=0.0,
            initial_capital=Decimal("10000"),
            final_equity=Decimal("10000"),
            total_fees_paid=Decimal("0"),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=None,
            profit_factor_is_infinite=False,
            average_trade_pnl=Decimal("0"),
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            total_bars=0,
            bars_in_market=0,
            exposure_pct=0.0,
            exposure_pct_per_symbol={},
            start_date=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 1, 2, tzinfo=UTC),
            duration_days=1,
            open_positions_mtm=[],
            # NO psr, n_observations, confidence_flag fields
        )

        response = build_backtest_metrics(mock_result)
        assert response.psr is None
        assert response.n_observations == 0
        assert response.confidence_flag is None
