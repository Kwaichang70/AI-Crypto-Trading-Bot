"""
tests/unit/test_metrics.py
---------------------------
Unit tests for all metric functions in packages/trading/metrics.py.

Every function under test is a pure function — no I/O, no state.
Tests are grouped by function name and cover:

- Happy path (representative numeric input)
- Degenerate / edge cases (empty inputs, zeros, single elements)
- Boundary conditions (division by zero guards, negative values)
- Known-good reference values for financial formulae

All monetary inputs use Decimal for precision; ratio outputs are float.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide, TimeFrame
from trading.metrics import (
    EquityCurvePoint,
    compute_calmar,
    compute_cagr,
    compute_exposure,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_profit_factor,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_trade_statistics,
)
from trading.models import TradeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_equity_curve(values: list[float]) -> list[EquityCurvePoint]:
    """
    Build an EquityCurvePoint sequence from a list of equity values.

    Timestamps are evenly spaced starting from a fixed UTC point.
    """
    base_ts = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        EquityCurvePoint(
            timestamp=base_ts.replace(hour=i % 24, day=(i // 24) + 1),
            equity=Decimal(str(v)),
        )
        for i, v in enumerate(values)
    ]


def _make_trade(realised_pnl: float, *, symbol: str = "BTC/USDT") -> TradeResult:
    """Build a minimal TradeResult with the given realised PnL."""
    now = datetime.now(tz=UTC)
    return TradeResult(
        run_id="test-run",
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=Decimal("50000"),
        exit_price=Decimal("51000") if realised_pnl >= 0 else Decimal("49000"),
        quantity=Decimal("0.01"),
        realised_pnl=Decimal(str(realised_pnl)),
        total_fees=Decimal("0.5"),
        entry_at=now,
        exit_at=now,
        strategy_id="test",
    )


# ===========================================================================
# compute_cagr
# ===========================================================================


class TestComputeCagr:
    """Tests for compute_cagr(initial, final, days)."""

    def test_ten_percent_annual_return(self) -> None:
        """
        10% total return over exactly 365 days should yield ~10% CAGR.
        CAGR = (11000/10000)^(365.25/365) - 1 ≈ 0.1004
        """
        result = compute_cagr(Decimal("10000"), Decimal("11000"), 365)
        assert abs(result - 0.10) < 0.002  # within 0.2% of 10%

    def test_flat_portfolio_zero_return(self) -> None:
        """No change in equity yields CAGR = 0.0."""
        result = compute_cagr(Decimal("10000"), Decimal("10000"), 365)
        assert abs(result) < 1e-9

    def test_zero_days_returns_zero(self) -> None:
        """Zero-day period is degenerate — returns 0.0 by convention."""
        result = compute_cagr(Decimal("10000"), Decimal("12000"), 0)
        assert result == 0.0

    def test_zero_initial_returns_zero(self) -> None:
        """Initial capital of 0 is invalid — returns 0.0 guard."""
        result = compute_cagr(Decimal("0"), Decimal("10000"), 365)
        assert result == 0.0

    def test_negative_initial_returns_zero(self) -> None:
        """Negative initial capital returns 0.0 guard."""
        result = compute_cagr(Decimal("-1000"), Decimal("500"), 180)
        assert result == 0.0

    def test_negative_final_returns_negative_one(self) -> None:
        """
        If final equity is negative (theoretical), ratio is non-positive
        and CAGR returns -1.0 (total loss sentinel).
        """
        result = compute_cagr(Decimal("10000"), Decimal("-1"), 365)
        assert result == -1.0

    def test_negative_final_near_zero_returns_negative_one(self) -> None:
        """Final equity of 0 yields ratio=0 which cannot be exponentiated — returns -1.0."""
        result = compute_cagr(Decimal("10000"), Decimal("0"), 365)
        assert result == -1.0

    def test_double_in_one_year(self) -> None:
        """Doubling in one year gives CAGR ≈ 100%."""
        result = compute_cagr(Decimal("10000"), Decimal("20000"), 365)
        assert abs(result - 1.0) < 0.005  # ≈ 100%

    def test_cagr_is_float(self) -> None:
        """Return type is always float."""
        result = compute_cagr(Decimal("10000"), Decimal("11000"), 365)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "initial,final,days,expected_approx",
        [
            (Decimal("10000"), Decimal("11000"), 365, 0.10),
            (Decimal("10000"), Decimal("12100"), 730, 0.10),  # 10% CAGR over 2 years
            (Decimal("10000"), Decimal("10000"), 365, 0.0),
        ],
    )
    def test_cagr_parametrized(
        self,
        initial: Decimal,
        final: Decimal,
        days: int,
        expected_approx: float,
    ) -> None:
        """Parametrized CAGR checks for known growth scenarios."""
        result = compute_cagr(initial, final, days)
        assert abs(result - expected_approx) < 0.005


# ===========================================================================
# compute_sharpe
# ===========================================================================


class TestComputeSharpe:
    """Tests for compute_sharpe(returns, periods_per_year)."""

    def test_constant_returns_zero_std_returns_zero(self) -> None:
        """Constant returns have zero std deviation — Sharpe returns 0.0."""
        returns = [0.01] * 50
        result = compute_sharpe(returns, periods_per_year=252.0)
        assert result == 0.0

    def test_single_return_too_few_observations(self) -> None:
        """Fewer than 2 observations returns 0.0 (cannot compute std)."""
        result = compute_sharpe([0.01], periods_per_year=252.0)
        assert result == 0.0

    def test_empty_returns_returns_zero(self) -> None:
        """Empty returns list returns 0.0."""
        result = compute_sharpe([], periods_per_year=252.0)
        assert result == 0.0

    def test_zero_periods_per_year_returns_zero(self) -> None:
        """periods_per_year of 0 or negative returns 0.0."""
        result = compute_sharpe([0.01, 0.02, -0.005], periods_per_year=0.0)
        assert result == 0.0

    def test_positive_mean_positive_sharpe(self) -> None:
        """Positively-biased returns produce a positive Sharpe ratio."""
        returns = [0.01, 0.02, 0.015, 0.012, 0.008, 0.011, 0.009]
        result = compute_sharpe(returns, periods_per_year=252.0)
        assert result > 0.0

    def test_negative_mean_negative_sharpe(self) -> None:
        """Negatively-biased returns produce a negative Sharpe ratio."""
        returns = [-0.01, -0.02, -0.015, -0.012, -0.008, -0.011, -0.009]
        result = compute_sharpe(returns, periods_per_year=252.0)
        assert result < 0.0

    def test_sharpe_is_float(self) -> None:
        """Return type is always float."""
        result = compute_sharpe([0.01, -0.01, 0.005], periods_per_year=252.0)
        assert isinstance(result, float)

    def test_sharpe_annualisation_scaling(self) -> None:
        """
        For a fixed mean/std ratio, doubling periods_per_year increases
        Sharpe by sqrt(2).
        """
        returns = [0.001, 0.002, -0.001, 0.0015, 0.0005, -0.0005, 0.0008, 0.0012]
        sharpe_252 = compute_sharpe(returns, periods_per_year=252.0)
        sharpe_1008 = compute_sharpe(returns, periods_per_year=1008.0)
        assert abs(sharpe_1008 / sharpe_252 - 2.0) < 0.01  # sqrt(1008/252) = 2


# ===========================================================================
# compute_sortino
# ===========================================================================


class TestComputeSortino:
    """Tests for compute_sortino(returns, periods_per_year, target_return)."""

    def test_all_positive_returns_infinite_sortino(self) -> None:
        """All returns above target → downside dev = 0 → Sortino = +inf."""
        returns = [0.01, 0.02, 0.015, 0.008, 0.005]
        result = compute_sortino(returns, periods_per_year=252.0)
        assert math.isinf(result) and result > 0

    def test_mixed_returns_finite_positive(self) -> None:
        """Mixed returns with positive mean yield a finite positive Sortino."""
        returns = [0.02, -0.01, 0.015, -0.005, 0.025, -0.003, 0.018]
        result = compute_sortino(returns, periods_per_year=252.0)
        assert math.isfinite(result)
        assert result > 0

    def test_all_negative_returns_negative_sortino(self) -> None:
        """All negative returns produce a negative Sortino ratio."""
        returns = [-0.01, -0.02, -0.015, -0.008, -0.005]
        result = compute_sortino(returns, periods_per_year=252.0)
        assert result < 0 or result == 0.0  # may clamp in extreme cases

    def test_single_return_returns_zero(self) -> None:
        """Fewer than 2 observations returns 0.0."""
        result = compute_sortino([0.01], periods_per_year=252.0)
        assert result == 0.0

    def test_empty_returns_zero(self) -> None:
        """Empty list returns 0.0."""
        result = compute_sortino([], periods_per_year=252.0)
        assert result == 0.0

    def test_custom_target_return(self) -> None:
        """
        With target_return = 0.01, returns above 0.01 don't contribute
        to downside. A set of returns all above 0.01 yields +inf.
        """
        returns = [0.02, 0.015, 0.03, 0.025]
        result = compute_sortino(returns, periods_per_year=252.0, target_return=0.01)
        assert math.isinf(result) and result > 0

    def test_sortino_is_float(self) -> None:
        """Return type is always float (including inf)."""
        result = compute_sortino([0.01, -0.005, 0.02], periods_per_year=252.0)
        assert isinstance(result, float)

    def test_sortino_greater_than_sharpe_for_asymmetric_returns(self) -> None:
        """
        For positively skewed returns (few large losses, many small gains),
        Sortino should exceed Sharpe because only downside is penalised.
        """
        returns = [0.02, 0.015, 0.025, 0.01, -0.05, 0.018, 0.022, 0.013]
        sharpe = compute_sharpe(returns, periods_per_year=252.0)
        sortino = compute_sortino(returns, periods_per_year=252.0)
        assert sortino > sharpe


# ===========================================================================
# compute_calmar
# ===========================================================================


class TestComputeCalmar:
    """Tests for compute_calmar(cagr, max_drawdown)."""

    def test_calmar_normal_case(self) -> None:
        """Calmar = CAGR / |max_drawdown| for normal inputs."""
        result = compute_calmar(cagr=0.20, max_drawdown=0.10)
        assert abs(result - 2.0) < 1e-9

    def test_calmar_zero_drawdown_returns_zero(self) -> None:
        """Zero drawdown guard: returns 0.0 (no division by zero)."""
        result = compute_calmar(cagr=0.15, max_drawdown=0.0)
        assert result == 0.0

    def test_calmar_near_zero_drawdown_returns_zero(self) -> None:
        """Near-zero drawdown (below 1e-12) still returns 0.0."""
        result = compute_calmar(cagr=0.15, max_drawdown=1e-15)
        assert result == 0.0

    def test_calmar_negative_cagr(self) -> None:
        """Negative CAGR with positive drawdown yields negative Calmar."""
        result = compute_calmar(cagr=-0.05, max_drawdown=0.10)
        assert result < 0

    def test_calmar_uses_absolute_drawdown(self) -> None:
        """Whether drawdown is passed as positive or negative, |dd| is used."""
        positive_dd = compute_calmar(cagr=0.20, max_drawdown=0.10)
        negative_dd = compute_calmar(cagr=0.20, max_drawdown=-0.10)
        assert abs(positive_dd - negative_dd) < 1e-9

    def test_calmar_is_float(self) -> None:
        """Return type is float."""
        result = compute_calmar(0.15, 0.10)
        assert isinstance(result, float)


# ===========================================================================
# compute_profit_factor
# ===========================================================================


class TestComputeProfitFactor:
    """Tests for compute_profit_factor(trades)."""

    def test_empty_trades_returns_zero(self) -> None:
        """No trades returns 0.0 by convention."""
        result = compute_profit_factor([])
        assert result == 0.0

    def test_all_winning_trades_returns_inf(self) -> None:
        """All winning trades with no losers returns float('inf')."""
        trades = [_make_trade(100.0), _make_trade(200.0), _make_trade(50.0)]
        result = compute_profit_factor(trades)
        assert math.isinf(result) and result > 0

    def test_all_losing_trades_returns_zero(self) -> None:
        """All losing trades with no winners returns 0.0."""
        trades = [_make_trade(-50.0), _make_trade(-30.0), _make_trade(-20.0)]
        result = compute_profit_factor(trades)
        assert result == 0.0

    def test_normal_mixed_trades(self) -> None:
        """
        100 USDT gross profit / 50 USDT gross loss = profit factor 2.0.
        """
        trades = [_make_trade(100.0), _make_trade(-50.0)]
        result = compute_profit_factor(trades)
        assert abs(result - 2.0) < 1e-9

    def test_break_even_trade_not_counted_as_win_or_loss(self) -> None:
        """
        A zero-PnL trade contributes nothing to gross profit or gross loss.
        This verifies the FIX-05 equivalent logic in profit_factor:
        gross_loss remains 50 (not 0) so profit_factor = 100/50 = 2.0.
        """
        trades = [
            _make_trade(100.0),
            _make_trade(-50.0),
            _make_trade(0.0),  # break-even
        ]
        result = compute_profit_factor(trades)
        assert abs(result - 2.0) < 1e-9

    def test_profit_factor_is_float(self) -> None:
        """Return type is always float."""
        result = compute_profit_factor([_make_trade(10.0), _make_trade(-5.0)])
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "wins,losses,expected",
        [
            ([100.0, 200.0], [-50.0, -100.0], 2.0),  # 300/150
            ([50.0], [-50.0], 1.0),  # exactly breakeven system
            ([10.0], [-100.0], 0.1),
        ],
    )
    def test_profit_factor_parametrized(
        self,
        wins: list[float],
        losses: list[float],
        expected: float,
    ) -> None:
        """Profit factor = gross_profit / gross_loss across known inputs."""
        trades = [_make_trade(p) for p in wins + losses]
        result = compute_profit_factor(trades)
        assert abs(result - expected) < 1e-9


# ===========================================================================
# compute_max_drawdown
# ===========================================================================


class TestComputeMaxDrawdown:
    """Tests for compute_max_drawdown(equity_curve)."""

    def test_empty_curve_returns_zero(self) -> None:
        """Empty equity curve returns 0.0."""
        result = compute_max_drawdown([])
        assert result == 0.0

    def test_single_point_returns_zero(self) -> None:
        """Single point — no drawdown possible."""
        result = compute_max_drawdown(_make_equity_curve([10000.0]))
        assert result == 0.0

    def test_monotonically_rising_curve_zero_drawdown(self) -> None:
        """Equity that never drops has 0% drawdown."""
        curve = _make_equity_curve([10000, 10100, 10200, 10300, 10400])
        result = compute_max_drawdown(curve)
        assert result == 0.0

    def test_simple_peak_valley_pattern(self) -> None:
        """
        Peak: 10_000, valley: 8_000 → max drawdown = 20%.
        """
        curve = _make_equity_curve([10000, 9500, 8000, 8500, 10000])
        result = compute_max_drawdown(curve)
        expected = (10000 - 8000) / 10000  # 0.20
        assert abs(result - expected) < 1e-6

    def test_drawdown_at_end_of_run(self) -> None:
        """
        Run ends in drawdown — the final trough should still be captured.
        Peak: 10_000, final: 7_000 → max drawdown = 30%.
        """
        curve = _make_equity_curve([10000, 9000, 8000, 7000])
        result = compute_max_drawdown(curve)
        expected = (10000 - 7000) / 10000  # 0.30
        assert abs(result - expected) < 1e-6

    def test_multiple_peaks_worst_case_selected(self) -> None:
        """
        Multiple drawdown episodes — only the worst is reported.
        Peak-1: 10_000 → 9_000 = 10%. Peak-2: 11_000 → 8_000 = 27.3%.
        """
        curve = _make_equity_curve([10000, 9000, 11000, 8000])
        result = compute_max_drawdown(curve)
        expected = (11000 - 8000) / 11000
        assert abs(result - expected) < 1e-6

    def test_max_drawdown_is_float(self) -> None:
        """Return type is float."""
        result = compute_max_drawdown(_make_equity_curve([10000, 9000]))
        assert isinstance(result, float)

    def test_max_drawdown_non_negative(self) -> None:
        """Max drawdown is always non-negative."""
        curve = _make_equity_curve([10000, 10050, 10200, 10300])
        result = compute_max_drawdown(curve)
        assert result >= 0.0


# ===========================================================================
# compute_max_drawdown_duration
# ===========================================================================


class TestComputeMaxDrawdownDuration:
    """Tests for compute_max_drawdown_duration(equity_curve)."""

    def test_empty_curve_returns_zero(self) -> None:
        """Empty equity curve returns 0."""
        result = compute_max_drawdown_duration([])
        assert result == 0

    def test_single_point_returns_zero(self) -> None:
        """Single point has no drawdown period."""
        result = compute_max_drawdown_duration(_make_equity_curve([10000.0]))
        assert result == 0

    def test_no_drawdown_returns_zero(self) -> None:
        """Monotonically rising equity has zero duration."""
        curve = _make_equity_curve([10000, 10100, 10200, 10300])
        result = compute_max_drawdown_duration(curve)
        assert result == 0

    def test_drawdown_recovered_mid_run(self) -> None:
        """
        Drawdown of 2 bars (indices 1, 2) followed by recovery at bar 3.
        Duration = 2.
        """
        # Peak at 10000, dip at 9500 (bar 1), 9000 (bar 2), recovery to 10000 (bar 3)
        curve = _make_equity_curve([10000, 9500, 9000, 10000, 10100])
        result = compute_max_drawdown_duration(curve)
        assert result == 2

    def test_drawdown_not_recovered_at_end(self) -> None:
        """
        Run ends while still in drawdown — the open drawdown is still counted.
        """
        # Peak at 10000 (bar 0), bars 1-3 are below peak = 3 bars duration
        curve = _make_equity_curve([10000, 9500, 9200, 9000])
        result = compute_max_drawdown_duration(curve)
        assert result == 3

    def test_longest_drawdown_selected(self) -> None:
        """
        Two drawdown episodes: first is 1 bar, second is 3 bars.
        Result must be 3.
        """
        # Episode 1: 10000 → 9000 → 10000 (1 bar below peak)
        # Episode 2: 10000 → 9500 → 9000 → 8500 → 10000 (3 bars below peak)
        curve = _make_equity_curve([
            10000,  # peak
            9000,   # ep1 bar1
            10000,  # ep1 recovery
            9500,   # ep2 bar1
            9000,   # ep2 bar2
            8500,   # ep2 bar3
            10000,  # ep2 recovery
        ])
        result = compute_max_drawdown_duration(curve)
        assert result == 3

    def test_duration_is_int(self) -> None:
        """Return type is int."""
        result = compute_max_drawdown_duration(_make_equity_curve([10000, 9000, 10000]))
        assert isinstance(result, int)


# ===========================================================================
# compute_exposure
# ===========================================================================


class TestComputeExposure:
    """Tests for compute_exposure(bars_in_market, total_bars)."""

    def test_half_exposure(self) -> None:
        """50 bars in market out of 100 total = 0.5 exposure."""
        result = compute_exposure(50, 100)
        assert abs(result - 0.5) < 1e-9

    def test_zero_exposure(self) -> None:
        """Never in market = 0.0 exposure."""
        result = compute_exposure(0, 100)
        assert result == 0.0

    def test_full_exposure(self) -> None:
        """Always in market = 1.0 exposure."""
        result = compute_exposure(100, 100)
        assert abs(result - 1.0) < 1e-9

    def test_zero_total_bars_returns_zero(self) -> None:
        """Guard against division by zero when total_bars = 0."""
        result = compute_exposure(0, 0)
        assert result == 0.0

    def test_bars_in_market_exceeds_total_clamped(self) -> None:
        """Exposure is clamped to 1.0 even if bars_in_market > total_bars."""
        result = compute_exposure(120, 100)
        assert result == 1.0

    def test_exposure_is_float(self) -> None:
        """Return type is float."""
        result = compute_exposure(50, 100)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "in_market,total,expected",
        [
            (0, 200, 0.0),
            (100, 200, 0.5),
            (200, 200, 1.0),
            (1, 1, 1.0),
        ],
    )
    def test_exposure_parametrized(
        self,
        in_market: int,
        total: int,
        expected: float,
    ) -> None:
        """Parametrized exposure checks across common scenarios."""
        result = compute_exposure(in_market, total)
        assert abs(result - expected) < 1e-9


# ===========================================================================
# compute_returns_from_equity
# ===========================================================================


class TestComputeReturnsFromEquity:
    """Tests for compute_returns_from_equity(equity_curve)."""

    def test_empty_curve_returns_empty(self) -> None:
        """Empty curve returns an empty list."""
        result = compute_returns_from_equity([])
        assert result == []

    def test_single_point_returns_empty(self) -> None:
        """Single point — no period-over-period return possible."""
        result = compute_returns_from_equity(_make_equity_curve([10000.0]))
        assert result == []

    def test_length_is_n_minus_one(self) -> None:
        """Returns list has len(equity_curve) - 1 elements."""
        curve = _make_equity_curve([10000, 10100, 10050, 10200])
        result = compute_returns_from_equity(curve)
        assert len(result) == 3

    def test_positive_growth(self) -> None:
        """10000 → 11000 in one step = +10% return."""
        curve = _make_equity_curve([10000, 11000])
        result = compute_returns_from_equity(curve)
        assert len(result) == 1
        assert abs(result[0] - 0.10) < 1e-9

    def test_decline(self) -> None:
        """10000 → 9000 in one step = -10% return."""
        curve = _make_equity_curve([10000, 9000])
        result = compute_returns_from_equity(curve)
        assert len(result) == 1
        assert abs(result[0] - (-0.10)) < 1e-9

    def test_flat_equity_zero_return(self) -> None:
        """Flat equity across bars produces all-zero returns."""
        curve = _make_equity_curve([10000, 10000, 10000, 10000])
        result = compute_returns_from_equity(curve)
        assert all(abs(r) < 1e-9 for r in result)

    def test_returns_are_floats(self) -> None:
        """All elements of the returned list are float."""
        curve = _make_equity_curve([10000, 10500, 9800])
        result = compute_returns_from_equity(curve)
        assert all(isinstance(r, float) for r in result)

    def test_known_values(self) -> None:
        """
        Manual verification:
          10000 → 10500: +5.0%
          10500 → 9450: -10.0%
          9450  → 9450: 0.0%
        """
        curve = _make_equity_curve([10000, 10500, 9450, 9450])
        result = compute_returns_from_equity(curve)
        assert abs(result[0] - 0.05) < 1e-9
        assert abs(result[1] - (-0.10)) < 1e-9
        assert abs(result[2] - 0.0) < 1e-9


# ===========================================================================
# compute_trade_statistics
# ===========================================================================


class TestComputeTradeStatistics:
    """Tests for the aggregate trade_statistics helper."""

    def test_empty_trades_returns_defaults(self) -> None:
        """No trades → all counters and metrics are zero/default."""
        stats = compute_trade_statistics([])
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0

    def test_win_rate_calculation(self) -> None:
        """3 wins + 1 loss = 75% win rate."""
        trades = [
            _make_trade(100.0),
            _make_trade(50.0),
            _make_trade(80.0),
            _make_trade(-40.0),
        ]
        stats = compute_trade_statistics(trades)
        assert stats.total_trades == 4
        assert stats.winning_trades == 3
        assert stats.losing_trades == 1
        assert abs(stats.win_rate - 0.75) < 1e-9

    def test_break_even_trade_not_win_or_loss(self) -> None:
        """
        FIX-05 regression: a break-even trade (realised_pnl == 0) must NOT
        increment winning_trades or losing_trades.
        """
        trades = [
            _make_trade(100.0),
            _make_trade(-50.0),
            _make_trade(0.0),  # break-even
        ]
        stats = compute_trade_statistics(trades)
        assert stats.total_trades == 3
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        # win_rate = 1/3
        assert abs(stats.win_rate - (1 / 3)) < 1e-6

    def test_largest_win_and_loss(self) -> None:
        """Largest win and largest loss are correctly identified."""
        trades = [
            _make_trade(200.0),
            _make_trade(50.0),
            _make_trade(-10.0),
            _make_trade(-100.0),
        ]
        stats = compute_trade_statistics(trades)
        assert stats.largest_win == Decimal("200.0")
        assert stats.largest_loss == Decimal("-100.0")

    def test_average_trade_pnl(self) -> None:
        """average_trade_pnl = total_pnl / total_trades."""
        trades = [_make_trade(100.0), _make_trade(-50.0)]
        stats = compute_trade_statistics(trades)
        # total_pnl = 50, total_trades = 2 → average = 25
        assert stats.average_trade_pnl == Decimal("25")
