"""
tests/unit/test_sprint49_m1_profit_factor.py
----------------------------------------------
Sprint 49 M1: build_backtest_metrics profit_factor propagation tests.

Verifies that BacktestResult with profit_factor=None + profit_factor_is_infinite=True
is correctly mapped through build_backtest_metrics into BacktestMetricsResponse
with the same wire shape, and that the field_serializer coerces legacy float('inf')
to None on the wire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from api.schemas import BacktestMetricsResponse
from api.services.run_persistence import build_backtest_metrics


class TestBuildBacktestMetricsProfitFactor:
    """Tests for profit_factor propagation through build_backtest_metrics."""

    def _make_result(
        self,
        *,
        profit_factor: float | None,
        profit_factor_is_infinite: bool,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            total_return_pct=0.1,
            cagr=0.1,
            initial_capital=Decimal("10000"),
            final_equity=Decimal("11000"),
            total_fees_paid=Decimal("50"),
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown_pct=0.05,
            max_drawdown_duration_bars=10,
            total_trades=3,
            winning_trades=3,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=profit_factor,
            profit_factor_is_infinite=profit_factor_is_infinite,
            average_trade_pnl=Decimal("333"),
            average_win=Decimal("333"),
            average_loss=Decimal("0"),
            largest_win=Decimal("400"),
            largest_loss=Decimal("0"),
            total_bars=100,
            bars_in_market=60,
            exposure_pct=0.6,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 2, 1, tzinfo=UTC),
            duration_days=31,
        )

    def test_build_backtest_metrics_propagates_inf_flag(self) -> None:
        """Winners-only result: profit_factor=None, profit_factor_is_infinite=True round-trips."""
        result = self._make_result(profit_factor=None, profit_factor_is_infinite=True)
        response = build_backtest_metrics(result)
        assert isinstance(response, BacktestMetricsResponse)
        assert response.profit_factor is None
        assert response.profit_factor_is_infinite is True

    def test_build_backtest_metrics_zero_trades_wire_shape(self) -> None:
        """Zero-trades result: profit_factor=None, profit_factor_is_infinite=False round-trips."""
        result = self._make_result(profit_factor=None, profit_factor_is_infinite=False)
        response = build_backtest_metrics(result)
        assert response.profit_factor is None
        assert response.profit_factor_is_infinite is False

    def test_build_backtest_metrics_normal_profit_factor(self) -> None:
        """Normal mixed result: profit_factor=2.5 passes through unchanged."""
        result = self._make_result(profit_factor=2.5, profit_factor_is_infinite=False)
        response = build_backtest_metrics(result)
        assert response.profit_factor == pytest.approx(2.5)
        assert response.profit_factor_is_infinite is False

    def test_build_backtest_metrics_serialise_profit_factor_coerces_legacy_inf(self) -> None:
        """Defensive serialiser coerces legacy float('inf') to None on the wire.

        The field_serializer on BacktestMetricsResponse.profit_factor must return None
        for any math.isinf(v) input. model_dump(mode="json") without by_alias=True
        always emits snake_case keys regardless of alias_generator=to_camel.
        """
        result = self._make_result(profit_factor=float("inf"), profit_factor_is_infinite=True)
        response = build_backtest_metrics(result)
        dumped = response.model_dump(mode="json")
        # model_dump(mode="json") without by_alias=True always emits snake_case
        assert dumped["profit_factor"] is None
        assert dumped["profit_factor_is_infinite"] is True
