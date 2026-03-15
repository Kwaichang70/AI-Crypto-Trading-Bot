"""
tests/unit/test_parameter_optimizer.py
---------------------------------------
Unit tests for ParameterOptimizer grid search class.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import TimeFrame
from tests.conftest import make_bars
from trading.optimizer import (
    DEFAULT_MAX_COMBINATIONS,
    SUPPORTED_METRICS,
    OptimizationEntry,
    OptimizationResult,
    ParameterOptimizer,
)
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading.strategy import BaseStrategy
from trading.models import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USD"
TF = TimeFrame.ONE_HOUR


def _bars_by_symbol(n: int = 200) -> dict[str, list[OHLCVBar]]:
    """Generate synthetic bars for testing."""
    return {SYMBOL: make_bars(n, symbol=SYMBOL, timeframe=TF)}


class _DummyStrategy(BaseStrategy):
    """Minimal strategy that always holds — for testing optimizer mechanics."""

    def on_bar(
        self,
        bars: Any,
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        return []


# ===================================================================
# TestParameterOptimizerInit
# ===================================================================


class TestParameterOptimizerInit:
    """Test ParameterOptimizer constructor validation."""

    def test_valid_grid_accepted(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
        )
        assert opt.total_combinations == 9

    def test_empty_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            ParameterOptimizer(
                strategy_cls=MACrossoverStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={},
            )

    def test_unsupported_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported rank_by"):
            ParameterOptimizer(
                strategy_cls=MACrossoverStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={"fast_period": [5, 10]},
                rank_by="invalid_metric",
            )

    def test_exceeds_max_combinations_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeding max_combinations"):
            ParameterOptimizer(
                strategy_cls=MACrossoverStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={
                    "fast_period": list(range(1, 11)),
                    "slow_period": list(range(1, 11)),
                },
                max_combinations=50,
            )

    def test_total_combinations_property(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={
                "fast_period": [5, 10],
                "slow_period": [30, 50, 100],
                "position_size": [500, 1000],
            },
        )
        assert opt.total_combinations == 2 * 3 * 2  # 12

    def test_top_n_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            ParameterOptimizer(
                strategy_cls=MACrossoverStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={"fast_period": [5]},
                top_n=0,
            )

    def test_max_combinations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_combinations must be >= 1"):
            ParameterOptimizer(
                strategy_cls=MACrossoverStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={"fast_period": [5]},
                max_combinations=0,
            )


# ===================================================================
# TestParameterOptimizerRun
# ===================================================================


class TestParameterOptimizerRun:
    """Test grid search execution."""

    @pytest.mark.asyncio
    async def test_grid_search_returns_ranked_results(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            top_n=10,
        )
        result = await opt.run(_bars_by_symbol(200))

        assert isinstance(result, OptimizationResult)
        assert result.strategy_name == "MACrossoverStrategy"
        assert result.total_combinations == 4
        assert result.failed_combinations == 0
        assert result.completed_combinations == 4
        assert result.rank_by == "sharpe_ratio"
        assert result.elapsed_seconds >= 0
        assert len(result.entries) == 4

        # Verify ranking order (descending sharpe)
        for i in range(len(result.entries) - 1):
            assert (
                result.entries[i].metrics["sharpe_ratio"]
                >= result.entries[i + 1].metrics["sharpe_ratio"]
            )

    @pytest.mark.asyncio
    async def test_top_n_limits_entries(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            top_n=2,
        )
        result = await opt.run(_bars_by_symbol(200))
        assert len(result.entries) == 2
        assert result.entries[0].rank == 1
        assert result.entries[1].rank == 2

    @pytest.mark.asyncio
    async def test_failed_combinations_counted(self) -> None:
        # fast_period=50 with slow_period=30 violates fast < slow constraint
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 50], "slow_period": [30]},
        )
        result = await opt.run(_bars_by_symbol(200))
        assert result.failed_combinations == 1
        assert result.completed_combinations == 1

    @pytest.mark.asyncio
    async def test_deterministic_seed(self) -> None:
        bars = _bars_by_symbol(200)
        grid = {"fast_period": [5, 10], "slow_period": [30, 50]}

        opt1 = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid=grid,
        )
        opt2 = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid=grid,
        )
        r1 = await opt1.run(bars)
        r2 = await opt2.run(bars)

        assert len(r1.entries) == len(r2.entries)
        for e1, e2 in zip(r1.entries, r2.entries):
            assert e1.params == e2.params
            assert e1.metrics == e2.metrics

    @pytest.mark.asyncio
    async def test_ascending_metric_drawdown(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            rank_by="max_drawdown_pct",
        )
        result = await opt.run(_bars_by_symbol(200))

        # max_drawdown_pct: lower is better → ascending order
        for i in range(len(result.entries) - 1):
            assert (
                result.entries[i].metrics["max_drawdown_pct"]
                <= result.entries[i + 1].metrics["max_drawdown_pct"]
            )

    @pytest.mark.asyncio
    async def test_all_supported_metrics_rank(self) -> None:
        bars = _bars_by_symbol(200)
        for metric in SUPPORTED_METRICS:
            opt = ParameterOptimizer(
                strategy_cls=_DummyStrategy,
                symbols=[SYMBOL],
                timeframe=TF,
                param_grid={"some_param": [1, 2]},
                rank_by=metric,
            )
            result = await opt.run(bars)
            assert result.rank_by == metric
            assert result.completed_combinations == 2

    @pytest.mark.asyncio
    async def test_single_combination(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [10], "slow_period": [50]},
        )
        result = await opt.run(_bars_by_symbol(200))
        assert result.total_combinations == 1
        assert result.completed_combinations == 1
        assert len(result.entries) == 1
        assert result.entries[0].rank == 1

    @pytest.mark.asyncio
    async def test_all_combinations_fail(self) -> None:
        # fast_period > slow_period for all combos
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [100, 200], "slow_period": [10]},
        )
        result = await opt.run(_bars_by_symbol(200))
        assert result.failed_combinations == 2
        assert result.completed_combinations == 0
        assert len(result.entries) == 0


# ===================================================================
# TestParameterOptimizerMetrics
# ===================================================================


class TestParameterOptimizerMetrics:
    """Test metric extraction from BacktestResult."""

    @pytest.mark.asyncio
    async def test_extract_metrics_all_fields(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [10], "slow_period": [50]},
        )
        result = await opt.run(_bars_by_symbol(200))
        assert len(result.entries) == 1

        metrics = result.entries[0].metrics
        expected_keys = {
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "total_return_pct",
            "cagr",
            "profit_factor",
            "win_rate",
            "max_drawdown_pct",
            "total_trades",
            "final_equity",
            "total_fees_paid",
            "exposure_pct",
        }
        assert set(metrics.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_metrics_types_are_float(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [10], "slow_period": [50]},
        )
        result = await opt.run(_bars_by_symbol(200))
        for key, value in result.entries[0].metrics.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"


# ===================================================================
# TestOptimizationResult
# ===================================================================


class TestOptimizationResult:
    """Test OptimizationResult and OptimizationEntry data classes."""

    def test_entry_is_frozen(self) -> None:
        entry = OptimizationEntry(rank=1, params={"a": 1}, metrics={"sharpe_ratio": 1.5})
        with pytest.raises(AttributeError):
            entry.rank = 2  # type: ignore[misc]

    def test_result_is_frozen(self) -> None:
        result = OptimizationResult(
            strategy_name="Test",
            symbols=["BTC/USD"],
            timeframe="1h",
            rank_by="sharpe_ratio",
            total_combinations=1,
            completed_combinations=1,
            failed_combinations=0,
            elapsed_seconds=0.1,
            entries=[],
        )
        with pytest.raises(AttributeError):
            result.strategy_name = "Other"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_result_fields_populated(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [10], "slow_period": [50]},
        )
        result = await opt.run(_bars_by_symbol(200))

        assert result.strategy_name == "MACrossoverStrategy"
        assert result.symbols == [SYMBOL]
        assert result.timeframe == "1h"
        assert result.rank_by == "sharpe_ratio"
        assert result.total_combinations == 1
        assert result.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_entry_params_match_grid(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=MACrossoverStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={"fast_period": [5, 10], "slow_period": [50]},
        )
        result = await opt.run(_bars_by_symbol(200))

        all_fast_periods = {e.params["fast_period"] for e in result.entries}
        assert all_fast_periods == {5, 10}
        for entry in result.entries:
            assert entry.params["slow_period"] == 50


# ===================================================================
# TestParameterOptimizerWithRSI
# ===================================================================


class TestParameterOptimizerWithRSI:
    """Test optimizer works with RSI strategy (different param schema)."""

    @pytest.mark.asyncio
    async def test_rsi_grid_search(self) -> None:
        opt = ParameterOptimizer(
            strategy_cls=RSIMeanReversionStrategy,
            symbols=[SYMBOL],
            timeframe=TF,
            param_grid={
                "rsi_period": [7, 14],
                "oversold": [25, 30],
                "overbought": [70, 75],
            },
        )
        result = await opt.run(_bars_by_symbol(200))
        assert result.total_combinations == 8
        assert result.completed_combinations == 8
        assert result.failed_combinations == 0
