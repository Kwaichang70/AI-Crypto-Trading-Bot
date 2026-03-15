"""
packages/trading/optimizer.py
------------------------------
Parameter optimization via grid search over strategy parameter space.

Generates all combinations from a parameter grid, runs a BacktestRunner
for each, and returns ranked results by a chosen performance metric.

Design principles:
- Data is fetched ONCE externally and passed in (no redundant exchange calls)
- Sequential execution (no parallelism for MVP)
- Hard cap on max_combinations to prevent combinatorial explosion
- Strategy-agnostic: works with any BaseStrategy subclass
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import BacktestResult
from trading.strategy import BaseStrategy

__all__ = ["ParameterOptimizer", "OptimizationResult", "OptimizationEntry"]

logger = structlog.get_logger(__name__)

# Metrics where lower is better (rank ascending)
_ASCENDING_METRICS: frozenset[str] = frozenset({"max_drawdown_pct"})

# All supported ranking metrics
SUPPORTED_METRICS: frozenset[str] = frozenset({
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "total_return_pct",
    "cagr",
    "profit_factor",
    "win_rate",
    "max_drawdown_pct",
})

DEFAULT_MAX_COMBINATIONS: int = 500


@dataclass(frozen=True)
class OptimizationEntry:
    """One parameter combination and its backtest result."""

    rank: int
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class OptimizationResult:
    """Complete optimization run output."""

    strategy_name: str
    symbols: list[str]
    timeframe: str
    rank_by: str
    total_combinations: int
    completed_combinations: int
    failed_combinations: int
    elapsed_seconds: float
    entries: list[OptimizationEntry]


class ParameterOptimizer:
    """
    Grid search optimizer over strategy parameter space.

    Parameters
    ----------
    strategy_cls :
        Strategy class to instantiate for each combination.
    symbols :
        Trading pairs.
    timeframe :
        Candle timeframe.
    param_grid :
        Parameter name -> list of values to search.
        Example: {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]}
    initial_capital :
        Starting cash. Default 10000.
    rank_by :
        Metric name to rank results by. Default "sharpe_ratio".
    top_n :
        Number of top results to return. Default 10.
    max_combinations :
        Hard cap on total combinations. Default 500.
    """

    def __init__(
        self,
        strategy_cls: type[BaseStrategy],
        symbols: list[str],
        timeframe: TimeFrame,
        param_grid: dict[str, list[Any]],
        initial_capital: Decimal = Decimal("10000"),
        rank_by: str = "sharpe_ratio",
        top_n: int = 10,
        max_combinations: int = DEFAULT_MAX_COMBINATIONS,
        maker_fee_bps: int = 10,
        taker_fee_bps: int = 15,
        slippage_bps: int = 5,
    ) -> None:
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter")
        if rank_by not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported rank_by metric: {rank_by!r}. "
                f"Supported: {sorted(SUPPORTED_METRICS)}"
            )
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if max_combinations < 1:
            raise ValueError(f"max_combinations must be >= 1, got {max_combinations}")

        self._strategy_cls = strategy_cls
        self._symbols = list(symbols)
        self._timeframe = timeframe
        self._param_grid = param_grid
        self._initial_capital = initial_capital
        self._rank_by = rank_by
        self._top_n = top_n
        self._max_combinations = max_combinations
        self._maker_fee_bps = maker_fee_bps
        self._taker_fee_bps = taker_fee_bps
        self._slippage_bps = slippage_bps

        # Pre-compute combinations and validate count
        self._param_names = list(param_grid.keys())
        self._param_values = list(param_grid.values())
        self._combinations = list(itertools.product(*self._param_values))
        if len(self._combinations) > max_combinations:
            raise ValueError(
                f"Parameter grid produces {len(self._combinations)} combinations, "
                f"exceeding max_combinations={max_combinations}. "
                f"Reduce the grid or increase max_combinations."
            )

        self._log = structlog.get_logger(__name__).bind(
            component="parameter_optimizer",
            strategy=strategy_cls.__name__,
            total_combinations=len(self._combinations),
        )

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations to evaluate."""
        return len(self._combinations)

    async def run(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> OptimizationResult:
        """
        Execute the grid search.

        Parameters
        ----------
        bars_by_symbol :
            Pre-fetched OHLCV bars keyed by symbol. Reused for every
            combination -- never re-fetched.

        Returns
        -------
        OptimizationResult
            Ranked results with top N entries.
        """
        self._log.info(
            "optimizer.starting",
            combinations=len(self._combinations),
            rank_by=self._rank_by,
        )
        start_time = time.monotonic()

        results: list[tuple[dict[str, Any], dict[str, float]]] = []
        failed = 0

        for idx, combo_values in enumerate(self._combinations):
            combo_params = dict(zip(self._param_names, combo_values))

            self._log.debug(
                "optimizer.running_combination",
                index=idx + 1,
                total=len(self._combinations),
                params=combo_params,
            )

            try:
                strategy_id = f"opt-{idx}"
                strategy = self._strategy_cls(
                    strategy_id=strategy_id,
                    params=combo_params,
                )

                trailing_stop_pct: float | None = combo_params.get(
                    "trailing_stop_pct"
                )

                runner = BacktestRunner(
                    strategies=[strategy],
                    symbols=self._symbols,
                    timeframe=self._timeframe,
                    initial_capital=self._initial_capital,
                    maker_fee_bps=self._maker_fee_bps,
                    taker_fee_bps=self._taker_fee_bps,
                    slippage_bps=self._slippage_bps,
                    trailing_stop_pct=trailing_stop_pct,
                    seed=42,
                )

                result = await runner.run(bars_by_symbol)
                metrics = self._extract_metrics(result)
                results.append((combo_params, metrics))

            except Exception:
                self._log.warning(
                    "optimizer.combination_failed",
                    index=idx + 1,
                    params=combo_params,
                    exc_info=True,
                )
                failed += 1

        # Rank results
        reverse = self._rank_by not in _ASCENDING_METRICS
        results.sort(
            key=lambda r: r[1].get(self._rank_by, float("-inf")),
            reverse=reverse,
        )

        # Build top-N entries
        entries = [
            OptimizationEntry(
                rank=i + 1,
                params=params,
                metrics=metrics,
            )
            for i, (params, metrics) in enumerate(results[: self._top_n])
        ]

        elapsed = time.monotonic() - start_time

        self._log.info(
            "optimizer.complete",
            completed=len(results),
            failed=failed,
            elapsed_seconds=round(elapsed, 2),
            best_metric=entries[0].metrics.get(self._rank_by) if entries else None,
        )

        return OptimizationResult(
            strategy_name=self._strategy_cls.__name__,
            symbols=self._symbols,
            timeframe=self._timeframe.value,
            rank_by=self._rank_by,
            total_combinations=len(self._combinations),
            completed_combinations=len(results),
            failed_combinations=failed,
            elapsed_seconds=round(elapsed, 2),
            entries=entries,
        )

    def _extract_metrics(self, result: BacktestResult) -> dict[str, float]:
        """Extract the standard metric dict from a BacktestResult.

        Rankable metrics (present in SUPPORTED_METRICS): sharpe_ratio, sortino_ratio,
        calmar_ratio, total_return_pct, cagr, profit_factor, win_rate, max_drawdown_pct.
        Informational-only (not rankable): total_trades, final_equity,
        total_fees_paid, exposure_pct.
        """
        return {
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "total_return_pct": result.total_return_pct,
            "cagr": result.cagr,
            "profit_factor": result.profit_factor,
            "win_rate": result.win_rate,
            "max_drawdown_pct": result.max_drawdown_pct,
            "total_trades": float(result.total_trades),
            "final_equity": float(result.final_equity),
            "total_fees_paid": float(result.total_fees_paid),
            "exposure_pct": result.exposure_pct,
        }
