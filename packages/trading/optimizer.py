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

import statistics

import structlog

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import BacktestResult
from trading.strategy import BaseStrategy
from trading.walk_forward import (
    WalkForwardValidator,
    deflated_sharpe_ratio,
)

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
    """One parameter combination and its backtest result.

    When walk-forward validation is enabled the ``metrics`` dict carries
    out-of-sample aggregates — mean/median Sharpe across test folds,
    in-sample Sharpe for comparison, and per-fold metric vectors under
    ``_per_fold``.  Ranking uses ``oos_<metric>`` keys so the best
    combination is the one with the strongest OOS signal, not the
    headline in-sample curve.
    """

    rank: int
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class OptimizationResult:
    """Complete optimization run output.

    Additional fields (present only when walk_forward=True):
        ``deflated_sharpe``  Sprint 41 QT-004 — observed-best Sharpe minus
                             the expected-maximum haircut over N trials.
                             Subtract this from the raw best Sharpe to
                             quantify multiple-testing bias.
        ``num_folds``        Number of walk-forward folds used.
    """

    strategy_name: str
    symbols: list[str]
    timeframe: str
    rank_by: str
    total_combinations: int
    completed_combinations: int
    failed_combinations: int
    elapsed_seconds: float
    entries: list[OptimizationEntry]
    walk_forward: bool = False
    num_folds: int = 0
    deflated_sharpe: float | None = None


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
        walk_forward: bool = False,
        walk_forward_folds: int = 4,
        walk_forward_train_fraction: float = 0.7,
        walk_forward_mode: str = "expanding",
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
        if walk_forward and walk_forward_mode not in ("expanding", "rolling"):
            raise ValueError(
                "walk_forward_mode must be 'expanding' or 'rolling', "
                f"got {walk_forward_mode!r}"
            )

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
        self._walk_forward = walk_forward
        self._walk_forward_folds = walk_forward_folds
        self._walk_forward_train_fraction = walk_forward_train_fraction
        self._walk_forward_mode = walk_forward_mode

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
            Ranked results with top N entries.  When ``walk_forward=True``,
            ranking is by out-of-sample metric and the result carries a
            Deflated Sharpe ratio so callers can gauge the multiple-
            testing bias eaten by the grid search.
        """
        self._log.info(
            "optimizer.starting",
            combinations=len(self._combinations),
            rank_by=self._rank_by,
            walk_forward=self._walk_forward,
        )
        start_time = time.monotonic()

        if self._walk_forward:
            result = await self._run_walk_forward(bars_by_symbol, start_time)
        else:
            result = await self._run_in_sample(bars_by_symbol, start_time)

        self._log.info(
            "optimizer.complete",
            completed=result.completed_combinations,
            failed=result.failed_combinations,
            elapsed_seconds=result.elapsed_seconds,
            walk_forward=result.walk_forward,
            deflated_sharpe=result.deflated_sharpe,
            best_metric=(
                result.entries[0].metrics.get(self._rank_by)
                if result.entries
                else None
            ),
        )
        return result

    # ------------------------------------------------------------------
    # In-sample path (legacy default; preserved behavior)
    # ------------------------------------------------------------------

    async def _run_in_sample(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
        start_time: float,
    ) -> OptimizationResult:
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
                result = await self._run_single_backtest(
                    combo_params=combo_params,
                    strategy_id=f"opt-{idx}",
                    bars_by_symbol=bars_by_symbol,
                )
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

        reverse = self._rank_by not in _ASCENDING_METRICS
        results.sort(
            key=lambda r: r[1].get(self._rank_by, float("-inf")),
            reverse=reverse,
        )

        entries = [
            OptimizationEntry(rank=i + 1, params=params, metrics=metrics)
            for i, (params, metrics) in enumerate(results[: self._top_n])
        ]

        elapsed = time.monotonic() - start_time
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
            walk_forward=False,
            num_folds=0,
            deflated_sharpe=None,
        )

    # ------------------------------------------------------------------
    # Walk-forward path (Sprint 41 QT-004)
    # ------------------------------------------------------------------

    async def _run_walk_forward(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
        start_time: float,
    ) -> OptimizationResult:
        from typing import cast

        validator = WalkForwardValidator(
            num_folds=self._walk_forward_folds,
            train_fraction=self._walk_forward_train_fraction,
            mode=cast("Any", self._walk_forward_mode),
        )
        folds = validator.split(bars_by_symbol)

        self._log.info(
            "optimizer.walk_forward_splits",
            num_folds=len(folds),
            mode=self._walk_forward_mode,
            train_fraction=self._walk_forward_train_fraction,
        )

        # For each combination run every fold's TEST window and aggregate
        # the resulting metrics.  The in-sample pass uses the full dataset
        # so callers can contrast overfit vs OOS Sharpe directly.
        results: list[tuple[dict[str, Any], dict[str, float]]] = []
        failed = 0

        for idx, combo_values in enumerate(self._combinations):
            combo_params = dict(zip(self._param_names, combo_values))

            try:
                # In-sample reference run (full series)
                is_result = await self._run_single_backtest(
                    combo_params=combo_params,
                    strategy_id=f"opt-{idx}-is",
                    bars_by_symbol=bars_by_symbol,
                )
                is_metrics = self._extract_metrics(is_result)

                # Out-of-sample per-fold runs
                fold_sharpes: list[float] = []
                fold_returns: list[float] = []
                fold_drawdowns: list[float] = []
                for fold in folds:
                    fold_result = await self._run_single_backtest(
                        combo_params=combo_params,
                        strategy_id=f"opt-{idx}-fold{fold.index}",
                        bars_by_symbol=fold.test_bars,
                    )
                    fold_sharpes.append(fold_result.sharpe_ratio)
                    fold_returns.append(fold_result.total_return_pct)
                    fold_drawdowns.append(fold_result.max_drawdown_pct)

                # Aggregate OOS metrics — median is used as the primary OOS
                # summary because one-off catastrophic folds (rare market
                # shock inside a single test window) should not dominate
                # ranking.  Mean is reported alongside for context.
                oos_sharpe_median = statistics.median(fold_sharpes)
                oos_sharpe_mean = statistics.mean(fold_sharpes)
                oos_return_mean = statistics.mean(fold_returns)
                oos_drawdown_worst = max(fold_drawdowns)

                metrics = dict(is_metrics)
                metrics["is_sharpe_ratio"] = is_result.sharpe_ratio
                metrics["oos_sharpe_median"] = oos_sharpe_median
                metrics["oos_sharpe_mean"] = oos_sharpe_mean
                metrics["oos_return_pct_mean"] = oos_return_mean
                metrics["oos_max_drawdown_pct_worst"] = oos_drawdown_worst
                # Convenience alias — ranking reads ``oos_{rank_by}`` when
                # walk-forward is on; keep ``oos_sharpe_ratio`` populated to
                # align with the default rank_by="sharpe_ratio" setting.
                metrics["oos_sharpe_ratio"] = oos_sharpe_median

                results.append((combo_params, metrics))

            except Exception:
                self._log.warning(
                    "optimizer.walk_forward_combination_failed",
                    index=idx + 1,
                    params=combo_params,
                    exc_info=True,
                )
                failed += 1

        # Rank by OOS variant of rank_by (falls back to IS metric when
        # the OOS alias is missing, e.g. rank_by metrics not yet mirrored).
        oos_rank_key = f"oos_{self._rank_by}"
        reverse = self._rank_by not in _ASCENDING_METRICS
        results.sort(
            key=lambda r: r[1].get(
                oos_rank_key,
                r[1].get(self._rank_by, float("-inf")),
            ),
            reverse=reverse,
        )

        entries = [
            OptimizationEntry(rank=i + 1, params=params, metrics=metrics)
            for i, (params, metrics) in enumerate(results[: self._top_n])
        ]

        # Deflated Sharpe across all completed trials (based on OOS medians)
        dsr: float | None = None
        if results:
            trial_sharpes = [m.get("oos_sharpe_ratio", 0.0) for _, m in results]
            if len(trial_sharpes) >= 2:
                sharpe_std = statistics.stdev(trial_sharpes)
                best_sharpe = max(trial_sharpes)
                dsr = deflated_sharpe_ratio(
                    observed_sharpe=best_sharpe,
                    sharpe_stddev=sharpe_std,
                    num_trials=len(trial_sharpes),
                )

        elapsed = time.monotonic() - start_time
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
            walk_forward=True,
            num_folds=len(folds),
            deflated_sharpe=dsr,
        )

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    async def _run_single_backtest(
        self,
        *,
        combo_params: dict[str, Any],
        strategy_id: str,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> BacktestResult:
        """Instantiate a strategy + runner for one backtest pass."""
        strategy = self._strategy_cls(
            strategy_id=strategy_id,
            params=combo_params,
        )
        trailing_stop_pct: float | None = combo_params.get("trailing_stop_pct")
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
        return await runner.run(bars_by_symbol)

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
