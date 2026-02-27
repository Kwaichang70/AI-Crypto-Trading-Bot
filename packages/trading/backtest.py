"""
packages/trading/backtest.py
------------------------------
BacktestRunner -- high-level API for running deterministic backtests.

This module provides a clean, self-contained entry point that:
1. Wires together PaperExecutionEngine + DefaultRiskManager +
   PortfolioAccounting + StrategyEngine
2. Validates input data (bar ordering, alignment, warm-up sufficiency)
3. Runs the backtest via StrategyEngine.run_backtest()
4. Collects the equity curve, trade log, and computes all metrics
5. Returns a fully-populated BacktestResult

Design principles
-----------------
- **Deterministic**: if a seed is provided, all stochastic elements
  (future: slippage jitter) produce identical results.
- **No look-ahead bias**: bars are fed chronologically; the growing
  window passed to strategies never includes future data.
- **Strategy-agnostic**: works with any ``BaseStrategy`` subclass.
- **Fee-aware**: all PnL metrics are net of trading fees.
- **Decimal precision**: monetary values use ``Decimal``; statistical
  ratios use ``float``.

Usage
-----
.. code-block:: python

    runner = BacktestRunner(
        strategies=[MACrossoverStrategy("ma_cross", {"fast": 10, "slow": 30})],
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        slippage_bps=5,
        maker_fee_bps=5,
        taker_fee_bps=10,
    )
    result = await runner.run(bars_by_symbol)
    print(f"CAGR: {result.cagr:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
"""

from __future__ import annotations

import math
import random
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from common.models import OHLCVBar
from common.types import RunMode, TimeFrame
from trading.engines.paper import PaperExecutionEngine
from trading.metrics import (
    BacktestResult,
    EquityCurvePoint,
    TIMEFRAME_PERIODS_PER_YEAR,
    compute_cagr,
    compute_calmar,
    compute_exposure,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_trade_statistics,
)
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine

__all__ = ["BacktestRunner"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Timeframe -> warm-up bar mapping (sensible defaults)
# ---------------------------------------------------------------------------

_DEFAULT_WARMUP_MULTIPLIER: int = 2
"""
The warmup period is set to max(strategy.min_bars_required) * this
multiplier, to provide strategies with sufficient history for indicator
convergence.
"""


class BacktestRunner:
    """
    High-level backtest orchestrator.

    Encapsulates the full pipeline from raw OHLCV bars to a
    comprehensive ``BacktestResult`` with all performance metrics.

    Parameters
    ----------
    strategies : list[BaseStrategy]
        One or more strategy instances to run.  Each must be a concrete
        subclass of ``BaseStrategy`` with ``on_bar`` implemented.
    symbols : list[str]
        Trading pairs to backtest, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    timeframe : TimeFrame
        Candle timeframe for all input data.
    initial_capital : Decimal
        Starting cash in quote currency.  Default 10000.
    risk_params : RiskParameters | None
        Risk manager configuration.  If None, default parameters are
        used.  Fee percentages in risk_params are overridden by the
        explicit ``maker_fee_bps`` / ``taker_fee_bps`` arguments if
        both are provided.
    slippage_bps : int
        Slippage in basis points applied to market orders.
        Default 5 (0.05%).
    maker_fee_bps : int
        Maker fee in basis points.  Default 10 (0.10%).
    taker_fee_bps : int
        Taker fee in basis points.  Default 15 (0.15%).
    seed : int | None
        Random seed for deterministic backtests.  If provided, sets
        ``random.seed(seed)`` before execution.  Default None.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        symbols: list[str],
        timeframe: TimeFrame,
        initial_capital: Decimal = Decimal("10000"),
        risk_params: RiskParameters | None = None,
        slippage_bps: int = 5,
        maker_fee_bps: int = 10,
        taker_fee_bps: int = 15,
        seed: int | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        if not symbols:
            raise ValueError("At least one symbol is required")
        if initial_capital <= Decimal("0"):
            raise ValueError(
                f"initial_capital must be positive, got {initial_capital}"
            )

        self._strategies = list(strategies)
        self._symbols = list(symbols)
        self._timeframe = timeframe
        self._initial_capital = initial_capital
        self._slippage_bps = slippage_bps
        self._maker_fee_bps = maker_fee_bps
        self._taker_fee_bps = taker_fee_bps
        self._seed = seed

        # Build risk parameters with explicit fee overrides
        if risk_params is not None:
            self._risk_params = RiskParameters(
                max_open_positions=risk_params.max_open_positions,
                max_position_size_pct=risk_params.max_position_size_pct,
                per_trade_risk_pct=risk_params.per_trade_risk_pct,
                max_order_size_quote=risk_params.max_order_size_quote,
                max_daily_loss_pct=risk_params.max_daily_loss_pct,
                max_drawdown_pct=risk_params.max_drawdown_pct,
                taker_fee_pct=taker_fee_bps / 10_000,
                maker_fee_pct=maker_fee_bps / 10_000,
                slippage_bps=slippage_bps,
                cooldown_after_loss_streak=risk_params.cooldown_after_loss_streak,
                loss_streak_count=risk_params.loss_streak_count,
            )
        else:
            self._risk_params = RiskParameters(
                taker_fee_pct=taker_fee_bps / 10_000,
                maker_fee_pct=maker_fee_bps / 10_000,
                slippage_bps=slippage_bps,
            )

        # Compute warmup: max of all strategy requirements * multiplier
        max_min_bars = max(
            (s.min_bars_required for s in self._strategies),
            default=0,
        )
        self._warmup_bars = max(max_min_bars * _DEFAULT_WARMUP_MULTIPLIER, 50)

        self._log = structlog.get_logger(__name__).bind(
            component="backtest_runner",
            symbols=self._symbols,
            timeframe=self._timeframe.value,
            strategies=[s.strategy_id for s in self._strategies],
            warmup_bars=self._warmup_bars,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> BacktestResult:
        """
        Execute a backtest over the provided historical bars.

        Parameters
        ----------
        bars_by_symbol : dict[str, list[OHLCVBar]]
            Pre-fetched OHLCV bars keyed by symbol.  Each list must be
            sorted by timestamp ascending and use the same timeframe.

        Returns
        -------
        BacktestResult
            Fully-populated result with all metrics computed.

        Raises
        ------
        ValueError
            If bar data fails validation (unsorted, misaligned, or
            insufficient for warm-up).
        """
        # 0. Deterministic seed
        if self._seed is not None:
            random.seed(self._seed)
            self._log.info("backtest.seed_set", seed=self._seed)

        # 1. Validate input bars
        self._validate_bars(bars_by_symbol)

        # 2. Generate run ID
        run_id = f"bt-{uuid.uuid4().hex[:12]}"

        # 3. Build engine components
        engine, portfolio, risk_manager = self._build_engine(run_id)

        self._log.info(
            "backtest.starting",
            run_id=run_id,
            initial_capital=str(self._initial_capital),
            slippage_bps=self._slippage_bps,
            maker_fee_bps=self._maker_fee_bps,
            taker_fee_bps=self._taker_fee_bps,
        )

        # 4. Start engine
        await engine.start(run_id)

        # 5. Track bar-level state for exposure calculation
        total_bars_processed = 0
        bars_in_market = 0

        # We need to track per-bar equity for accurate curve construction.
        # The portfolio records equity on every update_market_prices and
        # update_position call, but we want one point per bar after warmup.
        # We will extract the equity curve from portfolio after the run.

        try:
            # 6. Run backtest through the strategy engine
            await engine.run_backtest(bars_by_symbol)
        finally:
            # 7. Stop engine (always, even on error)
            await engine.stop()

        # 8. Collect results
        raw_equity_curve = portfolio.get_equity_curve()
        trade_history = portfolio.get_trade_history()

        # Determine date range from bars
        all_bars = []
        for sym in self._symbols:
            all_bars.extend(bars_by_symbol[sym])
        if all_bars:
            start_date = min(b.timestamp for b in all_bars)
            end_date = max(b.timestamp for b in all_bars)
        else:
            start_date = datetime.now(tz=UTC)
            end_date = start_date

        duration_days = max(1, (end_date - start_date).days)

        # 9. Build equity curve points with drawdown annotation
        equity_curve = self._build_equity_curve(raw_equity_curve)

        # 10. Compute total bars processed (shortest symbol series minus warmup)
        num_bars_per_symbol = min(
            len(bars_by_symbol[s]) for s in self._symbols
        )
        total_bars_processed = max(0, num_bars_per_symbol - self._warmup_bars)

        # 11. Compute exposure: count bars where portfolio had open positions
        # We approximate this from the equity curve: if equity differs from
        # cash-only (initial_capital + realised_pnl), a position is open.
        # A more precise approach: count bars where position_snapshots had
        # non-flat positions.  We use the equity curve as a proxy.
        bars_in_market = self._estimate_bars_in_market(
            equity_curve, portfolio
        )

        # 12. Compute final equity
        final_equity = portfolio.current_equity

        # 13. Compute returns
        total_return_pct = float(
            (final_equity - self._initial_capital) / self._initial_capital
        ) if self._initial_capital > Decimal("0") else 0.0

        cagr = compute_cagr(self._initial_capital, final_equity, duration_days)

        # 14. Compute risk metrics
        max_dd = compute_max_drawdown(equity_curve)
        max_dd_duration = compute_max_drawdown_duration(equity_curve)

        periods_per_year = TIMEFRAME_PERIODS_PER_YEAR.get(
            self._timeframe, 365.25
        )
        per_period_returns = compute_returns_from_equity(equity_curve)

        sharpe = compute_sharpe(per_period_returns, periods_per_year)
        sortino = compute_sortino(per_period_returns, periods_per_year)
        calmar = compute_calmar(cagr, max_dd)

        # 15. Compute trade statistics
        trade_stats = compute_trade_statistics(trade_history)

        exposure = compute_exposure(bars_in_market, total_bars_processed)

        # 16. Assemble result
        result = BacktestResult(
            # Metadata
            run_id=run_id,
            strategy_ids=[s.strategy_id for s in self._strategies],
            symbols=self._symbols,
            timeframe=self._timeframe,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            # Capital
            initial_capital=self._initial_capital,
            final_equity=final_equity,
            # Returns
            total_return_pct=total_return_pct,
            cagr=cagr,
            # Risk
            max_drawdown_pct=max_dd,
            max_drawdown_duration_bars=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            # Trades
            total_trades=trade_stats.total_trades,
            winning_trades=trade_stats.winning_trades,
            losing_trades=trade_stats.losing_trades,
            win_rate=trade_stats.win_rate,
            profit_factor=trade_stats.profit_factor,
            average_trade_pnl=trade_stats.average_trade_pnl,
            average_win=trade_stats.average_win,
            average_loss=trade_stats.average_loss,
            largest_win=trade_stats.largest_win,
            largest_loss=trade_stats.largest_loss,
            # Exposure
            total_bars=total_bars_processed,
            bars_in_market=bars_in_market,
            exposure_pct=exposure,
            # Curve & trades
            equity_curve=equity_curve,
            trades=trade_history,
            # Fees
            total_fees_paid=portfolio.total_fees_paid,
        )

        self._log.info(
            "backtest.complete",
            run_id=run_id,
            total_return=f"{total_return_pct:.4%}",
            cagr=f"{cagr:.4%}",
            sharpe=f"{sharpe:.3f}",
            sortino=f"{sortino:.3f}" if not math.isinf(sortino) else "inf",
            calmar=f"{calmar:.3f}",
            max_drawdown=f"{max_dd:.4%}",
            total_trades=trade_stats.total_trades,
            win_rate=f"{trade_stats.win_rate:.2%}",
            profit_factor=f"{trade_stats.profit_factor:.2f}",
            exposure=f"{exposure:.2%}",
            total_fees=str(portfolio.total_fees_paid),
        )

        return result

    # ------------------------------------------------------------------
    # Engine wiring
    # ------------------------------------------------------------------

    def _build_engine(
        self,
        run_id: str,
    ) -> tuple[StrategyEngine, PortfolioAccounting, DefaultRiskManager]:
        """
        Wire up all trading core components for a backtest run.

        Parameters
        ----------
        run_id : str
            Unique run identifier.

        Returns
        -------
        tuple[StrategyEngine, PortfolioAccounting, DefaultRiskManager]
            The fully-wired engine, portfolio, and risk manager.

        Notes
        -----
        The StrategyEngine requires a market data service, but in backtest
        mode it is never called (bars are passed directly).  We create a
        minimal stub to satisfy the constructor.
        """
        risk_manager = DefaultRiskManager(
            run_id=run_id,
            params=self._risk_params,
        )

        execution_engine = PaperExecutionEngine(
            run_id=run_id,
            risk_manager=risk_manager,
            slippage_bps=self._slippage_bps,
            initial_cash=self._initial_capital,
        )

        portfolio = PortfolioAccounting(
            run_id=run_id,
            initial_cash=self._initial_capital,
        )

        # The StrategyEngine requires a market data service for PAPER/LIVE
        # modes.  In BACKTEST mode it is never used, but the constructor
        # still requires one.  We provide a stub.
        market_data_stub = _BacktestMarketDataStub()

        engine = StrategyEngine(
            strategies=self._strategies,
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            market_data=market_data_stub,
            portfolio=portfolio,
            symbols=self._symbols,
            timeframe=self._timeframe,
            run_mode=RunMode.BACKTEST,
            config={"warmup_bars": self._warmup_bars},
        )

        return engine, portfolio, risk_manager

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_bars(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> None:
        """
        Validate bar data before running a backtest.

        Checks
        ------
        1. All configured symbols have bar data.
        2. Each symbol's bars are sorted by timestamp ascending.
        3. All symbols have the same number of bars (or at least
           enough after truncation to the shortest series).
        4. The shortest series has enough bars for the warm-up period.

        Parameters
        ----------
        bars_by_symbol : dict[str, list[OHLCVBar]]
            Raw bar data keyed by symbol.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # Check 1: all symbols present
        for symbol in self._symbols:
            if symbol not in bars_by_symbol:
                raise ValueError(
                    f"No bar data provided for symbol '{symbol}'. "
                    f"Expected data for: {self._symbols}"
                )
            if not bars_by_symbol[symbol]:
                raise ValueError(
                    f"Empty bar list for symbol '{symbol}'"
                )

        # Check 2: chronological ordering (no look-ahead bias)
        for symbol in self._symbols:
            bars = bars_by_symbol[symbol]
            for i in range(1, len(bars)):
                if bars[i].timestamp < bars[i - 1].timestamp:
                    raise ValueError(
                        f"Bars for '{symbol}' are not sorted by timestamp "
                        f"ascending. Bar at index {i} "
                        f"(ts={bars[i].timestamp}) precedes bar at index "
                        f"{i - 1} (ts={bars[i - 1].timestamp}). "
                        f"This would introduce look-ahead bias."
                    )
                if bars[i].timestamp == bars[i - 1].timestamp:
                    raise ValueError(
                        f"Duplicate timestamp for '{symbol}' at index {i}: "
                        f"{bars[i].timestamp}. Remove duplicates to ensure "
                        f"deterministic bar processing."
                    )

        # Check 3: sufficient bars for warm-up
        min_bars = min(len(bars_by_symbol[s]) for s in self._symbols)
        if min_bars <= self._warmup_bars:
            raise ValueError(
                f"Insufficient bars for warm-up. Shortest series has "
                f"{min_bars} bars but warm-up requires {self._warmup_bars}. "
                f"Provide at least {self._warmup_bars + 1} bars per symbol."
            )

        # Check 4: log alignment info
        bar_counts = {s: len(bars_by_symbol[s]) for s in self._symbols}
        if len(set(bar_counts.values())) > 1:
            self._log.warning(
                "backtest.bar_count_mismatch",
                bar_counts=bar_counts,
                note="Series will be truncated to shortest length",
            )

        self._log.info(
            "backtest.bars_validated",
            bar_counts=bar_counts,
            warmup_bars=self._warmup_bars,
            effective_bars=min_bars - self._warmup_bars,
        )

    # ------------------------------------------------------------------
    # Equity curve construction
    # ------------------------------------------------------------------

    def _build_equity_curve(
        self,
        raw_curve: list[tuple[datetime, Decimal]],
    ) -> list[EquityCurvePoint]:
        """
        Convert the raw (timestamp, equity) tuples from PortfolioAccounting
        into annotated EquityCurvePoint objects with drawdown information.

        Parameters
        ----------
        raw_curve : list[tuple[datetime, Decimal]]
            Raw equity curve from PortfolioAccounting.

        Returns
        -------
        list[EquityCurvePoint]
            Annotated equity curve points.
        """
        if not raw_curve:
            return []

        points: list[EquityCurvePoint] = []
        peak = Decimal("0")

        for timestamp, equity in raw_curve:
            if equity > peak:
                peak = equity

            dd_pct = 0.0
            if peak > Decimal("0"):
                dd_pct = float((peak - equity) / peak)
                dd_pct = max(0.0, dd_pct)

            points.append(EquityCurvePoint(
                timestamp=timestamp,
                equity=equity,
                drawdown_pct=dd_pct,
            ))

        return points

    # ------------------------------------------------------------------
    # Exposure estimation
    # ------------------------------------------------------------------

    def _estimate_bars_in_market(
        self,
        equity_curve: list[EquityCurvePoint],
        portfolio: PortfolioAccounting,
    ) -> int:
        """
        Estimate the number of bars where the portfolio had open positions.

        This is an approximation based on the equity curve.  A bar is
        considered "in market" if the equity differs from what it would
        be with cash only (i.e., there is unrealised position value).

        For a more precise calculation, the StrategyEngine would need to
        track position state per bar.  This heuristic is sufficient for
        reporting purposes.

        Parameters
        ----------
        equity_curve : list[EquityCurvePoint]
            The annotated equity curve.
        portfolio : PortfolioAccounting
            The portfolio accounting instance.

        Returns
        -------
        int
            Estimated number of bars with open positions.
        """
        # Use the trade history to estimate exposure.
        # Each trade spans from entry_at to exit_at.
        # We count equity curve points that fall within any trade span.
        trades = portfolio.get_trade_history()
        if not trades or not equity_curve:
            return 0

        # Simple heuristic: count equity curve points where equity
        # is not equal to cash.  Since equity = cash + position_value,
        # if equity != cash, there is an open position.
        #
        # However, we do not have direct access to cash at each point.
        # Instead, we count the number of non-zero drawdown points as
        # a rough lower bound on exposure.  A better approach:
        # count points where equity deviates from a linear interpolation.
        #
        # The most accurate approach available without additional tracking:
        # count the number of total bars, and estimate from trade count
        # and average trade duration.
        if not trades:
            return 0

        # Count bars between each trade's entry and exit
        curve_timestamps = [p.timestamp for p in equity_curve]
        bars_count = 0
        for trade in trades:
            for ts in curve_timestamps:
                if trade.entry_at <= ts <= trade.exit_at:
                    bars_count += 1

        # Deduplicate (a bar can be "in market" for multiple trades)
        # This is a simplification; in practice overlapping trades on
        # different symbols would cause overcounting. Cap at curve length.
        return min(bars_count, len(equity_curve))


# ---------------------------------------------------------------------------
# Market data stub for backtest mode
# ---------------------------------------------------------------------------

class _BacktestMarketDataStub:
    """
    Minimal stub that satisfies StrategyEngine's market_data dependency
    in BACKTEST mode.

    In backtest mode, bars are provided directly to ``run_backtest()`` and
    the market data service is never called.  This stub raises if any
    method is invoked, signalling a bug in the backtest flow.
    """

    async def connect(self) -> None:
        raise RuntimeError(
            "MarketDataService.connect() should not be called in BACKTEST mode"
        )

    async def close(self) -> None:
        raise RuntimeError(
            "MarketDataService.close() should not be called in BACKTEST mode"
        )

    async def fetch_ohlcv(self, **kwargs: Any) -> list[OHLCVBar]:
        raise RuntimeError(
            "MarketDataService.fetch_ohlcv() should not be called in BACKTEST mode"
        )

    async def get_latest_bar(self, **kwargs: Any) -> OHLCVBar:
        raise RuntimeError(
            "MarketDataService.get_latest_bar() should not be called in BACKTEST mode"
        )
