"""
packages/trading/strategy_engine.py
-------------------------------------
StrategyEngine -- central orchestrator that ties together all trading core
components for a complete trading run.

Responsibilities
----------------
1. **Run lifecycle**: ``start(run_id)`` -> bar loop -> ``stop()``
2. **Bar-by-bar processing**: fetch candles -> feed to strategies -> collect
   signals -> execute via engine -> route fills to portfolio & risk
3. **Component wiring**: holds references to all core components and mediates
   their interactions
4. **Multi-strategy support**: runs multiple strategies in parallel, each
   producing independent signals
5. **Fill routing**: routes fills from the execution engine to
   PortfolioAccounting and RiskManager

Run modes
---------
- BACKTEST: walk through pre-fetched historical bars deterministically
- PAPER: poll market data service for new bars on interval
- LIVE: same polling loop as PAPER but with real order placement

Safety invariants
-----------------
- In LIVE mode the risk manager kill-switch is checked before each bar
- Strategy exceptions are caught and logged without crashing the bar loop
- The bar loop never crashes -- all errors are logged and processing continues
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum, auto
from typing import Any

import structlog

from common.models import OHLCVBar
from common.types import OrderSide, RunMode, TimeFrame
from data.market_data import BaseMarketDataService, MarketDataError
from trading.execution import BaseExecutionEngine
from trading.models import Fill, Position, Signal, TradeResult
from trading.portfolio import PortfolioAccounting
from trading.risk import BaseRiskManager
from trading.strategy import BaseStrategy

__all__ = ["StrategyEngine", "EngineState"]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine state enum
# ---------------------------------------------------------------------------

class EngineState(StrEnum):
    """Lifecycle state of the StrategyEngine."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Timeframe -> seconds mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.ONE_MINUTE: 60,
    TimeFrame.THREE_MINUTES: 180,
    TimeFrame.FIVE_MINUTES: 300,
    TimeFrame.FIFTEEN_MINUTES: 900,
    TimeFrame.THIRTY_MINUTES: 1800,
    TimeFrame.ONE_HOUR: 3600,
    TimeFrame.FOUR_HOURS: 14400,
    TimeFrame.ONE_DAY: 86400,
    TimeFrame.ONE_WEEK: 604800,
}


class StrategyEngine:
    """
    Central orchestrator for a trading run.

    Wires together strategies, execution engine, risk manager, market data
    service, and portfolio accounting into a coherent bar-by-bar processing
    loop.  Supports BACKTEST, PAPER, and LIVE run modes through a unified
    async interface.

    Parameters
    ----------
    strategies :
        One or more strategy instances. Each produces independent signals
        on every bar.
    execution_engine :
        The execution engine (paper or live) that processes signals into
        orders and fills.
    risk_manager :
        Pre-trade risk gating and position sizing.
    market_data :
        Market data service for fetching OHLCV candles (used in PAPER/LIVE
        modes).
    portfolio :
        Portfolio accounting for equity tracking, PnL, and drawdown.
    symbols :
        List of trading pairs to monitor, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    timeframe :
        Candle timeframe for the run.
    run_mode :
        BACKTEST, PAPER, or LIVE.
    config :
        Optional configuration overrides. Recognised keys:
        - ``warmup_bars`` (int): number of bars required before strategies
          receive their first ``on_bar`` call. Default 50.
        - ``max_bars_history`` (int): maximum rolling window size for
          live/paper mode. Default 500.
        - ``poll_interval_seconds`` (float | None): override the default
          polling interval derived from ``timeframe``. Default None.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        execution_engine: BaseExecutionEngine,
        risk_manager: BaseRiskManager,
        market_data: BaseMarketDataService,
        portfolio: PortfolioAccounting,
        symbols: list[str],
        timeframe: TimeFrame,
        run_mode: RunMode,
        config: dict[str, Any] | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        if not symbols:
            raise ValueError("At least one symbol is required")

        self._strategies = list(strategies)
        self._execution_engine = execution_engine
        self._risk_manager = risk_manager
        self._market_data = market_data
        self._portfolio = portfolio
        self._symbols = list(symbols)
        self._timeframe = timeframe
        self._run_mode = run_mode
        self._config: dict[str, Any] = config or {}

        # Derived configuration
        self._warmup_bars: int = int(self._config.get("warmup_bars", 50))
        self._max_bars_history: int = int(
            self._config.get("max_bars_history", 500)
        )
        poll_override = self._config.get("poll_interval_seconds")
        self._poll_interval: float = (
            float(poll_override)
            if poll_override is not None
            else float(_TIMEFRAME_SECONDS.get(timeframe, 60))
        )

        # Run state
        self._state = EngineState.IDLE
        self._run_id: str | None = None
        self._bar_count: int = 0
        self._total_signals: int = 0
        self._total_orders: int = 0
        self._total_fills: int = 0
        self._stop_event: asyncio.Event = asyncio.Event()

        # Rolling bar windows for paper/live mode: symbol -> list[OHLCVBar]
        self._bar_windows: dict[str, list[OHLCVBar]] = {
            s: [] for s in self._symbols
        }

        self._log = structlog.get_logger(__name__).bind(
            component="strategy_engine",
            run_mode=run_mode.value,
            timeframe=timeframe.value,
            symbols=self._symbols,
            strategy_count=len(self._strategies),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        """Current lifecycle state of the engine."""
        return self._state

    @property
    def run_id(self) -> str | None:
        """Run identifier, set after ``start()`` is called."""
        return self._run_id

    @property
    def run_mode(self) -> RunMode:
        """The run mode (BACKTEST, PAPER, or LIVE)."""
        return self._run_mode

    @property
    def bar_count(self) -> int:
        """Number of bars processed so far."""
        return self._bar_count

    @property
    def portfolio(self) -> PortfolioAccounting:
        """Direct access to the portfolio accounting instance."""
        return self._portfolio

    # ------------------------------------------------------------------
    # Lifecycle: start
    # ------------------------------------------------------------------

    async def start(self, run_id: str) -> None:
        """
        Initialise all components and prepare for bar processing.

        This method transitions the engine from IDLE to RUNNING. It calls
        ``on_start`` on each strategy and the execution engine, and
        connects to the market data service for PAPER/LIVE modes.

        Parameters
        ----------
        run_id :
            Unique identifier for this trading run.

        Raises
        ------
        RuntimeError
            If the engine is not in IDLE state.
        """
        if self._state != EngineState.IDLE:
            raise RuntimeError(
                f"Cannot start engine in state {self._state.value}; "
                f"expected IDLE"
            )

        self._state = EngineState.STARTING
        self._run_id = run_id
        self._stop_event.clear()  # Reset for this run (supports restart)
        self._log = self._log.bind(run_id=run_id)

        self._log.info("engine.starting")

        try:
            # Start execution engine
            await self._execution_engine.on_start()

            # Connect market data service (paper/live only)
            if self._run_mode in (RunMode.PAPER, RunMode.LIVE):
                await self._market_data.connect()

            # Start strategies
            for strategy in self._strategies:
                try:
                    strategy.on_start(run_id)
                except Exception:
                    self._log.exception(
                        "engine.strategy_start_failed",
                        strategy_id=strategy.strategy_id,
                    )
                    raise

            self._state = EngineState.RUNNING
            self._log.info(
                "engine.started",
                strategies=[s.strategy_id for s in self._strategies],
            )

        except Exception:
            self._state = EngineState.ERROR
            self._log.exception("engine.start_failed")
            raise

    # ------------------------------------------------------------------
    # Lifecycle: stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """
        Gracefully shut down the engine.

        Stops all strategies, cancels open orders via the execution engine,
        closes the market data connection, and logs the final portfolio
        summary.
        """
        if self._state not in (EngineState.RUNNING, EngineState.ERROR):
            self._log.warning(
                "engine.stop_invalid_state",
                state=self._state.value,
            )
            return

        self._state = EngineState.STOPPING
        self._stop_event.set()
        self._log.info("engine.stopping")

        # Stop strategies
        for strategy in self._strategies:
            try:
                strategy.on_stop()
            except Exception:
                self._log.exception(
                    "engine.strategy_stop_failed",
                    strategy_id=strategy.strategy_id,
                )

        # Cancel open orders
        open_orders = self._execution_engine.get_open_orders()
        for order in open_orders:
            try:
                await self._execution_engine.cancel_order(order.order_id)
                self._log.info(
                    "engine.order_canceled_on_stop",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                )
            except Exception:
                self._log.exception(
                    "engine.cancel_failed_on_stop",
                    order_id=str(order.order_id),
                )

        # Stop execution engine
        await self._execution_engine.on_stop()

        # Close market data connection (paper/live only)
        if self._run_mode in (RunMode.PAPER, RunMode.LIVE):
            try:
                await self._market_data.close()
            except Exception:
                self._log.exception("engine.market_data_close_failed")

        # Log final summary
        summary = self._portfolio.get_summary()
        self._log.info(
            "engine.stopped",
            bars_processed=self._bar_count,
            total_signals=self._total_signals,
            total_orders=self._total_orders,
            total_fills=self._total_fills,
            portfolio_summary=summary,
        )

        self._state = EngineState.STOPPED

    # ------------------------------------------------------------------
    # Backtest entry point
    # ------------------------------------------------------------------

    async def run_backtest(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
    ) -> dict[str, Any]:
        """
        Walk through historical bars deterministically.

        Each bar step feeds a growing window of history to every strategy
        (preventing look-ahead bias). The async interface is maintained for
        uniformity with live mode, but no real I/O occurs.

        Parameters
        ----------
        bars_by_symbol :
            Pre-fetched OHLCV bars keyed by symbol. Each list must be
            sorted by timestamp ascending.

        Returns
        -------
        dict[str, Any]
            Portfolio summary at the end of the backtest.

        Raises
        ------
        RuntimeError
            If run mode is not BACKTEST.
        ValueError
            If no bars are provided for any configured symbol.
        """
        if self._run_mode != RunMode.BACKTEST:
            raise RuntimeError(
                f"run_backtest() requires RunMode.BACKTEST, "
                f"got {self._run_mode.value}"
            )

        if self._state != EngineState.RUNNING:
            raise RuntimeError(
                f"run_backtest() requires engine state RUNNING "
                f"(call start() first), got {self._state.value}"
            )

        # Validate that bars exist for all symbols
        for symbol in self._symbols:
            if symbol not in bars_by_symbol or not bars_by_symbol[symbol]:
                raise ValueError(
                    f"No bars provided for symbol {symbol}"
                )

        # Determine the number of bars to process (shortest series)
        num_bars = min(len(bars_by_symbol[s]) for s in self._symbols)

        self._log.info(
            "engine.backtest_starting",
            total_bars=num_bars,
            warmup_bars=self._warmup_bars,
        )

        # Walk forward bar by bar
        for bar_idx in range(num_bars):
            if self._stop_event.is_set():
                self._log.info(
                    "engine.backtest_stopped_early",
                    bar_index=bar_idx,
                )
                break

            # Build the current bar snapshot and the growing history
            current_bars: dict[str, OHLCVBar] = {}
            history_by_symbol: dict[str, list[OHLCVBar]] = {}

            for symbol in self._symbols:
                bar = bars_by_symbol[symbol][bar_idx]
                current_bars[symbol] = bar
                # Growing window: all bars up to and including current
                history_by_symbol[symbol] = bars_by_symbol[symbol][
                    : bar_idx + 1
                ]

            # Update last prices on the paper execution engine
            self._update_engine_prices(current_bars)

            # Skip strategy calls during warmup, but still update prices
            if bar_idx < self._warmup_bars:
                # Update market prices in portfolio during warmup
                prices = {
                    s: bar.close for s, bar in current_bars.items()
                }
                self._portfolio.update_market_prices(prices)
                continue

            # Process this bar
            await self._process_bar(current_bars, history_by_symbol)

        self._log.info(
            "engine.backtest_complete",
            bars_processed=self._bar_count,
            total_signals=self._total_signals,
            total_orders=self._total_orders,
        )

        return self._portfolio.get_summary()

    # ------------------------------------------------------------------
    # Paper / Live loop entry point
    # ------------------------------------------------------------------

    async def run_live_loop(self) -> None:
        """
        Poll for new bars on each timeframe interval and process them.

        Runs until ``stop()`` is called or the stop event is set. Each
        iteration fetches the latest bar for every symbol, appends it to
        the rolling window, and processes it through the strategy pipeline.

        Raises
        ------
        RuntimeError
            If run mode is not PAPER or LIVE.
        """
        if self._run_mode not in (RunMode.PAPER, RunMode.LIVE):
            raise RuntimeError(
                f"run_live_loop() requires RunMode.PAPER or LIVE, "
                f"got {self._run_mode.value}"
            )

        if self._state != EngineState.RUNNING:
            raise RuntimeError(
                f"run_live_loop() requires engine state RUNNING "
                f"(call start() first), got {self._state.value}"
            )

        self._log.info(
            "engine.live_loop_starting",
            poll_interval_seconds=self._poll_interval,
            warmup_bars=self._warmup_bars,
        )

        # Initial warmup: fetch recent bars for each symbol
        await self._warmup_bar_windows()

        # Main polling loop
        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                await self._poll_and_process()
            except Exception:
                self._log.exception("engine.live_loop_iteration_error")

            # Sleep until the next candle interval
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self._poll_interval - elapsed)

            if sleep_time > 0:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=sleep_time,
                    )
                except asyncio.TimeoutError:
                    # Normal: timeout means the stop event was not set
                    pass

        self._log.info("engine.live_loop_exited")

    # ------------------------------------------------------------------
    # Core bar processing
    # ------------------------------------------------------------------

    async def _process_bar(
        self,
        current_bars: dict[str, OHLCVBar],
        history_by_symbol: dict[str, list[OHLCVBar]],
    ) -> None:
        """
        Process a single bar across all strategies and symbols.

        This is the inner loop that drives the entire trading pipeline
        on each new candle.

        Parameters
        ----------
        current_bars :
            The latest bar for each symbol.
        history_by_symbol :
            Full bar history per symbol up to and including the current bar.
        """
        bar_start = time.monotonic()

        # Get a representative timestamp for logging
        first_bar = next(iter(current_bars.values()))
        bar_timestamp = first_bar.timestamp

        # 1. Update market prices in portfolio
        prices = {s: bar.close for s, bar in current_bars.items()}
        self._portfolio.update_market_prices(prices)

        # 2. Tick risk manager cooldown
        self._risk_manager.tick_cooldown()

        # 3. In LIVE mode, check kill switch before processing.
        # NOTE: PAPER mode deliberately continues through kill-switch events
        # so strategies can still be monitored and signals recorded. The paper
        # execution engine's pre_trade_check will block actual order placement.
        if self._run_mode == RunMode.LIVE and self._risk_manager.kill_switch_active:
            self._log.warning(
                "engine.bar_skipped_kill_switch",
                bar_timestamp=str(bar_timestamp),
            )
            return

        # Count only bars that reach strategy processing
        self._bar_count += 1

        # 4. For each strategy: call on_bar and process resulting signals
        bar_signals: list[Signal] = []
        bar_orders: int = 0
        bar_fills: int = 0

        for strategy in self._strategies:
            try:
                signals = self._call_strategy_on_bar(
                    strategy, history_by_symbol
                )
                bar_signals.extend(signals)
            except Exception:
                self._log.exception(
                    "engine.strategy_on_bar_error",
                    strategy_id=strategy.strategy_id,
                    bar_index=self._bar_count,
                )
                continue

        self._total_signals += len(bar_signals)

        # 5. Process each signal through the execution engine
        for signal in bar_signals:
            try:
                orders = await self._execution_engine.process_signal(signal)
                bar_orders += len(orders)

                # Route fills to portfolio and risk manager
                for order in orders:
                    fills = await self._execution_engine.get_fills(
                        order.order_id
                    )
                    bar_fills += len(fills)

                    for fill in fills:
                        symbol_bar = current_bars.get(fill.symbol)
                        if symbol_bar is None:
                            self._log.error(
                                "engine.fill_symbol_not_in_current_bars",
                                fill_id=str(fill.fill_id),
                                fill_symbol=fill.symbol,
                                available_symbols=list(current_bars.keys()),
                            )
                            continue
                        current_price = symbol_bar.close

                        # Capture position BEFORE fill for trade recording
                        pre_fill_pos = self._portfolio.get_position(fill.symbol)

                        self._portfolio.update_position(fill, current_price)

                        # Record trade if this fill closed/reduced a position
                        self._record_trade_if_closed(
                            fill=fill,
                            pre_fill_position=pre_fill_pos,
                            strategy_id=signal.strategy_id,
                        )

                        # Determine if this fill closed a position (for risk
                        # manager loss tracking). A SELL fill on a position
                        # that is now flat indicates a completed trade.
                        self._route_fill_to_risk_manager(fill, current_price)

            except Exception:
                self._log.exception(
                    "engine.signal_processing_error",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                )
                continue

        self._total_orders += bar_orders
        self._total_fills += bar_fills

        # 6. Check resting orders (paper engine limit order support)
        await self._check_resting_orders(current_bars)

        # 7. Log bar summary
        bar_elapsed_ms = (time.monotonic() - bar_start) * 1000
        self._log.debug(
            "engine.bar_processed",
            bar_index=self._bar_count,
            bar_timestamp=str(bar_timestamp),
            signals=len(bar_signals),
            orders=bar_orders,
            fills=bar_fills,
            elapsed_ms=round(bar_elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Strategy invocation
    # ------------------------------------------------------------------

    def _call_strategy_on_bar(
        self,
        strategy: BaseStrategy,
        history_by_symbol: dict[str, list[OHLCVBar]],
    ) -> list[Signal]:
        """
        Call a strategy's ``on_bar`` for each symbol and collect signals.

        Parameters
        ----------
        strategy :
            The strategy to invoke.
        history_by_symbol :
            Bar history for each symbol.

        Returns
        -------
        list[Signal]
            Signals produced by this strategy across all symbols.
        """
        all_signals: list[Signal] = []

        for symbol in self._symbols:
            bars = history_by_symbol.get(symbol, [])
            if not bars:
                continue

            signals = strategy.on_bar(bars)

            if signals:
                all_signals.extend(signals)

        return all_signals

    # ------------------------------------------------------------------
    # Fill routing
    # ------------------------------------------------------------------

    def _route_fill_to_risk_manager(
        self,
        fill: Fill,
        current_price: Decimal,
    ) -> None:
        """
        Route fill information to the risk manager for loss-streak tracking.

        This method examines the portfolio's position snapshot to determine
        whether the fill resulted in a closed (or partially closed) position
        and whether the trade was profitable.

        Parameters
        ----------
        fill :
            The fill event.
        current_price :
            Current market price for the fill's symbol.
        """
        # Only SELL fills can close positions (spot-only MVP)
        if fill.side != OrderSide.SELL:
            return

        # Check the position state after the fill has been applied to
        # portfolio. If the position is flat, a round trip was completed.
        position = self._portfolio.get_position(fill.symbol)
        if position is not None and position.is_flat:
            realised_pnl = position.realised_pnl
            is_loss = realised_pnl < Decimal("0")
            try:
                self._risk_manager.update_after_fill(
                    realised_pnl=realised_pnl,
                    is_loss=is_loss,
                )
            except Exception:
                self._log.exception(
                    "engine.risk_update_after_fill_error",
                    symbol=fill.symbol,
                )

    def _record_trade_if_closed(
        self,
        fill: Fill,
        pre_fill_position: Position | None,
        strategy_id: str,
    ) -> None:
        """
        Detect round-trip completion and record a TradeResult.

        Called after update_position() has applied the fill. Compares the
        pre-fill position state with the post-fill state to determine
        whether a position was fully or partially closed.

        Parameters
        ----------
        fill :
            The fill event that was just applied.
        pre_fill_position :
            The position snapshot captured BEFORE update_position() was called.
            None if no position existed for this symbol.
        strategy_id :
            The strategy that generated the signal leading to this fill.
        """
        # Only SELL fills can close long positions (spot-only MVP)
        if fill.side != OrderSide.SELL:
            return
        # No pre-existing position to close
        if pre_fill_position is None or pre_fill_position.is_flat:
            return

        closed_qty = min(fill.quantity, pre_fill_position.quantity)
        if closed_qty <= Decimal("0"):
            return

        # PnL for the closed portion
        pnl = (fill.price - pre_fill_position.average_entry_price) * closed_qty - fill.fee

        # Total fees for this trade: exit fill fee only.
        # Entry fees are already embedded in average_entry_price (all-in cost
        # basis), so adding them again would double-count.
        total_fees = fill.fee

        if self._run_id is None:
            self._log.error("engine.trade_record_no_run_id", symbol=fill.symbol)
            return

        now = datetime.now(tz=UTC)

        try:
            trade = TradeResult(
                run_id=self._run_id,
                symbol=fill.symbol,
                side=OrderSide.BUY,  # Opening side for long position (spot-only)
                # entry_price is the all-in cost basis (includes entry fees),
                # not the raw execution price. Matches portfolio VWAP calculation.
                entry_price=pre_fill_position.average_entry_price,
                exit_price=fill.price,
                quantity=closed_qty,
                realised_pnl=pnl,
                total_fees=total_fees,
                entry_at=pre_fill_position.opened_at,
                exit_at=now,
                strategy_id=strategy_id,
            )
            self._portfolio.record_trade(trade)
            self._log.info(
                "engine.trade_recorded",
                trade_id=str(trade.trade_id),
                symbol=trade.symbol,
                pnl=str(trade.realised_pnl),
                quantity=str(trade.quantity),
            )
        except Exception:
            self._log.exception(
                "engine.trade_record_error",
                symbol=fill.symbol,
            )

    # ------------------------------------------------------------------
    # Resting order check (paper engine)
    # ------------------------------------------------------------------

    async def _check_resting_orders(
        self,
        current_bars: dict[str, OHLCVBar],
    ) -> None:
        """
        Check and fill resting limit orders against current bar prices.

        Only applicable to PaperExecutionEngine which exposes
        ``check_resting_orders(symbol, price)``.

        Parameters
        ----------
        current_bars :
            Latest bar for each symbol.
        """
        check_fn = getattr(
            self._execution_engine, "check_resting_orders", None
        )
        if check_fn is None:
            return

        for symbol, bar in current_bars.items():
            try:
                filled_orders = await check_fn(symbol, bar.close)
                if filled_orders:
                    for order in filled_orders:
                        fills = await self._execution_engine.get_fills(
                            order.order_id
                        )
                        for fill in fills:
                            pre_fill_pos = self._portfolio.get_position(
                                fill.symbol
                            )
                            self._portfolio.update_position(
                                fill, bar.close
                            )
                            # TODO: track Order→strategy_id mapping for
                            # correct multi-strategy attribution on resting fills.
                            self._record_trade_if_closed(
                                fill=fill,
                                pre_fill_position=pre_fill_pos,
                                strategy_id=self._strategies[0].strategy_id,
                            )
                            self._route_fill_to_risk_manager(
                                fill, bar.close
                            )
                        self._total_fills += len(fills)
                    self._total_orders += len(filled_orders)
            except Exception:
                self._log.exception(
                    "engine.resting_order_check_error",
                    symbol=symbol,
                )

    # ------------------------------------------------------------------
    # Engine price updates (paper engine)
    # ------------------------------------------------------------------

    def _update_engine_prices(
        self,
        current_bars: dict[str, OHLCVBar],
    ) -> None:
        """
        Update last-known prices on the execution engine.

        PaperExecutionEngine requires ``set_last_price()`` to be called
        before signal processing. This method handles the dispatch.

        Parameters
        ----------
        current_bars :
            Latest bar for each symbol.
        """
        set_price_fn = getattr(
            self._execution_engine, "set_last_price", None
        )
        if set_price_fn is None:
            return

        for symbol, bar in current_bars.items():
            set_price_fn(symbol, bar.close)

    # ------------------------------------------------------------------
    # Paper/Live helpers
    # ------------------------------------------------------------------

    async def _warmup_bar_windows(self) -> None:
        """
        Fetch initial bar history for each symbol to satisfy strategy
        warm-up requirements.

        Uses ``market_data.fetch_ohlcv()`` with a limit equal to the
        configured warmup size.
        """
        fetch_limit = max(self._warmup_bars, 100)

        for symbol in self._symbols:
            try:
                bars = await self._market_data.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self._timeframe,
                    limit=fetch_limit,
                )
                self._bar_windows[symbol] = bars

                # Update engine prices with the most recent bar
                if bars:
                    set_price_fn = getattr(
                        self._execution_engine, "set_last_price", None
                    )
                    if set_price_fn is not None:
                        set_price_fn(symbol, bars[-1].close)

                self._log.info(
                    "engine.warmup_loaded",
                    symbol=symbol,
                    bars_loaded=len(bars),
                )
            except MarketDataError:
                self._log.exception(
                    "engine.warmup_fetch_failed",
                    symbol=symbol,
                )
            except Exception:
                self._log.exception(
                    "engine.warmup_unexpected_error",
                    symbol=symbol,
                )

    async def _poll_and_process(self) -> None:
        """
        Fetch the latest bar for each symbol, update the rolling window,
        and process the bar through the strategy pipeline.

        Deduplicates bars by checking whether the latest fetched bar's
        timestamp matches the last bar in the window.
        """
        current_bars: dict[str, OHLCVBar] = {}
        new_bar_found = False

        for symbol in self._symbols:
            try:
                latest_bar = await self._market_data.get_latest_bar(
                    symbol=symbol,
                    timeframe=self._timeframe,
                )

                window = self._bar_windows[symbol]

                # Deduplicate: only add if timestamp is newer
                if window and latest_bar.timestamp <= window[-1].timestamp:
                    # No new bar yet for this symbol
                    current_bars[symbol] = window[-1]
                    continue

                # Append new bar and trim to max window size
                window.append(latest_bar)
                if len(window) > self._max_bars_history:
                    self._bar_windows[symbol] = window[
                        -self._max_bars_history :
                    ]

                current_bars[symbol] = latest_bar
                new_bar_found = True

            except MarketDataError:
                self._log.exception(
                    "engine.poll_fetch_failed",
                    symbol=symbol,
                )
                # Use last known bar if available
                window = self._bar_windows.get(symbol, [])
                if window:
                    current_bars[symbol] = window[-1]
            except Exception:
                self._log.exception(
                    "engine.poll_unexpected_error",
                    symbol=symbol,
                )
                window = self._bar_windows.get(symbol, [])
                if window:
                    current_bars[symbol] = window[-1]

        # Only process if we got bars for all symbols and at least one is new
        if len(current_bars) < len(self._symbols):
            self._log.warning(
                "engine.poll_incomplete",
                symbols_received=len(current_bars),
                symbols_expected=len(self._symbols),
            )
            return

        if not new_bar_found:
            return

        # Update engine prices
        self._update_engine_prices(current_bars)

        # Build history windows for strategy calls
        history_by_symbol: dict[str, list[OHLCVBar]] = {
            s: list(self._bar_windows[s]) for s in self._symbols
        }

        # Process the bar
        await self._process_bar(current_bars, history_by_symbol)

    # ------------------------------------------------------------------
    # Summary / status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """
        Return the current engine status as a serialisable dictionary.

        Includes engine state, run metrics, and portfolio summary.

        Returns
        -------
        dict[str, Any]
            Engine status snapshot.
        """
        result: dict[str, Any] = {
            "state": self._state.value,
            "run_id": self._run_id,
            "run_mode": self._run_mode.value,
            "timeframe": self._timeframe.value,
            "symbols": self._symbols,
            "strategies": [s.strategy_id for s in self._strategies],
            "bar_count": self._bar_count,
            "total_signals": self._total_signals,
            "total_orders": self._total_orders,
            "total_fills": self._total_fills,
        }

        if self._state in (EngineState.RUNNING, EngineState.STOPPED):
            result["portfolio_summary"] = self._portfolio.get_summary()

        return result

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StrategyEngine("
            f"state={self._state.value!r}, "
            f"run_id={self._run_id!r}, "
            f"mode={self._run_mode.value!r}, "
            f"strategies={len(self._strategies)}, "
            f"symbols={self._symbols}, "
            f"bars={self._bar_count})"
        )
