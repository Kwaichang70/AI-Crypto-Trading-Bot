"""
tests/unit/test_strategy_engine_bar_loop.py
--------------------------------------------
Bar-loop unit tests for StrategyEngine.

Module under test
-----------------
packages/trading/strategy_engine.py

Test coverage
-------------
- _process_bar(): portfolio price update, risk tick, kill-switch behaviour,
  bar_count increment, strategy.on_bar dispatch, signal routing, fill routing,
  exception resilience, signal accumulation
- _call_strategy_on_bar(): correct bars passed per symbol, empty-bar skip,
  multi-symbol aggregation, None / empty return handling
- _route_fill_to_risk_manager(): BUY fill skipped, SELL on open position skipped,
  SELL on flat position calls update_after_fill with correct PnL / is_loss,
  exception in risk_manager does not propagate
- _check_resting_orders(): no-op when method absent, calls check_fn per symbol,
  routes fills to portfolio, exception resilience
- run_backtest() execution: warmup-bar skipping, portfolio summary return,
  early termination via stop_event, price updates during warmup

Async note
----------
pyproject.toml sets asyncio_mode = "auto"; no @pytest.mark.asyncio needed.

Design note on check_resting_orders
------------------------------------
The _make_engine() factory uses AsyncMock() for the execution engine.
AsyncMock auto-creates async attributes on first access, so
``getattr(engine._execution_engine, "check_resting_orders", None)``
returns an AsyncMock rather than None.  Tests that want the no-op path
(check_fn is None) explicitly set ``execution.check_resting_orders = None``.
Tests that exercise the path where resting orders are NOT involved in the
main assertion also set this to None to prevent the AsyncMock from generating
unawaited-coroutine warnings from the auto-created default return value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from common.models import OHLCVBar
from common.types import OrderSide, OrderType, RunMode, SignalDirection, TimeFrame
from trading.models import Fill, Order, Signal
from trading.strategy_engine import EngineState, StrategyEngine


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    run_mode: RunMode = RunMode.BACKTEST,
    symbols: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[StrategyEngine, dict[str, Any]]:
    """
    Create a StrategyEngine with fully mocked dependencies.

    Duplicated here so this test module is self-contained and independent
    of the lifecycle test module's private helpers.

    Returns the engine and a mocks dict keyed by dependency name.
    """
    strategy = MagicMock()
    strategy.strategy_id = "test_strategy"
    strategy.min_bars_required = 20
    strategy.on_start = MagicMock(return_value=None)
    strategy.on_stop = MagicMock(return_value=None)

    execution = AsyncMock()
    execution.on_start = AsyncMock(return_value=None)
    execution.on_stop = AsyncMock(return_value=None)
    execution.get_open_orders = MagicMock(return_value=[])
    execution.cancel_order = AsyncMock(return_value=None)

    market_data = AsyncMock()
    market_data.connect = AsyncMock(return_value=None)
    market_data.close = AsyncMock(return_value=None)

    risk_manager = MagicMock()
    risk_manager.kill_switch_active = False
    risk_manager.tick_cooldown = MagicMock(return_value=None)

    portfolio = MagicMock()
    portfolio.get_summary = MagicMock(return_value={
        "current_equity": "10000",
        "total_trades": 0,
    })

    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=symbols or ["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=run_mode,
        config=config,
    )

    mocks: dict[str, Any] = {
        "strategy": strategy,
        "execution": execution,
        "market_data": market_data,
        "risk_manager": risk_manager,
        "portfolio": portfolio,
    }
    return engine, mocks


def _make_bar(
    *,
    symbol: str = "BTC/USDT",
    close: str | Decimal = "105",
    open_: str | Decimal | None = None,
    high: str | Decimal | None = None,
    low: str | Decimal | None = None,
    volume: str | Decimal = "1000",
    timestamp: datetime | None = None,
) -> OHLCVBar:
    """
    Construct a minimal OHLCVBar for bar-loop tests.

    Defaults represent a single BTC/USDT bar.  ``open_``, ``high``, and
    ``low`` default to values derived from ``close`` so that the OHLCV
    consistency constraint (low <= open/close <= high) is always satisfied
    without callers having to specify every field.

    Parameters
    ----------
    symbol:
        Trading pair identifier.
    close:
        Close price.  Drives the default derivation of open_, high, and low
        when those are not provided.
    open_:
        Open price.  Defaults to ``close`` (flat candle open).
    high:
        High price.  Defaults to ``close * 1.01`` (1% above close).
    low:
        Low price.  Defaults to ``close * 0.99`` (1% below close).
    volume:
        Volume in base asset.
    timestamp:
        Bar open time in UTC.  Defaults to 2024-01-01 00:00:00 UTC.
    """
    ts = timestamp or datetime(2024, 1, 1, tzinfo=UTC)
    close_d = Decimal(str(close))
    open_d = Decimal(str(open_)) if open_ is not None else close_d
    high_d = Decimal(str(high)) if high is not None else (close_d * Decimal("1.01")).quantize(Decimal("0.01"))
    low_d = Decimal(str(low)) if low is not None else (close_d * Decimal("0.99")).quantize(Decimal("0.01"))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=ts,
        open=open_d,
        high=high_d,
        low=low_d,
        close=close_d,
        volume=Decimal(str(volume)),
    )


def _make_fill(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: str = "0.1",
    price: str = "105",
) -> Fill:
    """
    Construct a Fill for fill-routing tests.

    Parameters
    ----------
    symbol:
        Trading pair for the fill.
    side:
        BUY or SELL.
    quantity:
        Quantity in base asset.
    price:
        Execution price.
    """
    qty = Decimal(quantity)
    prc = Decimal(price)
    fee = qty * prc * Decimal("0.001")
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=qty,
        price=prc,
        fee=fee,
        fee_currency="USDT",
    )


def _make_order(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    run_id: str = "run-001",
) -> Order:
    """
    Construct a MARKET Order for signal-routing tests.

    Parameters
    ----------
    symbol:
        Trading pair for the order.
    side:
        BUY or SELL.
    run_id:
        Run identifier to embed in the client_order_id.
    """
    return Order(
        client_order_id=f"{run_id}-{uuid4().hex[:12]}",
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
    )


# ===========================================================================
# _process_bar
# ===========================================================================


class TestProcessBar:
    """
    Tests for StrategyEngine._process_bar().

    Each test starts the engine before invoking _process_bar directly so
    that the run_id and internal state are correctly initialised.

    The execution mock's check_resting_orders is set to None in tests that
    do not exercise the resting-order path.  This prevents AsyncMock from
    auto-creating a coroutine stub that generates unawaited-coroutine warnings
    when the _check_resting_orders helper iterates mock fill objects.
    """

    async def test_updates_portfolio_market_prices(self) -> None:
        """
        _process_bar() must call portfolio.update_market_prices with a dict
        mapping each symbol to the bar's close price.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        bar = _make_bar(close="105")
        current_bars = {"BTC/USDT": bar}
        history_by_symbol = {"BTC/USDT": [bar]}

        await engine._process_bar(current_bars, history_by_symbol)

        mocks["portfolio"].update_market_prices.assert_called_once_with(
            {"BTC/USDT": Decimal("105")}
        )

    async def test_ticks_risk_manager_cooldown_once(self) -> None:
        """
        _process_bar() must call risk_manager.tick_cooldown exactly once per
        bar invocation, regardless of whether signals are produced.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["risk_manager"].tick_cooldown.assert_called_once()

    async def test_kill_switch_in_live_mode_returns_early(self) -> None:
        """
        In LIVE mode, when risk_manager.kill_switch_active is True, _process_bar()
        must return before reaching strategy invocation and must NOT increment
        bar_count.

        The kill switch check comes after tick_cooldown but before bar_count
        increment and strategy dispatch.
        """
        engine, mocks = _make_engine(run_mode=RunMode.LIVE)
        await engine.start("run-001")
        mocks["risk_manager"].kill_switch_active = True
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        assert engine.bar_count == 0
        mocks["strategy"].on_bar.assert_not_called()

    async def test_kill_switch_in_paper_mode_does_not_skip(self) -> None:
        """
        In PAPER mode, kill_switch_active being True must NOT cause an early
        return.  bar_count must still increment and strategy.on_bar must be
        called.

        The paper execution engine's pre_trade_check blocks orders; the engine
        itself continues so strategies can be monitored.
        """
        engine, mocks = _make_engine(run_mode=RunMode.PAPER)
        await engine.start("run-001")
        mocks["risk_manager"].kill_switch_active = True
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        assert engine.bar_count == 1
        mocks["strategy"].on_bar.assert_called_once()

    async def test_bar_count_increments_by_one(self) -> None:
        """
        bar_count must increase by exactly 1 for each _process_bar() call that
        passes the kill-switch guard.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        assert engine.bar_count == 0

        bar1 = _make_bar(timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC))
        await engine._process_bar({"BTC/USDT": bar1}, {"BTC/USDT": [bar1]})
        assert engine.bar_count == 1

        bar2 = _make_bar(timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC))
        await engine._process_bar(
            {"BTC/USDT": bar2}, {"BTC/USDT": [bar1, bar2]}
        )
        assert engine.bar_count == 2

    async def test_calls_strategy_on_bar_for_configured_symbol(self) -> None:
        """
        _process_bar() must invoke strategy.on_bar with the bar history for
        the configured symbol.  No signals produced means no further processing.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        history = [bar]
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": history})

        mocks["strategy"].on_bar.assert_called_once_with(history)

    async def test_signal_routed_to_execution_process_signal(self) -> None:
        """
        When strategy.on_bar returns a Signal, _process_bar() must await
        execution_engine.process_signal with that signal.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        sig = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        mocks["strategy"].on_bar = MagicMock(return_value=[sig])
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["execution"].process_signal.assert_awaited_once_with(sig)

    async def test_fill_routed_to_portfolio_update_position(self) -> None:
        """
        When execution_engine returns an order and get_fills returns a Fill,
        _process_bar() must call portfolio.update_position with that fill and
        the bar's close price.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        sig = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        mocks["strategy"].on_bar = MagicMock(return_value=[sig])

        order = _make_order()
        fill = _make_fill(side=OrderSide.BUY)

        mocks["execution"].process_signal = AsyncMock(return_value=[order])
        mocks["execution"].get_fills = AsyncMock(return_value=[fill])
        mocks["execution"].check_resting_orders = None
        # BUY fill: _route_fill_to_risk_manager returns early, no position needed
        mocks["portfolio"].get_position = MagicMock(return_value=None)

        bar = _make_bar(close="105")
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["portfolio"].update_position.assert_called_once_with(
            fill, Decimal("105")
        )

    async def test_strategy_exception_does_not_crash_bar_processing(self) -> None:
        """
        When strategy.on_bar raises an exception, _process_bar() must catch it,
        log it, and continue processing.  bar_count must still increment.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(
            side_effect=RuntimeError("strategy exploded")
        )
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        # Must not raise
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        assert engine.bar_count == 1

    async def test_signal_processing_exception_does_not_crash_bar(self) -> None:
        """
        When execution_engine.process_signal raises, _process_bar() must catch it
        and continue to completion.  bar_count and _total_signals must still
        be updated correctly.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        sig = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        mocks["strategy"].on_bar = MagicMock(return_value=[sig])
        mocks["execution"].process_signal = AsyncMock(
            side_effect=RuntimeError("execution failed")
        )
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        # The signal was counted before execution was attempted
        assert engine.get_status()["total_signals"] == 1
        assert engine.bar_count == 1

    async def test_total_signals_accumulates_correctly(self) -> None:
        """
        _total_signals must accumulate across multiple _process_bar() calls.
        Two bars each producing one signal should yield _total_signals == 2.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        sig = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        mocks["strategy"].on_bar = MagicMock(return_value=[sig])
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar1 = _make_bar(timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC))
        bar2 = _make_bar(timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC))

        await engine._process_bar({"BTC/USDT": bar1}, {"BTC/USDT": [bar1]})
        await engine._process_bar(
            {"BTC/USDT": bar2}, {"BTC/USDT": [bar1, bar2]}
        )

        assert engine._total_signals == 2

    async def test_multiple_signals_from_one_strategy_all_processed(self) -> None:
        """
        When a strategy returns multiple signals in one bar, every signal must
        be submitted to execution_engine.process_signal.

        This uses two symbols so the strategy can generate one signal per
        symbol on a single bar.
        """
        engine, mocks = _make_engine(symbols=["BTC/USDT", "ETH/USDT"])
        await engine.start("run-001")

        sig_btc = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("500"),
        )
        sig_eth = Signal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        # on_bar is called twice (once per symbol); each call returns one signal
        mocks["strategy"].on_bar = MagicMock(
            side_effect=[[sig_btc], [sig_eth]]
        )
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar_btc = _make_bar(symbol="BTC/USDT", close="105")
        bar_eth = _make_bar(symbol="ETH/USDT", close="3000")
        current_bars = {"BTC/USDT": bar_btc, "ETH/USDT": bar_eth}
        history = {"BTC/USDT": [bar_btc], "ETH/USDT": [bar_eth]}

        await engine._process_bar(current_bars, history)

        assert mocks["execution"].process_signal.await_count == 2
        assert engine._total_signals == 2


# ===========================================================================
# _call_strategy_on_bar (tested via _process_bar)
# ===========================================================================


class TestCallStrategyOnBar:
    """
    Tests for StrategyEngine._call_strategy_on_bar() behaviour.

    Since _call_strategy_on_bar is a private synchronous method, tests drive
    it through _process_bar to exercise the realistic call path.

    All tests set execution.check_resting_orders = None to disable the
    resting-order path and keep assertions focused on strategy dispatch only.
    """

    async def test_on_bar_receives_correct_history_for_symbol(self) -> None:
        """
        strategy.on_bar must be called with the full growing history slice for
        the configured symbol, not just the current bar.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar1 = _make_bar(timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC))
        bar2 = _make_bar(timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC))
        history = [bar1, bar2]

        await engine._process_bar({"BTC/USDT": bar2}, {"BTC/USDT": history})

        mocks["strategy"].on_bar.assert_called_once_with(history)

    async def test_symbol_with_empty_history_is_skipped(self) -> None:
        """
        When history_by_symbol contains an empty list for a symbol,
        strategy.on_bar must NOT be called for that symbol.

        This prevents strategies from being called with zero bars, which could
        cause index errors in indicator calculations.
        """
        engine, mocks = _make_engine(symbols=["BTC/USDT", "ETH/USDT"])
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar_btc = _make_bar(symbol="BTC/USDT", close="105")
        bar_eth = _make_bar(symbol="ETH/USDT", close="3000")
        current_bars = {"BTC/USDT": bar_btc, "ETH/USDT": bar_eth}
        # ETH history is empty — on_bar must be called only once (for BTC)
        history = {"BTC/USDT": [bar_btc], "ETH/USDT": []}

        await engine._process_bar(current_bars, history)

        assert mocks["strategy"].on_bar.call_count == 1
        mocks["strategy"].on_bar.assert_called_with([bar_btc])

    async def test_signals_collected_from_multiple_symbols(self) -> None:
        """
        Signals from all symbols must be aggregated into a single list and all
        submitted to the execution engine.
        """
        engine, mocks = _make_engine(symbols=["BTC/USDT", "ETH/USDT"])
        await engine.start("run-001")

        sig_btc = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("500"),
        )
        sig_eth = Signal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("300"),
        )
        # Returns one signal per on_bar call (one per symbol)
        mocks["strategy"].on_bar = MagicMock(side_effect=[[sig_btc], [sig_eth]])
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar_btc = _make_bar(symbol="BTC/USDT", close="105")
        bar_eth = _make_bar(symbol="ETH/USDT", close="3000")
        current_bars = {"BTC/USDT": bar_btc, "ETH/USDT": bar_eth}
        history = {"BTC/USDT": [bar_btc], "ETH/USDT": [bar_eth]}

        await engine._process_bar(current_bars, history)

        assert engine._total_signals == 2

    async def test_empty_list_return_from_on_bar_produces_no_signals(self) -> None:
        """
        When strategy.on_bar returns an empty list, _call_strategy_on_bar must
        contribute zero signals to the bar's signal list.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["execution"].process_signal.assert_not_awaited()
        assert engine._total_signals == 0

    async def test_none_return_from_on_bar_treated_as_no_signal(self) -> None:
        """
        When strategy.on_bar returns None (a common strategy shorthand for HOLD),
        the engine must treat it as an empty signal list and not call
        process_signal.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=None)
        mocks["execution"].process_signal = AsyncMock(return_value=[])
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["execution"].process_signal.assert_not_awaited()
        assert engine._total_signals == 0


# ===========================================================================
# _route_fill_to_risk_manager
# ===========================================================================


class TestRouteFillToRiskManager:
    """
    Tests for StrategyEngine._route_fill_to_risk_manager().

    _route_fill_to_risk_manager is called directly (it is a synchronous
    private method) so tests can be precise about the state of the portfolio
    mock before the call.
    """

    async def test_buy_fill_does_not_call_update_after_fill(self) -> None:
        """
        A BUY fill must cause an immediate return — risk_manager.update_after_fill
        must NOT be called because spot BUY fills cannot close a position.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        fill = _make_fill(side=OrderSide.BUY)
        engine._route_fill_to_risk_manager(fill, Decimal("105"))

        mocks["risk_manager"].update_after_fill.assert_not_called()

    async def test_sell_fill_on_open_position_does_not_call_update_after_fill(
        self,
    ) -> None:
        """
        A SELL fill where the resulting position is still open (is_flat == False)
        must NOT call risk_manager.update_after_fill.

        Only a fully closed (flat) position triggers risk loss-streak tracking.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        position = MagicMock()
        position.is_flat = False
        position.realised_pnl = Decimal("50")
        mocks["portfolio"].get_position = MagicMock(return_value=position)

        fill = _make_fill(side=OrderSide.SELL)
        engine._route_fill_to_risk_manager(fill, Decimal("105"))

        mocks["risk_manager"].update_after_fill.assert_not_called()

    async def test_sell_fill_on_flat_position_calls_update_after_fill(self) -> None:
        """
        When a SELL fill closes the position (is_flat == True), the engine must
        call risk_manager.update_after_fill with the correct realised_pnl and
        is_loss flag.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        position = MagicMock()
        position.is_flat = True
        position.realised_pnl = Decimal("75")
        mocks["portfolio"].get_position = MagicMock(return_value=position)

        fill = _make_fill(side=OrderSide.SELL)
        engine._route_fill_to_risk_manager(fill, Decimal("105"))

        mocks["risk_manager"].update_after_fill.assert_called_once_with(
            realised_pnl=Decimal("75"),
            is_loss=False,
        )

    async def test_negative_pnl_sets_is_loss_true(self) -> None:
        """
        When realised_pnl is negative, update_after_fill must be called with
        is_loss=True.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        position = MagicMock()
        position.is_flat = True
        position.realised_pnl = Decimal("-30")
        mocks["portfolio"].get_position = MagicMock(return_value=position)

        fill = _make_fill(side=OrderSide.SELL)
        engine._route_fill_to_risk_manager(fill, Decimal("105"))

        mocks["risk_manager"].update_after_fill.assert_called_once_with(
            realised_pnl=Decimal("-30"),
            is_loss=True,
        )

    async def test_zero_pnl_is_not_a_loss(self) -> None:
        """
        When realised_pnl is exactly zero, is_loss must be False.

        The production code uses ``realised_pnl < Decimal("0")``, so a zero
        PnL (break-even trade) must NOT be flagged as a loss.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        position = MagicMock()
        position.is_flat = True
        position.realised_pnl = Decimal("0")
        mocks["portfolio"].get_position = MagicMock(return_value=position)

        fill = _make_fill(side=OrderSide.SELL)
        engine._route_fill_to_risk_manager(fill, Decimal("105"))

        mocks["risk_manager"].update_after_fill.assert_called_once_with(
            realised_pnl=Decimal("0"),
            is_loss=False,
        )

    async def test_exception_in_update_after_fill_does_not_propagate(self) -> None:
        """
        If risk_manager.update_after_fill raises, _route_fill_to_risk_manager
        must catch the exception and return silently.

        This guards the bar loop from being crashed by a risk manager bug.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        position = MagicMock()
        position.is_flat = True
        position.realised_pnl = Decimal("10")
        mocks["portfolio"].get_position = MagicMock(return_value=position)
        mocks["risk_manager"].update_after_fill = MagicMock(
            side_effect=RuntimeError("risk manager error")
        )

        fill = _make_fill(side=OrderSide.SELL)
        # Must not raise
        engine._route_fill_to_risk_manager(fill, Decimal("105"))


# ===========================================================================
# _check_resting_orders
# ===========================================================================


class TestCheckRestingOrders:
    """
    Tests for StrategyEngine._check_resting_orders().

    Tests cover the getattr-based method dispatch, fill routing from resting
    orders, and exception resilience.
    """

    async def test_no_op_when_check_resting_orders_absent(self) -> None:
        """
        When the execution engine does not expose a check_resting_orders method
        (i.e. getattr returns None), _check_resting_orders must return
        immediately without any portfolio or risk manager calls.

        Setting the attribute to None on the mock replicates the behaviour of
        an execution engine that does not implement resting-order simulation.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        # Explicitly set to None so getattr(..., None) returns None
        mocks["execution"].check_resting_orders = None

        bar = _make_bar()
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["portfolio"].update_position.assert_not_called()

    async def test_calls_check_fn_with_symbol_and_close_price(self) -> None:
        """
        _check_resting_orders must call check_resting_orders(symbol, bar.close)
        for each symbol in current_bars.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        mocks["execution"].check_resting_orders = AsyncMock(return_value=[])

        bar = _make_bar(close="105")
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["execution"].check_resting_orders.assert_awaited_once_with(
            "BTC/USDT", Decimal("105")
        )

    async def test_fills_from_resting_orders_routed_to_portfolio(self) -> None:
        """
        When check_resting_orders returns filled orders, _check_resting_orders
        must retrieve fills via get_fills and call portfolio.update_position for
        each fill.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        order = _make_order(side=OrderSide.BUY)
        fill = _make_fill(side=OrderSide.BUY, price="105")

        mocks["execution"].check_resting_orders = AsyncMock(return_value=[order])
        mocks["execution"].get_fills = AsyncMock(return_value=[fill])
        # BUY fill: _route_fill_to_risk_manager returns early, no position needed
        mocks["portfolio"].get_position = MagicMock(return_value=None)

        bar = _make_bar(close="105")
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["portfolio"].update_position.assert_called_once_with(
            fill, Decimal("105")
        )
        assert engine._total_fills == 1
        assert engine._total_orders == 1

    async def test_calls_check_fn_for_each_symbol(self) -> None:
        """
        With two symbols in current_bars, check_resting_orders must be called
        once per symbol with the correct (symbol, close_price) arguments.

        This exercises the per-symbol loop at production line 760 and ensures
        a bug short-circuiting after the first symbol would be caught.
        """
        engine, mocks = _make_engine(symbols=["BTC/USDT", "ETH/USDT"])
        await engine.start("run-001")

        mocks["execution"].check_resting_orders = AsyncMock(return_value=[])

        bar_btc = _make_bar(symbol="BTC/USDT", close="105")
        bar_eth = _make_bar(symbol="ETH/USDT", close="3000")
        current_bars = {"BTC/USDT": bar_btc, "ETH/USDT": bar_eth}

        await engine._check_resting_orders(current_bars)

        assert mocks["execution"].check_resting_orders.await_count == 2
        calls = mocks["execution"].check_resting_orders.await_args_list
        call_args = {(c.args[0], c.args[1]) for c in calls}
        assert ("BTC/USDT", Decimal("105")) in call_args
        assert ("ETH/USDT", Decimal("3000")) in call_args

    async def test_exception_in_check_fn_does_not_crash(self) -> None:
        """
        If check_resting_orders raises for a symbol, the exception must be
        caught and logged.  The method must return normally.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        async def _failing_check(symbol: str, price: Decimal) -> list[Order]:
            raise RuntimeError("check_resting_orders failed")

        mocks["execution"].check_resting_orders = _failing_check

        bar = _make_bar()
        # Must not raise
        await engine._check_resting_orders({"BTC/USDT": bar})


# ===========================================================================
# run_backtest() execution
# ===========================================================================


class TestRunBacktestExecution:
    """
    Tests for the bar-walk execution path inside run_backtest().

    Tests verify warmup-bar skipping, portfolio summary return value, early
    termination via the stop_event, and price updates during warmup.

    All tests disable resting-order checking by setting check_resting_orders
    to None so that AsyncMock does not auto-generate fill objects that cause
    unawaited-coroutine warnings in unrelated assertion paths.
    """

    async def test_warmup_bars_skipped_for_strategy_calls(self) -> None:
        """
        With warmup_bars=2 and 4 total bars, strategy.on_bar must be called
        exactly 2 times (bars at index 2 and 3 only).

        Bars 0 and 1 are warmup bars — prices are updated but strategies are
        not invoked.
        """
        engine, mocks = _make_engine(config={"warmup_bars": 2})
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None
        mocks["execution"].set_last_price = MagicMock()

        bars = [
            _make_bar(timestamp=datetime(2024, 1, 1, i, tzinfo=UTC))
            for i in range(4)
        ]
        await engine.run_backtest(bars_by_symbol={"BTC/USDT": bars})

        assert mocks["strategy"].on_bar.call_count == 2
        assert engine.bar_count == 2

    async def test_returns_portfolio_get_summary_result(self) -> None:
        """
        run_backtest() must return the dict produced by portfolio.get_summary().

        This is the primary result the BacktestRunner uses to populate the
        run record in the database.
        """
        expected = {"current_equity": "12500", "total_trades": 3}
        engine, mocks = _make_engine(config={"warmup_bars": 0})
        await engine.start("run-001")
        mocks["portfolio"].get_summary = MagicMock(return_value=expected)
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None
        mocks["execution"].set_last_price = MagicMock()

        bar = _make_bar()
        result = await engine.run_backtest(bars_by_symbol={"BTC/USDT": [bar]})

        assert result == expected

    async def test_early_termination_via_stop_event(self) -> None:
        """
        When _stop_event is set before run_backtest is called, the loop must
        check is_set() at the top of each iteration and exit immediately
        without processing any bars.
        """
        engine, mocks = _make_engine(config={"warmup_bars": 0})
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None
        mocks["execution"].set_last_price = MagicMock()

        bars = [
            _make_bar(timestamp=datetime(2024, 1, 1, i, tzinfo=UTC))
            for i in range(5)
        ]

        # Set the stop event before run_backtest is called — the loop checks
        # _stop_event.is_set() at the top of each iteration, so it exits
        # without processing any bar.
        engine._stop_event.set()

        await engine.run_backtest(bars_by_symbol={"BTC/USDT": bars})

        assert mocks["strategy"].on_bar.call_count == 0
        assert engine.bar_count == 0

    async def test_portfolio_prices_updated_during_warmup_bars(self) -> None:
        """
        During warmup bars, portfolio.update_market_prices must still be
        called so that equity and drawdown tracking remain accurate even
        before strategies are active.

        With warmup_bars=2 and 3 bars total, update_market_prices must be
        called 3 times: 2 warmup calls (in the warmup branch) + 1 call
        inside _process_bar for the non-warmup bar.
        """
        engine, mocks = _make_engine(config={"warmup_bars": 2})
        await engine.start("run-001")
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].check_resting_orders = None
        mocks["execution"].set_last_price = MagicMock()

        bars = [
            _make_bar(timestamp=datetime(2024, 1, 1, i, tzinfo=UTC))
            for i in range(3)
        ]

        await engine.run_backtest(bars_by_symbol={"BTC/USDT": bars})

        assert mocks["portfolio"].update_market_prices.call_count == 3
