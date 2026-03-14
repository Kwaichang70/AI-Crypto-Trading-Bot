"""
tests/unit/test_trade_recording.py
------------------------------------
Unit tests for StrategyEngine._record_trade_if_closed().

Module under test
-----------------
packages/trading/strategy_engine.py

Test coverage
-------------
- _record_trade_if_closed(): BUY fill skip, None pre-position skip,
  flat pre-position skip, full close recording, partial close recording,
  negative PnL detection, fee attribution, exception resilience
- _process_bar() integration: BUY+SELL sequence, BUY-only no trade,
  strategy_id propagation, multi-symbol recording, VWAP averaged entry
- _check_resting_orders() integration: resting SELL triggers record,
  resting BUY skip, strategy_id attribution from strategies[0]

Async note
----------
pyproject.toml sets asyncio_mode = "auto"; no @pytest.mark.asyncio needed.

Design note on Position objects
--------------------------------
Position is a Pydantic BaseModel, so we construct it directly with
field kwargs rather than using __new__() or SimpleNamespace. This
gives us real .is_flat property logic and proper Decimal arithmetic
throughout the tests.

Design note on _make_engine fixture pattern
--------------------------------------------
Each test class follows the same pattern as test_strategy_engine_bar_loop.py:
the _make_engine() factory returns (engine, mocks) where mocks is a dict of
all injected dependencies. This keeps tests self-contained with no
cross-test state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest

from common.models import OHLCVBar
from common.types import OrderSide, OrderType, RunMode, SignalDirection, TimeFrame
from trading.models import Fill, Order, Position, Signal, TradeResult
from trading.strategy_engine import StrategyEngine


# ---------------------------------------------------------------------------
# Factory helpers — mirrors test_strategy_engine_bar_loop.py conventions
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    run_mode: RunMode = RunMode.BACKTEST,
    symbols: list[str] | None = None,
    config: dict[str, Any] | None = None,
    strategy_id: str = "test_strategy",
) -> tuple[StrategyEngine, dict[str, Any]]:
    """
    Create a StrategyEngine with fully mocked dependencies.

    Returns the engine (not yet started) and a mocks dict keyed by
    dependency name.  Callers must call ``await engine.start(run_id)``
    before exercising bar-loop methods.
    """
    strategy = MagicMock()
    strategy.strategy_id = strategy_id
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
    risk_manager.update_after_fill = MagicMock(return_value=None)

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
    """Construct a minimal OHLCVBar; all OHLCV constraints satisfied by default."""
    ts = timestamp or datetime(2024, 1, 1, tzinfo=UTC)
    close_d = Decimal(str(close))
    open_d = Decimal(str(open_)) if open_ is not None else close_d
    high_d = (
        Decimal(str(high)) if high is not None
        else (close_d * Decimal("1.01")).quantize(Decimal("0.01"))
    )
    low_d = (
        Decimal(str(low)) if low is not None
        else (close_d * Decimal("0.99")).quantize(Decimal("0.01"))
    )
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
    quantity: str = "1.0",
    price: str = "100",
    fee: str | None = None,
) -> Fill:
    """
    Construct a Fill for trade-recording tests.

    Fee defaults to 0.1% of notional if not provided explicitly.
    """
    qty = Decimal(quantity)
    prc = Decimal(price)
    computed_fee = qty * prc * Decimal("0.001")
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=qty,
        price=prc,
        fee=Decimal(fee) if fee is not None else computed_fee,
        fee_currency="USDT",
    )


def _make_position(
    *,
    symbol: str = "BTC/USDT",
    run_id: str = "run-001",
    quantity: str = "1.0",
    average_entry_price: str = "100",
    current_price: str = "100",
    realised_pnl: str = "0",
    opened_at: datetime | None = None,
) -> Position:
    """
    Construct a real Position Pydantic model.

    Uses the actual Position class so that .is_flat, .quantity, and
    .average_entry_price have correct semantics throughout.
    """
    return Position(
        symbol=symbol,
        run_id=run_id,
        quantity=Decimal(quantity),
        average_entry_price=Decimal(average_entry_price),
        current_price=Decimal(current_price),
        realised_pnl=Decimal(realised_pnl),
        opened_at=opened_at or datetime(2024, 1, 1, tzinfo=UTC),
    )


def _make_order(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    run_id: str = "run-001",
) -> Order:
    """Construct a MARKET Order for signal-routing tests."""
    return Order(
        client_order_id=f"{run_id}-{uuid4().hex[:12]}",
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
    )


# ===========================================================================
# Class 1: TestRecordTradeIfClosed
# Direct unit tests for _record_trade_if_closed()
# ===========================================================================


class TestRecordTradeIfClosed:
    """
    Direct unit tests for StrategyEngine._record_trade_if_closed().

    Each test calls _record_trade_if_closed() with controlled inputs and
    verifies portfolio.record_trade() call behaviour. The engine must be
    started first so _run_id is set.
    """

    async def test_buy_fill_does_not_record_trade(self) -> None:
        """
        A BUY fill must cause an immediate return without recording a trade.

        Only SELL fills can close long positions in spot-only mode.
        portfolio.record_trade must never be called for a BUY fill.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        fill = _make_fill(side=OrderSide.BUY, quantity="1.0", price="100")
        pre_pos = _make_position(quantity="0")  # flat, but side check fires first

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        mocks["portfolio"].record_trade.assert_not_called()

    async def test_sell_fill_no_preexisting_position_does_not_record(self) -> None:
        """
        SELL fill with pre_fill_position=None must not record a trade.

        There is no long position to close if no position snapshot exists
        for the symbol before the fill is applied.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="110")

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=None,
            strategy_id="test_strategy",
        )

        mocks["portfolio"].record_trade.assert_not_called()

    async def test_sell_fill_flat_preexisting_position_does_not_record(
        self,
    ) -> None:
        """
        SELL fill where pre_fill_position.is_flat is True must not record.

        A flat position (quantity == 0) has nothing to close. This guard
        prevents spurious trade records when a SELL arrives with no inventory.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="110")
        pre_pos = _make_position(quantity="0")  # is_flat == True

        assert pre_pos.is_flat, "Precondition: position must be flat"

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        mocks["portfolio"].record_trade.assert_not_called()

    async def test_sell_fill_closes_position_records_trade(self) -> None:
        """
        A SELL fill that fully closes a position must call record_trade()
        once with a correctly-populated TradeResult.

        Verified fields:
        - symbol matches fill.symbol
        - side == OrderSide.BUY (opening side for a long position)
        - entry_price == pre_fill_position.average_entry_price
        - exit_price == fill.price
        - quantity == fill.quantity (full close)
        - realised_pnl == (exit_price - entry_price) * qty - fee
        - total_fees == fill.fee
        - strategy_id propagated correctly
        - entry_at == pre_fill_position.opened_at
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        entry_price = Decimal("100")
        exit_price = Decimal("120")
        qty = Decimal("1.0")
        opened_at = datetime(2024, 1, 1, tzinfo=UTC)

        fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="120")
        pre_pos = _make_position(
            quantity="1.0",
            average_entry_price="100",
            opened_at=opened_at,
        )

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="my_strategy",
        )

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]

        assert trade.symbol == "BTC/USDT"
        assert trade.side == OrderSide.BUY
        assert trade.entry_price == entry_price
        assert trade.exit_price == exit_price
        assert trade.quantity == qty
        # PnL = (120 - 100) * 1.0 - fee
        expected_pnl = (exit_price - entry_price) * qty - fill.fee
        assert trade.realised_pnl == expected_pnl
        assert trade.total_fees == fill.fee
        assert trade.strategy_id == "my_strategy"
        assert trade.entry_at == opened_at
        assert trade.run_id == "run-001"

    async def test_sell_fill_partial_close_records_trade_for_closed_portion(
        self,
    ) -> None:
        """
        When fill.quantity < position.quantity, only the closed portion is
        recorded. TradeResult.quantity must equal fill.quantity, not the
        full position size.

        Position qty=1.0, SELL fill qty=0.7 -> trade quantity == 0.7.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        fill = _make_fill(side=OrderSide.SELL, quantity="0.7", price="120")
        pre_pos = _make_position(quantity="1.0", average_entry_price="100")

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]

        assert trade.quantity == Decimal("0.7")
        # Closed portion only: (120 - 100) * 0.7 - fee
        expected_pnl = (
            (Decimal("120") - Decimal("100")) * Decimal("0.7") - fill.fee
        )
        assert trade.realised_pnl == expected_pnl

    async def test_trade_recorded_with_loss_has_negative_pnl(self) -> None:
        """
        When exit price is below entry price, realised_pnl must be negative.

        This verifies the PnL sign convention: losses produce negative values.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        # Entry at 100, exit at 80 -> loss of 20 per unit minus fee
        fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="80")
        pre_pos = _make_position(quantity="1.0", average_entry_price="100")

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]

        assert trade.realised_pnl < Decimal("0"), (
            f"Expected negative PnL for loss trade, got {trade.realised_pnl}"
        )

    async def test_trade_recorded_fees_are_exit_fee_only(self) -> None:
        """
        total_fees in the TradeResult must equal fill.fee only.

        Entry fees are already embedded in average_entry_price (all-in cost
        basis in PortfolioAccounting). Adding them again here would double-count.
        This test verifies that only the exit fill fee is attributed.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        # Explicit fee to verify exact fee attribution
        fill = _make_fill(
            side=OrderSide.SELL,
            quantity="1.0",
            price="120",
            fee="0.084",  # explicit: 0.7% of 120 = 0.084
        )
        pre_pos = _make_position(quantity="1.0", average_entry_price="100")

        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]
        assert trade.total_fees == Decimal("0.084")
        # Sanity: PnL incorporates the fee deduction
        expected_pnl = (Decimal("120") - Decimal("100")) * Decimal("1.0") - Decimal("0.084")
        assert trade.realised_pnl == expected_pnl

    async def test_exception_in_trade_creation_does_not_propagate(self) -> None:
        """
        If portfolio.record_trade() raises an exception, _record_trade_if_closed
        must catch it and return normally.

        This guards the bar loop from being crashed by a portfolio bug.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        mocks["portfolio"].record_trade = MagicMock(
            side_effect=RuntimeError("portfolio recording failed")
        )

        fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="120")
        pre_pos = _make_position(quantity="1.0", average_entry_price="100")

        # Must not raise despite record_trade raising
        engine._record_trade_if_closed(
            fill=fill,
            pre_fill_position=pre_pos,
            strategy_id="test_strategy",
        )

        # Verify the exception path was reached (record_trade was attempted)
        mocks["portfolio"].record_trade.assert_called_once()


# ===========================================================================
# Class 2: TestProcessBarRecordsTrades
# Integration-level: _process_bar() -> _record_trade_if_closed() path
# ===========================================================================


class TestProcessBarRecordsTrades:
    """
    Integration-level tests of _record_trade_if_closed() called via
    the full _process_bar() pipeline.

    Tests confirm that the engine correctly captures pre-fill position state
    and passes it through to _record_trade_if_closed() in the fill loop.
    All tests disable _check_resting_orders by setting check_resting_orders=None.
    """

    async def test_process_bar_buy_then_sell_records_one_trade(self) -> None:
        """
        Processing two bars — bar 1 with BUY fill, bar 2 with SELL fill —
        must result in exactly one record_trade() call after bar 2.

        BUY fills must not trigger record_trade. Only the SELL fill closing
        the position should produce a TradeResult.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        buy_signal = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        sell_signal = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        buy_order = _make_order(side=OrderSide.BUY)
        sell_order = _make_order(side=OrderSide.SELL)
        buy_fill = _make_fill(side=OrderSide.BUY, quantity="1.0", price="100")
        sell_fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="120")

        # Bar 1: BUY signal -> BUY fill. No position before this bar.
        mocks["strategy"].on_bar = MagicMock(return_value=[buy_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[buy_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[buy_fill])

        # Before bar 1: no position
        mocks["portfolio"].get_position = MagicMock(return_value=None)

        bar1 = _make_bar(
            close="100", timestamp=datetime(2024, 1, 1, 0, tzinfo=UTC)
        )
        await engine._process_bar({"BTC/USDT": bar1}, {"BTC/USDT": [bar1]})

        # After BUY fill: record_trade must NOT have been called yet
        mocks["portfolio"].record_trade.assert_not_called()

        # Bar 2: SELL signal -> SELL fill. Position now exists (post-BUY).
        mocks["strategy"].on_bar = MagicMock(return_value=[sell_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        # Pre-fill position: the open long position from bar 1
        open_position = _make_position(
            quantity="1.0",
            average_entry_price="100",
        )
        mocks["portfolio"].get_position = MagicMock(return_value=open_position)

        bar2 = _make_bar(
            close="120", timestamp=datetime(2024, 1, 1, 1, tzinfo=UTC)
        )
        await engine._process_bar(
            {"BTC/USDT": bar2}, {"BTC/USDT": [bar1, bar2]}
        )

        # After SELL fill: exactly one trade recorded
        mocks["portfolio"].record_trade.assert_called_once()

    async def test_process_bar_buy_no_sell_records_no_trade(self) -> None:
        """
        A single bar with only a BUY fill must not call record_trade() at all.

        Buying opens a position; no round-trip is completed until a SELL fill
        closes it.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        buy_signal = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            target_position=Decimal("1000"),
        )
        buy_order = _make_order(side=OrderSide.BUY)
        buy_fill = _make_fill(side=OrderSide.BUY, quantity="1.0", price="100")

        mocks["strategy"].on_bar = MagicMock(return_value=[buy_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[buy_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[buy_fill])
        mocks["portfolio"].get_position = MagicMock(return_value=None)

        bar = _make_bar(close="100")
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["portfolio"].record_trade.assert_not_called()

    async def test_process_bar_strategy_id_flows_to_trade(self) -> None:
        """
        The strategy_id from the signal must flow into the recorded TradeResult.

        Verifies the chain: signal.strategy_id -> _record_trade_if_closed(strategy_id)
        -> TradeResult.strategy_id.
        """
        engine, mocks = _make_engine(strategy_id="my_custom_strategy")
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        sell_signal = Signal(
            strategy_id="my_custom_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        sell_order = _make_order(side=OrderSide.SELL)
        sell_fill = _make_fill(side=OrderSide.SELL, quantity="1.0", price="120")

        mocks["strategy"].on_bar = MagicMock(return_value=[sell_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        open_position = _make_position(
            quantity="1.0",
            average_entry_price="100",
        )
        mocks["portfolio"].get_position = MagicMock(return_value=open_position)

        bar = _make_bar(close="120")
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]
        assert trade.strategy_id == "my_custom_strategy"

    async def test_process_bar_multiple_symbols_records_per_symbol(
        self,
    ) -> None:
        """
        With two symbols each receiving a SELL fill that closes a position,
        record_trade() must be called twice — once per symbol.

        This verifies per-symbol position isolation in the fill loop.
        """
        engine, mocks = _make_engine(symbols=["BTC/USDT", "ETH/USDT"])
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        sig_btc = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        sig_eth = Signal(
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        order_btc = _make_order(symbol="BTC/USDT", side=OrderSide.SELL)
        order_eth = _make_order(symbol="ETH/USDT", side=OrderSide.SELL)
        fill_btc = _make_fill(
            symbol="BTC/USDT", side=OrderSide.SELL, quantity="1.0", price="120"
        )
        fill_eth = _make_fill(
            symbol="ETH/USDT", side=OrderSide.SELL, quantity="2.0", price="3000"
        )

        # on_bar called once per symbol; each returns one signal
        mocks["strategy"].on_bar = MagicMock(
            side_effect=[[sig_btc], [sig_eth]]
        )
        mocks["execution"].process_signal = AsyncMock(
            side_effect=[[order_btc], [order_eth]]
        )
        mocks["execution"].get_fills = AsyncMock(
            side_effect=[[fill_btc], [fill_eth]]
        )

        pos_btc = _make_position(
            symbol="BTC/USDT", quantity="1.0", average_entry_price="100"
        )
        pos_eth = _make_position(
            symbol="ETH/USDT", quantity="2.0", average_entry_price="2500"
        )

        def _get_position(symbol: str) -> Position | None:
            return {"BTC/USDT": pos_btc, "ETH/USDT": pos_eth}.get(symbol)

        mocks["portfolio"].get_position = MagicMock(side_effect=_get_position)

        bar_btc = _make_bar(symbol="BTC/USDT", close="120")
        bar_eth = _make_bar(symbol="ETH/USDT", close="3000")
        current_bars = {"BTC/USDT": bar_btc, "ETH/USDT": bar_eth}
        history = {"BTC/USDT": [bar_btc], "ETH/USDT": [bar_eth]}

        await engine._process_bar(current_bars, history)

        assert mocks["portfolio"].record_trade.call_count == 2

    async def test_process_bar_multiple_buys_then_sell_uses_averaged_entry(
        self,
    ) -> None:
        """
        When a position was accumulated from multiple BUY fills (VWAP entry),
        the TradeResult's entry_price must equal the position's
        average_entry_price, not the price of either individual BUY fill.

        This test simulates the case where the position snapshot has already
        been updated with a VWAP-averaged average_entry_price from two prior
        BUY fills, and the engine sees that composite price in pre_fill_position.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None

        # VWAP of two buys: 0.5@100 + 0.5@200 = avg entry 150
        # (simplified: 50 + 100 = 150 per unit across 1.0 total qty)
        # We represent this as a pre-existing position with qty=1.0, avg=150
        vwap_entry = Decimal("150")
        sell_signal = Signal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
        )
        sell_order = _make_order(side=OrderSide.SELL)
        sell_fill = _make_fill(
            side=OrderSide.SELL, quantity="1.0", price="200"
        )

        mocks["strategy"].on_bar = MagicMock(return_value=[sell_signal])
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        # Position with VWAP-averaged entry from two prior BUY fills
        averaged_position = _make_position(
            quantity="1.0",
            average_entry_price=str(vwap_entry),
        )
        mocks["portfolio"].get_position = MagicMock(return_value=averaged_position)

        bar = _make_bar(close="200")
        await engine._process_bar({"BTC/USDT": bar}, {"BTC/USDT": [bar]})

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]

        assert trade.entry_price == vwap_entry, (
            f"Expected VWAP entry price {vwap_entry}, got {trade.entry_price}"
        )


# ===========================================================================
# Class 3: TestCheckRestingOrdersRecordsTrades
# Resting-order path: _check_resting_orders() -> _record_trade_if_closed()
# ===========================================================================


class TestCheckRestingOrdersRecordsTrades:
    """
    Tests for the trade-recording path triggered by resting order fills
    in _check_resting_orders().

    The resting-order code path uses strategies[0].strategy_id for trade
    attribution because resting fills do not carry a signal reference.
    """

    async def test_resting_sell_fill_closing_position_records_trade(
        self,
    ) -> None:
        """
        When a resting SELL order fills and the pre-fill position is open,
        _check_resting_orders() must trigger a record_trade() call.

        This mirrors the signal-path test but exercises the resting-order
        code branch in _check_resting_orders().
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        resting_order = _make_order(side=OrderSide.SELL)
        sell_fill = _make_fill(
            side=OrderSide.SELL, quantity="1.0", price="120"
        )

        mocks["execution"].check_resting_orders = AsyncMock(
            return_value=[resting_order]
        )
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        open_position = _make_position(
            quantity="1.0",
            average_entry_price="100",
        )
        mocks["portfolio"].get_position = MagicMock(return_value=open_position)

        bar = _make_bar(close="120")
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]
        assert trade.symbol == "BTC/USDT"
        assert trade.entry_price == Decimal("100")
        assert trade.exit_price == Decimal("120")

    async def test_resting_buy_fill_does_not_record_trade(self) -> None:
        """
        A resting BUY fill (e.g. a limit buy that executes) must not trigger
        record_trade().

        Only SELL fills can close long positions. BUY resting fills open or
        add to positions and must never produce a TradeResult.
        """
        engine, mocks = _make_engine()
        await engine.start("run-001")

        resting_order = _make_order(side=OrderSide.BUY)
        buy_fill = _make_fill(
            side=OrderSide.BUY, quantity="1.0", price="100"
        )

        mocks["execution"].check_resting_orders = AsyncMock(
            return_value=[resting_order]
        )
        mocks["execution"].get_fills = AsyncMock(return_value=[buy_fill])
        # No pre-existing position (fresh buy)
        mocks["portfolio"].get_position = MagicMock(return_value=None)

        bar = _make_bar(close="100")
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["portfolio"].record_trade.assert_not_called()

    async def test_resting_order_trade_uses_first_strategy_id(self) -> None:
        """
        Resting order fills must attribute the trade to strategies[0].strategy_id.

        Because resting fills are not linked to a specific signal, the engine
        uses the first registered strategy's ID as a fallback attribution.
        This test verifies that attribution is correctly forwarded to the
        TradeResult.
        """
        engine, mocks = _make_engine(strategy_id="primary_strategy")
        await engine.start("run-001")

        resting_order = _make_order(side=OrderSide.SELL)
        sell_fill = _make_fill(
            side=OrderSide.SELL, quantity="1.0", price="120"
        )

        mocks["execution"].check_resting_orders = AsyncMock(
            return_value=[resting_order]
        )
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        open_position = _make_position(
            quantity="1.0",
            average_entry_price="100",
        )
        mocks["portfolio"].get_position = MagicMock(return_value=open_position)

        bar = _make_bar(close="120")
        await engine._check_resting_orders({"BTC/USDT": bar})

        mocks["portfolio"].record_trade.assert_called_once()
        trade: TradeResult = mocks["portfolio"].record_trade.call_args[0][0]
        assert trade.strategy_id == "primary_strategy"
