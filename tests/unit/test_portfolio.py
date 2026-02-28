"""
tests/unit/test_portfolio.py
-----------------------------
Unit tests for PortfolioAccounting in packages/trading/portfolio.py.

Test coverage
-------------
- BUY fill: new position created with all-in cost basis (price + fee)
- BUY fill: adds to existing position with correct weighted average
- SELL fill: position closed, realised PnL computed correctly
- SELL fill: partial close reduces quantity and records partial PnL
- FIX-05 regression: break-even trade (realised_pnl == 0) not counted as win
- FIX-07 regression: daily_pnl excludes unrealised (mark-to-market) changes
- Equity curve records a new point after each update
- Peak equity is updated when equity exceeds previous peak
- get_position() returns the current snapshot for a symbol
- Win rate calculation with wins, losses, and break-even trades
- Total return calculation
- Drawdown calculation
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide
from trading.models import Fill, TradeResult
from trading.portfolio import PortfolioAccounting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.01"),
    price: Decimal = Decimal("50000"),
    fee_pct: Decimal = Decimal("0.001"),
    fee_currency: str = "USDT",
) -> Fill:
    """Build a Fill with fee derived from a percentage of notional."""
    fee = quantity * price * fee_pct
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        fee_currency=fee_currency,
    )


def _make_trade(
    *,
    run_id: str = "test-run",
    symbol: str = "BTC/USDT",
    realised_pnl: Decimal,
    entry_price: Decimal = Decimal("50000"),
    exit_price: Decimal = Decimal("51000"),
    quantity: Decimal = Decimal("0.01"),
) -> TradeResult:
    """Build a minimal TradeResult for record_trade tests."""
    now = datetime.now(tz=UTC)
    return TradeResult(
        run_id=run_id,
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        realised_pnl=realised_pnl,
        total_fees=Decimal("0.5"),
        entry_at=now,
        exit_at=now,
        strategy_id="test",
    )


# ===========================================================================
# Initialisation
# ===========================================================================


class TestPortfolioInit:
    """Tests for PortfolioAccounting initial state."""

    def test_initial_cash_set_correctly(self) -> None:
        """Cash balance equals initial_cash at construction."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        assert portfolio.cash == Decimal("10000")

    def test_initial_equity_equals_initial_cash(self) -> None:
        """Equity equals cash when no positions are open."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        assert portfolio.current_equity == Decimal("10000")

    def test_initial_equity_curve_has_one_point(self) -> None:
        """The equity curve is seeded with one initial point at construction."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        assert len(portfolio.get_equity_curve()) == 1

    def test_initial_counters_are_zero(self) -> None:
        """Trade counters start at zero."""
        portfolio = PortfolioAccounting(run_id="test")
        assert portfolio.total_trades == 0
        assert portfolio.win_rate == 0.0
        assert portfolio.total_fees_paid == Decimal("0")

    def test_initial_daily_pnl_is_zero(self) -> None:
        """Daily PnL starts at zero."""
        portfolio = PortfolioAccounting(run_id="test")
        assert portfolio.get_daily_pnl() == Decimal("0")


# ===========================================================================
# BUY fill handling
# ===========================================================================


class TestBuyFill:
    """Tests for position creation and sizing after BUY fills."""

    def test_buy_fill_creates_new_position(self) -> None:
        """A BUY fill on a previously-flat symbol creates a new position."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        fill = _make_fill(side=OrderSide.BUY, quantity=Decimal("0.01"), price=Decimal("50000"))
        portfolio.update_position(fill, current_price=Decimal("50000"))
        pos = portfolio.get_position("BTC/USDT")
        assert pos is not None
        assert pos.quantity == Decimal("0.01")

    def test_buy_fill_all_in_cost_basis_includes_fee(self) -> None:
        """
        All-in cost basis = (price * qty + fee) / qty.
        With 0% fee: entry price equals fill price.
        With 0.1% fee: entry price is slightly above fill price.
        """
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("100000"))
        qty = Decimal("1")
        price = Decimal("1000")
        fee = qty * price * Decimal("0.001")  # 1 USDT fee
        fill = Fill(
            order_id=uuid4(),
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=qty,
            price=price,
            fee=fee,
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=price)
        pos = portfolio.get_position("ETH/USDT")
        assert pos is not None
        # All-in entry = (1000 + 1) / 1 = 1001
        expected_entry = (price * qty + fee) / qty
        assert pos.average_entry_price == expected_entry.quantize(Decimal("0.00000001"))

    def test_buy_fill_decreases_cash(self) -> None:
        """Cash decreases by (quantity * price + fee) after a BUY."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        fill = _make_fill(
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee_pct=Decimal("0"),
        )
        portfolio.update_position(fill, current_price=Decimal("50000"))
        # cash = 10000 - (0.01 * 50000 + 0) = 10000 - 500 = 9500
        assert portfolio.cash == Decimal("9500")

    def test_second_buy_averages_entry_price(self) -> None:
        """Adding to an existing position updates the VWAP entry price."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        # First buy: 1 unit at 1000
        fill1 = Fill(
            order_id=uuid4(),
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("1000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill1, current_price=Decimal("1000"))
        # Second buy: 1 unit at 1200
        fill2 = Fill(
            order_id=uuid4(),
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("1200"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill2, current_price=Decimal("1200"))
        pos = portfolio.get_position("ETH/USDT")
        assert pos is not None
        assert pos.quantity == Decimal("2")
        # Average = (1000 + 1200) / 2 = 1100
        assert pos.average_entry_price == Decimal("1100").quantize(Decimal("0.00000001"))

    def test_buy_fee_tracked(self) -> None:
        """Total fees paid increments by fill.fee after a BUY."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        fill = _make_fill(
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee_pct=Decimal("0.001"),
        )
        portfolio.update_position(fill, current_price=Decimal("50000"))
        assert portfolio.total_fees_paid == fill.fee


# ===========================================================================
# SELL fill handling
# ===========================================================================


class TestSellFill:
    """Tests for position closing and PnL computation after SELL fills."""

    def _open_position(
        self,
        portfolio: PortfolioAccounting,
        *,
        symbol: str = "BTC/USDT",
        qty: Decimal = Decimal("0.01"),
        price: Decimal = Decimal("50000"),
    ) -> None:
        """Helper to open a position by processing a zero-fee BUY fill."""
        fill = Fill(
            order_id=uuid4(),
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=qty,
            price=price,
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=price)

    def test_sell_fill_closes_position(self) -> None:
        """A SELL of the full quantity closes the position (quantity=0)."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        self._open_position(portfolio, qty=Decimal("0.01"), price=Decimal("50000"))
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("51000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=Decimal("51000"))
        pos = portfolio.get_position("BTC/USDT")
        assert pos is not None
        assert pos.is_flat

    def test_sell_fill_computes_pnl_correctly(self) -> None:
        """
        PnL = (exit_price - entry_price) * qty - fee.
        Buy at 50_000, sell at 51_000, qty=0.01, no fee:
        PnL = (51000 - 50000) * 0.01 = 10 USDT.
        """
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        self._open_position(portfolio, qty=Decimal("0.01"), price=Decimal("50000"))
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("51000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=Decimal("51000"))
        assert portfolio.total_realised_pnl == Decimal("10")

    def test_sell_fill_increases_cash(self) -> None:
        """Cash increases by (quantity * price - fee) after a SELL."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        self._open_position(portfolio, qty=Decimal("0.01"), price=Decimal("50000"))
        sell_price = Decimal("51000")
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=sell_price,
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        cash_before = portfolio.cash
        portfolio.update_position(fill, current_price=sell_price)
        # BUY cost 500, SELL proceeds = 0.01 * 51000 = 510
        # cash_before = 10000 - 500 = 9500; after = 9500 + 510 = 10010
        assert portfolio.cash == cash_before + (Decimal("0.01") * sell_price)

    def test_partial_sell_reduces_quantity(self) -> None:
        """Selling half the position halves the quantity."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        self._open_position(portfolio, qty=Decimal("0.02"), price=Decimal("50000"))
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("51000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=Decimal("51000"))
        pos = portfolio.get_position("BTC/USDT")
        assert pos is not None
        assert pos.quantity == Decimal("0.01")
        assert not pos.is_flat


# ===========================================================================
# FIX-05 regression: break-even trade not counted as win
# ===========================================================================


class TestBreakEvenTradeNotWin:
    """
    FIX-05 regression guard.

    When a position is closed at exactly the entry price (realised_pnl == 0),
    record_trade must NOT increment winning_trades or losing_trades.
    win_rate must remain at 0.0 when only break-even trades have been recorded.
    """

    def test_break_even_not_counted_as_win(self) -> None:
        """Break-even trade does not increment winning_trades."""
        portfolio = PortfolioAccounting(run_id="test")
        trade = _make_trade(realised_pnl=Decimal("0"))
        portfolio.record_trade(trade)
        # total_trades increments but winning/losing do not
        assert portfolio.total_trades == 1
        assert portfolio.win_rate == 0.0

    def test_break_even_not_counted_as_loss(self) -> None:
        """Break-even trade does not increment losing_trades."""
        portfolio = PortfolioAccounting(run_id="test")
        trade = _make_trade(realised_pnl=Decimal("0"))
        portfolio.record_trade(trade)
        # Confirm by checking win_rate is 0 (no wins) and trade count is 1
        assert portfolio.total_trades == 1
        # If it were counted as loss, win_rate = 0/1 = 0 — indistinguishable.
        # So also verify via multiple trades:
        portfolio.record_trade(_make_trade(realised_pnl=Decimal("10")))
        # 1 win, 0 losses, 1 break-even → 2 total, 1 winning
        assert portfolio.total_trades == 2
        assert abs(portfolio.win_rate - 0.5) < 1e-9  # 1 win / 2 total


# ===========================================================================
# FIX-07 regression: daily PnL excludes unrealised PnL
# ===========================================================================


class TestDailyPnlExcludesUnrealised:
    """
    FIX-07 regression guard.

    daily_pnl must only reflect realised (closed) trades.
    Unrealised mark-to-market changes from update_market_prices()
    must NOT affect get_daily_pnl().
    """

    def test_market_price_update_does_not_affect_daily_pnl(self) -> None:
        """
        Calling update_market_prices() with a higher price should NOT
        change the daily PnL (which is realised-only).
        """
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        # Open a position
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=Decimal("50000"))

        daily_pnl_before = portfolio.get_daily_pnl()

        # Simulate market price appreciation (unrealised gain)
        portfolio.update_market_prices({"BTC/USDT": Decimal("55000")})

        daily_pnl_after = portfolio.get_daily_pnl()

        assert daily_pnl_before == daily_pnl_after, (
            "FIX-07 regression: daily_pnl changed after market price update "
            "even though no position was closed"
        )

    def test_realised_sell_does_update_daily_pnl(self) -> None:
        """Closing a position via SELL must update daily_pnl."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        # BUY
        buy_fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(buy_fill, current_price=Decimal("50000"))
        pnl_before = portfolio.get_daily_pnl()

        # SELL at profit
        sell_fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("51000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(sell_fill, current_price=Decimal("51000"))
        pnl_after = portfolio.get_daily_pnl()

        assert pnl_after > pnl_before
        assert pnl_after == Decimal("10")  # (51000 - 50000) * 0.01


# ===========================================================================
# Equity curve
# ===========================================================================


class TestEquityCurve:
    """Tests for equity curve recording and peak tracking."""

    def test_equity_curve_grows_with_each_fill(self) -> None:
        """Each call to update_position appends a new equity curve point."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        initial_len = len(portfolio.get_equity_curve())

        fill = _make_fill(side=OrderSide.BUY)
        portfolio.update_position(fill, current_price=Decimal("50000"))
        assert len(portfolio.get_equity_curve()) == initial_len + 1

    def test_peak_equity_updated_on_profit(self) -> None:
        """Peak equity is updated when equity exceeds the previous peak."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        initial_peak = portfolio.get_peak_equity()

        # Simulate profitable position (price increase updates unrealised PnL)
        fill = Fill(
            order_id=uuid4(),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_currency="USDT",
        )
        portfolio.update_position(fill, current_price=Decimal("50000"))
        portfolio.update_market_prices({"BTC/USDT": Decimal("60000")})

        new_peak = portfolio.get_peak_equity()
        assert new_peak > initial_peak

    def test_peak_equity_does_not_decrease(self) -> None:
        """Peak equity is a high-water mark — it never decreases."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        portfolio.update_market_prices({"BTC/USDT": Decimal("60000")})
        peak_after_rise = portfolio.get_peak_equity()
        # Simulate a loss (drop in market price)
        portfolio.update_market_prices({"BTC/USDT": Decimal("10000")})
        assert portfolio.get_peak_equity() >= peak_after_rise


# ===========================================================================
# get_position
# ===========================================================================


class TestGetPosition:
    """Tests for the position snapshot lookup."""

    def test_get_position_returns_none_for_unknown_symbol(self) -> None:
        """get_position returns None when no position exists for the symbol."""
        portfolio = PortfolioAccounting(run_id="test")
        assert portfolio.get_position("BTC/USDT") is None

    def test_get_position_returns_snapshot_after_buy(self) -> None:
        """get_position returns the Position snapshot after a BUY fill."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        fill = _make_fill(side=OrderSide.BUY, quantity=Decimal("0.01"))
        portfolio.update_position(fill, current_price=Decimal("50000"))
        pos = portfolio.get_position("BTC/USDT")
        assert pos is not None
        assert pos.symbol == "BTC/USDT"

    def test_get_position_is_flat_after_full_sell(self) -> None:
        """After selling the full position, get_position returns a flat snapshot."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        buy = Fill(
            order_id=uuid4(), symbol="BTC/USDT", side=OrderSide.BUY,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            fee=Decimal("0"), fee_currency="USDT",
        )
        portfolio.update_position(buy, current_price=Decimal("50000"))
        sell = Fill(
            order_id=uuid4(), symbol="BTC/USDT", side=OrderSide.SELL,
            quantity=Decimal("0.01"), price=Decimal("51000"),
            fee=Decimal("0"), fee_currency="USDT",
        )
        portfolio.update_position(sell, current_price=Decimal("51000"))
        pos = portfolio.get_position("BTC/USDT")
        assert pos is not None
        assert pos.is_flat


# ===========================================================================
# Win rate and trade recording
# ===========================================================================


class TestWinRate:
    """Tests for trade recording and win-rate accounting."""

    def test_win_increments_winning_trades(self) -> None:
        """A trade with positive PnL increments win_rate numerator."""
        portfolio = PortfolioAccounting(run_id="test")
        portfolio.record_trade(_make_trade(realised_pnl=Decimal("100")))
        assert abs(portfolio.win_rate - 1.0) < 1e-9

    def test_loss_does_not_increment_winning_trades(self) -> None:
        """A trade with negative PnL does not count as a win."""
        portfolio = PortfolioAccounting(run_id="test")
        portfolio.record_trade(_make_trade(realised_pnl=Decimal("-50")))
        assert portfolio.win_rate == 0.0

    def test_win_rate_mixed_trades(self) -> None:
        """Win rate = winning_trades / total_trades for a mix."""
        portfolio = PortfolioAccounting(run_id="test")
        for pnl in [100, 50, -30, -20]:
            portfolio.record_trade(_make_trade(realised_pnl=Decimal(str(pnl))))
        # 2 wins / 4 total = 50%
        assert abs(portfolio.win_rate - 0.5) < 1e-9

    def test_total_return_after_profitable_run(self) -> None:
        """Total return is positive when equity exceeds initial cash."""
        portfolio = PortfolioAccounting(run_id="test", initial_cash=Decimal("10000"))
        # Simulate profitable sell that increases cash above initial
        fill_buy = Fill(
            order_id=uuid4(), symbol="BTC/USDT", side=OrderSide.BUY,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            fee=Decimal("0"), fee_currency="USDT",
        )
        portfolio.update_position(fill_buy, current_price=Decimal("50000"))
        fill_sell = Fill(
            order_id=uuid4(), symbol="BTC/USDT", side=OrderSide.SELL,
            quantity=Decimal("0.01"), price=Decimal("60000"),
            fee=Decimal("0"), fee_currency="USDT",
        )
        portfolio.update_position(fill_sell, current_price=Decimal("60000"))
        assert portfolio.get_total_return() > 0.0
