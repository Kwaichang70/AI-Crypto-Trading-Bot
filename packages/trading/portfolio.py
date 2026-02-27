"""
packages/trading/portfolio.py
-------------------------------
Portfolio accounting: equity tracking, PnL calculation, and trade history.

This is a concrete class (not abstract) that serves as the single source
of truth for portfolio state within a trading run. It is updated on every
fill event and provides equity, drawdown, and PnL metrics to the
RiskManager and UI layer.

Design notes
------------
- All monetary values are in quote currency (e.g. USDT).
- Drawdown is computed from peak equity, which is updated on every
  equity recalculation.
- Daily PnL is reset at day boundaries by calling ``reset_daily_pnl()``.
- The equity curve is stored as a list of (timestamp, equity) tuples
  for charting and analysis.
- TradeResult records are created when positions are fully closed and
  represent completed round-trip trades.
"""

from __future__ import annotations

from datetime import UTC, datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import structlog

from common.types import OrderSide
from trading.models import Fill, Position, TradeResult

__all__ = ["PortfolioAccounting"]

logger = structlog.get_logger(__name__)

_PRECISION = Decimal("0.00000001")


class PortfolioAccounting:
    """
    Tracks equity curve, realised/unrealised PnL, fees, and drawdown
    for a single trading run.

    Parameters
    ----------
    run_id:
        Unique identifier for the trading run.
    initial_cash:
        Starting cash balance in quote currency.
    """

    def __init__(
        self,
        run_id: str,
        initial_cash: Decimal = Decimal("10000"),
    ) -> None:
        self._run_id = run_id
        self._initial_cash = initial_cash
        self._cash = initial_cash

        # Peak equity tracking (starts at initial cash)
        self._peak_equity = initial_cash

        # Daily PnL accumulator (reset at day boundary)
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date: date | None = None

        # Cumulative metrics
        self._total_realised_pnl = Decimal("0")
        self._total_fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0

        # Equity curve: list of (datetime, equity_value) tuples
        self._equity_curve: list[tuple[datetime, Decimal]] = [
            (datetime.now(tz=UTC), initial_cash),
        ]

        # Completed trade history
        self._trade_history: list[TradeResult] = []

        # Position snapshots: symbol -> last known Position (for unrealised PnL)
        self._position_snapshots: dict[str, Position] = {}

        self._log = structlog.get_logger(__name__).bind(
            run_id=run_id,
            component="portfolio_accounting",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def cash(self) -> Decimal:
        """Current cash balance in quote currency."""
        return self._cash

    @property
    def initial_cash(self) -> Decimal:
        """Starting cash balance."""
        return self._initial_cash

    @property
    def total_realised_pnl(self) -> Decimal:
        """Cumulative realised PnL across all closed trades."""
        return self._total_realised_pnl

    @property
    def total_fees_paid(self) -> Decimal:
        """Cumulative fees paid across all fills."""
        return self._total_fees_paid

    @property
    def total_trades(self) -> int:
        """Total number of completed round-trip trades."""
        return self._total_trades

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades. Returns 0.0 if no trades."""
        if self._total_trades == 0:
            return 0.0
        return self._winning_trades / self._total_trades

    @property
    def current_equity(self) -> Decimal:
        """
        Current total portfolio equity (cash + open position value).

        This is the public accessor for the equity calculation that
        consolidates all position snapshots.  Prefer this over calling
        ``get_equity(list(self._position_snapshots.values()))`` from
        outside the class.

        Returns
        -------
        Decimal:
            Total portfolio value in quote currency.
        """
        return self.get_equity(list(self._position_snapshots.values()))

    def get_position(self, symbol: str) -> Position | None:
        """
        Return the current position snapshot for a symbol, or None if
        no position exists for that symbol.

        Parameters
        ----------
        symbol:
            Trading pair identifier (e.g. "BTC/USDT").

        Returns
        -------
        Position | None:
            The current position snapshot, or None.
        """
        return self._position_snapshots.get(symbol)

    # ------------------------------------------------------------------
    # Core update methods
    # ------------------------------------------------------------------

    def update_position(
        self,
        fill: Fill,
        current_price: Decimal,
    ) -> None:
        """
        Update portfolio state after a fill event.

        This method:
        1. Auto-resets daily PnL if day boundary has been crossed.
        2. Updates the cash balance based on the fill.
        3. Tracks fees.
        4. Updates the position snapshot for the symbol.
        5. Recalculates unrealised PnL for the position.
        6. Records a point on the equity curve.
        7. Updates peak equity and daily PnL.

        Parameters
        ----------
        fill:
            The fill event to process.
        current_price:
            Current market price of the asset (for unrealised PnL).
        """
        # Auto-reset daily PnL if day boundary has been crossed
        self._ensure_daily_pnl_date()

        # Track fees
        self._total_fees_paid += fill.fee

        # Update cash based on fill side
        if fill.side == OrderSide.BUY:
            # Buying: cash decreases by (quantity * price + fee)
            cost = fill.quantity * fill.price + fill.fee
            self._cash -= cost
        elif fill.side == OrderSide.SELL:
            # Selling: cash increases by (quantity * price - fee)
            proceeds = fill.quantity * fill.price - fill.fee
            self._cash += proceeds

        # Update position snapshot
        position = self._position_snapshots.get(fill.symbol)
        now = datetime.now(tz=UTC)

        if fill.side == OrderSide.BUY:
            if position is None or position.is_flat:
                # New position -- use all-in cost basis (includes entry fee)
                all_in_entry_price = (
                    (fill.price * fill.quantity + fill.fee) / fill.quantity
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
                self._position_snapshots[fill.symbol] = Position(
                    symbol=fill.symbol,
                    run_id=self._run_id,
                    quantity=fill.quantity,
                    average_entry_price=all_in_entry_price,
                    current_price=current_price,
                    realised_pnl=Decimal("0"),
                    unrealised_pnl=(current_price - all_in_entry_price) * fill.quantity,
                    total_fees_paid=fill.fee,
                    opened_at=now,
                    updated_at=now,
                )
            else:
                # Add to existing position
                old_cost = position.quantity * position.average_entry_price
                new_cost = fill.quantity * fill.price + fill.fee
                total_qty = position.quantity + fill.quantity
                new_avg = (
                    (old_cost + new_cost) / total_qty
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

                unrealised = (current_price - new_avg) * total_qty

                self._position_snapshots[fill.symbol] = position.model_copy(
                    update={
                        "quantity": total_qty,
                        "average_entry_price": new_avg,
                        "current_price": current_price,
                        "unrealised_pnl": unrealised,
                        "total_fees_paid": position.total_fees_paid + fill.fee,
                        "updated_at": now,
                    }
                )

        elif fill.side == OrderSide.SELL:
            if position is not None and not position.is_flat:
                sell_qty = min(fill.quantity, position.quantity)
                pnl = (fill.price - position.average_entry_price) * sell_qty - fill.fee
                remaining_qty = position.quantity - sell_qty

                # Accumulate realised PnL
                self._total_realised_pnl += pnl

                # Update daily PnL (date already ensured at top of method)
                self._daily_pnl += pnl

                if remaining_qty <= Decimal("0"):
                    # Position fully closed
                    self._position_snapshots[fill.symbol] = position.model_copy(
                        update={
                            "quantity": Decimal("0"),
                            "current_price": current_price,
                            "realised_pnl": position.realised_pnl + pnl,
                            "unrealised_pnl": Decimal("0"),
                            "total_fees_paid": position.total_fees_paid + fill.fee,
                            "updated_at": now,
                        }
                    )
                else:
                    unrealised = (
                        (current_price - position.average_entry_price) * remaining_qty
                    )
                    self._position_snapshots[fill.symbol] = position.model_copy(
                        update={
                            "quantity": remaining_qty,
                            "current_price": current_price,
                            "realised_pnl": position.realised_pnl + pnl,
                            "unrealised_pnl": unrealised,
                            "total_fees_paid": position.total_fees_paid + fill.fee,
                            "updated_at": now,
                        }
                    )

        # Record equity curve point
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        self._equity_curve.append((now, current_equity))

        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        self._log.debug(
            "portfolio.position_updated",
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=str(fill.quantity),
            price=str(fill.price),
            fee=str(fill.fee),
            cash=str(self._cash),
            equity=str(current_equity),
        )

    def update_market_prices(
        self,
        prices: dict[str, Decimal],
    ) -> None:
        """
        Update unrealised PnL for all positions with current market prices.

        Call this on every bar to keep the equity curve and unrealised
        PnL accurate between fills.

        Parameters
        ----------
        prices:
            Mapping of symbol -> current market price.
        """
        now = datetime.now(tz=UTC)
        for symbol, price in prices.items():
            position = self._position_snapshots.get(symbol)
            if position is not None and not position.is_flat:
                unrealised = (
                    (price - position.average_entry_price) * position.quantity
                )
                self._position_snapshots[symbol] = position.model_copy(
                    update={
                        "current_price": price,
                        "unrealised_pnl": unrealised,
                        "updated_at": now,
                    }
                )

        # Record equity curve point
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        self._equity_curve.append((now, current_equity))

        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    # ------------------------------------------------------------------
    # Equity calculations
    # ------------------------------------------------------------------

    def get_equity(self, positions: list[Position]) -> Decimal:
        """
        Calculate total portfolio equity.

        Equity = cash + sum(position.notional_value for open positions)

        Parameters
        ----------
        positions:
            List of current positions (typically from the execution engine
            or from internal snapshots).

        Returns
        -------
        Decimal:
            Total portfolio value in quote currency.
        """
        position_value = Decimal("0")
        for pos in positions:
            if not pos.is_flat:
                position_value += pos.notional_value
        return self._cash + position_value

    def get_peak_equity(self) -> Decimal:
        """
        Return the highest equity value recorded since run start.

        Returns
        -------
        Decimal:
            Peak equity in quote currency.
        """
        return self._peak_equity

    def get_daily_pnl(self) -> Decimal:
        """
        Return the net PnL for the current trading day.

        This includes only realised PnL from closed positions.
        Unrealised PnL is not included in the daily figure.

        Does NOT auto-reset on day boundary -- call ``reset_daily_pnl()``
        explicitly, or rely on the auto-reset in ``update_position()``.

        Returns
        -------
        Decimal:
            Net daily PnL in quote currency.
        """
        return self._daily_pnl

    def get_drawdown(self) -> float:
        """
        Calculate the current drawdown as a percentage of peak equity.

        Drawdown = (peak_equity - current_equity) / peak_equity

        Returns
        -------
        float:
            Drawdown as a decimal fraction (e.g. 0.05 = 5%).
            Returns 0.0 if peak equity is zero or negative.
        """
        if self._peak_equity <= Decimal("0"):
            return 0.0

        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return max(0.0, float(drawdown))

    def get_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown from the equity curve.

        Iterates through the entire equity curve to find the worst
        peak-to-trough decline.

        Returns
        -------
        float:
            Maximum drawdown as a decimal fraction.
        """
        if len(self._equity_curve) < 2:
            return 0.0

        peak = Decimal("0")
        max_dd = Decimal("0")

        for _, equity in self._equity_curve:
            if equity > peak:
                peak = equity
            if peak > Decimal("0"):
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        return float(max_dd)

    def get_total_return(self) -> float:
        """
        Calculate the total return as a percentage of initial cash.

        Returns
        -------
        float:
            Total return as a decimal fraction (e.g. 0.10 = 10%).
        """
        if self._initial_cash <= Decimal("0"):
            return 0.0

        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return float(
            (current_equity - self._initial_cash) / self._initial_cash
        )

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeResult) -> None:
        """
        Record a completed round-trip trade.

        Updates win/loss counters and stores the trade in history.

        Parameters
        ----------
        trade:
            A fully populated TradeResult from a closed position.
        """
        self._trade_history.append(trade)
        self._total_trades += 1

        if trade.realised_pnl > Decimal("0"):
            self._winning_trades += 1
        elif trade.realised_pnl < Decimal("0"):
            self._losing_trades += 1
        # break-even (== 0): neither counter incremented

        self._log.info(
            "portfolio.trade_recorded",
            trade_id=str(trade.trade_id),
            symbol=trade.symbol,
            side=trade.side.value,
            entry_price=str(trade.entry_price),
            exit_price=str(trade.exit_price),
            quantity=str(trade.quantity),
            realised_pnl=str(trade.realised_pnl),
            return_pct=f"{trade.return_pct:.4%}",
        )

    def get_trade_history(self) -> list[TradeResult]:
        """
        Return the complete trade history for this run.

        Returns
        -------
        list[TradeResult]:
            All completed trades, ordered by recording time.
        """
        return list(self._trade_history)

    # ------------------------------------------------------------------
    # Daily PnL management
    # ------------------------------------------------------------------

    def reset_daily_pnl(self) -> None:
        """
        Reset the daily PnL accumulator.

        Call this at the start of each trading day (or at day boundary
        in backtesting). The previous day's PnL is logged before reset.
        """
        if self._daily_pnl != Decimal("0"):
            self._log.info(
                "portfolio.daily_pnl_reset",
                previous_daily_pnl=str(self._daily_pnl),
                date=str(self._daily_pnl_date),
            )
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date = datetime.now(tz=UTC).date()

    def _ensure_daily_pnl_date(self) -> None:
        """
        Auto-reset daily PnL if the date has rolled over.

        This provides automatic day-boundary handling even if
        reset_daily_pnl() is not explicitly called.
        """
        today = datetime.now(tz=UTC).date()
        if self._daily_pnl_date is None:
            self._daily_pnl_date = today
        elif self._daily_pnl_date != today:
            self._log.info(
                "portfolio.auto_daily_reset",
                previous_date=str(self._daily_pnl_date),
                new_date=str(today),
                previous_pnl=str(self._daily_pnl),
            )
            self._daily_pnl = Decimal("0")
            self._daily_pnl_date = today

    # ------------------------------------------------------------------
    # Equity curve access
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """
        Return the full equity curve as (timestamp, equity) tuples.

        Returns
        -------
        list[tuple[datetime, Decimal]]:
            Equity curve ordered by timestamp ascending.
        """
        return list(self._equity_curve)

    # ------------------------------------------------------------------
    # Summary / snapshot
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """
        Return a comprehensive portfolio summary dict.

        Suitable for API serialization, logging, and dashboard display.

        Returns
        -------
        dict[str, Any]:
            Portfolio summary with all key metrics.
        """
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return {
            "run_id": self._run_id,
            "initial_cash": str(self._initial_cash),
            "current_cash": str(self._cash),
            "current_equity": str(current_equity),
            "peak_equity": str(self._peak_equity),
            "total_return_pct": self.get_total_return(),
            "total_realised_pnl": str(self._total_realised_pnl),
            "total_fees_paid": str(self._total_fees_paid),
            "daily_pnl": str(self._daily_pnl),
            "drawdown_pct": self.get_drawdown(),
            "max_drawdown_pct": self.get_max_drawdown(),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": self.win_rate,
            "open_positions": len(
                [p for p in self._position_snapshots.values() if not p.is_flat]
            ),
            "equity_curve_length": len(self._equity_curve),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return (
            f"PortfolioAccounting("
            f"run_id={self._run_id!r}, "
            f"equity={current_equity}, "
            f"trades={self._total_trades}, "
            f"drawdown={self.get_drawdown():.2%})"
        )
