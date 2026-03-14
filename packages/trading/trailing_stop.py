"""
packages/trading/trailing_stop.py
----------------------------------
Trailing stop-loss manager — tracks price high-water marks and emits
synthetic SELL signals when price drops below the trailing threshold.

Designed to be composed into StrategyEngine._process_bar() without
modifying execution engines (paper or live).
"""
from __future__ import annotations

from decimal import Decimal

import structlog

from common.types import SignalDirection
from trading.models import Position, Signal

__all__ = ["TrailingStopManager"]

logger = structlog.get_logger(__name__)


class TrailingStopManager:
    """
    Track trailing stop-loss levels for open positions.

    For each symbol with an open position, the manager maintains the
    highest close price seen since entry. When the current price drops
    ``trailing_stop_pct`` (as a fraction, e.g. 0.03 = 3%) below that
    peak, a SELL signal is emitted to close the position.

    Parameters
    ----------
    trailing_stop_pct : float
        Percentage drop from peak that triggers the stop (fraction).
        E.g. 0.03 means a 3% trailing stop. Must be in [0.005, 0.50].
    strategy_id : str
        Strategy ID to use on emitted signals.
    """

    def __init__(self, trailing_stop_pct: float, strategy_id: str = "trailing_stop") -> None:
        if not (0.005 <= trailing_stop_pct <= 0.50):
            raise ValueError(
                f"trailing_stop_pct must be in [0.005, 0.50], got {trailing_stop_pct}"
            )
        self._trailing_stop_pct = Decimal(str(trailing_stop_pct))
        self._strategy_id = strategy_id
        # symbol -> highest close price since position opened
        self._peak_prices: dict[str, Decimal] = {}
        self._pending_stop_symbols: set[str] = set()
        self._log = structlog.get_logger(__name__).bind(
            component="trailing_stop",
            trailing_stop_pct=trailing_stop_pct,
        )

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def trailing_stop_pct(self) -> Decimal:
        return self._trailing_stop_pct

    @property
    def peak_prices(self) -> dict[str, Decimal]:
        """Current peak prices for all tracked symbols (read-only copy)."""
        return dict(self._peak_prices)

    @property
    def pending_stop_symbols(self) -> set[str]:
        """Symbols with emitted but unfilled stop signals (read-only copy)."""
        return set(self._pending_stop_symbols)

    def check(
        self,
        symbol: str,
        current_price: Decimal,
        position: Position | None,
    ) -> Signal | None:
        """
        Check trailing stop for a single symbol.

        Parameters
        ----------
        symbol : str
            The trading pair.
        current_price : Decimal
            Latest close price for this symbol.
        position : Position | None
            Current position for this symbol (None or flat = no tracking).

        Returns
        -------
        Signal | None
            A SELL signal if the trailing stop is triggered, else None.
        """
        # No position or flat — clean up tracking
        if position is None or position.is_flat:
            self._peak_prices.pop(symbol, None)
            self._pending_stop_symbols.discard(symbol)
            return None

        # Skip if a stop signal was already emitted and position hasn't closed yet
        if symbol in self._pending_stop_symbols:
            return None

        # Update high-water mark
        peak = self._peak_prices.get(symbol)
        if peak is None or current_price > peak:
            self._peak_prices[symbol] = current_price
            peak = current_price

        # Check if price has dropped trailing_stop_pct below peak
        stop_price = peak * (Decimal("1") - self._trailing_stop_pct)

        if current_price <= stop_price:
            self._log.info(
                "trailing_stop.triggered",
                symbol=symbol,
                peak=str(peak),
                stop_price=str(stop_price),
                current_price=str(current_price),
                drop_pct=str(
                    round(float((peak - current_price) / peak) * 100, 2)
                ),
            )
            # Clear tracking — position will be closed
            self._peak_prices.pop(symbol, None)
            self._pending_stop_symbols.add(symbol)

            return Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.SELL,
                target_position=Decimal("0"),
                confidence=1.0,
                metadata={
                    "trigger": "trailing_stop",
                    "peak_price": str(peak),
                    "stop_price": str(stop_price),
                    "current_price": str(current_price),
                    "trailing_stop_pct": str(self._trailing_stop_pct),
                },
            )

        return None

    def reset(self) -> None:
        """Clear all tracking state."""
        self._peak_prices.clear()
        self._pending_stop_symbols.clear()
