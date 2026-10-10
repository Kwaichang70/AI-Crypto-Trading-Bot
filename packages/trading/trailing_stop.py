"""
packages/trading/trailing_stop.py
----------------------------------
Trailing stop-loss manager — tracks price high-water marks and emits
synthetic SELL signals when price drops below the trailing threshold.

Designed to be composed into StrategyEngine._process_bar() without
modifying execution engines (paper or live).

Semantics (WP1.3b-a, SY-13b-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``HWM_0`` -- the initial high-water mark -- is the gross (fee-free) VWAP of
   the lot's opening-order fills (the real fill in live), set by
   :meth:`TrailingStopManager.open_lot`.
2. The mark ratchets **only at bar-close checks**, and only with closes of
   bars whose ``ts`` is later than the anchor bar (``bar_ts > anchor_bar_ts``);
   the close of the signal bar itself is a pre-entry price.
3. A trigger **keeps** the mark; a quantity change while pending does not
   re-seed it.  It is dropped only when the position is flat.
4. A resume seeds ``max(HWM_0, max close of bars after the anchor bar)``
   (:meth:`TrailingStopManager.seed_peak`, never lowers).
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from common.types import SignalDirection
from trading.models import Position, Signal

__all__ = ["TRAILING_PCT_RANGE", "TrailingStopManager"]

logger = structlog.get_logger(__name__)

# WP1.3a (SY-13a-01): exported so ``packages/trading/exit_config.py`` is the
# single source of truth for the *policy* bounds layered on top of this
# mechanical constructor range (see that module's docstring).
TRAILING_PCT_RANGE: tuple[float, float] = (0.005, 0.50)


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
    pending_stop_ttl : int
        Live-safety backstop (mirrors bracket_exit CR-003).  After a stop
        SELL is emitted the symbol is held "pending" so a slow fill does not
        trigger a duplicate order.  A partial fill that changes the position
        quantity is detected and trailing-stop tracking resumes immediately
        for the residual.  If the order appears stuck (quantity unchanged)
        the guard is force-cleared after this many :meth:`check` calls so a
        position can never remain permanently unprotected.  Must be >= 1.
        Default 3.
    """

    def __init__(
        self,
        trailing_stop_pct: float,
        strategy_id: str = "trailing_stop",
        pending_stop_ttl: int = 3,
    ) -> None:
        _lo, _hi = TRAILING_PCT_RANGE
        if not (_lo <= trailing_stop_pct <= _hi):
            raise ValueError(
                f"trailing_stop_pct must be in [{_lo}, {_hi}], got {trailing_stop_pct}"
            )
        if pending_stop_ttl < 1:
            raise ValueError(
                f"pending_stop_ttl must be >= 1, got {pending_stop_ttl}"
            )
        self._trailing_stop_pct = Decimal(str(trailing_stop_pct))
        self._strategy_id = strategy_id
        self._pending_stop_ttl = pending_stop_ttl
        # symbol -> highest close price since position opened
        self._peak_prices: dict[str, Decimal] = {}
        # symbol -> (lot_key, anchor_bar_ts) of the lot opened via open_lot()
        self._lots: dict[str, tuple[datetime, datetime | None]] = {}
        self._pending_stop_symbols: set[str] = set()
        # Position quantity captured when the stop was emitted — a later
        # change signals a partial fill (or re-entry) leaving a live size
        # that must resume trailing protection.
        self._pending_qty: dict[str, Decimal] = {}
        # check()-call count since pending was set, for the TTL backstop.
        self._pending_age: dict[str, int] = {}
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

    @property
    def pending_stop_ttl(self) -> int:
        """Checks before a stuck pending stop is force-cleared (CR-003 parity)."""
        return self._pending_stop_ttl

    def open_lot(
        self,
        symbol: str,
        *,
        lot_key: datetime,
        hwm0: Decimal,
        anchor_bar_ts: datetime | None,
    ) -> None:
        """
        Register a lot (flat -> open) and set its initial high-water mark.

        ``hwm0`` is the gross (fee-free) VWAP of the opening order's fills.
        Calling this again for the **same** ``lot_key`` (a further fill of the
        opening order) only raises the mark: ``hwm = max(hwm, hwm0)``.  A new
        ``lot_key`` starts a fresh lot (mark replaced, pending cleared).
        ``anchor_bar_ts`` is the open time of the anchor bar; only later bars
        ratchet the mark (see :meth:`check`).
        """
        existing = self._lots.get(symbol)
        if existing is not None and existing[0] == lot_key:
            current = self._peak_prices.get(symbol)
            if current is None or hwm0 > current:
                self._peak_prices[symbol] = hwm0
            return
        self._lots[symbol] = (lot_key, anchor_bar_ts)
        self._peak_prices[symbol] = hwm0
        self._clear_pending(symbol)

    def protective_level(self, symbol: str, position: Position | None) -> Decimal | None:
        """The current trailing stop price for an open lot, else ``None`` (pure)."""
        if position is None or position.is_flat:
            return None
        peak = self._peak_prices.get(symbol)
        if peak is None:
            return None
        return peak * (Decimal("1") - self._trailing_stop_pct)

    def clear_pending(self, symbol: str) -> None:
        """Drop the pending-stop guard for ``symbol`` (R-04: exit not submitted)."""
        self._clear_pending(symbol)

    def check(
        self,
        symbol: str,
        current_price: Decimal,
        position: Position | None,
        *,
        bar_ts: datetime | None = None,
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
        bar_ts : datetime | None
            Open time of the bar ``current_price`` closes.  For a lot opened
            via :meth:`open_lot` the mark ratchets only when ``bar_ts`` is
            ``None`` (legacy direct callers) or later than the lot's anchor
            bar.  The engine always passes ``bar.timestamp``.

        Returns
        -------
        Signal | None
            A SELL signal if the trailing stop is triggered, else None.
        """
        # No position or flat — clean up tracking (the stop filled fully)
        if position is None or position.is_flat:
            self._peak_prices.pop(symbol, None)
            self._lots.pop(symbol, None)
            self._clear_pending(symbol)
            return None

        # A stop was emitted but the position is not yet flat.  Decide whether
        # to keep waiting (order in flight) or resume protection (CR-003
        # parity with bracket_exit: a partial fill / size change must never
        # leave the residual permanently unprotected).
        if symbol in self._pending_stop_symbols:
            prev_qty = self._pending_qty.get(symbol)
            if prev_qty is not None and position.quantity != prev_qty:
                # Position size changed while pending: a partial fill left a
                # residual, or a re-entry increased it.  Resume tracking now
                # against the SAME high-water mark (WP1.3b-a: no re-seed from
                # the current price -- that would loosen the stop).
                self._log.info(
                    "trailing_stop.quantity_changed_resume",
                    symbol=symbol,
                    prev_quantity=str(prev_qty),
                    new_quantity=str(position.quantity),
                )
                self._clear_pending(symbol)
            else:
                # Order still in flight (quantity unchanged).  Suppress
                # duplicates up to the TTL, then force-clear defensively so a
                # stuck order cannot leave the position unprotected forever.
                self._pending_age[symbol] = self._pending_age.get(symbol, 0) + 1
                if self._pending_age[symbol] < self._pending_stop_ttl:
                    return None
                self._log.warning(
                    "trailing_stop.pending_ttl_expired",
                    symbol=symbol,
                    ttl=self._pending_stop_ttl,
                    note="resuming trailing-stop tracking; prior stop may be stuck",
                )
                self._clear_pending(symbol)
            # fall through: tracking resumes for the open/residual position

        # Update high-water mark.  A lot opened via open_lot() ratchets only
        # with closes of bars after its anchor bar; a legacy lot (direct use,
        # no open_lot) seeds from the current price as before.
        peak = self._peak_prices.get(symbol)
        lot = self._lots.get(symbol)
        may_ratchet = (
            lot is None
            or bar_ts is None
            or lot[1] is None
            or bar_ts > lot[1]
        )
        if peak is None:
            self._peak_prices[symbol] = current_price
            peak = current_price
        elif may_ratchet and current_price > peak:
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
            # The high-water mark is KEPT (WP1.3b-a): if the exit is not
            # submitted or only partly fills, the residual is protected against
            # the same mark.  It is dropped only when the position is flat.
            self._pending_stop_symbols.add(symbol)
            # Capture the position quantity so a later partial fill (which
            # changes it) is detectable, and reset the TTL age (CR-003 parity).
            self._pending_qty[symbol] = position.quantity
            self._pending_age[symbol] = 0

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

    def seed_peak(self, symbol: str, peak: Decimal) -> None:
        """
        Seed the trailing high-water mark for ``symbol`` after a resume
        rebuild (WP1.8a A-09/R§3).

        Called once, at the end of ``StrategyEngine._warmup_bar_windows``
        (``_restore_exit_anchors``), for every symbol with an open position
        restored via ``PortfolioAccounting.from_fills``.  The in-memory peak is
        always lost on restart; seeding it from the current price alone would
        loosen the stop (a resumed position could sit further from its
        true historical peak than before the restart).  WP1.3b-a: the caller
        passes ``max(HWM_0, max close of bars after the anchor bar)``, where
        ``HWM_0`` is the gross VWAP of the opening-order fills.

        Never lowers an already-tracked peak -- idempotent/safe to call
        more than once for the same symbol.

        Parameters
        ----------
        symbol:
            The trading pair being seeded.
        peak:
            The peak price to seed, typically
            ``max(HWM_0, max close of bars after the anchor bar)``.
        """
        current = self._peak_prices.get(symbol)
        if current is None or peak > current:
            self._peak_prices[symbol] = peak
        self._log.info(
            "trailing_stop.peak_seeded",
            symbol=symbol,
            seeded_peak=str(peak),
            effective_peak=str(self._peak_prices[symbol]),
        )

    def _clear_pending(self, symbol: str) -> None:
        """Drop all pending-stop tracking for ``symbol``."""
        self._pending_stop_symbols.discard(symbol)
        self._pending_qty.pop(symbol, None)
        self._pending_age.pop(symbol, None)

    def reset(self) -> None:
        """Clear all tracking state."""
        self._peak_prices.clear()
        self._lots.clear()
        self._pending_stop_symbols.clear()
        self._pending_qty.clear()
        self._pending_age.clear()
