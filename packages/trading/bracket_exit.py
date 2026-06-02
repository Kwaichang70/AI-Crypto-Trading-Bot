"""
packages/trading/bracket_exit.py
--------------------------------
Bracket exit manager — fixed-percentage *and* ATR-multiple stop-loss /
take-profit exits.  Emits synthetic SELL signals when an open position's
price crosses a stop-loss (below entry) or take-profit (above entry) level.

This is the fixed/volatility-scaled counterpart to
:class:`trading.trailing_stop.TrailingStopManager`.  Whereas the trailing
stop floats with the price peak, brackets are anchored to the position's
``average_entry_price`` and never move.

Designed to be composed into ``StrategyEngine._process_bar()`` without
modifying execution engines (paper or live), exactly like the trailing stop.

Two modes (``bracket_mode``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``"fixed"`` — levels are a fixed fraction of entry price::

      stop_price        = entry * (1 - stop_loss_pct)
      take_profit_price = entry * (1 + take_profit_pct)

* ``"atr"`` — levels are an ATR multiple away from entry (volatility
  adaptive).  The engine recomputes ATR each bar and passes it in::

      stop_price        = entry - atr_sl_multiplier * atr
      take_profit_price = entry + atr_tp_multiplier * atr

Precedence
~~~~~~~~~~
Stop-loss is checked **before** take-profit.  On a bar that gaps through
both levels the stop-loss wins (hard risk floor takes priority).

Exit-reason classification (CRITICAL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``trade_journal.ExitReasonDetector.detect()`` inspects the emitted signal:

1. if ``"trailing_stop"`` is a substring of ``strategy_id`` -> "trailing_stop"
2. elif ``metadata["stop_loss"]`` truthy -> "stop_loss"
3. elif ``metadata["take_profit"]`` truthy -> "take_profit"

Therefore this manager's ``strategy_id`` must **not** contain the substring
``"trailing_stop"`` (default ``"bracket_exit"``), and each emitted signal
sets **exactly one** of ``stop_loss`` / ``take_profit`` to ``True`` so the
detector labels the trade correctly.
"""
from __future__ import annotations

from decimal import Decimal

import structlog

from common.types import SignalDirection
from trading.models import Position, Signal

__all__ = ["BracketExitManager"]

logger = structlog.get_logger(__name__)

_VALID_MODES = frozenset({"fixed", "atr"})


class BracketExitManager:
    """
    Track fixed/ATR stop-loss and take-profit levels for open positions.

    For each symbol with an open (long, spot-only) position the manager
    derives a stop-loss level below entry and a take-profit level above
    entry.  When the current price crosses either level a full-close SELL
    signal is emitted.

    Parameters
    ----------
    stop_loss_pct : float | None
        Fixed-mode stop distance as a fraction of entry (e.g. 0.02 = 2%).
        Must be in [0.001, 0.95] when set.  Used only in ``"fixed"`` mode.
    take_profit_pct : float | None
        Fixed-mode take-profit distance as a fraction of entry (e.g.
        0.04 = 4%).  Must be in [0.001, 0.95] when set.  ``"fixed"`` mode.
    bracket_mode : str
        ``"fixed"`` (default) or ``"atr"``.
    atr_sl_multiplier : float | None
        ATR-mode stop distance multiplier (e.g. 2.0 = 2x ATR below entry).
        Must be in [0.1, 20.0] when set.  Used only in ``"atr"`` mode.
    atr_tp_multiplier : float | None
        ATR-mode take-profit multiplier (e.g. 3.0 = 3x ATR above entry).
        Must be in [0.1, 20.0] when set.  Used only in ``"atr"`` mode.
    atr_period : int
        ATR look-back period the engine should use when computing the ATR
        value passed to :meth:`check`.  Must be >= 2.  Default 14.
    strategy_id : str
        Strategy ID on emitted signals.  Must NOT contain the substring
        ``"trailing_stop"`` (see module docstring).  Default ``"bracket_exit"``.
    pending_exit_ttl : int
        Live-safety backstop (CR-003).  After an exit SELL is emitted the
        symbol is held "pending" so a slow fill does not trigger a duplicate
        order.  A **partial** fill that reduces the position quantity is
        detected and the residual is re-protected immediately.  If instead the
        order appears stuck (position quantity unchanged) the pending guard is
        force-cleared after this many :meth:`check` calls so a position can
        never remain permanently unprotected.  Must be >= 1.  Default 3.

    Raises
    ------
    ValueError
        On any invalid configuration.  The engine wraps construction in a
        try/except so a bad config disables brackets rather than crashing.
    """

    def __init__(
        self,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        bracket_mode: str = "fixed",
        atr_sl_multiplier: float | None = None,
        atr_tp_multiplier: float | None = None,
        atr_period: int = 14,
        strategy_id: str = "bracket_exit",
        pending_exit_ttl: int = 3,
    ) -> None:
        if "trailing_stop" in strategy_id:
            raise ValueError(
                "bracket_exit strategy_id must not contain 'trailing_stop' "
                f"(would corrupt exit-reason classification), got {strategy_id!r}"
            )
        if pending_exit_ttl < 1:
            raise ValueError(
                f"pending_exit_ttl must be >= 1, got {pending_exit_ttl}"
            )
        if bracket_mode not in _VALID_MODES:
            raise ValueError(
                f"bracket_mode must be one of {sorted(_VALID_MODES)}, got {bracket_mode!r}"
            )

        if bracket_mode == "fixed":
            if stop_loss_pct is None and take_profit_pct is None:
                raise ValueError(
                    "fixed bracket_mode requires at least one of "
                    "stop_loss_pct / take_profit_pct"
                )
            for name, value in (
                ("stop_loss_pct", stop_loss_pct),
                ("take_profit_pct", take_profit_pct),
            ):
                if value is not None and not (0.001 <= value <= 0.95):
                    raise ValueError(
                        f"{name} must be in [0.001, 0.95], got {value}"
                    )
        else:  # atr
            if atr_sl_multiplier is None and atr_tp_multiplier is None:
                raise ValueError(
                    "atr bracket_mode requires at least one of "
                    "atr_sl_multiplier / atr_tp_multiplier"
                )
            for name, value in (
                ("atr_sl_multiplier", atr_sl_multiplier),
                ("atr_tp_multiplier", atr_tp_multiplier),
            ):
                if value is not None and not (0.1 <= value <= 20.0):
                    raise ValueError(
                        f"{name} must be in [0.1, 20.0], got {value}"
                    )
            if atr_period < 2:
                raise ValueError(f"atr_period must be >= 2, got {atr_period}")

        self._mode = bracket_mode
        self._stop_loss_pct = (
            Decimal(str(stop_loss_pct)) if stop_loss_pct is not None else None
        )
        self._take_profit_pct = (
            Decimal(str(take_profit_pct)) if take_profit_pct is not None else None
        )
        self._atr_sl_multiplier = (
            Decimal(str(atr_sl_multiplier)) if atr_sl_multiplier is not None else None
        )
        self._atr_tp_multiplier = (
            Decimal(str(atr_tp_multiplier)) if atr_tp_multiplier is not None else None
        )
        self._atr_period = atr_period
        self._strategy_id = strategy_id
        self._pending_exit_ttl = pending_exit_ttl

        # Symbols with an emitted-but-unfilled exit (fill-latency guard).
        self._pending_exit_symbols: set[str] = set()
        # Position quantity captured when the exit was emitted — a later
        # decrease signals a partial fill that left a residual to re-protect.
        self._pending_qty: dict[str, Decimal] = {}
        # check()-call count since pending was set, for the TTL backstop.
        self._pending_age: dict[str, int] = {}
        # ATR-mode symbols already warned about a missing ATR (log once).
        self._atr_warned_symbols: set[str] = set()

        self._log = structlog.get_logger(__name__).bind(
            component="bracket_exit",
            bracket_mode=bracket_mode,
        )

    # ------------------------------------------------------------------ #
    # Read-only properties
    # ------------------------------------------------------------------ #

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def bracket_mode(self) -> str:
        return self._mode

    @property
    def atr_period(self) -> int:
        return self._atr_period

    @property
    def requires_atr(self) -> bool:
        """True when the engine must supply an ATR value to :meth:`check`."""
        return self._mode == "atr"

    @property
    def pending_exit_symbols(self) -> set[str]:
        """Symbols with emitted but unfilled exit signals (read-only copy)."""
        return set(self._pending_exit_symbols)

    @property
    def pending_exit_ttl(self) -> int:
        """Checks before a stuck pending exit is force-cleared (CR-003)."""
        return self._pending_exit_ttl

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #

    def check(
        self,
        symbol: str,
        current_price: Decimal,
        position: Position | None,
        *,
        atr_value: Decimal | None = None,
    ) -> Signal | None:
        """
        Check stop-loss / take-profit brackets for a single symbol.

        Parameters
        ----------
        symbol : str
            The trading pair.
        current_price : Decimal
            Latest close price for this symbol.
        position : Position | None
            Current position (None or flat = nothing to protect).
        atr_value : Decimal | None
            Latest ATR in price units.  Required in ``"atr"`` mode; ignored
            in ``"fixed"`` mode.  When None/<=0 in ATR mode, no exit is
            emitted (the position is held) and a warning is logged once.

        Returns
        -------
        Signal | None
            A full-close SELL signal if a bracket is breached, else None.
        """
        # No position or flat — clean up tracking (the exit filled fully).
        if position is None or position.is_flat:
            self._clear_pending(symbol)
            self._atr_warned_symbols.discard(symbol)
            return None

        # An exit was emitted but the position is not yet flat.  Decide
        # whether to keep waiting (order in flight) or re-protect a residual.
        if symbol in self._pending_exit_symbols:
            # ``prev_qty`` is only ever None if _pending_exit_symbols and
            # _pending_qty desynchronised — impossible in this class (_emit
            # writes both, _clear_pending clears both), and the TTL backstop
            # below would bound any future desync to ``pending_exit_ttl`` bars.
            prev_qty = self._pending_qty.get(symbol)
            if prev_qty is not None and position.quantity != prev_qty:
                # The position size CHANGED while the exit was pending: a
                # partial fill left a smaller residual, OR the strategy added
                # to the position (re-entry).  Either way the prior pending
                # exit no longer covers the live position — clear the guard
                # and fall through to re-protect the current size now.
                self._log.info(
                    "bracket_exit.quantity_changed_reprotect",
                    symbol=symbol,
                    prev_quantity=str(prev_qty),
                    new_quantity=str(position.quantity),
                    direction=(
                        "partial_fill"
                        if position.quantity < prev_qty
                        else "increase"
                    ),
                )
                self._clear_pending(symbol)
            else:
                # Order still in flight (quantity unchanged).  Wait up to the
                # TTL, then force-clear defensively so a stuck order cannot
                # leave the position permanently unprotected.
                self._pending_age[symbol] = self._pending_age.get(symbol, 0) + 1
                if self._pending_age[symbol] < self._pending_exit_ttl:
                    return None
                self._log.warning(
                    "bracket_exit.pending_ttl_expired",
                    symbol=symbol,
                    ttl=self._pending_exit_ttl,
                    note="re-evaluating brackets; prior exit may be stuck",
                )
                self._clear_pending(symbol)
            # fall through to re-evaluate brackets for the open/residual position

        entry = position.average_entry_price
        if entry <= Decimal(0):
            return None

        stop_price, take_profit_price = self._levels(symbol, entry, atr_value)
        if stop_price is None and take_profit_price is None:
            return None

        # Precedence: stop-loss first (hard risk floor wins on a gap).
        if stop_price is not None and current_price <= stop_price:
            return self._emit(
                symbol=symbol,
                entry=entry,
                current_price=current_price,
                level=stop_price,
                quantity=position.quantity,
                is_stop_loss=True,
            )

        if take_profit_price is not None and current_price >= take_profit_price:
            return self._emit(
                symbol=symbol,
                entry=entry,
                current_price=current_price,
                level=take_profit_price,
                quantity=position.quantity,
                is_stop_loss=False,
            )

        return None

    def _clear_pending(self, symbol: str) -> None:
        """Drop all pending-exit tracking for ``symbol``."""
        self._pending_exit_symbols.discard(symbol)
        self._pending_qty.pop(symbol, None)
        self._pending_age.pop(symbol, None)

    def _levels(
        self,
        symbol: str,
        entry: Decimal,
        atr_value: Decimal | None,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Return ``(stop_price, take_profit_price)`` for the active mode."""
        if self._mode == "fixed":
            stop = (
                entry * (Decimal("1") - self._stop_loss_pct)
                if self._stop_loss_pct is not None
                else None
            )
            tp = (
                entry * (Decimal("1") + self._take_profit_pct)
                if self._take_profit_pct is not None
                else None
            )
            return stop, tp

        # ATR mode.
        if atr_value is None or atr_value <= Decimal(0):
            if symbol not in self._atr_warned_symbols:
                self._atr_warned_symbols.add(symbol)
                self._log.warning(
                    "bracket_exit.atr_unavailable",
                    symbol=symbol,
                    note="holding position; no ATR bracket this bar",
                )
            return None, None

        stop = (
            entry - self._atr_sl_multiplier * atr_value
            if self._atr_sl_multiplier is not None
            else None
        )
        tp = (
            entry + self._atr_tp_multiplier * atr_value
            if self._atr_tp_multiplier is not None
            else None
        )
        # A stop computed below zero is meaningless — treat as no stop.
        if stop is not None and stop <= Decimal(0):
            stop = None
        return stop, tp

    def _emit(
        self,
        *,
        symbol: str,
        entry: Decimal,
        current_price: Decimal,
        level: Decimal,
        quantity: Decimal,
        is_stop_loss: bool,
    ) -> Signal:
        """Build the full-close SELL signal and mark the symbol pending."""
        self._pending_exit_symbols.add(symbol)
        # Capture the position quantity so a later partial fill (which reduces
        # it) is detectable, and reset the TTL age for this pending exit.
        self._pending_qty[symbol] = quantity
        self._pending_age[symbol] = 0
        trigger = "bracket_stop_loss" if is_stop_loss else "bracket_take_profit"

        self._log.info(
            f"bracket_exit.{trigger}",
            symbol=symbol,
            entry_price=str(entry),
            level_price=str(level),
            current_price=str(current_price),
        )

        metadata: dict[str, object] = {
            "trigger": trigger,
            "bracket_mode": self._mode,
            "entry_price": str(entry),
            "current_price": str(current_price),
        }
        # Exactly one of these is set so ExitReasonDetector classifies
        # the trade correctly (it checks stop_loss before take_profit).
        if is_stop_loss:
            metadata["stop_loss"] = True
            metadata["stop_price"] = str(level)
        else:
            metadata["take_profit"] = True
            metadata["take_profit_price"] = str(level)

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=SignalDirection.SELL,
            target_position=Decimal("0"),
            confidence=1.0,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Clear all tracking state."""
        self._pending_exit_symbols.clear()
        self._pending_qty.clear()
        self._pending_age.clear()
        self._atr_warned_symbols.clear()
