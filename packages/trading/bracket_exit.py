"""
packages/trading/bracket_exit.py
--------------------------------
Bracket exit manager — fixed-percentage *and* ATR-multiple stop-loss /
take-profit exits.  Emits synthetic SELL signals when an open position's
price crosses a stop-loss (below entry) or take-profit (above entry) level.

This is the fixed/volatility-scaled counterpart to
:class:`trading.trailing_stop.TrailingStopManager`.  Whereas the trailing
stop floats with the price peak, brackets are **anchored once per lot** and
never move (WP1.3b-a, see below).

Designed to be composed into ``StrategyEngine._process_bar()`` without
modifying execution engines (paper or live), exactly like the trailing stop.

Two modes (``bracket_mode``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``"fixed"`` — levels are a fixed fraction of the entry basis::

      stop_price        = entry * (1 - stop_loss_pct)
      take_profit_price = entry * (1 + take_profit_pct)

* ``"atr"`` — levels are an ATR multiple away from the entry basis
  (volatility adaptive).  The ATR is computed **once per lot** (the ATR of
  the signal bar, frozen) and never re-read::

      stop_price        = entry - atr_sl_multiplier * atr0
      take_profit_price = entry + atr_tp_multiplier * atr0

Anchoring (WP1.3b-a, SY-13b-01..03, D-13b-04: all modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. A *lot* runs from the BUY fill that takes a symbol from flat to open until
   it is flat again.  A partial SELL does not end it.  The *opening order* is
   the order of that first fill.
2. The entry basis ``E`` is the all-in blended average over the opening
   order's BUY fills (``portfolio.blend_entry_price``).  Adds from other
   orders never change ``E`` or the levels, up or down (user decision U-6).
3. Levels are stored as prices in a :class:`BracketAnchor`.  They change only
   when a further fill of the opening order re-blends ``E`` (the ATR stays
   frozen) or once on finalisation of a provisional/deferred anchor.
4. The anchor bar is a pure function of persisted data
   (:func:`select_anchor_bar`, ``anchor_at = min(order.created_at,
   first fill executed_at)``) so a resume recomputes exactly what the live
   run anchored.
5. A lot without an anchor (hook exception, direct unit use) is anchored
   lazily at the first :meth:`BracketExitManager.check` from the position's
   ``average_entry_price`` and the caller's ``atr_value`` (source ``late``).

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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal
from uuid import UUID

import structlog

from common.models import OHLCVBar
from common.types import OrderSide, SignalDirection
from trading.models import Fill, Order, Position, Signal
from trading.portfolio import blend_entry_price
from trading.recovery import replay_sort_key

__all__ = [
    "ATR_MULTIPLIER_RANGE",
    "ATR_PERIOD_MIN",
    "FIXED_PCT_RANGE",
    "VALID_BRACKET_MODES",
    "AnchorSource",
    "BracketAnchor",
    "BracketExitManager",
    "LotEntry",
    "anchor_bar_timestamp",
    "lot_entry_from_fills",
    "lot_entry_from_opening_fills",
    "open_lot_fills",
    "select_anchor_bar",
    "window_is_contiguous",
]

logger = structlog.get_logger(__name__)

_VALID_MODES = frozenset({"fixed", "atr"})

# WP1.3a (SY-13a-01): mechanical constructor ranges, exported so
# ``packages/trading/exit_config.py`` is the single source of truth for the
# *policy* bounds layered on top (see that module's module docstring).  These
# constants are also used directly below, so the constructor and the shared
# validator can never drift out of sync.
VALID_BRACKET_MODES: frozenset[str] = _VALID_MODES
FIXED_PCT_RANGE: tuple[float, float] = (0.001, 0.95)
ATR_MULTIPLIER_RANGE: tuple[float, float] = (0.1, 20.0)
ATR_PERIOD_MIN: int = 2

_PRECISION = Decimal("0.00000001")

#: FX-07: a finalisation that moves the stop DOWN by more than this many bps
#: (the interim stop was tighter than the exact one, i.e. the accepted RK-11
#: one-time loosening) is logged at warning level; any other move at info level.
_FINALIZE_WARN_DELTA_BPS: float = -25.0

#: Where a :class:`BracketAnchor` came from (logged and metric-tagged).
AnchorSource = Literal[
    "entry",
    "entry_provisional",
    "entry_deferred",
    "entry_unverified",
    "resume_recomputed",
    "resume_fallback_current_atr",
    "late",
]


@dataclass(frozen=True, slots=True)
class LotEntry:
    """Entry facts of the open lot, derived purely from persisted fills/orders.

    ``entry_basis`` is the all-in blend over the opening order's BUY fills
    (``blend_entry_price``); ``entry_gross_vwap`` is the fee-free VWAP of the
    same fills (the trailing stop's initial high-water mark).  ``anchor_at`` is
    ``min(order.created_at, first opening fill executed_at)`` (SY-13b-01).
    """

    symbol: str
    opening_order_id: UUID | None
    anchor_at: datetime
    entry_basis: Decimal
    entry_gross_vwap: Decimal
    basis_qty: Decimal


@dataclass(frozen=True, slots=True)
class BracketAnchor:
    """The frozen bracket levels of one lot (prices, not formulas)."""

    symbol: str
    lot_key: datetime
    opening_order_id: UUID | None
    anchor_at: datetime | None
    anchor_bar_ts: datetime | None
    entry_basis: Decimal
    basis_qty: Decimal
    atr: Decimal | None
    stop_price: Decimal | None
    take_profit_price: Decimal | None
    source: AnchorSource
    final: bool


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def select_anchor_bar(
    bars: Sequence[OHLCVBar], anchor_at: datetime, timeframe_s: int
) -> int | None:
    """Index of the last bar whose close (``timestamp + timeframe``) is at or
    before ``anchor_at`` (inclusive), or ``None`` when no bar qualifies.

    ``bars`` must be ascending by timestamp.  Pure (no clock).
    """
    tf = timedelta(seconds=timeframe_s)
    at = _as_utc(anchor_at)
    for i in range(len(bars) - 1, -1, -1):
        if _as_utc(bars[i].timestamp) + tf <= at:
            return i
    return None


def anchor_bar_timestamp(
    bars: Sequence[OHLCVBar], anchor_at: datetime, timeframe_s: int
) -> datetime | None:
    """Open time of the rule bar for ``anchor_at`` on the bar grid of ``bars``.

    Unlike :func:`select_anchor_bar` this is arithmetic: it also yields the
    timestamp when the rule bar is *newer* than the newest bar in the window
    (a bar boundary passed between the fetch and the order) or *older* than
    the oldest (a resumed lot older than the warmup window).  ``None`` for an
    empty ``bars``.
    """
    if not bars:
        return None
    tf = timedelta(seconds=timeframe_s)
    ref = _as_utc(bars[-1].timestamp)
    steps = (_as_utc(anchor_at) - tf - ref) // tf
    return ref + steps * tf


def window_is_contiguous(bars: Sequence[OHLCVBar], timeframe_s: int) -> bool:
    """True when every consecutive pair of bars is exactly one timeframe apart."""
    tf = timedelta(seconds=timeframe_s)
    return all(
        _as_utc(bars[i + 1].timestamp) - _as_utc(bars[i].timestamp) == tf
        for i in range(len(bars) - 1)
    )


def lot_entry_from_opening_fills(
    symbol: str, order: Order | None, fills: Sequence[Fill]
) -> LotEntry | None:
    """Build the :class:`LotEntry` from the opening order's BUY fills.

    The single computation shared by the live hook and the resume recompute
    (so both derive the same numbers).  ``None`` when there are no fills.
    """
    ordered = sorted(fills, key=replay_sort_key)
    if not ordered:
        return None
    basis = Decimal("0")
    qty = Decimal("0")
    gross_cost = Decimal("0")
    for fill in ordered:
        basis = blend_entry_price(qty, basis, fill)
        qty += fill.quantity
        gross_cost += fill.price * fill.quantity
    anchor_at = ordered[0].executed_at
    if order is not None:
        anchor_at = min(_as_utc(order.created_at), _as_utc(anchor_at))
    return LotEntry(
        symbol=symbol,
        opening_order_id=ordered[0].order_id,
        anchor_at=_as_utc(anchor_at),
        entry_basis=basis,
        entry_gross_vwap=(gross_cost / qty).quantize(_PRECISION, rounding=ROUND_HALF_UP),
        basis_qty=qty,
    )


def open_lot_fills(fills: Sequence[Fill], symbol: str) -> list[Fill]:
    """The opening order's BUY fills of the lot still open for ``symbol``.

    Replays ``fills`` in ``replay_sort_key`` order with the portfolio's own
    quantity arithmetic.  The lot opens at a BUY that takes the quantity from
    zero; only fills of that opening order are returned (adds from other
    orders are ignored, SY-13b-03).  Partial SELLs do not end the lot.
    ``[]`` when the symbol is flat.
    """
    qty = Decimal("0")
    opening_order_id: UUID | None = None
    opening_fills: list[Fill] = []
    for fill in sorted((f for f in fills if f.symbol == symbol), key=replay_sort_key):
        if fill.side == OrderSide.BUY:
            if qty <= Decimal("0"):
                opening_order_id = fill.order_id
                opening_fills = [fill]
                qty = fill.quantity
            else:
                qty += fill.quantity
                if fill.order_id == opening_order_id:
                    opening_fills.append(fill)
        elif fill.side == OrderSide.SELL:
            qty -= min(fill.quantity, qty)
    if qty <= Decimal("0"):
        return []
    return opening_fills


def lot_entry_from_fills(
    fills: Sequence[Fill], orders: Sequence[Order], symbol: str
) -> LotEntry | None:
    """Entry facts of the lot still open for ``symbol`` (``None`` when flat).

    Pure function of persisted fills/orders: the resume recompute (WP1.3b-a,
    SY-13b-01) and the live hook both end in :func:`lot_entry_from_opening_fills`.
    """
    opening_fills = open_lot_fills(fills, symbol)
    if not opening_fills:
        return None
    order_id = opening_fills[0].order_id
    order = next((o for o in orders if o.order_id == order_id), None)
    return lot_entry_from_opening_fills(symbol, order, opening_fills)


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
        ATR look-back period the engine uses when it freezes the lot's ATR at
        entry (and for the lazy ``late`` anchor's ``atr_value``).  Must be
        >= 2.  Default 14.
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
        On any invalid configuration.  WP1.3a (SY-13a-22): the engine no
        longer catches this to warn-and-disable brackets -- an invalid
        config is rejected at run creation by
        ``packages/trading/exit_config.py`` and, defensively, raises here
        straight into ``StrategyEngine.__init__`` (``ExitConfigError``,
        a ``ValueError`` subclass) so a bad config can never silently run
        with no exit.
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
                _lo, _hi = FIXED_PCT_RANGE
                if value is not None and not (_lo <= value <= _hi):
                    raise ValueError(
                        f"{name} must be in [{_lo}, {_hi}], got {value}"
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
                _lo, _hi = ATR_MULTIPLIER_RANGE
                if value is not None and not (_lo <= value <= _hi):
                    raise ValueError(
                        f"{name} must be in [{_lo}, {_hi}], got {value}"
                    )
            if atr_period < ATR_PERIOD_MIN:
                raise ValueError(f"atr_period must be >= {ATR_PERIOD_MIN}, got {atr_period}")

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
        # WP1.3b-a: the frozen per-lot levels (one anchor per open symbol).
        self._anchors: dict[str, BracketAnchor] = {}
        # Symbols already warned about a lazily-created ("late") anchor.
        self._late_warned: set[str] = set()

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
    def has_stop_loss(self) -> bool:
        """True iff the active mode configures a stop-loss leg (a TP-only bracket has none)."""
        if self._mode == "fixed":
            return self._stop_loss_pct is not None
        return self._atr_sl_multiplier is not None

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
        Check the lot's frozen stop-loss / take-profit levels for one symbol.

        Parameters
        ----------
        symbol : str
            The trading pair.
        current_price : Decimal
            Latest close price for this symbol.
        position : Position | None
            Current position (None or flat = nothing to protect; the anchor
            and any pending exit are dropped).
        atr_value : Decimal | None
            Only used when the lot has **no anchor** (lazy ``late`` path:
            hook exception, direct unit use): the ATR is then frozen at the
            first value passed.  Ignored for anchored lots -- their levels are
            prices and never follow a later ATR (WP1.3b-a, AC-13B-01).  In ATR
            mode, no anchor and no usable ATR means no exit and a warning
            (logged once).

        Returns
        -------
        Signal | None
            A full-close SELL signal if a bracket is breached, else None.
        """
        # No position or flat — clean up tracking (the exit filled fully).
        if position is None or position.is_flat:
            self._clear_pending(symbol)
            self._atr_warned_symbols.discard(symbol)
            self._late_warned.discard(symbol)
            self._anchors.pop(symbol, None)
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

        anchor = self._anchor_for(symbol, position, atr_value)
        if anchor is None:
            return None
        stop_price, take_profit_price = anchor.stop_price, anchor.take_profit_price
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
                anchor=anchor,
            )

        if take_profit_price is not None and current_price >= take_profit_price:
            return self._emit(
                symbol=symbol,
                entry=entry,
                current_price=current_price,
                level=take_profit_price,
                quantity=position.quantity,
                is_stop_loss=False,
                anchor=anchor,
            )

        return None

    # ------------------------------------------------------------------ #
    # Anchor book (WP1.3b-a)
    # ------------------------------------------------------------------ #

    def get_anchor(self, symbol: str) -> BracketAnchor | None:
        """The lot's anchor, or ``None`` (read-only; anchors are frozen)."""
        return self._anchors.get(symbol)

    def projected_levels(
        self, entry_price: Decimal, atr_value: Decimal | None
    ) -> tuple[Decimal | None, Decimal | None]:
        """Pure ``(stop, take_profit)`` for a hypothetical entry (WP1.10 hook).

        The exact formula the anchor uses, so a sizing step can price the stop
        before the order exists.  No state, no logging.
        """
        return self._compute_levels(entry_price, atr_value)

    def protective_level(self, symbol: str, position: Position | None) -> Decimal | None:
        """The anchored stop-loss price for an open lot, else ``None`` (pure)."""
        if position is None or position.is_flat:
            return None
        anchor = self._anchors.get(symbol)
        if anchor is None or anchor.lot_key != position.opened_at:
            return None
        return anchor.stop_price

    def open_lot(
        self,
        symbol: str,
        *,
        lot_key: datetime,
        opening_order_id: UUID | None,
        anchor_at: datetime | None,
        anchor_bar_ts: datetime | None,
        entry_basis: Decimal,
        basis_qty: Decimal,
        atr: Decimal | None,
        source: AnchorSource,
        final: bool,
    ) -> BracketAnchor:
        """Create (replace) the anchor for a lot that just went flat -> open."""
        stop, tp = self._compute_levels(entry_basis, atr)
        anchor = BracketAnchor(
            symbol=symbol,
            lot_key=lot_key,
            opening_order_id=opening_order_id,
            anchor_at=anchor_at,
            anchor_bar_ts=anchor_bar_ts,
            entry_basis=entry_basis,
            basis_qty=basis_qty,
            atr=atr,
            stop_price=stop,
            take_profit_price=tp,
            source=source,
            final=final,
        )
        self._anchors[symbol] = anchor
        self._clear_pending(symbol)
        self._atr_warned_symbols.discard(symbol)
        self._late_warned.discard(symbol)
        self._log_anchor(anchor)
        return anchor

    def add_opening_fill(
        self,
        symbol: str,
        *,
        order_id: UUID,
        entry_basis: Decimal,
        basis_qty: Decimal,
    ) -> BracketAnchor | None:
        """A further BUY fill in the lot.

        A fill of the **opening order** re-blends the entry basis (the ATR
        stays frozen) and recomputes the levels.  A fill of any other order is
        an add: the levels stay (user decision U-6) and
        ``bracket_exit.anchor_kept_on_add`` is logged.  No-op without an anchor.
        """
        anchor = self._anchors.get(symbol)
        if anchor is None:
            return None
        if anchor.opening_order_id != order_id:
            self._log.info(
                "bracket_exit.anchor_kept_on_add",
                symbol=symbol,
                stop_price=str(anchor.stop_price) if anchor.stop_price is not None else None,
                take_profit_price=(
                    str(anchor.take_profit_price)
                    if anchor.take_profit_price is not None
                    else None
                ),
            )
            return anchor
        stop, tp = self._compute_levels(entry_basis, anchor.atr)
        updated = replace(
            anchor,
            entry_basis=entry_basis,
            basis_qty=basis_qty,
            stop_price=stop,
            take_profit_price=tp,
        )
        self._anchors[symbol] = updated
        self._log_anchor(updated)
        return updated

    def arm_interim(
        self,
        symbol: str,
        *,
        atr: Decimal,
        anchor_bar_ts: datetime | None,
        source: AnchorSource,
    ) -> BracketAnchor | None:
        """Arm the INTERIM levels of a pending anchor that has none yet (FX-02/03).

        The interim ATR is a tightest-estimate stand-in while the exact window is
        unavailable (SEC-01).  It is written **once**: a no-op returning ``None``
        when the anchor is missing, already final, or already armed (carries an
        ATR or any level), so a pending anchor can never be re-armed or loosened
        bar to bar.  Otherwise the levels come from ``_compute_levels`` on the
        existing ``entry_basis``; the anchor stays ``final=False`` and is logged.
        """
        anchor = self._anchors.get(symbol)
        if anchor is None or anchor.final:
            return None
        if (
            anchor.atr is not None
            or anchor.stop_price is not None
            or anchor.take_profit_price is not None
        ):
            return None
        stop, tp = self._compute_levels(anchor.entry_basis, atr)
        updated = replace(
            anchor,
            atr=atr,
            anchor_bar_ts=anchor_bar_ts if anchor_bar_ts is not None else anchor.anchor_bar_ts,
            stop_price=stop,
            take_profit_price=tp,
            source=source,
            final=False,
        )
        self._anchors[symbol] = updated
        self._log_anchor(updated)
        return updated

    def finalize_anchor(
        self,
        symbol: str,
        *,
        atr: Decimal | None,
        anchor_bar_ts: datetime | None,
        source: AnchorSource,
    ) -> BracketAnchor | None:
        """Finalise a provisional/deferred anchor **exactly once**.

        A second call (or a call on an already-final anchor) is a no-op that
        returns ``None``.  ``atr=None`` keeps the existing ATR (and levels).
        Logs ``bracket_exit.anchor_finalized`` with ``prev_stop_price``,
        ``stop_price`` and ``delta_bps`` (WP13BA-SEC-07): warning when the stop
        moved down by more than 25 bps, info otherwise.
        """
        anchor = self._anchors.get(symbol)
        if anchor is None or anchor.final:
            return None
        new_atr = atr if atr is not None else anchor.atr
        stop, tp = self._compute_levels(anchor.entry_basis, new_atr)
        updated = replace(
            anchor,
            atr=new_atr,
            anchor_bar_ts=anchor_bar_ts if anchor_bar_ts is not None else anchor.anchor_bar_ts,
            stop_price=stop,
            take_profit_price=tp,
            source=source,
            final=True,
        )
        self._anchors[symbol] = updated
        self._log_anchor(updated)
        self._log_finalized(anchor, updated)
        return updated

    def _log_finalized(self, prev: BracketAnchor, new: BracketAnchor) -> None:
        """``bracket_exit.anchor_finalized``: the size of the interim -> final move."""
        delta_bps: float | None = None
        if (
            prev.stop_price is not None
            and new.stop_price is not None
            and new.entry_basis > Decimal(0)
        ):
            delta_bps = round(
                float((new.stop_price - prev.stop_price) / new.entry_basis * 10_000), 4
            )
        emit = (
            self._log.warning
            if delta_bps is not None and delta_bps < _FINALIZE_WARN_DELTA_BPS
            else self._log.info
        )
        emit(
            "bracket_exit.anchor_finalized",
            symbol=new.symbol,
            source=new.source,
            prev_stop_price=str(prev.stop_price) if prev.stop_price is not None else None,
            stop_price=str(new.stop_price) if new.stop_price is not None else None,
            delta_bps=delta_bps,
        )

    def restore_anchor(self, anchor: BracketAnchor) -> None:
        """Install an anchor recomputed by a resume (replaces any existing)."""
        self._anchors[anchor.symbol] = anchor
        self._clear_pending(anchor.symbol)
        self._log_anchor(anchor)

    def clear_pending(self, symbol: str) -> None:
        """Drop the pending-exit guard for ``symbol`` (R-04).

        Called by the engine when an exit signal was **not submitted** (empty
        order list, every order terminal and unfilled, or ``process_signal``
        raised): the next evaluation may then re-emit instead of waiting for
        the TTL.
        """
        self._clear_pending(symbol)

    def _log_anchor(self, anchor: BracketAnchor) -> None:
        self._log.info(
            "bracket_exit.anchored",
            symbol=anchor.symbol,
            source=anchor.source,
            final=anchor.final,
            anchor_bar_ts=str(anchor.anchor_bar_ts) if anchor.anchor_bar_ts else None,
            entry_basis=str(anchor.entry_basis),
            atr0=str(anchor.atr) if anchor.atr is not None else None,
            stop_price=str(anchor.stop_price) if anchor.stop_price is not None else None,
            take_profit_price=(
                str(anchor.take_profit_price) if anchor.take_profit_price is not None else None
            ),
        )

    def _anchor_for(
        self,
        symbol: str,
        position: Position,
        atr_value: Decimal | None,
    ) -> BracketAnchor | None:
        """The anchor protecting ``position`` (creating a ``late`` one lazily)."""
        anchor = self._anchors.get(symbol)
        if anchor is not None and anchor.lot_key != position.opened_at:
            self._log.warning(
                "bracket_exit.anchor_lot_mismatch",
                symbol=symbol,
                anchor_lot_key=str(anchor.lot_key),
                position_opened_at=str(position.opened_at),
            )
            del self._anchors[symbol]
            anchor = None
        if anchor is not None:
            return anchor

        entry = position.average_entry_price
        if self._mode == "atr" and (atr_value is None or atr_value <= Decimal(0)):
            if symbol not in self._atr_warned_symbols:
                self._atr_warned_symbols.add(symbol)
                self._log.warning(
                    "bracket_exit.atr_unavailable",
                    symbol=symbol,
                    note="holding position; no ATR bracket this bar",
                )
            return None
        atr0 = atr_value if self._mode == "atr" else None
        stop, tp = self._compute_levels(entry, atr0)
        late = BracketAnchor(
            symbol=symbol,
            lot_key=position.opened_at,
            opening_order_id=None,
            anchor_at=None,
            anchor_bar_ts=None,
            entry_basis=entry,
            basis_qty=position.quantity,
            atr=atr0,
            stop_price=stop,
            take_profit_price=tp,
            source="late",
            final=True,
        )
        self._anchors[symbol] = late
        if symbol not in self._late_warned:
            self._late_warned.add(symbol)
            self._log.warning(
                "bracket_exit.anchor_late",
                symbol=symbol,
                note="no entry anchor for the open lot; anchored from the average "
                "entry price at the first check",
            )
        self._log_anchor(late)
        return late

    def _clear_pending(self, symbol: str) -> None:
        """Drop all pending-exit tracking for ``symbol``."""
        self._pending_exit_symbols.discard(symbol)
        self._pending_qty.pop(symbol, None)
        self._pending_age.pop(symbol, None)

    def _compute_levels(
        self,
        entry: Decimal,
        atr_value: Decimal | None,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Return ``(stop_price, take_profit_price)`` for the active mode (pure)."""
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
        anchor: BracketAnchor,
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
            anchor_source=anchor.source,
        )

        metadata: dict[str, object] = {
            "trigger": trigger,
            "bracket_mode": self._mode,
            "entry_price": str(entry),
            "current_price": str(current_price),
            "anchor_source": anchor.source,
            "atr_anchor": str(anchor.atr) if anchor.atr is not None else None,
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
        self._anchors.clear()
        self._late_warned.clear()
