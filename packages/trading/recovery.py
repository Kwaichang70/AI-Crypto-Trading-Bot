"""
packages/trading/recovery.py
-----------------------------
Pure (no DB, no I/O) domain helpers for the WP1.8 orphan-recovery/resume
pipeline.

This module holds the value objects and validation logic that both the
paper boot-resume path (``apps.api.routers.runs.recover_orphaned_runs``)
and the live resume path (``apps.api.services.run_recovery``) share.
Keeping it dependency-free (no SQLAlchemy, no FastAPI) makes it directly
unit-testable and keeps ``packages/trading`` free of an ``apps`` import.

Scope note (WP1.8a vs 1.8b)
----------------------------
``check_fill_integrity`` implements every WP18-R-06 / O10 structural check
that can be evaluated from persisted fills/orders alone. The two checks
that require a live exchange scan (S2/S3 -- cancel-then-import,
order-level dedup against exchange trade history) are WP1.8b scope and
live in ``apps.api.services.run_recovery``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from common.types import OrderSide
from trading.models import Fill, Order

__all__ = [
    "ResumeRejected",
    "ResumeSnapshot",
    "check_fill_integrity",
    "replay_sort_key",
]

#: Default tolerance for quantity comparisons -- one satoshi-scale unit.
#: Matches the WP1.1 I8 tolerance convention (one amount step, or this
#: fallback when the step size is unknown).
_DEFAULT_TOLERANCE = Decimal("0.00000001")


def replay_sort_key(fill: Fill) -> tuple[datetime, int, str]:
    """Deterministic fill replay/audit ordering (WP1.8b S2-02).

    ``(executed_at, buy_before_sell, fill_id)`` -- when a BUY and a SELL
    share the exact same ``executed_at`` (two fills timestamped by the
    same exchange trade batch, or two synthesised fills sharing an
    order's ``updated_at``), the BUY is always replayed first. Without
    this tie-break the previous ``(executed_at, str(fill_id))`` key
    ordered same-timestamp fills by random UUID, so a legitimate
    BUY-then-SELL history could occasionally replay SELL-first and be
    rejected as an oversell by :func:`check_fill_integrity` (WP18a-S2-02,
    fail-closed but availability-costly). Shared by
    :func:`check_fill_integrity`, ``PortfolioAccounting.from_fills``,
    ``PaperExecutionEngine.restore_from_fills`` and the resume endpoint's
    own audit-payload reconstruction so every replay/report orders
    identically.
    """
    return (fill.executed_at, 0 if fill.side == OrderSide.BUY else 1, str(fill.fill_id))


class ResumeRejected(Exception):
    """
    Raised whenever a resume/rebuild must fail closed (S11: never
    overridable).

    Attributes
    ----------
    reason:
        A short machine-readable reason code (e.g. ``"fill_history_corrupt"``,
        ``"fill_history_partial"``, ``"exchange_scan_incomplete"``,
        ``"exchange_scan_failed"``, ``"order_cancel_failed"``,
        ``"order_cancel_timeout"``, ``"trade_fetch_failed"`` -- the last
        five raised by WP1.8b's ``apps.api.services.run_recovery
        .scan_and_import``, not this module). Callers surface this
        verbatim in the 409 response body and in the
        ``run_resume_rejected`` audit event payload.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ResumeSnapshot:
    """
    Everything :meth:`PortfolioAccounting.from_fills` and the paper/live
    resume paths need, loaded once from the database (WP1.8a §4/§5).

    Pure value object -- no DB session, no ORM types.  Built by
    ``apps.api.services.run_persistence.load_resume_snapshot``.
    """

    run_id: str
    initial_cash: Decimal
    fills: Sequence[Fill]
    orders: Sequence[Order]
    peak_equity_hint: Decimal | None
    max_bar_index: int


def check_fill_integrity(
    fills: Sequence[Fill],
    orders: Sequence[Order],
    *,
    symbols: Collection[str],
    tolerance: Decimal = _DEFAULT_TOLERANCE,
) -> None:
    """
    Validate persisted fill/order history before it seeds a resume replay
    (O10, WP18-R-06).

    Raises :class:`ResumeRejected` on the first violation found. Pure and
    deterministic: never mutates ``fills``/``orders``, and the result
    depends only on their contents.

    Checks (in replay order, ``(executed_at, fill_id)``):
    - ``quantity <= 0`` or ``price <= 0`` -> ``"fill_history_corrupt"``.
    - a duplicate ``(order_id, executed_at, quantity, price)`` tuple ->
      ``"fill_history_corrupt"``.
    - ``fill.symbol`` not one of the run's own ``symbols`` ->
      ``"fill_history_corrupt"``.
    - ``fee_currency`` equals the symbol's base asset, raw and never
      normalised (a base-currency fee must already have been netted out of
      ``quantity`` at fill time -- WP1.1 D6) -> ``"fill_history_corrupt"``.
    - a fill referencing an ``order_id`` absent from ``orders`` (an orphan
      fill) -> ``"fill_history_corrupt"``.
    - the parent order's ``symbol`` not one of the run's own ``symbols``
      (WP1.8a-round2 S-05/C-02) -> ``"fill_history_corrupt"``.
    - ``fill.symbol`` or ``fill.side`` disagreeing with its parent order's
      ``symbol``/``side`` (WP1.8a-round2 S-05/C-02 -- e.g. a 0-quantity
      opposite-side fill smuggled onto the wrong order) ->
      ``"fill_history_corrupt"``.
    - a SELL larger than the running own quantity for that symbol at that
      point in the replay (the live incremental clamp would silently hide
      this; a resume must not) -> ``"fill_history_corrupt"``.
    - per order, ``sum(fill.quantity for fills of that order)`` not within
      ``tolerance`` of ``order.filled_quantity`` (net vs gross -- e.g. the
      30s flush window lost a fill, S6) -> ``"fill_history_partial"``.

    Parameters
    ----------
    fills:
        Every persisted fill for the run, in any order.
    orders:
        Every persisted order for the run.
    symbols:
        The run's own configured symbol set.
    tolerance:
        Quantity comparison tolerance (default: one amount-step fallback,
        matching WP1.1 I8).
    """
    orders_by_id = {order.order_id: order for order in orders}
    ordered_fills = sorted(fills, key=replay_sort_key)

    # WP18a-S2-02 (INFO finding, promoted to a binding 1.8b condition): the
    # per-fill loop below only ever looks up ``orders_by_id`` for an order
    # that HAS a fill -- a foreign-symbol order with zero fills was
    # previously invisible to this check entirely (it has no bearing on
    # position math, but a resume must still fail closed on ANY row that
    # does not belong to this run, fill or no fill). Check every persisted
    # order's symbol up front, independent of whether it has fills.
    for persisted_order in orders:
        if persisted_order.symbol not in symbols:
            raise ResumeRejected("fill_history_corrupt")

    running_qty: dict[str, Decimal] = {}
    seen_keys: set[tuple[object, object, Decimal, Decimal]] = set()

    for fill in ordered_fills:
        if fill.quantity <= Decimal("0") or fill.price <= Decimal("0"):
            raise ResumeRejected("fill_history_corrupt")

        dedup_key = (fill.order_id, fill.executed_at, fill.quantity, fill.price)
        if dedup_key in seen_keys:
            raise ResumeRejected("fill_history_corrupt")
        seen_keys.add(dedup_key)

        if fill.symbol not in symbols:
            raise ResumeRejected("fill_history_corrupt")

        base_asset = fill.symbol.split("/")[0]
        # WP18a-P-03 (false positive, fixed 1.8b): a NON-ZERO base-currency
        # fee must already have been netted out of quantity at fill time
        # (WP1.1 D6's _normalize_trade_fee) -- that is the actual
        # corruption signal. A zero-fee record whose fee_currency
        # nonetheless reports the base asset is not: some exchanges
        # default the fee-currency field to the market base even when
        # cost=0 (nothing was charged, so D6 had nothing to normalise),
        # and rejecting that would fail a perfectly legitimate history
        # closed. Documented rule: only ``fee_currency == base_asset AND
        # fee > 0`` is corrupt.
        if fill.fee_currency == base_asset and fill.fee > Decimal("0"):
            raise ResumeRejected("fill_history_corrupt")

        order = orders_by_id.get(fill.order_id)
        if order is None:
            raise ResumeRejected("fill_history_corrupt")

        # WP1.8a-round2 (S-05/C-02): a fill's own symbol/side must be
        # internally consistent with its parent order (the order's OWN
        # symbol is already checked against ``symbols`` up front above,
        # WP18a-S2-02). Without this, a mislabelled or foreign-symbol fill
        # could pass every other check (a 0 vs 0.02 quantity mismatch is
        # otherwise invisible to the corrupt-history checks above) and
        # silently poison the rebuilt portfolio on resume.
        if fill.symbol != order.symbol or fill.side != order.side:
            raise ResumeRejected("fill_history_corrupt")

        held = running_qty.get(fill.symbol, Decimal("0"))
        if fill.side == OrderSide.BUY:
            running_qty[fill.symbol] = held + fill.quantity
        else:
            if fill.quantity > held + tolerance:
                raise ResumeRejected("fill_history_corrupt")
            running_qty[fill.symbol] = held - fill.quantity

    fills_by_order: dict[object, Decimal] = {}
    for fill in ordered_fills:
        fills_by_order[fill.order_id] = (
            fills_by_order.get(fill.order_id, Decimal("0")) + fill.quantity
        )

    for order in orders:
        summed = fills_by_order.get(order.order_id, Decimal("0"))
        if abs(summed - order.filled_quantity) > tolerance:
            raise ResumeRejected("fill_history_partial")
