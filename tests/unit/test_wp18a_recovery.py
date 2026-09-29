"""
tests/unit/test_wp18a_recovery.py
------------------------------------
Unit tests for ``packages/trading/recovery.py`` (WP1.8a: ``ResumeRejected``,
``ResumeSnapshot``, ``check_fill_integrity`` -- the O10/WP18-R-06 replay
integrity checks).

Mandatory: "every O10 case is rejected" (synthesis spec §6).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType
from trading.models import Fill, Order
from trading.recovery import ResumeRejected, check_fill_integrity

SYMBOL = "BTC/USDT"
SYMBOLS = {SYMBOL}


def _order(*, filled_quantity: str = "0.01", quantity: str = "0.01") -> Order:
    return Order(
        client_order_id=f"wp18a-{uuid4().hex[:12]}",
        run_id="wp18a-recovery-test",
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal(quantity),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal(filled_quantity),
    )


def _fill_for(
    order: Order,
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: str = "0.01",
    price: str = "50000",
    fee_currency: str = "USDT",
    executed_at: datetime | None = None,
) -> Fill:
    return Fill(
        order_id=order.order_id,
        symbol=order.symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal("0.5"),
        fee_currency=fee_currency,
        executed_at=executed_at or datetime(2026, 9, 20, tzinfo=UTC),
    )


class TestValidHistoryPasses:
    def test_consistent_fills_and_orders_pass(self) -> None:
        order = _order()
        fill = _fill_for(order)
        check_fill_integrity([fill], [order], symbols=SYMBOLS)  # no raise


class TestO10Violations:
    def test_negative_quantity_rejected(self) -> None:
        order = _order()
        # Pydantic itself forbids qty <= 0 (Field gt=0) -- exercise the
        # integrity check's own defence for a value that slipped through
        # (e.g. a future non-Pydantic loader), by asserting the check
        # itself rejects an out-of-range quantity built via model_copy
        # with validation disabled.
        fill = _fill_for(order).model_copy(update={"quantity": Decimal("0.01")})
        object.__setattr__(fill, "quantity", Decimal("-0.01"))
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_negative_price_rejected(self) -> None:
        order = _order()
        fill = _fill_for(order)
        object.__setattr__(fill, "price", Decimal("-1"))
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_symbol_not_in_run_rejected(self) -> None:
        order = _order()
        fill = _fill_for(order)
        object.__setattr__(fill, "symbol", "ETH/USDT")
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_fee_currency_equals_base_rejected(self) -> None:
        order = _order()
        fill = _fill_for(order, fee_currency="BTC")
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_orphan_order_id_rejected(self) -> None:
        order = _order()
        other_order = _order()
        fill = _fill_for(other_order)
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_sell_larger_than_running_qty_rejected(self) -> None:
        order = _order(filled_quantity="0.02", quantity="0.02")
        sell = _fill_for(
            order,
            side=OrderSide.SELL,
            quantity="0.02",
            executed_at=datetime(2026, 9, 20, 9, tzinfo=UTC),
        )
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([sell], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_duplicate_fill_tuple_rejected(self) -> None:
        order = _order(filled_quantity="0.02", quantity="0.02")
        fill = _fill_for(order)
        duplicate = fill.model_copy(update={"fill_id": uuid4()})
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill, duplicate], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_sum_fills_not_matching_filled_quantity_rejected(self) -> None:
        # order.filled_quantity says 0.02 but only one 0.01 fill is on file --
        # the 30s flush window lost the second fill (S6/WP18-R-06).
        order = _order(filled_quantity="0.02", quantity="0.02")
        fill = _fill_for(order, quantity="0.01")
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_partial"

    def test_sum_fills_within_tolerance_passes(self) -> None:
        order = _order(filled_quantity="0.01", quantity="0.01")
        fill = _fill_for(order, quantity="0.01000000005")
        check_fill_integrity(
            [fill], [order], symbols=SYMBOLS, tolerance=Decimal("0.0000001")
        )  # no raise


class TestS05FillOrderConsistency:
    """WP1.8a-round2 (S-05/C-02): a fill whose own side/symbol disagrees
    with its parent order, or an order whose symbol the run does not even
    trade, is invisible to every check above (the aggregate running-qty
    clamp and the per-order fill-sum check both operate on quantities
    alone -- a same-magnitude, wrong-side fill sails straight through
    both).  These are the "0.02-vs-0" cases flagged by security review:
    a fill correctly sized at 0.02 in isolation, but wrong-signed for its
    own order, silently corrupts the rebuilt per-symbol position from what
    should be 0 net to +/-0.02 (or vice versa) on resume.
    """

    def test_fill_side_disagrees_with_order_side_rejected(self) -> None:
        # order1: a legitimate BUY 0.02 that raises the symbol's running
        # qty to 0.02 -- this is what lets the corrupted fill below sneak
        # past the aggregate "SELL <= held" clamp (0.02 <= 0.02 passes).
        order1 = _order(filled_quantity="0.02", quantity="0.02")
        fill1 = _fill_for(order1, executed_at=datetime(2026, 9, 20, 8, tzinfo=UTC))

        # order2: also a BUY 0.02, fully filled -- but its OWN persisted
        # fill is mislabelled SELL.  Quantity (0.02) matches order2's own
        # filled_quantity exactly, so the per-order sum check passes too.
        # Only checking fill.side against order2.side catches this.
        order2 = _order(filled_quantity="0.02", quantity="0.02")
        fill2 = _fill_for(
            order2,
            side=OrderSide.SELL,
            quantity="0.02",
            executed_at=datetime(2026, 9, 20, 9, tzinfo=UTC),
        )
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill1, fill2], [order1, order2], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_fill_symbol_disagrees_with_order_symbol_rejected(self) -> None:
        symbols = {SYMBOL, "ETH/USDT"}
        order = _order()  # symbol=BTC/USDT
        fill = _fill_for(order)
        # fill.symbol is a DIFFERENT symbol the run also trades (so the
        # pre-existing "fill.symbol not in symbols" check does not fire),
        # but it disagrees with its own order's symbol.
        object.__setattr__(fill, "symbol", "ETH/USDT")
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=symbols)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_order_symbol_not_in_run_symbols_rejected(self) -> None:
        order = _order()  # symbol=BTC/USDT
        object.__setattr__(order, "symbol", "XRP/USDT")  # not a run symbol
        # fill.symbol is forged to BTC/USDT (a real run symbol) so the
        # pre-existing per-fill "fill.symbol not in symbols" check does
        # not fire -- only checking the ORDER's own symbol catches this.
        fill = _fill_for(order)
        object.__setattr__(fill, "symbol", SYMBOL)
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"


class TestWP18bBindingConditions:
    """WP1.8b: P-03 (fee_currency false positive), S2-02 (BUY-before-SELL
    tie-break; order-symbol check over ALL orders, not just orders with
    fills)."""

    def test_zero_fee_reported_in_base_currency_is_accepted(self) -> None:
        """P-03: a zero-fee record whose fee_currency nonetheless reports
        the base asset (some exchanges default the field even when
        cost=0) must NOT be rejected -- only a NON-ZERO base-currency fee
        (which D6 should already have netted out) is corruption."""
        order = _order()
        fill = _fill_for(order, fee_currency="BTC")
        object.__setattr__(fill, "fee", Decimal("0"))
        # Must not raise.
        check_fill_integrity([fill], [order], symbols=SYMBOLS)

    def test_nonzero_fee_reported_in_base_currency_still_rejected(self) -> None:
        """The documented rule's other half: fee > 0 with fee_currency ==
        base is still corrupt (unchanged from WP1.8a)."""
        order = _order()
        fill = _fill_for(order, fee_currency="BTC")
        object.__setattr__(fill, "fee", Decimal("0.0001"))
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([fill], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"

    def test_buy_before_sell_tie_break_on_equal_executed_at(self) -> None:
        """S2-02: a BUY and a SELL sharing the EXACT same executed_at must
        always replay BUY-first, regardless of fill_id ordering -- fed
        deliberately with fill_ids that would sort SELL-first under the
        OLD ``str(fill_id)`` tie-break (which would spuriously reject this
        legitimate history as an oversell)."""
        same_instant = datetime(2026, 9, 20, 12, tzinfo=UTC)
        buy_order = _order(filled_quantity="0.01", quantity="0.01")
        sell_order = _order(filled_quantity="0.01", quantity="0.01")
        object.__setattr__(sell_order, "side", OrderSide.SELL)
        buy = _fill_for(
            buy_order, side=OrderSide.BUY, quantity="0.01", price="49000", executed_at=same_instant
        )
        sell = _fill_for(
            sell_order,
            side=OrderSide.SELL,
            quantity="0.01",
            price="51000",
            executed_at=same_instant,
        )

        # Force fill_ids so a plain str(fill_id) sort would replay SELL
        # before BUY (the pre-1.8b failure mode -- a SELL against a still-
        # flat position is rejected as an oversell): keep drawing until
        # sell's id sorts lexically BEFORE buy's, deterministically
        # reproducing the old bug's worst case regardless of this run's
        # random UUIDs.
        object.__setattr__(sell, "fill_id", uuid4())
        object.__setattr__(buy, "fill_id", uuid4())
        while str(sell.fill_id) >= str(buy.fill_id):
            object.__setattr__(buy, "fill_id", uuid4())

        # Fed SELL-first in the input list too -- check_fill_integrity
        # must re-sort by replay_sort_key internally, not trust input order.
        check_fill_integrity([sell, buy], [buy_order, sell_order], symbols=SYMBOLS)

    def test_foreign_symbol_order_with_zero_fills_rejected(self) -> None:
        """S2-02: an order whose symbol the run does not trade, but which
        has NO fills at all, was previously invisible (the per-fill loop
        never inspects an order with zero fills) -- must now be rejected
        too."""
        order = _order()
        object.__setattr__(order, "symbol", "XRP/USDT")
        with pytest.raises(ResumeRejected) as exc_info:
            check_fill_integrity([], [order], symbols=SYMBOLS)
        assert exc_info.value.reason == "fill_history_corrupt"
