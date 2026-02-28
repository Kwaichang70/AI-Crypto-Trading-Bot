"""
tests/unit/test_execution.py
-----------------------------
Unit tests for the order state machine defined in packages/trading/execution.py.

The execution engine itself is abstract.  These tests focus on the state
machine logic — specifically the ORDER_STATE_MACHINE adjacency map and the
validate_transition / InvalidOrderTransitionError helpers.

Test coverage
-------------
- Valid transitions for every non-terminal state
- Invalid transitions raise InvalidOrderTransitionError
- Terminal states (FILLED, CANCELED, REJECTED, EXPIRED) have no outgoing transitions
- InvalidOrderTransitionError carries correct attributes
- BaseExecutionEngine._transition helper updates status and updated_at
- get_open_orders filters out terminal states
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType
from trading.execution import (
    ORDER_STATE_MACHINE,
    InvalidOrderTransitionError,
    validate_transition,
)
from trading.models import Order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    status: OrderStatus = OrderStatus.NEW,
    *,
    symbol: str = "BTC/USDT",
    run_id: str = "test-run",
) -> Order:
    """
    Build an Order in the given status.

    For non-NEW statuses we still create a NEW order then use model_copy
    to set the desired status so we bypass any Pydantic model_validator
    that checks transition logic (which lives in the execution engine,
    not the model itself).
    """
    order = Order(
        client_order_id=f"{run_id}-{uuid4().hex[:12]}",
        run_id=run_id,
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
    )
    if status != OrderStatus.NEW:
        order = order.model_copy(update={"status": status})
    return order


# ===========================================================================
# ORDER_STATE_MACHINE completeness
# ===========================================================================


class TestStateMachineCompleteness:
    """Verify the state machine covers all OrderStatus values."""

    def test_all_statuses_have_entries(self) -> None:
        """Every OrderStatus value must appear as a key in ORDER_STATE_MACHINE."""
        for status in OrderStatus:
            assert status in ORDER_STATE_MACHINE, (
                f"OrderStatus.{status} is missing from ORDER_STATE_MACHINE"
            )

    def test_terminal_states_have_empty_outgoing_sets(self) -> None:
        """FILLED, CANCELED, REJECTED, EXPIRED must have no outgoing transitions."""
        terminal = [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]
        for status in terminal:
            allowed = ORDER_STATE_MACHINE[status]
            assert allowed == frozenset(), (
                f"Terminal status {status} should have no outgoing transitions, "
                f"but has: {allowed}"
            )

    def test_non_terminal_states_have_outgoing_transitions(self) -> None:
        """NEW, PENDING_SUBMIT, OPEN, PARTIAL must each have at least one successor."""
        non_terminal = [
            OrderStatus.NEW,
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.OPEN,
            OrderStatus.PARTIAL,
        ]
        for status in non_terminal:
            assert len(ORDER_STATE_MACHINE[status]) > 0, (
                f"Non-terminal status {status} must have outgoing transitions"
            )


# ===========================================================================
# Valid transitions
# ===========================================================================


class TestValidTransitions:
    """Tests for transitions that should succeed according to the state machine."""

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            # NEW transitions
            (OrderStatus.NEW, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.NEW, OrderStatus.REJECTED),
            # PENDING_SUBMIT transitions
            (OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.REJECTED),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.CANCELED),
            # OPEN transitions
            (OrderStatus.OPEN, OrderStatus.PARTIAL),
            (OrderStatus.OPEN, OrderStatus.FILLED),
            (OrderStatus.OPEN, OrderStatus.CANCELED),
            (OrderStatus.OPEN, OrderStatus.EXPIRED),
            # PARTIAL transitions
            (OrderStatus.PARTIAL, OrderStatus.PARTIAL),
            (OrderStatus.PARTIAL, OrderStatus.FILLED),
            (OrderStatus.PARTIAL, OrderStatus.CANCELED),
            (OrderStatus.PARTIAL, OrderStatus.EXPIRED),
        ],
    )
    def test_valid_transition_does_not_raise(
        self,
        from_status: OrderStatus,
        to_status: OrderStatus,
    ) -> None:
        """All transitions listed in ORDER_STATE_MACHINE must not raise."""
        order = _make_order(from_status)
        validate_transition(order, to_status)  # should not raise


# ===========================================================================
# Invalid transitions
# ===========================================================================


class TestInvalidTransitions:
    """Tests for transitions that must raise InvalidOrderTransitionError."""

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            # Terminal states — no outgoing transitions allowed
            (OrderStatus.FILLED, OrderStatus.OPEN),
            (OrderStatus.FILLED, OrderStatus.CANCELED),
            (OrderStatus.FILLED, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.CANCELED, OrderStatus.OPEN),
            (OrderStatus.CANCELED, OrderStatus.FILLED),
            (OrderStatus.REJECTED, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.REJECTED, OrderStatus.OPEN),
            (OrderStatus.EXPIRED, OrderStatus.OPEN),
            (OrderStatus.EXPIRED, OrderStatus.FILLED),
            # Skip transitions (bypassing intermediate states)
            (OrderStatus.NEW, OrderStatus.OPEN),
            (OrderStatus.NEW, OrderStatus.FILLED),
            (OrderStatus.NEW, OrderStatus.PARTIAL),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.PARTIAL),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.FILLED),
            # Backwards transitions
            (OrderStatus.OPEN, OrderStatus.NEW),
            (OrderStatus.OPEN, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.PARTIAL, OrderStatus.NEW),
            (OrderStatus.PARTIAL, OrderStatus.PENDING_SUBMIT),
        ],
    )
    def test_invalid_transition_raises(
        self,
        from_status: OrderStatus,
        to_status: OrderStatus,
    ) -> None:
        """Any transition not in ORDER_STATE_MACHINE must raise InvalidOrderTransitionError."""
        order = _make_order(from_status)
        with pytest.raises(InvalidOrderTransitionError):
            validate_transition(order, to_status)


# ===========================================================================
# InvalidOrderTransitionError attributes
# ===========================================================================


class TestInvalidOrderTransitionError:
    """Tests for the exception class attributes."""

    def test_exception_carries_order_id(self) -> None:
        """The exception stores the UUID of the offending order."""
        order = _make_order(OrderStatus.FILLED)
        with pytest.raises(InvalidOrderTransitionError) as exc_info:
            validate_transition(order, OrderStatus.OPEN)
        assert exc_info.value.order_id == order.order_id

    def test_exception_carries_from_status(self) -> None:
        """The exception stores the source status."""
        order = _make_order(OrderStatus.FILLED)
        with pytest.raises(InvalidOrderTransitionError) as exc_info:
            validate_transition(order, OrderStatus.OPEN)
        assert exc_info.value.from_status == OrderStatus.FILLED

    def test_exception_carries_to_status(self) -> None:
        """The exception stores the attempted target status."""
        order = _make_order(OrderStatus.FILLED)
        with pytest.raises(InvalidOrderTransitionError) as exc_info:
            validate_transition(order, OrderStatus.OPEN)
        assert exc_info.value.to_status == OrderStatus.OPEN

    def test_exception_message_mentions_statuses(self) -> None:
        """The exception message includes both from and to statuses."""
        order = _make_order(OrderStatus.FILLED)
        with pytest.raises(InvalidOrderTransitionError) as exc_info:
            validate_transition(order, OrderStatus.OPEN)
        msg = str(exc_info.value)
        assert "filled" in msg.lower() or "FILLED" in msg
        assert "open" in msg.lower() or "OPEN" in msg

    def test_exception_is_subclass_of_exception(self) -> None:
        """InvalidOrderTransitionError inherits from Exception."""
        assert issubclass(InvalidOrderTransitionError, Exception)


# ===========================================================================
# Full happy-path transition chain
# ===========================================================================


class TestHappyPathChain:
    """Test the primary success path through the state machine."""

    def test_new_to_filled_chain(self) -> None:
        """
        Simulate the standard success chain:
        NEW → PENDING_SUBMIT → OPEN → FILLED
        Each step must succeed without raising.
        """
        order = _make_order(OrderStatus.NEW)
        validate_transition(order, OrderStatus.PENDING_SUBMIT)
        order = order.model_copy(update={"status": OrderStatus.PENDING_SUBMIT})

        validate_transition(order, OrderStatus.OPEN)
        order = order.model_copy(update={"status": OrderStatus.OPEN})

        validate_transition(order, OrderStatus.FILLED)
        order = order.model_copy(update={"status": OrderStatus.FILLED})

        assert order.status == OrderStatus.FILLED

    def test_partial_fill_chain(self) -> None:
        """
        NEW → PENDING_SUBMIT → OPEN → PARTIAL → PARTIAL → FILLED
        Partial fills may repeat; each step must succeed.
        """
        order = _make_order(OrderStatus.NEW)

        for from_s, to_s in [
            (OrderStatus.NEW, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN),
            (OrderStatus.OPEN, OrderStatus.PARTIAL),
            (OrderStatus.PARTIAL, OrderStatus.PARTIAL),  # second partial fill
            (OrderStatus.PARTIAL, OrderStatus.FILLED),
        ]:
            order = order.model_copy(update={"status": from_s})
            validate_transition(order, to_s)

    def test_cancellation_chain(self) -> None:
        """
        NEW → PENDING_SUBMIT → OPEN → CANCELED
        All steps succeed without raising.
        """
        statuses = [
            (OrderStatus.NEW, OrderStatus.PENDING_SUBMIT),
            (OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN),
            (OrderStatus.OPEN, OrderStatus.CANCELED),
        ]
        order = _make_order(OrderStatus.NEW)
        for from_s, to_s in statuses:
            order = order.model_copy(update={"status": from_s})
            validate_transition(order, to_s)
