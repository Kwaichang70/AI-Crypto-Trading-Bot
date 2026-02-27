"""
packages/trading/execution.py
------------------------------
Abstract ExecutionEngine base class and order state-machine helpers.

The state machine is the authoritative definition of valid order transitions.
Any attempt to make an illegal transition raises ``InvalidOrderTransitionError``.

Concrete subclasses:
  - PaperExecutionEngine  (packages/trading/engines/paper.py — to be implemented)
  - LiveExecutionEngine   (packages/trading/engines/live.py — to be implemented)
"""

from __future__ import annotations

import abc
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from common.types import OrderStatus
from trading.models import Fill, Order, Signal

__all__ = [
    "BaseExecutionEngine",
    "InvalidOrderTransitionError",
    "ORDER_STATE_MACHINE",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Order state machine — adjacency map
# ---------------------------------------------------------------------------
#
# Maps each OrderStatus to the set of statuses it is allowed to transition to.
# The ExecutionEngine MUST enforce these rules before updating any Order.
#
ORDER_STATE_MACHINE: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset({
        OrderStatus.PENDING_SUBMIT,
        OrderStatus.REJECTED,   # immediate rejection before submission
    }),
    OrderStatus.PENDING_SUBMIT: frozenset({
        OrderStatus.OPEN,
        OrderStatus.REJECTED,
        OrderStatus.CANCELED,
    }),
    OrderStatus.OPEN: frozenset({
        OrderStatus.PARTIAL,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    }),
    OrderStatus.PARTIAL: frozenset({
        OrderStatus.PARTIAL,   # additional partial fill
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    }),
    # Terminal states — no outgoing transitions
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


class InvalidOrderTransitionError(Exception):
    """
    Raised when a requested order state transition is not allowed
    by the state machine.

    Attributes
    ----------
    order_id:
        UUID of the order that triggered the error.
    from_status:
        Current (invalid source) status.
    to_status:
        Requested (invalid target) status.
    """

    def __init__(
        self,
        order_id: UUID,
        from_status: OrderStatus,
        to_status: OrderStatus,
    ) -> None:
        self.order_id = order_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Order {order_id}: invalid transition {from_status!r} -> {to_status!r}. "
            f"Allowed targets: {ORDER_STATE_MACHINE.get(from_status, frozenset())}"
        )


def validate_transition(order: Order, to_status: OrderStatus) -> None:
    """
    Assert that transitioning ``order`` to ``to_status`` is legal.

    Raises
    ------
    InvalidOrderTransitionError
        If the transition is not permitted by ``ORDER_STATE_MACHINE``.
    """
    allowed = ORDER_STATE_MACHINE.get(order.status, frozenset())
    if to_status not in allowed:
        raise InvalidOrderTransitionError(order.order_id, order.status, to_status)


class BaseExecutionEngine(abc.ABC):
    """
    Abstract execution engine.

    Responsibility
    --------------
    - Accept Signals from the StrategyEngine
    - Create Orders and drive them through the state machine
    - Emit Fill events consumed by PortfolioAccounting
    - Enforce idempotency via ``client_order_id``

    Thread / concurrency model
    --------------------------
    All public methods are async. Concrete implementations MUST NOT
    perform blocking I/O (use ``asyncio.to_thread`` for any blocking calls).

    Safety contract
    ---------------
    - LIVE mode requires that ``RiskManager.pre_trade_check`` returns
      ``RiskCheckResult(approved=True)`` before any order is submitted.
    - The engine records every order and fill regardless of outcome.
    """

    def __init__(self, run_id: str, config: dict[str, Any] | None = None) -> None:
        self._run_id = run_id
        self._config: dict[str, Any] = config or {}
        self._log = structlog.get_logger(__name__).bind(
            run_id=run_id,
            engine=self.__class__.__name__,
        )
        # In-memory order registry; persistence layer flushes to DB separately.
        self._orders: dict[UUID, Order] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange (live) or fill simulator (paper).

        Implementations MUST:
        1. Transition the order from NEW -> PENDING_SUBMIT before submission
        2. Return the updated Order with exchange_order_id set (if applicable)
        3. Raise on unrecoverable errors; transient errors should be retried
           with exponential backoff inside the implementation.

        Parameters
        ----------
        order:
            A fully validated Order with status=NEW.

        Returns
        -------
        Order:
            Updated order reflecting the submission outcome.
        """
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: UUID) -> Order:
        """
        Request cancellation of an open or partially-filled order.

        Parameters
        ----------
        order_id:
            Internal UUID of the order to cancel.

        Returns
        -------
        Order:
            Updated order with status CANCELED (or FILLED if a race occurred).
        """
        ...

    @abc.abstractmethod
    async def get_order(self, order_id: UUID) -> Order:
        """
        Fetch the current state of an order.

        For paper mode this reads from the in-memory registry.
        For live mode this calls the exchange API and reconciles state.

        Parameters
        ----------
        order_id:
            Internal UUID of the order to fetch.

        Returns
        -------
        Order:
            The most up-to-date Order record.
        """
        ...

    @abc.abstractmethod
    async def process_signal(self, signal: Signal) -> list[Order]:
        """
        Convert a trading Signal into zero or more submitted Orders.

        Implementations MUST:
        1. Determine whether a new order is needed (e.g. current position
           may already match the target)
        2. Run pre-trade risk check via the injected RiskManager
        3. Build and submit the Order(s)
        4. Return all Orders created (may be empty if signal is ignored)

        Parameters
        ----------
        signal:
            The Signal emitted by a strategy's ``on_bar`` call.

        Returns
        -------
        list[Order]:
            Orders created and submitted in response to the signal.
        """
        ...

    @abc.abstractmethod
    async def get_fills(self, order_id: UUID) -> list[Fill]:
        """
        Return all fills associated with an order.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        list[Fill]:
            Fills sorted by ``executed_at`` ascending.
        """
        ...

    # ------------------------------------------------------------------
    # State machine helpers (shared by all concrete engines)
    # ------------------------------------------------------------------

    def _transition(self, order: Order, to_status: OrderStatus) -> Order:
        """
        Validate and apply a state transition, returning a new Order copy.

        Uses Pydantic model_copy to preserve immutability.
        Logs every transition at DEBUG level for full auditability.

        Parameters
        ----------
        order:
            Current order object.
        to_status:
            Target status.

        Returns
        -------
        Order:
            New order object with updated status and ``updated_at``.

        Raises
        ------
        InvalidOrderTransitionError:
            If the transition is not permitted.
        """
        validate_transition(order, to_status)
        updated = order.model_copy(
            update={"status": to_status, "updated_at": datetime.now(tz=UTC)}
        )
        self._orders[updated.order_id] = updated
        self._log.debug(
            "order.transition",
            order_id=str(updated.order_id),
            from_status=order.status,
            to_status=to_status,
        )
        return updated

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """
        Called once when the trading run begins.
        Override to open exchange connections, warm up caches, etc.
        """
        self._log.info("execution_engine.started")

    async def on_stop(self) -> None:
        """
        Called on graceful shutdown.

        Implementations MUST:
        - Cancel all open orders or log them as needing manual review
        - Flush any pending fills to the persistence layer
        """
        self._log.info("execution_engine.stopping", open_orders=len(self._orders))

    def get_open_orders(self) -> list[Order]:
        """Return all orders that are not in a terminal state."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"run_id={self._run_id!r}, "
            f"orders={len(self._orders)})"
        )
