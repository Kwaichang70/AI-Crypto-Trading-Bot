"""
packages/common/events.py
--------------------------
Lightweight in-process event bus for fill routing and component decoupling.

This module provides:
1. A typed event hierarchy (frozen Pydantic models) for all trading domain events
2. An async-first EventBus with sync handler support
3. Priority-based subscriber ordering
4. Error isolation per handler (one failing handler does not block others)

Design principles
-----------------
- **Zero external dependencies**: pure stdlib + Pydantic (already in the project)
- **In-process only**: no Redis/RabbitMQ; MVP-scope infrastructure
- **Asyncio-safe**: all dispatch is via async; sync handlers are wrapped
  in asyncio.to_thread to prevent event-loop blocking.
- **Immutable events**: all event models are frozen dataclasses (Pydantic frozen=True)
- **Type-safe subscriptions**: subscribe by event class, receive only that type
- **Error isolation**: handler exceptions are logged, never propagated to publishers

Thread safety
-------------
This bus is designed for single-threaded asyncio use. No threading locks are
needed for the MVP. If multi-threaded access is required in the future, wrap
the subscriber registry with asyncio.Lock.

Usage
-----
    from common.events import EventBus, FillEvent

    bus = EventBus()

    async def on_fill(event: FillEvent):
        print(f"Fill received: {event.fill_id}")

    bus.subscribe(FillEvent, on_fill)
    await bus.publish(FillEvent(...))
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum, auto
from typing import Any, Callable, Coroutine, Literal, TypeAlias, TypeVar, cast
from uuid import UUID, uuid4

import structlog

from pydantic import BaseModel, Field

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection, TimeFrame

__all__ = [
    # Event base
    "BaseEvent",
    "EventPriority",
    # Concrete events
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "RiskEvent",
    # Bus
    "EventBus",
]

logger = structlog.get_logger(__name__)

# Type variable for event class registration
E = TypeVar("E", bound="BaseEvent")

# Handler type: sync or async callable that accepts exactly one BaseEvent argument
EventHandler: TypeAlias = (
    Callable[["BaseEvent"], None]
    | Callable[["BaseEvent"], Coroutine[Any, Any, None]]
)


# ---------------------------------------------------------------------------
# Event priority
# ---------------------------------------------------------------------------

class EventPriority(StrEnum):
    """
    Subscriber execution priority within a single event type.

    Handlers with CRITICAL priority run first, then HIGH, NORMAL, LOW.
    Within the same priority level, execution order is insertion order
    (first subscribed, first called).
    """

    CRITICAL = auto()  # Risk checks, kill-switch evaluation
    HIGH = auto()      # Portfolio accounting, position updates
    NORMAL = auto()    # Strategy engine, general consumers
    LOW = auto()       # Logging, metrics, analytics


# Map priority to sort key (lower = runs first)
_PRIORITY_ORDER: dict[EventPriority, int] = {
    EventPriority.CRITICAL: 0,
    EventPriority.HIGH: 1,
    EventPriority.NORMAL: 2,
    EventPriority.LOW: 3,
}


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class BaseEvent(BaseModel):
    """
    Base class for all domain events.

    Every event has:
    - A unique event_id (UUID4, auto-generated)
    - A timestamp (UTC, auto-generated)
    - An event_type string (derived from the class name)

    All events are frozen (immutable) to prevent downstream mutation.
    """

    # frozen=True and extra="forbid" are inherited by all subclasses.
    # Do NOT override model_config in subclasses without preserving both settings.
    model_config = {"frozen": True, "extra": "forbid"}

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp when the event was created",
    )

    @property
    def event_type(self) -> str:
        """Return the class name as the event type string."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Concrete event types
# ---------------------------------------------------------------------------

class BarEvent(BaseEvent):
    """
    Emitted when a new OHLCV bar is received and ready for processing.

    Published by the StrategyEngine at the start of each bar processing
    cycle, before strategies are invoked.
    """

    symbol: str = Field(description="Trading pair, e.g. 'BTC/USDT'")
    timeframe: TimeFrame = Field(description="Candle timeframe")
    bar_timestamp: datetime = Field(description="Bar open time in UTC")
    open: Decimal = Field(description="Open price")
    high: Decimal = Field(description="High price")
    low: Decimal = Field(description="Low price")
    close: Decimal = Field(description="Close price")
    volume: Decimal = Field(description="Volume in base asset")
    bar_index: int = Field(
        default=0,
        description="Sequential bar number within the run",
    )
    run_id: str = Field(
        default="",
        description="Trading run identifier",
    )


class SignalEvent(BaseEvent):
    """
    Emitted when a strategy produces a trading signal.

    Published by the StrategyEngine after each strategy's on_bar call.
    Contains the full signal data for downstream consumption by the
    execution engine, logging, and analytics.
    """

    strategy_id: str = Field(description="ID of the strategy that generated this signal")
    symbol: str = Field(description="Trading pair the signal applies to")
    direction: SignalDirection = Field(description="BUY / SELL / HOLD")
    target_position: Decimal = Field(
        description="Target notional size in quote currency (0 = flat)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strategy confidence in this signal [0.0, 1.0]",
    )
    run_id: str = Field(
        default="",
        description="Trading run identifier",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific context (indicator values, etc.)",
    )


class OrderEvent(BaseEvent):
    """
    Emitted when an order transitions to a new state in the state machine.

    Published by the ExecutionEngine on every state transition:
    NEW -> PENDING_SUBMIT -> OPEN -> PARTIAL -> FILLED/CANCELED/REJECTED

    This event captures the order snapshot at the moment of transition.
    """

    order_id: UUID = Field(description="Internal order UUID")
    client_order_id: str = Field(description="Idempotency key")
    run_id: str = Field(description="Run that generated this order")
    symbol: str = Field(description="Trading pair")
    side: OrderSide = Field(description="BUY or SELL")
    order_type: OrderType = Field(description="MARKET or LIMIT")
    quantity: Decimal = Field(description="Order size in base asset")
    price: Decimal | None = Field(
        default=None,
        description="Limit price (None for MARKET orders)",
    )
    status: OrderStatus = Field(description="Current state after transition")
    previous_status: OrderStatus | None = Field(
        default=None,
        description="State before this transition",
    )
    filled_quantity: Decimal = Field(
        default=Decimal("0"),
        description="Quantity filled so far",
    )
    average_fill_price: Decimal | None = Field(
        default=None,
        description="Volume-weighted average fill price",
    )
    exchange_order_id: str | None = Field(
        default=None,
        description="Exchange-assigned order ID",
    )


class FillEvent(BaseEvent):
    """
    Emitted when an order receives a fill (partial or complete).

    This is the primary event for fill routing. Published by the
    ExecutionEngine, consumed by PortfolioAccounting and RiskManager.

    Contains all data needed by downstream consumers to update
    positions, equity, PnL, and risk state.
    """

    fill_id: UUID = Field(
        default_factory=uuid4,
        description="Unique fill identifier",
    )
    order_id: UUID = Field(description="Parent order UUID")
    run_id: str = Field(description="Trading run identifier")
    symbol: str = Field(description="Trading pair")
    side: OrderSide = Field(description="BUY or SELL")
    quantity: Decimal = Field(description="Filled quantity in base asset")
    price: Decimal = Field(description="Execution price per unit")
    fee: Decimal = Field(
        default=Decimal("0"),
        description="Fee paid in quote asset",
    )
    fee_currency: str = Field(
        default="USDT",
        description="Currency in which fee was paid",
    )
    is_maker: bool = Field(
        default=False,
        description="True if this fill was a maker (resting limit order)",
    )
    strategy_id: str = Field(
        default="",
        description="Strategy that originated the signal leading to this fill",
    )
    market_price: Decimal = Field(
        description="Current market price at time of fill (for PnL calculations). "
                    "Must be > 0. Used by PortfolioAccounting for unrealized PnL.",
    )


class RiskEvent(BaseEvent):
    """
    Emitted when a risk condition is triggered or cleared.

    Published by the RiskManager when:
    - Kill switch is activated or reset
    - Cooldown period starts or expires
    - Daily loss limit is breached
    - Maximum drawdown is breached
    - A pre-trade check rejects an order
    """

    risk_type: str = Field(
        description=(
            "Type of risk event: 'kill_switch_activated', 'kill_switch_reset', "
            "'cooldown_activated', 'cooldown_expired', 'daily_loss_breached', "
            "'drawdown_breached', 'order_rejected'"
        ),
    )
    run_id: str = Field(description="Trading run identifier")
    severity: Literal["info", "warning", "critical"] = Field(
        default="warning",
        description="Event severity level.",
    )
    message: str = Field(
        default="",
        description="Human-readable description of the risk event",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data associated with the risk event",
    )


# ---------------------------------------------------------------------------
# Subscriber record
# ---------------------------------------------------------------------------

class _Subscription:
    """Internal record of a single event subscription."""

    __slots__ = ("handler", "priority", "sort_key", "is_async", "name")

    def __init__(
        self,
        handler: EventHandler,
        priority: EventPriority,
        insertion_order: int,
    ) -> None:
        self.handler = handler
        self.priority = priority
        self.sort_key = (_PRIORITY_ORDER[priority], insertion_order)
        self.is_async = inspect.iscoroutinefunction(handler)
        self.name = getattr(handler, "__qualname__", repr(handler))

    def __repr__(self) -> str:
        return (
            f"_Subscription(handler={self.name!r}, "
            f"priority={self.priority.value!r}, "
            f"async={self.is_async})"
        )


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Lightweight in-process event bus with typed pub/sub.

    Supports both sync and async handlers. Events are dispatched in
    priority order (CRITICAL first, then HIGH, NORMAL, LOW). Within
    the same priority level, handlers are called in subscription order.

    Error isolation: if a handler raises an exception, it is logged
    and the remaining handlers continue to execute.

    Parameters
    ----------
    name :
        Optional name for this bus instance (used in log messages).

    Examples
    --------
    >>> bus = EventBus(name="trading")
    >>> async def on_fill(event: FillEvent):
    ...     print(f"Fill: {event.symbol} {event.side}")
    >>> bus.subscribe(FillEvent, on_fill, priority=EventPriority.HIGH)
    >>> await bus.publish(FillEvent(
    ...     order_id=some_uuid,
    ...     run_id="run-001",
    ...     symbol="BTC/USDT",
    ...     side=OrderSide.BUY,
    ...     quantity=Decimal("0.1"),
    ...     price=Decimal("50000"),
    ... ))
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        # event_type (class) -> list of subscriptions
        self._subscribers: dict[type[BaseEvent], list[_Subscription]] = {}
        # Monotonically increasing counter for insertion-order tie-breaking
        self._insertion_counter: int = 0
        # Flag to prevent publish during bus teardown
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Bus instance name."""
        return self._name

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriptions across all event types."""
        return sum(len(subs) for subs in self._subscribers.values())

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: type[E],
        handler: EventHandler,
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """
        Register a handler for a specific event type.

        The handler will be called with the event instance as its sole
        argument whenever an event of the specified type (or a subclass)
        is published.

        Parameters
        ----------
        event_type :
            The event class to subscribe to.
        handler :
            Callable that accepts a single event argument. May be sync
            or async.
        priority :
            Execution priority. CRITICAL handlers run before HIGH,
            which run before NORMAL, which run before LOW.

        Raises
        ------
        TypeError
            If event_type is not a subclass of BaseEvent.
        ValueError
            If the bus is closed.
        """
        if self._closed:
            raise ValueError(
                f"EventBus '{self._name}' is closed; cannot subscribe"
            )

        if not (isinstance(event_type, type) and issubclass(event_type, BaseEvent)):
            raise TypeError(
                f"event_type must be a subclass of BaseEvent, "
                f"got {event_type!r}"
            )

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Duplicate-subscription guard: same handler object must not be registered twice
        existing = self._subscribers[event_type]
        if any(s.handler is handler for s in existing):
            logger.warning(
                "eventbus.duplicate_subscribe",
                bus=self._name,
                event_type=event_type.__name__,
                handler=getattr(handler, "__qualname__", repr(handler)),
            )
            return

        sub = _Subscription(
            handler=handler,
            priority=priority,
            insertion_order=self._insertion_counter,
        )
        self._insertion_counter += 1

        self._subscribers[event_type].append(sub)

        # Re-sort by priority (stable sort preserves insertion order within
        # the same priority level)
        self._subscribers[event_type].sort(key=lambda s: s.sort_key)

        logger.debug(
            "eventbus.subscribed",
            bus=self._name,
            event_type=event_type.__name__,
            handler=sub.name,
            priority=priority.value,
        )

    # ------------------------------------------------------------------
    # Unsubscribe
    # ------------------------------------------------------------------

    def unsubscribe(
        self,
        event_type: type[E],
        handler: EventHandler,
    ) -> bool:
        """
        Remove a handler from an event type.

        Parameters
        ----------
        event_type :
            The event class the handler was subscribed to.
        handler :
            The handler function to remove.

        Returns
        -------
        bool
            True if the handler was found and removed; False otherwise.
        """
        subs = self._subscribers.get(event_type, [])
        for i, sub in enumerate(subs):
            if sub.handler is handler:
                subs.pop(i)
                logger.debug(
                    "eventbus.unsubscribed",
                    bus=self._name,
                    event_type=event_type.__name__,
                    handler=sub.name,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, event: BaseEvent) -> int:
        """
        Publish an event to all subscribers of its type.

        Handlers are called in priority order. If a handler raises an
        exception, it is logged and remaining handlers continue.

        Parameters
        ----------
        event :
            The event instance to publish.

        Returns
        -------
        int
            Number of handlers that completed without raising an exception.

        Raises
        ------
        ValueError
            If the bus is closed.
        TypeError
            If event is not an instance of BaseEvent.
        """
        if self._closed:
            raise ValueError(
                f"EventBus '{self._name}' is closed; cannot publish"
            )

        if not isinstance(event, BaseEvent):
            raise TypeError(
                f"event must be an instance of BaseEvent, "
                f"got {type(event).__name__}"
            )

        event_type = type(event)
        # Snapshot the subscriber list before iteration to guard against
        # mutation if a handler calls unsubscribe() during dispatch (CR-EB-006)
        subs = list(self._subscribers.get(event_type, []))

        if not subs:
            logger.debug(
                "eventbus.no_subscribers",
                bus=self._name,
                event_type=event_type.__name__,
                event_id=str(event.event_id),
            )
            return 0

        handlers_succeeded = 0
        handlers_failed = 0
        for sub in subs:
            try:
                if sub.is_async:
                    await cast(Coroutine[Any, Any, None], sub.handler(event))
                else:
                    # Wrap sync handlers in asyncio.to_thread to avoid blocking
                    # the event loop (CR-EB-001). The event is a frozen Pydantic
                    # model so thread-safety of the event object is not a concern.
                    await asyncio.to_thread(sub.handler, event)
                handlers_succeeded += 1
            except Exception:
                handlers_failed += 1
                logger.exception(
                    "eventbus.handler_error",
                    bus=self._name,
                    event_type=event_type.__name__,
                    event_id=str(event.event_id),
                    handler=sub.name,
                    priority=sub.priority.value,
                )

        logger.debug(
            "eventbus.published",
            bus=self._name,
            event_type=event_type.__name__,
            event_id=str(event.event_id),
            handlers_succeeded=handlers_succeeded,
            handlers_failed=handlers_failed,
        )

        return handlers_succeeded

    # ------------------------------------------------------------------
    # Batch publish
    # ------------------------------------------------------------------

    async def publish_batch(self, events: list[BaseEvent]) -> int:
        """
        Publish multiple events sequentially.

        Events are published in list order. This is a convenience method
        for publishing a batch of events from a single bar processing cycle.

        Parameters
        ----------
        events :
            List of events to publish in order.

        Returns
        -------
        int
            Total number of handler invocations across all events.
        """
        total = 0
        for event in events:
            total += await self.publish(event)
        return total

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_subscribers(
        self,
        event_type: type[BaseEvent],
    ) -> list[tuple[str, EventPriority]]:
        """
        Return subscriber info for a given event type.

        Parameters
        ----------
        event_type :
            The event class to query.

        Returns
        -------
        list[tuple[str, EventPriority]]
            List of (handler_name, priority) tuples in execution order.
        """
        subs = self._subscribers.get(event_type, [])
        return [(sub.name, sub.priority) for sub in subs]

    def get_registered_event_types(self) -> list[str]:
        """
        Return all event type names that have at least one subscriber.

        Returns
        -------
        list[str]
            Sorted list of event type class names.
        """
        return sorted(
            et.__name__
            for et, subs in self._subscribers.items()
            if subs
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all subscribers from the bus.

        Does not close the bus -- new subscriptions can still be made.
        """
        count = self.subscriber_count
        self._subscribers.clear()
        logger.info(
            "eventbus.cleared",
            bus=self._name,
            removed_subscriptions=count,
        )

    def close(self) -> None:
        """
        Close the bus and remove all subscribers.

        After closing, publish() and subscribe() will raise ValueError.
        This is intended for graceful shutdown.
        """
        self.clear()
        self._closed = True
        logger.info(
            "eventbus.closed",
            bus=self._name,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EventBus(name={self._name!r}, "
            f"event_types={len(self._subscribers)}, "
            f"subscribers={self.subscriber_count}, "
            f"closed={self._closed})"
        )
