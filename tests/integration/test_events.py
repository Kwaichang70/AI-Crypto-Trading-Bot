"""
tests/integration/test_events.py
---------------------------------
Integration tests for the in-process event bus (packages/common/events.py).

Tests cover:
- Subscribe and publish: handler receives correct event
- Async and sync handler support
- Priority ordering: CRITICAL handlers fire before LOW
- Error isolation: one failing handler doesn't block others
- Duplicate subscribe guard
- Bus close prevents further publish/subscribe
- Batch publish dispatches all events in order
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest

from common.events import (
    BarEvent,
    BaseEvent,
    EventBus,
    EventPriority,
    FillEvent,
    RiskEvent,
    SignalEvent,
)
from common.types import OrderSide, SignalDirection, TimeFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fill_event() -> FillEvent:
    """Create a minimal FillEvent for testing."""
    return FillEvent(
        order_id=uuid4(),
        run_id="test-run-001",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("0.5"),
        market_price=Decimal("50000"),
    )


def _make_signal_event() -> SignalEvent:
    """Create a minimal SignalEvent for testing."""
    return SignalEvent(
        strategy_id="test-strategy",
        symbol="BTC/USDT",
        direction=SignalDirection.BUY,
        target_position=Decimal("1000"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusPublishSubscribe:
    """Basic pub/sub functionality."""

    async def test_handler_receives_published_event(self) -> None:
        """Subscribed handler should receive the published event."""
        bus = EventBus(name="test")
        received: list[FillEvent] = []

        async def on_fill(event: FillEvent) -> None:
            received.append(event)

        bus.subscribe(FillEvent, on_fill)
        fill = _make_fill_event()
        count = await bus.publish(fill)

        assert count == 1
        assert len(received) == 1
        assert received[0].symbol == "BTC/USDT"
        assert received[0].event_id == fill.event_id

    async def test_sync_handler_receives_event(self) -> None:
        """Synchronous handlers should work via asyncio.to_thread."""
        bus = EventBus(name="test")
        received: list[FillEvent] = []

        def on_fill_sync(event: FillEvent) -> None:
            received.append(event)

        bus.subscribe(FillEvent, on_fill_sync)
        fill = _make_fill_event()
        count = await bus.publish(fill)

        assert count == 1
        assert len(received) == 1

    async def test_no_cross_event_delivery(self) -> None:
        """Handler subscribed to FillEvent should NOT receive SignalEvent."""
        bus = EventBus(name="test")
        received: list[BaseEvent] = []

        async def on_fill(event: FillEvent) -> None:
            received.append(event)

        bus.subscribe(FillEvent, on_fill)
        signal = _make_signal_event()
        count = await bus.publish(signal)

        assert count == 0
        assert len(received) == 0

    async def test_no_subscribers_returns_zero(self) -> None:
        """Publishing to an event type with no subscribers should return 0."""
        bus = EventBus(name="test")
        fill = _make_fill_event()
        count = await bus.publish(fill)
        assert count == 0


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusPriority:
    """Priority-based dispatch ordering."""

    async def test_critical_before_low(self) -> None:
        """CRITICAL priority handlers should fire before LOW priority handlers."""
        bus = EventBus(name="test")
        order: list[str] = []

        async def low_handler(event: FillEvent) -> None:
            order.append("low")

        async def critical_handler(event: FillEvent) -> None:
            order.append("critical")

        # Subscribe low first, then critical
        bus.subscribe(FillEvent, low_handler, priority=EventPriority.LOW)
        bus.subscribe(FillEvent, critical_handler, priority=EventPriority.CRITICAL)

        await bus.publish(_make_fill_event())

        assert order == ["critical", "low"]

    async def test_full_priority_ordering(self) -> None:
        """All four priority levels should execute in correct order."""
        bus = EventBus(name="test")
        order: list[str] = []

        for name, prio in [
            ("low", EventPriority.LOW),
            ("normal", EventPriority.NORMAL),
            ("critical", EventPriority.CRITICAL),
            ("high", EventPriority.HIGH),
        ]:
            async def handler(event: FillEvent, n: str = name) -> None:
                order.append(n)
            bus.subscribe(FillEvent, handler, priority=prio)

        await bus.publish(_make_fill_event())
        assert order == ["critical", "high", "normal", "low"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusErrorIsolation:
    """Handler error isolation."""

    async def test_failing_handler_does_not_block_others(self) -> None:
        """If one handler raises, other handlers should still execute."""
        bus = EventBus(name="test")
        results: list[str] = []

        async def failing_handler(event: FillEvent) -> None:
            raise RuntimeError("intentional test error")

        async def good_handler(event: FillEvent) -> None:
            results.append("ok")

        bus.subscribe(FillEvent, failing_handler, priority=EventPriority.HIGH)
        bus.subscribe(FillEvent, good_handler, priority=EventPriority.LOW)

        count = await bus.publish(_make_fill_event())

        # good_handler succeeded, failing_handler did not
        assert count == 1
        assert results == ["ok"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusDuplicateGuard:
    """Duplicate subscription prevention."""

    async def test_duplicate_subscribe_ignored(self) -> None:
        """Subscribing the same handler twice should not result in double calls."""
        bus = EventBus(name="test")
        call_count = 0

        async def handler(event: FillEvent) -> None:
            nonlocal call_count
            call_count += 1

        bus.subscribe(FillEvent, handler)
        bus.subscribe(FillEvent, handler)  # Duplicate — should be ignored

        await bus.publish(_make_fill_event())
        assert call_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusLifecycle:
    """Bus lifecycle: close, clear."""

    async def test_close_prevents_publish(self) -> None:
        """Publishing after close() should raise ValueError."""
        bus = EventBus(name="test")
        bus.close()

        with pytest.raises(ValueError, match="closed"):
            await bus.publish(_make_fill_event())

    async def test_close_prevents_subscribe(self) -> None:
        """Subscribing after close() should raise ValueError."""
        bus = EventBus(name="test")
        bus.close()

        async def handler(event: FillEvent) -> None:
            pass

        with pytest.raises(ValueError, match="closed"):
            bus.subscribe(FillEvent, handler)

    async def test_publish_batch_dispatches_all(self) -> None:
        """publish_batch should dispatch all events in order."""
        bus = EventBus(name="test")
        received: list[str] = []

        async def on_fill(event: FillEvent) -> None:
            received.append(str(event.event_id))

        bus.subscribe(FillEvent, on_fill)

        fills = [_make_fill_event() for _ in range(3)]
        total = await bus.publish_batch(fills)

        assert total == 3
        assert len(received) == 3
        # Verify order matches input
        for i, fill in enumerate(fills):
            assert received[i] == str(fill.event_id)

    async def test_unsubscribe_removes_handler(self) -> None:
        """Unsubscribing should remove the handler from dispatch."""
        bus = EventBus(name="test")
        call_count = 0

        async def handler(event: FillEvent) -> None:
            nonlocal call_count
            call_count += 1

        bus.subscribe(FillEvent, handler)
        await bus.publish(_make_fill_event())
        assert call_count == 1

        removed = bus.unsubscribe(FillEvent, handler)
        assert removed is True

        await bus.publish(_make_fill_event())
        assert call_count == 1  # Not called again
