"""Unit tests for Event system."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.core.events import (
    EventType,
    Event,
    EventMetrics,
    EventHandler,
    EventBus,
    get_event_bus,
    reset_event_bus,
    emit,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_account_events(self):
        """Should have account event types."""
        assert EventType.ACCOUNT_CONNECTED.value == "account.connected"
        assert EventType.ACCOUNT_DISCONNECTED.value == "account.disconnected"
        assert EventType.ACCOUNT_ERROR.value == "account.error"

    def test_position_events(self):
        """Should have position event types."""
        assert EventType.POSITION_OPENED.value == "position.opened"
        assert EventType.POSITION_CLOSED.value == "position.closed"
        assert EventType.POSITION_UPDATED.value == "position.updated"

    def test_order_events(self):
        """Should have order event types."""
        assert EventType.ORDER_CREATED.value == "order.created"
        assert EventType.ORDER_FILLED.value == "order.filled"
        assert EventType.ORDER_CANCELLED.value == "order.cancelled"
        assert EventType.ORDER_REJECTED.value == "order.rejected"
        assert EventType.ORDER_ERROR.value == "order.error"

    def test_delta_events(self):
        """Should have delta event types."""
        assert EventType.DELTA_UPDATED.value == "delta.updated"
        assert EventType.DELTA_THRESHOLD_BREACHED.value == "delta.threshold_breached"
        assert EventType.DELTA_NEUTRAL_RESTORED.value == "delta.neutral_restored"

    def test_rebalance_events(self):
        """Should have rebalance event types."""
        assert EventType.REBALANCE_STARTED.value == "rebalance.started"
        assert EventType.REBALANCE_COMPLETED.value == "rebalance.completed"
        assert EventType.REBALANCE_FAILED.value == "rebalance.failed"

    def test_risk_events(self):
        """Should have risk event types."""
        assert EventType.RISK_ALERT.value == "risk.alert"
        assert EventType.RISK_HALT.value == "risk.halt"
        assert EventType.RISK_RESUME.value == "risk.resume"

    def test_system_events(self):
        """Should have system event types."""
        assert EventType.BOT_STARTED.value == "bot.started"
        assert EventType.BOT_STOPPED.value == "bot.stopped"
        assert EventType.BOT_ERROR.value == "bot.error"

    def test_health_events(self):
        """Should have health event types."""
        assert EventType.HEALTH_CHECK.value == "health.check"
        assert EventType.HEALTH_DEGRADED.value == "health.degraded"
        assert EventType.HEALTH_RECOVERED.value == "health.recovered"


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Should create event with defaults."""
        event = Event(type=EventType.BOT_STARTED)

        assert event.type == EventType.BOT_STARTED
        assert event.data == {}
        assert event.source == "system"
        assert event.timestamp > 0
        assert event.correlation_id is None

    def test_create_event_with_data(self):
        """Should create event with data."""
        data = {"account_id": "acc1", "balance": 1000}
        event = Event(
            type=EventType.ACCOUNT_CONNECTED,
            data=data,
            source="account_manager",
        )

        assert event.data == data
        assert event.source == "account_manager"

    def test_create_event_with_correlation_id(self):
        """Should create event with correlation ID."""
        event = Event(
            type=EventType.ORDER_CREATED,
            correlation_id="trade-123",
        )

        assert event.correlation_id == "trade-123"

    def test_to_dict(self):
        """Should convert to dictionary."""
        event = Event(
            type=EventType.POSITION_OPENED,
            data={"market": "BTC-USD"},
            source="position_manager",
            correlation_id="pos-456",
        )

        d = event.to_dict()

        assert d["type"] == "position.opened"
        assert d["data"] == {"market": "BTC-USD"}
        assert d["source"] == "position_manager"
        assert d["correlation_id"] == "pos-456"
        assert "timestamp" in d


class TestEventMetrics:
    """Tests for EventMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = EventMetrics()

        assert metrics.events_emitted == 0
        assert metrics.events_handled == 0
        assert metrics.handlers_called == 0
        assert metrics.errors == 0
        assert metrics.avg_handle_time_ms == 0.0

    def test_record_emit(self):
        """Should record event emission."""
        metrics = EventMetrics()

        metrics.record_emit()
        metrics.record_emit()

        assert metrics.events_emitted == 2

    def test_record_handle(self):
        """Should record event handling with timing."""
        metrics = EventMetrics()

        metrics.record_handle(10.0)
        metrics.record_handle(20.0)

        assert metrics.events_handled == 2
        assert metrics.avg_handle_time_ms == 15.0

    def test_record_handler_call(self):
        """Should record handler invocations."""
        metrics = EventMetrics()

        metrics.record_handler_call()
        metrics.record_handler_call()
        metrics.record_handler_call()

        assert metrics.handlers_called == 3

    def test_record_error(self):
        """Should record errors."""
        metrics = EventMetrics()

        metrics.record_error()

        assert metrics.errors == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = EventMetrics()
        metrics.record_emit()
        metrics.record_handle(5.0)

        d = metrics.to_dict()

        assert d["events_emitted"] == 1
        assert d["events_handled"] == 1
        assert d["avg_handle_time_ms"] == 5.0


class TestEventHandler:
    """Tests for EventHandler dataclass."""

    def test_create_handler(self):
        """Should create handler."""
        callback = MagicMock()
        handler = EventHandler(
            callback=callback,
            event_types={EventType.BOT_STARTED},
        )

        assert handler.callback == callback
        assert EventType.BOT_STARTED in handler.event_types
        assert handler.priority == 0
        assert handler.once is False

    def test_handler_with_priority(self):
        """Should create handler with priority."""
        handler = EventHandler(
            callback=MagicMock(),
            event_types={EventType.ORDER_FILLED},
            priority=10,
        )

        assert handler.priority == 10

    def test_handler_once(self):
        """Should create one-time handler."""
        handler = EventHandler(
            callback=MagicMock(),
            event_types={EventType.BOT_STARTED},
            once=True,
        )

        assert handler.once is True

    def test_matches_event_type(self):
        """Should match event type."""
        handler = EventHandler(
            callback=MagicMock(),
            event_types={EventType.ORDER_FILLED, EventType.ORDER_CANCELLED},
        )

        event1 = Event(type=EventType.ORDER_FILLED)
        event2 = Event(type=EventType.ORDER_CANCELLED)
        event3 = Event(type=EventType.BOT_STARTED)

        assert handler.matches(event1) is True
        assert handler.matches(event2) is True
        assert handler.matches(event3) is False

    def test_matches_with_filter(self):
        """Should apply filter function."""
        handler = EventHandler(
            callback=MagicMock(),
            event_types={EventType.ORDER_FILLED},
            filter_fn=lambda e: e.data.get("market") == "BTC-USD",
        )

        event1 = Event(type=EventType.ORDER_FILLED, data={"market": "BTC-USD"})
        event2 = Event(type=EventType.ORDER_FILLED, data={"market": "ETH-USD"})

        assert handler.matches(event1) is True
        assert handler.matches(event2) is False

    def test_matches_wrong_type_with_filter(self):
        """Should not match wrong type even with matching filter."""
        handler = EventHandler(
            callback=MagicMock(),
            event_types={EventType.ORDER_FILLED},
            filter_fn=lambda e: True,  # Always true filter
        )

        event = Event(type=EventType.BOT_STARTED)

        assert handler.matches(event) is False


class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def bus(self):
        """Create fresh event bus."""
        return EventBus()

    def test_create_bus(self, bus):
        """Should create event bus."""
        assert len(bus._handlers) == 0
        assert len(bus._history) == 0
        assert bus.max_history == 100

    def test_subscribe_single_event(self, bus):
        """Should subscribe to single event type."""
        callback = MagicMock()

        handler = bus.subscribe(EventType.BOT_STARTED, callback)

        assert handler in bus._handlers
        assert EventType.BOT_STARTED in handler.event_types

    def test_subscribe_multiple_events(self, bus):
        """Should subscribe to multiple event types."""
        callback = MagicMock()

        handler = bus.subscribe(
            [EventType.ORDER_FILLED, EventType.ORDER_CANCELLED],
            callback,
        )

        assert EventType.ORDER_FILLED in handler.event_types
        assert EventType.ORDER_CANCELLED in handler.event_types

    def test_subscribe_with_priority(self, bus):
        """Should subscribe with priority."""
        callback = MagicMock()

        handler = bus.subscribe(EventType.BOT_STARTED, callback, priority=10)

        assert handler.priority == 10

    def test_subscribe_once(self, bus):
        """Should subscribe for one-time handling."""
        callback = MagicMock()

        handler = bus.subscribe(EventType.BOT_STARTED, callback, once=True)

        assert handler.once is True

    def test_subscribe_with_filter(self, bus):
        """Should subscribe with filter function."""
        callback = MagicMock()
        filter_fn = lambda e: e.data.get("important")

        handler = bus.subscribe(EventType.BOT_STARTED, callback, filter_fn=filter_fn)

        assert handler.filter_fn == filter_fn

    def test_unsubscribe(self, bus):
        """Should unsubscribe handler."""
        callback = MagicMock()
        handler = bus.subscribe(EventType.BOT_STARTED, callback)

        result = bus.unsubscribe(handler)

        assert result is True
        assert handler not in bus._handlers

    def test_unsubscribe_not_found(self, bus):
        """Should return False for unknown handler."""
        handler = EventHandler(callback=MagicMock(), event_types={EventType.BOT_STARTED})

        result = bus.unsubscribe(handler)

        assert result is False

    def test_unsubscribe_all(self, bus):
        """Should unsubscribe all handlers."""
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.ORDER_FILLED, MagicMock())

        count = bus.unsubscribe_all()

        assert count == 2
        assert len(bus._handlers) == 0

    def test_unsubscribe_all_by_type(self, bus):
        """Should unsubscribe all handlers for specific type."""
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.ORDER_FILLED, MagicMock())

        count = bus.unsubscribe_all(EventType.BOT_STARTED)

        assert count == 2
        assert len(bus._handlers) == 1

    @pytest.mark.asyncio
    async def test_emit_event(self, bus):
        """Should emit event to handlers."""
        callback = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await bus.emit(EventType.BOT_STARTED, {"version": "1.0"})

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.type == EventType.BOT_STARTED
        assert event.data == {"version": "1.0"}

    @pytest.mark.asyncio
    async def test_emit_sync_handler(self, bus):
        """Should call sync handlers."""
        callback = MagicMock()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await bus.emit(EventType.BOT_STARTED)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_with_source(self, bus):
        """Should emit with source."""
        callback = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await bus.emit(EventType.BOT_STARTED, source="main")

        event = callback.call_args[0][0]
        assert event.source == "main"

    @pytest.mark.asyncio
    async def test_emit_with_correlation_id(self, bus):
        """Should emit with correlation ID."""
        callback = AsyncMock()
        bus.subscribe(EventType.ORDER_CREATED, callback)

        await bus.emit(
            EventType.ORDER_CREATED,
            correlation_id="trade-abc",
        )

        event = callback.call_args[0][0]
        assert event.correlation_id == "trade-abc"

    @pytest.mark.asyncio
    async def test_emit_event_object(self, bus):
        """Should emit pre-built event."""
        callback = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback)

        event = Event(type=EventType.BOT_STARTED, data={"test": True})
        await bus.emit_event(event)

        callback.assert_called_once()
        received = callback.call_args[0][0]
        assert received.data == {"test": True}

    @pytest.mark.asyncio
    async def test_emit_multiple_handlers(self, bus):
        """Should call multiple handlers."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback1)
        bus.subscribe(EventType.BOT_STARTED, callback2)

        await bus.emit(EventType.BOT_STARTED)

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_priority_ordering(self, bus):
        """Should call handlers in priority order."""
        order = []

        async def low_priority(e):
            order.append("low")

        async def high_priority(e):
            order.append("high")

        bus.subscribe(EventType.BOT_STARTED, low_priority, priority=0)
        bus.subscribe(EventType.BOT_STARTED, high_priority, priority=10)

        await bus.emit(EventType.BOT_STARTED)

        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_emit_once_handler_removed(self, bus):
        """Should remove one-time handlers after call."""
        callback = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback, once=True)

        await bus.emit(EventType.BOT_STARTED)
        await bus.emit(EventType.BOT_STARTED)

        assert callback.call_count == 1
        assert len(bus._handlers) == 0

    @pytest.mark.asyncio
    async def test_emit_filter_applied(self, bus):
        """Should apply filter function."""
        callback = AsyncMock()
        bus.subscribe(
            EventType.ORDER_FILLED,
            callback,
            filter_fn=lambda e: e.data.get("market") == "BTC-USD",
        )

        await bus.emit(EventType.ORDER_FILLED, {"market": "ETH-USD"})
        await bus.emit(EventType.ORDER_FILLED, {"market": "BTC-USD"})

        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_emit_handler_error_caught(self, bus):
        """Should catch handler errors."""
        async def failing_handler(e):
            raise ValueError("Handler error")

        bus.subscribe(EventType.BOT_STARTED, failing_handler)

        # Should not raise
        await bus.emit(EventType.BOT_STARTED)

        assert bus._metrics.errors == 1

    @pytest.mark.asyncio
    async def test_emit_updates_history(self, bus):
        """Should add events to history."""
        await bus.emit(EventType.BOT_STARTED)
        await bus.emit(EventType.ORDER_FILLED)

        assert len(bus._history) == 2
        assert bus._history[0].type == EventType.BOT_STARTED
        assert bus._history[1].type == EventType.ORDER_FILLED

    @pytest.mark.asyncio
    async def test_emit_history_limit(self, bus):
        """Should limit history size."""
        bus.max_history = 5

        for _ in range(10):
            await bus.emit(EventType.BOT_STARTED)

        assert len(bus._history) == 5

    @pytest.mark.asyncio
    async def test_emit_updates_metrics(self, bus):
        """Should update metrics."""
        callback = AsyncMock()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await bus.emit(EventType.BOT_STARTED)

        metrics = bus.get_metrics()
        assert metrics.events_emitted == 1
        assert metrics.events_handled == 1
        assert metrics.handlers_called == 1

    def test_on_decorator(self, bus):
        """Should work as decorator."""
        @bus.on(EventType.BOT_STARTED)
        async def handler(event):
            pass

        assert len(bus._handlers) == 1

    def test_on_decorator_with_priority(self, bus):
        """Should apply priority in decorator."""
        @bus.on(EventType.BOT_STARTED, priority=5)
        async def handler(event):
            pass

        assert bus._handlers[0].priority == 5

    def test_once_decorator(self, bus):
        """Should create one-time handler via decorator."""
        @bus.once(EventType.BOT_STARTED)
        async def handler(event):
            pass

        assert bus._handlers[0].once is True

    def test_get_handlers(self, bus):
        """Should get all handlers."""
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.ORDER_FILLED, MagicMock())

        handlers = bus.get_handlers()

        assert len(handlers) == 2

    def test_get_handlers_by_type(self, bus):
        """Should filter handlers by type."""
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.ORDER_FILLED, MagicMock())

        handlers = bus.get_handlers(EventType.BOT_STARTED)

        assert len(handlers) == 2

    @pytest.mark.asyncio
    async def test_get_history(self, bus):
        """Should get event history."""
        await bus.emit(EventType.BOT_STARTED)
        await bus.emit(EventType.ORDER_FILLED)
        await bus.emit(EventType.BOT_STOPPED)

        history = bus.get_history(limit=2)

        assert len(history) == 2
        # Most recent events
        assert history[-1].type == EventType.BOT_STOPPED

    @pytest.mark.asyncio
    async def test_get_history_by_type(self, bus):
        """Should filter history by type."""
        await bus.emit(EventType.BOT_STARTED)
        await bus.emit(EventType.ORDER_FILLED)
        await bus.emit(EventType.ORDER_FILLED)
        await bus.emit(EventType.BOT_STOPPED)

        history = bus.get_history(event_type=EventType.ORDER_FILLED)

        assert len(history) == 2
        assert all(e.type == EventType.ORDER_FILLED for e in history)

    def test_get_status(self, bus):
        """Should get bus status."""
        bus.subscribe(EventType.BOT_STARTED, MagicMock())
        bus.subscribe(EventType.ORDER_FILLED, MagicMock())

        status = bus.get_status()

        assert status["total_handlers"] == 2
        assert "handler_counts" in status
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_clear_history(self, bus):
        """Should clear history."""
        await bus.emit(EventType.BOT_STARTED)
        await bus.emit(EventType.ORDER_FILLED)

        bus.clear_history()

        assert len(bus._history) == 0

    def test_reset_metrics(self, bus):
        """Should reset metrics."""
        bus._metrics.events_emitted = 100

        bus.reset_metrics()

        assert bus._metrics.events_emitted == 0


class TestGlobalEventBus:
    """Tests for global event bus functions."""

    def setup_method(self):
        """Reset global bus before each test."""
        reset_event_bus()

    def test_get_event_bus_creates_singleton(self):
        """Should create singleton bus."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def test_reset_event_bus(self):
        """Should reset global bus."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()

        assert bus1 is not bus2

    @pytest.mark.asyncio
    async def test_emit_global(self):
        """Should emit on global bus."""
        callback = AsyncMock()
        bus = get_event_bus()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await emit(EventType.BOT_STARTED, {"test": True})

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_global_with_source(self):
        """Should emit with source on global bus."""
        callback = AsyncMock()
        bus = get_event_bus()
        bus.subscribe(EventType.BOT_STARTED, callback)

        await emit(EventType.BOT_STARTED, source="test")

        event = callback.call_args[0][0]
        assert event.source == "test"


class TestEventIntegration:
    """Integration tests for event system."""

    @pytest.fixture
    def bus(self):
        """Create fresh event bus."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_trade_flow_events(self, bus):
        """Should handle complete trade flow."""
        events_received = []

        async def trade_logger(event):
            events_received.append(event.type)

        bus.subscribe(
            [
                EventType.ORDER_CREATED,
                EventType.ORDER_FILLED,
                EventType.POSITION_OPENED,
            ],
            trade_logger,
        )

        # Simulate trade flow
        await bus.emit(EventType.ORDER_CREATED, {"order_id": "123"})
        await bus.emit(EventType.ORDER_FILLED, {"order_id": "123"})
        await bus.emit(EventType.POSITION_OPENED, {"position_id": "456"})

        assert events_received == [
            EventType.ORDER_CREATED,
            EventType.ORDER_FILLED,
            EventType.POSITION_OPENED,
        ]

    @pytest.mark.asyncio
    async def test_rebalance_flow(self, bus):
        """Should handle rebalance events."""
        events_received = []

        async def rebalance_handler(event):
            events_received.append((event.type, event.data.get("delta_pct")))

        bus.subscribe(
            [
                EventType.DELTA_THRESHOLD_BREACHED,
                EventType.REBALANCE_STARTED,
                EventType.REBALANCE_COMPLETED,
                EventType.DELTA_NEUTRAL_RESTORED,
            ],
            rebalance_handler,
        )

        await bus.emit(EventType.DELTA_THRESHOLD_BREACHED, {"delta_pct": 7.5})
        await bus.emit(EventType.REBALANCE_STARTED, {"delta_pct": 7.5})
        await bus.emit(EventType.REBALANCE_COMPLETED, {"delta_pct": 0.5})
        await bus.emit(EventType.DELTA_NEUTRAL_RESTORED, {"delta_pct": 0.5})

        assert len(events_received) == 4
        assert events_received[0] == (EventType.DELTA_THRESHOLD_BREACHED, 7.5)
        assert events_received[-1] == (EventType.DELTA_NEUTRAL_RESTORED, 0.5)

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self, bus):
        """Should notify all subscribers of same event."""
        notified = set()

        async def subscriber_a(e):
            notified.add("A")

        async def subscriber_b(e):
            notified.add("B")

        async def subscriber_c(e):
            notified.add("C")

        bus.subscribe(EventType.BOT_STARTED, subscriber_a)
        bus.subscribe(EventType.BOT_STARTED, subscriber_b)
        bus.subscribe(EventType.BOT_STARTED, subscriber_c)

        await bus.emit(EventType.BOT_STARTED)

        assert notified == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_correlation_tracking(self, bus):
        """Should track events by correlation ID."""
        correlated_events = []

        async def track_correlation(event):
            if event.correlation_id == "trade-xyz":
                correlated_events.append(event)

        bus.subscribe(
            [EventType.ORDER_CREATED, EventType.ORDER_FILLED, EventType.POSITION_OPENED],
            track_correlation,
        )

        await bus.emit(EventType.ORDER_CREATED, correlation_id="trade-xyz")
        await bus.emit(EventType.ORDER_CREATED, correlation_id="trade-abc")  # Different
        await bus.emit(EventType.ORDER_FILLED, correlation_id="trade-xyz")

        assert len(correlated_events) == 2
        assert all(e.correlation_id == "trade-xyz" for e in correlated_events)

    @pytest.mark.asyncio
    async def test_error_isolation(self, bus):
        """Should isolate errors between handlers."""
        results = []

        async def failing_handler(e):
            results.append("before_fail")
            raise ValueError("Error!")

        async def success_handler(e):
            results.append("success")

        bus.subscribe(EventType.BOT_STARTED, failing_handler, priority=10)
        bus.subscribe(EventType.BOT_STARTED, success_handler, priority=0)

        await bus.emit(EventType.BOT_STARTED)

        # Both handlers called despite first failing
        assert "before_fail" in results
        assert "success" in results
