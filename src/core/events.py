"""Event system for decoupled component communication."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import weakref

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for the trading bot."""

    # Account events
    ACCOUNT_CONNECTED = "account.connected"
    ACCOUNT_DISCONNECTED = "account.disconnected"
    ACCOUNT_ERROR = "account.error"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # Order events
    ORDER_CREATED = "order.created"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_ERROR = "order.error"

    # Trade events
    TRADE_EXECUTED = "trade.executed"

    # Delta events
    DELTA_UPDATED = "delta.updated"
    DELTA_THRESHOLD_BREACHED = "delta.threshold_breached"
    DELTA_NEUTRAL_RESTORED = "delta.neutral_restored"

    # Rebalance events
    REBALANCE_STARTED = "rebalance.started"
    REBALANCE_COMPLETED = "rebalance.completed"
    REBALANCE_FAILED = "rebalance.failed"

    # Risk events
    RISK_ALERT = "risk.alert"
    RISK_HALT = "risk.halt"
    RISK_RESUME = "risk.resume"

    # System events
    BOT_STARTED = "bot.started"
    BOT_STOPPED = "bot.stopped"
    BOT_ERROR = "bot.error"

    # Health events
    HEALTH_CHECK = "health.check"
    HEALTH_DEGRADED = "health.degraded"
    HEALTH_RECOVERED = "health.recovered"


@dataclass
class Event:
    """An event in the system."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


@dataclass
class EventMetrics:
    """Metrics for event system."""

    events_emitted: int = 0
    events_handled: int = 0
    handlers_called: int = 0
    errors: int = 0
    avg_handle_time_ms: float = 0.0
    _total_handle_time: float = 0.0

    def record_emit(self) -> None:
        """Record event emission."""
        self.events_emitted += 1

    def record_handle(self, duration_ms: float) -> None:
        """Record event handling."""
        self.events_handled += 1
        self._total_handle_time += duration_ms
        self.avg_handle_time_ms = self._total_handle_time / self.events_handled

    def record_handler_call(self) -> None:
        """Record handler invocation."""
        self.handlers_called += 1

    def record_error(self) -> None:
        """Record handler error."""
        self.errors += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "events_emitted": self.events_emitted,
            "events_handled": self.events_handled,
            "handlers_called": self.handlers_called,
            "errors": self.errors,
            "avg_handle_time_ms": round(self.avg_handle_time_ms, 2),
        }


@dataclass
class EventHandler:
    """A registered event handler."""

    callback: Callable
    event_types: Set[EventType]
    priority: int = 0  # Higher = earlier
    once: bool = False  # Remove after first call
    filter_fn: Optional[Callable[[Event], bool]] = None

    def matches(self, event: Event) -> bool:
        """Check if handler matches event.

        Args:
            event: Event to check

        Returns:
            True if handler should handle event
        """
        if event.type not in self.event_types:
            return False

        if self.filter_fn and not self.filter_fn(event):
            return False

        return True


@dataclass
class EventBus:
    """Central event bus for publishing and subscribing to events.

    Features:
    - Type-safe event handling
    - Priority-based handler ordering
    - Async handler support
    - One-time handlers
    - Event filtering
    - Metrics tracking
    """

    _handlers: List[EventHandler] = field(default_factory=list)
    _history: List[Event] = field(default_factory=list)
    _metrics: EventMetrics = field(default_factory=EventMetrics)
    max_history: int = 100
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._handlers = []
        self._history = []
        self._metrics = EventMetrics()
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        event_types: EventType | List[EventType],
        callback: Callable,
        priority: int = 0,
        once: bool = False,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> EventHandler:
        """Subscribe to event types.

        Args:
            event_types: Event type(s) to subscribe to
            callback: Handler function
            priority: Handler priority (higher = earlier)
            once: Remove handler after first call
            filter_fn: Optional filter function

        Returns:
            EventHandler for unsubscribing
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]

        handler = EventHandler(
            callback=callback,
            event_types=set(event_types),
            priority=priority,
            once=once,
            filter_fn=filter_fn,
        )

        self._handlers.append(handler)
        # Sort by priority (descending)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)

        logger.debug(f"Subscribed to events: {[e.value for e in event_types]}")
        return handler

    def unsubscribe(self, handler: EventHandler) -> bool:
        """Unsubscribe a handler.

        Args:
            handler: Handler to remove

        Returns:
            True if removed
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False

    def unsubscribe_all(self, event_type: Optional[EventType] = None) -> int:
        """Unsubscribe all handlers for event type.

        Args:
            event_type: Optional specific type (None = all)

        Returns:
            Number of handlers removed
        """
        if event_type is None:
            count = len(self._handlers)
            self._handlers.clear()
            return count

        original_count = len(self._handlers)
        self._handlers = [
            h for h in self._handlers
            if event_type not in h.event_types
        ]
        return original_count - len(self._handlers)

    async def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: str = "system",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit an event.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            correlation_id: Optional correlation ID
        """
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            correlation_id=correlation_id,
        )

        await self._process_event(event)

    async def emit_event(self, event: Event) -> None:
        """Emit a pre-built event.

        Args:
            event: Event to emit
        """
        await self._process_event(event)

    async def _process_event(self, event: Event) -> None:
        """Process an event through handlers."""
        start_time = time.time()
        self._metrics.record_emit()

        # Add to history
        self._history.append(event)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # Find matching handlers
        handlers_to_remove = []
        matching_handlers = [h for h in self._handlers if h.matches(event)]

        for handler in matching_handlers:
            try:
                self._metrics.record_handler_call()

                if asyncio.iscoroutinefunction(handler.callback):
                    await handler.callback(event)
                else:
                    handler.callback(event)

                if handler.once:
                    handlers_to_remove.append(handler)

            except Exception as e:
                self._metrics.record_error()
                logger.error(f"Event handler error for {event.type.value}: {e}")

        # Remove one-time handlers
        for handler in handlers_to_remove:
            self._handlers.remove(handler)

        duration_ms = (time.time() - start_time) * 1000
        self._metrics.record_handle(duration_ms)

        logger.debug(
            f"Event {event.type.value} processed by {len(matching_handlers)} handlers"
        )

    def on(
        self,
        event_types: EventType | List[EventType],
        priority: int = 0,
    ):
        """Decorator for subscribing to events.

        Args:
            event_types: Event type(s) to subscribe to
            priority: Handler priority

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.subscribe(event_types, func, priority=priority)
            return func
        return decorator

    def once(
        self,
        event_types: EventType | List[EventType],
        priority: int = 0,
    ):
        """Decorator for one-time event subscription.

        Args:
            event_types: Event type(s) to subscribe to
            priority: Handler priority

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.subscribe(event_types, func, priority=priority, once=True)
            return func
        return decorator

    def get_handlers(
        self,
        event_type: Optional[EventType] = None,
    ) -> List[EventHandler]:
        """Get registered handlers.

        Args:
            event_type: Optional filter by type

        Returns:
            List of handlers
        """
        if event_type is None:
            return list(self._handlers)

        return [h for h in self._handlers if event_type in h.event_types]

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 20,
    ) -> List[Event]:
        """Get event history.

        Args:
            event_type: Optional filter by type
            limit: Maximum events to return

        Returns:
            List of events
        """
        history = self._history
        if event_type:
            history = [e for e in history if e.type == event_type]

        return history[-limit:]

    def get_metrics(self) -> EventMetrics:
        """Get event metrics."""
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get event bus status.

        Returns:
            Status dictionary
        """
        handler_counts = {}
        for handler in self._handlers:
            for event_type in handler.event_types:
                type_name = event_type.value
                handler_counts[type_name] = handler_counts.get(type_name, 0) + 1

        return {
            "total_handlers": len(self._handlers),
            "handler_counts": handler_counts,
            "history_size": len(self._history),
            "metrics": self._metrics.to_dict(),
        }

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    def reset_metrics(self) -> None:
        """Reset event metrics."""
        self._metrics = EventMetrics()


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus.

    Returns:
        Global EventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset global event bus."""
    global _global_event_bus
    _global_event_bus = None


async def emit(
    event_type: EventType,
    data: Optional[Dict[str, Any]] = None,
    source: str = "system",
) -> None:
    """Emit event on global bus.

    Args:
        event_type: Type of event
        data: Event data
        source: Event source
    """
    bus = get_event_bus()
    await bus.emit(event_type, data, source)
