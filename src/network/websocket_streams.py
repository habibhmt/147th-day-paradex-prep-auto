"""
WebSocket Stream Manager for Paradex.

Handles real-time data streams with reconnection, heartbeat, and message processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import json
import logging
import time
import hashlib


logger = logging.getLogger(__name__)


class StreamType(Enum):
    """WebSocket stream types."""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINE = "kline"
    FUNDING = "funding"
    LIQUIDATIONS = "liquidations"
    POSITIONS = "positions"
    ORDERS = "orders"
    FILLS = "fills"
    ACCOUNT = "account"


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    DATA = "data"
    ERROR = "error"
    AUTH = "auth"


@dataclass
class StreamConfig:
    """WebSocket stream configuration."""
    url: str
    reconnect_enabled: bool = True
    reconnect_interval: float = 1.0
    reconnect_max_interval: float = 60.0
    reconnect_backoff: float = 2.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0
    message_queue_size: int = 10000
    max_subscriptions: int = 200
    compression: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "reconnect_enabled": self.reconnect_enabled,
            "reconnect_interval": self.reconnect_interval,
            "reconnect_max_interval": self.reconnect_max_interval,
            "reconnect_backoff": self.reconnect_backoff,
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "message_queue_size": self.message_queue_size,
            "max_subscriptions": self.max_subscriptions,
            "compression": self.compression,
        }


@dataclass
class Subscription:
    """WebSocket subscription."""
    id: str
    stream_type: StreamType
    symbol: Optional[str]
    params: dict = field(default_factory=dict)
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    message_count: int = 0
    last_message_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "stream_type": self.stream_type.value,
            "symbol": self.symbol,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "active": self.active,
            "message_count": self.message_count,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }


@dataclass
class StreamMessage:
    """WebSocket stream message."""
    message_type: MessageType
    stream_type: Optional[StreamType]
    symbol: Optional[str]
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)
    raw: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "message_type": self.message_type.value,
            "stream_type": self.stream_type.value if self.stream_type else None,
            "symbol": self.symbol,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TickerData:
    """Real-time ticker data."""
    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    change_percent_24h: Decimal
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "price": str(self.price),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "bid_size": str(self.bid_size),
            "ask_size": str(self.ask_size),
            "volume_24h": str(self.volume_24h),
            "high_24h": str(self.high_24h),
            "low_24h": str(self.low_24h),
            "change_24h": str(self.change_24h),
            "change_percent_24h": str(self.change_percent_24h),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderBookUpdate:
    """Order book update data."""
    symbol: str
    bids: list  # [(price, size), ...]
    asks: list  # [(price, size), ...]
    sequence: int
    timestamp: datetime
    is_snapshot: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "is_snapshot": self.is_snapshot,
        }


@dataclass
class TradeUpdate:
    """Trade update data."""
    symbol: str
    trade_id: str
    price: Decimal
    size: Decimal
    side: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": str(self.price),
            "size": str(self.size),
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class KlineUpdate:
    """Kline/candlestick update data."""
    symbol: str
    interval: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    is_closed: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "timestamp": self.timestamp.isoformat(),
            "is_closed": self.is_closed,
        }


@dataclass
class ConnectionStats:
    """WebSocket connection statistics."""
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    reconnect_count: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    errors: int = 0
    latency_ms: Optional[float] = None
    uptime_seconds: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "disconnected_at": self.disconnected_at.isoformat() if self.disconnected_at else None,
            "reconnect_count": self.reconnect_count,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "errors": self.errors,
            "latency_ms": self.latency_ms,
            "uptime_seconds": self.uptime_seconds,
        }


class MessageParser:
    """Parse WebSocket messages into typed objects."""

    def parse_ticker(self, data: dict, symbol: str) -> TickerData:
        """Parse ticker message."""
        return TickerData(
            symbol=symbol,
            price=Decimal(str(data.get("price", "0"))),
            bid=Decimal(str(data.get("bid", "0"))),
            ask=Decimal(str(data.get("ask", "0"))),
            bid_size=Decimal(str(data.get("bid_size", "0"))),
            ask_size=Decimal(str(data.get("ask_size", "0"))),
            volume_24h=Decimal(str(data.get("volume_24h", "0"))),
            high_24h=Decimal(str(data.get("high_24h", "0"))),
            low_24h=Decimal(str(data.get("low_24h", "0"))),
            change_24h=Decimal(str(data.get("change_24h", "0"))),
            change_percent_24h=Decimal(str(data.get("change_percent_24h", "0"))),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )

    def parse_orderbook(self, data: dict, symbol: str) -> OrderBookUpdate:
        """Parse order book message."""
        return OrderBookUpdate(
            symbol=symbol,
            bids=[(Decimal(str(p)), Decimal(str(s))) for p, s in data.get("bids", [])],
            asks=[(Decimal(str(p)), Decimal(str(s))) for p, s in data.get("asks", [])],
            sequence=data.get("sequence", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            is_snapshot=data.get("is_snapshot", False),
        )

    def parse_trade(self, data: dict, symbol: str) -> TradeUpdate:
        """Parse trade message."""
        return TradeUpdate(
            symbol=symbol,
            trade_id=data.get("trade_id", ""),
            price=Decimal(str(data.get("price", "0"))),
            size=Decimal(str(data.get("size", "0"))),
            side=data.get("side", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )

    def parse_kline(self, data: dict, symbol: str) -> KlineUpdate:
        """Parse kline message."""
        return KlineUpdate(
            symbol=symbol,
            interval=data.get("interval", "1m"),
            open=Decimal(str(data.get("open", "0"))),
            high=Decimal(str(data.get("high", "0"))),
            low=Decimal(str(data.get("low", "0"))),
            close=Decimal(str(data.get("close", "0"))),
            volume=Decimal(str(data.get("volume", "0"))),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            is_closed=data.get("is_closed", False),
        )


class SubscriptionManager:
    """Manage WebSocket subscriptions."""

    def __init__(self, max_subscriptions: int = 200):
        """Initialize subscription manager."""
        self.max_subscriptions = max_subscriptions
        self._subscriptions: dict[str, Subscription] = {}
        self._symbol_subscriptions: dict[str, list[str]] = {}
        self._type_subscriptions: dict[StreamType, list[str]] = {}

    def add(
        self,
        stream_type: StreamType,
        symbol: Optional[str] = None,
        params: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Add a subscription."""
        if len(self._subscriptions) >= self.max_subscriptions:
            raise ValueError(f"Maximum subscriptions ({self.max_subscriptions}) reached")

        sub_id = self._generate_id(stream_type, symbol, params or {})

        if sub_id in self._subscriptions:
            return self._subscriptions[sub_id]

        subscription = Subscription(
            id=sub_id,
            stream_type=stream_type,
            symbol=symbol,
            params=params or {},
            callback=callback,
        )

        self._subscriptions[sub_id] = subscription

        if symbol:
            if symbol not in self._symbol_subscriptions:
                self._symbol_subscriptions[symbol] = []
            self._symbol_subscriptions[symbol].append(sub_id)

        if stream_type not in self._type_subscriptions:
            self._type_subscriptions[stream_type] = []
        self._type_subscriptions[stream_type].append(sub_id)

        return subscription

    def remove(self, sub_id: str) -> bool:
        """Remove a subscription."""
        if sub_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[sub_id]

        if subscription.symbol and subscription.symbol in self._symbol_subscriptions:
            if sub_id in self._symbol_subscriptions[subscription.symbol]:
                self._symbol_subscriptions[subscription.symbol].remove(sub_id)

        if subscription.stream_type in self._type_subscriptions:
            if sub_id in self._type_subscriptions[subscription.stream_type]:
                self._type_subscriptions[subscription.stream_type].remove(sub_id)

        del self._subscriptions[sub_id]
        return True

    def get(self, sub_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(sub_id)

    def get_by_symbol(self, symbol: str) -> list[Subscription]:
        """Get subscriptions for a symbol."""
        sub_ids = self._symbol_subscriptions.get(symbol, [])
        return [self._subscriptions[sid] for sid in sub_ids if sid in self._subscriptions]

    def get_by_type(self, stream_type: StreamType) -> list[Subscription]:
        """Get subscriptions by stream type."""
        sub_ids = self._type_subscriptions.get(stream_type, [])
        return [self._subscriptions[sid] for sid in sub_ids if sid in self._subscriptions]

    def get_all(self) -> list[Subscription]:
        """Get all subscriptions."""
        return list(self._subscriptions.values())

    def get_active(self) -> list[Subscription]:
        """Get active subscriptions."""
        return [s for s in self._subscriptions.values() if s.active]

    def deactivate(self, sub_id: str) -> bool:
        """Deactivate a subscription."""
        if sub_id in self._subscriptions:
            self._subscriptions[sub_id].active = False
            return True
        return False

    def activate(self, sub_id: str) -> bool:
        """Activate a subscription."""
        if sub_id in self._subscriptions:
            self._subscriptions[sub_id].active = True
            return True
        return False

    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscriptions.clear()
        self._symbol_subscriptions.clear()
        self._type_subscriptions.clear()

    def count(self) -> int:
        """Get subscription count."""
        return len(self._subscriptions)

    def update_stats(self, sub_id: str) -> None:
        """Update subscription statistics."""
        if sub_id in self._subscriptions:
            sub = self._subscriptions[sub_id]
            sub.message_count += 1
            sub.last_message_at = datetime.now()

    def _generate_id(self, stream_type: StreamType, symbol: Optional[str], params: dict) -> str:
        """Generate subscription ID."""
        key = f"{stream_type.value}:{symbol or 'all'}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


class ReconnectionManager:
    """Manage WebSocket reconnection logic."""

    def __init__(
        self,
        initial_interval: float = 1.0,
        max_interval: float = 60.0,
        backoff: float = 2.0,
    ):
        """Initialize reconnection manager."""
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.backoff = backoff
        self._current_interval = initial_interval
        self._attempt_count = 0
        self._last_attempt: Optional[datetime] = None

    def get_delay(self) -> float:
        """Get next reconnection delay."""
        delay = min(self._current_interval, self.max_interval)
        self._current_interval *= self.backoff
        self._attempt_count += 1
        self._last_attempt = datetime.now()
        return delay

    def reset(self) -> None:
        """Reset reconnection state."""
        self._current_interval = self.initial_interval
        self._attempt_count = 0
        self._last_attempt = None

    def get_attempt_count(self) -> int:
        """Get reconnection attempt count."""
        return self._attempt_count

    def should_reconnect(self, max_attempts: Optional[int] = None) -> bool:
        """Check if should attempt reconnection."""
        if max_attempts is None:
            return True
        return self._attempt_count < max_attempts


class HeartbeatManager:
    """Manage WebSocket heartbeat."""

    def __init__(self, interval: float = 30.0, timeout: float = 10.0):
        """Initialize heartbeat manager."""
        self.interval = interval
        self.timeout = timeout
        self._last_ping: Optional[datetime] = None
        self._last_pong: Optional[datetime] = None
        self._pending_ping: bool = False
        self._latency_samples: list[float] = []

    def send_ping(self) -> dict:
        """Create ping message."""
        self._last_ping = datetime.now()
        self._pending_ping = True
        return {"type": "ping", "timestamp": time.time()}

    def receive_pong(self) -> Optional[float]:
        """Process pong response, return latency in ms."""
        if not self._pending_ping or not self._last_ping:
            return None

        self._last_pong = datetime.now()
        self._pending_ping = False
        latency = (self._last_pong - self._last_ping).total_seconds() * 1000

        self._latency_samples.append(latency)
        if len(self._latency_samples) > 100:
            self._latency_samples.pop(0)

        return latency

    def is_timeout(self) -> bool:
        """Check if heartbeat has timed out."""
        if not self._pending_ping or not self._last_ping:
            return False

        elapsed = (datetime.now() - self._last_ping).total_seconds()
        return elapsed > self.timeout

    def needs_ping(self) -> bool:
        """Check if ping should be sent."""
        if self._pending_ping:
            return False

        if self._last_pong is None:
            return True

        elapsed = (datetime.now() - self._last_pong).total_seconds()
        return elapsed >= self.interval

    def get_average_latency(self) -> Optional[float]:
        """Get average latency in ms."""
        if not self._latency_samples:
            return None
        return sum(self._latency_samples) / len(self._latency_samples)

    def reset(self) -> None:
        """Reset heartbeat state."""
        self._last_ping = None
        self._last_pong = None
        self._pending_ping = False


class MessageQueue:
    """Async message queue for stream processing."""

    def __init__(self, max_size: int = 10000):
        """Initialize message queue."""
        self.max_size = max_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._dropped_count = 0

    async def put(self, message: StreamMessage) -> bool:
        """Add message to queue."""
        if self._queue.full():
            try:
                self._queue.get_nowait()  # Drop oldest
                self._dropped_count += 1
            except asyncio.QueueEmpty:
                pass

        await self._queue.put(message)
        return True

    def put_nowait(self, message: StreamMessage) -> bool:
        """Add message without waiting."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            return False

    async def get(self, timeout: Optional[float] = None) -> Optional[StreamMessage]:
        """Get message from queue."""
        try:
            if timeout:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def get_nowait(self) -> Optional[StreamMessage]:
        """Get message without waiting."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def dropped_count(self) -> int:
        """Get count of dropped messages."""
        return self._dropped_count

    def clear(self) -> int:
        """Clear the queue, return count cleared."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count


class StreamProcessor:
    """Process stream messages with callbacks."""

    def __init__(self):
        """Initialize stream processor."""
        self._handlers: dict[StreamType, list[Callable]] = {}
        self._global_handlers: list[Callable] = []
        self._error_handlers: list[Callable] = []
        self._parser = MessageParser()

    def add_handler(self, stream_type: StreamType, handler: Callable) -> None:
        """Add handler for stream type."""
        if stream_type not in self._handlers:
            self._handlers[stream_type] = []
        self._handlers[stream_type].append(handler)

    def remove_handler(self, stream_type: StreamType, handler: Callable) -> bool:
        """Remove handler for stream type."""
        if stream_type in self._handlers and handler in self._handlers[stream_type]:
            self._handlers[stream_type].remove(handler)
            return True
        return False

    def add_global_handler(self, handler: Callable) -> None:
        """Add global message handler."""
        self._global_handlers.append(handler)

    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler."""
        self._error_handlers.append(handler)

    async def process(self, message: StreamMessage) -> None:
        """Process a stream message."""
        # Global handlers
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                await self._handle_error(e, message)

        # Type-specific handlers
        if message.stream_type and message.stream_type in self._handlers:
            parsed_data = self._parse_message(message)
            for handler in self._handlers[message.stream_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(parsed_data)
                    else:
                        handler(parsed_data)
                except Exception as e:
                    await self._handle_error(e, message)

    def _parse_message(self, message: StreamMessage) -> Any:
        """Parse message to typed object."""
        if not message.stream_type:
            return message.data

        symbol = message.symbol or ""

        if message.stream_type == StreamType.TICKER:
            return self._parser.parse_ticker(message.data, symbol)
        elif message.stream_type == StreamType.ORDERBOOK:
            return self._parser.parse_orderbook(message.data, symbol)
        elif message.stream_type == StreamType.TRADES:
            return self._parser.parse_trade(message.data, symbol)
        elif message.stream_type == StreamType.KLINE:
            return self._parser.parse_kline(message.data, symbol)

        return message.data

    async def _handle_error(self, error: Exception, message: StreamMessage) -> None:
        """Handle processing error."""
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error, message)
                else:
                    handler(error, message)
            except Exception:
                pass  # Ignore errors in error handlers


class WebSocketStreamManager:
    """Manage WebSocket streams for real-time data."""

    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize WebSocket stream manager."""
        self.config = config or StreamConfig(url="wss://ws.api.paradex.trade/v1")
        self._state = ConnectionState.DISCONNECTED
        self._subscriptions = SubscriptionManager(self.config.max_subscriptions)
        self._reconnection = ReconnectionManager(
            self.config.reconnect_interval,
            self.config.reconnect_max_interval,
            self.config.reconnect_backoff,
        )
        self._heartbeat = HeartbeatManager(
            self.config.heartbeat_interval,
            self.config.heartbeat_timeout,
        )
        self._message_queue = MessageQueue(self.config.message_queue_size)
        self._processor = StreamProcessor()
        self._stats = ConnectionStats()
        self._callbacks: dict[str, list[Callable]] = {
            "connect": [],
            "disconnect": [],
            "error": [],
            "reconnect": [],
        }

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def stats(self) -> ConnectionStats:
        """Get connection statistics."""
        if self._stats.connected_at and self._state == ConnectionState.CONNECTED:
            self._stats.uptime_seconds = (datetime.now() - self._stats.connected_at).total_seconds()
        self._stats.latency_ms = self._heartbeat.get_average_latency()
        return self._stats

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> bool:
        """Remove event callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            return True
        return False

    async def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to callbacks."""
        if event not in self._callbacks:
            return
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        if self._state in (ConnectionState.CONNECTING, ConnectionState.CONNECTED):
            return self._state == ConnectionState.CONNECTED

        self._state = ConnectionState.CONNECTING

        # Simulate connection for now (real implementation would use websockets library)
        self._state = ConnectionState.CONNECTED
        self._stats.connected_at = datetime.now()
        self._reconnection.reset()
        self._heartbeat.reset()

        await self._emit("connect")
        return True

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self._state == ConnectionState.DISCONNECTED:
            return

        self._state = ConnectionState.CLOSING
        self._stats.disconnected_at = datetime.now()
        self._state = ConnectionState.DISCONNECTED

        await self._emit("disconnect")

    async def reconnect(self) -> bool:
        """Reconnect to WebSocket server."""
        if not self.config.reconnect_enabled:
            return False

        self._state = ConnectionState.RECONNECTING
        self._stats.reconnect_count += 1

        delay = self._reconnection.get_delay()
        await asyncio.sleep(delay)

        success = await self.connect()

        if success:
            # Resubscribe to all active subscriptions
            for sub in self._subscriptions.get_active():
                await self._send_subscribe(sub)
            await self._emit("reconnect")

        return success

    def subscribe_ticker(
        self,
        symbol: str,
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to ticker updates."""
        return self._subscriptions.add(StreamType.TICKER, symbol, callback=callback)

    def subscribe_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to order book updates."""
        return self._subscriptions.add(
            StreamType.ORDERBOOK,
            symbol,
            params={"depth": depth},
            callback=callback,
        )

    def subscribe_trades(
        self,
        symbol: str,
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to trade updates."""
        return self._subscriptions.add(StreamType.TRADES, symbol, callback=callback)

    def subscribe_kline(
        self,
        symbol: str,
        interval: str = "1m",
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to kline/candlestick updates."""
        return self._subscriptions.add(
            StreamType.KLINE,
            symbol,
            params={"interval": interval},
            callback=callback,
        )

    def subscribe_funding(
        self,
        symbol: str,
        callback: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to funding rate updates."""
        return self._subscriptions.add(StreamType.FUNDING, symbol, callback=callback)

    def subscribe_positions(self, callback: Optional[Callable] = None) -> Subscription:
        """Subscribe to position updates."""
        return self._subscriptions.add(StreamType.POSITIONS, callback=callback)

    def subscribe_orders(self, callback: Optional[Callable] = None) -> Subscription:
        """Subscribe to order updates."""
        return self._subscriptions.add(StreamType.ORDERS, callback=callback)

    def subscribe_fills(self, callback: Optional[Callable] = None) -> Subscription:
        """Subscribe to fill updates."""
        return self._subscriptions.add(StreamType.FILLS, callback=callback)

    def subscribe_account(self, callback: Optional[Callable] = None) -> Subscription:
        """Subscribe to account updates."""
        return self._subscriptions.add(StreamType.ACCOUNT, callback=callback)

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a stream."""
        return self._subscriptions.remove(subscription_id)

    def unsubscribe_all(self, symbol: Optional[str] = None) -> int:
        """Unsubscribe from all streams for a symbol."""
        count = 0
        if symbol:
            for sub in self._subscriptions.get_by_symbol(symbol):
                if self._subscriptions.remove(sub.id):
                    count += 1
        else:
            count = self._subscriptions.count()
            self._subscriptions.clear()
        return count

    def get_subscriptions(self) -> list[Subscription]:
        """Get all subscriptions."""
        return self._subscriptions.get_all()

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def add_handler(self, stream_type: StreamType, handler: Callable) -> None:
        """Add message handler for stream type."""
        self._processor.add_handler(stream_type, handler)

    def remove_handler(self, stream_type: StreamType, handler: Callable) -> bool:
        """Remove message handler."""
        return self._processor.remove_handler(stream_type, handler)

    async def process_message(self, raw_message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            self._stats.messages_received += 1
            self._stats.bytes_received += len(raw_message)

            message = self._parse_raw_message(data, raw_message)

            if message.message_type == MessageType.PONG:
                self._heartbeat.receive_pong()
                return

            if message.message_type == MessageType.ERROR:
                self._stats.errors += 1
                await self._emit("error", message.data)
                return

            # Update subscription stats
            if message.stream_type:
                for sub in self._subscriptions.get_by_type(message.stream_type):
                    if sub.symbol is None or sub.symbol == message.symbol:
                        self._subscriptions.update_stats(sub.id)
                        if sub.callback:
                            try:
                                if asyncio.iscoroutinefunction(sub.callback):
                                    await sub.callback(message)
                                else:
                                    sub.callback(message)
                            except Exception as e:
                                logger.error(f"Error in subscription callback: {e}")

            # Process through handlers
            await self._processor.process(message)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            self._stats.errors += 1

    def _parse_raw_message(self, data: dict, raw: str) -> StreamMessage:
        """Parse raw message to StreamMessage."""
        msg_type = MessageType.DATA
        stream_type = None
        symbol = None

        if "type" in data:
            type_str = data["type"].lower()
            if type_str == "pong":
                msg_type = MessageType.PONG
            elif type_str == "error":
                msg_type = MessageType.ERROR

        if "channel" in data or "stream" in data:
            channel = data.get("channel", data.get("stream", ""))
            stream_type, symbol = self._parse_channel(channel)

        return StreamMessage(
            message_type=msg_type,
            stream_type=stream_type,
            symbol=symbol,
            data=data.get("data", data),
            raw=raw,
        )

    def _parse_channel(self, channel: str) -> tuple[Optional[StreamType], Optional[str]]:
        """Parse channel name to stream type and symbol."""
        parts = channel.split("@")

        if len(parts) >= 2:
            symbol = parts[0].upper()
            stream_name = parts[1].lower()
        else:
            symbol = None
            stream_name = channel.lower()

        stream_type_map = {
            "ticker": StreamType.TICKER,
            "orderbook": StreamType.ORDERBOOK,
            "depth": StreamType.ORDERBOOK,
            "trade": StreamType.TRADES,
            "trades": StreamType.TRADES,
            "kline": StreamType.KLINE,
            "candle": StreamType.KLINE,
            "funding": StreamType.FUNDING,
            "liquidation": StreamType.LIQUIDATIONS,
            "position": StreamType.POSITIONS,
            "order": StreamType.ORDERS,
            "fill": StreamType.FILLS,
            "account": StreamType.ACCOUNT,
        }

        stream_type = stream_type_map.get(stream_name)
        return stream_type, symbol

    async def _send_subscribe(self, subscription: Subscription) -> None:
        """Send subscription message."""
        message = {
            "type": "subscribe",
            "channel": self._build_channel(subscription),
            "params": subscription.params,
        }
        await self._send(message)

    def _build_channel(self, subscription: Subscription) -> str:
        """Build channel name for subscription."""
        if subscription.symbol:
            return f"{subscription.symbol.lower()}@{subscription.stream_type.value}"
        return subscription.stream_type.value

    async def _send(self, message: dict) -> None:
        """Send message to WebSocket."""
        raw = json.dumps(message)
        self._stats.messages_sent += 1
        self._stats.bytes_sent += len(raw)
        # Real implementation would send via websocket connection

    def get_summary(self) -> dict:
        """Get manager summary."""
        return {
            "state": self._state.value,
            "subscriptions": self._subscriptions.count(),
            "active_subscriptions": len(self._subscriptions.get_active()),
            "stats": self.stats.to_dict(),
            "queue_size": self._message_queue.size(),
            "dropped_messages": self._message_queue.dropped_count(),
        }


# Global instance
_stream_manager: Optional[WebSocketStreamManager] = None


def get_stream_manager() -> WebSocketStreamManager:
    """Get global stream manager instance."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = WebSocketStreamManager()
    return _stream_manager


def set_stream_manager(manager: WebSocketStreamManager) -> None:
    """Set global stream manager instance."""
    global _stream_manager
    _stream_manager = manager
