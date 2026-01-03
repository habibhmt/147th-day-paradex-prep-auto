"""Tests for WebSocket stream manager."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

from src.network.websocket_streams import (
    StreamType,
    ConnectionState,
    MessageType,
    StreamConfig,
    Subscription,
    StreamMessage,
    TickerData,
    OrderBookUpdate,
    TradeUpdate,
    KlineUpdate,
    ConnectionStats,
    MessageParser,
    SubscriptionManager,
    ReconnectionManager,
    HeartbeatManager,
    MessageQueue,
    StreamProcessor,
    WebSocketStreamManager,
    get_stream_manager,
    set_stream_manager,
)


class TestEnums:
    """Test enum classes."""

    def test_stream_type_values(self):
        """Test StreamType enum values."""
        assert StreamType.TICKER.value == "ticker"
        assert StreamType.ORDERBOOK.value == "orderbook"
        assert StreamType.TRADES.value == "trades"
        assert StreamType.KLINE.value == "kline"
        assert StreamType.FUNDING.value == "funding"
        assert StreamType.POSITIONS.value == "positions"
        assert StreamType.ORDERS.value == "orders"

    def test_connection_state_values(self):
        """Test ConnectionState enum values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.CLOSED.value == "closed"

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.SUBSCRIBE.value == "subscribe"
        assert MessageType.DATA.value == "data"
        assert MessageType.PING.value == "ping"
        assert MessageType.PONG.value == "pong"
        assert MessageType.ERROR.value == "error"


class TestStreamConfig:
    """Test StreamConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = StreamConfig(url="wss://test.com")
        assert config.url == "wss://test.com"
        assert config.reconnect_enabled is True
        assert config.reconnect_interval == 1.0
        assert config.heartbeat_interval == 30.0
        assert config.max_subscriptions == 200

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamConfig(
            url="wss://test.com",
            reconnect_interval=5.0,
            heartbeat_interval=60.0,
            max_subscriptions=100,
        )
        assert config.reconnect_interval == 5.0
        assert config.heartbeat_interval == 60.0
        assert config.max_subscriptions == 100

    def test_to_dict(self):
        """Test config to_dict."""
        config = StreamConfig(url="wss://test.com")
        result = config.to_dict()
        assert result["url"] == "wss://test.com"
        assert "reconnect_enabled" in result
        assert "heartbeat_interval" in result


class TestSubscription:
    """Test Subscription class."""

    def test_creation(self):
        """Test subscription creation."""
        sub = Subscription(
            id="test123",
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
        )
        assert sub.id == "test123"
        assert sub.stream_type == StreamType.TICKER
        assert sub.symbol == "BTC-USD-PERP"
        assert sub.active is True
        assert sub.message_count == 0

    def test_to_dict(self):
        """Test subscription to_dict."""
        sub = Subscription(
            id="test123",
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
        )
        result = sub.to_dict()
        assert result["id"] == "test123"
        assert result["stream_type"] == "ticker"
        assert result["symbol"] == "BTC-USD-PERP"


class TestStreamMessage:
    """Test StreamMessage class."""

    def test_creation(self):
        """Test message creation."""
        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        assert msg.message_type == MessageType.DATA
        assert msg.stream_type == StreamType.TICKER
        assert msg.symbol == "BTC-USD-PERP"
        assert msg.data["price"] == "50000"

    def test_to_dict(self):
        """Test message to_dict."""
        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        result = msg.to_dict()
        assert result["message_type"] == "data"
        assert result["stream_type"] == "ticker"


class TestTickerData:
    """Test TickerData class."""

    def test_creation(self):
        """Test ticker data creation."""
        ticker = TickerData(
            symbol="BTC-USD-PERP",
            price=Decimal("50000"),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            change_24h=Decimal("500"),
            change_percent_24h=Decimal("1.0"),
            timestamp=datetime.now(),
        )
        assert ticker.symbol == "BTC-USD-PERP"
        assert ticker.price == Decimal("50000")

    def test_to_dict(self):
        """Test ticker to_dict."""
        ticker = TickerData(
            symbol="BTC-USD-PERP",
            price=Decimal("50000"),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            change_24h=Decimal("500"),
            change_percent_24h=Decimal("1.0"),
            timestamp=datetime.now(),
        )
        result = ticker.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["price"] == "50000"


class TestOrderBookUpdate:
    """Test OrderBookUpdate class."""

    def test_creation(self):
        """Test order book update creation."""
        update = OrderBookUpdate(
            symbol="BTC-USD-PERP",
            bids=[(Decimal("49999"), Decimal("10"))],
            asks=[(Decimal("50001"), Decimal("10"))],
            sequence=123,
            timestamp=datetime.now(),
        )
        assert update.symbol == "BTC-USD-PERP"
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        assert update.sequence == 123

    def test_to_dict(self):
        """Test order book to_dict."""
        update = OrderBookUpdate(
            symbol="BTC-USD-PERP",
            bids=[(Decimal("49999"), Decimal("10"))],
            asks=[(Decimal("50001"), Decimal("10"))],
            sequence=123,
            timestamp=datetime.now(),
            is_snapshot=True,
        )
        result = update.to_dict()
        assert result["is_snapshot"] is True
        assert result["sequence"] == 123


class TestTradeUpdate:
    """Test TradeUpdate class."""

    def test_creation(self):
        """Test trade update creation."""
        trade = TradeUpdate(
            symbol="BTC-USD-PERP",
            trade_id="trade123",
            price=Decimal("50000"),
            size=Decimal("1.5"),
            side="buy",
            timestamp=datetime.now(),
        )
        assert trade.trade_id == "trade123"
        assert trade.side == "buy"

    def test_to_dict(self):
        """Test trade to_dict."""
        trade = TradeUpdate(
            symbol="BTC-USD-PERP",
            trade_id="trade123",
            price=Decimal("50000"),
            size=Decimal("1.5"),
            side="buy",
            timestamp=datetime.now(),
        )
        result = trade.to_dict()
        assert result["price"] == "50000"
        assert result["side"] == "buy"


class TestKlineUpdate:
    """Test KlineUpdate class."""

    def test_creation(self):
        """Test kline update creation."""
        kline = KlineUpdate(
            symbol="BTC-USD-PERP",
            interval="1h",
            open=Decimal("49500"),
            high=Decimal("50500"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(),
            is_closed=True,
        )
        assert kline.interval == "1h"
        assert kline.is_closed is True

    def test_to_dict(self):
        """Test kline to_dict."""
        kline = KlineUpdate(
            symbol="BTC-USD-PERP",
            interval="1h",
            open=Decimal("49500"),
            high=Decimal("50500"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(),
            is_closed=True,
        )
        result = kline.to_dict()
        assert result["interval"] == "1h"
        assert result["is_closed"] is True


class TestConnectionStats:
    """Test ConnectionStats class."""

    def test_defaults(self):
        """Test default stats."""
        stats = ConnectionStats()
        assert stats.reconnect_count == 0
        assert stats.messages_received == 0
        assert stats.errors == 0

    def test_to_dict(self):
        """Test stats to_dict."""
        stats = ConnectionStats(
            reconnect_count=5,
            messages_received=1000,
            errors=2,
        )
        result = stats.to_dict()
        assert result["reconnect_count"] == 5
        assert result["messages_received"] == 1000
        assert result["errors"] == 2


class TestMessageParser:
    """Test MessageParser class."""

    def test_parse_ticker(self):
        """Test ticker parsing."""
        parser = MessageParser()
        data = {
            "price": "50000",
            "bid": "49999",
            "ask": "50001",
            "bid_size": "10",
            "ask_size": "10",
            "volume_24h": "1000000",
            "high_24h": "51000",
            "low_24h": "49000",
            "change_24h": "500",
            "change_percent_24h": "1.0",
        }
        result = parser.parse_ticker(data, "BTC-USD-PERP")
        assert result.symbol == "BTC-USD-PERP"
        assert result.price == Decimal("50000")

    def test_parse_orderbook(self):
        """Test order book parsing."""
        parser = MessageParser()
        data = {
            "bids": [["49999", "10"], ["49998", "20"]],
            "asks": [["50001", "10"], ["50002", "20"]],
            "sequence": 123,
        }
        result = parser.parse_orderbook(data, "BTC-USD-PERP")
        assert result.symbol == "BTC-USD-PERP"
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.sequence == 123

    def test_parse_trade(self):
        """Test trade parsing."""
        parser = MessageParser()
        data = {
            "trade_id": "trade123",
            "price": "50000",
            "size": "1.5",
            "side": "buy",
        }
        result = parser.parse_trade(data, "BTC-USD-PERP")
        assert result.trade_id == "trade123"
        assert result.price == Decimal("50000")

    def test_parse_kline(self):
        """Test kline parsing."""
        parser = MessageParser()
        data = {
            "interval": "1h",
            "open": "49500",
            "high": "50500",
            "low": "49000",
            "close": "50000",
            "volume": "100000",
            "is_closed": True,
        }
        result = parser.parse_kline(data, "BTC-USD-PERP")
        assert result.interval == "1h"
        assert result.open == Decimal("49500")
        assert result.is_closed is True


class TestSubscriptionManager:
    """Test SubscriptionManager class."""

    def test_add_subscription(self):
        """Test adding subscription."""
        manager = SubscriptionManager()
        sub = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        assert sub.stream_type == StreamType.TICKER
        assert sub.symbol == "BTC-USD-PERP"
        assert manager.count() == 1

    def test_add_duplicate(self):
        """Test adding duplicate subscription."""
        manager = SubscriptionManager()
        sub1 = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        sub2 = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        assert sub1.id == sub2.id
        assert manager.count() == 1

    def test_remove_subscription(self):
        """Test removing subscription."""
        manager = SubscriptionManager()
        sub = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        assert manager.remove(sub.id) is True
        assert manager.count() == 0

    def test_remove_nonexistent(self):
        """Test removing nonexistent subscription."""
        manager = SubscriptionManager()
        assert manager.remove("nonexistent") is False

    def test_get_by_symbol(self):
        """Test getting subscriptions by symbol."""
        manager = SubscriptionManager()
        manager.add(StreamType.TICKER, "BTC-USD-PERP")
        manager.add(StreamType.ORDERBOOK, "BTC-USD-PERP")
        manager.add(StreamType.TICKER, "ETH-USD-PERP")

        btc_subs = manager.get_by_symbol("BTC-USD-PERP")
        assert len(btc_subs) == 2

    def test_get_by_type(self):
        """Test getting subscriptions by type."""
        manager = SubscriptionManager()
        manager.add(StreamType.TICKER, "BTC-USD-PERP")
        manager.add(StreamType.TICKER, "ETH-USD-PERP")
        manager.add(StreamType.ORDERBOOK, "BTC-USD-PERP")

        ticker_subs = manager.get_by_type(StreamType.TICKER)
        assert len(ticker_subs) == 2

    def test_max_subscriptions(self):
        """Test max subscriptions limit."""
        manager = SubscriptionManager(max_subscriptions=2)
        manager.add(StreamType.TICKER, "BTC-USD-PERP")
        manager.add(StreamType.TICKER, "ETH-USD-PERP")

        with pytest.raises(ValueError):
            manager.add(StreamType.TICKER, "SOL-USD-PERP")

    def test_activate_deactivate(self):
        """Test activate/deactivate subscription."""
        manager = SubscriptionManager()
        sub = manager.add(StreamType.TICKER, "BTC-USD-PERP")

        assert manager.deactivate(sub.id) is True
        assert sub.active is False

        assert manager.activate(sub.id) is True
        assert sub.active is True

    def test_get_active(self):
        """Test getting active subscriptions."""
        manager = SubscriptionManager()
        sub1 = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        sub2 = manager.add(StreamType.TICKER, "ETH-USD-PERP")
        manager.deactivate(sub1.id)

        active = manager.get_active()
        assert len(active) == 1
        assert active[0].id == sub2.id

    def test_clear(self):
        """Test clearing subscriptions."""
        manager = SubscriptionManager()
        manager.add(StreamType.TICKER, "BTC-USD-PERP")
        manager.add(StreamType.TICKER, "ETH-USD-PERP")
        manager.clear()
        assert manager.count() == 0

    def test_update_stats(self):
        """Test updating subscription stats."""
        manager = SubscriptionManager()
        sub = manager.add(StreamType.TICKER, "BTC-USD-PERP")
        manager.update_stats(sub.id)
        assert sub.message_count == 1
        assert sub.last_message_at is not None


class TestReconnectionManager:
    """Test ReconnectionManager class."""

    def test_initial_delay(self):
        """Test initial reconnection delay."""
        manager = ReconnectionManager(initial_interval=1.0)
        delay = manager.get_delay()
        assert delay == 1.0
        assert manager.get_attempt_count() == 1

    def test_backoff(self):
        """Test exponential backoff."""
        manager = ReconnectionManager(
            initial_interval=1.0,
            backoff=2.0,
            max_interval=60.0,
        )
        delay1 = manager.get_delay()
        delay2 = manager.get_delay()
        delay3 = manager.get_delay()
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_max_interval(self):
        """Test max interval limit."""
        manager = ReconnectionManager(
            initial_interval=50.0,
            backoff=2.0,
            max_interval=60.0,
        )
        delay1 = manager.get_delay()
        delay2 = manager.get_delay()
        assert delay1 == 50.0
        assert delay2 == 60.0  # Capped at max

    def test_reset(self):
        """Test reset."""
        manager = ReconnectionManager(initial_interval=1.0)
        manager.get_delay()
        manager.get_delay()
        manager.reset()
        assert manager.get_attempt_count() == 0
        assert manager.get_delay() == 1.0

    def test_should_reconnect(self):
        """Test should_reconnect logic."""
        manager = ReconnectionManager()
        assert manager.should_reconnect() is True
        assert manager.should_reconnect(max_attempts=10) is True

        for _ in range(10):
            manager.get_delay()

        assert manager.should_reconnect(max_attempts=10) is False


class TestHeartbeatManager:
    """Test HeartbeatManager class."""

    def test_send_ping(self):
        """Test sending ping."""
        manager = HeartbeatManager()
        ping = manager.send_ping()
        assert ping["type"] == "ping"
        assert "timestamp" in ping

    def test_receive_pong(self):
        """Test receiving pong."""
        manager = HeartbeatManager()
        manager.send_ping()
        latency = manager.receive_pong()
        assert latency is not None
        assert latency >= 0

    def test_receive_pong_without_ping(self):
        """Test receiving pong without pending ping."""
        manager = HeartbeatManager()
        latency = manager.receive_pong()
        assert latency is None

    def test_is_timeout(self):
        """Test timeout detection."""
        manager = HeartbeatManager(timeout=0.01)
        assert manager.is_timeout() is False

        manager.send_ping()
        import time
        time.sleep(0.02)
        assert manager.is_timeout() is True

    def test_needs_ping(self):
        """Test needs_ping logic."""
        manager = HeartbeatManager(interval=0.01)
        assert manager.needs_ping() is True

        manager.send_ping()
        assert manager.needs_ping() is False  # Pending ping

        manager.receive_pong()
        import time
        time.sleep(0.02)
        assert manager.needs_ping() is True

    def test_get_average_latency(self):
        """Test average latency calculation."""
        manager = HeartbeatManager()
        assert manager.get_average_latency() is None

        for _ in range(3):
            manager.send_ping()
            manager.receive_pong()

        avg = manager.get_average_latency()
        assert avg is not None
        assert avg >= 0

    def test_reset(self):
        """Test reset."""
        manager = HeartbeatManager()
        manager.send_ping()
        manager.reset()
        assert manager.needs_ping() is True


class TestMessageQueue:
    """Test MessageQueue class."""

    @pytest.mark.asyncio
    async def test_put_get(self):
        """Test put and get."""
        queue = MessageQueue()
        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        await queue.put(msg)
        assert queue.size() == 1

        result = await queue.get(timeout=1.0)
        assert result.data["price"] == "50000"
        assert queue.size() == 0

    def test_put_nowait(self):
        """Test put_nowait."""
        queue = MessageQueue()
        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        result = queue.put_nowait(msg)
        assert result is True
        assert queue.size() == 1

    def test_get_nowait(self):
        """Test get_nowait."""
        queue = MessageQueue()
        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        queue.put_nowait(msg)
        result = queue.get_nowait()
        assert result is not None
        assert result.data["price"] == "50000"

    def test_get_nowait_empty(self):
        """Test get_nowait on empty queue."""
        queue = MessageQueue()
        result = queue.get_nowait()
        assert result is None

    @pytest.mark.asyncio
    async def test_overflow(self):
        """Test queue overflow handling."""
        queue = MessageQueue(max_size=2)
        msg1 = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC",
            data={"id": 1},
        )
        msg2 = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="ETH",
            data={"id": 2},
        )
        msg3 = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="SOL",
            data={"id": 3},
        )

        await queue.put(msg1)
        await queue.put(msg2)
        await queue.put(msg3)

        assert queue.size() == 2
        assert queue.dropped_count() == 1

    def test_is_full_empty(self):
        """Test is_full and is_empty."""
        queue = MessageQueue(max_size=2)
        assert queue.is_empty() is True
        assert queue.is_full() is False

        queue.put_nowait(StreamMessage(
            message_type=MessageType.DATA,
            stream_type=None,
            symbol=None,
            data={},
        ))
        queue.put_nowait(StreamMessage(
            message_type=MessageType.DATA,
            stream_type=None,
            symbol=None,
            data={},
        ))

        assert queue.is_empty() is False
        assert queue.is_full() is True

    def test_clear(self):
        """Test clear."""
        queue = MessageQueue()
        for i in range(5):
            queue.put_nowait(StreamMessage(
                message_type=MessageType.DATA,
                stream_type=None,
                symbol=None,
                data={"i": i},
            ))

        count = queue.clear()
        assert count == 5
        assert queue.is_empty() is True


class TestStreamProcessor:
    """Test StreamProcessor class."""

    @pytest.mark.asyncio
    async def test_add_handler(self):
        """Test adding handler."""
        processor = StreamProcessor()
        results = []

        def handler(data):
            results.append(data)

        processor.add_handler(StreamType.TICKER, handler)

        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        await processor.process(msg)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_remove_handler(self):
        """Test removing handler."""
        processor = StreamProcessor()
        results = []

        def handler(data):
            results.append(data)

        processor.add_handler(StreamType.TICKER, handler)
        assert processor.remove_handler(StreamType.TICKER, handler) is True
        assert processor.remove_handler(StreamType.TICKER, handler) is False

    @pytest.mark.asyncio
    async def test_global_handler(self):
        """Test global handler."""
        processor = StreamProcessor()
        results = []

        def handler(msg):
            results.append(msg)

        processor.add_global_handler(handler)

        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        await processor.process(msg)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test async handler."""
        processor = StreamProcessor()
        results = []

        async def handler(data):
            await asyncio.sleep(0.01)
            results.append(data)

        processor.add_handler(StreamType.TICKER, handler)

        msg = StreamMessage(
            message_type=MessageType.DATA,
            stream_type=StreamType.TICKER,
            symbol="BTC-USD-PERP",
            data={"price": "50000"},
        )
        await processor.process(msg)

        assert len(results) == 1


class TestWebSocketStreamManager:
    """Test WebSocketStreamManager class."""

    def test_init(self):
        """Test initialization."""
        manager = WebSocketStreamManager()
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.is_connected is False

    def test_custom_config(self):
        """Test custom config."""
        config = StreamConfig(
            url="wss://custom.com",
            max_subscriptions=100,
        )
        manager = WebSocketStreamManager(config)
        assert manager.config.url == "wss://custom.com"
        assert manager.config.max_subscriptions == 100

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connect."""
        manager = WebSocketStreamManager()
        result = await manager.connect()
        assert result is True
        assert manager.state == ConnectionState.CONNECTED
        assert manager.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        manager = WebSocketStreamManager()
        await manager.connect()
        await manager.disconnect()
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect(self):
        """Test reconnect."""
        manager = WebSocketStreamManager()
        await manager.connect()
        await manager.disconnect()
        result = await manager.reconnect()
        assert result is True
        assert manager.state == ConnectionState.CONNECTED

    def test_subscribe_ticker(self):
        """Test ticker subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_ticker("BTC-USD-PERP")
        assert sub.stream_type == StreamType.TICKER
        assert sub.symbol == "BTC-USD-PERP"

    def test_subscribe_orderbook(self):
        """Test order book subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_orderbook("BTC-USD-PERP", depth=50)
        assert sub.stream_type == StreamType.ORDERBOOK
        assert sub.params["depth"] == 50

    def test_subscribe_trades(self):
        """Test trades subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_trades("BTC-USD-PERP")
        assert sub.stream_type == StreamType.TRADES

    def test_subscribe_kline(self):
        """Test kline subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_kline("BTC-USD-PERP", interval="1h")
        assert sub.stream_type == StreamType.KLINE
        assert sub.params["interval"] == "1h"

    def test_subscribe_positions(self):
        """Test positions subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_positions()
        assert sub.stream_type == StreamType.POSITIONS

    def test_subscribe_orders(self):
        """Test orders subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_orders()
        assert sub.stream_type == StreamType.ORDERS

    def test_unsubscribe(self):
        """Test unsubscribe."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_ticker("BTC-USD-PERP")
        result = manager.unsubscribe(sub.id)
        assert result is True
        assert len(manager.get_subscriptions()) == 0

    def test_unsubscribe_all(self):
        """Test unsubscribe all."""
        manager = WebSocketStreamManager()
        manager.subscribe_ticker("BTC-USD-PERP")
        manager.subscribe_orderbook("BTC-USD-PERP")
        manager.subscribe_ticker("ETH-USD-PERP")

        count = manager.unsubscribe_all("BTC-USD-PERP")
        assert count == 2
        assert len(manager.get_subscriptions()) == 1

    def test_unsubscribe_all_symbols(self):
        """Test unsubscribe all symbols."""
        manager = WebSocketStreamManager()
        manager.subscribe_ticker("BTC-USD-PERP")
        manager.subscribe_ticker("ETH-USD-PERP")

        count = manager.unsubscribe_all()
        assert count == 2
        assert len(manager.get_subscriptions()) == 0

    def test_get_subscription(self):
        """Test get subscription."""
        manager = WebSocketStreamManager()
        sub = manager.subscribe_ticker("BTC-USD-PERP")
        result = manager.get_subscription(sub.id)
        assert result is not None
        assert result.id == sub.id

    def test_add_handler(self):
        """Test adding handler."""
        manager = WebSocketStreamManager()
        called = []

        def handler(data):
            called.append(data)

        manager.add_handler(StreamType.TICKER, handler)
        assert manager.remove_handler(StreamType.TICKER, handler) is True

    @pytest.mark.asyncio
    async def test_on_event(self):
        """Test event callbacks."""
        manager = WebSocketStreamManager()
        events = []

        def on_connect():
            events.append("connect")

        def on_disconnect():
            events.append("disconnect")

        manager.on("connect", on_connect)
        manager.on("disconnect", on_disconnect)

        await manager.connect()
        assert "connect" in events

        await manager.disconnect()
        assert "disconnect" in events

    @pytest.mark.asyncio
    async def test_off_event(self):
        """Test removing event callback."""
        manager = WebSocketStreamManager()
        events = []

        def handler():
            events.append("called")

        manager.on("connect", handler)
        manager.off("connect", handler)

        await manager.connect()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_process_message_ticker(self):
        """Test processing ticker message."""
        manager = WebSocketStreamManager()
        manager.subscribe_ticker("BTC-USD-PERP")

        raw = '{"channel": "btc-usd-perp@ticker", "data": {"price": "50000"}}'
        await manager.process_message(raw)
        assert manager.stats.messages_received == 1

    @pytest.mark.asyncio
    async def test_process_message_pong(self):
        """Test processing pong message."""
        manager = WebSocketStreamManager()
        manager._heartbeat.send_ping()

        raw = '{"type": "pong"}'
        await manager.process_message(raw)
        assert manager.stats.messages_received == 1

    @pytest.mark.asyncio
    async def test_process_message_error(self):
        """Test processing error message."""
        manager = WebSocketStreamManager()
        errors = []

        async def on_error(data):
            errors.append(data)

        manager.on("error", on_error)

        raw = '{"type": "error", "data": {"message": "test error"}}'
        await manager.process_message(raw)
        assert manager.stats.errors == 1
        assert len(errors) == 1

    def test_get_summary(self):
        """Test get summary."""
        manager = WebSocketStreamManager()
        manager.subscribe_ticker("BTC-USD-PERP")
        manager.subscribe_orderbook("ETH-USD-PERP")

        summary = manager.get_summary()
        assert summary["state"] == "disconnected"
        assert summary["subscriptions"] == 2
        assert "stats" in summary

    def test_stats(self):
        """Test stats property."""
        manager = WebSocketStreamManager()
        stats = manager.stats
        assert stats.messages_received == 0
        assert stats.reconnect_count == 0


class TestGlobalInstance:
    """Test global instance functions."""

    def test_get_stream_manager(self):
        """Test getting global instance."""
        manager = get_stream_manager()
        assert manager is not None
        assert isinstance(manager, WebSocketStreamManager)

    def test_set_stream_manager(self):
        """Test setting global instance."""
        custom = WebSocketStreamManager()
        set_stream_manager(custom)
        assert get_stream_manager() is custom


class TestChannelParsing:
    """Test channel name parsing."""

    def test_parse_ticker_channel(self):
        """Test parsing ticker channel."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("btc-usd-perp@ticker")
        assert stream_type == StreamType.TICKER
        assert symbol == "BTC-USD-PERP"

    def test_parse_orderbook_channel(self):
        """Test parsing orderbook channel."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("eth-usd-perp@orderbook")
        assert stream_type == StreamType.ORDERBOOK
        assert symbol == "ETH-USD-PERP"

    def test_parse_depth_channel(self):
        """Test parsing depth channel (alias for orderbook)."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("btc-usd-perp@depth")
        assert stream_type == StreamType.ORDERBOOK

    def test_parse_trades_channel(self):
        """Test parsing trades channel."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("btc-usd-perp@trades")
        assert stream_type == StreamType.TRADES

    def test_parse_kline_channel(self):
        """Test parsing kline channel."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("btc-usd-perp@kline")
        assert stream_type == StreamType.KLINE

    def test_parse_unknown_channel(self):
        """Test parsing unknown channel."""
        manager = WebSocketStreamManager()
        stream_type, symbol = manager._parse_channel("unknown@channel")
        assert stream_type is None


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full connection and subscription workflow."""
        manager = WebSocketStreamManager()

        # Connect
        await manager.connect()
        assert manager.is_connected

        # Subscribe
        ticker_sub = manager.subscribe_ticker("BTC-USD-PERP")
        orderbook_sub = manager.subscribe_orderbook("BTC-USD-PERP", depth=20)

        assert len(manager.get_subscriptions()) == 2

        # Process messages
        messages = [
            '{"channel": "btc-usd-perp@ticker", "data": {"price": "50000"}}',
            '{"channel": "btc-usd-perp@orderbook", "data": {"bids": [], "asks": [], "sequence": 1}}',
        ]

        for msg in messages:
            await manager.process_message(msg)

        assert manager.stats.messages_received == 2

        # Unsubscribe
        manager.unsubscribe(ticker_sub.id)
        assert len(manager.get_subscriptions()) == 1

        # Disconnect
        await manager.disconnect()
        assert not manager.is_connected

    @pytest.mark.asyncio
    async def test_subscription_callbacks(self):
        """Test subscription callbacks."""
        manager = WebSocketStreamManager()
        await manager.connect()

        results = []

        def callback(msg):
            results.append(msg)

        manager.subscribe_ticker("BTC-USD-PERP", callback=callback)

        raw = '{"channel": "btc-usd-perp@ticker", "data": {"price": "50000"}}'
        await manager.process_message(raw)

        assert len(results) == 1

        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        """Test multiple symbol subscriptions."""
        manager = WebSocketStreamManager()
        await manager.connect()

        manager.subscribe_ticker("BTC-USD-PERP")
        manager.subscribe_ticker("ETH-USD-PERP")
        manager.subscribe_ticker("SOL-USD-PERP")

        assert len(manager.get_subscriptions()) == 3

        # Process messages for each
        for symbol in ["btc", "eth", "sol"]:
            raw = f'{{"channel": "{symbol}-usd-perp@ticker", "data": {{"price": "100"}}}}'
            await manager.process_message(raw)

        assert manager.stats.messages_received == 3

        await manager.disconnect()
