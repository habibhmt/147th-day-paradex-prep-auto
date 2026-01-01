"""Unit tests for WebSocket Manager."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from enum import Enum

from src.network.websocket_manager import (
    WebSocketManager,
    WebSocketSubscription,
)


# Mock the ParadexWebsocketChannel enum
class MockChannel(Enum):
    """Mock WebSocket channels."""

    POSITIONS = "positions"
    ORDERS = "orders"
    BALANCE_EVENTS = "balance_events"
    FUNDING_DATA = "funding_data"
    TRADES = "trades"
    FILLS = "fills"


class TestWebSocketSubscription:
    """Tests for WebSocketSubscription dataclass."""

    def test_create_subscription(self):
        """Should create subscription correctly."""
        callback = MagicMock()
        sub = WebSocketSubscription(
            account_id="acc1",
            channel=MockChannel.POSITIONS,
            callback=callback,
        )

        assert sub.account_id == "acc1"
        assert sub.channel == MockChannel.POSITIONS
        assert sub.callback == callback
        assert sub.active is True

    def test_subscription_with_params(self):
        """Should store params correctly."""
        sub = WebSocketSubscription(
            account_id="acc1",
            channel=MockChannel.TRADES,
            params={"market": "BTC-USD-PERP"},
        )

        assert sub.params == {"market": "BTC-USD-PERP"}

    def test_subscription_default_values(self):
        """Should have correct defaults."""
        sub = WebSocketSubscription(
            account_id="acc1",
            channel=MockChannel.ORDERS,
        )

        assert sub.params is None
        assert sub.callback is None
        assert sub.active is True


class TestWebSocketManager:
    """Tests for WebSocketManager."""

    @pytest.fixture
    def manager(self):
        """Create WebSocket manager."""
        return WebSocketManager()

    @pytest.fixture
    def mock_client(self):
        """Create mock Paradex client."""
        client = MagicMock()
        client.ws_client = MagicMock()
        client.ws_client.connect = AsyncMock()
        client.ws_client.subscribe = AsyncMock()
        client.ws_client.disconnect = AsyncMock()
        return client

    def test_initial_state(self, manager):
        """Should start with correct initial state."""
        assert manager.connection_count == 0
        assert len(manager.connected_accounts) == 0
        assert manager._running is False

    def test_max_connections_default(self, manager):
        """Should have default max connections."""
        assert manager.max_connections_per_ip == 20

    def test_reconnect_delay_default(self, manager):
        """Should have default reconnect delay."""
        assert manager.reconnect_delay == 5.0

    @pytest.mark.asyncio
    async def test_connect_account(self, manager, mock_client):
        """Should connect account successfully."""
        result = await manager.connect_account("acc1", mock_client)

        assert result is True
        assert manager.is_connected("acc1")
        assert manager.connection_count == 1
        mock_client.ws_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_account_already_connected(self, manager, mock_client):
        """Should handle already connected account."""
        await manager.connect_account("acc1", mock_client)

        # Connect again
        result = await manager.connect_account("acc1", mock_client)

        assert result is True
        # Should not call connect again
        assert mock_client.ws_client.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_account_failure(self, manager, mock_client):
        """Should handle connection failure."""
        mock_client.ws_client.connect.side_effect = Exception("Connection failed")

        result = await manager.connect_account("acc1", mock_client)

        assert result is False
        assert not manager.is_connected("acc1")

    @pytest.mark.asyncio
    async def test_disconnect_account(self, manager, mock_client):
        """Should disconnect account."""
        await manager.connect_account("acc1", mock_client)

        await manager.disconnect_account("acc1")

        assert not manager.is_connected("acc1")
        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_account(self, manager):
        """Should handle disconnecting nonexistent account."""
        # Should not raise
        await manager.disconnect_account("nonexistent")

    @pytest.mark.asyncio
    async def test_subscribe_positions(self, manager, mock_client):
        """Should subscribe to positions."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_positions("acc1", callback)

        assert result is True
        subs = manager.get_subscriptions("acc1")
        assert len(subs) == 1

    @pytest.mark.asyncio
    async def test_subscribe_positions_not_connected(self, manager):
        """Should fail subscription without connection."""
        callback = MagicMock()

        result = await manager.subscribe_positions("acc1", callback)

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_orders(self, manager, mock_client):
        """Should subscribe to orders."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_orders("acc1", callback)

        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_balances(self, manager, mock_client):
        """Should subscribe to balances."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_balances("acc1", callback)

        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_funding(self, manager, mock_client):
        """Should subscribe to funding data."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_funding("acc1", "BTC-USD-PERP", callback)

        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, manager, mock_client):
        """Should subscribe to trades."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_trades("acc1", "BTC-USD-PERP", callback)

        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_fills(self, manager, mock_client):
        """Should subscribe to fills."""
        await manager.connect_account("acc1", mock_client)
        callback = MagicMock()

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_fills("acc1", callback)

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_all(self, manager, mock_client):
        """Should disconnect all accounts."""
        await manager.connect_account("acc1", mock_client)

        mock_client2 = MagicMock()
        mock_client2.ws_client = MagicMock()
        mock_client2.ws_client.connect = AsyncMock()
        mock_client2.ws_client.disconnect = AsyncMock()
        await manager.connect_account("acc2", mock_client2)

        await manager.disconnect_all()

        assert manager.connection_count == 0
        assert manager._running is False

    def test_connected_accounts(self, manager):
        """Should return connected accounts list."""
        # Manually add to simulate connections
        manager._connections["acc1"] = MagicMock()
        manager._connections["acc2"] = MagicMock()

        accounts = manager.connected_accounts

        assert len(accounts) == 2
        assert "acc1" in accounts
        assert "acc2" in accounts

    def test_is_connected(self, manager):
        """Should check connection status."""
        manager._connections["acc1"] = MagicMock()

        assert manager.is_connected("acc1") is True
        assert manager.is_connected("acc2") is False

    def test_get_subscriptions_empty(self, manager):
        """Should return empty list for no subscriptions."""
        subs = manager.get_subscriptions("nonexistent")

        assert subs == []

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, manager, mock_client):
        """Should track multiple subscriptions."""
        await manager.connect_account("acc1", mock_client)

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            await manager.subscribe_positions("acc1", MagicMock())
            await manager.subscribe_orders("acc1", MagicMock())
            await manager.subscribe_fills("acc1", MagicMock())

        subs = manager.get_subscriptions("acc1")
        assert len(subs) == 3

    @pytest.mark.asyncio
    async def test_subscription_failure(self, manager, mock_client):
        """Should handle subscription failure."""
        await manager.connect_account("acc1", mock_client)
        mock_client.ws_client.subscribe.side_effect = Exception("Sub failed")

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            result = await manager.subscribe_positions("acc1", MagicMock())

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_with_disconnect_method(self, manager, mock_client):
        """Should use disconnect method when available."""
        await manager.connect_account("acc1", mock_client)

        await manager.disconnect_account("acc1")

        mock_client.ws_client.disconnect.assert_called_once()
        assert "acc1" not in manager._connections

    @pytest.mark.asyncio
    async def test_disconnect_cleanup_subscriptions(self, manager, mock_client):
        """Should cleanup subscriptions on disconnect."""
        await manager.connect_account("acc1", mock_client)

        with patch(
            "src.network.websocket_manager.ParadexWebsocketChannel",
            MockChannel,
        ):
            await manager.subscribe_positions("acc1", MagicMock())

        await manager.disconnect_account("acc1")

        # Subscriptions should be cleaned up
        assert "acc1" not in manager._subscriptions

    @pytest.mark.asyncio
    async def test_connect_multiple_accounts(self, manager, mock_client):
        """Should connect multiple accounts."""
        mock_client2 = MagicMock()
        mock_client2.ws_client = MagicMock()
        mock_client2.ws_client.connect = AsyncMock()

        await manager.connect_account("acc1", mock_client)
        await manager.connect_account("acc2", mock_client2)

        assert manager.connection_count == 2
        assert manager.is_connected("acc1")
        assert manager.is_connected("acc2")
