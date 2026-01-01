"""WebSocket connection management for real-time updates."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from paradex_py import ParadexSubkey
from paradex_py.api.ws_client import ParadexWebsocketChannel

logger = logging.getLogger(__name__)


@dataclass
class WebSocketSubscription:
    """Tracks a WebSocket subscription."""

    account_id: str
    channel: ParadexWebsocketChannel
    params: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None
    active: bool = True


@dataclass
class WebSocketManager:
    """Manages WebSocket connections for all accounts.

    Handles:
    - Connection lifecycle
    - Subscriptions to private/public channels
    - Reconnection on disconnect
    - Message routing to callbacks
    """

    max_connections_per_ip: int = 20
    reconnect_delay: float = 5.0

    _connections: Dict[str, Any] = field(default_factory=dict)
    _subscriptions: Dict[str, list[WebSocketSubscription]] = field(default_factory=dict)
    _running: bool = False
    _tasks: list[asyncio.Task] = field(default_factory=list)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._connections = {}
        self._subscriptions = {}
        self._running = False
        self._tasks = []

    async def connect_account(
        self,
        account_id: str,
        client: ParadexSubkey,
    ) -> bool:
        """Establish WebSocket connection for account.

        Args:
            account_id: Account identifier
            client: ParadexSubkey client

        Returns:
            True if connected successfully
        """
        if account_id in self._connections:
            logger.debug(f"Account {account_id} already connected")
            return True

        try:
            await client.ws_client.connect()
            self._connections[account_id] = client.ws_client
            self._subscriptions[account_id] = []
            logger.info(f"WebSocket connected for account: {account_id}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed for {account_id}: {e}")
            return False

    async def disconnect_account(self, account_id: str) -> None:
        """Disconnect WebSocket for account.

        Args:
            account_id: Account to disconnect
        """
        if account_id not in self._connections:
            return

        try:
            ws = self._connections[account_id]
            if hasattr(ws, "disconnect"):
                await ws.disconnect()
            elif hasattr(ws, "close"):
                await ws.close()
        except Exception as e:
            logger.warning(f"Error disconnecting {account_id}: {e}")
        finally:
            del self._connections[account_id]
            if account_id in self._subscriptions:
                del self._subscriptions[account_id]
            logger.info(f"WebSocket disconnected: {account_id}")

    async def subscribe_positions(
        self,
        account_id: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to position updates for account.

        Args:
            account_id: Account to subscribe
            callback: Function to call on position updates

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            logger.error(f"No connection for account: {account_id}")
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.POSITIONS,
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.POSITIONS,
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to positions: {account_id}")
            return True
        except Exception as e:
            logger.error(f"Position subscription failed for {account_id}: {e}")
            return False

    async def subscribe_orders(
        self,
        account_id: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to order updates for account.

        Args:
            account_id: Account to subscribe
            callback: Function to call on order updates

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.ORDERS,
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.ORDERS,
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to orders: {account_id}")
            return True
        except Exception as e:
            logger.error(f"Order subscription failed for {account_id}: {e}")
            return False

    async def subscribe_balances(
        self,
        account_id: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to balance updates for account.

        Args:
            account_id: Account to subscribe
            callback: Function to call on balance updates

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.BALANCE_EVENTS,
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.BALANCE_EVENTS,
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to balances: {account_id}")
            return True
        except Exception as e:
            logger.error(f"Balance subscription failed for {account_id}: {e}")
            return False

    async def subscribe_funding(
        self,
        account_id: str,
        market: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to funding rate updates for market.

        Args:
            account_id: Account (for connection)
            market: Market symbol (e.g., BTC-USD-PERP)
            callback: Function to call on funding updates

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.FUNDING_DATA,
                params={"market": market},
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.FUNDING_DATA,
                params={"market": market},
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to funding for {market}")
            return True
        except Exception as e:
            logger.error(f"Funding subscription failed: {e}")
            return False

    async def subscribe_trades(
        self,
        account_id: str,
        market: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to trade stream for market.

        Args:
            account_id: Account (for connection)
            market: Market symbol
            callback: Function to call on trades

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.TRADES,
                params={"market": market},
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.TRADES,
                params={"market": market},
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to trades for {market}")
            return True
        except Exception as e:
            logger.error(f"Trades subscription failed: {e}")
            return False

    async def subscribe_fills(
        self,
        account_id: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to fill notifications for account.

        Args:
            account_id: Account to subscribe
            callback: Function to call on fills

        Returns:
            True if subscribed successfully
        """
        if account_id not in self._connections:
            return False

        try:
            ws = self._connections[account_id]
            await ws.subscribe(
                ParadexWebsocketChannel.FILLS,
                callback=callback,
            )

            sub = WebSocketSubscription(
                account_id=account_id,
                channel=ParadexWebsocketChannel.FILLS,
                callback=callback,
            )
            self._subscriptions[account_id].append(sub)

            logger.info(f"Subscribed to fills: {account_id}")
            return True
        except Exception as e:
            logger.error(f"Fills subscription failed for {account_id}: {e}")
            return False

    async def disconnect_all(self) -> None:
        """Gracefully disconnect all WebSocket connections."""
        self._running = False

        # Cancel any running tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Disconnect all accounts
        for account_id in list(self._connections.keys()):
            await self.disconnect_account(account_id)

        logger.info("All WebSocket connections closed")

    @property
    def connected_accounts(self) -> list[str]:
        """Get list of connected account IDs."""
        return list(self._connections.keys())

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    def is_connected(self, account_id: str) -> bool:
        """Check if account is connected.

        Args:
            account_id: Account to check

        Returns:
            True if connected
        """
        return account_id in self._connections

    def get_subscriptions(self, account_id: str) -> list[WebSocketSubscription]:
        """Get subscriptions for account.

        Args:
            account_id: Account to query

        Returns:
            List of active subscriptions
        """
        return self._subscriptions.get(account_id, [])
