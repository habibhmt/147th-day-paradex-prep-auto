"""Order execution and management."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from src.core.account_manager import AccountManager
from src.network.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by Paradex."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class OrderRequest:
    """Request to submit an order."""

    account_id: str
    market: str
    side: OrderSide
    size: Decimal
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    client_id: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = False

    def __post_init__(self):
        """Generate client ID if not provided."""
        if not self.client_id:
            self.client_id = f"delta-{uuid.uuid4().hex[:8]}"


@dataclass
class OrderResult:
    """Result of order submission."""

    success: bool
    order_id: Optional[str] = None
    client_id: Optional[str] = None
    filled_size: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")
    status: OrderStatus = OrderStatus.PENDING
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "client_id": self.client_id,
            "filled_size": str(self.filled_size),
            "avg_price": str(self.avg_price),
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class OrderManager:
    """Handles order execution across accounts.

    Features:
    - Rate limit compliance
    - Batch order support
    - Order tracking
    - Error handling with retries
    """

    account_manager: AccountManager
    rate_limiter: RateLimiter
    max_retries: int = 3
    retry_delay: float = 0.5

    # Order tracking
    _pending_orders: Dict[str, OrderRequest] = field(default_factory=dict)
    _order_results: Dict[str, OrderResult] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._pending_orders = {}
        self._order_results = {}

    async def submit_order(
        self,
        order: OrderRequest,
        apply_randomization: bool = False,
    ) -> OrderResult:
        """Submit a single order with rate limiting.

        Args:
            order: Order request
            apply_randomization: Whether to randomize size (for anti-detection)

        Returns:
            OrderResult with submission outcome
        """
        # Get client
        client = self.account_manager.get_client(order.account_id)
        if not client:
            return OrderResult(
                success=False,
                client_id=order.client_id,
                error=f"No client for account: {order.account_id}",
                status=OrderStatus.FAILED,
            )

        # Acquire rate limit slot
        await self.rate_limiter.acquire_order_slot()

        # Track pending
        self._pending_orders[order.client_id] = order

        # Submit with retries
        for attempt in range(self.max_retries):
            try:
                result = await self._execute_order(client, order)

                # Track result
                self._order_results[order.client_id] = result
                del self._pending_orders[order.client_id]

                if result.success:
                    logger.info(
                        f"Order submitted: {order.account_id} {order.side.value} "
                        f"{order.size} {order.market}"
                    )
                return result

            except Exception as e:
                logger.warning(
                    f"Order attempt {attempt + 1} failed for {order.account_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # All retries failed
        result = OrderResult(
            success=False,
            client_id=order.client_id,
            error="Max retries exceeded",
            status=OrderStatus.FAILED,
        )
        self._order_results[order.client_id] = result
        if order.client_id in self._pending_orders:
            del self._pending_orders[order.client_id]

        return result

    async def _execute_order(
        self,
        client,
        order: OrderRequest,
    ) -> OrderResult:
        """Execute order on exchange.

        Args:
            client: Paradex client
            order: Order request

        Returns:
            OrderResult
        """
        try:
            # Build order params
            params = {
                "market": order.market,
                "side": order.side.value,
                "type": order.order_type.value,
                "size": str(order.size),
            }

            if order.price:
                params["price"] = str(order.price)

            if order.client_id:
                params["client_id"] = order.client_id

            if order.reduce_only:
                params["reduce_only"] = True

            if order.post_only and order.order_type == OrderType.LIMIT:
                params["post_only"] = True

            # Submit order
            response = await client.api_client.submit_order(**params)

            # Parse response
            if response:
                return OrderResult(
                    success=True,
                    order_id=response.get("id"),
                    client_id=order.client_id,
                    filled_size=Decimal(str(response.get("filled_size", "0"))),
                    avg_price=Decimal(str(response.get("avg_fill_price", "0"))),
                    status=OrderStatus.FILLED
                    if response.get("status") == "FILLED"
                    else OrderStatus.OPEN,
                )

            return OrderResult(
                success=False,
                client_id=order.client_id,
                error="Empty response from exchange",
                status=OrderStatus.FAILED,
            )

        except Exception as e:
            return OrderResult(
                success=False,
                client_id=order.client_id,
                error=str(e),
                status=OrderStatus.FAILED,
            )

    async def submit_batch(
        self,
        orders: List[OrderRequest],
    ) -> List[OrderResult]:
        """Submit batch of orders.

        Batch orders count as 1 rate limit unit for up to 50 orders.

        Args:
            orders: List of order requests

        Returns:
            List of OrderResults
        """
        if not orders:
            return []

        if len(orders) == 1:
            return [await self.submit_order(orders[0])]

        # Group by account (batch must be same account)
        by_account: Dict[str, List[OrderRequest]] = {}
        for order in orders:
            if order.account_id not in by_account:
                by_account[order.account_id] = []
            by_account[order.account_id].append(order)

        # Submit batches per account
        all_results = []
        for account_id, account_orders in by_account.items():
            results = await self._submit_account_batch(account_id, account_orders)
            all_results.extend(results)

        return all_results

    async def _submit_account_batch(
        self,
        account_id: str,
        orders: List[OrderRequest],
    ) -> List[OrderResult]:
        """Submit batch for single account.

        Args:
            account_id: Account ID
            orders: Orders for this account

        Returns:
            List of results
        """
        client = self.account_manager.get_client(account_id)
        if not client:
            return [
                OrderResult(
                    success=False,
                    client_id=o.client_id,
                    error=f"No client for account: {account_id}",
                    status=OrderStatus.FAILED,
                )
                for o in orders
            ]

        # Acquire single rate limit slot for batch
        await self.rate_limiter.acquire_order_slot()

        try:
            # Build batch params
            batch_orders = []
            for order in orders:
                params = {
                    "market": order.market,
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "size": str(order.size),
                }
                if order.price:
                    params["price"] = str(order.price)
                if order.client_id:
                    params["client_id"] = order.client_id
                batch_orders.append(params)

            # Submit batch
            response = await client.api_client.submit_orders_batch(batch_orders)

            # Parse responses
            results = []
            for i, order in enumerate(orders):
                if response and i < len(response):
                    r = response[i]
                    results.append(
                        OrderResult(
                            success=r.get("status") != "FAILED",
                            order_id=r.get("id"),
                            client_id=order.client_id,
                            filled_size=Decimal(str(r.get("filled_size", "0"))),
                            status=OrderStatus.FILLED
                            if r.get("status") == "FILLED"
                            else OrderStatus.OPEN,
                        )
                    )
                else:
                    results.append(
                        OrderResult(
                            success=False,
                            client_id=order.client_id,
                            error="No response for order",
                            status=OrderStatus.FAILED,
                        )
                    )

            logger.info(f"Batch submitted for {account_id}: {len(orders)} orders")
            return results

        except Exception as e:
            logger.error(f"Batch submission failed for {account_id}: {e}")
            return [
                OrderResult(
                    success=False,
                    client_id=o.client_id,
                    error=str(e),
                    status=OrderStatus.FAILED,
                )
                for o in orders
            ]

    async def cancel_order(
        self,
        account_id: str,
        order_id: str,
    ) -> bool:
        """Cancel specific order.

        Args:
            account_id: Account ID
            order_id: Order ID to cancel

        Returns:
            True if cancelled
        """
        client = self.account_manager.get_client(account_id)
        if not client:
            return False

        await self.rate_limiter.acquire_order_slot()

        try:
            await client.api_client.cancel_order(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(
        self,
        account_id: str,
        market: Optional[str] = None,
    ) -> int:
        """Cancel all orders for account.

        Args:
            account_id: Account ID
            market: Optional market filter

        Returns:
            Number of orders cancelled
        """
        client = self.account_manager.get_client(account_id)
        if not client:
            return 0

        await self.rate_limiter.acquire_order_slot()

        try:
            params = {}
            if market:
                params["market"] = market

            result = await client.api_client.cancel_all_orders(params)
            count = result.get("count", 0) if result else 0
            logger.info(f"Cancelled {count} orders for {account_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel orders for {account_id}: {e}")
            return 0

    async def get_open_orders(
        self,
        account_id: str,
        market: Optional[str] = None,
    ) -> List[dict]:
        """Get open orders for account.

        Args:
            account_id: Account ID
            market: Optional market filter

        Returns:
            List of open orders
        """
        client = self.account_manager.get_client(account_id)
        if not client:
            return []

        await self.rate_limiter.acquire_request_slot(account_id)

        try:
            params = {}
            if market:
                params["market"] = market

            orders = await client.api_client.fetch_orders(params)
            return orders or []
        except Exception as e:
            logger.error(f"Failed to fetch orders for {account_id}: {e}")
            return []

    def get_result(self, client_id: str) -> Optional[OrderResult]:
        """Get result for a submitted order.

        Args:
            client_id: Client order ID

        Returns:
            OrderResult if found
        """
        return self._order_results.get(client_id)

    def get_pending_count(self) -> int:
        """Get number of pending orders."""
        return len(self._pending_orders)

    def clear_results(self) -> None:
        """Clear stored order results."""
        self._order_results.clear()
