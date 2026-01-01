"""Rate limiting for Paradex API compliance."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration based on Paradex limits."""

    # Order endpoints: POST, DELETE, PUT /orders
    orders_per_second: int = 800

    # Private GET endpoints
    private_get_per_second: int = 120
    private_get_per_minute: int = 600

    # IP-based limit (applies to all accounts from same IP)
    requests_per_minute_ip: int = 1500

    # WebSocket limits
    ws_connections_per_second: int = 20
    ws_connections_per_minute: int = 600
    ws_message_backlog: int = 2000


@dataclass
class RateLimiter:
    """Manages API rate limits across all accounts.

    Paradex rate limits:
    - POST/DELETE/PUT orders: 800/sec (shared pool)
    - Private GET: 120/sec, 600/min per account
    - All requests: 1500/min per IP

    Strategy:
    - Use batch orders when possible (50 orders = 1 rate limit unit)
    - Track timestamps per category
    - Async wait when limit approached
    """

    config: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Timestamp tracking
    _order_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _request_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    _get_timestamps_per_account: dict = field(default_factory=dict)

    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._order_timestamps = deque(maxlen=1000)
        self._request_timestamps = deque(maxlen=2000)
        self._get_timestamps_per_account = {}
        self._lock = asyncio.Lock()

    def _clean_old_timestamps(
        self,
        timestamps: Deque[float],
        window_seconds: float,
    ) -> None:
        """Remove timestamps older than window."""
        cutoff = time.time() - window_seconds
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

    def _count_in_window(
        self,
        timestamps: Deque[float],
        window_seconds: float,
    ) -> int:
        """Count timestamps within window."""
        cutoff = time.time() - window_seconds
        return sum(1 for t in timestamps if t >= cutoff)

    async def acquire_order_slot(self) -> None:
        """Wait for available order slot.

        Orders share a pool of 800/sec across all accounts.
        """
        async with self._lock:
            self._clean_old_timestamps(self._order_timestamps, 1.0)

            while len(self._order_timestamps) >= self.config.orders_per_second:
                # Wait for oldest to expire
                oldest = self._order_timestamps[0]
                wait_time = oldest + 1.0 - time.time()
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.3f}s for order slot")
                    await asyncio.sleep(wait_time + 0.01)
                self._clean_old_timestamps(self._order_timestamps, 1.0)

            self._order_timestamps.append(time.time())
            self._request_timestamps.append(time.time())

    async def acquire_request_slot(self, account_id: str = "default") -> None:
        """Wait for available request slot.

        Args:
            account_id: Account making the request (for per-account limits)
        """
        async with self._lock:
            # Clean old timestamps
            self._clean_old_timestamps(self._request_timestamps, 60.0)

            # Check IP-based limit (1500/min)
            while self._count_in_window(self._request_timestamps, 60.0) >= self.config.requests_per_minute_ip:
                wait_time = 0.1
                logger.debug(f"Rate limit: IP limit reached, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                self._clean_old_timestamps(self._request_timestamps, 60.0)

            # Check per-account GET limits
            if account_id not in self._get_timestamps_per_account:
                self._get_timestamps_per_account[account_id] = deque(maxlen=700)

            account_timestamps = self._get_timestamps_per_account[account_id]
            self._clean_old_timestamps(account_timestamps, 60.0)

            # 120/sec limit
            count_1s = self._count_in_window(account_timestamps, 1.0)
            if count_1s >= self.config.private_get_per_second:
                await asyncio.sleep(0.05)

            # 600/min limit
            count_60s = self._count_in_window(account_timestamps, 60.0)
            if count_60s >= self.config.private_get_per_minute:
                await asyncio.sleep(0.5)

            account_timestamps.append(time.time())
            self._request_timestamps.append(time.time())

    def get_current_usage(self) -> dict:
        """Get current rate limit usage percentages.

        Returns:
            Dictionary with usage percentages
        """
        self._clean_old_timestamps(self._order_timestamps, 1.0)
        self._clean_old_timestamps(self._request_timestamps, 60.0)

        orders_1s = len(self._order_timestamps)
        requests_60s = self._count_in_window(self._request_timestamps, 60.0)

        return {
            "orders_per_second": {
                "current": orders_1s,
                "limit": self.config.orders_per_second,
                "usage_pct": (orders_1s / self.config.orders_per_second) * 100,
            },
            "requests_per_minute_ip": {
                "current": requests_60s,
                "limit": self.config.requests_per_minute_ip,
                "usage_pct": (requests_60s / self.config.requests_per_minute_ip) * 100,
            },
        }

    def should_use_batch(self, order_count: int) -> bool:
        """Recommend batch vs individual based on limits.

        A single batch counts as 1 rate limit unit regardless of order count.

        Args:
            order_count: Number of orders to submit

        Returns:
            True if batch recommended
        """
        # Always use batch for 3+ orders
        return order_count >= 3

    async def wait_for_capacity(self, required_slots: int = 1) -> None:
        """Wait until enough capacity is available.

        Args:
            required_slots: Number of slots needed
        """
        for _ in range(required_slots):
            await self.acquire_order_slot()

    def reset(self) -> None:
        """Reset all rate limit counters."""
        self._order_timestamps.clear()
        self._request_timestamps.clear()
        self._get_timestamps_per_account.clear()
        logger.info("Rate limiters reset")
