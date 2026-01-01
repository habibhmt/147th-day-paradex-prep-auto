"""Unit tests for Network module."""

import pytest
import asyncio
import time
from collections import deque

from src.network.rate_limiter import RateLimiter, RateLimitConfig


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_config(self):
        """Should have correct default values."""
        config = RateLimitConfig()

        assert config.orders_per_second == 800
        assert config.private_get_per_second == 120
        assert config.private_get_per_minute == 600
        assert config.requests_per_minute_ip == 1500
        assert config.ws_connections_per_second == 20

    def test_custom_config(self):
        """Should accept custom values."""
        config = RateLimitConfig(
            orders_per_second=400,
            private_get_per_second=60,
        )

        assert config.orders_per_second == 400
        assert config.private_get_per_second == 60


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter with low limits for testing."""
        config = RateLimitConfig(
            orders_per_second=10,
            private_get_per_second=5,
            private_get_per_minute=20,
            requests_per_minute_ip=50,
        )
        return RateLimiter(config=config)

    def test_initial_state(self, limiter):
        """Should start with empty counters."""
        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 0
        assert usage["requests_per_minute_ip"]["current"] == 0

    @pytest.mark.asyncio
    async def test_acquire_order_slot(self, limiter):
        """Should track order slots."""
        await limiter.acquire_order_slot()

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_order_slots(self, limiter):
        """Should track multiple order slots."""
        for _ in range(5):
            await limiter.acquire_order_slot()

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 5

    @pytest.mark.asyncio
    async def test_acquire_request_slot(self, limiter):
        """Should track request slots."""
        await limiter.acquire_request_slot("acc1")

        usage = limiter.get_current_usage()
        assert usage["requests_per_minute_ip"]["current"] == 1

    @pytest.mark.asyncio
    async def test_acquire_request_slot_per_account(self, limiter):
        """Should track per-account timestamps."""
        await limiter.acquire_request_slot("acc1")
        await limiter.acquire_request_slot("acc1")
        await limiter.acquire_request_slot("acc2")

        assert "acc1" in limiter._get_timestamps_per_account
        assert "acc2" in limiter._get_timestamps_per_account
        assert len(limiter._get_timestamps_per_account["acc1"]) == 2
        assert len(limiter._get_timestamps_per_account["acc2"]) == 1

    def test_should_use_batch_small(self, limiter):
        """Should not recommend batch for small counts."""
        assert limiter.should_use_batch(1) is False
        assert limiter.should_use_batch(2) is False

    def test_should_use_batch_large(self, limiter):
        """Should recommend batch for larger counts."""
        assert limiter.should_use_batch(3) is True
        assert limiter.should_use_batch(10) is True
        assert limiter.should_use_batch(50) is True

    def test_get_current_usage(self, limiter):
        """Should return usage dictionary."""
        usage = limiter.get_current_usage()

        assert "orders_per_second" in usage
        assert "requests_per_minute_ip" in usage
        assert "current" in usage["orders_per_second"]
        assert "limit" in usage["orders_per_second"]
        assert "usage_pct" in usage["orders_per_second"]

    def test_reset(self, limiter):
        """Should reset all counters."""
        # Add some timestamps manually
        limiter._order_timestamps.append(time.time())
        limiter._request_timestamps.append(time.time())
        limiter._get_timestamps_per_account["acc1"] = deque([time.time()])

        limiter.reset()

        assert len(limiter._order_timestamps) == 0
        assert len(limiter._request_timestamps) == 0
        assert len(limiter._get_timestamps_per_account) == 0

    @pytest.mark.asyncio
    async def test_wait_for_capacity(self, limiter):
        """Should acquire multiple slots."""
        await limiter.wait_for_capacity(3)

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 3

    def test_count_in_window(self, limiter):
        """Should count timestamps within window correctly."""
        now = time.time()
        timestamps = deque([
            now - 2.0,  # Outside 1s window
            now - 0.5,  # Inside 1s window
            now - 0.3,  # Inside 1s window
            now - 0.1,  # Inside 1s window
        ])

        count = limiter._count_in_window(timestamps, 1.0)
        assert count == 3

    def test_clean_old_timestamps(self, limiter):
        """Should remove old timestamps."""
        now = time.time()
        timestamps = deque([
            now - 5.0,  # Old
            now - 3.0,  # Old
            now - 0.5,  # Recent
            now - 0.1,  # Recent
        ])

        limiter._clean_old_timestamps(timestamps, 1.0)
        assert len(timestamps) == 2

    @pytest.mark.asyncio
    async def test_order_slot_tracking_with_requests(self, limiter):
        """Order slots should also count toward IP limit."""
        await limiter.acquire_order_slot()

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 1
        assert usage["requests_per_minute_ip"]["current"] == 1

    def test_usage_percentage_calculation(self, limiter):
        """Should calculate usage percentage correctly."""
        # Manually add 5 order timestamps (50% of limit of 10)
        now = time.time()
        for _ in range(5):
            limiter._order_timestamps.append(now)

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["usage_pct"] == 50.0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, limiter):
        """Should handle concurrent access safely."""
        async def acquire_slot():
            await limiter.acquire_order_slot()

        # Run 5 concurrent acquisitions
        await asyncio.gather(*[acquire_slot() for _ in range(5)])

        usage = limiter.get_current_usage()
        assert usage["orders_per_second"]["current"] == 5
