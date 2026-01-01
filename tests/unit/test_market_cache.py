"""Unit tests for Market data cache."""

import pytest
import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.network.market_cache import (
    CacheStrategy,
    CacheEntry,
    CacheConfig,
    CacheMetrics,
    PriceData,
    OrderbookData,
    FundingData,
    MarketCache,
    get_market_cache,
    reset_market_cache,
)


class TestCacheStrategy:
    """Tests for CacheStrategy enum."""

    def test_strategy_values(self):
        """Should have expected strategy values."""
        assert CacheStrategy.TTL.value == "ttl"
        assert CacheStrategy.LRU.value == "lru"
        assert CacheStrategy.LFU.value == "lfu"


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Should create cache entry."""
        entry = CacheEntry(key="test", value="data")

        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.access_count == 0
        assert entry.ttl == 60.0

    def test_entry_not_expired(self):
        """Should not be expired initially."""
        entry = CacheEntry(key="test", value="data", ttl=60.0)

        assert entry.is_expired is False

    def test_entry_expired(self):
        """Should detect expired entry."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time() - 100,
            ttl=60.0,
        )

        assert entry.is_expired is True

    def test_entry_age(self):
        """Should calculate age."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time() - 10,
        )

        assert entry.age >= 10

    def test_entry_access(self):
        """Should record access."""
        entry = CacheEntry(key="test", value="data")
        initial_count = entry.access_count

        result = entry.access()

        assert result == "data"
        assert entry.access_count == initial_count + 1


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = CacheConfig()

        assert config.max_entries == 1000
        assert config.default_ttl == 60.0
        assert config.strategy == CacheStrategy.TTL
        assert config.price_ttl == 1.0
        assert config.orderbook_ttl == 0.5
        assert config.funding_ttl == 60.0
        assert config.market_info_ttl == 300.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = CacheConfig(
            max_entries=500,
            strategy=CacheStrategy.LRU,
        )

        assert config.max_entries == 500
        assert config.strategy == CacheStrategy.LRU


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = CacheMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.expirations == 0

    def test_hit_rate_no_requests(self):
        """Should return 0% with no requests."""
        metrics = CacheMetrics()

        assert metrics.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Should calculate hit rate correctly."""
        metrics = CacheMetrics(hits=80, misses=20)

        assert metrics.hit_rate == 80.0

    def test_record_hit(self):
        """Should record hits."""
        metrics = CacheMetrics()

        metrics.record_hit()
        metrics.record_hit()

        assert metrics.hits == 2

    def test_record_miss(self):
        """Should record misses."""
        metrics = CacheMetrics()

        metrics.record_miss()

        assert metrics.misses == 1

    def test_record_eviction(self):
        """Should record evictions."""
        metrics = CacheMetrics()

        metrics.record_eviction()

        assert metrics.evictions == 1

    def test_record_expiration(self):
        """Should record expirations."""
        metrics = CacheMetrics()

        metrics.record_expiration()

        assert metrics.expirations == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = CacheMetrics(hits=10, misses=5)

        d = metrics.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["hit_rate"] == 66.67


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_create_price_data(self):
        """Should create price data."""
        data = PriceData(
            market="BTC-USD-PERP",
            price=Decimal("50000"),
            timestamp=time.time(),
        )

        assert data.market == "BTC-USD-PERP"
        assert data.price == Decimal("50000")

    def test_price_data_with_bid_ask(self):
        """Should include bid/ask."""
        data = PriceData(
            market="BTC-USD-PERP",
            price=Decimal("50000"),
            timestamp=time.time(),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
        )

        assert data.bid == Decimal("49990")
        assert data.ask == Decimal("50010")

    def test_to_dict(self):
        """Should convert to dictionary."""
        data = PriceData(
            market="BTC-USD-PERP",
            price=Decimal("50000"),
            timestamp=123456789.0,
        )

        d = data.to_dict()

        assert d["market"] == "BTC-USD-PERP"
        assert d["price"] == "50000"


class TestOrderbookData:
    """Tests for OrderbookData dataclass."""

    def test_create_orderbook(self):
        """Should create orderbook data."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[(Decimal("49990"), Decimal("1.5"))],
            asks=[(Decimal("50010"), Decimal("1.0"))],
            timestamp=time.time(),
        )

        assert data.market == "BTC-USD-PERP"

    def test_best_bid(self):
        """Should get best bid."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[
                (Decimal("49990"), Decimal("1.5")),
                (Decimal("49980"), Decimal("2.0")),
            ],
            asks=[],
            timestamp=time.time(),
        )

        assert data.best_bid == Decimal("49990")

    def test_best_ask(self):
        """Should get best ask."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[],
            asks=[
                (Decimal("50010"), Decimal("1.0")),
                (Decimal("50020"), Decimal("2.0")),
            ],
            timestamp=time.time(),
        )

        assert data.best_ask == Decimal("50010")

    def test_spread(self):
        """Should calculate spread."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[(Decimal("49990"), Decimal("1.5"))],
            asks=[(Decimal("50010"), Decimal("1.0"))],
            timestamp=time.time(),
        )

        assert data.spread == Decimal("20")

    def test_mid_price(self):
        """Should calculate mid price."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[(Decimal("49990"), Decimal("1.5"))],
            asks=[(Decimal("50010"), Decimal("1.0"))],
            timestamp=time.time(),
        )

        assert data.mid_price == Decimal("50000")

    def test_empty_orderbook(self):
        """Should handle empty orderbook."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[],
            asks=[],
            timestamp=time.time(),
        )

        assert data.best_bid is None
        assert data.best_ask is None
        assert data.spread is None
        assert data.mid_price is None

    def test_to_dict(self):
        """Should convert to dictionary."""
        data = OrderbookData(
            market="BTC-USD-PERP",
            bids=[(Decimal("49990"), Decimal("1.5"))],
            asks=[(Decimal("50010"), Decimal("1.0"))],
            timestamp=123456789.0,
        )

        d = data.to_dict()

        assert d["market"] == "BTC-USD-PERP"
        assert d["spread"] == "20"


class TestFundingData:
    """Tests for FundingData dataclass."""

    def test_create_funding_data(self):
        """Should create funding data."""
        data = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=time.time() + 3600,
        )

        assert data.market == "BTC-USD-PERP"
        assert data.funding_rate == Decimal("0.0001")

    def test_time_to_funding(self):
        """Should calculate time to funding."""
        future_time = time.time() + 3600
        data = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=future_time,
        )

        assert data.time_to_funding > 3500

    def test_time_to_funding_passed(self):
        """Should return 0 for passed funding time."""
        past_time = time.time() - 100
        data = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=past_time,
        )

        assert data.time_to_funding == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        data = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=time.time() + 3600,
        )

        d = data.to_dict()

        assert d["market"] == "BTC-USD-PERP"
        assert d["funding_rate"] == "0.0001"


class TestMarketCache:
    """Tests for MarketCache."""

    @pytest.fixture
    def cache(self):
        """Create fresh cache."""
        return MarketCache()

    @pytest.mark.asyncio
    async def test_create_cache(self, cache):
        """Should create cache."""
        assert len(cache._cache) == 0
        assert cache._cleanup_task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, cache):
        """Should start and stop cleanup task."""
        await cache.start()
        assert cache._cleanup_task is not None

        await cache.stop()
        assert cache._cleanup_task is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Should set and get values."""
        await cache.set("key1", "value1")

        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Should return None for missing key."""
        result = await cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_records_hit(self, cache):
        """Should record cache hit."""
        await cache.set("key1", "value1")

        await cache.get("key1")

        assert cache._metrics.hits == 1

    @pytest.mark.asyncio
    async def test_get_records_miss(self, cache):
        """Should record cache miss."""
        await cache.get("nonexistent")

        assert cache._metrics.misses == 1

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache):
        """Should set with custom TTL."""
        await cache.set("key1", "value1", ttl=30.0)

        entry = cache._cache.get("key1")
        assert entry.ttl == 30.0

    @pytest.mark.asyncio
    async def test_get_expired_entry(self, cache):
        """Should return None for expired entry."""
        cache._cache["expired"] = CacheEntry(
            key="expired",
            value="data",
            created_at=time.time() - 100,
            ttl=60.0,
        )

        result = await cache.get("expired")

        assert result is None
        assert cache._metrics.expirations == 1

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Should delete entry."""
        await cache.set("key1", "value1")

        result = await cache.delete("key1")

        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Should return False for nonexistent key."""
        result = await cache.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Should clear all entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        count = await cache.clear()

        assert count == 2
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_eviction_on_max_entries(self, cache):
        """Should evict when max entries reached."""
        cache.config.max_entries = 3

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # Should trigger eviction

        assert len(cache._cache) == 3
        assert cache._metrics.evictions == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Should evict least recently used."""
        cache = MarketCache(config=CacheConfig(
            max_entries=3,
            strategy=CacheStrategy.LRU,
        ))

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        await cache.set("key4", "value4")

        # key2 should be evicted (least recently used)
        assert await cache.get("key2") is None
        assert await cache.get("key1") is not None

    @pytest.mark.asyncio
    async def test_lfu_eviction(self):
        """Should evict least frequently used."""
        cache = MarketCache(config=CacheConfig(
            max_entries=3,
            strategy=CacheStrategy.LFU,
        ))

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 multiple times
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("key2")

        await cache.set("key4", "value4")

        # key3 should be evicted (least frequently used)
        assert await cache.get("key3") is None
        assert await cache.get("key1") is not None

    @pytest.mark.asyncio
    async def test_get_or_fetch_cached(self, cache):
        """Should return cached value."""
        await cache.set("key1", "cached_value")

        fetch_fn = AsyncMock(return_value="fetched_value")
        result = await cache.get_or_fetch("key1", fetch_fn)

        assert result == "cached_value"
        fetch_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_fetch_miss(self, cache):
        """Should fetch and cache on miss."""
        fetch_fn = AsyncMock(return_value="fetched_value")

        result = await cache.get_or_fetch("key1", fetch_fn)

        assert result == "fetched_value"
        fetch_fn.assert_called_once()

        # Should be cached now
        cached = await cache.get("key1")
        assert cached == "fetched_value"

    @pytest.mark.asyncio
    async def test_get_or_fetch_sync_function(self, cache):
        """Should work with sync fetch function."""
        def sync_fetch():
            return "sync_value"

        result = await cache.get_or_fetch("key1", sync_fetch)

        assert result == "sync_value"

    @pytest.mark.asyncio
    async def test_price_cache(self, cache):
        """Should cache price data."""
        price_data = PriceData(
            market="BTC-USD-PERP",
            price=Decimal("50000"),
            timestamp=time.time(),
        )

        await cache.set_price(price_data)
        result = await cache.get_price("BTC-USD-PERP")

        assert result == price_data

    @pytest.mark.asyncio
    async def test_orderbook_cache(self, cache):
        """Should cache orderbook data."""
        orderbook = OrderbookData(
            market="BTC-USD-PERP",
            bids=[(Decimal("49990"), Decimal("1.5"))],
            asks=[(Decimal("50010"), Decimal("1.0"))],
            timestamp=time.time(),
        )

        await cache.set_orderbook(orderbook)
        result = await cache.get_orderbook("BTC-USD-PERP")

        assert result == orderbook

    @pytest.mark.asyncio
    async def test_funding_cache(self, cache):
        """Should cache funding data."""
        funding = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=time.time() + 3600,
        )

        await cache.set_funding(funding)
        result = await cache.get_funding("BTC-USD-PERP")

        assert result == funding

    @pytest.mark.asyncio
    async def test_market_info_cache(self, cache):
        """Should cache market info."""
        info = {"symbol": "BTC-USD-PERP", "tick_size": "0.1"}

        await cache.set_market_info("BTC-USD-PERP", info)
        result = await cache.get_market_info("BTC-USD-PERP")

        assert result == info

    @pytest.mark.asyncio
    async def test_get_metrics(self, cache):
        """Should return metrics."""
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("nonexistent")

        metrics = cache.get_metrics()

        assert metrics.hits == 1
        assert metrics.misses == 1

    @pytest.mark.asyncio
    async def test_get_status(self, cache):
        """Should return status."""
        await cache.set("key1", "value1")

        status = cache.get_status()

        assert status["entries"] == 1
        assert status["max_entries"] == 1000
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_get_all_prices(self, cache):
        """Should get all cached prices."""
        price1 = PriceData(
            market="BTC-USD-PERP",
            price=Decimal("50000"),
            timestamp=time.time(),
        )
        price2 = PriceData(
            market="ETH-USD-PERP",
            price=Decimal("3000"),
            timestamp=time.time(),
        )

        await cache.set_price(price1)
        await cache.set_price(price2)

        all_prices = await cache.get_all_prices()

        assert len(all_prices) == 2
        assert "BTC-USD-PERP" in all_prices
        assert "ETH-USD-PERP" in all_prices

    @pytest.mark.asyncio
    async def test_get_all_funding(self, cache):
        """Should get all cached funding."""
        funding1 = FundingData(
            market="BTC-USD-PERP",
            funding_rate=Decimal("0.0001"),
            next_funding_time=time.time() + 3600,
        )
        funding2 = FundingData(
            market="ETH-USD-PERP",
            funding_rate=Decimal("0.0002"),
            next_funding_time=time.time() + 3600,
        )

        await cache.set_funding(funding1)
        await cache.set_funding(funding2)

        all_funding = await cache.get_all_funding()

        assert len(all_funding) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache):
        """Should cleanup expired entries."""
        # Add expired entry
        cache._cache["expired"] = CacheEntry(
            key="expired",
            value="data",
            created_at=time.time() - 100,
            ttl=60.0,
        )
        # Add valid entry
        await cache.set("valid", "data")

        count = await cache._cleanup_expired()

        assert count == 1
        assert "expired" not in cache._cache
        assert "valid" in cache._cache


class TestGlobalCache:
    """Tests for global cache functions."""

    def setup_method(self):
        """Reset global cache before each test."""
        reset_market_cache()

    def test_get_market_cache_creates_singleton(self):
        """Should create singleton cache."""
        cache1 = get_market_cache()
        cache2 = get_market_cache()

        assert cache1 is cache2

    def test_reset_market_cache(self):
        """Should reset global cache."""
        cache1 = get_market_cache()
        reset_market_cache()
        cache2 = get_market_cache()

        assert cache1 is not cache2
