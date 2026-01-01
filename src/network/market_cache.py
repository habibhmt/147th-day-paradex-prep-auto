"""Market data caching for optimized API usage."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache invalidation strategies."""

    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used


@dataclass
class CacheEntry:
    """A single cache entry."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 60.0  # Default 60 seconds

    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_accessed == 0:
            self.last_accessed = self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    def access(self) -> Any:
        """Record access and return value."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


@dataclass
class CacheConfig:
    """Configuration for market cache."""

    max_entries: int = 1000
    default_ttl: float = 60.0  # 60 seconds
    strategy: CacheStrategy = CacheStrategy.TTL
    cleanup_interval: float = 30.0
    price_ttl: float = 1.0  # 1 second for prices
    orderbook_ttl: float = 0.5  # 500ms for orderbook
    funding_ttl: float = 60.0  # 60 seconds for funding
    market_info_ttl: float = 300.0  # 5 minutes for market info


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def record_hit(self) -> None:
        """Record cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1

    def record_eviction(self) -> None:
        """Record cache eviction."""
        self.evictions += 1

    def record_expiration(self) -> None:
        """Record expiration."""
        self.expirations += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": round(self.hit_rate, 2),
            "total_entries": self.total_entries,
        }


@dataclass
class PriceData:
    """Cached price data."""

    market: str
    price: Decimal
    timestamp: float
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "price": str(self.price),
            "timestamp": self.timestamp,
            "bid": str(self.bid) if self.bid else None,
            "ask": str(self.ask) if self.ask else None,
            "volume_24h": str(self.volume_24h) if self.volume_24h else None,
        }


@dataclass
class OrderbookData:
    """Cached orderbook data."""

    market: str
    bids: List[Tuple[Decimal, Decimal]]  # (price, size)
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: float

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "bids": [(str(p), str(s)) for p, s in self.bids[:5]],
            "asks": [(str(p), str(s)) for p, s in self.asks[:5]],
            "best_bid": str(self.best_bid) if self.best_bid else None,
            "best_ask": str(self.best_ask) if self.best_ask else None,
            "spread": str(self.spread) if self.spread else None,
            "mid_price": str(self.mid_price) if self.mid_price else None,
            "timestamp": self.timestamp,
        }


@dataclass
class FundingData:
    """Cached funding rate data."""

    market: str
    funding_rate: Decimal
    next_funding_time: float
    predicted_rate: Optional[Decimal] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def time_to_funding(self) -> float:
        """Seconds until next funding."""
        return max(0, self.next_funding_time - time.time())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "funding_rate": str(self.funding_rate),
            "next_funding_time": self.next_funding_time,
            "predicted_rate": str(self.predicted_rate) if self.predicted_rate else None,
            "time_to_funding": self.time_to_funding,
            "timestamp": self.timestamp,
        }


@dataclass
class MarketCache:
    """In-memory cache for market data.

    Features:
    - Multiple TTL support for different data types
    - LRU/LFU eviction strategies
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Metrics tracking
    """

    config: CacheConfig = field(default_factory=CacheConfig)
    _cache: Dict[str, CacheEntry] = field(default_factory=dict)
    _metrics: CacheMetrics = field(default_factory=CacheMetrics)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cleanup_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._cache = {}
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._cleanup_task = None

    async def start(self) -> None:
        """Start cache cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Market cache started")

    async def stop(self) -> None:
        """Stop cache cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Market cache stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self._metrics.record_expiration()

            self._metrics.total_entries = len(self._cache)
            return len(expired_keys)

    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key from prefix and arguments."""
        return f"{prefix}:{':'.join(str(a) for a in args)}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._metrics.record_miss()
                return None

            if entry.is_expired:
                del self._cache[key]
                self._metrics.record_expiration()
                self._metrics.record_miss()
                return None

            self._metrics.record_hit()
            return entry.access()

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            if len(self._cache) >= self.config.max_entries:
                await self._evict_one()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.config.default_ttl,
            )
            self._metrics.total_entries = len(self._cache)

    async def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return

        if self.config.strategy == CacheStrategy.LRU:
            key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
        elif self.config.strategy == CacheStrategy.LFU:
            key = min(self._cache, key=lambda k: self._cache[k].access_count)
        else:  # TTL - evict oldest
            key = min(self._cache, key=lambda k: self._cache[k].created_at)

        del self._cache[key]
        self._metrics.record_eviction()

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics.total_entries = len(self._cache)
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._metrics.total_entries = 0
            return count

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable,
        ttl: Optional[float] = None,
    ) -> Any:
        """Get from cache or fetch and cache."""
        value = await self.get(key)
        if value is not None:
            return value

        # Fetch new value
        if asyncio.iscoroutinefunction(fetch_fn):
            value = await fetch_fn()
        else:
            value = fetch_fn()

        await self.set(key, value, ttl)
        return value

    # Price cache methods
    async def get_price(self, market: str) -> Optional[PriceData]:
        """Get cached price data."""
        key = self._make_key("price", market)
        return await self.get(key)

    async def set_price(self, data: PriceData) -> None:
        """Cache price data."""
        key = self._make_key("price", data.market)
        await self.set(key, data, self.config.price_ttl)

    # Orderbook cache methods
    async def get_orderbook(self, market: str) -> Optional[OrderbookData]:
        """Get cached orderbook data."""
        key = self._make_key("orderbook", market)
        return await self.get(key)

    async def set_orderbook(self, data: OrderbookData) -> None:
        """Cache orderbook data."""
        key = self._make_key("orderbook", data.market)
        await self.set(key, data, self.config.orderbook_ttl)

    # Funding cache methods
    async def get_funding(self, market: str) -> Optional[FundingData]:
        """Get cached funding data."""
        key = self._make_key("funding", market)
        return await self.get(key)

    async def set_funding(self, data: FundingData) -> None:
        """Cache funding data."""
        key = self._make_key("funding", data.market)
        await self.set(key, data, self.config.funding_ttl)

    # Market info cache methods
    async def get_market_info(self, market: str) -> Optional[dict]:
        """Get cached market info."""
        key = self._make_key("market_info", market)
        return await self.get(key)

    async def set_market_info(self, market: str, info: dict) -> None:
        """Cache market info."""
        key = self._make_key("market_info", market)
        await self.set(key, info, self.config.market_info_ttl)

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._metrics

    def get_status(self) -> dict:
        """Get cache status."""
        return {
            "entries": len(self._cache),
            "max_entries": self.config.max_entries,
            "strategy": self.config.strategy.value,
            "running": self._cleanup_task is not None,
            "metrics": self._metrics.to_dict(),
        }

    async def get_all_prices(self) -> Dict[str, PriceData]:
        """Get all cached prices."""
        async with self._lock:
            prices = {}
            prefix = "price:"
            for key, entry in self._cache.items():
                if key.startswith(prefix) and not entry.is_expired:
                    market = key[len(prefix):]
                    prices[market] = entry.value
            return prices

    async def get_all_funding(self) -> Dict[str, FundingData]:
        """Get all cached funding rates."""
        async with self._lock:
            funding = {}
            prefix = "funding:"
            for key, entry in self._cache.items():
                if key.startswith(prefix) and not entry.is_expired:
                    market = key[len(prefix):]
                    funding[market] = entry.value
            return funding


# Global cache instance
_global_cache: Optional[MarketCache] = None


def get_market_cache() -> MarketCache:
    """Get or create global market cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketCache()
    return _global_cache


def reset_market_cache() -> None:
    """Reset global market cache."""
    global _global_cache
    _global_cache = None
