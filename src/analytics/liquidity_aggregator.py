"""
Liquidity Aggregator Module.

Aggregates liquidity from multiple sources and markets,
calculates available liquidity at different price levels,
and provides insights for optimal order execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable


class LiquiditySource(Enum):
    """Source of liquidity."""

    ORDER_BOOK = "order_book"
    MARKET_MAKER = "market_maker"
    DARK_POOL = "dark_pool"
    INTERNAL = "internal"
    EXTERNAL = "external"


class LiquidityType(Enum):
    """Type of liquidity."""

    BID = "bid"
    ASK = "ask"
    BOTH = "both"


class AggregationMode(Enum):
    """Mode for aggregating liquidity."""

    SUM = "sum"  # Sum all liquidity
    WEIGHTED = "weighted"  # Weight by priority/quality
    BEST_PRICE = "best_price"  # Only best prices
    DEPTH_BASED = "depth_based"  # Consider depth levels


class LiquidityQuality(Enum):
    """Quality rating of liquidity."""

    HIGH = "high"  # Firm, reliable
    MEDIUM = "medium"  # Generally available
    LOW = "low"  # May be stale or unreliable
    INDICATIVE = "indicative"  # Not firm quotes


@dataclass
class LiquidityLevel:
    """Single level of liquidity at a price."""

    price: Decimal
    size: Decimal
    source: LiquiditySource
    quality: LiquidityQuality = LiquidityQuality.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    order_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": float(self.price),
            "size": float(self.size),
            "source": self.source.value,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "order_count": self.order_count,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedLevel:
    """Aggregated liquidity at a price point."""

    price: Decimal
    total_size: Decimal
    sources: list[LiquiditySource] = field(default_factory=list)
    weighted_quality: float = 0.5
    order_count: int = 0
    levels: list[LiquidityLevel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": float(self.price),
            "total_size": float(self.total_size),
            "sources": [s.value for s in self.sources],
            "weighted_quality": self.weighted_quality,
            "order_count": self.order_count,
            "levels": [l.to_dict() for l in self.levels],
        }


@dataclass
class LiquiditySnapshot:
    """Snapshot of liquidity for a market."""

    market: str
    timestamp: datetime
    bids: list[AggregatedLevel]
    asks: list[AggregatedLevel]
    total_bid_liquidity: Decimal = Decimal("0")
    total_ask_liquidity: Decimal = Decimal("0")
    spread_bps: float = 0.0
    imbalance_ratio: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp.isoformat(),
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
            "total_bid_liquidity": float(self.total_bid_liquidity),
            "total_ask_liquidity": float(self.total_ask_liquidity),
            "spread_bps": self.spread_bps,
            "imbalance_ratio": self.imbalance_ratio,
        }


@dataclass
class LiquidityMetrics:
    """Metrics about aggregated liquidity."""

    market: str
    timestamp: datetime
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    spread_bps: float
    bid_depth_1pct: Decimal  # Liquidity within 1% of mid
    ask_depth_1pct: Decimal
    bid_depth_5pct: Decimal  # Liquidity within 5% of mid
    ask_depth_5pct: Decimal
    total_bid_liquidity: Decimal
    total_ask_liquidity: Decimal
    imbalance_ratio: float  # bid / (bid + ask)
    bid_levels: int
    ask_levels: int
    avg_bid_quality: float
    avg_ask_quality: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp.isoformat(),
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "mid_price": float(self.mid_price),
            "spread_bps": self.spread_bps,
            "bid_depth_1pct": float(self.bid_depth_1pct),
            "ask_depth_1pct": float(self.ask_depth_1pct),
            "bid_depth_5pct": float(self.bid_depth_5pct),
            "ask_depth_5pct": float(self.ask_depth_5pct),
            "total_bid_liquidity": float(self.total_bid_liquidity),
            "total_ask_liquidity": float(self.total_ask_liquidity),
            "imbalance_ratio": self.imbalance_ratio,
            "bid_levels": self.bid_levels,
            "ask_levels": self.ask_levels,
            "avg_bid_quality": self.avg_bid_quality,
            "avg_ask_quality": self.avg_ask_quality,
        }


@dataclass
class FillEstimate:
    """Estimate for filling an order."""

    market: str
    side: LiquidityType
    size: Decimal
    avg_price: Decimal
    worst_price: Decimal
    slippage_bps: float
    levels_consumed: int
    unfilled_size: Decimal = Decimal("0")
    fills: list[tuple[Decimal, Decimal]] = field(default_factory=list)  # (price, size)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "side": self.side.value,
            "size": float(self.size),
            "avg_price": float(self.avg_price),
            "worst_price": float(self.worst_price),
            "slippage_bps": self.slippage_bps,
            "levels_consumed": self.levels_consumed,
            "unfilled_size": float(self.unfilled_size),
            "fills": [(float(p), float(s)) for p, s in self.fills],
        }


@dataclass
class LiquidityDepth:
    """Depth of liquidity at various price levels."""

    market: str
    timestamp: datetime
    price_levels: list[Decimal]  # Price points
    bid_cumulative: list[Decimal]  # Cumulative bid liquidity at each price
    ask_cumulative: list[Decimal]  # Cumulative ask liquidity at each price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp.isoformat(),
            "price_levels": [float(p) for p in self.price_levels],
            "bid_cumulative": [float(b) for b in self.bid_cumulative],
            "ask_cumulative": [float(a) for a in self.ask_cumulative],
        }


class LiquidityAggregator:
    """Aggregates liquidity from multiple sources."""

    def __init__(
        self,
        max_levels: int = 50,
        aggregation_mode: AggregationMode = AggregationMode.SUM,
        stale_threshold_ms: int = 5000,
    ):
        """
        Initialize aggregator.

        Args:
            max_levels: Maximum price levels to track
            aggregation_mode: How to aggregate liquidity
            stale_threshold_ms: Time after which data is considered stale
        """
        self.max_levels = max_levels
        self.aggregation_mode = aggregation_mode
        self.stale_threshold_ms = stale_threshold_ms

        # Market -> source -> levels
        self._bid_levels: dict[str, dict[LiquiditySource, list[LiquidityLevel]]] = {}
        self._ask_levels: dict[str, dict[LiquiditySource, list[LiquidityLevel]]] = {}

        # Source priorities (higher = better)
        self._source_priorities: dict[LiquiditySource, int] = {
            LiquiditySource.ORDER_BOOK: 10,
            LiquiditySource.MARKET_MAKER: 8,
            LiquiditySource.INTERNAL: 6,
            LiquiditySource.DARK_POOL: 4,
            LiquiditySource.EXTERNAL: 2,
        }

        # Quality weights
        self._quality_weights: dict[LiquidityQuality, float] = {
            LiquidityQuality.HIGH: 1.0,
            LiquidityQuality.MEDIUM: 0.7,
            LiquidityQuality.LOW: 0.4,
            LiquidityQuality.INDICATIVE: 0.1,
        }

        # Callbacks
        self._callbacks: list[Callable[[str, LiquiditySnapshot], None]] = []

        # Cache
        self._snapshot_cache: dict[str, LiquiditySnapshot] = {}
        self._metrics_cache: dict[str, LiquidityMetrics] = {}

    def add_liquidity(
        self,
        market: str,
        levels: list[LiquidityLevel],
        liquidity_type: LiquidityType,
    ) -> None:
        """
        Add liquidity levels for a market.

        Args:
            market: Market symbol
            levels: Liquidity levels to add
            liquidity_type: Bid or ask
        """
        if liquidity_type == LiquidityType.BID or liquidity_type == LiquidityType.BOTH:
            self._add_to_book(market, levels, self._bid_levels)

        if liquidity_type == LiquidityType.ASK or liquidity_type == LiquidityType.BOTH:
            self._add_to_book(market, levels, self._ask_levels)

        # Invalidate cache
        self._snapshot_cache.pop(market, None)
        self._metrics_cache.pop(market, None)

    def _add_to_book(
        self,
        market: str,
        levels: list[LiquidityLevel],
        book: dict[str, dict[LiquiditySource, list[LiquidityLevel]]],
    ) -> None:
        """Add levels to order book."""
        if market not in book:
            book[market] = {}

        for level in levels:
            if level.source not in book[market]:
                book[market][level.source] = []

            # Replace existing level at same price from same source
            existing = book[market][level.source]
            updated = [l for l in existing if l.price != level.price]
            updated.append(level)

            # Sort and trim
            updated.sort(key=lambda x: x.price, reverse=True)  # Descending for bids
            book[market][level.source] = updated[:self.max_levels]

    def update_order_book(
        self,
        market: str,
        bids: list[tuple[Decimal, Decimal]],  # (price, size)
        asks: list[tuple[Decimal, Decimal]],
        source: LiquiditySource = LiquiditySource.ORDER_BOOK,
        quality: LiquidityQuality = LiquidityQuality.HIGH,
    ) -> None:
        """
        Update full order book for a market.

        Args:
            market: Market symbol
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            source: Source of liquidity
            quality: Quality rating
        """
        now = datetime.now()

        bid_levels = [
            LiquidityLevel(price=p, size=s, source=source, quality=quality, timestamp=now)
            for p, s in bids
        ]
        ask_levels = [
            LiquidityLevel(price=p, size=s, source=source, quality=quality, timestamp=now)
            for p, s in asks
        ]

        # Clear existing and add new
        if market in self._bid_levels:
            self._bid_levels[market].pop(source, None)
        if market in self._ask_levels:
            self._ask_levels[market].pop(source, None)

        self.add_liquidity(market, bid_levels, LiquidityType.BID)
        self.add_liquidity(market, ask_levels, LiquidityType.ASK)

    def get_snapshot(self, market: str) -> LiquiditySnapshot | None:
        """
        Get aggregated liquidity snapshot for a market.

        Args:
            market: Market symbol

        Returns:
            Liquidity snapshot or None
        """
        if market in self._snapshot_cache:
            return self._snapshot_cache[market]

        if market not in self._bid_levels and market not in self._ask_levels:
            return None

        # Aggregate bids
        agg_bids = self._aggregate_levels(
            market, self._bid_levels.get(market, {}), is_bid=True
        )

        # Aggregate asks
        agg_asks = self._aggregate_levels(
            market, self._ask_levels.get(market, {}), is_bid=False
        )

        # Calculate totals
        total_bid = sum(level.total_size for level in agg_bids)
        total_ask = sum(level.total_size for level in agg_asks)

        # Calculate spread
        spread_bps = 0.0
        if agg_bids and agg_asks:
            best_bid = agg_bids[0].price
            best_ask = agg_asks[0].price
            mid = (best_bid + best_ask) / 2
            if mid > 0:
                spread_bps = float((best_ask - best_bid) / mid * 10000)

        # Calculate imbalance
        total = total_bid + total_ask
        imbalance = float(total_bid / total) if total > 0 else 0.5

        snapshot = LiquiditySnapshot(
            market=market,
            timestamp=datetime.now(),
            bids=agg_bids,
            asks=agg_asks,
            total_bid_liquidity=total_bid,
            total_ask_liquidity=total_ask,
            spread_bps=spread_bps,
            imbalance_ratio=imbalance,
        )

        self._snapshot_cache[market] = snapshot

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(market, snapshot)
            except Exception:
                pass

        return snapshot

    def _aggregate_levels(
        self,
        market: str,
        sources: dict[LiquiditySource, list[LiquidityLevel]],
        is_bid: bool,
    ) -> list[AggregatedLevel]:
        """Aggregate levels from multiple sources."""
        # Collect all levels
        all_levels: dict[Decimal, list[LiquidityLevel]] = {}

        for source, levels in sources.items():
            for level in levels:
                if level.price not in all_levels:
                    all_levels[level.price] = []
                all_levels[level.price].append(level)

        # Aggregate by price
        aggregated = []
        for price, levels in all_levels.items():
            agg = self._aggregate_at_price(price, levels)
            aggregated.append(agg)

        # Sort (descending for bids, ascending for asks)
        aggregated.sort(key=lambda x: x.price, reverse=is_bid)

        return aggregated[:self.max_levels]

    def _aggregate_at_price(
        self,
        price: Decimal,
        levels: list[LiquidityLevel],
    ) -> AggregatedLevel:
        """Aggregate multiple levels at same price."""
        if self.aggregation_mode == AggregationMode.SUM:
            total_size = sum(l.size for l in levels)
        elif self.aggregation_mode == AggregationMode.WEIGHTED:
            # Weight by source priority and quality
            total_size = Decimal("0")
            for level in levels:
                priority = self._source_priorities.get(level.source, 1)
                quality = self._quality_weights.get(level.quality, 0.5)
                weight = priority * quality / 10  # Normalize
                total_size += level.size * Decimal(str(weight))
        elif self.aggregation_mode == AggregationMode.BEST_PRICE:
            # Only use best quality level
            best = max(levels, key=lambda l: self._quality_weights.get(l.quality, 0))
            total_size = best.size
        else:  # DEPTH_BASED
            total_size = sum(l.size for l in levels)

        # Calculate weighted quality
        total_weight = sum(self._quality_weights.get(l.quality, 0.5) for l in levels)
        avg_quality = total_weight / len(levels) if levels else 0.5

        # Collect sources
        sources = list(set(l.source for l in levels))

        # Count orders
        order_count = sum(l.order_count for l in levels)

        return AggregatedLevel(
            price=price,
            total_size=total_size,
            sources=sources,
            weighted_quality=avg_quality,
            order_count=order_count,
            levels=levels,
        )

    def get_metrics(self, market: str) -> LiquidityMetrics | None:
        """
        Get liquidity metrics for a market.

        Args:
            market: Market symbol

        Returns:
            Liquidity metrics or None
        """
        if market in self._metrics_cache:
            return self._metrics_cache[market]

        snapshot = self.get_snapshot(market)
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return None

        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        mid_price = (best_bid + best_ask) / 2

        # Calculate depth at various levels
        bid_depth_1pct = self._calculate_depth(
            snapshot.bids, mid_price, Decimal("0.01"), is_bid=True
        )
        ask_depth_1pct = self._calculate_depth(
            snapshot.asks, mid_price, Decimal("0.01"), is_bid=False
        )
        bid_depth_5pct = self._calculate_depth(
            snapshot.bids, mid_price, Decimal("0.05"), is_bid=True
        )
        ask_depth_5pct = self._calculate_depth(
            snapshot.asks, mid_price, Decimal("0.05"), is_bid=False
        )

        # Calculate average quality
        bid_qualities = [l.weighted_quality for l in snapshot.bids]
        ask_qualities = [l.weighted_quality for l in snapshot.asks]
        avg_bid_quality = sum(bid_qualities) / len(bid_qualities) if bid_qualities else 0.5
        avg_ask_quality = sum(ask_qualities) / len(ask_qualities) if ask_qualities else 0.5

        metrics = LiquidityMetrics(
            market=market,
            timestamp=snapshot.timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread_bps=snapshot.spread_bps,
            bid_depth_1pct=bid_depth_1pct,
            ask_depth_1pct=ask_depth_1pct,
            bid_depth_5pct=bid_depth_5pct,
            ask_depth_5pct=ask_depth_5pct,
            total_bid_liquidity=snapshot.total_bid_liquidity,
            total_ask_liquidity=snapshot.total_ask_liquidity,
            imbalance_ratio=snapshot.imbalance_ratio,
            bid_levels=len(snapshot.bids),
            ask_levels=len(snapshot.asks),
            avg_bid_quality=avg_bid_quality,
            avg_ask_quality=avg_ask_quality,
        )

        self._metrics_cache[market] = metrics
        return metrics

    def _calculate_depth(
        self,
        levels: list[AggregatedLevel],
        mid_price: Decimal,
        pct: Decimal,
        is_bid: bool,
    ) -> Decimal:
        """Calculate cumulative depth within percentage of mid price."""
        total = Decimal("0")

        if is_bid:
            threshold = mid_price * (1 - pct)
            for level in levels:
                if level.price >= threshold:
                    total += level.total_size
        else:
            threshold = mid_price * (1 + pct)
            for level in levels:
                if level.price <= threshold:
                    total += level.total_size

        return total

    def estimate_fill(
        self,
        market: str,
        side: LiquidityType,
        size: Decimal,
    ) -> FillEstimate | None:
        """
        Estimate how an order would fill.

        Args:
            market: Market symbol
            side: Buy or sell
            size: Order size

        Returns:
            Fill estimate or None
        """
        snapshot = self.get_snapshot(market)
        if not snapshot:
            return None

        # Get relevant side
        if side == LiquidityType.BID:
            levels = snapshot.asks  # Buying from asks
        else:
            levels = snapshot.bids  # Selling to bids

        if not levels:
            return None

        # Simulate fill
        remaining = size
        fills: list[tuple[Decimal, Decimal]] = []
        total_value = Decimal("0")
        levels_consumed = 0
        first_price = levels[0].price
        worst_price = first_price

        for level in levels:
            if remaining <= 0:
                break

            fill_size = min(remaining, level.total_size)
            fills.append((level.price, fill_size))
            total_value += level.price * fill_size
            remaining -= fill_size
            levels_consumed += 1
            worst_price = level.price

        filled_size = size - remaining
        avg_price = total_value / filled_size if filled_size > 0 else first_price

        # Calculate slippage from first price
        if first_price > 0:
            slippage_bps = float(abs(avg_price - first_price) / first_price * 10000)
        else:
            slippage_bps = 0.0

        return FillEstimate(
            market=market,
            side=side,
            size=size,
            avg_price=avg_price,
            worst_price=worst_price,
            slippage_bps=slippage_bps,
            levels_consumed=levels_consumed,
            unfilled_size=remaining,
            fills=fills,
        )

    def get_depth(
        self,
        market: str,
        num_levels: int = 10,
        tick_size: Decimal | None = None,
    ) -> LiquidityDepth | None:
        """
        Get liquidity depth profile.

        Args:
            market: Market symbol
            num_levels: Number of price levels to return
            tick_size: Price granularity (optional)

        Returns:
            Depth profile or None
        """
        snapshot = self.get_snapshot(market)
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return None

        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        mid = (best_bid + best_ask) / 2

        # Generate price levels around mid
        if tick_size is None:
            tick_size = mid * Decimal("0.001")  # 0.1% default

        price_levels = []
        bid_cumulative = []
        ask_cumulative = []

        for i in range(-num_levels, num_levels + 1):
            price = mid + tick_size * i
            price_levels.append(price)

            # Cumulative bid liquidity at or above this price
            bid_cum = sum(
                l.total_size for l in snapshot.bids
                if l.price >= price
            )
            bid_cumulative.append(bid_cum)

            # Cumulative ask liquidity at or below this price
            ask_cum = sum(
                l.total_size for l in snapshot.asks
                if l.price <= price
            )
            ask_cumulative.append(ask_cum)

        return LiquidityDepth(
            market=market,
            timestamp=datetime.now(),
            price_levels=price_levels,
            bid_cumulative=bid_cumulative,
            ask_cumulative=ask_cumulative,
        )

    def get_best_prices(self, market: str) -> tuple[Decimal | None, Decimal | None]:
        """
        Get best bid and ask prices.

        Args:
            market: Market symbol

        Returns:
            Tuple of (best_bid, best_ask)
        """
        snapshot = self.get_snapshot(market)
        if not snapshot:
            return None, None

        best_bid = snapshot.bids[0].price if snapshot.bids else None
        best_ask = snapshot.asks[0].price if snapshot.asks else None

        return best_bid, best_ask

    def get_mid_price(self, market: str) -> Decimal | None:
        """
        Get mid price for a market.

        Args:
            market: Market symbol

        Returns:
            Mid price or None
        """
        best_bid, best_ask = self.get_best_prices(market)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def get_spread(self, market: str) -> tuple[Decimal | None, float | None]:
        """
        Get spread for a market.

        Args:
            market: Market symbol

        Returns:
            Tuple of (spread_abs, spread_bps)
        """
        best_bid, best_ask = self.get_best_prices(market)
        if best_bid is None or best_ask is None:
            return None, None

        spread_abs = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        spread_bps = float(spread_abs / mid * 10000) if mid > 0 else None

        return spread_abs, spread_bps

    def get_imbalance(self, market: str, depth_pct: Decimal = Decimal("0.01")) -> float | None:
        """
        Get order book imbalance.

        Args:
            market: Market symbol
            depth_pct: Percentage of mid price to consider

        Returns:
            Imbalance ratio (0.5 = balanced, >0.5 = more bids)
        """
        snapshot = self.get_snapshot(market)
        if not snapshot:
            return None

        mid = self.get_mid_price(market)
        if mid is None:
            return None

        bid_depth = self._calculate_depth(snapshot.bids, mid, depth_pct, is_bid=True)
        ask_depth = self._calculate_depth(snapshot.asks, mid, depth_pct, is_bid=False)

        total = bid_depth + ask_depth
        if total > 0:
            return float(bid_depth / total)
        return 0.5

    def set_source_priority(self, source: LiquiditySource, priority: int) -> None:
        """Set priority for a liquidity source."""
        self._source_priorities[source] = priority

    def get_source_priority(self, source: LiquiditySource) -> int:
        """Get priority for a liquidity source."""
        return self._source_priorities.get(source, 0)

    def set_quality_weight(self, quality: LiquidityQuality, weight: float) -> None:
        """Set weight for a quality rating."""
        self._quality_weights[quality] = weight

    def get_quality_weight(self, quality: LiquidityQuality) -> float:
        """Get weight for a quality rating."""
        return self._quality_weights.get(quality, 0.5)

    def add_callback(self, callback: Callable[[str, LiquiditySnapshot], None]) -> None:
        """Add callback for snapshot updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, LiquiditySnapshot], None]) -> bool:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def get_markets(self) -> list[str]:
        """Get list of markets with liquidity data."""
        markets = set(self._bid_levels.keys()) | set(self._ask_levels.keys())
        return list(markets)

    def clear_market(self, market: str) -> None:
        """Clear liquidity data for a market."""
        self._bid_levels.pop(market, None)
        self._ask_levels.pop(market, None)
        self._snapshot_cache.pop(market, None)
        self._metrics_cache.pop(market, None)

    def clear_all(self) -> None:
        """Clear all liquidity data."""
        self._bid_levels.clear()
        self._ask_levels.clear()
        self._snapshot_cache.clear()
        self._metrics_cache.clear()

    def clear_source(self, market: str, source: LiquiditySource) -> None:
        """Clear liquidity data from specific source."""
        if market in self._bid_levels:
            self._bid_levels[market].pop(source, None)
        if market in self._ask_levels:
            self._ask_levels[market].pop(source, None)
        self._snapshot_cache.pop(market, None)
        self._metrics_cache.pop(market, None)


class MultiMarketAggregator:
    """Aggregates liquidity across multiple markets."""

    def __init__(self, aggregator: LiquidityAggregator | None = None):
        """
        Initialize multi-market aggregator.

        Args:
            aggregator: Base aggregator (creates new if None)
        """
        self.aggregator = aggregator or LiquidityAggregator()
        self._cross_market_callbacks: list[Callable[[dict[str, LiquiditySnapshot]], None]] = []

    def update_all_markets(
        self,
        market_data: dict[str, tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]],
    ) -> dict[str, LiquiditySnapshot]:
        """
        Update all markets at once.

        Args:
            market_data: Dict of market -> (bids, asks)

        Returns:
            Dict of market -> snapshot
        """
        snapshots = {}

        for market, (bids, asks) in market_data.items():
            self.aggregator.update_order_book(market, bids, asks)
            snapshot = self.aggregator.get_snapshot(market)
            if snapshot:
                snapshots[market] = snapshot

        # Notify cross-market callbacks
        for callback in self._cross_market_callbacks:
            try:
                callback(snapshots)
            except Exception:
                pass

        return snapshots

    def get_all_metrics(self) -> dict[str, LiquidityMetrics]:
        """Get metrics for all markets."""
        metrics = {}
        for market in self.aggregator.get_markets():
            m = self.aggregator.get_metrics(market)
            if m:
                metrics[market] = m
        return metrics

    def get_best_liquidity_market(
        self,
        markets: list[str] | None = None,
        side: LiquidityType = LiquidityType.BOTH,
    ) -> str | None:
        """
        Find market with best liquidity.

        Args:
            markets: Markets to consider (all if None)
            side: Side to evaluate

        Returns:
            Best market or None
        """
        if markets is None:
            markets = self.aggregator.get_markets()

        best_market = None
        best_liquidity = Decimal("0")

        for market in markets:
            metrics = self.aggregator.get_metrics(market)
            if not metrics:
                continue

            if side == LiquidityType.BID:
                liquidity = metrics.total_bid_liquidity
            elif side == LiquidityType.ASK:
                liquidity = metrics.total_ask_liquidity
            else:
                liquidity = metrics.total_bid_liquidity + metrics.total_ask_liquidity

            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_market = market

        return best_market

    def get_tightest_spread_market(
        self,
        markets: list[str] | None = None,
    ) -> str | None:
        """
        Find market with tightest spread.

        Args:
            markets: Markets to consider (all if None)

        Returns:
            Best market or None
        """
        if markets is None:
            markets = self.aggregator.get_markets()

        best_market = None
        best_spread = float("inf")

        for market in markets:
            metrics = self.aggregator.get_metrics(market)
            if not metrics:
                continue

            if metrics.spread_bps < best_spread:
                best_spread = metrics.spread_bps
                best_market = market

        return best_market

    def get_cross_market_imbalance(self, markets: list[str] | None = None) -> float:
        """
        Calculate imbalance across multiple markets.

        Args:
            markets: Markets to include

        Returns:
            Cross-market imbalance ratio
        """
        if markets is None:
            markets = self.aggregator.get_markets()

        total_bid = Decimal("0")
        total_ask = Decimal("0")

        for market in markets:
            metrics = self.aggregator.get_metrics(market)
            if metrics:
                total_bid += metrics.total_bid_liquidity
                total_ask += metrics.total_ask_liquidity

        total = total_bid + total_ask
        if total > 0:
            return float(total_bid / total)
        return 0.5

    def add_cross_market_callback(
        self,
        callback: Callable[[dict[str, LiquiditySnapshot]], None],
    ) -> None:
        """Add callback for cross-market updates."""
        self._cross_market_callbacks.append(callback)

    def remove_cross_market_callback(
        self,
        callback: Callable[[dict[str, LiquiditySnapshot]], None],
    ) -> bool:
        """Remove cross-market callback."""
        if callback in self._cross_market_callbacks:
            self._cross_market_callbacks.remove(callback)
            return True
        return False


# Global instance
_aggregator: LiquidityAggregator | None = None
_multi_aggregator: MultiMarketAggregator | None = None


def get_aggregator() -> LiquidityAggregator:
    """Get global liquidity aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = LiquidityAggregator()
    return _aggregator


def get_multi_aggregator() -> MultiMarketAggregator:
    """Get global multi-market aggregator."""
    global _multi_aggregator
    if _multi_aggregator is None:
        _multi_aggregator = MultiMarketAggregator(get_aggregator())
    return _multi_aggregator


def reset_aggregators() -> None:
    """Reset global aggregators."""
    global _aggregator, _multi_aggregator
    _aggregator = None
    _multi_aggregator = None
