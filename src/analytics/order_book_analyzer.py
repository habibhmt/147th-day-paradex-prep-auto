"""Order Book Analyzer for market microstructure analysis.

This module provides comprehensive order book analysis including:
- Bid/ask spread analysis
- Depth and liquidity metrics
- Order imbalance detection
- Price impact estimation
- VWAP calculation
- Support/resistance level detection
- Market microstructure analysis
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class OrderBookSide(Enum):
    """Order book side."""

    BID = "bid"
    ASK = "ask"


class LiquidityLevel(Enum):
    """Liquidity level classification."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketCondition(Enum):
    """Market condition based on order book."""

    BALANCED = "balanced"
    BID_HEAVY = "bid_heavy"
    ASK_HEAVY = "ask_heavy"
    THIN = "thin"
    THICK = "thick"
    VOLATILE = "volatile"
    STABLE = "stable"


@dataclass
class OrderLevel:
    """Single order book level."""

    price: Decimal
    size: Decimal
    side: OrderBookSide
    order_count: int = 1
    timestamp: float = field(default_factory=time.time)

    @property
    def notional(self) -> Decimal:
        """Calculate notional value."""
        return self.price * self.size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": str(self.price),
            "size": str(self.size),
            "side": self.side.value,
            "order_count": self.order_count,
            "notional": str(self.notional),
        }


@dataclass
class SpreadMetrics:
    """Bid-ask spread metrics."""

    bid_price: Decimal = Decimal("0")
    ask_price: Decimal = Decimal("0")
    spread: Decimal = Decimal("0")
    spread_pct: float = 0.0
    spread_bps: float = 0.0  # Basis points
    mid_price: Decimal = Decimal("0")
    weighted_mid: Decimal = Decimal("0")
    microprice: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bid_price": str(self.bid_price),
            "ask_price": str(self.ask_price),
            "spread": str(self.spread),
            "spread_pct": self.spread_pct,
            "spread_bps": self.spread_bps,
            "mid_price": str(self.mid_price),
            "weighted_mid": str(self.weighted_mid),
            "microprice": str(self.microprice),
        }


@dataclass
class DepthMetrics:
    """Order book depth metrics."""

    bid_depth: Decimal = Decimal("0")
    ask_depth: Decimal = Decimal("0")
    total_depth: Decimal = Decimal("0")
    bid_depth_notional: Decimal = Decimal("0")
    ask_depth_notional: Decimal = Decimal("0")
    total_depth_notional: Decimal = Decimal("0")
    bid_levels: int = 0
    ask_levels: int = 0
    depth_ratio: float = 0.0
    depth_imbalance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bid_depth": str(self.bid_depth),
            "ask_depth": str(self.ask_depth),
            "total_depth": str(self.total_depth),
            "bid_depth_notional": str(self.bid_depth_notional),
            "ask_depth_notional": str(self.ask_depth_notional),
            "total_depth_notional": str(self.total_depth_notional),
            "bid_levels": self.bid_levels,
            "ask_levels": self.ask_levels,
            "depth_ratio": self.depth_ratio,
            "depth_imbalance": self.depth_imbalance,
        }


@dataclass
class ImbalanceMetrics:
    """Order book imbalance metrics."""

    volume_imbalance: float = 0.0
    notional_imbalance: float = 0.0
    order_count_imbalance: float = 0.0
    top_level_imbalance: float = 0.0
    weighted_imbalance: float = 0.0
    pressure_side: Optional[OrderBookSide] = None
    pressure_strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "volume_imbalance": self.volume_imbalance,
            "notional_imbalance": self.notional_imbalance,
            "order_count_imbalance": self.order_count_imbalance,
            "top_level_imbalance": self.top_level_imbalance,
            "weighted_imbalance": self.weighted_imbalance,
            "pressure_side": self.pressure_side.value if self.pressure_side else None,
            "pressure_strength": self.pressure_strength,
        }


@dataclass
class PriceImpact:
    """Price impact estimation."""

    size: Decimal = Decimal("0")
    side: OrderBookSide = OrderBookSide.BID
    average_price: Decimal = Decimal("0")
    worst_price: Decimal = Decimal("0")
    impact_pct: float = 0.0
    impact_bps: float = 0.0
    slippage: Decimal = Decimal("0")
    slippage_pct: float = 0.0
    levels_consumed: int = 0
    fully_filled: bool = True
    remaining_size: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "size": str(self.size),
            "side": self.side.value,
            "average_price": str(self.average_price),
            "worst_price": str(self.worst_price),
            "impact_pct": self.impact_pct,
            "impact_bps": self.impact_bps,
            "slippage": str(self.slippage),
            "slippage_pct": self.slippage_pct,
            "levels_consumed": self.levels_consumed,
            "fully_filled": self.fully_filled,
            "remaining_size": str(self.remaining_size),
        }


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""

    liquidity_level: LiquidityLevel = LiquidityLevel.MEDIUM
    liquidity_score: float = 0.0
    bid_liquidity: Decimal = Decimal("0")
    ask_liquidity: Decimal = Decimal("0")
    spread_contribution: float = 0.0
    depth_contribution: float = 0.0
    resilience: float = 0.0
    cost_to_trade_1pct: Decimal = Decimal("0")
    cost_to_trade_5pct: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "liquidity_level": self.liquidity_level.value,
            "liquidity_score": self.liquidity_score,
            "bid_liquidity": str(self.bid_liquidity),
            "ask_liquidity": str(self.ask_liquidity),
            "spread_contribution": self.spread_contribution,
            "depth_contribution": self.depth_contribution,
            "resilience": self.resilience,
            "cost_to_trade_1pct": str(self.cost_to_trade_1pct),
            "cost_to_trade_5pct": str(self.cost_to_trade_5pct),
        }


@dataclass
class SupportResistance:
    """Support and resistance levels."""

    support_levels: List[Decimal] = field(default_factory=list)
    resistance_levels: List[Decimal] = field(default_factory=list)
    strongest_support: Optional[Decimal] = None
    strongest_resistance: Optional[Decimal] = None
    support_strength: Dict[str, float] = field(default_factory=dict)
    resistance_strength: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "support_levels": [str(s) for s in self.support_levels],
            "resistance_levels": [str(r) for r in self.resistance_levels],
            "strongest_support": str(self.strongest_support) if self.strongest_support else None,
            "strongest_resistance": str(self.strongest_resistance) if self.strongest_resistance else None,
            "support_strength": {str(k): v for k, v in self.support_strength.items()},
            "resistance_strength": {str(k): v for k, v in self.resistance_strength.items()},
        }


@dataclass
class VWAPResult:
    """VWAP calculation result."""

    vwap: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    total_notional: Decimal = Decimal("0")
    deviation_from_mid: float = 0.0
    deviation_from_last: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vwap": str(self.vwap),
            "total_volume": str(self.total_volume),
            "total_notional": str(self.total_notional),
            "deviation_from_mid": self.deviation_from_mid,
            "deviation_from_last": self.deviation_from_last,
        }


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""

    market: str = ""
    timestamp: float = field(default_factory=time.time)
    bids: List[OrderLevel] = field(default_factory=list)
    asks: List[OrderLevel] = field(default_factory=list)
    spread: Optional[SpreadMetrics] = None
    depth: Optional[DepthMetrics] = None
    imbalance: Optional[ImbalanceMetrics] = None
    liquidity: Optional[LiquidityMetrics] = None

    @property
    def best_bid(self) -> Optional[OrderLevel]:
        """Get best bid."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderLevel]:
        """Get best ask."""
        return self.asks[0] if self.asks else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
            "spread": self.spread.to_dict() if self.spread else None,
            "depth": self.depth.to_dict() if self.depth else None,
            "imbalance": self.imbalance.to_dict() if self.imbalance else None,
            "liquidity": self.liquidity.to_dict() if self.liquidity else None,
        }


class OrderBookAnalyzer:
    """Order book analyzer for market microstructure analysis."""

    def __init__(
        self,
        depth_levels: int = 20,
        liquidity_thresholds: Optional[Dict[str, Decimal]] = None,
    ):
        """Initialize analyzer.

        Args:
            depth_levels: Number of levels to analyze
            liquidity_thresholds: Thresholds for liquidity classification
        """
        self.depth_levels = depth_levels
        self.liquidity_thresholds = liquidity_thresholds or {
            "very_low": Decimal("10000"),
            "low": Decimal("50000"),
            "medium": Decimal("200000"),
            "high": Decimal("500000"),
            "very_high": Decimal("1000000"),
        }

        self._snapshots: List[OrderBookSnapshot] = []
        self._max_snapshots = 100
        self._callbacks: List[Callable] = []

    def analyze(
        self,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
        market: str = "",
    ) -> OrderBookSnapshot:
        """Analyze order book.

        Args:
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            market: Market symbol

        Returns:
            Complete order book snapshot with analysis
        """
        bid_levels = [
            OrderLevel(price=p, size=s, side=OrderBookSide.BID)
            for p, s in sorted(bids, key=lambda x: x[0], reverse=True)[:self.depth_levels]
        ]

        ask_levels = [
            OrderLevel(price=p, size=s, side=OrderBookSide.ASK)
            for p, s in sorted(asks, key=lambda x: x[0])[:self.depth_levels]
        ]

        snapshot = OrderBookSnapshot(
            market=market,
            bids=bid_levels,
            asks=ask_levels,
        )

        snapshot.spread = self.calculate_spread(bid_levels, ask_levels)
        snapshot.depth = self.calculate_depth(bid_levels, ask_levels)
        snapshot.imbalance = self.calculate_imbalance(bid_levels, ask_levels)
        snapshot.liquidity = self.calculate_liquidity(snapshot)

        self._add_snapshot(snapshot)

        for callback in self._callbacks:
            callback(snapshot)

        return snapshot

    def calculate_spread(
        self,
        bids: List[OrderLevel],
        asks: List[OrderLevel],
    ) -> SpreadMetrics:
        """Calculate spread metrics.

        Args:
            bids: Bid levels
            asks: Ask levels

        Returns:
            Spread metrics
        """
        if not bids or not asks:
            return SpreadMetrics()

        best_bid = bids[0].price
        best_ask = asks[0].price
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2

        spread_pct = float(spread / mid_price * 100) if mid_price else 0
        spread_bps = spread_pct * 100

        bid_size = bids[0].size
        ask_size = asks[0].size
        total_size = bid_size + ask_size

        if total_size > 0:
            weighted_mid = (best_bid * ask_size + best_ask * bid_size) / total_size
            microprice = (best_bid * ask_size + best_ask * bid_size) / total_size
        else:
            weighted_mid = mid_price
            microprice = mid_price

        return SpreadMetrics(
            bid_price=best_bid,
            ask_price=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            spread_bps=spread_bps,
            mid_price=mid_price,
            weighted_mid=weighted_mid,
            microprice=microprice,
        )

    def calculate_depth(
        self,
        bids: List[OrderLevel],
        asks: List[OrderLevel],
    ) -> DepthMetrics:
        """Calculate depth metrics.

        Args:
            bids: Bid levels
            asks: Ask levels

        Returns:
            Depth metrics
        """
        bid_depth = sum(level.size for level in bids)
        ask_depth = sum(level.size for level in asks)
        bid_notional = sum(level.notional for level in bids)
        ask_notional = sum(level.notional for level in asks)

        total_depth = bid_depth + ask_depth
        total_notional = bid_notional + ask_notional

        depth_ratio = float(bid_depth / ask_depth) if ask_depth else 0
        depth_imbalance = float((bid_depth - ask_depth) / total_depth) if total_depth else 0

        return DepthMetrics(
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            bid_depth_notional=bid_notional,
            ask_depth_notional=ask_notional,
            total_depth_notional=total_notional,
            bid_levels=len(bids),
            ask_levels=len(asks),
            depth_ratio=depth_ratio,
            depth_imbalance=depth_imbalance,
        )

    def calculate_imbalance(
        self,
        bids: List[OrderLevel],
        asks: List[OrderLevel],
    ) -> ImbalanceMetrics:
        """Calculate imbalance metrics.

        Args:
            bids: Bid levels
            asks: Ask levels

        Returns:
            Imbalance metrics
        """
        bid_volume = sum(level.size for level in bids)
        ask_volume = sum(level.size for level in asks)
        total_volume = bid_volume + ask_volume

        bid_notional = sum(level.notional for level in bids)
        ask_notional = sum(level.notional for level in asks)
        total_notional = bid_notional + ask_notional

        bid_orders = sum(level.order_count for level in bids)
        ask_orders = sum(level.order_count for level in asks)
        total_orders = bid_orders + ask_orders

        volume_imbalance = float((bid_volume - ask_volume) / total_volume) if total_volume else 0
        notional_imbalance = float((bid_notional - ask_notional) / total_notional) if total_notional else 0
        order_count_imbalance = float((bid_orders - ask_orders) / total_orders) if total_orders else 0

        top_bid = bids[0].size if bids else Decimal("0")
        top_ask = asks[0].size if asks else Decimal("0")
        top_total = top_bid + top_ask
        top_level_imbalance = float((top_bid - top_ask) / top_total) if top_total else 0

        weights = [1 / (i + 1) for i in range(len(bids))]
        weighted_bid = sum(float(bids[i].size) * weights[i] for i in range(len(bids))) if bids else 0

        weights = [1 / (i + 1) for i in range(len(asks))]
        weighted_ask = sum(float(asks[i].size) * weights[i] for i in range(len(asks))) if asks else 0

        weighted_total = weighted_bid + weighted_ask
        weighted_imbalance = (weighted_bid - weighted_ask) / weighted_total if weighted_total else 0

        if volume_imbalance > 0.2:
            pressure_side = OrderBookSide.BID
            pressure_strength = abs(volume_imbalance)
        elif volume_imbalance < -0.2:
            pressure_side = OrderBookSide.ASK
            pressure_strength = abs(volume_imbalance)
        else:
            pressure_side = None
            pressure_strength = 0.0

        return ImbalanceMetrics(
            volume_imbalance=volume_imbalance,
            notional_imbalance=notional_imbalance,
            order_count_imbalance=order_count_imbalance,
            top_level_imbalance=top_level_imbalance,
            weighted_imbalance=weighted_imbalance,
            pressure_side=pressure_side,
            pressure_strength=pressure_strength,
        )

    def calculate_liquidity(
        self,
        snapshot: OrderBookSnapshot,
    ) -> LiquidityMetrics:
        """Calculate liquidity metrics.

        Args:
            snapshot: Order book snapshot

        Returns:
            Liquidity metrics
        """
        if not snapshot.depth:
            return LiquidityMetrics()

        total_notional = snapshot.depth.total_depth_notional

        if total_notional >= self.liquidity_thresholds["very_high"]:
            level = LiquidityLevel.VERY_HIGH
            score = 100.0
        elif total_notional >= self.liquidity_thresholds["high"]:
            level = LiquidityLevel.HIGH
            score = 80.0
        elif total_notional >= self.liquidity_thresholds["medium"]:
            level = LiquidityLevel.MEDIUM
            score = 60.0
        elif total_notional >= self.liquidity_thresholds["low"]:
            level = LiquidityLevel.LOW
            score = 40.0
        else:
            level = LiquidityLevel.VERY_LOW
            score = 20.0

        spread_pct = snapshot.spread.spread_pct if snapshot.spread else 0
        spread_contribution = max(0, 100 - spread_pct * 10)

        depth_contribution = min(100, float(total_notional / Decimal("100000")) * 10)

        return LiquidityMetrics(
            liquidity_level=level,
            liquidity_score=score,
            bid_liquidity=snapshot.depth.bid_depth_notional,
            ask_liquidity=snapshot.depth.ask_depth_notional,
            spread_contribution=spread_contribution,
            depth_contribution=depth_contribution,
            resilience=0.5,
            cost_to_trade_1pct=total_notional * Decimal("0.01"),
            cost_to_trade_5pct=total_notional * Decimal("0.05"),
        )

    def estimate_price_impact(
        self,
        size: Decimal,
        side: OrderBookSide,
        snapshot: Optional[OrderBookSnapshot] = None,
    ) -> PriceImpact:
        """Estimate price impact for a given order size.

        Args:
            size: Order size
            side: Order side (BID for buy, ASK for sell)
            snapshot: Order book snapshot (uses latest if not provided)

        Returns:
            Price impact estimation
        """
        if snapshot is None:
            if not self._snapshots:
                return PriceImpact(size=size, side=side)
            snapshot = self._snapshots[-1]

        levels = snapshot.asks if side == OrderBookSide.BID else snapshot.bids
        if not levels:
            return PriceImpact(size=size, side=side)

        remaining = size
        total_notional = Decimal("0")
        total_filled = Decimal("0")
        levels_consumed = 0
        worst_price = Decimal("0")

        for level in levels:
            if remaining <= 0:
                break

            fill_size = min(remaining, level.size)
            total_notional += fill_size * level.price
            total_filled += fill_size
            remaining -= fill_size
            levels_consumed += 1
            worst_price = level.price

        if total_filled > 0:
            avg_price = total_notional / total_filled
        else:
            avg_price = Decimal("0")

        reference_price = levels[0].price
        if reference_price > 0:
            impact_pct = float((worst_price - reference_price) / reference_price * 100)
            if side == OrderBookSide.ASK:
                impact_pct = -impact_pct
            slippage = avg_price - reference_price
            if side == OrderBookSide.ASK:
                slippage = -slippage
            slippage_pct = float(slippage / reference_price * 100)
        else:
            impact_pct = 0.0
            slippage = Decimal("0")
            slippage_pct = 0.0

        return PriceImpact(
            size=size,
            side=side,
            average_price=avg_price,
            worst_price=worst_price,
            impact_pct=abs(impact_pct),
            impact_bps=abs(impact_pct) * 100,
            slippage=abs(slippage),
            slippage_pct=abs(slippage_pct),
            levels_consumed=levels_consumed,
            fully_filled=remaining <= 0,
            remaining_size=max(Decimal("0"), remaining),
        )

    def calculate_vwap(
        self,
        levels: List[OrderLevel],
        up_to_size: Optional[Decimal] = None,
    ) -> VWAPResult:
        """Calculate VWAP for given levels.

        Args:
            levels: Order book levels
            up_to_size: Optional size limit

        Returns:
            VWAP result
        """
        if not levels:
            return VWAPResult()

        total_notional = Decimal("0")
        total_volume = Decimal("0")

        for level in levels:
            if up_to_size and total_volume >= up_to_size:
                break

            available = level.size
            if up_to_size:
                available = min(available, up_to_size - total_volume)

            total_notional += available * level.price
            total_volume += available

        if total_volume > 0:
            vwap = total_notional / total_volume
        else:
            vwap = Decimal("0")

        mid_price = levels[0].price if levels else Decimal("0")
        deviation_from_mid = float((vwap - mid_price) / mid_price * 100) if mid_price else 0

        return VWAPResult(
            vwap=vwap,
            total_volume=total_volume,
            total_notional=total_notional,
            deviation_from_mid=deviation_from_mid,
            deviation_from_last=0.0,
        )

    def find_support_resistance(
        self,
        snapshot: Optional[OrderBookSnapshot] = None,
        strength_threshold: Decimal = Decimal("2"),
    ) -> SupportResistance:
        """Find support and resistance levels.

        Args:
            snapshot: Order book snapshot
            strength_threshold: Minimum strength multiplier for significant levels

        Returns:
            Support and resistance levels
        """
        if snapshot is None:
            if not self._snapshots:
                return SupportResistance()
            snapshot = self._snapshots[-1]

        if not snapshot.bids and not snapshot.asks:
            return SupportResistance()

        avg_bid_size = sum(b.size for b in snapshot.bids) / len(snapshot.bids) if snapshot.bids else Decimal("1")
        avg_ask_size = sum(a.size for a in snapshot.asks) / len(snapshot.asks) if snapshot.asks else Decimal("1")

        support_levels = []
        support_strength = {}
        for bid in snapshot.bids:
            if bid.size >= avg_bid_size * strength_threshold:
                support_levels.append(bid.price)
                support_strength[str(bid.price)] = float(bid.size / avg_bid_size)

        resistance_levels = []
        resistance_strength = {}
        for ask in snapshot.asks:
            if ask.size >= avg_ask_size * strength_threshold:
                resistance_levels.append(ask.price)
                resistance_strength[str(ask.price)] = float(ask.size / avg_ask_size)

        strongest_support = max(support_levels, key=lambda x: support_strength.get(str(x), 0)) if support_levels else None
        strongest_resistance = min(resistance_levels, key=lambda x: resistance_strength.get(str(x), 0)) if resistance_levels else None

        return SupportResistance(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            strongest_support=strongest_support,
            strongest_resistance=strongest_resistance,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
        )

    def get_market_condition(
        self,
        snapshot: Optional[OrderBookSnapshot] = None,
    ) -> MarketCondition:
        """Determine market condition from order book.

        Args:
            snapshot: Order book snapshot

        Returns:
            Market condition
        """
        if snapshot is None:
            if not self._snapshots:
                return MarketCondition.BALANCED
            snapshot = self._snapshots[-1]

        if not snapshot.imbalance:
            return MarketCondition.BALANCED

        imbalance = snapshot.imbalance.volume_imbalance

        if snapshot.liquidity and snapshot.liquidity.liquidity_level in [LiquidityLevel.VERY_LOW, LiquidityLevel.LOW]:
            return MarketCondition.THIN

        if snapshot.liquidity and snapshot.liquidity.liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
            if abs(imbalance) < 0.1:
                return MarketCondition.THICK

        if imbalance > 0.3:
            return MarketCondition.BID_HEAVY
        elif imbalance < -0.3:
            return MarketCondition.ASK_HEAVY

        return MarketCondition.BALANCED

    def add_callback(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Add callback for new snapshots.

        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_latest_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Get latest snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self, limit: int = 10) -> List[OrderBookSnapshot]:
        """Get recent snapshots.

        Args:
            limit: Maximum number of snapshots

        Returns:
            List of snapshots
        """
        return self._snapshots[-limit:]

    def get_spread_history(self, limit: int = 10) -> List[float]:
        """Get spread history.

        Args:
            limit: Maximum number of entries

        Returns:
            List of spread percentages
        """
        spreads = []
        for snapshot in self._snapshots[-limit:]:
            if snapshot.spread:
                spreads.append(snapshot.spread.spread_pct)
        return spreads

    def get_average_spread(self) -> float:
        """Get average spread from history."""
        spreads = self.get_spread_history(limit=50)
        return statistics.mean(spreads) if spreads else 0.0

    def clear_history(self) -> None:
        """Clear snapshot history."""
        self._snapshots.clear()

    def _add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add snapshot to history.

        Args:
            snapshot: Snapshot to add
        """
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]


class RealTimeOrderBook:
    """Real-time order book with incremental updates."""

    def __init__(self, market: str, depth_levels: int = 20):
        """Initialize real-time order book.

        Args:
            market: Market symbol
            depth_levels: Number of levels to track
        """
        self.market = market
        self.depth_levels = depth_levels
        self._bids: Dict[Decimal, Decimal] = {}
        self._asks: Dict[Decimal, Decimal] = {}
        self._last_update = 0.0
        self._sequence = 0
        self._analyzer = OrderBookAnalyzer(depth_levels=depth_levels)
        self._callbacks: List[Callable] = []

    def set_snapshot(
        self,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
        sequence: int = 0,
    ) -> OrderBookSnapshot:
        """Set full order book snapshot.

        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            sequence: Sequence number

        Returns:
            Analysis snapshot
        """
        self._bids = {price: size for price, size in bids if size > 0}
        self._asks = {price: size for price, size in asks if size > 0}
        self._sequence = sequence
        self._last_update = time.time()

        return self._analyze()

    def apply_update(
        self,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
        sequence: int = 0,
    ) -> Optional[OrderBookSnapshot]:
        """Apply incremental update.

        Args:
            bids: List of (price, size) tuples (size=0 means delete)
            asks: List of (price, size) tuples (size=0 means delete)
            sequence: Sequence number

        Returns:
            Analysis snapshot or None if sequence mismatch
        """
        if sequence and self._sequence and sequence <= self._sequence:
            return None

        for price, size in bids:
            if size > 0:
                self._bids[price] = size
            elif price in self._bids:
                del self._bids[price]

        for price, size in asks:
            if size > 0:
                self._asks[price] = size
            elif price in self._asks:
                del self._asks[price]

        self._sequence = sequence
        self._last_update = time.time()

        return self._analyze()

    def _analyze(self) -> OrderBookSnapshot:
        """Analyze current order book state."""
        bids = [(price, size) for price, size in self._bids.items()]
        asks = [(price, size) for price, size in self._asks.items()]

        snapshot = self._analyzer.analyze(bids, asks, self.market)

        for callback in self._callbacks:
            callback(snapshot)

        return snapshot

    def get_best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid price and size."""
        if not self._bids:
            return None
        price = max(self._bids.keys())
        return (price, self._bids[price])

    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask price and size."""
        if not self._asks:
            return None
        price = min(self._asks.keys())
        return (price, self._asks[price])

    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid[0] + ask[0]) / 2
        return None

    def get_spread(self) -> Optional[Decimal]:
        """Get spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask[0] - bid[0]
        return None

    def add_callback(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Add update callback.

        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)

    def get_depth(self, side: OrderBookSide, levels: int = 10) -> List[Tuple[Decimal, Decimal]]:
        """Get order book depth.

        Args:
            side: Side to get
            levels: Number of levels

        Returns:
            List of (price, size) tuples
        """
        if side == OrderBookSide.BID:
            prices = sorted(self._bids.keys(), reverse=True)[:levels]
            return [(p, self._bids[p]) for p in prices]
        else:
            prices = sorted(self._asks.keys())[:levels]
            return [(p, self._asks[p]) for p in prices]

    @property
    def last_update(self) -> float:
        """Get last update timestamp."""
        return self._last_update

    @property
    def sequence(self) -> int:
        """Get current sequence number."""
        return self._sequence


class MultiMarketOrderBook:
    """Order book tracker for multiple markets."""

    def __init__(self, depth_levels: int = 20):
        """Initialize multi-market order book.

        Args:
            depth_levels: Number of levels per market
        """
        self.depth_levels = depth_levels
        self._books: Dict[str, RealTimeOrderBook] = {}
        self._callbacks: List[Callable] = []

    def get_or_create(self, market: str) -> RealTimeOrderBook:
        """Get or create order book for market.

        Args:
            market: Market symbol

        Returns:
            Real-time order book
        """
        if market not in self._books:
            self._books[market] = RealTimeOrderBook(market, self.depth_levels)
            for callback in self._callbacks:
                self._books[market].add_callback(callback)
        return self._books[market]

    def update(
        self,
        market: str,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
        is_snapshot: bool = False,
        sequence: int = 0,
    ) -> Optional[OrderBookSnapshot]:
        """Update order book for market.

        Args:
            market: Market symbol
            bids: Bid updates
            asks: Ask updates
            is_snapshot: Whether this is a full snapshot
            sequence: Sequence number

        Returns:
            Analysis snapshot
        """
        book = self.get_or_create(market)

        if is_snapshot:
            return book.set_snapshot(bids, asks, sequence)
        else:
            return book.apply_update(bids, asks, sequence)

    def get_all_snapshots(self) -> Dict[str, Optional[OrderBookSnapshot]]:
        """Get latest snapshots for all markets."""
        return {
            market: book._analyzer.get_latest_snapshot()
            for market, book in self._books.items()
        }

    def add_callback(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Add callback for all markets.

        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)
        for book in self._books.values():
            book.add_callback(callback)

    def get_market_count(self) -> int:
        """Get number of tracked markets."""
        return len(self._books)

    def get_markets(self) -> List[str]:
        """Get list of tracked markets."""
        return list(self._books.keys())

    def remove_market(self, market: str) -> bool:
        """Remove market from tracking.

        Args:
            market: Market to remove

        Returns:
            True if removed
        """
        if market in self._books:
            del self._books[market]
            return True
        return False

    def clear(self) -> None:
        """Clear all order books."""
        self._books.clear()


# Global order book analyzer instance
_order_book_analyzer: Optional[OrderBookAnalyzer] = None


def get_order_book_analyzer() -> OrderBookAnalyzer:
    """Get global order book analyzer."""
    global _order_book_analyzer
    if _order_book_analyzer is None:
        _order_book_analyzer = OrderBookAnalyzer()
    return _order_book_analyzer


def reset_order_book_analyzer() -> None:
    """Reset global order book analyzer."""
    global _order_book_analyzer
    _order_book_analyzer = None
