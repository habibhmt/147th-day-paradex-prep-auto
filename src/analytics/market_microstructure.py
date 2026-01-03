"""
Market Microstructure Analysis Module.

Analyzes market microstructure elements including:
- Bid-ask spread dynamics
- Market depth and liquidity
- Trade imbalance and flow toxicity
- Quote activity and market maker behavior
- Price impact estimation
- Order book dynamics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LiquidityState(Enum):
    """Market liquidity state."""
    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    NORMAL = "normal"
    ILLIQUID = "illiquid"
    CRITICALLY_ILLIQUID = "critically_illiquid"


class MarketMakerActivity(Enum):
    """Market maker activity level."""
    VERY_ACTIVE = "very_active"
    ACTIVE = "active"
    NORMAL = "normal"
    REDUCED = "reduced"
    ABSENT = "absent"


class FlowToxicity(Enum):
    """Order flow toxicity level."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SpreadRegime(Enum):
    """Spread regime classification."""
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    VERY_WIDE = "very_wide"


@dataclass
class SpreadMetrics:
    """Bid-ask spread metrics."""
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    spread_absolute: Decimal
    spread_bps: Decimal  # Basis points
    mid_price: Decimal
    spread_regime: SpreadRegime
    is_crossed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_price": str(self.bid_price),
            "ask_price": str(self.ask_price),
            "spread_absolute": str(self.spread_absolute),
            "spread_bps": str(self.spread_bps),
            "mid_price": str(self.mid_price),
            "spread_regime": self.spread_regime.value,
            "is_crossed": self.is_crossed
        }


@dataclass
class DepthMetrics:
    """Order book depth metrics."""
    symbol: str
    timestamp: datetime
    bid_depth_levels: int
    ask_depth_levels: int
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    bid_volume_1pct: Decimal  # Volume within 1% of mid
    ask_volume_1pct: Decimal
    bid_volume_5pct: Decimal  # Volume within 5% of mid
    ask_volume_5pct: Decimal
    depth_imbalance: Decimal  # Positive = more bids
    liquidity_state: LiquidityState

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_depth_levels": self.bid_depth_levels,
            "ask_depth_levels": self.ask_depth_levels,
            "total_bid_volume": str(self.total_bid_volume),
            "total_ask_volume": str(self.total_ask_volume),
            "bid_volume_1pct": str(self.bid_volume_1pct),
            "ask_volume_1pct": str(self.ask_volume_1pct),
            "bid_volume_5pct": str(self.bid_volume_5pct),
            "ask_volume_5pct": str(self.ask_volume_5pct),
            "depth_imbalance": str(self.depth_imbalance),
            "liquidity_state": self.liquidity_state.value
        }


@dataclass
class TradeImbalance:
    """Trade imbalance metrics."""
    symbol: str
    timestamp: datetime
    window_seconds: int
    buy_volume: Decimal
    sell_volume: Decimal
    buy_trades: int
    sell_trades: int
    net_volume: Decimal
    imbalance_ratio: Decimal  # -1 to 1
    vwap_buy: Decimal
    vwap_sell: Decimal
    flow_toxicity: FlowToxicity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "window_seconds": self.window_seconds,
            "buy_volume": str(self.buy_volume),
            "sell_volume": str(self.sell_volume),
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "net_volume": str(self.net_volume),
            "imbalance_ratio": str(self.imbalance_ratio),
            "vwap_buy": str(self.vwap_buy),
            "vwap_sell": str(self.vwap_sell),
            "flow_toxicity": self.flow_toxicity.value
        }


@dataclass
class QuoteActivity:
    """Quote activity metrics."""
    symbol: str
    timestamp: datetime
    window_seconds: int
    quote_updates: int
    quote_rate_per_second: Decimal
    bid_updates: int
    ask_updates: int
    quote_to_trade_ratio: Decimal
    average_quote_lifetime_ms: Decimal
    mm_activity: MarketMakerActivity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "window_seconds": self.window_seconds,
            "quote_updates": self.quote_updates,
            "quote_rate_per_second": str(self.quote_rate_per_second),
            "bid_updates": self.bid_updates,
            "ask_updates": self.ask_updates,
            "quote_to_trade_ratio": str(self.quote_to_trade_ratio),
            "average_quote_lifetime_ms": str(self.average_quote_lifetime_ms),
            "mm_activity": self.mm_activity.value
        }


@dataclass
class PriceImpact:
    """Price impact estimation."""
    symbol: str
    timestamp: datetime
    order_size: Decimal
    side: str  # "buy" or "sell"
    estimated_impact_bps: Decimal
    estimated_slippage: Decimal
    effective_spread_bps: Decimal
    market_impact_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "order_size": str(self.order_size),
            "side": self.side,
            "estimated_impact_bps": str(self.estimated_impact_bps),
            "estimated_slippage": str(self.estimated_slippage),
            "effective_spread_bps": str(self.effective_spread_bps),
            "market_impact_cost": str(self.market_impact_cost)
        }


@dataclass
class MicrostructureSnapshot:
    """Complete microstructure snapshot."""
    symbol: str
    timestamp: datetime
    spread_metrics: SpreadMetrics
    depth_metrics: DepthMetrics
    trade_imbalance: TradeImbalance | None
    quote_activity: QuoteActivity | None
    price_impact_buy: PriceImpact | None
    price_impact_sell: PriceImpact | None
    overall_quality_score: Decimal  # 0-100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "spread_metrics": self.spread_metrics.to_dict(),
            "depth_metrics": self.depth_metrics.to_dict(),
            "trade_imbalance": self.trade_imbalance.to_dict() if self.trade_imbalance else None,
            "quote_activity": self.quote_activity.to_dict() if self.quote_activity else None,
            "price_impact_buy": self.price_impact_buy.to_dict() if self.price_impact_buy else None,
            "price_impact_sell": self.price_impact_sell.to_dict() if self.price_impact_sell else None,
            "overall_quality_score": str(self.overall_quality_score)
        }


@dataclass
class QuoteRecord:
    """Record of a quote update."""
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal


@dataclass
class TradeRecord:
    """Record of a trade."""
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # "buy" or "sell"


class SpreadAnalyzer:
    """Analyzes bid-ask spread dynamics."""

    def __init__(
        self,
        tight_threshold_bps: Decimal = Decimal("5"),
        wide_threshold_bps: Decimal = Decimal("20"),
        very_wide_threshold_bps: Decimal = Decimal("50")
    ):
        """
        Initialize spread analyzer.

        Args:
            tight_threshold_bps: Spread below this is tight
            wide_threshold_bps: Spread above this is wide
            very_wide_threshold_bps: Spread above this is very wide
        """
        self.tight_threshold = tight_threshold_bps
        self.wide_threshold = wide_threshold_bps
        self.very_wide_threshold = very_wide_threshold_bps
        self._spread_history: dict[str, list[SpreadMetrics]] = {}
        self._max_history = 1000

    def analyze_spread(
        self,
        symbol: str,
        bid_price: Decimal,
        ask_price: Decimal,
        timestamp: datetime | None = None
    ) -> SpreadMetrics:
        """
        Analyze current spread.

        Args:
            symbol: Trading symbol
            bid_price: Current bid price
            ask_price: Current ask price
            timestamp: Timestamp (default: now)

        Returns:
            SpreadMetrics with analysis
        """
        timestamp = timestamp or datetime.now()

        # Check for crossed market
        is_crossed = bid_price > ask_price

        # Calculate spread
        spread_absolute = ask_price - bid_price
        mid_price = (bid_price + ask_price) / 2

        if mid_price > 0:
            spread_bps = (spread_absolute / mid_price) * 10000
        else:
            spread_bps = Decimal("0")

        # Determine spread regime
        if is_crossed:
            spread_regime = SpreadRegime.VERY_WIDE  # Anomaly
        elif spread_bps < self.tight_threshold:
            spread_regime = SpreadRegime.TIGHT
        elif spread_bps < self.wide_threshold:
            spread_regime = SpreadRegime.NORMAL
        elif spread_bps < self.very_wide_threshold:
            spread_regime = SpreadRegime.WIDE
        else:
            spread_regime = SpreadRegime.VERY_WIDE

        metrics = SpreadMetrics(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=bid_price,
            ask_price=ask_price,
            spread_absolute=spread_absolute,
            spread_bps=spread_bps,
            mid_price=mid_price,
            spread_regime=spread_regime,
            is_crossed=is_crossed
        )

        # Store history
        if symbol not in self._spread_history:
            self._spread_history[symbol] = []

        self._spread_history[symbol].append(metrics)

        # Limit history
        if len(self._spread_history[symbol]) > self._max_history:
            self._spread_history[symbol] = self._spread_history[symbol][-self._max_history:]

        return metrics

    def get_average_spread(
        self,
        symbol: str,
        window_seconds: int = 300
    ) -> Decimal | None:
        """
        Get average spread over time window.

        Args:
            symbol: Trading symbol
            window_seconds: Time window in seconds

        Returns:
            Average spread in bps or None
        """
        if symbol not in self._spread_history:
            return None

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [
            m.spread_bps for m in self._spread_history[symbol]
            if m.timestamp >= cutoff
        ]

        if not recent:
            return None

        return sum(recent) / len(recent)

    def get_spread_volatility(
        self,
        symbol: str,
        window_seconds: int = 300
    ) -> Decimal | None:
        """
        Get spread volatility (std dev) over time window.

        Args:
            symbol: Trading symbol
            window_seconds: Time window in seconds

        Returns:
            Spread standard deviation in bps or None
        """
        if symbol not in self._spread_history:
            return None

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [
            m.spread_bps for m in self._spread_history[symbol]
            if m.timestamp >= cutoff
        ]

        if len(recent) < 2:
            return None

        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)

        # Square root approximation using Newton's method
        if variance <= 0:
            return Decimal("0")

        std_dev = variance.sqrt()
        return std_dev


class DepthAnalyzer:
    """Analyzes order book depth and liquidity."""

    def __init__(
        self,
        highly_liquid_threshold: Decimal = Decimal("1000000"),
        liquid_threshold: Decimal = Decimal("500000"),
        illiquid_threshold: Decimal = Decimal("100000"),
        critical_threshold: Decimal = Decimal("10000")
    ):
        """
        Initialize depth analyzer.

        Args:
            highly_liquid_threshold: Volume above this is highly liquid
            liquid_threshold: Volume above this is liquid
            illiquid_threshold: Volume below this is illiquid
            critical_threshold: Volume below this is critically illiquid
        """
        self.highly_liquid = highly_liquid_threshold
        self.liquid = liquid_threshold
        self.illiquid = illiquid_threshold
        self.critical = critical_threshold

    def analyze_depth(
        self,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],  # [(price, size), ...]
        asks: list[tuple[Decimal, Decimal]],
        mid_price: Decimal | None = None,
        timestamp: datetime | None = None
    ) -> DepthMetrics:
        """
        Analyze order book depth.

        Args:
            symbol: Trading symbol
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            mid_price: Mid price (calculated if not provided)
            timestamp: Timestamp (default: now)

        Returns:
            DepthMetrics with analysis
        """
        timestamp = timestamp or datetime.now()

        # Calculate mid price if not provided
        if mid_price is None and bids and asks:
            mid_price = (bids[0][0] + asks[0][0]) / 2
        elif mid_price is None:
            mid_price = Decimal("0")

        # Calculate total volumes
        total_bid_volume = sum(price * size for price, size in bids)
        total_ask_volume = sum(price * size for price, size in asks)

        # Calculate volume within price ranges
        bid_vol_1pct = Decimal("0")
        bid_vol_5pct = Decimal("0")
        ask_vol_1pct = Decimal("0")
        ask_vol_5pct = Decimal("0")

        if mid_price > 0:
            price_1pct = mid_price * Decimal("0.01")
            price_5pct = mid_price * Decimal("0.05")

            for price, size in bids:
                notional = price * size
                if mid_price - price <= price_1pct:
                    bid_vol_1pct += notional
                if mid_price - price <= price_5pct:
                    bid_vol_5pct += notional

            for price, size in asks:
                notional = price * size
                if price - mid_price <= price_1pct:
                    ask_vol_1pct += notional
                if price - mid_price <= price_5pct:
                    ask_vol_5pct += notional

        # Calculate depth imbalance
        total_volume = total_bid_volume + total_ask_volume
        if total_volume > 0:
            depth_imbalance = (total_bid_volume - total_ask_volume) / total_volume
        else:
            depth_imbalance = Decimal("0")

        # Determine liquidity state
        liquidity_metric = total_bid_volume + total_ask_volume

        if liquidity_metric >= self.highly_liquid:
            liquidity_state = LiquidityState.HIGHLY_LIQUID
        elif liquidity_metric >= self.liquid:
            liquidity_state = LiquidityState.LIQUID
        elif liquidity_metric >= self.illiquid:
            liquidity_state = LiquidityState.NORMAL
        elif liquidity_metric >= self.critical:
            liquidity_state = LiquidityState.ILLIQUID
        else:
            liquidity_state = LiquidityState.CRITICALLY_ILLIQUID

        return DepthMetrics(
            symbol=symbol,
            timestamp=timestamp,
            bid_depth_levels=len(bids),
            ask_depth_levels=len(asks),
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            bid_volume_1pct=bid_vol_1pct,
            ask_volume_1pct=ask_vol_1pct,
            bid_volume_5pct=bid_vol_5pct,
            ask_volume_5pct=ask_vol_5pct,
            depth_imbalance=depth_imbalance,
            liquidity_state=liquidity_state
        )


class TradeFlowAnalyzer:
    """Analyzes trade flow and imbalance."""

    def __init__(
        self,
        high_toxicity_threshold: Decimal = Decimal("0.7"),
        moderate_toxicity_threshold: Decimal = Decimal("0.4")
    ):
        """
        Initialize trade flow analyzer.

        Args:
            high_toxicity_threshold: Imbalance above this is high toxicity
            moderate_toxicity_threshold: Imbalance above this is moderate
        """
        self.high_toxicity = high_toxicity_threshold
        self.moderate_toxicity = moderate_toxicity_threshold
        self._trade_history: dict[str, list[TradeRecord]] = {}
        self._max_history = 5000

    def record_trade(
        self,
        symbol: str,
        price: Decimal,
        size: Decimal,
        side: str,
        timestamp: datetime | None = None
    ) -> None:
        """
        Record a trade.

        Args:
            symbol: Trading symbol
            price: Trade price
            size: Trade size
            side: "buy" or "sell"
            timestamp: Trade timestamp
        """
        timestamp = timestamp or datetime.now()

        record = TradeRecord(
            timestamp=timestamp,
            price=price,
            size=size,
            side=side
        )

        if symbol not in self._trade_history:
            self._trade_history[symbol] = []

        self._trade_history[symbol].append(record)

        # Limit history
        if len(self._trade_history[symbol]) > self._max_history:
            self._trade_history[symbol] = self._trade_history[symbol][-self._max_history:]

    def analyze_imbalance(
        self,
        symbol: str,
        window_seconds: int = 60,
        timestamp: datetime | None = None
    ) -> TradeImbalance | None:
        """
        Analyze trade imbalance over time window.

        Args:
            symbol: Trading symbol
            window_seconds: Analysis window in seconds
            timestamp: End timestamp (default: now)

        Returns:
            TradeImbalance or None if no data
        """
        if symbol not in self._trade_history:
            return None

        timestamp = timestamp or datetime.now()
        cutoff = timestamp - timedelta(seconds=window_seconds)

        recent_trades = [
            t for t in self._trade_history[symbol]
            if t.timestamp >= cutoff and t.timestamp <= timestamp
        ]

        if not recent_trades:
            return None

        # Separate buys and sells
        buy_trades = [t for t in recent_trades if t.side == "buy"]
        sell_trades = [t for t in recent_trades if t.side == "sell"]

        buy_volume = sum(t.price * t.size for t in buy_trades)
        sell_volume = sum(t.price * t.size for t in sell_trades)

        # Calculate VWAPs
        buy_size_total = sum(t.size for t in buy_trades)
        sell_size_total = sum(t.size for t in sell_trades)

        vwap_buy = (
            sum(t.price * t.size for t in buy_trades) / buy_size_total
            if buy_size_total > 0 else Decimal("0")
        )
        vwap_sell = (
            sum(t.price * t.size for t in sell_trades) / sell_size_total
            if sell_size_total > 0 else Decimal("0")
        )

        # Calculate imbalance ratio
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            imbalance_ratio = (buy_volume - sell_volume) / total_volume
        else:
            imbalance_ratio = Decimal("0")

        # Determine flow toxicity
        abs_imbalance = abs(imbalance_ratio)
        if abs_imbalance >= Decimal("0.9"):
            flow_toxicity = FlowToxicity.VERY_HIGH
        elif abs_imbalance >= self.high_toxicity:
            flow_toxicity = FlowToxicity.HIGH
        elif abs_imbalance >= self.moderate_toxicity:
            flow_toxicity = FlowToxicity.MODERATE
        elif abs_imbalance >= Decimal("0.2"):
            flow_toxicity = FlowToxicity.LOW
        else:
            flow_toxicity = FlowToxicity.VERY_LOW

        return TradeImbalance(
            symbol=symbol,
            timestamp=timestamp,
            window_seconds=window_seconds,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_trades=len(buy_trades),
            sell_trades=len(sell_trades),
            net_volume=buy_volume - sell_volume,
            imbalance_ratio=imbalance_ratio,
            vwap_buy=vwap_buy,
            vwap_sell=vwap_sell,
            flow_toxicity=flow_toxicity
        )

    def get_vpin(
        self,
        symbol: str,
        bucket_size: Decimal = Decimal("10000"),
        num_buckets: int = 50
    ) -> Decimal | None:
        """
        Calculate Volume-synchronized Probability of Informed Trading (VPIN).

        Args:
            symbol: Trading symbol
            bucket_size: Volume per bucket
            num_buckets: Number of buckets to use

        Returns:
            VPIN value (0-1) or None
        """
        if symbol not in self._trade_history:
            return None

        trades = self._trade_history[symbol]
        if not trades:
            return None

        # Create volume buckets
        buckets: list[dict[str, Decimal]] = []
        current_bucket: dict[str, Decimal] = {"buy": Decimal("0"), "sell": Decimal("0")}
        current_volume = Decimal("0")

        for trade in trades:
            volume = trade.price * trade.size

            while volume > 0:
                remaining_in_bucket = bucket_size - current_volume
                add_volume = min(volume, remaining_in_bucket)

                current_bucket[trade.side] += add_volume
                current_volume += add_volume
                volume -= add_volume

                if current_volume >= bucket_size:
                    buckets.append(current_bucket.copy())
                    current_bucket = {"buy": Decimal("0"), "sell": Decimal("0")}
                    current_volume = Decimal("0")

        # Use last n buckets
        if len(buckets) < num_buckets:
            if len(buckets) == 0:
                return None
            num_buckets = len(buckets)

        recent_buckets = buckets[-num_buckets:]

        # Calculate VPIN
        total_imbalance = sum(
            abs(b["buy"] - b["sell"]) for b in recent_buckets
        )
        total_volume = sum(
            b["buy"] + b["sell"] for b in recent_buckets
        )

        if total_volume > 0:
            return total_imbalance / total_volume

        return Decimal("0")


class QuoteAnalyzer:
    """Analyzes quote activity and market maker behavior."""

    def __init__(
        self,
        very_active_threshold: int = 100,  # Quotes per second
        active_threshold: int = 50,
        reduced_threshold: int = 10,
        absent_threshold: int = 1
    ):
        """
        Initialize quote analyzer.

        Args:
            very_active_threshold: Quote rate for very active
            active_threshold: Quote rate for active
            reduced_threshold: Quote rate for reduced activity
            absent_threshold: Quote rate for absent
        """
        self.very_active = very_active_threshold
        self.active = active_threshold
        self.reduced = reduced_threshold
        self.absent = absent_threshold
        self._quote_history: dict[str, list[QuoteRecord]] = {}
        self._max_history = 10000

    def record_quote(
        self,
        symbol: str,
        bid_price: Decimal,
        ask_price: Decimal,
        bid_size: Decimal,
        ask_size: Decimal,
        timestamp: datetime | None = None
    ) -> None:
        """
        Record a quote update.

        Args:
            symbol: Trading symbol
            bid_price: Current bid price
            ask_price: Current ask price
            bid_size: Current bid size
            ask_size: Current ask size
            timestamp: Quote timestamp
        """
        timestamp = timestamp or datetime.now()

        record = QuoteRecord(
            timestamp=timestamp,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size
        )

        if symbol not in self._quote_history:
            self._quote_history[symbol] = []

        self._quote_history[symbol].append(record)

        # Limit history
        if len(self._quote_history[symbol]) > self._max_history:
            self._quote_history[symbol] = self._quote_history[symbol][-self._max_history:]

    def analyze_activity(
        self,
        symbol: str,
        window_seconds: int = 60,
        trade_count: int | None = None,
        timestamp: datetime | None = None
    ) -> QuoteActivity | None:
        """
        Analyze quote activity.

        Args:
            symbol: Trading symbol
            window_seconds: Analysis window in seconds
            trade_count: Number of trades in window (for ratio)
            timestamp: End timestamp (default: now)

        Returns:
            QuoteActivity or None if no data
        """
        if symbol not in self._quote_history:
            return None

        timestamp = timestamp or datetime.now()
        cutoff = timestamp - timedelta(seconds=window_seconds)

        recent_quotes = [
            q for q in self._quote_history[symbol]
            if q.timestamp >= cutoff and q.timestamp <= timestamp
        ]

        if not recent_quotes:
            return None

        # Count updates
        quote_updates = len(recent_quotes)
        quote_rate = Decimal(quote_updates) / Decimal(window_seconds)

        # Count bid vs ask updates
        bid_updates = 0
        ask_updates = 0

        for i in range(1, len(recent_quotes)):
            if recent_quotes[i].bid_price != recent_quotes[i - 1].bid_price:
                bid_updates += 1
            if recent_quotes[i].ask_price != recent_quotes[i - 1].ask_price:
                ask_updates += 1

        # Quote to trade ratio
        if trade_count and trade_count > 0:
            quote_to_trade = Decimal(quote_updates) / Decimal(trade_count)
        else:
            quote_to_trade = Decimal("0")

        # Average quote lifetime
        if len(recent_quotes) >= 2:
            total_lifetime_ms = Decimal("0")
            for i in range(1, len(recent_quotes)):
                diff = recent_quotes[i].timestamp - recent_quotes[i - 1].timestamp
                total_lifetime_ms += Decimal(diff.total_seconds() * 1000)
            avg_lifetime = total_lifetime_ms / (len(recent_quotes) - 1)
        else:
            avg_lifetime = Decimal("0")

        # Determine market maker activity
        rate_int = int(quote_rate)
        if rate_int >= self.very_active:
            mm_activity = MarketMakerActivity.VERY_ACTIVE
        elif rate_int >= self.active:
            mm_activity = MarketMakerActivity.ACTIVE
        elif rate_int >= self.reduced:
            mm_activity = MarketMakerActivity.NORMAL
        elif rate_int >= self.absent:
            mm_activity = MarketMakerActivity.REDUCED
        else:
            mm_activity = MarketMakerActivity.ABSENT

        return QuoteActivity(
            symbol=symbol,
            timestamp=timestamp,
            window_seconds=window_seconds,
            quote_updates=quote_updates,
            quote_rate_per_second=quote_rate,
            bid_updates=bid_updates,
            ask_updates=ask_updates,
            quote_to_trade_ratio=quote_to_trade,
            average_quote_lifetime_ms=avg_lifetime,
            mm_activity=mm_activity
        )


class PriceImpactEstimator:
    """Estimates price impact of orders."""

    def __init__(
        self,
        impact_coefficient: Decimal = Decimal("0.1"),
        permanent_impact_ratio: Decimal = Decimal("0.5")
    ):
        """
        Initialize price impact estimator.

        Args:
            impact_coefficient: Coefficient for impact calculation
            permanent_impact_ratio: Ratio of permanent to temporary impact
        """
        self.impact_coefficient = impact_coefficient
        self.permanent_ratio = permanent_impact_ratio

    def estimate_impact(
        self,
        symbol: str,
        order_size: Decimal,
        side: str,
        orderbook: list[tuple[Decimal, Decimal]],  # [(price, size), ...]
        mid_price: Decimal,
        daily_volume: Decimal | None = None,
        timestamp: datetime | None = None
    ) -> PriceImpact:
        """
        Estimate price impact of an order.

        Args:
            symbol: Trading symbol
            order_size: Order size in base currency
            side: "buy" or "sell"
            orderbook: Order book (asks for buy, bids for sell)
            mid_price: Current mid price
            daily_volume: Average daily volume (optional)
            timestamp: Timestamp (default: now)

        Returns:
            PriceImpact estimation
        """
        timestamp = timestamp or datetime.now()

        # Calculate slippage by walking the book
        remaining_size = order_size
        total_cost = Decimal("0")

        for price, size in orderbook:
            fill_size = min(remaining_size, size)
            total_cost += fill_size * price
            remaining_size -= fill_size

            if remaining_size <= 0:
                break

        # If order is larger than book
        if remaining_size > 0:
            # Estimate additional impact
            if orderbook:
                last_price = orderbook[-1][0]
                extra_impact = remaining_size * last_price * Decimal("1.1")  # 10% beyond book
                total_cost += extra_impact
            else:
                total_cost += remaining_size * mid_price * Decimal("1.5")  # 50% impact

        # Calculate metrics
        if order_size > 0:
            avg_fill_price = total_cost / order_size
        else:
            avg_fill_price = mid_price

        slippage = abs(avg_fill_price - mid_price)
        slippage_bps = (slippage / mid_price) * 10000 if mid_price > 0 else Decimal("0")

        # Estimate market impact using square root model
        if daily_volume and daily_volume > 0:
            participation_rate = (order_size * mid_price) / daily_volume
            # Square root impact model
            market_impact_bps = self.impact_coefficient * (participation_rate * 10000).sqrt()
        else:
            # Fallback to slippage-based estimate
            market_impact_bps = slippage_bps * Decimal("0.5")

        # Effective spread includes half-spread plus impact
        effective_spread_bps = slippage_bps + market_impact_bps

        # Total cost
        market_impact_cost = mid_price * order_size * (effective_spread_bps / 10000)

        return PriceImpact(
            symbol=symbol,
            timestamp=timestamp,
            order_size=order_size,
            side=side,
            estimated_impact_bps=market_impact_bps,
            estimated_slippage=slippage,
            effective_spread_bps=effective_spread_bps,
            market_impact_cost=market_impact_cost
        )


class MicrostructureAnalyzer:
    """Main microstructure analysis coordinator."""

    def __init__(
        self,
        spread_analyzer: SpreadAnalyzer | None = None,
        depth_analyzer: DepthAnalyzer | None = None,
        trade_flow_analyzer: TradeFlowAnalyzer | None = None,
        quote_analyzer: QuoteAnalyzer | None = None,
        price_impact_estimator: PriceImpactEstimator | None = None
    ):
        """
        Initialize microstructure analyzer.

        Args:
            spread_analyzer: SpreadAnalyzer instance
            depth_analyzer: DepthAnalyzer instance
            trade_flow_analyzer: TradeFlowAnalyzer instance
            quote_analyzer: QuoteAnalyzer instance
            price_impact_estimator: PriceImpactEstimator instance
        """
        self.spread = spread_analyzer or SpreadAnalyzer()
        self.depth = depth_analyzer or DepthAnalyzer()
        self.trade_flow = trade_flow_analyzer or TradeFlowAnalyzer()
        self.quote = quote_analyzer or QuoteAnalyzer()
        self.price_impact = price_impact_estimator or PriceImpactEstimator()

        self._snapshot_history: dict[str, list[MicrostructureSnapshot]] = {}
        self._max_history = 500

    def analyze(
        self,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        reference_order_size: Decimal | None = None,
        daily_volume: Decimal | None = None,
        timestamp: datetime | None = None
    ) -> MicrostructureSnapshot:
        """
        Perform complete microstructure analysis.

        Args:
            symbol: Trading symbol
            bids: Order book bids [(price, size), ...]
            asks: Order book asks [(price, size), ...]
            reference_order_size: Size for impact estimation
            daily_volume: Average daily volume
            timestamp: Analysis timestamp

        Returns:
            Complete MicrostructureSnapshot
        """
        timestamp = timestamp or datetime.now()

        # Get BBO
        bid_price = bids[0][0] if bids else Decimal("0")
        ask_price = asks[0][0] if asks else Decimal("0")

        # Analyze spread
        spread_metrics = self.spread.analyze_spread(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            timestamp=timestamp
        )

        # Analyze depth
        depth_metrics = self.depth.analyze_depth(
            symbol=symbol,
            bids=bids,
            asks=asks,
            mid_price=spread_metrics.mid_price,
            timestamp=timestamp
        )

        # Get trade imbalance if we have data
        trade_imbalance = self.trade_flow.analyze_imbalance(
            symbol=symbol,
            window_seconds=60,
            timestamp=timestamp
        )

        # Get quote activity if we have data
        trade_count = None
        if trade_imbalance:
            trade_count = trade_imbalance.buy_trades + trade_imbalance.sell_trades

        quote_activity = self.quote.analyze_activity(
            symbol=symbol,
            window_seconds=60,
            trade_count=trade_count,
            timestamp=timestamp
        )

        # Estimate price impact if order size provided
        price_impact_buy = None
        price_impact_sell = None

        if reference_order_size and reference_order_size > 0:
            if asks:
                price_impact_buy = self.price_impact.estimate_impact(
                    symbol=symbol,
                    order_size=reference_order_size,
                    side="buy",
                    orderbook=asks,
                    mid_price=spread_metrics.mid_price,
                    daily_volume=daily_volume,
                    timestamp=timestamp
                )

            if bids:
                # Reverse bids for sell impact
                price_impact_sell = self.price_impact.estimate_impact(
                    symbol=symbol,
                    order_size=reference_order_size,
                    side="sell",
                    orderbook=bids,
                    mid_price=spread_metrics.mid_price,
                    daily_volume=daily_volume,
                    timestamp=timestamp
                )

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            spread_metrics=spread_metrics,
            depth_metrics=depth_metrics,
            trade_imbalance=trade_imbalance,
            quote_activity=quote_activity
        )

        snapshot = MicrostructureSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            spread_metrics=spread_metrics,
            depth_metrics=depth_metrics,
            trade_imbalance=trade_imbalance,
            quote_activity=quote_activity,
            price_impact_buy=price_impact_buy,
            price_impact_sell=price_impact_sell,
            overall_quality_score=quality_score
        )

        # Store history
        if symbol not in self._snapshot_history:
            self._snapshot_history[symbol] = []

        self._snapshot_history[symbol].append(snapshot)

        if len(self._snapshot_history[symbol]) > self._max_history:
            self._snapshot_history[symbol] = self._snapshot_history[symbol][-self._max_history:]

        logger.debug(f"Microstructure snapshot for {symbol}: quality={quality_score}")

        return snapshot

    def _calculate_quality_score(
        self,
        spread_metrics: SpreadMetrics,
        depth_metrics: DepthMetrics,
        trade_imbalance: TradeImbalance | None,
        quote_activity: QuoteActivity | None
    ) -> Decimal:
        """
        Calculate overall market quality score.

        Args:
            spread_metrics: Spread analysis
            depth_metrics: Depth analysis
            trade_imbalance: Trade flow analysis
            quote_activity: Quote analysis

        Returns:
            Quality score from 0 to 100
        """
        score = Decimal("0")

        # Spread component (30 points max)
        spread_score = Decimal("30")
        if spread_metrics.spread_regime == SpreadRegime.TIGHT:
            spread_score = Decimal("30")
        elif spread_metrics.spread_regime == SpreadRegime.NORMAL:
            spread_score = Decimal("22")
        elif spread_metrics.spread_regime == SpreadRegime.WIDE:
            spread_score = Decimal("12")
        else:
            spread_score = Decimal("5")

        if spread_metrics.is_crossed:
            spread_score = Decimal("0")

        score += spread_score

        # Liquidity component (30 points max)
        liquidity_score = Decimal("30")
        if depth_metrics.liquidity_state == LiquidityState.HIGHLY_LIQUID:
            liquidity_score = Decimal("30")
        elif depth_metrics.liquidity_state == LiquidityState.LIQUID:
            liquidity_score = Decimal("24")
        elif depth_metrics.liquidity_state == LiquidityState.NORMAL:
            liquidity_score = Decimal("18")
        elif depth_metrics.liquidity_state == LiquidityState.ILLIQUID:
            liquidity_score = Decimal("10")
        else:
            liquidity_score = Decimal("3")

        score += liquidity_score

        # Flow toxicity component (20 points max)
        if trade_imbalance:
            toxicity_score = Decimal("20")
            if trade_imbalance.flow_toxicity == FlowToxicity.VERY_LOW:
                toxicity_score = Decimal("20")
            elif trade_imbalance.flow_toxicity == FlowToxicity.LOW:
                toxicity_score = Decimal("16")
            elif trade_imbalance.flow_toxicity == FlowToxicity.MODERATE:
                toxicity_score = Decimal("12")
            elif trade_imbalance.flow_toxicity == FlowToxicity.HIGH:
                toxicity_score = Decimal("6")
            else:
                toxicity_score = Decimal("2")

            score += toxicity_score
        else:
            score += Decimal("10")  # Neutral if no data

        # Market maker activity component (20 points max)
        if quote_activity:
            mm_score = Decimal("20")
            if quote_activity.mm_activity == MarketMakerActivity.VERY_ACTIVE:
                mm_score = Decimal("20")
            elif quote_activity.mm_activity == MarketMakerActivity.ACTIVE:
                mm_score = Decimal("16")
            elif quote_activity.mm_activity == MarketMakerActivity.NORMAL:
                mm_score = Decimal("12")
            elif quote_activity.mm_activity == MarketMakerActivity.REDUCED:
                mm_score = Decimal("6")
            else:
                mm_score = Decimal("2")

            score += mm_score
        else:
            score += Decimal("10")  # Neutral if no data

        return score

    def get_quality_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> list[tuple[datetime, Decimal]]:
        """
        Get quality score history.

        Args:
            symbol: Trading symbol
            limit: Maximum number of entries

        Returns:
            List of (timestamp, quality_score) tuples
        """
        if symbol not in self._snapshot_history:
            return []

        history = self._snapshot_history[symbol][-limit:]
        return [(s.timestamp, s.overall_quality_score) for s in history]

    def record_trade(
        self,
        symbol: str,
        price: Decimal,
        size: Decimal,
        side: str,
        timestamp: datetime | None = None
    ) -> None:
        """
        Record a trade for flow analysis.

        Args:
            symbol: Trading symbol
            price: Trade price
            size: Trade size
            side: "buy" or "sell"
            timestamp: Trade timestamp
        """
        self.trade_flow.record_trade(
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            timestamp=timestamp
        )

    def record_quote(
        self,
        symbol: str,
        bid_price: Decimal,
        ask_price: Decimal,
        bid_size: Decimal,
        ask_size: Decimal,
        timestamp: datetime | None = None
    ) -> None:
        """
        Record a quote for activity analysis.

        Args:
            symbol: Trading symbol
            bid_price: Bid price
            ask_price: Ask price
            bid_size: Bid size
            ask_size: Ask size
            timestamp: Quote timestamp
        """
        self.quote.record_quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=timestamp
        )
