"""Smart Order Router module.

This module provides intelligent order routing including:
- Order routing based on market conditions
- Order splitting strategies (TWAP, VWAP, POV)
- Price impact estimation
- Liquidity-aware routing
- Order scheduling and timing optimization
- Multi-venue routing simulation
"""

import time
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class OrderType(Enum):
    """Order type classification."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of volume
    ICEBERG = "iceberg"


class RoutingStrategy(Enum):
    """Order routing strategy."""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    NEUTRAL = "neutral"
    ADAPTIVE = "adaptive"
    STEALTH = "stealth"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class ExecutionUrgency(Enum):
    """Execution urgency level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketState:
    """Current market state for routing decisions."""

    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    last_price: Decimal
    daily_volume: Decimal = Decimal("0")
    volatility: float = 0.0
    spread_bps: float = 0.0
    liquidity_score: float = 50.0

    def __post_init__(self):
        """Calculate derived values."""
        if self.spread_bps == 0.0 and self.bid_price > 0:
            mid_price = (self.bid_price + self.ask_price) / 2
            spread = self.ask_price - self.bid_price
            self.spread_bps = float(spread / mid_price * 10000)

    @property
    def mid_price(self) -> Decimal:
        """Get mid price."""
        return (self.bid_price + self.ask_price) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bid_price": str(self.bid_price),
            "ask_price": str(self.ask_price),
            "bid_size": str(self.bid_size),
            "ask_size": str(self.ask_size),
            "last_price": str(self.last_price),
            "daily_volume": str(self.daily_volume),
            "volatility": self.volatility,
            "spread_bps": self.spread_bps,
            "liquidity_score": self.liquidity_score,
        }


@dataclass
class OrderSlice:
    """Individual order slice in a split execution."""

    sequence: int
    size: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.LIMIT
    scheduled_time: Optional[float] = None
    executed: bool = False
    fill_price: Optional[Decimal] = None
    fill_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "size": str(self.size),
            "price": str(self.price) if self.price else None,
            "order_type": self.order_type.value,
            "scheduled_time": self.scheduled_time,
            "executed": self.executed,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "fill_time": self.fill_time,
        }


@dataclass
class RoutingDecision:
    """Routing decision for an order."""

    strategy: RoutingStrategy
    order_type: OrderType
    price: Optional[Decimal] = None
    slices: List[OrderSlice] = field(default_factory=list)
    total_size: Decimal = Decimal("0")
    estimated_cost: Decimal = Decimal("0")
    estimated_slippage_bps: float = 0.0
    urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "order_type": self.order_type.value,
            "price": str(self.price) if self.price else None,
            "slices": [s.to_dict() for s in self.slices],
            "total_size": str(self.total_size),
            "estimated_cost": str(self.estimated_cost),
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "urgency": self.urgency.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order."""

    market: str
    side: OrderSide
    total_size: Decimal
    decision: RoutingDecision
    start_time: float
    end_time: float
    participation_rate: float = 0.0  # For POV
    interval_seconds: float = 60.0  # For TWAP
    price_band_pct: float = 0.5  # Price tolerance
    min_slice_size: Decimal = Decimal("0")
    max_slice_size: Decimal = Decimal("0")
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "side": self.side.value,
            "total_size": str(self.total_size),
            "decision": self.decision.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "participation_rate": self.participation_rate,
            "interval_seconds": self.interval_seconds,
            "price_band_pct": self.price_band_pct,
            "status": self.status,
        }


@dataclass
class ExecutionMetrics:
    """Metrics for execution quality."""

    avg_fill_price: Decimal = Decimal("0")
    vwap: Decimal = Decimal("0")
    slippage_bps: float = 0.0
    participation_rate: float = 0.0
    fill_rate: float = 0.0
    execution_time: float = 0.0
    num_fills: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_fill_price": str(self.avg_fill_price),
            "vwap": str(self.vwap),
            "slippage_bps": self.slippage_bps,
            "participation_rate": self.participation_rate,
            "fill_rate": self.fill_rate,
            "execution_time": self.execution_time,
            "num_fills": self.num_fills,
        }


class SmartOrderRouter:
    """Smart order router for intelligent order execution."""

    def __init__(
        self,
        default_slice_count: int = 10,
        max_impact_bps: float = 10.0,
        stealth_threshold_pct: float = 5.0,
    ):
        """Initialize smart order router.

        Args:
            default_slice_count: Default number of slices for split orders
            max_impact_bps: Maximum acceptable price impact in basis points
            stealth_threshold_pct: Volume threshold for stealth execution
        """
        self.default_slice_count = default_slice_count
        self.max_impact_bps = max_impact_bps
        self.stealth_threshold_pct = stealth_threshold_pct

        self._market_states: Dict[str, MarketState] = {}
        self._execution_plans: Dict[str, ExecutionPlan] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []

    def update_market_state(self, market: str, state: MarketState) -> None:
        """Update market state for routing decisions.

        Args:
            market: Market symbol
            state: Current market state
        """
        self._market_states[market] = state

    def get_market_state(self, market: str) -> Optional[MarketState]:
        """Get market state.

        Args:
            market: Market symbol

        Returns:
            Market state or None
        """
        return self._market_states.get(market)

    def estimate_price_impact(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
    ) -> float:
        """Estimate price impact of an order.

        Args:
            market: Market symbol
            side: Order side
            size: Order size

        Returns:
            Estimated price impact in basis points
        """
        state = self._market_states.get(market)
        if not state:
            return 0.0

        # Simple price impact model based on liquidity
        if side == OrderSide.BUY:
            available_liquidity = state.ask_size
        else:
            available_liquidity = state.bid_size

        if available_liquidity <= 0:
            return float("inf")

        # Impact increases with order size relative to liquidity
        size_ratio = float(size / available_liquidity)

        # Square-root market impact model
        base_impact = math.sqrt(size_ratio) * 100  # bps

        # Adjust for volatility
        volatility_factor = 1 + state.volatility

        # Adjust for spread
        spread_factor = 1 + state.spread_bps / 100

        return base_impact * volatility_factor * spread_factor

    def estimate_execution_cost(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> Decimal:
        """Estimate total execution cost.

        Args:
            market: Market symbol
            side: Order side
            size: Order size
            order_type: Order type

        Returns:
            Estimated cost in quote currency
        """
        state = self._market_states.get(market)
        if not state:
            return Decimal("0")

        # Base cost from spread
        half_spread = (state.ask_price - state.bid_price) / 2
        spread_cost = half_spread * size

        # Price impact cost
        impact_bps = self.estimate_price_impact(market, side, size)
        impact_cost = state.mid_price * size * Decimal(str(impact_bps / 10000))

        # Adjust for order type
        if order_type == OrderType.LIMIT:
            # Limit orders may avoid spread cost
            spread_cost = spread_cost * Decimal("0.2")
        elif order_type in [OrderType.TWAP, OrderType.VWAP]:
            # Split orders reduce impact
            spread_cost = spread_cost * Decimal("0.5")
            impact_cost = impact_cost * Decimal("0.6")

        return spread_cost + impact_cost

    def determine_strategy(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
    ) -> RoutingStrategy:
        """Determine optimal routing strategy.

        Args:
            market: Market symbol
            side: Order side
            size: Order size
            urgency: Execution urgency

        Returns:
            Recommended routing strategy
        """
        state = self._market_states.get(market)
        if not state:
            return RoutingStrategy.NEUTRAL

        # Calculate size as percentage of daily volume
        if state.daily_volume > 0:
            volume_pct = float(size / state.daily_volume) * 100
        else:
            volume_pct = 0

        # Large orders need stealth
        if volume_pct > self.stealth_threshold_pct:
            return RoutingStrategy.STEALTH

        # High urgency = aggressive
        if urgency == ExecutionUrgency.CRITICAL:
            return RoutingStrategy.AGGRESSIVE
        elif urgency == ExecutionUrgency.HIGH:
            return RoutingStrategy.AGGRESSIVE

        # Low liquidity = adaptive
        if state.liquidity_score < 30:
            return RoutingStrategy.ADAPTIVE

        # Wide spread = passive
        if state.spread_bps > 20:
            return RoutingStrategy.PASSIVE

        # High volatility = adaptive
        if state.volatility > 0.03:
            return RoutingStrategy.ADAPTIVE

        return RoutingStrategy.NEUTRAL

    def determine_order_type(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
        strategy: RoutingStrategy,
    ) -> OrderType:
        """Determine optimal order type.

        Args:
            market: Market symbol
            side: Order side
            size: Order size
            strategy: Routing strategy

        Returns:
            Recommended order type
        """
        state = self._market_states.get(market)
        if not state:
            return OrderType.LIMIT

        # Calculate size as percentage of liquidity
        if side == OrderSide.BUY:
            liquidity = state.ask_size
        else:
            liquidity = state.bid_size

        if liquidity > 0:
            size_ratio = float(size / liquidity)
        else:
            size_ratio = float("inf")

        # Large orders need splitting
        if size_ratio > 1.0:
            if strategy == RoutingStrategy.STEALTH:
                return OrderType.ICEBERG
            else:
                return OrderType.TWAP

        # Strategy-based selection
        if strategy == RoutingStrategy.AGGRESSIVE:
            return OrderType.MARKET
        elif strategy == RoutingStrategy.PASSIVE:
            return OrderType.LIMIT
        elif strategy == RoutingStrategy.STEALTH:
            return OrderType.ICEBERG

        # Default to limit for better control
        return OrderType.LIMIT

    def create_twap_slices(
        self,
        total_size: Decimal,
        duration_seconds: float,
        interval_seconds: float = 60.0,
        randomize: bool = True,
    ) -> List[OrderSlice]:
        """Create TWAP order slices.

        Args:
            total_size: Total order size
            duration_seconds: Execution duration
            interval_seconds: Time between slices
            randomize: Add randomization to sizes/times

        Returns:
            List of order slices
        """
        num_slices = max(1, int(duration_seconds / interval_seconds))
        base_size = total_size / num_slices

        slices = []
        current_time = time.time()

        for i in range(num_slices):
            # Add randomization if requested
            if randomize:
                size_variance = float(base_size) * 0.2  # ±20%
                import random
                size_adj = Decimal(str(random.uniform(-size_variance, size_variance)))
                size = base_size + size_adj

                time_variance = interval_seconds * 0.3  # ±30%
                time_adj = random.uniform(-time_variance, time_variance)
            else:
                size = base_size
                time_adj = 0

            scheduled_time = current_time + (i * interval_seconds) + time_adj

            slices.append(
                OrderSlice(
                    sequence=i + 1,
                    size=max(Decimal("0.001"), size),  # Minimum size
                    order_type=OrderType.LIMIT,
                    scheduled_time=scheduled_time,
                )
            )

        return slices

    def create_vwap_slices(
        self,
        total_size: Decimal,
        volume_profile: List[float],
        duration_seconds: float,
    ) -> List[OrderSlice]:
        """Create VWAP order slices based on volume profile.

        Args:
            total_size: Total order size
            volume_profile: Expected volume distribution (sums to 1)
            duration_seconds: Execution duration

        Returns:
            List of order slices
        """
        if not volume_profile:
            # Default to uniform distribution
            volume_profile = [1.0 / 10] * 10

        # Normalize profile
        total_weight = sum(volume_profile)
        if total_weight > 0:
            volume_profile = [v / total_weight for v in volume_profile]

        num_slices = len(volume_profile)
        interval = duration_seconds / num_slices
        current_time = time.time()

        slices = []
        for i, weight in enumerate(volume_profile):
            size = total_size * Decimal(str(weight))

            slices.append(
                OrderSlice(
                    sequence=i + 1,
                    size=max(Decimal("0.001"), size),
                    order_type=OrderType.LIMIT,
                    scheduled_time=current_time + (i * interval),
                )
            )

        return slices

    def create_pov_slices(
        self,
        total_size: Decimal,
        participation_rate: float,
        estimated_market_volume: Decimal,
        duration_seconds: float,
    ) -> List[OrderSlice]:
        """Create POV (Percentage of Volume) order slices.

        Args:
            total_size: Total order size
            participation_rate: Target participation rate (0-1)
            estimated_market_volume: Expected market volume
            duration_seconds: Execution duration

        Returns:
            List of order slices
        """
        # Calculate expected execution rate
        volume_per_second = estimated_market_volume / Decimal(str(duration_seconds))
        size_per_second = volume_per_second * Decimal(str(participation_rate))

        # Create slices every minute
        interval = 60.0
        num_slices = max(1, int(duration_seconds / interval))
        size_per_slice = size_per_second * Decimal(str(interval))

        current_time = time.time()
        slices = []
        remaining = total_size

        for i in range(num_slices):
            size = min(remaining, size_per_slice)
            if size <= 0:
                break

            slices.append(
                OrderSlice(
                    sequence=i + 1,
                    size=size,
                    order_type=OrderType.LIMIT,
                    scheduled_time=current_time + (i * interval),
                )
            )
            remaining -= size

        return slices

    def create_iceberg_slices(
        self,
        total_size: Decimal,
        visible_size: Decimal,
    ) -> List[OrderSlice]:
        """Create iceberg order slices.

        Args:
            total_size: Total order size
            visible_size: Visible slice size

        Returns:
            List of order slices (only one visible at a time)
        """
        num_slices = max(1, int(total_size / visible_size) + 1)
        slices = []
        remaining = total_size

        for i in range(num_slices):
            size = min(remaining, visible_size)
            if size <= 0:
                break

            slices.append(
                OrderSlice(
                    sequence=i + 1,
                    size=size,
                    order_type=OrderType.LIMIT,
                    # No scheduled time - triggered when previous fills
                )
            )
            remaining -= size

        return slices

    def route_order(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
        duration_seconds: Optional[float] = None,
    ) -> RoutingDecision:
        """Route an order intelligently.

        Args:
            market: Market symbol
            side: Order side
            size: Order size
            urgency: Execution urgency
            duration_seconds: Optional execution duration

        Returns:
            Routing decision
        """
        # Determine strategy
        strategy = self.determine_strategy(market, side, size, urgency)

        # Determine order type
        order_type = self.determine_order_type(market, side, size, strategy)

        # Get market state
        state = self._market_states.get(market)

        # Calculate price
        price = None
        if state:
            if side == OrderSide.BUY:
                if strategy == RoutingStrategy.AGGRESSIVE:
                    price = state.ask_price
                elif strategy == RoutingStrategy.PASSIVE:
                    price = state.bid_price
                else:
                    price = state.mid_price
            else:
                if strategy == RoutingStrategy.AGGRESSIVE:
                    price = state.bid_price
                elif strategy == RoutingStrategy.PASSIVE:
                    price = state.ask_price
                else:
                    price = state.mid_price

        # Create slices if needed
        slices = []
        if order_type == OrderType.TWAP:
            duration = duration_seconds or 3600  # Default 1 hour
            slices = self.create_twap_slices(size, duration)
        elif order_type == OrderType.VWAP:
            duration = duration_seconds or 3600
            slices = self.create_vwap_slices(size, [], duration)
        elif order_type == OrderType.ICEBERG:
            visible_size = size * Decimal("0.1")  # Show 10%
            slices = self.create_iceberg_slices(size, visible_size)

        # Estimate costs
        estimated_cost = self.estimate_execution_cost(market, side, size, order_type)
        estimated_slippage = self.estimate_price_impact(market, side, size)

        # Create decision
        decision = RoutingDecision(
            strategy=strategy,
            order_type=order_type,
            price=price,
            slices=slices,
            total_size=size,
            estimated_cost=estimated_cost,
            estimated_slippage_bps=estimated_slippage,
            urgency=urgency,
            reason=self._generate_reason(strategy, order_type, state),
        )

        # Notify callbacks
        for callback in self._callbacks:
            callback(market, decision)

        return decision

    def _generate_reason(
        self,
        strategy: RoutingStrategy,
        order_type: OrderType,
        state: Optional[MarketState],
    ) -> str:
        """Generate human-readable reason for routing decision."""
        reasons = []

        if strategy == RoutingStrategy.AGGRESSIVE:
            reasons.append("High urgency execution")
        elif strategy == RoutingStrategy.PASSIVE:
            reasons.append("Wide spread - using passive approach")
        elif strategy == RoutingStrategy.STEALTH:
            reasons.append("Large order - using stealth execution")
        elif strategy == RoutingStrategy.ADAPTIVE:
            reasons.append("Market conditions require adaptive approach")

        if order_type == OrderType.TWAP:
            reasons.append("Splitting order over time (TWAP)")
        elif order_type == OrderType.VWAP:
            reasons.append("Volume-weighted execution (VWAP)")
        elif order_type == OrderType.ICEBERG:
            reasons.append("Hidden size (Iceberg)")

        if state:
            if state.spread_bps > 20:
                reasons.append(f"Wide spread: {state.spread_bps:.1f}bps")
            if state.volatility > 0.02:
                reasons.append(f"Elevated volatility: {state.volatility:.1%}")

        return "; ".join(reasons) if reasons else "Standard execution"

    def create_execution_plan(
        self,
        market: str,
        side: OrderSide,
        size: Decimal,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
    ) -> ExecutionPlan:
        """Create a complete execution plan.

        Args:
            market: Market symbol
            side: Order side
            size: Order size
            start_time: When to start execution
            end_time: When to complete execution
            urgency: Execution urgency

        Returns:
            Execution plan
        """
        start = start_time or time.time()
        end = end_time or (start + 3600)  # Default 1 hour
        duration = end - start

        # Get routing decision
        decision = self.route_order(market, side, size, urgency, duration)

        # Create plan
        plan = ExecutionPlan(
            market=market,
            side=side,
            total_size=size,
            decision=decision,
            start_time=start,
            end_time=end,
            interval_seconds=duration / max(1, len(decision.slices)),
        )

        # Store plan
        plan_id = f"{market}_{int(start)}"
        self._execution_plans[plan_id] = plan

        return plan

    def get_execution_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get execution plan by ID."""
        return self._execution_plans.get(plan_id)

    def get_active_plans(self) -> List[ExecutionPlan]:
        """Get all active execution plans."""
        current_time = time.time()
        return [
            plan
            for plan in self._execution_plans.values()
            if plan.start_time <= current_time <= plan.end_time
            and plan.status != "completed"
        ]

    def calculate_execution_metrics(
        self,
        fills: List[Dict[str, Any]],
        benchmark_price: Decimal,
    ) -> ExecutionMetrics:
        """Calculate execution quality metrics.

        Args:
            fills: List of fill data
            benchmark_price: Benchmark price for comparison

        Returns:
            Execution metrics
        """
        if not fills:
            return ExecutionMetrics()

        # Calculate average fill price
        total_value = sum(
            Decimal(str(f.get("price", 0))) * Decimal(str(f.get("size", 0)))
            for f in fills
        )
        total_size = sum(Decimal(str(f.get("size", 0))) for f in fills)

        avg_fill_price = total_value / total_size if total_size > 0 else Decimal("0")

        # Calculate slippage
        if benchmark_price > 0:
            slippage_bps = float(
                (avg_fill_price - benchmark_price) / benchmark_price * 10000
            )
        else:
            slippage_bps = 0

        # Calculate execution time
        times = [f.get("timestamp", 0) for f in fills]
        execution_time = max(times) - min(times) if times else 0

        return ExecutionMetrics(
            avg_fill_price=avg_fill_price,
            vwap=avg_fill_price,  # Simplified
            slippage_bps=abs(slippage_bps),
            fill_rate=100.0 if fills else 0.0,
            execution_time=execution_time,
            num_fills=len(fills),
        )

    def add_callback(self, callback: Callable[[str, RoutingDecision], None]) -> None:
        """Add routing decision callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_markets(self) -> List[str]:
        """Get list of markets with state."""
        return list(self._market_states.keys())

    def clear_market(self, market: str) -> None:
        """Clear market data."""
        if market in self._market_states:
            del self._market_states[market]

    def clear_all(self) -> None:
        """Clear all data."""
        self._market_states.clear()
        self._execution_plans.clear()


class OrderRouter:
    """Multi-market order router."""

    def __init__(self):
        """Initialize order router."""
        self._router = SmartOrderRouter()
        self._market_priority: Dict[str, int] = {}

    def set_market_priority(self, market: str, priority: int) -> None:
        """Set market priority for routing.

        Args:
            market: Market symbol
            priority: Priority (higher = preferred)
        """
        self._market_priority[market] = priority

    def get_market_priority(self, market: str) -> int:
        """Get market priority."""
        return self._market_priority.get(market, 0)

    def update_market_state(self, market: str, state: MarketState) -> None:
        """Update market state."""
        self._router.update_market_state(market, state)

    def route(
        self,
        markets: List[str],
        side: OrderSide,
        total_size: Decimal,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
    ) -> Dict[str, RoutingDecision]:
        """Route order across multiple markets.

        Args:
            markets: List of market symbols
            side: Order side
            total_size: Total size to execute
            urgency: Execution urgency

        Returns:
            Dict of market -> routing decision
        """
        if not markets:
            return {}

        # Sort by priority and liquidity
        scored_markets = []
        for market in markets:
            state = self._router.get_market_state(market)
            priority = self._market_priority.get(market, 0)
            liquidity = float(state.liquidity_score) if state else 0
            score = priority * 100 + liquidity
            scored_markets.append((market, score, state))

        scored_markets.sort(key=lambda x: x[1], reverse=True)

        # Allocate size based on liquidity
        decisions = {}
        remaining = total_size

        for market, score, state in scored_markets:
            if remaining <= 0:
                break

            # Calculate allocation
            if state and state.daily_volume > 0:
                max_allocation = state.daily_volume * Decimal("0.01")  # Max 1% of volume
                allocation = min(remaining, max_allocation)
            else:
                allocation = remaining / len(markets)

            if allocation > 0:
                decision = self._router.route_order(market, side, allocation, urgency)
                decisions[market] = decision
                remaining -= allocation

        return decisions

    def get_best_market(
        self,
        markets: List[str],
        side: OrderSide,
    ) -> Optional[str]:
        """Get best market for execution.

        Args:
            markets: List of market symbols
            side: Order side

        Returns:
            Best market symbol or None
        """
        best_market = None
        best_score = float("-inf")

        for market in markets:
            state = self._router.get_market_state(market)
            if not state:
                continue

            # Score based on spread, liquidity, and priority
            score = -state.spread_bps + state.liquidity_score
            score += self._market_priority.get(market, 0) * 10

            if side == OrderSide.BUY:
                score += float(state.ask_size)
            else:
                score += float(state.bid_size)

            if score > best_score:
                best_score = score
                best_market = market

        return best_market


# Global smart order router instance
_smart_router: Optional[SmartOrderRouter] = None


def get_smart_router() -> SmartOrderRouter:
    """Get global smart order router."""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartOrderRouter()
    return _smart_router


def reset_smart_router() -> None:
    """Reset global smart order router."""
    global _smart_router
    _smart_router = None
