"""
Execution Optimizer Module.

Optimizes order execution by analyzing market conditions,
predicting execution outcomes, and selecting the best
execution strategy based on various cost models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable


class ExecutionObjective(Enum):
    """Execution optimization objective."""

    MINIMIZE_COST = "minimize_cost"  # Minimize total execution cost
    MINIMIZE_IMPACT = "minimize_impact"  # Minimize market impact
    MINIMIZE_TIME = "minimize_time"  # Minimize execution time
    MINIMIZE_RISK = "minimize_risk"  # Minimize execution risk
    MAXIMIZE_FILL = "maximize_fill"  # Maximize fill rate
    BALANCE = "balance"  # Balance all objectives


class ExecutionStrategy(Enum):
    """Execution strategy type."""

    IMMEDIATE = "immediate"  # Execute immediately at market
    PASSIVE = "passive"  # Use limit orders, wait for fills
    AGGRESSIVE = "aggressive"  # Take liquidity aggressively
    ADAPTIVE = "adaptive"  # Adapt based on market conditions
    SCHEDULED = "scheduled"  # Schedule execution over time
    OPPORTUNISTIC = "opportunistic"  # Wait for favorable conditions


class MarketCondition(Enum):
    """Current market condition."""

    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    AUCTION = "auction"


class ExecutionRisk(Enum):
    """Execution risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionCost:
    """Breakdown of execution costs."""

    spread_cost: Decimal = Decimal("0")
    impact_cost: Decimal = Decimal("0")
    timing_cost: Decimal = Decimal("0")
    opportunity_cost: Decimal = Decimal("0")
    fee_cost: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")

    @property
    def total_cost(self) -> Decimal:
        """Calculate total execution cost."""
        return (
            self.spread_cost
            + self.impact_cost
            + self.timing_cost
            + self.opportunity_cost
            + self.fee_cost
            + self.slippage_cost
        )

    @property
    def total_cost_bps(self) -> float:
        """Total cost in basis points (if set)."""
        return float(self.total_cost)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread_cost": float(self.spread_cost),
            "impact_cost": float(self.impact_cost),
            "timing_cost": float(self.timing_cost),
            "opportunity_cost": float(self.opportunity_cost),
            "fee_cost": float(self.fee_cost),
            "slippage_cost": float(self.slippage_cost),
            "total_cost": float(self.total_cost),
        }


@dataclass
class ExecutionPrediction:
    """Prediction for execution outcome."""

    strategy: ExecutionStrategy
    expected_price: Decimal
    expected_slippage_bps: float
    expected_fill_rate: float
    expected_duration_seconds: float
    cost_estimate: ExecutionCost
    risk_level: ExecutionRisk
    confidence: float = 0.5
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "expected_price": float(self.expected_price),
            "expected_slippage_bps": self.expected_slippage_bps,
            "expected_fill_rate": self.expected_fill_rate,
            "expected_duration_seconds": self.expected_duration_seconds,
            "cost_estimate": self.cost_estimate.to_dict(),
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class OptimizationResult:
    """Result of execution optimization."""

    market: str
    side: str
    size: Decimal
    objective: ExecutionObjective
    recommended_strategy: ExecutionStrategy
    predictions: list[ExecutionPrediction]
    market_condition: MarketCondition
    optimal_params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "side": self.side,
            "size": float(self.size),
            "objective": self.objective.value,
            "recommended_strategy": self.recommended_strategy.value,
            "predictions": [p.to_dict() for p in self.predictions],
            "market_condition": self.market_condition.value,
            "optimal_params": self.optimal_params,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MarketState:
    """Current market state for optimization."""

    market: str
    mid_price: Decimal
    spread_bps: float
    volatility: float  # Daily volatility in %
    liquidity_score: float  # 0-1 score
    volume_24h: Decimal
    bid_depth: Decimal
    ask_depth: Decimal
    trend_direction: float  # -1 to 1
    imbalance: float  # 0-1, 0.5 = balanced
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "mid_price": float(self.mid_price),
            "spread_bps": self.spread_bps,
            "volatility": self.volatility,
            "liquidity_score": self.liquidity_score,
            "volume_24h": float(self.volume_24h),
            "bid_depth": float(self.bid_depth),
            "ask_depth": float(self.ask_depth),
            "trend_direction": self.trend_direction,
            "imbalance": self.imbalance,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionBenchmark:
    """Benchmark for execution performance."""

    strategy: ExecutionStrategy
    avg_slippage_bps: float
    avg_fill_rate: float
    avg_duration_seconds: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "avg_slippage_bps": self.avg_slippage_bps,
            "avg_fill_rate": self.avg_fill_rate,
            "avg_duration_seconds": self.avg_duration_seconds,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ExecutionRecord:
    """Record of a single execution for analysis."""

    market: str
    side: str
    size: Decimal
    strategy: ExecutionStrategy
    entry_price: Decimal
    avg_fill_price: Decimal
    fill_rate: float
    duration_seconds: float
    slippage_bps: float
    cost: ExecutionCost
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "side": self.side,
            "size": float(self.size),
            "strategy": self.strategy.value,
            "entry_price": float(self.entry_price),
            "avg_fill_price": float(self.avg_fill_price),
            "fill_rate": self.fill_rate,
            "duration_seconds": self.duration_seconds,
            "slippage_bps": self.slippage_bps,
            "cost": self.cost.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


class ExecutionOptimizer:
    """Optimizes execution strategy based on market conditions."""

    def __init__(
        self,
        default_objective: ExecutionObjective = ExecutionObjective.BALANCE,
        impact_model_alpha: float = 0.1,
        fee_rate_bps: float = 5.0,
        max_participation_rate: float = 0.25,
    ):
        """
        Initialize optimizer.

        Args:
            default_objective: Default optimization objective
            impact_model_alpha: Market impact model coefficient
            fee_rate_bps: Trading fee rate in basis points
            max_participation_rate: Maximum participation rate of volume
        """
        self.default_objective = default_objective
        self.impact_model_alpha = impact_model_alpha
        self.fee_rate_bps = fee_rate_bps
        self.max_participation_rate = max_participation_rate

        # Market states
        self._market_states: dict[str, MarketState] = {}

        # Execution history
        self._execution_history: list[ExecutionRecord] = []

        # Benchmarks
        self._benchmarks: dict[str, dict[ExecutionStrategy, ExecutionBenchmark]] = {}

        # Strategy weights for different objectives
        self._objective_weights: dict[ExecutionObjective, dict[str, float]] = {
            ExecutionObjective.MINIMIZE_COST: {
                "spread": 0.3,
                "impact": 0.4,
                "timing": 0.1,
                "risk": 0.2,
            },
            ExecutionObjective.MINIMIZE_IMPACT: {
                "spread": 0.1,
                "impact": 0.6,
                "timing": 0.1,
                "risk": 0.2,
            },
            ExecutionObjective.MINIMIZE_TIME: {
                "spread": 0.1,
                "impact": 0.2,
                "timing": 0.5,
                "risk": 0.2,
            },
            ExecutionObjective.MINIMIZE_RISK: {
                "spread": 0.1,
                "impact": 0.2,
                "timing": 0.1,
                "risk": 0.6,
            },
            ExecutionObjective.MAXIMIZE_FILL: {
                "spread": 0.2,
                "impact": 0.2,
                "timing": 0.4,
                "risk": 0.2,
            },
            ExecutionObjective.BALANCE: {
                "spread": 0.25,
                "impact": 0.25,
                "timing": 0.25,
                "risk": 0.25,
            },
        }

        # Callbacks
        self._callbacks: list[Callable[[OptimizationResult], None]] = []

    def update_market_state(self, state: MarketState) -> None:
        """
        Update market state.

        Args:
            state: New market state
        """
        self._market_states[state.market] = state

    def get_market_state(self, market: str) -> MarketState | None:
        """Get market state."""
        return self._market_states.get(market)

    def detect_market_condition(self, market: str) -> MarketCondition:
        """
        Detect current market condition.

        Args:
            market: Market symbol

        Returns:
            Market condition
        """
        state = self._market_states.get(market)
        if not state:
            return MarketCondition.NORMAL

        # High volatility
        if state.volatility > 5.0:  # >5% daily
            return MarketCondition.HIGH_VOLATILITY

        # Low liquidity
        if state.liquidity_score < 0.3:
            return MarketCondition.LOW_LIQUIDITY

        # Trending
        if abs(state.trend_direction) > 0.6:
            return MarketCondition.TRENDING

        # Mean reverting
        if abs(state.trend_direction) < 0.2 and state.volatility < 2.0:
            return MarketCondition.MEAN_REVERTING

        return MarketCondition.NORMAL

    def estimate_impact(
        self,
        market: str,
        size: Decimal,
        side: str,
    ) -> float:
        """
        Estimate market impact in basis points.

        Uses square-root market impact model:
        Impact = alpha * sqrt(size / ADV)

        Args:
            market: Market symbol
            size: Order size
            side: 'buy' or 'sell'

        Returns:
            Estimated impact in basis points
        """
        state = self._market_states.get(market)
        if not state or state.volume_24h <= 0:
            return 10.0  # Default 10 bps

        # Participation rate
        participation = float(size) / float(state.volume_24h)

        # Square root model
        import math
        impact = self.impact_model_alpha * math.sqrt(participation) * 10000

        # Adjust for order book imbalance
        if side == "buy":
            # Buying into ask side
            if state.imbalance > 0.5:
                # More bids than asks, harder to buy
                impact *= 1 + (state.imbalance - 0.5)
        else:
            # Selling into bid side
            if state.imbalance < 0.5:
                # More asks than bids, harder to sell
                impact *= 1 + (0.5 - state.imbalance)

        return impact

    def estimate_cost(
        self,
        market: str,
        size: Decimal,
        side: str,
        strategy: ExecutionStrategy,
    ) -> ExecutionCost:
        """
        Estimate total execution cost.

        Args:
            market: Market symbol
            size: Order size
            side: 'buy' or 'sell'
            strategy: Execution strategy

        Returns:
            Execution cost breakdown
        """
        state = self._market_states.get(market)
        if not state:
            return ExecutionCost(fee_cost=Decimal(str(self.fee_rate_bps)))

        # Spread cost (half spread for immediate, less for passive)
        spread_multiplier = {
            ExecutionStrategy.IMMEDIATE: 0.5,
            ExecutionStrategy.AGGRESSIVE: 0.5,
            ExecutionStrategy.PASSIVE: 0.0,  # Maker orders avoid spread
            ExecutionStrategy.ADAPTIVE: 0.3,
            ExecutionStrategy.SCHEDULED: 0.25,
            ExecutionStrategy.OPPORTUNISTIC: 0.1,
        }
        spread_cost = Decimal(str(state.spread_bps * spread_multiplier.get(strategy, 0.5)))

        # Impact cost
        impact_multiplier = {
            ExecutionStrategy.IMMEDIATE: 1.0,
            ExecutionStrategy.AGGRESSIVE: 1.2,
            ExecutionStrategy.PASSIVE: 0.3,
            ExecutionStrategy.ADAPTIVE: 0.5,
            ExecutionStrategy.SCHEDULED: 0.4,
            ExecutionStrategy.OPPORTUNISTIC: 0.3,
        }
        raw_impact = self.estimate_impact(market, size, side)
        impact_cost = Decimal(str(raw_impact * impact_multiplier.get(strategy, 1.0)))

        # Timing cost (opportunity cost of waiting)
        timing_multiplier = {
            ExecutionStrategy.IMMEDIATE: 0.0,
            ExecutionStrategy.AGGRESSIVE: 0.0,
            ExecutionStrategy.PASSIVE: 2.0,
            ExecutionStrategy.ADAPTIVE: 1.0,
            ExecutionStrategy.SCHEDULED: 1.5,
            ExecutionStrategy.OPPORTUNISTIC: 3.0,
        }
        timing_cost = Decimal(str(state.volatility * 0.1 * timing_multiplier.get(strategy, 1.0)))

        # Fee cost
        fee_cost = Decimal(str(self.fee_rate_bps))

        # Slippage estimate
        slippage_multiplier = {
            ExecutionStrategy.IMMEDIATE: 1.0,
            ExecutionStrategy.AGGRESSIVE: 1.5,
            ExecutionStrategy.PASSIVE: 0.5,
            ExecutionStrategy.ADAPTIVE: 0.7,
            ExecutionStrategy.SCHEDULED: 0.6,
            ExecutionStrategy.OPPORTUNISTIC: 0.4,
        }
        slippage_cost = Decimal(str(raw_impact * 0.3 * slippage_multiplier.get(strategy, 1.0)))

        return ExecutionCost(
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            timing_cost=timing_cost,
            opportunity_cost=Decimal("0"),  # Calculated separately if needed
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
        )

    def predict_execution(
        self,
        market: str,
        size: Decimal,
        side: str,
        strategy: ExecutionStrategy,
    ) -> ExecutionPrediction:
        """
        Predict execution outcome for a strategy.

        Args:
            market: Market symbol
            size: Order size
            side: 'buy' or 'sell'
            strategy: Execution strategy

        Returns:
            Execution prediction
        """
        state = self._market_states.get(market)
        if not state:
            return ExecutionPrediction(
                strategy=strategy,
                expected_price=Decimal("0"),
                expected_slippage_bps=10.0,
                expected_fill_rate=0.9,
                expected_duration_seconds=60.0,
                cost_estimate=ExecutionCost(),
                risk_level=ExecutionRisk.MEDIUM,
                confidence=0.3,
            )

        # Cost estimate
        cost = self.estimate_cost(market, size, side, strategy)

        # Expected price
        impact_bps = float(cost.impact_cost + cost.slippage_cost)
        if side == "buy":
            expected_price = state.mid_price * (1 + Decimal(str(impact_bps / 10000)))
        else:
            expected_price = state.mid_price * (1 - Decimal(str(impact_bps / 10000)))

        # Fill rate estimate
        fill_rates = {
            ExecutionStrategy.IMMEDIATE: 1.0,
            ExecutionStrategy.AGGRESSIVE: 0.99,
            ExecutionStrategy.PASSIVE: 0.7,
            ExecutionStrategy.ADAPTIVE: 0.9,
            ExecutionStrategy.SCHEDULED: 0.95,
            ExecutionStrategy.OPPORTUNISTIC: 0.6,
        }
        fill_rate = fill_rates.get(strategy, 0.9)

        # Adjust for liquidity
        if state.liquidity_score < 0.5:
            fill_rate *= 0.9

        # Duration estimate (seconds)
        durations = {
            ExecutionStrategy.IMMEDIATE: 5.0,
            ExecutionStrategy.AGGRESSIVE: 10.0,
            ExecutionStrategy.PASSIVE: 300.0,
            ExecutionStrategy.ADAPTIVE: 60.0,
            ExecutionStrategy.SCHEDULED: 600.0,
            ExecutionStrategy.OPPORTUNISTIC: 900.0,
        }
        duration = durations.get(strategy, 60.0)

        # Risk level
        risk = self._assess_risk(market, size, strategy)

        # Confidence based on data quality
        confidence = 0.7 if state else 0.3
        # Reduce confidence if using benchmarks
        benchmark = self._get_benchmark(market, strategy)
        if benchmark and benchmark.sample_count > 10:
            confidence = min(confidence + 0.2, 0.95)

        return ExecutionPrediction(
            strategy=strategy,
            expected_price=expected_price,
            expected_slippage_bps=impact_bps,
            expected_fill_rate=fill_rate,
            expected_duration_seconds=duration,
            cost_estimate=cost,
            risk_level=risk,
            confidence=confidence,
        )

    def _assess_risk(
        self,
        market: str,
        size: Decimal,
        strategy: ExecutionStrategy,
    ) -> ExecutionRisk:
        """Assess execution risk level."""
        state = self._market_states.get(market)
        if not state:
            return ExecutionRisk.MEDIUM

        risk_score = 0.0

        # Size relative to liquidity
        total_depth = state.bid_depth + state.ask_depth
        if total_depth > 0:
            size_ratio = float(size / total_depth)
            if size_ratio > 0.5:
                risk_score += 3
            elif size_ratio > 0.2:
                risk_score += 2
            elif size_ratio > 0.1:
                risk_score += 1

        # Volatility risk
        if state.volatility > 5.0:
            risk_score += 2
        elif state.volatility > 3.0:
            risk_score += 1

        # Liquidity risk
        if state.liquidity_score < 0.3:
            risk_score += 2
        elif state.liquidity_score < 0.5:
            risk_score += 1

        # Strategy risk
        strategy_risk = {
            ExecutionStrategy.IMMEDIATE: 1,
            ExecutionStrategy.AGGRESSIVE: 2,
            ExecutionStrategy.PASSIVE: 1,
            ExecutionStrategy.ADAPTIVE: 0,
            ExecutionStrategy.SCHEDULED: 1,
            ExecutionStrategy.OPPORTUNISTIC: 2,
        }
        risk_score += strategy_risk.get(strategy, 1)

        # Convert to risk level
        if risk_score >= 6:
            return ExecutionRisk.CRITICAL
        elif risk_score >= 4:
            return ExecutionRisk.HIGH
        elif risk_score >= 2:
            return ExecutionRisk.MEDIUM
        return ExecutionRisk.LOW

    def optimize(
        self,
        market: str,
        size: Decimal,
        side: str,
        objective: ExecutionObjective | None = None,
    ) -> OptimizationResult:
        """
        Optimize execution strategy.

        Args:
            market: Market symbol
            size: Order size
            side: 'buy' or 'sell'
            objective: Optimization objective

        Returns:
            Optimization result with recommendations
        """
        if objective is None:
            objective = self.default_objective

        # Detect market condition
        condition = self.detect_market_condition(market)

        # Get predictions for all strategies
        predictions = []
        for strategy in ExecutionStrategy:
            pred = self.predict_execution(market, size, side, strategy)
            predictions.append(pred)

        # Score each strategy
        weights = self._objective_weights.get(objective, self._objective_weights[ExecutionObjective.BALANCE])
        best_strategy = ExecutionStrategy.ADAPTIVE
        best_score = float("inf")

        for pred in predictions:
            score = (
                float(pred.cost_estimate.spread_cost) * weights["spread"]
                + float(pred.cost_estimate.impact_cost) * weights["impact"]
                + float(pred.cost_estimate.timing_cost) * weights["timing"]
                + self._risk_score(pred.risk_level) * weights["risk"]
            )

            # Adjust for fill rate (lower is worse)
            score *= (2 - pred.expected_fill_rate)

            if score < best_score:
                best_score = score
                best_strategy = pred.strategy

        # Generate optimal parameters
        optimal_params = self._generate_optimal_params(
            market, size, side, best_strategy, condition
        )

        result = OptimizationResult(
            market=market,
            side=side,
            size=size,
            objective=objective,
            recommended_strategy=best_strategy,
            predictions=predictions,
            market_condition=condition,
            optimal_params=optimal_params,
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass

        return result

    def _risk_score(self, risk: ExecutionRisk) -> float:
        """Convert risk level to numeric score."""
        scores = {
            ExecutionRisk.LOW: 1,
            ExecutionRisk.MEDIUM: 2,
            ExecutionRisk.HIGH: 4,
            ExecutionRisk.CRITICAL: 8,
        }
        return scores.get(risk, 2)

    def _generate_optimal_params(
        self,
        market: str,
        size: Decimal,
        side: str,
        strategy: ExecutionStrategy,
        condition: MarketCondition,
    ) -> dict[str, Any]:
        """Generate optimal parameters for the strategy."""
        state = self._market_states.get(market)
        params: dict[str, Any] = {}

        if strategy == ExecutionStrategy.SCHEDULED:
            # TWAP/VWAP parameters
            duration = 600  # 10 minutes default
            if condition == MarketCondition.HIGH_VOLATILITY:
                duration = 300  # Shorter in volatile markets
            elif condition == MarketCondition.LOW_LIQUIDITY:
                duration = 900  # Longer in illiquid markets

            slices = min(max(int(float(size) / 1), 10), 50)
            params["duration_seconds"] = duration
            params["slice_count"] = slices
            params["randomize"] = True

        elif strategy == ExecutionStrategy.PASSIVE:
            # Limit order parameters
            if state:
                if side == "buy":
                    params["limit_price"] = float(state.mid_price * Decimal("0.9995"))
                else:
                    params["limit_price"] = float(state.mid_price * Decimal("1.0005"))
            params["timeout_seconds"] = 300
            params["allow_partial"] = True

        elif strategy == ExecutionStrategy.ADAPTIVE:
            # Adaptive parameters
            params["aggression_start"] = 0.3
            params["aggression_max"] = 0.8
            params["adapt_interval_seconds"] = 30
            params["use_vwap_target"] = True

        elif strategy == ExecutionStrategy.OPPORTUNISTIC:
            # Opportunistic parameters
            if state:
                target_improvement = state.spread_bps * 0.3
                params["target_price_improvement_bps"] = target_improvement
            params["max_wait_seconds"] = 900
            params["min_fill_pct"] = 0.5

        # Common parameters
        if state:
            max_size = float(state.volume_24h) * self.max_participation_rate / 24
            params["max_slice_size"] = min(float(size), max_size)

        return params

    def record_execution(self, record: ExecutionRecord) -> None:
        """
        Record an execution for analysis.

        Args:
            record: Execution record
        """
        self._execution_history.append(record)

        # Update benchmarks
        self._update_benchmark(record)

        # Trim history if too large
        if len(self._execution_history) > 10000:
            self._execution_history = self._execution_history[-5000:]

    def _update_benchmark(self, record: ExecutionRecord) -> None:
        """Update benchmark from execution record."""
        if record.market not in self._benchmarks:
            self._benchmarks[record.market] = {}

        if record.strategy not in self._benchmarks[record.market]:
            self._benchmarks[record.market][record.strategy] = ExecutionBenchmark(
                strategy=record.strategy,
                avg_slippage_bps=record.slippage_bps,
                avg_fill_rate=record.fill_rate,
                avg_duration_seconds=record.duration_seconds,
                sample_count=1,
            )
        else:
            bench = self._benchmarks[record.market][record.strategy]
            n = bench.sample_count

            # Running average
            bench.avg_slippage_bps = (bench.avg_slippage_bps * n + record.slippage_bps) / (n + 1)
            bench.avg_fill_rate = (bench.avg_fill_rate * n + record.fill_rate) / (n + 1)
            bench.avg_duration_seconds = (bench.avg_duration_seconds * n + record.duration_seconds) / (n + 1)
            bench.sample_count = n + 1
            bench.last_updated = datetime.now()

    def _get_benchmark(self, market: str, strategy: ExecutionStrategy) -> ExecutionBenchmark | None:
        """Get benchmark for market and strategy."""
        if market not in self._benchmarks:
            return None
        return self._benchmarks[market].get(strategy)

    def get_benchmarks(self, market: str) -> dict[ExecutionStrategy, ExecutionBenchmark]:
        """Get all benchmarks for a market."""
        return self._benchmarks.get(market, {})

    def get_execution_history(
        self,
        market: str | None = None,
        strategy: ExecutionStrategy | None = None,
        since: datetime | None = None,
    ) -> list[ExecutionRecord]:
        """
        Get execution history with optional filters.

        Args:
            market: Filter by market
            strategy: Filter by strategy
            since: Filter by timestamp

        Returns:
            List of execution records
        """
        results = self._execution_history

        if market:
            results = [r for r in results if r.market == market]
        if strategy:
            results = [r for r in results if r.strategy == strategy]
        if since:
            results = [r for r in results if r.timestamp >= since]

        return results

    def calculate_execution_quality(
        self,
        market: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Calculate execution quality metrics.

        Args:
            market: Market symbol
            since: Start time for analysis

        Returns:
            Quality metrics
        """
        if since is None:
            since = datetime.now() - timedelta(days=7)

        records = self.get_execution_history(market=market, since=since)

        if not records:
            return {
                "market": market,
                "sample_count": 0,
                "avg_slippage_bps": 0.0,
                "avg_fill_rate": 0.0,
                "avg_cost_bps": 0.0,
            }

        total_slippage = sum(r.slippage_bps for r in records)
        total_fill_rate = sum(r.fill_rate for r in records)
        total_cost = sum(float(r.cost.total_cost) for r in records)

        return {
            "market": market,
            "sample_count": len(records),
            "avg_slippage_bps": total_slippage / len(records),
            "avg_fill_rate": total_fill_rate / len(records),
            "avg_cost_bps": total_cost / len(records),
            "by_strategy": self._quality_by_strategy(records),
        }

    def _quality_by_strategy(
        self,
        records: list[ExecutionRecord],
    ) -> dict[str, dict[str, float]]:
        """Calculate quality metrics by strategy."""
        by_strategy: dict[ExecutionStrategy, list[ExecutionRecord]] = {}

        for record in records:
            if record.strategy not in by_strategy:
                by_strategy[record.strategy] = []
            by_strategy[record.strategy].append(record)

        result = {}
        for strategy, recs in by_strategy.items():
            result[strategy.value] = {
                "count": len(recs),
                "avg_slippage_bps": sum(r.slippage_bps for r in recs) / len(recs),
                "avg_fill_rate": sum(r.fill_rate for r in recs) / len(recs),
            }

        return result

    def add_callback(self, callback: Callable[[OptimizationResult], None]) -> None:
        """Add optimization callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[OptimizationResult], None]) -> bool:
        """Remove optimization callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def get_markets(self) -> list[str]:
        """Get list of markets with state."""
        return list(self._market_states.keys())

    def clear_market(self, market: str) -> None:
        """Clear market data."""
        self._market_states.pop(market, None)
        self._benchmarks.pop(market, None)

    def clear_all(self) -> None:
        """Clear all data."""
        self._market_states.clear()
        self._benchmarks.clear()
        self._execution_history.clear()


class ExecutionAnalyzer:
    """Analyzes execution performance."""

    def __init__(self, optimizer: ExecutionOptimizer | None = None):
        """
        Initialize analyzer.

        Args:
            optimizer: Execution optimizer (creates new if None)
        """
        self.optimizer = optimizer or ExecutionOptimizer()

    def analyze_slippage(
        self,
        market: str,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Analyze slippage patterns.

        Args:
            market: Market symbol
            lookback_hours: Hours to analyze

        Returns:
            Slippage analysis
        """
        since = datetime.now() - timedelta(hours=lookback_hours)
        records = self.optimizer.get_execution_history(market=market, since=since)

        if not records:
            return {"market": market, "error": "No data"}

        slippages = [r.slippage_bps for r in records]

        return {
            "market": market,
            "sample_count": len(records),
            "min_slippage_bps": min(slippages),
            "max_slippage_bps": max(slippages),
            "avg_slippage_bps": sum(slippages) / len(slippages),
            "median_slippage_bps": sorted(slippages)[len(slippages) // 2],
        }

    def compare_strategies(
        self,
        market: str,
        lookback_hours: int = 168,  # 1 week
    ) -> dict[str, Any]:
        """
        Compare execution strategies.

        Args:
            market: Market symbol
            lookback_hours: Hours to analyze

        Returns:
            Strategy comparison
        """
        benchmarks = self.optimizer.get_benchmarks(market)

        comparison = {}
        for strategy, bench in benchmarks.items():
            comparison[strategy.value] = {
                "avg_slippage_bps": bench.avg_slippage_bps,
                "avg_fill_rate": bench.avg_fill_rate,
                "avg_duration_seconds": bench.avg_duration_seconds,
                "sample_count": bench.sample_count,
            }

        # Rank by cost-efficiency
        if comparison:
            ranked = sorted(
                comparison.items(),
                key=lambda x: x[1]["avg_slippage_bps"] * (2 - x[1]["avg_fill_rate"]),
            )
            for i, (strategy, _) in enumerate(ranked):
                comparison[strategy]["rank"] = i + 1

        return {
            "market": market,
            "strategies": comparison,
        }

    def get_best_strategy(
        self,
        market: str,
        objective: ExecutionObjective = ExecutionObjective.BALANCE,
    ) -> ExecutionStrategy | None:
        """
        Get best strategy based on historical performance.

        Args:
            market: Market symbol
            objective: Optimization objective

        Returns:
            Best strategy or None
        """
        benchmarks = self.optimizer.get_benchmarks(market)

        if not benchmarks:
            return None

        best_strategy = None
        best_score = float("inf")

        for strategy, bench in benchmarks.items():
            # Need minimum samples
            if bench.sample_count < 5:
                continue

            # Score based on objective
            if objective == ExecutionObjective.MINIMIZE_COST:
                score = bench.avg_slippage_bps
            elif objective == ExecutionObjective.MINIMIZE_TIME:
                score = bench.avg_duration_seconds
            elif objective == ExecutionObjective.MAXIMIZE_FILL:
                score = -bench.avg_fill_rate  # Negative for maximization
            else:
                score = bench.avg_slippage_bps * (2 - bench.avg_fill_rate)

            if score < best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy


# Global instances
_optimizer: ExecutionOptimizer | None = None
_analyzer: ExecutionAnalyzer | None = None


def get_optimizer() -> ExecutionOptimizer:
    """Get global execution optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ExecutionOptimizer()
    return _optimizer


def get_analyzer() -> ExecutionAnalyzer:
    """Get global execution analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ExecutionAnalyzer(get_optimizer())
    return _analyzer


def reset_optimizer() -> None:
    """Reset global instances."""
    global _optimizer, _analyzer
    _optimizer = None
    _analyzer = None
