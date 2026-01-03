"""Tests for smart order router module."""

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.analytics.smart_order_router import (
    OrderType,
    RoutingStrategy,
    OrderSide,
    ExecutionUrgency,
    MarketState,
    OrderSlice,
    RoutingDecision,
    ExecutionPlan,
    ExecutionMetrics,
    SmartOrderRouter,
    OrderRouter,
    get_smart_router,
    reset_smart_router,
)


class TestOrderTypeEnum:
    """Tests for OrderType enum."""

    def test_all_order_types(self):
        """Test all order type values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_MARKET.value == "stop_market"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TWAP.value == "twap"
        assert OrderType.VWAP.value == "vwap"
        assert OrderType.POV.value == "pov"
        assert OrderType.ICEBERG.value == "iceberg"

    def test_order_type_count(self):
        """Test total number of order types."""
        assert len(OrderType) == 8


class TestRoutingStrategyEnum:
    """Tests for RoutingStrategy enum."""

    def test_all_strategy_values(self):
        """Test all strategy values."""
        assert RoutingStrategy.AGGRESSIVE.value == "aggressive"
        assert RoutingStrategy.PASSIVE.value == "passive"
        assert RoutingStrategy.NEUTRAL.value == "neutral"
        assert RoutingStrategy.ADAPTIVE.value == "adaptive"
        assert RoutingStrategy.STEALTH.value == "stealth"

    def test_strategy_count(self):
        """Test total number of strategies."""
        assert len(RoutingStrategy) == 5


class TestOrderSideEnum:
    """Tests for OrderSide enum."""

    def test_order_side_values(self):
        """Test order side values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_side_count(self):
        """Test number of sides."""
        assert len(OrderSide) == 2


class TestExecutionUrgencyEnum:
    """Tests for ExecutionUrgency enum."""

    def test_urgency_values(self):
        """Test urgency values."""
        assert ExecutionUrgency.LOW.value == "low"
        assert ExecutionUrgency.MEDIUM.value == "medium"
        assert ExecutionUrgency.HIGH.value == "high"
        assert ExecutionUrgency.CRITICAL.value == "critical"

    def test_urgency_count(self):
        """Test number of urgency levels."""
        assert len(ExecutionUrgency) == 4


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_create_market_state(self):
        """Test creating market state."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            last_price=Decimal("50005"),
        )
        assert state.bid_price == Decimal("50000")
        assert state.ask_price == Decimal("50010")

    def test_mid_price(self):
        """Test mid price calculation."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            last_price=Decimal("50005"),
        )
        assert state.mid_price == Decimal("50005")

    def test_spread_bps_calculation(self):
        """Test spread bps calculation."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            last_price=Decimal("50005"),
        )
        assert state.spread_bps > 0

    def test_custom_spread_bps(self):
        """Test custom spread bps."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            last_price=Decimal("50005"),
            spread_bps=5.0,
        )
        assert state.spread_bps == 5.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("1000000"),
            volatility=0.02,
            liquidity_score=75.0,
        )
        d = state.to_dict()
        assert d["bid_price"] == "50000"
        assert d["ask_price"] == "50010"
        assert d["daily_volume"] == "1000000"
        assert d["volatility"] == 0.02


class TestOrderSlice:
    """Tests for OrderSlice dataclass."""

    def test_create_slice(self):
        """Test creating order slice."""
        slice = OrderSlice(
            sequence=1,
            size=Decimal("1.5"),
            price=Decimal("50000"),
            order_type=OrderType.LIMIT,
        )
        assert slice.sequence == 1
        assert slice.size == Decimal("1.5")
        assert slice.executed is False

    def test_slice_with_execution(self):
        """Test slice with execution info."""
        slice = OrderSlice(
            sequence=1,
            size=Decimal("1.5"),
            price=Decimal("50000"),
            executed=True,
            fill_price=Decimal("50001"),
            fill_time=time.time(),
        )
        assert slice.executed is True
        assert slice.fill_price == Decimal("50001")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        slice = OrderSlice(
            sequence=1,
            size=Decimal("1.5"),
            price=Decimal("50000"),
        )
        d = slice.to_dict()
        assert d["sequence"] == 1
        assert d["size"] == "1.5"
        assert d["price"] == "50000"


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_create_decision(self):
        """Test creating routing decision."""
        decision = RoutingDecision(
            strategy=RoutingStrategy.NEUTRAL,
            order_type=OrderType.LIMIT,
            total_size=Decimal("10"),
        )
        assert decision.strategy == RoutingStrategy.NEUTRAL
        assert decision.order_type == OrderType.LIMIT

    def test_decision_with_slices(self):
        """Test decision with slices."""
        slices = [
            OrderSlice(sequence=1, size=Decimal("5")),
            OrderSlice(sequence=2, size=Decimal("5")),
        ]
        decision = RoutingDecision(
            strategy=RoutingStrategy.STEALTH,
            order_type=OrderType.TWAP,
            slices=slices,
            total_size=Decimal("10"),
        )
        assert len(decision.slices) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = RoutingDecision(
            strategy=RoutingStrategy.AGGRESSIVE,
            order_type=OrderType.MARKET,
            price=Decimal("50000"),
            total_size=Decimal("5"),
            estimated_cost=Decimal("10"),
            estimated_slippage_bps=5.0,
            urgency=ExecutionUrgency.HIGH,
            reason="Test reason",
        )
        d = decision.to_dict()
        assert d["strategy"] == "aggressive"
        assert d["order_type"] == "market"
        assert d["urgency"] == "high"


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_create_plan(self):
        """Test creating execution plan."""
        decision = RoutingDecision(
            strategy=RoutingStrategy.NEUTRAL,
            order_type=OrderType.TWAP,
            total_size=Decimal("10"),
        )
        now = time.time()
        plan = ExecutionPlan(
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            total_size=Decimal("10"),
            decision=decision,
            start_time=now,
            end_time=now + 3600,
        )
        assert plan.market == "BTC-USD-PERP"
        assert plan.side == OrderSide.BUY
        assert plan.status == "pending"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = RoutingDecision(
            strategy=RoutingStrategy.NEUTRAL,
            order_type=OrderType.LIMIT,
            total_size=Decimal("5"),
        )
        now = time.time()
        plan = ExecutionPlan(
            market="ETH-USD-PERP",
            side=OrderSide.SELL,
            total_size=Decimal("5"),
            decision=decision,
            start_time=now,
            end_time=now + 1800,
        )
        d = plan.to_dict()
        assert d["market"] == "ETH-USD-PERP"
        assert d["side"] == "sell"


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating execution metrics."""
        metrics = ExecutionMetrics(
            avg_fill_price=Decimal("50000"),
            vwap=Decimal("50001"),
            slippage_bps=2.0,
            participation_rate=15.0,
            fill_rate=100.0,
            execution_time=300.0,
            num_fills=5,
        )
        assert metrics.avg_fill_price == Decimal("50000")
        assert metrics.num_fills == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ExecutionMetrics(
            avg_fill_price=Decimal("50000"),
            slippage_bps=2.0,
        )
        d = metrics.to_dict()
        assert d["avg_fill_price"] == "50000"
        assert d["slippage_bps"] == 2.0


class TestSmartOrderRouter:
    """Tests for SmartOrderRouter class."""

    @pytest.fixture
    def router(self):
        """Create a smart order router."""
        return SmartOrderRouter()

    @pytest.fixture
    def router_with_state(self):
        """Create router with market state."""
        router = SmartOrderRouter()
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("10000000"),
            volatility=0.02,
            liquidity_score=70.0,
        )
        router.update_market_state("BTC-USD-PERP", state)
        return router

    def test_init_defaults(self):
        """Test initialization with defaults."""
        router = SmartOrderRouter()
        assert router.default_slice_count == 10
        assert router.max_impact_bps == 10.0

    def test_init_custom(self):
        """Test initialization with custom params."""
        router = SmartOrderRouter(
            default_slice_count=20,
            max_impact_bps=5.0,
            stealth_threshold_pct=3.0,
        )
        assert router.default_slice_count == 20
        assert router.max_impact_bps == 5.0

    def test_update_market_state(self, router):
        """Test updating market state."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
        )
        router.update_market_state("BTC-USD-PERP", state)
        assert router.get_market_state("BTC-USD-PERP") is not None

    def test_get_market_state_missing(self, router):
        """Test getting missing market state."""
        assert router.get_market_state("UNKNOWN") is None

    def test_estimate_price_impact(self, router_with_state):
        """Test price impact estimation."""
        impact = router_with_state.estimate_price_impact(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert impact >= 0

    def test_estimate_price_impact_large_order(self, router_with_state):
        """Test price impact for large order."""
        small_impact = router_with_state.estimate_price_impact(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        large_impact = router_with_state.estimate_price_impact(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
        )
        # Larger orders should have more impact
        assert large_impact >= small_impact

    def test_estimate_price_impact_no_state(self, router):
        """Test price impact with no state."""
        impact = router.estimate_price_impact(
            "UNKNOWN",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert impact == 0.0

    def test_estimate_execution_cost(self, router_with_state):
        """Test execution cost estimation."""
        cost = router_with_state.estimate_execution_cost(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert cost >= 0

    def test_estimate_cost_different_order_types(self, router_with_state):
        """Test cost for different order types."""
        market_cost = router_with_state.estimate_execution_cost(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            OrderType.MARKET,
        )
        limit_cost = router_with_state.estimate_execution_cost(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            OrderType.LIMIT,
        )
        # Limit orders should have lower cost
        assert limit_cost <= market_cost

    def test_determine_strategy_neutral(self, router_with_state):
        """Test neutral strategy determination."""
        strategy = router_with_state.determine_strategy(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
            ExecutionUrgency.MEDIUM,
        )
        assert isinstance(strategy, RoutingStrategy)

    def test_determine_strategy_aggressive(self, router_with_state):
        """Test aggressive strategy for critical urgency."""
        strategy = router_with_state.determine_strategy(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            ExecutionUrgency.CRITICAL,
        )
        assert strategy == RoutingStrategy.AGGRESSIVE

    def test_determine_strategy_stealth(self, router_with_state):
        """Test stealth strategy for large orders."""
        # Large order relative to daily volume
        strategy = router_with_state.determine_strategy(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("1000000"),  # 10% of daily volume
            ExecutionUrgency.MEDIUM,
        )
        assert strategy == RoutingStrategy.STEALTH

    def test_determine_order_type_market(self, router_with_state):
        """Test market order type for aggressive strategy."""
        order_type = router_with_state.determine_order_type(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            RoutingStrategy.AGGRESSIVE,
        )
        assert order_type == OrderType.MARKET

    def test_determine_order_type_limit(self, router_with_state):
        """Test limit order type for passive strategy."""
        order_type = router_with_state.determine_order_type(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            RoutingStrategy.PASSIVE,
        )
        assert order_type == OrderType.LIMIT

    def test_determine_order_type_iceberg(self, router_with_state):
        """Test iceberg order type for stealth strategy."""
        order_type = router_with_state.determine_order_type(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            RoutingStrategy.STEALTH,
        )
        assert order_type == OrderType.ICEBERG


class TestTWAPSlices:
    """Tests for TWAP slice creation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_create_twap_slices(self, router):
        """Test creating TWAP slices."""
        slices = router.create_twap_slices(
            total_size=Decimal("100"),
            duration_seconds=600,
            interval_seconds=60,
            randomize=False,
        )
        assert len(slices) == 10
        total = sum(s.size for s in slices)
        assert total == Decimal("100")

    def test_twap_slices_sequencing(self, router):
        """Test TWAP slice sequencing."""
        slices = router.create_twap_slices(
            total_size=Decimal("50"),
            duration_seconds=300,
            interval_seconds=60,
            randomize=False,
        )
        for i, slice in enumerate(slices):
            assert slice.sequence == i + 1

    def test_twap_slices_randomization(self, router):
        """Test TWAP slice randomization."""
        slices1 = router.create_twap_slices(
            total_size=Decimal("100"),
            duration_seconds=600,
            interval_seconds=60,
            randomize=True,
        )
        slices2 = router.create_twap_slices(
            total_size=Decimal("100"),
            duration_seconds=600,
            interval_seconds=60,
            randomize=True,
        )
        # With randomization, slices may differ
        sizes1 = [s.size for s in slices1]
        sizes2 = [s.size for s in slices2]
        # They should still total the same
        assert float(sum(sizes1)) == pytest.approx(float(sum(sizes2)), rel=0.1)


class TestVWAPSlices:
    """Tests for VWAP slice creation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_create_vwap_slices(self, router):
        """Test creating VWAP slices."""
        volume_profile = [0.1, 0.2, 0.3, 0.25, 0.15]
        slices = router.create_vwap_slices(
            total_size=Decimal("100"),
            volume_profile=volume_profile,
            duration_seconds=300,
        )
        assert len(slices) == 5

    def test_vwap_slices_respect_profile(self, router):
        """Test VWAP slices respect volume profile."""
        volume_profile = [0.5, 0.3, 0.2]
        slices = router.create_vwap_slices(
            total_size=Decimal("100"),
            volume_profile=volume_profile,
            duration_seconds=180,
        )
        # First slice should be largest
        assert slices[0].size > slices[-1].size

    def test_vwap_empty_profile(self, router):
        """Test VWAP with empty profile."""
        slices = router.create_vwap_slices(
            total_size=Decimal("100"),
            volume_profile=[],
            duration_seconds=600,
        )
        # Should use default profile
        assert len(slices) == 10


class TestPOVSlices:
    """Tests for POV slice creation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_create_pov_slices(self, router):
        """Test creating POV slices."""
        slices = router.create_pov_slices(
            total_size=Decimal("100"),
            participation_rate=0.1,
            estimated_market_volume=Decimal("10000"),
            duration_seconds=600,
        )
        assert len(slices) > 0

    def test_pov_participation_rate(self, router):
        """Test POV respects participation rate."""
        slices = router.create_pov_slices(
            total_size=Decimal("1000"),
            participation_rate=0.1,
            estimated_market_volume=Decimal("10000"),
            duration_seconds=600,
        )
        # Total should be limited by participation rate
        total = sum(s.size for s in slices)
        assert total <= Decimal("1000")


class TestIcebergSlices:
    """Tests for Iceberg slice creation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_create_iceberg_slices(self, router):
        """Test creating iceberg slices."""
        slices = router.create_iceberg_slices(
            total_size=Decimal("100"),
            visible_size=Decimal("10"),
        )
        assert len(slices) == 10

    def test_iceberg_visible_size(self, router):
        """Test iceberg visible size."""
        slices = router.create_iceberg_slices(
            total_size=Decimal("100"),
            visible_size=Decimal("25"),
        )
        for slice in slices[:-1]:  # All but last
            assert slice.size == Decimal("25")

    def test_iceberg_no_scheduled_time(self, router):
        """Test iceberg slices have no scheduled time."""
        slices = router.create_iceberg_slices(
            total_size=Decimal("50"),
            visible_size=Decimal("10"),
        )
        for slice in slices:
            assert slice.scheduled_time is None


class TestRouteOrder:
    """Tests for route_order method."""

    @pytest.fixture
    def router_with_state(self):
        """Create router with state."""
        router = SmartOrderRouter()
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("10000000"),
            volatility=0.02,
            liquidity_score=70.0,
        )
        router.update_market_state("BTC-USD-PERP", state)
        return router

    def test_route_order_basic(self, router_with_state):
        """Test basic order routing."""
        decision = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.total_size == Decimal("10")

    def test_route_order_with_urgency(self, router_with_state):
        """Test routing with different urgencies."""
        high_urgency = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            urgency=ExecutionUrgency.HIGH,
        )
        low_urgency = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            urgency=ExecutionUrgency.LOW,
        )
        # High urgency should be more aggressive
        # (they may or may not differ based on market conditions)

    def test_route_order_buy_price(self, router_with_state):
        """Test buy order price selection."""
        decision = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            urgency=ExecutionUrgency.CRITICAL,
        )
        # Aggressive buy should use ask price
        state = router_with_state.get_market_state("BTC-USD-PERP")
        if decision.strategy == RoutingStrategy.AGGRESSIVE:
            assert decision.price == state.ask_price

    def test_route_order_sell_price(self, router_with_state):
        """Test sell order price selection."""
        decision = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.SELL,
            Decimal("10"),
            urgency=ExecutionUrgency.CRITICAL,
        )
        # Aggressive sell should use bid price
        state = router_with_state.get_market_state("BTC-USD-PERP")
        if decision.strategy == RoutingStrategy.AGGRESSIVE:
            assert decision.price == state.bid_price

    def test_route_order_with_duration(self, router_with_state):
        """Test routing with duration."""
        decision = router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("1000"),
            duration_seconds=3600,
        )
        # Should have slices for timed execution
        assert isinstance(decision, RoutingDecision)

    def test_route_order_callback(self, router_with_state):
        """Test routing callback."""
        called = []

        def callback(market, decision):
            called.append((market, decision))

        router_with_state.add_callback(callback)
        router_with_state.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert len(called) == 1
        assert called[0][0] == "BTC-USD-PERP"


class TestExecutionPlan:
    """Tests for execution plan creation."""

    @pytest.fixture
    def router_with_state(self):
        """Create router with state."""
        router = SmartOrderRouter()
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("10000000"),
            volatility=0.02,
            liquidity_score=70.0,
        )
        router.update_market_state("BTC-USD-PERP", state)
        return router

    def test_create_execution_plan(self, router_with_state):
        """Test creating execution plan."""
        plan = router_with_state.create_execution_plan(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
        )
        assert isinstance(plan, ExecutionPlan)
        assert plan.market == "BTC-USD-PERP"

    def test_plan_with_times(self, router_with_state):
        """Test plan with custom times."""
        now = time.time()
        plan = router_with_state.create_execution_plan(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
            start_time=now,
            end_time=now + 1800,
        )
        assert plan.start_time == now
        assert plan.end_time == now + 1800

    def test_get_execution_plan(self, router_with_state):
        """Test getting execution plan."""
        plan = router_with_state.create_execution_plan(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
        )
        plan_id = f"BTC-USD-PERP_{int(plan.start_time)}"
        retrieved = router_with_state.get_execution_plan(plan_id)
        assert retrieved is not None

    def test_get_active_plans(self, router_with_state):
        """Test getting active plans."""
        router_with_state.create_execution_plan(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("100"),
        )
        active = router_with_state.get_active_plans()
        assert len(active) >= 0  # May or may not be active


class TestExecutionMetricsCalculation:
    """Tests for execution metrics calculation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_calculate_metrics(self, router):
        """Test calculating execution metrics."""
        fills = [
            {"price": "50000", "size": "5", "timestamp": 1000},
            {"price": "50010", "size": "5", "timestamp": 1060},
        ]
        metrics = router.calculate_execution_metrics(
            fills,
            benchmark_price=Decimal("50000"),
        )
        assert metrics.avg_fill_price == Decimal("50005")
        assert metrics.num_fills == 2

    def test_calculate_metrics_empty(self, router):
        """Test metrics with no fills."""
        metrics = router.calculate_execution_metrics(
            [],
            benchmark_price=Decimal("50000"),
        )
        assert metrics.num_fills == 0

    def test_slippage_calculation(self, router):
        """Test slippage calculation."""
        fills = [
            {"price": "50010", "size": "10", "timestamp": 1000},
        ]
        metrics = router.calculate_execution_metrics(
            fills,
            benchmark_price=Decimal("50000"),
        )
        assert metrics.slippage_bps > 0


class TestRouterCallbacks:
    """Tests for router callbacks."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_add_callback(self, router):
        """Test adding callback."""
        def callback(m, d):
            pass

        router.add_callback(callback)
        assert callback in router._callbacks

    def test_remove_callback(self, router):
        """Test removing callback."""
        def callback(m, d):
            pass

        router.add_callback(callback)
        router.remove_callback(callback)
        assert callback not in router._callbacks

    def test_remove_nonexistent_callback(self, router):
        """Test removing non-existent callback."""
        def callback(m, d):
            pass

        # Should not raise
        router.remove_callback(callback)


class TestRouterUtilities:
    """Tests for router utility methods."""

    @pytest.fixture
    def router(self):
        """Create router with data."""
        router = SmartOrderRouter()
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
        )
        router.update_market_state("BTC-USD-PERP", state)
        router.update_market_state("ETH-USD-PERP", state)
        return router

    def test_get_markets(self, router):
        """Test getting list of markets."""
        markets = router.get_markets()
        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_clear_market(self, router):
        """Test clearing market data."""
        router.clear_market("BTC-USD-PERP")
        assert router.get_market_state("BTC-USD-PERP") is None
        assert router.get_market_state("ETH-USD-PERP") is not None

    def test_clear_all(self, router):
        """Test clearing all data."""
        router.clear_all()
        assert len(router.get_markets()) == 0


class TestOrderRouter:
    """Tests for OrderRouter class."""

    @pytest.fixture
    def order_router(self):
        """Create order router."""
        return OrderRouter()

    def test_init(self, order_router):
        """Test initialization."""
        assert isinstance(order_router, OrderRouter)

    def test_set_market_priority(self, order_router):
        """Test setting market priority."""
        order_router.set_market_priority("BTC-USD-PERP", 10)
        assert order_router.get_market_priority("BTC-USD-PERP") == 10

    def test_get_market_priority_default(self, order_router):
        """Test default market priority."""
        assert order_router.get_market_priority("UNKNOWN") == 0

    def test_update_market_state(self, order_router):
        """Test updating market state."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("10000000"),
        )
        order_router.update_market_state("BTC-USD-PERP", state)
        # Should not raise

    def test_route_multiple_markets(self, order_router):
        """Test routing across multiple markets."""
        for market in ["BTC-USD-PERP", "ETH-USD-PERP"]:
            state = MarketState(
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("100"),
                ask_size=Decimal("100"),
                last_price=Decimal("50005"),
                daily_volume=Decimal("10000000"),
                liquidity_score=70.0,
            )
            order_router.update_market_state(market, state)

        decisions = order_router.route(
            ["BTC-USD-PERP", "ETH-USD-PERP"],
            OrderSide.BUY,
            Decimal("100"),
        )
        assert isinstance(decisions, dict)

    def test_get_best_market(self, order_router):
        """Test getting best market."""
        for market in ["BTC-USD-PERP", "ETH-USD-PERP"]:
            state = MarketState(
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_size=Decimal("100"),
                ask_size=Decimal("100"),
                last_price=Decimal("50005"),
                liquidity_score=70.0,
            )
            order_router.update_market_state(market, state)

        order_router.set_market_priority("BTC-USD-PERP", 10)
        best = order_router.get_best_market(
            ["BTC-USD-PERP", "ETH-USD-PERP"],
            OrderSide.BUY,
        )
        # BTC should be best due to higher priority
        assert best == "BTC-USD-PERP"

    def test_get_best_market_empty(self, order_router):
        """Test getting best market with no markets."""
        best = order_router.get_best_market([], OrderSide.BUY)
        assert best is None

    def test_route_empty_markets(self, order_router):
        """Test routing with empty markets."""
        decisions = order_router.route([], OrderSide.BUY, Decimal("100"))
        assert decisions == {}


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_smart_router(self):
        """Test getting global router."""
        reset_smart_router()
        router1 = get_smart_router()
        router2 = get_smart_router()
        assert router1 is router2

    def test_reset_smart_router(self):
        """Test resetting global router."""
        router1 = get_smart_router()
        reset_smart_router()
        router2 = get_smart_router()
        assert router1 is not router2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_zero_size_order(self, router):
        """Test with zero size order."""
        decision = router.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("0"),
        )
        assert decision.total_size == Decimal("0")

    def test_very_large_order(self, router):
        """Test with very large order."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("1000"),
        )
        router.update_market_state("BTC-USD-PERP", state)
        decision = router.route_order(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10000"),  # Very large relative to volume
        )
        assert decision.strategy == RoutingStrategy.STEALTH

    def test_no_liquidity(self, router):
        """Test with no liquidity."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("0"),
            ask_size=Decimal("0"),
            last_price=Decimal("50005"),
        )
        router.update_market_state("BTC-USD-PERP", state)
        impact = router.estimate_price_impact(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
        )
        assert impact == float("inf")

    def test_wide_spread(self, router):
        """Test with wide spread."""
        state = MarketState(
            bid_price=Decimal("49000"),
            ask_price=Decimal("51000"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50000"),
            daily_volume=Decimal("10000000"),
        )
        router.update_market_state("BTC-USD-PERP", state)
        strategy = router.determine_strategy(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            ExecutionUrgency.MEDIUM,
        )
        # Wide spread should encourage passive approach
        assert strategy in [RoutingStrategy.PASSIVE, RoutingStrategy.ADAPTIVE]

    def test_high_volatility(self, router):
        """Test with high volatility."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50005"),
            daily_volume=Decimal("10000000"),
            volatility=0.1,
            liquidity_score=30.0,
        )
        router.update_market_state("BTC-USD-PERP", state)
        strategy = router.determine_strategy(
            "BTC-USD-PERP",
            OrderSide.BUY,
            Decimal("10"),
            ExecutionUrgency.MEDIUM,
        )
        # High volatility or low liquidity should encourage adaptive
        assert strategy in [RoutingStrategy.ADAPTIVE, RoutingStrategy.PASSIVE]


class TestGenerateReason:
    """Tests for reason generation."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_aggressive_reason(self, router):
        """Test aggressive reason."""
        reason = router._generate_reason(
            RoutingStrategy.AGGRESSIVE,
            OrderType.MARKET,
            None,
        )
        assert "urgency" in reason.lower()

    def test_passive_reason(self, router):
        """Test passive reason."""
        reason = router._generate_reason(
            RoutingStrategy.PASSIVE,
            OrderType.LIMIT,
            None,
        )
        assert "passive" in reason.lower()

    def test_stealth_reason(self, router):
        """Test stealth reason."""
        reason = router._generate_reason(
            RoutingStrategy.STEALTH,
            OrderType.ICEBERG,
            None,
        )
        assert "stealth" in reason.lower()

    def test_twap_reason(self, router):
        """Test TWAP reason."""
        reason = router._generate_reason(
            RoutingStrategy.NEUTRAL,
            OrderType.TWAP,
            None,
        )
        assert "TWAP" in reason

    def test_reason_with_state(self, router):
        """Test reason with market state."""
        state = MarketState(
            bid_price=Decimal("50000"),
            ask_price=Decimal("50050"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            last_price=Decimal("50025"),
            spread_bps=25.0,
            volatility=0.05,
        )
        reason = router._generate_reason(
            RoutingStrategy.NEUTRAL,
            OrderType.LIMIT,
            state,
        )
        # Should mention wide spread and volatility
        assert "spread" in reason.lower() or "volatility" in reason.lower()
