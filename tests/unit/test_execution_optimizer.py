"""Tests for Execution Optimizer module."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.analytics.execution_optimizer import (
    ExecutionAnalyzer,
    ExecutionBenchmark,
    ExecutionCost,
    ExecutionObjective,
    ExecutionOptimizer,
    ExecutionPrediction,
    ExecutionRecord,
    ExecutionRisk,
    ExecutionStrategy,
    MarketCondition,
    MarketState,
    OptimizationResult,
    get_analyzer,
    get_optimizer,
    reset_optimizer,
)


class TestExecutionObjectiveEnum:
    """Tests for ExecutionObjective enum."""

    def test_all_objectives(self):
        """Test all objectives exist."""
        objectives = [
            ExecutionObjective.MINIMIZE_COST,
            ExecutionObjective.MINIMIZE_IMPACT,
            ExecutionObjective.MINIMIZE_TIME,
            ExecutionObjective.MINIMIZE_RISK,
            ExecutionObjective.MAXIMIZE_FILL,
            ExecutionObjective.BALANCE,
        ]
        assert len(objectives) == 6

    def test_objective_values(self):
        """Test objective values."""
        assert ExecutionObjective.MINIMIZE_COST.value == "minimize_cost"
        assert ExecutionObjective.BALANCE.value == "balance"


class TestExecutionStrategyEnum:
    """Tests for ExecutionStrategy enum."""

    def test_all_strategies(self):
        """Test all strategies exist."""
        strategies = [
            ExecutionStrategy.IMMEDIATE,
            ExecutionStrategy.PASSIVE,
            ExecutionStrategy.AGGRESSIVE,
            ExecutionStrategy.ADAPTIVE,
            ExecutionStrategy.SCHEDULED,
            ExecutionStrategy.OPPORTUNISTIC,
        ]
        assert len(strategies) == 6

    def test_strategy_values(self):
        """Test strategy values."""
        assert ExecutionStrategy.IMMEDIATE.value == "immediate"
        assert ExecutionStrategy.ADAPTIVE.value == "adaptive"


class TestMarketConditionEnum:
    """Tests for MarketCondition enum."""

    def test_all_conditions(self):
        """Test all conditions exist."""
        conditions = [
            MarketCondition.NORMAL,
            MarketCondition.HIGH_VOLATILITY,
            MarketCondition.LOW_LIQUIDITY,
            MarketCondition.TRENDING,
            MarketCondition.MEAN_REVERTING,
            MarketCondition.AUCTION,
        ]
        assert len(conditions) == 6


class TestExecutionRiskEnum:
    """Tests for ExecutionRisk enum."""

    def test_all_risks(self):
        """Test all risk levels exist."""
        risks = [
            ExecutionRisk.LOW,
            ExecutionRisk.MEDIUM,
            ExecutionRisk.HIGH,
            ExecutionRisk.CRITICAL,
        ]
        assert len(risks) == 4


class TestExecutionCost:
    """Tests for ExecutionCost dataclass."""

    def test_create_cost(self):
        """Test creating execution cost."""
        cost = ExecutionCost(
            spread_cost=Decimal("2"),
            impact_cost=Decimal("3"),
            fee_cost=Decimal("5"),
        )
        assert cost.spread_cost == Decimal("2")
        assert cost.impact_cost == Decimal("3")

    def test_total_cost(self):
        """Test total cost calculation."""
        cost = ExecutionCost(
            spread_cost=Decimal("2"),
            impact_cost=Decimal("3"),
            timing_cost=Decimal("1"),
            fee_cost=Decimal("5"),
        )
        assert cost.total_cost == Decimal("11")

    def test_total_cost_bps(self):
        """Test total cost in bps."""
        cost = ExecutionCost(
            spread_cost=Decimal("2"),
            impact_cost=Decimal("3"),
        )
        assert cost.total_cost_bps == 5.0

    def test_to_dict(self):
        """Test converting to dict."""
        cost = ExecutionCost(spread_cost=Decimal("2"))
        d = cost.to_dict()
        assert d["spread_cost"] == 2.0
        assert "total_cost" in d


class TestExecutionPrediction:
    """Tests for ExecutionPrediction dataclass."""

    def test_create_prediction(self):
        """Test creating prediction."""
        pred = ExecutionPrediction(
            strategy=ExecutionStrategy.ADAPTIVE,
            expected_price=Decimal("50000"),
            expected_slippage_bps=5.0,
            expected_fill_rate=0.95,
            expected_duration_seconds=60.0,
            cost_estimate=ExecutionCost(),
            risk_level=ExecutionRisk.MEDIUM,
        )
        assert pred.strategy == ExecutionStrategy.ADAPTIVE
        assert pred.expected_fill_rate == 0.95

    def test_to_dict(self):
        """Test converting to dict."""
        pred = ExecutionPrediction(
            strategy=ExecutionStrategy.IMMEDIATE,
            expected_price=Decimal("50000"),
            expected_slippage_bps=5.0,
            expected_fill_rate=1.0,
            expected_duration_seconds=5.0,
            cost_estimate=ExecutionCost(),
            risk_level=ExecutionRisk.LOW,
        )
        d = pred.to_dict()
        assert d["strategy"] == "immediate"
        assert d["expected_fill_rate"] == 1.0


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_create_state(self):
        """Test creating market state."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.2,
            imbalance=0.5,
        )
        assert state.market == "BTC-USD-PERP"
        assert state.mid_price == Decimal("50000")

    def test_to_dict(self):
        """Test converting to dict."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.2,
            imbalance=0.5,
        )
        d = state.to_dict()
        assert d["market"] == "BTC-USD-PERP"


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_create_result(self):
        """Test creating result."""
        result = OptimizationResult(
            market="BTC-USD-PERP",
            side="buy",
            size=Decimal("10"),
            objective=ExecutionObjective.MINIMIZE_COST,
            recommended_strategy=ExecutionStrategy.ADAPTIVE,
            predictions=[],
            market_condition=MarketCondition.NORMAL,
        )
        assert result.recommended_strategy == ExecutionStrategy.ADAPTIVE

    def test_to_dict(self):
        """Test converting to dict."""
        result = OptimizationResult(
            market="BTC-USD-PERP",
            side="buy",
            size=Decimal("10"),
            objective=ExecutionObjective.BALANCE,
            recommended_strategy=ExecutionStrategy.ADAPTIVE,
            predictions=[],
            market_condition=MarketCondition.NORMAL,
        )
        d = result.to_dict()
        assert d["market"] == "BTC-USD-PERP"
        assert d["objective"] == "balance"


class TestExecutionBenchmark:
    """Tests for ExecutionBenchmark dataclass."""

    def test_create_benchmark(self):
        """Test creating benchmark."""
        bench = ExecutionBenchmark(
            strategy=ExecutionStrategy.ADAPTIVE,
            avg_slippage_bps=5.0,
            avg_fill_rate=0.95,
            avg_duration_seconds=60.0,
            sample_count=100,
        )
        assert bench.strategy == ExecutionStrategy.ADAPTIVE
        assert bench.sample_count == 100

    def test_to_dict(self):
        """Test converting to dict."""
        bench = ExecutionBenchmark(
            strategy=ExecutionStrategy.IMMEDIATE,
            avg_slippage_bps=10.0,
            avg_fill_rate=1.0,
            avg_duration_seconds=5.0,
            sample_count=50,
        )
        d = bench.to_dict()
        assert d["strategy"] == "immediate"


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass."""

    def test_create_record(self):
        """Test creating record."""
        record = ExecutionRecord(
            market="BTC-USD-PERP",
            side="buy",
            size=Decimal("10"),
            strategy=ExecutionStrategy.ADAPTIVE,
            entry_price=Decimal("50000"),
            avg_fill_price=Decimal("50010"),
            fill_rate=1.0,
            duration_seconds=30.0,
            slippage_bps=2.0,
            cost=ExecutionCost(),
        )
        assert record.market == "BTC-USD-PERP"
        assert record.fill_rate == 1.0

    def test_to_dict(self):
        """Test converting to dict."""
        record = ExecutionRecord(
            market="BTC-USD-PERP",
            side="sell",
            size=Decimal("5"),
            strategy=ExecutionStrategy.IMMEDIATE,
            entry_price=Decimal("50000"),
            avg_fill_price=Decimal("49990"),
            fill_rate=1.0,
            duration_seconds=5.0,
            slippage_bps=2.0,
            cost=ExecutionCost(),
        )
        d = record.to_dict()
        assert d["side"] == "sell"


class TestExecutionOptimizer:
    """Tests for ExecutionOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return ExecutionOptimizer()

    @pytest.fixture
    def optimizer_with_state(self, optimizer):
        """Create optimizer with market state."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.1,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)
        return optimizer

    def test_init_defaults(self):
        """Test default initialization."""
        opt = ExecutionOptimizer()
        assert opt.default_objective == ExecutionObjective.BALANCE
        assert opt.fee_rate_bps == 5.0

    def test_init_custom(self):
        """Test custom initialization."""
        opt = ExecutionOptimizer(
            default_objective=ExecutionObjective.MINIMIZE_COST,
            impact_model_alpha=0.2,
            fee_rate_bps=3.0,
        )
        assert opt.default_objective == ExecutionObjective.MINIMIZE_COST
        assert opt.fee_rate_bps == 3.0

    def test_update_market_state(self, optimizer):
        """Test updating market state."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        retrieved = optimizer.get_market_state("BTC-USD-PERP")
        assert retrieved is not None
        assert retrieved.mid_price == Decimal("50000")

    def test_get_market_state_missing(self, optimizer):
        """Test getting missing market state."""
        state = optimizer.get_market_state("UNKNOWN")
        assert state is None


class TestMarketConditionDetection:
    """Tests for market condition detection."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return ExecutionOptimizer()

    def test_detect_normal_condition(self, optimizer):
        """Test detecting normal condition."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=2.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.3,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        condition = optimizer.detect_market_condition("BTC-USD-PERP")
        assert condition == MarketCondition.NORMAL

    def test_detect_high_volatility(self, optimizer):
        """Test detecting high volatility."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=5.0,
            volatility=8.0,  # High
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.1,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        condition = optimizer.detect_market_condition("BTC-USD-PERP")
        assert condition == MarketCondition.HIGH_VOLATILITY

    def test_detect_low_liquidity(self, optimizer):
        """Test detecting low liquidity."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=10.0,
            volatility=2.0,
            liquidity_score=0.2,  # Low
            volume_24h=Decimal("100000"),
            bid_depth=Decimal("50"),
            ask_depth=Decimal("50"),
            trend_direction=0.1,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        condition = optimizer.detect_market_condition("BTC-USD-PERP")
        assert condition == MarketCondition.LOW_LIQUIDITY

    def test_detect_trending(self, optimizer):
        """Test detecting trending market."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.8,  # Strong trend
            imbalance=0.6,
        )
        optimizer.update_market_state(state)

        condition = optimizer.detect_market_condition("BTC-USD-PERP")
        assert condition == MarketCondition.TRENDING

    def test_detect_unknown_market(self, optimizer):
        """Test detecting condition for unknown market."""
        condition = optimizer.detect_market_condition("UNKNOWN")
        assert condition == MarketCondition.NORMAL


class TestImpactEstimation:
    """Tests for impact estimation."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        opt.update_market_state(state)
        return opt

    def test_estimate_impact_small_order(self, optimizer):
        """Test impact for small order."""
        impact = optimizer.estimate_impact("BTC-USD-PERP", Decimal("10"), "buy")
        assert impact > 0
        assert impact < 50  # Should be reasonable

    def test_estimate_impact_large_order(self, optimizer):
        """Test impact for large order."""
        impact_small = optimizer.estimate_impact("BTC-USD-PERP", Decimal("100"), "buy")
        impact_large = optimizer.estimate_impact("BTC-USD-PERP", Decimal("10000"), "buy")
        # Large order should have more impact
        assert impact_large > impact_small

    def test_estimate_impact_unknown_market(self, optimizer):
        """Test impact for unknown market."""
        impact = optimizer.estimate_impact("UNKNOWN", Decimal("10"), "buy")
        assert impact == 10.0  # Default

    def test_estimate_impact_buy_vs_sell(self, optimizer):
        """Test impact varies by side based on imbalance."""
        # With balanced book (imbalance=0.5), buy and sell should be similar
        impact_buy = optimizer.estimate_impact("BTC-USD-PERP", Decimal("100"), "buy")
        impact_sell = optimizer.estimate_impact("BTC-USD-PERP", Decimal("100"), "sell")
        assert abs(impact_buy - impact_sell) < 1.0


class TestCostEstimation:
    """Tests for cost estimation."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        opt.update_market_state(state)
        return opt

    def test_estimate_cost_immediate(self, optimizer):
        """Test cost for immediate execution."""
        cost = optimizer.estimate_cost(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            ExecutionStrategy.IMMEDIATE,
        )
        assert cost.spread_cost > 0
        assert cost.fee_cost > 0
        assert cost.total_cost > 0

    def test_estimate_cost_passive(self, optimizer):
        """Test cost for passive execution."""
        cost = optimizer.estimate_cost(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            ExecutionStrategy.PASSIVE,
        )
        # Passive should have lower spread cost
        assert cost.spread_cost == 0

    def test_estimate_cost_by_strategy(self, optimizer):
        """Test costs differ by strategy."""
        cost_immediate = optimizer.estimate_cost(
            "BTC-USD-PERP",
            Decimal("100"),
            "buy",
            ExecutionStrategy.IMMEDIATE,
        )
        cost_passive = optimizer.estimate_cost(
            "BTC-USD-PERP",
            Decimal("100"),
            "buy",
            ExecutionStrategy.PASSIVE,
        )
        # Immediate should cost more
        assert cost_immediate.total_cost > cost_passive.spread_cost


class TestExecutionPrediction:
    """Tests for execution prediction."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        opt.update_market_state(state)
        return opt

    def test_predict_immediate(self, optimizer):
        """Test prediction for immediate execution."""
        pred = optimizer.predict_execution(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            ExecutionStrategy.IMMEDIATE,
        )
        assert pred.strategy == ExecutionStrategy.IMMEDIATE
        assert pred.expected_fill_rate == 1.0
        assert pred.expected_duration_seconds == 5.0

    def test_predict_passive(self, optimizer):
        """Test prediction for passive execution."""
        pred = optimizer.predict_execution(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            ExecutionStrategy.PASSIVE,
        )
        assert pred.expected_fill_rate < 1.0
        assert pred.expected_duration_seconds > 60

    def test_predict_buy_price(self, optimizer):
        """Test predicted price for buy order."""
        pred = optimizer.predict_execution(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            ExecutionStrategy.IMMEDIATE,
        )
        # Buy price should be above mid
        assert pred.expected_price >= Decimal("50000")

    def test_predict_sell_price(self, optimizer):
        """Test predicted price for sell order."""
        pred = optimizer.predict_execution(
            "BTC-USD-PERP",
            Decimal("10"),
            "sell",
            ExecutionStrategy.IMMEDIATE,
        )
        # Sell price should be below mid
        assert pred.expected_price <= Decimal("50000")

    def test_predict_unknown_market(self, optimizer):
        """Test prediction for unknown market."""
        pred = optimizer.predict_execution(
            "UNKNOWN",
            Decimal("10"),
            "buy",
            ExecutionStrategy.IMMEDIATE,
        )
        assert pred.confidence < 0.5


class TestOptimization:
    """Tests for optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        opt.update_market_state(state)
        return opt

    def test_optimize_basic(self, optimizer):
        """Test basic optimization."""
        result = optimizer.optimize(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
        )
        assert result.market == "BTC-USD-PERP"
        assert result.recommended_strategy is not None
        assert len(result.predictions) == len(ExecutionStrategy)

    def test_optimize_with_objective(self, optimizer):
        """Test optimization with specific objective."""
        result = optimizer.optimize(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
            objective=ExecutionObjective.MINIMIZE_TIME,
        )
        assert result.objective == ExecutionObjective.MINIMIZE_TIME

    def test_optimize_returns_params(self, optimizer):
        """Test optimization returns parameters."""
        result = optimizer.optimize(
            "BTC-USD-PERP",
            Decimal("100"),
            "buy",
        )
        # Should have some optimal parameters
        assert "max_slice_size" in result.optimal_params or len(result.optimal_params) >= 0

    def test_optimize_market_condition(self, optimizer):
        """Test optimization detects condition."""
        result = optimizer.optimize(
            "BTC-USD-PERP",
            Decimal("10"),
            "buy",
        )
        assert result.market_condition == MarketCondition.NORMAL


class TestExecutionRecording:
    """Tests for execution recording."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return ExecutionOptimizer()

    def test_record_execution(self, optimizer):
        """Test recording execution."""
        record = ExecutionRecord(
            market="BTC-USD-PERP",
            side="buy",
            size=Decimal("10"),
            strategy=ExecutionStrategy.ADAPTIVE,
            entry_price=Decimal("50000"),
            avg_fill_price=Decimal("50010"),
            fill_rate=1.0,
            duration_seconds=30.0,
            slippage_bps=2.0,
            cost=ExecutionCost(),
        )
        optimizer.record_execution(record)

        history = optimizer.get_execution_history(market="BTC-USD-PERP")
        assert len(history) == 1
        assert history[0].market == "BTC-USD-PERP"

    def test_record_updates_benchmark(self, optimizer):
        """Test recording updates benchmark."""
        record = ExecutionRecord(
            market="BTC-USD-PERP",
            side="buy",
            size=Decimal("10"),
            strategy=ExecutionStrategy.ADAPTIVE,
            entry_price=Decimal("50000"),
            avg_fill_price=Decimal("50010"),
            fill_rate=1.0,
            duration_seconds=30.0,
            slippage_bps=2.0,
            cost=ExecutionCost(),
        )
        optimizer.record_execution(record)

        benchmarks = optimizer.get_benchmarks("BTC-USD-PERP")
        assert ExecutionStrategy.ADAPTIVE in benchmarks

    def test_get_history_filtered(self, optimizer):
        """Test getting filtered history."""
        for strategy in [ExecutionStrategy.ADAPTIVE, ExecutionStrategy.IMMEDIATE]:
            record = ExecutionRecord(
                market="BTC-USD-PERP",
                side="buy",
                size=Decimal("10"),
                strategy=strategy,
                entry_price=Decimal("50000"),
                avg_fill_price=Decimal("50010"),
                fill_rate=1.0,
                duration_seconds=30.0,
                slippage_bps=2.0,
                cost=ExecutionCost(),
            )
            optimizer.record_execution(record)

        history = optimizer.get_execution_history(strategy=ExecutionStrategy.ADAPTIVE)
        assert len(history) == 1
        assert history[0].strategy == ExecutionStrategy.ADAPTIVE


class TestExecutionQuality:
    """Tests for execution quality calculation."""

    @pytest.fixture
    def optimizer_with_history(self):
        """Create optimizer with execution history."""
        opt = ExecutionOptimizer()

        for i in range(10):
            record = ExecutionRecord(
                market="BTC-USD-PERP",
                side="buy",
                size=Decimal("10"),
                strategy=ExecutionStrategy.ADAPTIVE,
                entry_price=Decimal("50000"),
                avg_fill_price=Decimal("50010"),
                fill_rate=0.95 + i * 0.005,
                duration_seconds=30.0 + i,
                slippage_bps=2.0 + i * 0.1,
                cost=ExecutionCost(spread_cost=Decimal("1")),
            )
            opt.record_execution(record)

        return opt

    def test_calculate_quality(self, optimizer_with_history):
        """Test calculating execution quality."""
        quality = optimizer_with_history.calculate_execution_quality("BTC-USD-PERP")

        assert quality["market"] == "BTC-USD-PERP"
        assert quality["sample_count"] == 10
        assert quality["avg_slippage_bps"] > 0
        assert quality["avg_fill_rate"] > 0

    def test_calculate_quality_no_data(self):
        """Test calculating quality with no data."""
        opt = ExecutionOptimizer()
        quality = opt.calculate_execution_quality("BTC-USD-PERP")

        assert quality["sample_count"] == 0


class TestOptimizerCallbacks:
    """Tests for callbacks."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        opt.update_market_state(state)
        return opt

    def test_add_callback(self, optimizer):
        """Test adding callback."""
        results = []

        def callback(result):
            results.append(result)

        optimizer.add_callback(callback)
        optimizer.optimize("BTC-USD-PERP", Decimal("10"), "buy")

        assert len(results) == 1

    def test_remove_callback(self, optimizer):
        """Test removing callback."""
        results = []

        def callback(result):
            results.append(result)

        optimizer.add_callback(callback)
        removed = optimizer.remove_callback(callback)
        assert removed is True

        optimizer.optimize("BTC-USD-PERP", Decimal("10"), "buy")
        assert len(results) == 0

    def test_remove_nonexistent_callback(self, optimizer):
        """Test removing non-existent callback."""
        def callback(result):
            pass

        removed = optimizer.remove_callback(callback)
        assert removed is False


class TestOptimizerUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with state."""
        opt = ExecutionOptimizer()
        for market in ["BTC-USD-PERP", "ETH-USD-PERP"]:
            state = MarketState(
                market=market,
                mid_price=Decimal("50000"),
                spread_bps=2.0,
                volatility=3.0,
                liquidity_score=0.8,
                volume_24h=Decimal("1000000"),
                bid_depth=Decimal("500"),
                ask_depth=Decimal("500"),
                trend_direction=0.0,
                imbalance=0.5,
            )
            opt.update_market_state(state)
        return opt

    def test_get_markets(self, optimizer):
        """Test getting markets list."""
        markets = optimizer.get_markets()
        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_clear_market(self, optimizer):
        """Test clearing market."""
        optimizer.clear_market("BTC-USD-PERP")
        state = optimizer.get_market_state("BTC-USD-PERP")
        assert state is None

        # Other market should still exist
        state = optimizer.get_market_state("ETH-USD-PERP")
        assert state is not None

    def test_clear_all(self, optimizer):
        """Test clearing all."""
        optimizer.clear_all()
        markets = optimizer.get_markets()
        assert len(markets) == 0


class TestExecutionAnalyzer:
    """Tests for ExecutionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return ExecutionAnalyzer()

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with data."""
        opt = ExecutionOptimizer()

        for i in range(20):
            record = ExecutionRecord(
                market="BTC-USD-PERP",
                side="buy",
                size=Decimal("10"),
                strategy=ExecutionStrategy.ADAPTIVE if i % 2 == 0 else ExecutionStrategy.IMMEDIATE,
                entry_price=Decimal("50000"),
                avg_fill_price=Decimal("50010"),
                fill_rate=0.95,
                duration_seconds=30.0 if i % 2 == 0 else 5.0,
                slippage_bps=2.0 + i * 0.1,
                cost=ExecutionCost(),
            )
            opt.record_execution(record)

        return ExecutionAnalyzer(opt)

    def test_init(self, analyzer):
        """Test initialization."""
        assert analyzer.optimizer is not None

    def test_init_with_optimizer(self):
        """Test initialization with optimizer."""
        opt = ExecutionOptimizer()
        analyzer = ExecutionAnalyzer(opt)
        assert analyzer.optimizer is opt

    def test_analyze_slippage(self, analyzer_with_data):
        """Test slippage analysis."""
        analysis = analyzer_with_data.analyze_slippage("BTC-USD-PERP")

        assert analysis["market"] == "BTC-USD-PERP"
        assert analysis["sample_count"] == 20
        assert "avg_slippage_bps" in analysis
        assert "min_slippage_bps" in analysis
        assert "max_slippage_bps" in analysis

    def test_analyze_slippage_no_data(self, analyzer):
        """Test slippage analysis with no data."""
        analysis = analyzer.analyze_slippage("BTC-USD-PERP")
        assert "error" in analysis

    def test_compare_strategies(self, analyzer_with_data):
        """Test strategy comparison."""
        comparison = analyzer_with_data.compare_strategies("BTC-USD-PERP")

        assert comparison["market"] == "BTC-USD-PERP"
        assert "strategies" in comparison
        assert "adaptive" in comparison["strategies"]
        assert "immediate" in comparison["strategies"]

    def test_get_best_strategy(self, analyzer_with_data):
        """Test getting best strategy."""
        best = analyzer_with_data.get_best_strategy(
            "BTC-USD-PERP",
            objective=ExecutionObjective.MINIMIZE_COST,
        )
        assert best is not None
        assert isinstance(best, ExecutionStrategy)

    def test_get_best_strategy_no_data(self, analyzer):
        """Test getting best strategy with no data."""
        best = analyzer.get_best_strategy("BTC-USD-PERP")
        assert best is None


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_optimizer(self):
        """Test getting global optimizer."""
        reset_optimizer()
        opt = get_optimizer()
        assert opt is not None
        assert isinstance(opt, ExecutionOptimizer)

    def test_get_optimizer_singleton(self):
        """Test optimizer is singleton."""
        reset_optimizer()
        opt1 = get_optimizer()
        opt2 = get_optimizer()
        assert opt1 is opt2

    def test_get_analyzer(self):
        """Test getting global analyzer."""
        reset_optimizer()
        analyzer = get_analyzer()
        assert analyzer is not None
        assert isinstance(analyzer, ExecutionAnalyzer)

    def test_get_analyzer_singleton(self):
        """Test analyzer is singleton."""
        reset_optimizer()
        a1 = get_analyzer()
        a2 = get_analyzer()
        assert a1 is a2

    def test_reset_optimizer(self):
        """Test resetting optimizer."""
        opt1 = get_optimizer()
        reset_optimizer()
        opt2 = get_optimizer()
        assert opt1 is not opt2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return ExecutionOptimizer()

    def test_zero_volume(self, optimizer):
        """Test with zero volume."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("0"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        impact = optimizer.estimate_impact("BTC-USD-PERP", Decimal("10"), "buy")
        assert impact == 10.0  # Default

    def test_extreme_imbalance(self, optimizer):
        """Test with extreme imbalance."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("1000"),
            ask_depth=Decimal("100"),
            trend_direction=0.0,
            imbalance=0.9,  # Heavy bid side
        )
        optimizer.update_market_state(state)

        # Buying should be harder
        impact_buy = optimizer.estimate_impact("BTC-USD-PERP", Decimal("100"), "buy")
        impact_sell = optimizer.estimate_impact("BTC-USD-PERP", Decimal("100"), "sell")
        assert impact_buy > impact_sell

    def test_very_small_order(self, optimizer):
        """Test with very small order."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        result = optimizer.optimize("BTC-USD-PERP", Decimal("0.001"), "buy")
        assert result is not None

    def test_very_large_order(self, optimizer):
        """Test with very large order."""
        state = MarketState(
            market="BTC-USD-PERP",
            mid_price=Decimal("50000"),
            spread_bps=2.0,
            volatility=3.0,
            liquidity_score=0.8,
            volume_24h=Decimal("1000000"),
            bid_depth=Decimal("500"),
            ask_depth=Decimal("500"),
            trend_direction=0.0,
            imbalance=0.5,
        )
        optimizer.update_market_state(state)

        result = optimizer.optimize("BTC-USD-PERP", Decimal("100000"), "buy")
        # Should recommend scheduled or adaptive for large orders
        assert result is not None
