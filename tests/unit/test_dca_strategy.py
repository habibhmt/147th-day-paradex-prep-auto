"""
Tests for DCA Strategy Engine.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategies.dca_strategy import (
    DCAType,
    DCAState,
    TriggerType,
    OrderStatus,
    DCAOrder,
    DCAConfig,
    DCAMetrics,
    PricePoint,
    DCACalculator,
    DCAStrategy,
    DCAStrategyManager,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestDCAType:
    """Tests for DCAType enum."""

    def test_all_types_defined(self):
        """Test all DCA types are defined."""
        assert DCAType.TIME_BASED.value == "time_based"
        assert DCAType.PRICE_BASED.value == "price_based"
        assert DCAType.HYBRID.value == "hybrid"
        assert DCAType.SMART.value == "smart"


class TestDCAState:
    """Tests for DCAState enum."""

    def test_all_states_defined(self):
        """Test all DCA states are defined."""
        assert DCAState.INACTIVE.value == "inactive"
        assert DCAState.ACTIVE.value == "active"
        assert DCAState.PAUSED.value == "paused"
        assert DCAState.COMPLETED.value == "completed"
        assert DCAState.CANCELLED.value == "cancelled"


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_all_triggers_defined(self):
        """Test all trigger types are defined."""
        assert TriggerType.SCHEDULED.value == "scheduled"
        assert TriggerType.PRICE_DROP.value == "price_drop"
        assert TriggerType.PRICE_TARGET.value == "price_target"
        assert TriggerType.MANUAL.value == "manual"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_all_statuses_defined(self):
        """Test all order statuses are defined."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PLACED.value == "placed"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.FAILED.value == "failed"
        assert OrderStatus.SKIPPED.value == "skipped"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDCAOrder:
    """Tests for DCAOrder dataclass."""

    def test_creation(self):
        """Test DCAOrder creation."""
        order = DCAOrder(
            order_id="test_1",
            sequence_number=1,
            trigger_type=TriggerType.SCHEDULED,
            planned_amount=Decimal("1000")
        )
        assert order.order_id == "test_1"
        assert order.sequence_number == 1
        assert order.status == OrderStatus.PENDING

    def test_to_dict(self):
        """Test conversion to dictionary."""
        order = DCAOrder(
            order_id="test_1",
            sequence_number=1,
            trigger_type=TriggerType.PRICE_DROP,
            planned_amount=Decimal("1000"),
            fill_price=Decimal("50000"),
            status=OrderStatus.FILLED
        )
        result = order.to_dict()
        assert result["order_id"] == "test_1"
        assert result["trigger_type"] == "price_drop"
        assert result["status"] == "filled"


class TestDCAConfig:
    """Tests for DCAConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            interval_hours=24,
            num_orders=10
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_total_amount(self):
        """Test invalid total amount."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("0"),
            num_orders=10
        )
        errors = config.validate()
        assert any("positive" in e for e in errors)

    def test_invalid_num_orders(self):
        """Test invalid number of orders."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=0
        )
        errors = config.validate()
        assert any("at least 1" in e for e in errors)

    def test_too_many_orders(self):
        """Test too many orders."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=1500
        )
        errors = config.validate()
        assert any("cannot exceed 1000" in e for e in errors)

    def test_invalid_price_drop_pct(self):
        """Test invalid price drop percentage."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.PRICE_BASED,
            total_amount=Decimal("10000"),
            num_orders=10,
            price_drop_pct=Decimal("60")  # Too high
        )
        errors = config.validate()
        assert any("too high" in e for e in errors)

    def test_invalid_amount_range(self):
        """Test invalid min/max amount range."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=10,
            max_single_amount=Decimal("500"),
            min_single_amount=Decimal("1000")  # Min > max
        )
        errors = config.validate()
        assert any("must be >=" in e for e in errors)

    def test_invalid_time_range(self):
        """Test invalid start/end time range."""
        now = datetime.now()
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=10,
            start_time=now + timedelta(hours=1),
            end_time=now  # End before start
        )
        errors = config.validate()
        assert any("after start" in e for e in errors)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.HYBRID,
            total_amount=Decimal("10000"),
            num_orders=10
        )
        result = config.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["dca_type"] == "hybrid"


class TestDCAMetrics:
    """Tests for DCAMetrics dataclass."""

    def test_creation(self):
        """Test DCAMetrics creation."""
        metrics = DCAMetrics()
        assert metrics.total_invested == Decimal("0")
        assert metrics.orders_executed == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DCAMetrics(
            total_invested=Decimal("5000"),
            total_size=Decimal("0.1"),
            average_price=Decimal("50000"),
            orders_executed=5
        )
        result = metrics.to_dict()
        assert result["total_invested"] == "5000"
        assert result["orders_executed"] == 5


class TestPricePoint:
    """Tests for PricePoint dataclass."""

    def test_creation(self):
        """Test PricePoint creation."""
        now = datetime.now()
        point = PricePoint(timestamp=now, price=Decimal("50000"))
        assert point.price == Decimal("50000")


# =============================================================================
# DCA Calculator Tests
# =============================================================================

class TestDCACalculator:
    """Tests for DCACalculator."""

    def test_calculate_fixed_amounts(self):
        """Test fixed amount calculation."""
        amounts = DCACalculator.calculate_fixed_amounts(
            total_amount=Decimal("10000"),
            num_orders=10
        )
        assert len(amounts) == 10
        assert sum(amounts) == Decimal("10000")
        assert all(a == Decimal("1000") for a in amounts)

    def test_calculate_fixed_amounts_single(self):
        """Test fixed amounts with single order."""
        amounts = DCACalculator.calculate_fixed_amounts(
            total_amount=Decimal("5000"),
            num_orders=1
        )
        assert len(amounts) == 1
        assert amounts[0] == Decimal("5000")

    def test_calculate_fixed_amounts_odd_total(self):
        """Test fixed amounts with non-divisible total."""
        amounts = DCACalculator.calculate_fixed_amounts(
            total_amount=Decimal("10000"),
            num_orders=3
        )
        assert len(amounts) == 3
        assert sum(amounts) == Decimal("10000")

    def test_calculate_weighted_amounts(self):
        """Test weighted amount calculation."""
        amounts = DCACalculator.calculate_weighted_amounts(
            total_amount=Decimal("10000"),
            num_orders=5,
            weight_start=Decimal("0.5"),
            weight_end=Decimal("1.5")
        )
        assert len(amounts) == 5
        assert sum(amounts) == Decimal("10000")
        # First should be smaller than last
        assert amounts[0] < amounts[-1]

    def test_calculate_weighted_amounts_single(self):
        """Test weighted amounts with single order."""
        amounts = DCACalculator.calculate_weighted_amounts(
            total_amount=Decimal("5000"),
            num_orders=1
        )
        assert len(amounts) == 1
        assert amounts[0] == Decimal("5000")

    def test_calculate_schedule(self):
        """Test schedule calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        schedule = DCACalculator.calculate_schedule(
            start_time=start,
            interval_hours=24,
            num_orders=5
        )
        assert len(schedule) == 5
        assert schedule[0] == start
        assert schedule[1] == start + timedelta(hours=24)

    def test_calculate_dip_amount_no_dip(self):
        """Test dip amount with no price drop."""
        amount = DCACalculator.calculate_dip_amount(
            base_amount=Decimal("1000"),
            current_price=Decimal("50000"),
            reference_price=Decimal("50000")
        )
        assert amount == Decimal("1000")

    def test_calculate_dip_amount_with_dip(self):
        """Test dip amount with price drop."""
        amount = DCACalculator.calculate_dip_amount(
            base_amount=Decimal("1000"),
            current_price=Decimal("45000"),  # 10% dip
            reference_price=Decimal("50000"),
            dip_multiplier=Decimal("2")
        )
        # Should be more than base amount
        assert amount > Decimal("1000")

    def test_calculate_dip_amount_respects_max(self):
        """Test dip amount respects max multiplier."""
        amount = DCACalculator.calculate_dip_amount(
            base_amount=Decimal("1000"),
            current_price=Decimal("25000"),  # 50% dip
            reference_price=Decimal("50000"),
            dip_multiplier=Decimal("2"),
            max_multiplier=Decimal("3")
        )
        assert amount <= Decimal("3000")

    def test_calculate_volatility_adjusted_normal(self):
        """Test volatility adjustment with normal volatility."""
        amount = DCACalculator.calculate_volatility_adjusted_amount(
            base_amount=Decimal("1000"),
            current_volatility=Decimal("0.02"),
            average_volatility=Decimal("0.02")  # Same
        )
        assert amount == Decimal("1000")

    def test_calculate_volatility_adjusted_high(self):
        """Test volatility adjustment with high volatility."""
        amount = DCACalculator.calculate_volatility_adjusted_amount(
            base_amount=Decimal("1000"),
            current_volatility=Decimal("0.04"),  # 2x average
            average_volatility=Decimal("0.02"),
            volatility_multiplier=Decimal("1.5")
        )
        assert amount > Decimal("1000")


# =============================================================================
# DCA Strategy Tests
# =============================================================================

class TestDCAStrategy:
    """Tests for DCAStrategy."""

    @pytest.fixture
    def config(self):
        """Create valid config."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            interval_hours=24,
            num_orders=10
        )

    @pytest.fixture
    def strategy(self, config):
        """Create strategy instance."""
        return DCAStrategy(config, "test_dca")

    def test_strategy_creation(self, strategy):
        """Test strategy creation."""
        assert strategy.strategy_id == "test_dca"
        assert strategy.state == DCAState.INACTIVE
        assert len(strategy._orders) == 10

    def test_strategy_invalid_config(self):
        """Test strategy with invalid config."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("0"),
            num_orders=10
        )
        with pytest.raises(ValueError):
            DCAStrategy(config)

    def test_start_strategy(self, strategy):
        """Test starting strategy."""
        strategy.start(Decimal("50000"))
        assert strategy.state == DCAState.ACTIVE

    def test_cannot_restart_active(self, strategy):
        """Test cannot restart active strategy."""
        strategy.start(Decimal("50000"))
        strategy.start(Decimal("51000"))  # Should be ignored
        assert strategy._current_price == Decimal("50000")

    def test_check_time_trigger(self, strategy):
        """Test time-based trigger."""
        strategy.start(Decimal("50000"))

        # Simulate time passing
        now = datetime.now()
        for order in strategy._orders:
            order.triggered_at = now - timedelta(hours=1)  # Past trigger time

        orders = strategy.check_triggers(Decimal("50000"), now)
        assert len(orders) > 0

    def test_check_trigger_respects_max_price(self, strategy):
        """Test trigger respects max price."""
        strategy.config.max_price = Decimal("49000")
        strategy.start(Decimal("48000"))

        # Set triggers to now
        now = datetime.now()
        for order in strategy._orders:
            order.triggered_at = now - timedelta(hours=1)

        # Price above max - should not trigger
        orders = strategy.check_triggers(Decimal("50000"), now)
        assert len(orders) == 0

    def test_on_order_filled(self, strategy):
        """Test handling order fill."""
        strategy.start(Decimal("50000"))

        # Get first order
        order = strategy._orders[0]
        order.status = OrderStatus.PLACED

        strategy.on_order_filled(
            order_id=order.order_id,
            fill_price=Decimal("49000"),
            size=Decimal("0.02")
        )

        assert strategy.metrics.orders_executed == 1
        assert strategy.metrics.total_invested == Decimal("980")
        assert order.status == OrderStatus.FILLED

    def test_on_order_failed(self, strategy):
        """Test handling order failure."""
        strategy.start(Decimal("50000"))

        order = strategy._orders[0]
        order.status = OrderStatus.PLACED

        strategy.on_order_failed(order.order_id, "Insufficient funds")

        assert strategy.metrics.orders_failed == 1
        assert order.status == OrderStatus.FAILED

    def test_skip_order(self, strategy):
        """Test skipping an order."""
        strategy.start(Decimal("50000"))

        order = strategy._orders[0]
        strategy.skip_order(order.order_id, "Skipped by user")

        assert strategy.metrics.orders_skipped == 1
        assert order.status == OrderStatus.SKIPPED

    def test_update_metrics(self, strategy):
        """Test metrics update."""
        strategy.start(Decimal("50000"))

        # Simulate a fill
        order = strategy._orders[0]
        order.status = OrderStatus.PLACED
        strategy.on_order_filled(
            order.order_id,
            Decimal("50000"),
            Decimal("0.02")
        )

        # Update with higher price
        strategy.update_metrics(Decimal("55000"))

        assert strategy.metrics.unrealized_pnl > Decimal("0")
        assert strategy.metrics.unrealized_pnl_pct > Decimal("0")

    def test_check_stop_loss(self, strategy):
        """Test stop loss check."""
        strategy.config.stop_loss_pct = Decimal("10")
        strategy.start(Decimal("50000"))

        # Simulate fill
        order = strategy._orders[0]
        order.status = OrderStatus.PLACED
        strategy.on_order_filled(
            order.order_id,
            Decimal("50000"),
            Decimal("0.02")
        )

        # Price drops 15%
        result = strategy.check_exit_conditions(Decimal("42500"))
        assert result["should_exit"]
        assert result["reason"] == "stop_loss"

    def test_check_take_profit(self, strategy):
        """Test take profit check."""
        strategy.config.take_profit_pct = Decimal("20")
        strategy.start(Decimal("50000"))

        # Simulate fill
        order = strategy._orders[0]
        order.status = OrderStatus.PLACED
        strategy.on_order_filled(
            order.order_id,
            Decimal("50000"),
            Decimal("0.02")
        )

        # Price rises 25%
        result = strategy.check_exit_conditions(Decimal("62500"))
        assert result["should_exit"]
        assert result["reason"] == "take_profit"

    def test_pause_resume(self, strategy):
        """Test pause and resume."""
        strategy.start(Decimal("50000"))

        strategy.pause()
        assert strategy.state == DCAState.PAUSED

        strategy.resume()
        assert strategy.state == DCAState.ACTIVE

    def test_cancel(self, strategy):
        """Test strategy cancellation."""
        strategy.start(Decimal("50000"))

        result = strategy.cancel()
        assert strategy.state == DCAState.CANCELLED
        assert "cancelled_orders" in result

    def test_trigger_manual_order(self, strategy):
        """Test manual order trigger."""
        strategy.start(Decimal("50000"))

        order = strategy.trigger_manual_order(
            Decimal("49000"),
            Decimal("500")
        )

        assert order is not None
        assert order.trigger_type == TriggerType.MANUAL
        assert order.actual_amount == Decimal("500")

    def test_get_status(self, strategy):
        """Test getting strategy status."""
        strategy.start(Decimal("50000"))

        status = strategy.get_status()
        assert status["strategy_id"] == "test_dca"
        assert status["state"] == "active"
        assert "metrics" in status
        assert "progress" in status

    def test_get_orders(self, strategy):
        """Test getting all orders."""
        orders = strategy.get_orders()
        assert len(orders) == 10
        assert all("order_id" in o for o in orders)

    def test_get_next_order(self, strategy):
        """Test getting next pending order."""
        strategy.start(Decimal("50000"))

        next_order = strategy.get_next_order()
        assert next_order is not None
        assert next_order.status == OrderStatus.PENDING


class TestDCAStrategyPriceBased:
    """Tests for price-based DCA strategy."""

    @pytest.fixture
    def config(self):
        """Create price-based config."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.PRICE_BASED,
            total_amount=Decimal("10000"),
            num_orders=10,
            price_drop_pct=Decimal("5")
        )

    @pytest.fixture
    def strategy(self, config):
        """Create strategy instance."""
        return DCAStrategy(config, "test_price_dca")

    def test_price_drop_trigger(self, strategy):
        """Test price drop trigger."""
        strategy.start(Decimal("50000"))

        # Price drops 5%
        orders = strategy.check_triggers(Decimal("47500"))
        assert len(orders) == 1
        assert orders[0].trigger_type == TriggerType.PRICE_DROP

    def test_no_trigger_small_drop(self, strategy):
        """Test no trigger on small price drop."""
        strategy.start(Decimal("50000"))

        # Price drops only 3%
        orders = strategy.check_triggers(Decimal("48500"))
        assert len(orders) == 0


class TestDCAStrategyHybrid:
    """Tests for hybrid DCA strategy."""

    @pytest.fixture
    def config(self):
        """Create hybrid config."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.HYBRID,
            total_amount=Decimal("10000"),
            interval_hours=24,
            num_orders=10,
            price_drop_pct=Decimal("5")
        )

    def test_hybrid_time_trigger(self, config):
        """Test hybrid strategy time trigger."""
        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))

        now = datetime.now()
        for order in strategy._orders:
            order.triggered_at = now - timedelta(hours=1)

        orders = strategy.check_triggers(Decimal("50000"), now)
        assert len(orders) > 0

    def test_hybrid_price_trigger(self, config):
        """Test hybrid strategy price trigger."""
        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))

        # Price drops 5%
        orders = strategy.check_triggers(Decimal("47500"))
        assert len(orders) > 0


class TestDCAStrategySmart:
    """Tests for smart DCA strategy."""

    @pytest.fixture
    def config(self):
        """Create smart config."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.SMART,
            total_amount=Decimal("10000"),
            interval_hours=24,
            num_orders=10,
            dip_multiplier=Decimal("2"),
            volatility_multiplier=Decimal("1.5")
        )

    def test_smart_weighted_amounts(self, config):
        """Test smart strategy uses weighted amounts."""
        strategy = DCAStrategy(config)
        amounts = [o.planned_amount for o in strategy._orders]
        # First orders should be smaller
        assert amounts[0] < amounts[-1]

    def test_smart_dip_adjustment(self, config):
        """Test smart strategy adjusts for dips."""
        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))

        now = datetime.now()
        strategy._orders[0].triggered_at = now - timedelta(hours=1)

        # Trigger at 10% dip
        orders = strategy.check_triggers(Decimal("45000"), now)
        if orders:
            # Amount should be adjusted up
            base = strategy._orders[0].planned_amount
            assert orders[0].actual_amount >= base


class TestDCAStrategyWithConstraints:
    """Tests for DCA strategy with constraints."""

    @pytest.fixture
    def config(self):
        """Create config with constraints."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=10,
            min_single_amount=Decimal("500"),
            max_single_amount=Decimal("2000"),
            max_price=Decimal("60000"),
            min_price=Decimal("40000")
        )

    def test_respects_min_amount(self, config):
        """Test strategy respects min amount."""
        strategy = DCAStrategy(config)
        for order in strategy._orders:
            assert order.planned_amount >= Decimal("500")

    def test_respects_max_amount(self, config):
        """Test strategy respects max amount."""
        strategy = DCAStrategy(config)
        for order in strategy._orders:
            assert order.planned_amount <= Decimal("2000")


# =============================================================================
# DCA Strategy Manager Tests
# =============================================================================

class TestDCAStrategyManager:
    """Tests for DCAStrategyManager."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return DCAStrategyManager()

    @pytest.fixture
    def config(self):
        """Create config."""
        return DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=10
        )

    def test_create_strategy(self, manager, config):
        """Test strategy creation."""
        strategy = manager.create_strategy(config, "test_1")
        assert strategy.strategy_id == "test_1"
        assert manager.get_strategy("test_1") is not None

    def test_get_strategy_not_found(self, manager):
        """Test getting non-existent strategy."""
        assert manager.get_strategy("nonexistent") is None

    def test_list_strategies(self, manager, config):
        """Test listing strategies."""
        manager.create_strategy(config, "test_1")
        config2 = DCAConfig(
            symbol="ETH-USD-PERP",
            dca_type=DCAType.PRICE_BASED,
            total_amount=Decimal("5000"),
            num_orders=5
        )
        manager.create_strategy(config2, "test_2")

        strategies = manager.list_strategies()
        assert len(strategies) == 2
        assert any(s["strategy_id"] == "test_1" for s in strategies)
        assert any(s["strategy_id"] == "test_2" for s in strategies)

    def test_get_active_strategies(self, manager, config):
        """Test getting active strategies."""
        strategy1 = manager.create_strategy(config, "test_1")
        strategy1.start(Decimal("50000"))

        strategy2 = manager.create_strategy(
            DCAConfig(
                symbol="ETH-USD-PERP",
                dca_type=DCAType.TIME_BASED,
                total_amount=Decimal("5000"),
                num_orders=5
            ),
            "test_2"
        )
        # test_2 not started

        active = manager.get_active_strategies()
        assert len(active) == 1
        assert active[0].strategy_id == "test_1"

    def test_remove_strategy(self, manager, config):
        """Test removing strategy."""
        manager.create_strategy(config, "test_1")
        assert manager.remove_strategy("test_1")
        assert manager.get_strategy("test_1") is None

    def test_remove_nonexistent_strategy(self, manager):
        """Test removing non-existent strategy."""
        assert not manager.remove_strategy("nonexistent")

    def test_check_all_triggers(self, manager, config):
        """Test checking triggers for all strategies."""
        strategy1 = manager.create_strategy(config, "test_1")
        strategy1.start(Decimal("50000"))

        # Set trigger time to past
        now = datetime.now()
        for order in strategy1._orders:
            order.triggered_at = now - timedelta(hours=1)

        prices = {"BTC-USD-PERP": Decimal("50000")}
        results = manager.check_all_triggers(prices, now)

        assert "test_1" in results
        assert len(results["test_1"]) > 0

    def test_get_total_metrics(self, manager, config):
        """Test aggregated metrics."""
        strategy1 = manager.create_strategy(config, "test_1")
        strategy1.start(Decimal("50000"))

        # Simulate fills
        strategy1._orders[0].status = OrderStatus.PLACED
        strategy1.on_order_filled(
            strategy1._orders[0].order_id,
            Decimal("50000"),
            Decimal("0.02")
        )
        strategy1.update_metrics(Decimal("50000"))

        metrics = manager.get_total_metrics()
        assert metrics["total_strategies"] == 1
        assert metrics["active_strategies"] == 1
        assert Decimal(metrics["total_invested"]) > Decimal("0")


# =============================================================================
# Integration Tests
# =============================================================================

class TestDCAIntegration:
    """Integration tests for DCA strategy."""

    def test_full_dca_lifecycle(self):
        """Test complete DCA lifecycle."""
        # Create config
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            interval_hours=1,
            num_orders=5,
            stop_loss_pct=Decimal("20"),
            take_profit_pct=Decimal("50")
        )

        # Create and start
        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))
        assert strategy.state == DCAState.ACTIVE

        # Simulate time passing and fills
        now = datetime.now()
        filled_count = 0

        for i, order in enumerate(strategy._orders):
            order.triggered_at = now - timedelta(hours=1)

            # Check trigger
            orders = strategy.check_triggers(Decimal("50000") - i * 500, now)
            if orders:
                order_to_fill = orders[0]
                # Simulate fill
                strategy.on_order_filled(
                    order_to_fill.order_id,
                    order_to_fill.planned_price,
                    order_to_fill.actual_amount / order_to_fill.planned_price
                )
                filled_count += 1

        # Check metrics
        assert strategy.metrics.orders_executed == filled_count
        assert strategy.metrics.total_invested > Decimal("0")

    def test_dca_with_price_drops(self):
        """Test DCA execution with price drops."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.PRICE_BASED,
            total_amount=Decimal("5000"),
            num_orders=5,
            price_drop_pct=Decimal("5")
        )

        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))

        # Simulate price drops
        prices = [
            Decimal("47500"),  # 5% drop - trigger
            Decimal("46000"),  # Not enough from last trigger
            Decimal("45125"),  # 5% from 47500 - trigger
        ]

        total_fills = 0
        for price in prices:
            orders = strategy.check_triggers(price)
            for order in orders:
                strategy.on_order_filled(
                    order.order_id,
                    price,
                    order.actual_amount / price
                )
                total_fills += 1

        assert total_fills >= 2

    def test_manager_multiple_strategies(self):
        """Test managing multiple strategies."""
        manager = DCAStrategyManager()

        # Create BTC strategy
        btc_config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("10000"),
            num_orders=10
        )
        btc = manager.create_strategy(btc_config, "btc_dca")
        btc.start(Decimal("50000"))

        # Create ETH strategy
        eth_config = DCAConfig(
            symbol="ETH-USD-PERP",
            dca_type=DCAType.PRICE_BASED,
            total_amount=Decimal("5000"),
            num_orders=5
        )
        eth = manager.create_strategy(eth_config, "eth_dca")
        eth.start(Decimal("3000"))

        # Check metrics
        metrics = manager.get_total_metrics()
        assert metrics["total_strategies"] == 2
        assert metrics["active_strategies"] == 2

    def test_strategy_completion(self):
        """Test strategy completion after all orders."""
        config = DCAConfig(
            symbol="BTC-USD-PERP",
            dca_type=DCAType.TIME_BASED,
            total_amount=Decimal("3000"),
            interval_hours=1,
            num_orders=3
        )

        strategy = DCAStrategy(config)
        strategy.start(Decimal("50000"))

        now = datetime.now()

        # Fill all orders
        for order in strategy._orders:
            order.triggered_at = now - timedelta(hours=1)
            orders = strategy.check_triggers(Decimal("50000"), now)
            if orders:
                strategy.on_order_filled(
                    orders[0].order_id,
                    Decimal("50000"),
                    orders[0].actual_amount / Decimal("50000")
                )

        # Check for completion
        strategy.check_triggers(Decimal("50000"))

        # All orders should be executed
        assert strategy.metrics.orders_executed == 3
        assert strategy.metrics.orders_pending == 0
