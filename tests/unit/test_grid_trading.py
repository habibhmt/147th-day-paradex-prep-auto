"""
Tests for Grid Trading Strategy Module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategies.grid_trading import (
    GridType,
    GridState,
    OrderSide,
    GridOrderStatus,
    GridLevel,
    GridConfig,
    GridMetrics,
    GridSnapshot,
    GridCalculator,
    GridOrderManager,
    GridStrategy,
    GridStrategyManager,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestGridType:
    """Tests for GridType enum."""

    def test_all_types_defined(self):
        """Test all grid types are defined."""
        assert GridType.ARITHMETIC.value == "arithmetic"
        assert GridType.GEOMETRIC.value == "geometric"
        assert GridType.CUSTOM.value == "custom"


class TestGridState:
    """Tests for GridState enum."""

    def test_all_states_defined(self):
        """Test all grid states are defined."""
        assert GridState.INACTIVE.value == "inactive"
        assert GridState.INITIALIZING.value == "initializing"
        assert GridState.ACTIVE.value == "active"
        assert GridState.PAUSED.value == "paused"
        assert GridState.CLOSING.value == "closing"
        assert GridState.CLOSED.value == "closed"
        assert GridState.ERROR.value == "error"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_sides_defined(self):
        """Test order sides are defined."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestGridOrderStatus:
    """Tests for GridOrderStatus enum."""

    def test_all_statuses_defined(self):
        """Test all order statuses are defined."""
        assert GridOrderStatus.PENDING.value == "pending"
        assert GridOrderStatus.OPEN.value == "open"
        assert GridOrderStatus.FILLED.value == "filled"
        assert GridOrderStatus.CANCELLED.value == "cancelled"
        assert GridOrderStatus.FAILED.value == "failed"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestGridLevel:
    """Tests for GridLevel dataclass."""

    def test_creation(self):
        """Test GridLevel creation."""
        level = GridLevel(
            level_id="test_1",
            price=Decimal("50000"),
            side=OrderSide.BUY,
            size=Decimal("0.1")
        )
        assert level.level_id == "test_1"
        assert level.price == Decimal("50000")
        assert level.side == OrderSide.BUY
        assert level.status == GridOrderStatus.PENDING

    def test_to_dict(self):
        """Test conversion to dictionary."""
        level = GridLevel(
            level_id="test_1",
            price=Decimal("50000"),
            side=OrderSide.SELL,
            size=Decimal("0.1"),
            order_id="order_123",
            status=GridOrderStatus.OPEN
        )
        result = level.to_dict()
        assert result["level_id"] == "test_1"
        assert result["side"] == "sell"
        assert result["status"] == "open"


class TestGridConfig:
    """Tests for GridConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000")
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_price_range(self):
        """Test invalid price range."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("45000"),
            lower_price=Decimal("55000"),  # Lower > upper
            num_grids=10,
            total_investment=Decimal("10000")
        )
        errors = config.validate()
        assert any("Upper price must be greater" in e for e in errors)

    def test_invalid_num_grids(self):
        """Test invalid number of grids."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=1,  # Too few
            total_investment=Decimal("10000")
        )
        errors = config.validate()
        assert any("at least 2" in e for e in errors)

    def test_too_many_grids(self):
        """Test too many grids."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=600,  # Too many
            total_investment=Decimal("10000")
        )
        errors = config.validate()
        assert any("cannot exceed 500" in e for e in errors)

    def test_invalid_investment(self):
        """Test invalid investment amount."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("0")
        )
        errors = config.validate()
        assert any("positive" in e for e in errors)

    def test_custom_grid_validation(self):
        """Test custom grid validation."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.CUSTOM,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=5,
            total_investment=Decimal("10000"),
            custom_levels=None  # Missing custom levels
        )
        errors = config.validate()
        assert any("Custom grid requires" in e for e in errors)

    def test_stop_loss_validation(self):
        """Test stop loss validation."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000"),
            stop_loss_price=Decimal("46000")  # Above lower price
        )
        errors = config.validate()
        assert any("Stop loss must be below" in e for e in errors)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.GEOMETRIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000")
        )
        result = config.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["grid_type"] == "geometric"


class TestGridMetrics:
    """Tests for GridMetrics dataclass."""

    def test_creation(self):
        """Test GridMetrics creation."""
        metrics = GridMetrics()
        assert metrics.total_pnl == Decimal("0")
        assert metrics.total_trades == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GridMetrics(
            total_pnl=Decimal("500"),
            total_trades=20
        )
        result = metrics.to_dict()
        assert result["total_pnl"] == "500"
        assert result["total_trades"] == 20


class TestGridSnapshot:
    """Tests for GridSnapshot dataclass."""

    def test_creation(self):
        """Test GridSnapshot creation."""
        metrics = GridMetrics()
        snapshot = GridSnapshot(
            timestamp=datetime.now(),
            current_price=Decimal("50000"),
            buy_levels=5,
            sell_levels=5,
            open_orders=10,
            filled_orders=5,
            position_size=Decimal("0.5"),
            position_value=Decimal("25000"),
            metrics=metrics
        )
        assert snapshot.buy_levels == 5
        assert snapshot.position_size == Decimal("0.5")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GridMetrics()
        snapshot = GridSnapshot(
            timestamp=datetime.now(),
            current_price=Decimal("50000"),
            buy_levels=5,
            sell_levels=5,
            open_orders=10,
            filled_orders=5,
            position_size=Decimal("0.5"),
            position_value=Decimal("25000"),
            metrics=metrics
        )
        result = snapshot.to_dict()
        assert result["buy_levels"] == 5
        assert "metrics" in result


# =============================================================================
# Grid Calculator Tests
# =============================================================================

class TestGridCalculator:
    """Tests for GridCalculator."""

    def test_arithmetic_levels_basic(self):
        """Test basic arithmetic level calculation."""
        levels = GridCalculator.calculate_arithmetic_levels(
            lower_price=Decimal("45000"),
            upper_price=Decimal("55000"),
            num_grids=11
        )
        assert len(levels) == 11
        assert levels[0] == Decimal("45000")
        assert levels[-1] == Decimal("55000")
        # Check equal spacing
        spacing = levels[1] - levels[0]
        for i in range(1, len(levels)):
            assert abs(levels[i] - levels[i-1] - spacing) < Decimal("0.01")

    def test_arithmetic_levels_two_grids(self):
        """Test arithmetic with minimum grids."""
        levels = GridCalculator.calculate_arithmetic_levels(
            lower_price=Decimal("100"),
            upper_price=Decimal("200"),
            num_grids=2
        )
        assert len(levels) == 2
        assert levels[0] == Decimal("100")
        assert levels[1] == Decimal("200")

    def test_geometric_levels_basic(self):
        """Test basic geometric level calculation."""
        levels = GridCalculator.calculate_geometric_levels(
            lower_price=Decimal("1000"),
            upper_price=Decimal("2000"),
            num_grids=5
        )
        assert len(levels) == 5
        assert levels[0] == Decimal("1000")
        # Check geometric spacing (equal percentage)
        for i in range(1, len(levels) - 1):
            ratio1 = levels[i] / levels[i-1]
            ratio2 = levels[i+1] / levels[i]
            assert abs(ratio1 - ratio2) < Decimal("0.01")

    def test_geometric_levels_two_grids(self):
        """Test geometric with minimum grids."""
        levels = GridCalculator.calculate_geometric_levels(
            lower_price=Decimal("100"),
            upper_price=Decimal("400"),
            num_grids=2
        )
        assert len(levels) == 2

    def test_order_sizes_balanced(self):
        """Test order size calculation with balanced levels."""
        levels = [
            Decimal("48000"),
            Decimal("49000"),
            Decimal("50000"),  # Current price
            Decimal("51000"),
            Decimal("52000")
        ]
        sizes = GridCalculator.calculate_order_sizes(
            levels=levels,
            total_investment=Decimal("10000"),
            current_price=Decimal("50000")
        )
        assert len(sizes) == 5
        assert all(s > Decimal("0") for s in sizes)

    def test_order_sizes_with_leverage(self):
        """Test order sizes with leverage."""
        levels = [Decimal("45000"), Decimal("50000"), Decimal("55000")]
        sizes_1x = GridCalculator.calculate_order_sizes(
            levels=levels,
            total_investment=Decimal("10000"),
            current_price=Decimal("50000"),
            leverage=Decimal("1")
        )
        sizes_2x = GridCalculator.calculate_order_sizes(
            levels=levels,
            total_investment=Decimal("10000"),
            current_price=Decimal("50000"),
            leverage=Decimal("2")
        )
        # 2x leverage should give roughly 2x sizes
        for s1, s2 in zip(sizes_1x, sizes_2x):
            if s1 > 0:
                assert abs(s2 / s1 - Decimal("2")) < Decimal("0.1")

    def test_grid_spacing_calculation(self):
        """Test grid spacing percentage calculation."""
        levels = [Decimal("100"), Decimal("110"), Decimal("121")]
        spacing = GridCalculator.calculate_grid_spacing_pct(levels)
        assert spacing > Decimal("9")  # ~10% spacing

    def test_grid_spacing_single_level(self):
        """Test grid spacing with single level."""
        spacing = GridCalculator.calculate_grid_spacing_pct([Decimal("100")])
        assert spacing == Decimal("0")

    def test_profit_estimation(self):
        """Test profit per grid estimation."""
        levels = [Decimal("100"), Decimal("105"), Decimal("110.25")]  # ~5% spacing
        profit = GridCalculator.estimate_profit_per_grid(
            levels=levels,
            fee_rate=Decimal("0.001")  # 0.1% fee
        )
        # Should be spacing - fees (~5% - 0.2%)
        assert profit > Decimal("4")


# =============================================================================
# Grid Order Manager Tests
# =============================================================================

class TestGridOrderManager:
    """Tests for GridOrderManager."""

    @pytest.fixture
    def manager(self):
        """Create order manager instance."""
        return GridOrderManager()

    @pytest.fixture
    def sample_level(self):
        """Create sample level."""
        return GridLevel(
            level_id="test_1",
            price=Decimal("50000"),
            side=OrderSide.BUY,
            size=Decimal("0.1")
        )

    def test_add_level(self, manager, sample_level):
        """Test adding a level."""
        manager.add_level(sample_level)
        assert manager.get_level("test_1") is not None

    def test_get_level_not_found(self, manager):
        """Test getting non-existent level."""
        assert manager.get_level("nonexistent") is None

    def test_update_order_id(self, manager, sample_level):
        """Test updating order ID."""
        manager.add_level(sample_level)
        manager.update_order_id("test_1", "order_123")
        level = manager.get_level("test_1")
        assert level.order_id == "order_123"
        assert level.status == GridOrderStatus.OPEN

    def test_get_level_by_order(self, manager, sample_level):
        """Test getting level by order ID."""
        manager.add_level(sample_level)
        manager.update_order_id("test_1", "order_123")
        level = manager.get_level_by_order("order_123")
        assert level is not None
        assert level.level_id == "test_1"

    def test_mark_filled(self, manager, sample_level):
        """Test marking level as filled."""
        manager.add_level(sample_level)
        manager.update_order_id("test_1", "order_123")
        manager.mark_filled("test_1", Decimal("50010"))
        level = manager.get_level("test_1")
        assert level.status == GridOrderStatus.FILLED
        assert level.fill_price == Decimal("50010")

    def test_mark_cancelled(self, manager, sample_level):
        """Test marking level as cancelled."""
        manager.add_level(sample_level)
        manager.update_order_id("test_1", "order_123")
        manager.mark_cancelled("test_1")
        level = manager.get_level("test_1")
        assert level.status == GridOrderStatus.CANCELLED
        assert level.order_id is None

    def test_get_open_levels(self, manager):
        """Test getting open levels."""
        for i in range(3):
            level = GridLevel(
                level_id=f"test_{i}",
                price=Decimal("50000") + i * 100,
                side=OrderSide.BUY,
                size=Decimal("0.1")
            )
            manager.add_level(level)
            manager.update_order_id(f"test_{i}", f"order_{i}")

        manager.mark_filled("test_0", Decimal("50000"))
        open_levels = manager.get_open_levels()
        assert len(open_levels) == 2

    def test_get_filled_levels(self, manager):
        """Test getting filled levels."""
        for i in range(3):
            level = GridLevel(
                level_id=f"test_{i}",
                price=Decimal("50000") + i * 100,
                side=OrderSide.BUY,
                size=Decimal("0.1")
            )
            manager.add_level(level)
            manager.update_order_id(f"test_{i}", f"order_{i}")

        manager.mark_filled("test_0", Decimal("50000"))
        manager.mark_filled("test_1", Decimal("50100"))
        filled_levels = manager.get_filled_levels()
        assert len(filled_levels) == 2

    def test_get_buy_sell_levels(self, manager):
        """Test getting buy and sell levels."""
        buy_level = GridLevel(
            level_id="buy_1",
            price=Decimal("49000"),
            side=OrderSide.BUY,
            size=Decimal("0.1")
        )
        sell_level = GridLevel(
            level_id="sell_1",
            price=Decimal("51000"),
            side=OrderSide.SELL,
            size=Decimal("0.1")
        )
        manager.add_level(buy_level)
        manager.add_level(sell_level)

        assert len(manager.get_buy_levels()) == 1
        assert len(manager.get_sell_levels()) == 1

    def test_get_all_levels_sorted(self, manager):
        """Test getting all levels sorted by price."""
        for price in [51000, 49000, 50000]:
            level = GridLevel(
                level_id=f"level_{price}",
                price=Decimal(str(price)),
                side=OrderSide.BUY,
                size=Decimal("0.1")
            )
            manager.add_level(level)

        all_levels = manager.get_all_levels()
        assert all_levels[0].price == Decimal("49000")
        assert all_levels[-1].price == Decimal("51000")

    def test_clear(self, manager, sample_level):
        """Test clearing all levels."""
        manager.add_level(sample_level)
        manager.clear()
        assert manager.get_level("test_1") is None

    def test_count_open_orders(self, manager):
        """Test counting open orders."""
        for i in range(5):
            level = GridLevel(
                level_id=f"test_{i}",
                price=Decimal("50000") + i * 100,
                side=OrderSide.BUY,
                size=Decimal("0.1")
            )
            manager.add_level(level)
            if i < 3:  # Only first 3 are open
                manager.update_order_id(f"test_{i}", f"order_{i}")

        assert manager.count_open_orders() == 3


# =============================================================================
# Grid Strategy Tests
# =============================================================================

class TestGridStrategy:
    """Tests for GridStrategy."""

    @pytest.fixture
    def config(self):
        """Create valid config."""
        return GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=11,
            total_investment=Decimal("10000")
        )

    @pytest.fixture
    def strategy(self, config):
        """Create strategy instance."""
        return GridStrategy(config, "test_strategy")

    def test_strategy_creation(self, strategy):
        """Test strategy creation."""
        assert strategy.strategy_id == "test_strategy"
        assert strategy.state == GridState.INACTIVE

    def test_strategy_invalid_config(self):
        """Test strategy with invalid config."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("45000"),
            lower_price=Decimal("55000"),  # Invalid
            num_grids=10,
            total_investment=Decimal("10000")
        )
        with pytest.raises(ValueError):
            GridStrategy(config)

    def test_initialize_grid(self, strategy):
        """Test grid initialization."""
        levels = strategy.initialize(Decimal("50000"))
        assert len(levels) > 0
        assert strategy.state == GridState.ACTIVE

    def test_initialize_creates_buy_and_sell(self, strategy):
        """Test initialization creates both buy and sell orders."""
        levels = strategy.initialize(Decimal("50000"))
        buy_levels = [l for l in levels if l.side == OrderSide.BUY]
        sell_levels = [l for l in levels if l.side == OrderSide.SELL]
        assert len(buy_levels) > 0
        assert len(sell_levels) > 0

    def test_cannot_reinitialize_active(self, strategy):
        """Test cannot reinitialize active grid."""
        strategy.initialize(Decimal("50000"))
        levels = strategy.initialize(Decimal("50000"))
        assert len(levels) == 0  # Should return empty

    def test_on_price_update_normal(self, strategy):
        """Test normal price update."""
        strategy.initialize(Decimal("50000"))
        result = strategy.on_price_update(Decimal("50500"))
        assert result["action"] == "none"
        assert not result["close_position"]

    def test_on_price_update_stop_loss(self, strategy):
        """Test stop loss trigger."""
        strategy.config.stop_loss_price = Decimal("44000")
        strategy.initialize(Decimal("50000"))
        result = strategy.on_price_update(Decimal("43000"))
        assert result["action"] == "stop_loss"
        assert result["close_position"]

    def test_on_price_update_take_profit(self, strategy):
        """Test take profit trigger."""
        strategy.config.take_profit_price = Decimal("60000")
        strategy.initialize(Decimal("50000"))
        result = strategy.on_price_update(Decimal("61000"))
        assert result["action"] == "take_profit"
        assert result["close_position"]

    def test_on_order_filled_buy(self, strategy):
        """Test handling buy order fill."""
        levels = strategy.initialize(Decimal("50000"))

        # Find a buy level and mark it as placed
        buy_level = next(l for l in levels if l.side == OrderSide.BUY)
        strategy.order_manager.update_order_id(buy_level.level_id, "order_123")

        # Simulate fill
        opposite = strategy.on_order_filled(
            order_id="order_123",
            fill_price=buy_level.price,
            fee=Decimal("0.5")
        )

        assert strategy.metrics.buy_trades == 1
        assert strategy.metrics.total_fees == Decimal("0.5")
        # Should create opposite (sell) order
        assert opposite is None or opposite.side == OrderSide.SELL

    def test_on_order_filled_sell(self, strategy):
        """Test handling sell order fill."""
        levels = strategy.initialize(Decimal("50000"))

        # First do a buy to have position
        buy_level = next(l for l in levels if l.side == OrderSide.BUY)
        strategy.order_manager.update_order_id(buy_level.level_id, "order_buy")
        strategy.on_order_filled("order_buy", buy_level.price)

        # Now do a sell
        sell_level = next(l for l in levels if l.side == OrderSide.SELL)
        strategy.order_manager.update_order_id(sell_level.level_id, "order_sell")
        strategy.on_order_filled("order_sell", sell_level.price)

        assert strategy.metrics.sell_trades == 1

    def test_get_snapshot(self, strategy):
        """Test getting grid snapshot."""
        strategy.initialize(Decimal("50000"))
        snapshot = strategy.get_snapshot()
        assert snapshot.current_price == Decimal("50000")
        assert snapshot.buy_levels > 0
        assert snapshot.sell_levels > 0

    def test_pause_grid(self, strategy):
        """Test pausing grid."""
        strategy.initialize(Decimal("50000"))

        # Mark some orders as open
        for level in strategy.order_manager.get_all_levels()[:3]:
            strategy.order_manager.update_order_id(level.level_id, f"order_{level.level_id}")

        cancelled = strategy.pause()
        assert len(cancelled) > 0
        assert strategy.state == GridState.PAUSED

    def test_resume_grid(self, strategy):
        """Test resuming grid."""
        strategy.initialize(Decimal("50000"))
        strategy.pause()
        to_place = strategy.resume()
        assert strategy.state == GridState.ACTIVE

    def test_close_grid(self, strategy):
        """Test closing grid."""
        strategy.initialize(Decimal("50000"))
        result = strategy.close()
        assert strategy.state == GridState.CLOSED
        assert "realized_pnl" in result
        assert "total_trades" in result

    def test_get_grid_info(self, strategy):
        """Test getting complete grid info."""
        strategy.initialize(Decimal("50000"))
        info = strategy.get_grid_info()
        assert info["strategy_id"] == "test_strategy"
        assert info["state"] == "active"
        assert "config" in info
        assert "metrics" in info
        assert "levels" in info

    def test_metrics_update(self, strategy):
        """Test metrics are updated after trades."""
        levels = strategy.initialize(Decimal("50000"))

        # Simulate multiple fills
        for i, level in enumerate(levels[:3]):
            strategy.order_manager.update_order_id(level.level_id, f"order_{i}")
            strategy.on_order_filled(f"order_{i}", level.price, Decimal("0.1"))

        assert strategy.metrics.total_trades == 3
        assert strategy.metrics.total_volume > Decimal("0")


class TestGridStrategyGeometric:
    """Tests for geometric grid strategy."""

    @pytest.fixture
    def config(self):
        """Create geometric config."""
        return GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.GEOMETRIC,
            upper_price=Decimal("60000"),
            lower_price=Decimal("40000"),
            num_grids=10,
            total_investment=Decimal("10000")
        )

    def test_geometric_initialization(self, config):
        """Test geometric grid initialization."""
        strategy = GridStrategy(config)
        levels = strategy.initialize(Decimal("50000"))
        assert len(levels) > 0

        # Check geometric spacing
        prices = sorted([l.price for l in levels])
        if len(prices) >= 3:
            ratio1 = prices[1] / prices[0]
            ratio2 = prices[2] / prices[1]
            assert abs(ratio1 - ratio2) < Decimal("0.1")


class TestGridStrategyCustom:
    """Tests for custom grid strategy."""

    @pytest.fixture
    def config(self):
        """Create custom config."""
        return GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.CUSTOM,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=5,
            total_investment=Decimal("10000"),
            custom_levels=[
                Decimal("45000"),
                Decimal("47500"),
                Decimal("50000"),
                Decimal("52500"),
                Decimal("55000")
            ]
        )

    def test_custom_initialization(self, config):
        """Test custom grid initialization."""
        strategy = GridStrategy(config)
        levels = strategy.initialize(Decimal("50000"))
        assert len(levels) > 0


class TestGridStrategyWithTrailingStop:
    """Tests for grid with trailing stop."""

    @pytest.fixture
    def config(self):
        """Create config with trailing stop."""
        return GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000"),
            trailing_stop_pct=Decimal("5")  # 5% trailing stop
        )

    def test_trailing_stop_updates(self, config):
        """Test trailing stop price updates."""
        strategy = GridStrategy(config)
        strategy.initialize(Decimal("50000"))

        # Price moves up
        strategy.on_price_update(Decimal("52000"))
        strategy.on_price_update(Decimal("54000"))

        # Trailing stop should be set
        assert strategy._trailing_stop_price is not None
        # Should be 5% below highest price (54000 * 0.95 = 51300)
        assert strategy._trailing_stop_price < Decimal("52000")

    def test_trailing_stop_triggers(self, config):
        """Test trailing stop trigger."""
        strategy = GridStrategy(config)
        strategy.initialize(Decimal("50000"))

        # Price moves up then down
        strategy.on_price_update(Decimal("55000"))
        result = strategy.on_price_update(Decimal("51000"))  # 7.27% drop

        assert result["action"] == "trailing_stop"
        assert result["close_position"]


# =============================================================================
# Grid Strategy Manager Tests
# =============================================================================

class TestGridStrategyManager:
    """Tests for GridStrategyManager."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return GridStrategyManager()

    @pytest.fixture
    def config(self):
        """Create config."""
        return GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000")
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
        config2 = GridConfig(
            symbol="ETH-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("4000"),
            lower_price=Decimal("3000"),
            num_grids=10,
            total_investment=Decimal("5000")
        )
        manager.create_strategy(config2, "test_2")

        strategies = manager.list_strategies()
        assert len(strategies) == 2
        assert any(s["strategy_id"] == "test_1" for s in strategies)
        assert any(s["strategy_id"] == "test_2" for s in strategies)

    def test_get_active_strategies(self, manager, config):
        """Test getting active strategies."""
        strategy1 = manager.create_strategy(config, "test_1")
        strategy1.initialize(Decimal("50000"))

        strategy2 = manager.create_strategy(
            GridConfig(
                symbol="ETH-USD-PERP",
                grid_type=GridType.ARITHMETIC,
                upper_price=Decimal("4000"),
                lower_price=Decimal("3000"),
                num_grids=10,
                total_investment=Decimal("5000")
            ),
            "test_2"
        )
        # test_2 not initialized, so not active

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

    def test_get_total_metrics(self, manager, config):
        """Test aggregated metrics."""
        strategy1 = manager.create_strategy(config, "test_1")
        strategy1.initialize(Decimal("50000"))

        strategy2 = manager.create_strategy(
            GridConfig(
                symbol="ETH-USD-PERP",
                grid_type=GridType.ARITHMETIC,
                upper_price=Decimal("4000"),
                lower_price=Decimal("3000"),
                num_grids=10,
                total_investment=Decimal("5000")
            ),
            "test_2"
        )
        strategy2.initialize(Decimal("3500"))

        metrics = manager.get_total_metrics()
        assert metrics["total_strategies"] == 2
        assert metrics["active_strategies"] == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestGridTradingIntegration:
    """Integration tests for grid trading."""

    def test_full_grid_lifecycle(self):
        """Test complete grid trading lifecycle."""
        # Create config
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=11,
            total_investment=Decimal("10000"),
            stop_loss_price=Decimal("43000"),
            take_profit_price=Decimal("60000")
        )

        # Create and initialize
        strategy = GridStrategy(config)
        levels = strategy.initialize(Decimal("50000"))
        assert len(levels) == 11

        # Simulate order placement
        for level in levels:
            strategy.order_manager.update_order_id(
                level.level_id,
                f"order_{level.level_id}"
            )

        # Simulate some fills
        buy_levels = [l for l in levels if l.side == OrderSide.BUY]
        sell_levels = [l for l in levels if l.side == OrderSide.SELL]

        # Fill a buy order
        if buy_levels:
            strategy.on_order_filled(
                f"order_{buy_levels[0].level_id}",
                buy_levels[0].price,
                Decimal("0.1")
            )

        # Fill a sell order
        if sell_levels:
            strategy.on_order_filled(
                f"order_{sell_levels[0].level_id}",
                sell_levels[0].price,
                Decimal("0.1")
            )

        # Check metrics
        assert strategy.metrics.total_trades == 2
        assert strategy.metrics.buy_trades == 1
        assert strategy.metrics.sell_trades == 1

        # Get snapshot
        snapshot = strategy.get_snapshot()
        assert snapshot.filled_orders == 2

        # Close strategy
        result = strategy.close()
        assert strategy.state == GridState.CLOSED
        assert result["total_trades"] == 2

    def test_grid_profit_calculation(self):
        """Test profit calculation after round trips."""
        config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("52000"),
            lower_price=Decimal("48000"),
            num_grids=5,
            total_investment=Decimal("10000")
        )

        strategy = GridStrategy(config)
        strategy.initialize(Decimal("50000"))

        # Simulate buy at 49000, sell at 51000 (round trip)
        levels = strategy.order_manager.get_all_levels()

        # Find buy level at 49000
        buy_level = next((l for l in levels if l.price == Decimal("49000")), None)
        if buy_level:
            strategy.order_manager.update_order_id(buy_level.level_id, "order_buy")
            strategy.on_order_filled("order_buy", Decimal("49000"))

            # Now price goes up, sell at 51000
            sell_level = next((l for l in levels if l.price == Decimal("51000")), None)
            if sell_level:
                strategy.order_manager.update_order_id(sell_level.level_id, "order_sell")
                strategy.on_order_filled("order_sell", Decimal("51000"))

                # Should have realized profit
                assert strategy.metrics.realized_pnl > Decimal("0")

    def test_multiple_strategies_manager(self):
        """Test managing multiple strategies."""
        manager = GridStrategyManager()

        # Create BTC strategy
        btc_config = GridConfig(
            symbol="BTC-USD-PERP",
            grid_type=GridType.ARITHMETIC,
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            num_grids=10,
            total_investment=Decimal("10000")
        )
        btc_strategy = manager.create_strategy(btc_config, "btc_grid")
        btc_strategy.initialize(Decimal("50000"))

        # Create ETH strategy
        eth_config = GridConfig(
            symbol="ETH-USD-PERP",
            grid_type=GridType.GEOMETRIC,
            upper_price=Decimal("4000"),
            lower_price=Decimal("3000"),
            num_grids=8,
            total_investment=Decimal("5000")
        )
        eth_strategy = manager.create_strategy(eth_config, "eth_grid")
        eth_strategy.initialize(Decimal("3500"))

        # Check manager state
        metrics = manager.get_total_metrics()
        assert metrics["total_strategies"] == 2
        assert metrics["active_strategies"] == 2

        # Close one strategy
        manager.remove_strategy("btc_grid")
        metrics = manager.get_total_metrics()
        assert metrics["total_strategies"] == 1
