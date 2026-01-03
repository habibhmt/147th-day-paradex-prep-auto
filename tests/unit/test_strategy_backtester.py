"""
Tests for Strategy Backtester Module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.analytics.strategy_backtester import (
    OrderType, OrderSide, OrderStatus, PositionSide, BacktestMode,
    SlippageModel, FillModel, OHLCV, Trade, Order, Position,
    EquityPoint, BacktestMetrics, BacktestConfig, BacktestResult,
    SlippageCalculator, CommissionCalculator, PositionManager,
    OrderManager, BaseStrategy, BacktestContext, StrategyBacktester,
    WalkForwardOptimizer, MonteCarloSimulator, PerformanceAnalyzer,
    get_backtester, set_backtester
)


# ============== Fixtures ==============

@pytest.fixture
def sample_bar():
    """Create sample OHLCV bar."""
    return OHLCV(
        timestamp=datetime(2024, 1, 1, 10, 0),
        open=Decimal("50000"),
        high=Decimal("50500"),
        low=Decimal("49500"),
        close=Decimal("50200"),
        volume=Decimal("1000")
    )


@pytest.fixture
def sample_bars():
    """Create multiple sample bars."""
    base = datetime(2024, 1, 1, 10, 0)
    bars = []
    price = Decimal("50000")

    for i in range(100):
        change = Decimal(str((i % 5 - 2) * 50))
        bars.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=price,
            high=price + Decimal("200"),
            low=price - Decimal("200"),
            close=price + change,
            volume=Decimal("1000")
        ))
        price = price + change

    return bars


@pytest.fixture
def backtest_config():
    """Create backtest configuration."""
    return BacktestConfig(
        initial_capital=Decimal("100000"),
        commission_rate=Decimal("0.001"),
        slippage_model=SlippageModel.PERCENTAGE,
        slippage_rate=Decimal("0.0005")
    )


@pytest.fixture
def backtester(backtest_config, sample_bars):
    """Create backtester with data."""
    bt = StrategyBacktester(backtest_config)
    bt.add_data("BTC-USD", sample_bars)
    return bt


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_order_type_values(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"

    def test_order_side_values(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status_values(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_position_side_values(self):
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"
        assert PositionSide.FLAT.value == "flat"

    def test_backtest_mode_values(self):
        assert BacktestMode.VECTORIZED.value == "vectorized"
        assert BacktestMode.EVENT_DRIVEN.value == "event_driven"

    def test_slippage_model_values(self):
        assert SlippageModel.NONE.value == "none"
        assert SlippageModel.FIXED.value == "fixed"
        assert SlippageModel.PERCENTAGE.value == "percentage"

    def test_fill_model_values(self):
        assert FillModel.IMMEDIATE.value == "immediate"
        assert FillModel.NEXT_BAR.value == "next_bar"


# ============== Data Class Tests ==============

class TestOHLCV:
    """Test OHLCV dataclass."""

    def test_creation(self, sample_bar):
        assert sample_bar.open == Decimal("50000")
        assert sample_bar.high == Decimal("50500")
        assert sample_bar.low == Decimal("49500")
        assert sample_bar.close == Decimal("50200")

    def test_to_dict(self, sample_bar):
        result = sample_bar.to_dict()
        assert "timestamp" in result
        assert result["open"] == "50000"
        assert result["close"] == "50200"


class TestTrade:
    """Test Trade dataclass."""

    def test_creation(self):
        trade = Trade(
            id="TRD-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            commission=Decimal("50"),
            slippage=Decimal("25")
        )
        assert trade.quantity == Decimal("1")
        assert trade.price == Decimal("50000")

    def test_to_dict(self):
        trade = Trade(
            id="TRD-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            commission=Decimal("50"),
            slippage=Decimal("25"),
            pnl=Decimal("1000")
        )
        result = trade.to_dict()
        assert result["id"] == "TRD-001"
        assert result["side"] == "buy"
        assert result["pnl"] == "1000"


class TestOrder:
    """Test Order dataclass."""

    def test_creation(self):
        order = Order(
            id="ORD-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1")
        )
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")

    def test_limit_order(self):
        order = Order(
            id="ORD-002",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("49000")
        )
        assert order.price == Decimal("49000")

    def test_to_dict(self):
        order = Order(
            id="ORD-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("2"),
            stop_price=Decimal("48000")
        )
        result = order.to_dict()
        assert result["order_type"] == "stop"
        assert result["stop_price"] == "48000"


class TestPosition:
    """Test Position dataclass."""

    def test_creation(self):
        position = Position(
            symbol="BTC-USD",
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            entry_time=datetime.now()
        )
        assert position.unrealized_pnl == Decimal("0")
        assert position.realized_pnl == Decimal("0")

    def test_to_dict(self):
        position = Position(
            symbol="BTC-USD",
            side=PositionSide.SHORT,
            quantity=Decimal("2"),
            entry_price=Decimal("51000"),
            entry_time=datetime.now(),
            unrealized_pnl=Decimal("2000")
        )
        result = position.to_dict()
        assert result["side"] == "short"
        assert result["unrealized_pnl"] == "2000"


class TestEquityPoint:
    """Test EquityPoint dataclass."""

    def test_creation(self):
        ep = EquityPoint(
            timestamp=datetime.now(),
            equity=Decimal("110000"),
            cash=Decimal("60000"),
            positions_value=Decimal("50000"),
            drawdown=Decimal("5000"),
            drawdown_pct=Decimal("0.05")
        )
        assert ep.equity == Decimal("110000")

    def test_to_dict(self):
        ep = EquityPoint(
            timestamp=datetime.now(),
            equity=Decimal("110000"),
            cash=Decimal("60000"),
            positions_value=Decimal("50000"),
            drawdown=Decimal("0"),
            drawdown_pct=Decimal("0")
        )
        result = ep.to_dict()
        assert "equity" in result
        assert result["equity"] == "110000"


class TestBacktestMetrics:
    """Test BacktestMetrics dataclass."""

    def test_creation(self):
        metrics = BacktestMetrics(
            total_return=Decimal("10000"),
            total_return_pct=Decimal("10"),
            annualized_return=Decimal("25"),
            volatility=Decimal("15"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("1.2"),
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=Decimal("5"),
            max_drawdown_duration=30,
            win_rate=Decimal("55"),
            profit_factor=Decimal("1.8"),
            avg_win=Decimal("500"),
            avg_loss=Decimal("300"),
            largest_win=Decimal("2000"),
            largest_loss=Decimal("-1000"),
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            avg_trade_duration=24.0,
            total_commission=Decimal("1000"),
            total_slippage=Decimal("500"),
            exposure_time=Decimal("80"),
            recovery_factor=Decimal("2.0"),
            expectancy=Decimal("100"),
            sqn=Decimal("2.5")
        )
        assert metrics.sharpe_ratio == Decimal("1.5")
        assert metrics.total_trades == 100

    def test_to_dict(self):
        metrics = BacktestMetrics(
            total_return=Decimal("10000"),
            total_return_pct=Decimal("10"),
            annualized_return=Decimal("25"),
            volatility=Decimal("15"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("1.2"),
            max_drawdown=Decimal("5000"),
            max_drawdown_pct=Decimal("5"),
            max_drawdown_duration=30,
            win_rate=Decimal("55"),
            profit_factor=Decimal("1.8"),
            avg_win=Decimal("500"),
            avg_loss=Decimal("300"),
            largest_win=Decimal("2000"),
            largest_loss=Decimal("-1000"),
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            avg_trade_duration=24.0,
            total_commission=Decimal("1000"),
            total_slippage=Decimal("500"),
            exposure_time=Decimal("80"),
            recovery_factor=Decimal("2.0"),
            expectancy=Decimal("100"),
            sqn=Decimal("2.5")
        )
        result = metrics.to_dict()
        assert "sharpe_ratio" in result
        assert result["total_trades"] == 100


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_defaults(self):
        config = BacktestConfig()
        assert config.initial_capital == Decimal("100000")
        assert config.commission_rate == Decimal("0.001")
        assert config.leverage == Decimal("1")

    def test_custom_config(self):
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            leverage=Decimal("2"),
            allow_shorting=False
        )
        assert config.initial_capital == Decimal("50000")
        assert config.leverage == Decimal("2")
        assert config.allow_shorting is False

    def test_to_dict(self):
        config = BacktestConfig()
        result = config.to_dict()
        assert "initial_capital" in result
        assert result["allow_shorting"] is True


# ============== Slippage Calculator Tests ==============

class TestSlippageCalculator:
    """Test SlippageCalculator."""

    def test_no_slippage(self):
        calc = SlippageCalculator(SlippageModel.NONE, Decimal("0"))
        slip = calc.calculate(Decimal("50000"), Decimal("1"), OrderSide.BUY)
        assert slip == Decimal("0")

    def test_fixed_slippage(self):
        calc = SlippageCalculator(SlippageModel.FIXED, Decimal("10"))
        slip = calc.calculate(Decimal("50000"), Decimal("1"), OrderSide.BUY)
        assert slip == Decimal("10")

    def test_percentage_slippage(self):
        calc = SlippageCalculator(SlippageModel.PERCENTAGE, Decimal("0.001"))
        slip = calc.calculate(Decimal("50000"), Decimal("1"), OrderSide.BUY)
        assert slip == Decimal("50")

    def test_volume_based_slippage(self):
        calc = SlippageCalculator(SlippageModel.VOLUME_BASED, Decimal("0.01"))
        slip = calc.calculate(
            Decimal("50000"), Decimal("100"), OrderSide.BUY,
            volume=Decimal("10000")
        )
        assert slip > Decimal("0")

    def test_spread_based_slippage(self):
        calc = SlippageCalculator(SlippageModel.SPREAD_BASED, Decimal("0.001"))
        slip = calc.calculate(
            Decimal("50000"), Decimal("1"), OrderSide.BUY,
            spread=Decimal("20")
        )
        assert slip == Decimal("10")

    def test_apply_buy(self):
        calc = SlippageCalculator(SlippageModel.PERCENTAGE, Decimal("0.001"))
        price = calc.apply(Decimal("50000"), Decimal("1"), OrderSide.BUY)
        assert price == Decimal("50050")

    def test_apply_sell(self):
        calc = SlippageCalculator(SlippageModel.PERCENTAGE, Decimal("0.001"))
        price = calc.apply(Decimal("50000"), Decimal("1"), OrderSide.SELL)
        assert price == Decimal("49950")


# ============== Commission Calculator Tests ==============

class TestCommissionCalculator:
    """Test CommissionCalculator."""

    def test_percentage_commission(self):
        calc = CommissionCalculator(rate=Decimal("0.001"))
        comm = calc.calculate(Decimal("50000"), Decimal("1"))
        assert comm == Decimal("50")

    def test_per_share_commission(self):
        calc = CommissionCalculator(per_share=Decimal("0.005"))
        comm = calc.calculate(Decimal("50000"), Decimal("100"))
        assert comm == Decimal("0.5")

    def test_min_commission(self):
        calc = CommissionCalculator(
            rate=Decimal("0.001"),
            min_commission=Decimal("5")
        )
        comm = calc.calculate(Decimal("100"), Decimal("1"))
        assert comm == Decimal("5")

    def test_max_commission(self):
        calc = CommissionCalculator(
            rate=Decimal("0.001"),
            max_commission=Decimal("100")
        )
        comm = calc.calculate(Decimal("500000"), Decimal("1"))
        assert comm == Decimal("100")


# ============== Position Manager Tests ==============

class TestPositionManager:
    """Test PositionManager."""

    def test_open_long_position(self):
        pm = PositionManager()
        position = pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("50000"), datetime.now(), Decimal("50")
        )
        assert position.side == PositionSide.LONG
        assert position.quantity == Decimal("1")

    def test_open_short_position(self):
        pm = PositionManager()
        position = pm.open_position(
            "BTC-USD", PositionSide.SHORT, Decimal("2"),
            Decimal("51000"), datetime.now(), Decimal("51")
        )
        assert position.side == PositionSide.SHORT
        assert position.quantity == Decimal("2")

    def test_add_to_position(self):
        pm = PositionManager()
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("50000"), datetime.now(), Decimal("50")
        )
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("52000"), datetime.now(), Decimal("52")
        )
        position = pm.get_position("BTC-USD")
        assert position.quantity == Decimal("2")
        assert position.entry_price == Decimal("51000")  # Average

    def test_close_position(self):
        pm = PositionManager()
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("50000"), datetime.now(), Decimal("50")
        )
        pnl = pm.close_position(
            "BTC-USD", Decimal("1"), Decimal("51000"),
            datetime.now(), Decimal("51")
        )
        assert pnl == Decimal("949")  # 1000 - 51 commission
        assert pm.get_position("BTC-USD") is None

    def test_partial_close(self):
        pm = PositionManager()
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("2"),
            Decimal("50000"), datetime.now(), Decimal("100")
        )
        pnl = pm.close_position(
            "BTC-USD", Decimal("1"), Decimal("51000"),
            datetime.now(), Decimal("51")
        )
        position = pm.get_position("BTC-USD")
        assert position.quantity == Decimal("1")

    def test_update_unrealized(self):
        pm = PositionManager()
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("50000"), datetime.now(), Decimal("50")
        )
        pm.update_unrealized("BTC-USD", Decimal("52000"))
        position = pm.get_position("BTC-USD")
        assert position.unrealized_pnl == Decimal("2000")

    def test_get_total_value(self):
        pm = PositionManager()
        pm.open_position(
            "BTC-USD", PositionSide.LONG, Decimal("1"),
            Decimal("50000"), datetime.now(), Decimal("50")
        )
        pm.open_position(
            "ETH-USD", PositionSide.LONG, Decimal("10"),
            Decimal("3000"), datetime.now(), Decimal("30")
        )
        total = pm.get_total_value({
            "BTC-USD": Decimal("51000"),
            "ETH-USD": Decimal("3100")
        })
        assert total == Decimal("82000")


# ============== Order Manager Tests ==============

class TestOrderManager:
    """Test OrderManager."""

    @pytest.fixture
    def order_manager(self):
        slippage = SlippageCalculator(SlippageModel.PERCENTAGE, Decimal("0.0005"))
        commission = CommissionCalculator(rate=Decimal("0.001"))
        return OrderManager(slippage, commission, FillModel.NEXT_BAR)

    def test_create_market_order(self, order_manager):
        order = order_manager.create_order(
            datetime.now(), "BTC-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("1")
        )
        assert order.id == "ORD-000001"
        assert order.status == OrderStatus.PENDING

    def test_create_limit_order(self, order_manager):
        order = order_manager.create_order(
            datetime.now(), "BTC-USD", OrderSide.BUY,
            OrderType.LIMIT, Decimal("1"),
            price=Decimal("49000")
        )
        assert order.price == Decimal("49000")

    def test_cancel_order(self, order_manager):
        order = order_manager.create_order(
            datetime.now(), "BTC-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("1")
        )
        assert order_manager.cancel_order(order.id)
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent(self, order_manager):
        assert order_manager.cancel_order("INVALID") is False

    def test_cancel_all(self, order_manager):
        order_manager.create_order(
            datetime.now(), "BTC-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("1")
        )
        order_manager.create_order(
            datetime.now(), "ETH-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("10")
        )
        order_manager.cancel_all()
        assert len(order_manager.pending_orders) == 0

    def test_cancel_by_symbol(self, order_manager):
        order_manager.create_order(
            datetime.now(), "BTC-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("1")
        )
        order_manager.create_order(
            datetime.now(), "ETH-USD", OrderSide.BUY,
            OrderType.MARKET, Decimal("10")
        )
        order_manager.cancel_all("BTC-USD")
        assert len(order_manager.pending_orders) == 1


# ============== Base Strategy Tests ==============

class TestBaseStrategy:
    """Test BaseStrategy."""

    def test_creation(self):
        strategy = BaseStrategy("TestStrategy")
        assert strategy.name == "TestStrategy"
        assert strategy.parameters == {}

    def test_lifecycle_methods(self):
        strategy = BaseStrategy()
        context = Mock()
        bar = Mock()
        trade = Mock()
        order = Mock()

        # Should not raise
        strategy.initialize(context)
        strategy.on_bar(context, bar)
        strategy.on_trade(context, trade)
        strategy.on_order(context, order)
        strategy.finalize(context)


# ============== Backtest Context Tests ==============

class TestBacktestContext:
    """Test BacktestContext."""

    @pytest.fixture
    def context(self):
        order_manager = Mock()
        position_manager = Mock()
        return BacktestContext(
            timestamp=datetime.now(),
            cash=Decimal("100000"),
            equity=Decimal("100000"),
            positions={},
            current_prices={"BTC-USD": Decimal("50000")},
            bars_history={"BTC-USD": []},
            order_manager=order_manager,
            position_manager=position_manager
        )

    def test_buy(self, context):
        context.order_manager.create_order = Mock(return_value=Mock())
        context.buy("BTC-USD", Decimal("1"))
        context.order_manager.create_order.assert_called_once()

    def test_sell(self, context):
        context.order_manager.create_order = Mock(return_value=Mock())
        context.sell("BTC-USD", Decimal("1"))
        context.order_manager.create_order.assert_called_once()

    def test_close_position_long(self, context):
        context.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            entry_time=datetime.now()
        )
        context.order_manager.create_order = Mock(return_value=Mock())
        context.close_position("BTC-USD")
        context.order_manager.create_order.assert_called_once()

    def test_close_no_position(self, context):
        result = context.close_position("ETH-USD")
        assert result is None

    def test_get_position_value(self, context):
        context.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            side=PositionSide.LONG,
            quantity=Decimal("2"),
            entry_price=Decimal("50000"),
            entry_time=datetime.now()
        )
        value = context.get_position_value("BTC-USD")
        assert value == Decimal("100000")


# ============== Strategy Backtester Tests ==============

class TestStrategyBacktester:
    """Test StrategyBacktester."""

    def test_init(self, backtest_config):
        bt = StrategyBacktester(backtest_config)
        assert bt.config == backtest_config
        assert bt.data == {}
        assert bt.strategies == []

    def test_add_data(self, backtester, sample_bars):
        assert "BTC-USD" in backtester.data
        assert len(backtester.data["BTC-USD"]) == len(sample_bars)

    def test_add_strategy(self, backtester):
        strategy = BaseStrategy("Test")
        backtester.add_strategy(strategy)
        assert len(backtester.strategies) == 1

    def test_register_callback(self, backtester):
        callback = Mock()
        backtester.register_callback("on_bar", callback)
        assert callback in backtester.callbacks["on_bar"]

    def test_run_no_data(self, backtest_config):
        bt = StrategyBacktester(backtest_config)
        bt.add_strategy(BaseStrategy())
        with pytest.raises(ValueError, match="No data"):
            bt.run()

    def test_run_no_strategy(self, backtester):
        with pytest.raises(ValueError, match="No strategies"):
            backtester.run()

    def test_run_basic(self, backtester):
        backtester.add_strategy(BaseStrategy())
        result = backtester.run()
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_run_with_trades(self, backtester):
        class SimpleStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Simple")
                self.bought = False

            def on_bar(self, context, bar):
                if not self.bought:
                    context.buy("BTC-USD", Decimal("0.1"))
                    self.bought = True

        backtester.add_strategy(SimpleStrategy())
        result = backtester.run()
        assert len(result.orders) > 0

    def test_metrics_calculation(self, backtester):
        backtester.add_strategy(BaseStrategy())
        result = backtester.run()
        assert result.metrics.total_trades >= 0
        assert result.metrics.max_drawdown >= Decimal("0")


# ============== Walk Forward Optimizer Tests ==============

class TestWalkForwardOptimizer:
    """Test WalkForwardOptimizer."""

    @pytest.fixture
    def optimizer(self, backtester):
        backtester.add_strategy(BaseStrategy())
        return WalkForwardOptimizer(backtester, in_sample_pct=0.7, num_folds=2)

    def test_init(self, optimizer):
        assert optimizer.in_sample_pct == 0.7
        assert optimizer.num_folds == 2

    def test_generate_combinations(self, optimizer):
        param_grid = {
            "param1": [1, 2],
            "param2": ["a", "b"]
        }
        combos = optimizer._generate_combinations(param_grid)
        assert len(combos) == 4

    def test_empty_param_grid(self, optimizer):
        combos = optimizer._generate_combinations({})
        assert len(combos) == 1
        assert combos[0] == {}


# ============== Monte Carlo Simulator Tests ==============

class TestMonteCarloSimulator:
    """Test MonteCarloSimulator."""

    @pytest.fixture
    def backtest_result(self, backtest_config, sample_bars):
        bt = StrategyBacktester(backtest_config)
        bt.add_data("BTC-USD", sample_bars)
        bt.add_strategy(BaseStrategy())
        return bt.run()

    def test_init(self, backtest_result):
        mc = MonteCarloSimulator(backtest_result, num_simulations=100)
        assert mc.num_simulations == 100

    def test_trade_shuffle_empty(self, backtest_result):
        mc = MonteCarloSimulator(backtest_result, num_simulations=10)
        result = mc.run_trade_shuffle()
        # May be empty if no trades
        assert isinstance(result, dict)

    def test_bootstrap_insufficient_data(self, backtest_config):
        # Create minimal data
        bars = [
            OHLCV(datetime(2024, 1, 1), Decimal("50000"), Decimal("50100"),
                  Decimal("49900"), Decimal("50050"), Decimal("100"))
        ]
        bt = StrategyBacktester(backtest_config)
        bt.add_data("BTC-USD", bars)
        bt.add_strategy(BaseStrategy())
        result = bt.run()

        mc = MonteCarloSimulator(result, num_simulations=10)
        bootstrap_result = mc.run_bootstrap(block_size=20)
        assert bootstrap_result == {}


# ============== Performance Analyzer Tests ==============

class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer."""

    @pytest.fixture
    def analyzer(self, backtest_config, sample_bars):
        bt = StrategyBacktester(backtest_config)
        bt.add_data("BTC-USD", sample_bars)
        bt.add_strategy(BaseStrategy())
        result = bt.run()
        return PerformanceAnalyzer(result)

    def test_monthly_returns(self, analyzer):
        monthly = analyzer.monthly_returns()
        assert isinstance(monthly, dict)

    def test_yearly_returns(self, analyzer):
        yearly = analyzer.yearly_returns()
        assert isinstance(yearly, dict)

    def test_drawdown_analysis(self, analyzer):
        drawdowns = analyzer.drawdown_analysis()
        assert isinstance(drawdowns, list)

    def test_trade_analysis(self, analyzer):
        analysis = analyzer.trade_analysis()
        assert isinstance(analysis, dict)
        # May be empty if no trades with BaseStrategy
        if analysis:
            assert "total_trades" in analysis

    def test_risk_metrics(self, analyzer):
        risk = analyzer.risk_metrics()
        assert isinstance(risk, dict)


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global backtester instance."""

    def test_get_backtester(self):
        bt = get_backtester()
        assert isinstance(bt, StrategyBacktester)

    def test_set_backtester(self):
        custom = StrategyBacktester(BacktestConfig(
            initial_capital=Decimal("50000")
        ))
        set_backtester(custom)
        bt = get_backtester()
        assert bt.config.initial_capital == Decimal("50000")


# ============== Integration Tests ==============

class TestBacktestIntegration:
    """Integration tests for full backtest."""

    def test_full_backtest_flow(self, sample_bars):
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005")
        )

        bt = StrategyBacktester(config)
        bt.add_data("BTC-USD", sample_bars)

        class TrendStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Trend")
                self.position = None

            def on_bar(self, context, bar):
                if len(context.bars_history.get("BTC-USD", [])) < 5:
                    return

                # Simple momentum
                recent = context.bars_history["BTC-USD"][-5:]
                if recent[-1].close > recent[0].close:
                    if "BTC-USD" not in context.positions:
                        context.buy("BTC-USD", Decimal("0.5"))
                else:
                    if "BTC-USD" in context.positions:
                        context.close_position("BTC-USD")

        bt.add_strategy(TrendStrategy())
        result = bt.run()

        assert result.start_date < result.end_date
        assert len(result.equity_curve) > 0
        assert result.config == config

    def test_multi_symbol_backtest(self):
        config = BacktestConfig()
        bt = StrategyBacktester(config)

        base = datetime(2024, 1, 1)
        btc_bars = [
            OHLCV(base + timedelta(hours=i), Decimal("50000"),
                  Decimal("50100"), Decimal("49900"), Decimal("50050"),
                  Decimal("1000"))
            for i in range(50)
        ]
        eth_bars = [
            OHLCV(base + timedelta(hours=i), Decimal("3000"),
                  Decimal("3010"), Decimal("2990"), Decimal("3005"),
                  Decimal("5000"))
            for i in range(50)
        ]

        bt.add_data("BTC-USD", btc_bars)
        bt.add_data("ETH-USD", eth_bars)
        bt.add_strategy(BaseStrategy())

        result = bt.run()
        assert "BTC-USD" in result.symbols
        assert "ETH-USD" in result.symbols


# ============== Edge Cases ==============

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_equity_curve_metrics(self):
        bt = StrategyBacktester()
        metrics = bt._empty_metrics()
        assert metrics.total_return == Decimal("0")
        assert metrics.total_trades == 0

    def test_single_bar_backtest(self, backtest_config):
        bt = StrategyBacktester(backtest_config)
        bars = [
            OHLCV(datetime(2024, 1, 1), Decimal("50000"),
                  Decimal("50100"), Decimal("49900"), Decimal("50050"),
                  Decimal("1000"))
        ]
        bt.add_data("BTC-USD", bars)
        bt.add_strategy(BaseStrategy())
        result = bt.run()
        assert len(result.equity_curve) == 1

    def test_zero_volume_bar(self, backtest_config):
        bt = StrategyBacktester(backtest_config)
        bars = [
            OHLCV(datetime(2024, 1, 1), Decimal("50000"),
                  Decimal("50100"), Decimal("49900"), Decimal("50050"),
                  Decimal("0"))
        ]
        bt.add_data("BTC-USD", bars)
        bt.add_strategy(BaseStrategy())
        result = bt.run()
        assert result is not None


# ============== Result Serialization Tests ==============

class TestResultSerialization:
    """Test result serialization."""

    def test_backtest_result_to_dict(self, backtester):
        backtester.add_strategy(BaseStrategy())
        result = backtester.run()
        data = result.to_dict()

        assert "metrics" in data
        assert "equity_curve" in data
        assert "trades" in data
        assert "config" in data
        assert "start_date" in data
        assert "symbols" in data
