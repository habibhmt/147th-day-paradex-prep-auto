"""Tests for Backtesting Engine Module"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.backtesting.engine import (
    OrderType, OrderSide, OrderStatus, PositionSide,
    OHLCV, BacktestOrder, Trade, Position, BacktestConfig,
    BacktestMetrics, EquityCurve, OrderManager, PositionManager,
    MetricsCalculator, BacktestEngine,
)


class TestEnums:
    def test_order_type(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"

    def test_order_side(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"

    def test_position_side(self):
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"


class TestOHLCV:
    def test_creation(self):
        bar = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("103"),
            volume=Decimal("1000"),
        )
        assert bar.open == Decimal("100")
        assert bar.close == Decimal("103")

    def test_to_dict(self):
        bar = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("103"),
            volume=Decimal("1000"),
        )
        result = bar.to_dict()
        assert result["open"] == "100"


class TestBacktestConfig:
    def test_valid_config(self):
        config = BacktestConfig(initial_capital=Decimal("50000"))
        assert config.initial_capital == Decimal("50000")

    def test_invalid_initial_capital(self):
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=Decimal("0"))

    def test_invalid_commission_rate(self):
        with pytest.raises(ValueError):
            BacktestConfig(commission_rate=-0.1)

    def test_invalid_leverage(self):
        with pytest.raises(ValueError):
            BacktestConfig(leverage=0)


class TestOrderManager:
    def test_create_order(self):
        config = BacktestConfig()
        manager = OrderManager(config)
        order = manager.create_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        assert order.symbol == "ETH-USD"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING

    def test_fill_order(self):
        config = BacktestConfig(use_slippage=False, use_commission=False)
        manager = OrderManager(config)
        order = manager.create_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        filled = manager.fill_order(order.order_id, Decimal("2000"), Decimal("1"), datetime.now())
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_price == Decimal("2000")

    def test_fill_with_slippage(self):
        config = BacktestConfig(use_slippage=True, slippage_rate=0.01)
        manager = OrderManager(config)
        order = manager.create_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        filled = manager.fill_order(order.order_id, Decimal("2000"), Decimal("1"), datetime.now())
        assert filled.filled_price > Decimal("2000")

    def test_fill_with_commission(self):
        config = BacktestConfig(use_commission=True, commission_rate=0.001)
        manager = OrderManager(config)
        order = manager.create_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        filled = manager.fill_order(order.order_id, Decimal("2000"), Decimal("1"), datetime.now())
        assert filled.commission > 0

    def test_cancel_order(self):
        config = BacktestConfig()
        manager = OrderManager(config)
        order = manager.create_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        result = manager.cancel_order(order.order_id)
        assert result
        assert order.status == OrderStatus.CANCELLED

    def test_get_pending_orders(self):
        config = BacktestConfig()
        manager = OrderManager(config)
        manager.create_order("ETH-USD", OrderSide.BUY, Decimal("1"))
        manager.create_order("BTC-USD", OrderSide.SELL, Decimal("0.1"))
        pending = manager.get_pending_orders()
        assert len(pending) == 2


class TestPositionManager:
    def test_new_long_position(self):
        config = BacktestConfig()
        manager = PositionManager(config)
        pnl, pos = manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("2000"))
        assert pos.side == PositionSide.LONG
        assert pos.quantity == Decimal("1")
        assert pnl == Decimal("0")

    def test_new_short_position(self):
        config = BacktestConfig(allow_shorting=True)
        manager = PositionManager(config)
        pnl, pos = manager.update_position("ETH-USD", OrderSide.SELL, Decimal("1"), Decimal("2000"))
        assert pos.side == PositionSide.SHORT
        assert pos.quantity == Decimal("1")

    def test_close_long_profit(self):
        config = BacktestConfig()
        manager = PositionManager(config)
        manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("2000"))
        pnl, pos = manager.update_position("ETH-USD", OrderSide.SELL, Decimal("1"), Decimal("2100"))
        assert pnl == Decimal("100")
        assert pos.side == PositionSide.FLAT

    def test_close_long_loss(self):
        config = BacktestConfig()
        manager = PositionManager(config)
        manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("2000"))
        pnl, pos = manager.update_position("ETH-USD", OrderSide.SELL, Decimal("1"), Decimal("1900"))
        assert pnl == Decimal("-100")

    def test_close_short_profit(self):
        config = BacktestConfig(allow_shorting=True)
        manager = PositionManager(config)
        manager.update_position("ETH-USD", OrderSide.SELL, Decimal("1"), Decimal("2000"))
        pnl, pos = manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("1900"))
        assert pnl == Decimal("100")

    def test_update_unrealized_pnl(self):
        config = BacktestConfig()
        manager = PositionManager(config)
        manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("2000"))
        manager.update_unrealized_pnl("ETH-USD", Decimal("2100"))
        pos = manager.get_position("ETH-USD")
        assert pos.unrealized_pnl == Decimal("100")

    def test_get_total_unrealized_pnl(self):
        config = BacktestConfig()
        manager = PositionManager(config)
        manager.update_position("ETH-USD", OrderSide.BUY, Decimal("1"), Decimal("2000"))
        manager.update_unrealized_pnl("ETH-USD", Decimal("2100"))
        total = manager.get_total_unrealized_pnl()
        assert total == Decimal("100")


class TestMetricsCalculator:
    def test_calculate_returns(self):
        equity_curve = [
            EquityCurve(datetime.now(), Decimal("10000"), 0, 0),
            EquityCurve(datetime.now(), Decimal("10500"), 0, 0),
            EquityCurve(datetime.now(), Decimal("10200"), 0, 0),
        ]
        returns = MetricsCalculator.calculate_returns(equity_curve)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.05, rel=0.01)

    def test_calculate_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_empty(self):
        sharpe = MetricsCalculator.calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_calculate_sortino_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        sortino = MetricsCalculator.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_calculate_max_drawdown(self):
        equity_curve = [
            EquityCurve(datetime.now(), Decimal("10000"), 0, 0),
            EquityCurve(datetime.now(), Decimal("11000"), 0, 0),
            EquityCurve(datetime.now(), Decimal("9500"), 0, 0),
            EquityCurve(datetime.now(), Decimal("10500"), 0, 0),
        ]
        max_dd, duration = MetricsCalculator.calculate_max_drawdown(equity_curve)
        assert max_dd > 0
        assert max_dd == pytest.approx(0.136, rel=0.01)

    def test_calculate_trade_metrics(self):
        trades = [
            Trade("t1", "o1", "ETH", OrderSide.BUY, Decimal("1"), Decimal("2000"), datetime.now(), Decimal("2"), pnl=Decimal("100")),
            Trade("t2", "o2", "ETH", OrderSide.SELL, Decimal("1"), Decimal("2100"), datetime.now(), Decimal("2"), pnl=Decimal("-50")),
        ]
        metrics = MetricsCalculator.calculate_trade_metrics(trades)
        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1

    def test_calculate_trade_metrics_empty(self):
        metrics = MetricsCalculator.calculate_trade_metrics([])
        assert metrics["total_trades"] == 0


class TestBacktestEngine:
    def test_creation(self):
        engine = BacktestEngine()
        assert engine.equity == Decimal("10000")

    def test_creation_with_config(self):
        config = BacktestConfig(initial_capital=Decimal("50000"))
        engine = BacktestEngine(config)
        assert engine.equity == Decimal("50000")

    def test_load_data(self):
        engine = BacktestEngine()
        bars = [
            OHLCV(datetime.now(), Decimal("100"), Decimal("105"), Decimal("98"), Decimal("103"), Decimal("1000")),
        ]
        engine.load_data("ETH-USD", bars)
        assert "ETH-USD" in engine.data

    def test_buy_order(self):
        engine = BacktestEngine()
        engine.current_bar = OHLCV(datetime.now(), Decimal("100"), Decimal("105"), Decimal("98"), Decimal("103"), Decimal("1000"))
        order = engine.buy("ETH-USD", Decimal("1"))
        assert order.side == OrderSide.BUY

    def test_sell_order(self):
        engine = BacktestEngine()
        engine.current_bar = OHLCV(datetime.now(), Decimal("100"), Decimal("105"), Decimal("98"), Decimal("103"), Decimal("1000"))
        order = engine.sell("ETH-USD", Decimal("1"))
        assert order.side == OrderSide.SELL

    def test_run_no_data(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError):
            engine.run()

    def test_run_simple(self):
        engine = BacktestEngine(BacktestConfig(use_commission=False, use_slippage=False))
        now = datetime.now()
        bars = [
            OHLCV(now, Decimal("100"), Decimal("105"), Decimal("98"), Decimal("103"), Decimal("1000")),
            OHLCV(now + timedelta(days=1), Decimal("103"), Decimal("110"), Decimal("102"), Decimal("108"), Decimal("1200")),
            OHLCV(now + timedelta(days=2), Decimal("108"), Decimal("112"), Decimal("105"), Decimal("110"), Decimal("1100")),
        ]
        engine.load_data("ETH-USD", bars)
        
        def strategy(eng, bar, idx):
            if idx == 0:
                eng.buy("ETH-USD", Decimal("10"))
        
        engine.set_strategy(strategy)
        metrics = engine.run()
        assert isinstance(metrics, BacktestMetrics)

    def test_get_position(self):
        engine = BacktestEngine()
        pos = engine.get_position("ETH-USD")
        assert pos is None

    def test_get_results(self):
        engine = BacktestEngine()
        results = engine.get_results()
        assert "equity" in results
        assert "config" in results


class TestBacktestIntegration:
    def test_simple_buy_and_hold(self):
        config = BacktestConfig(
            initial_capital=Decimal("10000"),
            use_commission=False,
            use_slippage=False,
        )
        engine = BacktestEngine(config)
        now = datetime.now()
        bars = []
        for i in range(10):
            price = Decimal(str(100 + i * 2))
            bars.append(OHLCV(
                now + timedelta(days=i),
                price, price + 2, price - 1, price + 1, Decimal("1000")
            ))
        engine.load_data("ETH-USD", bars)
        
        def strategy(eng, bar, idx):
            if idx == 0:
                eng.buy("ETH-USD", Decimal("50"))  # Buy 50 units at ~100
        
        engine.set_strategy(strategy)
        metrics = engine.run()
        assert metrics.total_trades > 0 or metrics.total_return >= 0

    def test_long_short_strategy(self):
        config = BacktestConfig(
            initial_capital=Decimal("10000"),
            allow_shorting=True,
            use_commission=False,
            use_slippage=False,
        )
        engine = BacktestEngine(config)
        now = datetime.now()
        bars = []
        for i in range(20):
            price = Decimal(str(100 + (i % 10) * 2))
            bars.append(OHLCV(
                now + timedelta(days=i),
                price, price + 2, price - 1, price + 1, Decimal("1000")
            ))
        engine.load_data("ETH-USD", bars)
        
        def strategy(eng, bar, idx):
            pos = eng.get_position("ETH-USD")
            if idx % 5 == 0:
                if pos is None or pos.side == PositionSide.FLAT:
                    eng.buy("ETH-USD", Decimal("10"))
            elif idx % 5 == 3:
                if pos and pos.side == PositionSide.LONG:
                    eng.sell("ETH-USD", Decimal("10"))
        
        engine.set_strategy(strategy)
        metrics = engine.run()
        assert isinstance(metrics, BacktestMetrics)
