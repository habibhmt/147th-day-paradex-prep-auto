"""Tests for order flow analysis module."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.analytics.order_flow import (
    OrderFlowType,
    AggresionType,
    MarketPressure,
    ImbalanceType,
    TradeFlow,
    VolumeDelta,
    FootprintLevel,
    FootprintBar,
    VolumeProfile,
    OrderFlowMetrics,
    DeltaCalculator,
    FootprintBuilder,
    VolumeProfileBuilder,
    LargeTradeFinder,
    OrderFlowAggregator,
    PressureAnalyzer,
    OrderFlowAnalyzer,
    get_order_flow_analyzer,
    set_order_flow_analyzer,
)


class TestEnums:
    """Test enum classes."""

    def test_order_flow_type_values(self):
        """Test OrderFlowType enum values."""
        assert OrderFlowType.BUY_MARKET.value == "buy_market"
        assert OrderFlowType.SELL_MARKET.value == "sell_market"
        assert OrderFlowType.BUY_LIQUIDATION.value == "buy_liquidation"

    def test_aggression_type_values(self):
        """Test AggresionType enum values."""
        assert AggresionType.AGGRESSIVE_BUY.value == "aggressive_buy"
        assert AggresionType.PASSIVE_SELL.value == "passive_sell"

    def test_market_pressure_values(self):
        """Test MarketPressure enum values."""
        assert MarketPressure.STRONG_BUY.value == "strong_buy"
        assert MarketPressure.SELL.value == "sell"
        assert MarketPressure.NEUTRAL.value == "neutral"

    def test_imbalance_type_values(self):
        """Test ImbalanceType enum values."""
        assert ImbalanceType.BUYING.value == "buying"
        assert ImbalanceType.SELLING.value == "selling"
        assert ImbalanceType.BALANCED.value == "balanced"


class TestTradeFlow:
    """Test TradeFlow class."""

    def test_creation(self):
        """Test TradeFlow creation."""
        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="buy",
        )
        assert trade.price == Decimal("50000")
        assert trade.side == "buy"

    def test_to_dict(self):
        """Test TradeFlow to_dict."""
        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="sell",
            is_liquidation=True,
        )
        result = trade.to_dict()
        assert result["price"] == "50000"
        assert result["side"] == "sell"
        assert result["is_liquidation"] is True

    def test_notional(self):
        """Test notional value calculation."""
        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="buy",
        )
        assert trade.notional == Decimal("25000")


class TestVolumeDelta:
    """Test VolumeDelta class."""

    def test_creation(self):
        """Test VolumeDelta creation."""
        delta = VolumeDelta(
            timestamp=datetime.now(),
            buy_volume=Decimal("100"),
            sell_volume=Decimal("80"),
            delta=Decimal("20"),
            cumulative_delta=Decimal("50"),
            total_volume=Decimal("180"),
        )
        assert delta.delta == Decimal("20")
        assert delta.cumulative_delta == Decimal("50")

    def test_to_dict(self):
        """Test VolumeDelta to_dict."""
        delta = VolumeDelta(
            timestamp=datetime.now(),
            buy_volume=Decimal("100"),
            sell_volume=Decimal("80"),
        )
        result = delta.to_dict()
        assert result["buy_volume"] == "100"


class TestFootprintLevel:
    """Test FootprintLevel class."""

    def test_creation(self):
        """Test FootprintLevel creation."""
        level = FootprintLevel(
            price=Decimal("50000"),
            buy_volume=Decimal("10"),
            sell_volume=Decimal("5"),
            buy_trades=20,
            sell_trades=10,
        )
        assert level.price == Decimal("50000")
        assert level.buy_trades == 20

    def test_to_dict(self):
        """Test FootprintLevel to_dict."""
        level = FootprintLevel(
            price=Decimal("50000"),
            delta=Decimal("5"),
        )
        result = level.to_dict()
        assert result["price"] == "50000"


class TestFootprintBar:
    """Test FootprintBar class."""

    def test_creation(self):
        """Test FootprintBar creation."""
        bar = FootprintBar(
            timestamp=datetime.now(),
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
            total_buy_volume=Decimal("100"),
            total_sell_volume=Decimal("80"),
            delta=Decimal("20"),
        )
        assert bar.delta == Decimal("20")

    def test_to_dict(self):
        """Test FootprintBar to_dict."""
        bar = FootprintBar(
            timestamp=datetime.now(),
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
        )
        result = bar.to_dict()
        assert result["open"] == "50000"


class TestVolumeProfile:
    """Test VolumeProfile class."""

    def test_creation(self):
        """Test VolumeProfile creation."""
        profile = VolumeProfile(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            poc_price=Decimal("50000"),
            total_volume=Decimal("1000"),
        )
        assert profile.poc_price == Decimal("50000")

    def test_to_dict(self):
        """Test VolumeProfile to_dict."""
        profile = VolumeProfile(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
        )
        result = profile.to_dict()
        assert "start_time" in result
        assert "end_time" in result


class TestOrderFlowMetrics:
    """Test OrderFlowMetrics class."""

    def test_creation(self):
        """Test OrderFlowMetrics creation."""
        metrics = OrderFlowMetrics(
            symbol="BTC-USD-PERP",
            cvd=Decimal("100"),
            buy_volume=Decimal("500"),
            sell_volume=Decimal("400"),
            pressure=MarketPressure.BUY,
        )
        assert metrics.symbol == "BTC-USD-PERP"
        assert metrics.cvd == Decimal("100")

    def test_to_dict(self):
        """Test OrderFlowMetrics to_dict."""
        metrics = OrderFlowMetrics(
            symbol="BTC-USD-PERP",
            pressure=MarketPressure.STRONG_BUY,
        )
        result = metrics.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["pressure"] == "strong_buy"


def create_trades(count: int = 10, side: str = "buy") -> list[TradeFlow]:
    """Create sample trades."""
    base_time = datetime.now()
    trades = []
    for i in range(count):
        trades.append(TradeFlow(
            timestamp=base_time + timedelta(seconds=i),
            price=Decimal("50000") + Decimal(i),
            size=Decimal("0.1"),
            side=side,
            trade_id=f"trade_{i}",
        ))
    return trades


class TestDeltaCalculator:
    """Test DeltaCalculator class."""

    def test_process_buy_trade(self):
        """Test processing buy trade."""
        calc = DeltaCalculator()
        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("1"),
            side="buy",
        )
        delta = calc.process_trade(trade)
        assert delta.delta == Decimal("1")
        assert delta.buy_volume == Decimal("1")
        assert delta.sell_volume == Decimal("0")
        assert calc.cvd == Decimal("1")

    def test_process_sell_trade(self):
        """Test processing sell trade."""
        calc = DeltaCalculator()
        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("1"),
            side="sell",
        )
        delta = calc.process_trade(trade)
        assert delta.delta == Decimal("-1")
        assert delta.buy_volume == Decimal("0")
        assert delta.sell_volume == Decimal("1")
        assert calc.cvd == Decimal("-1")

    def test_cumulative_delta(self):
        """Test cumulative delta."""
        calc = DeltaCalculator()
        trades = create_trades(5, "buy") + create_trades(3, "sell")

        for trade in trades:
            calc.process_trade(trade)

        # 5 buys * 0.1 - 3 sells * 0.1 = 0.5 - 0.3 = 0.2
        assert calc.cvd == Decimal("0.2")

    def test_reset(self):
        """Test reset."""
        calc = DeltaCalculator()
        calc.process_trade(TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("1"),
            side="buy",
        ))
        calc.reset()
        assert calc.cvd == Decimal("0")

    def test_calculate_for_period(self):
        """Test calculating for period."""
        calc = DeltaCalculator()
        trades = create_trades(5)
        deltas = calc.calculate_for_period(trades)
        assert len(deltas) == 5


class TestFootprintBuilder:
    """Test FootprintBuilder class."""

    def test_build_bar(self):
        """Test building footprint bar."""
        builder = FootprintBuilder(tick_size=Decimal("10"))
        trades = [
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"),
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("0.5"), "sell"),
            TradeFlow(datetime.now(), Decimal("50010"), Decimal("2"), "buy"),
        ]
        bar = builder.build_bar(trades, datetime.now())

        assert bar.total_buy_volume == Decimal("3")
        assert bar.total_sell_volume == Decimal("0.5")
        assert bar.delta == Decimal("2.5")

    def test_build_bar_empty(self):
        """Test building bar with no trades."""
        builder = FootprintBuilder()
        bar = builder.build_bar([], datetime.now())
        assert bar.open == Decimal("0")
        assert bar.delta == Decimal("0")

    def test_poc_calculation(self):
        """Test POC calculation."""
        builder = FootprintBuilder(tick_size=Decimal("10"))
        # More volume at 50010
        trades = [
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"),
            TradeFlow(datetime.now(), Decimal("50010"), Decimal("5"), "buy"),
            TradeFlow(datetime.now(), Decimal("50010"), Decimal("3"), "sell"),
        ]
        bar = builder.build_bar(trades, datetime.now())
        assert bar.poc_price == Decimal("50010")


class TestVolumeProfileBuilder:
    """Test VolumeProfileBuilder class."""

    def test_build_profile(self):
        """Test building volume profile."""
        builder = VolumeProfileBuilder(tick_size=Decimal("10"))
        trades = create_trades(20)
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()

        profile = builder.build(trades, start, end)

        assert profile.total_volume > 0
        assert profile.poc_price is not None

    def test_build_empty_profile(self):
        """Test building empty profile."""
        builder = VolumeProfileBuilder()
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()

        profile = builder.build([], start, end)
        assert profile.total_volume == Decimal("0")
        assert profile.poc_price is None

    def test_value_area_calculation(self):
        """Test value area calculation."""
        builder = VolumeProfileBuilder(tick_size=Decimal("10"))
        trades = []
        base_time = datetime.now()

        # Create trades with different volumes at different prices
        for i in range(100):
            trades.append(TradeFlow(
                timestamp=base_time + timedelta(seconds=i),
                price=Decimal("50000") + Decimal(i // 10) * 10,
                size=Decimal("1"),
                side="buy",
            ))

        profile = builder.build(trades, base_time, base_time + timedelta(minutes=5))

        assert profile.value_area_high is not None
        assert profile.value_area_low is not None


class TestLargeTradeFinder:
    """Test LargeTradeFinder class."""

    def test_with_min_size(self):
        """Test with minimum size threshold."""
        finder = LargeTradeFinder(min_size=Decimal("10"))

        small_trade = TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy")
        large_trade = TradeFlow(datetime.now(), Decimal("50000"), Decimal("15"), "buy")

        assert finder.is_large_trade(small_trade) is False
        assert finder.is_large_trade(large_trade) is True

    def test_percentile_based(self):
        """Test percentile-based detection."""
        finder = LargeTradeFinder(threshold_percentile=Decimal("90"))

        # Add many small trades first
        for i in range(100):
            finder.add_trade(TradeFlow(
                datetime.now(),
                Decimal("50000"),
                Decimal("0.1"),
                "buy",
            ))

        # Now a large trade should be detected
        large = TradeFlow(datetime.now(), Decimal("50000"), Decimal("10"), "buy")
        assert finder.add_trade(large) is True

    def test_find_large_trades(self):
        """Test finding large trades in list."""
        finder = LargeTradeFinder(min_size=Decimal("5"))

        trades = [
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"),
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("10"), "buy"),
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("2"), "sell"),
            TradeFlow(datetime.now(), Decimal("50000"), Decimal("8"), "sell"),
        ]

        large = finder.find_large_trades(trades)
        assert len(large) == 2

    def test_reset(self):
        """Test reset."""
        finder = LargeTradeFinder()
        finder.add_trade(TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"))
        finder.reset()
        assert len(finder._size_history) == 0


class TestOrderFlowAggregator:
    """Test OrderFlowAggregator class."""

    def test_add_trade(self):
        """Test adding trades."""
        agg = OrderFlowAggregator(window_minutes=5)
        base_time = datetime.now()

        result = agg.add_trade(TradeFlow(base_time, Decimal("50000"), Decimal("1"), "buy"))
        assert result is None  # Window not complete

    def test_window_completion(self):
        """Test window completion."""
        agg = OrderFlowAggregator(window_minutes=1)
        base_time = datetime.now()

        # Add trade in first window
        agg.add_trade(TradeFlow(base_time, Decimal("50000"), Decimal("1"), "buy"))

        # Add trade outside window - should complete previous window
        result = agg.add_trade(TradeFlow(
            base_time + timedelta(minutes=2),
            Decimal("50000"),
            Decimal("1"),
            "buy",
        ))

        assert result is not None
        assert len(result) == 1

    def test_get_current_window(self):
        """Test getting current window."""
        agg = OrderFlowAggregator()
        agg.add_trade(TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"))
        agg.add_trade(TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "sell"))

        window = agg.get_current_window()
        assert len(window) == 2

    def test_clear(self):
        """Test clearing aggregator."""
        agg = OrderFlowAggregator()
        agg.add_trade(TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy"))
        agg.clear()
        assert len(agg.get_current_window()) == 0


class TestPressureAnalyzer:
    """Test PressureAnalyzer class."""

    def test_strong_buy(self):
        """Test strong buy pressure."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("80"), Decimal("20"))
        assert pressure == MarketPressure.STRONG_BUY

    def test_buy(self):
        """Test buy pressure."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("60"), Decimal("40"))
        assert pressure == MarketPressure.BUY

    def test_neutral(self):
        """Test neutral pressure."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("50"), Decimal("50"))
        assert pressure == MarketPressure.NEUTRAL

    def test_sell(self):
        """Test sell pressure."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("40"), Decimal("60"))
        assert pressure == MarketPressure.SELL

    def test_strong_sell(self):
        """Test strong sell pressure."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("20"), Decimal("80"))
        assert pressure == MarketPressure.STRONG_SELL

    def test_zero_volume(self):
        """Test with zero volume."""
        analyzer = PressureAnalyzer()
        pressure = analyzer.analyze(Decimal("0"), Decimal("0"))
        assert pressure == MarketPressure.NEUTRAL


class TestOrderFlowAnalyzer:
    """Test OrderFlowAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = OrderFlowAnalyzer()
        assert analyzer is not None

    def test_add_trade(self):
        """Test adding trade."""
        analyzer = OrderFlowAnalyzer()
        trade = TradeFlow(datetime.now(), Decimal("50000"), Decimal("1"), "buy")
        analyzer.add_trade("BTC-USD-PERP", trade)

        metrics = analyzer.get_metrics("BTC-USD-PERP")
        assert metrics is not None

    def test_get_metrics(self):
        """Test getting metrics."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10, "buy"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        for trade in create_trades(5, "sell"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        metrics = analyzer.get_metrics("BTC-USD-PERP")
        assert metrics.buy_trades == 10
        assert metrics.sell_trades == 5

    def test_get_cvd(self):
        """Test getting CVD."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10, "buy"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        cvd = analyzer.get_cvd("BTC-USD-PERP")
        assert cvd == Decimal("1")  # 10 * 0.1

    def test_get_delta_series(self):
        """Test getting delta series."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(20):
            analyzer.add_trade("BTC-USD-PERP", trade)

        deltas = analyzer.get_delta_series("BTC-USD-PERP", limit=10)
        assert len(deltas) == 10

    def test_build_footprint(self):
        """Test building footprint."""
        analyzer = OrderFlowAnalyzer(tick_size=Decimal("10"))

        for trade in create_trades(20):
            analyzer.add_trade("BTC-USD-PERP", trade)

        bar = analyzer.build_footprint("BTC-USD-PERP", datetime.now())
        assert bar is not None

    def test_build_volume_profile(self):
        """Test building volume profile."""
        analyzer = OrderFlowAnalyzer(tick_size=Decimal("10"))
        base_time = datetime.now()

        for trade in create_trades(20):
            analyzer.add_trade("BTC-USD-PERP", trade)

        profile = analyzer.build_volume_profile(
            "BTC-USD-PERP",
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=1),
        )
        assert profile is not None

    def test_get_large_trades(self):
        """Test getting large trades."""
        analyzer = OrderFlowAnalyzer(large_trade_min=Decimal("5"))

        for trade in create_trades(10):
            analyzer.add_trade("BTC-USD-PERP", trade)

        # Add a large trade
        large = TradeFlow(datetime.now(), Decimal("50000"), Decimal("10"), "buy")
        analyzer.add_trade("BTC-USD-PERP", large)

        large_trades = analyzer.get_large_trades("BTC-USD-PERP")
        assert len(large_trades) == 1

    def test_get_pressure(self):
        """Test getting pressure."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(80, "buy"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        for trade in create_trades(20, "sell"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        pressure = analyzer.get_pressure("BTC-USD-PERP")
        assert pressure in (MarketPressure.BUY, MarketPressure.STRONG_BUY)

    def test_get_imbalance(self):
        """Test getting imbalance."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(80, "buy"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        for trade in create_trades(20, "sell"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        imbalance = analyzer.get_imbalance("BTC-USD-PERP")
        assert imbalance == ImbalanceType.BUYING

    def test_add_callback(self):
        """Test adding callback."""
        analyzer = OrderFlowAnalyzer(large_trade_min=Decimal("5"))
        events = []

        def callback(event, data):
            events.append(event)

        analyzer.add_callback(callback)

        # Add large trade to trigger callback
        large = TradeFlow(datetime.now(), Decimal("50000"), Decimal("10"), "buy")
        analyzer.add_trade("BTC-USD-PERP", large)

        assert "large_trade" in events

    def test_remove_callback(self):
        """Test removing callback."""
        analyzer = OrderFlowAnalyzer()

        def callback(event, data):
            pass

        analyzer.add_callback(callback)
        assert analyzer.remove_callback(callback) is True
        assert analyzer.remove_callback(callback) is False

    def test_clear(self):
        """Test clearing data."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10):
            analyzer.add_trade("BTC-USD-PERP", trade)

        analyzer.clear("BTC-USD-PERP")
        assert analyzer.get_metrics("BTC-USD-PERP") is None

    def test_clear_all(self):
        """Test clearing all data."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10):
            analyzer.add_trade("BTC-USD-PERP", trade)
            analyzer.add_trade("ETH-USD-PERP", trade)

        analyzer.clear()
        assert analyzer.get_metrics("BTC-USD-PERP") is None
        assert analyzer.get_metrics("ETH-USD-PERP") is None

    def test_get_summary(self):
        """Test getting summary."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10):
            analyzer.add_trade("BTC-USD-PERP", trade)

        summary = analyzer.get_summary()
        assert "BTC-USD-PERP" in summary["symbols"]
        assert summary["total_symbols"] == 1

    def test_get_summary_by_symbol(self):
        """Test getting summary by symbol."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(10):
            analyzer.add_trade("BTC-USD-PERP", trade)

        summary = analyzer.get_summary("BTC-USD-PERP")
        assert summary["symbol"] == "BTC-USD-PERP"
        assert summary["trade_count"] == 10


class TestGlobalInstance:
    """Test global instance functions."""

    def test_get_order_flow_analyzer(self):
        """Test getting global instance."""
        analyzer = get_order_flow_analyzer()
        assert analyzer is not None
        assert isinstance(analyzer, OrderFlowAnalyzer)

    def test_set_order_flow_analyzer(self):
        """Test setting global instance."""
        custom = OrderFlowAnalyzer()
        set_order_flow_analyzer(custom)
        assert get_order_flow_analyzer() is custom


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete order flow analysis workflow."""
        analyzer = OrderFlowAnalyzer(tick_size=Decimal("10"))

        # Simulate trading session
        base_time = datetime.now()

        # Add buy pressure
        for i in range(50):
            analyzer.add_trade("BTC-USD-PERP", TradeFlow(
                timestamp=base_time + timedelta(seconds=i),
                price=Decimal("50000") + Decimal(i % 10),
                size=Decimal("0.5"),
                side="buy",
            ))

        # Add sell pressure
        for i in range(30):
            analyzer.add_trade("BTC-USD-PERP", TradeFlow(
                timestamp=base_time + timedelta(seconds=50 + i),
                price=Decimal("50010") - Decimal(i % 10),
                size=Decimal("0.3"),
                side="sell",
            ))

        # Check metrics
        metrics = analyzer.get_metrics("BTC-USD-PERP")
        assert metrics.buy_volume > metrics.sell_volume
        assert metrics.pressure in (MarketPressure.BUY, MarketPressure.STRONG_BUY)

        # Build footprint
        bar = analyzer.build_footprint("BTC-USD-PERP", datetime.now())
        assert bar.delta > 0

        # Build volume profile
        profile = analyzer.build_volume_profile(
            "BTC-USD-PERP",
            base_time - timedelta(minutes=5),
            datetime.now(),
        )
        assert profile.total_volume > 0

    def test_multiple_symbols(self):
        """Test with multiple symbols."""
        analyzer = OrderFlowAnalyzer()

        for trade in create_trades(20, "buy"):
            analyzer.add_trade("BTC-USD-PERP", trade)

        for trade in create_trades(20, "sell"):
            analyzer.add_trade("ETH-USD-PERP", trade)

        btc_metrics = analyzer.get_metrics("BTC-USD-PERP")
        eth_metrics = analyzer.get_metrics("ETH-USD-PERP")

        assert btc_metrics.imbalance == ImbalanceType.BUYING
        assert eth_metrics.imbalance == ImbalanceType.SELLING


class TestEdgeCases:
    """Test edge cases."""

    def test_nonexistent_symbol(self):
        """Test with nonexistent symbol."""
        analyzer = OrderFlowAnalyzer()
        assert analyzer.get_metrics("FAKE") is None
        assert analyzer.get_cvd("FAKE") == Decimal("0")
        assert analyzer.get_pressure("FAKE") == MarketPressure.NEUTRAL
        assert analyzer.get_imbalance("FAKE") == ImbalanceType.BALANCED

    def test_liquidation_trades(self):
        """Test liquidation trade handling."""
        analyzer = OrderFlowAnalyzer()

        trade = TradeFlow(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            size=Decimal("5"),
            side="sell",
            is_liquidation=True,
        )
        analyzer.add_trade("BTC-USD-PERP", trade)

        metrics = analyzer.get_metrics("BTC-USD-PERP")
        assert metrics.liquidation_volume == Decimal("5")

    def test_zero_volume(self):
        """Test with no trades."""
        analyzer = OrderFlowAnalyzer()
        summary = analyzer.get_summary("BTC-USD-PERP")
        assert summary == {}
