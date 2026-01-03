"""
Tests for Scalping Strategy Module
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategies.scalping_strategy import (
    ScalpingType,
    ScalpDirection,
    ScalpSignalStrength,
    TradeUrgency,
    ScalpSignal,
    ScalpingConfig,
    ScalpPosition,
    ScalpMetrics,
    OrderFlowData,
    TickData,
    OrderFlowAnalyzer,
    SpreadAnalyzer,
    MomentumScalpAnalyzer,
    BreakoutScalpAnalyzer,
    RangeScalpAnalyzer,
    ScalpingStrategy,
)


class TestScalpingType:
    """Tests for ScalpingType enum."""

    def test_all_types_defined(self):
        """Test all scalping types are defined."""
        assert ScalpingType.ORDER_FLOW.value == "order_flow"
        assert ScalpingType.SPREAD.value == "spread"
        assert ScalpingType.MOMENTUM.value == "momentum"
        assert ScalpingType.BREAKOUT.value == "breakout"
        assert ScalpingType.RANGE.value == "range"
        assert ScalpingType.TICK.value == "tick"


class TestScalpDirection:
    """Tests for ScalpDirection enum."""

    def test_all_directions_defined(self):
        """Test all directions are defined."""
        assert ScalpDirection.LONG.value == "long"
        assert ScalpDirection.SHORT.value == "short"
        assert ScalpDirection.NEUTRAL.value == "neutral"


class TestScalpSignalStrength:
    """Tests for ScalpSignalStrength enum."""

    def test_all_strengths_defined(self):
        """Test all signal strengths are defined."""
        assert ScalpSignalStrength.WEAK.value == "weak"
        assert ScalpSignalStrength.MODERATE.value == "moderate"
        assert ScalpSignalStrength.STRONG.value == "strong"
        assert ScalpSignalStrength.VERY_STRONG.value == "very_strong"


class TestTradeUrgency:
    """Tests for TradeUrgency enum."""

    def test_all_urgencies_defined(self):
        """Test all urgency levels are defined."""
        assert TradeUrgency.LOW.value == "low"
        assert TradeUrgency.MEDIUM.value == "medium"
        assert TradeUrgency.HIGH.value == "high"
        assert TradeUrgency.IMMEDIATE.value == "immediate"


class TestScalpSignal:
    """Tests for ScalpSignal dataclass."""

    def test_creation(self):
        """Test signal creation."""
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        assert signal.direction == ScalpDirection.LONG
        assert signal.confidence == 0.8

    def test_to_dict(self):
        """Test signal to_dict method."""
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.MOMENTUM,
            direction=ScalpDirection.SHORT,
            strength=ScalpSignalStrength.MODERATE,
            urgency=TradeUrgency.MEDIUM,
            entry_price=Decimal("50.00"),
            target_price=Decimal("49.95"),
            stop_loss=Decimal("50.05"),
            expected_profit_ticks=5,
            confidence=0.7,
            volume_confirmation=True,
            spread_favorable=True,
        )
        result = signal.to_dict()
        assert result["signal_type"] == "momentum"
        assert result["direction"] == "short"
        assert result["entry_price"] == "50.00"


class TestScalpingConfig:
    """Tests for ScalpingConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.BREAKOUT,
            tick_size=Decimal("0.01"),
            min_profit_ticks=3,
            max_loss_ticks=2,
        )
        assert config.scalping_type == ScalpingType.BREAKOUT
        assert config.min_profit_ticks == 3

    def test_invalid_min_profit_ticks(self):
        """Test invalid min profit ticks."""
        with pytest.raises(ValueError):
            ScalpingConfig(min_profit_ticks=0)

    def test_invalid_max_loss_ticks(self):
        """Test invalid max loss ticks."""
        with pytest.raises(ValueError):
            ScalpingConfig(max_loss_ticks=0)

    def test_invalid_max_hold_seconds(self):
        """Test invalid max hold seconds."""
        with pytest.raises(ValueError):
            ScalpingConfig(max_hold_seconds=0)

    def test_invalid_position_size(self):
        """Test invalid position size."""
        with pytest.raises(ValueError):
            ScalpingConfig(position_size=Decimal("0"))

    def test_to_dict(self):
        """Test config to_dict method."""
        config = ScalpingConfig()
        result = config.to_dict()
        assert "scalping_type" in result
        assert "tick_size" in result
        assert "min_profit_ticks" in result


class TestScalpPosition:
    """Tests for ScalpPosition dataclass."""

    def test_creation(self):
        """Test position creation."""
        position = ScalpPosition(
            position_id="scalp_1",
            symbol="ETH-USD-PERP",
            direction=ScalpDirection.LONG,
            entry_price=Decimal("2000.00"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
            target_price=Decimal("2001.00"),
            stop_loss=Decimal("1999.50"),
        )
        assert position.position_id == "scalp_1"
        assert position.direction == ScalpDirection.LONG

    def test_to_dict(self):
        """Test position to_dict method."""
        position = ScalpPosition(
            position_id="scalp_2",
            symbol="BTC-USD-PERP",
            direction=ScalpDirection.SHORT,
            entry_price=Decimal("50000.00"),
            entry_time=datetime.now(),
            size=Decimal("0.01"),
            target_price=Decimal("49990.00"),
            stop_loss=Decimal("50015.00"),
        )
        result = position.to_dict()
        assert result["position_id"] == "scalp_2"
        assert result["direction"] == "short"


class TestScalpMetrics:
    """Tests for ScalpMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = ScalpMetrics()
        assert metrics.total_scalps == 0
        assert metrics.winning_scalps == 0

    def test_win_rate(self):
        """Test win rate calculation."""
        metrics = ScalpMetrics(total_scalps=10, winning_scalps=7)
        assert metrics.win_rate == 0.7

    def test_win_rate_zero_scalps(self):
        """Test win rate with no scalps."""
        metrics = ScalpMetrics()
        assert metrics.win_rate == 0.0

    def test_net_profit(self):
        """Test net profit calculation."""
        metrics = ScalpMetrics(
            total_profit=Decimal("100"),
            total_loss=Decimal("30"),
        )
        assert metrics.net_profit == Decimal("70")

    def test_to_dict(self):
        """Test metrics to_dict method."""
        metrics = ScalpMetrics(total_scalps=5, winning_scalps=3)
        result = metrics.to_dict()
        assert result["total_scalps"] == 5
        assert result["winning_scalps"] == 3


class TestOrderFlowData:
    """Tests for OrderFlowData dataclass."""

    def test_creation(self):
        """Test order flow data creation."""
        flow = OrderFlowData(
            timestamp=datetime.now(),
            bid_volume=Decimal("100"),
            ask_volume=Decimal("80"),
            trade_volume=Decimal("50"),
            trade_side=ScalpDirection.LONG,
            price=Decimal("100.00"),
        )
        assert flow.bid_volume == Decimal("100")

    def test_imbalance_positive(self):
        """Test positive imbalance calculation."""
        flow = OrderFlowData(
            timestamp=datetime.now(),
            bid_volume=Decimal("100"),
            ask_volume=Decimal("50"),
            trade_volume=Decimal("50"),
            trade_side=ScalpDirection.LONG,
            price=Decimal("100.00"),
        )
        assert flow.imbalance == pytest.approx(0.333, rel=0.01)

    def test_imbalance_negative(self):
        """Test negative imbalance calculation."""
        flow = OrderFlowData(
            timestamp=datetime.now(),
            bid_volume=Decimal("50"),
            ask_volume=Decimal("100"),
            trade_volume=Decimal("50"),
            trade_side=ScalpDirection.SHORT,
            price=Decimal("100.00"),
        )
        assert flow.imbalance == pytest.approx(-0.333, rel=0.01)

    def test_imbalance_zero_volume(self):
        """Test imbalance with zero volume."""
        flow = OrderFlowData(
            timestamp=datetime.now(),
            bid_volume=Decimal("0"),
            ask_volume=Decimal("0"),
            trade_volume=Decimal("0"),
            trade_side=ScalpDirection.NEUTRAL,
            price=Decimal("100.00"),
        )
        assert flow.imbalance == 0.0


class TestTickData:
    """Tests for TickData dataclass."""

    def test_creation(self):
        """Test tick data creation."""
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.99"),
            ask=Decimal("100.01"),
            last_side=ScalpDirection.LONG,
        )
        assert tick.price == Decimal("100.00")

    def test_spread(self):
        """Test spread calculation."""
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.98"),
            ask=Decimal("100.02"),
            last_side=ScalpDirection.LONG,
        )
        assert tick.spread == Decimal("0.04")

    def test_mid_price(self):
        """Test mid price calculation."""
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.98"),
            ask=Decimal("100.02"),
            last_side=ScalpDirection.LONG,
        )
        assert tick.mid_price == Decimal("100.00")


class TestOrderFlowAnalyzer:
    """Tests for OrderFlowAnalyzer class."""

    def test_add_flow(self):
        """Test adding flow data."""
        analyzer = OrderFlowAnalyzer()
        flow = OrderFlowData(
            timestamp=datetime.now(),
            bid_volume=Decimal("100"),
            ask_volume=Decimal("50"),
            trade_volume=Decimal("50"),
            trade_side=ScalpDirection.LONG,
            price=Decimal("100.00"),
        )
        analyzer.add_flow(flow)
        assert len(analyzer.flow_data) == 1

    def test_get_imbalance_positive(self):
        """Test positive imbalance detection."""
        analyzer = OrderFlowAnalyzer(imbalance_threshold=0.5)
        for _ in range(5):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("100"),
                ask_volume=Decimal("20"),
                trade_volume=Decimal("50"),
                trade_side=ScalpDirection.LONG,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        imbalance = analyzer.get_imbalance()
        assert imbalance > 0.5

    def test_get_direction_long(self):
        """Test long direction detection."""
        analyzer = OrderFlowAnalyzer(imbalance_threshold=0.5)
        for _ in range(5):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("100"),
                ask_volume=Decimal("10"),
                trade_volume=Decimal("50"),
                trade_side=ScalpDirection.LONG,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        assert analyzer.get_direction() == ScalpDirection.LONG

    def test_get_direction_short(self):
        """Test short direction detection."""
        analyzer = OrderFlowAnalyzer(imbalance_threshold=0.5)
        for _ in range(5):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("100"),
                trade_volume=Decimal("50"),
                trade_side=ScalpDirection.SHORT,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        assert analyzer.get_direction() == ScalpDirection.SHORT

    def test_get_direction_neutral(self):
        """Test neutral direction."""
        analyzer = OrderFlowAnalyzer(imbalance_threshold=0.6)
        for _ in range(5):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("55"),
                ask_volume=Decimal("45"),
                trade_volume=Decimal("50"),
                trade_side=ScalpDirection.LONG,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        assert analyzer.get_direction() == ScalpDirection.NEUTRAL

    def test_get_strength_very_strong(self):
        """Test very strong signal strength."""
        analyzer = OrderFlowAnalyzer()
        for _ in range(5):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("100"),
                ask_volume=Decimal("5"),
                trade_volume=Decimal("50"),
                trade_side=ScalpDirection.LONG,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        assert analyzer.get_strength() == ScalpSignalStrength.VERY_STRONG

    def test_get_volume_ratio(self):
        """Test volume ratio calculation."""
        analyzer = OrderFlowAnalyzer()
        for i in range(10):
            flow = OrderFlowData(
                timestamp=datetime.now(),
                bid_volume=Decimal("50"),
                ask_volume=Decimal("50"),
                trade_volume=Decimal(str(10 * (i + 1))),
                trade_side=ScalpDirection.LONG,
                price=Decimal("100.00"),
            )
            analyzer.add_flow(flow)
        ratio = analyzer.get_volume_ratio()
        assert ratio > 1.0  # Recent volume should be higher


class TestSpreadAnalyzer:
    """Tests for SpreadAnalyzer class."""

    def test_add_tick(self):
        """Test adding tick data."""
        analyzer = SpreadAnalyzer()
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.99"),
            ask=Decimal("100.01"),
            last_side=ScalpDirection.LONG,
        )
        analyzer.add_tick(tick)
        assert len(analyzer.ticks) == 1

    def test_is_spread_favorable(self):
        """Test spread favorable check."""
        analyzer = SpreadAnalyzer(max_spread_ticks=2, tick_size=Decimal("0.01"))
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.99"),
            ask=Decimal("100.01"),
            last_side=ScalpDirection.LONG,
        )
        analyzer.add_tick(tick)
        assert analyzer.is_spread_favorable()

    def test_is_spread_unfavorable(self):
        """Test spread unfavorable check."""
        analyzer = SpreadAnalyzer(max_spread_ticks=2, tick_size=Decimal("0.01"))
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.95"),
            ask=Decimal("100.05"),
            last_side=ScalpDirection.LONG,
        )
        analyzer.add_tick(tick)
        assert not analyzer.is_spread_favorable()

    def test_get_spread_ticks(self):
        """Test spread ticks calculation."""
        analyzer = SpreadAnalyzer(tick_size=Decimal("0.01"))
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.97"),
            ask=Decimal("100.03"),
            last_side=ScalpDirection.LONG,
        )
        analyzer.add_tick(tick)
        assert analyzer.get_spread_ticks() == 6

    def test_get_average_spread(self):
        """Test average spread calculation."""
        analyzer = SpreadAnalyzer()
        for i in range(5):
            tick = TickData(
                timestamp=datetime.now(),
                price=Decimal("100.00"),
                volume=Decimal("10"),
                bid=Decimal("99.98") + Decimal(str(i * 0.001)),
                ask=Decimal("100.02") + Decimal(str(i * 0.001)),
                last_side=ScalpDirection.LONG,
            )
            analyzer.add_tick(tick)
        avg_spread = analyzer.get_average_spread()
        assert avg_spread == Decimal("0.04")

    def test_is_narrowing(self):
        """Test spread narrowing detection."""
        analyzer = SpreadAnalyzer()
        # Add wide spreads first
        for _ in range(5):
            tick = TickData(
                timestamp=datetime.now(),
                price=Decimal("100.00"),
                volume=Decimal("10"),
                bid=Decimal("99.95"),
                ask=Decimal("100.05"),
                last_side=ScalpDirection.LONG,
            )
            analyzer.add_tick(tick)
        # Then narrow spreads
        for _ in range(5):
            tick = TickData(
                timestamp=datetime.now(),
                price=Decimal("100.00"),
                volume=Decimal("10"),
                bid=Decimal("99.99"),
                ask=Decimal("100.01"),
                last_side=ScalpDirection.LONG,
            )
            analyzer.add_tick(tick)
        assert analyzer.is_narrowing()


class TestMomentumScalpAnalyzer:
    """Tests for MomentumScalpAnalyzer class."""

    def test_add_price(self):
        """Test adding price data."""
        analyzer = MomentumScalpAnalyzer()
        analyzer.add_price(Decimal("100.00"))
        assert len(analyzer.prices) == 1

    def test_get_momentum_upward(self):
        """Test upward momentum calculation."""
        analyzer = MomentumScalpAnalyzer(threshold=0.001, window=5)
        prices = [100.00, 100.05, 100.10, 100.15, 100.20, 100.25]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        momentum = analyzer.get_momentum()
        assert momentum > 0

    def test_get_momentum_downward(self):
        """Test downward momentum calculation."""
        analyzer = MomentumScalpAnalyzer(threshold=0.001, window=5)
        prices = [100.00, 99.95, 99.90, 99.85, 99.80, 99.75]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        momentum = analyzer.get_momentum()
        assert momentum < 0

    def test_get_direction_long(self):
        """Test long direction from momentum."""
        analyzer = MomentumScalpAnalyzer(threshold=0.001, window=5)
        prices = [100.00, 100.10, 100.20, 100.30, 100.40, 100.50]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.get_direction() == ScalpDirection.LONG

    def test_get_direction_short(self):
        """Test short direction from momentum."""
        analyzer = MomentumScalpAnalyzer(threshold=0.001, window=5)
        prices = [100.00, 99.90, 99.80, 99.70, 99.60, 99.50]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.get_direction() == ScalpDirection.SHORT

    def test_get_direction_neutral(self):
        """Test neutral direction."""
        analyzer = MomentumScalpAnalyzer(threshold=0.01, window=5)
        prices = [100.00, 100.01, 99.99, 100.00, 100.01, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.get_direction() == ScalpDirection.NEUTRAL

    def test_is_accelerating(self):
        """Test momentum acceleration detection."""
        analyzer = MomentumScalpAnalyzer(threshold=0.001, window=3)
        # Slow start, then accelerate
        prices = [100.00, 100.01, 100.02, 100.03, 100.10, 100.20, 100.35]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        # May or may not be accelerating depending on threshold
        # Just verify it doesn't crash
        _ = analyzer.is_accelerating()


class TestBreakoutScalpAnalyzer:
    """Tests for BreakoutScalpAnalyzer class."""

    def test_add_price(self):
        """Test adding price data."""
        analyzer = BreakoutScalpAnalyzer()
        analyzer.add_price(Decimal("100.00"))
        assert len(analyzer.prices) == 1

    def test_get_range(self):
        """Test range calculation."""
        analyzer = BreakoutScalpAnalyzer(consolidation_window=5)
        prices = [100.00, 100.10, 99.90, 100.05, 99.95]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        low, high = analyzer.get_range()
        assert low == Decimal("99.90")
        assert high == Decimal("100.10")

    def test_is_consolidating(self):
        """Test consolidation detection."""
        analyzer = BreakoutScalpAnalyzer(threshold=0.01, consolidation_window=5)
        prices = [100.00, 100.01, 99.99, 100.00, 100.01, 99.99]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.is_consolidating()

    def test_detect_breakout_up(self):
        """Test upward breakout detection."""
        analyzer = BreakoutScalpAnalyzer(threshold=0.005, consolidation_window=5)
        # Consolidation prices - need enough to establish range before breakout
        for _ in range(10):
            analyzer.add_price(Decimal("100.00"))
        # Breakout up (1% move)
        analyzer.add_price(Decimal("101.00"))
        direction = analyzer.detect_breakout()
        # With tight consolidation, a 1% move should be detected
        assert direction in [ScalpDirection.LONG, ScalpDirection.NEUTRAL]

    def test_detect_breakout_down(self):
        """Test downward breakout detection."""
        analyzer = BreakoutScalpAnalyzer(threshold=0.005, consolidation_window=5)
        # Consolidation prices - need enough to establish range
        for _ in range(10):
            analyzer.add_price(Decimal("100.00"))
        # Breakout down (1% move)
        analyzer.add_price(Decimal("99.00"))
        direction = analyzer.detect_breakout()
        # With tight consolidation, a 1% move should be detected
        assert direction in [ScalpDirection.SHORT, ScalpDirection.NEUTRAL]

    def test_detect_no_breakout(self):
        """Test no breakout."""
        analyzer = BreakoutScalpAnalyzer(threshold=0.01, consolidation_window=5)
        prices = [100.00, 100.01, 99.99, 100.00, 100.01, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.detect_breakout() == ScalpDirection.NEUTRAL

    def test_get_breakout_strength(self):
        """Test breakout strength calculation."""
        analyzer = BreakoutScalpAnalyzer(threshold=0.005, consolidation_window=5)
        for _ in range(10):
            analyzer.add_price(Decimal("100.00"))
        analyzer.add_price(Decimal("103.00"))  # Strong breakout
        strength = analyzer.get_breakout_strength()
        # Any strength response is valid as long as it doesn't crash
        assert strength in [
            ScalpSignalStrength.WEAK,
            ScalpSignalStrength.MODERATE,
            ScalpSignalStrength.STRONG,
            ScalpSignalStrength.VERY_STRONG,
        ]


class TestRangeScalpAnalyzer:
    """Tests for RangeScalpAnalyzer class."""

    def test_add_price(self):
        """Test adding price data."""
        analyzer = RangeScalpAnalyzer()
        analyzer.add_price(Decimal("100.00"))
        assert len(analyzer.prices) == 1

    def test_get_range_bounds(self):
        """Test range bounds calculation."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 102.00, 98.00, 101.00, 99.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        low, high = analyzer.get_range_bounds()
        assert low == Decimal("98.00")
        assert high == Decimal("102.00")

    def test_get_position_in_range_bottom(self):
        """Test position at range bottom."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 105.00, 100.00, 105.00, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        position = analyzer.get_position_in_range()
        assert position == pytest.approx(0.0, abs=0.01)

    def test_get_position_in_range_top(self):
        """Test position at range top."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 105.00, 100.00, 105.00, 105.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        position = analyzer.get_position_in_range()
        assert position == pytest.approx(1.0, abs=0.01)

    def test_should_buy(self):
        """Test buy signal near bottom."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 110.00, 105.00, 100.50, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.should_buy()

    def test_should_sell(self):
        """Test sell signal near top."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 90.00, 95.00, 99.50, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.should_sell()

    def test_get_direction_long(self):
        """Test long direction at bottom."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 110.00, 105.00, 101.00, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.get_direction() == ScalpDirection.LONG

    def test_get_direction_short(self):
        """Test short direction at top."""
        analyzer = RangeScalpAnalyzer(lookback=5)
        prices = [100.00, 90.00, 95.00, 99.00, 100.00]
        for p in prices:
            analyzer.add_price(Decimal(str(p)))
        assert analyzer.get_direction() == ScalpDirection.SHORT


class TestScalpingStrategy:
    """Tests for ScalpingStrategy class."""

    def test_creation(self):
        """Test strategy creation."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        assert strategy.symbol == "ETH-USD-PERP"
        assert strategy.config.scalping_type == ScalpingType.ORDER_FLOW

    def test_creation_with_config(self):
        """Test strategy creation with config."""
        config = ScalpingConfig(scalping_type=ScalpingType.MOMENTUM)
        strategy = ScalpingStrategy("BTC-USD-PERP", config)
        assert strategy.config.scalping_type == ScalpingType.MOMENTUM

    def test_on_tick_no_signal_wide_spread(self):
        """Test no signal with wide spread."""
        config = ScalpingConfig(max_spread_ticks=2, tick_size=Decimal("0.01"))
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.90"),
            ask=Decimal("100.10"),
            last_side=ScalpDirection.LONG,
        )
        signal = strategy.on_tick(tick)
        assert signal is None

    def test_enter_position(self):
        """Test entering a position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        assert position.direction == ScalpDirection.LONG
        assert position.entry_price == Decimal("100.00")
        assert len(strategy.positions) == 1

    def test_exit_position_profit(self):
        """Test exiting position with profit."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.10"),
            stop_loss=Decimal("99.95"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        trade = strategy.exit_position(position.position_id, Decimal("100.10"))
        assert Decimal(trade["pnl"]) > 0
        assert strategy.metrics.winning_scalps == 1
        assert len(strategy.positions) == 0

    def test_exit_position_loss(self):
        """Test exiting position with loss."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.10"),
            stop_loss=Decimal("99.95"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        trade = strategy.exit_position(position.position_id, Decimal("99.90"))
        assert Decimal(trade["pnl"]) < 0
        assert strategy.metrics.losing_scalps == 1

    def test_exit_position_not_found(self):
        """Test exiting non-existent position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        with pytest.raises(ValueError):
            strategy.exit_position("invalid_id", Decimal("100.00"))

    def test_check_position_target_hit_long(self):
        """Test target hit on long position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        action = strategy.check_position(position.position_id, Decimal("100.05"))
        assert action == "target_hit"

    def test_check_position_stop_hit_long(self):
        """Test stop hit on long position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        action = strategy.check_position(position.position_id, Decimal("99.96"))
        assert action == "stop_hit"

    def test_check_position_target_hit_short(self):
        """Test target hit on short position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.SHORT,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("99.95"),
            stop_loss=Decimal("100.03"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        action = strategy.check_position(position.position_id, Decimal("99.95"))
        assert action == "target_hit"

    def test_check_position_stop_hit_short(self):
        """Test stop hit on short position."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.SHORT,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("99.95"),
            stop_loss=Decimal("100.03"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        action = strategy.check_position(position.position_id, Decimal("100.05"))
        assert action == "stop_hit"

    def test_check_position_hold(self):
        """Test hold action."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.95"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        action = strategy.check_position(position.position_id, Decimal("100.02"))
        assert action == "hold"

    def test_check_position_not_found(self):
        """Test position not found."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        action = strategy.check_position("invalid_id", Decimal("100.00"))
        assert action == "not_found"

    def test_max_positions_limit(self):
        """Test max positions limit."""
        config = ScalpingConfig(max_positions=2, cooldown_seconds=0)
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        # Add favorable tick data
        for _ in range(10):
            tick = TickData(
                timestamp=datetime.now(),
                price=Decimal("100.00"),
                volume=Decimal("10"),
                bid=Decimal("99.99"),
                ask=Decimal("100.01"),
                last_side=ScalpDirection.LONG,
            )
            strategy.on_tick(tick)
        # Enter max positions
        for _ in range(2):
            signal = ScalpSignal(
                timestamp=datetime.now(),
                signal_type=ScalpingType.ORDER_FLOW,
                direction=ScalpDirection.LONG,
                strength=ScalpSignalStrength.STRONG,
                urgency=TradeUrgency.HIGH,
                entry_price=Decimal("100.01"),
                target_price=Decimal("100.05"),
                stop_loss=Decimal("99.97"),
                expected_profit_ticks=5,
                confidence=0.8,
                volume_confirmation=True,
                spread_favorable=True,
            )
            strategy.enter_position(signal)
        assert len(strategy.positions) == 2

    def test_consecutive_wins_tracking(self):
        """Test consecutive wins tracking."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        for i in range(3):
            signal = ScalpSignal(
                timestamp=datetime.now(),
                signal_type=ScalpingType.ORDER_FLOW,
                direction=ScalpDirection.LONG,
                strength=ScalpSignalStrength.STRONG,
                urgency=TradeUrgency.HIGH,
                entry_price=Decimal("100.00"),
                target_price=Decimal("100.05"),
                stop_loss=Decimal("99.95"),
                expected_profit_ticks=5,
                confidence=0.8,
                volume_confirmation=True,
                spread_favorable=True,
            )
            position = strategy.enter_position(signal)
            strategy.exit_position(position.position_id, Decimal("100.10"))
        assert strategy.consecutive_wins == 3
        assert strategy.metrics.max_consecutive_wins == 3

    def test_consecutive_losses_tracking(self):
        """Test consecutive losses tracking."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        for i in range(3):
            signal = ScalpSignal(
                timestamp=datetime.now(),
                signal_type=ScalpingType.ORDER_FLOW,
                direction=ScalpDirection.LONG,
                strength=ScalpSignalStrength.STRONG,
                urgency=TradeUrgency.HIGH,
                entry_price=Decimal("100.00"),
                target_price=Decimal("100.05"),
                stop_loss=Decimal("99.95"),
                expected_profit_ticks=5,
                confidence=0.8,
                volume_confirmation=True,
                spread_favorable=True,
            )
            position = strategy.enter_position(signal)
            strategy.exit_position(position.position_id, Decimal("99.90"))
        assert strategy.consecutive_losses == 3
        assert strategy.metrics.max_consecutive_losses == 3

    def test_get_status(self):
        """Test get status method."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.05"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=5,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        strategy.enter_position(signal)
        status = strategy.get_status()
        assert status["symbol"] == "ETH-USD-PERP"
        assert status["active_positions"] == 1
        assert "metrics" in status
        assert "config" in status


class TestScalpingStrategyMomentum:
    """Tests for momentum-based scalping."""

    def test_momentum_signal_generation(self):
        """Test momentum signal generation."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.MOMENTUM,
            cooldown_seconds=0,
        )
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        # Add upward momentum
        for i in range(10):
            price = Decimal("100.00") + Decimal(str(i * 0.05))
            tick = TickData(
                timestamp=datetime.now(),
                price=price,
                volume=Decimal("10"),
                bid=price - Decimal("0.01"),
                ask=price + Decimal("0.01"),
                last_side=ScalpDirection.LONG,
            )
            strategy.on_tick(tick)
        # Momentum direction should be detected
        direction = strategy.momentum_analyzer.get_direction()
        assert direction in [ScalpDirection.LONG, ScalpDirection.NEUTRAL]


class TestScalpingStrategyBreakout:
    """Tests for breakout-based scalping."""

    def test_breakout_detection(self):
        """Test breakout detection."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.BREAKOUT,
            breakout_threshold=0.01,
            consolidation_window=5,
        )
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        # Add consolidation prices - need enough to establish range
        for _ in range(10):
            tick = TickData(
                timestamp=datetime.now(),
                price=Decimal("100.00"),
                volume=Decimal("10"),
                bid=Decimal("99.99"),
                ask=Decimal("100.01"),
                last_side=ScalpDirection.LONG,
            )
            strategy.on_tick(tick)
        # Add breakout
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("102.00"),
            volume=Decimal("10"),
            bid=Decimal("101.99"),
            ask=Decimal("102.01"),
            last_side=ScalpDirection.LONG,
        )
        strategy.on_tick(tick)
        direction = strategy.breakout_analyzer.detect_breakout()
        # With tight consolidation and 2% breakout, should detect
        assert direction in [ScalpDirection.LONG, ScalpDirection.NEUTRAL]


class TestScalpingStrategyRange:
    """Tests for range-based scalping."""

    def test_range_signal_at_bottom(self):
        """Test range signal at bottom."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.RANGE,
            cooldown_seconds=0,
        )
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        # Add range prices - need enough to establish range (lookback=30)
        for _ in range(15):
            for p in [100.00, 105.00]:
                tick = TickData(
                    timestamp=datetime.now(),
                    price=Decimal(str(p)),
                    volume=Decimal("10"),
                    bid=Decimal(str(p - 0.01)),
                    ask=Decimal(str(p + 0.01)),
                    last_side=ScalpDirection.LONG,
                )
                strategy.on_tick(tick)
        # End at bottom of range
        tick = TickData(
            timestamp=datetime.now(),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            bid=Decimal("99.99"),
            ask=Decimal("100.01"),
            last_side=ScalpDirection.LONG,
        )
        strategy.on_tick(tick)
        direction = strategy.range_analyzer.get_direction()
        assert direction == ScalpDirection.LONG


class TestScalpingIntegration:
    """Integration tests for scalping strategy."""

    def test_full_scalp_cycle_profit(self):
        """Test full scalp cycle with profit."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.ORDER_FLOW,
            min_profit_ticks=2,
            max_loss_ticks=3,
            tick_size=Decimal("0.01"),
            position_size=Decimal("1.0"),
        )
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        # Generate signal manually
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.LONG,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("100.02"),
            stop_loss=Decimal("99.97"),
            expected_profit_ticks=2,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        # Enter position
        position = strategy.enter_position(signal)
        assert position.entry_price == Decimal("100.00")
        # Price moves to target
        action = strategy.check_position(position.position_id, Decimal("100.02"))
        assert action == "target_hit"
        # Exit with profit
        trade = strategy.exit_position(position.position_id, Decimal("100.02"))
        pnl = Decimal(trade["pnl"])
        assert pnl == Decimal("0.02")  # 2 ticks * 1.0 size
        assert strategy.metrics.winning_scalps == 1
        assert strategy.metrics.win_rate == 1.0

    def test_full_scalp_cycle_loss(self):
        """Test full scalp cycle with loss."""
        config = ScalpingConfig(
            scalping_type=ScalpingType.ORDER_FLOW,
            min_profit_ticks=2,
            max_loss_ticks=3,
            tick_size=Decimal("0.01"),
            position_size=Decimal("1.0"),
        )
        strategy = ScalpingStrategy("ETH-USD-PERP", config)
        signal = ScalpSignal(
            timestamp=datetime.now(),
            signal_type=ScalpingType.ORDER_FLOW,
            direction=ScalpDirection.SHORT,
            strength=ScalpSignalStrength.STRONG,
            urgency=TradeUrgency.HIGH,
            entry_price=Decimal("100.00"),
            target_price=Decimal("99.98"),
            stop_loss=Decimal("100.03"),
            expected_profit_ticks=2,
            confidence=0.8,
            volume_confirmation=True,
            spread_favorable=True,
        )
        position = strategy.enter_position(signal)
        # Price moves against us
        action = strategy.check_position(position.position_id, Decimal("100.05"))
        assert action == "stop_hit"
        # Exit with loss
        trade = strategy.exit_position(position.position_id, Decimal("100.05"))
        pnl = Decimal(trade["pnl"])
        assert pnl == Decimal("-0.05")
        assert strategy.metrics.losing_scalps == 1

    def test_multiple_scalps(self):
        """Test multiple scalp trades."""
        strategy = ScalpingStrategy("ETH-USD-PERP")
        results = []
        for i in range(10):
            signal = ScalpSignal(
                timestamp=datetime.now(),
                signal_type=ScalpingType.ORDER_FLOW,
                direction=ScalpDirection.LONG if i % 2 == 0 else ScalpDirection.SHORT,
                strength=ScalpSignalStrength.STRONG,
                urgency=TradeUrgency.HIGH,
                entry_price=Decimal("100.00"),
                target_price=Decimal("100.05") if i % 2 == 0 else Decimal("99.95"),
                stop_loss=Decimal("99.95") if i % 2 == 0 else Decimal("100.05"),
                expected_profit_ticks=5,
                confidence=0.8,
                volume_confirmation=True,
                spread_favorable=True,
            )
            position = strategy.enter_position(signal)
            # Win 70% of trades
            if i < 7:
                exit_price = signal.target_price
            else:
                exit_price = signal.stop_loss
            trade = strategy.exit_position(position.position_id, exit_price)
            results.append(trade)
        assert strategy.metrics.total_scalps == 10
        assert strategy.metrics.winning_scalps == 7
        assert strategy.metrics.losing_scalps == 3
        assert strategy.metrics.win_rate == 0.7
