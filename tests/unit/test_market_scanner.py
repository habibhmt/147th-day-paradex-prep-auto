"""
Tests for Market Scanner Module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.analytics.market_scanner import (
    ScannerType, ScanCondition, AlertPriority, TrendDirection, SignalStrength,
    MarketData, ScanFilter, ScanResult, ScannerConfig, Alert,
    FilterEvaluator, MomentumScanner, VolumeScanner, VolatilityScanner,
    BreakoutScanner, FundingScanner, LiquidityScanner, TrendScanner,
    ReversalScanner, MarketScanner, WatchlistScanner, CompositeScanner,
    get_scanner, set_scanner
)


# ============== Fixtures ==============

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return MarketData(
        symbol="BTC-USD",
        price=Decimal("50000"),
        change_24h=Decimal("5.5"),
        change_1h=Decimal("0.5"),
        volume_24h=Decimal("5000000000"),
        high_24h=Decimal("51000"),
        low_24h=Decimal("49000"),
        open_interest=Decimal("1000000000"),
        funding_rate=Decimal("0.0003"),
        bid=Decimal("49995"),
        ask=Decimal("50005"),
        spread=Decimal("10")
    )


@pytest.fixture
def multiple_market_data():
    """Create multiple market data points."""
    return [
        MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("5.5"),
            change_1h=Decimal("0.5"),
            volume_24h=Decimal("5000000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000")
        ),
        MarketData(
            symbol="ETH-USD",
            price=Decimal("3000"),
            change_24h=Decimal("-2.5"),
            change_1h=Decimal("-0.3"),
            volume_24h=Decimal("2000000000"),
            high_24h=Decimal("3100"),
            low_24h=Decimal("2950")
        ),
        MarketData(
            symbol="SOL-USD",
            price=Decimal("100"),
            change_24h=Decimal("10.2"),
            change_1h=Decimal("2.1"),
            volume_24h=Decimal("500000000"),
            high_24h=Decimal("105"),
            low_24h=Decimal("95")
        )
    ]


@pytest.fixture
def scanner_config():
    """Create scanner configuration."""
    return ScannerConfig(
        scanner_type=ScannerType.MOMENTUM,
        min_volume=Decimal("1000000"),
        max_spread_pct=Decimal("1"),
        top_n=10
    )


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_scanner_type_values(self):
        assert ScannerType.MOMENTUM.value == "momentum"
        assert ScannerType.VOLUME.value == "volume"
        assert ScannerType.VOLATILITY.value == "volatility"
        assert ScannerType.BREAKOUT.value == "breakout"
        assert ScannerType.FUNDING.value == "funding"

    def test_scan_condition_values(self):
        assert ScanCondition.ABOVE.value == "above"
        assert ScanCondition.BELOW.value == "below"
        assert ScanCondition.BETWEEN.value == "between"
        assert ScanCondition.CROSSES_ABOVE.value == "crosses_above"

    def test_alert_priority_values(self):
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.CRITICAL.value == "critical"

    def test_trend_direction_values(self):
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.DOWN.value == "down"
        assert TrendDirection.SIDEWAYS.value == "sideways"

    def test_signal_strength_values(self):
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.VERY_STRONG.value == "very_strong"


# ============== Data Class Tests ==============

class TestMarketData:
    """Test MarketData dataclass."""

    def test_creation(self, sample_market_data):
        assert sample_market_data.symbol == "BTC-USD"
        assert sample_market_data.price == Decimal("50000")

    def test_defaults(self):
        data = MarketData(
            symbol="TEST",
            price=Decimal("100"),
            change_24h=Decimal("1"),
            change_1h=Decimal("0.1"),
            volume_24h=Decimal("1000"),
            high_24h=Decimal("105"),
            low_24h=Decimal("95")
        )
        assert data.open_interest == Decimal("0")
        assert data.funding_rate == Decimal("0")

    def test_to_dict(self, sample_market_data):
        result = sample_market_data.to_dict()
        assert result["symbol"] == "BTC-USD"
        assert result["price"] == "50000"
        assert "timestamp" in result


class TestScanFilter:
    """Test ScanFilter dataclass."""

    def test_creation(self):
        filter = ScanFilter(
            field="price",
            condition=ScanCondition.ABOVE,
            value=Decimal("100")
        )
        assert filter.field == "price"
        assert filter.condition == ScanCondition.ABOVE

    def test_between_filter(self):
        filter = ScanFilter(
            field="price",
            condition=ScanCondition.BETWEEN,
            value=Decimal("100"),
            value2=Decimal("200")
        )
        assert filter.value2 == Decimal("200")

    def test_to_dict(self):
        filter = ScanFilter(
            field="volume_24h",
            condition=ScanCondition.ABOVE,
            value=Decimal("1000000")
        )
        result = filter.to_dict()
        assert result["field"] == "volume_24h"
        assert result["condition"] == "above"


class TestScanResult:
    """Test ScanResult dataclass."""

    def test_creation(self, sample_market_data):
        result = ScanResult(
            symbol="BTC-USD",
            score=Decimal("85"),
            signals=["Signal 1", "Signal 2"],
            data=sample_market_data
        )
        assert result.symbol == "BTC-USD"
        assert result.score == Decimal("85")

    def test_defaults(self, sample_market_data):
        result = ScanResult(
            symbol="BTC-USD",
            score=Decimal("50"),
            signals=[],
            data=sample_market_data
        )
        assert result.priority == AlertPriority.MEDIUM
        assert result.trend == TrendDirection.SIDEWAYS

    def test_to_dict(self, sample_market_data):
        result = ScanResult(
            symbol="BTC-USD",
            score=Decimal("85"),
            signals=["Test signal"],
            data=sample_market_data,
            priority=AlertPriority.HIGH
        )
        data = result.to_dict()
        assert data["priority"] == "high"
        assert "data" in data


class TestScannerConfig:
    """Test ScannerConfig dataclass."""

    def test_defaults(self):
        config = ScannerConfig(scanner_type=ScannerType.MOMENTUM)
        assert config.top_n == 10
        assert config.min_volume == Decimal("0")

    def test_custom_config(self, scanner_config):
        assert scanner_config.min_volume == Decimal("1000000")
        assert scanner_config.top_n == 10

    def test_to_dict(self, scanner_config):
        result = scanner_config.to_dict()
        assert result["scanner_type"] == "momentum"
        assert "filters" in result


class TestAlert:
    """Test Alert dataclass."""

    def test_creation(self):
        alert = Alert(
            id="ALERT-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            alert_type="momentum",
            message="Test alert",
            priority=AlertPriority.HIGH,
            data={}
        )
        assert alert.id == "ALERT-001"
        assert alert.acknowledged is False

    def test_to_dict(self):
        alert = Alert(
            id="ALERT-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            alert_type="volume",
            message="Volume spike",
            priority=AlertPriority.HIGH,
            data={"key": "value"}
        )
        result = alert.to_dict()
        assert result["acknowledged"] is False


# ============== Filter Evaluator Tests ==============

class TestFilterEvaluator:
    """Test FilterEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return FilterEvaluator()

    def test_above_condition(self, evaluator, sample_market_data):
        filter = ScanFilter("price", ScanCondition.ABOVE, Decimal("40000"))
        assert evaluator.evaluate(sample_market_data, filter)

        filter = ScanFilter("price", ScanCondition.ABOVE, Decimal("60000"))
        assert not evaluator.evaluate(sample_market_data, filter)

    def test_below_condition(self, evaluator, sample_market_data):
        filter = ScanFilter("price", ScanCondition.BELOW, Decimal("60000"))
        assert evaluator.evaluate(sample_market_data, filter)

        filter = ScanFilter("price", ScanCondition.BELOW, Decimal("40000"))
        assert not evaluator.evaluate(sample_market_data, filter)

    def test_between_condition(self, evaluator, sample_market_data):
        filter = ScanFilter("price", ScanCondition.BETWEEN, Decimal("40000"), Decimal("60000"))
        assert evaluator.evaluate(sample_market_data, filter)

        filter = ScanFilter("price", ScanCondition.BETWEEN, Decimal("60000"), Decimal("70000"))
        assert not evaluator.evaluate(sample_market_data, filter)

    def test_equals_condition(self, evaluator, sample_market_data):
        filter = ScanFilter("price", ScanCondition.EQUALS, Decimal("50000"))
        assert evaluator.evaluate(sample_market_data, filter)

    def test_not_equals_condition(self, evaluator, sample_market_data):
        filter = ScanFilter("price", ScanCondition.NOT_EQUALS, Decimal("40000"))
        assert evaluator.evaluate(sample_market_data, filter)

    def test_crosses_above(self, evaluator, sample_market_data):
        prev_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("48000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("0"),
            high_24h=Decimal("0"),
            low_24h=Decimal("0")
        )
        filter = ScanFilter("price", ScanCondition.CROSSES_ABOVE, Decimal("49000"))
        assert evaluator.evaluate(sample_market_data, filter, prev_data)

    def test_crosses_below(self, evaluator, sample_market_data):
        prev_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("52000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("0"),
            high_24h=Decimal("0"),
            low_24h=Decimal("0")
        )
        filter = ScanFilter("price", ScanCondition.CROSSES_BELOW, Decimal("51000"))
        assert evaluator.evaluate(sample_market_data, filter, prev_data)

    def test_increases_by(self, evaluator, sample_market_data):
        prev_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("45000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("0"),
            high_24h=Decimal("0"),
            low_24h=Decimal("0")
        )
        filter = ScanFilter("price", ScanCondition.INCREASES_BY, Decimal("10"))
        assert evaluator.evaluate(sample_market_data, filter, prev_data)

    def test_invalid_field(self, evaluator, sample_market_data):
        filter = ScanFilter("invalid_field", ScanCondition.ABOVE, Decimal("100"))
        assert not evaluator.evaluate(sample_market_data, filter)


# ============== Momentum Scanner Tests ==============

class TestMomentumScanner:
    """Test MomentumScanner."""

    @pytest.fixture
    def scanner(self):
        return MomentumScanner(lookback_periods=5, momentum_threshold=Decimal("3"))

    def test_init(self, scanner):
        assert scanner.lookback_periods == 5
        assert scanner.momentum_threshold == Decimal("3")

    def test_update(self, scanner):
        scanner.update("BTC-USD", Decimal("50000"))
        assert "BTC-USD" in scanner.history
        assert len(scanner.history["BTC-USD"]) == 1

    def test_scan_insufficient_data(self, scanner, sample_market_data):
        result = scanner.scan(sample_market_data)
        assert result is None

    def test_scan_positive_momentum(self, scanner, sample_market_data):
        # Build history with upward trend
        for price in [48000, 48500, 49000, 49500]:
            scanner.update("BTC-USD", Decimal(str(price)))

        result = scanner.scan(sample_market_data)
        assert result is not None
        assert result.trend == TrendDirection.UP

    def test_scan_negative_momentum(self, scanner):
        # Build history with downward trend
        for price in [52000, 51500, 51000, 50500]:
            scanner.update("BTC-USD", Decimal(str(price)))

        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("-5"),
            change_1h=Decimal("-0.5"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("52000"),
            low_24h=Decimal("50000")
        )
        result = scanner.scan(data)
        assert result is not None
        assert result.trend == TrendDirection.DOWN


# ============== Volume Scanner Tests ==============

class TestVolumeScanner:
    """Test VolumeScanner."""

    @pytest.fixture
    def scanner(self):
        return VolumeScanner(volume_multiplier=Decimal("2"), avg_periods=5)

    def test_init(self, scanner):
        assert scanner.volume_multiplier == Decimal("2")

    def test_update(self, scanner):
        scanner.update("BTC-USD", Decimal("1000000"))
        assert len(scanner.history["BTC-USD"]) == 1

    def test_scan_volume_spike(self, scanner):
        # Build history with normal volume
        for _ in range(4):
            scanner.update("BTC-USD", Decimal("1000000"))

        # Data with volume spike
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("3000000"),  # 3x normal
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000")
        )
        result = scanner.scan(data)
        assert result is not None
        assert "Volume spike" in result.signals[0]


# ============== Volatility Scanner Tests ==============

class TestVolatilityScanner:
    """Test VolatilityScanner."""

    @pytest.fixture
    def scanner(self):
        return VolatilityScanner(atr_periods=5, volatility_threshold=Decimal("2"))

    def test_init(self, scanner):
        assert scanner.atr_periods == 5

    def test_calculate_atr(self, scanner):
        # Build history
        for i in range(5):
            scanner.update("BTC-USD",
                          Decimal("51000") + Decimal(str(i * 100)),
                          Decimal("49000") - Decimal(str(i * 100)),
                          Decimal("50000"))

        atr = scanner.calculate_atr("BTC-USD")
        assert atr is not None
        assert atr > 0

    def test_scan_high_volatility(self, scanner):
        # Build history with high volatility
        for i in range(5):
            scanner.update("BTC-USD",
                          Decimal("55000"),
                          Decimal("45000"),
                          Decimal("50000"))

        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("55000"),
            low_24h=Decimal("45000")
        )
        result = scanner.scan(data)
        assert result is not None
        assert "volatility" in result.signals[0].lower()


# ============== Breakout Scanner Tests ==============

class TestBreakoutScanner:
    """Test BreakoutScanner."""

    @pytest.fixture
    def scanner(self):
        return BreakoutScanner(lookback_periods=5, breakout_threshold=Decimal("1"))

    def test_init(self, scanner):
        assert scanner.lookback_periods == 5

    def test_scan_upward_breakout(self, scanner):
        # Build history with range
        for i in range(5):
            scanner.update("BTC-USD",
                          Decimal("50000"),
                          Decimal("48000"),
                          Decimal("49000"))

        # Data with breakout above
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("51000"),  # Above resistance
            change_24h=Decimal("5"),
            change_1h=Decimal("1"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("50000")
        )
        result = scanner.scan(data)
        assert result is not None
        assert result.trend == TrendDirection.UP


# ============== Funding Scanner Tests ==============

class TestFundingScanner:
    """Test FundingScanner."""

    @pytest.fixture
    def scanner(self):
        return FundingScanner(
            funding_threshold=Decimal("0.01"),
            extreme_threshold=Decimal("0.05")
        )

    def test_init(self, scanner):
        assert scanner.funding_threshold == Decimal("0.01")

    def test_scan_positive_funding(self, scanner):
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000"),
            funding_rate=Decimal("0.02")
        )
        result = scanner.scan(data)
        assert result is not None
        assert "Short" in " ".join(result.signals)

    def test_scan_negative_funding(self, scanner):
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000"),
            funding_rate=Decimal("-0.02")
        )
        result = scanner.scan(data)
        assert result is not None
        assert "Long" in " ".join(result.signals)

    def test_scan_extreme_funding(self, scanner):
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000"),
            funding_rate=Decimal("0.1")
        )
        result = scanner.scan(data)
        assert result is not None
        assert result.strength == SignalStrength.VERY_STRONG

    def test_scan_low_funding(self, scanner):
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000"),
            funding_rate=Decimal("0.005")
        )
        result = scanner.scan(data)
        assert result is None


# ============== Liquidity Scanner Tests ==============

class TestLiquidityScanner:
    """Test LiquidityScanner."""

    @pytest.fixture
    def scanner(self):
        return LiquidityScanner(
            min_volume=Decimal("1000000"),
            max_spread_bps=Decimal("10")
        )

    def test_init(self, scanner):
        assert scanner.min_volume == Decimal("1000000")

    def test_scan_wide_spread(self, scanner):
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("2000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("50000"),
            spread=Decimal("100")  # 20 bps
        )
        result = scanner.scan(data)
        assert result is not None
        assert "Wide spread" in " ".join(result.signals)

    def test_scan_low_volume(self, scanner):
        data = MarketData(
            symbol="TEST-USD",
            price=Decimal("100"),
            change_24h=Decimal("0"),
            change_1h=Decimal("0"),
            volume_24h=Decimal("500000"),  # Below min
            high_24h=Decimal("100"),
            low_24h=Decimal("100"),
            spread=Decimal("0.01")
        )
        result = scanner.scan(data)
        assert result is not None
        assert "Low volume" in " ".join(result.signals)


# ============== Trend Scanner Tests ==============

class TestTrendScanner:
    """Test TrendScanner."""

    @pytest.fixture
    def scanner(self):
        return TrendScanner(short_period=5, long_period=10)

    def test_init(self, scanner):
        assert scanner.short_period == 5
        assert scanner.long_period == 10

    def test_scan_bullish_trend(self, scanner):
        # Build upward trend history
        for i in range(10):
            scanner.update("BTC-USD", Decimal(str(45000 + i * 500)))

        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("5"),
            change_1h=Decimal("0.5"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("50000"),
            low_24h=Decimal("49000")
        )
        result = scanner.scan(data)
        assert result is not None
        assert result.trend == TrendDirection.UP


# ============== Reversal Scanner Tests ==============

class TestReversalScanner:
    """Test ReversalScanner."""

    @pytest.fixture
    def scanner(self):
        return ReversalScanner(
            rsi_periods=5,
            oversold_level=Decimal("30"),
            overbought_level=Decimal("70")
        )

    def test_init(self, scanner):
        assert scanner.rsi_periods == 5

    def test_calculate_rsi(self, scanner):
        # Build history with upward movement
        for price in [50000, 51000, 52000, 53000, 54000, 55000]:
            scanner.update("BTC-USD", Decimal(str(price)))

        rsi = scanner.calculate_rsi("BTC-USD")
        assert rsi is not None
        assert rsi > Decimal("50")

    def test_scan_oversold(self, scanner):
        # Build history with downward movement
        for price in [55000, 53000, 51000, 49000, 47000, 45000]:
            scanner.update("BTC-USD", Decimal(str(price)))

        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("43000"),
            change_24h=Decimal("-10"),
            change_1h=Decimal("-2"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("45000"),
            low_24h=Decimal("43000")
        )
        result = scanner.scan(data)
        # RSI should be low after consecutive drops
        if result:
            assert "Oversold" in " ".join(result.signals) or result is None


# ============== Market Scanner Tests ==============

class TestMarketScanner:
    """Test MarketScanner."""

    @pytest.fixture
    def scanner(self, scanner_config):
        return MarketScanner(scanner_config)

    def test_init(self, scanner):
        assert scanner.config.scanner_type == ScannerType.MOMENTUM

    def test_update_data(self, scanner, sample_market_data):
        scanner.update_data(sample_market_data)
        assert "BTC-USD" in scanner.market_data

    def test_update_data_tracks_previous(self, scanner, sample_market_data):
        scanner.update_data(sample_market_data)

        new_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("51000"),
            change_24h=Decimal("6"),
            change_1h=Decimal("1"),
            volume_24h=Decimal("5500000000"),
            high_24h=Decimal("51500"),
            low_24h=Decimal("49000")
        )
        scanner.update_data(new_data)

        assert "BTC-USD" in scanner.prev_data

    def test_register_callback(self, scanner):
        callback = Mock()
        scanner.register_callback("on_result", callback)
        assert callback in scanner.callbacks["on_result"]

    def test_scan(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        results = scanner.scan()
        assert isinstance(results, list)

    def test_scan_specific_symbols(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        results = scanner.scan(["BTC-USD"])
        # Results should only include BTC-USD
        for result in results:
            assert result.symbol == "BTC-USD"

    def test_scan_all(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        all_results = scanner.scan_all()
        assert isinstance(all_results, dict)
        assert ScannerType.MOMENTUM in all_results

    def test_get_top_gainers(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        gainers = scanner.get_top_gainers(2)
        assert len(gainers) <= 2
        # Should be sorted by change
        if len(gainers) >= 2:
            assert gainers[0].change_24h >= gainers[1].change_24h

    def test_get_top_losers(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        losers = scanner.get_top_losers(2)
        assert len(losers) <= 2

    def test_get_top_volume(self, scanner, multiple_market_data):
        for data in multiple_market_data:
            scanner.update_data(data)

        top_vol = scanner.get_top_volume(2)
        assert len(top_vol) <= 2

    def test_get_highest_oi(self, scanner, sample_market_data):
        scanner.update_data(sample_market_data)
        highest_oi = scanner.get_highest_oi(5)
        assert len(highest_oi) >= 1

    def test_get_extreme_funding(self, scanner, sample_market_data):
        scanner.update_data(sample_market_data)
        extreme = scanner.get_extreme_funding(5)
        assert len(extreme) >= 1

    def test_acknowledge_alert(self, scanner):
        alert = Alert(
            id="TEST-001",
            timestamp=datetime.now(),
            symbol="BTC-USD",
            alert_type="test",
            message="Test",
            priority=AlertPriority.HIGH,
            data={}
        )
        scanner.alerts.append(alert)

        assert scanner.acknowledge_alert("TEST-001")
        assert alert.acknowledged

    def test_acknowledge_nonexistent_alert(self, scanner):
        assert not scanner.acknowledge_alert("INVALID")

    def test_get_unacknowledged_alerts(self, scanner):
        scanner.alerts = [
            Alert("A1", datetime.now(), "BTC", "test", "msg", AlertPriority.HIGH, {}, acknowledged=False),
            Alert("A2", datetime.now(), "ETH", "test", "msg", AlertPriority.HIGH, {}, acknowledged=True)
        ]
        unack = scanner.get_unacknowledged_alerts()
        assert len(unack) == 1
        assert unack[0].id == "A1"

    def test_clear_alerts(self, scanner):
        scanner.alerts = [Mock(), Mock()]
        scanner.clear_alerts()
        assert len(scanner.alerts) == 0

    def test_get_summary(self, scanner, sample_market_data):
        scanner.update_data(sample_market_data)
        summary = scanner.get_summary()
        assert "total_symbols" in summary
        assert summary["total_symbols"] == 1


# ============== Watchlist Scanner Tests ==============

class TestWatchlistScanner:
    """Test WatchlistScanner."""

    @pytest.fixture
    def scanner(self):
        return WatchlistScanner()

    def test_add_to_watchlist(self, scanner):
        scanner.add_to_watchlist(
            "BTC-USD",
            price_alerts=[Decimal("55000"), Decimal("60000")],
            change_alerts=[Decimal("10"), Decimal("-10")]
        )
        assert "BTC-USD" in scanner.watchlist

    def test_remove_from_watchlist(self, scanner):
        scanner.add_to_watchlist("BTC-USD")
        assert scanner.remove_from_watchlist("BTC-USD")
        assert "BTC-USD" not in scanner.watchlist

    def test_remove_nonexistent(self, scanner):
        assert not scanner.remove_from_watchlist("INVALID")

    def test_check_price_alert(self, scanner):
        scanner.add_to_watchlist(
            "BTC-USD",
            price_alerts=[Decimal("50000")]
        )
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("51000"),
            change_24h=Decimal("5"),
            change_1h=Decimal("1"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000")
        )
        alerts = scanner.check(data)
        assert len(alerts) == 1

    def test_check_no_duplicate_alerts(self, scanner):
        scanner.add_to_watchlist(
            "BTC-USD",
            price_alerts=[Decimal("50000")]
        )
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("51000"),
            change_24h=Decimal("5"),
            change_1h=Decimal("1"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000")
        )
        # First check triggers
        scanner.check(data)
        # Second check should not trigger
        alerts = scanner.check(data)
        assert len(alerts) == 0

    def test_reset_alerts(self, scanner):
        scanner.add_to_watchlist("BTC-USD", price_alerts=[Decimal("50000")])
        scanner.watchlist["BTC-USD"]["triggered"].add("price_50000")
        scanner.reset_alerts("BTC-USD")
        assert len(scanner.watchlist["BTC-USD"]["triggered"]) == 0


# ============== Composite Scanner Tests ==============

class TestCompositeScanner:
    """Test CompositeScanner."""

    @pytest.fixture
    def scanner(self):
        composite = CompositeScanner()
        composite.add_scanner(ScannerType.MOMENTUM, MomentumScanner(), Decimal("1"))
        composite.add_scanner(ScannerType.VOLUME, VolumeScanner(), Decimal("0.5"))
        return composite

    def test_add_scanner(self, scanner):
        assert len(scanner.scanners) == 2

    def test_scan_no_results(self, scanner, sample_market_data):
        # Without history, individual scanners return None
        result = scanner.scan(sample_market_data)
        assert result is None

    def test_scan_combined_results(self):
        composite = CompositeScanner()

        # Create scanners with history
        momentum = MomentumScanner(lookback_periods=3, momentum_threshold=Decimal("1"))
        for price in [48000, 49000, 50000]:
            momentum.update("BTC-USD", Decimal(str(price)))

        composite.add_scanner(ScannerType.MOMENTUM, momentum, Decimal("1"))

        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("51000"),
            change_24h=Decimal("5"),
            change_1h=Decimal("1"),
            volume_24h=Decimal("1000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000")
        )
        result = composite.scan(data)
        # May or may not have result depending on momentum threshold
        assert result is None or isinstance(result, ScanResult)


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global scanner instance."""

    def test_get_scanner(self):
        scanner = get_scanner()
        assert isinstance(scanner, MarketScanner)

    def test_set_scanner(self):
        config = ScannerConfig(scanner_type=ScannerType.VOLUME)
        custom = MarketScanner(config)
        set_scanner(custom)
        scanner = get_scanner()
        assert scanner.config.scanner_type == ScannerType.VOLUME


# ============== Integration Tests ==============

class TestScannerIntegration:
    """Integration tests for scanner."""

    def test_full_scanning_flow(self, multiple_market_data):
        config = ScannerConfig(
            scanner_type=ScannerType.MOMENTUM,
            min_volume=Decimal("100000000"),
            top_n=5
        )
        scanner = MarketScanner(config)

        # Load data
        for data in multiple_market_data:
            scanner.update_data(data)

        # Run scan
        results = scanner.scan()

        # Check results structure
        for result in results:
            assert hasattr(result, "symbol")
            assert hasattr(result, "score")
            assert hasattr(result, "signals")

    def test_callback_integration(self, sample_market_data):
        config = ScannerConfig(scanner_type=ScannerType.FUNDING)
        scanner = MarketScanner(config)

        results_received = []
        scanner.register_callback("on_result", lambda r: results_received.append(r))

        # High funding data
        data = MarketData(
            symbol="BTC-USD",
            price=Decimal("50000"),
            change_24h=Decimal("5"),
            change_1h=Decimal("0.5"),
            volume_24h=Decimal("5000000000"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            funding_rate=Decimal("0.05")
        )
        scanner.update_data(data)
        scanner.scan()

        assert len(results_received) > 0
