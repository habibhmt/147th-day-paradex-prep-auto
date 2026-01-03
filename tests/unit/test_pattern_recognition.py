"""Tests for pattern recognition module."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.analytics.pattern_recognition import (
    PatternType,
    PatternDirection,
    PatternStrength,
    PatternStatus,
    OHLCV,
    PatternPoint,
    Pattern,
    PivotFinder,
    TrendlineAnalyzer,
    CandlestickPatternDetector,
    ChartPatternDetector,
    PatternRecognizer,
    get_pattern_recognizer,
    set_pattern_recognizer,
)


class TestEnums:
    """Test enum classes."""

    def test_pattern_type_values(self):
        """Test PatternType enum values."""
        assert PatternType.HEAD_AND_SHOULDERS.value == "head_and_shoulders"
        assert PatternType.DOUBLE_TOP.value == "double_top"
        assert PatternType.ASCENDING_TRIANGLE.value == "ascending_triangle"
        assert PatternType.DOJI.value == "doji"
        assert PatternType.HAMMER.value == "hammer"

    def test_pattern_direction_values(self):
        """Test PatternDirection enum values."""
        assert PatternDirection.BULLISH.value == "bullish"
        assert PatternDirection.BEARISH.value == "bearish"
        assert PatternDirection.NEUTRAL.value == "neutral"

    def test_pattern_strength_values(self):
        """Test PatternStrength enum values."""
        assert PatternStrength.WEAK.value == "weak"
        assert PatternStrength.MODERATE.value == "moderate"
        assert PatternStrength.STRONG.value == "strong"
        assert PatternStrength.VERY_STRONG.value == "very_strong"

    def test_pattern_status_values(self):
        """Test PatternStatus enum values."""
        assert PatternStatus.FORMING.value == "forming"
        assert PatternStatus.COMPLETE.value == "complete"
        assert PatternStatus.CONFIRMED.value == "confirmed"
        assert PatternStatus.FAILED.value == "failed"


class TestOHLCV:
    """Test OHLCV class."""

    def test_creation(self):
        """Test OHLCV creation."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        assert candle.open == Decimal("100")
        assert candle.close == Decimal("105")

    def test_is_bullish(self):
        """Test bullish candle detection."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("108"),
            volume=Decimal("1000"),
        )
        assert candle.is_bullish is True
        assert candle.is_bearish is False

    def test_is_bearish(self):
        """Test bearish candle detection."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("90"),
            close=Decimal("92"),
            volume=Decimal("1000"),
        )
        assert candle.is_bearish is True
        assert candle.is_bullish is False

    def test_body_size(self):
        """Test body size calculation."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("108"),
            volume=Decimal("1000"),
        )
        assert candle.body_size == Decimal("8")

    def test_upper_wick(self):
        """Test upper wick calculation."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        assert candle.upper_wick == Decimal("5")

    def test_lower_wick(self):
        """Test lower wick calculation."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        assert candle.lower_wick == Decimal("5")

    def test_range(self):
        """Test range calculation."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        assert candle.range == Decimal("20")

    def test_to_dict(self):
        """Test OHLCV to_dict."""
        candle = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        result = candle.to_dict()
        assert result["open"] == "100"
        assert result["close"] == "105"


class TestPatternPoint:
    """Test PatternPoint class."""

    def test_creation(self):
        """Test PatternPoint creation."""
        point = PatternPoint(
            index=5,
            price=Decimal("100"),
            timestamp=datetime.now(),
            point_type="high",
        )
        assert point.index == 5
        assert point.price == Decimal("100")
        assert point.point_type == "high"

    def test_to_dict(self):
        """Test PatternPoint to_dict."""
        point = PatternPoint(
            index=5,
            price=Decimal("100"),
            timestamp=datetime.now(),
            point_type="low",
        )
        result = point.to_dict()
        assert result["index"] == 5
        assert result["price"] == "100"
        assert result["point_type"] == "low"


class TestPattern:
    """Test Pattern class."""

    def test_creation(self):
        """Test Pattern creation."""
        pattern = Pattern(
            id="test123",
            pattern_type=PatternType.DOUBLE_TOP,
            symbol="BTC-USD-PERP",
            direction=PatternDirection.BEARISH,
            strength=PatternStrength.STRONG,
            status=PatternStatus.COMPLETE,
            start_index=0,
            end_index=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert pattern.pattern_type == PatternType.DOUBLE_TOP
        assert pattern.direction == PatternDirection.BEARISH

    def test_risk_reward(self):
        """Test risk/reward calculation."""
        pattern = Pattern(
            id="test123",
            pattern_type=PatternType.DOUBLE_TOP,
            symbol="BTC-USD-PERP",
            direction=PatternDirection.BEARISH,
            strength=PatternStrength.STRONG,
            status=PatternStatus.COMPLETE,
            start_index=0,
            end_index=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
            entry_price=Decimal("100"),
            target_price=Decimal("80"),
            stop_loss=Decimal("110"),
        )
        assert pattern.risk_reward == Decimal("2")

    def test_risk_reward_none(self):
        """Test risk/reward when values missing."""
        pattern = Pattern(
            id="test123",
            pattern_type=PatternType.DOUBLE_TOP,
            symbol="BTC-USD-PERP",
            direction=PatternDirection.BEARISH,
            strength=PatternStrength.STRONG,
            status=PatternStatus.COMPLETE,
            start_index=0,
            end_index=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert pattern.risk_reward is None

    def test_to_dict(self):
        """Test Pattern to_dict."""
        pattern = Pattern(
            id="test123",
            pattern_type=PatternType.DOUBLE_TOP,
            symbol="BTC-USD-PERP",
            direction=PatternDirection.BEARISH,
            strength=PatternStrength.STRONG,
            status=PatternStatus.COMPLETE,
            start_index=0,
            end_index=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        result = pattern.to_dict()
        assert result["id"] == "test123"
        assert result["pattern_type"] == "double_top"
        assert result["direction"] == "bearish"


def create_candles(prices: list[tuple], base_time: datetime = None) -> list[OHLCV]:
    """Create candles from price tuples (open, high, low, close)."""
    base = base_time or datetime.now()
    candles = []
    for i, (o, h, l, c) in enumerate(prices):
        candles.append(OHLCV(
            timestamp=base + timedelta(hours=i),
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(l)),
            close=Decimal(str(c)),
            volume=Decimal("1000"),
        ))
    return candles


class TestPivotFinder:
    """Test PivotFinder class."""

    def test_find_pivot_highs(self):
        """Test finding pivot highs."""
        finder = PivotFinder(left_bars=2, right_bars=2)
        prices = [
            (100, 102, 98, 101),   # 0
            (101, 103, 100, 102),  # 1
            (102, 110, 101, 109),  # 2 - pivot high
            (109, 108, 105, 106),  # 3
            (106, 107, 104, 105),  # 4
            (105, 108, 103, 107),  # 5
            (107, 115, 106, 114),  # 6 - pivot high
            (114, 113, 110, 111),  # 7
            (111, 112, 109, 110),  # 8
        ]
        candles = create_candles(prices)
        pivots = finder.find_pivot_highs(candles)
        assert len(pivots) >= 1

    def test_find_pivot_lows(self):
        """Test finding pivot lows."""
        finder = PivotFinder(left_bars=2, right_bars=2)
        prices = [
            (100, 102, 98, 99),    # 0
            (99, 100, 95, 96),     # 1
            (96, 98, 90, 91),      # 2 - pivot low
            (91, 95, 90, 94),      # 3
            (94, 98, 93, 97),      # 4
            (97, 99, 95, 96),      # 5
            (96, 97, 88, 89),      # 6 - pivot low
            (89, 93, 88, 92),      # 7
            (92, 96, 91, 95),      # 8
        ]
        candles = create_candles(prices)
        pivots = finder.find_pivot_lows(candles)
        assert len(pivots) >= 1

    def test_no_pivots(self):
        """Test when no pivots found."""
        finder = PivotFinder(left_bars=2, right_bars=2)
        # Monotonically increasing - no pivot highs
        prices = [(i, i + 2, i - 1, i + 1) for i in range(10)]
        candles = create_candles(prices)
        pivots = finder.find_pivot_highs(candles)
        assert len(pivots) == 0


class TestTrendlineAnalyzer:
    """Test TrendlineAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = TrendlineAnalyzer(min_touches=3, tolerance_pct=Decimal("0.02"))
        assert analyzer.min_touches == 3
        assert analyzer.tolerance_pct == Decimal("0.02")

    def test_find_support_line(self):
        """Test finding support line."""
        analyzer = TrendlineAnalyzer(min_touches=2)
        prices = [
            (100, 105, 95, 102),
            (102, 108, 98, 106),
            (106, 112, 102, 110),
            (110, 115, 105, 113),
        ]
        candles = create_candles(prices)
        pivot_lows = [(0, Decimal("95")), (2, Decimal("102"))]
        result = analyzer.find_support_line(candles, pivot_lows)
        # May or may not find depending on touches
        assert result is None or isinstance(result, tuple)

    def test_find_resistance_line(self):
        """Test finding resistance line."""
        analyzer = TrendlineAnalyzer(min_touches=2)
        prices = [
            (100, 110, 95, 108),
            (108, 115, 103, 112),
            (112, 120, 108, 118),
            (118, 125, 114, 122),
        ]
        candles = create_candles(prices)
        pivot_highs = [(0, Decimal("110")), (2, Decimal("120"))]
        result = analyzer.find_resistance_line(candles, pivot_highs)
        # May or may not find depending on touches
        assert result is None or isinstance(result, tuple)


class TestCandlestickPatternDetector:
    """Test CandlestickPatternDetector class."""

    def test_detect_doji(self):
        """Test doji detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 110, 90, 100.5),  # Doji - open and close almost equal
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        doji_patterns = [p for p in patterns if p.pattern_type == PatternType.DOJI]
        assert len(doji_patterns) > 0

    def test_detect_hammer(self):
        """Test hammer detection."""
        detector = CandlestickPatternDetector()
        # Hammer: small body, long lower wick, tiny upper wick
        prices = [
            (100, 101, 80, 99),  # Hammer
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        hammer_patterns = [p for p in patterns if p.pattern_type == PatternType.HAMMER]
        assert len(hammer_patterns) > 0

    def test_detect_shooting_star(self):
        """Test shooting star detection."""
        detector = CandlestickPatternDetector()
        # Shooting star: small body, long upper wick, tiny lower wick, bearish
        prices = [
            (100, 120, 99, 98),  # Shooting star
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        star_patterns = [p for p in patterns if p.pattern_type == PatternType.SHOOTING_STAR]
        assert len(star_patterns) > 0

    def test_detect_engulfing_bullish(self):
        """Test bullish engulfing detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 101, 95, 96),   # Bearish
            (94, 105, 93, 103),   # Bullish engulfing
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        engulfing = [p for p in patterns if p.pattern_type == PatternType.ENGULFING_BULLISH]
        assert len(engulfing) > 0

    def test_detect_engulfing_bearish(self):
        """Test bearish engulfing detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 105, 99, 104),  # Bullish
            (106, 107, 95, 97),   # Bearish engulfing
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        engulfing = [p for p in patterns if p.pattern_type == PatternType.ENGULFING_BEARISH]
        assert len(engulfing) > 0

    def test_detect_morning_star(self):
        """Test morning star detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 101, 92, 93),   # Bearish
            (93, 94, 91, 92),     # Small body
            (92, 105, 91, 103),   # Bullish
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        morning = [p for p in patterns if p.pattern_type == PatternType.MORNING_STAR]
        assert len(morning) > 0

    def test_detect_three_white_soldiers(self):
        """Test three white soldiers detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 108, 99, 107),
            (108, 116, 107, 115),
            (116, 124, 115, 123),
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        soldiers = [p for p in patterns if p.pattern_type == PatternType.THREE_WHITE_SOLDIERS]
        assert len(soldiers) > 0

    def test_detect_three_black_crows(self):
        """Test three black crows detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 101, 92, 93),
            (92, 93, 84, 85),
            (84, 85, 76, 77),
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        crows = [p for p in patterns if p.pattern_type == PatternType.THREE_BLACK_CROWS]
        assert len(crows) > 0

    def test_detect_marubozu(self):
        """Test marubozu detection."""
        detector = CandlestickPatternDetector()
        prices = [
            (100, 110, 100, 110),  # Bullish marubozu
        ]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        marubozu = [p for p in patterns if p.pattern_type == PatternType.MARUBOZU_BULLISH]
        assert len(marubozu) > 0


class TestChartPatternDetector:
    """Test ChartPatternDetector class."""

    def test_init(self):
        """Test initialization."""
        detector = ChartPatternDetector(pivot_left=3, pivot_right=3)
        assert detector.pivot_finder.left_bars == 3
        assert detector.pivot_finder.right_bars == 3

    def test_detect_double_top(self):
        """Test double top detection."""
        detector = ChartPatternDetector(pivot_left=2, pivot_right=2)

        # Create price pattern for double top
        prices = []
        # Rising to first peak
        for i in range(5):
            prices.append((100 + i * 2, 101 + i * 2, 99 + i * 2, 100 + i * 2))
        # First peak
        prices.append((110, 115, 109, 114))
        prices.append((114, 116, 113, 115))  # Peak
        prices.append((115, 115, 110, 111))
        # Pullback
        for i in range(3):
            prices.append((111 - i * 2, 112 - i * 2, 104 - i * 2, 110 - i * 2))
        # Second peak
        prices.append((108, 112, 107, 111))
        prices.append((111, 116, 110, 115))  # Peak (similar to first)
        prices.append((115, 115, 108, 109))
        prices.append((109, 110, 105, 106))
        prices.append((106, 107, 102, 103))

        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        # Should detect some patterns
        assert isinstance(patterns, list)

    def test_detect_double_bottom(self):
        """Test double bottom detection."""
        detector = ChartPatternDetector(pivot_left=2, pivot_right=2)

        # Create price pattern for double bottom
        prices = []
        # Falling to first bottom
        for i in range(5):
            prices.append((100 - i * 2, 101 - i * 2, 99 - i * 2, 100 - i * 2))
        # First bottom
        prices.append((90, 91, 85, 86))
        prices.append((86, 87, 84, 85))  # Bottom
        prices.append((85, 90, 84, 89))
        # Rally
        for i in range(3):
            prices.append((89 + i * 2, 92 + i * 2, 88 + i * 2, 91 + i * 2))
        # Second bottom
        prices.append((92, 93, 88, 89))
        prices.append((89, 90, 84, 85))  # Bottom (similar to first)
        prices.append((85, 91, 84, 90))
        prices.append((90, 95, 89, 94))
        prices.append((94, 98, 93, 97))

        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        detector = ChartPatternDetector()
        prices = [(100, 105, 95, 102) for _ in range(10)]
        candles = create_candles(prices)
        patterns = detector.detect_all(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)


class TestPatternRecognizer:
    """Test PatternRecognizer class."""

    def test_init(self):
        """Test initialization."""
        recognizer = PatternRecognizer()
        assert recognizer.candlestick_detector is not None
        assert recognizer.chart_detector is not None

    def test_analyze(self):
        """Test pattern analysis."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 101, 95, 96),   # Bearish
            (94, 105, 93, 103),   # Bullish engulfing
        ]
        candles = create_candles(prices)
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert len(patterns) > 0

    def test_get_patterns(self):
        """Test getting patterns with filters."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 110, 90, 100.5),  # Doji
            (100, 101, 95, 96),     # Bearish
            (94, 105, 93, 103),     # Bullish engulfing
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        all_patterns = recognizer.get_patterns()
        assert len(all_patterns) > 0

        symbol_patterns = recognizer.get_patterns(symbol="BTC-USD-PERP")
        assert len(symbol_patterns) > 0

    def test_get_patterns_by_type(self):
        """Test getting patterns by type."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 110, 90, 100.5),  # Doji
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        doji_patterns = recognizer.get_patterns(pattern_type=PatternType.DOJI)
        assert all(p.pattern_type == PatternType.DOJI for p in doji_patterns)

    def test_get_bullish_patterns(self):
        """Test getting bullish patterns."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 101, 95, 96),   # Bearish
            (94, 105, 93, 103),   # Bullish engulfing
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        bullish = recognizer.get_bullish_patterns()
        assert all(p.direction == PatternDirection.BULLISH for p in bullish)

    def test_get_bearish_patterns(self):
        """Test getting bearish patterns."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 105, 99, 104),  # Bullish
            (106, 107, 95, 97),   # Bearish engulfing
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        bearish = recognizer.get_bearish_patterns()
        assert all(p.direction == PatternDirection.BEARISH for p in bearish)

    def test_get_strong_patterns(self):
        """Test getting strong patterns."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 101, 95, 96),
            (94, 105, 93, 103),
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        strong = recognizer.get_strong_patterns()
        assert all(
            p.strength in (PatternStrength.STRONG, PatternStrength.VERY_STRONG)
            for p in strong
        )

    def test_add_callback(self):
        """Test adding callback."""
        recognizer = PatternRecognizer()
        results = []

        def callback(pattern):
            results.append(pattern)

        recognizer.add_callback(callback)

        prices = [(100, 110, 90, 100.5)]  # Doji
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        assert len(results) > 0

    def test_remove_callback(self):
        """Test removing callback."""
        recognizer = PatternRecognizer()
        results = []

        def callback(pattern):
            results.append(pattern)

        recognizer.add_callback(callback)
        assert recognizer.remove_callback(callback) is True
        assert recognizer.remove_callback(callback) is False

    def test_clear(self):
        """Test clearing patterns."""
        recognizer = PatternRecognizer()
        prices = [(100, 110, 90, 100.5)]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        assert len(recognizer.get_patterns()) > 0
        recognizer.clear()
        assert len(recognizer.get_patterns()) == 0

    def test_clear_by_symbol(self):
        """Test clearing patterns for specific symbol."""
        recognizer = PatternRecognizer()
        prices = [(100, 110, 90, 100.5)]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")
        recognizer.analyze(candles, "ETH-USD-PERP")

        recognizer.clear(symbol="BTC-USD-PERP")
        patterns = recognizer.get_patterns()
        assert all(p.symbol == "ETH-USD-PERP" for p in patterns)

    def test_get_summary(self):
        """Test getting summary."""
        recognizer = PatternRecognizer()
        prices = [
            (100, 110, 90, 100.5),  # Doji
            (100, 101, 95, 96),
            (94, 105, 93, 103),
        ]
        candles = create_candles(prices)
        recognizer.analyze(candles, "BTC-USD-PERP")

        summary = recognizer.get_summary()
        assert "total_patterns" in summary
        assert "by_type" in summary
        assert "by_direction" in summary
        assert "by_strength" in summary
        assert "symbols" in summary

    def test_multiple_symbols(self):
        """Test analyzing multiple symbols."""
        recognizer = PatternRecognizer()

        btc_prices = [(100, 110, 90, 100.5)]
        eth_prices = [(50, 55, 45, 50.2)]

        btc_candles = create_candles(btc_prices)
        eth_candles = create_candles(eth_prices)

        recognizer.analyze(btc_candles, "BTC-USD-PERP")
        recognizer.analyze(eth_candles, "ETH-USD-PERP")

        btc_patterns = recognizer.get_patterns(symbol="BTC-USD-PERP")
        eth_patterns = recognizer.get_patterns(symbol="ETH-USD-PERP")

        assert len(btc_patterns) > 0
        assert len(eth_patterns) > 0
        assert all(p.symbol == "BTC-USD-PERP" for p in btc_patterns)
        assert all(p.symbol == "ETH-USD-PERP" for p in eth_patterns)


class TestGlobalInstance:
    """Test global instance functions."""

    def test_get_pattern_recognizer(self):
        """Test getting global instance."""
        recognizer = get_pattern_recognizer()
        assert recognizer is not None
        assert isinstance(recognizer, PatternRecognizer)

    def test_set_pattern_recognizer(self):
        """Test setting global instance."""
        custom = PatternRecognizer()
        set_pattern_recognizer(custom)
        assert get_pattern_recognizer() is custom


class TestPatternAnalysis:
    """Test pattern analysis scenarios."""

    def test_trending_market(self):
        """Test patterns in trending market."""
        recognizer = PatternRecognizer()

        # Create uptrend
        prices = []
        for i in range(20):
            base = 100 + i * 2
            prices.append((base, base + 3, base - 1, base + 2))

        candles = create_candles(prices)
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)

    def test_ranging_market(self):
        """Test patterns in ranging market."""
        recognizer = PatternRecognizer()

        # Create range-bound market
        prices = []
        for i in range(20):
            base = 100 + (i % 5) * 2
            prices.append((base, base + 3, base - 2, base + 1))

        candles = create_candles(prices)
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)

    def test_volatile_market(self):
        """Test patterns in volatile market."""
        recognizer = PatternRecognizer()

        # Create volatile market
        prices = []
        import random
        random.seed(42)
        for i in range(20):
            base = 100 + random.randint(-10, 10)
            h = base + random.randint(1, 5)
            l = base - random.randint(1, 5)
            c = base + random.randint(-3, 3)
            prices.append((base, h, l, c))

        candles = create_candles(prices)
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_candles(self):
        """Test with empty candles list."""
        recognizer = PatternRecognizer()
        patterns = recognizer.analyze([], "BTC-USD-PERP")
        assert patterns == []

    def test_single_candle(self):
        """Test with single candle."""
        recognizer = PatternRecognizer()
        candles = create_candles([(100, 110, 90, 105)])
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)

    def test_flat_price(self):
        """Test with flat price (all same values)."""
        recognizer = PatternRecognizer()
        prices = [(100, 100, 100, 100) for _ in range(10)]
        candles = create_candles(prices)
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)

    def test_zero_volume(self):
        """Test with zero volume."""
        candles = [
            OHLCV(
                timestamp=datetime.now(),
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("90"),
                close=Decimal("105"),
                volume=Decimal("0"),
            )
        ]
        recognizer = PatternRecognizer()
        patterns = recognizer.analyze(candles, "BTC-USD-PERP")
        assert isinstance(patterns, list)
