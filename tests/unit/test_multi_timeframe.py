"""Tests for multi-timeframe analysis module."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.analytics.multi_timeframe import (
    Timeframe,
    TrendDirection,
    BiasStrength,
    SignalType,
    OHLCV,
    TimeframeData,
    MTFSignal,
    MTFAnalysis,
    TrendAnalyzer,
    SupportResistanceFinder,
    IndicatorCalculator,
    TimeframeCorrelator,
    MTFSignalGenerator,
    MultiTimeframeAnalyzer,
    get_mtf_analyzer,
    set_mtf_analyzer,
)


class TestEnums:
    """Test enum classes."""

    def test_timeframe_values(self):
        """Test Timeframe enum values."""
        assert Timeframe.M1.value == "1m"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.D1.value == "1d"
        assert Timeframe.W1.value == "1w"

    def test_timeframe_minutes(self):
        """Test timeframe minutes property."""
        assert Timeframe.M1.minutes == 1
        assert Timeframe.M5.minutes == 5
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440

    def test_trend_direction_values(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.STRONG_UP.value == "strong_up"
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.NEUTRAL.value == "neutral"
        assert TrendDirection.DOWN.value == "down"
        assert TrendDirection.STRONG_DOWN.value == "strong_down"

    def test_bias_strength_values(self):
        """Test BiasStrength enum values."""
        assert BiasStrength.VERY_STRONG.value == "very_strong"
        assert BiasStrength.STRONG.value == "strong"
        assert BiasStrength.MODERATE.value == "moderate"
        assert BiasStrength.WEAK.value == "weak"
        assert BiasStrength.NEUTRAL.value == "neutral"

    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"


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


class TestTimeframeData:
    """Test TimeframeData class."""

    def test_creation(self):
        """Test TimeframeData creation."""
        data = TimeframeData(
            timeframe=Timeframe.H1,
            trend=TrendDirection.UP,
            trend_strength=Decimal("50"),
        )
        assert data.timeframe == Timeframe.H1
        assert data.trend == TrendDirection.UP

    def test_to_dict(self):
        """Test TimeframeData to_dict."""
        data = TimeframeData(
            timeframe=Timeframe.H4,
            trend=TrendDirection.STRONG_UP,
            trend_strength=Decimal("75"),
            support_levels=[Decimal("100"), Decimal("95")],
            resistance_levels=[Decimal("110"), Decimal("115")],
        )
        result = data.to_dict()
        assert result["timeframe"] == "4h"
        assert result["trend"] == "strong_up"
        assert len(result["support_levels"]) == 2


class TestMTFSignal:
    """Test MTFSignal class."""

    def test_creation(self):
        """Test MTFSignal creation."""
        signal = MTFSignal(
            symbol="BTC-USD-PERP",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.8"),
            timeframes_aligned=[Timeframe.H1, Timeframe.H4],
            timeframes_opposing=[Timeframe.M15],
        )
        assert signal.symbol == "BTC-USD-PERP"
        assert signal.signal_type == SignalType.BUY
        assert len(signal.timeframes_aligned) == 2

    def test_to_dict(self):
        """Test MTFSignal to_dict."""
        signal = MTFSignal(
            symbol="BTC-USD-PERP",
            signal_type=SignalType.SELL,
            confidence=Decimal("0.7"),
            timeframes_aligned=[Timeframe.H4, Timeframe.D1],
            timeframes_opposing=[],
            entry_price=Decimal("50000"),
            stop_loss=Decimal("51000"),
            take_profit=Decimal("48000"),
        )
        result = signal.to_dict()
        assert result["signal_type"] == "sell"
        assert result["entry_price"] == "50000"


class TestMTFAnalysis:
    """Test MTFAnalysis class."""

    def test_creation(self):
        """Test MTFAnalysis creation."""
        analysis = MTFAnalysis(
            symbol="BTC-USD-PERP",
            overall_trend=TrendDirection.UP,
            overall_bias=SignalType.BUY,
            bias_strength=BiasStrength.STRONG,
            confluence_score=Decimal("75"),
            timeframe_analyses={},
        )
        assert analysis.symbol == "BTC-USD-PERP"
        assert analysis.overall_trend == TrendDirection.UP

    def test_to_dict(self):
        """Test MTFAnalysis to_dict."""
        analysis = MTFAnalysis(
            symbol="ETH-USD-PERP",
            overall_trend=TrendDirection.STRONG_DOWN,
            overall_bias=SignalType.SELL,
            bias_strength=BiasStrength.VERY_STRONG,
            confluence_score=Decimal("85"),
            timeframe_analyses={},
        )
        result = analysis.to_dict()
        assert result["overall_trend"] == "strong_down"
        assert result["bias_strength"] == "very_strong"


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


def create_uptrend_candles(count: int = 30) -> list[OHLCV]:
    """Create uptrending candles."""
    prices = []
    for i in range(count):
        base = 100 + i * 2
        prices.append((base, base + 3, base - 1, base + 2))
    return create_candles(prices)


def create_downtrend_candles(count: int = 30) -> list[OHLCV]:
    """Create downtrending candles."""
    prices = []
    for i in range(count):
        base = 200 - i * 2
        prices.append((base, base + 1, base - 3, base - 2))
    return create_candles(prices)


def create_ranging_candles(count: int = 30) -> list[OHLCV]:
    """Create ranging candles."""
    prices = []
    for i in range(count):
        base = 100 + (i % 5)
        prices.append((base, base + 2, base - 2, base + 1))
    return create_candles(prices)


class TestTrendAnalyzer:
    """Test TrendAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = TrendAnalyzer(sma_period=20, ema_period=9)
        assert analyzer.sma_period == 20
        assert analyzer.ema_period == 9

    def test_analyze_uptrend(self):
        """Test uptrend analysis."""
        analyzer = TrendAnalyzer()
        candles = create_uptrend_candles()
        trend, strength = analyzer.analyze(candles)
        assert trend in (TrendDirection.UP, TrendDirection.STRONG_UP)
        assert strength >= 0

    def test_analyze_downtrend(self):
        """Test downtrend analysis."""
        analyzer = TrendAnalyzer()
        candles = create_downtrend_candles()
        trend, strength = analyzer.analyze(candles)
        assert trend in (TrendDirection.DOWN, TrendDirection.STRONG_DOWN)
        assert strength >= 0

    def test_analyze_ranging(self):
        """Test ranging market analysis."""
        analyzer = TrendAnalyzer()
        candles = create_ranging_candles()
        trend, strength = analyzer.analyze(candles)
        # Ranging markets should be neutral or weak trend
        assert trend in (
            TrendDirection.NEUTRAL, TrendDirection.UP, TrendDirection.DOWN
        )

    def test_insufficient_data(self):
        """Test with insufficient data."""
        analyzer = TrendAnalyzer()
        candles = create_candles([(100, 105, 95, 102) for _ in range(5)])
        trend, strength = analyzer.analyze(candles)
        assert trend == TrendDirection.NEUTRAL
        assert strength == Decimal("0")


class TestSupportResistanceFinder:
    """Test SupportResistanceFinder class."""

    def test_init(self):
        """Test initialization."""
        finder = SupportResistanceFinder(sensitivity=5, max_levels=10)
        assert finder.sensitivity == 5
        assert finder.max_levels == 10

    def test_find_levels(self):
        """Test finding levels."""
        finder = SupportResistanceFinder(sensitivity=2)
        prices = [
            (100, 102, 98, 101),
            (101, 103, 99, 102),
            (102, 110, 100, 108),  # High
            (108, 109, 105, 106),
            (106, 108, 104, 105),
            (105, 107, 95, 96),    # Low
            (96, 98, 94, 97),
            (97, 100, 96, 99),
        ]
        candles = create_candles(prices)
        supports, resistances = finder.find_levels(candles)
        assert isinstance(supports, list)
        assert isinstance(resistances, list)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        finder = SupportResistanceFinder()
        candles = create_candles([(100, 105, 95, 102)])
        supports, resistances = finder.find_levels(candles)
        assert supports == []
        assert resistances == []


class TestIndicatorCalculator:
    """Test IndicatorCalculator class."""

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        calc = IndicatorCalculator()
        candles = create_uptrend_candles(20)
        rsi = calc.calculate_rsi(candles)
        assert rsi is not None
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        calc = IndicatorCalculator()
        candles = create_candles([(100, 105, 95, 102) for _ in range(5)])
        rsi = calc.calculate_rsi(candles)
        assert rsi is None

    def test_calculate_macd(self):
        """Test MACD calculation."""
        calc = IndicatorCalculator()
        candles = create_uptrend_candles(40)
        macd, signal, hist = calc.calculate_macd(candles)
        assert macd is not None

    def test_calculate_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        calc = IndicatorCalculator()
        candles = create_candles([(100, 105, 95, 102) for _ in range(10)])
        macd, signal, hist = calc.calculate_macd(candles)
        assert macd is None


class TestTimeframeCorrelator:
    """Test TimeframeCorrelator class."""

    def test_init(self):
        """Test initialization."""
        weights = {Timeframe.H1: Decimal("0.5"), Timeframe.D1: Decimal("0.5")}
        correlator = TimeframeCorrelator(weights)
        assert correlator.weights == weights

    def test_calculate_confluence_bullish(self):
        """Test confluence calculation with bullish bias."""
        correlator = TimeframeCorrelator()
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.UP,
                trend_strength=Decimal("50"),
            ),
            Timeframe.H4: TimeframeData(
                timeframe=Timeframe.H4,
                trend=TrendDirection.STRONG_UP,
                trend_strength=Decimal("70"),
            ),
            Timeframe.D1: TimeframeData(
                timeframe=Timeframe.D1,
                trend=TrendDirection.UP,
                trend_strength=Decimal("60"),
            ),
        }
        trend, strength, confluence = correlator.calculate_confluence(analyses)
        assert trend in (TrendDirection.UP, TrendDirection.STRONG_UP)

    def test_calculate_confluence_bearish(self):
        """Test confluence calculation with bearish bias."""
        correlator = TimeframeCorrelator()
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.DOWN,
                trend_strength=Decimal("50"),
            ),
            Timeframe.H4: TimeframeData(
                timeframe=Timeframe.H4,
                trend=TrendDirection.STRONG_DOWN,
                trend_strength=Decimal("70"),
            ),
        }
        trend, strength, confluence = correlator.calculate_confluence(analyses)
        assert trend in (TrendDirection.DOWN, TrendDirection.STRONG_DOWN)

    def test_calculate_confluence_mixed(self):
        """Test confluence with mixed signals."""
        correlator = TimeframeCorrelator()
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.UP,
                trend_strength=Decimal("50"),
            ),
            Timeframe.H4: TimeframeData(
                timeframe=Timeframe.H4,
                trend=TrendDirection.DOWN,
                trend_strength=Decimal("50"),
            ),
        }
        trend, strength, confluence = correlator.calculate_confluence(analyses)
        assert isinstance(trend, TrendDirection)

    def test_find_aligned_timeframes(self):
        """Test finding aligned timeframes."""
        correlator = TimeframeCorrelator()
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.UP,
                trend_strength=Decimal("50"),
            ),
            Timeframe.H4: TimeframeData(
                timeframe=Timeframe.H4,
                trend=TrendDirection.STRONG_UP,
                trend_strength=Decimal("70"),
            ),
            Timeframe.D1: TimeframeData(
                timeframe=Timeframe.D1,
                trend=TrendDirection.DOWN,
                trend_strength=Decimal("30"),
            ),
        }
        aligned, opposing = correlator.find_aligned_timeframes(
            analyses, TrendDirection.UP
        )
        assert Timeframe.H1 in aligned
        assert Timeframe.H4 in aligned
        assert Timeframe.D1 in opposing


class TestMTFSignalGenerator:
    """Test MTFSignalGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = MTFSignalGenerator(min_confluence=Decimal("60"))
        assert generator.min_confluence == Decimal("60")

    def test_generate_buy_signal(self):
        """Test generating buy signal."""
        generator = MTFSignalGenerator(min_confluence=Decimal("50"))
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.UP,
                trend_strength=Decimal("50"),
                support_levels=[Decimal("95"), Decimal("90")],
                resistance_levels=[Decimal("110"), Decimal("115")],
                candles=create_uptrend_candles(5),
            ),
        }
        signals = generator.generate(
            "BTC-USD-PERP",
            analyses,
            TrendDirection.UP,
            Decimal("70"),
        )
        assert len(signals) > 0
        assert signals[0].signal_type == SignalType.BUY

    def test_generate_sell_signal(self):
        """Test generating sell signal."""
        generator = MTFSignalGenerator(min_confluence=Decimal("50"))
        analyses = {
            Timeframe.H1: TimeframeData(
                timeframe=Timeframe.H1,
                trend=TrendDirection.DOWN,
                trend_strength=Decimal("50"),
                support_levels=[Decimal("90"), Decimal("85")],
                resistance_levels=[Decimal("110"), Decimal("115")],
                candles=create_downtrend_candles(5),
            ),
        }
        signals = generator.generate(
            "BTC-USD-PERP",
            analyses,
            TrendDirection.DOWN,
            Decimal("70"),
        )
        assert len(signals) > 0
        assert signals[0].signal_type == SignalType.SELL

    def test_no_signal_low_confluence(self):
        """Test no signal when confluence is low."""
        generator = MTFSignalGenerator(min_confluence=Decimal("80"))
        signals = generator.generate(
            "BTC-USD-PERP",
            {},
            TrendDirection.UP,
            Decimal("50"),
        )
        assert len(signals) == 0


class TestMultiTimeframeAnalyzer:
    """Test MultiTimeframeAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = MultiTimeframeAnalyzer()
        assert len(analyzer.timeframes) > 0

    def test_init_custom_timeframes(self):
        """Test initialization with custom timeframes."""
        timeframes = [Timeframe.M5, Timeframe.H1]
        analyzer = MultiTimeframeAnalyzer(timeframes=timeframes)
        assert analyzer.timeframes == timeframes

    def test_analyze(self):
        """Test analysis."""
        analyzer = MultiTimeframeAnalyzer()
        data = {
            Timeframe.H1: create_uptrend_candles(30),
            Timeframe.H4: create_uptrend_candles(30),
        }
        result = analyzer.analyze("BTC-USD-PERP", data)
        assert result.symbol == "BTC-USD-PERP"
        assert isinstance(result.overall_trend, TrendDirection)
        assert isinstance(result.overall_bias, SignalType)

    def test_analyze_multiple_symbols(self):
        """Test analyzing multiple symbols."""
        analyzer = MultiTimeframeAnalyzer()

        btc_data = {Timeframe.H1: create_uptrend_candles()}
        eth_data = {Timeframe.H1: create_downtrend_candles()}

        analyzer.analyze("BTC-USD-PERP", btc_data)
        analyzer.analyze("ETH-USD-PERP", eth_data)

        btc = analyzer.get_analysis("BTC-USD-PERP")
        eth = analyzer.get_analysis("ETH-USD-PERP")

        assert btc is not None
        assert eth is not None
        assert btc.symbol == "BTC-USD-PERP"
        assert eth.symbol == "ETH-USD-PERP"

    def test_get_trend(self):
        """Test getting trend."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)
        trend = analyzer.get_trend("BTC-USD-PERP")
        assert trend in (TrendDirection.UP, TrendDirection.STRONG_UP, TrendDirection.NEUTRAL)

    def test_get_bias(self):
        """Test getting bias."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)
        bias = analyzer.get_bias("BTC-USD-PERP")
        assert isinstance(bias, SignalType)

    def test_get_confluence(self):
        """Test getting confluence score."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)
        confluence = analyzer.get_confluence("BTC-USD-PERP")
        assert confluence is not None
        assert confluence >= 0

    def test_get_signals(self):
        """Test getting signals."""
        analyzer = MultiTimeframeAnalyzer(min_confluence=Decimal("0"))
        data = {
            Timeframe.H1: create_uptrend_candles(),
            Timeframe.H4: create_uptrend_candles(),
        }
        analyzer.analyze("BTC-USD-PERP", data)
        signals = analyzer.get_signals("BTC-USD-PERP")
        assert isinstance(signals, list)

    def test_add_callback(self):
        """Test adding callback."""
        analyzer = MultiTimeframeAnalyzer()
        results = []

        def callback(analysis):
            results.append(analysis)

        analyzer.add_callback(callback)
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)

        assert len(results) == 1

    def test_remove_callback(self):
        """Test removing callback."""
        analyzer = MultiTimeframeAnalyzer()
        results = []

        def callback(analysis):
            results.append(analysis)

        analyzer.add_callback(callback)
        assert analyzer.remove_callback(callback) is True
        assert analyzer.remove_callback(callback) is False

    def test_clear(self):
        """Test clearing analyses."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)

        assert analyzer.get_analysis("BTC-USD-PERP") is not None
        analyzer.clear()
        assert analyzer.get_analysis("BTC-USD-PERP") is None

    def test_clear_by_symbol(self):
        """Test clearing by symbol."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)
        analyzer.analyze("ETH-USD-PERP", data)

        analyzer.clear(symbol="BTC-USD-PERP")
        assert analyzer.get_analysis("BTC-USD-PERP") is None
        assert analyzer.get_analysis("ETH-USD-PERP") is not None

    def test_get_summary(self):
        """Test getting summary."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)
        analyzer.analyze("ETH-USD-PERP", data)

        summary = analyzer.get_summary()
        assert "symbols_analyzed" in summary
        assert "total_symbols" in summary
        assert summary["total_symbols"] == 2

    def test_get_summary_by_symbol(self):
        """Test getting summary for specific symbol."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        analyzer.analyze("BTC-USD-PERP", data)

        summary = analyzer.get_summary("BTC-USD-PERP")
        assert summary["symbol"] == "BTC-USD-PERP"
        assert "trend" in summary
        assert "bias" in summary
        assert "confluence" in summary


class TestGlobalInstance:
    """Test global instance functions."""

    def test_get_mtf_analyzer(self):
        """Test getting global instance."""
        analyzer = get_mtf_analyzer()
        assert analyzer is not None
        assert isinstance(analyzer, MultiTimeframeAnalyzer)

    def test_set_mtf_analyzer(self):
        """Test setting global instance."""
        custom = MultiTimeframeAnalyzer()
        set_mtf_analyzer(custom)
        assert get_mtf_analyzer() is custom


class TestIntegration:
    """Integration tests."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        analyzer = MultiTimeframeAnalyzer(min_confluence=Decimal("0"))

        # Create multi-timeframe data
        data = {
            Timeframe.M15: create_uptrend_candles(50),
            Timeframe.H1: create_uptrend_candles(50),
            Timeframe.H4: create_uptrend_candles(50),
            Timeframe.D1: create_uptrend_candles(50),
        }

        # Analyze
        result = analyzer.analyze("BTC-USD-PERP", data)

        # Verify result structure
        assert result.symbol == "BTC-USD-PERP"
        assert isinstance(result.overall_trend, TrendDirection)
        assert isinstance(result.overall_bias, SignalType)
        assert isinstance(result.bias_strength, BiasStrength)
        assert result.confluence_score >= 0
        assert len(result.timeframe_analyses) == 4

        # Verify timeframe analyses
        for tf, tf_data in result.timeframe_analyses.items():
            assert isinstance(tf_data.trend, TrendDirection)
            assert len(tf_data.candles) > 0

        # Verify to_dict works
        result_dict = result.to_dict()
        assert "symbol" in result_dict
        assert "overall_trend" in result_dict
        assert "timeframe_analyses" in result_dict

    def test_mixed_trends(self):
        """Test analysis with mixed trends."""
        analyzer = MultiTimeframeAnalyzer()

        data = {
            Timeframe.H1: create_uptrend_candles(),
            Timeframe.H4: create_downtrend_candles(),
            Timeframe.D1: create_ranging_candles(),
        }

        result = analyzer.analyze("BTC-USD-PERP", data)
        assert result is not None

    def test_empty_data(self):
        """Test analysis with empty data."""
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze("BTC-USD-PERP", {})
        assert result is not None
        assert result.overall_trend == TrendDirection.NEUTRAL


class TestEdgeCases:
    """Test edge cases."""

    def test_single_timeframe(self):
        """Test analysis with single timeframe."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: create_uptrend_candles()}
        result = analyzer.analyze("BTC-USD-PERP", data)
        assert result is not None

    def test_empty_candles_list(self):
        """Test with empty candles list."""
        analyzer = MultiTimeframeAnalyzer()
        data = {Timeframe.H1: []}
        result = analyzer.analyze("BTC-USD-PERP", data)
        assert result is not None

    def test_very_short_data(self):
        """Test with very short data."""
        analyzer = MultiTimeframeAnalyzer()
        data = {
            Timeframe.H1: create_candles([(100, 105, 95, 102)])
        }
        result = analyzer.analyze("BTC-USD-PERP", data)
        assert result is not None

    def test_nonexistent_symbol(self):
        """Test getting analysis for nonexistent symbol."""
        analyzer = MultiTimeframeAnalyzer()
        assert analyzer.get_analysis("FAKE-SYMBOL") is None
        assert analyzer.get_trend("FAKE-SYMBOL") is None
        assert analyzer.get_bias("FAKE-SYMBOL") is None
        assert analyzer.get_confluence("FAKE-SYMBOL") is None
        assert analyzer.get_signals("FAKE-SYMBOL") == []
