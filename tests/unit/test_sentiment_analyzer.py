"""
Tests for Sentiment Analyzer Module.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

from src.analytics.sentiment_analyzer import (
    SentimentLevel, SentimentSource, TrendBias,
    SentimentData, SentimentScore, FundingRateSentiment, LongShortRatio,
    LiquidationData, FearGreedIndex, SentimentSignal,
    FundingRateAnalyzer, LongShortAnalyzer, LiquidationAnalyzer,
    VolumeProfileAnalyzer, FearGreedCalculator, SentimentAggregator,
    SignalGenerator, SentimentAnalyzer,
    get_sentiment_analyzer, set_sentiment_analyzer
)


# ============== Fixtures ==============

@pytest.fixture
def funding_analyzer():
    return FundingRateAnalyzer()


@pytest.fixture
def ls_analyzer():
    return LongShortAnalyzer()


@pytest.fixture
def liquidation_analyzer():
    return LiquidationAnalyzer()


@pytest.fixture
def volume_analyzer():
    return VolumeProfileAnalyzer()


@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_sentiment_level_values(self):
        assert SentimentLevel.EXTREME_FEAR.value == "extreme_fear"
        assert SentimentLevel.FEAR.value == "fear"
        assert SentimentLevel.NEUTRAL.value == "neutral"
        assert SentimentLevel.GREED.value == "greed"
        assert SentimentLevel.EXTREME_GREED.value == "extreme_greed"

    def test_sentiment_source_values(self):
        assert SentimentSource.FUNDING_RATE.value == "funding_rate"
        assert SentimentSource.LONG_SHORT_RATIO.value == "long_short_ratio"
        assert SentimentSource.LIQUIDATIONS.value == "liquidations"

    def test_trend_bias_values(self):
        assert TrendBias.BULLISH.value == "bullish"
        assert TrendBias.BEARISH.value == "bearish"
        assert TrendBias.NEUTRAL.value == "neutral"


# ============== Data Class Tests ==============

class TestSentimentData:
    """Test SentimentData dataclass."""

    def test_creation(self):
        data = SentimentData(
            source=SentimentSource.FUNDING_RATE,
            symbol="BTC-USD",
            value=Decimal("0.0003"),
            timestamp=datetime.now()
        )
        assert data.source == SentimentSource.FUNDING_RATE
        assert data.symbol == "BTC-USD"

    def test_to_dict(self):
        data = SentimentData(
            source=SentimentSource.VOLUME,
            symbol="ETH-USD",
            value=Decimal("1000000"),
            timestamp=datetime.now(),
            metadata={"type": "24h"}
        )
        result = data.to_dict()
        assert result["source"] == "volume"
        assert "metadata" in result


class TestSentimentScore:
    """Test SentimentScore dataclass."""

    def test_creation(self):
        score = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("75"),
            level=SentimentLevel.GREED,
            bias=TrendBias.BULLISH,
            confidence=Decimal("80"),
            sources=[SentimentSource.FUNDING_RATE, SentimentSource.VOLUME]
        )
        assert score.score == Decimal("75")
        assert score.level == SentimentLevel.GREED

    def test_to_dict(self):
        score = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("50"),
            level=SentimentLevel.NEUTRAL,
            bias=TrendBias.NEUTRAL,
            confidence=Decimal("60"),
            sources=[SentimentSource.FUNDING_RATE]
        )
        result = score.to_dict()
        assert result["level"] == "neutral"
        assert result["bias"] == "neutral"


class TestFundingRateSentiment:
    """Test FundingRateSentiment dataclass."""

    def test_creation(self):
        sentiment = FundingRateSentiment(
            symbol="BTC-USD",
            funding_rate=Decimal("0.0005"),
            annualized_rate=Decimal("0.5475"),
            sentiment_score=Decimal("70"),
            bias=TrendBias.BULLISH
        )
        assert sentiment.funding_rate == Decimal("0.0005")

    def test_to_dict(self):
        sentiment = FundingRateSentiment(
            symbol="BTC-USD",
            funding_rate=Decimal("0.001"),
            annualized_rate=Decimal("1.095"),
            sentiment_score=Decimal("100"),
            bias=TrendBias.BULLISH
        )
        result = sentiment.to_dict()
        assert "annualized_rate" in result


class TestLongShortRatio:
    """Test LongShortRatio dataclass."""

    def test_creation(self):
        ls = LongShortRatio(
            symbol="BTC-USD",
            long_ratio=Decimal("60"),
            short_ratio=Decimal("40"),
            ratio=Decimal("1.5"),
            sentiment_score=Decimal("50"),
            bias=TrendBias.NEUTRAL
        )
        assert ls.ratio == Decimal("1.5")

    def test_to_dict(self):
        ls = LongShortRatio(
            symbol="BTC-USD",
            long_ratio=Decimal("70"),
            short_ratio=Decimal("30"),
            ratio=Decimal("2.33"),
            sentiment_score=Decimal("80"),
            bias=TrendBias.BEARISH
        )
        result = ls.to_dict()
        assert "long_ratio" in result


class TestLiquidationData:
    """Test LiquidationData dataclass."""

    def test_creation(self):
        liq = LiquidationData(
            symbol="BTC-USD",
            long_liquidations=Decimal("5000000"),
            short_liquidations=Decimal("2000000"),
            total_liquidations=Decimal("7000000"),
            liquidation_ratio=Decimal("2.5"),
            sentiment_impact=Decimal("-50")
        )
        assert liq.total_liquidations == Decimal("7000000")

    def test_to_dict(self):
        liq = LiquidationData(
            symbol="BTC-USD",
            long_liquidations=Decimal("1000000"),
            short_liquidations=Decimal("3000000"),
            total_liquidations=Decimal("4000000"),
            liquidation_ratio=Decimal("0.33"),
            sentiment_impact=Decimal("75")
        )
        result = liq.to_dict()
        assert "liquidation_ratio" in result


class TestFearGreedIndex:
    """Test FearGreedIndex dataclass."""

    def test_creation(self):
        fgi = FearGreedIndex(
            value=Decimal("25"),
            level=SentimentLevel.FEAR,
            previous_value=Decimal("30"),
            change=Decimal("-5")
        )
        assert fgi.level == SentimentLevel.FEAR

    def test_to_dict(self):
        fgi = FearGreedIndex(
            value=Decimal("75"),
            level=SentimentLevel.GREED,
            previous_value=Decimal("70"),
            change=Decimal("5")
        )
        result = fgi.to_dict()
        assert result["level"] == "greed"


class TestSentimentSignal:
    """Test SentimentSignal dataclass."""

    def test_creation(self):
        signal = SentimentSignal(
            symbol="BTC-USD",
            signal_type="buy",
            strength=Decimal("75"),
            sentiment_score=Decimal("-60"),
            reasoning=["Extreme fear", "Contrarian buy"]
        )
        assert signal.signal_type == "buy"

    def test_to_dict(self):
        signal = SentimentSignal(
            symbol="BTC-USD",
            signal_type="sell",
            strength=Decimal("80"),
            sentiment_score=Decimal("70"),
            reasoning=["Extreme greed"]
        )
        result = signal.to_dict()
        assert "reasoning" in result


# ============== Funding Rate Analyzer Tests ==============

class TestFundingRateAnalyzer:
    """Test FundingRateAnalyzer."""

    def test_init(self, funding_analyzer):
        assert funding_analyzer.extreme_threshold == Decimal("0.001")
        assert funding_analyzer.high_threshold == Decimal("0.0005")

    def test_add_data(self, funding_analyzer):
        funding_analyzer.add_data("BTC-USD", Decimal("0.0003"))
        assert "BTC-USD" in funding_analyzer.history
        assert len(funding_analyzer.history["BTC-USD"]) == 1

    def test_analyze_positive_funding(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("0.0006"))
        assert result.bias == TrendBias.BULLISH
        assert result.sentiment_score > Decimal("0")

    def test_analyze_negative_funding(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("-0.0006"))
        assert result.bias == TrendBias.BEARISH
        assert result.sentiment_score < Decimal("0")

    def test_analyze_neutral_funding(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("0.0001"))
        assert result.bias == TrendBias.NEUTRAL

    def test_analyze_extreme_positive(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("0.002"))
        assert result.sentiment_score == Decimal("100")

    def test_analyze_extreme_negative(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("-0.002"))
        assert result.sentiment_score == Decimal("-100")

    def test_annualized_rate(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC-USD", Decimal("0.001"))
        assert result.annualized_rate == Decimal("1.095")

    def test_get_average(self, funding_analyzer):
        for rate in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]:
            funding_analyzer.add_data("BTC-USD", Decimal(str(rate)))

        avg = funding_analyzer.get_average("BTC-USD", periods=5)
        assert avg == Decimal("0.0003")

    def test_get_average_no_data(self, funding_analyzer):
        result = funding_analyzer.get_average("UNKNOWN")
        assert result is None


# ============== Long Short Analyzer Tests ==============

class TestLongShortAnalyzer:
    """Test LongShortAnalyzer."""

    def test_init(self, ls_analyzer):
        assert ls_analyzer.extreme_long_ratio == Decimal("2.0")

    def test_analyze_balanced(self, ls_analyzer):
        result = ls_analyzer.analyze("BTC-USD", Decimal("50"), Decimal("50"))
        assert result.ratio == Decimal("1")
        assert result.bias == TrendBias.NEUTRAL

    def test_analyze_long_heavy(self, ls_analyzer):
        result = ls_analyzer.analyze("BTC-USD", Decimal("75"), Decimal("25"))
        assert result.ratio == Decimal("3")
        assert result.bias == TrendBias.BEARISH  # Contrarian

    def test_analyze_short_heavy(self, ls_analyzer):
        result = ls_analyzer.analyze("BTC-USD", Decimal("25"), Decimal("75"))
        assert float(result.ratio) < 0.5
        assert result.bias == TrendBias.BULLISH  # Contrarian

    def test_analyze_zero_short(self, ls_analyzer):
        result = ls_analyzer.analyze("BTC-USD", Decimal("100"), Decimal("0"))
        assert result.ratio == Decimal("999")


# ============== Liquidation Analyzer Tests ==============

class TestLiquidationAnalyzer:
    """Test LiquidationAnalyzer."""

    def test_init(self, liquidation_analyzer):
        assert liquidation_analyzer.significant_threshold == Decimal("1000000")

    def test_analyze_long_cascade(self, liquidation_analyzer):
        result = liquidation_analyzer.analyze(
            "BTC-USD",
            Decimal("5000000"),  # Long liquidations
            Decimal("1000000")   # Short liquidations
        )
        assert result.liquidation_ratio == Decimal("5")
        assert result.sentiment_impact < Decimal("0")  # Bearish

    def test_analyze_short_cascade(self, liquidation_analyzer):
        result = liquidation_analyzer.analyze(
            "BTC-USD",
            Decimal("1000000"),  # Long liquidations
            Decimal("5000000")   # Short liquidations
        )
        assert float(result.liquidation_ratio) < 1
        assert result.sentiment_impact > Decimal("0")  # Bullish

    def test_analyze_insignificant(self, liquidation_analyzer):
        result = liquidation_analyzer.analyze(
            "BTC-USD",
            Decimal("100000"),
            Decimal("100000")
        )
        assert result.sentiment_impact == Decimal("0")

    def test_analyze_zero_short(self, liquidation_analyzer):
        result = liquidation_analyzer.analyze(
            "BTC-USD",
            Decimal("1000000"),
            Decimal("0")
        )
        assert result.liquidation_ratio == Decimal("999")


# ============== Volume Profile Analyzer Tests ==============

class TestVolumeProfileAnalyzer:
    """Test VolumeProfileAnalyzer."""

    def test_init(self, volume_analyzer):
        assert volume_analyzer.periods == 20

    def test_add_data(self, volume_analyzer):
        volume_analyzer.add_data(
            "BTC-USD",
            Decimal("50000"),
            Decimal("1000000"),
            Decimal("600000"),
            Decimal("400000")
        )
        assert "BTC-USD" in volume_analyzer.history
        assert len(volume_analyzer.history["BTC-USD"]) == 1

    def test_get_volume_sentiment_bullish(self, volume_analyzer):
        # Add buy-heavy data
        for i in range(5):
            volume_analyzer.add_data(
                "BTC-USD",
                Decimal("50000"),
                Decimal("1000000"),
                Decimal("700000"),  # 70% buy
                Decimal("300000")   # 30% sell
            )

        sentiment = volume_analyzer.get_volume_sentiment("BTC-USD")
        assert sentiment > Decimal("0")

    def test_get_volume_sentiment_bearish(self, volume_analyzer):
        # Add sell-heavy data
        for i in range(5):
            volume_analyzer.add_data(
                "BTC-USD",
                Decimal("50000"),
                Decimal("1000000"),
                Decimal("300000"),  # 30% buy
                Decimal("700000")   # 70% sell
            )

        sentiment = volume_analyzer.get_volume_sentiment("BTC-USD")
        assert sentiment < Decimal("0")

    def test_get_volume_sentiment_no_data(self, volume_analyzer):
        result = volume_analyzer.get_volume_sentiment("UNKNOWN")
        assert result is None

    def test_get_volume_trend(self, volume_analyzer):
        # Add increasing volume
        for i in range(10):
            volume_analyzer.add_data(
                "BTC-USD",
                Decimal("50000"),
                Decimal(str(1000000 + i * 200000))
            )

        trend = volume_analyzer.get_volume_trend("BTC-USD")
        assert trend is not None


# ============== Fear Greed Calculator Tests ==============

class TestFearGreedCalculator:
    """Test FearGreedCalculator."""

    def test_calculate_no_components(self):
        calc = FearGreedCalculator()
        result = calc.calculate()
        assert result.value == Decimal("50")
        assert result.level == SentimentLevel.NEUTRAL

    def test_update_component(self):
        calc = FearGreedCalculator()
        calc.update_component("volatility", Decimal("80"))
        assert "volatility" in calc.components

    def test_calculate_extreme_fear(self):
        calc = FearGreedCalculator()
        calc.update_component("comp1", Decimal("10"))
        calc.update_component("comp2", Decimal("15"))
        result = calc.calculate()
        assert result.level == SentimentLevel.EXTREME_FEAR

    def test_calculate_extreme_greed(self):
        calc = FearGreedCalculator()
        calc.update_component("comp1", Decimal("90"))
        calc.update_component("comp2", Decimal("85"))
        result = calc.calculate()
        assert result.level == SentimentLevel.EXTREME_GREED

    def test_tracks_history(self):
        calc = FearGreedCalculator()
        calc.update_component("comp1", Decimal("50"))
        calc.calculate()
        calc.update_component("comp1", Decimal("60"))
        result = calc.calculate()
        assert result.previous_value == Decimal("50")
        assert result.change == Decimal("10")


# ============== Sentiment Aggregator Tests ==============

class TestSentimentAggregator:
    """Test SentimentAggregator."""

    def test_update(self):
        agg = SentimentAggregator()
        agg.update("BTC-USD", SentimentSource.FUNDING_RATE, Decimal("50"))
        assert "BTC-USD" in agg.data
        assert SentimentSource.FUNDING_RATE in agg.data["BTC-USD"]

    def test_set_weight(self):
        agg = SentimentAggregator()
        agg.set_weight(SentimentSource.FUNDING_RATE, Decimal("0.5"))
        assert agg.weights[SentimentSource.FUNDING_RATE] == Decimal("0.5")

    def test_aggregate_no_data(self):
        agg = SentimentAggregator()
        result = agg.aggregate("UNKNOWN")
        assert result.score == Decimal("0")
        assert result.confidence == Decimal("0")

    def test_aggregate_single_source(self):
        agg = SentimentAggregator()
        agg.update("BTC-USD", SentimentSource.FUNDING_RATE, Decimal("75"))
        result = agg.aggregate("BTC-USD")
        assert result.score == Decimal("75")
        assert len(result.sources) == 1

    def test_aggregate_multiple_sources(self):
        agg = SentimentAggregator()
        agg.update("BTC-USD", SentimentSource.FUNDING_RATE, Decimal("80"))
        agg.update("BTC-USD", SentimentSource.LONG_SHORT_RATIO, Decimal("60"))
        result = agg.aggregate("BTC-USD")
        assert result.score > Decimal("0")
        assert len(result.sources) == 2

    def test_score_to_level_extreme_fear(self):
        agg = SentimentAggregator()
        level = agg._score_to_level(Decimal("-70"))
        assert level == SentimentLevel.EXTREME_FEAR

    def test_score_to_level_fear(self):
        agg = SentimentAggregator()
        level = agg._score_to_level(Decimal("-30"))
        assert level == SentimentLevel.FEAR

    def test_score_to_level_greed(self):
        agg = SentimentAggregator()
        level = agg._score_to_level(Decimal("50"))
        assert level == SentimentLevel.GREED

    def test_score_to_bias(self):
        agg = SentimentAggregator()
        assert agg._score_to_bias(Decimal("30")) == TrendBias.BULLISH
        assert agg._score_to_bias(Decimal("-30")) == TrendBias.BEARISH
        assert agg._score_to_bias(Decimal("0")) == TrendBias.NEUTRAL


# ============== Signal Generator Tests ==============

class TestSignalGenerator:
    """Test SignalGenerator."""

    def test_init(self):
        gen = SignalGenerator()
        assert gen.buy_threshold == Decimal("-50")
        assert gen.sell_threshold == Decimal("50")

    def test_generate_buy_signal(self):
        gen = SignalGenerator()
        sentiment = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("-60"),
            level=SentimentLevel.EXTREME_FEAR,
            bias=TrendBias.BEARISH,
            confidence=Decimal("80"),
            sources=[SentimentSource.FUNDING_RATE]
        )
        signal = gen.generate(sentiment)
        assert signal.signal_type == "buy"
        assert signal.strength > Decimal("0")

    def test_generate_sell_signal(self):
        gen = SignalGenerator()
        sentiment = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("70"),
            level=SentimentLevel.GREED,
            bias=TrendBias.BULLISH,
            confidence=Decimal("80"),
            sources=[SentimentSource.FUNDING_RATE]
        )
        signal = gen.generate(sentiment)
        assert signal.signal_type == "sell"

    def test_generate_hold_signal_neutral(self):
        gen = SignalGenerator()
        sentiment = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("20"),
            level=SentimentLevel.NEUTRAL,
            bias=TrendBias.NEUTRAL,
            confidence=Decimal("80"),
            sources=[SentimentSource.FUNDING_RATE]
        )
        signal = gen.generate(sentiment)
        assert signal.signal_type == "hold"

    def test_generate_hold_low_confidence(self):
        gen = SignalGenerator(min_confidence=Decimal("70"))
        sentiment = SentimentScore(
            symbol="BTC-USD",
            score=Decimal("-80"),
            level=SentimentLevel.EXTREME_FEAR,
            bias=TrendBias.BEARISH,
            confidence=Decimal("50"),  # Below threshold
            sources=[SentimentSource.FUNDING_RATE]
        )
        signal = gen.generate(sentiment)
        assert signal.signal_type == "hold"


# ============== Sentiment Analyzer Tests ==============

class TestSentimentAnalyzer:
    """Test SentimentAnalyzer."""

    def test_init(self, sentiment_analyzer):
        assert sentiment_analyzer.funding_analyzer is not None
        assert sentiment_analyzer.aggregator is not None

    def test_register_callback(self, sentiment_analyzer):
        callback = Mock()
        sentiment_analyzer.register_callback("on_sentiment_update", callback)
        assert callback in sentiment_analyzer.callbacks["on_sentiment_update"]

    def test_update_funding_rate(self, sentiment_analyzer):
        result = sentiment_analyzer.update_funding_rate("BTC-USD", Decimal("0.0008"))
        assert result.funding_rate == Decimal("0.0008")

    def test_update_long_short_ratio(self, sentiment_analyzer):
        result = sentiment_analyzer.update_long_short_ratio(
            "BTC-USD", Decimal("60"), Decimal("40")
        )
        assert result.ratio == Decimal("1.5")

    def test_update_liquidations(self, sentiment_analyzer):
        result = sentiment_analyzer.update_liquidations(
            "BTC-USD", Decimal("2000000"), Decimal("1000000")
        )
        assert result.total_liquidations == Decimal("3000000")

    def test_update_volume(self, sentiment_analyzer):
        sentiment_analyzer.update_volume(
            "BTC-USD",
            Decimal("50000"),
            Decimal("1000000"),
            Decimal("600000"),
            Decimal("400000")
        )
        # No error means success
        assert True

    def test_get_sentiment(self, sentiment_analyzer):
        sentiment_analyzer.update_funding_rate("BTC-USD", Decimal("0.0005"))
        result = sentiment_analyzer.get_sentiment("BTC-USD")
        assert result.symbol == "BTC-USD"

    def test_get_signal(self, sentiment_analyzer):
        sentiment_analyzer.update_funding_rate("BTC-USD", Decimal("0.002"))
        signal = sentiment_analyzer.get_signal("BTC-USD")
        assert signal.symbol == "BTC-USD"

    def test_get_fear_greed_index(self, sentiment_analyzer):
        sentiment_analyzer.update_fear_greed_component("volatility", Decimal("30"))
        result = sentiment_analyzer.get_fear_greed_index()
        assert result.value == Decimal("30")

    def test_get_market_sentiment_summary(self, sentiment_analyzer):
        sentiment_analyzer.update_funding_rate("BTC-USD", Decimal("0.0008"))
        sentiment_analyzer.update_funding_rate("ETH-USD", Decimal("-0.0006"))

        summary = sentiment_analyzer.get_market_sentiment_summary(["BTC-USD", "ETH-USD"])
        assert summary["total_symbols"] == 2
        assert "market_bias" in summary

    def test_callbacks_triggered(self, sentiment_analyzer):
        callback = Mock()
        sentiment_analyzer.register_callback("on_sentiment_update", callback)

        sentiment_analyzer.update_funding_rate("BTC-USD", Decimal("0.0005"))
        sentiment_analyzer.get_sentiment("BTC-USD")

        assert callback.called


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global sentiment analyzer."""

    def test_get_sentiment_analyzer(self):
        analyzer = get_sentiment_analyzer()
        assert isinstance(analyzer, SentimentAnalyzer)

    def test_set_sentiment_analyzer(self):
        custom = SentimentAnalyzer()
        custom.update_funding_rate("TEST", Decimal("0.001"))
        set_sentiment_analyzer(custom)

        analyzer = get_sentiment_analyzer()
        assert "TEST" in analyzer.aggregator.data


# ============== Integration Tests ==============

class TestSentimentIntegration:
    """Integration tests."""

    def test_full_sentiment_flow(self):
        analyzer = SentimentAnalyzer()

        # Update multiple sources
        analyzer.update_funding_rate("BTC-USD", Decimal("0.0008"))
        analyzer.update_long_short_ratio("BTC-USD", Decimal("65"), Decimal("35"))
        analyzer.update_liquidations("BTC-USD", Decimal("3000000"), Decimal("1500000"))

        # Add volume data
        for i in range(5):
            analyzer.update_volume(
                "BTC-USD",
                Decimal("50000"),
                Decimal("1000000"),
                Decimal("600000"),
                Decimal("400000")
            )

        # Get aggregated sentiment
        sentiment = analyzer.get_sentiment("BTC-USD")
        assert len(sentiment.sources) >= 3

        # Get signal
        signal = analyzer.get_signal("BTC-USD")
        assert signal.symbol == "BTC-USD"

    def test_market_summary_flow(self):
        analyzer = SentimentAnalyzer()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        # Update data for each symbol
        for i, symbol in enumerate(symbols):
            funding = Decimal(str((i - 1) * 0.0005))
            analyzer.update_funding_rate(symbol, funding)

        # Get summary
        summary = analyzer.get_market_sentiment_summary(symbols)
        assert summary["total_symbols"] == 3
        assert "bullish_count" in summary
        assert "bearish_count" in summary
