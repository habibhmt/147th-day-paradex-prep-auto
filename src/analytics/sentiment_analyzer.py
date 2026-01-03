"""
Sentiment Analyzer Module.

Analyze market sentiment from various sources and generate
trading signals based on sentiment indicators.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import math


class SentimentLevel(Enum):
    """Sentiment level."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class SentimentSource(Enum):
    """Sentiment data source."""
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LONG_SHORT_RATIO = "long_short_ratio"
    LIQUIDATIONS = "liquidations"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    VOLATILITY = "volatility"
    SOCIAL_MEDIA = "social_media"


class TrendBias(Enum):
    """Market trend bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SentimentData:
    """Raw sentiment data point."""
    source: SentimentSource
    symbol: str
    value: Decimal
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "symbol": self.symbol,
            "value": str(self.value),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SentimentScore:
    """Calculated sentiment score."""
    symbol: str
    score: Decimal  # -100 to +100
    level: SentimentLevel
    bias: TrendBias
    confidence: Decimal
    sources: list[SentimentSource]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": str(self.score),
            "level": self.level.value,
            "bias": self.bias.value,
            "confidence": str(self.confidence),
            "sources": [s.value for s in self.sources],
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class FundingRateSentiment:
    """Sentiment derived from funding rates."""
    symbol: str
    funding_rate: Decimal
    annualized_rate: Decimal
    sentiment_score: Decimal
    bias: TrendBias
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "funding_rate": str(self.funding_rate),
            "annualized_rate": str(self.annualized_rate),
            "sentiment_score": str(self.sentiment_score),
            "bias": self.bias.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LongShortRatio:
    """Long/Short ratio data."""
    symbol: str
    long_ratio: Decimal
    short_ratio: Decimal
    ratio: Decimal
    sentiment_score: Decimal
    bias: TrendBias
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "long_ratio": str(self.long_ratio),
            "short_ratio": str(self.short_ratio),
            "ratio": str(self.ratio),
            "sentiment_score": str(self.sentiment_score),
            "bias": self.bias.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LiquidationData:
    """Liquidation data."""
    symbol: str
    long_liquidations: Decimal
    short_liquidations: Decimal
    total_liquidations: Decimal
    liquidation_ratio: Decimal
    sentiment_impact: Decimal
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "long_liquidations": str(self.long_liquidations),
            "short_liquidations": str(self.short_liquidations),
            "total_liquidations": str(self.total_liquidations),
            "liquidation_ratio": str(self.liquidation_ratio),
            "sentiment_impact": str(self.sentiment_impact),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class FearGreedIndex:
    """Fear and Greed Index."""
    value: Decimal  # 0-100
    level: SentimentLevel
    previous_value: Decimal
    change: Decimal
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "value": str(self.value),
            "level": self.level.value,
            "previous_value": str(self.previous_value),
            "change": str(self.change),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SentimentSignal:
    """Trading signal from sentiment."""
    symbol: str
    signal_type: str  # "buy", "sell", "hold"
    strength: Decimal  # 0-100
    sentiment_score: Decimal
    reasoning: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "strength": str(self.strength),
            "sentiment_score": str(self.sentiment_score),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }


class FundingRateAnalyzer:
    """Analyze funding rates for sentiment."""

    def __init__(
        self,
        extreme_threshold: Decimal = Decimal("0.001"),  # 0.1%
        high_threshold: Decimal = Decimal("0.0005")     # 0.05%
    ):
        self.extreme_threshold = extreme_threshold
        self.high_threshold = high_threshold
        self.history: dict[str, list[Decimal]] = {}

    def add_data(self, symbol: str, funding_rate: Decimal):
        """Add funding rate data."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(funding_rate)
        if len(self.history[symbol]) > 100:
            self.history[symbol].pop(0)

    def analyze(self, symbol: str, funding_rate: Decimal) -> FundingRateSentiment:
        """Analyze funding rate sentiment."""
        self.add_data(symbol, funding_rate)

        # Calculate annualized rate (8-hour funding * 3 * 365)
        annualized = funding_rate * Decimal("1095")

        # Calculate sentiment score (-100 to +100)
        if abs(funding_rate) >= self.extreme_threshold:
            sentiment_score = Decimal("100") if funding_rate > 0 else Decimal("-100")
        elif abs(funding_rate) >= self.high_threshold:
            sentiment_score = Decimal("70") if funding_rate > 0 else Decimal("-70")
        else:
            sentiment_score = funding_rate / self.high_threshold * Decimal("70")

        # Determine bias
        if funding_rate > self.high_threshold:
            bias = TrendBias.BULLISH  # Longs paying shorts - bullish sentiment
        elif funding_rate < -self.high_threshold:
            bias = TrendBias.BEARISH  # Shorts paying longs - bearish sentiment
        else:
            bias = TrendBias.NEUTRAL

        return FundingRateSentiment(
            symbol=symbol,
            funding_rate=funding_rate,
            annualized_rate=annualized,
            sentiment_score=sentiment_score,
            bias=bias
        )

    def get_average(self, symbol: str, periods: int = 10) -> Optional[Decimal]:
        """Get average funding rate."""
        history = self.history.get(symbol, [])
        if not history:
            return None

        recent = history[-periods:]
        return sum(recent) / Decimal(str(len(recent)))


class LongShortAnalyzer:
    """Analyze long/short ratios."""

    def __init__(
        self,
        extreme_long_ratio: Decimal = Decimal("2.0"),
        extreme_short_ratio: Decimal = Decimal("0.5")
    ):
        self.extreme_long_ratio = extreme_long_ratio
        self.extreme_short_ratio = extreme_short_ratio
        self.history: dict[str, list[Decimal]] = {}

    def analyze(
        self,
        symbol: str,
        long_ratio: Decimal,
        short_ratio: Decimal
    ) -> LongShortRatio:
        """Analyze long/short ratio."""
        if short_ratio == 0:
            ratio = Decimal("999")
        else:
            ratio = long_ratio / short_ratio

        # Track history
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(ratio)
        if len(self.history[symbol]) > 100:
            self.history[symbol].pop(0)

        # Calculate sentiment score
        if ratio >= self.extreme_long_ratio:
            sentiment_score = Decimal("100")
        elif ratio <= self.extreme_short_ratio:
            sentiment_score = Decimal("-100")
        else:
            # Normalize to -100 to +100
            mid = (self.extreme_long_ratio + self.extreme_short_ratio) / Decimal("2")
            range_val = self.extreme_long_ratio - mid
            sentiment_score = (ratio - mid) / range_val * Decimal("100")

        # Determine bias (contrarian view)
        if ratio >= self.extreme_long_ratio:
            bias = TrendBias.BEARISH  # Too many longs - contrarian bearish
        elif ratio <= self.extreme_short_ratio:
            bias = TrendBias.BULLISH  # Too many shorts - contrarian bullish
        else:
            bias = TrendBias.NEUTRAL

        return LongShortRatio(
            symbol=symbol,
            long_ratio=long_ratio,
            short_ratio=short_ratio,
            ratio=ratio,
            sentiment_score=sentiment_score,
            bias=bias
        )


class LiquidationAnalyzer:
    """Analyze liquidation data."""

    def __init__(
        self,
        significant_threshold: Decimal = Decimal("1000000"),
        extreme_ratio: Decimal = Decimal("3.0")
    ):
        self.significant_threshold = significant_threshold
        self.extreme_ratio = extreme_ratio
        self.history: dict[str, list[dict]] = {}

    def analyze(
        self,
        symbol: str,
        long_liquidations: Decimal,
        short_liquidations: Decimal
    ) -> LiquidationData:
        """Analyze liquidation data."""
        total = long_liquidations + short_liquidations

        # Calculate ratio
        if short_liquidations == 0:
            ratio = Decimal("999") if long_liquidations > 0 else Decimal("1")
        else:
            ratio = long_liquidations / short_liquidations

        # Track history
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append({
            "long": long_liquidations,
            "short": short_liquidations,
            "total": total,
            "ratio": ratio
        })
        if len(self.history[symbol]) > 100:
            self.history[symbol].pop(0)

        # Calculate sentiment impact
        if total >= self.significant_threshold:
            if ratio >= self.extreme_ratio:
                sentiment_impact = Decimal("-100")  # Long cascade - bearish
            elif ratio <= Decimal("1") / self.extreme_ratio:
                sentiment_impact = Decimal("100")   # Short cascade - bullish
            else:
                sentiment_impact = (Decimal("1") - ratio) / ratio * Decimal("50")
        else:
            sentiment_impact = Decimal("0")

        return LiquidationData(
            symbol=symbol,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            total_liquidations=total,
            liquidation_ratio=ratio,
            sentiment_impact=sentiment_impact
        )


class VolumeProfileAnalyzer:
    """Analyze volume profile for sentiment."""

    def __init__(self, periods: int = 20):
        self.periods = periods
        self.history: dict[str, list[dict]] = {}

    def add_data(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        buy_volume: Optional[Decimal] = None,
        sell_volume: Optional[Decimal] = None
    ):
        """Add volume data."""
        if symbol not in self.history:
            self.history[symbol] = []

        data = {
            "price": price,
            "volume": volume,
            "buy_volume": buy_volume or volume / Decimal("2"),
            "sell_volume": sell_volume or volume / Decimal("2"),
            "timestamp": datetime.now()
        }
        self.history[symbol].append(data)
        if len(self.history[symbol]) > self.periods:
            self.history[symbol].pop(0)

    def get_volume_sentiment(self, symbol: str) -> Optional[Decimal]:
        """Get volume-based sentiment."""
        history = self.history.get(symbol, [])
        if len(history) < 2:
            return None

        total_buy = sum(h["buy_volume"] for h in history)
        total_sell = sum(h["sell_volume"] for h in history)

        if total_buy + total_sell == 0:
            return Decimal("0")

        # Buy/sell imbalance as sentiment
        sentiment = (total_buy - total_sell) / (total_buy + total_sell) * Decimal("100")
        return sentiment

    def get_volume_trend(self, symbol: str) -> Optional[TrendBias]:
        """Get volume trend."""
        history = self.history.get(symbol, [])
        if len(history) < self.periods // 2:
            return None

        mid = len(history) // 2
        recent_vol = sum(h["volume"] for h in history[mid:])
        earlier_vol = sum(h["volume"] for h in history[:mid])

        if earlier_vol == 0:
            return TrendBias.NEUTRAL

        change = (recent_vol - earlier_vol) / earlier_vol

        if change > Decimal("0.3"):  # 30% increase
            return TrendBias.BULLISH
        elif change < Decimal("-0.3"):
            return TrendBias.BEARISH
        return TrendBias.NEUTRAL


class FearGreedCalculator:
    """Calculate Fear and Greed Index."""

    def __init__(self):
        self.history: list[Decimal] = []
        self.components: dict[str, Decimal] = {}

    def update_component(self, name: str, value: Decimal):
        """Update a component value (0-100)."""
        self.components[name] = max(Decimal("0"), min(Decimal("100"), value))

    def calculate(self) -> FearGreedIndex:
        """Calculate Fear and Greed Index."""
        if not self.components:
            return FearGreedIndex(
                value=Decimal("50"),
                level=SentimentLevel.NEUTRAL,
                previous_value=Decimal("50"),
                change=Decimal("0")
            )

        # Average all components
        total = sum(self.components.values())
        value = total / Decimal(str(len(self.components)))

        # Track history
        previous = self.history[-1] if self.history else Decimal("50")
        self.history.append(value)
        if len(self.history) > 100:
            self.history.pop(0)

        # Determine level
        if value <= Decimal("20"):
            level = SentimentLevel.EXTREME_FEAR
        elif value <= Decimal("40"):
            level = SentimentLevel.FEAR
        elif value <= Decimal("60"):
            level = SentimentLevel.NEUTRAL
        elif value <= Decimal("80"):
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED

        return FearGreedIndex(
            value=value,
            level=level,
            previous_value=previous,
            change=value - previous
        )


class SentimentAggregator:
    """Aggregate sentiment from multiple sources."""

    def __init__(self):
        self.weights: dict[SentimentSource, Decimal] = {
            SentimentSource.FUNDING_RATE: Decimal("0.25"),
            SentimentSource.LONG_SHORT_RATIO: Decimal("0.20"),
            SentimentSource.LIQUIDATIONS: Decimal("0.15"),
            SentimentSource.VOLUME: Decimal("0.15"),
            SentimentSource.PRICE_ACTION: Decimal("0.15"),
            SentimentSource.VOLATILITY: Decimal("0.10")
        }
        self.data: dict[str, dict[SentimentSource, Decimal]] = {}

    def update(self, symbol: str, source: SentimentSource, score: Decimal):
        """Update sentiment score for source."""
        if symbol not in self.data:
            self.data[symbol] = {}
        self.data[symbol][source] = score

    def set_weight(self, source: SentimentSource, weight: Decimal):
        """Set weight for source."""
        self.weights[source] = weight

    def aggregate(self, symbol: str) -> SentimentScore:
        """Aggregate all sentiment sources."""
        scores = self.data.get(symbol, {})

        if not scores:
            return SentimentScore(
                symbol=symbol,
                score=Decimal("0"),
                level=SentimentLevel.NEUTRAL,
                bias=TrendBias.NEUTRAL,
                confidence=Decimal("0"),
                sources=[]
            )

        # Weighted average
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for source, score in scores.items():
            weight = self.weights.get(source, Decimal("0.1"))
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = Decimal("0")

        # Confidence based on number of sources
        confidence = Decimal(str(len(scores))) / Decimal(str(len(self.weights))) * Decimal("100")

        # Determine level and bias
        level = self._score_to_level(final_score)
        bias = self._score_to_bias(final_score)

        return SentimentScore(
            symbol=symbol,
            score=final_score,
            level=level,
            bias=bias,
            confidence=confidence,
            sources=list(scores.keys())
        )

    def _score_to_level(self, score: Decimal) -> SentimentLevel:
        """Convert score to sentiment level."""
        if score <= Decimal("-60"):
            return SentimentLevel.EXTREME_FEAR
        elif score <= Decimal("-20"):
            return SentimentLevel.FEAR
        elif score <= Decimal("20"):
            return SentimentLevel.NEUTRAL
        elif score <= Decimal("60"):
            return SentimentLevel.GREED
        return SentimentLevel.EXTREME_GREED

    def _score_to_bias(self, score: Decimal) -> TrendBias:
        """Convert score to trend bias."""
        if score > Decimal("20"):
            return TrendBias.BULLISH
        elif score < Decimal("-20"):
            return TrendBias.BEARISH
        return TrendBias.NEUTRAL


class SignalGenerator:
    """Generate trading signals from sentiment."""

    def __init__(
        self,
        buy_threshold: Decimal = Decimal("-50"),
        sell_threshold: Decimal = Decimal("50"),
        min_confidence: Decimal = Decimal("60")
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence

    def generate(self, sentiment: SentimentScore) -> SentimentSignal:
        """Generate trading signal from sentiment."""
        reasoning = []

        # Determine signal type
        if sentiment.confidence < self.min_confidence:
            signal_type = "hold"
            strength = Decimal("0")
            reasoning.append(f"Insufficient confidence: {sentiment.confidence:.1f}%")
        elif sentiment.score <= self.buy_threshold:
            signal_type = "buy"
            strength = min(Decimal("100"), abs(sentiment.score))
            reasoning.append(f"Extreme fear detected: {sentiment.score:.1f}")
            reasoning.append("Contrarian buy opportunity")
        elif sentiment.score >= self.sell_threshold:
            signal_type = "sell"
            strength = min(Decimal("100"), sentiment.score)
            reasoning.append(f"Extreme greed detected: {sentiment.score:.1f}")
            reasoning.append("Contrarian sell opportunity")
        else:
            signal_type = "hold"
            strength = Decimal("0")
            reasoning.append(f"Neutral sentiment: {sentiment.score:.1f}")

        # Add source info
        reasoning.append(f"Sources: {', '.join(s.value for s in sentiment.sources)}")

        return SentimentSignal(
            symbol=sentiment.symbol,
            signal_type=signal_type,
            strength=strength,
            sentiment_score=sentiment.score,
            reasoning=reasoning
        )


class SentimentAnalyzer:
    """Main sentiment analyzer."""

    def __init__(self):
        self.funding_analyzer = FundingRateAnalyzer()
        self.ls_analyzer = LongShortAnalyzer()
        self.liquidation_analyzer = LiquidationAnalyzer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.fear_greed = FearGreedCalculator()
        self.aggregator = SentimentAggregator()
        self.signal_gen = SignalGenerator()
        self.callbacks: dict[str, list[Callable]] = {
            "on_sentiment_update": [],
            "on_signal": []
        }

    def register_callback(self, event: str, callback: Callable):
        """Register callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def update_funding_rate(self, symbol: str, funding_rate: Decimal):
        """Update funding rate data."""
        result = self.funding_analyzer.analyze(symbol, funding_rate)
        self.aggregator.update(
            symbol,
            SentimentSource.FUNDING_RATE,
            result.sentiment_score
        )
        return result

    def update_long_short_ratio(
        self,
        symbol: str,
        long_ratio: Decimal,
        short_ratio: Decimal
    ):
        """Update long/short ratio."""
        result = self.ls_analyzer.analyze(symbol, long_ratio, short_ratio)
        self.aggregator.update(
            symbol,
            SentimentSource.LONG_SHORT_RATIO,
            result.sentiment_score
        )
        return result

    def update_liquidations(
        self,
        symbol: str,
        long_liquidations: Decimal,
        short_liquidations: Decimal
    ):
        """Update liquidation data."""
        result = self.liquidation_analyzer.analyze(
            symbol, long_liquidations, short_liquidations
        )
        self.aggregator.update(
            symbol,
            SentimentSource.LIQUIDATIONS,
            result.sentiment_impact
        )
        return result

    def update_volume(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        buy_volume: Optional[Decimal] = None,
        sell_volume: Optional[Decimal] = None
    ):
        """Update volume data."""
        self.volume_analyzer.add_data(symbol, price, volume, buy_volume, sell_volume)
        sentiment = self.volume_analyzer.get_volume_sentiment(symbol)
        if sentiment is not None:
            self.aggregator.update(symbol, SentimentSource.VOLUME, sentiment)

    def get_sentiment(self, symbol: str) -> SentimentScore:
        """Get aggregated sentiment for symbol."""
        sentiment = self.aggregator.aggregate(symbol)

        for cb in self.callbacks["on_sentiment_update"]:
            cb(sentiment)

        return sentiment

    def get_signal(self, symbol: str) -> SentimentSignal:
        """Get trading signal based on sentiment."""
        sentiment = self.get_sentiment(symbol)
        signal = self.signal_gen.generate(sentiment)

        for cb in self.callbacks["on_signal"]:
            cb(signal)

        return signal

    def get_fear_greed_index(self) -> FearGreedIndex:
        """Get Fear and Greed Index."""
        return self.fear_greed.calculate()

    def update_fear_greed_component(self, name: str, value: Decimal):
        """Update Fear and Greed component."""
        self.fear_greed.update_component(name, value)

    def get_market_sentiment_summary(self, symbols: list[str]) -> dict:
        """Get summary of market sentiment."""
        sentiments = [self.get_sentiment(s) for s in symbols]

        bullish = sum(1 for s in sentiments if s.bias == TrendBias.BULLISH)
        bearish = sum(1 for s in sentiments if s.bias == TrendBias.BEARISH)
        neutral = sum(1 for s in sentiments if s.bias == TrendBias.NEUTRAL)

        avg_score = sum(s.score for s in sentiments) / Decimal(str(len(sentiments))) if sentiments else Decimal("0")

        return {
            "total_symbols": len(symbols),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "average_score": str(avg_score),
            "market_bias": "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral"),
            "fear_greed": self.get_fear_greed_index().to_dict()
        }


# Global instance
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get global sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def set_sentiment_analyzer(analyzer: SentimentAnalyzer):
    """Set global sentiment analyzer."""
    global _analyzer
    _analyzer = analyzer
