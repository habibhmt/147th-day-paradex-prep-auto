"""
Funding Rate Analyzer Module.

Analyzes funding rates across markets to optimize
delta-neutral strategies and predict funding payments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable


class FundingDirection(Enum):
    """Direction of funding payment."""

    LONG_PAYS_SHORT = "long_pays_short"  # Positive funding
    SHORT_PAYS_LONG = "short_pays_long"  # Negative funding
    NEUTRAL = "neutral"  # Near zero


class FundingTrend(Enum):
    """Trend of funding rate."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"


class FundingRegime(Enum):
    """Funding rate regime."""

    HIGH_POSITIVE = "high_positive"  # > 0.05%
    POSITIVE = "positive"  # 0.01% to 0.05%
    NEUTRAL = "neutral"  # -0.01% to 0.01%
    NEGATIVE = "negative"  # -0.05% to -0.01%
    HIGH_NEGATIVE = "high_negative"  # < -0.05%


@dataclass
class FundingRate:
    """Single funding rate data point."""

    market: str
    rate: Decimal  # Percentage (0.01 = 0.01%)
    timestamp: datetime
    next_funding_time: datetime | None = None
    predicted_rate: Decimal | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None

    @property
    def rate_bps(self) -> float:
        """Funding rate in basis points."""
        return float(self.rate * 10000)

    @property
    def direction(self) -> FundingDirection:
        """Get funding direction."""
        if self.rate > Decimal("0.001"):
            return FundingDirection.LONG_PAYS_SHORT
        elif self.rate < Decimal("-0.001"):
            return FundingDirection.SHORT_PAYS_LONG
        return FundingDirection.NEUTRAL

    @property
    def annualized_rate(self) -> float:
        """Calculate annualized funding rate (assuming 8h intervals)."""
        return float(self.rate * 3 * 365 * 100)  # 3 times per day * 365 days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "rate": float(self.rate),
            "rate_bps": self.rate_bps,
            "timestamp": self.timestamp.isoformat(),
            "next_funding_time": self.next_funding_time.isoformat() if self.next_funding_time else None,
            "predicted_rate": float(self.predicted_rate) if self.predicted_rate else None,
            "direction": self.direction.value,
            "annualized_rate": self.annualized_rate,
        }


@dataclass
class FundingHistory:
    """Historical funding rate data."""

    market: str
    rates: list[FundingRate]
    start_time: datetime
    end_time: datetime

    @property
    def avg_rate(self) -> Decimal:
        """Average funding rate."""
        if not self.rates:
            return Decimal("0")
        return sum(r.rate for r in self.rates) / len(self.rates)

    @property
    def max_rate(self) -> Decimal:
        """Maximum funding rate."""
        if not self.rates:
            return Decimal("0")
        return max(r.rate for r in self.rates)

    @property
    def min_rate(self) -> Decimal:
        """Minimum funding rate."""
        if not self.rates:
            return Decimal("0")
        return min(r.rate for r in self.rates)

    @property
    def cumulative_rate(self) -> Decimal:
        """Cumulative funding rate over period."""
        return sum(r.rate for r in self.rates)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "rates_count": len(self.rates),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "avg_rate": float(self.avg_rate),
            "max_rate": float(self.max_rate),
            "min_rate": float(self.min_rate),
            "cumulative_rate": float(self.cumulative_rate),
        }


@dataclass
class FundingPayment:
    """Estimated funding payment."""

    market: str
    position_size: Decimal
    position_side: str  # 'long' or 'short'
    funding_rate: Decimal
    payment: Decimal  # Positive = receive, negative = pay
    timestamp: datetime

    @property
    def is_receiving(self) -> bool:
        """Whether the position receives funding."""
        return self.payment > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "position_size": float(self.position_size),
            "position_side": self.position_side,
            "funding_rate": float(self.funding_rate),
            "payment": float(self.payment),
            "is_receiving": self.is_receiving,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FundingOpportunity:
    """Funding rate arbitrage/optimization opportunity."""

    market: str
    current_rate: Decimal
    avg_rate_24h: Decimal
    predicted_rate: Decimal
    recommended_side: str  # 'long' or 'short'
    expected_payment_8h: Decimal
    expected_payment_24h: Decimal
    confidence: float
    regime: FundingRegime
    trend: FundingTrend
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "current_rate": float(self.current_rate),
            "avg_rate_24h": float(self.avg_rate_24h),
            "predicted_rate": float(self.predicted_rate),
            "recommended_side": self.recommended_side,
            "expected_payment_8h": float(self.expected_payment_8h),
            "expected_payment_24h": float(self.expected_payment_24h),
            "confidence": self.confidence,
            "regime": self.regime.value,
            "trend": self.trend.value,
            "notes": self.notes,
        }


@dataclass
class FundingAnalysis:
    """Complete funding rate analysis for a market."""

    market: str
    timestamp: datetime
    current_rate: FundingRate
    regime: FundingRegime
    trend: FundingTrend
    avg_rate_1h: Decimal
    avg_rate_4h: Decimal
    avg_rate_24h: Decimal
    avg_rate_7d: Decimal
    volatility: float  # Std dev of rates
    mean_reversion_score: float  # -1 to 1
    opportunity: FundingOpportunity | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp.isoformat(),
            "current_rate": self.current_rate.to_dict(),
            "regime": self.regime.value,
            "trend": self.trend.value,
            "avg_rate_1h": float(self.avg_rate_1h),
            "avg_rate_4h": float(self.avg_rate_4h),
            "avg_rate_24h": float(self.avg_rate_24h),
            "avg_rate_7d": float(self.avg_rate_7d),
            "volatility": self.volatility,
            "mean_reversion_score": self.mean_reversion_score,
            "opportunity": self.opportunity.to_dict() if self.opportunity else None,
        }


@dataclass
class CrossMarketFunding:
    """Cross-market funding comparison."""

    timestamp: datetime
    markets: list[str]
    rates: dict[str, Decimal]
    best_long_market: str  # Best market for long (lowest funding)
    best_short_market: str  # Best market for short (highest funding)
    spread: Decimal  # Difference between best markets
    opportunities: list[FundingOpportunity]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "markets": self.markets,
            "rates": {k: float(v) for k, v in self.rates.items()},
            "best_long_market": self.best_long_market,
            "best_short_market": self.best_short_market,
            "spread": float(self.spread),
            "opportunities": [o.to_dict() for o in self.opportunities],
        }


class FundingRateAnalyzer:
    """Analyzes funding rates for delta-neutral strategies."""

    def __init__(
        self,
        history_window_hours: int = 168,  # 7 days
        prediction_lookback: int = 24,  # 24 intervals
        high_funding_threshold: Decimal = Decimal("0.0005"),  # 0.05%
        min_opportunity_threshold: Decimal = Decimal("0.0001"),  # 0.01%
    ):
        """
        Initialize analyzer.

        Args:
            history_window_hours: Hours of history to maintain
            prediction_lookback: Number of intervals for prediction
            high_funding_threshold: Threshold for high funding
            min_opportunity_threshold: Minimum rate for opportunity
        """
        self.history_window_hours = history_window_hours
        self.prediction_lookback = prediction_lookback
        self.high_funding_threshold = high_funding_threshold
        self.min_opportunity_threshold = min_opportunity_threshold

        # Store funding rates by market
        self._rates: dict[str, list[FundingRate]] = {}

        # Callbacks
        self._callbacks: list[Callable[[str, FundingAnalysis], None]] = []

        # Cache
        self._analysis_cache: dict[str, FundingAnalysis] = {}

    def add_rate(self, rate: FundingRate) -> None:
        """
        Add a funding rate observation.

        Args:
            rate: Funding rate data
        """
        if rate.market not in self._rates:
            self._rates[rate.market] = []

        self._rates[rate.market].append(rate)

        # Sort by timestamp
        self._rates[rate.market].sort(key=lambda x: x.timestamp)

        # Trim old data
        cutoff = datetime.now() - timedelta(hours=self.history_window_hours)
        self._rates[rate.market] = [r for r in self._rates[rate.market] if r.timestamp > cutoff]

        # Invalidate cache
        self._analysis_cache.pop(rate.market, None)

    def update_rates(
        self,
        market: str,
        rates: list[tuple[datetime, Decimal]],
    ) -> None:
        """
        Update funding rates for a market.

        Args:
            market: Market symbol
            rates: List of (timestamp, rate) tuples
        """
        for timestamp, rate_value in rates:
            rate = FundingRate(
                market=market,
                rate=rate_value,
                timestamp=timestamp,
            )
            self.add_rate(rate)

    def get_current_rate(self, market: str) -> FundingRate | None:
        """
        Get current funding rate for a market.

        Args:
            market: Market symbol

        Returns:
            Current funding rate or None
        """
        if market not in self._rates or not self._rates[market]:
            return None
        return self._rates[market][-1]

    def get_history(
        self,
        market: str,
        hours: int | None = None,
    ) -> FundingHistory | None:
        """
        Get funding rate history.

        Args:
            market: Market symbol
            hours: Hours of history (None for all)

        Returns:
            Funding history or None
        """
        if market not in self._rates or not self._rates[market]:
            return None

        rates = self._rates[market]

        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            rates = [r for r in rates if r.timestamp > cutoff]

        if not rates:
            return None

        return FundingHistory(
            market=market,
            rates=rates,
            start_time=rates[0].timestamp,
            end_time=rates[-1].timestamp,
        )

    def detect_regime(self, market: str) -> FundingRegime:
        """
        Detect current funding regime.

        Args:
            market: Market symbol

        Returns:
            Funding regime
        """
        rate = self.get_current_rate(market)
        if not rate:
            return FundingRegime.NEUTRAL

        if rate.rate > self.high_funding_threshold:
            return FundingRegime.HIGH_POSITIVE
        elif rate.rate > Decimal("0.0001"):
            return FundingRegime.POSITIVE
        elif rate.rate < -self.high_funding_threshold:
            return FundingRegime.HIGH_NEGATIVE
        elif rate.rate < Decimal("-0.0001"):
            return FundingRegime.NEGATIVE
        return FundingRegime.NEUTRAL

    def detect_trend(self, market: str, lookback_count: int = 6) -> FundingTrend:
        """
        Detect funding rate trend.

        Args:
            market: Market symbol
            lookback_count: Number of recent rates to analyze

        Returns:
            Funding trend
        """
        if market not in self._rates:
            return FundingTrend.STABLE

        rates = self._rates[market][-lookback_count:]
        if len(rates) < 3:
            return FundingTrend.STABLE

        # Calculate changes
        changes = []
        for i in range(1, len(rates)):
            change = rates[i].rate - rates[i - 1].rate
            changes.append(change)

        if not changes:
            return FundingTrend.STABLE

        avg_change = sum(changes) / len(changes)

        # Check for volatility
        import math
        variance = sum((c - avg_change) ** 2 for c in changes) / len(changes)
        std_dev = Decimal(str(math.sqrt(float(variance))))

        if std_dev > Decimal("0.0002"):
            return FundingTrend.VOLATILE

        if avg_change > Decimal("0.00005"):
            return FundingTrend.RISING
        elif avg_change < Decimal("-0.00005"):
            return FundingTrend.FALLING
        return FundingTrend.STABLE

    def calculate_average(
        self,
        market: str,
        hours: int,
    ) -> Decimal:
        """
        Calculate average funding rate over period.

        Args:
            market: Market symbol
            hours: Hours to average

        Returns:
            Average funding rate
        """
        history = self.get_history(market, hours)
        if not history:
            return Decimal("0")
        return history.avg_rate

    def predict_rate(self, market: str) -> Decimal:
        """
        Predict next funding rate.

        Uses simple moving average with mean reversion.

        Args:
            market: Market symbol

        Returns:
            Predicted funding rate
        """
        if market not in self._rates:
            return Decimal("0")

        rates = self._rates[market][-self.prediction_lookback:]
        if len(rates) < 3:
            return rates[-1].rate if rates else Decimal("0")

        # Simple EMA
        alpha = Decimal("0.3")
        ema = rates[0].rate
        for r in rates[1:]:
            ema = alpha * r.rate + (1 - alpha) * ema

        # Add mean reversion towards zero
        mean_reversion = -ema * Decimal("0.1")

        return ema + mean_reversion

    def calculate_volatility(self, market: str, hours: int = 24) -> float:
        """
        Calculate funding rate volatility.

        Args:
            market: Market symbol
            hours: Hours of history

        Returns:
            Standard deviation of rates
        """
        history = self.get_history(market, hours)
        if not history or len(history.rates) < 2:
            return 0.0

        rates = [float(r.rate) for r in history.rates]
        mean = sum(rates) / len(rates)

        import math
        variance = sum((r - mean) ** 2 for r in rates) / len(rates)
        return math.sqrt(variance)

    def calculate_mean_reversion_score(self, market: str) -> float:
        """
        Calculate mean reversion score.

        -1 = strongly mean reverting
        0 = random
        1 = strongly trending

        Args:
            market: Market symbol

        Returns:
            Mean reversion score
        """
        if market not in self._rates:
            return 0.0

        rates = self._rates[market][-24:]  # Last 24 intervals
        if len(rates) < 5:
            return 0.0

        # Calculate autocorrelation
        rate_values = [float(r.rate) for r in rates]
        mean = sum(rate_values) / len(rate_values)
        centered = [r - mean for r in rate_values]

        if len(centered) < 2:
            return 0.0

        # Lag-1 autocorrelation
        numerator = sum(centered[i] * centered[i + 1] for i in range(len(centered) - 1))
        denominator = sum(c ** 2 for c in centered)

        if denominator == 0:
            return 0.0

        autocorr = numerator / denominator

        # Convert to mean reversion score
        # Negative autocorr = mean reverting
        return -autocorr

    def estimate_payment(
        self,
        market: str,
        position_size: Decimal,
        position_side: str,
        mark_price: Decimal | None = None,
    ) -> FundingPayment | None:
        """
        Estimate funding payment for a position.

        Args:
            market: Market symbol
            position_size: Position size in base currency
            position_side: 'long' or 'short'
            mark_price: Mark price (uses last known if None)

        Returns:
            Funding payment estimate or None
        """
        rate = self.get_current_rate(market)
        if not rate:
            return None

        # Get mark price
        if mark_price is None:
            mark_price = rate.mark_price or Decimal("1")

        # Calculate notional value
        notional = position_size * mark_price

        # Calculate payment
        # Long pays when rate is positive, receives when negative
        # Short pays when rate is negative, receives when positive
        if position_side == "long":
            payment = -notional * rate.rate
        else:  # short
            payment = notional * rate.rate

        return FundingPayment(
            market=market,
            position_size=position_size,
            position_side=position_side,
            funding_rate=rate.rate,
            payment=payment,
            timestamp=datetime.now(),
        )

    def find_opportunity(
        self,
        market: str,
        position_size: Decimal = Decimal("1000"),
    ) -> FundingOpportunity | None:
        """
        Find funding rate opportunity.

        Args:
            market: Market symbol
            position_size: Reference position size

        Returns:
            Opportunity or None
        """
        rate = self.get_current_rate(market)
        if not rate:
            return None

        # Get averages
        avg_24h = self.calculate_average(market, 24)
        predicted = self.predict_rate(market)
        regime = self.detect_regime(market)
        trend = self.detect_trend(market)

        # Determine recommended side
        if rate.rate > self.min_opportunity_threshold:
            recommended_side = "short"  # Receive funding
            expected_8h = position_size * rate.rate
        elif rate.rate < -self.min_opportunity_threshold:
            recommended_side = "long"  # Receive funding
            expected_8h = position_size * abs(rate.rate)
        else:
            return None  # No clear opportunity

        expected_24h = expected_8h * 3

        # Calculate confidence
        confidence = 0.5

        # Higher confidence if regime and trend align
        if regime in [FundingRegime.HIGH_POSITIVE, FundingRegime.HIGH_NEGATIVE]:
            confidence += 0.2

        if trend == FundingTrend.STABLE:
            confidence += 0.1

        # Lower confidence if volatile
        if trend == FundingTrend.VOLATILE:
            confidence -= 0.2

        confidence = max(0.1, min(0.95, confidence))

        return FundingOpportunity(
            market=market,
            current_rate=rate.rate,
            avg_rate_24h=avg_24h,
            predicted_rate=predicted,
            recommended_side=recommended_side,
            expected_payment_8h=expected_8h,
            expected_payment_24h=expected_24h,
            confidence=confidence,
            regime=regime,
            trend=trend,
        )

    def analyze(self, market: str) -> FundingAnalysis | None:
        """
        Complete funding analysis for a market.

        Args:
            market: Market symbol

        Returns:
            Complete analysis or None
        """
        if market in self._analysis_cache:
            return self._analysis_cache[market]

        rate = self.get_current_rate(market)
        if not rate:
            return None

        regime = self.detect_regime(market)
        trend = self.detect_trend(market)
        opportunity = self.find_opportunity(market)

        analysis = FundingAnalysis(
            market=market,
            timestamp=datetime.now(),
            current_rate=rate,
            regime=regime,
            trend=trend,
            avg_rate_1h=self.calculate_average(market, 1),
            avg_rate_4h=self.calculate_average(market, 4),
            avg_rate_24h=self.calculate_average(market, 24),
            avg_rate_7d=self.calculate_average(market, 168),
            volatility=self.calculate_volatility(market),
            mean_reversion_score=self.calculate_mean_reversion_score(market),
            opportunity=opportunity,
        )

        self._analysis_cache[market] = analysis

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(market, analysis)
            except Exception:
                pass

        return analysis

    def compare_markets(
        self,
        markets: list[str] | None = None,
    ) -> CrossMarketFunding | None:
        """
        Compare funding rates across markets.

        Args:
            markets: Markets to compare (all if None)

        Returns:
            Cross-market comparison or None
        """
        if markets is None:
            markets = list(self._rates.keys())

        if not markets:
            return None

        rates: dict[str, Decimal] = {}
        opportunities: list[FundingOpportunity] = []

        for market in markets:
            rate = self.get_current_rate(market)
            if rate:
                rates[market] = rate.rate
                opp = self.find_opportunity(market)
                if opp:
                    opportunities.append(opp)

        if not rates:
            return None

        # Find best markets
        sorted_by_rate = sorted(rates.items(), key=lambda x: x[1])
        best_long = sorted_by_rate[0][0]  # Lowest rate (pay less when long)
        best_short = sorted_by_rate[-1][0]  # Highest rate (receive more when short)

        spread = rates[best_short] - rates[best_long]

        return CrossMarketFunding(
            timestamp=datetime.now(),
            markets=markets,
            rates=rates,
            best_long_market=best_long,
            best_short_market=best_short,
            spread=spread,
            opportunities=opportunities,
        )

    def get_best_funding_market(
        self,
        side: str,
        markets: list[str] | None = None,
    ) -> str | None:
        """
        Get best market for a position side.

        Args:
            side: 'long' or 'short'
            markets: Markets to consider

        Returns:
            Best market symbol or None
        """
        comparison = self.compare_markets(markets)
        if not comparison:
            return None

        if side == "long":
            return comparison.best_long_market
        return comparison.best_short_market

    def add_callback(self, callback: Callable[[str, FundingAnalysis], None]) -> None:
        """Add analysis callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, FundingAnalysis], None]) -> bool:
        """Remove analysis callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def get_markets(self) -> list[str]:
        """Get list of markets with data."""
        return list(self._rates.keys())

    def clear_market(self, market: str) -> None:
        """Clear data for a market."""
        self._rates.pop(market, None)
        self._analysis_cache.pop(market, None)

    def clear_all(self) -> None:
        """Clear all data."""
        self._rates.clear()
        self._analysis_cache.clear()


# Global instance
_analyzer: FundingRateAnalyzer | None = None


def get_funding_analyzer() -> FundingRateAnalyzer:
    """Get global funding analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FundingRateAnalyzer()
    return _analyzer


def reset_funding_analyzer() -> None:
    """Reset global funding analyzer."""
    global _analyzer
    _analyzer = None
