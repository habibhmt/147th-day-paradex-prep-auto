"""Market Correlations module for cross-market analysis.

This module provides comprehensive correlation analysis including:
- Correlation matrix between markets
- Rolling correlations with configurable windows
- Lead-lag analysis between markets
- Cointegration testing for pairs
- Market relationship detection
- Pair selection for delta-neutral strategies
- Regime-based correlation analysis
"""

import time
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class CorrelationType(Enum):
    """Type of correlation calculation."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class RelationshipStrength(Enum):
    """Strength of market relationship."""

    VERY_STRONG = "very_strong"  # |r| > 0.8
    STRONG = "strong"  # |r| > 0.6
    MODERATE = "moderate"  # |r| > 0.4
    WEAK = "weak"  # |r| > 0.2
    NEGLIGIBLE = "negligible"  # |r| <= 0.2


class RelationshipDirection(Enum):
    """Direction of market relationship."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CointegrationResult(Enum):
    """Result of cointegration test."""

    COINTEGRATED = "cointegrated"
    NOT_COINTEGRATED = "not_cointegrated"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PriceData:
    """Price data point for a market."""

    market: str
    timestamp: float
    price: Decimal
    volume: Decimal = Decimal("0")
    returns: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "timestamp": self.timestamp,
            "price": str(self.price),
            "volume": str(self.volume),
            "returns": self.returns,
        }


@dataclass
class CorrelationPair:
    """Correlation between two markets."""

    market_a: str
    market_b: str
    correlation: float
    p_value: float = 0.0
    sample_size: int = 0
    correlation_type: CorrelationType = CorrelationType.PEARSON
    timestamp: float = field(default_factory=time.time)

    @property
    def strength(self) -> RelationshipStrength:
        """Get relationship strength."""
        abs_corr = abs(self.correlation)
        if abs_corr > 0.8:
            return RelationshipStrength.VERY_STRONG
        elif abs_corr > 0.6:
            return RelationshipStrength.STRONG
        elif abs_corr > 0.4:
            return RelationshipStrength.MODERATE
        elif abs_corr > 0.2:
            return RelationshipStrength.WEAK
        else:
            return RelationshipStrength.NEGLIGIBLE

    @property
    def direction(self) -> RelationshipDirection:
        """Get relationship direction."""
        if self.correlation > 0.1:
            return RelationshipDirection.POSITIVE
        elif self.correlation < -0.1:
            return RelationshipDirection.NEGATIVE
        else:
            return RelationshipDirection.NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "sample_size": self.sample_size,
            "strength": self.strength.value,
            "direction": self.direction.value,
            "correlation_type": self.correlation_type.value,
        }


@dataclass
class RollingCorrelation:
    """Rolling correlation between two markets."""

    market_a: str
    market_b: str
    window_size: int
    correlations: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def current(self) -> float:
        """Get current correlation."""
        return self.correlations[-1] if self.correlations else 0.0

    @property
    def mean(self) -> float:
        """Get mean correlation."""
        return statistics.mean(self.correlations) if self.correlations else 0.0

    @property
    def std(self) -> float:
        """Get standard deviation of correlations."""
        return statistics.stdev(self.correlations) if len(self.correlations) > 1 else 0.0

    @property
    def min(self) -> float:
        """Get minimum correlation."""
        return min(self.correlations) if self.correlations else 0.0

    @property
    def max(self) -> float:
        """Get maximum correlation."""
        return max(self.correlations) if self.correlations else 0.0

    def is_stable(self, threshold: float = 0.2) -> bool:
        """Check if correlation is stable."""
        return self.std < threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "window_size": self.window_size,
            "current": self.current,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "is_stable": self.is_stable(),
            "sample_count": len(self.correlations),
        }


@dataclass
class LeadLagAnalysis:
    """Lead-lag analysis between two markets."""

    leader: str
    follower: str
    lag_periods: int
    correlation_at_lag: float
    optimal_lag: int = 0
    max_correlation: float = 0.0
    lag_correlations: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "leader": self.leader,
            "follower": self.follower,
            "lag_periods": self.lag_periods,
            "correlation_at_lag": self.correlation_at_lag,
            "optimal_lag": self.optimal_lag,
            "max_correlation": self.max_correlation,
            "lag_correlations": self.lag_correlations,
        }


@dataclass
class CointegrationTest:
    """Cointegration test result."""

    market_a: str
    market_b: str
    test_statistic: float = 0.0
    critical_value_1pct: float = -3.43
    critical_value_5pct: float = -2.86
    critical_value_10pct: float = -2.57
    p_value: float = 1.0
    result: CointegrationResult = CointegrationResult.INCONCLUSIVE
    hedge_ratio: float = 1.0
    spread_mean: float = 0.0
    spread_std: float = 1.0

    @property
    def is_cointegrated(self) -> bool:
        """Check if pair is cointegrated at 5% level."""
        return self.result == CointegrationResult.COINTEGRATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "result": self.result.value,
            "is_cointegrated": self.is_cointegrated,
            "hedge_ratio": self.hedge_ratio,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
        }


@dataclass
class CorrelationMatrix:
    """Correlation matrix for multiple markets."""

    markets: List[str] = field(default_factory=list)
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_correlation(self, market_a: str, market_b: str) -> float:
        """Get correlation between two markets."""
        if market_a == market_b:
            return 1.0
        if market_a in self.matrix and market_b in self.matrix[market_a]:
            return self.matrix[market_a][market_b]
        if market_b in self.matrix and market_a in self.matrix[market_b]:
            return self.matrix[market_b][market_a]
        return 0.0

    def get_most_correlated(self, market: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get most correlated markets."""
        correlations = []
        for other in self.markets:
            if other != market:
                corr = self.get_correlation(market, other)
                correlations.append((other, abs(corr)))
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations[:n]

    def get_least_correlated(self, market: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get least correlated markets."""
        correlations = []
        for other in self.markets:
            if other != market:
                corr = self.get_correlation(market, other)
                correlations.append((other, abs(corr)))
        correlations.sort(key=lambda x: x[1])
        return correlations[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "markets": self.markets,
            "matrix": self.matrix,
            "timestamp": self.timestamp,
        }


@dataclass
class PairScore:
    """Score for a potential trading pair."""

    market_a: str
    market_b: str
    correlation_score: float = 0.0
    cointegration_score: float = 0.0
    stability_score: float = 0.0
    liquidity_score: float = 0.0
    total_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "correlation_score": self.correlation_score,
            "cointegration_score": self.cointegration_score,
            "stability_score": self.stability_score,
            "liquidity_score": self.liquidity_score,
            "total_score": self.total_score,
            "rank": self.rank,
        }


class CorrelationCalculator:
    """Calculator for market correlations."""

    def __init__(self, min_sample_size: int = 30):
        """Initialize calculator.

        Args:
            min_sample_size: Minimum samples for valid correlation
        """
        self.min_sample_size = min_sample_size

    def calculate_pearson(
        self,
        x: List[float],
        y: List[float],
    ) -> Tuple[float, float]:
        """Calculate Pearson correlation.

        Args:
            x: First series
            y: Second series

        Returns:
            Tuple of (correlation, p_value)
        """
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0, 1.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
        denominator_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

        if denominator_x == 0 or denominator_y == 0:
            return 0.0, 1.0

        correlation = numerator / (denominator_x * denominator_y)

        # Approximate p-value using t-distribution
        if abs(correlation) >= 0.9999:
            p_value = 0.0
        else:
            denom = 1 - correlation ** 2
            if denom <= 0:
                p_value = 0.0
            else:
                t_stat = correlation * math.sqrt(n - 2) / math.sqrt(denom)
                # Simplified p-value approximation
                p_value = 2 * (1 - min(0.9999, abs(t_stat) / (abs(t_stat) + n)))

        return correlation, p_value

    def calculate_spearman(
        self,
        x: List[float],
        y: List[float],
    ) -> Tuple[float, float]:
        """Calculate Spearman rank correlation.

        Args:
            x: First series
            y: Second series

        Returns:
            Tuple of (correlation, p_value)
        """
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0, 1.0

        # Calculate ranks
        rank_x = self._get_ranks(x)
        rank_y = self._get_ranks(y)

        return self.calculate_pearson(rank_x, rank_y)

    def _get_ranks(self, data: List[float]) -> List[float]:
        """Get ranks for data."""
        n = len(data)
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank + 1)
        return ranks

    def calculate_kendall(
        self,
        x: List[float],
        y: List[float],
    ) -> Tuple[float, float]:
        """Calculate Kendall's tau correlation.

        Args:
            x: First series
            y: Second series

        Returns:
            Tuple of (correlation, p_value)
        """
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0, 1.0

        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                x_diff = x[i] - x[j]
                y_diff = y[i] - y[j]
                product = x_diff * y_diff
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.0, 1.0

        tau = (concordant - discordant) / total

        # Simplified p-value
        p_value = 1.0 - min(0.99, abs(tau) * n / 10)

        return tau, p_value


class MarketCorrelationAnalyzer:
    """Analyzer for market correlations."""

    def __init__(
        self,
        window_size: int = 100,
        min_sample_size: int = 30,
    ):
        """Initialize analyzer.

        Args:
            window_size: Default window size for rolling calculations
            min_sample_size: Minimum samples for valid correlation
        """
        self.window_size = window_size
        self.min_sample_size = min_sample_size
        self._calculator = CorrelationCalculator(min_sample_size)
        self._price_data: Dict[str, List[PriceData]] = {}
        self._returns_data: Dict[str, List[float]] = {}
        self._correlation_cache: Dict[str, CorrelationPair] = {}
        self._rolling_cache: Dict[str, RollingCorrelation] = {}
        self._callbacks: List[Callable] = []

    def add_price(
        self,
        market: str,
        timestamp: float,
        price: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> Optional[float]:
        """Add price data point.

        Args:
            market: Market symbol
            timestamp: Timestamp
            price: Price value
            volume: Volume

        Returns:
            Return if calculable
        """
        if market not in self._price_data:
            self._price_data[market] = []
            self._returns_data[market] = []

        returns = 0.0
        if self._price_data[market]:
            last_price = self._price_data[market][-1].price
            if last_price > 0:
                returns = float((price - last_price) / last_price)

        data = PriceData(
            market=market,
            timestamp=timestamp,
            price=price,
            volume=volume,
            returns=returns,
        )

        self._price_data[market].append(data)
        if returns != 0.0 or self._returns_data[market]:
            self._returns_data[market].append(returns)

        # Trim to window size
        max_size = self.window_size * 2
        if len(self._price_data[market]) > max_size:
            self._price_data[market] = self._price_data[market][-max_size:]
            self._returns_data[market] = self._returns_data[market][-max_size:]

        return returns

    def add_prices_batch(
        self,
        prices: Dict[str, List[Tuple[float, Decimal]]],
    ) -> None:
        """Add batch of prices.

        Args:
            prices: Dict of market -> [(timestamp, price), ...]
        """
        for market, price_list in prices.items():
            for timestamp, price in price_list:
                self.add_price(market, timestamp, price)

    def calculate_correlation(
        self,
        market_a: str,
        market_b: str,
        correlation_type: CorrelationType = CorrelationType.PEARSON,
    ) -> CorrelationPair:
        """Calculate correlation between two markets.

        Args:
            market_a: First market
            market_b: Second market
            correlation_type: Type of correlation

        Returns:
            Correlation pair
        """
        returns_a = self._returns_data.get(market_a, [])
        returns_b = self._returns_data.get(market_b, [])

        min_len = min(len(returns_a), len(returns_b))

        if min_len < self.min_sample_size:
            return CorrelationPair(
                market_a=market_a,
                market_b=market_b,
                correlation=0.0,
                sample_size=min_len,
                correlation_type=correlation_type,
            )

        # Use most recent data
        x = returns_a[-min_len:]
        y = returns_b[-min_len:]

        if correlation_type == CorrelationType.PEARSON:
            corr, p_value = self._calculator.calculate_pearson(x, y)
        elif correlation_type == CorrelationType.SPEARMAN:
            corr, p_value = self._calculator.calculate_spearman(x, y)
        else:
            corr, p_value = self._calculator.calculate_kendall(x, y)

        result = CorrelationPair(
            market_a=market_a,
            market_b=market_b,
            correlation=corr,
            p_value=p_value,
            sample_size=min_len,
            correlation_type=correlation_type,
        )

        cache_key = f"{market_a}_{market_b}_{correlation_type.value}"
        self._correlation_cache[cache_key] = result

        return result

    def calculate_correlation_matrix(
        self,
        markets: Optional[List[str]] = None,
        correlation_type: CorrelationType = CorrelationType.PEARSON,
    ) -> CorrelationMatrix:
        """Calculate correlation matrix.

        Args:
            markets: Markets to include (all if None)
            correlation_type: Type of correlation

        Returns:
            Correlation matrix
        """
        if markets is None:
            markets = list(self._price_data.keys())

        matrix = CorrelationMatrix(markets=markets)

        for i, market_a in enumerate(markets):
            matrix.matrix[market_a] = {}
            for j, market_b in enumerate(markets):
                if i <= j:
                    if market_a == market_b:
                        matrix.matrix[market_a][market_b] = 1.0
                    else:
                        pair = self.calculate_correlation(market_a, market_b, correlation_type)
                        matrix.matrix[market_a][market_b] = pair.correlation

        return matrix

    def calculate_rolling_correlation(
        self,
        market_a: str,
        market_b: str,
        window_size: Optional[int] = None,
    ) -> RollingCorrelation:
        """Calculate rolling correlation.

        Args:
            market_a: First market
            market_b: Second market
            window_size: Window size (default: self.window_size)

        Returns:
            Rolling correlation
        """
        window = window_size or self.window_size

        returns_a = self._returns_data.get(market_a, [])
        returns_b = self._returns_data.get(market_b, [])

        min_len = min(len(returns_a), len(returns_b))

        result = RollingCorrelation(
            market_a=market_a,
            market_b=market_b,
            window_size=window,
        )

        if min_len < window:
            return result

        for i in range(window, min_len + 1):
            x = returns_a[i - window:i]
            y = returns_b[i - window:i]
            corr, _ = self._calculator.calculate_pearson(x, y)
            result.correlations.append(corr)

            # Use timestamp from market_a if available
            if self._price_data.get(market_a) and len(self._price_data[market_a]) >= i:
                result.timestamps.append(self._price_data[market_a][i - 1].timestamp)

        cache_key = f"{market_a}_{market_b}_rolling_{window}"
        self._rolling_cache[cache_key] = result

        return result

    def analyze_lead_lag(
        self,
        market_a: str,
        market_b: str,
        max_lag: int = 10,
    ) -> LeadLagAnalysis:
        """Analyze lead-lag relationship.

        Args:
            market_a: First market
            market_b: Second market
            max_lag: Maximum lag to test

        Returns:
            Lead-lag analysis
        """
        returns_a = self._returns_data.get(market_a, [])
        returns_b = self._returns_data.get(market_b, [])

        min_len = min(len(returns_a), len(returns_b))

        lag_correlations = {}
        max_corr = 0.0
        optimal_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # market_a leads (market_b lags behind)
                x = returns_a[:min_len + lag]
                y = returns_b[-lag:min_len]
            elif lag > 0:
                # market_b leads (market_a lags behind)
                x = returns_a[lag:min_len]
                y = returns_b[:min_len - lag]
            else:
                x = returns_a[:min_len]
                y = returns_b[:min_len]

            if len(x) >= self.min_sample_size:
                corr, _ = self._calculator.calculate_pearson(x, y)
                lag_correlations[lag] = corr

                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    optimal_lag = lag

        if optimal_lag < 0:
            leader = market_a
            follower = market_b
        elif optimal_lag > 0:
            leader = market_b
            follower = market_a
        else:
            leader = market_a
            follower = market_b

        return LeadLagAnalysis(
            leader=leader,
            follower=follower,
            lag_periods=abs(optimal_lag),
            correlation_at_lag=max_corr,
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
            lag_correlations=lag_correlations,
        )

    def test_cointegration(
        self,
        market_a: str,
        market_b: str,
    ) -> CointegrationTest:
        """Test for cointegration between markets.

        Args:
            market_a: First market
            market_b: Second market

        Returns:
            Cointegration test result
        """
        prices_a = [float(p.price) for p in self._price_data.get(market_a, [])]
        prices_b = [float(p.price) for p in self._price_data.get(market_b, [])]

        min_len = min(len(prices_a), len(prices_b))

        if min_len < self.min_sample_size:
            return CointegrationTest(
                market_a=market_a,
                market_b=market_b,
                result=CointegrationResult.INCONCLUSIVE,
            )

        x = prices_a[-min_len:]
        y = prices_b[-min_len:]

        # Simple linear regression for hedge ratio
        mean_x = sum(x) / min_len
        mean_y = sum(y) / min_len

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(min_len))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(min_len))

        if denominator == 0:
            hedge_ratio = 1.0
        else:
            hedge_ratio = numerator / denominator

        # Calculate spread
        spread = [y[i] - hedge_ratio * x[i] for i in range(min_len)]
        spread_mean = sum(spread) / min_len
        spread_var = sum((s - spread_mean) ** 2 for s in spread) / min_len
        spread_std = math.sqrt(spread_var) if spread_var > 0 else 1.0

        # Simplified ADF test statistic approximation
        # In practice, use statsmodels.tsa.stattools.adfuller
        dspread = [spread[i] - spread[i - 1] for i in range(1, min_len)]
        spread_lag = spread[:-1]

        if len(dspread) < 10:
            return CointegrationTest(
                market_a=market_a,
                market_b=market_b,
                result=CointegrationResult.INCONCLUSIVE,
            )

        # OLS: dspread = alpha + beta * spread_lag + error
        mean_dspread = sum(dspread) / len(dspread)
        mean_lag = sum(spread_lag) / len(spread_lag)

        num = sum((dspread[i] - mean_dspread) * (spread_lag[i] - mean_lag) for i in range(len(dspread)))
        den = sum((spread_lag[i] - mean_lag) ** 2 for i in range(len(dspread)))

        if den == 0:
            beta = 0.0
        else:
            beta = num / den

        # Residuals and standard error
        alpha = mean_dspread - beta * mean_lag
        residuals = [dspread[i] - alpha - beta * spread_lag[i] for i in range(len(dspread))]
        sse = sum(r ** 2 for r in residuals)
        mse = sse / (len(residuals) - 2) if len(residuals) > 2 else 1.0
        se_beta = math.sqrt(mse / den) if den > 0 and mse > 0 else 1.0

        if se_beta == 0:
            test_stat = 0.0
        else:
            test_stat = beta / se_beta

        # Determine result based on critical values
        if test_stat < -3.43:
            result = CointegrationResult.COINTEGRATED
            p_value = 0.01
        elif test_stat < -2.86:
            result = CointegrationResult.COINTEGRATED
            p_value = 0.05
        elif test_stat < -2.57:
            result = CointegrationResult.INCONCLUSIVE
            p_value = 0.1
        else:
            result = CointegrationResult.NOT_COINTEGRATED
            p_value = 0.5

        return CointegrationTest(
            market_a=market_a,
            market_b=market_b,
            test_statistic=test_stat,
            p_value=p_value,
            result=result,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

    def find_best_pairs(
        self,
        markets: Optional[List[str]] = None,
        top_n: int = 10,
        min_correlation: float = 0.5,
    ) -> List[PairScore]:
        """Find best trading pairs.

        Args:
            markets: Markets to consider
            top_n: Number of pairs to return
            min_correlation: Minimum absolute correlation

        Returns:
            List of pair scores
        """
        if markets is None:
            markets = list(self._price_data.keys())

        pairs = []

        for i, market_a in enumerate(markets):
            for j in range(i + 1, len(markets)):
                market_b = markets[j]

                # Calculate correlation
                corr_pair = self.calculate_correlation(market_a, market_b)

                if abs(corr_pair.correlation) < min_correlation:
                    continue

                # Calculate rolling correlation for stability
                rolling = self.calculate_rolling_correlation(market_a, market_b)

                # Calculate cointegration
                coint = self.test_cointegration(market_a, market_b)

                # Score components
                correlation_score = abs(corr_pair.correlation) * 100
                stability_score = (1 - rolling.std) * 100 if rolling.std < 1 else 0
                cointegration_score = 100 if coint.is_cointegrated else 0

                # Simple liquidity score based on data availability
                data_a = len(self._price_data.get(market_a, []))
                data_b = len(self._price_data.get(market_b, []))
                liquidity_score = min(100, (data_a + data_b) / 2)

                # Weighted total
                total_score = (
                    correlation_score * 0.3 +
                    cointegration_score * 0.4 +
                    stability_score * 0.2 +
                    liquidity_score * 0.1
                )

                pairs.append(PairScore(
                    market_a=market_a,
                    market_b=market_b,
                    correlation_score=correlation_score,
                    cointegration_score=cointegration_score,
                    stability_score=stability_score,
                    liquidity_score=liquidity_score,
                    total_score=total_score,
                ))

        # Sort by total score
        pairs.sort(key=lambda x: x.total_score, reverse=True)

        # Assign ranks
        for i, pair in enumerate(pairs[:top_n]):
            pair.rank = i + 1

        return pairs[:top_n]

    def get_highly_correlated_pairs(
        self,
        threshold: float = 0.8,
    ) -> List[CorrelationPair]:
        """Get highly correlated pairs.

        Args:
            threshold: Minimum absolute correlation

        Returns:
            List of highly correlated pairs
        """
        pairs = []
        markets = list(self._price_data.keys())

        for i, market_a in enumerate(markets):
            for j in range(i + 1, len(markets)):
                market_b = markets[j]
                corr = self.calculate_correlation(market_a, market_b)
                if abs(corr.correlation) >= threshold:
                    pairs.append(corr)

        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return pairs

    def get_uncorrelated_pairs(
        self,
        threshold: float = 0.2,
    ) -> List[CorrelationPair]:
        """Get uncorrelated pairs (for diversification).

        Args:
            threshold: Maximum absolute correlation

        Returns:
            List of uncorrelated pairs
        """
        pairs = []
        markets = list(self._price_data.keys())

        for i, market_a in enumerate(markets):
            for j in range(i + 1, len(markets)):
                market_b = markets[j]
                corr = self.calculate_correlation(market_a, market_b)
                if abs(corr.correlation) <= threshold:
                    pairs.append(corr)

        pairs.sort(key=lambda x: abs(x.correlation))
        return pairs

    def add_callback(self, callback: Callable) -> None:
        """Add callback for correlation updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_markets(self) -> List[str]:
        """Get list of tracked markets."""
        return list(self._price_data.keys())

    def get_sample_size(self, market: str) -> int:
        """Get sample size for market."""
        return len(self._returns_data.get(market, []))

    def clear_market(self, market: str) -> None:
        """Clear data for a market."""
        if market in self._price_data:
            del self._price_data[market]
        if market in self._returns_data:
            del self._returns_data[market]

    def clear_all(self) -> None:
        """Clear all data."""
        self._price_data.clear()
        self._returns_data.clear()
        self._correlation_cache.clear()
        self._rolling_cache.clear()


# Global correlation analyzer instance
_correlation_analyzer: Optional[MarketCorrelationAnalyzer] = None


def get_correlation_analyzer() -> MarketCorrelationAnalyzer:
    """Get global correlation analyzer."""
    global _correlation_analyzer
    if _correlation_analyzer is None:
        _correlation_analyzer = MarketCorrelationAnalyzer()
    return _correlation_analyzer


def reset_correlation_analyzer() -> None:
    """Reset global correlation analyzer."""
    global _correlation_analyzer
    _correlation_analyzer = None
