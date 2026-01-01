"""Volatility Analyzer module for market volatility analysis.

This module provides comprehensive volatility analysis including:
- Historical volatility (realized volatility)
- Average True Range (ATR) calculation
- Volatility regime detection
- GARCH-like volatility forecasting
- Volatility term structure
- Volatility clustering analysis
- Intraday volatility patterns
"""

import time
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class VolatilityType(Enum):
    """Type of volatility calculation."""

    REALIZED = "realized"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"
    YANG_ZHANG = "yang_zhang"
    ATR = "atr"


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class TrendDirection(Enum):
    """Volatility trend direction."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class OHLC:
    """OHLC price data."""

    timestamp: float
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")

    @property
    def range(self) -> Decimal:
        """Get price range."""
        return self.high - self.low

    @property
    def range_pct(self) -> float:
        """Get range as percentage of open."""
        if self.open > 0:
            return float(self.range / self.open * 100)
        return 0.0

    @property
    def true_range(self) -> Decimal:
        """Get true range (same as range without previous close)."""
        return self.range

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "range": str(self.range),
            "range_pct": self.range_pct,
        }


@dataclass
class VolatilityMetrics:
    """Volatility metrics."""

    realized_volatility: float = 0.0
    annualized_volatility: float = 0.0
    atr: Decimal = Decimal("0")
    atr_pct: float = 0.0
    daily_range_avg: float = 0.0
    daily_range_std: float = 0.0
    volatility_type: VolatilityType = VolatilityType.REALIZED
    period_days: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "realized_volatility": self.realized_volatility,
            "annualized_volatility": self.annualized_volatility,
            "atr": str(self.atr),
            "atr_pct": self.atr_pct,
            "daily_range_avg": self.daily_range_avg,
            "daily_range_std": self.daily_range_std,
            "volatility_type": self.volatility_type.value,
            "period_days": self.period_days,
        }


@dataclass
class VolatilityRegimeInfo:
    """Volatility regime information."""

    regime: VolatilityRegime = VolatilityRegime.NORMAL
    current_volatility: float = 0.0
    regime_percentile: float = 50.0
    regime_duration: int = 0
    previous_regime: Optional[VolatilityRegime] = None
    regime_thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "current_volatility": self.current_volatility,
            "regime_percentile": self.regime_percentile,
            "regime_duration": self.regime_duration,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "regime_thresholds": self.regime_thresholds,
        }


@dataclass
class VolatilityForecast:
    """Volatility forecast."""

    forecast_periods: int = 1
    predicted_volatility: float = 0.0
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    model_type: str = "ewma"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast_periods": self.forecast_periods,
            "predicted_volatility": self.predicted_volatility,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "model_type": self.model_type,
        }


@dataclass
class VolatilityTermStructure:
    """Volatility term structure."""

    short_term: float = 0.0  # 7 days
    medium_term: float = 0.0  # 30 days
    long_term: float = 0.0  # 90 days
    term_spread: float = 0.0  # long - short
    is_inverted: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "short_term": self.short_term,
            "medium_term": self.medium_term,
            "long_term": self.long_term,
            "term_spread": self.term_spread,
            "is_inverted": self.is_inverted,
        }


@dataclass
class VolatilityClustering:
    """Volatility clustering analysis."""

    cluster_coefficient: float = 0.0  # Autocorrelation of squared returns
    persistence: float = 0.0  # How long high vol periods last
    mean_reversion_speed: float = 0.0
    half_life: float = 0.0  # Days to revert halfway to mean
    is_clustering: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_coefficient": self.cluster_coefficient,
            "persistence": self.persistence,
            "mean_reversion_speed": self.mean_reversion_speed,
            "half_life": self.half_life,
            "is_clustering": self.is_clustering,
        }


@dataclass
class IntradayPattern:
    """Intraday volatility pattern."""

    hour_volatilities: Dict[int, float] = field(default_factory=dict)
    most_volatile_hour: int = 0
    least_volatile_hour: int = 0
    asian_session_vol: float = 0.0  # 0-8 UTC
    european_session_vol: float = 0.0  # 8-16 UTC
    us_session_vol: float = 0.0  # 16-24 UTC

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hour_volatilities": self.hour_volatilities,
            "most_volatile_hour": self.most_volatile_hour,
            "least_volatile_hour": self.least_volatile_hour,
            "asian_session_vol": self.asian_session_vol,
            "european_session_vol": self.european_session_vol,
            "us_session_vol": self.us_session_vol,
        }


@dataclass
class VolatilityReport:
    """Complete volatility analysis report."""

    market: str = ""
    metrics: Optional[VolatilityMetrics] = None
    regime: Optional[VolatilityRegimeInfo] = None
    forecast: Optional[VolatilityForecast] = None
    term_structure: Optional[VolatilityTermStructure] = None
    clustering: Optional[VolatilityClustering] = None
    intraday: Optional[IntradayPattern] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "regime": self.regime.to_dict() if self.regime else None,
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "term_structure": self.term_structure.to_dict() if self.term_structure else None,
            "clustering": self.clustering.to_dict() if self.clustering else None,
            "intraday": self.intraday.to_dict() if self.intraday else None,
            "timestamp": self.timestamp,
        }


class VolatilityCalculator:
    """Calculator for volatility metrics."""

    def __init__(self, trading_days_per_year: int = 365):
        """Initialize calculator.

        Args:
            trading_days_per_year: Trading days for annualization (crypto = 365)
        """
        self.trading_days_per_year = trading_days_per_year

    def calculate_realized_volatility(
        self,
        returns: List[float],
        annualize: bool = True,
    ) -> float:
        """Calculate realized volatility from returns.

        Args:
            returns: List of returns
            annualize: Whether to annualize

        Returns:
            Volatility as percentage
        """
        if len(returns) < 2:
            return 0.0

        std = statistics.stdev(returns)

        if annualize:
            std = std * math.sqrt(self.trading_days_per_year)

        return std * 100  # As percentage

    def calculate_parkinson_volatility(
        self,
        ohlc_data: List[OHLC],
        annualize: bool = True,
    ) -> float:
        """Calculate Parkinson volatility from high-low range.

        Args:
            ohlc_data: OHLC data
            annualize: Whether to annualize

        Returns:
            Volatility as percentage
        """
        if len(ohlc_data) < 2:
            return 0.0

        log_hl_sq = []
        for bar in ohlc_data:
            if bar.low > 0 and bar.high > 0:
                log_hl = math.log(float(bar.high / bar.low))
                log_hl_sq.append(log_hl ** 2)

        if not log_hl_sq:
            return 0.0

        variance = sum(log_hl_sq) / (4 * math.log(2) * len(log_hl_sq))
        vol = math.sqrt(variance)

        if annualize:
            vol = vol * math.sqrt(self.trading_days_per_year)

        return vol * 100

    def calculate_garman_klass_volatility(
        self,
        ohlc_data: List[OHLC],
        annualize: bool = True,
    ) -> float:
        """Calculate Garman-Klass volatility.

        Args:
            ohlc_data: OHLC data
            annualize: Whether to annualize

        Returns:
            Volatility as percentage
        """
        if len(ohlc_data) < 2:
            return 0.0

        variance_sum = 0.0
        count = 0

        for bar in ohlc_data:
            if bar.low > 0 and bar.high > 0 and bar.open > 0 and bar.close > 0:
                log_hl = math.log(float(bar.high / bar.low))
                log_co = math.log(float(bar.close / bar.open))
                variance_sum += 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
                count += 1

        if count == 0:
            return 0.0

        variance = variance_sum / count
        vol = math.sqrt(max(0, variance))

        if annualize:
            vol = vol * math.sqrt(self.trading_days_per_year)

        return vol * 100

    def calculate_atr(
        self,
        ohlc_data: List[OHLC],
        period: int = 14,
    ) -> Decimal:
        """Calculate Average True Range.

        Args:
            ohlc_data: OHLC data
            period: ATR period

        Returns:
            ATR value
        """
        if len(ohlc_data) < 2:
            return Decimal("0")

        true_ranges = []

        for i, bar in enumerate(ohlc_data):
            if i == 0:
                tr = bar.high - bar.low
            else:
                prev_close = ohlc_data[i - 1].close
                tr = max(
                    bar.high - bar.low,
                    abs(bar.high - prev_close),
                    abs(bar.low - prev_close),
                )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges)

        # Use EMA for smoother ATR
        multiplier = Decimal(str(2 / (period + 1)))
        atr = sum(true_ranges[:period]) / period

        for tr in true_ranges[period:]:
            atr = (tr - atr) * multiplier + atr

        return atr

    def calculate_ewma_volatility(
        self,
        returns: List[float],
        lambda_param: float = 0.94,
        annualize: bool = True,
    ) -> float:
        """Calculate EWMA volatility (RiskMetrics style).

        Args:
            returns: List of returns
            lambda_param: Decay factor (0.94 typical)
            annualize: Whether to annualize

        Returns:
            Volatility as percentage
        """
        if len(returns) < 2:
            return 0.0

        # Initial variance (sample variance of first returns)
        variance = statistics.variance(returns[:min(10, len(returns))])

        # EWMA update
        for r in returns:
            variance = lambda_param * variance + (1 - lambda_param) * r ** 2

        vol = math.sqrt(variance)

        if annualize:
            vol = vol * math.sqrt(self.trading_days_per_year)

        return vol * 100


class VolatilityAnalyzer:
    """Analyzer for market volatility."""

    def __init__(
        self,
        regime_thresholds: Optional[Dict[str, float]] = None,
        trading_days_per_year: int = 365,
    ):
        """Initialize analyzer.

        Args:
            regime_thresholds: Custom regime thresholds (percentiles)
            trading_days_per_year: Trading days for annualization
        """
        self.regime_thresholds = regime_thresholds or {
            "very_low": 10.0,
            "low": 25.0,
            "normal_low": 40.0,
            "normal_high": 60.0,
            "high": 75.0,
            "very_high": 90.0,
        }
        self.trading_days_per_year = trading_days_per_year
        self._calculator = VolatilityCalculator(trading_days_per_year)
        self._ohlc_data: Dict[str, List[OHLC]] = {}
        self._returns_data: Dict[str, List[float]] = {}
        self._volatility_history: Dict[str, List[float]] = {}
        self._max_history = 1000
        self._callbacks: List[Callable] = []

    def add_ohlc(
        self,
        market: str,
        timestamp: float,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> Optional[float]:
        """Add OHLC data.

        Args:
            market: Market symbol
            timestamp: Timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Volume

        Returns:
            Return if calculable
        """
        if market not in self._ohlc_data:
            self._ohlc_data[market] = []
            self._returns_data[market] = []
            self._volatility_history[market] = []

        ohlc = OHLC(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

        returns = 0.0
        if self._ohlc_data[market]:
            prev_close = self._ohlc_data[market][-1].close
            if prev_close > 0:
                returns = float((close_price - prev_close) / prev_close)

        self._ohlc_data[market].append(ohlc)
        if returns != 0.0 or self._returns_data[market]:
            self._returns_data[market].append(returns)

        # Trim history
        if len(self._ohlc_data[market]) > self._max_history:
            self._ohlc_data[market] = self._ohlc_data[market][-self._max_history:]
            self._returns_data[market] = self._returns_data[market][-self._max_history:]

        return returns

    def add_price(
        self,
        market: str,
        timestamp: float,
        price: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> Optional[float]:
        """Add simple price data (uses as OHLC with same values).

        Args:
            market: Market symbol
            timestamp: Timestamp
            price: Price
            volume: Volume

        Returns:
            Return if calculable
        """
        return self.add_ohlc(market, timestamp, price, price, price, price, volume)

    def calculate_volatility(
        self,
        market: str,
        period: int = 20,
        vol_type: VolatilityType = VolatilityType.REALIZED,
    ) -> VolatilityMetrics:
        """Calculate volatility metrics.

        Args:
            market: Market symbol
            period: Lookback period
            vol_type: Type of volatility calculation

        Returns:
            Volatility metrics
        """
        ohlc_data = self._ohlc_data.get(market, [])
        returns = self._returns_data.get(market, [])

        if len(returns) < 2:
            return VolatilityMetrics(period_days=0)

        # Use most recent data
        recent_ohlc = ohlc_data[-period:] if len(ohlc_data) >= period else ohlc_data
        recent_returns = returns[-period:] if len(returns) >= period else returns

        # Calculate based on type
        if vol_type == VolatilityType.REALIZED:
            vol = self._calculator.calculate_realized_volatility(recent_returns, annualize=False)
            ann_vol = self._calculator.calculate_realized_volatility(recent_returns, annualize=True)
        elif vol_type == VolatilityType.PARKINSON:
            vol = self._calculator.calculate_parkinson_volatility(recent_ohlc, annualize=False)
            ann_vol = self._calculator.calculate_parkinson_volatility(recent_ohlc, annualize=True)
        elif vol_type == VolatilityType.GARMAN_KLASS:
            vol = self._calculator.calculate_garman_klass_volatility(recent_ohlc, annualize=False)
            ann_vol = self._calculator.calculate_garman_klass_volatility(recent_ohlc, annualize=True)
        else:
            vol = self._calculator.calculate_realized_volatility(recent_returns, annualize=False)
            ann_vol = self._calculator.calculate_realized_volatility(recent_returns, annualize=True)

        # Calculate ATR
        atr = self._calculator.calculate_atr(recent_ohlc, min(14, len(recent_ohlc)))
        atr_pct = 0.0
        if recent_ohlc and recent_ohlc[-1].close > 0:
            atr_pct = float(atr / recent_ohlc[-1].close * 100)

        # Daily range statistics
        ranges = [bar.range_pct for bar in recent_ohlc]
        range_avg = statistics.mean(ranges) if ranges else 0.0
        range_std = statistics.stdev(ranges) if len(ranges) > 1 else 0.0

        # Store in history
        self._volatility_history[market].append(ann_vol)
        if len(self._volatility_history[market]) > self._max_history:
            self._volatility_history[market] = self._volatility_history[market][-self._max_history:]

        return VolatilityMetrics(
            realized_volatility=vol,
            annualized_volatility=ann_vol,
            atr=atr,
            atr_pct=atr_pct,
            daily_range_avg=range_avg,
            daily_range_std=range_std,
            volatility_type=vol_type,
            period_days=len(recent_returns),
        )

    def detect_regime(
        self,
        market: str,
        lookback: int = 100,
    ) -> VolatilityRegimeInfo:
        """Detect current volatility regime.

        Args:
            market: Market symbol
            lookback: Historical lookback for percentile calculation

        Returns:
            Regime information
        """
        vol_history = self._volatility_history.get(market, [])

        if len(vol_history) < 5:
            return VolatilityRegimeInfo()

        current_vol = vol_history[-1]

        # Calculate percentile
        history = vol_history[-lookback:] if len(vol_history) >= lookback else vol_history
        sorted_history = sorted(history)
        percentile_idx = 0
        for i, v in enumerate(sorted_history):
            if v <= current_vol:
                percentile_idx = i
        percentile = (percentile_idx / len(sorted_history)) * 100

        # Determine regime
        if percentile <= self.regime_thresholds["very_low"]:
            regime = VolatilityRegime.VERY_LOW
        elif percentile <= self.regime_thresholds["low"]:
            regime = VolatilityRegime.LOW
        elif percentile <= self.regime_thresholds["normal_high"]:
            regime = VolatilityRegime.NORMAL
        elif percentile <= self.regime_thresholds["high"]:
            regime = VolatilityRegime.HIGH
        elif percentile <= self.regime_thresholds["very_high"]:
            regime = VolatilityRegime.VERY_HIGH
        else:
            regime = VolatilityRegime.EXTREME

        # Calculate regime duration (how long in current regime)
        duration = 1
        for v in reversed(vol_history[:-1]):
            v_percentile = (sorted(history).index(min(history, key=lambda x: abs(x - v))) / len(history)) * 100
            if v_percentile <= self.regime_thresholds["very_low"]:
                v_regime = VolatilityRegime.VERY_LOW
            elif v_percentile <= self.regime_thresholds["low"]:
                v_regime = VolatilityRegime.LOW
            elif v_percentile <= self.regime_thresholds["normal_high"]:
                v_regime = VolatilityRegime.NORMAL
            elif v_percentile <= self.regime_thresholds["high"]:
                v_regime = VolatilityRegime.HIGH
            elif v_percentile <= self.regime_thresholds["very_high"]:
                v_regime = VolatilityRegime.VERY_HIGH
            else:
                v_regime = VolatilityRegime.EXTREME

            if v_regime == regime:
                duration += 1
            else:
                break

        return VolatilityRegimeInfo(
            regime=regime,
            current_volatility=current_vol,
            regime_percentile=percentile,
            regime_duration=duration,
            regime_thresholds=self.regime_thresholds,
        )

    def forecast_volatility(
        self,
        market: str,
        periods: int = 5,
        lambda_param: float = 0.94,
    ) -> VolatilityForecast:
        """Forecast future volatility using EWMA.

        Args:
            market: Market symbol
            periods: Forecast horizon
            lambda_param: EWMA decay parameter

        Returns:
            Volatility forecast
        """
        returns = self._returns_data.get(market, [])

        if len(returns) < 10:
            return VolatilityForecast(forecast_periods=periods)

        # Calculate EWMA volatility
        ewma_vol = self._calculator.calculate_ewma_volatility(
            returns,
            lambda_param=lambda_param,
            annualize=True,
        )

        # Historical volatility for confidence interval
        vol_history = self._volatility_history.get(market, [])
        if len(vol_history) > 10:
            vol_std = statistics.stdev(vol_history[-50:])
        else:
            vol_std = ewma_vol * 0.2  # Default 20% uncertainty

        # Confidence interval widens with forecast horizon
        horizon_factor = math.sqrt(periods)

        return VolatilityForecast(
            forecast_periods=periods,
            predicted_volatility=ewma_vol,
            confidence_low=max(0, ewma_vol - 1.96 * vol_std * horizon_factor),
            confidence_high=ewma_vol + 1.96 * vol_std * horizon_factor,
            model_type="ewma",
        )

    def calculate_term_structure(
        self,
        market: str,
    ) -> VolatilityTermStructure:
        """Calculate volatility term structure.

        Args:
            market: Market symbol

        Returns:
            Term structure
        """
        returns = self._returns_data.get(market, [])

        if len(returns) < 90:
            return VolatilityTermStructure()

        # Calculate volatility at different windows
        short_vol = self._calculator.calculate_realized_volatility(returns[-7:], annualize=True)
        medium_vol = self._calculator.calculate_realized_volatility(returns[-30:], annualize=True)
        long_vol = self._calculator.calculate_realized_volatility(returns[-90:], annualize=True)

        term_spread = long_vol - short_vol
        is_inverted = short_vol > long_vol * 1.1  # 10% threshold

        return VolatilityTermStructure(
            short_term=short_vol,
            medium_term=medium_vol,
            long_term=long_vol,
            term_spread=term_spread,
            is_inverted=is_inverted,
        )

    def analyze_clustering(
        self,
        market: str,
    ) -> VolatilityClustering:
        """Analyze volatility clustering.

        Args:
            market: Market symbol

        Returns:
            Clustering analysis
        """
        returns = self._returns_data.get(market, [])

        if len(returns) < 30:
            return VolatilityClustering()

        # Squared returns for volatility clustering
        squared_returns = [r ** 2 for r in returns]

        # Calculate autocorrelation of squared returns (lag 1)
        n = len(squared_returns)
        mean_sq = sum(squared_returns) / n

        numerator = sum(
            (squared_returns[i] - mean_sq) * (squared_returns[i + 1] - mean_sq)
            for i in range(n - 1)
        )
        denominator = sum((sr - mean_sq) ** 2 for sr in squared_returns)

        if denominator == 0:
            cluster_coef = 0.0
        else:
            cluster_coef = numerator / denominator

        # Persistence: higher autocorrelation = more persistence
        persistence = max(0, min(1, cluster_coef))

        # Mean reversion speed (1 - persistence)
        mean_reversion = 1 - persistence

        # Half-life in days
        if persistence > 0 and persistence < 1:
            half_life = math.log(0.5) / math.log(persistence) if persistence > 0 else float('inf')
        else:
            half_life = 0.0

        # Clustering if autocorrelation > 0.2
        is_clustering = cluster_coef > 0.2

        return VolatilityClustering(
            cluster_coefficient=cluster_coef,
            persistence=persistence,
            mean_reversion_speed=mean_reversion,
            half_life=abs(half_life) if half_life != float('inf') else 999,
            is_clustering=is_clustering,
        )

    def analyze_intraday_pattern(
        self,
        market: str,
    ) -> IntradayPattern:
        """Analyze intraday volatility patterns.

        Args:
            market: Market symbol

        Returns:
            Intraday pattern
        """
        ohlc_data = self._ohlc_data.get(market, [])

        if len(ohlc_data) < 24:
            return IntradayPattern()

        # Group by hour
        hour_ranges: Dict[int, List[float]] = {h: [] for h in range(24)}

        for bar in ohlc_data:
            from datetime import datetime
            hour = datetime.fromtimestamp(bar.timestamp).hour
            hour_ranges[hour].append(bar.range_pct)

        # Calculate average volatility per hour
        hour_volatilities = {}
        for hour, ranges in hour_ranges.items():
            if ranges:
                hour_volatilities[hour] = statistics.mean(ranges)
            else:
                hour_volatilities[hour] = 0.0

        # Find most/least volatile hours
        non_zero = {h: v for h, v in hour_volatilities.items() if v > 0}
        most_volatile = max(non_zero, key=non_zero.get) if non_zero else 0
        least_volatile = min(non_zero, key=non_zero.get) if non_zero else 0

        # Session volatilities
        asian = [v for h, v in hour_volatilities.items() if 0 <= h < 8]
        european = [v for h, v in hour_volatilities.items() if 8 <= h < 16]
        us = [v for h, v in hour_volatilities.items() if 16 <= h < 24]

        return IntradayPattern(
            hour_volatilities=hour_volatilities,
            most_volatile_hour=most_volatile,
            least_volatile_hour=least_volatile,
            asian_session_vol=statistics.mean(asian) if asian else 0.0,
            european_session_vol=statistics.mean(european) if european else 0.0,
            us_session_vol=statistics.mean(us) if us else 0.0,
        )

    def get_full_report(
        self,
        market: str,
    ) -> VolatilityReport:
        """Get comprehensive volatility report.

        Args:
            market: Market symbol

        Returns:
            Full volatility report
        """
        return VolatilityReport(
            market=market,
            metrics=self.calculate_volatility(market),
            regime=self.detect_regime(market),
            forecast=self.forecast_volatility(market),
            term_structure=self.calculate_term_structure(market),
            clustering=self.analyze_clustering(market),
            intraday=self.analyze_intraday_pattern(market),
        )

    def get_volatility_trend(
        self,
        market: str,
        lookback: int = 20,
    ) -> TrendDirection:
        """Get volatility trend direction.

        Args:
            market: Market symbol
            lookback: Lookback period

        Returns:
            Trend direction
        """
        vol_history = self._volatility_history.get(market, [])

        if len(vol_history) < lookback:
            return TrendDirection.STABLE

        recent = vol_history[-lookback:]

        # Simple linear regression slope
        n = len(recent)
        mean_x = (n - 1) / 2
        mean_y = sum(recent) / n

        numerator = sum((i - mean_x) * (recent[i] - mean_y) for i in range(n))
        denominator = sum((i - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return TrendDirection.STABLE

        slope = numerator / denominator

        # Normalize by mean volatility
        relative_slope = slope / (mean_y if mean_y > 0 else 1)

        if relative_slope > 0.05:
            return TrendDirection.INCREASING
        elif relative_slope < -0.05:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE

    def compare_volatility(
        self,
        market_a: str,
        market_b: str,
    ) -> Dict[str, Any]:
        """Compare volatility between two markets.

        Args:
            market_a: First market
            market_b: Second market

        Returns:
            Comparison results
        """
        metrics_a = self.calculate_volatility(market_a)
        metrics_b = self.calculate_volatility(market_b)

        ratio = 0.0
        if metrics_b.annualized_volatility > 0:
            ratio = metrics_a.annualized_volatility / metrics_b.annualized_volatility

        return {
            "market_a": market_a,
            "market_b": market_b,
            "volatility_a": metrics_a.annualized_volatility,
            "volatility_b": metrics_b.annualized_volatility,
            "volatility_ratio": ratio,
            "higher_volatility": market_a if ratio > 1 else market_b,
        }

    def add_callback(self, callback: Callable) -> None:
        """Add callback for volatility updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_markets(self) -> List[str]:
        """Get list of tracked markets."""
        return list(self._ohlc_data.keys())

    def get_sample_size(self, market: str) -> int:
        """Get sample size for market."""
        return len(self._ohlc_data.get(market, []))

    def clear_market(self, market: str) -> None:
        """Clear data for a market."""
        for store in [self._ohlc_data, self._returns_data, self._volatility_history]:
            if market in store:
                del store[market]

    def clear_all(self) -> None:
        """Clear all data."""
        self._ohlc_data.clear()
        self._returns_data.clear()
        self._volatility_history.clear()


# Global volatility analyzer instance
_volatility_analyzer: Optional[VolatilityAnalyzer] = None


def get_volatility_analyzer() -> VolatilityAnalyzer:
    """Get global volatility analyzer."""
    global _volatility_analyzer
    if _volatility_analyzer is None:
        _volatility_analyzer = VolatilityAnalyzer()
    return _volatility_analyzer


def reset_volatility_analyzer() -> None:
    """Reset global volatility analyzer."""
    global _volatility_analyzer
    _volatility_analyzer = None
