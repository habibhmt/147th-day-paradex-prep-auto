"""Market Regime Detector module.

This module provides market regime detection including:
- Market regime classification (trending, ranging, volatile, calm)
- Regime transition detection
- Regime-specific strategy recommendations
- HMM-like regime detection
- Regime persistence tracking
- Multi-factor regime analysis
"""

import time
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class MarketRegime(Enum):
    """Market regime classification."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Trend strength classification."""

    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class VolatilityState(Enum):
    """Volatility state classification."""

    EXTREME = "extreme"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    VERY_LOW = "very_low"


class MomentumState(Enum):
    """Momentum state classification."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class RegimeIndicators:
    """Indicators used for regime detection."""

    trend_direction: float = 0.0  # -1 to 1
    trend_strength: float = 0.0  # 0 to 1
    volatility_percentile: float = 50.0
    momentum: float = 0.0
    mean_reversion_score: float = 0.0
    correlation_with_market: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "volatility_percentile": self.volatility_percentile,
            "momentum": self.momentum,
            "mean_reversion_score": self.mean_reversion_score,
            "correlation_with_market": self.correlation_with_market,
        }


@dataclass
class RegimeState:
    """Current regime state."""

    regime: MarketRegime
    confidence: float  # 0-100
    trend_strength: TrendStrength
    volatility_state: VolatilityState
    momentum_state: MomentumState
    indicators: Optional[RegimeIndicators] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "trend_strength": self.trend_strength.value,
            "volatility_state": self.volatility_state.value,
            "momentum_state": self.momentum_state.value,
            "indicators": self.indicators.to_dict() if self.indicators else None,
            "timestamp": self.timestamp,
        }


@dataclass
class RegimeTransition:
    """Regime transition event."""

    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: float
    confidence: float = 0.0
    duration_in_previous: float = 0.0  # seconds in previous regime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "duration_in_previous": self.duration_in_previous,
        }


@dataclass
class RegimePersistence:
    """Track regime persistence over time."""

    regime: MarketRegime
    start_time: float
    last_update: float
    confidence_history: List[float] = field(default_factory=list)
    sample_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.last_update - self.start_time

    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration_seconds / 3600

    @property
    def avg_confidence(self) -> float:
        """Get average confidence."""
        return statistics.mean(self.confidence_history) if self.confidence_history else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "duration_hours": self.duration_hours,
            "sample_count": self.sample_count,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class StrategyRecommendation:
    """Strategy recommendation based on regime."""

    regime: MarketRegime
    recommended_strategies: List[str]
    avoid_strategies: List[str]
    position_sizing_factor: float = 1.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "recommended_strategies": self.recommended_strategies,
            "avoid_strategies": self.avoid_strategies,
            "position_sizing_factor": self.position_sizing_factor,
            "notes": self.notes,
        }


@dataclass
class RegimeReport:
    """Complete regime analysis report."""

    market: str
    current_state: RegimeState
    persistence: Optional[RegimePersistence] = None
    recent_transitions: List[RegimeTransition] = field(default_factory=list)
    recommendation: Optional[StrategyRecommendation] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "current_state": self.current_state.to_dict(),
            "persistence": self.persistence.to_dict() if self.persistence else None,
            "recent_transitions": [t.to_dict() for t in self.recent_transitions],
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "timestamp": self.timestamp,
        }


class RegimeDetector:
    """Market regime detector."""

    def __init__(
        self,
        trend_threshold: float = 0.02,
        volatility_lookback: int = 20,
        momentum_lookback: int = 10,
    ):
        """Initialize regime detector.

        Args:
            trend_threshold: Threshold for trend detection
            volatility_lookback: Lookback for volatility calculation
            momentum_lookback: Lookback for momentum calculation
        """
        self.trend_threshold = trend_threshold
        self.volatility_lookback = volatility_lookback
        self.momentum_lookback = momentum_lookback

        self._prices: Dict[str, List[Decimal]] = {}
        self._returns: Dict[str, List[float]] = {}
        self._volatility_history: Dict[str, List[float]] = {}
        self._regime_history: Dict[str, List[RegimeState]] = {}
        self._persistence: Dict[str, RegimePersistence] = {}
        self._transitions: Dict[str, List[RegimeTransition]] = {}
        self._callbacks: List[Callable] = []
        self._max_history = 500

        # Strategy recommendations per regime
        self._recommendations = self._init_recommendations()

    def _init_recommendations(self) -> Dict[MarketRegime, StrategyRecommendation]:
        """Initialize strategy recommendations."""
        return {
            MarketRegime.STRONG_UPTREND: StrategyRecommendation(
                regime=MarketRegime.STRONG_UPTREND,
                recommended_strategies=["trend_following", "momentum", "breakout"],
                avoid_strategies=["mean_reversion", "counter_trend"],
                position_sizing_factor=1.2,
                notes="Strong trend - maximize position in direction",
            ),
            MarketRegime.UPTREND: StrategyRecommendation(
                regime=MarketRegime.UPTREND,
                recommended_strategies=["trend_following", "pullback_entry"],
                avoid_strategies=["counter_trend"],
                position_sizing_factor=1.0,
                notes="Moderate uptrend - follow with caution",
            ),
            MarketRegime.RANGING: StrategyRecommendation(
                regime=MarketRegime.RANGING,
                recommended_strategies=["mean_reversion", "range_trading", "grid"],
                avoid_strategies=["trend_following", "breakout"],
                position_sizing_factor=0.8,
                notes="Range-bound - trade the range",
            ),
            MarketRegime.DOWNTREND: StrategyRecommendation(
                regime=MarketRegime.DOWNTREND,
                recommended_strategies=["trend_following", "short_bias"],
                avoid_strategies=["counter_trend", "long_only"],
                position_sizing_factor=1.0,
                notes="Moderate downtrend - short bias",
            ),
            MarketRegime.STRONG_DOWNTREND: StrategyRecommendation(
                regime=MarketRegime.STRONG_DOWNTREND,
                recommended_strategies=["trend_following", "short_only"],
                avoid_strategies=["long_only", "mean_reversion"],
                position_sizing_factor=1.2,
                notes="Strong downtrend - maximize short position",
            ),
            MarketRegime.HIGH_VOLATILITY: StrategyRecommendation(
                regime=MarketRegime.HIGH_VOLATILITY,
                recommended_strategies=["volatility_selling", "straddle"],
                avoid_strategies=["trend_following"],
                position_sizing_factor=0.5,
                notes="High volatility - reduce size, widen stops",
            ),
            MarketRegime.LOW_VOLATILITY: StrategyRecommendation(
                regime=MarketRegime.LOW_VOLATILITY,
                recommended_strategies=["breakout_anticipation", "volatility_buying"],
                avoid_strategies=["range_trading"],
                position_sizing_factor=1.0,
                notes="Low volatility - prepare for breakout",
            ),
            MarketRegime.UNKNOWN: StrategyRecommendation(
                regime=MarketRegime.UNKNOWN,
                recommended_strategies=["delta_neutral", "market_making"],
                avoid_strategies=["directional"],
                position_sizing_factor=0.5,
                notes="Unknown regime - stay neutral",
            ),
        }

    def add_price(
        self,
        market: str,
        price: Decimal,
        timestamp: Optional[float] = None,
    ) -> Optional[RegimeState]:
        """Add price and update regime detection.

        Args:
            market: Market symbol
            price: Price
            timestamp: Optional timestamp

        Returns:
            Current regime state if enough data
        """
        ts = timestamp or time.time()

        if market not in self._prices:
            self._prices[market] = []
            self._returns[market] = []
            self._volatility_history[market] = []
            self._regime_history[market] = []
            self._transitions[market] = []

        # Calculate return
        returns = 0.0
        if self._prices[market]:
            prev_price = self._prices[market][-1]
            if prev_price > 0:
                returns = float((price - prev_price) / prev_price)

        self._prices[market].append(price)
        if returns != 0.0 or self._returns[market]:
            self._returns[market].append(returns)

        # Trim history
        if len(self._prices[market]) > self._max_history:
            self._prices[market] = self._prices[market][-self._max_history:]
            self._returns[market] = self._returns[market][-self._max_history:]

        # Detect regime if enough data
        if len(self._returns[market]) >= self.volatility_lookback:
            state = self._detect_regime(market, ts)

            # Track persistence
            self._update_persistence(market, state)

            # Store in history
            self._regime_history[market].append(state)
            if len(self._regime_history[market]) > self._max_history:
                self._regime_history[market] = self._regime_history[market][-self._max_history:]

            # Notify callbacks
            for callback in self._callbacks:
                callback(market, state)

            return state

        return None

    def _detect_regime(self, market: str, timestamp: float) -> RegimeState:
        """Detect current market regime.

        Args:
            market: Market symbol
            timestamp: Current timestamp

        Returns:
            Regime state
        """
        returns = self._returns.get(market, [])
        prices = self._prices.get(market, [])

        if len(returns) < self.volatility_lookback:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0,
                trend_strength=TrendStrength.NONE,
                volatility_state=VolatilityState.NORMAL,
                momentum_state=MomentumState.NEUTRAL,
                timestamp=timestamp,
            )

        # Calculate indicators
        indicators = self._calculate_indicators(returns, prices)

        # Determine regime components
        trend_strength = self._classify_trend_strength(indicators.trend_strength)
        volatility_state = self._classify_volatility(indicators.volatility_percentile)
        momentum_state = self._classify_momentum(indicators.momentum)

        # Determine overall regime
        regime, confidence = self._determine_regime(
            indicators.trend_direction,
            indicators.trend_strength,
            indicators.volatility_percentile,
            indicators.momentum,
        )

        return RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            momentum_state=momentum_state,
            indicators=indicators,
            timestamp=timestamp,
        )

    def _calculate_indicators(
        self,
        returns: List[float],
        prices: List[Decimal],
    ) -> RegimeIndicators:
        """Calculate regime indicators.

        Args:
            returns: List of returns
            prices: List of prices

        Returns:
            Regime indicators
        """
        recent_returns = returns[-self.volatility_lookback:]

        # Trend direction and strength (using linear regression slope)
        n = len(recent_returns)
        mean_x = (n - 1) / 2
        mean_y = sum(recent_returns) / n

        numerator = sum((i - mean_x) * (recent_returns[i] - mean_y) for i in range(n))
        denominator = sum((i - mean_x) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0

        # Normalize trend direction to -1 to 1
        trend_direction = max(-1, min(1, slope * 100))

        # Trend strength from R-squared
        if len(recent_returns) > 1:
            predicted = [mean_y + slope * (i - mean_x) for i in range(n)]
            ss_res = sum((recent_returns[i] - predicted[i]) ** 2 for i in range(n))
            ss_tot = sum((r - mean_y) ** 2 for r in recent_returns)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_strength = max(0, min(1, abs(r_squared)))
        else:
            trend_strength = 0

        # Volatility percentile
        current_vol = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0
        self._volatility_history[prices[0] if prices else "default"] = self._volatility_history.get(
            prices[0] if prices else "default", []
        )

        vol_history = [current_vol]  # Simplified
        if len(self._volatility_history.get("_all", [])) > 0:
            vol_history = self._volatility_history["_all"]
        else:
            if "_all" not in self._volatility_history:
                self._volatility_history["_all"] = []
            self._volatility_history["_all"].append(current_vol)
            if len(self._volatility_history["_all"]) > 100:
                self._volatility_history["_all"] = self._volatility_history["_all"][-100:]
            vol_history = self._volatility_history["_all"]

        sorted_vols = sorted(vol_history)
        if current_vol in sorted_vols:
            percentile_idx = sorted_vols.index(current_vol)
        else:
            percentile_idx = len([v for v in sorted_vols if v <= current_vol])
        volatility_percentile = (percentile_idx / len(sorted_vols)) * 100 if sorted_vols else 50

        # Momentum (cumulative return over lookback)
        momentum_returns = returns[-self.momentum_lookback:] if len(returns) >= self.momentum_lookback else returns
        momentum = sum(momentum_returns) if momentum_returns else 0

        # Mean reversion score (how far from moving average)
        if len(prices) >= 20:
            ma20 = sum(prices[-20:]) / 20
            current_price = prices[-1]
            mean_reversion_score = float((current_price - ma20) / ma20 * 100) if ma20 > 0 else 0
        else:
            mean_reversion_score = 0

        return RegimeIndicators(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
            momentum=momentum,
            mean_reversion_score=mean_reversion_score,
        )

    def _classify_trend_strength(self, strength: float) -> TrendStrength:
        """Classify trend strength."""
        if strength >= 0.8:
            return TrendStrength.VERY_STRONG
        elif strength >= 0.6:
            return TrendStrength.STRONG
        elif strength >= 0.4:
            return TrendStrength.MODERATE
        elif strength >= 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE

    def _classify_volatility(self, percentile: float) -> VolatilityState:
        """Classify volatility state."""
        if percentile >= 90:
            return VolatilityState.EXTREME
        elif percentile >= 75:
            return VolatilityState.HIGH
        elif percentile >= 25:
            return VolatilityState.NORMAL
        elif percentile >= 10:
            return VolatilityState.LOW
        else:
            return VolatilityState.VERY_LOW

    def _classify_momentum(self, momentum: float) -> MomentumState:
        """Classify momentum state."""
        if momentum >= 0.05:
            return MomentumState.STRONG_BULLISH
        elif momentum >= 0.02:
            return MomentumState.BULLISH
        elif momentum <= -0.05:
            return MomentumState.STRONG_BEARISH
        elif momentum <= -0.02:
            return MomentumState.BEARISH
        else:
            return MomentumState.NEUTRAL

    def _determine_regime(
        self,
        trend_direction: float,
        trend_strength: float,
        volatility_percentile: float,
        momentum: float,
    ) -> Tuple[MarketRegime, float]:
        """Determine overall market regime.

        Args:
            trend_direction: Trend direction (-1 to 1)
            trend_strength: Trend strength (0 to 1)
            volatility_percentile: Volatility percentile
            momentum: Momentum value

        Returns:
            Tuple of (regime, confidence)
        """
        # High volatility overrides other regimes
        if volatility_percentile >= 85:
            return MarketRegime.HIGH_VOLATILITY, min(100, volatility_percentile)

        # Low volatility
        if volatility_percentile <= 15:
            return MarketRegime.LOW_VOLATILITY, min(100, 100 - volatility_percentile)

        # Trend-based regimes
        if trend_direction > 0.3 and trend_strength > 0.5:
            if trend_direction > 0.6 and trend_strength > 0.7:
                return MarketRegime.STRONG_UPTREND, min(100, (trend_direction + trend_strength) * 50)
            else:
                return MarketRegime.UPTREND, min(100, (trend_direction + trend_strength) * 40)
        elif trend_direction < -0.3 and trend_strength > 0.5:
            if trend_direction < -0.6 and trend_strength > 0.7:
                return MarketRegime.STRONG_DOWNTREND, min(100, (abs(trend_direction) + trend_strength) * 50)
            else:
                return MarketRegime.DOWNTREND, min(100, (abs(trend_direction) + trend_strength) * 40)
        else:
            # Ranging market
            confidence = min(100, (1 - trend_strength) * 100)
            return MarketRegime.RANGING, confidence

    def _update_persistence(self, market: str, state: RegimeState) -> None:
        """Update regime persistence tracking.

        Args:
            market: Market symbol
            state: Current regime state
        """
        current_persistence = self._persistence.get(market)

        if current_persistence is None or current_persistence.regime != state.regime:
            # New regime or first regime
            if current_persistence:
                # Record transition
                transition = RegimeTransition(
                    from_regime=current_persistence.regime,
                    to_regime=state.regime,
                    timestamp=state.timestamp,
                    confidence=state.confidence,
                    duration_in_previous=current_persistence.duration_seconds,
                )
                self._transitions[market].append(transition)
                if len(self._transitions[market]) > 100:
                    self._transitions[market] = self._transitions[market][-100:]

            # Start new persistence
            self._persistence[market] = RegimePersistence(
                regime=state.regime,
                start_time=state.timestamp,
                last_update=state.timestamp,
                confidence_history=[state.confidence],
                sample_count=1,
            )
        else:
            # Update existing persistence
            current_persistence.last_update = state.timestamp
            current_persistence.confidence_history.append(state.confidence)
            current_persistence.sample_count += 1

            # Trim confidence history
            if len(current_persistence.confidence_history) > 100:
                current_persistence.confidence_history = current_persistence.confidence_history[-100:]

    def get_current_regime(self, market: str) -> Optional[RegimeState]:
        """Get current regime state.

        Args:
            market: Market symbol

        Returns:
            Current regime state or None
        """
        history = self._regime_history.get(market, [])
        return history[-1] if history else None

    def get_regime_history(
        self,
        market: str,
        limit: int = 50,
    ) -> List[RegimeState]:
        """Get regime history.

        Args:
            market: Market symbol
            limit: Maximum entries

        Returns:
            List of regime states
        """
        return self._regime_history.get(market, [])[-limit:]

    def get_persistence(self, market: str) -> Optional[RegimePersistence]:
        """Get current regime persistence.

        Args:
            market: Market symbol

        Returns:
            Regime persistence or None
        """
        return self._persistence.get(market)

    def get_transitions(
        self,
        market: str,
        limit: int = 20,
    ) -> List[RegimeTransition]:
        """Get recent regime transitions.

        Args:
            market: Market symbol
            limit: Maximum transitions

        Returns:
            List of transitions
        """
        return self._transitions.get(market, [])[-limit:]

    def get_recommendation(
        self,
        market: str,
    ) -> Optional[StrategyRecommendation]:
        """Get strategy recommendation for current regime.

        Args:
            market: Market symbol

        Returns:
            Strategy recommendation or None
        """
        state = self.get_current_regime(market)
        if state:
            return self._recommendations.get(state.regime)
        return None

    def get_full_report(self, market: str) -> RegimeReport:
        """Get complete regime report.

        Args:
            market: Market symbol

        Returns:
            Regime report
        """
        state = self.get_current_regime(market)
        if state is None:
            state = RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0,
                trend_strength=TrendStrength.NONE,
                volatility_state=VolatilityState.NORMAL,
                momentum_state=MomentumState.NEUTRAL,
            )

        return RegimeReport(
            market=market,
            current_state=state,
            persistence=self.get_persistence(market),
            recent_transitions=self.get_transitions(market),
            recommendation=self.get_recommendation(market),
        )

    def get_regime_distribution(
        self,
        market: str,
    ) -> Dict[str, float]:
        """Get distribution of regimes over history.

        Args:
            market: Market symbol

        Returns:
            Dict of regime -> percentage
        """
        history = self._regime_history.get(market, [])
        if not history:
            return {}

        counts = {}
        for state in history:
            regime_name = state.regime.value
            counts[regime_name] = counts.get(regime_name, 0) + 1

        total = len(history)
        return {k: (v / total) * 100 for k, v in counts.items()}

    def add_callback(self, callback: Callable[[str, RegimeState], None]) -> None:
        """Add regime change callback.

        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_markets(self) -> List[str]:
        """Get list of tracked markets."""
        return list(self._prices.keys())

    def get_sample_count(self, market: str) -> int:
        """Get number of samples for market."""
        return len(self._prices.get(market, []))

    def clear_market(self, market: str) -> None:
        """Clear data for market."""
        for store in [
            self._prices,
            self._returns,
            self._regime_history,
            self._persistence,
            self._transitions,
        ]:
            if market in store:
                del store[market]

    def clear_all(self) -> None:
        """Clear all data."""
        self._prices.clear()
        self._returns.clear()
        self._volatility_history.clear()
        self._regime_history.clear()
        self._persistence.clear()
        self._transitions.clear()


class MultiMarketRegimeAnalyzer:
    """Analyze regimes across multiple markets."""

    def __init__(self):
        """Initialize multi-market analyzer."""
        self._detectors: Dict[str, RegimeDetector] = {}

    def get_or_create_detector(self, market: str) -> RegimeDetector:
        """Get or create detector for market.

        Args:
            market: Market symbol

        Returns:
            Regime detector
        """
        if market not in self._detectors:
            self._detectors[market] = RegimeDetector()
        return self._detectors[market]

    def add_price(
        self,
        market: str,
        price: Decimal,
        timestamp: Optional[float] = None,
    ) -> Optional[RegimeState]:
        """Add price for market.

        Args:
            market: Market symbol
            price: Price
            timestamp: Timestamp

        Returns:
            Regime state if available
        """
        detector = self.get_or_create_detector(market)
        return detector.add_price(market, price, timestamp)

    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary across all markets.

        Returns:
            Summary dict
        """
        summary = {
            "markets": {},
            "regime_counts": {},
            "overall_sentiment": "neutral",
        }

        bullish_count = 0
        bearish_count = 0

        for market, detector in self._detectors.items():
            state = detector.get_current_regime(market)
            if state:
                summary["markets"][market] = state.regime.value

                if state.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND]:
                    bullish_count += 1
                elif state.regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND]:
                    bearish_count += 1

                regime_name = state.regime.value
                summary["regime_counts"][regime_name] = summary["regime_counts"].get(regime_name, 0) + 1

        total = len(self._detectors)
        if total > 0:
            if bullish_count > bearish_count + total * 0.2:
                summary["overall_sentiment"] = "bullish"
            elif bearish_count > bullish_count + total * 0.2:
                summary["overall_sentiment"] = "bearish"

        return summary

    def get_aligned_markets(
        self,
        regime: MarketRegime,
    ) -> List[str]:
        """Get markets in specific regime.

        Args:
            regime: Target regime

        Returns:
            List of market symbols
        """
        aligned = []
        for market, detector in self._detectors.items():
            state = detector.get_current_regime(market)
            if state and state.regime == regime:
                aligned.append(market)
        return aligned

    def get_market_count(self) -> int:
        """Get number of tracked markets."""
        return len(self._detectors)


# Global regime detector instance
_regime_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    """Get global regime detector."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector


def reset_regime_detector() -> None:
    """Reset global regime detector."""
    global _regime_detector
    _regime_detector = None
