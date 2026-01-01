"""Signal Generator module for trading signals.

This module provides comprehensive signal generation including:
- Technical indicator-based signals
- Signal combination and weighting
- Signal strength and confidence scoring
- Entry/exit signal generation
- Multi-timeframe signal analysis
- Signal persistence tracking
- Custom signal rules
"""

import time
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics


class SignalType(Enum):
    """Type of trading signal."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class SignalStrength(Enum):
    """Strength of signal."""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class IndicatorType(Enum):
    """Type of technical indicator."""

    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


class TimeFrame(Enum):
    """Timeframe for analysis."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class PriceBar:
    """Price bar data."""

    timestamp: float
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }


@dataclass
class IndicatorValue:
    """Value from an indicator."""

    name: str
    value: float
    indicator_type: IndicatorType
    signal: SignalType = SignalType.NEUTRAL
    strength: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "indicator_type": self.indicator_type.value,
            "signal": self.signal.value,
            "strength": self.strength,
        }


@dataclass
class Signal:
    """Trading signal."""

    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-100
    source: str = ""
    price: Decimal = Decimal("0")
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "source": self.source,
            "price": str(self.price),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class CompositeSignal:
    """Composite signal from multiple sources."""

    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    contributing_signals: List[Signal] = field(default_factory=list)
    agreement_score: float = 0.0  # How much signals agree
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "num_contributing": len(self.contributing_signals),
            "timestamp": self.timestamp,
        }


@dataclass
class SignalRule:
    """Custom signal rule."""

    name: str
    condition: Callable[..., bool]
    signal_type: SignalType
    weight: float = 1.0
    enabled: bool = True

    def evaluate(self, *args, **kwargs) -> Optional[Signal]:
        """Evaluate rule.

        Returns:
            Signal if condition met
        """
        if not self.enabled:
            return None

        try:
            if self.condition(*args, **kwargs):
                return Signal(
                    signal_type=self.signal_type,
                    strength=SignalStrength.MODERATE,
                    confidence=50.0 * self.weight,
                    source=self.name,
                )
        except Exception:
            pass

        return None


@dataclass
class SignalPersistence:
    """Track signal persistence over time."""

    signal_type: SignalType
    first_seen: float
    last_seen: float
    occurrence_count: int = 1
    max_strength: SignalStrength = SignalStrength.WEAK
    avg_confidence: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.last_seen - self.first_seen

    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration_seconds / 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "duration_minutes": self.duration_minutes,
            "occurrence_count": self.occurrence_count,
            "max_strength": self.max_strength.value,
            "avg_confidence": self.avg_confidence,
        }


class TechnicalIndicators:
    """Technical indicator calculations."""

    @staticmethod
    def sma(prices: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average.

        Args:
            prices: List of prices
            period: Period

        Returns:
            SMA value or None
        """
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def ema(prices: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate Exponential Moving Average.

        Args:
            prices: List of prices
            period: Period

        Returns:
            EMA value or None
        """
        if len(prices) < period:
            return None

        multiplier = Decimal(str(2 / (period + 1)))
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def rsi(prices: List[Decimal], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index.

        Args:
            prices: List of prices
            period: Period

        Returns:
            RSI value (0-100) or None
        """
        if len(prices) < period + 1:
            return None

        changes = [float(prices[i] - prices[i - 1]) for i in range(1, len(prices))]

        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        prices: List[Decimal],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """Calculate MACD.

        Args:
            prices: List of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, signal line, histogram) or None
        """
        if len(prices) < slow + signal:
            return None

        fast_ema = TechnicalIndicators.ema(prices, fast)
        slow_ema = TechnicalIndicators.ema(prices, slow)

        if fast_ema is None or slow_ema is None:
            return None

        macd_line = fast_ema - slow_ema

        # Calculate signal line from MACD history
        macd_history = []
        for i in range(slow, len(prices) + 1):
            fast_e = TechnicalIndicators.ema(prices[:i], fast)
            slow_e = TechnicalIndicators.ema(prices[:i], slow)
            if fast_e and slow_e:
                macd_history.append(fast_e - slow_e)

        if len(macd_history) < signal:
            return (macd_line, macd_line, Decimal("0"))

        signal_line = sum(macd_history[-signal:]) / signal
        histogram = macd_line - signal_line

        return (macd_line, signal_line, histogram)

    @staticmethod
    def bollinger_bands(
        prices: List[Decimal],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """Calculate Bollinger Bands.

        Args:
            prices: List of prices
            period: Period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper, middle, lower) or None
        """
        if len(prices) < period:
            return None

        recent = prices[-period:]
        middle = sum(recent) / period

        variance = sum((p - middle) ** 2 for p in recent) / period
        std = Decimal(str(math.sqrt(float(variance))))

        upper = middle + std * Decimal(str(std_dev))
        lower = middle - std * Decimal(str(std_dev))

        return (upper, middle, lower)

    @staticmethod
    def stochastic(
        highs: List[Decimal],
        lows: List[Decimal],
        closes: List[Decimal],
        k_period: int = 14,
        d_period: int = 3,
    ) -> Optional[Tuple[float, float]]:
        """Calculate Stochastic Oscillator.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D) or None
        """
        if len(closes) < k_period:
            return None

        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])

        if highest_high == lowest_low:
            return (50.0, 50.0)

        current_close = closes[-1]
        k = float((current_close - lowest_low) / (highest_high - lowest_low) * 100)

        # Calculate %D (SMA of %K)
        k_values = []
        for i in range(k_period, len(closes) + 1):
            hh = max(highs[i - k_period:i])
            ll = min(lows[i - k_period:i])
            if hh != ll:
                k_values.append(float((closes[i - 1] - ll) / (hh - ll) * 100))

        d = sum(k_values[-d_period:]) / d_period if len(k_values) >= d_period else k

        return (k, d)

    @staticmethod
    def atr(
        highs: List[Decimal],
        lows: List[Decimal],
        closes: List[Decimal],
        period: int = 14,
    ) -> Optional[Decimal]:
        """Calculate Average True Range.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: Period

        Returns:
            ATR value or None
        """
        if len(closes) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        return sum(true_ranges[-period:]) / period


class SignalGenerator:
    """Generator for trading signals."""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
    ):
        """Initialize signal generator.

        Args:
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            stoch_oversold: Stochastic oversold threshold
            stoch_overbought: Stochastic overbought threshold
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought

        self._bars: Dict[str, List[PriceBar]] = {}
        self._signals: Dict[str, List[Signal]] = {}
        self._persistence: Dict[str, Dict[str, SignalPersistence]] = {}
        self._custom_rules: List[SignalRule] = []
        self._callbacks: List[Callable] = []
        self._max_bars = 500

    def add_bar(
        self,
        market: str,
        timestamp: float,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> List[Signal]:
        """Add price bar and generate signals.

        Args:
            market: Market symbol
            timestamp: Timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Volume

        Returns:
            List of generated signals
        """
        if market not in self._bars:
            self._bars[market] = []
            self._signals[market] = []
            self._persistence[market] = {}

        bar = PriceBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

        self._bars[market].append(bar)

        # Trim history
        if len(self._bars[market]) > self._max_bars:
            self._bars[market] = self._bars[market][-self._max_bars:]

        # Generate signals
        signals = self._generate_signals(market)

        # Track persistence
        for signal in signals:
            self._update_persistence(market, signal)

        # Store signals
        self._signals[market].extend(signals)
        if len(self._signals[market]) > self._max_bars:
            self._signals[market] = self._signals[market][-self._max_bars:]

        # Notify callbacks
        for callback in self._callbacks:
            for signal in signals:
                callback(market, signal)

        return signals

    def _generate_signals(self, market: str) -> List[Signal]:
        """Generate signals from indicators.

        Args:
            market: Market symbol

        Returns:
            List of signals
        """
        bars = self._bars.get(market, [])
        if len(bars) < 26:  # Need enough data for MACD
            return []

        signals = []
        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        current_price = closes[-1]

        # RSI Signal
        rsi = TechnicalIndicators.rsi(closes)
        if rsi is not None:
            signal = self._evaluate_rsi(rsi, current_price)
            if signal:
                signals.append(signal)

        # MACD Signal
        macd_result = TechnicalIndicators.macd(closes)
        if macd_result:
            signal = self._evaluate_macd(macd_result, current_price)
            if signal:
                signals.append(signal)

        # Bollinger Bands Signal
        bb = TechnicalIndicators.bollinger_bands(closes)
        if bb:
            signal = self._evaluate_bollinger(bb, current_price)
            if signal:
                signals.append(signal)

        # Stochastic Signal
        stoch = TechnicalIndicators.stochastic(highs, lows, closes)
        if stoch:
            signal = self._evaluate_stochastic(stoch, current_price)
            if signal:
                signals.append(signal)

        # Moving Average Crossover
        sma_short = TechnicalIndicators.sma(closes, 10)
        sma_long = TechnicalIndicators.sma(closes, 30)
        if sma_short and sma_long:
            signal = self._evaluate_ma_crossover(sma_short, sma_long, current_price)
            if signal:
                signals.append(signal)

        # Custom rules
        for rule in self._custom_rules:
            rule_signal = rule.evaluate(bars=bars, closes=closes, highs=highs, lows=lows)
            if rule_signal:
                signals.append(rule_signal)

        return signals

    def _evaluate_rsi(self, rsi: float, price: Decimal) -> Optional[Signal]:
        """Evaluate RSI for signal.

        Args:
            rsi: RSI value
            price: Current price

        Returns:
            Signal or None
        """
        if rsi <= self.rsi_oversold:
            strength_val = (self.rsi_oversold - rsi) / self.rsi_oversold
            return Signal(
                signal_type=SignalType.LONG,
                strength=self._get_strength(strength_val),
                confidence=min(100, 50 + strength_val * 50),
                source="rsi",
                price=price,
                metadata={"rsi": rsi},
            )
        elif rsi >= self.rsi_overbought:
            strength_val = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return Signal(
                signal_type=SignalType.SHORT,
                strength=self._get_strength(strength_val),
                confidence=min(100, 50 + strength_val * 50),
                source="rsi",
                price=price,
                metadata={"rsi": rsi},
            )
        return None

    def _evaluate_macd(
        self,
        macd_result: Tuple[Decimal, Decimal, Decimal],
        price: Decimal,
    ) -> Optional[Signal]:
        """Evaluate MACD for signal.

        Args:
            macd_result: MACD values
            price: Current price

        Returns:
            Signal or None
        """
        macd_line, signal_line, histogram = macd_result

        if histogram > 0 and macd_line > signal_line:
            strength_val = min(1, float(abs(histogram)) / 100)
            return Signal(
                signal_type=SignalType.LONG,
                strength=self._get_strength(strength_val),
                confidence=50 + strength_val * 30,
                source="macd",
                price=price,
                metadata={"macd": float(macd_line), "signal": float(signal_line)},
            )
        elif histogram < 0 and macd_line < signal_line:
            strength_val = min(1, float(abs(histogram)) / 100)
            return Signal(
                signal_type=SignalType.SHORT,
                strength=self._get_strength(strength_val),
                confidence=50 + strength_val * 30,
                source="macd",
                price=price,
                metadata={"macd": float(macd_line), "signal": float(signal_line)},
            )
        return None

    def _evaluate_bollinger(
        self,
        bb: Tuple[Decimal, Decimal, Decimal],
        price: Decimal,
    ) -> Optional[Signal]:
        """Evaluate Bollinger Bands for signal.

        Args:
            bb: Bollinger Bands values
            price: Current price

        Returns:
            Signal or None
        """
        upper, middle, lower = bb

        if price <= lower:
            width = upper - lower
            if width > 0:
                strength_val = min(1, float((lower - price) / width))
            else:
                strength_val = 0.5
            return Signal(
                signal_type=SignalType.LONG,
                strength=self._get_strength(strength_val),
                confidence=55 + strength_val * 25,
                source="bollinger",
                price=price,
                metadata={"upper": float(upper), "lower": float(lower)},
            )
        elif price >= upper:
            width = upper - lower
            if width > 0:
                strength_val = min(1, float((price - upper) / width))
            else:
                strength_val = 0.5
            return Signal(
                signal_type=SignalType.SHORT,
                strength=self._get_strength(strength_val),
                confidence=55 + strength_val * 25,
                source="bollinger",
                price=price,
                metadata={"upper": float(upper), "lower": float(lower)},
            )
        return None

    def _evaluate_stochastic(
        self,
        stoch: Tuple[float, float],
        price: Decimal,
    ) -> Optional[Signal]:
        """Evaluate Stochastic for signal.

        Args:
            stoch: Stochastic values
            price: Current price

        Returns:
            Signal or None
        """
        k, d = stoch

        if k <= self.stoch_oversold and d <= self.stoch_oversold:
            strength_val = (self.stoch_oversold - min(k, d)) / self.stoch_oversold
            return Signal(
                signal_type=SignalType.LONG,
                strength=self._get_strength(strength_val),
                confidence=50 + strength_val * 30,
                source="stochastic",
                price=price,
                metadata={"k": k, "d": d},
            )
        elif k >= self.stoch_overbought and d >= self.stoch_overbought:
            strength_val = (max(k, d) - self.stoch_overbought) / (100 - self.stoch_overbought)
            return Signal(
                signal_type=SignalType.SHORT,
                strength=self._get_strength(strength_val),
                confidence=50 + strength_val * 30,
                source="stochastic",
                price=price,
                metadata={"k": k, "d": d},
            )
        return None

    def _evaluate_ma_crossover(
        self,
        sma_short: Decimal,
        sma_long: Decimal,
        price: Decimal,
    ) -> Optional[Signal]:
        """Evaluate MA crossover for signal.

        Args:
            sma_short: Short SMA
            sma_long: Long SMA
            price: Current price

        Returns:
            Signal or None
        """
        if sma_short > sma_long:
            diff = float((sma_short - sma_long) / sma_long * 100)
            strength_val = min(1, diff / 5)  # 5% diff = full strength
            return Signal(
                signal_type=SignalType.LONG,
                strength=self._get_strength(strength_val),
                confidence=45 + strength_val * 25,
                source="ma_crossover",
                price=price,
                metadata={"sma_short": float(sma_short), "sma_long": float(sma_long)},
            )
        elif sma_short < sma_long:
            diff = float((sma_long - sma_short) / sma_long * 100)
            strength_val = min(1, diff / 5)
            return Signal(
                signal_type=SignalType.SHORT,
                strength=self._get_strength(strength_val),
                confidence=45 + strength_val * 25,
                source="ma_crossover",
                price=price,
                metadata={"sma_short": float(sma_short), "sma_long": float(sma_long)},
            )
        return None

    def _get_strength(self, value: float) -> SignalStrength:
        """Get strength enum from value.

        Args:
            value: Strength value (0-1)

        Returns:
            Signal strength
        """
        if value >= 0.8:
            return SignalStrength.VERY_STRONG
        elif value >= 0.6:
            return SignalStrength.STRONG
        elif value >= 0.4:
            return SignalStrength.MODERATE
        elif value >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _update_persistence(self, market: str, signal: Signal) -> None:
        """Update signal persistence tracking.

        Args:
            market: Market symbol
            signal: Signal
        """
        key = f"{signal.signal_type.value}_{signal.source}"

        if key not in self._persistence[market]:
            self._persistence[market][key] = SignalPersistence(
                signal_type=signal.signal_type,
                first_seen=signal.timestamp,
                last_seen=signal.timestamp,
                max_strength=signal.strength,
                avg_confidence=signal.confidence,
            )
        else:
            p = self._persistence[market][key]
            p.last_seen = signal.timestamp
            p.occurrence_count += 1
            if self._strength_to_value(signal.strength) > self._strength_to_value(p.max_strength):
                p.max_strength = signal.strength
            # Update rolling average confidence
            p.avg_confidence = (p.avg_confidence * (p.occurrence_count - 1) + signal.confidence) / p.occurrence_count

    def _strength_to_value(self, strength: SignalStrength) -> int:
        """Convert strength to numeric value."""
        mapping = {
            SignalStrength.VERY_WEAK: 1,
            SignalStrength.WEAK: 2,
            SignalStrength.MODERATE: 3,
            SignalStrength.STRONG: 4,
            SignalStrength.VERY_STRONG: 5,
        }
        return mapping.get(strength, 0)

    def get_composite_signal(
        self,
        market: str,
        lookback: int = 5,
    ) -> CompositeSignal:
        """Get composite signal from recent signals.

        Args:
            market: Market symbol
            lookback: Number of recent bars to consider

        Returns:
            Composite signal
        """
        signals = self._signals.get(market, [])[-lookback * 5:]  # ~5 signals per bar

        if not signals:
            return CompositeSignal(
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.VERY_WEAK,
                confidence=0,
            )

        # Count signal types
        long_count = sum(1 for s in signals if s.signal_type == SignalType.LONG)
        short_count = sum(1 for s in signals if s.signal_type == SignalType.SHORT)
        total = long_count + short_count

        if total == 0:
            return CompositeSignal(
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.VERY_WEAK,
                confidence=0,
            )

        # Determine dominant signal
        if long_count > short_count:
            signal_type = SignalType.LONG
            agreement = long_count / total
            relevant = [s for s in signals if s.signal_type == SignalType.LONG]
        elif short_count > long_count:
            signal_type = SignalType.SHORT
            agreement = short_count / total
            relevant = [s for s in signals if s.signal_type == SignalType.SHORT]
        else:
            signal_type = SignalType.NEUTRAL
            agreement = 0.5
            relevant = signals

        # Calculate average confidence
        avg_confidence = statistics.mean([s.confidence for s in relevant]) if relevant else 0

        # Determine strength from agreement
        strength = self._get_strength(agreement)

        return CompositeSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            contributing_signals=relevant,
            agreement_score=agreement * 100,
        )

    def add_custom_rule(self, rule: SignalRule) -> None:
        """Add custom signal rule.

        Args:
            rule: Signal rule
        """
        self._custom_rules.append(rule)

    def remove_custom_rule(self, name: str) -> bool:
        """Remove custom rule by name.

        Args:
            name: Rule name

        Returns:
            True if removed
        """
        for i, rule in enumerate(self._custom_rules):
            if rule.name == name:
                self._custom_rules.pop(i)
                return True
        return False

    def add_callback(self, callback: Callable[[str, Signal], None]) -> None:
        """Add signal callback.

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

    def get_signals(self, market: str, limit: int = 50) -> List[Signal]:
        """Get recent signals.

        Args:
            market: Market symbol
            limit: Maximum number of signals

        Returns:
            List of signals
        """
        return self._signals.get(market, [])[-limit:]

    def get_persistence(self, market: str) -> Dict[str, SignalPersistence]:
        """Get signal persistence data.

        Args:
            market: Market symbol

        Returns:
            Persistence data
        """
        return self._persistence.get(market, {})

    def get_markets(self) -> List[str]:
        """Get list of markets."""
        return list(self._bars.keys())

    def get_bar_count(self, market: str) -> int:
        """Get number of bars for market."""
        return len(self._bars.get(market, []))

    def clear_market(self, market: str) -> None:
        """Clear data for market."""
        for store in [self._bars, self._signals, self._persistence]:
            if market in store:
                del store[market]

    def clear_all(self) -> None:
        """Clear all data."""
        self._bars.clear()
        self._signals.clear()
        self._persistence.clear()


class MultiTimeframeSignals:
    """Multi-timeframe signal analysis."""

    def __init__(self, timeframes: Optional[List[TimeFrame]] = None):
        """Initialize multi-timeframe analyzer.

        Args:
            timeframes: Timeframes to analyze
        """
        self.timeframes = timeframes or [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]
        self._generators: Dict[TimeFrame, SignalGenerator] = {}

        for tf in self.timeframes:
            self._generators[tf] = SignalGenerator()

    def add_bar(
        self,
        market: str,
        timeframe: TimeFrame,
        timestamp: float,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> List[Signal]:
        """Add bar to specific timeframe.

        Args:
            market: Market symbol
            timeframe: Timeframe
            timestamp: Timestamp
            open_price: Open
            high_price: High
            low_price: Low
            close_price: Close
            volume: Volume

        Returns:
            Generated signals
        """
        if timeframe not in self._generators:
            return []

        return self._generators[timeframe].add_bar(
            market, timestamp, open_price, high_price, low_price, close_price, volume
        )

    def get_alignment(self, market: str) -> Dict[str, Any]:
        """Get timeframe alignment analysis.

        Args:
            market: Market symbol

        Returns:
            Alignment analysis
        """
        composites = {}
        for tf, gen in self._generators.items():
            composites[tf.value] = gen.get_composite_signal(market)

        # Check alignment
        signal_types = [c.signal_type for c in composites.values()]
        long_count = sum(1 for s in signal_types if s == SignalType.LONG)
        short_count = sum(1 for s in signal_types if s == SignalType.SHORT)
        total = len(signal_types)

        if long_count == total:
            alignment = "fully_bullish"
        elif short_count == total:
            alignment = "fully_bearish"
        elif long_count > short_count:
            alignment = "mostly_bullish"
        elif short_count > long_count:
            alignment = "mostly_bearish"
        else:
            alignment = "mixed"

        return {
            "alignment": alignment,
            "timeframe_signals": {tf: c.to_dict() for tf, c in composites.items()},
            "bullish_count": long_count,
            "bearish_count": short_count,
            "confidence": max(long_count, short_count) / total * 100 if total > 0 else 0,
        }

    def get_generator(self, timeframe: TimeFrame) -> Optional[SignalGenerator]:
        """Get generator for timeframe."""
        return self._generators.get(timeframe)


# Global signal generator instance
_signal_generator: Optional[SignalGenerator] = None


def get_signal_generator() -> SignalGenerator:
    """Get global signal generator."""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SignalGenerator()
    return _signal_generator


def reset_signal_generator() -> None:
    """Reset global signal generator."""
    global _signal_generator
    _signal_generator = None
