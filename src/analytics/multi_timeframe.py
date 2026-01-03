"""
Multi-Timeframe Analysis for Paradex.

Analyzes multiple timeframes to find confluence signals and trend alignment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import logging


logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Trading timeframes."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"

    @property
    def minutes(self) -> int:
        """Get timeframe duration in minutes."""
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
        }
        return mapping.get(self.value, 1)


class TrendDirection(Enum):
    """Trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class BiasStrength(Enum):
    """Overall bias strength."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


class SignalType(Enum):
    """Signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }


@dataclass
class TimeframeData:
    """Data for a single timeframe."""
    timeframe: Timeframe
    candles: list[OHLCV] = field(default_factory=list)
    trend: TrendDirection = TrendDirection.NEUTRAL
    trend_strength: Decimal = Decimal("0")
    support_levels: list[Decimal] = field(default_factory=list)
    resistance_levels: list[Decimal] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe.value,
            "candle_count": len(self.candles),
            "trend": self.trend.value,
            "trend_strength": str(self.trend_strength),
            "support_levels": [str(s) for s in self.support_levels],
            "resistance_levels": [str(r) for r in self.resistance_levels],
            "indicators": self.indicators,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class MTFSignal:
    """Multi-timeframe signal."""
    symbol: str
    signal_type: SignalType
    confidence: Decimal
    timeframes_aligned: list[Timeframe]
    timeframes_opposing: list[Timeframe]
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    risk_reward: Optional[Decimal] = None
    notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": str(self.confidence),
            "timeframes_aligned": [tf.value for tf in self.timeframes_aligned],
            "timeframes_opposing": [tf.value for tf in self.timeframes_opposing],
            "entry_price": str(self.entry_price) if self.entry_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "risk_reward": str(self.risk_reward) if self.risk_reward else None,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MTFAnalysis:
    """Multi-timeframe analysis result."""
    symbol: str
    overall_trend: TrendDirection
    overall_bias: SignalType
    bias_strength: BiasStrength
    confluence_score: Decimal
    timeframe_analyses: dict[Timeframe, TimeframeData]
    signals: list[MTFSignal] = field(default_factory=list)
    key_levels: dict = field(default_factory=dict)
    analysis_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "overall_trend": self.overall_trend.value,
            "overall_bias": self.overall_bias.value,
            "bias_strength": self.bias_strength.value,
            "confluence_score": str(self.confluence_score),
            "timeframe_analyses": {
                tf.value: data.to_dict()
                for tf, data in self.timeframe_analyses.items()
            },
            "signals": [s.to_dict() for s in self.signals],
            "key_levels": self.key_levels,
            "analysis_time": self.analysis_time.isoformat(),
        }


class TrendAnalyzer:
    """Analyze trend for a single timeframe."""

    def __init__(self, sma_period: int = 20, ema_period: int = 9):
        """Initialize trend analyzer."""
        self.sma_period = sma_period
        self.ema_period = ema_period

    def analyze(self, candles: list[OHLCV]) -> tuple[TrendDirection, Decimal]:
        """Analyze trend direction and strength."""
        if len(candles) < max(self.sma_period, self.ema_period):
            return TrendDirection.NEUTRAL, Decimal("0")

        # Calculate SMA
        closes = [c.close for c in candles]
        sma = sum(closes[-self.sma_period:]) / self.sma_period

        # Calculate EMA
        ema = self._calculate_ema(closes, self.ema_period)

        # Current price
        current_price = closes[-1]

        # Calculate trend strength
        price_vs_sma = (current_price - sma) / sma * 100 if sma != 0 else Decimal("0")
        price_vs_ema = (current_price - ema) / ema * 100 if ema != 0 else Decimal("0")

        # Higher highs / lower lows analysis
        recent_highs = [c.high for c in candles[-10:]]
        recent_lows = [c.low for c in candles[-10:]]

        higher_highs = all(
            recent_highs[i] >= recent_highs[i - 1]
            for i in range(1, len(recent_highs))
        )
        lower_lows = all(
            recent_lows[i] <= recent_lows[i - 1]
            for i in range(1, len(recent_lows))
        )

        # Determine trend
        avg_deviation = (abs(price_vs_sma) + abs(price_vs_ema)) / 2

        if price_vs_sma > Decimal("2") and price_vs_ema > Decimal("1") and higher_highs:
            return TrendDirection.STRONG_UP, min(avg_deviation, Decimal("100"))
        elif price_vs_sma > Decimal("0.5") and price_vs_ema > Decimal("0"):
            return TrendDirection.UP, min(avg_deviation, Decimal("100"))
        elif price_vs_sma < Decimal("-2") and price_vs_ema < Decimal("-1") and lower_lows:
            return TrendDirection.STRONG_DOWN, min(avg_deviation, Decimal("100"))
        elif price_vs_sma < Decimal("-0.5") and price_vs_ema < Decimal("0"):
            return TrendDirection.DOWN, min(avg_deviation, Decimal("100"))
        else:
            return TrendDirection.NEUTRAL, Decimal("0")

    def _calculate_ema(self, prices: list[Decimal], period: int) -> Decimal:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else Decimal("0")

        multiplier = Decimal(2) / (Decimal(period) + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema


class SupportResistanceFinder:
    """Find support and resistance levels."""

    def __init__(self, sensitivity: int = 3, max_levels: int = 5):
        """Initialize support/resistance finder."""
        self.sensitivity = sensitivity
        self.max_levels = max_levels

    def find_levels(self, candles: list[OHLCV]) -> tuple[list[Decimal], list[Decimal]]:
        """Find support and resistance levels."""
        if len(candles) < self.sensitivity * 2:
            return [], []

        supports = []
        resistances = []

        # Find pivot points
        for i in range(self.sensitivity, len(candles) - self.sensitivity):
            is_support = True
            is_resistance = True

            for j in range(i - self.sensitivity, i + self.sensitivity + 1):
                if j == i:
                    continue
                if candles[j].low < candles[i].low:
                    is_support = False
                if candles[j].high > candles[i].high:
                    is_resistance = False

            if is_support:
                supports.append(candles[i].low)
            if is_resistance:
                resistances.append(candles[i].high)

        # Cluster nearby levels
        supports = self._cluster_levels(supports)
        resistances = self._cluster_levels(resistances)

        # Sort and limit
        supports = sorted(supports, reverse=True)[:self.max_levels]
        resistances = sorted(resistances)[:self.max_levels]

        return supports, resistances

    def _cluster_levels(self, levels: list[Decimal], threshold_pct: Decimal = Decimal("0.5")) -> list[Decimal]:
        """Cluster nearby levels."""
        if not levels:
            return []

        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if current_cluster:
                avg = sum(current_cluster) / len(current_cluster)
                threshold = avg * threshold_pct / 100
                if abs(level - avg) <= threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            else:
                current_cluster = [level]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters


class IndicatorCalculator:
    """Calculate common indicators for MTF analysis."""

    def calculate_rsi(self, candles: list[OHLCV], period: int = 14) -> Optional[Decimal]:
        """Calculate RSI."""
        if len(candles) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(candles)):
            change = candles[i].close - candles[i - 1].close
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        if len(gains) < period:
            return None

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        candles: list[OHLCV],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate MACD line, signal line, and histogram."""
        if len(candles) < slow + signal:
            return None, None, None

        closes = [c.close for c in candles]

        # Calculate EMAs
        fast_ema = self._ema(closes, fast)
        slow_ema = self._ema(closes, slow)

        if fast_ema is None or slow_ema is None:
            return None, None, None

        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD)
        macd_values = []
        for i in range(slow - 1, len(closes)):
            f_ema = self._ema(closes[:i + 1], fast)
            s_ema = self._ema(closes[:i + 1], slow)
            if f_ema and s_ema:
                macd_values.append(f_ema - s_ema)

        if len(macd_values) < signal:
            return macd_line, None, None

        signal_line = sum(macd_values[-signal:]) / signal
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _ema(self, prices: list[Decimal], period: int) -> Optional[Decimal]:
        """Calculate EMA."""
        if len(prices) < period:
            return None

        multiplier = Decimal(2) / (Decimal(period) + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema


class TimeframeCorrelator:
    """Correlate analysis across timeframes."""

    def __init__(self, timeframe_weights: Optional[dict[Timeframe, Decimal]] = None):
        """Initialize correlator with timeframe weights."""
        self.weights = timeframe_weights or {
            Timeframe.M5: Decimal("0.05"),
            Timeframe.M15: Decimal("0.10"),
            Timeframe.H1: Decimal("0.20"),
            Timeframe.H4: Decimal("0.30"),
            Timeframe.D1: Decimal("0.35"),
        }

    def calculate_confluence(
        self,
        analyses: dict[Timeframe, TimeframeData],
    ) -> tuple[TrendDirection, BiasStrength, Decimal]:
        """Calculate confluence score across timeframes."""
        if not analyses:
            return TrendDirection.NEUTRAL, BiasStrength.NEUTRAL, Decimal("0")

        bullish_score = Decimal("0")
        bearish_score = Decimal("0")
        total_weight = Decimal("0")

        for timeframe, data in analyses.items():
            weight = self.weights.get(timeframe, Decimal("0.1"))
            total_weight += weight

            if data.trend in (TrendDirection.STRONG_UP, TrendDirection.UP):
                strength_mult = Decimal("1.5") if data.trend == TrendDirection.STRONG_UP else Decimal("1")
                bullish_score += weight * strength_mult * (1 + data.trend_strength / 100)
            elif data.trend in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN):
                strength_mult = Decimal("1.5") if data.trend == TrendDirection.STRONG_DOWN else Decimal("1")
                bearish_score += weight * strength_mult * (1 + data.trend_strength / 100)

        if total_weight == 0:
            return TrendDirection.NEUTRAL, BiasStrength.NEUTRAL, Decimal("0")

        bullish_pct = bullish_score / total_weight * 100
        bearish_pct = bearish_score / total_weight * 100

        confluence = max(bullish_pct, bearish_pct)

        # Determine overall trend
        if bullish_pct > bearish_pct + Decimal("30"):
            trend = TrendDirection.STRONG_UP
        elif bullish_pct > bearish_pct + Decimal("10"):
            trend = TrendDirection.UP
        elif bearish_pct > bullish_pct + Decimal("30"):
            trend = TrendDirection.STRONG_DOWN
        elif bearish_pct > bullish_pct + Decimal("10"):
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL

        # Determine strength
        if confluence >= Decimal("80"):
            strength = BiasStrength.VERY_STRONG
        elif confluence >= Decimal("60"):
            strength = BiasStrength.STRONG
        elif confluence >= Decimal("40"):
            strength = BiasStrength.MODERATE
        elif confluence >= Decimal("20"):
            strength = BiasStrength.WEAK
        else:
            strength = BiasStrength.NEUTRAL

        return trend, strength, confluence

    def find_aligned_timeframes(
        self,
        analyses: dict[Timeframe, TimeframeData],
        direction: TrendDirection,
    ) -> tuple[list[Timeframe], list[Timeframe]]:
        """Find aligned and opposing timeframes."""
        aligned = []
        opposing = []

        is_bullish = direction in (TrendDirection.STRONG_UP, TrendDirection.UP)
        is_bearish = direction in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN)

        for timeframe, data in analyses.items():
            data_bullish = data.trend in (TrendDirection.STRONG_UP, TrendDirection.UP)
            data_bearish = data.trend in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN)

            if (is_bullish and data_bullish) or (is_bearish and data_bearish):
                aligned.append(timeframe)
            elif (is_bullish and data_bearish) or (is_bearish and data_bullish):
                opposing.append(timeframe)

        return aligned, opposing


class MTFSignalGenerator:
    """Generate trading signals from MTF analysis."""

    def __init__(self, min_confluence: Decimal = Decimal("50")):
        """Initialize signal generator."""
        self.min_confluence = min_confluence

    def generate(
        self,
        symbol: str,
        analyses: dict[Timeframe, TimeframeData],
        overall_trend: TrendDirection,
        confluence: Decimal,
    ) -> list[MTFSignal]:
        """Generate trading signals."""
        signals = []

        if confluence < self.min_confluence:
            return signals

        # Determine signal type
        if overall_trend in (TrendDirection.STRONG_UP, TrendDirection.UP):
            signal_type = SignalType.BUY
        elif overall_trend in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN):
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        if signal_type == SignalType.HOLD:
            return signals

        # Find aligned/opposing timeframes
        aligned = []
        opposing = []
        for tf, data in analyses.items():
            data_bullish = data.trend in (TrendDirection.STRONG_UP, TrendDirection.UP)
            data_bearish = data.trend in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN)

            if signal_type == SignalType.BUY:
                if data_bullish:
                    aligned.append(tf)
                elif data_bearish:
                    opposing.append(tf)
            else:
                if data_bearish:
                    aligned.append(tf)
                elif data_bullish:
                    opposing.append(tf)

        # Calculate entry, SL, TP from key levels
        entry_price, stop_loss, take_profit = self._calculate_levels(
            analyses, signal_type
        )

        # Calculate confidence
        alignment_ratio = Decimal(len(aligned)) / Decimal(max(len(aligned) + len(opposing), 1))
        confidence = min(
            confluence / 100 * alignment_ratio,
            Decimal("1")
        )

        # Calculate risk/reward
        risk_reward = None
        if entry_price and stop_loss and take_profit:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                risk_reward = reward / risk

        notes = []
        if len(aligned) >= 3:
            notes.append(f"Strong confluence: {len(aligned)} timeframes aligned")
        if opposing:
            notes.append(f"Caution: {len(opposing)} timeframes opposing")

        signal = MTFSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            timeframes_aligned=aligned,
            timeframes_opposing=opposing,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            notes=notes,
        )
        signals.append(signal)

        return signals

    def _calculate_levels(
        self,
        analyses: dict[Timeframe, TimeframeData],
        signal_type: SignalType,
    ) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, SL, TP from analyses."""
        all_supports = []
        all_resistances = []

        for data in analyses.values():
            all_supports.extend(data.support_levels)
            all_resistances.extend(data.resistance_levels)

        if not all_supports or not all_resistances:
            return None, None, None

        current_price = None
        for data in analyses.values():
            if data.candles:
                current_price = data.candles[-1].close
                break

        if current_price is None:
            return None, None, None

        if signal_type == SignalType.BUY:
            # Entry at current price
            entry = current_price
            # Stop below nearest support
            nearby_supports = [s for s in all_supports if s < current_price]
            stop = min(nearby_supports) if nearby_supports else current_price * Decimal("0.98")
            # Target at nearest resistance
            nearby_resistances = [r for r in all_resistances if r > current_price]
            target = min(nearby_resistances) if nearby_resistances else current_price * Decimal("1.05")
        else:
            # Entry at current price
            entry = current_price
            # Stop above nearest resistance
            nearby_resistances = [r for r in all_resistances if r > current_price]
            stop = min(nearby_resistances) if nearby_resistances else current_price * Decimal("1.02")
            # Target at nearest support
            nearby_supports = [s for s in all_supports if s < current_price]
            target = max(nearby_supports) if nearby_supports else current_price * Decimal("0.95")

        return entry, stop, target


class MultiTimeframeAnalyzer:
    """Main multi-timeframe analysis engine."""

    def __init__(
        self,
        timeframes: Optional[list[Timeframe]] = None,
        min_confluence: Decimal = Decimal("50"),
    ):
        """Initialize MTF analyzer."""
        self.timeframes = timeframes or [
            Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1
        ]
        self.trend_analyzer = TrendAnalyzer()
        self.sr_finder = SupportResistanceFinder()
        self.indicator_calc = IndicatorCalculator()
        self.correlator = TimeframeCorrelator()
        self.signal_gen = MTFSignalGenerator(min_confluence)
        self._analyses: dict[str, MTFAnalysis] = {}
        self._callbacks: list[Callable] = []

    def analyze(
        self,
        symbol: str,
        data: dict[Timeframe, list[OHLCV]],
    ) -> MTFAnalysis:
        """Perform multi-timeframe analysis."""
        timeframe_analyses: dict[Timeframe, TimeframeData] = {}

        for timeframe, candles in data.items():
            if not candles:
                continue

            # Analyze trend
            trend, strength = self.trend_analyzer.analyze(candles)

            # Find S/R levels
            supports, resistances = self.sr_finder.find_levels(candles)

            # Calculate indicators
            indicators = {}
            rsi = self.indicator_calc.calculate_rsi(candles)
            if rsi is not None:
                indicators["rsi"] = float(rsi)

            macd, signal, hist = self.indicator_calc.calculate_macd(candles)
            if macd is not None:
                indicators["macd"] = {
                    "macd": float(macd),
                    "signal": float(signal) if signal else None,
                    "histogram": float(hist) if hist else None,
                }

            tf_data = TimeframeData(
                timeframe=timeframe,
                candles=candles,
                trend=trend,
                trend_strength=strength,
                support_levels=supports,
                resistance_levels=resistances,
                indicators=indicators,
            )
            timeframe_analyses[timeframe] = tf_data

        # Calculate confluence
        overall_trend, bias_strength, confluence = self.correlator.calculate_confluence(
            timeframe_analyses
        )

        # Determine overall bias
        if overall_trend in (TrendDirection.STRONG_UP, TrendDirection.UP):
            overall_bias = SignalType.BUY
        elif overall_trend in (TrendDirection.STRONG_DOWN, TrendDirection.DOWN):
            overall_bias = SignalType.SELL
        else:
            overall_bias = SignalType.HOLD

        # Generate signals
        signals = self.signal_gen.generate(
            symbol, timeframe_analyses, overall_trend, confluence
        )

        # Compile key levels
        key_levels = self._compile_key_levels(timeframe_analyses)

        analysis = MTFAnalysis(
            symbol=symbol,
            overall_trend=overall_trend,
            overall_bias=overall_bias,
            bias_strength=bias_strength,
            confluence_score=confluence,
            timeframe_analyses=timeframe_analyses,
            signals=signals,
            key_levels=key_levels,
        )

        self._analyses[symbol] = analysis

        # Trigger callbacks
        for callback in self._callbacks:
            callback(analysis)

        return analysis

    def _compile_key_levels(
        self,
        analyses: dict[Timeframe, TimeframeData],
    ) -> dict:
        """Compile key support/resistance levels."""
        all_supports = []
        all_resistances = []

        for data in analyses.values():
            all_supports.extend([(s, data.timeframe) for s in data.support_levels])
            all_resistances.extend([(r, data.timeframe) for r in data.resistance_levels])

        # Group by price level
        support_clusters = self._cluster_with_timeframes(all_supports)
        resistance_clusters = self._cluster_with_timeframes(all_resistances)

        return {
            "supports": support_clusters[:5],
            "resistances": resistance_clusters[:5],
        }

    def _cluster_with_timeframes(
        self,
        levels: list[tuple[Decimal, Timeframe]],
        threshold_pct: Decimal = Decimal("0.5"),
    ) -> list[dict]:
        """Cluster levels and track which timeframes confirm them."""
        if not levels:
            return []

        levels = sorted(levels, key=lambda x: x[0])
        clusters = []
        current_prices = [levels[0][0]]
        current_tfs = [levels[0][1]]

        for price, tf in levels[1:]:
            avg = sum(current_prices) / len(current_prices)
            threshold = avg * threshold_pct / 100

            if abs(price - avg) <= threshold:
                current_prices.append(price)
                if tf not in current_tfs:
                    current_tfs.append(tf)
            else:
                clusters.append({
                    "price": str(sum(current_prices) / len(current_prices)),
                    "timeframes": [tf.value for tf in current_tfs],
                    "strength": len(current_tfs),
                })
                current_prices = [price]
                current_tfs = [tf]

        if current_prices:
            clusters.append({
                "price": str(sum(current_prices) / len(current_prices)),
                "timeframes": [tf.value for tf in current_tfs],
                "strength": len(current_tfs),
            })

        # Sort by strength (number of confirming timeframes)
        clusters.sort(key=lambda x: x["strength"], reverse=True)

        return clusters

    def get_analysis(self, symbol: str) -> Optional[MTFAnalysis]:
        """Get latest analysis for symbol."""
        return self._analyses.get(symbol)

    def get_trend(self, symbol: str) -> Optional[TrendDirection]:
        """Get overall trend for symbol."""
        analysis = self._analyses.get(symbol)
        return analysis.overall_trend if analysis else None

    def get_bias(self, symbol: str) -> Optional[SignalType]:
        """Get overall bias for symbol."""
        analysis = self._analyses.get(symbol)
        return analysis.overall_bias if analysis else None

    def get_confluence(self, symbol: str) -> Optional[Decimal]:
        """Get confluence score for symbol."""
        analysis = self._analyses.get(symbol)
        return analysis.confluence_score if analysis else None

    def get_signals(self, symbol: str) -> list[MTFSignal]:
        """Get signals for symbol."""
        analysis = self._analyses.get(symbol)
        return analysis.signals if analysis else []

    def add_callback(self, callback: Callable) -> None:
        """Add analysis callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> bool:
        """Remove analysis callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear analyses."""
        if symbol:
            self._analyses.pop(symbol, None)
        else:
            self._analyses.clear()

    def get_summary(self, symbol: Optional[str] = None) -> dict:
        """Get analysis summary."""
        if symbol:
            analysis = self._analyses.get(symbol)
            if analysis:
                return {
                    "symbol": symbol,
                    "trend": analysis.overall_trend.value,
                    "bias": analysis.overall_bias.value,
                    "confluence": str(analysis.confluence_score),
                    "timeframes_analyzed": len(analysis.timeframe_analyses),
                    "signals": len(analysis.signals),
                }
            return {}

        return {
            "symbols_analyzed": list(self._analyses.keys()),
            "total_symbols": len(self._analyses),
            "analyses": {
                sym: {
                    "trend": a.overall_trend.value,
                    "bias": a.overall_bias.value,
                    "confluence": str(a.confluence_score),
                }
                for sym, a in self._analyses.items()
            },
        }


# Global instance
_mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get global MTF analyzer instance."""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer


def set_mtf_analyzer(analyzer: MultiTimeframeAnalyzer) -> None:
    """Set global MTF analyzer instance."""
    global _mtf_analyzer
    _mtf_analyzer = analyzer
