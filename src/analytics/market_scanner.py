"""
Market Scanner Module.

Comprehensive market scanning for identifying trading opportunities
based on various criteria and filters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import math


class ScannerType(Enum):
    """Scanner type."""
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    TREND = "trend"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"
    PATTERN = "pattern"
    FUNDING = "funding"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"


class ScanCondition(Enum):
    """Scan condition types."""
    ABOVE = "above"
    BELOW = "below"
    BETWEEN = "between"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    INCREASES_BY = "increases_by"
    DECREASES_BY = "decreases_by"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    TOP_N = "top_n"
    BOTTOM_N = "bottom_n"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """Trend direction."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class SignalStrength(Enum):
    """Signal strength."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class MarketData:
    """Market data point."""
    symbol: str
    price: Decimal
    change_24h: Decimal
    change_1h: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    open_interest: Decimal = Decimal("0")
    funding_rate: Decimal = Decimal("0")
    bid: Decimal = Decimal("0")
    ask: Decimal = Decimal("0")
    spread: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": str(self.price),
            "change_24h": str(self.change_24h),
            "change_1h": str(self.change_1h),
            "volume_24h": str(self.volume_24h),
            "high_24h": str(self.high_24h),
            "low_24h": str(self.low_24h),
            "open_interest": str(self.open_interest),
            "funding_rate": str(self.funding_rate),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "spread": str(self.spread),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScanFilter:
    """Scan filter criteria."""
    field: str
    condition: ScanCondition
    value: Any
    value2: Optional[Any] = None  # For BETWEEN condition

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "condition": self.condition.value,
            "value": str(self.value) if isinstance(self.value, Decimal) else self.value,
            "value2": str(self.value2) if isinstance(self.value2, Decimal) else self.value2
        }


@dataclass
class ScanResult:
    """Result from scanner."""
    symbol: str
    score: Decimal
    signals: list[str]
    data: MarketData
    timestamp: datetime = field(default_factory=datetime.now)
    priority: AlertPriority = AlertPriority.MEDIUM
    trend: TrendDirection = TrendDirection.SIDEWAYS
    strength: SignalStrength = SignalStrength.MODERATE
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": str(self.score),
            "signals": self.signals,
            "data": self.data.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "trend": self.trend.value,
            "strength": self.strength.value,
            "metadata": self.metadata
        }


@dataclass
class ScannerConfig:
    """Scanner configuration."""
    scanner_type: ScannerType
    filters: list[ScanFilter] = field(default_factory=list)
    min_volume: Decimal = Decimal("0")
    max_spread_pct: Decimal = Decimal("1")  # 1%
    include_symbols: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    top_n: int = 10
    refresh_interval: int = 60  # seconds

    def to_dict(self) -> dict:
        return {
            "scanner_type": self.scanner_type.value,
            "filters": [f.to_dict() for f in self.filters],
            "min_volume": str(self.min_volume),
            "max_spread_pct": str(self.max_spread_pct),
            "include_symbols": self.include_symbols,
            "exclude_symbols": self.exclude_symbols,
            "top_n": self.top_n,
            "refresh_interval": self.refresh_interval
        }


@dataclass
class Alert:
    """Scanner alert."""
    id: str
    timestamp: datetime
    symbol: str
    alert_type: str
    message: str
    priority: AlertPriority
    data: dict
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "alert_type": self.alert_type,
            "message": self.message,
            "priority": self.priority.value,
            "data": self.data,
            "acknowledged": self.acknowledged
        }


class FilterEvaluator:
    """Evaluate scan filters."""

    def evaluate(
        self,
        data: MarketData,
        filter: ScanFilter,
        prev_data: Optional[MarketData] = None
    ) -> bool:
        """Evaluate if data passes filter."""
        value = self._get_field_value(data, filter.field)
        if value is None:
            return False

        if filter.condition == ScanCondition.ABOVE:
            return value > Decimal(str(filter.value))

        if filter.condition == ScanCondition.BELOW:
            return value < Decimal(str(filter.value))

        if filter.condition == ScanCondition.BETWEEN:
            return Decimal(str(filter.value)) <= value <= Decimal(str(filter.value2))

        if filter.condition == ScanCondition.EQUALS:
            return value == Decimal(str(filter.value))

        if filter.condition == ScanCondition.NOT_EQUALS:
            return value != Decimal(str(filter.value))

        if filter.condition == ScanCondition.CROSSES_ABOVE:
            if prev_data is None:
                return False
            prev_value = self._get_field_value(prev_data, filter.field)
            if prev_value is None:
                return False
            threshold = Decimal(str(filter.value))
            return prev_value <= threshold and value > threshold

        if filter.condition == ScanCondition.CROSSES_BELOW:
            if prev_data is None:
                return False
            prev_value = self._get_field_value(prev_data, filter.field)
            if prev_value is None:
                return False
            threshold = Decimal(str(filter.value))
            return prev_value >= threshold and value < threshold

        if filter.condition == ScanCondition.INCREASES_BY:
            if prev_data is None:
                return False
            prev_value = self._get_field_value(prev_data, filter.field)
            if prev_value is None or prev_value == 0:
                return False
            change = (value - prev_value) / prev_value * Decimal("100")
            return change >= Decimal(str(filter.value))

        if filter.condition == ScanCondition.DECREASES_BY:
            if prev_data is None:
                return False
            prev_value = self._get_field_value(prev_data, filter.field)
            if prev_value is None or prev_value == 0:
                return False
            change = (prev_value - value) / prev_value * Decimal("100")
            return change >= Decimal(str(filter.value))

        return False

    def _get_field_value(self, data: MarketData, field: str) -> Optional[Decimal]:
        """Get field value from market data."""
        return getattr(data, field, None)


class MomentumScanner:
    """Scanner for momentum signals."""

    def __init__(
        self,
        lookback_periods: int = 14,
        momentum_threshold: Decimal = Decimal("5")
    ):
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.lookback_periods:
            self.history[symbol].pop(0)

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for momentum signals."""
        self.update(data.symbol, data.price)

        history = self.history.get(data.symbol, [])
        if len(history) < 2:
            return None

        # Calculate momentum
        first_price = history[0]
        if first_price == 0:
            return None

        momentum = (data.price - first_price) / first_price * Decimal("100")

        if abs(momentum) >= self.momentum_threshold:
            signals = []
            trend = TrendDirection.SIDEWAYS
            strength = SignalStrength.MODERATE

            if momentum > 0:
                signals.append(f"Positive momentum: {momentum:.2f}%")
                trend = TrendDirection.UP
            else:
                signals.append(f"Negative momentum: {momentum:.2f}%")
                trend = TrendDirection.DOWN

            if abs(momentum) >= self.momentum_threshold * Decimal("2"):
                strength = SignalStrength.STRONG
            if abs(momentum) >= self.momentum_threshold * Decimal("3"):
                strength = SignalStrength.VERY_STRONG

            return ScanResult(
                symbol=data.symbol,
                score=abs(momentum),
                signals=signals,
                data=data,
                trend=trend,
                strength=strength,
                metadata={"momentum_pct": str(momentum)}
            )

        return None


class VolumeScanner:
    """Scanner for volume anomalies."""

    def __init__(
        self,
        volume_multiplier: Decimal = Decimal("2"),
        avg_periods: int = 20
    ):
        self.volume_multiplier = volume_multiplier
        self.avg_periods = avg_periods
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, volume: Decimal):
        """Update volume history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(volume)
        if len(self.history[symbol]) > self.avg_periods:
            self.history[symbol].pop(0)

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for volume spikes."""
        self.update(data.symbol, data.volume_24h)

        history = self.history.get(data.symbol, [])
        if len(history) < 2:
            return None

        avg_volume = sum(history[:-1]) / Decimal(str(len(history) - 1))
        if avg_volume == 0:
            return None

        volume_ratio = data.volume_24h / avg_volume

        if volume_ratio >= self.volume_multiplier:
            signals = [f"Volume spike: {volume_ratio:.1f}x average"]

            strength = SignalStrength.MODERATE
            if volume_ratio >= self.volume_multiplier * Decimal("2"):
                strength = SignalStrength.STRONG
            if volume_ratio >= self.volume_multiplier * Decimal("3"):
                strength = SignalStrength.VERY_STRONG

            return ScanResult(
                symbol=data.symbol,
                score=volume_ratio,
                signals=signals,
                data=data,
                strength=strength,
                priority=AlertPriority.HIGH if strength == SignalStrength.VERY_STRONG else AlertPriority.MEDIUM,
                metadata={"volume_ratio": str(volume_ratio), "avg_volume": str(avg_volume)}
            )

        return None


class VolatilityScanner:
    """Scanner for volatility signals."""

    def __init__(
        self,
        atr_periods: int = 14,
        volatility_threshold: Decimal = Decimal("5")
    ):
        self.atr_periods = atr_periods
        self.volatility_threshold = volatility_threshold
        self.history: dict[str, list[dict]] = {}

    def update(self, symbol: str, high: Decimal, low: Decimal, close: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append({"high": high, "low": low, "close": close})
        if len(self.history[symbol]) > self.atr_periods:
            self.history[symbol].pop(0)

    def calculate_atr(self, symbol: str) -> Optional[Decimal]:
        """Calculate Average True Range."""
        history = self.history.get(symbol, [])
        if len(history) < 2:
            return None

        true_ranges = []
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]

            tr1 = curr["high"] - curr["low"]
            tr2 = abs(curr["high"] - prev["close"])
            tr3 = abs(curr["low"] - prev["close"])

            true_ranges.append(max(tr1, tr2, tr3))

        if not true_ranges:
            return None

        return sum(true_ranges) / Decimal(str(len(true_ranges)))

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for volatility signals."""
        self.update(data.symbol, data.high_24h, data.low_24h, data.price)

        atr = self.calculate_atr(data.symbol)
        if atr is None or data.price == 0:
            return None

        volatility_pct = atr / data.price * Decimal("100")

        if volatility_pct >= self.volatility_threshold:
            signals = [f"High volatility: ATR {volatility_pct:.2f}%"]

            strength = SignalStrength.MODERATE
            if volatility_pct >= self.volatility_threshold * Decimal("2"):
                strength = SignalStrength.STRONG

            return ScanResult(
                symbol=data.symbol,
                score=volatility_pct,
                signals=signals,
                data=data,
                strength=strength,
                metadata={"atr": str(atr), "volatility_pct": str(volatility_pct)}
            )

        return None


class BreakoutScanner:
    """Scanner for breakout signals."""

    def __init__(
        self,
        lookback_periods: int = 20,
        breakout_threshold: Decimal = Decimal("1")
    ):
        self.lookback_periods = lookback_periods
        self.breakout_threshold = breakout_threshold
        self.history: dict[str, list[dict]] = {}

    def update(self, symbol: str, high: Decimal, low: Decimal, close: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append({"high": high, "low": low, "close": close})
        if len(self.history[symbol]) > self.lookback_periods:
            self.history[symbol].pop(0)

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for breakout signals."""
        self.update(data.symbol, data.high_24h, data.low_24h, data.price)

        history = self.history.get(data.symbol, [])
        if len(history) < self.lookback_periods:
            return None

        # Calculate channel
        highs = [h["high"] for h in history[:-1]]
        lows = [h["low"] for h in history[:-1]]

        resistance = max(highs)
        support = min(lows)

        signals = []
        trend = TrendDirection.SIDEWAYS
        strength = SignalStrength.WEAK

        # Check for breakout
        if resistance > 0:
            breakout_above = (data.price - resistance) / resistance * Decimal("100")
            if breakout_above >= self.breakout_threshold:
                signals.append(f"Breakout above resistance: {breakout_above:.2f}%")
                trend = TrendDirection.UP
                strength = SignalStrength.STRONG

        breakout_below = (support - data.price) / support * Decimal("100") if support > 0 else Decimal("0")
        if breakout_below >= self.breakout_threshold:
            signals.append(f"Breakout below support: {breakout_below:.2f}%")
            trend = TrendDirection.DOWN
            strength = SignalStrength.STRONG

        if signals:
            return ScanResult(
                symbol=data.symbol,
                score=max(breakout_above if breakout_above > 0 else Decimal("0"),
                         breakout_below if breakout_below > 0 else Decimal("0")),
                signals=signals,
                data=data,
                trend=trend,
                strength=strength,
                priority=AlertPriority.HIGH,
                metadata={
                    "resistance": str(resistance),
                    "support": str(support)
                }
            )

        return None


class FundingScanner:
    """Scanner for funding rate opportunities."""

    def __init__(
        self,
        funding_threshold: Decimal = Decimal("0.01"),  # 0.01% = 1 bps
        extreme_threshold: Decimal = Decimal("0.05")   # 0.05% = 5 bps
    ):
        self.funding_threshold = funding_threshold
        self.extreme_threshold = extreme_threshold

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for funding rate signals."""
        funding = abs(data.funding_rate)

        if funding < self.funding_threshold:
            return None

        signals = []
        strength = SignalStrength.MODERATE
        priority = AlertPriority.MEDIUM

        if data.funding_rate > 0:
            signals.append(f"High positive funding: {data.funding_rate * 100:.4f}%")
            signals.append("Opportunity: Short to receive funding")
        else:
            signals.append(f"High negative funding: {data.funding_rate * 100:.4f}%")
            signals.append("Opportunity: Long to receive funding")

        if funding >= self.extreme_threshold:
            strength = SignalStrength.VERY_STRONG
            priority = AlertPriority.HIGH
            signals.append("⚠️ Extreme funding rate")

        return ScanResult(
            symbol=data.symbol,
            score=funding * Decimal("10000"),  # Convert to bps
            signals=signals,
            data=data,
            strength=strength,
            priority=priority,
            metadata={"funding_bps": str(data.funding_rate * Decimal("10000"))}
        )


class LiquidityScanner:
    """Scanner for liquidity conditions."""

    def __init__(
        self,
        min_volume: Decimal = Decimal("1000000"),
        max_spread_bps: Decimal = Decimal("10")
    ):
        self.min_volume = min_volume
        self.max_spread_bps = max_spread_bps

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for liquidity signals."""
        signals = []

        # Check spread
        if data.price > 0 and data.spread > 0:
            spread_bps = data.spread / data.price * Decimal("10000")

            if spread_bps > self.max_spread_bps:
                signals.append(f"Wide spread: {spread_bps:.1f} bps")

            if spread_bps < Decimal("2"):
                signals.append(f"Tight spread: {spread_bps:.1f} bps")

        # Check volume
        if data.volume_24h < self.min_volume:
            signals.append(f"Low volume: ${data.volume_24h:,.0f}")

        if data.volume_24h >= self.min_volume * Decimal("10"):
            signals.append(f"High liquidity: ${data.volume_24h:,.0f}")

        if not signals:
            return None

        has_warning = any("Wide" in s or "Low" in s for s in signals)

        return ScanResult(
            symbol=data.symbol,
            score=data.volume_24h / self.min_volume,
            signals=signals,
            data=data,
            priority=AlertPriority.LOW if not has_warning else AlertPriority.MEDIUM,
            metadata={"spread_bps": str(data.spread / data.price * Decimal("10000")) if data.price > 0 else "0"}
        )


class TrendScanner:
    """Scanner for trend signals."""

    def __init__(
        self,
        short_period: int = 10,
        long_period: int = 30
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.long_period:
            self.history[symbol].pop(0)

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for trend signals."""
        self.update(data.symbol, data.price)

        history = self.history.get(data.symbol, [])
        if len(history) < self.long_period:
            return None

        short_ma = sum(history[-self.short_period:]) / Decimal(str(self.short_period))
        long_ma = sum(history) / Decimal(str(self.long_period))

        if long_ma == 0:
            return None

        ma_diff_pct = (short_ma - long_ma) / long_ma * Decimal("100")

        signals = []
        trend = TrendDirection.SIDEWAYS
        strength = SignalStrength.WEAK

        if ma_diff_pct > Decimal("1"):
            signals.append(f"Bullish trend: SMA({self.short_period}) > SMA({self.long_period})")
            trend = TrendDirection.UP
            strength = SignalStrength.MODERATE
            if ma_diff_pct > Decimal("3"):
                strength = SignalStrength.STRONG
        elif ma_diff_pct < Decimal("-1"):
            signals.append(f"Bearish trend: SMA({self.short_period}) < SMA({self.long_period})")
            trend = TrendDirection.DOWN
            strength = SignalStrength.MODERATE
            if ma_diff_pct < Decimal("-3"):
                strength = SignalStrength.STRONG

        if signals:
            return ScanResult(
                symbol=data.symbol,
                score=abs(ma_diff_pct),
                signals=signals,
                data=data,
                trend=trend,
                strength=strength,
                metadata={
                    "short_ma": str(short_ma),
                    "long_ma": str(long_ma),
                    "ma_diff_pct": str(ma_diff_pct)
                }
            )

        return None


class ReversalScanner:
    """Scanner for potential reversal signals."""

    def __init__(
        self,
        rsi_periods: int = 14,
        oversold_level: Decimal = Decimal("30"),
        overbought_level: Decimal = Decimal("70")
    ):
        self.rsi_periods = rsi_periods
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.rsi_periods + 1:
            self.history[symbol].pop(0)

    def calculate_rsi(self, symbol: str) -> Optional[Decimal]:
        """Calculate RSI."""
        history = self.history.get(symbol, [])
        if len(history) < self.rsi_periods + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(history)):
            change = history[i] - history[i-1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        avg_gain = sum(gains) / Decimal(str(len(gains)))
        avg_loss = sum(losses) / Decimal(str(len(losses)))

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Scan for reversal signals."""
        self.update(data.symbol, data.price)

        rsi = self.calculate_rsi(data.symbol)
        if rsi is None:
            return None

        signals = []
        trend = TrendDirection.SIDEWAYS
        strength = SignalStrength.WEAK

        if rsi <= self.oversold_level:
            signals.append(f"Oversold: RSI = {rsi:.1f}")
            signals.append("Potential bullish reversal")
            trend = TrendDirection.DOWN  # Currently down, may reverse
            strength = SignalStrength.MODERATE
            if rsi <= Decimal("20"):
                strength = SignalStrength.STRONG

        if rsi >= self.overbought_level:
            signals.append(f"Overbought: RSI = {rsi:.1f}")
            signals.append("Potential bearish reversal")
            trend = TrendDirection.UP  # Currently up, may reverse
            strength = SignalStrength.MODERATE
            if rsi >= Decimal("80"):
                strength = SignalStrength.STRONG

        if signals:
            return ScanResult(
                symbol=data.symbol,
                score=abs(rsi - Decimal("50")),
                signals=signals,
                data=data,
                trend=trend,
                strength=strength,
                metadata={"rsi": str(rsi)}
            )

        return None


class MarketScanner:
    """Main market scanner orchestrator."""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig(scanner_type=ScannerType.MOMENTUM)
        self.market_data: dict[str, MarketData] = {}
        self.prev_data: dict[str, MarketData] = {}
        self.results: list[ScanResult] = []
        self.alerts: list[Alert] = []
        self.alert_counter = 0
        self.filter_evaluator = FilterEvaluator()
        self.callbacks: dict[str, list[Callable]] = {
            "on_result": [],
            "on_alert": []
        }

        # Initialize scanners
        self.scanners = {
            ScannerType.MOMENTUM: MomentumScanner(),
            ScannerType.VOLUME: VolumeScanner(),
            ScannerType.VOLATILITY: VolatilityScanner(),
            ScannerType.BREAKOUT: BreakoutScanner(),
            ScannerType.FUNDING: FundingScanner(),
            ScannerType.LIQUIDITY: LiquidityScanner(),
            ScannerType.TREND: TrendScanner(),
            ScannerType.REVERSAL: ReversalScanner()
        }

    def update_data(self, data: MarketData):
        """Update market data."""
        symbol = data.symbol
        if symbol in self.market_data:
            self.prev_data[symbol] = self.market_data[symbol]
        self.market_data[symbol] = data

    def register_callback(self, event: str, callback: Callable):
        """Register callback for events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def scan(self, symbols: Optional[list[str]] = None) -> list[ScanResult]:
        """Run scanner on specified or all symbols."""
        symbols = symbols or list(self.market_data.keys())
        self.results = []

        for symbol in symbols:
            data = self.market_data.get(symbol)
            if not data:
                continue

            # Apply filters
            if not self._passes_filters(data):
                continue

            # Run appropriate scanner
            scanner = self.scanners.get(self.config.scanner_type)
            if scanner:
                result = scanner.scan(data)
                if result:
                    self.results.append(result)
                    self._check_alert(result)

                    for cb in self.callbacks["on_result"]:
                        cb(result)

        # Sort by score and limit
        self.results.sort(key=lambda r: r.score, reverse=True)
        self.results = self.results[:self.config.top_n]

        return self.results

    def scan_all(self, symbols: Optional[list[str]] = None) -> dict[ScannerType, list[ScanResult]]:
        """Run all scanners."""
        symbols = symbols or list(self.market_data.keys())
        all_results: dict[ScannerType, list[ScanResult]] = {}

        for scanner_type, scanner in self.scanners.items():
            results = []

            for symbol in symbols:
                data = self.market_data.get(symbol)
                if not data:
                    continue

                if not self._passes_basic_filters(data):
                    continue

                result = scanner.scan(data)
                if result:
                    results.append(result)

            results.sort(key=lambda r: r.score, reverse=True)
            all_results[scanner_type] = results[:self.config.top_n]

        return all_results

    def _passes_filters(self, data: MarketData) -> bool:
        """Check if data passes all filters."""
        # Check include/exclude lists
        if self.config.include_symbols and data.symbol not in self.config.include_symbols:
            return False
        if data.symbol in self.config.exclude_symbols:
            return False

        # Check basic filters
        if not self._passes_basic_filters(data):
            return False

        # Check custom filters
        for filter in self.config.filters:
            prev_data = self.prev_data.get(data.symbol)
            if not self.filter_evaluator.evaluate(data, filter, prev_data):
                return False

        return True

    def _passes_basic_filters(self, data: MarketData) -> bool:
        """Check basic volume and spread filters."""
        if data.volume_24h < self.config.min_volume:
            return False

        if data.price > 0:
            spread_pct = data.spread / data.price * Decimal("100")
            if spread_pct > self.config.max_spread_pct:
                return False

        return True

    def _check_alert(self, result: ScanResult):
        """Check if result should trigger alert."""
        if result.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]:
            self.alert_counter += 1
            alert = Alert(
                id=f"ALERT-{self.alert_counter:06d}",
                timestamp=datetime.now(),
                symbol=result.symbol,
                alert_type=self.config.scanner_type.value,
                message="; ".join(result.signals),
                priority=result.priority,
                data=result.to_dict()
            )
            self.alerts.append(alert)

            for cb in self.callbacks["on_alert"]:
                cb(alert)

    def get_top_gainers(self, n: int = 10) -> list[MarketData]:
        """Get top gaining symbols."""
        sorted_data = sorted(
            self.market_data.values(),
            key=lambda d: d.change_24h,
            reverse=True
        )
        return sorted_data[:n]

    def get_top_losers(self, n: int = 10) -> list[MarketData]:
        """Get top losing symbols."""
        sorted_data = sorted(
            self.market_data.values(),
            key=lambda d: d.change_24h
        )
        return sorted_data[:n]

    def get_top_volume(self, n: int = 10) -> list[MarketData]:
        """Get top volume symbols."""
        sorted_data = sorted(
            self.market_data.values(),
            key=lambda d: d.volume_24h,
            reverse=True
        )
        return sorted_data[:n]

    def get_highest_oi(self, n: int = 10) -> list[MarketData]:
        """Get highest open interest symbols."""
        sorted_data = sorted(
            self.market_data.values(),
            key=lambda d: d.open_interest,
            reverse=True
        )
        return sorted_data[:n]

    def get_extreme_funding(self, n: int = 10) -> list[MarketData]:
        """Get symbols with extreme funding rates."""
        sorted_data = sorted(
            self.market_data.values(),
            key=lambda d: abs(d.funding_rate),
            reverse=True
        )
        return sorted_data[:n]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_unacknowledged_alerts(self) -> list[Alert]:
        """Get unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []

    def get_summary(self) -> dict:
        """Get scanner summary."""
        return {
            "total_symbols": len(self.market_data),
            "active_alerts": len(self.get_unacknowledged_alerts()),
            "last_scan_results": len(self.results),
            "scanner_type": self.config.scanner_type.value,
            "config": self.config.to_dict()
        }


class WatchlistScanner:
    """Scanner for user watchlist."""

    def __init__(self):
        self.watchlist: dict[str, dict] = {}
        self.alerts: list[Alert] = []
        self.alert_counter = 0

    def add_to_watchlist(
        self,
        symbol: str,
        price_alerts: Optional[list[Decimal]] = None,
        change_alerts: Optional[list[Decimal]] = None,
        volume_alerts: Optional[list[Decimal]] = None
    ):
        """Add symbol to watchlist with alerts."""
        self.watchlist[symbol] = {
            "price_alerts": price_alerts or [],
            "change_alerts": change_alerts or [],
            "volume_alerts": volume_alerts or [],
            "triggered": set()
        }

    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove symbol from watchlist."""
        if symbol in self.watchlist:
            del self.watchlist[symbol]
            return True
        return False

    def check(self, data: MarketData) -> list[Alert]:
        """Check watchlist alerts."""
        if data.symbol not in self.watchlist:
            return []

        alerts = []
        config = self.watchlist[data.symbol]

        # Price alerts
        for target_price in config["price_alerts"]:
            key = f"price_{target_price}"
            if key not in config["triggered"]:
                if data.price >= target_price:
                    self.alert_counter += 1
                    alert = Alert(
                        id=f"WL-{self.alert_counter:06d}",
                        timestamp=datetime.now(),
                        symbol=data.symbol,
                        alert_type="price_target",
                        message=f"Price reached ${target_price}",
                        priority=AlertPriority.HIGH,
                        data=data.to_dict()
                    )
                    alerts.append(alert)
                    config["triggered"].add(key)

        # Change alerts
        for target_change in config["change_alerts"]:
            key = f"change_{target_change}"
            if key not in config["triggered"]:
                if (target_change > 0 and data.change_24h >= target_change) or \
                   (target_change < 0 and data.change_24h <= target_change):
                    self.alert_counter += 1
                    alert = Alert(
                        id=f"WL-{self.alert_counter:06d}",
                        timestamp=datetime.now(),
                        symbol=data.symbol,
                        alert_type="change_alert",
                        message=f"24h change: {data.change_24h}%",
                        priority=AlertPriority.HIGH,
                        data=data.to_dict()
                    )
                    alerts.append(alert)
                    config["triggered"].add(key)

        self.alerts.extend(alerts)
        return alerts

    def reset_alerts(self, symbol: str):
        """Reset triggered alerts for symbol."""
        if symbol in self.watchlist:
            self.watchlist[symbol]["triggered"] = set()


class CompositeScanner:
    """Combine multiple scanners for comprehensive analysis."""

    def __init__(self):
        self.scanners: list[tuple[ScannerType, Any, Decimal]] = []  # (type, scanner, weight)

    def add_scanner(self, scanner_type: ScannerType, scanner: Any, weight: Decimal = Decimal("1")):
        """Add scanner with weight."""
        self.scanners.append((scanner_type, scanner, weight))

    def scan(self, data: MarketData) -> Optional[ScanResult]:
        """Run all scanners and combine results."""
        results = []
        total_weight = Decimal("0")

        for scanner_type, scanner, weight in self.scanners:
            result = scanner.scan(data)
            if result:
                results.append((result, weight))
                total_weight += weight

        if not results:
            return None

        # Combine signals and calculate weighted score
        all_signals = []
        weighted_score = Decimal("0")
        highest_priority = AlertPriority.LOW
        strongest = SignalStrength.WEAK

        for result, weight in results:
            all_signals.extend(result.signals)
            weighted_score += result.score * weight

            if result.priority.value > highest_priority.value:
                highest_priority = result.priority

            if list(SignalStrength).index(result.strength) > list(SignalStrength).index(strongest):
                strongest = result.strength

        if total_weight > 0:
            weighted_score /= total_weight

        return ScanResult(
            symbol=data.symbol,
            score=weighted_score,
            signals=all_signals,
            data=data,
            priority=highest_priority,
            strength=strongest,
            metadata={"num_scanners": len(results)}
        )


# Global instance
_scanner: Optional[MarketScanner] = None


def get_scanner() -> MarketScanner:
    """Get global scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = MarketScanner()
    return _scanner


def set_scanner(scanner: MarketScanner):
    """Set global scanner instance."""
    global _scanner
    _scanner = scanner
