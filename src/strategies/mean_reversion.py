"""
Mean Reversion Trading Strategy Module.

Implements mean reversion strategies that capitalize on
price deviations from statistical averages:
- Bollinger Bands mean reversion
- RSI mean reversion
- Z-score based trading
- Statistical arbitrage
- Pairs trading foundation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
import math

logger = logging.getLogger(__name__)


class MeanReversionType(Enum):
    """Type of mean reversion strategy."""
    BOLLINGER = "bollinger"          # Bollinger Bands based
    RSI = "rsi"                      # RSI extremes
    ZSCORE = "zscore"                # Z-score deviation
    KELTNER = "keltner"              # Keltner Channels
    PERCENTAGE = "percentage"         # Simple percentage deviation


class DeviationState(Enum):
    """Price deviation state from mean."""
    EXTREMELY_OVERBOUGHT = "extremely_overbought"
    OVERBOUGHT = "overbought"
    ABOVE_MEAN = "above_mean"
    AT_MEAN = "at_mean"
    BELOW_MEAN = "below_mean"
    OVERSOLD = "oversold"
    EXTREMELY_OVERSOLD = "extremely_oversold"


class EntrySignal(Enum):
    """Entry signal type."""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    NO_SIGNAL = "no_signal"


class ExitSignal(Enum):
    """Exit signal type."""
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal."""
    timestamp: datetime
    symbol: str
    entry_signal: EntrySignal
    exit_signal: ExitSignal
    deviation_state: DeviationState
    current_price: Decimal
    mean_price: Decimal
    deviation_pct: Decimal
    z_score: Decimal | None = None
    upper_band: Decimal | None = None
    lower_band: Decimal | None = None
    target_price: Decimal | None = None
    stop_loss: Decimal | None = None
    confidence: Decimal = Decimal("50")
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "entry_signal": self.entry_signal.value,
            "exit_signal": self.exit_signal.value,
            "deviation_state": self.deviation_state.value,
            "current_price": str(self.current_price),
            "mean_price": str(self.mean_price),
            "deviation_pct": str(self.deviation_pct),
            "z_score": str(self.z_score) if self.z_score else None,
            "upper_band": str(self.upper_band) if self.upper_band else None,
            "lower_band": str(self.lower_band) if self.lower_band else None,
            "target_price": str(self.target_price) if self.target_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "confidence": str(self.confidence),
            "reason": self.reason
        }


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""
    symbol: str
    strategy_type: MeanReversionType
    # Moving average settings
    ma_period: int = 20
    ma_type: str = "sma"  # sma or ema
    # Bollinger Bands settings
    bb_std_dev: Decimal = Decimal("2")
    bb_extreme_std_dev: Decimal = Decimal("3")
    # RSI settings
    rsi_period: int = 14
    rsi_overbought: Decimal = Decimal("70")
    rsi_oversold: Decimal = Decimal("30")
    rsi_extreme_overbought: Decimal = Decimal("80")
    rsi_extreme_oversold: Decimal = Decimal("20")
    # Z-score settings
    zscore_lookback: int = 20
    zscore_entry_threshold: Decimal = Decimal("2")
    zscore_exit_threshold: Decimal = Decimal("0.5")
    zscore_extreme_threshold: Decimal = Decimal("3")
    # Keltner settings
    keltner_atr_period: int = 14
    keltner_atr_multiplier: Decimal = Decimal("2")
    # Percentage deviation settings
    entry_deviation_pct: Decimal = Decimal("3")
    exit_deviation_pct: Decimal = Decimal("0.5")
    extreme_deviation_pct: Decimal = Decimal("5")
    # Risk management
    stop_loss_pct: Decimal = Decimal("3")
    take_profit_pct: Decimal = Decimal("2")
    max_position_size: Decimal | None = None
    # Filters
    min_volume_ratio: Decimal = Decimal("0.8")
    require_volume_confirmation: bool = True

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if self.ma_period < 2:
            errors.append("MA period must be at least 2")

        if self.bb_std_dev <= Decimal("0"):
            errors.append("Bollinger std dev must be positive")

        if self.rsi_overbought <= self.rsi_oversold:
            errors.append("RSI overbought must be greater than oversold")

        if self.zscore_entry_threshold <= Decimal("0"):
            errors.append("Z-score entry threshold must be positive")

        if self.entry_deviation_pct <= Decimal("0"):
            errors.append("Entry deviation must be positive")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "strategy_type": self.strategy_type.value,
            "ma_period": self.ma_period,
            "ma_type": self.ma_type,
            "bb_std_dev": str(self.bb_std_dev),
            "rsi_period": self.rsi_period,
            "rsi_overbought": str(self.rsi_overbought),
            "rsi_oversold": str(self.rsi_oversold),
            "zscore_entry_threshold": str(self.zscore_entry_threshold),
            "entry_deviation_pct": str(self.entry_deviation_pct),
            "stop_loss_pct": str(self.stop_loss_pct),
            "take_profit_pct": str(self.take_profit_pct)
        }


@dataclass
class MeanReversionMetrics:
    """Performance metrics for mean reversion strategy."""
    total_signals: int = 0
    long_entries: int = 0
    short_entries: int = 0
    successful_reversions: int = 0
    failed_reversions: int = 0
    total_pnl: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    avg_reversion_time_hours: Decimal = Decimal("0")
    avg_deviation_at_entry: Decimal = Decimal("0")
    max_deviation_seen: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_signals": self.total_signals,
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "successful_reversions": self.successful_reversions,
            "failed_reversions": self.failed_reversions,
            "total_pnl": str(self.total_pnl),
            "win_rate": str(self.win_rate),
            "avg_reversion_time_hours": str(self.avg_reversion_time_hours),
            "avg_deviation_at_entry": str(self.avg_deviation_at_entry),
            "max_deviation_seen": str(self.max_deviation_seen),
            "last_updated": self.last_updated.isoformat()
        }


class StatisticalCalculator:
    """Statistical calculations for mean reversion."""

    @staticmethod
    def calculate_mean(values: list[Decimal]) -> Decimal:
        """Calculate arithmetic mean."""
        if not values:
            return Decimal("0")
        return sum(values) / len(values)

    @staticmethod
    def calculate_std_dev(values: list[Decimal], mean: Decimal | None = None) -> Decimal:
        """Calculate standard deviation."""
        if len(values) < 2:
            return Decimal("0")

        if mean is None:
            mean = StatisticalCalculator.calculate_mean(values)

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance.sqrt() if variance > 0 else Decimal("0")

    @staticmethod
    def calculate_z_score(
        value: Decimal,
        mean: Decimal,
        std_dev: Decimal
    ) -> Decimal:
        """Calculate z-score."""
        if std_dev == 0:
            return Decimal("0")
        return (value - mean) / std_dev

    @staticmethod
    def calculate_ema(values: list[Decimal], period: int) -> Decimal | None:
        """Calculate exponential moving average."""
        if len(values) < period:
            return None

        multiplier = Decimal("2") / (period + 1)
        ema = values[0]

        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))

        return ema

    @staticmethod
    def calculate_sma(values: list[Decimal], period: int) -> Decimal | None:
        """Calculate simple moving average."""
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
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
        return Decimal("100") - (Decimal("100") / (1 + rs))

    @staticmethod
    def calculate_atr(
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
        period: int = 14
    ) -> Decimal | None:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        if len(true_ranges) < period:
            return None

        return sum(true_ranges[-period:]) / period

    @staticmethod
    def calculate_bollinger_bands(
        prices: list[Decimal],
        period: int = 20,
        std_multiplier: Decimal = Decimal("2")
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            return None, None, None

        middle = StatisticalCalculator.calculate_sma(prices, period)
        if middle is None:
            return None, None, None

        std = StatisticalCalculator.calculate_std_dev(prices[-period:], middle)

        upper = middle + (std_multiplier * std)
        lower = middle - (std_multiplier * std)

        return upper, middle, lower

    @staticmethod
    def calculate_keltner_channels(
        closes: list[Decimal],
        highs: list[Decimal],
        lows: list[Decimal],
        ema_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal("2")
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """
        Calculate Keltner Channels.

        Returns:
            Tuple of (upper_channel, middle, lower_channel)
        """
        middle = StatisticalCalculator.calculate_ema(closes, ema_period)
        atr = StatisticalCalculator.calculate_atr(highs, lows, closes, atr_period)

        if middle is None or atr is None:
            return None, None, None

        upper = middle + (atr_multiplier * atr)
        lower = middle - (atr_multiplier * atr)

        return upper, middle, lower


class MeanReversionAnalyzer:
    """Analyzes price for mean reversion opportunities."""

    def __init__(self, config: MeanReversionConfig):
        """Initialize analyzer."""
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {', '.join(errors)}")

        self.config = config
        self._price_history: list[Decimal] = []
        self._high_history: list[Decimal] = []
        self._low_history: list[Decimal] = []
        self._volume_history: list[Decimal] = []
        self._max_history = 500
        self._signals: list[MeanReversionSignal] = []

    def add_price(
        self,
        price: Decimal,
        high: Decimal | None = None,
        low: Decimal | None = None,
        volume: Decimal | None = None
    ) -> None:
        """Add a price point to history."""
        self._price_history.append(price)
        self._high_history.append(high or price)
        self._low_history.append(low or price)
        self._volume_history.append(volume or Decimal("1"))

        if len(self._price_history) > self._max_history:
            self._price_history = self._price_history[-self._max_history:]
            self._high_history = self._high_history[-self._max_history:]
            self._low_history = self._low_history[-self._max_history:]
            self._volume_history = self._volume_history[-self._max_history:]

    def analyze(self) -> MeanReversionSignal | None:
        """
        Analyze current price for mean reversion signal.

        Returns:
            MeanReversionSignal or None
        """
        if len(self._price_history) < self.config.ma_period:
            return None

        current_price = self._price_history[-1]

        if self.config.strategy_type == MeanReversionType.BOLLINGER:
            return self._analyze_bollinger(current_price)
        elif self.config.strategy_type == MeanReversionType.RSI:
            return self._analyze_rsi(current_price)
        elif self.config.strategy_type == MeanReversionType.ZSCORE:
            return self._analyze_zscore(current_price)
        elif self.config.strategy_type == MeanReversionType.KELTNER:
            return self._analyze_keltner(current_price)
        elif self.config.strategy_type == MeanReversionType.PERCENTAGE:
            return self._analyze_percentage(current_price)

        return None

    def _analyze_bollinger(self, current_price: Decimal) -> MeanReversionSignal | None:
        """Analyze using Bollinger Bands."""
        upper, middle, lower = StatisticalCalculator.calculate_bollinger_bands(
            self._price_history,
            self.config.ma_period,
            self.config.bb_std_dev
        )

        if upper is None or middle is None or lower is None:
            return None

        # Calculate extreme bands
        std = StatisticalCalculator.calculate_std_dev(
            self._price_history[-self.config.ma_period:],
            middle
        )
        extreme_upper = middle + (self.config.bb_extreme_std_dev * std)
        extreme_lower = middle - (self.config.bb_extreme_std_dev * std)

        # Determine deviation state
        deviation_pct = ((current_price - middle) / middle * 100) if middle > 0 else Decimal("0")

        if current_price > extreme_upper:
            state = DeviationState.EXTREMELY_OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("80")
        elif current_price > upper:
            state = DeviationState.OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("65")
        elif current_price < extreme_lower:
            state = DeviationState.EXTREMELY_OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("80")
        elif current_price < lower:
            state = DeviationState.OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("65")
        elif current_price > middle:
            state = DeviationState.ABOVE_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        elif current_price < middle:
            state = DeviationState.BELOW_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        else:
            state = DeviationState.AT_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")

        # Determine exit signal
        if abs(current_price - middle) < std * Decimal("0.5"):
            exit_signal = ExitSignal.EXIT_LONG if current_price >= middle else ExitSignal.EXIT_SHORT
        else:
            exit_signal = ExitSignal.HOLD

        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            entry_signal=entry,
            exit_signal=exit_signal,
            deviation_state=state,
            current_price=current_price,
            mean_price=middle,
            deviation_pct=deviation_pct,
            upper_band=upper,
            lower_band=lower,
            target_price=middle,
            stop_loss=self._calculate_stop_loss(current_price, entry),
            confidence=confidence,
            reason=f"BB: price={current_price:.2f}, mean={middle:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_rsi(self, current_price: Decimal) -> MeanReversionSignal | None:
        """Analyze using RSI."""
        rsi = StatisticalCalculator.calculate_rsi(
            self._price_history,
            self.config.rsi_period
        )

        if rsi is None:
            return None

        # Calculate mean for reference
        mean = StatisticalCalculator.calculate_sma(
            self._price_history,
            self.config.ma_period
        )
        if mean is None:
            mean = current_price

        deviation_pct = ((current_price - mean) / mean * 100) if mean > 0 else Decimal("0")

        # Determine state based on RSI
        if rsi >= self.config.rsi_extreme_overbought:
            state = DeviationState.EXTREMELY_OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("85")
        elif rsi >= self.config.rsi_overbought:
            state = DeviationState.OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("70")
        elif rsi <= self.config.rsi_extreme_oversold:
            state = DeviationState.EXTREMELY_OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("85")
        elif rsi <= self.config.rsi_oversold:
            state = DeviationState.OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("70")
        elif rsi > Decimal("50"):
            state = DeviationState.ABOVE_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        elif rsi < Decimal("50"):
            state = DeviationState.BELOW_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        else:
            state = DeviationState.AT_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")

        # Exit when RSI returns to middle zone
        if Decimal("40") <= rsi <= Decimal("60"):
            exit_signal = ExitSignal.EXIT_LONG if rsi >= Decimal("50") else ExitSignal.EXIT_SHORT
        else:
            exit_signal = ExitSignal.HOLD

        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            entry_signal=entry,
            exit_signal=exit_signal,
            deviation_state=state,
            current_price=current_price,
            mean_price=mean,
            deviation_pct=deviation_pct,
            z_score=Decimal(str((rsi - 50) / 20)),  # Approximate z-score from RSI
            target_price=mean,
            stop_loss=self._calculate_stop_loss(current_price, entry),
            confidence=confidence,
            reason=f"RSI: {rsi:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_zscore(self, current_price: Decimal) -> MeanReversionSignal | None:
        """Analyze using Z-score."""
        if len(self._price_history) < self.config.zscore_lookback:
            return None

        prices = self._price_history[-self.config.zscore_lookback:]
        mean = StatisticalCalculator.calculate_mean(prices)
        std = StatisticalCalculator.calculate_std_dev(prices, mean)

        if std == 0:
            return None

        z_score = StatisticalCalculator.calculate_z_score(current_price, mean, std)
        deviation_pct = ((current_price - mean) / mean * 100) if mean > 0 else Decimal("0")

        # Determine state based on z-score
        if z_score >= self.config.zscore_extreme_threshold:
            state = DeviationState.EXTREMELY_OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("90")
        elif z_score >= self.config.zscore_entry_threshold:
            state = DeviationState.OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("75")
        elif z_score <= -self.config.zscore_extreme_threshold:
            state = DeviationState.EXTREMELY_OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("90")
        elif z_score <= -self.config.zscore_entry_threshold:
            state = DeviationState.OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("75")
        elif z_score > Decimal("0"):
            state = DeviationState.ABOVE_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        elif z_score < Decimal("0"):
            state = DeviationState.BELOW_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        else:
            state = DeviationState.AT_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")

        # Exit when z-score returns near zero
        if abs(z_score) <= self.config.zscore_exit_threshold:
            exit_signal = ExitSignal.EXIT_LONG if z_score >= 0 else ExitSignal.EXIT_SHORT
        else:
            exit_signal = ExitSignal.HOLD

        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            entry_signal=entry,
            exit_signal=exit_signal,
            deviation_state=state,
            current_price=current_price,
            mean_price=mean,
            deviation_pct=deviation_pct,
            z_score=z_score,
            upper_band=mean + (self.config.zscore_entry_threshold * std),
            lower_band=mean - (self.config.zscore_entry_threshold * std),
            target_price=mean,
            stop_loss=self._calculate_stop_loss(current_price, entry),
            confidence=confidence,
            reason=f"Z-score: {z_score:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_keltner(self, current_price: Decimal) -> MeanReversionSignal | None:
        """Analyze using Keltner Channels."""
        upper, middle, lower = StatisticalCalculator.calculate_keltner_channels(
            self._price_history,
            self._high_history,
            self._low_history,
            self.config.ma_period,
            self.config.keltner_atr_period,
            self.config.keltner_atr_multiplier
        )

        if upper is None or middle is None or lower is None:
            return None

        deviation_pct = ((current_price - middle) / middle * 100) if middle > 0 else Decimal("0")

        # Determine state
        if current_price > upper:
            state = DeviationState.OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("70")
        elif current_price < lower:
            state = DeviationState.OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("70")
        elif current_price > middle:
            state = DeviationState.ABOVE_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        elif current_price < middle:
            state = DeviationState.BELOW_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        else:
            state = DeviationState.AT_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")

        # Exit signal
        channel_width = upper - lower
        if abs(current_price - middle) < channel_width * Decimal("0.2"):
            exit_signal = ExitSignal.EXIT_LONG if current_price >= middle else ExitSignal.EXIT_SHORT
        else:
            exit_signal = ExitSignal.HOLD

        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            entry_signal=entry,
            exit_signal=exit_signal,
            deviation_state=state,
            current_price=current_price,
            mean_price=middle,
            deviation_pct=deviation_pct,
            upper_band=upper,
            lower_band=lower,
            target_price=middle,
            stop_loss=self._calculate_stop_loss(current_price, entry),
            confidence=confidence,
            reason=f"Keltner: price={current_price:.2f}, middle={middle:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_percentage(self, current_price: Decimal) -> MeanReversionSignal | None:
        """Analyze using simple percentage deviation."""
        if self.config.ma_type == "ema":
            mean = StatisticalCalculator.calculate_ema(
                self._price_history,
                self.config.ma_period
            )
        else:
            mean = StatisticalCalculator.calculate_sma(
                self._price_history,
                self.config.ma_period
            )

        if mean is None or mean == 0:
            return None

        deviation_pct = (current_price - mean) / mean * 100

        # Determine state
        if deviation_pct >= self.config.extreme_deviation_pct:
            state = DeviationState.EXTREMELY_OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("80")
        elif deviation_pct >= self.config.entry_deviation_pct:
            state = DeviationState.OVERBOUGHT
            entry = EntrySignal.SHORT_ENTRY
            confidence = Decimal("65")
        elif deviation_pct <= -self.config.extreme_deviation_pct:
            state = DeviationState.EXTREMELY_OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("80")
        elif deviation_pct <= -self.config.entry_deviation_pct:
            state = DeviationState.OVERSOLD
            entry = EntrySignal.LONG_ENTRY
            confidence = Decimal("65")
        elif deviation_pct > Decimal("0"):
            state = DeviationState.ABOVE_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        elif deviation_pct < Decimal("0"):
            state = DeviationState.BELOW_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")
        else:
            state = DeviationState.AT_MEAN
            entry = EntrySignal.NO_SIGNAL
            confidence = Decimal("50")

        # Exit when deviation shrinks
        if abs(deviation_pct) <= self.config.exit_deviation_pct:
            exit_signal = ExitSignal.EXIT_LONG if deviation_pct >= 0 else ExitSignal.EXIT_SHORT
        else:
            exit_signal = ExitSignal.HOLD

        upper = mean * (1 + self.config.entry_deviation_pct / 100)
        lower = mean * (1 - self.config.entry_deviation_pct / 100)

        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            entry_signal=entry,
            exit_signal=exit_signal,
            deviation_state=state,
            current_price=current_price,
            mean_price=mean,
            deviation_pct=deviation_pct,
            upper_band=upper,
            lower_band=lower,
            target_price=mean,
            stop_loss=self._calculate_stop_loss(current_price, entry),
            confidence=confidence,
            reason=f"Deviation: {deviation_pct:.2f}%"
        )

        self._signals.append(signal)
        return signal

    def _calculate_stop_loss(
        self,
        current_price: Decimal,
        entry: EntrySignal
    ) -> Decimal | None:
        """Calculate stop loss price."""
        if entry == EntrySignal.LONG_ENTRY:
            return current_price * (1 - self.config.stop_loss_pct / 100)
        elif entry == EntrySignal.SHORT_ENTRY:
            return current_price * (1 + self.config.stop_loss_pct / 100)
        return None

    def get_recent_signals(self, count: int = 10) -> list[MeanReversionSignal]:
        """Get recent signals."""
        return self._signals[-count:]

    def get_current_deviation(self) -> Decimal | None:
        """Get current deviation from mean."""
        if not self._signals:
            return None
        return self._signals[-1].deviation_pct


class MeanReversionStrategy:
    """Complete mean reversion trading strategy."""

    def __init__(
        self,
        config: MeanReversionConfig,
        strategy_id: str | None = None
    ):
        """Initialize mean reversion strategy."""
        self.strategy_id = strategy_id or f"meanrev_{config.symbol}"
        self.config = config
        self.analyzer = MeanReversionAnalyzer(config)
        self.metrics = MeanReversionMetrics()

        self._in_position = False
        self._position_side: str | None = None
        self._entry_price: Decimal | None = None
        self._entry_time: datetime | None = None
        self._position_size: Decimal = Decimal("0")
        self._entry_deviation: Decimal = Decimal("0")

    def on_price(
        self,
        price: Decimal,
        high: Decimal | None = None,
        low: Decimal | None = None,
        volume: Decimal | None = None
    ) -> MeanReversionSignal | None:
        """
        Process new price.

        Args:
            price: Current price
            high: High price (optional)
            low: Low price (optional)
            volume: Volume (optional)

        Returns:
            Signal if generated
        """
        self.analyzer.add_price(price, high, low, volume)
        signal = self.analyzer.analyze()

        if signal:
            self.metrics.total_signals += 1

            # Track max deviation
            if abs(signal.deviation_pct) > self.metrics.max_deviation_seen:
                self.metrics.max_deviation_seen = abs(signal.deviation_pct)

        return signal

    def enter_position(
        self,
        side: str,
        price: Decimal,
        size: Decimal,
        deviation: Decimal
    ) -> None:
        """Enter a position."""
        self._in_position = True
        self._position_side = side
        self._entry_price = price
        self._entry_time = datetime.now()
        self._position_size = size
        self._entry_deviation = deviation

        if side == "long":
            self.metrics.long_entries += 1
        else:
            self.metrics.short_entries += 1

        # Update average deviation at entry
        total_entries = self.metrics.long_entries + self.metrics.short_entries
        self.metrics.avg_deviation_at_entry = (
            (self.metrics.avg_deviation_at_entry * (total_entries - 1) + abs(deviation))
            / total_entries
        )

    def exit_position(
        self,
        exit_price: Decimal,
        reverted_to_mean: bool = True
    ) -> Decimal:
        """
        Exit position.

        Args:
            exit_price: Exit price
            reverted_to_mean: Whether price reverted to mean

        Returns:
            Realized PnL
        """
        if not self._in_position or self._entry_price is None:
            return Decimal("0")

        # Calculate PnL
        if self._position_side == "long":
            pnl = (exit_price - self._entry_price) * self._position_size
        else:
            pnl = (self._entry_price - exit_price) * self._position_size

        self.metrics.total_pnl += pnl

        # Track success/failure
        if reverted_to_mean and pnl > 0:
            self.metrics.successful_reversions += 1
        else:
            self.metrics.failed_reversions += 1

        # Update win rate
        total = self.metrics.successful_reversions + self.metrics.failed_reversions
        if total > 0:
            self.metrics.win_rate = Decimal(self.metrics.successful_reversions) / total * 100

        # Update average reversion time
        if self._entry_time and reverted_to_mean:
            reversion_time = (datetime.now() - self._entry_time).total_seconds() / 3600
            total_reverts = self.metrics.successful_reversions
            self.metrics.avg_reversion_time_hours = (
                (self.metrics.avg_reversion_time_hours * (total_reverts - 1) + Decimal(str(reversion_time)))
                / total_reverts
            ) if total_reverts > 0 else Decimal(str(reversion_time))

        # Reset position state
        self._in_position = False
        self._position_side = None
        self._entry_price = None
        self._entry_time = None
        self._position_size = Decimal("0")

        self.metrics.last_updated = datetime.now()

        return pnl

    def should_enter(self, signal: MeanReversionSignal) -> bool:
        """Check if should enter based on signal."""
        if self._in_position:
            return False

        if signal.entry_signal == EntrySignal.NO_SIGNAL:
            return False

        if signal.confidence < Decimal("60"):
            return False

        return True

    def should_exit(self, signal: MeanReversionSignal) -> bool:
        """Check if should exit based on signal."""
        if not self._in_position:
            return False

        # Exit on opposite signal or reversion to mean
        if self._position_side == "long":
            if signal.exit_signal == ExitSignal.EXIT_LONG:
                return True
            if signal.deviation_state in [DeviationState.AT_MEAN, DeviationState.ABOVE_MEAN]:
                return True
        elif self._position_side == "short":
            if signal.exit_signal == ExitSignal.EXIT_SHORT:
                return True
            if signal.deviation_state in [DeviationState.AT_MEAN, DeviationState.BELOW_MEAN]:
                return True

        return False

    def check_stop_loss(self, current_price: Decimal) -> bool:
        """Check if stop loss is hit."""
        if not self._in_position or self._entry_price is None:
            return False

        if self._position_side == "long":
            stop = self._entry_price * (1 - self.config.stop_loss_pct / 100)
            return current_price <= stop
        else:
            stop = self._entry_price * (1 + self.config.stop_loss_pct / 100)
            return current_price >= stop

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.config.symbol,
            "strategy_type": self.config.strategy_type.value,
            "in_position": self._in_position,
            "position_side": self._position_side,
            "entry_price": str(self._entry_price) if self._entry_price else None,
            "position_size": str(self._position_size),
            "entry_deviation": str(self._entry_deviation),
            "current_deviation": str(self.analyzer.get_current_deviation()) if self.analyzer.get_current_deviation() else None,
            "metrics": self.metrics.to_dict()
        }
