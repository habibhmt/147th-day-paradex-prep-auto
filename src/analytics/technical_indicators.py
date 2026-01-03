"""
Technical Indicators Library.

Comprehensive library of technical analysis indicators
for trading signal generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
import math


class IndicatorType(Enum):
    """Indicator type classification."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


class SignalType(Enum):
    """Signal type."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    name: str
    value: Decimal
    signal: SignalType = SignalType.NEUTRAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": str(self.value),
            "signal": self.signal.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class OHLCV:
    """OHLCV data point."""
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: Optional[datetime] = None


# ============== Moving Averages ==============

class SMA:
    """Simple Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return SMA."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)

        if len(self.values) < self.period:
            return None

        return sum(self.values) / Decimal(str(self.period))

    def current(self) -> Optional[Decimal]:
        """Get current SMA value."""
        if len(self.values) < self.period:
            return None
        return sum(self.values) / Decimal(str(self.period))


class EMA:
    """Exponential Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.multiplier = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
        self.ema: Optional[Decimal] = None
        self.count = 0

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return EMA."""
        self.count += 1

        if self.ema is None:
            self.ema = value
        else:
            self.ema = (value * self.multiplier) + (self.ema * (Decimal("1") - self.multiplier))

        if self.count < self.period:
            return None

        return self.ema

    def current(self) -> Optional[Decimal]:
        """Get current EMA value."""
        return self.ema if self.count >= self.period else None


class WMA:
    """Weighted Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return WMA."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)

        if len(self.values) < self.period:
            return None

        weights = range(1, self.period + 1)
        total_weight = sum(weights)
        weighted_sum = sum(v * Decimal(str(w)) for v, w in zip(self.values, weights))

        return weighted_sum / Decimal(str(total_weight))


class DEMA:
    """Double Exponential Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return DEMA."""
        ema1 = self.ema1.update(value)
        if ema1 is None:
            return None

        ema2 = self.ema2.update(ema1)
        if ema2 is None:
            return None

        return Decimal("2") * ema1 - ema2


class TEMA:
    """Triple Exponential Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
        self.ema3 = EMA(period)

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return TEMA."""
        ema1 = self.ema1.update(value)
        if ema1 is None:
            return None

        ema2 = self.ema2.update(ema1)
        if ema2 is None:
            return None

        ema3 = self.ema3.update(ema2)
        if ema3 is None:
            return None

        return Decimal("3") * ema1 - Decimal("3") * ema2 + ema3


class VWMA:
    """Volume Weighted Moving Average."""

    def __init__(self, period: int = 20):
        self.period = period
        self.prices: list[Decimal] = []
        self.volumes: list[Decimal] = []

    def update(self, price: Decimal, volume: Decimal) -> Optional[Decimal]:
        """Update with new value and return VWMA."""
        self.prices.append(price)
        self.volumes.append(volume)

        if len(self.prices) > self.period:
            self.prices.pop(0)
            self.volumes.pop(0)

        if len(self.prices) < self.period:
            return None

        total_volume = sum(self.volumes)
        if total_volume == 0:
            return None

        weighted_sum = sum(p * v for p, v in zip(self.prices, self.volumes))
        return weighted_sum / total_volume


# ============== Momentum Indicators ==============

class RSI:
    """Relative Strength Index."""

    def __init__(self, period: int = 14):
        self.period = period
        self.gains: list[Decimal] = []
        self.losses: list[Decimal] = []
        self.prev_value: Optional[Decimal] = None
        self.avg_gain: Optional[Decimal] = None
        self.avg_loss: Optional[Decimal] = None

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update with new value and return RSI."""
        if self.prev_value is not None:
            change = value - self.prev_value
            gain = change if change > 0 else Decimal("0")
            loss = abs(change) if change < 0 else Decimal("0")

            self.gains.append(gain)
            self.losses.append(loss)

            if len(self.gains) > self.period:
                self.gains.pop(0)
                self.losses.pop(0)

        self.prev_value = value

        if len(self.gains) < self.period:
            return None

        if self.avg_gain is None:
            self.avg_gain = sum(self.gains) / Decimal(str(self.period))
            self.avg_loss = sum(self.losses) / Decimal(str(self.period))
        else:
            self.avg_gain = (self.avg_gain * Decimal(str(self.period - 1)) + self.gains[-1]) / Decimal(str(self.period))
            self.avg_loss = (self.avg_loss * Decimal(str(self.period - 1)) + self.losses[-1]) / Decimal(str(self.period))

        if self.avg_loss == 0:
            return Decimal("100")

        rs = self.avg_gain / self.avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    def get_signal(self) -> SignalType:
        """Get trading signal based on RSI."""
        rsi = self.update(self.prev_value) if self.prev_value else None
        if rsi is None:
            return SignalType.NEUTRAL
        if rsi < Decimal("30"):
            return SignalType.BUY
        if rsi > Decimal("70"):
            return SignalType.SELL
        return SignalType.NEUTRAL


class StochasticRSI:
    """Stochastic RSI."""

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3):
        self.rsi = RSI(rsi_period)
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.rsi_values: list[Decimal] = []
        self.k_values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[tuple[Decimal, Decimal]]:
        """Update and return (K, D)."""
        rsi = self.rsi.update(value)
        if rsi is None:
            return None

        self.rsi_values.append(rsi)
        if len(self.rsi_values) > self.stoch_period:
            self.rsi_values.pop(0)

        if len(self.rsi_values) < self.stoch_period:
            return None

        highest = max(self.rsi_values)
        lowest = min(self.rsi_values)

        if highest == lowest:
            k = Decimal("50")
        else:
            k = (rsi - lowest) / (highest - lowest) * Decimal("100")

        self.k_values.append(k)
        if len(self.k_values) > self.d_period:
            self.k_values.pop(0)

        d = sum(self.k_values) / Decimal(str(len(self.k_values)))

        return k, d


class MACD:
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast_ema = EMA(fast)
        self.slow_ema = EMA(slow)
        self.signal_ema = EMA(signal)
        self.macd_line: Optional[Decimal] = None
        self.signal_line: Optional[Decimal] = None
        self.histogram: Optional[Decimal] = None

    def update(self, value: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (MACD, Signal, Histogram)."""
        fast = self.fast_ema.update(value)
        slow = self.slow_ema.update(value)

        if fast is None or slow is None:
            return None

        self.macd_line = fast - slow
        self.signal_line = self.signal_ema.update(self.macd_line)

        if self.signal_line is None:
            return None

        self.histogram = self.macd_line - self.signal_line

        return self.macd_line, self.signal_line, self.histogram

    def get_signal(self) -> SignalType:
        """Get trading signal."""
        if self.histogram is None:
            return SignalType.NEUTRAL
        if self.histogram > 0 and self.macd_line > self.signal_line:
            return SignalType.BUY
        if self.histogram < 0 and self.macd_line < self.signal_line:
            return SignalType.SELL
        return SignalType.NEUTRAL


class Momentum:
    """Momentum indicator."""

    def __init__(self, period: int = 10):
        self.period = period
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update and return momentum."""
        self.values.append(value)
        if len(self.values) > self.period + 1:
            self.values.pop(0)

        if len(self.values) <= self.period:
            return None

        return value - self.values[0]


class ROC:
    """Rate of Change."""

    def __init__(self, period: int = 10):
        self.period = period
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update and return ROC percentage."""
        self.values.append(value)
        if len(self.values) > self.period + 1:
            self.values.pop(0)

        if len(self.values) <= self.period:
            return None

        prev_value = self.values[0]
        if prev_value == 0:
            return None

        return ((value - prev_value) / prev_value) * Decimal("100")


class CCI:
    """Commodity Channel Index."""

    def __init__(self, period: int = 20):
        self.period = period
        self.typical_prices: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[Decimal]:
        """Update and return CCI."""
        tp = (high + low + close) / Decimal("3")
        self.typical_prices.append(tp)

        if len(self.typical_prices) > self.period:
            self.typical_prices.pop(0)

        if len(self.typical_prices) < self.period:
            return None

        sma = sum(self.typical_prices) / Decimal(str(self.period))

        # Mean deviation
        mean_dev = sum(abs(tp - sma) for tp in self.typical_prices) / Decimal(str(self.period))

        if mean_dev == 0:
            return Decimal("0")

        cci = (tp - sma) / (Decimal("0.015") * mean_dev)
        return cci


class WilliamsR:
    """Williams %R."""

    def __init__(self, period: int = 14):
        self.period = period
        self.highs: list[Decimal] = []
        self.lows: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[Decimal]:
        """Update and return Williams %R."""
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > self.period:
            self.highs.pop(0)
            self.lows.pop(0)

        if len(self.highs) < self.period:
            return None

        highest = max(self.highs)
        lowest = min(self.lows)

        if highest == lowest:
            return Decimal("-50")

        return ((highest - close) / (highest - lowest)) * Decimal("-100")


# ============== Volatility Indicators ==============

class ATR:
    """Average True Range."""

    def __init__(self, period: int = 14):
        self.period = period
        self.true_ranges: list[Decimal] = []
        self.prev_close: Optional[Decimal] = None
        self.atr: Optional[Decimal] = None

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[Decimal]:
        """Update and return ATR."""
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)
        else:
            tr = high - low

        self.prev_close = close
        self.true_ranges.append(tr)

        if len(self.true_ranges) > self.period:
            self.true_ranges.pop(0)

        if len(self.true_ranges) < self.period:
            return None

        if self.atr is None:
            self.atr = sum(self.true_ranges) / Decimal(str(self.period))
        else:
            self.atr = (self.atr * Decimal(str(self.period - 1)) + tr) / Decimal(str(self.period))

        return self.atr


class BollingerBands:
    """Bollinger Bands."""

    def __init__(self, period: int = 20, std_dev: Decimal = Decimal("2")):
        self.period = period
        self.std_dev = std_dev
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (upper, middle, lower) bands."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)

        if len(self.values) < self.period:
            return None

        middle = sum(self.values) / Decimal(str(self.period))

        # Standard deviation
        variance = sum((v - middle) ** 2 for v in self.values) / Decimal(str(self.period))
        std = Decimal(str(math.sqrt(float(variance))))

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        return upper, middle, lower

    def get_signal(self, current_price: Decimal) -> SignalType:
        """Get signal based on band position."""
        result = self.update(current_price)
        if result is None:
            return SignalType.NEUTRAL

        upper, middle, lower = result
        if current_price <= lower:
            return SignalType.BUY
        if current_price >= upper:
            return SignalType.SELL
        return SignalType.NEUTRAL


class KeltnerChannels:
    """Keltner Channels."""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: Decimal = Decimal("2")):
        self.ema = EMA(ema_period)
        self.atr = ATR(atr_period)
        self.multiplier = multiplier

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (upper, middle, lower)."""
        middle = self.ema.update(close)
        atr = self.atr.update(high, low, close)

        if middle is None or atr is None:
            return None

        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)

        return upper, middle, lower


class DonchianChannels:
    """Donchian Channels."""

    def __init__(self, period: int = 20):
        self.period = period
        self.highs: list[Decimal] = []
        self.lows: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (upper, middle, lower)."""
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > self.period:
            self.highs.pop(0)
            self.lows.pop(0)

        if len(self.highs) < self.period:
            return None

        upper = max(self.highs)
        lower = min(self.lows)
        middle = (upper + lower) / Decimal("2")

        return upper, middle, lower


class StandardDeviation:
    """Standard Deviation indicator."""

    def __init__(self, period: int = 20):
        self.period = period
        self.values: list[Decimal] = []

    def update(self, value: Decimal) -> Optional[Decimal]:
        """Update and return standard deviation."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)

        if len(self.values) < self.period:
            return None

        mean = sum(self.values) / Decimal(str(self.period))
        variance = sum((v - mean) ** 2 for v in self.values) / Decimal(str(self.period))

        return Decimal(str(math.sqrt(float(variance))))


# ============== Volume Indicators ==============

class OBV:
    """On Balance Volume."""

    def __init__(self):
        self.obv = Decimal("0")
        self.prev_close: Optional[Decimal] = None

    def update(self, close: Decimal, volume: Decimal) -> Decimal:
        """Update and return OBV."""
        if self.prev_close is not None:
            if close > self.prev_close:
                self.obv += volume
            elif close < self.prev_close:
                self.obv -= volume
            # If equal, OBV stays the same

        self.prev_close = close
        return self.obv


class VWAP:
    """Volume Weighted Average Price."""

    def __init__(self):
        self.cumulative_tp_vol = Decimal("0")
        self.cumulative_vol = Decimal("0")

    def update(self, high: Decimal, low: Decimal, close: Decimal, volume: Decimal) -> Decimal:
        """Update and return VWAP."""
        typical_price = (high + low + close) / Decimal("3")
        self.cumulative_tp_vol += typical_price * volume
        self.cumulative_vol += volume

        if self.cumulative_vol == 0:
            return typical_price

        return self.cumulative_tp_vol / self.cumulative_vol

    def reset(self):
        """Reset VWAP (typically at start of new period)."""
        self.cumulative_tp_vol = Decimal("0")
        self.cumulative_vol = Decimal("0")


class MFI:
    """Money Flow Index."""

    def __init__(self, period: int = 14):
        self.period = period
        self.typical_prices: list[Decimal] = []
        self.volumes: list[Decimal] = []
        self.prev_tp: Optional[Decimal] = None

    def update(self, high: Decimal, low: Decimal, close: Decimal, volume: Decimal) -> Optional[Decimal]:
        """Update and return MFI."""
        tp = (high + low + close) / Decimal("3")

        if self.prev_tp is not None:
            self.typical_prices.append(tp)
            self.volumes.append(volume if tp > self.prev_tp else -volume)

        self.prev_tp = tp

        if len(self.typical_prices) > self.period:
            self.typical_prices.pop(0)
            self.volumes.pop(0)

        if len(self.typical_prices) < self.period:
            return None

        positive_mf = sum(
            tp * abs(v) for tp, v in zip(self.typical_prices, self.volumes) if v > 0
        )
        negative_mf = sum(
            tp * abs(v) for tp, v in zip(self.typical_prices, self.volumes) if v < 0
        )

        if negative_mf == 0:
            return Decimal("100")

        mf_ratio = positive_mf / negative_mf
        mfi = Decimal("100") - (Decimal("100") / (Decimal("1") + mf_ratio))

        return mfi


class ChaikinMoneyFlow:
    """Chaikin Money Flow."""

    def __init__(self, period: int = 20):
        self.period = period
        self.mfv: list[Decimal] = []
        self.volumes: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal, volume: Decimal) -> Optional[Decimal]:
        """Update and return CMF."""
        if high == low:
            mf_multiplier = Decimal("0")
        else:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)

        mfv = mf_multiplier * volume
        self.mfv.append(mfv)
        self.volumes.append(volume)

        if len(self.mfv) > self.period:
            self.mfv.pop(0)
            self.volumes.pop(0)

        if len(self.mfv) < self.period:
            return None

        sum_vol = sum(self.volumes)
        if sum_vol == 0:
            return Decimal("0")

        return sum(self.mfv) / sum_vol


class AccumulationDistribution:
    """Accumulation/Distribution Line."""

    def __init__(self):
        self.ad_line = Decimal("0")

    def update(self, high: Decimal, low: Decimal, close: Decimal, volume: Decimal) -> Decimal:
        """Update and return A/D Line."""
        if high == low:
            mf_multiplier = Decimal("0")
        else:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)

        mfv = mf_multiplier * volume
        self.ad_line += mfv

        return self.ad_line


# ============== Trend Indicators ==============

class ADX:
    """Average Directional Index."""

    def __init__(self, period: int = 14):
        self.period = period
        self.prev_high: Optional[Decimal] = None
        self.prev_low: Optional[Decimal] = None
        self.plus_dm: list[Decimal] = []
        self.minus_dm: list[Decimal] = []
        self.tr: list[Decimal] = []
        self.prev_close: Optional[Decimal] = None
        self.dx_values: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (ADX, +DI, -DI)."""
        if self.prev_high is not None:
            # True Range
            tr = max(
                high - low,
                abs(high - self.prev_close) if self.prev_close else Decimal("0"),
                abs(low - self.prev_close) if self.prev_close else Decimal("0")
            )
            self.tr.append(tr)

            # Directional Movement
            up_move = high - self.prev_high
            down_move = self.prev_low - low

            plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
            minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")

            self.plus_dm.append(plus_dm)
            self.minus_dm.append(minus_dm)

        self.prev_high = high
        self.prev_low = low
        self.prev_close = close

        if len(self.plus_dm) > self.period:
            self.plus_dm.pop(0)
            self.minus_dm.pop(0)
            self.tr.pop(0)

        if len(self.plus_dm) < self.period:
            return None

        # Smoothed values
        atr = sum(self.tr) / Decimal(str(self.period))
        if atr == 0:
            return None

        plus_di = (sum(self.plus_dm) / Decimal(str(self.period))) / atr * Decimal("100")
        minus_di = (sum(self.minus_dm) / Decimal(str(self.period))) / atr * Decimal("100")

        # DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None

        dx = abs(plus_di - minus_di) / di_sum * Decimal("100")
        self.dx_values.append(dx)

        if len(self.dx_values) > self.period:
            self.dx_values.pop(0)

        if len(self.dx_values) < self.period:
            return None

        adx = sum(self.dx_values) / Decimal(str(len(self.dx_values)))

        return adx, plus_di, minus_di


class Aroon:
    """Aroon Indicator."""

    def __init__(self, period: int = 25):
        self.period = period
        self.highs: list[Decimal] = []
        self.lows: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """Update and return (Aroon Up, Aroon Down, Oscillator)."""
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > self.period + 1:
            self.highs.pop(0)
            self.lows.pop(0)

        if len(self.highs) < self.period + 1:
            return None

        # Find periods since highest high and lowest low
        highest_idx = self.highs.index(max(self.highs))
        lowest_idx = self.lows.index(min(self.lows))

        periods_since_high = len(self.highs) - 1 - highest_idx
        periods_since_low = len(self.lows) - 1 - lowest_idx

        aroon_up = ((Decimal(str(self.period)) - Decimal(str(periods_since_high))) /
                   Decimal(str(self.period))) * Decimal("100")
        aroon_down = ((Decimal(str(self.period)) - Decimal(str(periods_since_low))) /
                     Decimal(str(self.period))) * Decimal("100")

        oscillator = aroon_up - aroon_down

        return aroon_up, aroon_down, oscillator


class ParabolicSAR:
    """Parabolic SAR."""

    def __init__(self, af_start: Decimal = Decimal("0.02"), af_increment: Decimal = Decimal("0.02"),
                 af_max: Decimal = Decimal("0.2")):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.af = af_start
        self.ep = Decimal("0")
        self.sar: Optional[Decimal] = None
        self.is_long = True
        self.prev_high: Optional[Decimal] = None
        self.prev_low: Optional[Decimal] = None

    def update(self, high: Decimal, low: Decimal) -> Optional[Decimal]:
        """Update and return SAR."""
        if self.sar is None:
            self.sar = low
            self.ep = high
            self.prev_high = high
            self.prev_low = low
            return self.sar

        # Update SAR
        new_sar = self.sar + self.af * (self.ep - self.sar)

        if self.is_long:
            new_sar = min(new_sar, self.prev_low, low if self.prev_low else low)
            if low < new_sar:
                # Switch to short
                self.is_long = False
                new_sar = self.ep
                self.ep = low
                self.af = self.af_start
            else:
                if high > self.ep:
                    self.ep = high
                    self.af = min(self.af + self.af_increment, self.af_max)
        else:
            new_sar = max(new_sar, self.prev_high, high if self.prev_high else high)
            if high > new_sar:
                # Switch to long
                self.is_long = True
                new_sar = self.ep
                self.ep = high
                self.af = self.af_start
            else:
                if low < self.ep:
                    self.ep = low
                    self.af = min(self.af + self.af_increment, self.af_max)

        self.sar = new_sar
        self.prev_high = high
        self.prev_low = low

        return self.sar


class Ichimoku:
    """Ichimoku Cloud."""

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.senkou_b_period = senkou_b
        self.highs: list[Decimal] = []
        self.lows: list[Decimal] = []
        self.closes: list[Decimal] = []

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> Optional[dict]:
        """Update and return Ichimoku components."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        max_period = max(self.tenkan_period, self.kijun_period, self.senkou_b_period)
        if len(self.highs) > max_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        if len(self.highs) < self.senkou_b_period:
            return None

        # Tenkan-sen (Conversion Line)
        tenkan_high = max(self.highs[-self.tenkan_period:])
        tenkan_low = min(self.lows[-self.tenkan_period:])
        tenkan_sen = (tenkan_high + tenkan_low) / Decimal("2")

        # Kijun-sen (Base Line)
        kijun_high = max(self.highs[-self.kijun_period:])
        kijun_low = min(self.lows[-self.kijun_period:])
        kijun_sen = (kijun_high + kijun_low) / Decimal("2")

        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan_sen + kijun_sen) / Decimal("2")

        # Senkou Span B (Leading Span B)
        senkou_b_high = max(self.highs[-self.senkou_b_period:])
        senkou_b_low = min(self.lows[-self.senkou_b_period:])
        senkou_b = (senkou_b_high + senkou_b_low) / Decimal("2")

        # Chikou Span (Lagging Span) - current close
        chikou = close

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_a,
            "senkou_span_b": senkou_b,
            "chikou_span": chikou
        }


# ============== Indicator Collection ==============

class TechnicalIndicators:
    """Collection of all technical indicators."""

    def __init__(self):
        self.indicators: dict[str, Any] = {}

    def add_sma(self, name: str, period: int) -> SMA:
        """Add SMA indicator."""
        indicator = SMA(period)
        self.indicators[name] = indicator
        return indicator

    def add_ema(self, name: str, period: int) -> EMA:
        """Add EMA indicator."""
        indicator = EMA(period)
        self.indicators[name] = indicator
        return indicator

    def add_rsi(self, name: str, period: int = 14) -> RSI:
        """Add RSI indicator."""
        indicator = RSI(period)
        self.indicators[name] = indicator
        return indicator

    def add_macd(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9) -> MACD:
        """Add MACD indicator."""
        indicator = MACD(fast, slow, signal)
        self.indicators[name] = indicator
        return indicator

    def add_bollinger(self, name: str, period: int = 20, std_dev: Decimal = Decimal("2")) -> BollingerBands:
        """Add Bollinger Bands indicator."""
        indicator = BollingerBands(period, std_dev)
        self.indicators[name] = indicator
        return indicator

    def add_atr(self, name: str, period: int = 14) -> ATR:
        """Add ATR indicator."""
        indicator = ATR(period)
        self.indicators[name] = indicator
        return indicator

    def add_adx(self, name: str, period: int = 14) -> ADX:
        """Add ADX indicator."""
        indicator = ADX(period)
        self.indicators[name] = indicator
        return indicator

    def get(self, name: str) -> Optional[Any]:
        """Get indicator by name."""
        return self.indicators.get(name)

    def update_all(self, ohlcv: OHLCV) -> dict[str, Any]:
        """Update all indicators with OHLCV data."""
        results = {}

        for name, indicator in self.indicators.items():
            if isinstance(indicator, (SMA, EMA, WMA, DEMA, TEMA, RSI, Momentum, ROC)):
                results[name] = indicator.update(ohlcv.close)
            elif isinstance(indicator, MACD):
                results[name] = indicator.update(ohlcv.close)
            elif isinstance(indicator, BollingerBands):
                results[name] = indicator.update(ohlcv.close)
            elif isinstance(indicator, (ATR, ADX, CCI, WilliamsR)):
                results[name] = indicator.update(ohlcv.high, ohlcv.low, ohlcv.close)
            elif isinstance(indicator, (KeltnerChannels, DonchianChannels)):
                results[name] = indicator.update(ohlcv.high, ohlcv.low, ohlcv.close if hasattr(indicator, 'ema') else None)
            elif isinstance(indicator, (OBV, VWAP, MFI, ChaikinMoneyFlow, AccumulationDistribution)):
                results[name] = indicator.update(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

        return results


# Global instance
_indicators: Optional[TechnicalIndicators] = None


def get_indicators() -> TechnicalIndicators:
    """Get global indicators instance."""
    global _indicators
    if _indicators is None:
        _indicators = TechnicalIndicators()
    return _indicators


def set_indicators(indicators: TechnicalIndicators):
    """Set global indicators instance."""
    global _indicators
    _indicators = indicators
