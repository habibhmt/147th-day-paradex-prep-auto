"""
Momentum Trading Strategy Module.

Implements various momentum-based trading strategies:
- Trend following with moving averages
- Breakout trading
- Momentum oscillators (RSI, MACD, Stochastic)
- Volume-confirmed momentum
- Multi-timeframe momentum
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


class MomentumType(Enum):
    """Type of momentum strategy."""
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    OSCILLATOR = "oscillator"
    VOLUME_MOMENTUM = "volume_momentum"
    MULTI_TIMEFRAME = "multi_timeframe"


class TrendDirection(Enum):
    """Trend direction."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class SignalType(Enum):
    """Signal type."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class PositionState(Enum):
    """Position state."""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class PriceBar:
    """OHLCV price bar."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume)
        }


@dataclass
class MomentumSignal:
    """Momentum trading signal."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    trend_direction: TrendDirection
    strength: Decimal  # 0-100
    price: Decimal
    target_price: Decimal | None = None
    stop_loss: Decimal | None = None
    indicators: dict[str, Decimal] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "trend_direction": self.trend_direction.value,
            "strength": str(self.strength),
            "price": str(self.price),
            "target_price": str(self.target_price) if self.target_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "indicators": {k: str(v) for k, v in self.indicators.items()},
            "reason": self.reason
        }


@dataclass
class MomentumConfig:
    """Configuration for momentum strategy."""
    symbol: str
    momentum_type: MomentumType
    # MA settings
    fast_ma_period: int = 9
    slow_ma_period: int = 21
    signal_ma_period: int = 9
    # RSI settings
    rsi_period: int = 14
    rsi_overbought: Decimal = Decimal("70")
    rsi_oversold: Decimal = Decimal("30")
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Breakout settings
    breakout_period: int = 20
    breakout_confirmation_bars: int = 2
    # Volume settings
    volume_ma_period: int = 20
    volume_multiplier: Decimal = Decimal("1.5")
    # Risk settings
    stop_loss_pct: Decimal = Decimal("2")
    take_profit_pct: Decimal = Decimal("4")
    trailing_stop_pct: Decimal | None = None
    max_position_size: Decimal | None = None
    # Filter settings
    min_trend_strength: Decimal = Decimal("25")
    min_volume_ratio: Decimal = Decimal("0.8")

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if self.fast_ma_period >= self.slow_ma_period:
            errors.append("Fast MA period must be less than slow MA period")

        if self.rsi_period < 2:
            errors.append("RSI period must be at least 2")

        if self.rsi_overbought <= self.rsi_oversold:
            errors.append("RSI overbought must be greater than oversold")

        if self.stop_loss_pct <= Decimal("0"):
            errors.append("Stop loss must be positive")

        if self.take_profit_pct <= Decimal("0"):
            errors.append("Take profit must be positive")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "momentum_type": self.momentum_type.value,
            "fast_ma_period": self.fast_ma_period,
            "slow_ma_period": self.slow_ma_period,
            "signal_ma_period": self.signal_ma_period,
            "rsi_period": self.rsi_period,
            "rsi_overbought": str(self.rsi_overbought),
            "rsi_oversold": str(self.rsi_oversold),
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "breakout_period": self.breakout_period,
            "volume_ma_period": self.volume_ma_period,
            "volume_multiplier": str(self.volume_multiplier),
            "stop_loss_pct": str(self.stop_loss_pct),
            "take_profit_pct": str(self.take_profit_pct),
            "trailing_stop_pct": str(self.trailing_stop_pct) if self.trailing_stop_pct else None,
            "max_position_size": str(self.max_position_size) if self.max_position_size else None,
            "min_trend_strength": str(self.min_trend_strength),
            "min_volume_ratio": str(self.min_volume_ratio)
        }


@dataclass
class MomentumMetrics:
    """Performance metrics for momentum strategy."""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl),
            "win_rate": str(self.win_rate),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "profit_factor": str(self.profit_factor),
            "max_drawdown": str(self.max_drawdown),
            "sharpe_ratio": str(self.sharpe_ratio),
            "last_updated": self.last_updated.isoformat()
        }


class TechnicalIndicators:
    """Technical indicator calculations."""

    @staticmethod
    def sma(prices: list[Decimal], period: int) -> Decimal | None:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def ema(prices: list[Decimal], period: int) -> Decimal | None:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        multiplier = Decimal("2") / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    @staticmethod
    def rsi(prices: list[Decimal], period: int = 14) -> Decimal | None:
        """Calculate Relative Strength Index."""
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
        rsi = Decimal("100") - (Decimal("100") / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        prices: list[Decimal],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """
        Calculate MACD.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < slow_period + signal_period:
            return None, None, None

        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)

        if fast_ema is None or slow_ema is None:
            return None, None, None

        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD)
        # This is simplified - in practice you'd track MACD history
        signal_line = macd_line * Decimal("0.9")  # Approximation
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple[Decimal | None, Decimal | None]:
        """
        Calculate Stochastic Oscillator.

        Returns:
            Tuple of (%K, %D)
        """
        if len(closes) < k_period:
            return None, None

        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])
        current_close = closes[-1]

        if highest_high == lowest_low:
            return Decimal("50"), Decimal("50")

        k = (current_close - lowest_low) / (highest_high - lowest_low) * 100

        # %D is SMA of %K (simplified)
        d = k  # Would need K history for proper calculation

        return k, d

    @staticmethod
    def atr(
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
    def bollinger_bands(
        prices: list[Decimal],
        period: int = 20,
        std_dev: Decimal = Decimal("2")
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            return None, None, None

        middle = TechnicalIndicators.sma(prices, period)
        if middle is None:
            return None, None, None

        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in prices[-period:]) / period
        std = variance.sqrt() if variance > 0 else Decimal("0")

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    @staticmethod
    def adx(
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
        period: int = 14
    ) -> Decimal | None:
        """Calculate Average Directional Index."""
        if len(closes) < period * 2:
            return None

        # Simplified ADX calculation
        atr = TechnicalIndicators.atr(highs, lows, closes, period)
        if atr is None or atr == 0:
            return None

        plus_dm = []
        minus_dm = []

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(Decimal("0"))

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(Decimal("0"))

        if len(plus_dm) < period:
            return None

        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return Decimal("0")

        dx = abs(plus_di - minus_di) / di_sum * 100

        return dx


class MomentumAnalyzer:
    """Analyzes momentum and generates signals."""

    def __init__(self, config: MomentumConfig):
        """Initialize momentum analyzer."""
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {', '.join(errors)}")

        self.config = config
        self._price_history: list[PriceBar] = []
        self._max_history = 500
        self._signals: list[MomentumSignal] = []

    def add_bar(self, bar: PriceBar) -> None:
        """Add a price bar to history."""
        self._price_history.append(bar)
        if len(self._price_history) > self._max_history:
            self._price_history = self._price_history[-self._max_history:]

    def get_closes(self) -> list[Decimal]:
        """Get closing prices."""
        return [bar.close for bar in self._price_history]

    def get_highs(self) -> list[Decimal]:
        """Get high prices."""
        return [bar.high for bar in self._price_history]

    def get_lows(self) -> list[Decimal]:
        """Get low prices."""
        return [bar.low for bar in self._price_history]

    def get_volumes(self) -> list[Decimal]:
        """Get volumes."""
        return [bar.volume for bar in self._price_history]

    def analyze(self) -> MomentumSignal | None:
        """
        Analyze current momentum and generate signal.

        Returns:
            MomentumSignal or None
        """
        if len(self._price_history) < self.config.slow_ma_period:
            return None

        closes = self.get_closes()
        current_price = closes[-1]

        # Calculate indicators based on momentum type
        if self.config.momentum_type == MomentumType.TREND_FOLLOWING:
            return self._analyze_trend_following(closes, current_price)
        elif self.config.momentum_type == MomentumType.BREAKOUT:
            return self._analyze_breakout(current_price)
        elif self.config.momentum_type == MomentumType.OSCILLATOR:
            return self._analyze_oscillator(closes, current_price)
        elif self.config.momentum_type == MomentumType.VOLUME_MOMENTUM:
            return self._analyze_volume_momentum(closes, current_price)
        elif self.config.momentum_type == MomentumType.MULTI_TIMEFRAME:
            return self._analyze_multi_timeframe(closes, current_price)

        return None

    def _analyze_trend_following(
        self,
        closes: list[Decimal],
        current_price: Decimal
    ) -> MomentumSignal | None:
        """Analyze using trend following strategy."""
        fast_ma = TechnicalIndicators.ema(closes, self.config.fast_ma_period)
        slow_ma = TechnicalIndicators.ema(closes, self.config.slow_ma_period)

        if fast_ma is None or slow_ma is None:
            return None

        # Determine trend direction
        ma_diff = fast_ma - slow_ma
        ma_diff_pct = (ma_diff / slow_ma * 100) if slow_ma > 0 else Decimal("0")

        if ma_diff_pct > Decimal("2"):
            trend = TrendDirection.STRONG_BULLISH
        elif ma_diff_pct > Decimal("0.5"):
            trend = TrendDirection.BULLISH
        elif ma_diff_pct < Decimal("-2"):
            trend = TrendDirection.STRONG_BEARISH
        elif ma_diff_pct < Decimal("-0.5"):
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL

        # Generate signal
        if trend in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
            signal_type = SignalType.BUY if trend == TrendDirection.BULLISH else SignalType.STRONG_BUY
            strength = min(abs(ma_diff_pct) * 10, Decimal("100"))
        elif trend in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
            signal_type = SignalType.SELL if trend == TrendDirection.BEARISH else SignalType.STRONG_SELL
            strength = min(abs(ma_diff_pct) * 10, Decimal("100"))
        else:
            signal_type = SignalType.HOLD
            strength = Decimal("50")

        # Calculate stop loss and take profit
        atr = TechnicalIndicators.atr(
            self.get_highs(),
            self.get_lows(),
            closes,
            14
        ) or current_price * Decimal("0.02")

        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = current_price - (atr * 2)
            target = current_price + (atr * 3)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            stop_loss = current_price + (atr * 2)
            target = current_price - (atr * 3)
        else:
            stop_loss = None
            target = None

        signal = MomentumSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            signal_type=signal_type,
            trend_direction=trend,
            strength=strength,
            price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            indicators={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "ma_diff_pct": ma_diff_pct
            },
            reason=f"MA crossover: fast={fast_ma:.2f}, slow={slow_ma:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_breakout(self, current_price: Decimal) -> MomentumSignal | None:
        """Analyze using breakout strategy."""
        if len(self._price_history) < self.config.breakout_period:
            return None

        highs = self.get_highs()
        lows = self.get_lows()
        closes = self.get_closes()

        period_high = max(highs[-self.config.breakout_period:-1])
        period_low = min(lows[-self.config.breakout_period:-1])

        # Check for breakout
        breakout_up = current_price > period_high
        breakout_down = current_price < period_low

        if breakout_up:
            signal_type = SignalType.STRONG_BUY
            trend = TrendDirection.STRONG_BULLISH
            breakout_pct = (current_price - period_high) / period_high * 100
            strength = min(breakout_pct * 20, Decimal("100"))
            reason = f"Breakout above {period_high:.2f}"
        elif breakout_down:
            signal_type = SignalType.STRONG_SELL
            trend = TrendDirection.STRONG_BEARISH
            breakout_pct = (period_low - current_price) / period_low * 100
            strength = min(breakout_pct * 20, Decimal("100"))
            reason = f"Breakdown below {period_low:.2f}"
        else:
            signal_type = SignalType.HOLD
            trend = TrendDirection.NEUTRAL
            strength = Decimal("50")
            reason = "No breakout"

        atr = TechnicalIndicators.atr(highs, lows, closes, 14) or current_price * Decimal("0.02")

        signal = MomentumSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            signal_type=signal_type,
            trend_direction=trend,
            strength=strength,
            price=current_price,
            target_price=current_price + (atr * 3) if breakout_up else (
                current_price - (atr * 3) if breakout_down else None
            ),
            stop_loss=current_price - (atr * 2) if breakout_up else (
                current_price + (atr * 2) if breakout_down else None
            ),
            indicators={
                "period_high": period_high,
                "period_low": period_low
            },
            reason=reason
        )

        self._signals.append(signal)
        return signal

    def _analyze_oscillator(
        self,
        closes: list[Decimal],
        current_price: Decimal
    ) -> MomentumSignal | None:
        """Analyze using oscillator strategy (RSI, MACD)."""
        rsi = TechnicalIndicators.rsi(closes, self.config.rsi_period)
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(
            closes,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )

        if rsi is None:
            return None

        # Determine signal from RSI
        if rsi < self.config.rsi_oversold:
            rsi_signal = SignalType.BUY
            trend = TrendDirection.BULLISH
        elif rsi > self.config.rsi_overbought:
            rsi_signal = SignalType.SELL
            trend = TrendDirection.BEARISH
        else:
            rsi_signal = SignalType.HOLD
            trend = TrendDirection.NEUTRAL

        # Combine with MACD if available
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and rsi_signal == SignalType.BUY:
                signal_type = SignalType.STRONG_BUY
                trend = TrendDirection.STRONG_BULLISH
            elif macd < macd_signal and rsi_signal == SignalType.SELL:
                signal_type = SignalType.STRONG_SELL
                trend = TrendDirection.STRONG_BEARISH
            else:
                signal_type = rsi_signal
        else:
            signal_type = rsi_signal

        # Calculate strength based on RSI extremes
        if rsi < Decimal("30"):
            strength = (Decimal("30") - rsi) * 3
        elif rsi > Decimal("70"):
            strength = (rsi - Decimal("70")) * 3
        else:
            strength = Decimal("50")
        strength = min(strength, Decimal("100"))

        indicators = {"rsi": rsi}
        if macd is not None:
            indicators["macd"] = macd
        if macd_signal is not None:
            indicators["macd_signal"] = macd_signal

        signal = MomentumSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            signal_type=signal_type,
            trend_direction=trend,
            strength=strength,
            price=current_price,
            indicators=indicators,
            reason=f"RSI: {rsi:.2f}"
        )

        self._signals.append(signal)
        return signal

    def _analyze_volume_momentum(
        self,
        closes: list[Decimal],
        current_price: Decimal
    ) -> MomentumSignal | None:
        """Analyze using volume-confirmed momentum."""
        volumes = self.get_volumes()

        if len(volumes) < self.config.volume_ma_period:
            return None

        volume_ma = TechnicalIndicators.sma(volumes, self.config.volume_ma_period)
        current_volume = volumes[-1]

        if volume_ma is None or volume_ma == 0:
            return None

        volume_ratio = current_volume / volume_ma

        # Price momentum
        price_change = closes[-1] - closes[-2] if len(closes) >= 2 else Decimal("0")
        price_change_pct = (price_change / closes[-2] * 100) if closes[-2] > 0 else Decimal("0")

        # Volume-confirmed move
        high_volume = volume_ratio >= self.config.volume_multiplier

        if high_volume and price_change_pct > Decimal("0.5"):
            signal_type = SignalType.STRONG_BUY
            trend = TrendDirection.STRONG_BULLISH
            strength = min(volume_ratio * 30 + price_change_pct * 10, Decimal("100"))
            reason = f"Volume surge ({volume_ratio:.1f}x) with bullish move"
        elif high_volume and price_change_pct < Decimal("-0.5"):
            signal_type = SignalType.STRONG_SELL
            trend = TrendDirection.STRONG_BEARISH
            strength = min(volume_ratio * 30 + abs(price_change_pct) * 10, Decimal("100"))
            reason = f"Volume surge ({volume_ratio:.1f}x) with bearish move"
        elif price_change_pct > Decimal("0.3"):
            signal_type = SignalType.BUY
            trend = TrendDirection.BULLISH
            strength = Decimal("60")
            reason = "Bullish momentum"
        elif price_change_pct < Decimal("-0.3"):
            signal_type = SignalType.SELL
            trend = TrendDirection.BEARISH
            strength = Decimal("60")
            reason = "Bearish momentum"
        else:
            signal_type = SignalType.HOLD
            trend = TrendDirection.NEUTRAL
            strength = Decimal("50")
            reason = "No significant momentum"

        signal = MomentumSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            signal_type=signal_type,
            trend_direction=trend,
            strength=strength,
            price=current_price,
            indicators={
                "volume_ratio": volume_ratio,
                "price_change_pct": price_change_pct,
                "volume_ma": volume_ma
            },
            reason=reason
        )

        self._signals.append(signal)
        return signal

    def _analyze_multi_timeframe(
        self,
        closes: list[Decimal],
        current_price: Decimal
    ) -> MomentumSignal | None:
        """Analyze using multi-timeframe approach."""
        # Short-term (fast MA)
        fast_ma = TechnicalIndicators.ema(closes, self.config.fast_ma_period)
        # Medium-term (slow MA)
        slow_ma = TechnicalIndicators.ema(closes, self.config.slow_ma_period)
        # Long-term (using 2x slow MA period)
        long_ma = TechnicalIndicators.sma(closes, self.config.slow_ma_period * 2)

        if fast_ma is None or slow_ma is None or long_ma is None:
            return None

        # Determine alignment
        all_bullish = fast_ma > slow_ma > long_ma
        all_bearish = fast_ma < slow_ma < long_ma

        if all_bullish:
            signal_type = SignalType.STRONG_BUY
            trend = TrendDirection.STRONG_BULLISH
            alignment = Decimal("100")
            reason = "All timeframes aligned bullish"
        elif all_bearish:
            signal_type = SignalType.STRONG_SELL
            trend = TrendDirection.STRONG_BEARISH
            alignment = Decimal("100")
            reason = "All timeframes aligned bearish"
        elif fast_ma > slow_ma:
            signal_type = SignalType.BUY
            trend = TrendDirection.BULLISH
            alignment = Decimal("66")
            reason = "Short and medium term bullish"
        elif fast_ma < slow_ma:
            signal_type = SignalType.SELL
            trend = TrendDirection.BEARISH
            alignment = Decimal("66")
            reason = "Short and medium term bearish"
        else:
            signal_type = SignalType.HOLD
            trend = TrendDirection.NEUTRAL
            alignment = Decimal("33")
            reason = "Mixed signals"

        signal = MomentumSignal(
            timestamp=datetime.now(),
            symbol=self.config.symbol,
            signal_type=signal_type,
            trend_direction=trend,
            strength=alignment,
            price=current_price,
            indicators={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "long_ma": long_ma
            },
            reason=reason
        )

        self._signals.append(signal)
        return signal

    def get_recent_signals(self, count: int = 10) -> list[MomentumSignal]:
        """Get recent signals."""
        return self._signals[-count:]

    def get_current_trend(self) -> TrendDirection:
        """Get current trend direction."""
        if not self._signals:
            return TrendDirection.NEUTRAL
        return self._signals[-1].trend_direction


class MomentumStrategy:
    """Complete momentum trading strategy."""

    def __init__(
        self,
        config: MomentumConfig,
        strategy_id: str | None = None
    ):
        """Initialize momentum strategy."""
        self.strategy_id = strategy_id or f"momentum_{config.symbol}"
        self.config = config
        self.analyzer = MomentumAnalyzer(config)
        self.metrics = MomentumMetrics()

        self._position_state = PositionState.FLAT
        self._entry_price: Decimal | None = None
        self._position_size: Decimal = Decimal("0")
        self._trailing_stop: Decimal | None = None
        self._highest_since_entry: Decimal = Decimal("0")
        self._lowest_since_entry: Decimal = Decimal("999999999")

    def on_bar(self, bar: PriceBar) -> MomentumSignal | None:
        """
        Process new price bar.

        Args:
            bar: New price bar

        Returns:
            Signal if generated
        """
        self.analyzer.add_bar(bar)

        # Update trailing stop
        self._update_trailing_stop(bar.close)

        # Generate signal
        signal = self.analyzer.analyze()

        if signal:
            self.metrics.total_signals += 1
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                self.metrics.buy_signals += 1
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                self.metrics.sell_signals += 1

        return signal

    def _update_trailing_stop(self, current_price: Decimal) -> None:
        """Update trailing stop if enabled."""
        if not self.config.trailing_stop_pct:
            return

        if self._position_state == PositionState.LONG:
            if current_price > self._highest_since_entry:
                self._highest_since_entry = current_price
                self._trailing_stop = current_price * (1 - self.config.trailing_stop_pct / 100)
        elif self._position_state == PositionState.SHORT:
            if current_price < self._lowest_since_entry:
                self._lowest_since_entry = current_price
                self._trailing_stop = current_price * (1 + self.config.trailing_stop_pct / 100)

    def enter_position(
        self,
        side: str,
        price: Decimal,
        size: Decimal
    ) -> None:
        """Enter a position."""
        self._position_state = PositionState.LONG if side == "long" else PositionState.SHORT
        self._entry_price = price
        self._position_size = size
        self._highest_since_entry = price
        self._lowest_since_entry = price

    def exit_position(self, exit_price: Decimal) -> Decimal:
        """
        Exit position and calculate PnL.

        Returns:
            Realized PnL
        """
        if self._entry_price is None:
            return Decimal("0")

        if self._position_state == PositionState.LONG:
            pnl = (exit_price - self._entry_price) * self._position_size
        elif self._position_state == PositionState.SHORT:
            pnl = (self._entry_price - exit_price) * self._position_size
        else:
            pnl = Decimal("0")

        # Update metrics
        self.metrics.total_pnl += pnl
        if pnl > Decimal("0"):
            self.metrics.winning_trades += 1
            # Update avg win
            total_wins = self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl
            self.metrics.avg_win = total_wins / self.metrics.winning_trades
        else:
            self.metrics.losing_trades += 1
            # Update avg loss
            total_losses = self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)
            if self.metrics.losing_trades > 0:
                self.metrics.avg_loss = total_losses / self.metrics.losing_trades

        # Update win rate
        total_trades = self.metrics.winning_trades + self.metrics.losing_trades
        if total_trades > 0:
            self.metrics.win_rate = Decimal(self.metrics.winning_trades) / total_trades * 100

        # Update profit factor
        if self.metrics.avg_loss > 0 and self.metrics.losing_trades > 0:
            total_wins = self.metrics.avg_win * self.metrics.winning_trades
            total_losses = self.metrics.avg_loss * self.metrics.losing_trades
            if total_losses > 0:
                self.metrics.profit_factor = total_wins / total_losses

        # Reset position state
        self._position_state = PositionState.FLAT
        self._entry_price = None
        self._position_size = Decimal("0")
        self._trailing_stop = None

        self.metrics.last_updated = datetime.now()

        return pnl

    def check_stops(self, current_price: Decimal) -> dict[str, Any]:
        """
        Check if any stops are hit.

        Returns:
            Dict with stop info if triggered
        """
        if self._position_state == PositionState.FLAT:
            return {"triggered": False}

        if self._entry_price is None:
            return {"triggered": False}

        # Check trailing stop
        if self._trailing_stop:
            if self._position_state == PositionState.LONG and current_price <= self._trailing_stop:
                return {
                    "triggered": True,
                    "type": "trailing_stop",
                    "price": self._trailing_stop
                }
            elif self._position_state == PositionState.SHORT and current_price >= self._trailing_stop:
                return {
                    "triggered": True,
                    "type": "trailing_stop",
                    "price": self._trailing_stop
                }

        # Check fixed stop loss
        if self._position_state == PositionState.LONG:
            stop_price = self._entry_price * (1 - self.config.stop_loss_pct / 100)
            if current_price <= stop_price:
                return {"triggered": True, "type": "stop_loss", "price": stop_price}

            target_price = self._entry_price * (1 + self.config.take_profit_pct / 100)
            if current_price >= target_price:
                return {"triggered": True, "type": "take_profit", "price": target_price}

        elif self._position_state == PositionState.SHORT:
            stop_price = self._entry_price * (1 + self.config.stop_loss_pct / 100)
            if current_price >= stop_price:
                return {"triggered": True, "type": "stop_loss", "price": stop_price}

            target_price = self._entry_price * (1 - self.config.take_profit_pct / 100)
            if current_price <= target_price:
                return {"triggered": True, "type": "take_profit", "price": target_price}

        return {"triggered": False}

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.config.symbol,
            "momentum_type": self.config.momentum_type.value,
            "position_state": self._position_state.value,
            "entry_price": str(self._entry_price) if self._entry_price else None,
            "position_size": str(self._position_size),
            "trailing_stop": str(self._trailing_stop) if self._trailing_stop else None,
            "current_trend": self.analyzer.get_current_trend().value,
            "metrics": self.metrics.to_dict()
        }
