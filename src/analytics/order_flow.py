"""
Order Flow Analysis for Paradex.

Analyzes order flow, volume delta, footprint charts, and market pressure.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import logging


logger = logging.getLogger(__name__)


class OrderFlowType(Enum):
    """Order flow event types."""
    BUY_MARKET = "buy_market"
    SELL_MARKET = "sell_market"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_LIQUIDATION = "buy_liquidation"
    SELL_LIQUIDATION = "sell_liquidation"


class AggresionType(Enum):
    """Aggression type in order flow."""
    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"
    PASSIVE_BUY = "passive_buy"
    PASSIVE_SELL = "passive_sell"


class MarketPressure(Enum):
    """Market pressure direction."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ImbalanceType(Enum):
    """Order book imbalance type."""
    BUYING = "buying"
    SELLING = "selling"
    BALANCED = "balanced"


@dataclass
class TradeFlow:
    """Single trade in order flow."""
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # buy or sell
    is_liquidation: bool = False
    trade_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "size": str(self.size),
            "side": self.side,
            "is_liquidation": self.is_liquidation,
            "trade_id": self.trade_id,
        }

    @property
    def notional(self) -> Decimal:
        """Get notional value."""
        return self.price * self.size


@dataclass
class VolumeDelta:
    """Volume delta for a time period."""
    timestamp: datetime
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    delta: Decimal = Decimal("0")
    cumulative_delta: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "buy_volume": str(self.buy_volume),
            "sell_volume": str(self.sell_volume),
            "delta": str(self.delta),
            "cumulative_delta": str(self.cumulative_delta),
            "total_volume": str(self.total_volume),
        }


@dataclass
class FootprintLevel:
    """Single price level in footprint chart."""
    price: Decimal
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    buy_trades: int = 0
    sell_trades: int = 0
    delta: Decimal = Decimal("0")
    imbalance: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "price": str(self.price),
            "buy_volume": str(self.buy_volume),
            "sell_volume": str(self.sell_volume),
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "delta": str(self.delta),
            "imbalance": str(self.imbalance),
        }


@dataclass
class FootprintBar:
    """Footprint chart bar with price levels."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    levels: dict[Decimal, FootprintLevel] = field(default_factory=dict)
    total_buy_volume: Decimal = Decimal("0")
    total_sell_volume: Decimal = Decimal("0")
    delta: Decimal = Decimal("0")
    poc_price: Optional[Decimal] = None  # Point of Control

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "levels": {str(p): l.to_dict() for p, l in self.levels.items()},
            "total_buy_volume": str(self.total_buy_volume),
            "total_sell_volume": str(self.total_sell_volume),
            "delta": str(self.delta),
            "poc_price": str(self.poc_price) if self.poc_price else None,
        }


@dataclass
class VolumeProfile:
    """Volume profile for a time period."""
    start_time: datetime
    end_time: datetime
    levels: dict[Decimal, FootprintLevel] = field(default_factory=dict)
    poc_price: Optional[Decimal] = None
    value_area_high: Optional[Decimal] = None
    value_area_low: Optional[Decimal] = None
    total_volume: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "levels": {str(p): l.to_dict() for p, l in self.levels.items()},
            "poc_price": str(self.poc_price) if self.poc_price else None,
            "value_area_high": str(self.value_area_high) if self.value_area_high else None,
            "value_area_low": str(self.value_area_low) if self.value_area_low else None,
            "total_volume": str(self.total_volume),
        }


@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics."""
    symbol: str
    cvd: Decimal = Decimal("0")
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    buy_trades: int = 0
    sell_trades: int = 0
    avg_buy_size: Decimal = Decimal("0")
    avg_sell_size: Decimal = Decimal("0")
    large_buy_count: int = 0
    large_sell_count: int = 0
    liquidation_volume: Decimal = Decimal("0")
    pressure: MarketPressure = MarketPressure.NEUTRAL
    imbalance: ImbalanceType = ImbalanceType.BALANCED
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "cvd": str(self.cvd),
            "buy_volume": str(self.buy_volume),
            "sell_volume": str(self.sell_volume),
            "total_volume": str(self.total_volume),
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "avg_buy_size": str(self.avg_buy_size),
            "avg_sell_size": str(self.avg_sell_size),
            "large_buy_count": self.large_buy_count,
            "large_sell_count": self.large_sell_count,
            "liquidation_volume": str(self.liquidation_volume),
            "pressure": self.pressure.value,
            "imbalance": self.imbalance.value,
            "timestamp": self.timestamp.isoformat(),
        }


class DeltaCalculator:
    """Calculate volume delta and CVD."""

    def __init__(self):
        """Initialize delta calculator."""
        self._cvd = Decimal("0")

    def reset(self) -> None:
        """Reset CVD."""
        self._cvd = Decimal("0")

    def process_trade(self, trade: TradeFlow) -> VolumeDelta:
        """Process a single trade and update CVD."""
        if trade.side == "buy":
            delta = trade.size
            buy_vol = trade.size
            sell_vol = Decimal("0")
        else:
            delta = -trade.size
            buy_vol = Decimal("0")
            sell_vol = trade.size

        self._cvd += delta

        return VolumeDelta(
            timestamp=trade.timestamp,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            delta=delta,
            cumulative_delta=self._cvd,
            total_volume=trade.size,
        )

    def calculate_for_period(self, trades: list[TradeFlow]) -> list[VolumeDelta]:
        """Calculate delta for a list of trades."""
        deltas = []
        for trade in trades:
            deltas.append(self.process_trade(trade))
        return deltas

    @property
    def cvd(self) -> Decimal:
        """Get current CVD."""
        return self._cvd


class FootprintBuilder:
    """Build footprint charts from trade data."""

    def __init__(self, tick_size: Decimal = Decimal("1")):
        """Initialize footprint builder."""
        self.tick_size = tick_size

    def build_bar(
        self,
        trades: list[TradeFlow],
        timestamp: datetime,
    ) -> FootprintBar:
        """Build a footprint bar from trades."""
        if not trades:
            return FootprintBar(
                timestamp=timestamp,
                open=Decimal("0"),
                high=Decimal("0"),
                low=Decimal("0"),
                close=Decimal("0"),
            )

        levels: dict[Decimal, FootprintLevel] = {}

        for trade in trades:
            # Round price to tick size
            price_level = self._round_to_tick(trade.price)

            if price_level not in levels:
                levels[price_level] = FootprintLevel(price=price_level)

            level = levels[price_level]
            if trade.side == "buy":
                level.buy_volume += trade.size
                level.buy_trades += 1
            else:
                level.sell_volume += trade.size
                level.sell_trades += 1

        # Calculate delta and imbalance for each level
        for level in levels.values():
            level.delta = level.buy_volume - level.sell_volume
            total = level.buy_volume + level.sell_volume
            if total > 0:
                level.imbalance = level.delta / total * 100

        # Calculate bar totals
        total_buy = sum(l.buy_volume for l in levels.values())
        total_sell = sum(l.sell_volume for l in levels.values())

        # Find POC (Point of Control - highest volume level)
        poc_price = None
        if levels:
            poc_level = max(
                levels.values(),
                key=lambda l: l.buy_volume + l.sell_volume
            )
            poc_price = poc_level.price

        prices = [t.price for t in trades]

        return FootprintBar(
            timestamp=timestamp,
            open=trades[0].price,
            high=max(prices),
            low=min(prices),
            close=trades[-1].price,
            levels=levels,
            total_buy_volume=total_buy,
            total_sell_volume=total_sell,
            delta=total_buy - total_sell,
            poc_price=poc_price,
        )

    def _round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        return (price / self.tick_size).quantize(Decimal("1")) * self.tick_size


class VolumeProfileBuilder:
    """Build volume profile from trade data."""

    def __init__(
        self,
        tick_size: Decimal = Decimal("1"),
        value_area_pct: Decimal = Decimal("70"),
    ):
        """Initialize volume profile builder."""
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct

    def build(
        self,
        trades: list[TradeFlow],
        start_time: datetime,
        end_time: datetime,
    ) -> VolumeProfile:
        """Build volume profile from trades."""
        levels: dict[Decimal, FootprintLevel] = {}

        for trade in trades:
            price_level = self._round_to_tick(trade.price)

            if price_level not in levels:
                levels[price_level] = FootprintLevel(price=price_level)

            level = levels[price_level]
            if trade.side == "buy":
                level.buy_volume += trade.size
                level.buy_trades += 1
            else:
                level.sell_volume += trade.size
                level.sell_trades += 1

        # Calculate delta for each level
        for level in levels.values():
            level.delta = level.buy_volume - level.sell_volume

        total_volume = sum(l.buy_volume + l.sell_volume for l in levels.values())

        # Find POC
        poc_price = None
        if levels:
            poc_level = max(
                levels.values(),
                key=lambda l: l.buy_volume + l.sell_volume
            )
            poc_price = poc_level.price

        # Calculate Value Area
        vah, val = self._calculate_value_area(levels, total_volume)

        return VolumeProfile(
            start_time=start_time,
            end_time=end_time,
            levels=levels,
            poc_price=poc_price,
            value_area_high=vah,
            value_area_low=val,
            total_volume=total_volume,
        )

    def _calculate_value_area(
        self,
        levels: dict[Decimal, FootprintLevel],
        total_volume: Decimal,
    ) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate Value Area High and Low."""
        if not levels or total_volume == 0:
            return None, None

        target_volume = total_volume * self.value_area_pct / 100

        # Sort levels by volume
        sorted_levels = sorted(
            levels.items(),
            key=lambda x: x[1].buy_volume + x[1].sell_volume,
            reverse=True,
        )

        accumulated = Decimal("0")
        included_prices = []

        for price, level in sorted_levels:
            accumulated += level.buy_volume + level.sell_volume
            included_prices.append(price)
            if accumulated >= target_volume:
                break

        if included_prices:
            return max(included_prices), min(included_prices)

        return None, None

    def _round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        return (price / self.tick_size).quantize(Decimal("1")) * self.tick_size


class LargeTradeFinder:
    """Find large trades (whales) in order flow."""

    def __init__(
        self,
        threshold_percentile: Decimal = Decimal("95"),
        min_size: Optional[Decimal] = None,
    ):
        """Initialize large trade finder."""
        self.threshold_percentile = threshold_percentile
        self.min_size = min_size
        self._size_history: list[Decimal] = []

    def add_trade(self, trade: TradeFlow) -> bool:
        """Add trade and return if it's a large trade."""
        self._size_history.append(trade.size)

        # Keep only recent history
        if len(self._size_history) > 1000:
            self._size_history = self._size_history[-1000:]

        return self.is_large_trade(trade)

    def is_large_trade(self, trade: TradeFlow) -> bool:
        """Check if trade is a large trade."""
        if self.min_size and trade.size >= self.min_size:
            return True

        if len(self._size_history) < 10:
            return False

        # Calculate percentile threshold
        sorted_sizes = sorted(self._size_history)
        idx = int(len(sorted_sizes) * float(self.threshold_percentile) / 100)
        threshold = sorted_sizes[min(idx, len(sorted_sizes) - 1)]

        return trade.size >= threshold

    def find_large_trades(self, trades: list[TradeFlow]) -> list[TradeFlow]:
        """Find all large trades in list."""
        return [t for t in trades if self.is_large_trade(t)]

    def reset(self) -> None:
        """Reset size history."""
        self._size_history = []


class OrderFlowAggregator:
    """Aggregate order flow data."""

    def __init__(self, window_minutes: int = 5):
        """Initialize aggregator."""
        self.window_minutes = window_minutes
        self._trades: list[TradeFlow] = []
        self._window_start: Optional[datetime] = None

    def add_trade(self, trade: TradeFlow) -> Optional[list[TradeFlow]]:
        """Add trade and return completed window if any."""
        if self._window_start is None:
            self._window_start = trade.timestamp

        window_end = self._window_start + timedelta(minutes=self.window_minutes)

        if trade.timestamp >= window_end:
            # Window complete
            completed = self._trades.copy()
            self._trades = [trade]
            self._window_start = trade.timestamp
            return completed

        self._trades.append(trade)
        return None

    def get_current_window(self) -> list[TradeFlow]:
        """Get trades in current window."""
        return self._trades.copy()

    def clear(self) -> None:
        """Clear aggregator."""
        self._trades = []
        self._window_start = None


class PressureAnalyzer:
    """Analyze market pressure from order flow."""

    def __init__(
        self,
        strong_threshold: Decimal = Decimal("70"),
        weak_threshold: Decimal = Decimal("55"),
    ):
        """Initialize pressure analyzer."""
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

    def analyze(
        self,
        buy_volume: Decimal,
        sell_volume: Decimal,
    ) -> MarketPressure:
        """Analyze market pressure."""
        total = buy_volume + sell_volume
        if total == 0:
            return MarketPressure.NEUTRAL

        buy_pct = buy_volume / total * 100
        sell_pct = sell_volume / total * 100

        if buy_pct >= self.strong_threshold:
            return MarketPressure.STRONG_BUY
        elif buy_pct >= self.weak_threshold:
            return MarketPressure.BUY
        elif sell_pct >= self.strong_threshold:
            return MarketPressure.STRONG_SELL
        elif sell_pct >= self.weak_threshold:
            return MarketPressure.SELL

        return MarketPressure.NEUTRAL

    def analyze_delta(self, cvd: Decimal, window_delta: Decimal) -> MarketPressure:
        """Analyze pressure from delta."""
        if window_delta > Decimal("0") and cvd > Decimal("0"):
            if window_delta / (abs(cvd) + 1) > Decimal("0.2"):
                return MarketPressure.STRONG_BUY
            return MarketPressure.BUY
        elif window_delta < Decimal("0") and cvd < Decimal("0"):
            if abs(window_delta) / (abs(cvd) + 1) > Decimal("0.2"):
                return MarketPressure.STRONG_SELL
            return MarketPressure.SELL

        return MarketPressure.NEUTRAL


class OrderFlowAnalyzer:
    """Main order flow analysis engine."""

    def __init__(
        self,
        tick_size: Decimal = Decimal("1"),
        large_trade_min: Optional[Decimal] = None,
    ):
        """Initialize order flow analyzer."""
        self.tick_size = tick_size
        self._delta_calc = DeltaCalculator()
        self._footprint_builder = FootprintBuilder(tick_size)
        self._volume_profile_builder = VolumeProfileBuilder(tick_size)
        self._large_trade_finder = LargeTradeFinder(min_size=large_trade_min)
        self._pressure_analyzer = PressureAnalyzer()
        self._trades: dict[str, list[TradeFlow]] = {}
        self._metrics: dict[str, OrderFlowMetrics] = {}
        self._callbacks: list[Callable] = []

    def add_trade(self, symbol: str, trade: TradeFlow) -> None:
        """Add trade to analyzer."""
        if symbol not in self._trades:
            self._trades[symbol] = []

        self._trades[symbol].append(trade)

        # Keep limited history
        if len(self._trades[symbol]) > 10000:
            self._trades[symbol] = self._trades[symbol][-10000:]

        # Check for large trade
        if self._large_trade_finder.add_trade(trade):
            self._notify("large_trade", {"symbol": symbol, "trade": trade})

        # Update metrics
        self._update_metrics(symbol)

    def _update_metrics(self, symbol: str) -> None:
        """Update metrics for symbol."""
        trades = self._trades.get(symbol, [])

        # Get recent trades (last 100)
        recent = trades[-100:] if len(trades) > 100 else trades

        buy_trades = [t for t in recent if t.side == "buy"]
        sell_trades = [t for t in recent if t.side == "sell"]

        buy_volume = sum(t.size for t in buy_trades)
        sell_volume = sum(t.size for t in sell_trades)
        total_volume = buy_volume + sell_volume

        cvd = buy_volume - sell_volume

        avg_buy = buy_volume / len(buy_trades) if buy_trades else Decimal("0")
        avg_sell = sell_volume / len(sell_trades) if sell_trades else Decimal("0")

        large_buys = len(self._large_trade_finder.find_large_trades(buy_trades))
        large_sells = len(self._large_trade_finder.find_large_trades(sell_trades))

        liq_volume = sum(t.size for t in recent if t.is_liquidation)

        pressure = self._pressure_analyzer.analyze(buy_volume, sell_volume)

        if buy_volume > sell_volume * Decimal("1.2"):
            imbalance = ImbalanceType.BUYING
        elif sell_volume > buy_volume * Decimal("1.2"):
            imbalance = ImbalanceType.SELLING
        else:
            imbalance = ImbalanceType.BALANCED

        self._metrics[symbol] = OrderFlowMetrics(
            symbol=symbol,
            cvd=cvd,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            buy_trades=len(buy_trades),
            sell_trades=len(sell_trades),
            avg_buy_size=avg_buy,
            avg_sell_size=avg_sell,
            large_buy_count=large_buys,
            large_sell_count=large_sells,
            liquidation_volume=liq_volume,
            pressure=pressure,
            imbalance=imbalance,
        )

    def get_metrics(self, symbol: str) -> Optional[OrderFlowMetrics]:
        """Get metrics for symbol."""
        return self._metrics.get(symbol)

    def get_cvd(self, symbol: str) -> Decimal:
        """Get CVD for symbol."""
        metrics = self._metrics.get(symbol)
        return metrics.cvd if metrics else Decimal("0")

    def get_delta_series(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[VolumeDelta]:
        """Get delta series for symbol."""
        trades = self._trades.get(symbol, [])[-limit:]
        calc = DeltaCalculator()
        return calc.calculate_for_period(trades)

    def build_footprint(
        self,
        symbol: str,
        timestamp: datetime,
        minutes: int = 5,
    ) -> FootprintBar:
        """Build footprint bar."""
        trades = self._trades.get(symbol, [])
        cutoff = timestamp - timedelta(minutes=minutes)
        period_trades = [t for t in trades if cutoff <= t.timestamp <= timestamp]
        return self._footprint_builder.build_bar(period_trades, timestamp)

    def build_volume_profile(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> VolumeProfile:
        """Build volume profile."""
        trades = self._trades.get(symbol, [])
        period_trades = [
            t for t in trades
            if start_time <= t.timestamp <= end_time
        ]
        return self._volume_profile_builder.build(period_trades, start_time, end_time)

    def get_large_trades(
        self,
        symbol: str,
        limit: int = 20,
    ) -> list[TradeFlow]:
        """Get large trades for symbol."""
        trades = self._trades.get(symbol, [])
        large = self._large_trade_finder.find_large_trades(trades)
        return sorted(large, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_pressure(self, symbol: str) -> MarketPressure:
        """Get market pressure for symbol."""
        metrics = self._metrics.get(symbol)
        return metrics.pressure if metrics else MarketPressure.NEUTRAL

    def get_imbalance(self, symbol: str) -> ImbalanceType:
        """Get order imbalance for symbol."""
        metrics = self._metrics.get(symbol)
        return metrics.imbalance if metrics else ImbalanceType.BALANCED

    def add_callback(self, callback: Callable) -> None:
        """Add event callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> bool:
        """Remove event callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def _notify(self, event: str, data: Any) -> None:
        """Notify callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear data."""
        if symbol:
            self._trades.pop(symbol, None)
            self._metrics.pop(symbol, None)
        else:
            self._trades.clear()
            self._metrics.clear()
            self._delta_calc.reset()
            self._large_trade_finder.reset()

    def get_summary(self, symbol: Optional[str] = None) -> dict:
        """Get analysis summary."""
        if symbol:
            metrics = self._metrics.get(symbol)
            if metrics:
                return {
                    "symbol": symbol,
                    "trade_count": len(self._trades.get(symbol, [])),
                    "metrics": metrics.to_dict(),
                }
            return {}

        return {
            "symbols": list(self._trades.keys()),
            "total_symbols": len(self._trades),
            "summaries": {
                s: {
                    "trade_count": len(t),
                    "pressure": self._metrics[s].pressure.value if s in self._metrics else "unknown",
                }
                for s, t in self._trades.items()
            },
        }


# Global instance
_order_flow_analyzer: Optional[OrderFlowAnalyzer] = None


def get_order_flow_analyzer() -> OrderFlowAnalyzer:
    """Get global order flow analyzer instance."""
    global _order_flow_analyzer
    if _order_flow_analyzer is None:
        _order_flow_analyzer = OrderFlowAnalyzer()
    return _order_flow_analyzer


def set_order_flow_analyzer(analyzer: OrderFlowAnalyzer) -> None:
    """Set global order flow analyzer instance."""
    global _order_flow_analyzer
    _order_flow_analyzer = analyzer
