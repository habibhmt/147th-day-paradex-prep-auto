"""
Scalping Strategy Module

High-frequency scalping strategy for quick profits from small price movements.
Includes order flow scalping, spread scalping, and momentum scalping.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import time


class ScalpingType(Enum):
    """Types of scalping strategies."""
    ORDER_FLOW = "order_flow"  # Trade based on order flow imbalance
    SPREAD = "spread"  # Capture bid-ask spread
    MOMENTUM = "momentum"  # Quick momentum trades
    BREAKOUT = "breakout"  # Micro breakout scalping
    RANGE = "range"  # Range-bound scalping
    TICK = "tick"  # Tick-by-tick scalping


class ScalpDirection(Enum):
    """Direction of scalp trade."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ScalpSignalStrength(Enum):
    """Strength of scalping signal."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TradeUrgency(Enum):
    """Urgency level for scalp entry."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class ScalpSignal:
    """Scalping signal with entry/exit parameters."""
    timestamp: datetime
    signal_type: ScalpingType
    direction: ScalpDirection
    strength: ScalpSignalStrength
    urgency: TradeUrgency
    entry_price: Decimal
    target_price: Decimal
    stop_loss: Decimal
    expected_profit_ticks: int
    confidence: float
    volume_confirmation: bool
    spread_favorable: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert signal to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "urgency": self.urgency.value,
            "entry_price": str(self.entry_price),
            "target_price": str(self.target_price),
            "stop_loss": str(self.stop_loss),
            "expected_profit_ticks": self.expected_profit_ticks,
            "confidence": self.confidence,
            "volume_confirmation": self.volume_confirmation,
            "spread_favorable": self.spread_favorable,
            "metadata": self.metadata,
        }


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy."""
    scalping_type: ScalpingType = ScalpingType.ORDER_FLOW
    tick_size: Decimal = Decimal("0.01")
    min_profit_ticks: int = 2
    max_loss_ticks: int = 3
    max_hold_seconds: int = 60
    min_volume_ratio: float = 1.5
    max_spread_ticks: int = 2
    position_size: Decimal = Decimal("0.1")
    max_positions: int = 3
    cooldown_seconds: int = 5
    use_trailing_stop: bool = True
    trailing_stop_ticks: int = 1

    # Order flow parameters
    flow_imbalance_threshold: float = 0.6
    flow_window_seconds: int = 10

    # Momentum parameters
    momentum_threshold: float = 0.0002
    momentum_window: int = 5

    # Breakout parameters
    breakout_threshold: float = 0.0005
    consolidation_window: int = 20

    def __post_init__(self):
        """Validate configuration."""
        if self.min_profit_ticks < 1:
            raise ValueError("min_profit_ticks must be >= 1")
        if self.max_loss_ticks < 1:
            raise ValueError("max_loss_ticks must be >= 1")
        if self.max_hold_seconds < 1:
            raise ValueError("max_hold_seconds must be >= 1")
        if self.position_size <= 0:
            raise ValueError("position_size must be positive")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "scalping_type": self.scalping_type.value,
            "tick_size": str(self.tick_size),
            "min_profit_ticks": self.min_profit_ticks,
            "max_loss_ticks": self.max_loss_ticks,
            "max_hold_seconds": self.max_hold_seconds,
            "min_volume_ratio": self.min_volume_ratio,
            "max_spread_ticks": self.max_spread_ticks,
            "position_size": str(self.position_size),
            "max_positions": self.max_positions,
            "cooldown_seconds": self.cooldown_seconds,
            "use_trailing_stop": self.use_trailing_stop,
            "trailing_stop_ticks": self.trailing_stop_ticks,
            "flow_imbalance_threshold": self.flow_imbalance_threshold,
            "flow_window_seconds": self.flow_window_seconds,
            "momentum_threshold": self.momentum_threshold,
            "momentum_window": self.momentum_window,
            "breakout_threshold": self.breakout_threshold,
            "consolidation_window": self.consolidation_window,
        }


@dataclass
class ScalpPosition:
    """Active scalp position."""
    position_id: str
    symbol: str
    direction: ScalpDirection
    entry_price: Decimal
    entry_time: datetime
    size: Decimal
    target_price: Decimal
    stop_loss: Decimal
    trailing_stop: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "size": str(self.size),
            "target_price": str(self.target_price),
            "stop_loss": str(self.stop_loss),
            "trailing_stop": str(self.trailing_stop) if self.trailing_stop else None,
            "realized_pnl": str(self.realized_pnl),
        }


@dataclass
class ScalpMetrics:
    """Scalping performance metrics."""
    total_scalps: int = 0
    winning_scalps: int = 0
    losing_scalps: int = 0
    total_profit: Decimal = Decimal("0")
    total_loss: Decimal = Decimal("0")
    avg_profit_ticks: float = 0.0
    avg_loss_ticks: float = 0.0
    avg_hold_time_seconds: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    scalps_per_hour: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_scalps": self.total_scalps,
            "winning_scalps": self.winning_scalps,
            "losing_scalps": self.losing_scalps,
            "total_profit": str(self.total_profit),
            "total_loss": str(self.total_loss),
            "avg_profit_ticks": self.avg_profit_ticks,
            "avg_loss_ticks": self.avg_loss_ticks,
            "avg_hold_time_seconds": self.avg_hold_time_seconds,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "scalps_per_hour": self.scalps_per_hour,
        }

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_scalps == 0:
            return 0.0
        return self.winning_scalps / self.total_scalps

    @property
    def net_profit(self) -> Decimal:
        """Calculate net profit."""
        return self.total_profit - self.total_loss


@dataclass
class OrderFlowData:
    """Order flow data point."""
    timestamp: datetime
    bid_volume: Decimal
    ask_volume: Decimal
    trade_volume: Decimal
    trade_side: ScalpDirection
    price: Decimal

    @property
    def imbalance(self) -> float:
        """Calculate order flow imbalance."""
        total = float(self.bid_volume + self.ask_volume)
        if total == 0:
            return 0.0
        return float(self.bid_volume - self.ask_volume) / total


@dataclass
class TickData:
    """Single tick data point."""
    timestamp: datetime
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    last_side: ScalpDirection

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


class OrderFlowAnalyzer:
    """Analyzes order flow for scalping signals."""

    def __init__(self, window_seconds: int = 10, imbalance_threshold: float = 0.6):
        """Initialize order flow analyzer."""
        self.window_seconds = window_seconds
        self.imbalance_threshold = imbalance_threshold
        self.flow_data: list[OrderFlowData] = []

    def add_flow(self, flow: OrderFlowData) -> None:
        """Add order flow data point."""
        self.flow_data.append(flow)
        self._cleanup_old_data()

    def _cleanup_old_data(self) -> None:
        """Remove data outside window."""
        if not self.flow_data:
            return
        cutoff = datetime.now().timestamp() - self.window_seconds
        self.flow_data = [
            f for f in self.flow_data
            if f.timestamp.timestamp() > cutoff
        ]

    def get_imbalance(self) -> float:
        """Calculate current order flow imbalance."""
        if not self.flow_data:
            return 0.0
        total_bid = sum(float(f.bid_volume) for f in self.flow_data)
        total_ask = sum(float(f.ask_volume) for f in self.flow_data)
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total

    def get_direction(self) -> ScalpDirection:
        """Get suggested trade direction from order flow."""
        imbalance = self.get_imbalance()
        if imbalance > self.imbalance_threshold:
            return ScalpDirection.LONG
        elif imbalance < -self.imbalance_threshold:
            return ScalpDirection.SHORT
        return ScalpDirection.NEUTRAL

    def get_strength(self) -> ScalpSignalStrength:
        """Get signal strength from imbalance."""
        imbalance = abs(self.get_imbalance())
        if imbalance >= 0.9:
            return ScalpSignalStrength.VERY_STRONG
        elif imbalance >= 0.8:
            return ScalpSignalStrength.STRONG
        elif imbalance >= 0.7:
            return ScalpSignalStrength.MODERATE
        return ScalpSignalStrength.WEAK

    def get_volume_ratio(self) -> float:
        """Calculate volume acceleration ratio."""
        if len(self.flow_data) < 2:
            return 1.0
        recent_volume = sum(float(f.trade_volume) for f in self.flow_data[-5:])
        older_volume = sum(float(f.trade_volume) for f in self.flow_data[:-5]) or 1
        return recent_volume / (older_volume / max(len(self.flow_data) - 5, 1) * 5)


class SpreadAnalyzer:
    """Analyzes spread patterns for scalping."""

    def __init__(self, max_spread_ticks: int = 2, tick_size: Decimal = Decimal("0.01")):
        """Initialize spread analyzer."""
        self.max_spread_ticks = max_spread_ticks
        self.tick_size = tick_size
        self.ticks: list[TickData] = []

    def add_tick(self, tick: TickData) -> None:
        """Add tick data."""
        self.ticks.append(tick)
        if len(self.ticks) > 100:
            self.ticks = self.ticks[-100:]

    def is_spread_favorable(self) -> bool:
        """Check if spread is favorable for scalping."""
        if not self.ticks:
            return False
        current_spread = self.ticks[-1].spread
        return current_spread <= self.tick_size * self.max_spread_ticks

    def get_spread_ticks(self) -> int:
        """Get current spread in ticks."""
        if not self.ticks:
            return 0
        return int(self.ticks[-1].spread / self.tick_size)

    def get_average_spread(self) -> Decimal:
        """Calculate average spread."""
        if not self.ticks:
            return Decimal("0")
        return sum(t.spread for t in self.ticks) / len(self.ticks)

    def is_narrowing(self) -> bool:
        """Check if spread is narrowing."""
        if len(self.ticks) < 10:
            return False
        recent_avg = sum(t.spread for t in self.ticks[-5:]) / 5
        older_avg = sum(t.spread for t in self.ticks[-10:-5]) / 5
        return recent_avg < older_avg


class MomentumScalpAnalyzer:
    """Analyzes micro-momentum for scalping."""

    def __init__(self, threshold: float = 0.0002, window: int = 5):
        """Initialize momentum scalp analyzer."""
        self.threshold = threshold
        self.window = window
        self.prices: list[Decimal] = []
        self.timestamps: list[datetime] = []

    def add_price(self, price: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Add price data point."""
        self.prices.append(price)
        self.timestamps.append(timestamp or datetime.now())
        if len(self.prices) > 50:
            self.prices = self.prices[-50:]
            self.timestamps = self.timestamps[-50:]

    def get_momentum(self) -> float:
        """Calculate current momentum."""
        if len(self.prices) < self.window:
            return 0.0
        recent = self.prices[-self.window:]
        if recent[0] == 0:
            return 0.0
        return float((recent[-1] - recent[0]) / recent[0])

    def get_direction(self) -> ScalpDirection:
        """Get momentum direction."""
        momentum = self.get_momentum()
        if momentum > self.threshold:
            return ScalpDirection.LONG
        elif momentum < -self.threshold:
            return ScalpDirection.SHORT
        return ScalpDirection.NEUTRAL

    def get_acceleration(self) -> float:
        """Calculate momentum acceleration."""
        if len(self.prices) < self.window * 2:
            return 0.0
        recent_momentum = self._calc_momentum(self.prices[-self.window:])
        older_momentum = self._calc_momentum(self.prices[-self.window*2:-self.window])
        return recent_momentum - older_momentum

    def _calc_momentum(self, prices: list[Decimal]) -> float:
        """Calculate momentum for price slice."""
        if not prices or prices[0] == 0:
            return 0.0
        return float((prices[-1] - prices[0]) / prices[0])

    def is_accelerating(self) -> bool:
        """Check if momentum is accelerating."""
        return abs(self.get_acceleration()) > self.threshold / 2


class BreakoutScalpAnalyzer:
    """Analyzes micro-breakouts for scalping."""

    def __init__(self, threshold: float = 0.0005, consolidation_window: int = 20):
        """Initialize breakout analyzer."""
        self.threshold = threshold
        self.consolidation_window = consolidation_window
        self.prices: list[Decimal] = []
        self.highs: list[Decimal] = []
        self.lows: list[Decimal] = []

    def add_price(self, price: Decimal, high: Optional[Decimal] = None,
                  low: Optional[Decimal] = None) -> None:
        """Add price data."""
        self.prices.append(price)
        self.highs.append(high or price)
        self.lows.append(low or price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]
            self.highs = self.highs[-100:]
            self.lows = self.lows[-100:]

    def get_range(self) -> tuple[Decimal, Decimal]:
        """Get consolidation range."""
        if len(self.highs) < self.consolidation_window:
            return Decimal("0"), Decimal("0")
        recent_highs = self.highs[-self.consolidation_window:]
        recent_lows = self.lows[-self.consolidation_window:]
        return min(recent_lows), max(recent_highs)

    def is_consolidating(self) -> bool:
        """Check if price is in consolidation."""
        if len(self.prices) < self.consolidation_window:
            return False
        low, high = self.get_range()
        if low == 0:
            return False
        range_pct = float((high - low) / low)
        return range_pct < self.threshold * 2

    def detect_breakout(self) -> ScalpDirection:
        """Detect breakout direction."""
        if not self.prices or len(self.prices) < self.consolidation_window + 1:
            return ScalpDirection.NEUTRAL
        low, high = self.get_range()
        current = self.prices[-1]
        if high == 0:
            return ScalpDirection.NEUTRAL
        upper_break = float((current - high) / high)
        lower_break = float((low - current) / low) if low > 0 else 0
        if upper_break > self.threshold:
            return ScalpDirection.LONG
        elif lower_break > self.threshold:
            return ScalpDirection.SHORT
        return ScalpDirection.NEUTRAL

    def get_breakout_strength(self) -> ScalpSignalStrength:
        """Get breakout signal strength."""
        if not self.prices:
            return ScalpSignalStrength.WEAK
        low, high = self.get_range()
        current = self.prices[-1]
        if high == 0 or low == 0:
            return ScalpSignalStrength.WEAK
        if current > high:
            break_pct = float((current - high) / high)
        elif current < low:
            break_pct = float((low - current) / low)
        else:
            return ScalpSignalStrength.WEAK
        if break_pct > self.threshold * 4:
            return ScalpSignalStrength.VERY_STRONG
        elif break_pct > self.threshold * 2:
            return ScalpSignalStrength.STRONG
        elif break_pct > self.threshold:
            return ScalpSignalStrength.MODERATE
        return ScalpSignalStrength.WEAK


class RangeScalpAnalyzer:
    """Analyzes price ranges for range scalping."""

    def __init__(self, lookback: int = 30):
        """Initialize range analyzer."""
        self.lookback = lookback
        self.prices: list[Decimal] = []

    def add_price(self, price: Decimal) -> None:
        """Add price data point."""
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_range_bounds(self) -> tuple[Decimal, Decimal]:
        """Get current range bounds."""
        if len(self.prices) < self.lookback:
            return Decimal("0"), Decimal("0")
        recent = self.prices[-self.lookback:]
        return min(recent), max(recent)

    def get_position_in_range(self) -> float:
        """Get current position in range (0-1)."""
        if not self.prices:
            return 0.5
        low, high = self.get_range_bounds()
        if high == low:
            return 0.5
        return float((self.prices[-1] - low) / (high - low))

    def should_buy(self, threshold: float = 0.2) -> bool:
        """Check if near range bottom."""
        return self.get_position_in_range() < threshold

    def should_sell(self, threshold: float = 0.8) -> bool:
        """Check if near range top."""
        return self.get_position_in_range() > threshold

    def get_direction(self) -> ScalpDirection:
        """Get trade direction based on range position."""
        position = self.get_position_in_range()
        if position < 0.2:
            return ScalpDirection.LONG
        elif position > 0.8:
            return ScalpDirection.SHORT
        return ScalpDirection.NEUTRAL


class ScalpingStrategy:
    """Main scalping strategy implementation."""

    def __init__(self, symbol: str, config: Optional[ScalpingConfig] = None):
        """Initialize scalping strategy."""
        self.symbol = symbol
        self.config = config or ScalpingConfig()
        self.positions: dict[str, ScalpPosition] = {}
        self.metrics = ScalpMetrics()
        self.last_scalp_time: Optional[datetime] = None
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.position_counter = 0

        # Initialize analyzers based on type
        self.order_flow_analyzer = OrderFlowAnalyzer(
            window_seconds=self.config.flow_window_seconds,
            imbalance_threshold=self.config.flow_imbalance_threshold,
        )
        self.spread_analyzer = SpreadAnalyzer(
            max_spread_ticks=self.config.max_spread_ticks,
            tick_size=self.config.tick_size,
        )
        self.momentum_analyzer = MomentumScalpAnalyzer(
            threshold=self.config.momentum_threshold,
            window=self.config.momentum_window,
        )
        self.breakout_analyzer = BreakoutScalpAnalyzer(
            threshold=self.config.breakout_threshold,
            consolidation_window=self.config.consolidation_window,
        )
        self.range_analyzer = RangeScalpAnalyzer(lookback=30)

        # Trade history
        self.trade_history: list[dict] = []
        self.start_time = datetime.now()

    def on_tick(self, tick: TickData) -> Optional[ScalpSignal]:
        """Process tick data and generate signal if conditions met."""
        # Update analyzers
        self.spread_analyzer.add_tick(tick)
        self.momentum_analyzer.add_price(tick.price, tick.timestamp)
        self.breakout_analyzer.add_price(tick.price)
        self.range_analyzer.add_price(tick.price)

        # Check if spread is favorable
        if not self.spread_analyzer.is_spread_favorable():
            return None

        # Check cooldown
        if not self._check_cooldown():
            return None

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return None

        # Generate signal based on strategy type
        return self._generate_signal(tick)

    def on_order_flow(self, flow: OrderFlowData) -> None:
        """Process order flow data."""
        self.order_flow_analyzer.add_flow(flow)

    def _generate_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate scalping signal based on strategy type."""
        if self.config.scalping_type == ScalpingType.ORDER_FLOW:
            return self._generate_order_flow_signal(tick)
        elif self.config.scalping_type == ScalpingType.SPREAD:
            return self._generate_spread_signal(tick)
        elif self.config.scalping_type == ScalpingType.MOMENTUM:
            return self._generate_momentum_signal(tick)
        elif self.config.scalping_type == ScalpingType.BREAKOUT:
            return self._generate_breakout_signal(tick)
        elif self.config.scalping_type == ScalpingType.RANGE:
            return self._generate_range_signal(tick)
        elif self.config.scalping_type == ScalpingType.TICK:
            return self._generate_tick_signal(tick)
        return None

    def _generate_order_flow_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from order flow analysis."""
        direction = self.order_flow_analyzer.get_direction()
        if direction == ScalpDirection.NEUTRAL:
            return None
        strength = self.order_flow_analyzer.get_strength()
        if strength == ScalpSignalStrength.WEAK:
            return None
        volume_ratio = self.order_flow_analyzer.get_volume_ratio()
        volume_confirmation = volume_ratio >= self.config.min_volume_ratio
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.ORDER_FLOW,
            direction=direction,
            strength=strength,
            urgency=self._get_urgency(strength),
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=self.config.min_profit_ticks,
            confidence=self._calculate_confidence(strength, volume_confirmation),
            volume_confirmation=volume_confirmation,
            spread_favorable=True,
            metadata={"imbalance": self.order_flow_analyzer.get_imbalance()},
        )

    def _generate_spread_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from spread analysis."""
        if not self.spread_analyzer.is_narrowing():
            return None
        # Spread scalping aims to capture the spread
        direction = tick.last_side
        if direction == ScalpDirection.NEUTRAL:
            return None
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.SPREAD,
            direction=direction,
            strength=ScalpSignalStrength.MODERATE,
            urgency=TradeUrgency.HIGH,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=1,
            confidence=0.6,
            volume_confirmation=True,
            spread_favorable=True,
            metadata={"spread": str(tick.spread)},
        )

    def _generate_momentum_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from momentum analysis."""
        direction = self.momentum_analyzer.get_direction()
        if direction == ScalpDirection.NEUTRAL:
            return None
        if not self.momentum_analyzer.is_accelerating():
            return None
        momentum = self.momentum_analyzer.get_momentum()
        strength = self._momentum_to_strength(abs(momentum))
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.MOMENTUM,
            direction=direction,
            strength=strength,
            urgency=TradeUrgency.HIGH,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=self.config.min_profit_ticks,
            confidence=0.7,
            volume_confirmation=True,
            spread_favorable=True,
            metadata={"momentum": momentum},
        )

    def _generate_breakout_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from breakout analysis."""
        direction = self.breakout_analyzer.detect_breakout()
        if direction == ScalpDirection.NEUTRAL:
            return None
        strength = self.breakout_analyzer.get_breakout_strength()
        if strength == ScalpSignalStrength.WEAK:
            return None
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.BREAKOUT,
            direction=direction,
            strength=strength,
            urgency=TradeUrgency.IMMEDIATE,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=self.config.min_profit_ticks * 2,
            confidence=0.75,
            volume_confirmation=True,
            spread_favorable=True,
            metadata={"range": self.breakout_analyzer.get_range()},
        )

    def _generate_range_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from range analysis."""
        direction = self.range_analyzer.get_direction()
        if direction == ScalpDirection.NEUTRAL:
            return None
        position_in_range = self.range_analyzer.get_position_in_range()
        strength = self._range_position_to_strength(position_in_range, direction)
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.RANGE,
            direction=direction,
            strength=strength,
            urgency=TradeUrgency.MEDIUM,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=self.config.min_profit_ticks,
            confidence=0.65,
            volume_confirmation=True,
            spread_favorable=True,
            metadata={"range_position": position_in_range},
        )

    def _generate_tick_signal(self, tick: TickData) -> Optional[ScalpSignal]:
        """Generate signal from tick analysis."""
        # Simple tick scalping based on last trade side
        direction = tick.last_side
        if direction == ScalpDirection.NEUTRAL:
            return None
        entry_price = tick.ask if direction == ScalpDirection.LONG else tick.bid
        target_price, stop_loss = self._calculate_targets(entry_price, direction)
        return ScalpSignal(
            timestamp=tick.timestamp,
            signal_type=ScalpingType.TICK,
            direction=direction,
            strength=ScalpSignalStrength.MODERATE,
            urgency=TradeUrgency.IMMEDIATE,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_ticks=1,
            confidence=0.55,
            volume_confirmation=float(tick.volume) > 0,
            spread_favorable=True,
            metadata={"tick_volume": str(tick.volume)},
        )

    def _calculate_targets(self, entry_price: Decimal,
                           direction: ScalpDirection) -> tuple[Decimal, Decimal]:
        """Calculate target price and stop loss."""
        profit_distance = self.config.tick_size * self.config.min_profit_ticks
        loss_distance = self.config.tick_size * self.config.max_loss_ticks
        if direction == ScalpDirection.LONG:
            target = entry_price + profit_distance
            stop = entry_price - loss_distance
        else:
            target = entry_price - profit_distance
            stop = entry_price + loss_distance
        return target, stop

    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        if self.last_scalp_time is None:
            return True
        elapsed = (datetime.now() - self.last_scalp_time).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def _get_urgency(self, strength: ScalpSignalStrength) -> TradeUrgency:
        """Convert signal strength to urgency."""
        if strength == ScalpSignalStrength.VERY_STRONG:
            return TradeUrgency.IMMEDIATE
        elif strength == ScalpSignalStrength.STRONG:
            return TradeUrgency.HIGH
        elif strength == ScalpSignalStrength.MODERATE:
            return TradeUrgency.MEDIUM
        return TradeUrgency.LOW

    def _momentum_to_strength(self, momentum: float) -> ScalpSignalStrength:
        """Convert momentum value to signal strength."""
        if momentum > self.config.momentum_threshold * 4:
            return ScalpSignalStrength.VERY_STRONG
        elif momentum > self.config.momentum_threshold * 2:
            return ScalpSignalStrength.STRONG
        elif momentum > self.config.momentum_threshold:
            return ScalpSignalStrength.MODERATE
        return ScalpSignalStrength.WEAK

    def _range_position_to_strength(self, position: float,
                                     direction: ScalpDirection) -> ScalpSignalStrength:
        """Convert range position to signal strength."""
        if direction == ScalpDirection.LONG:
            if position < 0.1:
                return ScalpSignalStrength.VERY_STRONG
            elif position < 0.15:
                return ScalpSignalStrength.STRONG
            elif position < 0.2:
                return ScalpSignalStrength.MODERATE
        else:
            if position > 0.9:
                return ScalpSignalStrength.VERY_STRONG
            elif position > 0.85:
                return ScalpSignalStrength.STRONG
            elif position > 0.8:
                return ScalpSignalStrength.MODERATE
        return ScalpSignalStrength.WEAK

    def _calculate_confidence(self, strength: ScalpSignalStrength,
                               volume_confirmation: bool) -> float:
        """Calculate signal confidence."""
        base_confidence = {
            ScalpSignalStrength.WEAK: 0.4,
            ScalpSignalStrength.MODERATE: 0.55,
            ScalpSignalStrength.STRONG: 0.7,
            ScalpSignalStrength.VERY_STRONG: 0.85,
        }[strength]
        if volume_confirmation:
            base_confidence += 0.1
        return min(base_confidence, 0.95)

    def enter_position(self, signal: ScalpSignal) -> ScalpPosition:
        """Enter a new scalp position."""
        self.position_counter += 1
        position_id = f"scalp_{self.position_counter}_{int(time.time())}"
        position = ScalpPosition(
            position_id=position_id,
            symbol=self.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            entry_time=signal.timestamp,
            size=self.config.position_size,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
        )
        if self.config.use_trailing_stop:
            position.trailing_stop = signal.stop_loss
        self.positions[position_id] = position
        self.last_scalp_time = signal.timestamp
        return position

    def exit_position(self, position_id: str, exit_price: Decimal,
                      exit_time: Optional[datetime] = None) -> dict:
        """Exit a scalp position."""
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        position = self.positions.pop(position_id)
        exit_time = exit_time or datetime.now()
        # Calculate PnL
        if position.direction == ScalpDirection.LONG:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        # Update metrics
        self.metrics.total_scalps += 1
        hold_time = (exit_time - position.entry_time).total_seconds()
        if pnl > 0:
            self.metrics.winning_scalps += 1
            self.metrics.total_profit += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if self.consecutive_wins > self.metrics.max_consecutive_wins:
                self.metrics.max_consecutive_wins = self.consecutive_wins
        else:
            self.metrics.losing_scalps += 1
            self.metrics.total_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if self.consecutive_losses > self.metrics.max_consecutive_losses:
                self.metrics.max_consecutive_losses = self.consecutive_losses
        # Update average metrics
        self._update_average_metrics(pnl, hold_time)
        # Record trade
        trade_record = {
            "position_id": position_id,
            "symbol": self.symbol,
            "direction": position.direction.value,
            "entry_price": str(position.entry_price),
            "exit_price": str(exit_price),
            "entry_time": position.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "hold_time_seconds": hold_time,
            "pnl": str(pnl),
            "size": str(position.size),
        }
        self.trade_history.append(trade_record)
        return trade_record

    def _update_average_metrics(self, pnl: Decimal, hold_time: float) -> None:
        """Update rolling average metrics."""
        n = self.metrics.total_scalps
        # Update average hold time
        self.metrics.avg_hold_time_seconds = (
            (self.metrics.avg_hold_time_seconds * (n - 1) + hold_time) / n
        )
        # Update average profit/loss ticks
        ticks = int(abs(pnl) / self.config.tick_size / self.config.position_size)
        if pnl > 0:
            wins = self.metrics.winning_scalps
            self.metrics.avg_profit_ticks = (
                (self.metrics.avg_profit_ticks * (wins - 1) + ticks) / wins
            )
        else:
            losses = self.metrics.losing_scalps
            if losses > 0:
                self.metrics.avg_loss_ticks = (
                    (self.metrics.avg_loss_ticks * (losses - 1) + ticks) / losses
                )
        # Update profit factor
        if self.metrics.total_loss > 0:
            self.metrics.profit_factor = float(
                self.metrics.total_profit / self.metrics.total_loss
            )
        # Update scalps per hour
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if elapsed_hours > 0:
            self.metrics.scalps_per_hour = n / elapsed_hours

    def check_position(self, position_id: str, current_price: Decimal) -> str:
        """Check position status and return action."""
        if position_id not in self.positions:
            return "not_found"
        position = self.positions[position_id]
        # Check time limit
        elapsed = (datetime.now() - position.entry_time).total_seconds()
        if elapsed >= self.config.max_hold_seconds:
            return "timeout"
        # Check targets
        if position.direction == ScalpDirection.LONG:
            if current_price >= position.target_price:
                return "target_hit"
            if current_price <= position.stop_loss:
                return "stop_hit"
            # Update trailing stop
            if self.config.use_trailing_stop and position.trailing_stop:
                new_stop = current_price - self.config.tick_size * self.config.trailing_stop_ticks
                if new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
                if current_price <= position.trailing_stop:
                    return "trailing_stop_hit"
        else:
            if current_price <= position.target_price:
                return "target_hit"
            if current_price >= position.stop_loss:
                return "stop_hit"
            # Update trailing stop
            if self.config.use_trailing_stop and position.trailing_stop:
                new_stop = current_price + self.config.tick_size * self.config.trailing_stop_ticks
                if new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop
                if current_price >= position.trailing_stop:
                    return "trailing_stop_hit"
        return "hold"

    def get_status(self) -> dict:
        """Get strategy status."""
        return {
            "symbol": self.symbol,
            "scalping_type": self.config.scalping_type.value,
            "active_positions": len(self.positions),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "metrics": self.metrics.to_dict(),
            "win_rate": self.metrics.win_rate,
            "net_profit": str(self.metrics.net_profit),
            "config": self.config.to_dict(),
        }
