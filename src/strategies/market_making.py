"""
Market Making Strategy Module

Professional market making with inventory management, spread optimization,
and risk controls. Supports multiple quoting strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import time
import math


class MarketMakingType(Enum):
    """Types of market making strategies."""
    BASIC = "basic"  # Simple two-sided quotes
    INVENTORY = "inventory"  # Inventory-aware quoting
    AVELLANEDA_STOIKOV = "avellaneda_stoikov"  # Academic optimal market making
    ADAPTIVE = "adaptive"  # Volatility-adaptive spreads
    INFORMATION = "information"  # Information-based market making


class QuoteStatus(Enum):
    """Status of a quote."""
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIAL = "partial"


class InventoryState(Enum):
    """Inventory state classification."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    EXTREME_LONG = "extreme_long"
    EXTREME_SHORT = "extreme_short"


class SpreadState(Enum):
    """Spread condition state."""
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    VERY_WIDE = "very_wide"


@dataclass
class Quote:
    """A market making quote."""
    quote_id: str
    symbol: str
    side: str  # "bid" or "ask"
    price: Decimal
    size: Decimal
    timestamp: datetime
    status: QuoteStatus = QuoteStatus.ACTIVE
    filled_size: Decimal = Decimal("0")
    expiry: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert quote to dictionary."""
        return {
            "quote_id": self.quote_id,
            "symbol": self.symbol,
            "side": self.side,
            "price": str(self.price),
            "size": str(self.size),
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "filled_size": str(self.filled_size),
            "expiry": self.expiry.isoformat() if self.expiry else None,
        }

    @property
    def remaining_size(self) -> Decimal:
        """Get remaining unfilled size."""
        return self.size - self.filled_size


@dataclass
class QuotePair:
    """A pair of bid and ask quotes."""
    bid: Quote
    ask: Quote
    spread: Decimal
    mid_price: Decimal
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert quote pair to dictionary."""
        return {
            "bid": self.bid.to_dict(),
            "ask": self.ask.to_dict(),
            "spread": str(self.spread),
            "mid_price": str(self.mid_price),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MarketMakingConfig:
    """Configuration for market making strategy."""
    strategy_type: MarketMakingType = MarketMakingType.INVENTORY
    base_spread: Decimal = Decimal("0.001")  # 0.1%
    min_spread: Decimal = Decimal("0.0002")  # 0.02%
    max_spread: Decimal = Decimal("0.01")  # 1%
    quote_size: Decimal = Decimal("0.1")
    max_position: Decimal = Decimal("10.0")
    inventory_skew: float = 0.5  # Inventory adjustment factor
    volatility_multiplier: float = 2.0
    quote_refresh_seconds: int = 5
    quote_expiry_seconds: int = 30
    max_quotes_per_side: int = 5
    gamma: float = 0.1  # Risk aversion for A-S model
    kappa: float = 1.5  # Order arrival intensity
    eta: float = 0.01  # Fill probability decay

    def __post_init__(self):
        """Validate configuration."""
        if self.base_spread <= 0:
            raise ValueError("base_spread must be positive")
        if self.min_spread <= 0:
            raise ValueError("min_spread must be positive")
        if self.min_spread > self.base_spread:
            raise ValueError("min_spread cannot exceed base_spread")
        if self.max_spread < self.base_spread:
            raise ValueError("max_spread cannot be less than base_spread")
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "strategy_type": self.strategy_type.value,
            "base_spread": str(self.base_spread),
            "min_spread": str(self.min_spread),
            "max_spread": str(self.max_spread),
            "quote_size": str(self.quote_size),
            "max_position": str(self.max_position),
            "inventory_skew": self.inventory_skew,
            "volatility_multiplier": self.volatility_multiplier,
            "quote_refresh_seconds": self.quote_refresh_seconds,
            "quote_expiry_seconds": self.quote_expiry_seconds,
            "max_quotes_per_side": self.max_quotes_per_side,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "eta": self.eta,
        }


@dataclass
class InventoryMetrics:
    """Inventory tracking metrics."""
    position: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_buys: int = 0
    total_sells: int = 0
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "position": str(self.position),
            "avg_entry_price": str(self.avg_entry_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "total_buys": self.total_buys,
            "total_sells": self.total_sells,
            "buy_volume": str(self.buy_volume),
            "sell_volume": str(self.sell_volume),
        }

    @property
    def net_volume(self) -> Decimal:
        """Calculate net volume."""
        return self.buy_volume - self.sell_volume

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL."""
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class MarketMakingMetrics:
    """Overall market making performance metrics."""
    total_quotes: int = 0
    filled_quotes: int = 0
    cancelled_quotes: int = 0
    total_volume: Decimal = Decimal("0")
    spread_captured: Decimal = Decimal("0")
    avg_spread: float = 0.0
    fill_rate: float = 0.0
    profit_per_volume: float = 0.0
    time_in_market_pct: float = 0.0
    inventory_turnover: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_quotes": self.total_quotes,
            "filled_quotes": self.filled_quotes,
            "cancelled_quotes": self.cancelled_quotes,
            "total_volume": str(self.total_volume),
            "spread_captured": str(self.spread_captured),
            "avg_spread": self.avg_spread,
            "fill_rate": self.fill_rate,
            "profit_per_volume": self.profit_per_volume,
            "time_in_market_pct": self.time_in_market_pct,
            "inventory_turnover": self.inventory_turnover,
        }


@dataclass
class MarketData:
    """Market data snapshot."""
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume_24h: Decimal = Decimal("0")
    volatility: float = 0.0

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage."""
        if self.mid_price == 0:
            return 0.0
        return float(self.spread / self.mid_price)


class SpreadCalculator:
    """Calculates optimal spreads based on various factors."""

    def __init__(self, config: MarketMakingConfig):
        """Initialize spread calculator."""
        self.config = config

    def calculate_base_spread(self, market: MarketData) -> Decimal:
        """Calculate base spread."""
        return self.config.base_spread

    def calculate_volatility_spread(self, market: MarketData) -> Decimal:
        """Calculate volatility-adjusted spread."""
        vol_adjustment = Decimal(str(market.volatility * self.config.volatility_multiplier))
        spread = self.config.base_spread + vol_adjustment
        return max(self.config.min_spread, min(spread, self.config.max_spread))

    def calculate_inventory_spread(self, market: MarketData,
                                    inventory: InventoryMetrics) -> tuple[Decimal, Decimal]:
        """Calculate inventory-adjusted bid/ask spreads."""
        base = self.calculate_volatility_spread(market)
        position_ratio = float(inventory.position / self.config.max_position)
        skew = Decimal(str(position_ratio * self.config.inventory_skew))
        # Widen spread on the side we don't want to fill
        bid_spread = base + (skew * base) if skew > 0 else base
        ask_spread = base - (skew * base) if skew < 0 else base
        # Ensure minimum spread
        bid_spread = max(self.config.min_spread, bid_spread)
        ask_spread = max(self.config.min_spread, ask_spread)
        return bid_spread, ask_spread

    def calculate_avellaneda_stoikov(self, market: MarketData,
                                      inventory: InventoryMetrics,
                                      time_horizon: float) -> tuple[Decimal, Decimal]:
        """Calculate A-S optimal market making spreads."""
        s = float(market.mid_price)
        q = float(inventory.position)
        sigma = max(market.volatility, 0.001)  # Prevent division by zero
        gamma = self.config.gamma
        kappa = self.config.kappa
        T = max(time_horizon, 0.001)
        # Reservation price
        r = s - q * gamma * (sigma ** 2) * T
        # Optimal spread
        spread = gamma * (sigma ** 2) * T + (2 / gamma) * math.log(1 + gamma / kappa)
        half_spread = spread / 2
        # Bid and ask around reservation price
        bid_price = Decimal(str(r - half_spread))
        ask_price = Decimal(str(r + half_spread))
        bid_spread = market.mid_price - bid_price
        ask_spread = ask_price - market.mid_price
        # Apply bounds
        bid_spread = max(self.config.min_spread, min(bid_spread, self.config.max_spread))
        ask_spread = max(self.config.min_spread, min(ask_spread, self.config.max_spread))
        return bid_spread, ask_spread


class InventoryManager:
    """Manages inventory for market making."""

    def __init__(self, max_position: Decimal):
        """Initialize inventory manager."""
        self.max_position = max_position
        self.metrics = InventoryMetrics()

    def record_fill(self, side: str, price: Decimal, size: Decimal) -> None:
        """Record a fill and update inventory."""
        if side == "bid":  # We bought
            old_position = self.metrics.position
            old_cost = old_position * self.metrics.avg_entry_price
            new_cost = size * price
            self.metrics.position += size
            if self.metrics.position != 0:
                self.metrics.avg_entry_price = (old_cost + new_cost) / self.metrics.position
            self.metrics.total_buys += 1
            self.metrics.buy_volume += size
        else:  # We sold
            if self.metrics.position > 0:
                pnl = (price - self.metrics.avg_entry_price) * min(size, self.metrics.position)
                self.metrics.realized_pnl += pnl
            self.metrics.position -= size
            self.metrics.total_sells += 1
            self.metrics.sell_volume += size

    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized PnL based on current price."""
        if self.metrics.position != 0:
            self.metrics.unrealized_pnl = (
                (current_price - self.metrics.avg_entry_price) * self.metrics.position
            )

    def get_state(self) -> InventoryState:
        """Get current inventory state."""
        ratio = float(self.metrics.position / self.max_position)
        if ratio > 0.8:
            return InventoryState.EXTREME_LONG
        elif ratio > 0.3:
            return InventoryState.LONG
        elif ratio < -0.8:
            return InventoryState.EXTREME_SHORT
        elif ratio < -0.3:
            return InventoryState.SHORT
        return InventoryState.NEUTRAL

    def can_buy(self, size: Decimal) -> bool:
        """Check if can place buy order."""
        return self.metrics.position + size <= self.max_position

    def can_sell(self, size: Decimal) -> bool:
        """Check if can place sell order."""
        return self.metrics.position - size >= -self.max_position

    def get_suggested_size(self, side: str) -> Decimal:
        """Get suggested order size based on inventory."""
        state = self.get_state()
        base_size = self.max_position / 10  # 10% of max position
        if side == "bid":
            if state == InventoryState.EXTREME_LONG:
                return Decimal("0")
            elif state == InventoryState.LONG:
                return base_size / 2
            elif state == InventoryState.SHORT:
                return base_size * 2
            elif state == InventoryState.EXTREME_SHORT:
                return base_size * 3
        else:  # ask
            if state == InventoryState.EXTREME_SHORT:
                return Decimal("0")
            elif state == InventoryState.SHORT:
                return base_size / 2
            elif state == InventoryState.LONG:
                return base_size * 2
            elif state == InventoryState.EXTREME_LONG:
                return base_size * 3
        return base_size


class QuoteManager:
    """Manages quote lifecycle."""

    def __init__(self, symbol: str):
        """Initialize quote manager."""
        self.symbol = symbol
        self.active_quotes: dict[str, Quote] = {}
        self.quote_history: list[Quote] = []
        self.quote_counter = 0

    def create_quote(self, side: str, price: Decimal, size: Decimal,
                     expiry_seconds: Optional[int] = None) -> Quote:
        """Create a new quote."""
        self.quote_counter += 1
        quote_id = f"q_{self.symbol}_{side}_{self.quote_counter}_{int(time.time())}"
        now = datetime.now()
        expiry = None
        if expiry_seconds:
            expiry = datetime.fromtimestamp(now.timestamp() + expiry_seconds)
        quote = Quote(
            quote_id=quote_id,
            symbol=self.symbol,
            side=side,
            price=price,
            size=size,
            timestamp=now,
            status=QuoteStatus.ACTIVE,
            expiry=expiry,
        )
        self.active_quotes[quote_id] = quote
        return quote

    def fill_quote(self, quote_id: str, fill_size: Decimal) -> Optional[Quote]:
        """Fill a quote (partial or full)."""
        if quote_id not in self.active_quotes:
            return None
        quote = self.active_quotes[quote_id]
        quote.filled_size += fill_size
        if quote.filled_size >= quote.size:
            quote.status = QuoteStatus.FILLED
            del self.active_quotes[quote_id]
            self.quote_history.append(quote)
        else:
            quote.status = QuoteStatus.PARTIAL
        return quote

    def cancel_quote(self, quote_id: str) -> Optional[Quote]:
        """Cancel a quote."""
        if quote_id not in self.active_quotes:
            return None
        quote = self.active_quotes.pop(quote_id)
        quote.status = QuoteStatus.CANCELLED
        self.quote_history.append(quote)
        return quote

    def cancel_all(self, side: Optional[str] = None) -> list[Quote]:
        """Cancel all quotes, optionally filtering by side."""
        cancelled = []
        to_cancel = [
            q for q in self.active_quotes.values()
            if side is None or q.side == side
        ]
        for quote in to_cancel:
            cancelled.append(self.cancel_quote(quote.quote_id))
        return [q for q in cancelled if q is not None]

    def expire_quotes(self) -> list[Quote]:
        """Expire quotes past their expiry time."""
        expired = []
        now = datetime.now()
        to_expire = [
            q for q in self.active_quotes.values()
            if q.expiry and q.expiry < now
        ]
        for quote in to_expire:
            quote = self.active_quotes.pop(quote.quote_id)
            quote.status = QuoteStatus.EXPIRED
            self.quote_history.append(quote)
            expired.append(quote)
        return expired

    def get_active_quotes(self, side: Optional[str] = None) -> list[Quote]:
        """Get active quotes, optionally filtering by side."""
        if side:
            return [q for q in self.active_quotes.values() if q.side == side]
        return list(self.active_quotes.values())

    def get_best_bid(self) -> Optional[Quote]:
        """Get best bid quote."""
        bids = [q for q in self.active_quotes.values() if q.side == "bid"]
        if not bids:
            return None
        return max(bids, key=lambda q: q.price)

    def get_best_ask(self) -> Optional[Quote]:
        """Get best ask quote."""
        asks = [q for q in self.active_quotes.values() if q.side == "ask"]
        if not asks:
            return None
        return min(asks, key=lambda q: q.price)


class MarketMakingStrategy:
    """Main market making strategy implementation."""

    def __init__(self, symbol: str, config: Optional[MarketMakingConfig] = None):
        """Initialize market making strategy."""
        self.symbol = symbol
        self.config = config or MarketMakingConfig()
        self.quote_manager = QuoteManager(symbol)
        self.inventory_manager = InventoryManager(self.config.max_position)
        self.spread_calculator = SpreadCalculator(self.config)
        self.metrics = MarketMakingMetrics()
        self.last_quote_time: Optional[datetime] = None
        self.start_time = datetime.now()
        self.total_time_quoted = 0.0
        self.price_history: list[Decimal] = []

    def on_market_data(self, market: MarketData) -> Optional[QuotePair]:
        """Process market data and update quotes."""
        # Update inventory PnL
        self.inventory_manager.update_unrealized_pnl(market.mid_price)
        # Track price history for volatility
        self.price_history.append(market.mid_price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        # Check if we should refresh quotes
        if not self._should_refresh_quotes():
            return None
        # Expire old quotes
        self.quote_manager.expire_quotes()
        # Generate new quotes
        return self._generate_quotes(market)

    def _should_refresh_quotes(self) -> bool:
        """Check if quotes should be refreshed."""
        if self.last_quote_time is None:
            return True
        elapsed = (datetime.now() - self.last_quote_time).total_seconds()
        return elapsed >= self.config.quote_refresh_seconds

    def _generate_quotes(self, market: MarketData) -> QuotePair:
        """Generate bid and ask quotes."""
        # Cancel existing quotes
        self.quote_manager.cancel_all()
        # Calculate spreads based on strategy type
        if self.config.strategy_type == MarketMakingType.BASIC:
            bid_spread = ask_spread = self.spread_calculator.calculate_base_spread(market)
        elif self.config.strategy_type == MarketMakingType.ADAPTIVE:
            bid_spread = ask_spread = self.spread_calculator.calculate_volatility_spread(market)
        elif self.config.strategy_type == MarketMakingType.INVENTORY:
            bid_spread, ask_spread = self.spread_calculator.calculate_inventory_spread(
                market, self.inventory_manager.metrics
            )
        elif self.config.strategy_type == MarketMakingType.AVELLANEDA_STOIKOV:
            time_horizon = 1.0  # 1 hour horizon
            bid_spread, ask_spread = self.spread_calculator.calculate_avellaneda_stoikov(
                market, self.inventory_manager.metrics, time_horizon
            )
        else:  # INFORMATION
            bid_spread, ask_spread = self.spread_calculator.calculate_inventory_spread(
                market, self.inventory_manager.metrics
            )
        # Calculate prices
        bid_price = market.mid_price - bid_spread
        ask_price = market.mid_price + ask_spread
        # Get sizes based on inventory
        bid_size = self._get_quote_size("bid")
        ask_size = self._get_quote_size("ask")
        # Create quotes
        bid_quote = self.quote_manager.create_quote(
            "bid", bid_price, bid_size, self.config.quote_expiry_seconds
        )
        ask_quote = self.quote_manager.create_quote(
            "ask", ask_price, ask_size, self.config.quote_expiry_seconds
        )
        self.metrics.total_quotes += 2
        self.last_quote_time = datetime.now()
        spread = ask_price - bid_price
        return QuotePair(
            bid=bid_quote,
            ask=ask_quote,
            spread=spread,
            mid_price=market.mid_price,
            timestamp=datetime.now(),
        )

    def _get_quote_size(self, side: str) -> Decimal:
        """Get quote size based on inventory state."""
        suggested = self.inventory_manager.get_suggested_size(side)
        return max(suggested, self.config.quote_size)

    def on_fill(self, quote_id: str, fill_price: Decimal, fill_size: Decimal) -> dict:
        """Process a fill event."""
        quote = self.quote_manager.fill_quote(quote_id, fill_size)
        if not quote:
            return {"error": "Quote not found"}
        # Update inventory
        self.inventory_manager.record_fill(quote.side, fill_price, fill_size)
        # Update metrics
        self.metrics.filled_quotes += 1
        self.metrics.total_volume += fill_size
        if self.metrics.total_quotes > 0:
            self.metrics.fill_rate = self.metrics.filled_quotes / self.metrics.total_quotes
        return {
            "quote_id": quote_id,
            "side": quote.side,
            "fill_price": str(fill_price),
            "fill_size": str(fill_size),
            "remaining_size": str(quote.remaining_size),
            "status": quote.status.value,
        }

    def get_quotes(self) -> dict:
        """Get current quote state."""
        best_bid = self.quote_manager.get_best_bid()
        best_ask = self.quote_manager.get_best_ask()
        return {
            "best_bid": best_bid.to_dict() if best_bid else None,
            "best_ask": best_ask.to_dict() if best_ask else None,
            "spread": str(best_ask.price - best_bid.price) if best_bid and best_ask else None,
            "active_bids": len(self.quote_manager.get_active_quotes("bid")),
            "active_asks": len(self.quote_manager.get_active_quotes("ask")),
        }

    def get_inventory(self) -> dict:
        """Get inventory state."""
        return {
            "metrics": self.inventory_manager.metrics.to_dict(),
            "state": self.inventory_manager.get_state().value,
            "can_buy": self.inventory_manager.can_buy(self.config.quote_size),
            "can_sell": self.inventory_manager.can_sell(self.config.quote_size),
        }

    def get_status(self) -> dict:
        """Get strategy status."""
        return {
            "symbol": self.symbol,
            "strategy_type": self.config.strategy_type.value,
            "quotes": self.get_quotes(),
            "inventory": self.get_inventory(),
            "metrics": self.metrics.to_dict(),
            "config": self.config.to_dict(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

    def cancel_all_quotes(self) -> int:
        """Cancel all active quotes."""
        cancelled = self.quote_manager.cancel_all()
        self.metrics.cancelled_quotes += len(cancelled)
        return len(cancelled)


class MultiLevelMarketMaker:
    """Market maker with multiple price levels."""

    def __init__(self, symbol: str, config: Optional[MarketMakingConfig] = None):
        """Initialize multi-level market maker."""
        self.symbol = symbol
        self.config = config or MarketMakingConfig()
        self.base_strategy = MarketMakingStrategy(symbol, config)
        self.levels = self.config.max_quotes_per_side

    def generate_ladder(self, market: MarketData) -> list[QuotePair]:
        """Generate multi-level quote ladder."""
        quotes = []
        base_spread = float(self.config.base_spread)
        for i in range(self.levels):
            # Increase spread for each level
            level_spread = Decimal(str(base_spread * (1 + i * 0.5)))
            # Decrease size for outer levels
            level_size = self.config.quote_size / Decimal(str(1 + i * 0.3))
            bid_price = market.mid_price - level_spread
            ask_price = market.mid_price + level_spread
            bid_quote = self.base_strategy.quote_manager.create_quote(
                "bid", bid_price, level_size, self.config.quote_expiry_seconds
            )
            ask_quote = self.base_strategy.quote_manager.create_quote(
                "ask", ask_price, level_size, self.config.quote_expiry_seconds
            )
            quotes.append(QuotePair(
                bid=bid_quote,
                ask=ask_quote,
                spread=level_spread * 2,
                mid_price=market.mid_price,
                timestamp=datetime.now(),
            ))
        return quotes

    def get_status(self) -> dict:
        """Get multi-level status."""
        base_status = self.base_strategy.get_status()
        base_status["levels"] = self.levels
        base_status["total_active_quotes"] = len(
            self.base_strategy.quote_manager.get_active_quotes()
        )
        return base_status


class GridMarketMaker:
    """Grid-based market maker with fixed price levels."""

    def __init__(self, symbol: str, grid_size: int = 10,
                 grid_spacing: Decimal = Decimal("0.001"),
                 size_per_level: Decimal = Decimal("0.1")):
        """Initialize grid market maker."""
        self.symbol = symbol
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.size_per_level = size_per_level
        self.quote_manager = QuoteManager(symbol)
        self.inventory_manager = InventoryManager(Decimal(str(grid_size * 2)))
        self.grid_prices: list[Decimal] = []

    def initialize_grid(self, center_price: Decimal) -> list[Quote]:
        """Initialize grid around center price."""
        self.grid_prices = []
        quotes = []
        # Create buy grid below center
        for i in range(1, self.grid_size + 1):
            price = center_price * (1 - self.grid_spacing * i)
            self.grid_prices.append(price)
            quote = self.quote_manager.create_quote("bid", price, self.size_per_level)
            quotes.append(quote)
        # Create sell grid above center
        for i in range(1, self.grid_size + 1):
            price = center_price * (1 + self.grid_spacing * i)
            self.grid_prices.append(price)
            quote = self.quote_manager.create_quote("ask", price, self.size_per_level)
            quotes.append(quote)
        return quotes

    def on_fill(self, quote_id: str, fill_price: Decimal, fill_size: Decimal) -> Optional[Quote]:
        """Handle fill and create opposite order."""
        quote = self.quote_manager.fill_quote(quote_id, fill_size)
        if not quote:
            return None
        # Update inventory
        self.inventory_manager.record_fill(quote.side, fill_price, fill_size)
        # Create opposite order
        if quote.side == "bid":
            new_price = fill_price * (1 + self.grid_spacing)
            new_quote = self.quote_manager.create_quote("ask", new_price, fill_size)
        else:
            new_price = fill_price * (1 - self.grid_spacing)
            new_quote = self.quote_manager.create_quote("bid", new_price, fill_size)
        return new_quote

    def get_grid_status(self) -> dict:
        """Get grid status."""
        return {
            "symbol": self.symbol,
            "grid_size": self.grid_size,
            "grid_spacing": str(self.grid_spacing),
            "size_per_level": str(self.size_per_level),
            "active_bids": len(self.quote_manager.get_active_quotes("bid")),
            "active_asks": len(self.quote_manager.get_active_quotes("ask")),
            "inventory": self.inventory_manager.metrics.to_dict(),
        }
