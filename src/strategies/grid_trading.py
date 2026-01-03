"""
Grid Trading Strategy Module.

Implements various grid trading strategies:
- Arithmetic grid (equal price spacing)
- Geometric grid (percentage-based spacing)
- Dynamic grid (adapts to volatility)
- Trailing grid (follows trend)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class GridType(Enum):
    """Type of grid spacing."""
    ARITHMETIC = "arithmetic"  # Equal dollar spacing
    GEOMETRIC = "geometric"    # Equal percentage spacing
    CUSTOM = "custom"          # User-defined levels


class GridState(Enum):
    """State of the grid strategy."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class GridOrderStatus(Enum):
    """Status of a grid order."""
    PENDING = "pending"      # Waiting to be placed
    OPEN = "open"            # Order is active on exchange
    FILLED = "filled"        # Order was filled
    CANCELLED = "cancelled"  # Order was cancelled
    FAILED = "failed"        # Order placement failed


@dataclass
class GridLevel:
    """A single level in the grid."""
    level_id: str
    price: Decimal
    side: OrderSide
    size: Decimal
    order_id: str | None = None
    status: GridOrderStatus = GridOrderStatus.PENDING
    filled_at: datetime | None = None
    fill_price: Decimal | None = None
    pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level_id": self.level_id,
            "price": str(self.price),
            "side": self.side.value,
            "size": str(self.size),
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "pnl": str(self.pnl)
        }


@dataclass
class GridConfig:
    """Configuration for a grid strategy."""
    symbol: str
    grid_type: GridType
    upper_price: Decimal
    lower_price: Decimal
    num_grids: int
    total_investment: Decimal
    # Optional parameters
    leverage: Decimal = Decimal("1")
    min_profit_per_grid: Decimal = Decimal("0")  # Minimum profit per grid in %
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    trailing_stop_pct: Decimal | None = None
    rebalance_threshold: Decimal = Decimal("0.1")  # 10% deviation triggers rebalance
    max_open_orders: int = 100
    custom_levels: list[Decimal] | None = None

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.upper_price <= self.lower_price:
            errors.append("Upper price must be greater than lower price")

        if self.num_grids < 2:
            errors.append("Number of grids must be at least 2")

        if self.num_grids > 500:
            errors.append("Number of grids cannot exceed 500")

        if self.total_investment <= Decimal("0"):
            errors.append("Total investment must be positive")

        if self.leverage < Decimal("1"):
            errors.append("Leverage must be at least 1")

        if self.grid_type == GridType.CUSTOM:
            if not self.custom_levels or len(self.custom_levels) < 2:
                errors.append("Custom grid requires at least 2 price levels")
            elif self.custom_levels:
                sorted_levels = sorted(self.custom_levels)
                if sorted_levels != self.custom_levels:
                    errors.append("Custom levels must be sorted in ascending order")

        if self.take_profit_price and self.take_profit_price <= self.upper_price:
            errors.append("Take profit must be above upper grid price")

        if self.stop_loss_price and self.stop_loss_price >= self.lower_price:
            errors.append("Stop loss must be below lower grid price")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "grid_type": self.grid_type.value,
            "upper_price": str(self.upper_price),
            "lower_price": str(self.lower_price),
            "num_grids": self.num_grids,
            "total_investment": str(self.total_investment),
            "leverage": str(self.leverage),
            "min_profit_per_grid": str(self.min_profit_per_grid),
            "take_profit_price": str(self.take_profit_price) if self.take_profit_price else None,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "trailing_stop_pct": str(self.trailing_stop_pct) if self.trailing_stop_pct else None,
            "rebalance_threshold": str(self.rebalance_threshold),
            "max_open_orders": self.max_open_orders,
            "custom_levels": [str(l) for l in self.custom_levels] if self.custom_levels else None
        }


@dataclass
class GridMetrics:
    """Performance metrics for the grid strategy."""
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    avg_profit_per_trade: Decimal = Decimal("0")
    grid_profit_pct: Decimal = Decimal("0")
    apr: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    uptime_seconds: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_pnl": str(self.total_pnl),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_trades": self.total_trades,
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "total_volume": str(self.total_volume),
            "total_fees": str(self.total_fees),
            "win_rate": str(self.win_rate),
            "avg_profit_per_trade": str(self.avg_profit_per_trade),
            "grid_profit_pct": str(self.grid_profit_pct),
            "apr": str(self.apr),
            "max_drawdown": str(self.max_drawdown),
            "uptime_seconds": self.uptime_seconds,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class GridSnapshot:
    """Snapshot of grid state at a point in time."""
    timestamp: datetime
    current_price: Decimal
    buy_levels: int
    sell_levels: int
    open_orders: int
    filled_orders: int
    position_size: Decimal
    position_value: Decimal
    metrics: GridMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_price": str(self.current_price),
            "buy_levels": self.buy_levels,
            "sell_levels": self.sell_levels,
            "open_orders": self.open_orders,
            "filled_orders": self.filled_orders,
            "position_size": str(self.position_size),
            "position_value": str(self.position_value),
            "metrics": self.metrics.to_dict()
        }


class GridCalculator:
    """Calculates grid levels and order sizes."""

    @staticmethod
    def calculate_arithmetic_levels(
        lower_price: Decimal,
        upper_price: Decimal,
        num_grids: int
    ) -> list[Decimal]:
        """
        Calculate arithmetic grid levels (equal spacing).

        Args:
            lower_price: Lower bound of grid
            upper_price: Upper bound of grid
            num_grids: Number of grid lines

        Returns:
            List of price levels
        """
        if num_grids < 2:
            return [lower_price, upper_price]

        step = (upper_price - lower_price) / (num_grids - 1)
        levels = []

        for i in range(num_grids):
            price = lower_price + (step * i)
            levels.append(price.quantize(Decimal("0.01")))

        return levels

    @staticmethod
    def calculate_geometric_levels(
        lower_price: Decimal,
        upper_price: Decimal,
        num_grids: int
    ) -> list[Decimal]:
        """
        Calculate geometric grid levels (equal percentage spacing).

        Args:
            lower_price: Lower bound of grid
            upper_price: Upper bound of grid
            num_grids: Number of grid lines

        Returns:
            List of price levels
        """
        if num_grids < 2:
            return [lower_price, upper_price]

        # Calculate the ratio
        import math
        ratio = math.pow(float(upper_price / lower_price), 1 / (num_grids - 1))

        levels = []
        current = lower_price

        for i in range(num_grids):
            levels.append(current.quantize(Decimal("0.01")))
            current = current * Decimal(str(ratio))

        return levels

    @staticmethod
    def calculate_order_sizes(
        levels: list[Decimal],
        total_investment: Decimal,
        current_price: Decimal,
        leverage: Decimal = Decimal("1")
    ) -> list[Decimal]:
        """
        Calculate order sizes for each level.

        Args:
            levels: Price levels
            total_investment: Total capital to deploy
            current_price: Current market price
            leverage: Leverage multiplier

        Returns:
            List of order sizes for each level
        """
        if not levels:
            return []

        # Count buy levels (below current price) and sell levels
        buy_levels = [l for l in levels if l < current_price]
        sell_levels = [l for l in levels if l >= current_price]

        # Effective investment with leverage
        effective_investment = total_investment * leverage

        # Split investment between buy and sell sides
        if buy_levels and sell_levels:
            buy_investment = effective_investment / 2
            sell_investment = effective_investment / 2
        elif buy_levels:
            buy_investment = effective_investment
            sell_investment = Decimal("0")
        else:
            buy_investment = Decimal("0")
            sell_investment = effective_investment

        sizes = []

        for level in levels:
            if level < current_price and len(buy_levels) > 0:
                # Buy order - size in base currency
                order_investment = buy_investment / len(buy_levels)
                size = order_investment / level
            elif len(sell_levels) > 0:
                # Sell order - size in base currency
                order_investment = sell_investment / len(sell_levels)
                size = order_investment / level
            else:
                size = Decimal("0")

            sizes.append(size.quantize(Decimal("0.0001")))

        return sizes

    @staticmethod
    def calculate_grid_spacing_pct(levels: list[Decimal]) -> Decimal:
        """
        Calculate average grid spacing as percentage.

        Args:
            levels: Price levels

        Returns:
            Average spacing in percentage
        """
        if len(levels) < 2:
            return Decimal("0")

        total_pct = Decimal("0")
        for i in range(1, len(levels)):
            pct = (levels[i] - levels[i - 1]) / levels[i - 1] * 100
            total_pct += pct

        return total_pct / (len(levels) - 1)

    @staticmethod
    def estimate_profit_per_grid(
        levels: list[Decimal],
        fee_rate: Decimal = Decimal("0.001")  # 0.1% per trade
    ) -> Decimal:
        """
        Estimate profit per grid round trip.

        Args:
            levels: Price levels
            fee_rate: Trading fee rate

        Returns:
            Estimated profit percentage per grid
        """
        if len(levels) < 2:
            return Decimal("0")

        avg_spacing = GridCalculator.calculate_grid_spacing_pct(levels)
        total_fees = fee_rate * 2 * 100  # Buy and sell fees in %

        return avg_spacing - total_fees


class GridOrderManager:
    """Manages grid orders."""

    def __init__(self):
        """Initialize order manager."""
        self._levels: dict[str, GridLevel] = {}  # level_id -> GridLevel
        self._order_to_level: dict[str, str] = {}  # order_id -> level_id

    def add_level(self, level: GridLevel) -> None:
        """
        Add a grid level.

        Args:
            level: Grid level to add
        """
        self._levels[level.level_id] = level
        if level.order_id:
            self._order_to_level[level.order_id] = level.level_id

    def get_level(self, level_id: str) -> GridLevel | None:
        """
        Get level by ID.

        Args:
            level_id: Level identifier

        Returns:
            GridLevel or None
        """
        return self._levels.get(level_id)

    def get_level_by_order(self, order_id: str) -> GridLevel | None:
        """
        Get level by order ID.

        Args:
            order_id: Order identifier

        Returns:
            GridLevel or None
        """
        level_id = self._order_to_level.get(order_id)
        if level_id:
            return self._levels.get(level_id)
        return None

    def update_order_id(self, level_id: str, order_id: str) -> None:
        """
        Update order ID for a level.

        Args:
            level_id: Level identifier
            order_id: New order ID
        """
        level = self._levels.get(level_id)
        if level:
            # Remove old mapping
            if level.order_id and level.order_id in self._order_to_level:
                del self._order_to_level[level.order_id]

            level.order_id = order_id
            level.status = GridOrderStatus.OPEN
            self._order_to_level[order_id] = level_id

    def mark_filled(
        self,
        level_id: str,
        fill_price: Decimal,
        timestamp: datetime | None = None
    ) -> None:
        """
        Mark a level as filled.

        Args:
            level_id: Level identifier
            fill_price: Fill price
            timestamp: Fill timestamp
        """
        level = self._levels.get(level_id)
        if level:
            level.status = GridOrderStatus.FILLED
            level.fill_price = fill_price
            level.filled_at = timestamp or datetime.now()

    def mark_cancelled(self, level_id: str) -> None:
        """
        Mark a level as cancelled.

        Args:
            level_id: Level identifier
        """
        level = self._levels.get(level_id)
        if level:
            level.status = GridOrderStatus.CANCELLED
            if level.order_id and level.order_id in self._order_to_level:
                del self._order_to_level[level.order_id]
            level.order_id = None

    def get_open_levels(self) -> list[GridLevel]:
        """Get all open levels."""
        return [l for l in self._levels.values() if l.status == GridOrderStatus.OPEN]

    def get_filled_levels(self) -> list[GridLevel]:
        """Get all filled levels."""
        return [l for l in self._levels.values() if l.status == GridOrderStatus.FILLED]

    def get_pending_levels(self) -> list[GridLevel]:
        """Get all pending levels."""
        return [l for l in self._levels.values() if l.status == GridOrderStatus.PENDING]

    def get_buy_levels(self) -> list[GridLevel]:
        """Get all buy levels."""
        return [l for l in self._levels.values() if l.side == OrderSide.BUY]

    def get_sell_levels(self) -> list[GridLevel]:
        """Get all sell levels."""
        return [l for l in self._levels.values() if l.side == OrderSide.SELL]

    def get_all_levels(self) -> list[GridLevel]:
        """Get all levels sorted by price."""
        return sorted(self._levels.values(), key=lambda l: l.price)

    def clear(self) -> None:
        """Clear all levels."""
        self._levels.clear()
        self._order_to_level.clear()

    def count_open_orders(self) -> int:
        """Count open orders."""
        return len(self.get_open_levels())


class GridStrategy:
    """Main grid trading strategy implementation."""

    def __init__(
        self,
        config: GridConfig,
        strategy_id: str | None = None
    ):
        """
        Initialize grid strategy.

        Args:
            config: Grid configuration
            strategy_id: Unique strategy identifier
        """
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {', '.join(errors)}")

        self.strategy_id = strategy_id or str(uuid4())[:8]
        self.config = config
        self.state = GridState.INACTIVE
        self.order_manager = GridOrderManager()
        self.metrics = GridMetrics()

        self._start_time: datetime | None = None
        self._current_price: Decimal = Decimal("0")
        self._position_size: Decimal = Decimal("0")
        self._avg_entry_price: Decimal = Decimal("0")
        self._highest_price_seen: Decimal = Decimal("0")
        self._trailing_stop_price: Decimal | None = None
        self._snapshots: list[GridSnapshot] = []
        self._max_snapshots = 1000

        # Initialize levels
        self._initialize_levels()

    def _initialize_levels(self) -> None:
        """Initialize grid levels based on configuration."""
        # Calculate price levels
        if self.config.grid_type == GridType.ARITHMETIC:
            levels = GridCalculator.calculate_arithmetic_levels(
                self.config.lower_price,
                self.config.upper_price,
                self.config.num_grids
            )
        elif self.config.grid_type == GridType.GEOMETRIC:
            levels = GridCalculator.calculate_geometric_levels(
                self.config.lower_price,
                self.config.upper_price,
                self.config.num_grids
            )
        elif self.config.grid_type == GridType.CUSTOM and self.config.custom_levels:
            levels = self.config.custom_levels
        else:
            levels = GridCalculator.calculate_arithmetic_levels(
                self.config.lower_price,
                self.config.upper_price,
                self.config.num_grids
            )

        self._price_levels = levels

    def initialize(self, current_price: Decimal) -> list[GridLevel]:
        """
        Initialize the grid with current market price.

        Args:
            current_price: Current market price

        Returns:
            List of levels to place orders for
        """
        if self.state not in [GridState.INACTIVE, GridState.CLOSED]:
            logger.warning(f"Cannot initialize grid in state {self.state}")
            return []

        self.state = GridState.INITIALIZING
        self._current_price = current_price
        self._start_time = datetime.now()
        self._highest_price_seen = current_price

        # Calculate order sizes
        sizes = GridCalculator.calculate_order_sizes(
            self._price_levels,
            self.config.total_investment,
            current_price,
            self.config.leverage
        )

        # Create grid levels
        self.order_manager.clear()
        levels_to_place = []

        for i, (price, size) in enumerate(zip(self._price_levels, sizes)):
            if size <= Decimal("0"):
                continue

            side = OrderSide.BUY if price < current_price else OrderSide.SELL

            level = GridLevel(
                level_id=f"{self.strategy_id}_{i}",
                price=price,
                side=side,
                size=size
            )

            self.order_manager.add_level(level)
            levels_to_place.append(level)

        self.state = GridState.ACTIVE
        logger.info(
            f"Grid initialized with {len(levels_to_place)} levels, "
            f"current price: {current_price}"
        )

        return levels_to_place

    def on_price_update(self, price: Decimal) -> dict[str, Any]:
        """
        Handle price update.

        Args:
            price: New market price

        Returns:
            Dict with actions to take (orders to place/cancel, etc.)
        """
        if self.state != GridState.ACTIVE:
            return {"action": "none", "reason": f"Grid not active: {self.state}"}

        self._current_price = price
        actions: dict[str, Any] = {
            "action": "none",
            "orders_to_place": [],
            "orders_to_cancel": [],
            "close_position": False
        }

        # Check stop loss
        if self.config.stop_loss_price and price <= self.config.stop_loss_price:
            actions["action"] = "stop_loss"
            actions["close_position"] = True
            self.state = GridState.CLOSING
            return actions

        # Check take profit
        if self.config.take_profit_price and price >= self.config.take_profit_price:
            actions["action"] = "take_profit"
            actions["close_position"] = True
            self.state = GridState.CLOSING
            return actions

        # Update trailing stop if enabled
        if self.config.trailing_stop_pct:
            self._update_trailing_stop(price)
            if self._trailing_stop_price and price <= self._trailing_stop_price:
                actions["action"] = "trailing_stop"
                actions["close_position"] = True
                self.state = GridState.CLOSING
                return actions

        # Update unrealized PnL
        self._update_unrealized_pnl(price)

        return actions

    def on_order_filled(
        self,
        order_id: str,
        fill_price: Decimal,
        fee: Decimal = Decimal("0"),
        timestamp: datetime | None = None
    ) -> GridLevel | None:
        """
        Handle order fill event.

        Args:
            order_id: Filled order ID
            fill_price: Actual fill price
            fee: Trading fee
            timestamp: Fill timestamp

        Returns:
            The opposite level to place (for grid continuation) or None
        """
        level = self.order_manager.get_level_by_order(order_id)
        if not level:
            logger.warning(f"Unknown order filled: {order_id}")
            return None

        self.order_manager.mark_filled(level.level_id, fill_price, timestamp)

        # Update metrics
        self.metrics.total_trades += 1
        self.metrics.total_fees += fee
        volume = fill_price * level.size
        self.metrics.total_volume += volume

        if level.side == OrderSide.BUY:
            self.metrics.buy_trades += 1
            self._position_size += level.size
            self._update_avg_entry(fill_price, level.size, "buy")
        else:
            self.metrics.sell_trades += 1
            old_size = self._position_size
            self._position_size -= level.size

            # Calculate realized PnL for sell
            if old_size > Decimal("0") and self._avg_entry_price > Decimal("0"):
                pnl = (fill_price - self._avg_entry_price) * level.size - fee
                level.pnl = pnl
                self.metrics.realized_pnl += pnl

        # Create opposite order for grid continuation
        opposite_level = self._create_opposite_level(level, fill_price)

        logger.info(
            f"Order filled: {level.side.value} at {fill_price}, "
            f"position: {self._position_size}"
        )

        return opposite_level

    def _create_opposite_level(
        self,
        filled_level: GridLevel,
        fill_price: Decimal
    ) -> GridLevel | None:
        """
        Create opposite order after a fill.

        Args:
            filled_level: The level that was filled
            fill_price: Fill price

        Returns:
            New level to place or None
        """
        # Find adjacent level for opposite order
        all_levels = self.order_manager.get_all_levels()
        filled_idx = None

        for i, level in enumerate(all_levels):
            if level.level_id == filled_level.level_id:
                filled_idx = i
                break

        if filled_idx is None:
            return None

        # Determine opposite level price
        if filled_level.side == OrderSide.BUY:
            # After buy, place sell at next higher level
            if filled_idx < len(all_levels) - 1:
                next_level = all_levels[filled_idx + 1]
                new_price = next_level.price
            else:
                return None
            new_side = OrderSide.SELL
        else:
            # After sell, place buy at next lower level
            if filled_idx > 0:
                prev_level = all_levels[filled_idx - 1]
                new_price = prev_level.price
            else:
                return None
            new_side = OrderSide.BUY

        # Create new level
        new_level = GridLevel(
            level_id=f"{filled_level.level_id}_r{self.metrics.total_trades}",
            price=new_price,
            side=new_side,
            size=filled_level.size
        )

        self.order_manager.add_level(new_level)
        return new_level

    def _update_avg_entry(
        self,
        price: Decimal,
        size: Decimal,
        side: str
    ) -> None:
        """Update average entry price."""
        if side == "buy":
            if self._position_size > size:
                # Average with existing position
                old_value = self._avg_entry_price * (self._position_size - size)
                new_value = price * size
                self._avg_entry_price = (old_value + new_value) / self._position_size
            else:
                self._avg_entry_price = price

    def _update_trailing_stop(self, price: Decimal) -> None:
        """Update trailing stop price."""
        if not self.config.trailing_stop_pct:
            return

        if price > self._highest_price_seen:
            self._highest_price_seen = price
            self._trailing_stop_price = price * (1 - self.config.trailing_stop_pct / 100)

    def _update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized PnL."""
        if self._position_size > Decimal("0") and self._avg_entry_price > Decimal("0"):
            self.metrics.unrealized_pnl = (
                (current_price - self._avg_entry_price) * self._position_size
            )
        else:
            self.metrics.unrealized_pnl = Decimal("0")

        self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl

    def get_snapshot(self) -> GridSnapshot:
        """Get current grid snapshot."""
        buy_levels = self.order_manager.get_buy_levels()
        sell_levels = self.order_manager.get_sell_levels()
        open_orders = self.order_manager.get_open_levels()
        filled_orders = self.order_manager.get_filled_levels()

        # Update metrics
        if self._start_time:
            self.metrics.uptime_seconds = int(
                (datetime.now() - self._start_time).total_seconds()
            )

        if self.metrics.total_trades > 0:
            self.metrics.avg_profit_per_trade = (
                self.metrics.realized_pnl / self.metrics.total_trades
            )

        if self.config.total_investment > Decimal("0"):
            self.metrics.grid_profit_pct = (
                self.metrics.total_pnl / self.config.total_investment * 100
            )

            # Calculate APR
            if self.metrics.uptime_seconds > 0:
                days = Decimal(self.metrics.uptime_seconds) / Decimal("86400")
                if days > 0:
                    daily_return = self.metrics.grid_profit_pct / days
                    self.metrics.apr = daily_return * 365

        self.metrics.last_updated = datetime.now()

        snapshot = GridSnapshot(
            timestamp=datetime.now(),
            current_price=self._current_price,
            buy_levels=len(buy_levels),
            sell_levels=len(sell_levels),
            open_orders=len(open_orders),
            filled_orders=len(filled_orders),
            position_size=self._position_size,
            position_value=self._position_size * self._current_price,
            metrics=self.metrics
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

        return snapshot

    def pause(self) -> list[GridLevel]:
        """
        Pause the grid (cancel open orders).

        Returns:
            List of levels to cancel
        """
        if self.state != GridState.ACTIVE:
            return []

        self.state = GridState.PAUSED
        return self.order_manager.get_open_levels()

    def resume(self) -> list[GridLevel]:
        """
        Resume the grid.

        Returns:
            List of levels to place orders for
        """
        if self.state != GridState.PAUSED:
            return []

        self.state = GridState.ACTIVE

        # Re-place cancelled orders based on current price
        levels_to_place = []
        for level in self.order_manager.get_all_levels():
            if level.status in [GridOrderStatus.CANCELLED, GridOrderStatus.PENDING]:
                # Determine if order should be placed
                if level.side == OrderSide.BUY and level.price < self._current_price:
                    levels_to_place.append(level)
                elif level.side == OrderSide.SELL and level.price > self._current_price:
                    levels_to_place.append(level)

        return levels_to_place

    def close(self) -> dict[str, Any]:
        """
        Close the grid strategy.

        Returns:
            Dict with close details
        """
        self.state = GridState.CLOSING

        open_levels = self.order_manager.get_open_levels()

        result = {
            "orders_to_cancel": [l.level_id for l in open_levels],
            "position_to_close": self._position_size,
            "realized_pnl": self.metrics.realized_pnl,
            "unrealized_pnl": self.metrics.unrealized_pnl,
            "total_trades": self.metrics.total_trades
        }

        self.state = GridState.CLOSED
        return result

    def get_grid_info(self) -> dict[str, Any]:
        """Get complete grid information."""
        return {
            "strategy_id": self.strategy_id,
            "state": self.state.value,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "levels": [l.to_dict() for l in self.order_manager.get_all_levels()],
            "position_size": str(self._position_size),
            "avg_entry_price": str(self._avg_entry_price),
            "current_price": str(self._current_price),
            "trailing_stop_price": str(self._trailing_stop_price) if self._trailing_stop_price else None,
            "price_levels": [str(p) for p in self._price_levels]
        }


class GridStrategyManager:
    """Manages multiple grid strategies."""

    def __init__(self):
        """Initialize strategy manager."""
        self._strategies: dict[str, GridStrategy] = {}

    def create_strategy(
        self,
        config: GridConfig,
        strategy_id: str | None = None
    ) -> GridStrategy:
        """
        Create a new grid strategy.

        Args:
            config: Grid configuration
            strategy_id: Optional strategy ID

        Returns:
            Created GridStrategy
        """
        strategy = GridStrategy(config, strategy_id)
        self._strategies[strategy.strategy_id] = strategy
        logger.info(f"Created grid strategy: {strategy.strategy_id}")
        return strategy

    def get_strategy(self, strategy_id: str) -> GridStrategy | None:
        """
        Get strategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            GridStrategy or None
        """
        return self._strategies.get(strategy_id)

    def list_strategies(self) -> list[dict[str, Any]]:
        """List all strategies with summary info."""
        return [
            {
                "strategy_id": s.strategy_id,
                "symbol": s.config.symbol,
                "state": s.state.value,
                "pnl": str(s.metrics.total_pnl),
                "trades": s.metrics.total_trades
            }
            for s in self._strategies.values()
        ]

    def get_active_strategies(self) -> list[GridStrategy]:
        """Get all active strategies."""
        return [s for s in self._strategies.values() if s.state == GridState.ACTIVE]

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy.

        Args:
            strategy_id: Strategy to remove

        Returns:
            True if removed
        """
        if strategy_id in self._strategies:
            strategy = self._strategies[strategy_id]
            if strategy.state not in [GridState.CLOSED, GridState.INACTIVE]:
                strategy.close()
            del self._strategies[strategy_id]
            return True
        return False

    def get_total_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics across all strategies."""
        total_pnl = Decimal("0")
        total_trades = 0
        total_volume = Decimal("0")
        active_count = 0

        for strategy in self._strategies.values():
            total_pnl += strategy.metrics.total_pnl
            total_trades += strategy.metrics.total_trades
            total_volume += strategy.metrics.total_volume
            if strategy.state == GridState.ACTIVE:
                active_count += 1

        return {
            "total_strategies": len(self._strategies),
            "active_strategies": active_count,
            "total_pnl": str(total_pnl),
            "total_trades": total_trades,
            "total_volume": str(total_volume)
        }
