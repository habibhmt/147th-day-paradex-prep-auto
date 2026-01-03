"""
Dollar Cost Averaging (DCA) Strategy Engine.

Implements various DCA strategies:
- Time-based DCA (regular intervals)
- Price-based DCA (buy dips)
- Hybrid DCA (time + price triggers)
- Smart DCA (adaptive amounts)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class DCAType(Enum):
    """Type of DCA strategy."""
    TIME_BASED = "time_based"          # Fixed intervals
    PRICE_BASED = "price_based"        # Buy on price drops
    HYBRID = "hybrid"                  # Both time and price
    SMART = "smart"                    # Adaptive amounts


class DCAState(Enum):
    """State of the DCA strategy."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TriggerType(Enum):
    """Type of DCA trigger."""
    SCHEDULED = "scheduled"           # Time-based trigger
    PRICE_DROP = "price_drop"         # Price drop trigger
    PRICE_TARGET = "price_target"     # Price target reached
    MANUAL = "manual"                 # Manual trigger


class OrderStatus(Enum):
    """Status of a DCA order."""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DCAOrder:
    """A single DCA order."""
    order_id: str
    sequence_number: int
    trigger_type: TriggerType
    planned_amount: Decimal
    actual_amount: Decimal | None = None
    planned_price: Decimal | None = None
    fill_price: Decimal | None = None
    size: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    triggered_at: datetime | None = None
    filled_at: datetime | None = None
    exchange_order_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "sequence_number": self.sequence_number,
            "trigger_type": self.trigger_type.value,
            "planned_amount": str(self.planned_amount),
            "actual_amount": str(self.actual_amount) if self.actual_amount else None,
            "planned_price": str(self.planned_price) if self.planned_price else None,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "size": str(self.size) if self.size else None,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "exchange_order_id": self.exchange_order_id,
            "notes": self.notes
        }


@dataclass
class DCAConfig:
    """Configuration for a DCA strategy."""
    symbol: str
    dca_type: DCAType
    total_amount: Decimal                    # Total amount to invest
    # Time-based settings
    interval_hours: int = 24                 # Hours between purchases
    num_orders: int = 10                     # Number of DCA orders
    # Price-based settings
    price_drop_pct: Decimal = Decimal("5")   # Buy when price drops this %
    price_target: Decimal | None = None      # Optional target price
    max_single_amount: Decimal | None = None # Max per order
    min_single_amount: Decimal | None = None # Min per order
    # Smart DCA settings
    volatility_multiplier: Decimal = Decimal("1.5")  # Increase amount on high volatility
    dip_multiplier: Decimal = Decimal("2")           # Increase amount on dips
    # Risk settings
    stop_loss_pct: Decimal | None = None
    take_profit_pct: Decimal | None = None
    max_price: Decimal | None = None         # Don't buy above this price
    min_price: Decimal | None = None         # Don't buy below this price
    # Execution settings
    use_limit_orders: bool = False
    limit_offset_pct: Decimal = Decimal("0.1")  # % below market for limit orders
    start_time: datetime | None = None
    end_time: datetime | None = None

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.total_amount <= Decimal("0"):
            errors.append("Total amount must be positive")

        if self.num_orders < 1:
            errors.append("Number of orders must be at least 1")

        if self.num_orders > 1000:
            errors.append("Number of orders cannot exceed 1000")

        if self.interval_hours < 1 and self.dca_type == DCAType.TIME_BASED:
            errors.append("Interval must be at least 1 hour for time-based DCA")

        if self.price_drop_pct <= Decimal("0"):
            errors.append("Price drop percentage must be positive")

        if self.price_drop_pct > Decimal("50"):
            errors.append("Price drop percentage too high (max 50%)")

        if self.max_single_amount and self.min_single_amount:
            if self.max_single_amount < self.min_single_amount:
                errors.append("Max single amount must be >= min single amount")

        if self.max_price and self.min_price:
            if self.max_price < self.min_price:
                errors.append("Max price must be >= min price")

        if self.start_time and self.end_time:
            if self.end_time <= self.start_time:
                errors.append("End time must be after start time")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "dca_type": self.dca_type.value,
            "total_amount": str(self.total_amount),
            "interval_hours": self.interval_hours,
            "num_orders": self.num_orders,
            "price_drop_pct": str(self.price_drop_pct),
            "price_target": str(self.price_target) if self.price_target else None,
            "max_single_amount": str(self.max_single_amount) if self.max_single_amount else None,
            "min_single_amount": str(self.min_single_amount) if self.min_single_amount else None,
            "volatility_multiplier": str(self.volatility_multiplier),
            "dip_multiplier": str(self.dip_multiplier),
            "stop_loss_pct": str(self.stop_loss_pct) if self.stop_loss_pct else None,
            "take_profit_pct": str(self.take_profit_pct) if self.take_profit_pct else None,
            "max_price": str(self.max_price) if self.max_price else None,
            "min_price": str(self.min_price) if self.min_price else None,
            "use_limit_orders": self.use_limit_orders,
            "limit_offset_pct": str(self.limit_offset_pct),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class DCAMetrics:
    """Performance metrics for the DCA strategy."""
    total_invested: Decimal = Decimal("0")
    total_size: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_pct: Decimal = Decimal("0")
    orders_executed: int = 0
    orders_pending: int = 0
    orders_skipped: int = 0
    orders_failed: int = 0
    lowest_buy_price: Decimal = Decimal("0")
    highest_buy_price: Decimal = Decimal("0")
    time_in_market_hours: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_invested": str(self.total_invested),
            "total_size": str(self.total_size),
            "average_price": str(self.average_price),
            "current_value": str(self.current_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_pct": str(self.unrealized_pnl_pct),
            "orders_executed": self.orders_executed,
            "orders_pending": self.orders_pending,
            "orders_skipped": self.orders_skipped,
            "orders_failed": self.orders_failed,
            "lowest_buy_price": str(self.lowest_buy_price),
            "highest_buy_price": str(self.highest_buy_price),
            "time_in_market_hours": self.time_in_market_hours,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class PricePoint:
    """A price data point."""
    timestamp: datetime
    price: Decimal


class DCACalculator:
    """Calculates DCA amounts and timing."""

    @staticmethod
    def calculate_fixed_amounts(
        total_amount: Decimal,
        num_orders: int
    ) -> list[Decimal]:
        """
        Calculate fixed amounts for each order.

        Args:
            total_amount: Total amount to invest
            num_orders: Number of orders

        Returns:
            List of amounts per order
        """
        if num_orders <= 0:
            return []

        amount_per_order = total_amount / num_orders
        amounts = [amount_per_order.quantize(Decimal("0.01"))] * num_orders

        # Adjust last order for rounding
        total = sum(amounts)
        if total != total_amount:
            amounts[-1] += (total_amount - total)

        return amounts

    @staticmethod
    def calculate_weighted_amounts(
        total_amount: Decimal,
        num_orders: int,
        weight_start: Decimal = Decimal("0.5"),
        weight_end: Decimal = Decimal("1.5")
    ) -> list[Decimal]:
        """
        Calculate weighted amounts (increasing over time).

        Args:
            total_amount: Total amount to invest
            num_orders: Number of orders
            weight_start: Starting weight multiplier
            weight_end: Ending weight multiplier

        Returns:
            List of weighted amounts per order
        """
        if num_orders <= 0:
            return []

        if num_orders == 1:
            return [total_amount]

        # Generate weights
        weights = []
        for i in range(num_orders):
            ratio = Decimal(i) / Decimal(num_orders - 1)
            weight = weight_start + (weight_end - weight_start) * ratio
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        # Calculate amounts
        amounts = []
        for norm_weight in normalized:
            amount = (total_amount * norm_weight).quantize(Decimal("0.01"))
            amounts.append(amount)

        # Adjust for rounding
        total = sum(amounts)
        if total != total_amount:
            amounts[-1] += (total_amount - total)

        return amounts

    @staticmethod
    def calculate_schedule(
        start_time: datetime,
        interval_hours: int,
        num_orders: int
    ) -> list[datetime]:
        """
        Calculate order schedule.

        Args:
            start_time: Start time for DCA
            interval_hours: Hours between orders
            num_orders: Number of orders

        Returns:
            List of scheduled times
        """
        schedule = []
        current = start_time

        for _ in range(num_orders):
            schedule.append(current)
            current += timedelta(hours=interval_hours)

        return schedule

    @staticmethod
    def calculate_dip_amount(
        base_amount: Decimal,
        current_price: Decimal,
        reference_price: Decimal,
        dip_multiplier: Decimal = Decimal("2"),
        max_multiplier: Decimal = Decimal("3")
    ) -> Decimal:
        """
        Calculate amount adjustment based on price dip.

        Args:
            base_amount: Base order amount
            current_price: Current price
            reference_price: Reference price (e.g., moving average)
            dip_multiplier: Multiplier per 10% dip
            max_multiplier: Maximum multiplier

        Returns:
            Adjusted amount
        """
        if reference_price <= Decimal("0"):
            return base_amount

        # Calculate dip percentage
        dip_pct = (reference_price - current_price) / reference_price * 100

        if dip_pct <= Decimal("0"):
            return base_amount

        # Calculate multiplier (linear increase per 10% dip)
        multiplier = Decimal("1") + (dip_pct / 10) * (dip_multiplier - 1)
        multiplier = min(multiplier, max_multiplier)

        return (base_amount * multiplier).quantize(Decimal("0.01"))

    @staticmethod
    def calculate_volatility_adjusted_amount(
        base_amount: Decimal,
        current_volatility: Decimal,
        average_volatility: Decimal,
        volatility_multiplier: Decimal = Decimal("1.5"),
        max_multiplier: Decimal = Decimal("2")
    ) -> Decimal:
        """
        Adjust amount based on volatility.

        Args:
            base_amount: Base order amount
            current_volatility: Current volatility
            average_volatility: Average volatility
            volatility_multiplier: Base multiplier for high volatility
            max_multiplier: Maximum multiplier

        Returns:
            Adjusted amount
        """
        if average_volatility <= Decimal("0"):
            return base_amount

        vol_ratio = current_volatility / average_volatility

        if vol_ratio <= Decimal("1"):
            return base_amount

        # Increase amount during high volatility (better prices)
        multiplier = Decimal("1") + (vol_ratio - 1) * (volatility_multiplier - 1)
        multiplier = min(multiplier, max_multiplier)

        return (base_amount * multiplier).quantize(Decimal("0.01"))


class DCAStrategy:
    """Main DCA strategy implementation."""

    def __init__(
        self,
        config: DCAConfig,
        strategy_id: str | None = None
    ):
        """
        Initialize DCA strategy.

        Args:
            config: DCA configuration
            strategy_id: Unique strategy identifier
        """
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {', '.join(errors)}")

        self.strategy_id = strategy_id or str(uuid4())[:8]
        self.config = config
        self.state = DCAState.INACTIVE
        self.metrics = DCAMetrics()

        self._orders: list[DCAOrder] = []
        self._current_price: Decimal = Decimal("0")
        self._reference_price: Decimal = Decimal("0")
        self._last_price_drop_trigger: Decimal | None = None
        self._start_time: datetime | None = None
        self._first_fill_time: datetime | None = None
        self._price_history: list[PricePoint] = []
        self._max_price_history = 1000

        # Initialize orders
        self._initialize_orders()

    def _initialize_orders(self) -> None:
        """Initialize DCA orders based on configuration."""
        # Calculate amounts
        if self.config.dca_type == DCAType.SMART:
            amounts = DCACalculator.calculate_weighted_amounts(
                self.config.total_amount,
                self.config.num_orders
            )
        else:
            amounts = DCACalculator.calculate_fixed_amounts(
                self.config.total_amount,
                self.config.num_orders
            )

        # Calculate schedule for time-based
        start_time = self.config.start_time or datetime.now()
        schedule = DCACalculator.calculate_schedule(
            start_time,
            self.config.interval_hours,
            self.config.num_orders
        )

        # Create orders
        for i, (amount, scheduled_time) in enumerate(zip(amounts, schedule)):
            # Apply min/max constraints
            if self.config.max_single_amount:
                amount = min(amount, self.config.max_single_amount)
            if self.config.min_single_amount:
                amount = max(amount, self.config.min_single_amount)

            order = DCAOrder(
                order_id=f"{self.strategy_id}_{i}",
                sequence_number=i + 1,
                trigger_type=TriggerType.SCHEDULED,
                planned_amount=amount,
                triggered_at=scheduled_time if self.config.dca_type == DCAType.TIME_BASED else None
            )
            self._orders.append(order)

        # Update metrics
        self.metrics.orders_pending = len(self._orders)

    def start(self, current_price: Decimal) -> None:
        """
        Start the DCA strategy.

        Args:
            current_price: Current market price
        """
        if self.state != DCAState.INACTIVE:
            logger.warning(f"Cannot start DCA in state {self.state}")
            return

        self._current_price = current_price
        self._reference_price = current_price
        self._last_price_drop_trigger = current_price
        self._start_time = datetime.now()
        self.state = DCAState.ACTIVE

        logger.info(f"DCA strategy started: {self.strategy_id} at price {current_price}")

    def check_triggers(
        self,
        current_price: Decimal,
        current_time: datetime | None = None,
        volatility: Decimal | None = None
    ) -> list[DCAOrder]:
        """
        Check for DCA triggers and return orders to execute.

        Args:
            current_price: Current market price
            current_time: Current time
            volatility: Current volatility (optional)

        Returns:
            List of orders to execute
        """
        if self.state != DCAState.ACTIVE:
            return []

        current_time = current_time or datetime.now()
        self._current_price = current_price

        # Record price history
        self._price_history.append(PricePoint(current_time, current_price))
        if len(self._price_history) > self._max_price_history:
            self._price_history = self._price_history[-self._max_price_history:]

        # Check end time
        if self.config.end_time and current_time >= self.config.end_time:
            self._complete_strategy()
            return []

        # Check price limits
        if self.config.max_price and current_price > self.config.max_price:
            return []  # Price too high

        if self.config.min_price and current_price < self.config.min_price:
            return []  # Price too low (might indicate issue)

        orders_to_execute = []
        pending_orders = [o for o in self._orders if o.status == OrderStatus.PENDING]

        if not pending_orders:
            self._complete_strategy()
            return []

        for order in pending_orders:
            should_trigger = False
            trigger_type = order.trigger_type

            # Time-based trigger
            if self.config.dca_type in [DCAType.TIME_BASED, DCAType.HYBRID]:
                if order.triggered_at and current_time >= order.triggered_at:
                    should_trigger = True
                    trigger_type = TriggerType.SCHEDULED

            # Price-based trigger
            if self.config.dca_type in [DCAType.PRICE_BASED, DCAType.HYBRID]:
                if self._check_price_trigger(current_price):
                    should_trigger = True
                    trigger_type = TriggerType.PRICE_DROP

            # Price target trigger
            if self.config.price_target and current_price <= self.config.price_target:
                should_trigger = True
                trigger_type = TriggerType.PRICE_TARGET

            if should_trigger:
                # Calculate actual amount
                actual_amount = self._calculate_order_amount(
                    order.planned_amount,
                    current_price,
                    volatility
                )

                order.trigger_type = trigger_type
                order.actual_amount = actual_amount
                order.planned_price = current_price
                order.triggered_at = current_time
                order.status = OrderStatus.PLACED

                orders_to_execute.append(order)

                # Only execute one order per check for price-based
                if self.config.dca_type == DCAType.PRICE_BASED:
                    self._last_price_drop_trigger = current_price
                    break

        return orders_to_execute

    def _check_price_trigger(self, current_price: Decimal) -> bool:
        """Check if price drop trigger is met."""
        if self._last_price_drop_trigger is None:
            return False

        drop_pct = (self._last_price_drop_trigger - current_price) / self._last_price_drop_trigger * 100

        return drop_pct >= self.config.price_drop_pct

    def _calculate_order_amount(
        self,
        base_amount: Decimal,
        current_price: Decimal,
        volatility: Decimal | None = None
    ) -> Decimal:
        """Calculate actual order amount based on conditions."""
        amount = base_amount

        # Smart DCA adjustments
        if self.config.dca_type == DCAType.SMART:
            # Adjust for dip
            if self._reference_price > Decimal("0"):
                amount = DCACalculator.calculate_dip_amount(
                    amount,
                    current_price,
                    self._reference_price,
                    self.config.dip_multiplier
                )

            # Adjust for volatility
            if volatility:
                avg_vol = self._calculate_average_volatility()
                if avg_vol > Decimal("0"):
                    amount = DCACalculator.calculate_volatility_adjusted_amount(
                        amount,
                        volatility,
                        avg_vol,
                        self.config.volatility_multiplier
                    )

        # Apply constraints
        if self.config.max_single_amount:
            amount = min(amount, self.config.max_single_amount)
        if self.config.min_single_amount:
            amount = max(amount, self.config.min_single_amount)

        return amount

    def _calculate_average_volatility(self) -> Decimal:
        """Calculate average price volatility from history."""
        if len(self._price_history) < 2:
            return Decimal("0")

        volatilities = []
        for i in range(1, len(self._price_history)):
            prev_price = self._price_history[i - 1].price
            curr_price = self._price_history[i].price
            if prev_price > Decimal("0"):
                change = abs(curr_price - prev_price) / prev_price
                volatilities.append(change)

        if not volatilities:
            return Decimal("0")

        return sum(volatilities) / len(volatilities)

    def on_order_filled(
        self,
        order_id: str,
        fill_price: Decimal,
        size: Decimal,
        timestamp: datetime | None = None
    ) -> None:
        """
        Handle order fill event.

        Args:
            order_id: Order identifier
            fill_price: Actual fill price
            size: Filled size
            timestamp: Fill timestamp
        """
        order = self._get_order(order_id)
        if not order:
            logger.warning(f"Unknown order filled: {order_id}")
            return

        timestamp = timestamp or datetime.now()

        order.fill_price = fill_price
        order.size = size
        order.filled_at = timestamp
        order.status = OrderStatus.FILLED

        # Update metrics
        self.metrics.orders_executed += 1
        self.metrics.orders_pending -= 1
        invested = fill_price * size
        self.metrics.total_invested += invested
        self.metrics.total_size += size

        # Update average price
        if self.metrics.total_size > Decimal("0"):
            self.metrics.average_price = self.metrics.total_invested / self.metrics.total_size

        # Update price range
        if self.metrics.lowest_buy_price == Decimal("0") or fill_price < self.metrics.lowest_buy_price:
            self.metrics.lowest_buy_price = fill_price
        if fill_price > self.metrics.highest_buy_price:
            self.metrics.highest_buy_price = fill_price

        # Track first fill time
        if not self._first_fill_time:
            self._first_fill_time = timestamp

        # Update reference price (moving average)
        self._update_reference_price(fill_price)

        logger.info(
            f"DCA order filled: {order_id} at {fill_price}, "
            f"total invested: {self.metrics.total_invested}"
        )

    def on_order_failed(self, order_id: str, reason: str = "") -> None:
        """
        Handle order failure.

        Args:
            order_id: Order identifier
            reason: Failure reason
        """
        order = self._get_order(order_id)
        if not order:
            return

        order.status = OrderStatus.FAILED
        order.notes = reason

        self.metrics.orders_failed += 1
        self.metrics.orders_pending -= 1

        logger.warning(f"DCA order failed: {order_id}, reason: {reason}")

    def skip_order(self, order_id: str, reason: str = "") -> None:
        """
        Skip a DCA order.

        Args:
            order_id: Order identifier
            reason: Skip reason
        """
        order = self._get_order(order_id)
        if not order:
            return

        order.status = OrderStatus.SKIPPED
        order.notes = reason

        self.metrics.orders_skipped += 1
        self.metrics.orders_pending -= 1

    def _get_order(self, order_id: str) -> DCAOrder | None:
        """Get order by ID."""
        for order in self._orders:
            if order.order_id == order_id:
                return order
        return None

    def _update_reference_price(self, new_price: Decimal) -> None:
        """Update reference price (exponential moving average)."""
        alpha = Decimal("0.2")  # Smoothing factor
        self._reference_price = (alpha * new_price +
                                 (1 - alpha) * self._reference_price)

    def update_metrics(self, current_price: Decimal) -> None:
        """
        Update metrics with current price.

        Args:
            current_price: Current market price
        """
        self._current_price = current_price

        if self.metrics.total_size > Decimal("0"):
            self.metrics.current_value = current_price * self.metrics.total_size
            self.metrics.unrealized_pnl = self.metrics.current_value - self.metrics.total_invested

            if self.metrics.total_invested > Decimal("0"):
                self.metrics.unrealized_pnl_pct = (
                    self.metrics.unrealized_pnl / self.metrics.total_invested * 100
                )

        # Update time in market
        if self._first_fill_time:
            delta = datetime.now() - self._first_fill_time
            self.metrics.time_in_market_hours = int(delta.total_seconds() / 3600)

        self.metrics.last_updated = datetime.now()

    def check_exit_conditions(self, current_price: Decimal) -> dict[str, Any]:
        """
        Check if exit conditions are met.

        Args:
            current_price: Current market price

        Returns:
            Dict with exit info if conditions met
        """
        if self.state != DCAState.ACTIVE:
            return {"should_exit": False}

        self.update_metrics(current_price)

        # Check stop loss
        if self.config.stop_loss_pct:
            if self.metrics.unrealized_pnl_pct <= -self.config.stop_loss_pct:
                return {
                    "should_exit": True,
                    "reason": "stop_loss",
                    "pnl": self.metrics.unrealized_pnl,
                    "pnl_pct": self.metrics.unrealized_pnl_pct
                }

        # Check take profit
        if self.config.take_profit_pct:
            if self.metrics.unrealized_pnl_pct >= self.config.take_profit_pct:
                return {
                    "should_exit": True,
                    "reason": "take_profit",
                    "pnl": self.metrics.unrealized_pnl,
                    "pnl_pct": self.metrics.unrealized_pnl_pct
                }

        return {"should_exit": False}

    def _complete_strategy(self) -> None:
        """Mark strategy as completed."""
        if self.state == DCAState.ACTIVE:
            self.state = DCAState.COMPLETED
            logger.info(f"DCA strategy completed: {self.strategy_id}")

    def pause(self) -> None:
        """Pause the strategy."""
        if self.state == DCAState.ACTIVE:
            self.state = DCAState.PAUSED
            logger.info(f"DCA strategy paused: {self.strategy_id}")

    def resume(self) -> None:
        """Resume the strategy."""
        if self.state == DCAState.PAUSED:
            self.state = DCAState.ACTIVE
            logger.info(f"DCA strategy resumed: {self.strategy_id}")

    def cancel(self) -> dict[str, Any]:
        """
        Cancel the strategy.

        Returns:
            Summary of cancelled strategy
        """
        self.state = DCAState.CANCELLED

        # Cancel pending orders
        cancelled_orders = []
        for order in self._orders:
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.SKIPPED
                order.notes = "Strategy cancelled"
                cancelled_orders.append(order.order_id)

        return {
            "strategy_id": self.strategy_id,
            "cancelled_orders": cancelled_orders,
            "total_invested": str(self.metrics.total_invested),
            "total_size": str(self.metrics.total_size),
            "average_price": str(self.metrics.average_price)
        }

    def trigger_manual_order(
        self,
        current_price: Decimal,
        amount: Decimal | None = None
    ) -> DCAOrder | None:
        """
        Trigger a manual DCA order.

        Args:
            current_price: Current market price
            amount: Custom amount (uses default if not specified)

        Returns:
            Order to execute or None
        """
        if self.state != DCAState.ACTIVE:
            return None

        pending_orders = [o for o in self._orders if o.status == OrderStatus.PENDING]
        if not pending_orders:
            return None

        order = pending_orders[0]
        order.trigger_type = TriggerType.MANUAL
        order.actual_amount = amount or order.planned_amount
        order.planned_price = current_price
        order.triggered_at = datetime.now()
        order.status = OrderStatus.PLACED

        return order

    def get_status(self) -> dict[str, Any]:
        """Get current strategy status."""
        self.update_metrics(self._current_price)

        return {
            "strategy_id": self.strategy_id,
            "symbol": self.config.symbol,
            "state": self.state.value,
            "dca_type": self.config.dca_type.value,
            "current_price": str(self._current_price),
            "reference_price": str(self._reference_price),
            "metrics": self.metrics.to_dict(),
            "progress": {
                "executed": self.metrics.orders_executed,
                "pending": self.metrics.orders_pending,
                "skipped": self.metrics.orders_skipped,
                "failed": self.metrics.orders_failed,
                "total": len(self._orders),
                "completion_pct": str(
                    Decimal(self.metrics.orders_executed) / Decimal(len(self._orders)) * 100
                    if self._orders else Decimal("0")
                )
            },
            "config": self.config.to_dict()
        }

    def get_orders(self) -> list[dict[str, Any]]:
        """Get all orders."""
        return [o.to_dict() for o in self._orders]

    def get_next_order(self) -> DCAOrder | None:
        """Get next pending order."""
        for order in self._orders:
            if order.status == OrderStatus.PENDING:
                return order
        return None


class DCAStrategyManager:
    """Manages multiple DCA strategies."""

    def __init__(self):
        """Initialize strategy manager."""
        self._strategies: dict[str, DCAStrategy] = {}

    def create_strategy(
        self,
        config: DCAConfig,
        strategy_id: str | None = None
    ) -> DCAStrategy:
        """
        Create a new DCA strategy.

        Args:
            config: DCA configuration
            strategy_id: Optional strategy ID

        Returns:
            Created DCAStrategy
        """
        strategy = DCAStrategy(config, strategy_id)
        self._strategies[strategy.strategy_id] = strategy
        logger.info(f"Created DCA strategy: {strategy.strategy_id}")
        return strategy

    def get_strategy(self, strategy_id: str) -> DCAStrategy | None:
        """
        Get strategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            DCAStrategy or None
        """
        return self._strategies.get(strategy_id)

    def list_strategies(self) -> list[dict[str, Any]]:
        """List all strategies with summary info."""
        return [
            {
                "strategy_id": s.strategy_id,
                "symbol": s.config.symbol,
                "state": s.state.value,
                "dca_type": s.config.dca_type.value,
                "invested": str(s.metrics.total_invested),
                "progress": f"{s.metrics.orders_executed}/{len(s._orders)}"
            }
            for s in self._strategies.values()
        ]

    def get_active_strategies(self) -> list[DCAStrategy]:
        """Get all active strategies."""
        return [s for s in self._strategies.values() if s.state == DCAState.ACTIVE]

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
            if strategy.state == DCAState.ACTIVE:
                strategy.cancel()
            del self._strategies[strategy_id]
            return True
        return False

    def check_all_triggers(
        self,
        prices: dict[str, Decimal],
        current_time: datetime | None = None
    ) -> dict[str, list[DCAOrder]]:
        """
        Check triggers for all active strategies.

        Args:
            prices: Dict of symbol -> current price
            current_time: Current time

        Returns:
            Dict of strategy_id -> orders to execute
        """
        results = {}

        for strategy in self.get_active_strategies():
            symbol = strategy.config.symbol
            if symbol in prices:
                orders = strategy.check_triggers(prices[symbol], current_time)
                if orders:
                    results[strategy.strategy_id] = orders

        return results

    def get_total_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics across all strategies."""
        total_invested = Decimal("0")
        total_value = Decimal("0")
        total_orders = 0
        active_count = 0

        for strategy in self._strategies.values():
            total_invested += strategy.metrics.total_invested
            total_value += strategy.metrics.current_value
            total_orders += strategy.metrics.orders_executed
            if strategy.state == DCAState.ACTIVE:
                active_count += 1

        total_pnl = total_value - total_invested
        pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else Decimal("0")

        return {
            "total_strategies": len(self._strategies),
            "active_strategies": active_count,
            "total_invested": str(total_invested),
            "total_value": str(total_value),
            "total_pnl": str(total_pnl),
            "pnl_pct": str(pnl_pct),
            "total_orders_executed": total_orders
        }
