"""
Strategy Backtester Module.

Comprehensive backtesting engine for evaluating trading strategies
against historical data with realistic simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import random
import math


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class BacktestMode(Enum):
    """Backtesting mode."""
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    TICK_BY_TICK = "tick_by_tick"


class SlippageModel(Enum):
    """Slippage model type."""
    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_BASED = "volume_based"
    SPREAD_BASED = "spread_based"


class FillModel(Enum):
    """Order fill model."""
    IMMEDIATE = "immediate"
    NEXT_BAR = "next_bar"
    LIMIT_CHECK = "limit_check"
    PROBABILISTIC = "probabilistic"


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume)
        }


@dataclass
class Trade:
    """Executed trade."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    slippage: Decimal
    pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "pnl": str(self.pnl)
        }


@dataclass
class Order:
    """Order for backtesting."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    filled_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    expiry: Optional[datetime] = None
    trail_amount: Optional[Decimal] = None
    trail_percent: Optional[Decimal] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "filled_price": str(self.filled_price),
            "commission": str(self.commission)
        }


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    entry_time: datetime
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    commission_paid: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "commission_paid": str(self.commission_paid)
        }


@dataclass
class EquityPoint:
    """Equity curve point."""
    timestamp: datetime
    equity: Decimal
    cash: Decimal
    positions_value: Decimal
    drawdown: Decimal
    drawdown_pct: Decimal

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": str(self.equity),
            "cash": str(self.cash),
            "positions_value": str(self.positions_value),
            "drawdown": str(self.drawdown),
            "drawdown_pct": str(self.drawdown_pct)
        }


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    max_drawdown_duration: int  # days
    win_rate: Decimal
    profit_factor: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float  # hours
    total_commission: Decimal
    total_slippage: Decimal
    exposure_time: Decimal  # percentage
    recovery_factor: Decimal
    expectancy: Decimal
    sqn: Decimal  # System Quality Number

    def to_dict(self) -> dict:
        return {
            "total_return": str(self.total_return),
            "total_return_pct": str(self.total_return_pct),
            "annualized_return": str(self.annualized_return),
            "volatility": str(self.volatility),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "total_commission": str(self.total_commission),
            "total_slippage": str(self.total_slippage),
            "exposure_time": str(self.exposure_time),
            "recovery_factor": str(self.recovery_factor),
            "expectancy": str(self.expectancy),
            "sqn": str(self.sqn)
        }


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: Decimal = Decimal("100000")
    commission_rate: Decimal = Decimal("0.001")  # 0.1%
    slippage_model: SlippageModel = SlippageModel.PERCENTAGE
    slippage_rate: Decimal = Decimal("0.0005")  # 0.05%
    fill_model: FillModel = FillModel.NEXT_BAR
    margin_rate: Decimal = Decimal("0.1")  # 10% margin
    leverage: Decimal = Decimal("1")
    allow_shorting: bool = True
    fractional_shares: bool = True
    risk_free_rate: Decimal = Decimal("0.02")  # 2% annual

    def to_dict(self) -> dict:
        return {
            "initial_capital": str(self.initial_capital),
            "commission_rate": str(self.commission_rate),
            "slippage_model": self.slippage_model.value,
            "slippage_rate": str(self.slippage_rate),
            "fill_model": self.fill_model.value,
            "margin_rate": str(self.margin_rate),
            "leverage": str(self.leverage),
            "allow_shorting": self.allow_shorting,
            "fractional_shares": self.fractional_shares,
            "risk_free_rate": str(self.risk_free_rate)
        }


@dataclass
class BacktestResult:
    """Complete backtest result."""
    metrics: BacktestMetrics
    equity_curve: list[EquityPoint]
    trades: list[Trade]
    orders: list[Order]
    positions_history: list[dict]
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    symbols: list[str]

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "equity_curve": [e.to_dict() for e in self.equity_curve],
            "trades": [t.to_dict() for t in self.trades],
            "orders": [o.to_dict() for o in self.orders],
            "positions_history": self.positions_history,
            "config": self.config.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "symbols": self.symbols
        }


class SlippageCalculator:
    """Calculate slippage based on model."""

    def __init__(self, model: SlippageModel, rate: Decimal):
        self.model = model
        self.rate = rate

    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: OrderSide,
        volume: Optional[Decimal] = None,
        spread: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate slippage amount."""
        if self.model == SlippageModel.NONE:
            return Decimal("0")

        if self.model == SlippageModel.FIXED:
            return self.rate

        if self.model == SlippageModel.PERCENTAGE:
            return price * self.rate

        if self.model == SlippageModel.VOLUME_BASED:
            if volume and volume > 0:
                impact = float(quantity) / float(volume)
                return price * Decimal(str(impact)) * self.rate * Decimal("10")
            return price * self.rate

        if self.model == SlippageModel.SPREAD_BASED:
            if spread:
                return spread * Decimal("0.5")
            return price * self.rate

        return Decimal("0")

    def apply(
        self,
        price: Decimal,
        quantity: Decimal,
        side: OrderSide,
        volume: Optional[Decimal] = None,
        spread: Optional[Decimal] = None
    ) -> Decimal:
        """Apply slippage to price."""
        slippage = self.calculate(price, quantity, side, volume, spread)

        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage


class CommissionCalculator:
    """Calculate trading commissions."""

    def __init__(
        self,
        rate: Decimal = Decimal("0.001"),
        min_commission: Decimal = Decimal("0"),
        max_commission: Optional[Decimal] = None,
        per_share: Optional[Decimal] = None
    ):
        self.rate = rate
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.per_share = per_share

    def calculate(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Calculate commission for trade."""
        if self.per_share:
            commission = quantity * self.per_share
        else:
            commission = price * quantity * self.rate

        commission = max(commission, self.min_commission)

        if self.max_commission:
            commission = min(commission, self.max_commission)

        return commission


class PositionManager:
    """Manage positions during backtest."""

    def __init__(self, allow_shorting: bool = True):
        self.positions: dict[str, Position] = {}
        self.allow_shorting = allow_shorting
        self.position_history: list[dict] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime,
        commission: Decimal
    ) -> Position:
        """Open or add to position."""
        existing = self.positions.get(symbol)

        if existing:
            if existing.side == side:
                # Average in
                total_qty = existing.quantity + quantity
                avg_price = (
                    existing.entry_price * existing.quantity +
                    price * quantity
                ) / total_qty
                existing.quantity = total_qty
                existing.entry_price = avg_price
                existing.commission_paid += commission
                return existing
            else:
                # Close existing first
                self.close_position(symbol, existing.quantity, price, timestamp, commission / 2)
                quantity = quantity - existing.quantity
                if quantity <= 0:
                    return existing

        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            commission_paid=commission
        )
        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime,
        commission: Decimal
    ) -> Optional[Decimal]:
        """Close or reduce position, return realized PnL."""
        position = self.positions.get(symbol)
        if not position:
            return None

        close_qty = min(quantity, position.quantity)

        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * close_qty
        else:
            pnl = (position.entry_price - price) * close_qty

        pnl -= commission
        position.realized_pnl += pnl
        position.quantity -= close_qty
        position.commission_paid += commission

        # Record in history
        self.position_history.append({
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "action": "close" if position.quantity == 0 else "reduce",
            "quantity": str(close_qty),
            "price": str(price),
            "pnl": str(pnl)
        })

        if position.quantity == 0:
            del self.positions[symbol]

        return pnl

    def update_unrealized(self, symbol: str, current_price: Decimal):
        """Update unrealized PnL."""
        position = self.positions.get(symbol)
        if not position:
            return

        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

    def get_total_value(self, prices: dict[str, Decimal]) -> Decimal:
        """Get total positions value."""
        total = Decimal("0")
        for symbol, position in self.positions.items():
            if symbol in prices:
                total += position.quantity * prices[symbol]
        return total

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL."""
        return sum(p.unrealized_pnl for p in self.positions.values())


class OrderManager:
    """Manage orders during backtest."""

    def __init__(
        self,
        slippage_calc: SlippageCalculator,
        commission_calc: CommissionCalculator,
        fill_model: FillModel
    ):
        self.slippage_calc = slippage_calc
        self.commission_calc = commission_calc
        self.fill_model = fill_model
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.order_counter = 0

    def create_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        expiry: Optional[datetime] = None,
        trail_amount: Optional[Decimal] = None,
        trail_percent: Optional[Decimal] = None
    ) -> Order:
        """Create new order."""
        self.order_counter += 1
        order = Order(
            id=f"ORD-{self.order_counter:06d}",
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            expiry=expiry,
            trail_amount=trail_amount,
            trail_percent=trail_percent
        )
        self.pending_orders.append(order)
        return order

    def process_orders(
        self,
        bar: OHLCV,
        volume: Optional[Decimal] = None
    ) -> list[tuple[Order, Decimal, Decimal]]:
        """Process pending orders against bar, return (order, fill_price, slippage)."""
        fills = []
        remaining = []

        for order in self.pending_orders:
            if order.symbol != bar.timestamp:
                # Skip orders for other symbols
                pass

            # Check expiry
            if order.expiry and bar.timestamp > order.expiry:
                order.status = OrderStatus.EXPIRED
                self.filled_orders.append(order)
                continue

            fill_price, slippage = self._check_fill(order, bar, volume)

            if fill_price is not None:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = fill_price
                order.commission = self.commission_calc.calculate(
                    fill_price, order.quantity
                )
                self.filled_orders.append(order)
                fills.append((order, fill_price, slippage))
            else:
                remaining.append(order)

        self.pending_orders = remaining
        return fills

    def _check_fill(
        self,
        order: Order,
        bar: OHLCV,
        volume: Optional[Decimal]
    ) -> tuple[Optional[Decimal], Decimal]:
        """Check if order can be filled, return (fill_price, slippage)."""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at open of next bar
            if self.fill_model == FillModel.NEXT_BAR:
                base_price = bar.open
            else:
                base_price = bar.close

            slippage = self.slippage_calc.calculate(
                base_price, order.quantity, order.side, volume
            )
            fill_price = self.slippage_calc.apply(
                base_price, order.quantity, order.side, volume
            )
            return fill_price, slippage

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if bar.low <= order.price:
                    # Limit buy fills at limit price or better
                    fill_price = min(order.price, bar.open) if bar.open <= order.price else order.price
                    return fill_price, Decimal("0")
            else:
                if bar.high >= order.price:
                    fill_price = max(order.price, bar.open) if bar.open >= order.price else order.price
                    return fill_price, Decimal("0")

        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                if bar.high >= order.stop_price:
                    base_price = max(bar.open, order.stop_price)
                    slippage = self.slippage_calc.calculate(
                        base_price, order.quantity, order.side, volume
                    )
                    fill_price = self.slippage_calc.apply(
                        base_price, order.quantity, order.side, volume
                    )
                    return fill_price, slippage
            else:
                if bar.low <= order.stop_price:
                    base_price = min(bar.open, order.stop_price)
                    slippage = self.slippage_calc.calculate(
                        base_price, order.quantity, order.side, volume
                    )
                    fill_price = self.slippage_calc.apply(
                        base_price, order.quantity, order.side, volume
                    )
                    return fill_price, slippage

        if order.order_type == OrderType.TAKE_PROFIT:
            if order.side == OrderSide.SELL:
                if bar.high >= order.price:
                    fill_price = order.price
                    return fill_price, Decimal("0")
            else:
                if bar.low <= order.price:
                    fill_price = order.price
                    return fill_price, Decimal("0")

        return None, Decimal("0")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        for i, order in enumerate(self.pending_orders):
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                self.filled_orders.append(order)
                self.pending_orders.pop(i)
                return True
        return False

    def cancel_all(self, symbol: Optional[str] = None):
        """Cancel all pending orders."""
        for order in self.pending_orders:
            if symbol is None or order.symbol == symbol:
                order.status = OrderStatus.CANCELLED
                self.filled_orders.append(order)

        if symbol:
            self.pending_orders = [o for o in self.pending_orders if o.symbol != symbol]
        else:
            self.pending_orders = []


class BaseStrategy:
    """Base class for backtesting strategies."""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.parameters: dict[str, Any] = {}

    def initialize(self, context: "BacktestContext"):
        """Initialize strategy with context."""
        pass

    def on_bar(self, context: "BacktestContext", bar: OHLCV):
        """Called on each bar."""
        pass

    def on_trade(self, context: "BacktestContext", trade: Trade):
        """Called when trade is executed."""
        pass

    def on_order(self, context: "BacktestContext", order: Order):
        """Called when order status changes."""
        pass

    def finalize(self, context: "BacktestContext"):
        """Called at end of backtest."""
        pass


@dataclass
class BacktestContext:
    """Context passed to strategy during backtest."""
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    positions: dict[str, Position]
    current_prices: dict[str, Decimal]
    bars_history: dict[str, list[OHLCV]]

    # Methods for strategy to use
    order_manager: Optional[Any] = None
    position_manager: Optional[Any] = None

    def buy(
        self,
        symbol: str,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None
    ) -> Order:
        """Place buy order."""
        return self.order_manager.create_order(
            timestamp=self.timestamp,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

    def sell(
        self,
        symbol: str,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None
    ) -> Order:
        """Place sell order."""
        return self.order_manager.create_order(
            timestamp=self.timestamp,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close position for symbol."""
        position = self.positions.get(symbol)
        if not position or position.quantity == 0:
            return None

        if position.side == PositionSide.LONG:
            return self.sell(symbol, position.quantity)
        else:
            return self.buy(symbol, position.quantity)

    def get_position_value(self, symbol: str) -> Decimal:
        """Get current position value."""
        position = self.positions.get(symbol)
        if not position:
            return Decimal("0")

        price = self.current_prices.get(symbol, Decimal("0"))
        return position.quantity * price


class StrategyBacktester:
    """Main backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data: dict[str, list[OHLCV]] = {}
        self.strategies: list[BaseStrategy] = []
        self.callbacks: dict[str, list[Callable]] = {
            "on_bar": [],
            "on_trade": [],
            "on_equity_update": []
        }

    def add_data(self, symbol: str, bars: list[OHLCV]):
        """Add historical data for symbol."""
        self.data[symbol] = sorted(bars, key=lambda b: b.timestamp)

    def add_strategy(self, strategy: BaseStrategy):
        """Add strategy to backtest."""
        self.strategies.append(strategy)

    def register_callback(self, event: str, callback: Callable):
        """Register callback for event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def run(self) -> BacktestResult:
        """Run backtest."""
        if not self.data:
            raise ValueError("No data loaded")

        if not self.strategies:
            raise ValueError("No strategies added")

        # Initialize components
        slippage_calc = SlippageCalculator(
            self.config.slippage_model,
            self.config.slippage_rate
        )
        commission_calc = CommissionCalculator(self.config.commission_rate)
        order_manager = OrderManager(slippage_calc, commission_calc, self.config.fill_model)
        position_manager = PositionManager(self.config.allow_shorting)

        # Initialize state
        cash = self.config.initial_capital
        equity_curve: list[EquityPoint] = []
        trades: list[Trade] = []
        trade_counter = 0

        # Get all timestamps
        all_bars = []
        for symbol, bars in self.data.items():
            for bar in bars:
                all_bars.append((bar.timestamp, symbol, bar))
        all_bars.sort(key=lambda x: x[0])

        # Create context
        context = BacktestContext(
            timestamp=all_bars[0][0] if all_bars else datetime.now(),
            cash=cash,
            equity=cash,
            positions={},
            current_prices={},
            bars_history={s: [] for s in self.data.keys()},
            order_manager=order_manager,
            position_manager=position_manager
        )

        # Initialize strategies
        for strategy in self.strategies:
            strategy.initialize(context)

        # Track equity high for drawdown
        equity_high = cash

        # Main loop
        for timestamp, symbol, bar in all_bars:
            context.timestamp = timestamp
            context.current_prices[symbol] = bar.close
            context.bars_history[symbol].append(bar)

            # Process pending orders
            fills = order_manager.process_orders(bar)

            for order, fill_price, slippage in fills:
                # Update position
                if order.side == OrderSide.BUY:
                    position_manager.open_position(
                        order.symbol,
                        PositionSide.LONG,
                        order.quantity,
                        fill_price,
                        timestamp,
                        order.commission
                    )
                    cash -= fill_price * order.quantity + order.commission
                else:
                    pnl = position_manager.close_position(
                        order.symbol,
                        order.quantity,
                        fill_price,
                        timestamp,
                        order.commission
                    )
                    if pnl is None:
                        # Short sale
                        position_manager.open_position(
                            order.symbol,
                            PositionSide.SHORT,
                            order.quantity,
                            fill_price,
                            timestamp,
                            order.commission
                        )
                        cash += fill_price * order.quantity - order.commission
                    else:
                        cash += fill_price * order.quantity - order.commission

                # Create trade record
                trade_counter += 1
                trade = Trade(
                    id=f"TRD-{trade_counter:06d}",
                    timestamp=timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=order.commission,
                    slippage=slippage
                )
                trades.append(trade)

                # Notify strategies
                for strategy in self.strategies:
                    strategy.on_trade(context, trade)

                # Callbacks
                for cb in self.callbacks["on_trade"]:
                    cb(trade)

            # Update unrealized PnL
            for sym in position_manager.positions:
                if sym in context.current_prices:
                    position_manager.update_unrealized(sym, context.current_prices[sym])

            # Calculate equity
            positions_value = position_manager.get_total_value(context.current_prices)
            equity = cash + position_manager.get_total_unrealized_pnl()

            # Update context
            context.cash = cash
            context.equity = equity
            context.positions = dict(position_manager.positions)

            # Track drawdown
            equity_high = max(equity_high, equity)
            drawdown = equity_high - equity
            drawdown_pct = drawdown / equity_high if equity_high > 0 else Decimal("0")

            # Record equity point
            equity_point = EquityPoint(
                timestamp=timestamp,
                equity=equity,
                cash=cash,
                positions_value=positions_value,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct
            )
            equity_curve.append(equity_point)

            # Call strategies
            for strategy in self.strategies:
                strategy.on_bar(context, bar)

            # Callbacks
            for cb in self.callbacks["on_bar"]:
                cb(bar)
            for cb in self.callbacks["on_equity_update"]:
                cb(equity_point)

        # Finalize strategies
        for strategy in self.strategies:
            strategy.finalize(context)

        # Calculate metrics
        metrics = self._calculate_metrics(
            equity_curve,
            trades,
            self.config.initial_capital
        )

        # Build result
        start_date = all_bars[0][0] if all_bars else datetime.now()
        end_date = all_bars[-1][0] if all_bars else datetime.now()

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            orders=order_manager.filled_orders,
            positions_history=position_manager.position_history,
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            symbols=list(self.data.keys())
        )

    def _calculate_metrics(
        self,
        equity_curve: list[EquityPoint],
        trades: list[Trade],
        initial_capital: Decimal
    ) -> BacktestMetrics:
        """Calculate comprehensive metrics."""
        if not equity_curve:
            return self._empty_metrics()

        # Returns
        final_equity = equity_curve[-1].equity
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * Decimal("100")

        # Annualized return
        if len(equity_curve) > 1:
            days = (equity_curve[-1].timestamp - equity_curve[0].timestamp).days
            if days > 0:
                annual_factor = Decimal("365") / Decimal(str(days))
                annualized_return = (
                    (final_equity / initial_capital) ** annual_factor - Decimal("1")
                ) * Decimal("100")
            else:
                annualized_return = Decimal("0")
        else:
            annualized_return = Decimal("0")

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev = float(equity_curve[i-1].equity)
            curr = float(equity_curve[i].equity)
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        # Volatility
        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            volatility = Decimal(str(math.sqrt(variance) * math.sqrt(252)))  # Annualized
        else:
            volatility = Decimal("0")

        # Sharpe ratio
        risk_free_daily = float(self.config.risk_free_rate) / 252
        if daily_returns and float(volatility) > 0:
            excess_returns = [r - risk_free_daily for r in daily_returns]
            sharpe = Decimal(str(
                (sum(excess_returns) / len(excess_returns)) /
                (float(volatility) / math.sqrt(252))
            ))
        else:
            sharpe = Decimal("0")

        # Sortino ratio (downside deviation)
        if daily_returns:
            negative_returns = [r for r in daily_returns if r < risk_free_daily]
            if negative_returns:
                downside_var = sum((r - risk_free_daily) ** 2 for r in negative_returns) / len(negative_returns)
                downside_dev = math.sqrt(downside_var) * math.sqrt(252)
                if downside_dev > 0:
                    sortino = Decimal(str(float(annualized_return) / 100 / downside_dev))
                else:
                    sortino = Decimal("0")
            else:
                sortino = Decimal("999")  # No downside
        else:
            sortino = Decimal("0")

        # Max drawdown
        max_dd = max(e.drawdown for e in equity_curve)
        max_dd_pct = max(e.drawdown_pct for e in equity_curve) * Decimal("100")

        # Max drawdown duration
        max_dd_duration = 0
        current_dd_start = None
        for ep in equity_curve:
            if ep.drawdown > 0:
                if current_dd_start is None:
                    current_dd_start = ep.timestamp
            else:
                if current_dd_start:
                    duration = (ep.timestamp - current_dd_start).days
                    max_dd_duration = max(max_dd_duration, duration)
                    current_dd_start = None

        # Calmar ratio
        if float(max_dd_pct) > 0:
            calmar = annualized_return / max_dd_pct
        else:
            calmar = Decimal("0")

        # Trade statistics
        winning_trades = []
        losing_trades = []
        total_commission = Decimal("0")
        total_slippage = Decimal("0")

        for trade in trades:
            total_commission += trade.commission
            total_slippage += trade.slippage
            if trade.pnl > 0:
                winning_trades.append(trade)
            elif trade.pnl < 0:
                losing_trades.append(trade)

        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        win_rate = Decimal(str(num_winning / total_trades * 100)) if total_trades > 0 else Decimal("0")

        avg_win = (
            sum(t.pnl for t in winning_trades) / Decimal(str(num_winning))
            if num_winning > 0 else Decimal("0")
        )
        avg_loss = (
            abs(sum(t.pnl for t in losing_trades)) / Decimal(str(num_losing))
            if num_losing > 0 else Decimal("0")
        )

        largest_win = max((t.pnl for t in trades), default=Decimal("0"))
        largest_loss = min((t.pnl for t in trades), default=Decimal("0"))

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Average trade duration
        if trades:
            # Simplified - would need proper tracking in real implementation
            avg_duration = 24.0  # placeholder
        else:
            avg_duration = 0.0

        # Exposure time
        exposed_bars = sum(1 for e in equity_curve if e.positions_value > 0)
        exposure_time = Decimal(str(exposed_bars / len(equity_curve) * 100)) if equity_curve else Decimal("0")

        # Recovery factor
        if float(max_dd) > 0:
            recovery_factor = total_return / max_dd
        else:
            recovery_factor = Decimal("0")

        # Expectancy
        if total_trades > 0:
            expectancy = total_return / Decimal(str(total_trades))
        else:
            expectancy = Decimal("0")

        # SQN (System Quality Number)
        if daily_returns and len(daily_returns) > 1:
            mean_r = sum(daily_returns) / len(daily_returns)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1))
            if std_r > 0:
                sqn = Decimal(str((mean_r / std_r) * math.sqrt(len(daily_returns))))
            else:
                sqn = Decimal("0")
        else:
            sqn = Decimal("0")

        return BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=total_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            avg_trade_duration=avg_duration,
            total_commission=total_commission,
            total_slippage=total_slippage,
            exposure_time=exposure_time,
            recovery_factor=recovery_factor,
            expectancy=expectancy,
            sqn=sqn
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics."""
        return BacktestMetrics(
            total_return=Decimal("0"),
            total_return_pct=Decimal("0"),
            annualized_return=Decimal("0"),
            volatility=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            max_drawdown_duration=0,
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            avg_win=Decimal("0"),
            avg_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_trade_duration=0.0,
            total_commission=Decimal("0"),
            total_slippage=Decimal("0"),
            exposure_time=Decimal("0"),
            recovery_factor=Decimal("0"),
            expectancy=Decimal("0"),
            sqn=Decimal("0")
        )


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters."""

    def __init__(
        self,
        backtester: StrategyBacktester,
        in_sample_pct: float = 0.7,
        num_folds: int = 5
    ):
        self.backtester = backtester
        self.in_sample_pct = in_sample_pct
        self.num_folds = num_folds
        self.results: list[dict] = []

    def optimize(
        self,
        strategy_class: type,
        param_grid: dict[str, list[Any]],
        objective: str = "sharpe_ratio"
    ) -> dict[str, Any]:
        """Run walk-forward optimization."""
        # Get all timestamps
        all_timestamps = set()
        for bars in self.backtester.data.values():
            for bar in bars:
                all_timestamps.add(bar.timestamp)
        sorted_timestamps = sorted(all_timestamps)

        if len(sorted_timestamps) < self.num_folds * 10:
            raise ValueError("Not enough data for walk-forward optimization")

        # Split into folds
        fold_size = len(sorted_timestamps) // self.num_folds
        best_params: dict[str, Any] = {}
        best_score = float("-inf")

        # Generate parameter combinations
        param_combinations = self._generate_combinations(param_grid)

        for fold in range(self.num_folds):
            fold_start = fold * fold_size
            fold_end = (fold + 1) * fold_size if fold < self.num_folds - 1 else len(sorted_timestamps)

            # Split into in-sample and out-of-sample
            is_end = fold_start + int((fold_end - fold_start) * self.in_sample_pct)
            in_sample_dates = set(sorted_timestamps[fold_start:is_end])
            out_sample_dates = set(sorted_timestamps[is_end:fold_end])

            # Test each parameter combination on in-sample
            fold_best_params = None
            fold_best_score = float("-inf")

            for params in param_combinations:
                score = self._evaluate_params(
                    strategy_class,
                    params,
                    in_sample_dates,
                    objective
                )
                if score > fold_best_score:
                    fold_best_score = score
                    fold_best_params = params

            # Evaluate best params on out-of-sample
            if fold_best_params:
                oos_score = self._evaluate_params(
                    strategy_class,
                    fold_best_params,
                    out_sample_dates,
                    objective
                )

                self.results.append({
                    "fold": fold,
                    "params": fold_best_params,
                    "in_sample_score": fold_best_score,
                    "out_of_sample_score": oos_score
                })

                if oos_score > best_score:
                    best_score = oos_score
                    best_params = fold_best_params

        return best_params

    def _generate_combinations(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        self._recursive_combine(keys, values, 0, {}, combinations)
        return combinations

    def _recursive_combine(
        self,
        keys: list[str],
        values: list[list[Any]],
        index: int,
        current: dict[str, Any],
        results: list[dict[str, Any]]
    ):
        """Recursively generate combinations."""
        if index == len(keys):
            results.append(dict(current))
            return

        for value in values[index]:
            current[keys[index]] = value
            self._recursive_combine(keys, values, index + 1, current, results)

    def _evaluate_params(
        self,
        strategy_class: type,
        params: dict[str, Any],
        dates: set[datetime],
        objective: str
    ) -> float:
        """Evaluate parameters on subset of data."""
        # Filter data to dates
        filtered_data = {}
        for symbol, bars in self.backtester.data.items():
            filtered = [b for b in bars if b.timestamp in dates]
            if filtered:
                filtered_data[symbol] = filtered

        if not filtered_data:
            return float("-inf")

        # Create new backtester with filtered data
        bt = StrategyBacktester(self.backtester.config)
        for symbol, bars in filtered_data.items():
            bt.add_data(symbol, bars)

        # Create strategy with params
        strategy = strategy_class(**params)
        bt.add_strategy(strategy)

        try:
            result = bt.run()
            return float(getattr(result.metrics, objective, 0))
        except Exception:
            return float("-inf")


class MonteCarloSimulator:
    """Monte Carlo simulation for backtest analysis."""

    def __init__(self, backtest_result: BacktestResult, num_simulations: int = 1000):
        self.result = backtest_result
        self.num_simulations = num_simulations
        self.simulations: list[list[Decimal]] = []

    def run_trade_shuffle(self) -> dict:
        """Run Monte Carlo with shuffled trade sequence."""
        if not self.result.trades:
            return {}

        trade_pnls = [t.pnl for t in self.result.trades]
        initial_equity = float(self.result.config.initial_capital)

        final_equities = []
        max_drawdowns = []

        for _ in range(self.num_simulations):
            shuffled = trade_pnls.copy()
            random.shuffle(shuffled)

            equity = initial_equity
            peak = equity
            max_dd = 0.0
            curve = [equity]

            for pnl in shuffled:
                equity += float(pnl)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                curve.append(equity)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            self.simulations.append([Decimal(str(e)) for e in curve])

        return {
            "median_equity": Decimal(str(sorted(final_equities)[len(final_equities) // 2])),
            "percentile_5": Decimal(str(sorted(final_equities)[int(len(final_equities) * 0.05)])),
            "percentile_95": Decimal(str(sorted(final_equities)[int(len(final_equities) * 0.95)])),
            "median_max_dd": Decimal(str(sorted(max_drawdowns)[len(max_drawdowns) // 2])),
            "worst_max_dd": Decimal(str(max(max_drawdowns))),
            "probability_profit": Decimal(str(sum(1 for e in final_equities if e > initial_equity) / len(final_equities)))
        }

    def run_bootstrap(self, block_size: int = 20) -> dict:
        """Run block bootstrap simulation."""
        if len(self.result.equity_curve) < block_size:
            return {}

        # Calculate returns
        returns = []
        for i in range(1, len(self.result.equity_curve)):
            prev = float(self.result.equity_curve[i-1].equity)
            curr = float(self.result.equity_curve[i].equity)
            if prev > 0:
                returns.append(curr / prev - 1)

        if len(returns) < block_size:
            return {}

        initial_equity = float(self.result.config.initial_capital)
        final_equities = []

        for _ in range(self.num_simulations):
            # Sample blocks with replacement
            equity = initial_equity
            remaining = len(returns)

            while remaining > 0:
                start = random.randint(0, len(returns) - block_size)
                block = returns[start:start + min(block_size, remaining)]

                for r in block:
                    equity *= (1 + r)

                remaining -= len(block)

            final_equities.append(equity)

        return {
            "median_equity": Decimal(str(sorted(final_equities)[len(final_equities) // 2])),
            "percentile_5": Decimal(str(sorted(final_equities)[int(len(final_equities) * 0.05)])),
            "percentile_95": Decimal(str(sorted(final_equities)[int(len(final_equities) * 0.95)])),
            "std_final_equity": Decimal(str(
                math.sqrt(sum((e - sum(final_equities)/len(final_equities))**2 for e in final_equities) / len(final_equities))
            ))
        }


class PerformanceAnalyzer:
    """Analyze backtest performance in detail."""

    def __init__(self, result: BacktestResult):
        self.result = result

    def monthly_returns(self) -> dict[str, Decimal]:
        """Calculate monthly returns."""
        if not self.result.equity_curve:
            return {}

        monthly: dict[str, list[EquityPoint]] = {}
        for ep in self.result.equity_curve:
            key = ep.timestamp.strftime("%Y-%m")
            if key not in monthly:
                monthly[key] = []
            monthly[key].append(ep)

        returns = {}
        prev_equity = self.result.config.initial_capital

        for month in sorted(monthly.keys()):
            points = monthly[month]
            end_equity = points[-1].equity
            ret = (end_equity - prev_equity) / prev_equity * Decimal("100")
            returns[month] = ret
            prev_equity = end_equity

        return returns

    def yearly_returns(self) -> dict[str, Decimal]:
        """Calculate yearly returns."""
        if not self.result.equity_curve:
            return {}

        yearly: dict[str, list[EquityPoint]] = {}
        for ep in self.result.equity_curve:
            key = ep.timestamp.strftime("%Y")
            if key not in yearly:
                yearly[key] = []
            yearly[key].append(ep)

        returns = {}
        prev_equity = self.result.config.initial_capital

        for year in sorted(yearly.keys()):
            points = yearly[year]
            end_equity = points[-1].equity
            ret = (end_equity - prev_equity) / prev_equity * Decimal("100")
            returns[year] = ret
            prev_equity = end_equity

        return returns

    def drawdown_analysis(self) -> list[dict]:
        """Analyze drawdown periods."""
        if not self.result.equity_curve:
            return []

        drawdowns = []
        in_drawdown = False
        dd_start = None
        dd_peak = Decimal("0")

        for ep in self.result.equity_curve:
            if ep.drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = ep.timestamp
                    dd_peak = ep.equity + ep.drawdown
            else:
                if in_drawdown:
                    in_drawdown = False
                    drawdowns.append({
                        "start": dd_start.isoformat() if dd_start else None,
                        "end": ep.timestamp.isoformat(),
                        "duration_days": (ep.timestamp - dd_start).days if dd_start else 0,
                        "max_drawdown": str(dd_peak - ep.equity),
                        "max_drawdown_pct": str((dd_peak - ep.equity) / dd_peak * 100) if dd_peak > 0 else "0"
                    })

        return sorted(drawdowns, key=lambda x: float(x["max_drawdown_pct"]), reverse=True)[:10]

    def trade_analysis(self) -> dict:
        """Detailed trade analysis."""
        if not self.result.trades:
            return {}

        by_side = {"buy": [], "sell": []}
        by_symbol: dict[str, list[Trade]] = {}

        for trade in self.result.trades:
            by_side[trade.side.value].append(trade)
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)

        return {
            "total_trades": len(self.result.trades),
            "buy_trades": len(by_side["buy"]),
            "sell_trades": len(by_side["sell"]),
            "trades_per_symbol": {s: len(t) for s, t in by_symbol.items()},
            "avg_trade_size": str(
                sum(t.quantity * t.price for t in self.result.trades) / Decimal(str(len(self.result.trades)))
            ),
            "total_volume": str(sum(t.quantity * t.price for t in self.result.trades))
        }

    def risk_metrics(self) -> dict:
        """Calculate additional risk metrics."""
        if not self.result.equity_curve:
            return {}

        returns = []
        for i in range(1, len(self.result.equity_curve)):
            prev = float(self.result.equity_curve[i-1].equity)
            curr = float(self.result.equity_curve[i].equity)
            if prev > 0:
                returns.append(curr / prev - 1)

        if not returns:
            return {}

        sorted_returns = sorted(returns)

        # VaR
        var_95 = sorted_returns[int(len(returns) * 0.05)]
        var_99 = sorted_returns[int(len(returns) * 0.01)]

        # CVaR (Expected Shortfall)
        cvar_95_returns = sorted_returns[:int(len(returns) * 0.05)]
        cvar_95 = sum(cvar_95_returns) / len(cvar_95_returns) if cvar_95_returns else 0

        # Skewness
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        if std_r > 0:
            skewness = sum((r - mean_r) ** 3 for r in returns) / (len(returns) * std_r ** 3)
            kurtosis = sum((r - mean_r) ** 4 for r in returns) / (len(returns) * std_r ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0

        return {
            "var_95": Decimal(str(var_95 * 100)),
            "var_99": Decimal(str(var_99 * 100)),
            "cvar_95": Decimal(str(cvar_95 * 100)),
            "skewness": Decimal(str(skewness)),
            "kurtosis": Decimal(str(kurtosis)),
            "positive_days": sum(1 for r in returns if r > 0),
            "negative_days": sum(1 for r in returns if r < 0),
            "best_day": Decimal(str(max(returns) * 100)),
            "worst_day": Decimal(str(min(returns) * 100))
        }


# Global instance
_backtester: Optional[StrategyBacktester] = None


def get_backtester() -> StrategyBacktester:
    """Get global backtester instance."""
    global _backtester
    if _backtester is None:
        _backtester = StrategyBacktester()
    return _backtester


def set_backtester(backtester: StrategyBacktester):
    """Set global backtester instance."""
    global _backtester
    _backtester = backtester
