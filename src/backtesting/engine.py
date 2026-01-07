"""
Backtesting Engine Module

Professional backtesting engine for strategy evaluation with
realistic market simulation and comprehensive metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable
import math


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
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


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }


@dataclass
class BacktestOrder:
    """Order for backtesting."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    filled_price: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "filled_price": str(self.filled_price),
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "commission": str(self.commission),
        }


@dataclass
class Trade:
    """Executed trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal
    pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "commission": str(self.commission),
            "pnl": str(self.pnl),
        }


@dataclass
class Position:
    """Current position."""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
        }


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: Decimal = Decimal("10000")
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    leverage: float = 1.0
    margin_rate: float = 0.1
    allow_shorting: bool = True
    use_slippage: bool = True
    use_commission: bool = True
    fill_at_next_bar: bool = False
    position_sizing: str = "fixed"  # "fixed", "percent", "risk"
    max_position_pct: float = 1.0  # Max position as % of equity

    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_rate < 0:
            raise ValueError("commission_rate cannot be negative")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "initial_capital": str(self.initial_capital),
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "leverage": self.leverage,
            "margin_rate": self.margin_rate,
            "allow_shorting": self.allow_shorting,
            "use_slippage": self.use_slippage,
            "use_commission": self.use_commission,
            "fill_at_next_bar": self.fill_at_next_bar,
            "position_sizing": self.position_sizing,
            "max_position_pct": self.max_position_pct,
        }


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics."""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: Decimal = Decimal("0")
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_commission": str(self.total_commission),
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
        }


@dataclass
class EquityCurve:
    """Equity curve data point."""
    timestamp: datetime
    equity: Decimal
    drawdown: float
    returns: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": str(self.equity),
            "drawdown": self.drawdown,
            "returns": self.returns,
        }


class OrderManager:
    """Manages orders during backtesting."""

    def __init__(self, config: BacktestConfig):
        """Initialize order manager."""
        self.config = config
        self.orders: dict[str, BacktestOrder] = {}
        self.pending_orders: list[str] = []
        self.order_counter = 0

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
    ) -> BacktestOrder:
        """Create a new order."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        order = BacktestOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            created_at=timestamp or datetime.now(),
        )
        self.orders[order_id] = order
        self.pending_orders.append(order_id)
        return order

    def fill_order(
        self,
        order_id: str,
        fill_price: Decimal,
        fill_quantity: Decimal,
        timestamp: datetime,
    ) -> Optional[BacktestOrder]:
        """Fill an order."""
        if order_id not in self.orders:
            return None
        order = self.orders[order_id]
        # Apply slippage
        if self.config.use_slippage:
            slippage = fill_price * Decimal(str(self.config.slippage_rate))
            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage
        order.filled_quantity = fill_quantity
        order.filled_price = fill_price
        order.filled_at = timestamp
        # Calculate commission
        if self.config.use_commission:
            order.commission = fill_price * fill_quantity * Decimal(str(self.config.commission_rate))
        if fill_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        if order_id in self.pending_orders:
            self.pending_orders.remove(order_id)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            if order_id in self.pending_orders:
                self.pending_orders.remove(order_id)
            return True
        return False

    def get_pending_orders(self) -> list[BacktestOrder]:
        """Get pending orders."""
        return [self.orders[oid] for oid in self.pending_orders if oid in self.orders]

    def process_pending_orders(
        self,
        bar: OHLCV,
        current_price: Decimal,
        timestamp: datetime,
    ) -> list[BacktestOrder]:
        """Process pending orders against current bar."""
        filled = []
        for order_id in self.pending_orders.copy():
            order = self.orders[order_id]
            fill_price = None
            if order.order_type == OrderType.MARKET:
                fill_price = current_price
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.price >= bar.low:
                    fill_price = min(order.price, current_price)
                elif order.side == OrderSide.SELL and order.price <= bar.high:
                    fill_price = max(order.price, current_price)
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                    fill_price = max(order.stop_price, current_price)
                elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                    fill_price = min(order.stop_price, current_price)
            if fill_price:
                self.fill_order(order_id, fill_price, order.quantity, timestamp)
                filled.append(order)
        return filled


class PositionManager:
    """Manages positions during backtesting."""

    def __init__(self, config: BacktestConfig):
        """Initialize position manager."""
        self.config = config
        self.positions: dict[str, Position] = {}

    def update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[Decimal, Position]:
        """Update position after trade."""
        realized_pnl = Decimal("0")
        if symbol not in self.positions:
            # New position
            pos_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
            )
        else:
            position = self.positions[symbol]
            if side == OrderSide.BUY:
                if position.side == PositionSide.LONG:
                    # Add to long
                    total_cost = position.entry_price * position.quantity + price * quantity
                    position.quantity += quantity
                    position.entry_price = total_cost / position.quantity
                elif position.side == PositionSide.SHORT:
                    # Close short or flip
                    if quantity >= position.quantity:
                        # Close short, possibly go long
                        realized_pnl = (position.entry_price - price) * position.quantity
                        remaining = quantity - position.quantity
                        if remaining > 0:
                            position.side = PositionSide.LONG
                            position.quantity = remaining
                            position.entry_price = price
                        else:
                            position.side = PositionSide.FLAT
                            position.quantity = Decimal("0")
                    else:
                        # Reduce short
                        realized_pnl = (position.entry_price - price) * quantity
                        position.quantity -= quantity
                else:
                    # From flat to long
                    position.side = PositionSide.LONG
                    position.quantity = quantity
                    position.entry_price = price
            else:  # SELL
                if position.side == PositionSide.SHORT:
                    # Add to short
                    total_cost = position.entry_price * position.quantity + price * quantity
                    position.quantity += quantity
                    position.entry_price = total_cost / position.quantity
                elif position.side == PositionSide.LONG:
                    # Close long or flip
                    if quantity >= position.quantity:
                        # Close long, possibly go short
                        realized_pnl = (price - position.entry_price) * position.quantity
                        remaining = quantity - position.quantity
                        if remaining > 0 and self.config.allow_shorting:
                            position.side = PositionSide.SHORT
                            position.quantity = remaining
                            position.entry_price = price
                        else:
                            position.side = PositionSide.FLAT
                            position.quantity = Decimal("0")
                    else:
                        # Reduce long
                        realized_pnl = (price - position.entry_price) * quantity
                        position.quantity -= quantity
                else:
                    # From flat to short
                    if self.config.allow_shorting:
                        position.side = PositionSide.SHORT
                        position.quantity = quantity
                        position.entry_price = price
            position.realized_pnl += realized_pnl
        return realized_pnl, self.positions[symbol]

    def update_unrealized_pnl(self, symbol: str, current_price: Decimal) -> None:
        """Update unrealized PnL for a position."""
        if symbol not in self.positions:
            return
        position = self.positions[symbol]
        position.current_price = current_price
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        elif position.side == PositionSide.SHORT:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        else:
            position.unrealized_pnl = Decimal("0")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_total_realized_pnl(self) -> Decimal:
        """Get total realized PnL."""
        return sum(p.realized_pnl for p in self.positions.values())

    def close_all_positions(self, prices: dict[str, Decimal]) -> Decimal:
        """Close all positions at given prices."""
        total_pnl = Decimal("0")
        for symbol, position in self.positions.items():
            if position.side != PositionSide.FLAT:
                price = prices.get(symbol, position.current_price)
                if position.side == PositionSide.LONG:
                    pnl = (price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - price) * position.quantity
                total_pnl += pnl
                position.realized_pnl += pnl
                position.unrealized_pnl = Decimal("0")
                position.side = PositionSide.FLAT
                position.quantity = Decimal("0")
        return total_pnl


class MetricsCalculator:
    """Calculates backtesting metrics."""

    @staticmethod
    def calculate_returns(equity_curve: list[EquityCurve]) -> list[float]:
        """Calculate period returns from equity curve."""
        if len(equity_curve) < 2:
            return []
        returns = []
        for i in range(1, len(equity_curve)):
            prev_eq = float(equity_curve[i-1].equity)
            curr_eq = float(equity_curve[i].equity)
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)
        return returns

    @staticmethod
    def calculate_sharpe_ratio(
        returns: list[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1))
        if std_dev == 0:
            return 0.0
        excess_return = mean_return - risk_free_rate / periods_per_year
        return (excess_return / std_dev) * math.sqrt(periods_per_year)

    @staticmethod
    def calculate_sortino_ratio(
        returns: list[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sortino ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0
        downside_std = math.sqrt(sum(r ** 2 for r in negative_returns) / len(negative_returns))
        if downside_std == 0:
            return 0.0
        excess_return = mean_return - risk_free_rate / periods_per_year
        return (excess_return / downside_std) * math.sqrt(periods_per_year)

    @staticmethod
    def calculate_max_drawdown(equity_curve: list[EquityCurve]) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not equity_curve:
            return 0.0, 0
        peak = float(equity_curve[0].equity)
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        for i, point in enumerate(equity_curve):
            equity = float(point.equity)
            if equity > peak:
                peak = equity
                current_dd_start = i
            drawdown = (peak - equity) / peak if peak > 0 else 0
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_duration = i - current_dd_start
        return max_dd, max_dd_duration

    @staticmethod
    def calculate_trade_metrics(trades: list[Trade]) -> dict:
        """Calculate trade-based metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }
        pnls = [float(t.pnl) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
        }


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize backtesting engine."""
        self.config = config or BacktestConfig()
        self.order_manager = OrderManager(self.config)
        self.position_manager = PositionManager(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityCurve] = []
        self.trade_counter = 0
        self.current_bar: Optional[OHLCV] = None
        self.bar_index = 0
        self.data: dict[str, list[OHLCV]] = {}
        self.strategy_callback: Optional[Callable] = None

    def load_data(self, symbol: str, data: list[OHLCV]) -> None:
        """Load OHLCV data for a symbol."""
        self.data[symbol] = data

    def set_strategy(self, callback: Callable) -> None:
        """Set strategy callback function."""
        self.strategy_callback = callback

    def buy(
        self,
        symbol: str,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
    ) -> BacktestOrder:
        """Place a buy order."""
        return self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            price=price,
            timestamp=self.current_bar.timestamp if self.current_bar else datetime.now(),
        )

    def sell(
        self,
        symbol: str,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
    ) -> BacktestOrder:
        """Place a sell order."""
        return self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price,
            timestamp=self.current_bar.timestamp if self.current_bar else datetime.now(),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.order_manager.cancel_order(order_id)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position."""
        return self.position_manager.get_position(symbol)

    def run(self, symbols: Optional[list[str]] = None) -> BacktestMetrics:
        """Run the backtest."""
        if not self.data:
            raise ValueError("No data loaded")
        symbols = symbols or list(self.data.keys())
        if not symbols:
            raise ValueError("No symbols to backtest")
        # Get the primary symbol's data
        primary_symbol = symbols[0]
        bars = self.data.get(primary_symbol, [])
        if not bars:
            raise ValueError(f"No data for symbol {primary_symbol}")
        # Initialize equity curve
        self.equity_curve = []
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades = []
        # Run through each bar
        for i, bar in enumerate(bars):
            self.bar_index = i
            self.current_bar = bar
            # Process pending orders
            filled_orders = self.order_manager.process_pending_orders(
                bar, bar.close, bar.timestamp
            )
            # Process filled orders
            for order in filled_orders:
                self._process_filled_order(order)
            # Update unrealized PnL
            for symbol in symbols:
                current_price = bar.close
                if symbol in self.data and i < len(self.data[symbol]):
                    current_price = self.data[symbol][i].close
                self.position_manager.update_unrealized_pnl(symbol, current_price)
            # Calculate equity
            unrealized = self.position_manager.get_total_unrealized_pnl()
            self.equity = self.cash + unrealized
            # Record equity curve
            prev_equity = self.equity_curve[-1].equity if self.equity_curve else self.config.initial_capital
            returns = float((self.equity - prev_equity) / prev_equity) if prev_equity > 0 else 0
            peak = max(float(e.equity) for e in self.equity_curve) if self.equity_curve else float(self.equity)
            drawdown = (peak - float(self.equity)) / peak if peak > 0 else 0
            self.equity_curve.append(EquityCurve(
                timestamp=bar.timestamp,
                equity=self.equity,
                drawdown=drawdown,
                returns=returns,
            ))
            # Execute strategy
            if self.strategy_callback:
                self.strategy_callback(self, bar, i)
        # Close all positions at end
        final_prices = {}
        for symbol in symbols:
            if symbol in self.data and self.data[symbol]:
                final_prices[symbol] = self.data[symbol][-1].close
        final_pnl = self.position_manager.close_all_positions(final_prices)
        self.cash += final_pnl
        self.equity = self.cash
        # Calculate metrics
        return self._calculate_final_metrics()

    def _process_filled_order(self, order: BacktestOrder) -> None:
        """Process a filled order."""
        realized_pnl, position = self.position_manager.update_position(
            order.symbol,
            order.side,
            order.filled_quantity,
            order.filled_price,
        )
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= order.filled_price * order.filled_quantity
        else:
            self.cash += order.filled_price * order.filled_quantity
        self.cash -= order.commission
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"trade_{self.trade_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            timestamp=order.filled_at or datetime.now(),
            commission=order.commission,
            pnl=realized_pnl,
        )
        self.trades.append(trade)

    def _calculate_final_metrics(self) -> BacktestMetrics:
        """Calculate final backtest metrics."""
        returns = self.metrics_calculator.calculate_returns(self.equity_curve)
        max_dd, max_dd_duration = self.metrics_calculator.calculate_max_drawdown(self.equity_curve)
        trade_metrics = self.metrics_calculator.calculate_trade_metrics(self.trades)
        total_return = float((self.equity - self.config.initial_capital) / self.config.initial_capital)
        # Calculate annualized return
        if self.equity_curve and len(self.equity_curve) > 1:
            days = (self.equity_curve[-1].timestamp - self.equity_curve[0].timestamp).days
            years = days / 365 if days > 0 else 1
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annual_return = 0
        # Calculate volatility
        volatility = math.sqrt(sum(r ** 2 for r in returns) / len(returns)) * math.sqrt(252) if returns else 0
        total_commission = sum(t.commission for t in self.trades)
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=self.metrics_calculator.calculate_sharpe_ratio(returns),
            sortino_ratio=self.metrics_calculator.calculate_sortino_ratio(returns),
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            avg_trade_pnl=trade_metrics["avg_pnl"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            total_trades=trade_metrics["total_trades"],
            winning_trades=trade_metrics["winning_trades"],
            losing_trades=trade_metrics["losing_trades"],
            total_commission=total_commission,
            calmar_ratio=annual_return / max_dd if max_dd > 0 else 0,
            volatility=volatility,
        )

    def get_results(self) -> dict:
        """Get backtest results."""
        return {
            "equity": str(self.equity),
            "cash": str(self.cash),
            "trades_count": len(self.trades),
            "trades": [t.to_dict() for t in self.trades[-10:]],  # Last 10 trades
            "positions": {s: p.to_dict() for s, p in self.position_manager.positions.items()},
            "equity_curve_length": len(self.equity_curve),
            "config": self.config.to_dict(),
        }
