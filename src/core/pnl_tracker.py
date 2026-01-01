"""PnL (Profit and Loss) tracking for trading operations."""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PnLType(Enum):
    """Types of PnL."""

    REALIZED = "realized"  # Closed positions
    UNREALIZED = "unrealized"  # Open positions
    FUNDING = "funding"  # Funding payments
    FEES = "fees"  # Trading fees
    TOTAL = "total"  # Combined


class TradeDirection(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecord:
    """Record of a single trade."""

    trade_id: str
    market: str
    direction: TradeDirection
    entry_price: Decimal
    entry_size: Decimal
    entry_time: float
    exit_price: Optional[Decimal] = None
    exit_size: Optional[Decimal] = None
    exit_time: Optional[float] = None
    fees_paid: Decimal = Decimal("0")
    funding_paid: Decimal = Decimal("0")
    is_closed: bool = False

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value at entry."""
        return self.entry_size * self.entry_price

    @property
    def holding_time(self) -> float:
        """Calculate holding time in seconds."""
        end_time = self.exit_time or time.time()
        return end_time - self.entry_time

    @property
    def holding_hours(self) -> float:
        """Calculate holding time in hours."""
        return self.holding_time / 3600

    @property
    def realized_pnl(self) -> Decimal:
        """Calculate realized PnL."""
        if not self.is_closed or not self.exit_price:
            return Decimal("0")

        size = self.exit_size or self.entry_size

        if self.direction == TradeDirection.LONG:
            pnl = (self.exit_price - self.entry_price) * size
        else:
            pnl = (self.entry_price - self.exit_price) * size

        return pnl - self.fees_paid - self.funding_paid

    @property
    def realized_pnl_pct(self) -> float:
        """Calculate realized PnL percentage."""
        if self.notional_value == 0:
            return 0.0
        return float(self.realized_pnl / self.notional_value * 100)

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL at current price."""
        if self.is_closed:
            return Decimal("0")

        if self.direction == TradeDirection.LONG:
            pnl = (current_price - self.entry_price) * self.entry_size
        else:
            pnl = (self.entry_price - current_price) * self.entry_size

        return pnl - self.fees_paid - self.funding_paid

    def calculate_unrealized_pnl_pct(self, current_price: Decimal) -> float:
        """Calculate unrealized PnL percentage."""
        if self.notional_value == 0:
            return 0.0
        return float(self.calculate_unrealized_pnl(current_price) / self.notional_value * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "market": self.market,
            "direction": self.direction.value,
            "entry_price": str(self.entry_price),
            "entry_size": str(self.entry_size),
            "entry_time": self.entry_time,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_time": self.exit_time,
            "fees_paid": str(self.fees_paid),
            "funding_paid": str(self.funding_paid),
            "is_closed": self.is_closed,
            "realized_pnl": str(self.realized_pnl),
            "holding_hours": round(self.holding_hours, 2),
        }


@dataclass
class PnLSnapshot:
    """Snapshot of PnL at a point in time."""

    timestamp: float
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees: Decimal
    funding: Decimal
    equity: Decimal
    open_positions: int

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_pnl(self) -> Decimal:
        """Calculate net PnL after fees and funding."""
        return self.total_pnl - self.fees - self.funding

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "fees": str(self.fees),
            "funding": str(self.funding),
            "equity": str(self.equity),
            "open_positions": self.open_positions,
        }


@dataclass
class DailyPnL:
    """Daily PnL summary."""

    date: str  # YYYY-MM-DD
    starting_equity: Decimal
    ending_equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees: Decimal
    funding: Decimal
    trades_count: int
    winners: int
    losers: int

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_pnl(self) -> Decimal:
        """Calculate net PnL."""
        return self.total_pnl - self.fees - self.funding

    @property
    def return_pct(self) -> float:
        """Calculate daily return percentage."""
        if self.starting_equity == 0:
            return 0.0
        return float((self.ending_equity - self.starting_equity) / self.starting_equity * 100)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winners + self.losers
        if total == 0:
            return 0.0
        return self.winners / total * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "starting_equity": str(self.starting_equity),
            "ending_equity": str(self.ending_equity),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "return_pct": round(self.return_pct, 4),
            "trades_count": self.trades_count,
            "win_rate": round(self.win_rate, 2),
        }


@dataclass
class PnLSummary:
    """Summary of PnL statistics."""

    total_realized: Decimal = Decimal("0")
    total_unrealized: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_funding: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    avg_holding_time: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL."""
        return self.total_realized + self.total_unrealized

    @property
    def net_pnl(self) -> Decimal:
        """Calculate net PnL."""
        return self.total_pnl - self.total_fees - self.total_funding

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total * 100

    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade."""
        if self.total_trades == 0:
            return Decimal("0")
        return self.net_pnl / self.total_trades

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "total_realized": str(self.total_realized),
            "total_unrealized": str(self.total_unrealized),
            "total_fees": str(self.total_fees),
            "total_funding": str(self.total_funding),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "expectancy": str(self.expectancy),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
        }


@dataclass
class PnLTracker:
    """Tracker for profit and loss.

    Features:
    - Real-time PnL calculation
    - Trade history management
    - Daily/Weekly/Monthly summaries
    - Drawdown tracking
    - Performance metrics
    """

    initial_equity: Decimal = Decimal("10000")
    _trades: Dict[str, TradeRecord] = field(default_factory=dict)
    _closed_trades: List[TradeRecord] = field(default_factory=list)
    _snapshots: List[PnLSnapshot] = field(default_factory=list)
    _daily_pnl: Dict[str, DailyPnL] = field(default_factory=dict)
    _equity_history: List[Tuple[float, Decimal]] = field(default_factory=list)
    _peak_equity: Decimal = field(default=Decimal("0"))
    _total_fees: Decimal = field(default=Decimal("0"))
    _total_funding: Decimal = field(default=Decimal("0"))

    def __post_init__(self):
        """Initialize tracker."""
        self._trades = {}
        self._closed_trades = []
        self._snapshots = []
        self._daily_pnl = {}
        self._equity_history = [(time.time(), self.initial_equity)]
        self._peak_equity = self.initial_equity
        self._total_fees = Decimal("0")
        self._total_funding = Decimal("0")

    def open_trade(
        self,
        trade_id: str,
        market: str,
        direction: TradeDirection,
        entry_price: Decimal,
        entry_size: Decimal,
        fees: Decimal = Decimal("0"),
    ) -> TradeRecord:
        """Record a new trade opening."""
        trade = TradeRecord(
            trade_id=trade_id,
            market=market,
            direction=direction,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_time=time.time(),
            fees_paid=fees,
        )

        self._trades[trade_id] = trade
        self._total_fees += fees

        logger.info(f"Opened trade {trade_id}: {direction.value} {entry_size} {market} @ {entry_price}")

        return trade

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_size: Optional[Decimal] = None,
        fees: Decimal = Decimal("0"),
    ) -> Optional[TradeRecord]:
        """Record a trade closing."""
        trade = self._trades.get(trade_id)
        if not trade:
            logger.warning(f"Trade {trade_id} not found")
            return None

        trade.exit_price = exit_price
        trade.exit_size = exit_size or trade.entry_size
        trade.exit_time = time.time()
        trade.fees_paid += fees
        trade.is_closed = True

        self._total_fees += fees

        # Move to closed trades
        del self._trades[trade_id]
        self._closed_trades.append(trade)

        logger.info(f"Closed trade {trade_id}: PnL = {trade.realized_pnl}")

        return trade

    def add_funding(
        self,
        trade_id: str,
        funding_amount: Decimal,
    ) -> None:
        """Record funding payment for a trade."""
        trade = self._trades.get(trade_id)
        if trade:
            trade.funding_paid += funding_amount
            self._total_funding += funding_amount

    def update_equity(
        self,
        current_equity: Decimal,
    ) -> None:
        """Update current equity value."""
        timestamp = time.time()
        self._equity_history.append((timestamp, current_equity))

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def get_unrealized_pnl(
        self,
        prices: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate total unrealized PnL."""
        total = Decimal("0")

        for trade in self._trades.values():
            price = prices.get(trade.market)
            if price:
                total += trade.calculate_unrealized_pnl(price)

        return total

    def get_realized_pnl(self) -> Decimal:
        """Calculate total realized PnL."""
        return sum(t.realized_pnl for t in self._closed_trades)

    def get_total_pnl(
        self,
        prices: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate total PnL (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl(prices)

    def get_net_pnl(
        self,
        prices: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate net PnL after fees and funding."""
        return self.get_total_pnl(prices) - self._total_fees - self._total_funding

    def get_current_equity(
        self,
        prices: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate current equity."""
        return self.initial_equity + self.get_net_pnl(prices)

    def get_drawdown(
        self,
        prices: Dict[str, Decimal],
    ) -> Tuple[Decimal, float]:
        """Calculate current drawdown.

        Returns:
            Tuple of (drawdown_amount, drawdown_percent)
        """
        current = self.get_current_equity(prices)
        drawdown = self._peak_equity - current

        if self._peak_equity > 0:
            drawdown_pct = float(drawdown / self._peak_equity * 100)
        else:
            drawdown_pct = 0.0

        return drawdown, drawdown_pct

    def get_max_drawdown(self) -> Tuple[Decimal, float]:
        """Calculate maximum drawdown from equity history."""
        if len(self._equity_history) < 2:
            return Decimal("0"), 0.0

        peak = Decimal("0")
        max_dd = Decimal("0")
        max_dd_pct = 0.0

        for _, equity in self._equity_history:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = float(drawdown / peak * 100) if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def take_snapshot(
        self,
        prices: Dict[str, Decimal],
    ) -> PnLSnapshot:
        """Take a snapshot of current PnL state."""
        snapshot = PnLSnapshot(
            timestamp=time.time(),
            realized_pnl=self.get_realized_pnl(),
            unrealized_pnl=self.get_unrealized_pnl(prices),
            fees=self._total_fees,
            funding=self._total_funding,
            equity=self.get_current_equity(prices),
            open_positions=len(self._trades),
        )

        self._snapshots.append(snapshot)
        return snapshot

    def get_summary(
        self,
        prices: Dict[str, Decimal],
    ) -> PnLSummary:
        """Get complete PnL summary."""
        # Calculate win/loss statistics
        winners = [t for t in self._closed_trades if t.realized_pnl > 0]
        losers = [t for t in self._closed_trades if t.realized_pnl < 0]

        total_wins = sum(t.realized_pnl for t in winners)
        total_losses = abs(sum(t.realized_pnl for t in losers))

        avg_win = total_wins / len(winners) if winners else Decimal("0")
        avg_loss = total_losses / len(losers) if losers else Decimal("0")

        largest_win = max((t.realized_pnl for t in winners), default=Decimal("0"))
        largest_loss = min((t.realized_pnl for t in losers), default=Decimal("0"))

        profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0.0

        avg_holding = 0.0
        if self._closed_trades:
            avg_holding = sum(t.holding_hours for t in self._closed_trades) / len(self._closed_trades)

        max_dd, max_dd_pct = self.get_max_drawdown()

        return PnLSummary(
            total_realized=self.get_realized_pnl(),
            total_unrealized=self.get_unrealized_pnl(prices),
            total_fees=self._total_fees,
            total_funding=self._total_funding,
            total_trades=len(self._closed_trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_time=avg_holding,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
        )

    def get_open_trades(self) -> List[TradeRecord]:
        """Get all open trades."""
        return list(self._trades.values())

    def get_closed_trades(self) -> List[TradeRecord]:
        """Get all closed trades."""
        return self._closed_trades.copy()

    def get_trades_by_market(self, market: str) -> List[TradeRecord]:
        """Get trades for a specific market."""
        all_trades = list(self._trades.values()) + self._closed_trades
        return [t for t in all_trades if t.market == market]

    def get_snapshots(self) -> List[PnLSnapshot]:
        """Get all snapshots."""
        return self._snapshots.copy()

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._trades.clear()
        self._closed_trades.clear()
        self._snapshots.clear()
        self._daily_pnl.clear()
        self._equity_history = [(time.time(), self.initial_equity)]
        self._peak_equity = self.initial_equity
        self._total_fees = Decimal("0")
        self._total_funding = Decimal("0")


@dataclass
class MultiAccountPnLTracker:
    """PnL tracker for multiple accounts."""

    _trackers: Dict[str, PnLTracker] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize trackers."""
        self._trackers = {}

    def add_account(
        self,
        account_id: str,
        initial_equity: Decimal = Decimal("10000"),
    ) -> PnLTracker:
        """Add account tracker."""
        tracker = PnLTracker(initial_equity=initial_equity)
        self._trackers[account_id] = tracker
        return tracker

    def get_tracker(self, account_id: str) -> Optional[PnLTracker]:
        """Get tracker for an account."""
        return self._trackers.get(account_id)

    def get_combined_pnl(
        self,
        prices: Dict[str, Decimal],
    ) -> Decimal:
        """Get combined PnL across all accounts."""
        return sum(t.get_total_pnl(prices) for t in self._trackers.values())

    def get_combined_summary(
        self,
        prices: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Get combined summary across all accounts."""
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")
        total_fees = Decimal("0")
        total_funding = Decimal("0")
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        for tracker in self._trackers.values():
            summary = tracker.get_summary(prices)
            total_realized += summary.total_realized
            total_unrealized += summary.total_unrealized
            total_fees += summary.total_fees
            total_funding += summary.total_funding
            total_trades += summary.total_trades
            winning_trades += summary.winning_trades
            losing_trades += summary.losing_trades

        return {
            "accounts": len(self._trackers),
            "total_pnl": str(total_realized + total_unrealized),
            "net_pnl": str(total_realized + total_unrealized - total_fees - total_funding),
            "total_fees": str(total_fees),
            "total_funding": str(total_funding),
            "total_trades": total_trades,
            "win_rate": round(winning_trades / (winning_trades + losing_trades) * 100, 2) if winning_trades + losing_trades > 0 else 0,
        }

    def get_delta_exposure(
        self,
        prices: Dict[str, Decimal],
    ) -> Dict[str, Decimal]:
        """Get delta exposure across accounts."""
        exposure = {}

        for account_id, tracker in self._trackers.items():
            for trade in tracker.get_open_trades():
                market = trade.market
                if market not in exposure:
                    exposure[market] = Decimal("0")

                price = prices.get(market, trade.entry_price)
                notional = trade.entry_size * price

                if trade.direction == TradeDirection.LONG:
                    exposure[market] += notional
                else:
                    exposure[market] -= notional

        return exposure


@dataclass
class PnLReporter:
    """Reporter for PnL data."""

    tracker: PnLTracker

    def generate_daily_report(
        self,
        date: str,
        prices: Dict[str, Decimal],
    ) -> DailyPnL:
        """Generate daily PnL report."""
        # Filter trades for the date
        day_trades = [
            t for t in self.tracker.get_closed_trades()
            if time.strftime("%Y-%m-%d", time.localtime(t.exit_time)) == date
        ]

        winners = [t for t in day_trades if t.realized_pnl > 0]
        losers = [t for t in day_trades if t.realized_pnl < 0]

        # Get starting and ending equity for the day
        # This is simplified - in production would track actual values
        ending_equity = self.tracker.get_current_equity(prices)
        realized = sum(t.realized_pnl for t in day_trades)
        starting_equity = ending_equity - realized

        return DailyPnL(
            date=date,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            realized_pnl=realized,
            unrealized_pnl=self.tracker.get_unrealized_pnl(prices),
            fees=sum(t.fees_paid for t in day_trades),
            funding=sum(t.funding_paid for t in day_trades),
            trades_count=len(day_trades),
            winners=len(winners),
            losers=len(losers),
        )

    def format_summary(
        self,
        prices: Dict[str, Decimal],
    ) -> str:
        """Format summary as text."""
        summary = self.tracker.get_summary(prices)

        lines = [
            "=== PnL Summary ===",
            f"Total PnL: ${summary.total_pnl}",
            f"Net PnL: ${summary.net_pnl}",
            f"Realized: ${summary.total_realized}",
            f"Unrealized: ${summary.total_unrealized}",
            "",
            f"Total Trades: {summary.total_trades}",
            f"Win Rate: {summary.win_rate:.1f}%",
            f"Profit Factor: {summary.profit_factor:.2f}",
            "",
            f"Largest Win: ${summary.largest_win}",
            f"Largest Loss: ${summary.largest_loss}",
            f"Avg Win: ${summary.avg_win}",
            f"Avg Loss: ${summary.avg_loss}",
            "",
            f"Max Drawdown: ${summary.max_drawdown} ({summary.max_drawdown_pct:.1f}%)",
            f"Total Fees: ${summary.total_fees}",
            f"Total Funding: ${summary.total_funding}",
        ]

        return "\n".join(lines)


# Global tracker instance
_global_tracker: Optional[PnLTracker] = None


def get_pnl_tracker() -> PnLTracker:
    """Get or create global PnL tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PnLTracker()
    return _global_tracker


def reset_pnl_tracker() -> None:
    """Reset global PnL tracker."""
    global _global_tracker
    _global_tracker = None
