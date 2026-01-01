"""Trade analytics for performance analysis."""

import logging
import statistics
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Time frame for analysis."""

    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    ALL_TIME = "all"


class TradeOutcome(Enum):
    """Trade outcome."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeData:
    """Data for a single trade."""

    trade_id: str
    market: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    entry_time: float
    exit_time: float
    pnl: Decimal
    fees: Decimal
    funding: Decimal = Decimal("0")

    @property
    def net_pnl(self) -> Decimal:
        """Calculate net PnL after fees."""
        return self.pnl - self.fees - self.funding

    @property
    def pnl_pct(self) -> float:
        """Calculate PnL percentage."""
        notional = self.entry_price * self.size
        if notional == 0:
            return 0.0
        return float(self.pnl / notional * 100)

    @property
    def holding_time(self) -> float:
        """Calculate holding time in hours."""
        return (self.exit_time - self.entry_time) / 3600

    @property
    def outcome(self) -> TradeOutcome:
        """Determine trade outcome."""
        if self.net_pnl > 0:
            return TradeOutcome.WIN
        elif self.net_pnl < 0:
            return TradeOutcome.LOSS
        else:
            return TradeOutcome.BREAKEVEN

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "market": self.market,
            "direction": self.direction,
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "size": str(self.size),
            "pnl": str(self.pnl),
            "net_pnl": str(self.net_pnl),
            "pnl_pct": round(self.pnl_pct, 4),
            "holding_time": round(self.holding_time, 2),
            "outcome": self.outcome.value,
        }


@dataclass
class TradeStatistics:
    """Statistics for a set of trades."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    avg_pnl_pct: float = 0.0
    avg_holding_time: float = 0.0
    longest_trade: float = 0.0
    shortest_trade: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return float(self.gross_profit / abs(self.gross_loss))

    @property
    def payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        if self.avg_loss == 0:
            return float('inf') if self.avg_win > 0 else 0.0
        return float(self.avg_win / abs(self.avg_loss))

    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade."""
        if self.total_trades == 0:
            return Decimal("0")
        return self.net_pnl / self.total_trades

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float('inf') else "inf",
            "payoff_ratio": round(self.payoff_ratio, 2) if self.payoff_ratio != float('inf') else "inf",
            "expectancy": str(self.expectancy),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "avg_holding_time": round(self.avg_holding_time, 2),
        }


@dataclass
class StreakAnalysis:
    """Analysis of winning/losing streaks."""

    current_streak: int = 0
    current_streak_type: Optional[TradeOutcome] = None
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    win_streaks: List[int] = field(default_factory=list)
    loss_streaks: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize lists."""
        if self.win_streaks is None:
            self.win_streaks = []
        if self.loss_streaks is None:
            self.loss_streaks = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_streak": self.current_streak,
            "current_streak_type": self.current_streak_type.value if self.current_streak_type else None,
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "avg_win_streak": round(self.avg_win_streak, 2),
            "avg_loss_streak": round(self.avg_loss_streak, 2),
        }


@dataclass
class TimeAnalysis:
    """Analysis of trading by time."""

    best_hour: Optional[int] = None
    worst_hour: Optional[int] = None
    best_day: Optional[str] = None
    worst_day: Optional[str] = None
    hourly_pnl: Dict[int, Decimal] = field(default_factory=dict)
    daily_pnl: Dict[str, Decimal] = field(default_factory=dict)
    hourly_trades: Dict[int, int] = field(default_factory=dict)
    daily_trades: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize dicts."""
        if self.hourly_pnl is None:
            self.hourly_pnl = {}
        if self.daily_pnl is None:
            self.daily_pnl = {}
        if self.hourly_trades is None:
            self.hourly_trades = {}
        if self.daily_trades is None:
            self.daily_trades = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "best_hour": self.best_hour,
            "worst_hour": self.worst_hour,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "hourly_pnl": {k: str(v) for k, v in self.hourly_pnl.items()},
            "daily_pnl": {k: str(v) for k, v in self.daily_pnl.items()},
        }


@dataclass
class MarketAnalysis:
    """Analysis of trading by market."""

    market_stats: Dict[str, TradeStatistics] = field(default_factory=dict)
    best_market: Optional[str] = None
    worst_market: Optional[str] = None
    most_traded: Optional[str] = None

    def __post_init__(self):
        """Initialize dict."""
        if self.market_stats is None:
            self.market_stats = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "best_market": self.best_market,
            "worst_market": self.worst_market,
            "most_traded": self.most_traded,
            "market_stats": {k: v.to_dict() for k, v in self.market_stats.items()},
        }


@dataclass
class TradeAnalyzer:
    """Analyzer for trade performance.

    Features:
    - Trade statistics calculation
    - Win/loss streak analysis
    - Time-based analysis
    - Market-based analysis
    - Performance patterns
    """

    _trades: List[TradeData] = field(default_factory=list)

    def __post_init__(self):
        """Initialize trades list."""
        self._trades = []

    def add_trade(self, trade: TradeData) -> None:
        """Add a trade for analysis."""
        self._trades.append(trade)

    def add_trades(self, trades: List[TradeData]) -> None:
        """Add multiple trades."""
        self._trades.extend(trades)

    def clear_trades(self) -> None:
        """Clear all trades."""
        self._trades.clear()

    def get_trades(
        self,
        market: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[TradeData]:
        """Get filtered trades."""
        trades = self._trades

        if market:
            trades = [t for t in trades if t.market == market]

        if start_time:
            trades = [t for t in trades if t.entry_time >= start_time]

        if end_time:
            trades = [t for t in trades if t.exit_time <= end_time]

        return trades

    def calculate_statistics(
        self,
        trades: Optional[List[TradeData]] = None,
    ) -> TradeStatistics:
        """Calculate trade statistics."""
        trades = trades or self._trades

        if not trades:
            return TradeStatistics()

        stats = TradeStatistics(total_trades=len(trades))

        winners = []
        losers = []

        for trade in trades:
            stats.total_pnl += trade.pnl
            stats.total_fees += trade.fees
            stats.net_pnl += trade.net_pnl

            if trade.outcome == TradeOutcome.WIN:
                stats.winning_trades += 1
                stats.gross_profit += trade.net_pnl
                winners.append(trade)

                if trade.net_pnl > stats.largest_win:
                    stats.largest_win = trade.net_pnl

            elif trade.outcome == TradeOutcome.LOSS:
                stats.losing_trades += 1
                stats.gross_loss += trade.net_pnl  # Negative value
                losers.append(trade)

                if trade.net_pnl < stats.largest_loss:
                    stats.largest_loss = trade.net_pnl

            else:
                stats.breakeven_trades += 1

        # Calculate averages
        if winners:
            stats.avg_win = stats.gross_profit / len(winners)

        if losers:
            stats.avg_loss = stats.gross_loss / len(losers)

        if trades:
            pnl_pcts = [t.pnl_pct for t in trades]
            stats.avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts)

            holding_times = [t.holding_time for t in trades]
            stats.avg_holding_time = sum(holding_times) / len(holding_times)
            stats.longest_trade = max(holding_times)
            stats.shortest_trade = min(holding_times)

        return stats

    def analyze_streaks(
        self,
        trades: Optional[List[TradeData]] = None,
    ) -> StreakAnalysis:
        """Analyze winning and losing streaks."""
        trades = trades or self._trades

        if not trades:
            return StreakAnalysis()

        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)

        analysis = StreakAnalysis()
        current_streak = 0
        current_type = None
        win_streaks = []
        loss_streaks = []

        for trade in sorted_trades:
            if trade.outcome == TradeOutcome.BREAKEVEN:
                continue

            if current_type == trade.outcome:
                current_streak += 1
            else:
                # Save previous streak
                if current_type == TradeOutcome.WIN and current_streak > 0:
                    win_streaks.append(current_streak)
                elif current_type == TradeOutcome.LOSS and current_streak > 0:
                    loss_streaks.append(current_streak)

                current_streak = 1
                current_type = trade.outcome

        # Save final streak
        if current_type == TradeOutcome.WIN and current_streak > 0:
            win_streaks.append(current_streak)
        elif current_type == TradeOutcome.LOSS and current_streak > 0:
            loss_streaks.append(current_streak)

        analysis.current_streak = current_streak
        analysis.current_streak_type = current_type
        analysis.win_streaks = win_streaks
        analysis.loss_streaks = loss_streaks

        if win_streaks:
            analysis.longest_win_streak = max(win_streaks)
            analysis.avg_win_streak = sum(win_streaks) / len(win_streaks)

        if loss_streaks:
            analysis.longest_loss_streak = max(loss_streaks)
            analysis.avg_loss_streak = sum(loss_streaks) / len(loss_streaks)

        return analysis

    def analyze_by_time(
        self,
        trades: Optional[List[TradeData]] = None,
    ) -> TimeAnalysis:
        """Analyze trades by time."""
        trades = trades or self._trades

        if not trades:
            return TimeAnalysis()

        analysis = TimeAnalysis()

        for trade in trades:
            # Get hour and day of week
            hour = int(time.strftime("%H", time.localtime(trade.entry_time)))
            day = time.strftime("%A", time.localtime(trade.entry_time))

            # Hourly stats
            if hour not in analysis.hourly_pnl:
                analysis.hourly_pnl[hour] = Decimal("0")
                analysis.hourly_trades[hour] = 0
            analysis.hourly_pnl[hour] += trade.net_pnl
            analysis.hourly_trades[hour] += 1

            # Daily stats
            if day not in analysis.daily_pnl:
                analysis.daily_pnl[day] = Decimal("0")
                analysis.daily_trades[day] = 0
            analysis.daily_pnl[day] += trade.net_pnl
            analysis.daily_trades[day] += 1

        # Find best/worst
        if analysis.hourly_pnl:
            analysis.best_hour = max(analysis.hourly_pnl, key=lambda h: analysis.hourly_pnl[h])
            analysis.worst_hour = min(analysis.hourly_pnl, key=lambda h: analysis.hourly_pnl[h])

        if analysis.daily_pnl:
            analysis.best_day = max(analysis.daily_pnl, key=lambda d: analysis.daily_pnl[d])
            analysis.worst_day = min(analysis.daily_pnl, key=lambda d: analysis.daily_pnl[d])

        return analysis

    def analyze_by_market(
        self,
        trades: Optional[List[TradeData]] = None,
    ) -> MarketAnalysis:
        """Analyze trades by market."""
        trades = trades or self._trades

        if not trades:
            return MarketAnalysis()

        analysis = MarketAnalysis()

        # Group by market
        market_trades: Dict[str, List[TradeData]] = {}
        for trade in trades:
            if trade.market not in market_trades:
                market_trades[trade.market] = []
            market_trades[trade.market].append(trade)

        # Calculate stats for each market
        for market, market_trade_list in market_trades.items():
            analysis.market_stats[market] = self.calculate_statistics(market_trade_list)

        # Find best/worst/most traded
        if analysis.market_stats:
            analysis.best_market = max(
                analysis.market_stats,
                key=lambda m: analysis.market_stats[m].net_pnl
            )
            analysis.worst_market = min(
                analysis.market_stats,
                key=lambda m: analysis.market_stats[m].net_pnl
            )
            analysis.most_traded = max(
                analysis.market_stats,
                key=lambda m: analysis.market_stats[m].total_trades
            )

        return analysis

    def calculate_sharpe_ratio(
        self,
        trades: Optional[List[TradeData]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sharpe ratio from trades."""
        trades = trades or self._trades

        if len(trades) < 2:
            return 0.0

        # Get returns
        returns = [float(t.pnl_pct) for t in trades]

        avg_return = sum(returns) / len(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return 0.0

        # Annualized
        sharpe = (avg_return - risk_free_rate) / std_return
        sharpe *= (periods_per_year ** 0.5)  # Annualize

        return round(sharpe, 4)

    def calculate_sortino_ratio(
        self,
        trades: Optional[List[TradeData]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        trades = trades or self._trades

        if len(trades) < 2:
            return 0.0

        returns = [float(t.pnl_pct) for t in trades]
        avg_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]

        if not negative_returns:
            return float('inf') if avg_return > risk_free_rate else 0.0

        downside_dev = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5

        if downside_dev == 0:
            return 0.0

        sortino = (avg_return - risk_free_rate) / downside_dev
        sortino *= (periods_per_year ** 0.5)

        return round(sortino, 4)

    def calculate_calmar_ratio(
        self,
        trades: Optional[List[TradeData]] = None,
        initial_equity: Decimal = Decimal("10000"),
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        trades = trades or self._trades

        if not trades:
            return 0.0

        # Calculate total return
        total_pnl = sum(t.net_pnl for t in trades)
        total_return = float(total_pnl / initial_equity * 100)

        # Calculate max drawdown
        equity = initial_equity
        peak = equity
        max_drawdown = Decimal("0")

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)

        for trade in sorted_trades:
            equity += trade.net_pnl
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_dd_pct = float(max_drawdown / peak * 100) if peak > 0 else 0

        if max_dd_pct == 0:
            return float('inf') if total_return > 0 else 0.0

        return round(total_return / max_dd_pct, 4)

    def get_full_report(
        self,
        initial_equity: Decimal = Decimal("10000"),
    ) -> Dict[str, Any]:
        """Get comprehensive analytics report."""
        stats = self.calculate_statistics()
        streaks = self.analyze_streaks()
        time_analysis = self.analyze_by_time()
        market_analysis = self.analyze_by_market()

        return {
            "statistics": stats.to_dict(),
            "streaks": streaks.to_dict(),
            "time_analysis": time_analysis.to_dict(),
            "market_analysis": market_analysis.to_dict(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(initial_equity=initial_equity),
            "total_trades": len(self._trades),
        }


@dataclass
class RealTimeAnalytics:
    """Real-time analytics for live trading."""

    _trades: List[TradeData] = field(default_factory=list)
    _window_size: int = 50  # Rolling window
    _update_callbacks: List[Callable[[TradeStatistics], None]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize."""
        self._trades = []
        self._update_callbacks = []

    def add_callback(self, callback: Callable[[TradeStatistics], None]) -> None:
        """Add update callback."""
        self._update_callbacks.append(callback)

    def on_trade_complete(self, trade: TradeData) -> TradeStatistics:
        """Handle completed trade and update stats."""
        self._trades.append(trade)

        # Keep only recent trades for rolling stats
        if len(self._trades) > self._window_size:
            self._trades = self._trades[-self._window_size:]

        analyzer = TradeAnalyzer()
        analyzer.add_trades(self._trades)
        stats = analyzer.calculate_statistics()

        # Notify callbacks
        for callback in self._update_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return stats

    def get_current_stats(self) -> TradeStatistics:
        """Get current rolling statistics."""
        analyzer = TradeAnalyzer()
        analyzer.add_trades(self._trades)
        return analyzer.calculate_statistics()


# Global analyzer instance
_global_analyzer: Optional[TradeAnalyzer] = None


def get_trade_analyzer() -> TradeAnalyzer:
    """Get or create global trade analyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = TradeAnalyzer()
    return _global_analyzer


def reset_trade_analyzer() -> None:
    """Reset global trade analyzer."""
    global _global_analyzer
    _global_analyzer = None
