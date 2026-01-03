"""
Trade Journal System for Paradex.

Tracks trades, notes, tags, and provides performance analysis and trade review.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import logging
import json
import hashlib


logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class TradeOutcome(Enum):
    """Trade outcome."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


class EmotionalState(Enum):
    """Trader emotional state."""
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    PATIENT = "patient"
    IMPATIENT = "impatient"


class SetupQuality(Enum):
    """Trade setup quality."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ExecutionQuality(Enum):
    """Execution quality assessment."""
    PERFECT = "perfect"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    BAD = "bad"


@dataclass
class TradeNote:
    """Note attached to a trade."""
    id: str
    content: str
    note_type: str  # entry, exit, observation, lesson
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "note_type": self.note_type,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TradeTag:
    """Tag for categorizing trades."""
    name: str
    color: str = "#808080"
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "color": self.color,
            "description": self.description,
        }


@dataclass
class TradeScreenshot:
    """Screenshot attached to trade."""
    id: str
    path: str
    caption: str = ""
    timeframe: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "path": self.path,
            "caption": self.caption,
            "timeframe": self.timeframe,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TradeEntry:
    """Trade entry record."""
    id: str
    symbol: str
    direction: TradeDirection
    status: TradeStatus
    entry_price: Decimal
    entry_time: datetime
    size: Decimal
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    pnl: Decimal = Decimal("0")
    pnl_percent: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    outcome: TradeOutcome = TradeOutcome.PENDING
    setup_quality: Optional[SetupQuality] = None
    execution_quality: Optional[ExecutionQuality] = None
    emotional_state: Optional[EmotionalState] = None
    strategy: str = ""
    timeframe: str = ""
    setup_reason: str = ""
    exit_reason: str = ""
    notes: list[TradeNote] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    screenshots: list[TradeScreenshot] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "status": self.status.value,
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "size": str(self.size),
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "pnl": str(self.pnl),
            "pnl_percent": str(self.pnl_percent),
            "fees": str(self.fees),
            "outcome": self.outcome.value,
            "setup_quality": self.setup_quality.value if self.setup_quality else None,
            "execution_quality": self.execution_quality.value if self.execution_quality else None,
            "emotional_state": self.emotional_state.value if self.emotional_state else None,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "setup_reason": self.setup_reason,
            "exit_reason": self.exit_reason,
            "notes": [n.to_dict() for n in self.notes],
            "tags": self.tags,
            "screenshots": [s.to_dict() for s in self.screenshots],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def duration(self) -> Optional[timedelta]:
        """Get trade duration."""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None

    @property
    def risk_reward_actual(self) -> Optional[Decimal]:
        """Get actual risk/reward achieved."""
        if not self.stop_loss or not self.exit_price:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return None
        reward = abs(self.exit_price - self.entry_price)
        return reward / risk


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "total_pnl": str(self.total_pnl),
            "total_fees": str(self.total_fees),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "notes": self.notes,
        }


@dataclass
class WeeklyReview:
    """Weekly trading review."""
    week_start: datetime
    week_end: datetime
    total_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    lessons_learned: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    goals_achieved: list[str] = field(default_factory=list)
    goals_next_week: list[str] = field(default_factory=list)
    best_trades: list[str] = field(default_factory=list)
    worst_trades: list[str] = field(default_factory=list)
    overall_rating: int = 5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "total_trades": self.total_trades,
            "total_pnl": str(self.total_pnl),
            "win_rate": str(self.win_rate),
            "lessons_learned": self.lessons_learned,
            "improvements": self.improvements,
            "goals_achieved": self.goals_achieved,
            "goals_next_week": self.goals_next_week,
            "best_trades": self.best_trades,
            "worst_trades": self.worst_trades,
            "overall_rating": self.overall_rating,
        }


class TradeAnalyzer:
    """Analyze trade journal entries."""

    def calculate_stats(self, trades: list[TradeEntry]) -> dict:
        """Calculate overall statistics."""
        if not trades:
            return self._empty_stats()

        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return self._empty_stats()

        wins = [t for t in closed_trades if t.outcome == TradeOutcome.WIN]
        losses = [t for t in closed_trades if t.outcome == TradeOutcome.LOSS]
        breakeven = [t for t in closed_trades if t.outcome == TradeOutcome.BREAKEVEN]

        total_pnl = sum(t.pnl for t in closed_trades)
        total_fees = sum(t.fees for t in closed_trades)

        gross_profit = sum(t.pnl for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else Decimal("0")

        win_rate = Decimal(len(wins)) / Decimal(len(closed_trades)) * 100 if closed_trades else Decimal("0")
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("0")

        avg_win = gross_profit / len(wins) if wins else Decimal("0")
        avg_loss = gross_loss / len(losses) if losses else Decimal("0")

        expectancy = Decimal("0")
        if closed_trades:
            expectancy = total_pnl / len(closed_trades)

        largest_win = max((t.pnl for t in wins), default=Decimal("0"))
        largest_loss = min((t.pnl for t in losses), default=Decimal("0"))

        # Duration stats
        durations = [t.duration for t in closed_trades if t.duration]
        avg_duration = sum(d.total_seconds() for d in durations) / len(durations) if durations else 0

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": len(breakeven),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_pnl": float(total_pnl),
            "total_fees": float(total_fees),
            "net_pnl": float(total_pnl - total_fees),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "average_win": float(avg_win),
            "average_loss": float(avg_loss),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "expectancy": float(expectancy),
            "average_duration_seconds": avg_duration,
        }

    def _empty_stats(self) -> dict:
        """Return empty statistics."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_pnl": 0,
            "total_fees": 0,
            "net_pnl": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "average_win": 0,
            "average_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "expectancy": 0,
            "average_duration_seconds": 0,
        }

    def stats_by_symbol(self, trades: list[TradeEntry]) -> dict[str, dict]:
        """Calculate stats grouped by symbol."""
        by_symbol: dict[str, list[TradeEntry]] = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)

        return {symbol: self.calculate_stats(t_list) for symbol, t_list in by_symbol.items()}

    def stats_by_strategy(self, trades: list[TradeEntry]) -> dict[str, dict]:
        """Calculate stats grouped by strategy."""
        by_strategy: dict[str, list[TradeEntry]] = {}
        for trade in trades:
            strategy = trade.strategy or "unknown"
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(trade)

        return {strategy: self.calculate_stats(t_list) for strategy, t_list in by_strategy.items()}

    def stats_by_timeframe(self, trades: list[TradeEntry]) -> dict[str, dict]:
        """Calculate stats grouped by timeframe."""
        by_tf: dict[str, list[TradeEntry]] = {}
        for trade in trades:
            tf = trade.timeframe or "unknown"
            if tf not in by_tf:
                by_tf[tf] = []
            by_tf[tf].append(trade)

        return {tf: self.calculate_stats(t_list) for tf, t_list in by_tf.items()}

    def stats_by_day_of_week(self, trades: list[TradeEntry]) -> dict[str, dict]:
        """Calculate stats grouped by day of week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day: dict[str, list[TradeEntry]] = {day: [] for day in days}

        for trade in trades:
            day = days[trade.entry_time.weekday()]
            by_day[day].append(trade)

        return {day: self.calculate_stats(t_list) for day, t_list in by_day.items()}

    def stats_by_hour(self, trades: list[TradeEntry]) -> dict[int, dict]:
        """Calculate stats grouped by hour."""
        by_hour: dict[int, list[TradeEntry]] = {h: [] for h in range(24)}

        for trade in trades:
            hour = trade.entry_time.hour
            by_hour[hour].append(trade)

        return {hour: self.calculate_stats(t_list) for hour, t_list in by_hour.items()}

    def equity_curve(self, trades: list[TradeEntry], initial_balance: Decimal = Decimal("10000")) -> list[dict]:
        """Calculate equity curve from trades."""
        closed = sorted(
            [t for t in trades if t.status == TradeStatus.CLOSED and t.exit_time],
            key=lambda t: t.exit_time
        )

        curve = [{"time": datetime.now().isoformat(), "equity": float(initial_balance)}]
        equity = initial_balance

        for trade in closed:
            equity += trade.pnl
            curve.append({
                "time": trade.exit_time.isoformat(),
                "equity": float(equity),
                "trade_id": trade.id,
            })

        return curve

    def drawdown_analysis(self, trades: list[TradeEntry], initial_balance: Decimal = Decimal("10000")) -> dict:
        """Analyze drawdowns."""
        curve = self.equity_curve(trades, initial_balance)

        if len(curve) < 2:
            return {"max_drawdown": 0, "max_drawdown_pct": 0, "drawdown_periods": []}

        peak = Decimal(str(curve[0]["equity"]))
        max_dd = Decimal("0")
        max_dd_pct = Decimal("0")
        drawdown_periods = []
        current_dd_start = None

        for point in curve[1:]:
            equity = Decimal(str(point["equity"]))
            if equity > peak:
                if current_dd_start:
                    drawdown_periods.append({
                        "start": current_dd_start,
                        "end": point["time"],
                        "drawdown": float(peak - equity),
                    })
                    current_dd_start = None
                peak = equity
            else:
                dd = peak - equity
                dd_pct = dd / peak * 100 if peak > 0 else Decimal("0")
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct
                if current_dd_start is None:
                    current_dd_start = point["time"]

        return {
            "max_drawdown": float(max_dd),
            "max_drawdown_pct": float(max_dd_pct),
            "drawdown_periods": drawdown_periods,
        }


class TagManager:
    """Manage trade tags."""

    def __init__(self):
        """Initialize tag manager."""
        self._tags: dict[str, TradeTag] = {}
        self._default_tags()

    def _default_tags(self) -> None:
        """Create default tags."""
        defaults = [
            TradeTag("trend-following", "#4CAF50", "Trades following the trend"),
            TradeTag("counter-trend", "#F44336", "Counter-trend trades"),
            TradeTag("breakout", "#2196F3", "Breakout trades"),
            TradeTag("reversal", "#FF9800", "Reversal pattern trades"),
            TradeTag("scalp", "#9C27B0", "Short-term scalp trades"),
            TradeTag("swing", "#00BCD4", "Multi-day swing trades"),
            TradeTag("news-driven", "#795548", "News catalyst trades"),
            TradeTag("technical", "#607D8B", "Pure technical analysis"),
            TradeTag("revenge", "#E91E63", "Emotional revenge trade"),
            TradeTag("fomo", "#FF5722", "FOMO entry"),
        ]
        for tag in defaults:
            self._tags[tag.name] = tag

    def add(self, name: str, color: str = "#808080", description: str = "") -> TradeTag:
        """Add a new tag."""
        tag = TradeTag(name=name, color=color, description=description)
        self._tags[name] = tag
        return tag

    def remove(self, name: str) -> bool:
        """Remove a tag."""
        if name in self._tags:
            del self._tags[name]
            return True
        return False

    def get(self, name: str) -> Optional[TradeTag]:
        """Get tag by name."""
        return self._tags.get(name)

    def get_all(self) -> list[TradeTag]:
        """Get all tags."""
        return list(self._tags.values())

    def search(self, query: str) -> list[TradeTag]:
        """Search tags by name."""
        query = query.lower()
        return [t for t in self._tags.values() if query in t.name.lower()]


class TradeJournal:
    """Main trade journal system."""

    def __init__(self):
        """Initialize trade journal."""
        self._trades: dict[str, TradeEntry] = {}
        self._daily_stats: dict[str, DailyStats] = {}
        self._weekly_reviews: dict[str, WeeklyReview] = {}
        self._analyzer = TradeAnalyzer()
        self._tag_manager = TagManager()
        self._callbacks: list[Callable] = []

    def add_trade(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: Decimal,
        size: Decimal,
        entry_time: Optional[datetime] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        strategy: str = "",
        timeframe: str = "",
        setup_reason: str = "",
        setup_quality: Optional[SetupQuality] = None,
        emotional_state: Optional[EmotionalState] = None,
        tags: Optional[list[str]] = None,
    ) -> TradeEntry:
        """Add a new trade to the journal."""
        trade_id = self._generate_id(symbol, entry_time or datetime.now())

        trade = TradeEntry(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(),
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            timeframe=timeframe,
            setup_reason=setup_reason,
            setup_quality=setup_quality,
            emotional_state=emotional_state,
            tags=tags or [],
        )

        self._trades[trade_id] = trade
        self._notify("trade_added", trade)

        return trade

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_time: Optional[datetime] = None,
        fees: Decimal = Decimal("0"),
        exit_reason: str = "",
        execution_quality: Optional[ExecutionQuality] = None,
    ) -> Optional[TradeEntry]:
        """Close a trade."""
        if trade_id not in self._trades:
            return None

        trade = self._trades[trade_id]
        if trade.status != TradeStatus.OPEN:
            return None

        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.now()
        trade.fees = fees
        trade.exit_reason = exit_reason
        trade.execution_quality = execution_quality
        trade.status = TradeStatus.CLOSED
        trade.updated_at = datetime.now()

        # Calculate PnL
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.size - fees
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size - fees

        # Calculate PnL percent
        notional = trade.entry_price * trade.size
        if notional > 0:
            trade.pnl_percent = trade.pnl / notional * 100

        # Determine outcome
        if trade.pnl > Decimal("0.01"):
            trade.outcome = TradeOutcome.WIN
        elif trade.pnl < Decimal("-0.01"):
            trade.outcome = TradeOutcome.LOSS
        else:
            trade.outcome = TradeOutcome.BREAKEVEN

        self._notify("trade_closed", trade)
        return trade

    def update_trade(
        self,
        trade_id: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        strategy: Optional[str] = None,
        setup_quality: Optional[SetupQuality] = None,
        emotional_state: Optional[EmotionalState] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[TradeEntry]:
        """Update trade details."""
        if trade_id not in self._trades:
            return None

        trade = self._trades[trade_id]

        if stop_loss is not None:
            trade.stop_loss = stop_loss
        if take_profit is not None:
            trade.take_profit = take_profit
        if strategy is not None:
            trade.strategy = strategy
        if setup_quality is not None:
            trade.setup_quality = setup_quality
        if emotional_state is not None:
            trade.emotional_state = emotional_state
        if tags is not None:
            trade.tags = tags

        trade.updated_at = datetime.now()
        self._notify("trade_updated", trade)

        return trade

    def add_note(
        self,
        trade_id: str,
        content: str,
        note_type: str = "observation",
    ) -> Optional[TradeNote]:
        """Add a note to a trade."""
        if trade_id not in self._trades:
            return None

        note_id = hashlib.md5(f"{trade_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        note = TradeNote(id=note_id, content=content, note_type=note_type)
        self._trades[trade_id].notes.append(note)
        self._trades[trade_id].updated_at = datetime.now()

        return note

    def add_screenshot(
        self,
        trade_id: str,
        path: str,
        caption: str = "",
        timeframe: str = "",
    ) -> Optional[TradeScreenshot]:
        """Add a screenshot to a trade."""
        if trade_id not in self._trades:
            return None

        ss_id = hashlib.md5(f"{trade_id}:{path}".encode()).hexdigest()[:8]
        screenshot = TradeScreenshot(
            id=ss_id,
            path=path,
            caption=caption,
            timeframe=timeframe,
        )
        self._trades[trade_id].screenshots.append(screenshot)
        self._trades[trade_id].updated_at = datetime.now()

        return screenshot

    def add_tag(self, trade_id: str, tag: str) -> bool:
        """Add tag to trade."""
        if trade_id not in self._trades:
            return False

        if tag not in self._trades[trade_id].tags:
            self._trades[trade_id].tags.append(tag)
            self._trades[trade_id].updated_at = datetime.now()

        return True

    def remove_tag(self, trade_id: str, tag: str) -> bool:
        """Remove tag from trade."""
        if trade_id not in self._trades:
            return False

        if tag in self._trades[trade_id].tags:
            self._trades[trade_id].tags.remove(tag)
            self._trades[trade_id].updated_at = datetime.now()
            return True

        return False

    def get_trade(self, trade_id: str) -> Optional[TradeEntry]:
        """Get trade by ID."""
        return self._trades.get(trade_id)

    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[TradeStatus] = None,
        outcome: Optional[TradeOutcome] = None,
        direction: Optional[TradeDirection] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
    ) -> list[TradeEntry]:
        """Get trades with filters."""
        trades = list(self._trades.values())

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if status:
            trades = [t for t in trades if t.status == status]
        if outcome:
            trades = [t for t in trades if t.outcome == outcome]
        if direction:
            trades = [t for t in trades if t.direction == direction]
        if strategy:
            trades = [t for t in trades if t.strategy == strategy]
        if start_date:
            trades = [t for t in trades if t.entry_time >= start_date]
        if end_date:
            trades = [t for t in trades if t.entry_time <= end_date]
        if tags:
            trades = [t for t in trades if any(tag in t.tags for tag in tags)]

        return sorted(trades, key=lambda t: t.entry_time, reverse=True)

    def get_open_trades(self) -> list[TradeEntry]:
        """Get all open trades."""
        return self.get_trades(status=TradeStatus.OPEN)

    def get_closed_trades(self) -> list[TradeEntry]:
        """Get all closed trades."""
        return self.get_trades(status=TradeStatus.CLOSED)

    def get_winning_trades(self) -> list[TradeEntry]:
        """Get all winning trades."""
        return self.get_trades(outcome=TradeOutcome.WIN)

    def get_losing_trades(self) -> list[TradeEntry]:
        """Get all losing trades."""
        return self.get_trades(outcome=TradeOutcome.LOSS)

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade."""
        if trade_id in self._trades:
            del self._trades[trade_id]
            self._notify("trade_deleted", trade_id)
            return True
        return False

    def get_stats(self) -> dict:
        """Get overall statistics."""
        return self._analyzer.calculate_stats(list(self._trades.values()))

    def get_stats_by_symbol(self) -> dict[str, dict]:
        """Get stats by symbol."""
        return self._analyzer.stats_by_symbol(list(self._trades.values()))

    def get_stats_by_strategy(self) -> dict[str, dict]:
        """Get stats by strategy."""
        return self._analyzer.stats_by_strategy(list(self._trades.values()))

    def get_stats_by_timeframe(self) -> dict[str, dict]:
        """Get stats by timeframe."""
        return self._analyzer.stats_by_timeframe(list(self._trades.values()))

    def get_stats_by_day(self) -> dict[str, dict]:
        """Get stats by day of week."""
        return self._analyzer.stats_by_day_of_week(list(self._trades.values()))

    def get_stats_by_hour(self) -> dict[int, dict]:
        """Get stats by hour."""
        return self._analyzer.stats_by_hour(list(self._trades.values()))

    def get_equity_curve(self, initial_balance: Decimal = Decimal("10000")) -> list[dict]:
        """Get equity curve."""
        return self._analyzer.equity_curve(list(self._trades.values()), initial_balance)

    def get_drawdown_analysis(self, initial_balance: Decimal = Decimal("10000")) -> dict:
        """Get drawdown analysis."""
        return self._analyzer.drawdown_analysis(list(self._trades.values()), initial_balance)

    def get_daily_stats(self, date: datetime) -> DailyStats:
        """Get or calculate daily stats."""
        date_key = date.strftime("%Y-%m-%d")

        if date_key in self._daily_stats:
            return self._daily_stats[date_key]

        # Calculate daily stats
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        trades = self.get_trades(start_date=day_start, end_date=day_end, status=TradeStatus.CLOSED)
        stats = self._analyzer.calculate_stats(trades)

        daily = DailyStats(
            date=day_start,
            total_trades=stats["total_trades"],
            winning_trades=stats["winning_trades"],
            losing_trades=stats["losing_trades"],
            breakeven_trades=stats["breakeven_trades"],
            total_pnl=Decimal(str(stats["total_pnl"])),
            total_fees=Decimal(str(stats["total_fees"])),
            largest_win=Decimal(str(stats["largest_win"])),
            largest_loss=Decimal(str(stats["largest_loss"])),
            win_rate=Decimal(str(stats["win_rate"])),
            profit_factor=Decimal(str(stats["profit_factor"])),
            average_win=Decimal(str(stats["average_win"])),
            average_loss=Decimal(str(stats["average_loss"])),
        )

        self._daily_stats[date_key] = daily
        return daily

    def add_weekly_review(
        self,
        week_start: datetime,
        lessons_learned: list[str] = None,
        improvements: list[str] = None,
        goals_achieved: list[str] = None,
        goals_next_week: list[str] = None,
        best_trades: list[str] = None,
        worst_trades: list[str] = None,
        overall_rating: int = 5,
    ) -> WeeklyReview:
        """Add weekly review."""
        week_end = week_start + timedelta(days=6)
        trades = self.get_trades(
            start_date=week_start,
            end_date=week_end,
            status=TradeStatus.CLOSED,
        )
        stats = self._analyzer.calculate_stats(trades)

        review = WeeklyReview(
            week_start=week_start,
            week_end=week_end,
            total_trades=stats["total_trades"],
            total_pnl=Decimal(str(stats["total_pnl"])),
            win_rate=Decimal(str(stats["win_rate"])),
            lessons_learned=lessons_learned or [],
            improvements=improvements or [],
            goals_achieved=goals_achieved or [],
            goals_next_week=goals_next_week or [],
            best_trades=best_trades or [],
            worst_trades=worst_trades or [],
            overall_rating=overall_rating,
        )

        week_key = week_start.strftime("%Y-%W")
        self._weekly_reviews[week_key] = review

        return review

    def get_weekly_review(self, week_start: datetime) -> Optional[WeeklyReview]:
        """Get weekly review."""
        week_key = week_start.strftime("%Y-%W")
        return self._weekly_reviews.get(week_key)

    @property
    def tag_manager(self) -> TagManager:
        """Get tag manager."""
        return self._tag_manager

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

    def _generate_id(self, symbol: str, timestamp: datetime) -> str:
        """Generate trade ID."""
        key = f"{symbol}:{timestamp.isoformat()}:{len(self._trades)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def export_json(self) -> str:
        """Export journal to JSON."""
        data = {
            "trades": [t.to_dict() for t in self._trades.values()],
            "daily_stats": [s.to_dict() for s in self._daily_stats.values()],
            "weekly_reviews": [r.to_dict() for r in self._weekly_reviews.values()],
            "tags": [t.to_dict() for t in self._tag_manager.get_all()],
        }
        return json.dumps(data, indent=2)

    def get_summary(self) -> dict:
        """Get journal summary."""
        stats = self.get_stats()
        return {
            "total_trades": len(self._trades),
            "open_trades": len(self.get_open_trades()),
            "closed_trades": len(self.get_closed_trades()),
            "winning_trades": stats["winning_trades"],
            "losing_trades": stats["losing_trades"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "profit_factor": stats["profit_factor"],
        }


# Global instance
_trade_journal: Optional[TradeJournal] = None


def get_trade_journal() -> TradeJournal:
    """Get global trade journal instance."""
    global _trade_journal
    if _trade_journal is None:
        _trade_journal = TradeJournal()
    return _trade_journal


def set_trade_journal(journal: TradeJournal) -> None:
    """Set global trade journal instance."""
    global _trade_journal
    _trade_journal = journal
