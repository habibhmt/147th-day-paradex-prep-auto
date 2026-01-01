"""Performance analytics for trading bot."""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Metrics for a set of trades."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    avg_trade_size: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.largest_loss == 0:
            return float("inf") if self.largest_win > 0 else 0.0
        return float(self.largest_win / abs(self.largest_loss))

    @property
    def net_pnl(self) -> Decimal:
        """Net PnL after fees."""
        return self.total_pnl - self.total_fees

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "total_volume": str(self.total_volume),
            "total_fees": str(self.total_fees),
            "avg_trade_size": str(self.avg_trade_size),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "profit_factor": round(self.profit_factor, 2),
        }


@dataclass
class DeltaMetrics:
    """Delta neutrality metrics."""

    checks_count: int = 0
    neutral_count: int = 0
    rebalance_count: int = 0
    avg_delta_pct: float = 0.0
    max_delta_pct: float = 0.0
    time_in_neutral_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "checks_count": self.checks_count,
            "neutral_count": self.neutral_count,
            "rebalance_count": self.rebalance_count,
            "avg_delta_pct": round(self.avg_delta_pct, 2),
            "max_delta_pct": round(self.max_delta_pct, 2),
            "time_in_neutral_pct": round(self.time_in_neutral_pct, 2),
        }


@dataclass
class XPMetrics:
    """XP earning metrics."""

    estimated_xp: float = 0.0
    volume_contribution: Decimal = Decimal("0")
    position_hours: float = 0.0
    efficiency_score: float = 0.0  # XP per volume

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "estimated_xp": round(self.estimated_xp, 0),
            "volume_contribution": str(self.volume_contribution),
            "position_hours": round(self.position_hours, 2),
            "efficiency_score": round(self.efficiency_score, 4),
        }


@dataclass
class PerformanceAnalyzer:
    """Analyzes trading performance.

    Features:
    - Trade performance metrics
    - Delta neutrality tracking
    - XP efficiency analysis
    - Time-based reports
    - Account comparisons
    """

    # Tracking data
    _trade_pnls: List[Decimal] = field(default_factory=list)
    _delta_checks: List[tuple] = field(default_factory=list)  # (timestamp, delta_pct, is_neutral)
    _hourly_volume: Dict[str, Decimal] = field(default_factory=dict)
    _account_metrics: Dict[str, TradeMetrics] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._trade_pnls = []
        self._delta_checks = []
        self._hourly_volume = {}
        self._account_metrics = {}

    def record_trade(
        self,
        account_id: str,
        pnl: Decimal,
        volume: Decimal,
        fee: Decimal = Decimal("0"),
    ) -> None:
        """Record a trade for analytics.

        Args:
            account_id: Account that traded
            pnl: Trade PnL
            volume: Trade volume
            fee: Trade fee
        """
        self._trade_pnls.append(pnl)

        # Update account metrics
        if account_id not in self._account_metrics:
            self._account_metrics[account_id] = TradeMetrics()

        metrics = self._account_metrics[account_id]
        metrics.total_trades += 1
        metrics.total_pnl += pnl
        metrics.total_volume += volume
        metrics.total_fees += fee

        if pnl > 0:
            metrics.winning_trades += 1
            if pnl > metrics.largest_win:
                metrics.largest_win = pnl
        elif pnl < 0:
            metrics.losing_trades += 1
            if pnl < metrics.largest_loss:
                metrics.largest_loss = pnl

        if metrics.total_trades > 0:
            metrics.avg_trade_size = metrics.total_volume / metrics.total_trades

        # Track hourly volume
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        if hour_key not in self._hourly_volume:
            self._hourly_volume[hour_key] = Decimal("0")
        self._hourly_volume[hour_key] += volume

    def record_delta_check(
        self,
        delta_pct: float,
        is_neutral: bool,
    ) -> None:
        """Record a delta neutrality check.

        Args:
            delta_pct: Delta deviation percentage
            is_neutral: Whether within neutral threshold
        """
        self._delta_checks.append((time.time(), delta_pct, is_neutral))

    def record_rebalance(self) -> None:
        """Record a rebalance event."""
        # This is tracked via delta checks with the transition to neutral

    def get_trade_metrics(
        self,
        account_id: Optional[str] = None,
    ) -> TradeMetrics:
        """Get trade performance metrics.

        Args:
            account_id: Specific account or all accounts

        Returns:
            TradeMetrics
        """
        if account_id:
            return self._account_metrics.get(account_id, TradeMetrics())

        # Aggregate all accounts
        total = TradeMetrics()
        for metrics in self._account_metrics.values():
            total.total_trades += metrics.total_trades
            total.winning_trades += metrics.winning_trades
            total.losing_trades += metrics.losing_trades
            total.total_pnl += metrics.total_pnl
            total.total_volume += metrics.total_volume
            total.total_fees += metrics.total_fees
            if metrics.largest_win > total.largest_win:
                total.largest_win = metrics.largest_win
            if metrics.largest_loss < total.largest_loss:
                total.largest_loss = metrics.largest_loss

        if total.total_trades > 0:
            total.avg_trade_size = total.total_volume / total.total_trades

        return total

    def get_delta_metrics(self) -> DeltaMetrics:
        """Get delta neutrality metrics.

        Returns:
            DeltaMetrics
        """
        if not self._delta_checks:
            return DeltaMetrics()

        total_delta = sum(abs(d[1]) for d in self._delta_checks)
        neutral_count = sum(1 for d in self._delta_checks if d[2])

        return DeltaMetrics(
            checks_count=len(self._delta_checks),
            neutral_count=neutral_count,
            avg_delta_pct=total_delta / len(self._delta_checks),
            max_delta_pct=max(abs(d[1]) for d in self._delta_checks),
            time_in_neutral_pct=(neutral_count / len(self._delta_checks)) * 100,
        )

    def get_xp_metrics(
        self,
        weekly_pool: int = 4_000_000,
    ) -> XPMetrics:
        """Get XP earning metrics.

        Args:
            weekly_pool: Total weekly XP pool

        Returns:
            XPMetrics
        """
        total_volume = sum(m.total_volume for m in self._account_metrics.values())

        # Simple XP estimation (actual formula is more complex)
        # Assume we have ~10% of total platform volume (very rough estimate)
        estimated_xp = float(total_volume) / 1_000_000 * weekly_pool * 0.01

        # Calculate efficiency
        efficiency = estimated_xp / float(total_volume) if total_volume > 0 else 0

        return XPMetrics(
            estimated_xp=estimated_xp,
            volume_contribution=total_volume,
            efficiency_score=efficiency,
        )

    def get_hourly_volume(self, hours: int = 24) -> Dict[str, Decimal]:
        """Get hourly volume breakdown.

        Args:
            hours: Number of hours to include

        Returns:
            Dictionary of {hour_key: volume}
        """
        now = datetime.now()
        result = {}

        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            result[hour_key] = self._hourly_volume.get(hour_key, Decimal("0"))

        return result

    def get_account_comparison(self) -> List[Dict]:
        """Compare performance across accounts.

        Returns:
            List of account metrics
        """
        result = []
        for account_id, metrics in self._account_metrics.items():
            data = metrics.to_dict()
            data["account_id"] = account_id
            result.append(data)

        # Sort by total PnL
        result.sort(key=lambda x: Decimal(x["total_pnl"]), reverse=True)
        return result

    def get_summary(self) -> Dict:
        """Get performance summary.

        Returns:
            Summary dictionary
        """
        trade_metrics = self.get_trade_metrics()
        delta_metrics = self.get_delta_metrics()
        xp_metrics = self.get_xp_metrics()

        return {
            "trade_performance": trade_metrics.to_dict(),
            "delta_neutrality": delta_metrics.to_dict(),
            "xp_metrics": xp_metrics.to_dict(),
            "accounts_tracked": len(self._account_metrics),
            "timestamp": time.time(),
        }

    def reset(self) -> None:
        """Reset all analytics data."""
        self._trade_pnls.clear()
        self._delta_checks.clear()
        self._hourly_volume.clear()
        self._account_metrics.clear()
        logger.info("Analytics data reset")

    def get_streaks(self) -> Dict:
        """Calculate winning/losing streaks.

        Returns:
            Streak information
        """
        if not self._trade_pnls:
            return {"current_streak": 0, "max_win_streak": 0, "max_loss_streak": 0}

        max_win = 0
        max_loss = 0
        current = 0
        current_type = None

        for pnl in self._trade_pnls:
            if pnl > 0:
                if current_type == "win":
                    current += 1
                else:
                    current = 1
                    current_type = "win"
                max_win = max(max_win, current)
            elif pnl < 0:
                if current_type == "loss":
                    current += 1
                else:
                    current = 1
                    current_type = "loss"
                max_loss = max(max_loss, current)
            else:
                current = 0
                current_type = None

        return {
            "current_streak": current if current_type else 0,
            "current_streak_type": current_type or "none",
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
        }

    def get_drawdown(self) -> Dict:
        """Calculate drawdown metrics.

        Returns:
            Drawdown information
        """
        if not self._trade_pnls:
            return {"max_drawdown": "0", "current_drawdown": "0"}

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for pnl in self._trade_pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        current_dd = peak - cumulative if peak > 0 else Decimal("0")

        return {
            "max_drawdown": str(max_dd),
            "current_drawdown": str(current_dd),
            "peak_equity": str(peak),
            "current_equity": str(cumulative),
        }
