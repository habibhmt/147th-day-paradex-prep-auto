"""Performance metrics for trading systems."""

import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"


@dataclass
class EquityPoint:
    """A point in equity curve."""

    timestamp: float
    equity: Decimal
    drawdown: Decimal = Decimal("0")
    drawdown_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "equity": str(self.equity),
            "drawdown": str(self.drawdown),
            "drawdown_pct": round(self.drawdown_pct, 4),
        }


@dataclass
class DrawdownPeriod:
    """A drawdown period."""

    start_time: float
    end_time: Optional[float] = None
    peak_equity: Decimal = Decimal("0")
    trough_equity: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    duration_days: float = 0.0
    recovered: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "duration_days": round(self.duration_days, 2),
            "recovered": self.recovered,
        }


@dataclass
class ReturnMetrics:
    """Return-based metrics."""

    total_return: Decimal = Decimal("0")
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    daily_return_avg: float = 0.0
    monthly_return_avg: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_days: int = 0
    negative_days: int = 0

    @property
    def positive_days_pct(self) -> float:
        """Percentage of positive days."""
        total = self.positive_days + self.negative_days
        if total == 0:
            return 0.0
        return self.positive_days / total * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": str(self.total_return),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return": round(self.annualized_return, 4),
            "daily_return_avg": round(self.daily_return_avg, 6),
            "best_day": round(self.best_day, 4),
            "worst_day": round(self.worst_day, 4),
            "positive_days_pct": round(self.positive_days_pct, 2),
        }


@dataclass
class RiskMetrics:
    """Risk-based metrics."""

    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0  # days
    ulcer_index: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "volatility": round(self.volatility, 6),
            "annualized_volatility": round(self.annualized_volatility, 4),
            "downside_volatility": round(self.downside_volatility, 6),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_drawdown_duration": round(self.max_drawdown_duration, 2),
            "ulcer_index": round(self.ulcer_index, 4),
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
        }


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    gain_to_pain: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    martin_ratio: float = 0.0  # Ulcer performance index

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "omega_ratio": round(self.omega_ratio, 4),
            "gain_to_pain": round(self.gain_to_pain, 4),
            "martin_ratio": round(self.martin_ratio, 4),
        }


@dataclass
class EfficiencyMetrics:
    """Trading efficiency metrics."""

    profit_factor: float = 0.0
    win_rate: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: Decimal = Decimal("0")
    kelly_criterion: float = 0.0
    trades_per_day: float = 0.0
    avg_trade_duration: float = 0.0  # hours
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "profit_factor": round(self.profit_factor, 4),
            "win_rate": round(self.win_rate, 4),
            "payoff_ratio": round(self.payoff_ratio, 4),
            "expectancy": str(self.expectancy),
            "kelly_criterion": round(self.kelly_criterion, 4),
            "trades_per_day": round(self.trades_per_day, 2),
        }


@dataclass
class ConsistencyMetrics:
    """Consistency and stability metrics."""

    return_stability: float = 0.0  # R-squared of equity curve
    recovery_factor: float = 0.0  # Net profit / Max DD
    system_quality: float = 0.0  # SQN
    k_ratio: float = 0.0  # Linearity of returns
    tail_ratio: float = 0.0  # Right tail / Left tail

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "return_stability": round(self.return_stability, 4),
            "recovery_factor": round(self.recovery_factor, 4),
            "system_quality": round(self.system_quality, 4),
            "k_ratio": round(self.k_ratio, 4),
            "tail_ratio": round(self.tail_ratio, 4),
        }


@dataclass
class PerformanceReport:
    """Complete performance report."""

    period_start: float
    period_end: float
    initial_equity: Decimal
    final_equity: Decimal
    return_metrics: ReturnMetrics
    risk_metrics: RiskMetrics
    risk_adjusted: RiskAdjustedMetrics
    efficiency: EfficiencyMetrics
    consistency: ConsistencyMetrics
    drawdown_periods: List[DrawdownPeriod] = field(default_factory=list)

    def __post_init__(self):
        """Initialize lists."""
        if self.drawdown_periods is None:
            self.drawdown_periods = []

    @property
    def period_days(self) -> float:
        """Calculate period in days."""
        return (self.period_end - self.period_start) / 86400

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "period_days": round(self.period_days, 2),
            "initial_equity": str(self.initial_equity),
            "final_equity": str(self.final_equity),
            "return_metrics": self.return_metrics.to_dict(),
            "risk_metrics": self.risk_metrics.to_dict(),
            "risk_adjusted": self.risk_adjusted.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "consistency": self.consistency.to_dict(),
        }


@dataclass
class PerformanceCalculator:
    """Calculator for performance metrics.

    Features:
    - Return metrics calculation
    - Risk metrics (volatility, VaR, drawdown)
    - Risk-adjusted ratios (Sharpe, Sortino, Calmar)
    - Efficiency metrics (profit factor, win rate)
    - Consistency metrics (R-squared, SQN)
    """

    _equity_curve: List[EquityPoint] = field(default_factory=list)
    _daily_returns: List[float] = field(default_factory=list)
    risk_free_rate: float = 0.0
    periods_per_year: int = 252  # Trading days

    def __post_init__(self):
        """Initialize."""
        self._equity_curve = []
        self._daily_returns = []

    def add_equity_point(
        self,
        timestamp: float,
        equity: Decimal,
    ) -> None:
        """Add equity point."""
        # Calculate drawdown
        peak = max((p.equity for p in self._equity_curve), default=equity)
        if equity > peak:
            peak = equity

        drawdown = peak - equity
        drawdown_pct = float(drawdown / peak * 100) if peak > 0 else 0.0

        point = EquityPoint(
            timestamp=timestamp,
            equity=equity,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )

        # Calculate daily return if we have previous point
        if self._equity_curve:
            prev_equity = self._equity_curve[-1].equity
            if prev_equity > 0:
                daily_return = float((equity - prev_equity) / prev_equity)
                self._daily_returns.append(daily_return)

        self._equity_curve.append(point)

    def calculate_return_metrics(self) -> ReturnMetrics:
        """Calculate return-based metrics."""
        if not self._equity_curve:
            return ReturnMetrics()

        metrics = ReturnMetrics()

        initial = self._equity_curve[0].equity
        final = self._equity_curve[-1].equity

        metrics.total_return = final - initial
        metrics.total_return_pct = float(metrics.total_return / initial * 100) if initial > 0 else 0

        # Calculate annualized return
        if len(self._equity_curve) > 1:
            days = (self._equity_curve[-1].timestamp - self._equity_curve[0].timestamp) / 86400
            if days > 0:
                years = days / 365
                if years > 0:
                    total_return = float(final / initial) if initial > 0 else 1
                    metrics.annualized_return = (total_return ** (1 / years) - 1) * 100

        # Daily returns stats
        if self._daily_returns:
            metrics.daily_return_avg = sum(self._daily_returns) / len(self._daily_returns) * 100
            metrics.best_day = max(self._daily_returns) * 100
            metrics.worst_day = min(self._daily_returns) * 100

            metrics.positive_days = sum(1 for r in self._daily_returns if r > 0)
            metrics.negative_days = sum(1 for r in self._daily_returns if r < 0)

        return metrics

    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate risk-based metrics."""
        if not self._daily_returns:
            return RiskMetrics()

        metrics = RiskMetrics()

        returns = self._daily_returns

        # Volatility
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            metrics.volatility = variance ** 0.5 * 100
            metrics.annualized_volatility = metrics.volatility * (self.periods_per_year ** 0.5)

        # Downside volatility
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) > 1:
            neg_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
            metrics.downside_volatility = (neg_variance ** 0.5) * 100

        # Drawdown metrics from equity curve
        if self._equity_curve:
            max_dd = max(p.drawdown for p in self._equity_curve)
            max_dd_pct = max(p.drawdown_pct for p in self._equity_curve)
            metrics.max_drawdown = max_dd
            metrics.max_drawdown_pct = max_dd_pct

            dd_pcts = [p.drawdown_pct for p in self._equity_curve if p.drawdown_pct > 0]
            if dd_pcts:
                metrics.avg_drawdown = sum(dd_pcts) / len(dd_pcts)

            # Ulcer Index
            squared_dd = sum(p.drawdown_pct ** 2 for p in self._equity_curve)
            metrics.ulcer_index = (squared_dd / len(self._equity_curve)) ** 0.5

        # VaR calculation
        if len(returns) >= 20:
            sorted_returns = sorted(returns)
            n = len(sorted_returns)

            idx_95 = int(n * 0.05)
            idx_99 = int(n * 0.01)

            metrics.var_95 = -sorted_returns[idx_95] * 100
            metrics.var_99 = -sorted_returns[max(0, idx_99)] * 100

            # CVaR
            tail_returns = sorted_returns[:idx_95]
            if tail_returns:
                metrics.cvar_95 = -sum(tail_returns) / len(tail_returns) * 100

        return metrics

    def calculate_risk_adjusted_metrics(
        self,
        return_metrics: ReturnMetrics = None,
        risk_metrics: RiskMetrics = None,
    ) -> RiskAdjustedMetrics:
        """Calculate risk-adjusted metrics."""
        if return_metrics is None:
            return_metrics = self.calculate_return_metrics()
        if risk_metrics is None:
            risk_metrics = self.calculate_risk_metrics()

        metrics = RiskAdjustedMetrics()

        # Sharpe Ratio
        if risk_metrics.annualized_volatility > 0:
            excess_return = return_metrics.annualized_return - self.risk_free_rate
            metrics.sharpe_ratio = excess_return / risk_metrics.annualized_volatility

        # Sortino Ratio
        if risk_metrics.downside_volatility > 0:
            excess_return = return_metrics.daily_return_avg - self.risk_free_rate / self.periods_per_year
            ann_downside = risk_metrics.downside_volatility * (self.periods_per_year ** 0.5)
            metrics.sortino_ratio = (excess_return * self.periods_per_year) / ann_downside if ann_downside > 0 else 0

        # Calmar Ratio
        if risk_metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = return_metrics.annualized_return / risk_metrics.max_drawdown_pct

        # Omega Ratio
        threshold = self.risk_free_rate / self.periods_per_year
        gains = sum(r - threshold for r in self._daily_returns if r > threshold)
        losses = sum(threshold - r for r in self._daily_returns if r < threshold)
        if losses > 0:
            metrics.omega_ratio = gains / losses

        # Gain to Pain Ratio
        positive_returns = sum(r for r in self._daily_returns if r > 0)
        negative_returns = sum(abs(r) for r in self._daily_returns if r < 0)
        if negative_returns > 0:
            metrics.gain_to_pain = positive_returns / negative_returns

        # Martin Ratio (Ulcer Performance Index)
        if risk_metrics.ulcer_index > 0:
            metrics.martin_ratio = return_metrics.annualized_return / risk_metrics.ulcer_index

        return metrics

    def calculate_efficiency_metrics(
        self,
        trades: List[Dict[str, Any]],
    ) -> EfficiencyMetrics:
        """Calculate trading efficiency metrics."""
        if not trades:
            return EfficiencyMetrics()

        metrics = EfficiencyMetrics()

        # Win rate
        winners = [t for t in trades if float(t.get("pnl", 0)) > 0]
        losers = [t for t in trades if float(t.get("pnl", 0)) < 0]

        if winners or losers:
            metrics.win_rate = len(winners) / (len(winners) + len(losers)) * 100

        # Profit factor
        gross_profit = sum(float(t.get("pnl", 0)) for t in winners)
        gross_loss = abs(sum(float(t.get("pnl", 0)) for t in losers))

        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Payoff ratio
        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        if avg_loss > 0:
            metrics.payoff_ratio = avg_win / avg_loss

        # Expectancy
        total_pnl = sum(float(t.get("pnl", 0)) for t in trades)
        metrics.expectancy = Decimal(str(total_pnl / len(trades)))

        # Kelly Criterion
        if metrics.payoff_ratio > 0:
            w = metrics.win_rate / 100
            b = metrics.payoff_ratio
            metrics.kelly_criterion = (w * b - (1 - w)) / b if b > 0 else 0

        # Trading frequency
        if self._equity_curve and len(self._equity_curve) > 1:
            days = (self._equity_curve[-1].timestamp - self._equity_curve[0].timestamp) / 86400
            if days > 0:
                metrics.trades_per_day = len(trades) / days

        return metrics

    def calculate_consistency_metrics(self) -> ConsistencyMetrics:
        """Calculate consistency metrics."""
        if len(self._equity_curve) < 10:
            return ConsistencyMetrics()

        metrics = ConsistencyMetrics()

        # Return stability (R-squared of equity curve)
        n = len(self._equity_curve)
        x_values = list(range(n))
        y_values = [float(p.equity) for p in self._equity_curve]

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = 0

        if ss_tot > 0:
            # Linear regression
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            den = sum((x - x_mean) ** 2 for x in x_values)

            if den > 0:
                slope = num / den
                intercept = y_mean - slope * x_mean

                ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
                metrics.return_stability = 1 - (ss_res / ss_tot)

        # Recovery factor
        if self._equity_curve:
            initial = self._equity_curve[0].equity
            final = self._equity_curve[-1].equity
            net_profit = final - initial
            max_dd = max(p.drawdown for p in self._equity_curve)

            if max_dd > 0:
                metrics.recovery_factor = float(net_profit / max_dd)

        # System Quality Number (SQN)
        if self._daily_returns and len(self._daily_returns) > 1:
            mean_return = sum(self._daily_returns) / len(self._daily_returns)
            variance = sum((r - mean_return) ** 2 for r in self._daily_returns) / (len(self._daily_returns) - 1)
            std = variance ** 0.5

            if std > 0:
                metrics.system_quality = (mean_return / std) * (len(self._daily_returns) ** 0.5)

        # K-Ratio
        if len(self._equity_curve) > 2:
            n = len(self._equity_curve)
            log_returns = []

            for i in range(1, n):
                prev = float(self._equity_curve[i-1].equity)
                curr = float(self._equity_curve[i].equity)
                if prev > 0 and curr > 0:
                    log_returns.append(math.log(curr / prev))

            if log_returns:
                # Cumulative returns
                cum_returns = []
                cum = 0
                for r in log_returns:
                    cum += r
                    cum_returns.append(cum)

                # Linear regression slope
                x_mean = (len(cum_returns) - 1) / 2
                y_mean = sum(cum_returns) / len(cum_returns)

                num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(cum_returns))
                den = sum((i - x_mean) ** 2 for i in range(len(cum_returns)))

                if den > 0:
                    slope = num / den
                    # Standard error of slope
                    y_pred = [slope * i + (y_mean - slope * x_mean) for i in range(len(cum_returns))]
                    mse = sum((y - yp) ** 2 for y, yp in zip(cum_returns, y_pred)) / (len(cum_returns) - 2)

                    se_slope = (mse / den) ** 0.5 if den > 0 and mse > 0 else 0

                    if se_slope > 0:
                        metrics.k_ratio = slope / se_slope

        # Tail ratio
        if self._daily_returns and len(self._daily_returns) >= 20:
            sorted_returns = sorted(self._daily_returns)
            n = len(sorted_returns)

            right_tail = sorted_returns[int(n * 0.95):]
            left_tail = sorted_returns[:int(n * 0.05)]

            avg_right = sum(right_tail) / len(right_tail) if right_tail else 0
            avg_left = abs(sum(left_tail) / len(left_tail)) if left_tail else 0

            if avg_left > 0:
                metrics.tail_ratio = avg_right / avg_left

        return metrics

    def get_full_report(
        self,
        trades: List[Dict[str, Any]] = None,
    ) -> PerformanceReport:
        """Generate complete performance report."""
        if not self._equity_curve:
            return PerformanceReport(
                period_start=time.time(),
                period_end=time.time(),
                initial_equity=Decimal("0"),
                final_equity=Decimal("0"),
                return_metrics=ReturnMetrics(),
                risk_metrics=RiskMetrics(),
                risk_adjusted=RiskAdjustedMetrics(),
                efficiency=EfficiencyMetrics(),
                consistency=ConsistencyMetrics(),
            )

        return_metrics = self.calculate_return_metrics()
        risk_metrics = self.calculate_risk_metrics()
        risk_adjusted = self.calculate_risk_adjusted_metrics(return_metrics, risk_metrics)
        efficiency = self.calculate_efficiency_metrics(trades or [])
        consistency = self.calculate_consistency_metrics()

        return PerformanceReport(
            period_start=self._equity_curve[0].timestamp,
            period_end=self._equity_curve[-1].timestamp,
            initial_equity=self._equity_curve[0].equity,
            final_equity=self._equity_curve[-1].equity,
            return_metrics=return_metrics,
            risk_metrics=risk_metrics,
            risk_adjusted=risk_adjusted,
            efficiency=efficiency,
            consistency=consistency,
        )

    def reset(self) -> None:
        """Reset calculator."""
        self._equity_curve.clear()
        self._daily_returns.clear()


# Global calculator instance
_global_calculator: Optional[PerformanceCalculator] = None


def get_performance_calculator() -> PerformanceCalculator:
    """Get or create global performance calculator."""
    global _global_calculator
    if _global_calculator is None:
        _global_calculator = PerformanceCalculator()
    return _global_calculator


def reset_performance_calculator() -> None:
    """Reset global performance calculator."""
    global _global_calculator
    _global_calculator = None
