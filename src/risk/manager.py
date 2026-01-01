"""Risk management for delta-neutral trading."""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk alert levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """Actions to take on risk triggers."""

    WARN = "warn"
    REDUCE = "reduce"
    HALT = "halt"
    CLOSE_ALL = "close_all"


@dataclass
class RiskAlert:
    """A risk alert."""

    level: RiskLevel
    action: RiskAction
    message: str
    metric: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "action": self.action.value,
            "message": self.message,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Position limits
    max_position_size: Decimal = Decimal("50000")  # Max per account
    max_total_exposure: Decimal = Decimal("200000")  # Total across accounts
    max_leverage: float = 20.0

    # Loss limits
    max_drawdown_pct: float = 10.0  # Max daily drawdown %
    max_loss_per_trade: Decimal = Decimal("1000")  # Max loss per trade
    max_daily_loss: Decimal = Decimal("5000")  # Max daily loss

    # Delta limits
    max_delta_pct: float = 10.0  # Max delta deviation %
    critical_delta_pct: float = 20.0  # Critical delta level

    # Rate limits
    max_trades_per_hour: int = 100
    min_trade_interval: float = 5.0  # Minimum seconds between trades

    # Circuit breakers
    halt_on_critical: bool = True
    auto_reduce_on_high: bool = True


@dataclass
class RiskManager:
    """Manages trading risk.

    Features:
    - Position size limits
    - Drawdown protection
    - Delta deviation alerts
    - Circuit breakers
    - Trade frequency limits
    """

    config: RiskConfig = field(default_factory=RiskConfig)

    # State tracking
    _daily_pnl: Decimal = Decimal("0")
    _daily_high: Decimal = Decimal("0")
    _current_drawdown: float = 0.0
    _trade_timestamps: List[float] = field(default_factory=list)
    _alerts: List[RiskAlert] = field(default_factory=list)
    _is_halted: bool = False
    _halt_reason: str = ""

    # Callbacks
    on_alert: Optional[Callable[[RiskAlert], None]] = None
    on_halt: Optional[Callable[[str], None]] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._daily_pnl = Decimal("0")
        self._daily_high = Decimal("0")
        self._current_drawdown = 0.0
        self._trade_timestamps = []
        self._alerts = []
        self._is_halted = False
        self._halt_reason = ""

    def check_position_size(
        self,
        size: Decimal,
        account_id: str,
    ) -> Optional[RiskAlert]:
        """Check if position size is within limits.

        Args:
            size: Proposed position size
            account_id: Account ID

        Returns:
            RiskAlert if limit exceeded, None otherwise
        """
        if size > self.config.max_position_size:
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                action=RiskAction.REDUCE,
                message=f"Position size {size} exceeds max {self.config.max_position_size}",
                metric="position_size",
                current_value=float(size),
                threshold=float(self.config.max_position_size),
            )
            self._add_alert(alert)
            return alert

        return None

    def check_total_exposure(
        self,
        total_exposure: Decimal,
    ) -> Optional[RiskAlert]:
        """Check if total exposure is within limits.

        Args:
            total_exposure: Total exposure across all accounts

        Returns:
            RiskAlert if limit exceeded
        """
        if total_exposure > self.config.max_total_exposure:
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                action=RiskAction.REDUCE,
                message=f"Total exposure {total_exposure} exceeds max {self.config.max_total_exposure}",
                metric="total_exposure",
                current_value=float(total_exposure),
                threshold=float(self.config.max_total_exposure),
            )
            self._add_alert(alert)
            return alert

        return None

    def check_leverage(
        self,
        leverage: float,
        account_id: str,
    ) -> Optional[RiskAlert]:
        """Check if leverage is within limits.

        Args:
            leverage: Account leverage
            account_id: Account ID

        Returns:
            RiskAlert if limit exceeded
        """
        if leverage > self.config.max_leverage:
            alert = RiskAlert(
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                message=f"Leverage {leverage}x exceeds max {self.config.max_leverage}x",
                metric="leverage",
                current_value=leverage,
                threshold=self.config.max_leverage,
            )
            self._add_alert(alert)
            return alert

        return None

    def check_delta(
        self,
        delta_pct: float,
        market: str,
    ) -> Optional[RiskAlert]:
        """Check if delta deviation is acceptable.

        Args:
            delta_pct: Delta deviation percentage
            market: Market symbol

        Returns:
            RiskAlert if delta too high
        """
        abs_delta = abs(delta_pct)

        if abs_delta >= self.config.critical_delta_pct:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                action=RiskAction.HALT if self.config.halt_on_critical else RiskAction.REDUCE,
                message=f"Critical delta deviation: {abs_delta:.1f}% for {market}",
                metric="delta_pct",
                current_value=abs_delta,
                threshold=self.config.critical_delta_pct,
            )
            self._add_alert(alert)
            if self.config.halt_on_critical:
                self._halt(f"Critical delta: {abs_delta:.1f}%")
            return alert

        if abs_delta >= self.config.max_delta_pct:
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                action=RiskAction.REDUCE if self.config.auto_reduce_on_high else RiskAction.WARN,
                message=f"High delta deviation: {abs_delta:.1f}% for {market}",
                metric="delta_pct",
                current_value=abs_delta,
                threshold=self.config.max_delta_pct,
            )
            self._add_alert(alert)
            return alert

        return None

    def update_pnl(self, pnl_change: Decimal) -> Optional[RiskAlert]:
        """Update daily PnL and check drawdown.

        Args:
            pnl_change: PnL change (positive or negative)

        Returns:
            RiskAlert if drawdown limit hit
        """
        self._daily_pnl += pnl_change

        # Update high water mark
        if self._daily_pnl > self._daily_high:
            self._daily_high = self._daily_pnl

        # Calculate drawdown
        if self._daily_high > 0:
            self._current_drawdown = float(
                (self._daily_high - self._daily_pnl) / self._daily_high * 100
            )

        # Check drawdown limit
        if self._current_drawdown >= self.config.max_drawdown_pct:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                action=RiskAction.HALT,
                message=f"Max drawdown reached: {self._current_drawdown:.1f}%",
                metric="drawdown_pct",
                current_value=self._current_drawdown,
                threshold=self.config.max_drawdown_pct,
            )
            self._add_alert(alert)
            self._halt(f"Drawdown limit: {self._current_drawdown:.1f}%")
            return alert

        # Check single trade loss
        if pnl_change < 0 and abs(pnl_change) > self.config.max_loss_per_trade:
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                action=RiskAction.WARN,
                message=f"Large trade loss: {pnl_change}",
                metric="trade_loss",
                current_value=float(abs(pnl_change)),
                threshold=float(self.config.max_loss_per_trade),
            )
            self._add_alert(alert)
            return alert

        # Check daily loss
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= self.config.max_daily_loss:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                action=RiskAction.HALT,
                message=f"Max daily loss reached: {self._daily_pnl}",
                metric="daily_loss",
                current_value=float(abs(self._daily_pnl)),
                threshold=float(self.config.max_daily_loss),
            )
            self._add_alert(alert)
            self._halt(f"Daily loss limit: {self._daily_pnl}")
            return alert

        return None

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        if self._is_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # Check trade frequency
        now = time.time()
        recent_trades = [t for t in self._trade_timestamps if now - t < 3600]
        self._trade_timestamps = recent_trades

        if len(recent_trades) >= self.config.max_trades_per_hour:
            return False, f"Max trades per hour ({self.config.max_trades_per_hour}) reached"

        # Check minimum interval
        if recent_trades and now - recent_trades[-1] < self.config.min_trade_interval:
            return False, f"Min trade interval not met"

        return True, "OK"

    def record_trade(self) -> None:
        """Record a trade for frequency tracking."""
        self._trade_timestamps.append(time.time())

    def _halt(self, reason: str) -> None:
        """Halt trading.

        Args:
            reason: Halt reason
        """
        self._is_halted = True
        self._halt_reason = reason
        logger.critical(f"Trading halted: {reason}")

        if self.on_halt:
            try:
                self.on_halt(reason)
            except Exception as e:
                logger.error(f"Halt callback error: {e}")

    def resume(self) -> bool:
        """Resume trading after halt.

        Returns:
            True if resumed
        """
        if not self._is_halted:
            return False

        self._is_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
        return True

    def _add_alert(self, alert: RiskAlert) -> None:
        """Add alert to history and notify.

        Args:
            alert: Alert to add
        """
        self._alerts.append(alert)
        logger.warning(f"Risk alert: {alert.message}")

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def reset_daily(self) -> None:
        """Reset daily metrics (call at day boundary)."""
        self._daily_pnl = Decimal("0")
        self._daily_high = Decimal("0")
        self._current_drawdown = 0.0
        self._trade_timestamps.clear()
        logger.info("Daily risk metrics reset")

    def get_alerts(
        self,
        level: Optional[RiskLevel] = None,
        limit: int = 50,
    ) -> List[RiskAlert]:
        """Get recent alerts.

        Args:
            level: Filter by level
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        alerts = self._alerts[-limit:]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    def get_status(self) -> Dict:
        """Get current risk status.

        Returns:
            Status dictionary
        """
        now = time.time()
        recent_trades = len([t for t in self._trade_timestamps if now - t < 3600])

        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": str(self._daily_pnl),
            "current_drawdown": self._current_drawdown,
            "trades_this_hour": recent_trades,
            "max_trades_per_hour": self.config.max_trades_per_hour,
            "recent_alerts": len(self._alerts),
            "config": {
                "max_position_size": str(self.config.max_position_size),
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_delta_pct": self.config.max_delta_pct,
            },
        }

    def validate_order(
        self,
        size: Decimal,
        leverage: float,
        account_id: str,
    ) -> tuple[bool, List[RiskAlert]]:
        """Validate order against risk limits.

        Args:
            size: Order size
            leverage: Account leverage
            account_id: Account ID

        Returns:
            Tuple of (is_valid, alerts)
        """
        alerts = []

        # Check if can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            alerts.append(RiskAlert(
                level=RiskLevel.HIGH,
                action=RiskAction.HALT,
                message=reason,
                metric="can_trade",
                current_value=0,
                threshold=1,
            ))
            return False, alerts

        # Check position size
        pos_alert = self.check_position_size(size, account_id)
        if pos_alert:
            alerts.append(pos_alert)
            if pos_alert.action == RiskAction.HALT:
                return False, alerts

        # Check leverage
        lev_alert = self.check_leverage(leverage, account_id)
        if lev_alert:
            alerts.append(lev_alert)

        # Return valid if no critical alerts
        has_critical = any(a.level == RiskLevel.CRITICAL for a in alerts)
        return not has_critical, alerts
