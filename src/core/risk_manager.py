"""Risk management for trading operations."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk management actions."""

    NONE = "none"
    ALERT = "alert"
    REDUCE = "reduce"
    CLOSE = "close"
    HALT = "halt"
    EMERGENCY_CLOSE = "emergency_close"


class RiskLevel(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Types of risk."""

    POSITION = "position"
    EXPOSURE = "exposure"
    DRAWDOWN = "drawdown"
    DELTA = "delta"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"


@dataclass
class RiskLimit:
    """A risk limit configuration."""

    name: str
    risk_type: RiskType
    threshold_warning: Decimal
    threshold_critical: Decimal
    action_warning: RiskAction = RiskAction.ALERT
    action_critical: RiskAction = RiskAction.REDUCE
    enabled: bool = True
    cooldown_seconds: float = 60.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "risk_type": self.risk_type.value,
            "threshold_warning": str(self.threshold_warning),
            "threshold_critical": str(self.threshold_critical),
            "action_warning": self.action_warning.value,
            "action_critical": self.action_critical.value,
            "enabled": self.enabled,
        }


@dataclass
class RiskViolation:
    """A risk limit violation."""

    limit: RiskLimit
    current_value: Decimal
    level: RiskLevel
    action: RiskAction
    timestamp: float = field(default_factory=time.time)
    message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "limit_name": self.limit.name,
            "risk_type": self.limit.risk_type.value,
            "current_value": str(self.current_value),
            "threshold": str(self.limit.threshold_critical if self.level == RiskLevel.CRITICAL else self.limit.threshold_warning),
            "level": self.level.value,
            "action": self.action.value,
            "timestamp": self.timestamp,
            "message": self.message,
        }


@dataclass
class RiskMetrics:
    """Current risk metrics."""

    total_exposure: Decimal = Decimal("0")
    net_delta: Decimal = Decimal("0")
    delta_pct: Decimal = Decimal("0")
    gross_exposure: Decimal = Decimal("0")
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    position_count: int = 0
    concentration_pct: float = 0.0
    largest_position_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_exposure": str(self.total_exposure),
            "net_delta": str(self.net_delta),
            "delta_pct": str(self.delta_pct),
            "gross_exposure": str(self.gross_exposure),
            "long_exposure": str(self.long_exposure),
            "short_exposure": str(self.short_exposure),
            "current_drawdown": str(self.current_drawdown),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "position_count": self.position_count,
            "concentration_pct": round(self.concentration_pct, 4),
            "largest_position_pct": round(self.largest_position_pct, 4),
        }


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    # Exposure limits
    max_total_exposure: Decimal = Decimal("100000")
    max_position_size: Decimal = Decimal("10000")
    max_concentration_pct: float = 25.0  # Max single position as % of total

    # Delta limits (for delta-neutral)
    max_delta_pct: float = 5.0  # Max delta deviation percentage
    delta_alert_pct: float = 3.0

    # Drawdown limits
    max_drawdown_pct: float = 10.0
    drawdown_alert_pct: float = 5.0

    # Daily loss limits
    max_daily_loss: Decimal = Decimal("1000")
    max_daily_loss_pct: float = 5.0

    # Position limits
    max_positions: int = 20
    max_leverage: float = 10.0

    # Action thresholds
    halt_trading_drawdown_pct: float = 15.0
    reduce_size_drawdown_pct: float = 10.0


@dataclass
class RiskManager:
    """Manager for trading risk.

    Features:
    - Multiple risk limit types
    - Real-time risk monitoring
    - Automatic risk actions
    - Violation tracking
    - Alert system
    """

    config: RiskConfig = field(default_factory=RiskConfig)
    _limits: List[RiskLimit] = field(default_factory=list)
    _violations: List[RiskViolation] = field(default_factory=list)
    _last_check: Dict[str, float] = field(default_factory=dict)
    _is_halted: bool = False
    _alert_callbacks: List[Callable[[RiskViolation], None]] = field(default_factory=list)
    _peak_equity: Decimal = Decimal("0")
    _daily_pnl: Decimal = Decimal("0")
    _daily_reset_time: float = 0.0

    def __post_init__(self):
        """Initialize risk manager."""
        self._limits = []
        self._violations = []
        self._last_check = {}
        self._is_halted = False
        self._alert_callbacks = []
        self._peak_equity = Decimal("0")
        self._daily_pnl = Decimal("0")
        self._daily_reset_time = time.time()
        self._setup_default_limits()

    def _setup_default_limits(self) -> None:
        """Setup default risk limits."""
        # Exposure limit
        self.add_limit(RiskLimit(
            name="total_exposure",
            risk_type=RiskType.EXPOSURE,
            threshold_warning=self.config.max_total_exposure * Decimal("0.8"),
            threshold_critical=self.config.max_total_exposure,
            action_warning=RiskAction.ALERT,
            action_critical=RiskAction.REDUCE,
        ))

        # Delta limit
        self.add_limit(RiskLimit(
            name="delta_deviation",
            risk_type=RiskType.DELTA,
            threshold_warning=Decimal(str(self.config.delta_alert_pct)),
            threshold_critical=Decimal(str(self.config.max_delta_pct)),
            action_warning=RiskAction.ALERT,
            action_critical=RiskAction.REDUCE,
        ))

        # Drawdown limit
        self.add_limit(RiskLimit(
            name="drawdown",
            risk_type=RiskType.DRAWDOWN,
            threshold_warning=Decimal(str(self.config.drawdown_alert_pct)),
            threshold_critical=Decimal(str(self.config.max_drawdown_pct)),
            action_warning=RiskAction.ALERT,
            action_critical=RiskAction.REDUCE,
        ))

        # Halt trading limit
        self.add_limit(RiskLimit(
            name="halt_threshold",
            risk_type=RiskType.DRAWDOWN,
            threshold_warning=Decimal(str(self.config.max_drawdown_pct)),
            threshold_critical=Decimal(str(self.config.halt_trading_drawdown_pct)),
            action_warning=RiskAction.REDUCE,
            action_critical=RiskAction.HALT,
        ))

        # Concentration limit
        self.add_limit(RiskLimit(
            name="concentration",
            risk_type=RiskType.CONCENTRATION,
            threshold_warning=Decimal(str(self.config.max_concentration_pct * 0.8)),
            threshold_critical=Decimal(str(self.config.max_concentration_pct)),
            action_warning=RiskAction.ALERT,
            action_critical=RiskAction.REDUCE,
        ))

    def add_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit."""
        self._limits.append(limit)

    def remove_limit(self, name: str) -> bool:
        """Remove a risk limit by name."""
        for i, limit in enumerate(self._limits):
            if limit.name == name:
                del self._limits[i]
                return True
        return False

    def get_limit(self, name: str) -> Optional[RiskLimit]:
        """Get a limit by name."""
        for limit in self._limits:
            if limit.name == name:
                return limit
        return None

    def add_alert_callback(
        self,
        callback: Callable[[RiskViolation], None],
    ) -> None:
        """Add callback for risk alerts."""
        self._alert_callbacks.append(callback)

    def update_peak_equity(self, equity: Decimal) -> None:
        """Update peak equity for drawdown calculation."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily PnL."""
        # Check if day has rolled over
        current_day = time.strftime("%Y-%m-%d")
        reset_day = time.strftime("%Y-%m-%d", time.localtime(self._daily_reset_time))

        if current_day != reset_day:
            self._daily_pnl = Decimal("0")
            self._daily_reset_time = time.time()

        self._daily_pnl = pnl

    def check_risks(
        self,
        metrics: RiskMetrics,
    ) -> List[RiskViolation]:
        """Check all risk limits and return violations."""
        violations = []

        for limit in self._limits:
            if not limit.enabled:
                continue

            # Check cooldown
            if self._is_on_cooldown(limit.name):
                continue

            violation = self._check_limit(limit, metrics)
            if violation:
                violations.append(violation)
                self._violations.append(violation)
                self._last_check[limit.name] = time.time()

                # Trigger callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

                # Handle halt action
                if violation.action == RiskAction.HALT:
                    self._is_halted = True
                    logger.critical(f"Trading HALTED: {violation.message}")

        return violations

    def _check_limit(
        self,
        limit: RiskLimit,
        metrics: RiskMetrics,
    ) -> Optional[RiskViolation]:
        """Check a single limit against metrics."""
        value = self._get_metric_value(limit.risk_type, metrics)
        if value is None:
            return None

        # Check critical threshold
        if value >= limit.threshold_critical:
            return RiskViolation(
                limit=limit,
                current_value=value,
                level=RiskLevel.CRITICAL,
                action=limit.action_critical,
                message=f"{limit.name} critical: {value} >= {limit.threshold_critical}",
            )

        # Check warning threshold
        if value >= limit.threshold_warning:
            return RiskViolation(
                limit=limit,
                current_value=value,
                level=RiskLevel.HIGH,
                action=limit.action_warning,
                message=f"{limit.name} warning: {value} >= {limit.threshold_warning}",
            )

        return None

    def _get_metric_value(
        self,
        risk_type: RiskType,
        metrics: RiskMetrics,
    ) -> Optional[Decimal]:
        """Get metric value for a risk type."""
        if risk_type == RiskType.EXPOSURE:
            return metrics.total_exposure
        elif risk_type == RiskType.DELTA:
            return abs(metrics.delta_pct)
        elif risk_type == RiskType.DRAWDOWN:
            return Decimal(str(metrics.drawdown_pct))
        elif risk_type == RiskType.CONCENTRATION:
            return Decimal(str(metrics.concentration_pct))
        elif risk_type == RiskType.POSITION:
            return Decimal(str(metrics.position_count))
        return None

    def _is_on_cooldown(self, limit_name: str) -> bool:
        """Check if limit is on cooldown."""
        last_time = self._last_check.get(limit_name, 0)
        limit = self.get_limit(limit_name)

        if not limit:
            return False

        return time.time() - last_time < limit.cooldown_seconds

    def validate_order(
        self,
        order_size: Decimal,
        order_notional: Decimal,
        current_exposure: Decimal,
    ) -> Tuple[bool, List[str]]:
        """Validate an order against risk limits."""
        errors = []

        if self._is_halted:
            errors.append("Trading is halted due to risk breach")
            return False, errors

        # Check position size
        if order_notional > self.config.max_position_size:
            errors.append(f"Order size {order_notional} exceeds max {self.config.max_position_size}")

        # Check total exposure
        new_exposure = current_exposure + order_notional
        if new_exposure > self.config.max_total_exposure:
            errors.append(f"New exposure {new_exposure} would exceed max {self.config.max_total_exposure}")

        # Check position count (simplified)
        if self.config.max_positions <= 0:
            errors.append("Maximum positions reached")

        return len(errors) == 0, errors

    def validate_position(
        self,
        position_value: Decimal,
        total_equity: Decimal,
    ) -> Tuple[bool, List[str]]:
        """Validate a position against concentration limits."""
        errors = []

        if total_equity <= 0:
            return True, errors

        concentration = float(position_value / total_equity * 100)

        if concentration > self.config.max_concentration_pct:
            errors.append(f"Position concentration {concentration:.1f}% exceeds max {self.config.max_concentration_pct}%")

        return len(errors) == 0, errors

    def calculate_metrics(
        self,
        positions: List[Dict[str, Any]],
        equity: Decimal,
    ) -> RiskMetrics:
        """Calculate risk metrics from positions."""
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        largest_position = Decimal("0")

        for pos in positions:
            notional = Decimal(str(pos.get("notional", 0)))
            is_long = pos.get("is_long", True)

            if is_long:
                long_exposure += notional
            else:
                short_exposure += notional

            if notional > largest_position:
                largest_position = notional

        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        delta_pct = Decimal("0")
        if gross > 0:
            delta_pct = (abs(net) / gross * 100).quantize(Decimal("0.0001"))

        # Calculate drawdown
        drawdown = self._peak_equity - equity if self._peak_equity > equity else Decimal("0")
        drawdown_pct = float(drawdown / self._peak_equity * 100) if self._peak_equity > 0 else 0.0

        # Calculate concentration
        concentration = float(largest_position / equity * 100) if equity > 0 else 0.0

        return RiskMetrics(
            total_exposure=gross,
            net_delta=net,
            delta_pct=delta_pct,
            gross_exposure=gross,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            current_drawdown=drawdown,
            drawdown_pct=drawdown_pct,
            max_drawdown_pct=max(drawdown_pct, 0),
            position_count=len(positions),
            concentration_pct=concentration,
            largest_position_pct=float(largest_position / gross * 100) if gross > 0 else 0.0,
        )

    def get_recommended_action(
        self,
        metrics: RiskMetrics,
    ) -> Tuple[RiskAction, str]:
        """Get recommended action based on current metrics."""
        if self._is_halted:
            return RiskAction.HALT, "Trading is halted"

        # Check drawdown
        if metrics.drawdown_pct >= self.config.halt_trading_drawdown_pct:
            return RiskAction.HALT, f"Drawdown {metrics.drawdown_pct:.1f}% exceeds halt threshold"

        if metrics.drawdown_pct >= self.config.reduce_size_drawdown_pct:
            return RiskAction.REDUCE, f"Drawdown {metrics.drawdown_pct:.1f}% exceeds reduce threshold"

        # Check delta
        if float(metrics.delta_pct) >= self.config.max_delta_pct:
            return RiskAction.REDUCE, f"Delta {metrics.delta_pct}% exceeds max"

        # Check exposure
        if metrics.total_exposure >= self.config.max_total_exposure:
            return RiskAction.REDUCE, f"Exposure {metrics.total_exposure} exceeds max"

        # Check concentration
        if metrics.concentration_pct >= self.config.max_concentration_pct:
            return RiskAction.ALERT, f"Concentration {metrics.concentration_pct:.1f}% too high"

        return RiskAction.NONE, "Risk levels acceptable"

    def calculate_position_reduction(
        self,
        current_size: Decimal,
        reduction_pct: float = 25.0,
    ) -> Decimal:
        """Calculate recommended position reduction."""
        reduction = current_size * Decimal(str(reduction_pct / 100))
        return reduction.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)

    def calculate_safe_size(
        self,
        current_exposure: Decimal,
        equity: Decimal,
    ) -> Decimal:
        """Calculate safe position size given current exposure."""
        remaining = self.config.max_total_exposure - current_exposure
        max_from_equity = equity * Decimal(str(self.config.max_concentration_pct / 100))

        return min(remaining, max_from_equity, self.config.max_position_size)

    def resume_trading(self) -> bool:
        """Resume trading after halt."""
        self._is_halted = False
        logger.info("Trading resumed")
        return True

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        return not self._is_halted

    def get_violations(
        self,
        limit: int = 100,
    ) -> List[RiskViolation]:
        """Get recent violations."""
        return self._violations[-limit:]

    def clear_violations(self) -> None:
        """Clear violation history."""
        self._violations.clear()

    def get_status(self) -> dict:
        """Get risk manager status."""
        return {
            "is_halted": self._is_halted,
            "active_limits": len([l for l in self._limits if l.enabled]),
            "total_violations": len(self._violations),
            "peak_equity": str(self._peak_equity),
            "daily_pnl": str(self._daily_pnl),
        }


@dataclass
class DeltaNeutralRiskManager(RiskManager):
    """Risk manager specialized for delta-neutral strategies."""

    target_delta: Decimal = Decimal("0")
    rebalance_threshold: Decimal = Decimal("5")  # 5% delta deviation

    def check_delta_balance(
        self,
        long_exposure: Decimal,
        short_exposure: Decimal,
    ) -> Tuple[bool, Decimal, str]:
        """Check if positions are delta balanced.

        Returns:
            Tuple of (is_balanced, delta_pct, message)
        """
        total = long_exposure + short_exposure
        if total == 0:
            return True, Decimal("0"), "No positions"

        net_delta = long_exposure - short_exposure
        delta_pct = abs(net_delta) / total * 100

        is_balanced = delta_pct <= self.rebalance_threshold
        message = f"Delta: {delta_pct:.2f}% ({'balanced' if is_balanced else 'needs rebalance'})"

        return is_balanced, delta_pct, message

    def calculate_rebalance_needed(
        self,
        long_exposure: Decimal,
        short_exposure: Decimal,
    ) -> Tuple[str, Decimal]:
        """Calculate rebalance needed to reach target delta.

        Returns:
            Tuple of (side, amount) where side is 'long' or 'short'
        """
        current_delta = long_exposure - short_exposure
        target = self.target_delta

        diff = target - current_delta

        if diff > 0:
            return "long", abs(diff)
        elif diff < 0:
            return "short", abs(diff)
        else:
            return "none", Decimal("0")


@dataclass
class EmergencyRiskManager:
    """Emergency risk management for critical situations."""

    _emergency_mode: bool = False
    _close_all_callback: Optional[Callable] = None

    def set_close_all_callback(
        self,
        callback: Callable,
    ) -> None:
        """Set callback for closing all positions."""
        self._close_all_callback = callback

    def trigger_emergency(
        self,
        reason: str,
    ) -> None:
        """Trigger emergency mode."""
        logger.critical(f"EMERGENCY TRIGGERED: {reason}")
        self._emergency_mode = True

        if self._close_all_callback:
            try:
                self._close_all_callback()
            except Exception as e:
                logger.error(f"Emergency close failed: {e}")

    def is_emergency(self) -> bool:
        """Check if in emergency mode."""
        return self._emergency_mode

    def clear_emergency(self) -> None:
        """Clear emergency mode."""
        self._emergency_mode = False
        logger.info("Emergency mode cleared")


# Global risk manager instance
_global_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create global risk manager."""
    global _global_risk_manager
    if _global_risk_manager is None:
        _global_risk_manager = RiskManager()
    return _global_risk_manager


def reset_risk_manager() -> None:
    """Reset global risk manager."""
    global _global_risk_manager
    _global_risk_manager = None
