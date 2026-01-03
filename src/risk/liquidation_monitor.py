"""
Liquidation Monitor Module

Monitors positions for liquidation risk and provides alerts and protection mechanisms.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable
import time


class RiskLevel(Enum):
    """Risk level classification."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"


class AlertType(Enum):
    """Type of liquidation alert."""
    MARGIN_WARNING = "margin_warning"
    MARGIN_CALL = "margin_call"
    APPROACHING_LIQUIDATION = "approaching_liquidation"
    IMMINENT_LIQUIDATION = "imminent_liquidation"
    POSITION_CLOSED = "position_closed"
    AUTO_DELEVERAGED = "auto_deleveraged"


class ProtectionAction(Enum):
    """Actions to protect against liquidation."""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    ADD_MARGIN = "add_margin"
    CLOSE_POSITION = "close_position"
    HEDGE = "hedge"


@dataclass
class MarginInfo:
    """Margin information for a position."""
    initial_margin: Decimal
    maintenance_margin: Decimal
    available_margin: Decimal
    used_margin: Decimal
    margin_ratio: float
    leverage: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "initial_margin": str(self.initial_margin),
            "maintenance_margin": str(self.maintenance_margin),
            "available_margin": str(self.available_margin),
            "used_margin": str(self.used_margin),
            "margin_ratio": self.margin_ratio,
            "leverage": self.leverage,
        }


@dataclass
class LiquidationPrice:
    """Liquidation price calculation result."""
    long_liquidation_price: Optional[Decimal]
    short_liquidation_price: Optional[Decimal]
    distance_to_liquidation_pct: float
    price_buffer: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "long_liquidation_price": str(self.long_liquidation_price) if self.long_liquidation_price else None,
            "short_liquidation_price": str(self.short_liquidation_price) if self.short_liquidation_price else None,
            "distance_to_liquidation_pct": self.distance_to_liquidation_pct,
            "price_buffer": str(self.price_buffer),
        }


@dataclass
class PositionRisk:
    """Risk assessment for a position."""
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    margin_info: MarginInfo
    liquidation_price: LiquidationPrice
    risk_level: RiskLevel
    health_score: float
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "margin_info": self.margin_info.to_dict(),
            "liquidation_price": self.liquidation_price.to_dict(),
            "risk_level": self.risk_level.value,
            "health_score": self.health_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LiquidationAlert:
    """Alert for liquidation risk."""
    alert_id: str
    alert_type: AlertType
    symbol: str
    position_risk: PositionRisk
    message: str
    suggested_action: ProtectionAction
    suggested_amount: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "symbol": self.symbol,
            "position_risk": self.position_risk.to_dict(),
            "message": self.message,
            "suggested_action": self.suggested_action.value,
            "suggested_amount": str(self.suggested_amount) if self.suggested_amount else None,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class MonitorConfig:
    """Configuration for liquidation monitor."""
    # Margin thresholds
    margin_warning_threshold: float = 0.5  # 50% margin usage
    margin_call_threshold: float = 0.7  # 70% margin usage
    liquidation_warning_threshold: float = 0.85  # 85% margin usage
    critical_threshold: float = 0.95  # 95% margin usage

    # Price distance thresholds
    safe_distance_pct: float = 0.1  # 10% from liquidation
    warning_distance_pct: float = 0.05  # 5% from liquidation
    critical_distance_pct: float = 0.02  # 2% from liquidation

    # Auto-protection settings
    auto_reduce_enabled: bool = False
    auto_reduce_threshold: float = 0.9  # 90% margin usage
    auto_reduce_percentage: float = 0.25  # Reduce by 25%

    # Monitoring settings
    check_interval_seconds: int = 5
    alert_cooldown_seconds: int = 60

    def __post_init__(self):
        """Validate configuration."""
        if self.margin_warning_threshold >= self.margin_call_threshold:
            raise ValueError("warning threshold must be less than call threshold")
        if self.margin_call_threshold >= self.liquidation_warning_threshold:
            raise ValueError("call threshold must be less than liquidation threshold")
        if not (0 < self.auto_reduce_percentage <= 1):
            raise ValueError("auto_reduce_percentage must be between 0 and 1")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "margin_warning_threshold": self.margin_warning_threshold,
            "margin_call_threshold": self.margin_call_threshold,
            "liquidation_warning_threshold": self.liquidation_warning_threshold,
            "critical_threshold": self.critical_threshold,
            "safe_distance_pct": self.safe_distance_pct,
            "warning_distance_pct": self.warning_distance_pct,
            "critical_distance_pct": self.critical_distance_pct,
            "auto_reduce_enabled": self.auto_reduce_enabled,
            "auto_reduce_threshold": self.auto_reduce_threshold,
            "auto_reduce_percentage": self.auto_reduce_percentage,
            "check_interval_seconds": self.check_interval_seconds,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
        }


class LiquidationCalculator:
    """Calculates liquidation prices and risk metrics."""

    def __init__(self, maintenance_margin_rate: float = 0.005):
        """Initialize calculator."""
        self.maintenance_margin_rate = maintenance_margin_rate

    def calculate_liquidation_price(
        self,
        side: str,
        entry_price: Decimal,
        size: Decimal,
        margin: Decimal,
        maintenance_rate: Optional[float] = None,
    ) -> LiquidationPrice:
        """Calculate liquidation price for a position."""
        rate = maintenance_rate or self.maintenance_margin_rate
        position_value = entry_price * size

        if side == "long":
            # Long liquidation: entry - (margin - maintenance) / size
            maintenance = position_value * Decimal(str(rate))
            price_drop = (margin - maintenance) / size if size > 0 else Decimal("0")
            liq_price = entry_price - price_drop
            long_liq = max(Decimal("0"), liq_price)
            short_liq = None
            distance = float((entry_price - long_liq) / entry_price) if entry_price > 0 else 0
            buffer = entry_price - long_liq
        else:
            # Short liquidation: entry + (margin - maintenance) / size
            maintenance = position_value * Decimal(str(rate))
            price_rise = (margin - maintenance) / size if size > 0 else Decimal("0")
            liq_price = entry_price + price_rise
            long_liq = None
            short_liq = liq_price
            distance = float((short_liq - entry_price) / entry_price) if entry_price > 0 else 0
            buffer = short_liq - entry_price

        return LiquidationPrice(
            long_liquidation_price=long_liq,
            short_liquidation_price=short_liq,
            distance_to_liquidation_pct=distance,
            price_buffer=buffer,
        )

    def calculate_margin_ratio(
        self,
        position_value: Decimal,
        margin: Decimal,
        unrealized_pnl: Decimal,
    ) -> float:
        """Calculate margin ratio (used margin / available margin)."""
        equity = margin + unrealized_pnl
        if equity <= 0:
            return 1.0
        return float(position_value * Decimal(str(self.maintenance_margin_rate)) / equity)

    def calculate_leverage(
        self,
        position_value: Decimal,
        margin: Decimal,
    ) -> float:
        """Calculate current leverage."""
        if margin <= 0:
            return 0.0
        return float(position_value / margin)

    def calculate_health_score(
        self,
        margin_ratio: float,
        distance_to_liquidation: float,
    ) -> float:
        """Calculate health score (0-100)."""
        # Weight margin ratio more heavily
        margin_score = max(0, (1 - margin_ratio) * 60)
        distance_score = min(40, distance_to_liquidation * 400)
        return min(100, margin_score + distance_score)


class RiskClassifier:
    """Classifies risk levels based on metrics."""

    def __init__(self, config: MonitorConfig):
        """Initialize classifier."""
        self.config = config

    def classify_by_margin(self, margin_ratio: float) -> RiskLevel:
        """Classify risk by margin ratio."""
        if margin_ratio >= 1.0:
            return RiskLevel.LIQUIDATION
        elif margin_ratio >= self.config.critical_threshold:
            return RiskLevel.CRITICAL
        elif margin_ratio >= self.config.liquidation_warning_threshold:
            return RiskLevel.HIGH
        elif margin_ratio >= self.config.margin_call_threshold:
            return RiskLevel.MEDIUM
        elif margin_ratio >= self.config.margin_warning_threshold:
            return RiskLevel.LOW
        return RiskLevel.SAFE

    def classify_by_distance(self, distance_pct: float) -> RiskLevel:
        """Classify risk by distance to liquidation."""
        if distance_pct <= 0:
            return RiskLevel.LIQUIDATION
        elif distance_pct <= self.config.critical_distance_pct:
            return RiskLevel.CRITICAL
        elif distance_pct <= self.config.warning_distance_pct:
            return RiskLevel.HIGH
        elif distance_pct <= self.config.safe_distance_pct:
            return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    def classify_overall(
        self,
        margin_ratio: float,
        distance_pct: float,
    ) -> RiskLevel:
        """Get overall risk level (worst of both)."""
        margin_risk = self.classify_by_margin(margin_ratio)
        distance_risk = self.classify_by_distance(distance_pct)
        # Return worst risk level
        risk_order = [
            RiskLevel.SAFE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
            RiskLevel.LIQUIDATION,
        ]
        margin_idx = risk_order.index(margin_risk)
        distance_idx = risk_order.index(distance_risk)
        return risk_order[max(margin_idx, distance_idx)]


class AlertManager:
    """Manages liquidation alerts."""

    def __init__(self, cooldown_seconds: int = 60):
        """Initialize alert manager."""
        self.cooldown_seconds = cooldown_seconds
        self.alerts: dict[str, LiquidationAlert] = {}
        self.alert_history: list[LiquidationAlert] = []
        self.last_alert_time: dict[str, datetime] = {}
        self.alert_counter = 0
        self.callbacks: list[Callable[[LiquidationAlert], None]] = []

    def register_callback(self, callback: Callable[[LiquidationAlert], None]) -> None:
        """Register alert callback."""
        self.callbacks.append(callback)

    def can_alert(self, symbol: str, alert_type: AlertType) -> bool:
        """Check if can send alert (respecting cooldown)."""
        key = f"{symbol}_{alert_type.value}"
        if key not in self.last_alert_time:
            return True
        elapsed = (datetime.now() - self.last_alert_time[key]).total_seconds()
        return elapsed >= self.cooldown_seconds

    def create_alert(
        self,
        alert_type: AlertType,
        symbol: str,
        position_risk: PositionRisk,
        message: str,
        suggested_action: ProtectionAction,
        suggested_amount: Optional[Decimal] = None,
    ) -> Optional[LiquidationAlert]:
        """Create and dispatch alert."""
        if not self.can_alert(symbol, alert_type):
            return None

        self.alert_counter += 1
        alert_id = f"liq_alert_{self.alert_counter}_{int(time.time())}"

        alert = LiquidationAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            symbol=symbol,
            position_risk=position_risk,
            message=message,
            suggested_action=suggested_action,
            suggested_amount=suggested_amount,
        )

        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[f"{symbol}_{alert_type.value}"] = datetime.now()

        # Dispatch to callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception:
                pass  # Don't let callback errors break monitoring

        return alert

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            return True
        return False

    def get_active_alerts(self, symbol: Optional[str] = None) -> list[LiquidationAlert]:
        """Get active (unacknowledged) alerts."""
        alerts = [a for a in self.alerts.values() if not a.acknowledged]
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        return alerts

    def clear_acknowledged(self) -> int:
        """Clear acknowledged alerts."""
        to_remove = [aid for aid, a in self.alerts.items() if a.acknowledged]
        for aid in to_remove:
            del self.alerts[aid]
        return len(to_remove)


@dataclass
class Position:
    """Position data for monitoring."""
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    margin: Decimal
    leverage: float = 1.0


class LiquidationMonitor:
    """Main liquidation monitoring system."""

    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize monitor."""
        self.config = config or MonitorConfig()
        self.calculator = LiquidationCalculator()
        self.classifier = RiskClassifier(self.config)
        self.alert_manager = AlertManager(self.config.alert_cooldown_seconds)
        self.positions: dict[str, Position] = {}
        self.current_prices: dict[str, Decimal] = {}
        self.position_risks: dict[str, PositionRisk] = {}
        self.auto_actions_taken: list[dict] = []
        self.start_time = datetime.now()

    def add_position(self, position: Position) -> None:
        """Add or update position to monitor."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> bool:
        """Remove position from monitoring."""
        if symbol in self.positions:
            del self.positions[symbol]
            if symbol in self.position_risks:
                del self.position_risks[symbol]
            return True
        return False

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRisk]:
        """Update price and assess risk."""
        self.current_prices[symbol] = price
        if symbol not in self.positions:
            return None
        return self._assess_position(symbol)

    def _assess_position(self, symbol: str) -> PositionRisk:
        """Assess risk for a position."""
        position = self.positions[symbol]
        current_price = self.current_prices.get(symbol, position.entry_price)

        # Calculate unrealized PnL
        if position.side == "long":
            unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            unrealized_pnl = (position.entry_price - current_price) * position.size

        # Calculate position value
        position_value = current_price * position.size

        # Calculate liquidation price
        liq_price = self.calculator.calculate_liquidation_price(
            position.side,
            position.entry_price,
            position.size,
            position.margin,
        )

        # Calculate margin metrics
        margin_ratio = self.calculator.calculate_margin_ratio(
            position_value,
            position.margin,
            unrealized_pnl,
        )
        leverage = self.calculator.calculate_leverage(position_value, position.margin)

        # Create margin info
        initial_margin = position_value / Decimal(str(position.leverage))
        maintenance_margin = position_value * Decimal(str(self.calculator.maintenance_margin_rate))
        equity = position.margin + unrealized_pnl
        available_margin = equity - maintenance_margin

        margin_info = MarginInfo(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            available_margin=max(Decimal("0"), available_margin),
            used_margin=maintenance_margin,
            margin_ratio=margin_ratio,
            leverage=leverage,
        )

        # Calculate distance to liquidation
        if position.side == "long" and liq_price.long_liquidation_price:
            distance = float((current_price - liq_price.long_liquidation_price) / current_price) if current_price > 0 else 0
        elif position.side == "short" and liq_price.short_liquidation_price:
            distance = float((liq_price.short_liquidation_price - current_price) / current_price) if current_price > 0 else 0
        else:
            distance = 1.0

        # Classify risk
        risk_level = self.classifier.classify_overall(margin_ratio, distance)

        # Calculate health score
        health_score = self.calculator.calculate_health_score(margin_ratio, distance)

        # Create position risk
        position_risk = PositionRisk(
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=risk_level,
            health_score=health_score,
            timestamp=datetime.now(),
        )

        self.position_risks[symbol] = position_risk

        # Generate alerts if needed
        self._check_and_alert(position_risk)

        # Auto-protect if enabled
        if self.config.auto_reduce_enabled:
            self._auto_protect(position_risk)

        return position_risk

    def _check_and_alert(self, risk: PositionRisk) -> None:
        """Check risk and create alerts if needed."""
        if risk.risk_level == RiskLevel.LIQUIDATION:
            self.alert_manager.create_alert(
                AlertType.POSITION_CLOSED,
                risk.symbol,
                risk,
                f"Position {risk.symbol} has been liquidated!",
                ProtectionAction.NONE,
            )
        elif risk.risk_level == RiskLevel.CRITICAL:
            self.alert_manager.create_alert(
                AlertType.IMMINENT_LIQUIDATION,
                risk.symbol,
                risk,
                f"CRITICAL: {risk.symbol} liquidation imminent! Health: {risk.health_score:.1f}%",
                ProtectionAction.CLOSE_POSITION,
            )
        elif risk.risk_level == RiskLevel.HIGH:
            reduce_amount = risk.size * Decimal(str(self.config.auto_reduce_percentage))
            self.alert_manager.create_alert(
                AlertType.APPROACHING_LIQUIDATION,
                risk.symbol,
                risk,
                f"WARNING: {risk.symbol} approaching liquidation. Health: {risk.health_score:.1f}%",
                ProtectionAction.REDUCE_POSITION,
                reduce_amount,
            )
        elif risk.risk_level == RiskLevel.MEDIUM:
            self.alert_manager.create_alert(
                AlertType.MARGIN_CALL,
                risk.symbol,
                risk,
                f"Margin call: {risk.symbol} margin ratio at {risk.margin_info.margin_ratio:.1%}",
                ProtectionAction.ADD_MARGIN,
            )
        elif risk.risk_level == RiskLevel.LOW:
            self.alert_manager.create_alert(
                AlertType.MARGIN_WARNING,
                risk.symbol,
                risk,
                f"Low margin warning: {risk.symbol} margin usage elevated",
                ProtectionAction.NONE,
            )

    def _auto_protect(self, risk: PositionRisk) -> None:
        """Auto-protect position if enabled."""
        if risk.margin_info.margin_ratio >= self.config.auto_reduce_threshold:
            reduce_amount = risk.size * Decimal(str(self.config.auto_reduce_percentage))
            self.auto_actions_taken.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": risk.symbol,
                "action": ProtectionAction.REDUCE_POSITION.value,
                "amount": str(reduce_amount),
                "reason": f"Auto-reduce triggered at {risk.margin_info.margin_ratio:.1%} margin ratio",
            })

    def assess_all(self) -> dict[str, PositionRisk]:
        """Assess all monitored positions."""
        for symbol in self.positions:
            self._assess_position(symbol)
        return self.position_risks.copy()

    def get_position_risk(self, symbol: str) -> Optional[PositionRisk]:
        """Get risk assessment for a position."""
        return self.position_risks.get(symbol)

    def get_all_risks(self) -> dict[str, PositionRisk]:
        """Get all risk assessments."""
        return self.position_risks.copy()

    def get_high_risk_positions(self) -> list[PositionRisk]:
        """Get positions with high or critical risk."""
        return [
            r for r in self.position_risks.values()
            if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.LIQUIDATION]
        ]

    def get_alerts(self, symbol: Optional[str] = None) -> list[LiquidationAlert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts(symbol)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id)

    def register_alert_callback(self, callback: Callable[[LiquidationAlert], None]) -> None:
        """Register callback for alerts."""
        self.alert_manager.register_callback(callback)

    def get_status(self) -> dict:
        """Get monitor status."""
        return {
            "positions_monitored": len(self.positions),
            "high_risk_count": len(self.get_high_risk_positions()),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "auto_actions_taken": len(self.auto_actions_taken),
            "config": self.config.to_dict(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


class PortfolioRiskMonitor:
    """Monitor risk across entire portfolio."""

    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize portfolio monitor."""
        self.monitor = LiquidationMonitor(config)
        self.portfolio_margin: Decimal = Decimal("0")
        self.total_position_value: Decimal = Decimal("0")

    def set_portfolio_margin(self, margin: Decimal) -> None:
        """Set total portfolio margin."""
        self.portfolio_margin = margin

    def add_position(self, position: Position) -> None:
        """Add position to monitor."""
        self.monitor.add_position(position)
        self._update_totals()

    def remove_position(self, symbol: str) -> bool:
        """Remove position from monitor."""
        result = self.monitor.remove_position(symbol)
        self._update_totals()
        return result

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRisk]:
        """Update price for a position."""
        result = self.monitor.update_price(symbol, price)
        self._update_totals()
        return result

    def _update_totals(self) -> None:
        """Update total position value."""
        total = Decimal("0")
        for symbol, position in self.monitor.positions.items():
            price = self.monitor.current_prices.get(symbol, position.entry_price)
            total += price * position.size
        self.total_position_value = total

    def get_portfolio_leverage(self) -> float:
        """Get overall portfolio leverage."""
        if self.portfolio_margin <= 0:
            return 0.0
        return float(self.total_position_value / self.portfolio_margin)

    def get_portfolio_health(self) -> float:
        """Get overall portfolio health score."""
        if not self.monitor.position_risks:
            return 100.0
        scores = [r.health_score for r in self.monitor.position_risks.values()]
        return sum(scores) / len(scores)

    def get_worst_position(self) -> Optional[PositionRisk]:
        """Get position with worst health."""
        if not self.monitor.position_risks:
            return None
        return min(
            self.monitor.position_risks.values(),
            key=lambda r: r.health_score,
        )

    def get_status(self) -> dict:
        """Get portfolio status."""
        return {
            "portfolio_margin": str(self.portfolio_margin),
            "total_position_value": str(self.total_position_value),
            "portfolio_leverage": self.get_portfolio_leverage(),
            "portfolio_health": self.get_portfolio_health(),
            "worst_position": self.get_worst_position().symbol if self.get_worst_position() else None,
            "monitor_status": self.monitor.get_status(),
        }
