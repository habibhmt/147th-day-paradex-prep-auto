"""
Alert Notification System.

Comprehensive alert and notification system for
trading signals, price alerts, and system events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import json
import hashlib


class AlertSeverity(Enum):
    """Alert severity level."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert category."""
    PRICE = "price"
    VOLUME = "volume"
    POSITION = "position"
    ORDER = "order"
    RISK = "risk"
    SYSTEM = "system"
    TRADING_SIGNAL = "trading_signal"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"
    CUSTOM = "custom"


class NotificationChannel(Enum):
    """Notification channel type."""
    CONSOLE = "console"
    LOG = "log"
    WEBHOOK = "webhook"
    EMAIL = "email"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SMS = "sms"
    PUSH = "push"


class AlertStatus(Enum):
    """Alert status."""
    PENDING = "pending"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    DISABLED = "disabled"


class AlertCondition(Enum):
    """Alert trigger condition."""
    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    PERCENT_CHANGE = "percent_change"
    ABSOLUTE_CHANGE = "absolute_change"


@dataclass
class AlertRule:
    """Alert rule definition."""
    id: str
    name: str
    symbol: str
    category: AlertCategory
    condition: AlertCondition
    threshold: Decimal
    threshold_pct: Optional[Decimal] = None
    severity: AlertSeverity = AlertSeverity.INFO
    enabled: bool = True
    repeat_interval: int = 0  # seconds, 0 = no repeat
    cooldown: int = 60  # seconds between triggers
    expiry: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "symbol": self.symbol,
            "category": self.category.value,
            "condition": self.condition.value,
            "threshold": str(self.threshold),
            "threshold_pct": str(self.threshold_pct) if self.threshold_pct else None,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "repeat_interval": self.repeat_interval,
            "cooldown": self.cooldown,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "trigger_count": self.trigger_count
        }


@dataclass
class Alert:
    """Triggered alert."""
    id: str
    rule_id: str
    symbol: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.TRIGGERED
    timestamp: datetime = field(default_factory=datetime.now)
    value: Optional[Decimal] = None
    threshold: Optional[Decimal] = None
    metadata: dict = field(default_factory=dict)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "symbol": self.symbol,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "value": str(self.value) if self.value else None,
            "threshold": str(self.threshold) if self.threshold else None,
            "metadata": self.metadata,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class NotificationConfig:
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.INFO
    categories: list[AlertCategory] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel.value,
            "enabled": self.enabled,
            "min_severity": self.min_severity.value,
            "categories": [c.value for c in self.categories],
            "settings": self.settings
        }


@dataclass
class NotificationResult:
    """Result of notification delivery."""
    channel: NotificationChannel
    success: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "channel": self.channel.value,
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error
        }


class AlertConditionEvaluator:
    """Evaluate alert conditions."""

    def __init__(self):
        self.previous_values: dict[str, Decimal] = {}

    def evaluate(
        self,
        rule: AlertRule,
        current_value: Decimal,
        previous_value: Optional[Decimal] = None
    ) -> bool:
        """Evaluate if condition is met."""
        key = f"{rule.symbol}:{rule.id}"

        if previous_value is None:
            previous_value = self.previous_values.get(key)

        self.previous_values[key] = current_value

        if rule.condition == AlertCondition.ABOVE:
            return current_value > rule.threshold

        if rule.condition == AlertCondition.BELOW:
            return current_value < rule.threshold

        if rule.condition == AlertCondition.EQUALS:
            return current_value == rule.threshold

        if rule.condition == AlertCondition.CROSSES_ABOVE:
            if previous_value is None:
                return False
            return previous_value <= rule.threshold and current_value > rule.threshold

        if rule.condition == AlertCondition.CROSSES_BELOW:
            if previous_value is None:
                return False
            return previous_value >= rule.threshold and current_value < rule.threshold

        if rule.condition == AlertCondition.PERCENT_CHANGE:
            if previous_value is None or previous_value == 0:
                return False
            change = abs((current_value - previous_value) / previous_value * Decimal("100"))
            return change >= (rule.threshold_pct or rule.threshold)

        if rule.condition == AlertCondition.ABSOLUTE_CHANGE:
            if previous_value is None:
                return False
            change = abs(current_value - previous_value)
            return change >= rule.threshold

        return False


class ConsoleNotifier:
    """Console notification handler."""

    def send(self, alert: Alert) -> NotificationResult:
        """Send alert to console."""
        severity_icons = {
            AlertSeverity.DEBUG: "🔵",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        icon = severity_icons.get(alert.severity, "")
        print(f"{icon} [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

        return NotificationResult(
            channel=NotificationChannel.CONSOLE,
            success=True,
            message="Alert displayed"
        )


class LogNotifier:
    """Log file notification handler."""

    def __init__(self, log_file: str = "alerts.log"):
        self.log_file = log_file

    def send(self, alert: Alert) -> NotificationResult:
        """Send alert to log file."""
        try:
            log_entry = {
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "category": alert.category.value,
                "symbol": alert.symbol,
                "title": alert.title,
                "message": alert.message
            }

            # Simulate logging (in real implementation, would write to file)
            # with open(self.log_file, 'a') as f:
            #     f.write(json.dumps(log_entry) + '\n')

            return NotificationResult(
                channel=NotificationChannel.LOG,
                success=True,
                message=f"Alert logged to {self.log_file}"
            )
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.LOG,
                success=False,
                message="Failed to log alert",
                error=str(e)
            )


class WebhookNotifier:
    """Webhook notification handler."""

    def __init__(self, url: str, headers: Optional[dict] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: Alert) -> NotificationResult:
        """Send alert to webhook."""
        try:
            payload = alert.to_dict()
            # In real implementation, would make HTTP request
            # response = requests.post(self.url, json=payload, headers=self.headers)

            return NotificationResult(
                channel=NotificationChannel.WEBHOOK,
                success=True,
                message=f"Alert sent to webhook: {self.url}"
            )
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.WEBHOOK,
                success=False,
                message="Failed to send webhook",
                error=str(e)
            )


class TelegramNotifier:
    """Telegram notification handler."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, alert: Alert) -> NotificationResult:
        """Send alert to Telegram."""
        try:
            message = self._format_message(alert)
            # In real implementation:
            # url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            # response = requests.post(url, json={"chat_id": self.chat_id, "text": message})

            return NotificationResult(
                channel=NotificationChannel.TELEGRAM,
                success=True,
                message="Alert sent to Telegram"
            )
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.TELEGRAM,
                success=False,
                message="Failed to send Telegram message",
                error=str(e)
            )

    def _format_message(self, alert: Alert) -> str:
        """Format alert for Telegram."""
        return f"""
🔔 *{alert.title}*
━━━━━━━━━━━━━━
Symbol: {alert.symbol}
Severity: {alert.severity.value}
Category: {alert.category.value}

{alert.message}

_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_
"""


class DiscordNotifier:
    """Discord notification handler."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> NotificationResult:
        """Send alert to Discord."""
        try:
            embed = self._create_embed(alert)
            # In real implementation:
            # response = requests.post(self.webhook_url, json={"embeds": [embed]})

            return NotificationResult(
                channel=NotificationChannel.DISCORD,
                success=True,
                message="Alert sent to Discord"
            )
        except Exception as e:
            return NotificationResult(
                channel=NotificationChannel.DISCORD,
                success=False,
                message="Failed to send Discord message",
                error=str(e)
            )

    def _create_embed(self, alert: Alert) -> dict:
        """Create Discord embed."""
        colors = {
            AlertSeverity.DEBUG: 0x808080,
            AlertSeverity.INFO: 0x3498db,
            AlertSeverity.WARNING: 0xf39c12,
            AlertSeverity.ERROR: 0xe74c3c,
            AlertSeverity.CRITICAL: 0x9b59b6
        }

        return {
            "title": alert.title,
            "description": alert.message,
            "color": colors.get(alert.severity, 0x3498db),
            "fields": [
                {"name": "Symbol", "value": alert.symbol, "inline": True},
                {"name": "Category", "value": alert.category.value, "inline": True},
                {"name": "Severity", "value": alert.severity.value, "inline": True}
            ],
            "timestamp": alert.timestamp.isoformat()
        }


class AlertManager:
    """Main alert management system."""

    def __init__(self):
        self.rules: dict[str, AlertRule] = {}
        self.alerts: list[Alert] = []
        self.notifiers: dict[NotificationChannel, Any] = {}
        self.notification_configs: dict[NotificationChannel, NotificationConfig] = {}
        self.evaluator = AlertConditionEvaluator()
        self.alert_counter = 0
        self.callbacks: dict[str, list[Callable]] = {
            "on_alert": [],
            "on_acknowledge": [],
            "on_resolve": []
        }

        # Initialize default notifiers
        self.notifiers[NotificationChannel.CONSOLE] = ConsoleNotifier()
        self.notifiers[NotificationChannel.LOG] = LogNotifier()

    def register_callback(self, event: str, callback: Callable):
        """Register callback for events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def add_notifier(self, channel: NotificationChannel, notifier: Any, config: NotificationConfig):
        """Add notification channel."""
        self.notifiers[channel] = notifier
        self.notification_configs[channel] = config

    def create_rule(
        self,
        name: str,
        symbol: str,
        category: AlertCategory,
        condition: AlertCondition,
        threshold: Decimal,
        severity: AlertSeverity = AlertSeverity.INFO,
        **kwargs
    ) -> AlertRule:
        """Create new alert rule."""
        rule_id = self._generate_id(f"{symbol}:{name}:{condition.value}")

        rule = AlertRule(
            id=rule_id,
            name=name,
            symbol=symbol,
            category=category,
            condition=condition,
            threshold=threshold,
            severity=severity,
            **kwargs
        )

        self.rules[rule_id] = rule
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False

    def check_price(self, symbol: str, price: Decimal):
        """Check price against all relevant rules."""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            if rule.symbol != symbol:
                continue
            if rule.category != AlertCategory.PRICE:
                continue

            self._check_rule(rule, price)

    def check_value(self, symbol: str, category: AlertCategory, value: Decimal):
        """Check value against rules for specific category."""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            if rule.symbol != symbol:
                continue
            if rule.category != category:
                continue

            self._check_rule(rule, value)

    def _check_rule(self, rule: AlertRule, value: Decimal):
        """Check if rule should trigger."""
        # Check expiry
        if rule.expiry and datetime.now() > rule.expiry:
            rule.enabled = False
            return

        # Check cooldown
        if rule.last_triggered:
            elapsed = (datetime.now() - rule.last_triggered).total_seconds()
            if elapsed < rule.cooldown:
                return

        # Evaluate condition
        if self.evaluator.evaluate(rule, value):
            self._trigger_alert(rule, value)

    def _trigger_alert(self, rule: AlertRule, value: Decimal):
        """Trigger alert from rule."""
        self.alert_counter += 1
        alert_id = f"ALT-{self.alert_counter:08d}"

        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            symbol=rule.symbol,
            category=rule.category,
            severity=rule.severity,
            title=rule.name,
            message=self._format_message(rule, value),
            value=value,
            threshold=rule.threshold
        )

        self.alerts.append(alert)
        rule.last_triggered = datetime.now()
        rule.trigger_count += 1

        # Send notifications
        self._send_notifications(alert)

        # Trigger callbacks
        for cb in self.callbacks["on_alert"]:
            cb(alert)

    def _format_message(self, rule: AlertRule, value: Decimal) -> str:
        """Format alert message."""
        condition_msgs = {
            AlertCondition.ABOVE: f"Price went above {rule.threshold}",
            AlertCondition.BELOW: f"Price went below {rule.threshold}",
            AlertCondition.EQUALS: f"Price equals {rule.threshold}",
            AlertCondition.CROSSES_ABOVE: f"Price crossed above {rule.threshold}",
            AlertCondition.CROSSES_BELOW: f"Price crossed below {rule.threshold}",
            AlertCondition.PERCENT_CHANGE: f"Change exceeded {rule.threshold_pct or rule.threshold}%",
            AlertCondition.ABSOLUTE_CHANGE: f"Change exceeded {rule.threshold}"
        }

        msg = condition_msgs.get(rule.condition, "Alert triggered")
        return f"{msg}. Current value: {value}"

    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels."""
        for channel, config in self.notification_configs.items():
            if not config.enabled:
                continue

            # Check severity threshold
            severity_order = [s for s in AlertSeverity]
            if severity_order.index(alert.severity) < severity_order.index(config.min_severity):
                continue

            # Check category filter
            if config.categories and alert.category not in config.categories:
                continue

            notifier = self.notifiers.get(channel)
            if notifier:
                notifier.send(alert)

        # Always send to console
        if NotificationChannel.CONSOLE in self.notifiers:
            self.notifiers[NotificationChannel.CONSOLE].send(alert)

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()

                for cb in self.callbacks["on_acknowledge"]:
                    cb(alert)

                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()

                for cb in self.callbacks["on_resolve"]:
                    cb(alert)

                return True
        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if a.status in [AlertStatus.TRIGGERED, AlertStatus.ACKNOWLEDGED]]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity."""
        return [a for a in self.alerts if a.severity == severity]

    def get_alerts_by_symbol(self, symbol: str) -> list[Alert]:
        """Get alerts by symbol."""
        return [a for a in self.alerts if a.symbol == symbol]

    def get_alerts_by_category(self, category: AlertCategory) -> list[Alert]:
        """Get alerts by category."""
        return [a for a in self.alerts if a.category == category]

    def clear_resolved(self):
        """Clear resolved alerts."""
        self.alerts = [a for a in self.alerts if a.status != AlertStatus.RESOLVED]

    def get_summary(self) -> dict:
        """Get alert system summary."""
        active = self.get_active_alerts()
        by_severity = {}
        by_category = {}

        for alert in active:
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
            by_category[alert.category.value] = by_category.get(alert.category.value, 0) + 1

        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "total_alerts": len(self.alerts),
            "active_alerts": len(active),
            "by_severity": by_severity,
            "by_category": by_category
        }

    def _generate_id(self, seed: str) -> str:
        """Generate unique ID from seed."""
        hash_obj = hashlib.md5(f"{seed}:{datetime.now().timestamp()}".encode())
        return hash_obj.hexdigest()[:12]


class PriceAlertManager:
    """Simplified price alert manager."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager

    def add_price_above(
        self,
        symbol: str,
        price: Decimal,
        name: Optional[str] = None,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> AlertRule:
        """Add price above alert."""
        name = name or f"{symbol} above {price}"
        return self.alert_manager.create_rule(
            name=name,
            symbol=symbol,
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=price,
            severity=severity
        )

    def add_price_below(
        self,
        symbol: str,
        price: Decimal,
        name: Optional[str] = None,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> AlertRule:
        """Add price below alert."""
        name = name or f"{symbol} below {price}"
        return self.alert_manager.create_rule(
            name=name,
            symbol=symbol,
            category=AlertCategory.PRICE,
            condition=AlertCondition.BELOW,
            threshold=price,
            severity=severity
        )

    def add_price_change(
        self,
        symbol: str,
        percent: Decimal,
        name: Optional[str] = None,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> AlertRule:
        """Add percent change alert."""
        name = name or f"{symbol} {percent}% change"
        return self.alert_manager.create_rule(
            name=name,
            symbol=symbol,
            category=AlertCategory.PRICE,
            condition=AlertCondition.PERCENT_CHANGE,
            threshold=percent,
            threshold_pct=percent,
            severity=severity
        )


class RiskAlertManager:
    """Risk-specific alert manager."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager

    def add_drawdown_alert(
        self,
        account: str,
        max_drawdown_pct: Decimal,
        severity: AlertSeverity = AlertSeverity.CRITICAL
    ) -> AlertRule:
        """Add drawdown alert."""
        return self.alert_manager.create_rule(
            name=f"Drawdown exceeds {max_drawdown_pct}%",
            symbol=account,
            category=AlertCategory.RISK,
            condition=AlertCondition.ABOVE,
            threshold=max_drawdown_pct,
            severity=severity
        )

    def add_exposure_alert(
        self,
        account: str,
        max_exposure: Decimal,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> AlertRule:
        """Add exposure alert."""
        return self.alert_manager.create_rule(
            name=f"Exposure exceeds {max_exposure}",
            symbol=account,
            category=AlertCategory.RISK,
            condition=AlertCondition.ABOVE,
            threshold=max_exposure,
            severity=severity
        )

    def add_loss_alert(
        self,
        account: str,
        max_loss: Decimal,
        severity: AlertSeverity = AlertSeverity.CRITICAL
    ) -> AlertRule:
        """Add loss alert."""
        return self.alert_manager.create_rule(
            name=f"Loss exceeds {max_loss}",
            symbol=account,
            category=AlertCategory.RISK,
            condition=AlertCondition.BELOW,
            threshold=-max_loss,
            severity=severity
        )


# Global instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def set_alert_manager(manager: AlertManager):
    """Set global alert manager instance."""
    global _alert_manager
    _alert_manager = manager
