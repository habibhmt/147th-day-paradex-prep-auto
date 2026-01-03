"""
Tests for Alert Notification System.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.analytics.alert_system import (
    AlertSeverity, AlertCategory, NotificationChannel, AlertStatus, AlertCondition,
    AlertRule, Alert, NotificationConfig, NotificationResult,
    AlertConditionEvaluator, ConsoleNotifier, LogNotifier, WebhookNotifier,
    TelegramNotifier, DiscordNotifier, AlertManager, PriceAlertManager,
    RiskAlertManager, get_alert_manager, set_alert_manager
)


# ============== Fixtures ==============

@pytest.fixture
def alert_manager():
    return AlertManager()


@pytest.fixture
def sample_rule():
    return AlertRule(
        id="test-rule-001",
        name="BTC Price Alert",
        symbol="BTC-USD",
        category=AlertCategory.PRICE,
        condition=AlertCondition.ABOVE,
        threshold=Decimal("50000"),
        severity=AlertSeverity.WARNING
    )


@pytest.fixture
def sample_alert():
    return Alert(
        id="ALT-00000001",
        rule_id="test-rule-001",
        symbol="BTC-USD",
        category=AlertCategory.PRICE,
        severity=AlertSeverity.WARNING,
        title="BTC Price Alert",
        message="Price went above 50000. Current value: 51000",
        value=Decimal("51000"),
        threshold=Decimal("50000")
    )


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_alert_severity_values(self):
        assert AlertSeverity.DEBUG.value == "debug"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_category_values(self):
        assert AlertCategory.PRICE.value == "price"
        assert AlertCategory.RISK.value == "risk"
        assert AlertCategory.TRADING_SIGNAL.value == "trading_signal"

    def test_notification_channel_values(self):
        assert NotificationChannel.CONSOLE.value == "console"
        assert NotificationChannel.TELEGRAM.value == "telegram"
        assert NotificationChannel.DISCORD.value == "discord"

    def test_alert_status_values(self):
        assert AlertStatus.PENDING.value == "pending"
        assert AlertStatus.TRIGGERED.value == "triggered"
        assert AlertStatus.RESOLVED.value == "resolved"

    def test_alert_condition_values(self):
        assert AlertCondition.ABOVE.value == "above"
        assert AlertCondition.CROSSES_ABOVE.value == "crosses_above"


# ============== Data Class Tests ==============

class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_creation(self, sample_rule):
        assert sample_rule.id == "test-rule-001"
        assert sample_rule.symbol == "BTC-USD"
        assert sample_rule.enabled is True

    def test_defaults(self):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="ETH-USD",
            category=AlertCategory.VOLUME,
            condition=AlertCondition.BELOW,
            threshold=Decimal("1000000")
        )
        assert rule.severity == AlertSeverity.INFO
        assert rule.enabled is True
        assert rule.trigger_count == 0

    def test_to_dict(self, sample_rule):
        result = sample_rule.to_dict()
        assert result["id"] == "test-rule-001"
        assert result["category"] == "price"
        assert result["condition"] == "above"


class TestAlert:
    """Test Alert dataclass."""

    def test_creation(self, sample_alert):
        assert sample_alert.id == "ALT-00000001"
        assert sample_alert.status == AlertStatus.TRIGGERED

    def test_to_dict(self, sample_alert):
        result = sample_alert.to_dict()
        assert result["severity"] == "warning"
        assert result["status"] == "triggered"


class TestNotificationConfig:
    """Test NotificationConfig dataclass."""

    def test_creation(self):
        config = NotificationConfig(
            channel=NotificationChannel.TELEGRAM,
            enabled=True,
            min_severity=AlertSeverity.WARNING,
            categories=[AlertCategory.PRICE, AlertCategory.RISK]
        )
        assert config.channel == NotificationChannel.TELEGRAM
        assert len(config.categories) == 2

    def test_to_dict(self):
        config = NotificationConfig(
            channel=NotificationChannel.DISCORD,
            min_severity=AlertSeverity.ERROR
        )
        result = config.to_dict()
        assert result["channel"] == "discord"
        assert result["min_severity"] == "error"


class TestNotificationResult:
    """Test NotificationResult dataclass."""

    def test_creation(self):
        result = NotificationResult(
            channel=NotificationChannel.WEBHOOK,
            success=True,
            message="Sent successfully"
        )
        assert result.success is True

    def test_failed_result(self):
        result = NotificationResult(
            channel=NotificationChannel.EMAIL,
            success=False,
            message="Failed to send",
            error="Connection refused"
        )
        assert result.success is False
        assert result.error is not None


# ============== Alert Condition Evaluator Tests ==============

class TestAlertConditionEvaluator:
    """Test AlertConditionEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return AlertConditionEvaluator()

    def test_above_condition(self, evaluator, sample_rule):
        # Price above threshold
        assert evaluator.evaluate(sample_rule, Decimal("51000"))
        # Price below threshold
        assert not evaluator.evaluate(sample_rule, Decimal("49000"))

    def test_below_condition(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="ETH-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.BELOW,
            threshold=Decimal("3000")
        )
        assert evaluator.evaluate(rule, Decimal("2900"))
        assert not evaluator.evaluate(rule, Decimal("3100"))

    def test_equals_condition(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.EQUALS,
            threshold=Decimal("50000")
        )
        assert evaluator.evaluate(rule, Decimal("50000"))
        assert not evaluator.evaluate(rule, Decimal("50001"))

    def test_crosses_above(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.CROSSES_ABOVE,
            threshold=Decimal("50000")
        )
        # First call sets previous
        assert not evaluator.evaluate(rule, Decimal("49000"))
        # Now crosses above
        assert evaluator.evaluate(rule, Decimal("51000"))
        # Already above, doesn't cross
        assert not evaluator.evaluate(rule, Decimal("52000"))

    def test_crosses_below(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.CROSSES_BELOW,
            threshold=Decimal("50000")
        )
        # First call sets previous
        assert not evaluator.evaluate(rule, Decimal("51000"))
        # Now crosses below
        assert evaluator.evaluate(rule, Decimal("49000"))

    def test_percent_change(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.PERCENT_CHANGE,
            threshold=Decimal("5"),
            threshold_pct=Decimal("5")
        )
        # First call sets previous
        assert not evaluator.evaluate(rule, Decimal("50000"))
        # 10% change - triggers
        assert evaluator.evaluate(rule, Decimal("55000"))

    def test_absolute_change(self, evaluator):
        rule = AlertRule(
            id="r1",
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABSOLUTE_CHANGE,
            threshold=Decimal("1000")
        )
        # First call sets previous
        assert not evaluator.evaluate(rule, Decimal("50000"))
        # Change of 2000 - triggers
        assert evaluator.evaluate(rule, Decimal("52000"))


# ============== Notifier Tests ==============

class TestConsoleNotifier:
    """Test ConsoleNotifier."""

    def test_send(self, sample_alert, capsys):
        notifier = ConsoleNotifier()
        result = notifier.send(sample_alert)

        assert result.success is True
        assert result.channel == NotificationChannel.CONSOLE

        captured = capsys.readouterr()
        assert "BTC Price Alert" in captured.out


class TestLogNotifier:
    """Test LogNotifier."""

    def test_send(self, sample_alert):
        notifier = LogNotifier("test.log")
        result = notifier.send(sample_alert)

        assert result.success is True
        assert result.channel == NotificationChannel.LOG


class TestWebhookNotifier:
    """Test WebhookNotifier."""

    def test_send(self, sample_alert):
        notifier = WebhookNotifier("http://example.com/webhook")
        result = notifier.send(sample_alert)

        assert result.success is True
        assert result.channel == NotificationChannel.WEBHOOK


class TestTelegramNotifier:
    """Test TelegramNotifier."""

    def test_send(self, sample_alert):
        notifier = TelegramNotifier("bot_token", "chat_id")
        result = notifier.send(sample_alert)

        assert result.success is True
        assert result.channel == NotificationChannel.TELEGRAM

    def test_format_message(self, sample_alert):
        notifier = TelegramNotifier("bot_token", "chat_id")
        message = notifier._format_message(sample_alert)

        assert "BTC Price Alert" in message
        assert "BTC-USD" in message


class TestDiscordNotifier:
    """Test DiscordNotifier."""

    def test_send(self, sample_alert):
        notifier = DiscordNotifier("http://discord.webhook")
        result = notifier.send(sample_alert)

        assert result.success is True
        assert result.channel == NotificationChannel.DISCORD

    def test_create_embed(self, sample_alert):
        notifier = DiscordNotifier("http://discord.webhook")
        embed = notifier._create_embed(sample_alert)

        assert embed["title"] == "BTC Price Alert"
        assert "fields" in embed


# ============== Alert Manager Tests ==============

class TestAlertManager:
    """Test AlertManager."""

    def test_init(self, alert_manager):
        assert alert_manager.rules == {}
        assert NotificationChannel.CONSOLE in alert_manager.notifiers

    def test_register_callback(self, alert_manager):
        callback = Mock()
        alert_manager.register_callback("on_alert", callback)
        assert callback in alert_manager.callbacks["on_alert"]

    def test_create_rule(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test Alert",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000")
        )

        assert rule.name == "Test Alert"
        assert rule.id in alert_manager.rules

    def test_remove_rule(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000")
        )

        assert alert_manager.remove_rule(rule.id)
        assert rule.id not in alert_manager.rules

    def test_remove_nonexistent_rule(self, alert_manager):
        assert not alert_manager.remove_rule("nonexistent")

    def test_enable_disable_rule(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000")
        )

        alert_manager.disable_rule(rule.id)
        assert not alert_manager.rules[rule.id].enabled

        alert_manager.enable_rule(rule.id)
        assert alert_manager.rules[rule.id].enabled

    def test_check_price_triggers_alert(self, alert_manager):
        rule = alert_manager.create_rule(
            name="BTC Above 50k",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        # Price below threshold - no alert
        alert_manager.check_price("BTC-USD", Decimal("49000"))
        assert len(alert_manager.alerts) == 0

        # Price above threshold - alert triggered
        alert_manager.check_price("BTC-USD", Decimal("51000"))
        assert len(alert_manager.alerts) == 1
        assert alert_manager.alerts[0].symbol == "BTC-USD"

    def test_check_value(self, alert_manager):
        rule = alert_manager.create_rule(
            name="High Volume",
            symbol="BTC-USD",
            category=AlertCategory.VOLUME,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("1000000"),
            cooldown=0
        )

        alert_manager.check_value("BTC-USD", AlertCategory.VOLUME, Decimal("2000000"))
        assert len(alert_manager.alerts) == 1

    def test_cooldown_prevents_repeat(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=60
        )

        # First trigger
        alert_manager.check_price("BTC-USD", Decimal("51000"))
        assert len(alert_manager.alerts) == 1

        # Second trigger within cooldown - should not create new alert
        alert_manager.check_price("BTC-USD", Decimal("52000"))
        assert len(alert_manager.alerts) == 1

    def test_acknowledge_alert(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_id = alert_manager.alerts[0].id

        assert alert_manager.acknowledge(alert_id)
        assert alert_manager.alerts[0].status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_id = alert_manager.alerts[0].id

        assert alert_manager.resolve(alert_id)
        assert alert_manager.alerts[0].status == AlertStatus.RESOLVED

    def test_get_active_alerts(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_manager.check_price("BTC-USD", Decimal("52000"))

        # Resolve one
        alert_manager.resolve(alert_manager.alerts[0].id)

        active = alert_manager.get_active_alerts()
        assert len(active) == 1

    def test_get_alerts_by_severity(self, alert_manager):
        alert_manager.create_rule(
            name="Warning",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            severity=AlertSeverity.WARNING,
            cooldown=0
        )
        alert_manager.create_rule(
            name="Critical",
            symbol="ETH-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("3000"),
            severity=AlertSeverity.CRITICAL,
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_manager.check_price("ETH-USD", Decimal("3100"))

        warnings = alert_manager.get_alerts_by_severity(AlertSeverity.WARNING)
        criticals = alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)

        assert len(warnings) == 1
        assert len(criticals) == 1

    def test_get_alerts_by_symbol(self, alert_manager):
        alert_manager.create_rule(
            name="BTC Alert",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )
        alert_manager.create_rule(
            name="ETH Alert",
            symbol="ETH-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("3000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_manager.check_price("ETH-USD", Decimal("3100"))

        btc_alerts = alert_manager.get_alerts_by_symbol("BTC-USD")
        assert len(btc_alerts) == 1
        assert btc_alerts[0].symbol == "BTC-USD"

    def test_clear_resolved(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        alert_manager.check_price("BTC-USD", Decimal("52000"))

        alert_manager.resolve(alert_manager.alerts[0].id)
        alert_manager.clear_resolved()

        assert len(alert_manager.alerts) == 1

    def test_get_summary(self, alert_manager):
        rule = alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))

        summary = alert_manager.get_summary()
        assert summary["total_rules"] == 1
        assert summary["active_alerts"] == 1

    def test_callbacks_triggered(self, alert_manager):
        callback = Mock()
        alert_manager.register_callback("on_alert", callback)

        alert_manager.create_rule(
            name="Test",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            cooldown=0
        )

        alert_manager.check_price("BTC-USD", Decimal("51000"))
        assert callback.called


# ============== Price Alert Manager Tests ==============

class TestPriceAlertManager:
    """Test PriceAlertManager."""

    @pytest.fixture
    def price_alert_manager(self, alert_manager):
        return PriceAlertManager(alert_manager)

    def test_add_price_above(self, price_alert_manager, alert_manager):
        rule = price_alert_manager.add_price_above("BTC-USD", Decimal("50000"))
        assert rule.condition == AlertCondition.ABOVE
        assert rule.id in alert_manager.rules

    def test_add_price_below(self, price_alert_manager, alert_manager):
        rule = price_alert_manager.add_price_below("BTC-USD", Decimal("45000"))
        assert rule.condition == AlertCondition.BELOW

    def test_add_price_change(self, price_alert_manager, alert_manager):
        rule = price_alert_manager.add_price_change("BTC-USD", Decimal("5"))
        assert rule.condition == AlertCondition.PERCENT_CHANGE


# ============== Risk Alert Manager Tests ==============

class TestRiskAlertManager:
    """Test RiskAlertManager."""

    @pytest.fixture
    def risk_alert_manager(self, alert_manager):
        return RiskAlertManager(alert_manager)

    def test_add_drawdown_alert(self, risk_alert_manager, alert_manager):
        rule = risk_alert_manager.add_drawdown_alert("account1", Decimal("10"))
        assert rule.category == AlertCategory.RISK
        assert rule.severity == AlertSeverity.CRITICAL

    def test_add_exposure_alert(self, risk_alert_manager, alert_manager):
        rule = risk_alert_manager.add_exposure_alert("account1", Decimal("100000"))
        assert rule.category == AlertCategory.RISK

    def test_add_loss_alert(self, risk_alert_manager, alert_manager):
        rule = risk_alert_manager.add_loss_alert("account1", Decimal("5000"))
        assert rule.category == AlertCategory.RISK


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global alert manager."""

    def test_get_alert_manager(self):
        manager = get_alert_manager()
        assert isinstance(manager, AlertManager)

    def test_set_alert_manager(self):
        custom = AlertManager()
        custom.create_rule(
            name="Custom",
            symbol="TEST",
            category=AlertCategory.CUSTOM,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("100")
        )
        set_alert_manager(custom)

        manager = get_alert_manager()
        assert len(manager.rules) == 1


# ============== Integration Tests ==============

class TestAlertIntegration:
    """Integration tests."""

    def test_full_alert_flow(self):
        manager = AlertManager()

        # Add notification config
        config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True,
            min_severity=AlertSeverity.INFO
        )
        manager.add_notifier(NotificationChannel.LOG, LogNotifier(), config)

        # Create rules
        manager.create_rule(
            name="BTC Above 50k",
            symbol="BTC-USD",
            category=AlertCategory.PRICE,
            condition=AlertCondition.ABOVE,
            threshold=Decimal("50000"),
            severity=AlertSeverity.WARNING,
            cooldown=0
        )

        # Check prices
        manager.check_price("BTC-USD", Decimal("49000"))  # No alert
        manager.check_price("BTC-USD", Decimal("51000"))  # Alert triggered

        # Verify
        assert len(manager.alerts) == 1
        assert manager.alerts[0].status == AlertStatus.TRIGGERED

        # Acknowledge and resolve
        manager.acknowledge(manager.alerts[0].id)
        assert manager.alerts[0].status == AlertStatus.ACKNOWLEDGED

        manager.resolve(manager.alerts[0].id)
        assert manager.alerts[0].status == AlertStatus.RESOLVED

    def test_multi_symbol_monitoring(self):
        manager = AlertManager()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        thresholds = [50000, 3000, 100]

        for symbol, threshold in zip(symbols, thresholds):
            manager.create_rule(
                name=f"{symbol} Price Alert",
                symbol=symbol,
                category=AlertCategory.PRICE,
                condition=AlertCondition.ABOVE,
                threshold=Decimal(str(threshold)),
                cooldown=0
            )

        # Trigger some alerts
        manager.check_price("BTC-USD", Decimal("51000"))
        manager.check_price("ETH-USD", Decimal("3100"))
        manager.check_price("SOL-USD", Decimal("95"))  # Below threshold

        assert len(manager.alerts) == 2

        summary = manager.get_summary()
        assert summary["active_alerts"] == 2
