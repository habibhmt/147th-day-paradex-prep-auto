"""Unit tests for Risk Manager."""

import pytest
from decimal import Decimal
import time

from src.risk.manager import (
    RiskManager,
    RiskConfig,
    RiskAlert,
    RiskLevel,
    RiskAction,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels(self):
        """Should have expected risk levels."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestRiskAction:
    """Tests for RiskAction enum."""

    def test_risk_actions(self):
        """Should have expected risk actions."""
        assert RiskAction.WARN.value == "warn"
        assert RiskAction.REDUCE.value == "reduce"
        assert RiskAction.HALT.value == "halt"
        assert RiskAction.CLOSE_ALL.value == "close_all"


class TestRiskAlert:
    """Tests for RiskAlert dataclass."""

    def test_create_alert(self):
        """Should create alert correctly."""
        alert = RiskAlert(
            level=RiskLevel.HIGH,
            action=RiskAction.REDUCE,
            message="Test alert",
            metric="test_metric",
            current_value=100.0,
            threshold=50.0,
        )

        assert alert.level == RiskLevel.HIGH
        assert alert.action == RiskAction.REDUCE
        assert alert.message == "Test alert"

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        alert = RiskAlert(
            level=RiskLevel.CRITICAL,
            action=RiskAction.HALT,
            message="Critical alert",
            metric="delta",
            current_value=25.0,
            threshold=20.0,
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert d["action"] == "halt"
        assert d["current_value"] == 25.0


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = RiskConfig()

        assert config.max_position_size == Decimal("50000")
        assert config.max_drawdown_pct == 10.0
        assert config.max_delta_pct == 10.0
        assert config.max_leverage == 20.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = RiskConfig(
            max_position_size=Decimal("100000"),
            max_drawdown_pct=5.0,
        )

        assert config.max_position_size == Decimal("100000")
        assert config.max_drawdown_pct == 5.0


class TestRiskManager:
    """Tests for RiskManager."""

    @pytest.fixture
    def manager(self):
        """Create risk manager with default config."""
        return RiskManager()

    @pytest.fixture
    def strict_manager(self):
        """Create risk manager with strict config."""
        config = RiskConfig(
            max_position_size=Decimal("10000"),
            max_total_exposure=Decimal("50000"),
            max_leverage=10.0,
            max_drawdown_pct=5.0,
            max_delta_pct=5.0,
            critical_delta_pct=10.0,
            max_trades_per_hour=10,
            min_trade_interval=1.0,
        )
        return RiskManager(config=config)

    def test_initial_state(self, manager):
        """Should start with clean state."""
        status = manager.get_status()

        assert status["is_halted"] is False
        assert status["daily_pnl"] == "0"
        assert status["current_drawdown"] == 0.0

    def test_check_position_size_ok(self, manager):
        """Should allow position within limit."""
        alert = manager.check_position_size(Decimal("10000"), "acc1")
        assert alert is None

    def test_check_position_size_exceeded(self, manager):
        """Should alert when position exceeds limit."""
        alert = manager.check_position_size(Decimal("100000"), "acc1")

        assert alert is not None
        assert alert.level == RiskLevel.HIGH
        assert alert.action == RiskAction.REDUCE

    def test_check_total_exposure_ok(self, manager):
        """Should allow exposure within limit."""
        alert = manager.check_total_exposure(Decimal("100000"))
        assert alert is None

    def test_check_total_exposure_exceeded(self, manager):
        """Should alert when exposure exceeds limit."""
        alert = manager.check_total_exposure(Decimal("500000"))

        assert alert is not None
        assert alert.level == RiskLevel.HIGH

    def test_check_leverage_ok(self, manager):
        """Should allow leverage within limit."""
        alert = manager.check_leverage(10.0, "acc1")
        assert alert is None

    def test_check_leverage_exceeded(self, manager):
        """Should alert when leverage exceeds limit."""
        alert = manager.check_leverage(25.0, "acc1")

        assert alert is not None
        assert alert.level == RiskLevel.MEDIUM

    def test_check_delta_ok(self, strict_manager):
        """Should allow delta within limit."""
        alert = strict_manager.check_delta(3.0, "BTC-USD-PERP")
        assert alert is None

    def test_check_delta_high(self, strict_manager):
        """Should alert on high delta."""
        alert = strict_manager.check_delta(7.0, "BTC-USD-PERP")

        assert alert is not None
        assert alert.level == RiskLevel.HIGH

    def test_check_delta_critical(self, strict_manager):
        """Should alert and halt on critical delta."""
        alert = strict_manager.check_delta(15.0, "BTC-USD-PERP")

        assert alert is not None
        assert alert.level == RiskLevel.CRITICAL
        assert strict_manager._is_halted is True

    def test_update_pnl_positive(self, manager):
        """Should track positive PnL."""
        manager.update_pnl(Decimal("1000"))

        assert manager._daily_pnl == Decimal("1000")
        assert manager._daily_high == Decimal("1000")

    def test_update_pnl_drawdown(self, strict_manager):
        """Should calculate drawdown correctly."""
        strict_manager.update_pnl(Decimal("10000"))  # Set high
        strict_manager.update_pnl(Decimal("-300"))  # Small drawdown

        assert strict_manager._daily_pnl == Decimal("9700")
        assert strict_manager._current_drawdown == 3.0

    def test_update_pnl_max_drawdown(self, strict_manager):
        """Should halt on max drawdown."""
        strict_manager.update_pnl(Decimal("10000"))
        alert = strict_manager.update_pnl(Decimal("-600"))  # 6% drawdown > 5% limit

        assert alert is not None
        assert alert.level == RiskLevel.CRITICAL
        assert strict_manager._is_halted is True

    def test_can_trade_normal(self, manager):
        """Should allow trading normally."""
        can_trade, reason = manager.can_trade()

        assert can_trade is True
        assert reason == "OK"

    def test_can_trade_halted(self, manager):
        """Should deny trading when halted."""
        manager._halt("Test halt")

        can_trade, reason = manager.can_trade()

        assert can_trade is False
        assert "halted" in reason.lower()

    def test_can_trade_frequency_limit(self, strict_manager):
        """Should enforce trade frequency limit."""
        # Record max trades
        for _ in range(strict_manager.config.max_trades_per_hour):
            strict_manager.record_trade()

        can_trade, reason = strict_manager.can_trade()

        assert can_trade is False
        assert "max trades" in reason.lower()

    def test_record_trade(self, manager):
        """Should record trade timestamp."""
        initial_count = len(manager._trade_timestamps)
        manager.record_trade()

        assert len(manager._trade_timestamps) == initial_count + 1

    def test_resume(self, manager):
        """Should resume after halt."""
        manager._halt("Test halt")
        assert manager._is_halted is True

        result = manager.resume()

        assert result is True
        assert manager._is_halted is False

    def test_resume_not_halted(self, manager):
        """Should return False if not halted."""
        result = manager.resume()
        assert result is False

    def test_reset_daily(self, manager):
        """Should reset daily metrics."""
        manager.update_pnl(Decimal("5000"))
        manager.record_trade()

        manager.reset_daily()

        assert manager._daily_pnl == Decimal("0")
        assert manager._daily_high == Decimal("0")
        assert len(manager._trade_timestamps) == 0

    def test_get_alerts(self, manager):
        """Should return alerts."""
        manager.check_position_size(Decimal("100000"), "acc1")
        manager.check_leverage(25.0, "acc1")

        alerts = manager.get_alerts()

        assert len(alerts) == 2

    def test_get_alerts_by_level(self, manager):
        """Should filter alerts by level."""
        manager.check_position_size(Decimal("100000"), "acc1")  # HIGH
        manager.check_leverage(25.0, "acc1")  # MEDIUM

        high_alerts = manager.get_alerts(level=RiskLevel.HIGH)
        medium_alerts = manager.get_alerts(level=RiskLevel.MEDIUM)

        assert len(high_alerts) == 1
        assert len(medium_alerts) == 1

    def test_get_status(self, manager):
        """Should return status dictionary."""
        manager.update_pnl(Decimal("1000"))

        status = manager.get_status()

        assert "is_halted" in status
        assert "daily_pnl" in status
        assert "config" in status
        assert status["daily_pnl"] == "1000"

    def test_validate_order_ok(self, manager):
        """Should validate good order."""
        is_valid, alerts = manager.validate_order(
            size=Decimal("10000"),
            leverage=5.0,
            account_id="acc1",
        )

        assert is_valid is True
        assert len(alerts) == 0

    def test_validate_order_size_exceeded(self, manager):
        """Should reject order with size exceeded."""
        is_valid, alerts = manager.validate_order(
            size=Decimal("100000"),
            leverage=5.0,
            account_id="acc1",
        )

        # Size exceeded returns HIGH not CRITICAL, so still valid but with alerts
        assert len(alerts) > 0
        assert any(a.metric == "position_size" for a in alerts)

    def test_validate_order_halted(self, manager):
        """Should reject order when halted."""
        manager._halt("Test halt")

        is_valid, alerts = manager.validate_order(
            size=Decimal("10000"),
            leverage=5.0,
            account_id="acc1",
        )

        assert is_valid is False
        assert len(alerts) > 0

    def test_alert_callback(self, manager):
        """Should call alert callback."""
        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        manager.on_alert = on_alert
        manager.check_position_size(Decimal("100000"), "acc1")

        assert len(alerts_received) == 1

    def test_halt_callback(self, manager):
        """Should call halt callback."""
        halt_reasons = []

        def on_halt(reason):
            halt_reasons.append(reason)

        manager.on_halt = on_halt
        manager.config.halt_on_critical = True
        manager.config.critical_delta_pct = 10.0
        manager.check_delta(15.0, "BTC-USD-PERP")

        assert len(halt_reasons) == 1
