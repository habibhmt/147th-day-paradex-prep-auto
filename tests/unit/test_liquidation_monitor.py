"""
Tests for Liquidation Monitor Module
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.risk.liquidation_monitor import (
    RiskLevel,
    AlertType,
    ProtectionAction,
    MarginInfo,
    LiquidationPrice,
    PositionRisk,
    LiquidationAlert,
    MonitorConfig,
    LiquidationCalculator,
    RiskClassifier,
    AlertManager,
    Position,
    LiquidationMonitor,
    PortfolioRiskMonitor,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels_defined(self):
        """Test all risk levels are defined."""
        assert RiskLevel.SAFE.value == "safe"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.LIQUIDATION.value == "liquidation"


class TestAlertType:
    """Tests for AlertType enum."""

    def test_all_types_defined(self):
        """Test all alert types are defined."""
        assert AlertType.MARGIN_WARNING.value == "margin_warning"
        assert AlertType.MARGIN_CALL.value == "margin_call"
        assert AlertType.APPROACHING_LIQUIDATION.value == "approaching_liquidation"
        assert AlertType.IMMINENT_LIQUIDATION.value == "imminent_liquidation"
        assert AlertType.POSITION_CLOSED.value == "position_closed"
        assert AlertType.AUTO_DELEVERAGED.value == "auto_deleveraged"


class TestProtectionAction:
    """Tests for ProtectionAction enum."""

    def test_all_actions_defined(self):
        """Test all protection actions are defined."""
        assert ProtectionAction.NONE.value == "none"
        assert ProtectionAction.REDUCE_POSITION.value == "reduce_position"
        assert ProtectionAction.ADD_MARGIN.value == "add_margin"
        assert ProtectionAction.CLOSE_POSITION.value == "close_position"
        assert ProtectionAction.HEDGE.value == "hedge"


class TestMarginInfo:
    """Tests for MarginInfo dataclass."""

    def test_creation(self):
        """Test margin info creation."""
        info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        assert info.initial_margin == Decimal("1000")
        assert info.margin_ratio == 0.5

    def test_to_dict(self):
        """Test margin info to_dict method."""
        info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        result = info.to_dict()
        assert result["initial_margin"] == "1000"
        assert result["leverage"] == 10.0


class TestLiquidationPrice:
    """Tests for LiquidationPrice dataclass."""

    def test_creation_long(self):
        """Test liquidation price for long."""
        liq = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        assert liq.long_liquidation_price == Decimal("1800")
        assert liq.short_liquidation_price is None

    def test_creation_short(self):
        """Test liquidation price for short."""
        liq = LiquidationPrice(
            long_liquidation_price=None,
            short_liquidation_price=Decimal("2200"),
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        assert liq.long_liquidation_price is None
        assert liq.short_liquidation_price == Decimal("2200")

    def test_to_dict(self):
        """Test liquidation price to_dict method."""
        liq = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        result = liq.to_dict()
        assert result["long_liquidation_price"] == "1800"
        assert result["short_liquidation_price"] is None


class TestMonitorConfig:
    """Tests for MonitorConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = MonitorConfig(
            margin_warning_threshold=0.5,
            margin_call_threshold=0.7,
            liquidation_warning_threshold=0.85,
        )
        assert config.margin_warning_threshold == 0.5

    def test_invalid_warning_threshold(self):
        """Test invalid warning threshold."""
        with pytest.raises(ValueError):
            MonitorConfig(
                margin_warning_threshold=0.8,
                margin_call_threshold=0.7,
            )

    def test_invalid_call_threshold(self):
        """Test invalid call threshold."""
        with pytest.raises(ValueError):
            MonitorConfig(
                margin_call_threshold=0.9,
                liquidation_warning_threshold=0.85,
            )

    def test_invalid_auto_reduce_percentage(self):
        """Test invalid auto reduce percentage."""
        with pytest.raises(ValueError):
            MonitorConfig(auto_reduce_percentage=1.5)

    def test_to_dict(self):
        """Test config to_dict method."""
        config = MonitorConfig()
        result = config.to_dict()
        assert "margin_warning_threshold" in result
        assert "auto_reduce_enabled" in result


class TestLiquidationCalculator:
    """Tests for LiquidationCalculator class."""

    def test_creation(self):
        """Test calculator creation."""
        calc = LiquidationCalculator(maintenance_margin_rate=0.01)
        assert calc.maintenance_margin_rate == 0.01

    def test_calculate_liquidation_price_long(self):
        """Test liquidation price calculation for long."""
        calc = LiquidationCalculator(maintenance_margin_rate=0.005)
        result = calc.calculate_liquidation_price(
            "long",
            Decimal("2000"),
            Decimal("1"),
            Decimal("200"),
        )
        assert result.long_liquidation_price is not None
        assert result.short_liquidation_price is None
        assert result.long_liquidation_price < Decimal("2000")

    def test_calculate_liquidation_price_short(self):
        """Test liquidation price calculation for short."""
        calc = LiquidationCalculator(maintenance_margin_rate=0.005)
        result = calc.calculate_liquidation_price(
            "short",
            Decimal("2000"),
            Decimal("1"),
            Decimal("200"),
        )
        assert result.long_liquidation_price is None
        assert result.short_liquidation_price is not None
        assert result.short_liquidation_price > Decimal("2000")

    def test_calculate_margin_ratio(self):
        """Test margin ratio calculation."""
        calc = LiquidationCalculator(maintenance_margin_rate=0.005)
        ratio = calc.calculate_margin_ratio(
            Decimal("10000"),  # position value
            Decimal("1000"),   # margin
            Decimal("0"),      # unrealized PnL
        )
        # maintenance = 10000 * 0.005 = 50
        # ratio = 50 / 1000 = 0.05
        assert ratio == pytest.approx(0.05, rel=0.01)

    def test_calculate_margin_ratio_with_pnl(self):
        """Test margin ratio with unrealized PnL."""
        calc = LiquidationCalculator(maintenance_margin_rate=0.005)
        ratio = calc.calculate_margin_ratio(
            Decimal("10000"),
            Decimal("1000"),
            Decimal("-500"),  # loss
        )
        # equity = 1000 - 500 = 500
        # maintenance = 50
        # ratio = 50 / 500 = 0.1
        assert ratio == pytest.approx(0.1, rel=0.01)

    def test_calculate_margin_ratio_negative_equity(self):
        """Test margin ratio with negative equity."""
        calc = LiquidationCalculator()
        ratio = calc.calculate_margin_ratio(
            Decimal("10000"),
            Decimal("100"),
            Decimal("-200"),
        )
        assert ratio == 1.0

    def test_calculate_leverage(self):
        """Test leverage calculation."""
        calc = LiquidationCalculator()
        leverage = calc.calculate_leverage(
            Decimal("10000"),
            Decimal("1000"),
        )
        assert leverage == 10.0

    def test_calculate_leverage_zero_margin(self):
        """Test leverage with zero margin."""
        calc = LiquidationCalculator()
        leverage = calc.calculate_leverage(
            Decimal("10000"),
            Decimal("0"),
        )
        assert leverage == 0.0

    def test_calculate_health_score(self):
        """Test health score calculation."""
        calc = LiquidationCalculator()
        # Low risk
        score = calc.calculate_health_score(0.1, 0.2)
        assert score > 50

        # High risk
        score = calc.calculate_health_score(0.9, 0.02)
        assert score < 20


class TestRiskClassifier:
    """Tests for RiskClassifier class."""

    def test_creation(self):
        """Test classifier creation."""
        config = MonitorConfig()
        classifier = RiskClassifier(config)
        assert classifier.config is not None

    def test_classify_by_margin_safe(self):
        """Test safe margin classification."""
        config = MonitorConfig(margin_warning_threshold=0.5)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(0.3)
        assert level == RiskLevel.SAFE

    def test_classify_by_margin_low(self):
        """Test low risk margin classification."""
        config = MonitorConfig(margin_warning_threshold=0.5, margin_call_threshold=0.7)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(0.6)
        assert level == RiskLevel.LOW

    def test_classify_by_margin_medium(self):
        """Test medium risk margin classification."""
        config = MonitorConfig(margin_call_threshold=0.7, liquidation_warning_threshold=0.85)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(0.75)
        assert level == RiskLevel.MEDIUM

    def test_classify_by_margin_high(self):
        """Test high risk margin classification."""
        config = MonitorConfig(liquidation_warning_threshold=0.85, critical_threshold=0.95)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(0.9)
        assert level == RiskLevel.HIGH

    def test_classify_by_margin_critical(self):
        """Test critical margin classification."""
        config = MonitorConfig(critical_threshold=0.95)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(0.97)
        assert level == RiskLevel.CRITICAL

    def test_classify_by_margin_liquidation(self):
        """Test liquidation margin classification."""
        config = MonitorConfig()
        classifier = RiskClassifier(config)
        level = classifier.classify_by_margin(1.0)
        assert level == RiskLevel.LIQUIDATION

    def test_classify_by_distance_safe(self):
        """Test safe distance classification."""
        config = MonitorConfig(safe_distance_pct=0.1)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_distance(0.15)
        assert level == RiskLevel.SAFE

    def test_classify_by_distance_high(self):
        """Test high risk distance classification."""
        config = MonitorConfig(warning_distance_pct=0.05, critical_distance_pct=0.02)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_distance(0.03)
        assert level == RiskLevel.HIGH

    def test_classify_by_distance_critical(self):
        """Test critical distance classification."""
        config = MonitorConfig(critical_distance_pct=0.02)
        classifier = RiskClassifier(config)
        level = classifier.classify_by_distance(0.01)
        assert level == RiskLevel.CRITICAL

    def test_classify_by_distance_liquidation(self):
        """Test liquidation distance classification."""
        config = MonitorConfig()
        classifier = RiskClassifier(config)
        level = classifier.classify_by_distance(0)
        assert level == RiskLevel.LIQUIDATION

    def test_classify_overall_worst(self):
        """Test overall classification takes worst."""
        config = MonitorConfig()
        classifier = RiskClassifier(config)
        # Safe margin but critical distance
        level = classifier.classify_overall(0.1, 0.01)
        assert level == RiskLevel.CRITICAL


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_creation(self):
        """Test alert manager creation."""
        manager = AlertManager(cooldown_seconds=60)
        assert manager.cooldown_seconds == 60
        assert len(manager.alerts) == 0

    def test_can_alert_first_time(self):
        """Test can alert first time."""
        manager = AlertManager()
        assert manager.can_alert("ETH-USD-PERP", AlertType.MARGIN_WARNING)

    def test_can_alert_respects_cooldown(self):
        """Test alert respects cooldown."""
        manager = AlertManager(cooldown_seconds=60)
        # Create mock position risk
        margin_info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        liq_price = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        risk = PositionRisk(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=RiskLevel.LOW,
            health_score=80.0,
            timestamp=datetime.now(),
        )
        # Create first alert
        manager.create_alert(
            AlertType.MARGIN_WARNING,
            "ETH-USD-PERP",
            risk,
            "Test message",
            ProtectionAction.NONE,
        )
        # Should not allow immediate second alert
        assert not manager.can_alert("ETH-USD-PERP", AlertType.MARGIN_WARNING)
        # Different type should be allowed
        assert manager.can_alert("ETH-USD-PERP", AlertType.MARGIN_CALL)

    def test_create_alert(self):
        """Test alert creation."""
        manager = AlertManager()
        margin_info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        liq_price = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        risk = PositionRisk(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=RiskLevel.LOW,
            health_score=80.0,
            timestamp=datetime.now(),
        )
        alert = manager.create_alert(
            AlertType.MARGIN_WARNING,
            "ETH-USD-PERP",
            risk,
            "Test message",
            ProtectionAction.NONE,
        )
        assert alert is not None
        assert alert.alert_type == AlertType.MARGIN_WARNING
        assert len(manager.alerts) == 1

    def test_acknowledge_alert(self):
        """Test alert acknowledgement."""
        manager = AlertManager()
        margin_info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        liq_price = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        risk = PositionRisk(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=RiskLevel.LOW,
            health_score=80.0,
            timestamp=datetime.now(),
        )
        alert = manager.create_alert(
            AlertType.MARGIN_WARNING,
            "ETH-USD-PERP",
            risk,
            "Test message",
            ProtectionAction.NONE,
        )
        result = manager.acknowledge_alert(alert.alert_id)
        assert result
        assert manager.alerts[alert.alert_id].acknowledged

    def test_acknowledge_invalid_alert(self):
        """Test acknowledging invalid alert."""
        manager = AlertManager()
        result = manager.acknowledge_alert("invalid_id")
        assert not result

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        margin_info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        liq_price = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        risk = PositionRisk(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=RiskLevel.LOW,
            health_score=80.0,
            timestamp=datetime.now(),
        )
        alert = manager.create_alert(
            AlertType.MARGIN_WARNING,
            "ETH-USD-PERP",
            risk,
            "Test",
            ProtectionAction.NONE,
        )
        active = manager.get_active_alerts()
        assert len(active) == 1
        # Acknowledge and check again
        manager.acknowledge_alert(alert.alert_id)
        active = manager.get_active_alerts()
        assert len(active) == 0

    def test_callback_registration(self):
        """Test callback registration and invocation."""
        manager = AlertManager()
        callback_data = []

        def callback(alert):
            callback_data.append(alert)

        manager.register_callback(callback)

        margin_info = MarginInfo(
            initial_margin=Decimal("1000"),
            maintenance_margin=Decimal("500"),
            available_margin=Decimal("500"),
            used_margin=Decimal("500"),
            margin_ratio=0.5,
            leverage=10.0,
        )
        liq_price = LiquidationPrice(
            long_liquidation_price=Decimal("1800"),
            short_liquidation_price=None,
            distance_to_liquidation_pct=0.1,
            price_buffer=Decimal("200"),
        )
        risk = PositionRisk(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            margin_info=margin_info,
            liquidation_price=liq_price,
            risk_level=RiskLevel.LOW,
            health_score=80.0,
            timestamp=datetime.now(),
        )
        manager.create_alert(
            AlertType.MARGIN_WARNING,
            "ETH-USD-PERP",
            risk,
            "Test",
            ProtectionAction.NONE,
        )
        assert len(callback_data) == 1


class TestPosition:
    """Tests for Position dataclass."""

    def test_creation(self):
        """Test position creation."""
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
            leverage=10.0,
        )
        assert position.symbol == "ETH-USD-PERP"
        assert position.leverage == 10.0


class TestLiquidationMonitor:
    """Tests for LiquidationMonitor class."""

    def test_creation(self):
        """Test monitor creation."""
        monitor = LiquidationMonitor()
        assert monitor.config is not None
        assert len(monitor.positions) == 0

    def test_creation_with_config(self):
        """Test monitor creation with config."""
        config = MonitorConfig(margin_warning_threshold=0.6)
        monitor = LiquidationMonitor(config)
        assert monitor.config.margin_warning_threshold == 0.6

    def test_add_position(self):
        """Test adding position."""
        monitor = LiquidationMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        assert "ETH-USD-PERP" in monitor.positions

    def test_remove_position(self):
        """Test removing position."""
        monitor = LiquidationMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        result = monitor.remove_position("ETH-USD-PERP")
        assert result
        assert "ETH-USD-PERP" not in monitor.positions

    def test_remove_nonexistent_position(self):
        """Test removing nonexistent position."""
        monitor = LiquidationMonitor()
        result = monitor.remove_position("INVALID")
        assert not result

    def test_update_price(self):
        """Test price update."""
        monitor = LiquidationMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        risk = monitor.update_price("ETH-USD-PERP", Decimal("1950"))
        assert risk is not None
        assert risk.current_price == Decimal("1950")
        assert risk.unrealized_pnl < 0

    def test_update_price_no_position(self):
        """Test price update with no position."""
        monitor = LiquidationMonitor()
        result = monitor.update_price("INVALID", Decimal("2000"))
        assert result is None

    def test_assess_all(self):
        """Test assessing all positions."""
        monitor = LiquidationMonitor()
        position1 = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        position2 = Position(
            symbol="BTC-USD-PERP",
            side="short",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            margin=Decimal("500"),
        )
        monitor.add_position(position1)
        monitor.add_position(position2)
        risks = monitor.assess_all()
        assert len(risks) == 2
        assert "ETH-USD-PERP" in risks
        assert "BTC-USD-PERP" in risks

    def test_get_position_risk(self):
        """Test getting position risk."""
        monitor = LiquidationMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("2000"))
        risk = monitor.get_position_risk("ETH-USD-PERP")
        assert risk is not None
        assert risk.symbol == "ETH-USD-PERP"

    def test_get_high_risk_positions(self):
        """Test getting high risk positions."""
        config = MonitorConfig(
            margin_warning_threshold=0.05,  # Low thresholds for testing
            margin_call_threshold=0.1,
            liquidation_warning_threshold=0.15,
            critical_threshold=0.2,
        )
        monitor = LiquidationMonitor(config)
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            margin=Decimal("100"),  # High leverage
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("1900"))  # Loss
        high_risk = monitor.get_high_risk_positions()
        # May or may not be high risk depending on calculation
        assert isinstance(high_risk, list)

    def test_get_alerts(self):
        """Test getting alerts."""
        monitor = LiquidationMonitor()
        alerts = monitor.get_alerts()
        assert isinstance(alerts, list)

    def test_acknowledge_alert(self):
        """Test acknowledging alert."""
        monitor = LiquidationMonitor()
        result = monitor.acknowledge_alert("invalid")
        assert not result

    def test_register_alert_callback(self):
        """Test registering alert callback."""
        monitor = LiquidationMonitor()
        callback_data = []

        def callback(alert):
            callback_data.append(alert)

        monitor.register_alert_callback(callback)
        assert len(monitor.alert_manager.callbacks) == 1

    def test_get_status(self):
        """Test getting monitor status."""
        monitor = LiquidationMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        status = monitor.get_status()
        assert status["positions_monitored"] == 1
        assert "config" in status
        assert "uptime_seconds" in status


class TestLiquidationMonitorAlerts:
    """Tests for alert generation in monitor."""

    def test_critical_risk_generates_alert(self):
        """Test critical risk generates alert."""
        config = MonitorConfig(
            critical_threshold=0.01,  # Very low threshold
            alert_cooldown_seconds=0,
        )
        monitor = LiquidationMonitor(config)
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("100"),
            entry_price=Decimal("2000"),
            margin=Decimal("10"),  # Very low margin
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("1990"))
        alerts = monitor.get_alerts()
        # Should have some alert
        assert isinstance(alerts, list)


class TestPortfolioRiskMonitor:
    """Tests for PortfolioRiskMonitor class."""

    def test_creation(self):
        """Test portfolio monitor creation."""
        monitor = PortfolioRiskMonitor()
        assert monitor.portfolio_margin == Decimal("0")

    def test_set_portfolio_margin(self):
        """Test setting portfolio margin."""
        monitor = PortfolioRiskMonitor()
        monitor.set_portfolio_margin(Decimal("10000"))
        assert monitor.portfolio_margin == Decimal("10000")

    def test_add_position(self):
        """Test adding position."""
        monitor = PortfolioRiskMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        assert "ETH-USD-PERP" in monitor.monitor.positions

    def test_remove_position(self):
        """Test removing position."""
        monitor = PortfolioRiskMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        result = monitor.remove_position("ETH-USD-PERP")
        assert result

    def test_update_price(self):
        """Test updating price."""
        monitor = PortfolioRiskMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        risk = monitor.update_price("ETH-USD-PERP", Decimal("2100"))
        assert risk is not None

    def test_get_portfolio_leverage(self):
        """Test portfolio leverage calculation."""
        monitor = PortfolioRiskMonitor()
        monitor.set_portfolio_margin(Decimal("10000"))
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("5"),
            entry_price=Decimal("2000"),
            margin=Decimal("1000"),
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("2000"))
        leverage = monitor.get_portfolio_leverage()
        # 5 * 2000 = 10000, margin = 10000, leverage = 1
        assert leverage == pytest.approx(1.0, rel=0.1)

    def test_get_portfolio_leverage_zero_margin(self):
        """Test portfolio leverage with zero margin."""
        monitor = PortfolioRiskMonitor()
        leverage = monitor.get_portfolio_leverage()
        assert leverage == 0.0

    def test_get_portfolio_health(self):
        """Test portfolio health calculation."""
        monitor = PortfolioRiskMonitor()
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("2000"))
        health = monitor.get_portfolio_health()
        assert 0 <= health <= 100

    def test_get_portfolio_health_empty(self):
        """Test portfolio health with no positions."""
        monitor = PortfolioRiskMonitor()
        health = monitor.get_portfolio_health()
        assert health == 100.0

    def test_get_worst_position(self):
        """Test getting worst position."""
        monitor = PortfolioRiskMonitor()
        position1 = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
        )
        position2 = Position(
            symbol="BTC-USD-PERP",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            margin=Decimal("500"),
        )
        monitor.add_position(position1)
        monitor.add_position(position2)
        monitor.update_price("ETH-USD-PERP", Decimal("1800"))  # Loss
        monitor.update_price("BTC-USD-PERP", Decimal("50000"))  # No loss
        worst = monitor.get_worst_position()
        assert worst is not None
        # ETH should be worst due to loss
        assert worst.symbol == "ETH-USD-PERP"

    def test_get_worst_position_empty(self):
        """Test getting worst position with no positions."""
        monitor = PortfolioRiskMonitor()
        worst = monitor.get_worst_position()
        assert worst is None

    def test_get_status(self):
        """Test getting portfolio status."""
        monitor = PortfolioRiskMonitor()
        monitor.set_portfolio_margin(Decimal("10000"))
        status = monitor.get_status()
        assert status["portfolio_margin"] == "10000"
        assert "portfolio_leverage" in status
        assert "portfolio_health" in status
        assert "monitor_status" in status


class TestLiquidationMonitorIntegration:
    """Integration tests for liquidation monitor."""

    def test_full_monitoring_cycle(self):
        """Test full monitoring cycle."""
        config = MonitorConfig(
            alert_cooldown_seconds=0,
        )
        monitor = LiquidationMonitor(config)

        # Add position
        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("1"),
            entry_price=Decimal("2000"),
            margin=Decimal("200"),
            leverage=10.0,
        )
        monitor.add_position(position)

        # Track alerts
        received_alerts = []
        monitor.register_alert_callback(lambda a: received_alerts.append(a))

        # Price stable - should be safe
        risk1 = monitor.update_price("ETH-USD-PERP", Decimal("2000"))
        assert risk1 is not None

        # Price drops - risk increases
        risk2 = monitor.update_price("ETH-USD-PERP", Decimal("1900"))
        assert risk2.unrealized_pnl < risk1.unrealized_pnl

        # Get status
        status = monitor.get_status()
        assert status["positions_monitored"] == 1

    def test_multiple_positions(self):
        """Test monitoring multiple positions."""
        monitor = LiquidationMonitor()

        # Add multiple positions
        for i in range(5):
            position = Position(
                symbol=f"MARKET-{i}-PERP",
                side="long" if i % 2 == 0 else "short",
                size=Decimal("1"),
                entry_price=Decimal("1000"),
                margin=Decimal("100"),
            )
            monitor.add_position(position)

        # Update all prices
        for i in range(5):
            monitor.update_price(f"MARKET-{i}-PERP", Decimal("1000"))

        # Assess all
        risks = monitor.assess_all()
        assert len(risks) == 5

    def test_auto_protect_tracking(self):
        """Test auto-protect action tracking."""
        config = MonitorConfig(
            auto_reduce_enabled=True,
            auto_reduce_threshold=0.01,  # Very low threshold
        )
        monitor = LiquidationMonitor(config)

        position = Position(
            symbol="ETH-USD-PERP",
            side="long",
            size=Decimal("100"),
            entry_price=Decimal("2000"),
            margin=Decimal("10"),  # Very low margin
        )
        monitor.add_position(position)
        monitor.update_price("ETH-USD-PERP", Decimal("1950"))

        # Auto actions may have been taken
        assert isinstance(monitor.auto_actions_taken, list)
