"""Unit tests for Risk Manager."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.core.risk_manager import (
    RiskAction,
    RiskLevel,
    RiskType,
    RiskLimit,
    RiskViolation,
    RiskMetrics,
    RiskConfig,
    RiskManager,
    DeltaNeutralRiskManager,
    EmergencyRiskManager,
    get_risk_manager,
    reset_risk_manager,
)


class TestRiskAction:
    """Tests for RiskAction enum."""

    def test_action_values(self):
        """Should have expected action values."""
        assert RiskAction.NONE.value == "none"
        assert RiskAction.ALERT.value == "alert"
        assert RiskAction.REDUCE.value == "reduce"
        assert RiskAction.CLOSE.value == "close"
        assert RiskAction.HALT.value == "halt"
        assert RiskAction.EMERGENCY_CLOSE.value == "emergency_close"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_level_values(self):
        """Should have expected level values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestRiskType:
    """Tests for RiskType enum."""

    def test_type_values(self):
        """Should have expected type values."""
        assert RiskType.POSITION.value == "position"
        assert RiskType.EXPOSURE.value == "exposure"
        assert RiskType.DRAWDOWN.value == "drawdown"
        assert RiskType.DELTA.value == "delta"
        assert RiskType.VOLATILITY.value == "volatility"


class TestRiskLimit:
    """Tests for RiskLimit dataclass."""

    def test_create_limit(self):
        """Should create risk limit."""
        limit = RiskLimit(
            name="exposure_limit",
            risk_type=RiskType.EXPOSURE,
            threshold_warning=Decimal("80000"),
            threshold_critical=Decimal("100000"),
        )

        assert limit.name == "exposure_limit"
        assert limit.threshold_warning == Decimal("80000")
        assert limit.enabled is True

    def test_limit_with_actions(self):
        """Should set custom actions."""
        limit = RiskLimit(
            name="drawdown_limit",
            risk_type=RiskType.DRAWDOWN,
            threshold_warning=Decimal("5"),
            threshold_critical=Decimal("10"),
            action_warning=RiskAction.ALERT,
            action_critical=RiskAction.HALT,
        )

        assert limit.action_warning == RiskAction.ALERT
        assert limit.action_critical == RiskAction.HALT

    def test_to_dict(self):
        """Should convert to dictionary."""
        limit = RiskLimit(
            name="test_limit",
            risk_type=RiskType.EXPOSURE,
            threshold_warning=Decimal("80"),
            threshold_critical=Decimal("100"),
        )

        d = limit.to_dict()

        assert d["name"] == "test_limit"
        assert d["risk_type"] == "exposure"


class TestRiskViolation:
    """Tests for RiskViolation dataclass."""

    def test_create_violation(self):
        """Should create violation."""
        limit = RiskLimit(
            name="test",
            risk_type=RiskType.EXPOSURE,
            threshold_warning=Decimal("80"),
            threshold_critical=Decimal("100"),
        )

        violation = RiskViolation(
            limit=limit,
            current_value=Decimal("110"),
            level=RiskLevel.CRITICAL,
            action=RiskAction.REDUCE,
            message="Exposure exceeded",
        )

        assert violation.current_value == Decimal("110")
        assert violation.level == RiskLevel.CRITICAL

    def test_to_dict(self):
        """Should convert to dictionary."""
        limit = RiskLimit(
            name="test",
            risk_type=RiskType.EXPOSURE,
            threshold_warning=Decimal("80"),
            threshold_critical=Decimal("100"),
        )

        violation = RiskViolation(
            limit=limit,
            current_value=Decimal("110"),
            level=RiskLevel.CRITICAL,
            action=RiskAction.REDUCE,
        )

        d = violation.to_dict()

        assert d["limit_name"] == "test"
        assert d["level"] == "critical"


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_create_metrics(self):
        """Should create metrics."""
        metrics = RiskMetrics(
            total_exposure=Decimal("50000"),
            net_delta=Decimal("1000"),
            delta_pct=Decimal("2"),
            long_exposure=Decimal("25500"),
            short_exposure=Decimal("24500"),
        )

        assert metrics.total_exposure == Decimal("50000")
        assert metrics.delta_pct == Decimal("2")

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = RiskMetrics(
            total_exposure=Decimal("50000"),
            drawdown_pct=5.5,
        )

        d = metrics.to_dict()

        assert "total_exposure" in d
        assert d["drawdown_pct"] == 5.5


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = RiskConfig()

        assert config.max_total_exposure == Decimal("100000")
        assert config.max_delta_pct == 5.0
        assert config.max_drawdown_pct == 10.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = RiskConfig(
            max_total_exposure=Decimal("500000"),
            max_delta_pct=3.0,
            halt_trading_drawdown_pct=20.0,
        )

        assert config.max_total_exposure == Decimal("500000")
        assert config.max_delta_pct == 3.0


class TestRiskManager:
    """Tests for RiskManager."""

    @pytest.fixture
    def manager(self):
        """Create risk manager."""
        return RiskManager()

    def test_create_manager(self, manager):
        """Should create manager with default limits."""
        assert len(manager._limits) > 0

    def test_add_limit(self, manager):
        """Should add custom limit."""
        initial_count = len(manager._limits)

        manager.add_limit(RiskLimit(
            name="custom_limit",
            risk_type=RiskType.VOLATILITY,
            threshold_warning=Decimal("20"),
            threshold_critical=Decimal("30"),
        ))

        assert len(manager._limits) == initial_count + 1

    def test_remove_limit(self, manager):
        """Should remove limit by name."""
        manager.add_limit(RiskLimit(
            name="to_remove",
            risk_type=RiskType.VOLATILITY,
            threshold_warning=Decimal("20"),
            threshold_critical=Decimal("30"),
        ))

        result = manager.remove_limit("to_remove")

        assert result is True
        assert manager.get_limit("to_remove") is None

    def test_get_limit(self, manager):
        """Should get limit by name."""
        limit = manager.get_limit("total_exposure")

        assert limit is not None
        assert limit.risk_type == RiskType.EXPOSURE

    def test_check_risks_no_violations(self, manager):
        """Should return empty when no violations."""
        metrics = RiskMetrics(
            total_exposure=Decimal("10000"),
            delta_pct=Decimal("1"),
            drawdown_pct=1.0,
        )

        violations = manager.check_risks(metrics)

        assert len(violations) == 0

    def test_check_risks_exposure_warning(self, manager):
        """Should detect exposure warning."""
        metrics = RiskMetrics(
            total_exposure=Decimal("85000"),
            delta_pct=Decimal("1"),
            drawdown_pct=1.0,
        )

        violations = manager.check_risks(metrics)

        assert len(violations) >= 1
        assert any(v.limit.name == "total_exposure" for v in violations)

    def test_check_risks_exposure_critical(self, manager):
        """Should detect exposure critical."""
        metrics = RiskMetrics(
            total_exposure=Decimal("120000"),
            delta_pct=Decimal("1"),
            drawdown_pct=1.0,
        )

        violations = manager.check_risks(metrics)

        critical_violations = [v for v in violations if v.level == RiskLevel.CRITICAL]
        assert len(critical_violations) >= 1

    def test_validate_order_allowed(self, manager):
        """Should allow valid order."""
        is_valid, errors = manager.validate_order(
            order_size=Decimal("1"),
            order_notional=Decimal("5000"),
            current_exposure=Decimal("20000"),
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_order_exceeds_position_size(self, manager):
        """Should reject order exceeding position size."""
        is_valid, errors = manager.validate_order(
            order_size=Decimal("1"),
            order_notional=Decimal("15000"),
            current_exposure=Decimal("20000"),
        )

        assert is_valid is False
        assert any("size" in e.lower() for e in errors)

    def test_validate_order_when_halted(self, manager):
        """Should reject all orders when halted."""
        manager._is_halted = True

        is_valid, errors = manager.validate_order(
            order_size=Decimal("1"),
            order_notional=Decimal("100"),
            current_exposure=Decimal("0"),
        )

        assert is_valid is False
        assert any("halted" in e.lower() for e in errors)

    def test_validate_position_concentration(self, manager):
        """Should validate position concentration."""
        is_valid, errors = manager.validate_position(
            position_value=Decimal("30000"),
            total_equity=Decimal("100000"),
        )

        assert is_valid is False
        assert any("concentration" in e.lower() for e in errors)

    def test_calculate_metrics(self, manager):
        """Should calculate metrics from positions."""
        positions = [
            {"notional": 25000, "is_long": True},
            {"notional": 25000, "is_long": True},
            {"notional": 24000, "is_long": False},
            {"notional": 24000, "is_long": False},
        ]

        metrics = manager.calculate_metrics(positions, Decimal("100000"))

        assert metrics.long_exposure == Decimal("50000")
        assert metrics.short_exposure == Decimal("48000")
        assert metrics.position_count == 4

    def test_get_recommended_action_none(self, manager):
        """Should recommend no action when safe."""
        metrics = RiskMetrics(
            drawdown_pct=1.0,
            delta_pct=Decimal("1"),
            total_exposure=Decimal("10000"),
            concentration_pct=5.0,
        )

        action, message = manager.get_recommended_action(metrics)

        assert action == RiskAction.NONE

    def test_get_recommended_action_halt(self, manager):
        """Should recommend halt on critical drawdown."""
        metrics = RiskMetrics(
            drawdown_pct=20.0,
        )

        action, message = manager.get_recommended_action(metrics)

        assert action == RiskAction.HALT

    def test_calculate_position_reduction(self, manager):
        """Should calculate reduction amount."""
        reduction = manager.calculate_position_reduction(
            current_size=Decimal("10000"),
            reduction_pct=25.0,
        )

        assert reduction == Decimal("2500")

    def test_resume_trading(self, manager):
        """Should resume trading after halt."""
        manager._is_halted = True

        result = manager.resume_trading()

        assert result is True
        assert manager._is_halted is False

    def test_is_trading_allowed(self, manager):
        """Should check if trading allowed."""
        assert manager.is_trading_allowed() is True

        manager._is_halted = True
        assert manager.is_trading_allowed() is False

    def test_get_status(self, manager):
        """Should get manager status."""
        status = manager.get_status()

        assert "is_halted" in status
        assert "active_limits" in status


class TestDeltaNeutralRiskManager:
    """Tests for DeltaNeutralRiskManager."""

    @pytest.fixture
    def manager(self):
        """Create delta-neutral risk manager."""
        return DeltaNeutralRiskManager()

    def test_check_delta_balance_balanced(self, manager):
        """Should detect balanced positions."""
        is_balanced, delta_pct, message = manager.check_delta_balance(
            long_exposure=Decimal("50000"),
            short_exposure=Decimal("50000"),
        )

        assert is_balanced is True
        assert delta_pct == Decimal("0")

    def test_check_delta_balance_unbalanced(self, manager):
        """Should detect unbalanced positions."""
        manager.rebalance_threshold = Decimal("5")

        is_balanced, delta_pct, message = manager.check_delta_balance(
            long_exposure=Decimal("60000"),
            short_exposure=Decimal("40000"),
        )

        assert is_balanced is False
        assert delta_pct == Decimal("20")

    def test_calculate_rebalance_need_long(self, manager):
        """Should calculate need for more long."""
        side, amount = manager.calculate_rebalance_needed(
            long_exposure=Decimal("40000"),
            short_exposure=Decimal("50000"),
        )

        assert side == "long"
        assert amount == Decimal("10000")

    def test_calculate_rebalance_need_short(self, manager):
        """Should calculate need for more short."""
        side, amount = manager.calculate_rebalance_needed(
            long_exposure=Decimal("60000"),
            short_exposure=Decimal("50000"),
        )

        assert side == "short"
        assert amount == Decimal("10000")


class TestEmergencyRiskManager:
    """Tests for EmergencyRiskManager."""

    @pytest.fixture
    def manager(self):
        """Create emergency risk manager."""
        return EmergencyRiskManager()

    def test_trigger_emergency(self, manager):
        """Should trigger emergency mode."""
        manager.trigger_emergency("Test emergency")

        assert manager.is_emergency() is True

    def test_emergency_callback(self, manager):
        """Should call close all callback."""
        callback = MagicMock()
        manager.set_close_all_callback(callback)

        manager.trigger_emergency("Market crash")

        callback.assert_called_once()

    def test_clear_emergency(self, manager):
        """Should clear emergency mode."""
        manager._emergency_mode = True

        manager.clear_emergency()

        assert manager.is_emergency() is False


class TestGlobalRiskManager:
    """Tests for global risk manager functions."""

    def test_get_risk_manager(self):
        """Should get or create manager."""
        reset_risk_manager()

        m1 = get_risk_manager()
        m2 = get_risk_manager()

        assert m1 is m2

    def test_reset_risk_manager(self):
        """Should reset manager."""
        m1 = get_risk_manager()
        reset_risk_manager()
        m2 = get_risk_manager()

        assert m1 is not m2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_exposure(self):
        """Should handle zero exposure."""
        manager = RiskManager()
        positions = []

        metrics = manager.calculate_metrics(positions, Decimal("100000"))

        assert metrics.total_exposure == Decimal("0")

    def test_zero_equity(self):
        """Should handle zero equity."""
        manager = RiskManager()

        is_valid, errors = manager.validate_position(
            position_value=Decimal("1000"),
            total_equity=Decimal("0"),
        )

        assert is_valid is True

    def test_disabled_limit(self):
        """Should skip disabled limits."""
        manager = RiskManager()

        for limit in manager._limits:
            limit.enabled = False

        metrics = RiskMetrics(
            total_exposure=Decimal("999999"),
            delta_pct=Decimal("99"),
            drawdown_pct=99.0,
        )

        violations = manager.check_risks(metrics)

        assert len(violations) == 0
