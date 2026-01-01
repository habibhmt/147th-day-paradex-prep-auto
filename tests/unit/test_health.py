"""Unit tests for Health Monitoring."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.monitoring.health import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    ComponentHealth,
    create_db_health_check,
    create_rate_limit_health_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_statuses(self):
        """Should have expected health statuses."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_create_health_check(self):
        """Should create health check correctly."""
        check = HealthCheck(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=50.0,
        )

        assert check.name == "test"
        assert check.status == HealthStatus.HEALTHY
        assert check.latency_ms == 50.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        check = HealthCheck(
            name="api",
            status=HealthStatus.DEGRADED,
            message="Slow response",
            latency_ms=2500.0,
            details={"endpoint": "/health"},
        )

        d = check.to_dict()

        assert d["name"] == "api"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 2500.0
        assert d["details"]["endpoint"] == "/health"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_component(self):
        """Should create component correctly."""
        def check_fn():
            return HealthCheck("test", HealthStatus.HEALTHY)

        component = ComponentHealth(
            name="test",
            check_fn=check_fn,
            interval=60.0,
        )

        assert component.name == "test"
        assert component.interval == 60.0
        assert component.consecutive_failures == 0


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create health monitor."""
        return HealthMonitor(check_interval=1.0)

    def test_initial_state(self, monitor):
        """Should start with empty state."""
        status = monitor.get_status()

        assert status["is_running"] is False
        assert status["component_count"] == 0

    def test_register_component(self, monitor):
        """Should register component."""
        def check_fn():
            return HealthCheck("test", HealthStatus.HEALTHY)

        monitor.register_component("test", check_fn)

        assert "test" in monitor._components
        assert monitor._components["test"].name == "test"

    def test_register_component_with_interval(self, monitor):
        """Should use custom interval."""
        def check_fn():
            return HealthCheck("test", HealthStatus.HEALTHY)

        monitor.register_component("test", check_fn, interval=120.0)

        assert monitor._components["test"].interval == 120.0

    def test_unregister_component(self, monitor):
        """Should unregister component."""
        def check_fn():
            return HealthCheck("test", HealthStatus.HEALTHY)

        monitor.register_component("test", check_fn)
        result = monitor.unregister_component("test")

        assert result is True
        assert "test" not in monitor._components

    def test_unregister_nonexistent(self, monitor):
        """Should return False for nonexistent component."""
        result = monitor.unregister_component("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_component(self, monitor):
        """Should check component health."""
        def check_fn():
            return HealthCheck("test", HealthStatus.HEALTHY, "OK")

        monitor.register_component("test", check_fn)
        result = await monitor.check_component("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_check_component_async(self, monitor):
        """Should handle async check functions."""
        async def check_fn():
            await asyncio.sleep(0.01)
            return HealthCheck("test", HealthStatus.HEALTHY)

        monitor.register_component("test", check_fn)
        result = await monitor.check_component("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_component_error(self, monitor):
        """Should handle check errors."""
        def check_fn():
            raise Exception("Test error")

        monitor.register_component("test", check_fn)
        result = await monitor.check_component("test")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Test error" in result.message

    @pytest.mark.asyncio
    async def test_check_nonexistent(self, monitor):
        """Should return None for nonexistent component."""
        result = await monitor.check_component("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_all(self, monitor):
        """Should check all components."""
        monitor.register_component(
            "comp1",
            lambda: HealthCheck("comp1", HealthStatus.HEALTHY),
        )
        monitor.register_component(
            "comp2",
            lambda: HealthCheck("comp2", HealthStatus.DEGRADED),
        )

        results = await monitor.check_all()

        assert len(results) == 2
        assert "comp1" in results
        assert "comp2" in results

    def test_get_overall_status_empty(self, monitor):
        """Should return unknown for empty monitor."""
        status = monitor.get_overall_status()
        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_overall_status_healthy(self, monitor):
        """Should return healthy when all healthy."""
        monitor.register_component(
            "comp1",
            lambda: HealthCheck("comp1", HealthStatus.HEALTHY),
        )
        monitor.register_component(
            "comp2",
            lambda: HealthCheck("comp2", HealthStatus.HEALTHY),
        )
        await monitor.check_all()

        status = monitor.get_overall_status()
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_overall_status_degraded(self, monitor):
        """Should return degraded when any degraded."""
        monitor.register_component(
            "comp1",
            lambda: HealthCheck("comp1", HealthStatus.HEALTHY),
        )
        monitor.register_component(
            "comp2",
            lambda: HealthCheck("comp2", HealthStatus.DEGRADED),
        )
        await monitor.check_all()

        status = monitor.get_overall_status()
        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_get_overall_status_unhealthy(self, monitor):
        """Should return unhealthy when any unhealthy."""
        monitor.register_component(
            "comp1",
            lambda: HealthCheck("comp1", HealthStatus.HEALTHY),
        )
        monitor.register_component(
            "comp2",
            lambda: HealthCheck("comp2", HealthStatus.UNHEALTHY),
        )
        await monitor.check_all()

        status = monitor.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Should start and stop monitoring."""
        monitor.register_component(
            "test",
            lambda: HealthCheck("test", HealthStatus.HEALTHY),
        )

        await monitor.start()
        assert monitor._running is True

        await asyncio.sleep(0.1)

        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_status_change_callback(self, monitor):
        """Should call callback on status change."""
        changes = []

        def on_change(name, status):
            changes.append((name, status))

        monitor.on_status_change = on_change

        call_count = 0
        def check_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return HealthCheck("test", HealthStatus.HEALTHY)
            return HealthCheck("test", HealthStatus.DEGRADED)

        monitor.register_component("test", check_fn)

        # First check
        await monitor.check_component("test")
        # Second check - status changes
        await monitor.check_component("test")

        assert len(changes) == 1
        assert changes[0] == ("test", HealthStatus.DEGRADED)

    @pytest.mark.asyncio
    async def test_consecutive_failures(self, monitor):
        """Should track consecutive failures."""
        def check_fn():
            return HealthCheck("test", HealthStatus.UNHEALTHY)

        monitor.register_component("test", check_fn)

        for _ in range(5):
            await monitor.check_component("test")

        assert monitor._components["test"].consecutive_failures == 5

    @pytest.mark.asyncio
    async def test_failure_reset_on_healthy(self, monitor):
        """Should reset failures when healthy."""
        call_count = 0
        def check_fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return HealthCheck("test", HealthStatus.UNHEALTHY)
            return HealthCheck("test", HealthStatus.HEALTHY)

        monitor.register_component("test", check_fn)

        for _ in range(4):
            await monitor.check_component("test")

        assert monitor._components["test"].consecutive_failures == 0

    def test_get_status(self, monitor):
        """Should return status dictionary."""
        monitor.register_component(
            "test",
            lambda: HealthCheck("test", HealthStatus.HEALTHY),
        )

        status = monitor.get_status()

        assert "overall" in status
        assert "is_running" in status
        assert "component_count" in status
        assert "components" in status


class TestHealthCheckFactories:
    """Tests for health check factory functions."""

    def test_create_db_health_check(self):
        """Should create DB health check."""
        mock_db = MagicMock()
        mock_db.get_statistics.return_value = {"total_trades": 100}

        check_fn = create_db_health_check(mock_db)
        result = check_fn()

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert "100 trades" in result.message

    def test_create_db_health_check_error(self):
        """Should handle DB errors."""
        mock_db = MagicMock()
        mock_db.get_statistics.side_effect = Exception("Connection error")

        check_fn = create_db_health_check(mock_db)
        result = check_fn()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection error" in result.message

    def test_create_rate_limit_health_check(self):
        """Should create rate limit health check."""
        mock_limiter = MagicMock()
        mock_limiter.get_current_usage.return_value = {
            "orders_per_second": {"usage_pct": 30.0},
            "requests_per_minute_ip": {"usage_pct": 20.0},
        }

        check_fn = create_rate_limit_health_check(mock_limiter)
        result = check_fn()

        assert result.name == "rate_limiter"
        assert result.status == HealthStatus.HEALTHY

    def test_create_rate_limit_health_check_high_usage(self):
        """Should detect high rate limit usage."""
        mock_limiter = MagicMock()
        mock_limiter.get_current_usage.return_value = {
            "orders_per_second": {"usage_pct": 95.0},
            "requests_per_minute_ip": {"usage_pct": 50.0},
        }

        check_fn = create_rate_limit_health_check(mock_limiter)
        result = check_fn()

        assert result.status == HealthStatus.DEGRADED
        assert "95%" in result.message
