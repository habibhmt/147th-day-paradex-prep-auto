"""Unit tests for Circuit Breaker."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from src.network.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerOpen,
    CircuitMetrics,
    CircuitState,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self):
        """Should have expected state values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.half_open_max_calls == 3

    def test_custom_config(self):
        """Should accept custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=10.0,
        )

        assert config.failure_threshold == 3
        assert config.timeout == 10.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout=30.0,
        )

        d = config.to_dict()

        assert d["failure_threshold"] == 5
        assert d["timeout"] == 30.0


class TestCircuitMetrics:
    """Tests for CircuitMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = CircuitMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.rejected_calls == 0

    def test_success_rate_no_calls(self):
        """Should return 100% with no calls."""
        metrics = CircuitMetrics()
        assert metrics.success_rate == 100.0

    def test_success_rate_calculation(self):
        """Should calculate success rate correctly."""
        metrics = CircuitMetrics(
            total_calls=100,
            successful_calls=75,
            failed_calls=25,
        )

        assert metrics.success_rate == 75.0

    def test_failure_rate_no_calls(self):
        """Should return 0% with no calls."""
        metrics = CircuitMetrics()
        assert metrics.failure_rate == 0.0

    def test_failure_rate_calculation(self):
        """Should calculate failure rate correctly."""
        metrics = CircuitMetrics(
            total_calls=100,
            successful_calls=80,
            failed_calls=20,
        )

        assert metrics.failure_rate == 20.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = CircuitMetrics(
            total_calls=50,
            successful_calls=45,
            failed_calls=5,
        )

        d = metrics.to_dict()

        assert d["total_calls"] == 50
        assert d["success_rate"] == 90.0


class TestCircuitBreakerOpen:
    """Tests for CircuitBreakerOpen exception."""

    def test_exception_message(self):
        """Should create exception with message."""
        exc = CircuitBreakerOpen("api", 10.5)

        assert "api" in str(exc)
        assert "10.5" in str(exc)

    def test_exception_attributes(self):
        """Should have correct attributes."""
        exc = CircuitBreakerOpen("test", 5.0)

        assert exc.breaker_name == "test"
        assert exc.retry_after == 5.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker with fast timeouts."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.1,  # Fast timeout for tests
            half_open_max_calls=2,
        )
        return CircuitBreaker(name="test", config=config)

    def test_initial_state(self, breaker):
        """Should start in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    def test_name(self, breaker):
        """Should have correct name."""
        assert breaker.name == "test"

    @pytest.mark.asyncio
    async def test_can_execute_when_closed(self, breaker):
        """Should allow execution when closed."""
        result = await breaker.can_execute()
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_success(self, breaker):
        """Should execute function successfully."""
        async def success_func():
            return "success"

        result = await breaker.execute(success_func)

        assert result == "success"
        assert breaker._metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, breaker):
        """Should execute sync function."""
        def sync_func():
            return 42

        result = await breaker.execute(sync_func)

        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_with_args(self, breaker):
        """Should pass args to function."""
        async def add(a, b):
            return a + b

        result = await breaker.execute(add, 2, 3)

        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self, breaker):
        """Should pass kwargs to function."""
        async def greet(name="World"):
            return f"Hello, {name}"

        result = await breaker.execute(greet, name="Test")

        assert result == "Hello, Test"

    @pytest.mark.asyncio
    async def test_failure_tracking(self, breaker):
        """Should track failures."""
        async def failing_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker._failure_count == 2
        assert breaker.state == CircuitState.CLOSED  # Not yet at threshold

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        """Should open after reaching failure threshold."""
        async def failing_func():
            raise ValueError("Test error")

        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_rejects_when_open(self, breaker):
        """Should reject calls when open."""
        async def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Should now reject
        with pytest.raises(CircuitBreakerOpen):
            await breaker.execute(failing_func)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, breaker):
        """Should transition to half-open after timeout."""
        async def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should be half-open now
        await breaker._check_timeout()
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold(self, breaker):
        """Should close after success threshold in half-open."""
        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "ok"

        # Open then half-open
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        await asyncio.sleep(0.15)

        # Execute successes (2 = success_threshold)
        await breaker.execute(success_func)
        await breaker.execute(success_func)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self, breaker):
        """Should reopen on failure in half-open."""
        async def failing_func():
            raise ValueError("Test error")

        # Open then half-open
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        await asyncio.sleep(0.15)
        await breaker._check_timeout()

        # Fail in half-open
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, breaker):
        """Should reset failure count on success."""
        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "ok"

        # Some failures
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)

        assert breaker._failure_count == 2

        # Success should reset
        await breaker.execute(success_func)

        assert breaker._failure_count == 0

    def test_reset(self, breaker):
        """Should reset circuit breaker."""
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 10

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_get_time_until_retry_when_closed(self, breaker):
        """Should return 0 when closed."""
        assert breaker.get_time_until_retry() == 0.0

    @pytest.mark.asyncio
    async def test_get_time_until_retry_when_open(self, breaker):
        """Should return remaining time when open."""
        async def failing_func():
            raise ValueError("Test error")

        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        retry_time = breaker.get_time_until_retry()

        assert retry_time > 0
        assert retry_time <= breaker.config.timeout

    def test_get_status(self, breaker):
        """Should return status dictionary."""
        status = breaker.get_status()

        assert "name" in status
        assert "state" in status
        assert "failure_count" in status
        assert "config" in status
        assert "metrics" in status

    def test_on_state_change_callback(self, breaker):
        """Should call state change callback."""
        callback = MagicMock()
        breaker.on_state_change(callback)

        # Store the callback
        assert "state_change" in breaker._callbacks

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, breaker):
        """Should track metrics correctly."""
        async def success_func():
            return "ok"

        await breaker.execute(success_func)
        await breaker.execute(success_func)

        metrics = breaker.get_metrics()

        assert metrics.total_calls == 2
        assert metrics.successful_calls == 2

    @pytest.mark.asyncio
    async def test_rejected_calls_metric(self, breaker):
        """Should track rejected calls."""
        async def failing_func():
            raise ValueError("Test error")

        # Open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Try when open
        with pytest.raises(CircuitBreakerOpen):
            await breaker.execute(failing_func)

        assert breaker._metrics.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_exclude_exceptions(self):
        """Should not count excluded exceptions."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            exclude_exceptions=[ValueError],
        )
        breaker = CircuitBreaker(name="test", config=config)

        async def value_error_func():
            raise ValueError("Excluded")

        # ValueError should not count toward failures
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(value_error_func)

        # Should still be closed
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_max_calls(self, breaker):
        """Should limit calls in half-open state."""
        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "ok"

        # Open then half-open
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        await asyncio.sleep(0.15)

        # Use up half-open calls (max = 2)
        await breaker.execute(success_func)
        await breaker.execute(success_func)  # This closes it

        # Now it's closed again
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry."""
        return CircuitBreakerRegistry()

    def test_initial_state(self, registry):
        """Should start empty."""
        assert registry.count == 0

    def test_get_or_create(self, registry):
        """Should create new breaker."""
        breaker = registry.get_or_create("api")

        assert breaker is not None
        assert breaker.name == "api"
        assert registry.count == 1

    def test_get_or_create_returns_existing(self, registry):
        """Should return existing breaker."""
        breaker1 = registry.get_or_create("api")
        breaker2 = registry.get_or_create("api")

        assert breaker1 is breaker2

    def test_get_or_create_with_config(self, registry):
        """Should use custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = registry.get_or_create("api", config=config)

        assert breaker.config.failure_threshold == 10

    def test_get(self, registry):
        """Should get existing breaker."""
        registry.get_or_create("api")

        breaker = registry.get("api")

        assert breaker is not None

    def test_get_nonexistent(self, registry):
        """Should return None for nonexistent."""
        breaker = registry.get("nonexistent")
        assert breaker is None

    def test_remove(self, registry):
        """Should remove breaker."""
        registry.get_or_create("api")

        result = registry.remove("api")

        assert result is True
        assert registry.get("api") is None

    def test_remove_nonexistent(self, registry):
        """Should return False for nonexistent."""
        result = registry.remove("nonexistent")
        assert result is False

    def test_reset_all(self, registry):
        """Should reset all breakers."""
        breaker1 = registry.get_or_create("api1")
        breaker2 = registry.get_or_create("api2")

        breaker1._state = CircuitState.OPEN
        breaker2._state = CircuitState.OPEN

        registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_get_all_status(self, registry):
        """Should get status of all breakers."""
        registry.get_or_create("api1")
        registry.get_or_create("api2")

        status = registry.get_all_status()

        assert "api1" in status
        assert "api2" in status

    def test_get_open_breakers(self, registry):
        """Should return open breakers."""
        breaker1 = registry.get_or_create("api1")
        registry.get_or_create("api2")

        breaker1._state = CircuitState.OPEN

        open_breakers = registry.get_open_breakers()

        assert "api1" in open_breakers
        assert "api2" not in open_breakers
