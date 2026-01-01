"""Unit tests for Retry mechanism."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.network.retry import (
    RetryHandler,
    RetryConfig,
    RetryResult,
    RetryMetrics,
    RetryExhausted,
    RetryPolicy,
    BackoffStrategy,
    retry,
)


class TestBackoffStrategy:
    """Tests for BackoffStrategy enum."""

    def test_strategy_values(self):
        """Should have expected strategy values."""
        assert BackoffStrategy.FIXED.value == "fixed"
        assert BackoffStrategy.LINEAR.value == "linear"
        assert BackoffStrategy.EXPONENTIAL.value == "exponential"
        assert BackoffStrategy.FIBONACCI.value == "fibonacci"


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.jitter is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            backoff_strategy=BackoffStrategy.LINEAR,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.backoff_strategy == BackoffStrategy.LINEAR

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        config = RetryConfig(max_attempts=3)

        d = config.to_dict()

        assert d["max_attempts"] == 3
        assert "backoff_strategy" in d


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_successful_result(self):
        """Should create successful result."""
        result = RetryResult(
            success=True,
            result="data",
            attempts=1,
        )

        assert result.success is True
        assert result.result == "data"

    def test_failed_result(self):
        """Should create failed result."""
        exc = ValueError("Test error")
        result = RetryResult(
            success=False,
            attempts=3,
            last_exception=exc,
            exceptions=[exc],
        )

        assert result.success is False
        assert result.last_exception == exc

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = RetryResult(
            success=True,
            attempts=2,
            total_delay=1.5,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["attempts"] == 2


class TestRetryMetrics:
    """Tests for RetryMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = RetryMetrics()

        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.total_retries == 0

    def test_success_rate_no_operations(self):
        """Should return 100% with no operations."""
        metrics = RetryMetrics()
        assert metrics.success_rate == 100.0

    def test_success_rate_calculation(self):
        """Should calculate success rate correctly."""
        metrics = RetryMetrics(
            total_operations=100,
            successful_operations=80,
            failed_operations=20,
        )

        assert metrics.success_rate == 80.0

    def test_avg_retries(self):
        """Should calculate average retries."""
        metrics = RetryMetrics(
            total_operations=10,
            total_retries=20,
        )

        assert metrics.avg_retries == 2.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = RetryMetrics(
            total_operations=50,
            successful_operations=45,
        )

        d = metrics.to_dict()

        assert d["total_operations"] == 50
        assert d["success_rate"] == 90.0


class TestRetryExhausted:
    """Tests for RetryExhausted exception."""

    def test_exception_message(self):
        """Should create exception with message."""
        exc = RetryExhausted(3, ValueError("Test"))

        assert "3" in str(exc)
        assert "Test" in str(exc)

    def test_exception_attributes(self):
        """Should have correct attributes."""
        inner = ValueError("Inner")
        exc = RetryExhausted(5, inner)

        assert exc.attempts == 5
        assert exc.last_exception == inner


class TestRetryHandler:
    """Tests for RetryHandler."""

    @pytest.fixture
    def handler(self):
        """Create retry handler with fast config."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,  # Fast for tests
            max_delay=0.1,
            jitter=False,  # Deterministic for tests
        )
        return RetryHandler(config=config)

    @pytest.fixture
    def jitter_handler(self):
        """Create retry handler with jitter."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            jitter=True,
            jitter_range=0.5,
        )
        return RetryHandler(config=config)

    def test_calculate_delay_fixed(self):
        """Should calculate fixed delay."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.FIXED,
            jitter=False,
        )
        handler = RetryHandler(config=config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 1.0
        assert handler.calculate_delay(5) == 1.0

    def test_calculate_delay_linear(self):
        """Should calculate linear delay."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=1.0,
            backoff_strategy=BackoffStrategy.LINEAR,
            jitter=False,
        )
        handler = RetryHandler(config=config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 2.0
        assert handler.calculate_delay(2) == 3.0

    def test_calculate_delay_exponential(self):
        """Should calculate exponential delay."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        handler = RetryHandler(config=config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 2.0
        assert handler.calculate_delay(2) == 4.0
        assert handler.calculate_delay(3) == 8.0

    def test_calculate_delay_fibonacci(self):
        """Should calculate Fibonacci delay."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.FIBONACCI,
            jitter=False,
        )
        handler = RetryHandler(config=config)

        assert handler.calculate_delay(0) == 1.0  # fib(0) = 1
        assert handler.calculate_delay(1) == 1.0  # fib(1) = 1
        assert handler.calculate_delay(2) == 2.0  # fib(2) = 2
        assert handler.calculate_delay(3) == 3.0  # fib(3) = 3
        assert handler.calculate_delay(4) == 5.0  # fib(4) = 5

    def test_calculate_delay_max_cap(self):
        """Should cap at max delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=10.0,
            jitter=False,
        )
        handler = RetryHandler(config=config)

        # Would be 10.0 but capped at 5.0
        assert handler.calculate_delay(1) == 5.0

    def test_calculate_delay_with_jitter(self, jitter_handler):
        """Should add jitter to delay."""
        delays = [jitter_handler.calculate_delay(1) for _ in range(10)]

        # With jitter, delays should vary
        assert len(set(delays)) > 1

    def test_should_retry_default(self, handler):
        """Should retry all exceptions by default."""
        assert handler.should_retry(ValueError("test")) is True
        assert handler.should_retry(TypeError("test")) is True

    def test_should_retry_specific_exceptions(self):
        """Should only retry specified exceptions."""
        config = RetryConfig(
            retry_exceptions=[ValueError],
        )
        handler = RetryHandler(config=config)

        assert handler.should_retry(ValueError("test")) is True
        assert handler.should_retry(TypeError("test")) is False

    def test_should_not_retry_excluded(self):
        """Should not retry excluded exceptions."""
        config = RetryConfig(
            exclude_exceptions=[KeyboardInterrupt],
        )
        handler = RetryHandler(config=config)

        assert handler.should_retry(ValueError("test")) is True
        assert handler.should_retry(KeyboardInterrupt()) is False

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self, handler):
        """Should succeed on first try."""
        async def success_func():
            return "success"

        result = await handler.execute(success_func)

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, handler):
        """Should handle sync functions."""
        def sync_func():
            return 42

        result = await handler.execute(sync_func)

        assert result.success is True
        assert result.result == 42

    @pytest.mark.asyncio
    async def test_execute_with_args(self, handler):
        """Should pass arguments."""
        async def add(a, b):
            return a + b

        result = await handler.execute(add, 2, 3)

        assert result.result == 5

    @pytest.mark.asyncio
    async def test_execute_retry_then_succeed(self, handler):
        """Should retry and eventually succeed."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = await handler.execute(flaky_func)

        assert result.success is True
        assert result.attempts == 2
        assert len(result.exceptions) == 1

    @pytest.mark.asyncio
    async def test_execute_all_retries_fail(self, handler):
        """Should fail after all retries."""
        async def always_fail():
            raise ValueError("Always fails")

        result = await handler.execute(always_fail)

        assert result.success is False
        assert result.attempts == 3
        assert len(result.exceptions) == 3

    @pytest.mark.asyncio
    async def test_execute_tracks_delay(self, handler):
        """Should track total delay."""
        call_count = 0

        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fail")
            return "ok"

        result = await handler.execute(fail_twice)

        assert result.total_delay > 0

    @pytest.mark.asyncio
    async def test_execute_or_raise_success(self, handler):
        """Should return result on success."""
        async def success_func():
            return "data"

        result = await handler.execute_or_raise(success_func)

        assert result == "data"

    @pytest.mark.asyncio
    async def test_execute_or_raise_failure(self, handler):
        """Should raise RetryExhausted on failure."""
        async def always_fail():
            raise ValueError("Fail")

        with pytest.raises(RetryExhausted) as exc_info:
            await handler.execute_or_raise(always_fail)

        assert exc_info.value.attempts == 3

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, handler):
        """Should track metrics correctly."""
        async def success_func():
            return "ok"

        await handler.execute(success_func)
        await handler.execute(success_func)

        metrics = handler.get_metrics()

        assert metrics.total_operations == 2
        assert metrics.successful_operations == 2

    @pytest.mark.asyncio
    async def test_metrics_failed_operations(self, handler):
        """Should track failed operations."""
        async def always_fail():
            raise ValueError("Fail")

        await handler.execute(always_fail)

        metrics = handler.get_metrics()

        assert metrics.failed_operations == 1
        assert metrics.total_retries == 3  # All attempts are retries

    def test_reset_metrics(self, handler):
        """Should reset metrics."""
        handler._metrics.total_operations = 100

        handler.reset_metrics()

        assert handler._metrics.total_operations == 0


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_fast_policy(self):
        """Should create fast retry policy."""
        config = RetryPolicy.fast()

        assert config.max_attempts == 3
        assert config.base_delay == 0.1
        assert config.max_delay == 1.0

    def test_standard_policy(self):
        """Should create standard retry policy."""
        config = RetryPolicy.standard()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0

    def test_aggressive_policy(self):
        """Should create aggressive retry policy."""
        config = RetryPolicy.aggressive()

        assert config.max_attempts == 5

    def test_patient_policy(self):
        """Should create patient retry policy."""
        config = RetryPolicy.patient()

        assert config.max_attempts == 10
        assert config.backoff_strategy == BackoffStrategy.LINEAR

    def test_websocket_policy(self):
        """Should create websocket retry policy."""
        config = RetryPolicy.websocket()

        assert config.backoff_strategy == BackoffStrategy.FIBONACCI


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Should work as decorator."""
        @retry(max_attempts=3, base_delay=0.01)
        async def decorated_func():
            return "success"

        result = await decorated_func()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_retry(self):
        """Should retry decorated function."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def flaky_decorated():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "ok"

        result = await flaky_decorated()

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_exhausted(self):
        """Should raise after retries exhausted."""
        @retry(max_attempts=2, base_delay=0.01)
        async def always_fails():
            raise ValueError("Fail")

        with pytest.raises(RetryExhausted):
            await always_fails()

    @pytest.mark.asyncio
    async def test_decorator_with_strategy(self):
        """Should use specified strategy."""
        @retry(
            max_attempts=3,
            base_delay=0.01,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        async def func():
            return "ok"

        result = await func()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_decorator_with_exception_filter(self):
        """Should only retry specified exceptions."""
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retry_exceptions=[ValueError],
        )
        async def type_error_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("Should not retry")

        with pytest.raises(RetryExhausted):
            await type_error_func()

        # Should only try once since TypeError not in retry list
        assert call_count == 1
