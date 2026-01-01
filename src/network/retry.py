"""Retry mechanism for handling transient failures."""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""

    FIXED = "fixed"  # Fixed delay
    LINEAR = "linear"  # Linear increase
    EXPONENTIAL = "exponential"  # Exponential backoff
    FIBONACCI = "fibonacci"  # Fibonacci sequence


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0  # For exponential/linear
    jitter: bool = True  # Add randomness
    jitter_range: float = 0.25  # ±25% jitter
    retry_exceptions: List[Type[Exception]] = field(default_factory=list)
    exclude_exceptions: List[Type[Exception]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_strategy": self.backoff_strategy.value,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter": self.jitter,
        }


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any = None
    attempts: int = 0
    total_delay: float = 0.0
    last_exception: Optional[Exception] = None
    exceptions: List[Exception] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_delay": round(self.total_delay, 3),
            "had_errors": len(self.exceptions) > 0,
        }


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_retries: int = 0
    total_delay: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100

    @property
    def avg_retries(self) -> float:
        """Calculate average retries per operation."""
        if self.total_operations == 0:
            return 0.0
        return self.total_retries / self.total_operations

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "total_retries": self.total_retries,
            "success_rate": round(self.success_rate, 2),
            "avg_retries": round(self.avg_retries, 2),
        }


class RetryExhausted(Exception):
    """Exception raised when all retries are exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted. Last error: {last_exception}"
        )


@dataclass
class RetryHandler:
    """Handles retry logic with configurable backoff.

    Features:
    - Multiple backoff strategies
    - Configurable jitter
    - Exception filtering
    - Metrics tracking
    """

    config: RetryConfig = field(default_factory=RetryConfig)
    _metrics: RetryMetrics = field(default_factory=RetryMetrics)
    _fibonacci_cache: List[int] = field(default_factory=lambda: [1, 1])

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._metrics = RetryMetrics()
        self._fibonacci_cache = [1, 1]

    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n]

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        strategy = self.config.backoff_strategy

        if strategy == BackoffStrategy.FIXED:
            delay = self.config.base_delay

        elif strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * (1 + attempt * self.config.backoff_multiplier)

        elif strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)

        elif strategy == BackoffStrategy.FIBONACCI:
            delay = self.config.base_delay * self._get_fibonacci(attempt)

        else:
            delay = self.config.base_delay

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry.

        Args:
            exception: Exception that occurred

        Returns:
            True if should retry
        """
        # Check exclusions first
        for exc_type in self.config.exclude_exceptions:
            if isinstance(exception, exc_type):
                return False

        # If retry_exceptions is specified, only retry those
        if self.config.retry_exceptions:
            for exc_type in self.config.retry_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False

        # Default: retry all exceptions
        return True

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> RetryResult:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with outcome
        """
        self._metrics.total_operations += 1
        result = RetryResult(success=False)
        total_delay = 0.0

        for attempt in range(self.config.max_attempts):
            result.attempts = attempt + 1

            try:
                if asyncio.iscoroutinefunction(func):
                    outcome = await func(*args, **kwargs)
                else:
                    outcome = func(*args, **kwargs)

                result.success = True
                result.result = outcome
                result.total_delay = total_delay
                self._metrics.successful_operations += 1
                self._metrics.total_delay += total_delay

                if attempt > 0:
                    logger.info(f"Succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                result.exceptions.append(e)
                result.last_exception = e
                self._metrics.total_retries += 1

                if not self.should_retry(e):
                    logger.warning(f"Non-retryable exception: {e}")
                    break

                if attempt < self.config.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    total_delay += delay
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")

        result.total_delay = total_delay
        self._metrics.failed_operations += 1
        self._metrics.total_delay += total_delay

        return result

    async def execute_or_raise(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function and raise if all retries fail.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            RetryExhausted: If all retries fail
        """
        result = await self.execute(func, *args, **kwargs)

        if result.success:
            return result.result

        raise RetryExhausted(result.attempts, result.last_exception)

    def get_metrics(self) -> RetryMetrics:
        """Get retry metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics = RetryMetrics()


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    retry_exceptions: Optional[List[Type[Exception]]] = None,
    exclude_exceptions: Optional[List[Type[Exception]]] = None,
):
    """Decorator for adding retry logic to functions.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        backoff_strategy: Backoff strategy to use
        retry_exceptions: Only retry these exceptions
        exclude_exceptions: Never retry these exceptions

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_strategy=backoff_strategy,
            retry_exceptions=retry_exceptions or [],
            exclude_exceptions=exclude_exceptions or [],
        )
        handler = RetryHandler(config=config)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            return await handler.execute_or_raise(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            return asyncio.run(handler.execute_or_raise(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@dataclass
class RetryPolicy:
    """Named retry policies for different scenarios."""

    @staticmethod
    def fast() -> RetryConfig:
        """Fast retry for quick failures."""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )

    @staticmethod
    def standard() -> RetryConfig:
        """Standard retry for typical API calls."""
        return RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        )

    @staticmethod
    def aggressive() -> RetryConfig:
        """Aggressive retry for important operations."""
        return RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
        )

    @staticmethod
    def patient() -> RetryConfig:
        """Patient retry for long operations."""
        return RetryConfig(
            max_attempts=10,
            base_delay=5.0,
            max_delay=300.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )

    @staticmethod
    def websocket() -> RetryConfig:
        """Retry policy for WebSocket connections."""
        return RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.FIBONACCI,
        )
