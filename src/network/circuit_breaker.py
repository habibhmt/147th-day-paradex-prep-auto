"""Circuit breaker pattern implementation for API resilience."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout: float = 30.0  # Seconds in open state before half-open
    half_open_max_calls: int = 3  # Max calls allowed in half-open
    exclude_exceptions: List[type] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout": self.timeout,
            "half_open_max_calls": self.half_open_max_calls,
        }


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": round(self.success_rate, 2),
            "failure_rate": round(self.failure_rate, 2),
            "state_changes": self.state_changes,
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, breaker_name: str, retry_after: float):
        self.breaker_name = breaker_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open. Retry after {retry_after:.1f}s"
        )


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting API calls.

    Implements the circuit breaker pattern to prevent cascade failures.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting all calls
    - HALF_OPEN: Testing if service recovered
    """

    name: str = "default"
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0
    _open_time: float = 0.0
    _half_open_calls: int = 0
    _metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _callbacks: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._open_time = 0.0
        self._half_open_calls = 0
        self._metrics = CircuitMetrics()
        self._lock = asyncio.Lock()
        self._callbacks = {}

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes.

        Args:
            callback: Function(name, old_state, new_state)
        """
        self._callbacks["state_change"] = callback

    def on_open(self, callback: Callable) -> None:
        """Register callback for circuit opening.

        Args:
            callback: Function(name, failure_count)
        """
        self._callbacks["open"] = callback

    def on_close(self, callback: Callable) -> None:
        """Register callback for circuit closing.

        Args:
            callback: Function(name)
        """
        self._callbacks["close"] = callback

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state.

        Args:
            new_state: New state to transition to
        """
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change = time.time()

        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}"
        )

        # Call callbacks
        if "state_change" in self._callbacks:
            try:
                self._callbacks["state_change"](self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        if new_state == CircuitState.OPEN and "open" in self._callbacks:
            try:
                self._callbacks["open"](self.name, self._failure_count)
            except Exception as e:
                logger.error(f"Open callback error: {e}")

        if new_state == CircuitState.CLOSED and "close" in self._callbacks:
            try:
                self._callbacks["close"](self.name)
            except Exception as e:
                logger.error(f"Close callback error: {e}")

    async def _check_timeout(self) -> None:
        """Check if open circuit should transition to half-open."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._open_time
            if elapsed >= self.config.timeout:
                await self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
                self._success_count = 0

    async def _record_success(self) -> None:
        """Record a successful call."""
        self._metrics.successful_calls += 1
        self._metrics.last_success_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                await self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
                self._success_count = 0

        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    async def _record_failure(self, exception: Exception) -> None:
        """Record a failed call.

        Args:
            exception: The exception that caused the failure
        """
        # Check if exception should be excluded
        if any(isinstance(exception, exc) for exc in self.config.exclude_exceptions):
            return

        self._metrics.failed_calls += 1
        self._metrics.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self._failure_count += 1

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                self._open_time = time.time()

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens
            await self._transition_to(CircuitState.OPEN)
            self._open_time = time.time()

    async def can_execute(self) -> bool:
        """Check if a call can be executed.

        Returns:
            True if call is allowed
        """
        async with self._lock:
            await self._check_timeout()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        async with self._lock:
            await self._check_timeout()
            self._metrics.total_calls += 1

            if self._state == CircuitState.OPEN:
                retry_after = self.config.timeout - (time.time() - self._open_time)
                self._metrics.rejected_calls += 1
                raise CircuitBreakerOpen(self.name, max(0, retry_after))

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_calls += 1
                    raise CircuitBreakerOpen(self.name, self.config.timeout)
                self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(e)
            raise

    def get_time_until_retry(self) -> float:
        """Get seconds until retry is allowed.

        Returns:
            Seconds until retry, 0 if retry allowed now
        """
        if self._state != CircuitState.OPEN:
            return 0.0

        elapsed = time.time() - self._open_time
        remaining = self.config.timeout - elapsed
        return max(0.0, remaining)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._open_time = 0.0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "time_until_retry": self.get_time_until_retry(),
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
        }


@dataclass
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _breakers: Dict[str, CircuitBreaker] = field(default_factory=dict)
    _default_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._breakers = {}
        self._default_config = CircuitBreakerConfig()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name.

        Args:
            name: Circuit breaker name
            config: Optional custom config

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            CircuitBreaker or None
        """
        return self._breakers.get(name)

    def remove(self, name: str) -> bool:
        """Remove a circuit breaker.

        Args:
            name: Circuit breaker name

        Returns:
            True if removed
        """
        if name in self._breakers:
            del self._breakers[name]
            return True
        return False

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary of statuses
        """
        return {name: cb.get_status() for name, cb in self._breakers.items()}

    def get_open_breakers(self) -> List[str]:
        """Get names of open circuit breakers.

        Returns:
            List of open breaker names
        """
        return [
            name for name, cb in self._breakers.items()
            if cb.is_open
        ]

    @property
    def count(self) -> int:
        """Get number of registered breakers."""
        return len(self._breakers)
