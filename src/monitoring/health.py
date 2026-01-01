"""Health check and system monitoring."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class ComponentHealth:
    """Health of a system component."""

    name: str
    check_fn: Callable[[], HealthCheck]
    interval: float = 30.0  # Check interval in seconds
    last_check: Optional[HealthCheck] = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class HealthMonitor:
    """Monitors system health across components.

    Features:
    - Periodic health checks
    - Component status tracking
    - Alert on degradation
    - API connectivity monitoring
    - WebSocket status
    - Database health
    """

    check_interval: float = 30.0  # Default check interval
    alert_on_unhealthy: bool = True

    _components: Dict[str, ComponentHealth] = field(default_factory=dict)
    _running: bool = False
    _task: Optional[asyncio.Task] = None

    # Callbacks
    on_status_change: Optional[Callable[[str, HealthStatus], None]] = None
    on_alert: Optional[Callable[[HealthCheck], None]] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._components = {}
        self._running = False
        self._task = None

    def register_component(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
        interval: Optional[float] = None,
    ) -> None:
        """Register a component for health monitoring.

        Args:
            name: Component name
            check_fn: Function that returns HealthCheck
            interval: Check interval (uses default if not specified)
        """
        self._components[name] = ComponentHealth(
            name=name,
            check_fn=check_fn,
            interval=interval or self.check_interval,
        )
        logger.info(f"Registered health check: {name}")

    def unregister_component(self, name: str) -> bool:
        """Unregister a component.

        Args:
            name: Component name

        Returns:
            True if removed
        """
        if name in self._components:
            del self._components[name]
            return True
        return False

    async def check_component(self, name: str) -> Optional[HealthCheck]:
        """Check health of a specific component.

        Args:
            name: Component name

        Returns:
            HealthCheck result
        """
        component = self._components.get(name)
        if not component:
            return None

        start_time = time.time()
        try:
            # Run check
            if asyncio.iscoroutinefunction(component.check_fn):
                result = await component.check_fn()
            else:
                result = component.check_fn()

            result.latency_ms = (time.time() - start_time) * 1000

            # Track status changes
            old_status = component.last_check.status if component.last_check else None
            if old_status and old_status != result.status:
                logger.info(
                    f"Health status change: {name} "
                    f"{old_status.value} -> {result.status.value}"
                )
                if self.on_status_change:
                    self.on_status_change(name, result.status)

            # Track failures
            if result.status == HealthStatus.UNHEALTHY:
                component.consecutive_failures += 1
                if (
                    self.alert_on_unhealthy
                    and component.consecutive_failures >= component.max_failures
                    and self.on_alert
                ):
                    self.on_alert(result)
            else:
                component.consecutive_failures = 0

            component.last_check = result
            return result

        except Exception as e:
            logger.error(f"Health check error for {name}: {e}")
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
            component.consecutive_failures += 1
            component.last_check = result
            return result

    async def check_all(self) -> Dict[str, HealthCheck]:
        """Check health of all components.

        Returns:
            Dictionary of {component_name: HealthCheck}
        """
        results = {}
        for name in self._components:
            result = await self.check_component(name)
            if result:
                results[name] = result
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.

        Returns:
            Overall HealthStatus
        """
        if not self._components:
            return HealthStatus.UNKNOWN

        statuses = []
        for component in self._components.values():
            if component.last_check:
                statuses.append(component.last_check.status)

        if not statuses:
            return HealthStatus.UNKNOWN

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY

        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    async def start(self) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_checks())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _run_checks(self) -> None:
        """Background task for periodic checks."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)

    def get_status(self) -> Dict:
        """Get full health status.

        Returns:
            Status dictionary
        """
        components = {}
        for name, component in self._components.items():
            if component.last_check:
                components[name] = component.last_check.to_dict()
            else:
                components[name] = {
                    "status": "unknown",
                    "message": "Not checked yet",
                }

        return {
            "overall": self.get_overall_status().value,
            "is_running": self._running,
            "component_count": len(self._components),
            "components": components,
            "timestamp": time.time(),
        }


# Pre-built health check functions

def create_api_health_check(client) -> Callable:
    """Create health check for API connectivity.

    Args:
        client: Paradex client

    Returns:
        Check function
    """
    async def check() -> HealthCheck:
        try:
            start = time.time()
            # Simple ping - fetch system time
            await client.api_client.fetch_system_time()
            latency = (time.time() - start) * 1000

            if latency > 5000:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.DEGRADED,
                    message=f"High latency: {latency:.0f}ms",
                    latency_ms=latency,
                )

            return HealthCheck(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API responding",
                latency_ms=latency,
            )
        except Exception as e:
            return HealthCheck(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API error: {e}",
            )

    return check


def create_ws_health_check(ws_manager) -> Callable:
    """Create health check for WebSocket connections.

    Args:
        ws_manager: WebSocket manager

    Returns:
        Check function
    """
    def check() -> HealthCheck:
        try:
            connected = ws_manager.connected_count
            total = ws_manager.total_connections

            if connected == 0:
                return HealthCheck(
                    name="websocket",
                    status=HealthStatus.UNHEALTHY,
                    message="No WebSocket connections",
                    details={"connected": 0, "total": total},
                )

            if connected < total:
                return HealthCheck(
                    name="websocket",
                    status=HealthStatus.DEGRADED,
                    message=f"Only {connected}/{total} connected",
                    details={"connected": connected, "total": total},
                )

            return HealthCheck(
                name="websocket",
                status=HealthStatus.HEALTHY,
                message=f"All {connected} connections active",
                details={"connected": connected, "total": total},
            )
        except Exception as e:
            return HealthCheck(
                name="websocket",
                status=HealthStatus.UNHEALTHY,
                message=f"WS check error: {e}",
            )

    return check


def create_db_health_check(db) -> Callable:
    """Create health check for database.

    Args:
        db: Database instance

    Returns:
        Check function
    """
    def check() -> HealthCheck:
        try:
            start = time.time()
            # Simple query
            stats = db.get_statistics()
            latency = (time.time() - start) * 1000

            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message=f"DB ok, {stats.get('total_trades', 0)} trades",
                latency_ms=latency,
                details=stats,
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"DB error: {e}",
            )

    return check


def create_rate_limit_health_check(rate_limiter) -> Callable:
    """Create health check for rate limiter.

    Args:
        rate_limiter: Rate limiter instance

    Returns:
        Check function
    """
    def check() -> HealthCheck:
        try:
            usage = rate_limiter.get_current_usage()
            order_usage = usage["orders_per_second"]["usage_pct"]
            request_usage = usage["requests_per_minute_ip"]["usage_pct"]

            max_usage = max(order_usage, request_usage)

            if max_usage > 90:
                return HealthCheck(
                    name="rate_limiter",
                    status=HealthStatus.DEGRADED,
                    message=f"High rate limit usage: {max_usage:.0f}%",
                    details=usage,
                )

            return HealthCheck(
                name="rate_limiter",
                status=HealthStatus.HEALTHY,
                message=f"Rate limit usage: {max_usage:.0f}%",
                details=usage,
            )
        except Exception as e:
            return HealthCheck(
                name="rate_limiter",
                status=HealthStatus.UNHEALTHY,
                message=f"Rate limiter error: {e}",
            )

    return check
