"""Order execution engine for managing trade execution."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Execution states."""

    PENDING = "pending"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options."""

    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""

    max_retries: int = 3
    retry_delay: float = 0.5
    timeout: float = 30.0
    max_slippage_pct: float = 0.5
    enable_pre_trade_checks: bool = True
    enable_post_trade_validation: bool = True
    batch_size: int = 10
    rate_limit_delay: float = 0.1


@dataclass
class ExecutionMetrics:
    """Metrics for execution engine."""

    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_execution_time_ms: float = 0.0
    avg_slippage_pct: float = 0.0
    _total_execution_time: float = 0.0
    _total_slippage: float = 0.0
    _slippage_count: int = 0

    def record_order(
        self,
        success: bool,
        volume: Decimal,
        fees: Decimal,
        execution_time_ms: float,
        slippage_pct: float = 0.0,
    ) -> None:
        """Record order execution."""
        self.total_orders += 1
        if success:
            self.successful_orders += 1
        else:
            self.failed_orders += 1

        self.total_volume += volume
        self.total_fees += fees
        self._total_execution_time += execution_time_ms
        self.avg_execution_time_ms = self._total_execution_time / self.total_orders

        if slippage_pct != 0:
            self._total_slippage += slippage_pct
            self._slippage_count += 1
            self.avg_slippage_pct = self._total_slippage / self._slippage_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_orders == 0:
            return 100.0
        return (self.successful_orders / self.total_orders) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": round(self.success_rate, 2),
            "total_volume": str(self.total_volume),
            "total_fees": str(self.total_fees),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "avg_slippage_pct": round(self.avg_slippage_pct, 4),
        }


@dataclass
class OrderRequest:
    """Order request for execution."""

    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    account_id: str = ""
    market: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    size: Decimal = field(default_factory=lambda: Decimal("0"))
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize metadata."""
        if self.metadata is None:
            self.metadata = {}

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate order request."""
        if not self.account_id:
            return False, "Account ID required"
        if not self.market:
            return False, "Market required"
        if self.size <= 0:
            return False, "Size must be positive"
        if self.order_type == OrderType.LIMIT and self.price is None:
            return False, "Limit orders require price"
        if self.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT) and self.stop_price is None:
            return False, "Stop orders require stop price"
        return True, None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "account_id": self.account_id,
            "market": self.market,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": str(self.size),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
        }


@dataclass
class ExecutionResult:
    """Result of order execution."""

    order_id: str
    state: ExecutionState
    filled_size: Decimal = field(default_factory=lambda: Decimal("0"))
    filled_price: Optional[Decimal] = None
    fees: Decimal = field(default_factory=lambda: Decimal("0"))
    exchange_order_id: Optional[str] = None
    execution_time_ms: float = 0.0
    slippage_pct: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    fills: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize fills list."""
        if self.fills is None:
            self.fills = []

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.state in (ExecutionState.FILLED, ExecutionState.PARTIAL)

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.state in (
            ExecutionState.FILLED,
            ExecutionState.CANCELLED,
            ExecutionState.REJECTED,
            ExecutionState.EXPIRED,
            ExecutionState.FAILED,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "state": self.state.value,
            "filled_size": str(self.filled_size),
            "filled_price": str(self.filled_price) if self.filled_price else None,
            "fees": str(self.fees),
            "exchange_order_id": self.exchange_order_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "slippage_pct": round(self.slippage_pct, 4),
            "error": self.error,
            "fills_count": len(self.fills),
        }


@dataclass
class ExecutionEngine:
    """Engine for executing orders.

    Features:
    - Order validation
    - Retry with backoff
    - Slippage control
    - Batch execution
    - Metrics tracking
    """

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    _metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    _pending_orders: Dict[str, OrderRequest] = field(default_factory=dict)
    _results: Dict[str, ExecutionResult] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _executor: Optional[Callable] = None
    _validator: Optional[Callable] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._metrics = ExecutionMetrics()
        self._pending_orders = {}
        self._results = {}
        self._lock = asyncio.Lock()

    def set_executor(self, executor: Callable) -> None:
        """Set the order execution function."""
        self._executor = executor

    def set_validator(self, validator: Callable) -> None:
        """Set the order validation function."""
        self._validator = validator

    async def submit_order(self, request: OrderRequest) -> ExecutionResult:
        """Submit an order for execution."""
        start_time = time.time()

        # Validate request
        is_valid, error = request.validate()
        if not is_valid:
            return ExecutionResult(
                order_id=request.order_id,
                state=ExecutionState.REJECTED,
                error=error,
            )

        # Pre-trade validation
        if self.config.enable_pre_trade_checks and self._validator:
            try:
                is_valid = await self._validate_order(request)
                if not is_valid:
                    return ExecutionResult(
                        order_id=request.order_id,
                        state=ExecutionState.REJECTED,
                        error="Pre-trade validation failed",
                    )
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return ExecutionResult(
                    order_id=request.order_id,
                    state=ExecutionState.FAILED,
                    error=str(e),
                )

        # Store pending order
        self._pending_orders[request.order_id] = request

        # Execute with retries
        result = await self._execute_with_retry(request)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time_ms

        # Store result
        self._results[request.order_id] = result

        # Record metrics
        self._metrics.record_order(
            success=result.is_success,
            volume=result.filled_size,
            fees=result.fees,
            execution_time_ms=execution_time_ms,
            slippage_pct=result.slippage_pct,
        )

        # Remove from pending
        del self._pending_orders[request.order_id]

        return result

    async def _validate_order(self, request: OrderRequest) -> bool:
        """Run pre-trade validation."""
        if not self._validator:
            return True

        if asyncio.iscoroutinefunction(self._validator):
            return await self._validator(request)
        return self._validator(request)

    async def _execute_with_retry(self, request: OrderRequest) -> ExecutionResult:
        """Execute order with retries."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = await self._execute_order(request)
                if result.state != ExecutionState.FAILED:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return ExecutionResult(
            order_id=request.order_id,
            state=ExecutionState.FAILED,
            error=f"Failed after {self.config.max_retries} attempts: {last_error}",
        )

    async def _execute_order(self, request: OrderRequest) -> ExecutionResult:
        """Execute a single order."""
        if not self._executor:
            # Simulate execution for testing
            return ExecutionResult(
                order_id=request.order_id,
                state=ExecutionState.FILLED,
                filled_size=request.size,
                filled_price=request.price or Decimal("100"),
                fees=request.size * Decimal("0.001"),
            )

        if asyncio.iscoroutinefunction(self._executor):
            return await self._executor(request)
        return self._executor(request)

    async def submit_batch(
        self,
        requests: List[OrderRequest],
    ) -> List[ExecutionResult]:
        """Submit multiple orders."""
        results = []

        for i in range(0, len(requests), self.config.batch_size):
            batch = requests[i:i + self.config.batch_size]

            # Execute batch in parallel
            tasks = [self.submit_order(req) for req in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Rate limit between batches
            if i + self.config.batch_size < len(requests):
                await asyncio.sleep(self.config.rate_limit_delay)

        return results

    async def cancel_order(
        self,
        order_id: str,
        account_id: str = None,
    ) -> ExecutionResult:
        """Cancel a pending order."""
        if order_id in self._pending_orders:
            request = self._pending_orders.pop(order_id)
            return ExecutionResult(
                order_id=order_id,
                state=ExecutionState.CANCELLED,
            )

        return ExecutionResult(
            order_id=order_id,
            state=ExecutionState.FAILED,
            error="Order not found",
        )

    async def cancel_all(
        self,
        account_id: str = None,
        market: str = None,
    ) -> List[ExecutionResult]:
        """Cancel all pending orders."""
        results = []

        for order_id, request in list(self._pending_orders.items()):
            if account_id and request.account_id != account_id:
                continue
            if market and request.market != market:
                continue

            result = await self.cancel_order(order_id)
            results.append(result)

        return results

    def get_order_result(self, order_id: str) -> Optional[ExecutionResult]:
        """Get result for an order."""
        return self._results.get(order_id)

    def get_pending_orders(
        self,
        account_id: str = None,
    ) -> List[OrderRequest]:
        """Get pending orders."""
        if account_id:
            return [o for o in self._pending_orders.values() if o.account_id == account_id]
        return list(self._pending_orders.values())

    def calculate_slippage(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
    ) -> float:
        """Calculate slippage percentage."""
        if expected_price == 0:
            return 0.0
        return float(abs(actual_price - expected_price) / expected_price * 100)

    def check_slippage(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
    ) -> Tuple[bool, float]:
        """Check if slippage is within limits."""
        slippage = self.calculate_slippage(expected_price, actual_price)
        return slippage <= self.config.max_slippage_pct, slippage

    def get_metrics(self) -> ExecutionMetrics:
        """Get execution metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self._metrics = ExecutionMetrics()

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "pending_orders": len(self._pending_orders),
            "completed_orders": len(self._results),
            "config": {
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "max_slippage_pct": self.config.max_slippage_pct,
            },
            "metrics": self._metrics.to_dict(),
        }


@dataclass
class SmartOrderRouter:
    """Smart order router for optimal execution."""

    _venues: List[str] = field(default_factory=list)
    _weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize."""
        if self._venues is None:
            self._venues = []
        if self._weights is None:
            self._weights = {}

    def add_venue(self, venue: str, weight: float = 1.0) -> None:
        """Add execution venue."""
        if venue not in self._venues:
            self._venues.append(venue)
        self._weights[venue] = weight

    def remove_venue(self, venue: str) -> bool:
        """Remove execution venue."""
        if venue in self._venues:
            self._venues.remove(venue)
            del self._weights[venue]
            return True
        return False

    def get_best_venue(self) -> Optional[str]:
        """Get best venue by weight."""
        if not self._venues:
            return None
        return max(self._venues, key=lambda v: self._weights.get(v, 0))

    def get_venues(self) -> List[str]:
        """Get all venues."""
        return list(self._venues)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "venues": self._venues,
            "weights": self._weights,
        }


# Global execution engine
_global_engine: Optional[ExecutionEngine] = None


def get_execution_engine() -> ExecutionEngine:
    """Get or create global execution engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = ExecutionEngine()
    return _global_engine


def reset_execution_engine() -> None:
    """Reset global execution engine."""
    global _global_engine
    _global_engine = None
