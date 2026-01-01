"""Unit tests for Execution engine."""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

from src.core.execution_engine import (
    ExecutionState,
    OrderSide,
    OrderType,
    TimeInForce,
    ExecutionConfig,
    ExecutionMetrics,
    OrderRequest,
    ExecutionResult,
    ExecutionEngine,
    SmartOrderRouter,
    get_execution_engine,
    reset_execution_engine,
)


class TestExecutionState:
    """Tests for ExecutionState enum."""

    def test_state_values(self):
        """Should have expected state values."""
        assert ExecutionState.PENDING.value == "pending"
        assert ExecutionState.SUBMITTED.value == "submitted"
        assert ExecutionState.FILLED.value == "filled"
        assert ExecutionState.CANCELLED.value == "cancelled"
        assert ExecutionState.REJECTED.value == "rejected"
        assert ExecutionState.FAILED.value == "failed"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_side_values(self):
        """Should have expected side values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderType:
    """Tests for OrderType enum."""

    def test_type_values(self):
        """Should have expected type values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_MARKET.value == "stop_market"
        assert OrderType.STOP_LIMIT.value == "stop_limit"


class TestTimeInForce:
    """Tests for TimeInForce enum."""

    def test_tif_values(self):
        """Should have expected TIF values."""
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"


class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = ExecutionConfig()

        assert config.max_retries == 3
        assert config.retry_delay == 0.5
        assert config.timeout == 30.0
        assert config.max_slippage_pct == 0.5
        assert config.enable_pre_trade_checks is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = ExecutionConfig(
            max_retries=5,
            max_slippage_pct=1.0,
        )

        assert config.max_retries == 5
        assert config.max_slippage_pct == 1.0


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = ExecutionMetrics()

        assert metrics.total_orders == 0
        assert metrics.successful_orders == 0
        assert metrics.total_volume == Decimal("0")

    def test_record_successful_order(self):
        """Should record successful order."""
        metrics = ExecutionMetrics()

        metrics.record_order(
            success=True,
            volume=Decimal("1000"),
            fees=Decimal("1"),
            execution_time_ms=50.0,
        )

        assert metrics.total_orders == 1
        assert metrics.successful_orders == 1
        assert metrics.total_volume == Decimal("1000")

    def test_record_failed_order(self):
        """Should record failed order."""
        metrics = ExecutionMetrics()

        metrics.record_order(
            success=False,
            volume=Decimal("0"),
            fees=Decimal("0"),
            execution_time_ms=50.0,
        )

        assert metrics.total_orders == 1
        assert metrics.failed_orders == 1

    def test_success_rate(self):
        """Should calculate success rate."""
        metrics = ExecutionMetrics(
            total_orders=10,
            successful_orders=8,
        )

        assert metrics.success_rate == 80.0

    def test_success_rate_no_orders(self):
        """Should return 100% with no orders."""
        metrics = ExecutionMetrics()

        assert metrics.success_rate == 100.0

    def test_avg_slippage(self):
        """Should calculate average slippage."""
        metrics = ExecutionMetrics()

        metrics.record_order(True, Decimal("100"), Decimal("0"), 10, slippage_pct=0.1)
        metrics.record_order(True, Decimal("100"), Decimal("0"), 10, slippage_pct=0.3)

        assert metrics.avg_slippage_pct == 0.2

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = ExecutionMetrics()
        metrics.record_order(True, Decimal("1000"), Decimal("1"), 50)

        d = metrics.to_dict()

        assert d["total_orders"] == 1
        assert d["total_volume"] == "1000"


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_create_request(self):
        """Should create order request."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
        )

        assert request.account_id == "acc1"
        assert request.market == "BTC-USD-PERP"
        assert request.size == Decimal("1.5")

    def test_order_id_generated(self):
        """Should generate unique order ID."""
        request1 = OrderRequest(account_id="acc1", market="BTC", size=Decimal("1"))
        request2 = OrderRequest(account_id="acc1", market="BTC", size=Decimal("1"))

        assert request1.order_id != request2.order_id

    def test_validate_valid_market_order(self):
        """Should validate valid market order."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("1"),
        )

        is_valid, error = request.validate()

        assert is_valid is True
        assert error is None

    def test_validate_missing_account(self):
        """Should reject missing account ID."""
        request = OrderRequest(
            market="BTC-USD-PERP",
            size=Decimal("1"),
        )

        is_valid, error = request.validate()

        assert is_valid is False
        assert "Account ID" in error

    def test_validate_missing_market(self):
        """Should reject missing market."""
        request = OrderRequest(
            account_id="acc1",
            size=Decimal("1"),
        )

        is_valid, error = request.validate()

        assert is_valid is False
        assert "Market" in error

    def test_validate_zero_size(self):
        """Should reject zero size."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("0"),
        )

        is_valid, error = request.validate()

        assert is_valid is False
        assert "Size" in error

    def test_validate_limit_without_price(self):
        """Should reject limit order without price."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            order_type=OrderType.LIMIT,
            size=Decimal("1"),
        )

        is_valid, error = request.validate()

        assert is_valid is False
        assert "price" in error.lower()

    def test_validate_stop_without_stop_price(self):
        """Should reject stop order without stop price."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            order_type=OrderType.STOP_MARKET,
            size=Decimal("1"),
        )

        is_valid, error = request.validate()

        assert is_valid is False
        assert "stop price" in error.lower()

    def test_to_dict(self):
        """Should convert to dictionary."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
        )

        d = request.to_dict()

        assert d["account_id"] == "acc1"
        assert d["market"] == "BTC-USD-PERP"
        assert d["size"] == "1.5"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_result(self):
        """Should create execution result."""
        result = ExecutionResult(
            order_id="order123",
            state=ExecutionState.FILLED,
            filled_size=Decimal("1.5"),
            filled_price=Decimal("50000"),
        )

        assert result.order_id == "order123"
        assert result.filled_size == Decimal("1.5")

    def test_is_success_filled(self):
        """Should detect successful fill."""
        result = ExecutionResult(
            order_id="order123",
            state=ExecutionState.FILLED,
        )

        assert result.is_success is True

    def test_is_success_partial(self):
        """Should detect partial fill as success."""
        result = ExecutionResult(
            order_id="order123",
            state=ExecutionState.PARTIAL,
        )

        assert result.is_success is True

    def test_is_success_failed(self):
        """Should detect failure."""
        result = ExecutionResult(
            order_id="order123",
            state=ExecutionState.FAILED,
        )

        assert result.is_success is False

    def test_is_complete(self):
        """Should detect complete states."""
        filled = ExecutionResult(order_id="1", state=ExecutionState.FILLED)
        cancelled = ExecutionResult(order_id="2", state=ExecutionState.CANCELLED)
        pending = ExecutionResult(order_id="3", state=ExecutionState.PENDING)

        assert filled.is_complete is True
        assert cancelled.is_complete is True
        assert pending.is_complete is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ExecutionResult(
            order_id="order123",
            state=ExecutionState.FILLED,
            filled_size=Decimal("1.5"),
            execution_time_ms=50.0,
        )

        d = result.to_dict()

        assert d["order_id"] == "order123"
        assert d["state"] == "filled"


class TestExecutionEngine:
    """Tests for ExecutionEngine."""

    @pytest.fixture
    def engine(self):
        """Create fresh execution engine."""
        return ExecutionEngine()

    @pytest.fixture
    def valid_request(self):
        """Create valid order request."""
        return OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("1"),
        )

    @pytest.mark.asyncio
    async def test_create_engine(self, engine):
        """Should create engine."""
        assert len(engine._pending_orders) == 0
        assert len(engine._results) == 0

    @pytest.mark.asyncio
    async def test_submit_valid_order(self, engine, valid_request):
        """Should submit and execute valid order."""
        result = await engine.submit_order(valid_request)

        assert result.is_success is True
        assert result.state == ExecutionState.FILLED
        assert result.order_id == valid_request.order_id

    @pytest.mark.asyncio
    async def test_submit_invalid_order(self, engine):
        """Should reject invalid order."""
        request = OrderRequest(
            account_id="",  # Invalid - empty
            market="BTC-USD-PERP",
            size=Decimal("1"),
        )

        result = await engine.submit_order(request)

        assert result.state == ExecutionState.REJECTED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_submit_with_custom_executor(self, engine, valid_request):
        """Should use custom executor."""
        custom_result = ExecutionResult(
            order_id=valid_request.order_id,
            state=ExecutionState.FILLED,
            filled_size=Decimal("1"),
            filled_price=Decimal("50000"),
        )
        executor = AsyncMock(return_value=custom_result)
        engine.set_executor(executor)

        result = await engine.submit_order(valid_request)

        executor.assert_called_once()
        assert result == custom_result

    @pytest.mark.asyncio
    async def test_submit_with_validation(self, engine, valid_request):
        """Should run pre-trade validation."""
        validator = AsyncMock(return_value=True)
        engine.set_validator(validator)

        await engine.submit_order(valid_request)

        validator.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_validation_failure(self, engine, valid_request):
        """Should reject on validation failure."""
        validator = AsyncMock(return_value=False)
        engine.set_validator(validator)

        result = await engine.submit_order(valid_request)

        assert result.state == ExecutionState.REJECTED
        assert "validation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_submit_retry_on_failure(self, engine, valid_request):
        """Should retry on failure."""
        engine.config.max_retries = 3
        engine.config.retry_delay = 0.01

        call_count = 0
        async def flaky_executor(request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return ExecutionResult(
                order_id=request.order_id,
                state=ExecutionState.FILLED,
                filled_size=request.size,
            )

        engine.set_executor(flaky_executor)

        result = await engine.submit_order(valid_request)

        assert call_count == 3
        assert result.is_success is True

    @pytest.mark.asyncio
    async def test_submit_batch(self, engine):
        """Should submit batch of orders."""
        requests = [
            OrderRequest(
                account_id="acc1",
                market="BTC-USD-PERP",
                side=OrderSide.BUY,
                size=Decimal("1"),
            )
            for _ in range(5)
        ]

        results = await engine.submit_batch(requests)

        assert len(results) == 5
        assert all(r.is_success for r in results)

    @pytest.mark.asyncio
    async def test_cancel_pending_order(self, engine, valid_request):
        """Should cancel pending order."""
        engine._pending_orders[valid_request.order_id] = valid_request

        result = await engine.cancel_order(valid_request.order_id)

        assert result.state == ExecutionState.CANCELLED
        assert valid_request.order_id not in engine._pending_orders

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, engine):
        """Should fail for nonexistent order."""
        result = await engine.cancel_order("nonexistent")

        assert result.state == ExecutionState.FAILED

    @pytest.mark.asyncio
    async def test_cancel_all(self, engine):
        """Should cancel all pending orders."""
        for i in range(3):
            request = OrderRequest(
                account_id="acc1",
                market="BTC-USD-PERP",
                size=Decimal("1"),
            )
            engine._pending_orders[request.order_id] = request

        results = await engine.cancel_all()

        assert len(results) == 3
        assert all(r.state == ExecutionState.CANCELLED for r in results)

    @pytest.mark.asyncio
    async def test_cancel_all_by_account(self, engine):
        """Should cancel orders for specific account."""
        for account_id in ["acc1", "acc1", "acc2"]:
            request = OrderRequest(
                account_id=account_id,
                market="BTC-USD-PERP",
                size=Decimal("1"),
            )
            engine._pending_orders[request.order_id] = request

        results = await engine.cancel_all(account_id="acc1")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_order_result(self, engine, valid_request):
        """Should get stored result."""
        await engine.submit_order(valid_request)

        result = engine.get_order_result(valid_request.order_id)

        assert result is not None
        assert result.order_id == valid_request.order_id

    @pytest.mark.asyncio
    async def test_get_pending_orders(self, engine):
        """Should get pending orders."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("1"),
        )
        engine._pending_orders[request.order_id] = request

        pending = engine.get_pending_orders()

        assert len(pending) == 1

    def test_calculate_slippage(self, engine):
        """Should calculate slippage."""
        expected = Decimal("100")
        actual = Decimal("101")

        slippage = engine.calculate_slippage(expected, actual)

        assert slippage == 1.0  # 1%

    def test_check_slippage_within_limit(self, engine):
        """Should pass when slippage within limit."""
        engine.config.max_slippage_pct = 0.5

        within, slippage = engine.check_slippage(
            Decimal("100"),
            Decimal("100.4"),
        )

        assert within is True
        assert slippage == 0.4

    def test_check_slippage_exceeds_limit(self, engine):
        """Should fail when slippage exceeds limit."""
        engine.config.max_slippage_pct = 0.5

        within, slippage = engine.check_slippage(
            Decimal("100"),
            Decimal("101"),
        )

        assert within is False
        assert slippage == 1.0

    @pytest.mark.asyncio
    async def test_metrics_tracked(self, engine, valid_request):
        """Should track execution metrics."""
        await engine.submit_order(valid_request)

        metrics = engine.get_metrics()

        assert metrics.total_orders == 1
        assert metrics.successful_orders == 1

    def test_reset_metrics(self, engine):
        """Should reset metrics."""
        engine._metrics.total_orders = 100

        engine.reset_metrics()

        assert engine._metrics.total_orders == 0

    @pytest.mark.asyncio
    async def test_get_status(self, engine, valid_request):
        """Should get engine status."""
        await engine.submit_order(valid_request)

        status = engine.get_status()

        assert status["completed_orders"] == 1
        assert "metrics" in status


class TestSmartOrderRouter:
    """Tests for SmartOrderRouter."""

    @pytest.fixture
    def router(self):
        """Create router."""
        return SmartOrderRouter()

    def test_add_venue(self, router):
        """Should add venue."""
        router.add_venue("venue1", weight=0.8)

        assert "venue1" in router.get_venues()
        assert router._weights["venue1"] == 0.8

    def test_remove_venue(self, router):
        """Should remove venue."""
        router.add_venue("venue1")

        result = router.remove_venue("venue1")

        assert result is True
        assert "venue1" not in router.get_venues()

    def test_remove_nonexistent_venue(self, router):
        """Should return False for nonexistent venue."""
        result = router.remove_venue("nonexistent")

        assert result is False

    def test_get_best_venue(self, router):
        """Should get venue with highest weight."""
        router.add_venue("venue1", weight=0.5)
        router.add_venue("venue2", weight=0.9)
        router.add_venue("venue3", weight=0.7)

        best = router.get_best_venue()

        assert best == "venue2"

    def test_get_best_venue_empty(self, router):
        """Should return None when no venues."""
        best = router.get_best_venue()

        assert best is None

    def test_to_dict(self, router):
        """Should convert to dictionary."""
        router.add_venue("venue1", weight=0.8)

        d = router.to_dict()

        assert "venue1" in d["venues"]
        assert d["weights"]["venue1"] == 0.8


class TestGlobalExecutionEngine:
    """Tests for global execution engine."""

    def setup_method(self):
        """Reset global engine before each test."""
        reset_execution_engine()

    def test_get_execution_engine_creates_singleton(self):
        """Should create singleton engine."""
        engine1 = get_execution_engine()
        engine2 = get_execution_engine()

        assert engine1 is engine2

    def test_reset_execution_engine(self):
        """Should reset global engine."""
        engine1 = get_execution_engine()
        reset_execution_engine()
        engine2 = get_execution_engine()

        assert engine1 is not engine2
