"""Unit tests for Order Manager."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

from src.core.order_manager import (
    OrderManager,
    OrderRequest,
    OrderResult,
    OrderType,
    OrderSide,
    OrderStatus,
)


class TestOrderType:
    """Tests for OrderType enum."""

    def test_order_types(self):
        """Should have expected order types."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_order_sides(self):
        """Should have expected order sides."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_order_statuses(self):
        """Should have expected order statuses."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.OPEN.value == "OPEN"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.FAILED.value == "FAILED"


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_create_order_request(self):
        """Should create order request correctly."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
        )

        assert request.account_id == "acc1"
        assert request.market == "BTC-USD-PERP"
        assert request.side == OrderSide.BUY
        assert request.size == Decimal("1.5")
        assert request.order_type == OrderType.MARKET
        assert request.client_id is not None

    def test_auto_generated_client_id(self):
        """Should generate client ID automatically."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.SELL,
            size=Decimal("1"),
        )

        assert request.client_id.startswith("delta-")
        assert len(request.client_id) == 14  # "delta-" + 8 hex chars

    def test_custom_client_id(self):
        """Should use custom client ID if provided."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
            client_id="custom-123",
        )

        assert request.client_id == "custom-123"

    def test_limit_order_with_price(self):
        """Should create limit order with price."""
        request = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
        )

        assert request.order_type == OrderType.LIMIT
        assert request.price == Decimal("50000")


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_create_success_result(self):
        """Should create success result correctly."""
        result = OrderResult(
            success=True,
            order_id="order-123",
            client_id="client-456",
            filled_size=Decimal("1.5"),
            avg_price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.filled_size == Decimal("1.5")
        assert result.status == OrderStatus.FILLED

    def test_create_failure_result(self):
        """Should create failure result correctly."""
        result = OrderResult(
            success=False,
            error="Insufficient margin",
            status=OrderStatus.FAILED,
        )

        assert result.success is False
        assert result.error == "Insufficient margin"
        assert result.status == OrderStatus.FAILED

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = OrderResult(
            success=True,
            order_id="order-123",
            client_id="client-456",
            filled_size=Decimal("1.5"),
            avg_price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["order_id"] == "order-123"
        assert d["filled_size"] == "1.5"
        assert d["status"] == "FILLED"


class TestOrderManager:
    """Tests for OrderManager."""

    @pytest.fixture
    def mock_account_manager(self):
        """Create mock account manager."""
        manager = MagicMock()
        manager.get_client.return_value = None
        return manager

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock()
        limiter.acquire_order_slot = AsyncMock()
        limiter.acquire_request_slot = AsyncMock()
        return limiter

    @pytest.fixture
    def order_manager(self, mock_account_manager, mock_rate_limiter):
        """Create order manager."""
        return OrderManager(
            account_manager=mock_account_manager,
            rate_limiter=mock_rate_limiter,
        )

    def test_initial_state(self, order_manager):
        """Should start with empty state."""
        assert order_manager.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_submit_order_no_client(self, order_manager):
        """Should fail when no client for account."""
        order = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
        )

        result = await order_manager.submit_order(order)

        assert result.success is False
        assert "No client" in result.error
        assert result.status == OrderStatus.FAILED

    @pytest.mark.asyncio
    async def test_submit_order_success(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should submit order successfully."""
        # Mock client
        mock_client = MagicMock()
        mock_client.api_client.submit_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "FILLED",
            "filled_size": "1.5",
            "avg_fill_price": "50000",
        })
        mock_account_manager.get_client.return_value = mock_client

        order = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
        )

        result = await order_manager.submit_order(order)

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.status == OrderStatus.FILLED
        mock_rate_limiter.acquire_order_slot.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_batch_empty(self, order_manager):
        """Should handle empty batch."""
        results = await order_manager.submit_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_submit_batch_single(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should use single order for batch of 1."""
        mock_client = MagicMock()
        mock_client.api_client.submit_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "FILLED",
            "filled_size": "1",
        })
        mock_account_manager.get_client.return_value = mock_client

        order = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
        )

        results = await order_manager.submit_batch([order])

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_submit_batch_multiple(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should group batch by account."""
        mock_client = MagicMock()
        mock_client.api_client.submit_orders_batch = AsyncMock(return_value=[
            {"id": "order-1", "status": "FILLED", "filled_size": "1"},
            {"id": "order-2", "status": "FILLED", "filled_size": "2"},
        ])
        mock_account_manager.get_client.return_value = mock_client

        orders = [
            OrderRequest("acc1", "BTC-USD-PERP", OrderSide.BUY, Decimal("1")),
            OrderRequest("acc1", "BTC-USD-PERP", OrderSide.SELL, Decimal("2")),
        ]

        results = await order_manager.submit_batch(orders)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cancel_order_no_client(self, order_manager):
        """Should fail when no client for account."""
        result = await order_manager.cancel_order("acc1", "order-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should cancel order successfully."""
        mock_client = MagicMock()
        mock_client.api_client.cancel_order = AsyncMock()
        mock_account_manager.get_client.return_value = mock_client

        result = await order_manager.cancel_order("acc1", "order-123")

        assert result is True
        mock_client.api_client.cancel_order.assert_called_once_with("order-123")

    @pytest.mark.asyncio
    async def test_cancel_all_orders_no_client(self, order_manager):
        """Should return 0 when no client."""
        result = await order_manager.cancel_all_orders("acc1")
        assert result == 0

    @pytest.mark.asyncio
    async def test_cancel_all_orders_success(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should cancel all orders successfully."""
        mock_client = MagicMock()
        mock_client.api_client.cancel_all_orders = AsyncMock(return_value={"count": 5})
        mock_account_manager.get_client.return_value = mock_client

        result = await order_manager.cancel_all_orders("acc1")

        assert result == 5

    @pytest.mark.asyncio
    async def test_get_open_orders_no_client(self, order_manager):
        """Should return empty list when no client."""
        result = await order_manager.get_open_orders("acc1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_open_orders_success(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should fetch open orders successfully."""
        mock_client = MagicMock()
        mock_client.api_client.fetch_orders = AsyncMock(return_value=[
            {"id": "order-1", "status": "OPEN"},
            {"id": "order-2", "status": "OPEN"},
        ])
        mock_account_manager.get_client.return_value = mock_client

        result = await order_manager.get_open_orders("acc1")

        assert len(result) == 2

    def test_get_result_not_found(self, order_manager):
        """Should return None for unknown order."""
        result = order_manager.get_result("nonexistent")
        assert result is None

    def test_get_pending_count(self, order_manager):
        """Should return pending order count."""
        order = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
        )
        order_manager._pending_orders["test-1"] = order
        order_manager._pending_orders["test-2"] = order

        assert order_manager.get_pending_count() == 2

    def test_clear_results(self, order_manager):
        """Should clear all order results."""
        result = OrderResult(success=True, order_id="order-1")
        order_manager._order_results["test-1"] = result

        order_manager.clear_results()

        assert len(order_manager._order_results) == 0

    @pytest.mark.asyncio
    async def test_submit_order_api_error_returns_failed(self, order_manager, mock_account_manager, mock_rate_limiter):
        """Should return failed result on API error."""
        mock_client = MagicMock()
        mock_client.api_client.submit_order = AsyncMock(side_effect=Exception("Network error"))
        mock_account_manager.get_client.return_value = mock_client

        order_manager.retry_delay = 0.01  # Fast retries for test

        order = OrderRequest(
            account_id="acc1",
            market="BTC-USD-PERP",
            side=OrderSide.BUY,
            size=Decimal("1"),
        )

        result = await order_manager.submit_order(order)

        # Note: _execute_order catches exceptions and returns failed result
        # So the outer retry doesn't see exceptions, just failed results
        assert result.success is False
        assert result.status == OrderStatus.FAILED
