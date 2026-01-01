"""Unit tests for Rebalancing Engine."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
import time
import asyncio

from src.rebalancing.engine import (
    RebalancingEngine,
    RebalanceAction,
    RebalanceResult,
    RebalanceTrigger,
)
from src.core.delta_calculator import DeltaReport
from src.strategies.base import StrategyAllocation


class TestRebalanceAction:
    """Tests for RebalanceAction dataclass."""

    def test_create_action(self):
        """Should create rebalance action correctly."""
        action = RebalanceAction(
            account_id="acc1",
            market="BTC-USD-PERP",
            action="decrease",
            side="LONG",
            size_change=Decimal("500"),
            reason="Delta too high",
        )

        assert action.account_id == "acc1"
        assert action.market == "BTC-USD-PERP"
        assert action.action == "decrease"
        assert action.size_change == Decimal("500")

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        action = RebalanceAction(
            account_id="acc1",
            market="BTC-USD-PERP",
            action="increase",
            side="SHORT",
            size_change=Decimal("1000"),
        )

        d = action.to_dict()

        assert d["account_id"] == "acc1"
        assert d["size_change"] == "1000"
        assert d["side"] == "SHORT"


class TestRebalanceResult:
    """Tests for RebalanceResult dataclass."""

    def test_create_result(self):
        """Should create rebalance result correctly."""
        result = RebalanceResult(
            success=True,
            trigger=RebalanceTrigger.THRESHOLD_EXCEEDED,
            actions_planned=4,
            actions_executed=4,
            delta_before=Decimal("1000"),
            delta_after=Decimal("0"),
        )

        assert result.success is True
        assert result.trigger == RebalanceTrigger.THRESHOLD_EXCEEDED
        assert result.actions_planned == 4
        assert result.actions_executed == 4

    def test_result_with_errors(self):
        """Should track errors correctly."""
        result = RebalanceResult(
            success=False,
            trigger=RebalanceTrigger.MANUAL,
            actions_planned=4,
            actions_executed=2,
            delta_before=Decimal("1000"),
            errors=["Order rejected", "Insufficient margin"],
        )

        assert result.success is False
        assert len(result.errors) == 2


class TestRebalanceTrigger:
    """Tests for RebalanceTrigger enum."""

    def test_trigger_values(self):
        """Should have expected trigger values."""
        assert RebalanceTrigger.THRESHOLD_EXCEEDED.value == "threshold"
        assert RebalanceTrigger.SCHEDULED.value == "scheduled"
        assert RebalanceTrigger.MANUAL.value == "manual"
        assert RebalanceTrigger.FUNDING_CHANGE.value == "funding"
        assert RebalanceTrigger.POSITION_CLOSED.value == "position_closed"


class TestRebalancingEngine:
    """Tests for RebalancingEngine."""

    @pytest.fixture
    def mock_delta_calculator(self):
        """Create mock delta calculator."""
        calc = MagicMock()
        calc.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("5000"),
            total_short=Decimal("5000"),
            net_delta=Decimal("0"),
            delta_percentage=0.0,
            is_neutral=True,
            accounts_long=["acc1", "acc2"],
            accounts_short=["acc3", "acc4"],
        )
        return calc

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        manager = MagicMock()
        manager.submit_batch = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy."""
        strategy = MagicMock()
        strategy.get_rebalance_allocations.return_value = []
        return strategy

    @pytest.fixture
    def engine(self, mock_delta_calculator, mock_order_manager, mock_strategy):
        """Create rebalancing engine."""
        return RebalancingEngine(
            delta_calculator=mock_delta_calculator,
            order_manager=mock_order_manager,
            strategy=mock_strategy,
            threshold_pct=5.0,
            min_rebalance_interval=300.0,
        )

    @pytest.mark.asyncio
    async def test_check_and_rebalance_neutral(self, engine):
        """Should not rebalance when already neutral."""
        result = await engine.check_and_rebalance("BTC-USD-PERP")

        assert result is None  # No rebalance needed

    @pytest.mark.asyncio
    async def test_check_and_rebalance_needed(self, engine, mock_delta_calculator, mock_strategy, mock_order_manager):
        """Should rebalance when delta exceeds threshold."""
        # Set up imbalanced delta report
        mock_delta_calculator.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("6000"),
            total_short=Decimal("4000"),
            net_delta=Decimal("2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1", "acc2"],
            accounts_short=["acc3", "acc4"],
        )

        # Set up allocations
        mock_strategy.get_rebalance_allocations.return_value = [
            StrategyAllocation("acc1", "LONG", Decimal("-500"), Decimal("1")),
            StrategyAllocation("acc2", "LONG", Decimal("-500"), Decimal("1")),
        ]

        # Mock order results
        mock_order_result = MagicMock()
        mock_order_result.success = True
        mock_order_result.error = None
        mock_order_manager.submit_batch.return_value = [mock_order_result, mock_order_result]

        result = await engine.check_and_rebalance("BTC-USD-PERP")

        assert result is not None
        assert result.actions_planned == 2

    @pytest.mark.asyncio
    async def test_check_and_rebalance_interval_limit(self, engine, mock_delta_calculator):
        """Should respect minimum rebalance interval."""
        # Set up imbalanced delta
        mock_delta_calculator.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("6000"),
            total_short=Decimal("4000"),
            net_delta=Decimal("2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1"],
            accounts_short=["acc2"],
        )

        # Simulate recent rebalance
        engine._last_rebalance["BTC-USD-PERP"] = time.time()

        result = await engine.check_and_rebalance("BTC-USD-PERP")

        # Should not rebalance due to interval limit
        assert result is None

    @pytest.mark.asyncio
    async def test_check_and_rebalance_force(self, engine, mock_delta_calculator, mock_strategy, mock_order_manager):
        """Force should bypass checks."""
        # Neutral delta
        mock_delta_calculator.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("5000"),
            total_short=Decimal("5000"),
            net_delta=Decimal("0"),
            delta_percentage=0.0,
            is_neutral=True,
            accounts_long=["acc1"],
            accounts_short=["acc2"],
        )

        # Set up minimal allocations for forced rebalance
        mock_strategy.get_rebalance_allocations.return_value = [
            StrategyAllocation("acc1", "LONG", Decimal("100"), Decimal("1")),
        ]
        mock_result = MagicMock(success=True, error=None)
        mock_order_manager.submit_batch.return_value = [mock_result]

        result = await engine.check_and_rebalance("BTC-USD-PERP", force=True)

        # Should rebalance even though neutral
        assert result is not None

    @pytest.mark.asyncio
    async def test_manual_rebalance(self, engine, mock_delta_calculator, mock_strategy, mock_order_manager):
        """Manual rebalance should use force."""
        mock_strategy.get_rebalance_allocations.return_value = [
            StrategyAllocation("acc1", "LONG", Decimal("100"), Decimal("1")),
        ]
        mock_result = MagicMock(success=True, error=None)
        mock_order_manager.submit_batch.return_value = [mock_result]

        result = await engine.manual_rebalance("BTC-USD-PERP")

        assert result is not None
        assert result.trigger == RebalanceTrigger.MANUAL

    def test_set_threshold(self, engine, mock_delta_calculator):
        """Should update threshold in both engine and calculator."""
        engine.set_threshold(10.0)

        assert engine.threshold_pct == 10.0
        mock_delta_calculator.set_threshold.assert_called_once_with(10.0)

    def test_reset_timers(self, engine):
        """Should clear all rebalance timers."""
        engine._last_rebalance["BTC-USD-PERP"] = time.time()
        engine._last_rebalance["ETH-USD-PERP"] = time.time()

        engine.reset_timers()

        assert len(engine._last_rebalance) == 0

    def test_get_time_until_next_rebalance_immediate(self, engine):
        """Should return 0 when no recent rebalance."""
        remaining = engine.get_time_until_next_rebalance("BTC-USD-PERP")

        assert remaining == 0

    def test_get_time_until_next_rebalance_waiting(self, engine):
        """Should return remaining time when waiting."""
        engine._last_rebalance["BTC-USD-PERP"] = time.time()

        remaining = engine.get_time_until_next_rebalance("BTC-USD-PERP")

        # Should be close to min_rebalance_interval (300s)
        assert 299 < remaining <= 300

    def test_get_rebalance_history(self, engine):
        """Should return recent rebalance history."""
        # Add some history
        for i in range(5):
            engine._rebalance_history.append(
                RebalanceResult(
                    success=True,
                    trigger=RebalanceTrigger.THRESHOLD_EXCEEDED,
                    actions_planned=2,
                    actions_executed=2,
                    delta_before=Decimal("100"),
                )
            )

        history = engine.get_rebalance_history(limit=3)

        assert len(history) == 3

    def test_allocations_to_orders_increase_long(self, engine):
        """Should convert increase long to BUY order."""
        allocations = [
            StrategyAllocation("acc1", "LONG", Decimal("500"), Decimal("1")),
        ]

        orders = engine._allocations_to_orders("BTC-USD-PERP", allocations)

        assert len(orders) == 1
        assert orders[0].account_id == "acc1"
        assert orders[0].side.value == "BUY"
        assert orders[0].size == Decimal("500")

    def test_allocations_to_orders_decrease_long(self, engine):
        """Should convert decrease long to SELL order."""
        allocations = [
            StrategyAllocation("acc1", "LONG", Decimal("-500"), Decimal("1")),
        ]

        orders = engine._allocations_to_orders("BTC-USD-PERP", allocations)

        assert len(orders) == 1
        assert orders[0].side.value == "SELL"
        assert orders[0].size == Decimal("500")  # Absolute value

    def test_allocations_to_orders_increase_short(self, engine):
        """Should convert increase short to SELL order."""
        allocations = [
            StrategyAllocation("acc1", "SHORT", Decimal("500"), Decimal("1")),
        ]

        orders = engine._allocations_to_orders("BTC-USD-PERP", allocations)

        assert len(orders) == 1
        assert orders[0].side.value == "SELL"

    def test_allocations_to_orders_decrease_short(self, engine):
        """Should convert decrease short to BUY order."""
        allocations = [
            StrategyAllocation("acc1", "SHORT", Decimal("-500"), Decimal("1")),
        ]

        orders = engine._allocations_to_orders("BTC-USD-PERP", allocations)

        assert len(orders) == 1
        assert orders[0].side.value == "BUY"

    @pytest.mark.asyncio
    async def test_callbacks_on_rebalance(self, engine, mock_delta_calculator, mock_strategy, mock_order_manager):
        """Should invoke callbacks during rebalance."""
        # Set up imbalanced delta
        mock_delta_calculator.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("6000"),
            total_short=Decimal("4000"),
            net_delta=Decimal("2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1"],
            accounts_short=["acc2"],
        )

        mock_strategy.get_rebalance_allocations.return_value = [
            StrategyAllocation("acc1", "LONG", Decimal("-500"), Decimal("1")),
        ]
        mock_result = MagicMock(success=True, error=None)
        mock_order_manager.submit_batch.return_value = [mock_result]

        # Set up callbacks
        on_start = MagicMock()
        on_complete = MagicMock()
        engine.on_rebalance_start = on_start
        engine.on_rebalance_complete = on_complete

        await engine.check_and_rebalance("BTC-USD-PERP")

        on_start.assert_called_once()
        on_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_callbacks(self, engine, mock_delta_calculator, mock_strategy, mock_order_manager):
        """Should handle async callbacks."""
        mock_delta_calculator.calculate_delta.return_value = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("6000"),
            total_short=Decimal("4000"),
            net_delta=Decimal("2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1"],
            accounts_short=["acc2"],
        )

        mock_strategy.get_rebalance_allocations.return_value = [
            StrategyAllocation("acc1", "LONG", Decimal("-500"), Decimal("1")),
        ]
        mock_result = MagicMock(success=True, error=None)
        mock_order_manager.submit_batch.return_value = [mock_result]

        async_callback = AsyncMock()
        engine.on_rebalance_start = async_callback

        await engine.check_and_rebalance("BTC-USD-PERP")

        async_callback.assert_called_once()
