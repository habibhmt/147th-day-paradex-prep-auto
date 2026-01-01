"""Unit tests for trading strategies."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.strategies.base import BaseStrategy, StrategyAllocation
from src.strategies.simple_5050 import Simple5050Strategy
from src.strategies.funding_based import FundingBasedStrategy
from src.strategies.random_split import RandomSplitStrategy
from src.core.delta_calculator import DeltaReport


@dataclass
class MockAccount:
    """Mock account for testing."""
    account_id: str
    leverage: float = 1.0
    role: str = "NEUTRAL"


class TestStrategyAllocation:
    """Tests for StrategyAllocation dataclass."""

    def test_create_allocation(self):
        """Should create allocation correctly."""
        alloc = StrategyAllocation(
            account_id="acc1",
            side="LONG",
            size=Decimal("1000"),
            leverage=Decimal("2"),
        )

        assert alloc.account_id == "acc1"
        assert alloc.side == "LONG"
        assert alloc.size == Decimal("1000")
        assert alloc.leverage == Decimal("2")


class TestSimple5050Strategy:
    """Tests for Simple5050Strategy."""

    @pytest.fixture
    def mock_account_manager(self):
        """Create mock account manager."""
        manager = MagicMock()
        manager.get_active_accounts.return_value = [
            MockAccount("acc1"),
            MockAccount("acc2"),
            MockAccount("acc3"),
            MockAccount("acc4"),
        ]
        manager.active_count = 4
        return manager

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager."""
        return MagicMock()

    @pytest.fixture
    def strategy(self, mock_account_manager, mock_position_manager):
        """Create 50/50 strategy."""
        return Simple5050Strategy(mock_account_manager, mock_position_manager)

    def test_calculate_allocations_even_split(self, strategy):
        """Should split evenly with even number of accounts."""
        allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))

        long_allocs = [a for a in allocations if a.side == "LONG"]
        short_allocs = [a for a in allocations if a.side == "SHORT"]

        assert len(long_allocs) == 2
        assert len(short_allocs) == 2

    def test_calculate_allocations_delta_neutral(self, strategy):
        """Should produce delta-neutral allocations."""
        allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))

        total_long = sum(a.size for a in allocations if a.side == "LONG")
        total_short = sum(a.size for a in allocations if a.side == "SHORT")

        assert total_long == total_short

    def test_calculate_allocations_total_size(self, strategy):
        """Total allocation should match requested size."""
        total_size = Decimal("10000")
        allocations = strategy.calculate_allocations("BTC-USD-PERP", total_size)

        total_long = sum(a.size for a in allocations if a.side == "LONG")
        total_short = sum(a.size for a in allocations if a.side == "SHORT")

        # Total should be split evenly
        assert total_long == total_size / 2
        assert total_short == total_size / 2

    def test_rebalance_allocations_neutral_returns_empty(self, strategy):
        """Should return empty list when already neutral."""
        report = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("5000"),
            total_short=Decimal("5000"),
            net_delta=Decimal("0"),
            delta_percentage=0.0,
            is_neutral=True,
            accounts_long=["acc1", "acc2"],
            accounts_short=["acc3", "acc4"],
        )

        allocations = strategy.get_rebalance_allocations("BTC-USD-PERP", report)

        assert allocations == []

    def test_rebalance_allocations_long_heavy(self, strategy):
        """Should reduce longs when long-heavy."""
        report = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("6000"),
            total_short=Decimal("4000"),
            net_delta=Decimal("2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1", "acc2"],
            accounts_short=["acc3", "acc4"],
        )

        allocations = strategy.get_rebalance_allocations("BTC-USD-PERP", report)

        # Should reduce long positions
        assert len(allocations) == 2
        for alloc in allocations:
            assert alloc.side == "LONG"
            assert alloc.size < 0  # Negative = reduce

    def test_rebalance_allocations_short_heavy(self, strategy):
        """Should reduce shorts when short-heavy."""
        report = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("4000"),
            total_short=Decimal("6000"),
            net_delta=Decimal("-2000"),
            delta_percentage=20.0,
            is_neutral=False,
            accounts_long=["acc1", "acc2"],
            accounts_short=["acc3", "acc4"],
        )

        allocations = strategy.get_rebalance_allocations("BTC-USD-PERP", report)

        # Should reduce short positions
        assert len(allocations) == 2
        for alloc in allocations:
            assert alloc.side == "SHORT"
            assert alloc.size < 0  # Negative = reduce

    def test_requires_minimum_accounts(self, mock_account_manager, mock_position_manager):
        """Should require at least 2 accounts."""
        mock_account_manager.get_active_accounts.return_value = [MockAccount("acc1")]
        mock_account_manager.active_count = 1
        strategy = Simple5050Strategy(mock_account_manager, mock_position_manager)

        with pytest.raises(ValueError, match="at least 2"):
            strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))


class TestFundingBasedStrategy:
    """Tests for FundingBasedStrategy."""

    @pytest.fixture
    def mock_account_manager(self):
        """Create mock account manager."""
        manager = MagicMock()
        manager.get_active_accounts.return_value = [
            MockAccount("acc1"),
            MockAccount("acc2"),
            MockAccount("acc3"),
            MockAccount("acc4"),
        ]
        manager.active_count = 4
        return manager

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager."""
        return MagicMock()

    @pytest.fixture
    def strategy(self, mock_account_manager, mock_position_manager):
        """Create funding-based strategy."""
        return FundingBasedStrategy(mock_account_manager, mock_position_manager)

    def test_calculate_allocations_delta_neutral(self, strategy):
        """Allocations should be delta neutral."""
        allocations = strategy.calculate_allocations(
            "BTC-USD-PERP",
            Decimal("10000"),
        )

        long_size = sum(a.size for a in allocations if a.side == "LONG")
        short_size = sum(a.size for a in allocations if a.side == "SHORT")

        # Should be delta neutral
        assert long_size == short_size

    def test_calculate_allocations_all_accounts_used(self, strategy):
        """All accounts should get allocations."""
        allocations = strategy.calculate_allocations(
            "BTC-USD-PERP",
            Decimal("10000"),
        )

        assert len(allocations) == 4

    def test_calculate_allocations_correct_total(self, strategy):
        """Total size should match requested."""
        total_size = Decimal("10000")
        allocations = strategy.calculate_allocations(
            "BTC-USD-PERP",
            total_size,
        )

        total = sum(a.size for a in allocations)
        assert total == total_size


class TestRandomSplitStrategy:
    """Tests for RandomSplitStrategy."""

    @pytest.fixture
    def mock_account_manager(self):
        """Create mock account manager."""
        manager = MagicMock()
        manager.get_active_accounts.return_value = [
            MockAccount("acc1"),
            MockAccount("acc2"),
            MockAccount("acc3"),
            MockAccount("acc4"),
        ]
        manager.active_count = 4
        return manager

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager."""
        return MagicMock()

    @pytest.fixture
    def strategy(self, mock_account_manager, mock_position_manager):
        """Create random split strategy."""
        return RandomSplitStrategy(
            mock_account_manager,
            mock_position_manager,
            size_variance=0.15,
        )

    def test_calculate_allocations_delta_neutral(self, strategy):
        """Random split should still be delta neutral."""
        allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))

        total_long = sum(a.size for a in allocations if a.side == "LONG")
        total_short = sum(a.size for a in allocations if a.side == "SHORT")

        # Should be equal (delta neutral) - allow tiny precision difference
        assert abs(total_long - total_short) < Decimal("0.001")

    def test_calculate_allocations_applies_variance(self, strategy):
        """Should apply size variance to individual allocations."""
        # Run multiple times to check variance
        all_sizes = []
        for _ in range(10):
            allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))
            sizes = [a.size for a in allocations]
            all_sizes.extend(sizes)

        # With variance, sizes should not all be identical
        unique_sizes = set(all_sizes)
        assert len(unique_sizes) > 1

    def test_calculate_allocations_all_accounts_used(self, strategy):
        """All accounts should get allocations."""
        allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))

        assert len(allocations) == 4

    def test_calculate_allocations_has_both_sides(self, strategy):
        """Should have both long and short allocations."""
        allocations = strategy.calculate_allocations("BTC-USD-PERP", Decimal("10000"))

        longs = [a for a in allocations if a.side == "LONG"]
        shorts = [a for a in allocations if a.side == "SHORT"]

        assert len(longs) >= 1
        assert len(shorts) >= 1
