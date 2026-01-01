"""Base strategy interface for delta-neutral strategies."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from src.core.account_manager import AccountManager
from src.core.delta_calculator import DeltaReport
from src.core.position_manager import PositionManager

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Allocation for a single account in a strategy."""

    account_id: str
    side: str  # "LONG" or "SHORT"
    size: Decimal
    leverage: Decimal = Decimal("1")
    priority: int = 1  # Higher = execute first

    @property
    def is_long(self) -> bool:
        """Check if this is a long allocation."""
        return self.side.upper() == "LONG"

    @property
    def is_short(self) -> bool:
        """Check if this is a short allocation."""
        return self.side.upper() == "SHORT"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "side": self.side,
            "size": str(self.size),
            "leverage": str(self.leverage),
            "priority": self.priority,
        }


class BaseStrategy(ABC):
    """Abstract base class for delta-neutral strategies.

    All strategies must:
    1. Maintain delta neutrality (net exposure = 0)
    2. Distribute positions across accounts
    3. Handle rebalancing when needed

    Subclasses implement specific allocation logic.
    """

    def __init__(
        self,
        account_manager: AccountManager,
        position_manager: PositionManager,
    ):
        """Initialize strategy.

        Args:
            account_manager: Account manager instance
            position_manager: Position manager instance
        """
        self.account_manager = account_manager
        self.position_manager = position_manager
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._name

    @abstractmethod
    def calculate_allocations(
        self,
        market: str,
        total_size: Decimal,
    ) -> List[StrategyAllocation]:
        """Calculate position allocations for all accounts.

        Must return allocations where sum(longs) = sum(shorts).

        Args:
            market: Market symbol to trade
            total_size: Total position size (long + short)

        Returns:
            List of allocations for each account
        """
        pass

    @abstractmethod
    def get_rebalance_allocations(
        self,
        market: str,
        delta_report: DeltaReport,
    ) -> List[StrategyAllocation]:
        """Calculate allocations to restore delta neutrality.

        Called when positions drift from neutral.

        Args:
            market: Market symbol
            delta_report: Current delta exposure report

        Returns:
            List of adjustment allocations
        """
        pass

    def validate_delta_neutrality(
        self,
        allocations: List[StrategyAllocation],
        tolerance: Decimal = Decimal("0.0001"),
    ) -> bool:
        """Verify allocations sum to zero net delta.

        Args:
            allocations: List of allocations to validate
            tolerance: Acceptable deviation from zero

        Returns:
            True if allocations are delta neutral
        """
        total_long = sum(a.size for a in allocations if a.is_long)
        total_short = sum(a.size for a in allocations if a.is_short)

        diff = abs(total_long - total_short)
        is_neutral = diff <= tolerance

        if not is_neutral:
            logger.warning(
                f"Strategy {self.name}: Allocations not neutral! "
                f"Long={total_long}, Short={total_short}, Diff={diff}"
            )

        return is_neutral

    def get_active_accounts(self) -> List:
        """Get list of active accounts for trading.

        Returns:
            List of active AccountState objects
        """
        return self.account_manager.get_active_accounts()

    def get_account_count(self) -> int:
        """Get number of active accounts.

        Returns:
            Number of active accounts
        """
        return len(self.get_active_accounts())

    def validate_account_count(self, min_accounts: int = 2) -> bool:
        """Validate minimum account requirement.

        Args:
            min_accounts: Minimum required accounts

        Returns:
            True if enough accounts
        """
        count = self.get_account_count()
        if count < min_accounts:
            logger.error(
                f"Strategy {self.name} requires at least {min_accounts} accounts, "
                f"but only {count} available"
            )
            return False
        return True

    def split_accounts(
        self,
        n_long: Optional[int] = None,
    ) -> tuple[List, List]:
        """Split accounts into long and short groups.

        Args:
            n_long: Number of accounts for long side (default: half)

        Returns:
            Tuple of (long_accounts, short_accounts)
        """
        accounts = self.get_active_accounts()
        n = len(accounts)

        if n_long is None:
            n_long = n // 2

        # Ensure at least 1 on each side
        n_long = max(1, min(n_long, n - 1))

        return accounts[:n_long], accounts[n_long:]

    def calculate_per_account_size(
        self,
        total_size: Decimal,
        n_long: int,
        n_short: int,
    ) -> tuple[Decimal, Decimal]:
        """Calculate size per account for each side.

        For delta neutrality: total_long = total_short = total_size / 2

        Args:
            total_size: Total desired exposure
            n_long: Number of long accounts
            n_short: Number of short accounts

        Returns:
            Tuple of (size_per_long, size_per_short)
        """
        if n_long == 0 or n_short == 0:
            raise ValueError("Need at least 1 account on each side")

        per_side = total_size / 2
        size_per_long = per_side / Decimal(n_long)
        size_per_short = per_side / Decimal(n_short)

        return size_per_long, size_per_short

    def summary(self) -> dict:
        """Get strategy summary.

        Returns:
            Summary dictionary
        """
        return {
            "name": self.name,
            "active_accounts": self.get_account_count(),
            "description": self.__doc__ or "No description",
        }
