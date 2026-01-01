"""Simple 50/50 delta-neutral strategy."""

import logging
from decimal import Decimal
from typing import List

from src.core.account_manager import AccountManager, AccountRole
from src.core.delta_calculator import DeltaReport
from src.core.position_manager import PositionManager
from src.strategies.base import BaseStrategy, StrategyAllocation

logger = logging.getLogger(__name__)


class Simple5050Strategy(BaseStrategy):
    """Simple 50/50 split strategy.

    Divides accounts into two equal groups:
    - First half: Long positions
    - Second half: Short positions

    Each account gets equal size within its group.
    Total long = Total short (delta neutral).

    Best for:
    - Simple, predictable behavior
    - Equal XP distribution across accounts
    - When funding rate is near zero
    """

    def __init__(
        self,
        account_manager: AccountManager,
        position_manager: PositionManager,
    ):
        """Initialize 50/50 strategy."""
        super().__init__(account_manager, position_manager)

    def calculate_allocations(
        self,
        market: str,
        total_size: Decimal,
    ) -> List[StrategyAllocation]:
        """Calculate 50/50 split allocations.

        Args:
            market: Market to trade
            total_size: Total position size (long + short)

        Returns:
            List of allocations
        """
        if not self.validate_account_count(2):
            raise ValueError("Need at least 2 accounts for 50/50 strategy")

        accounts = self.get_active_accounts()
        n = len(accounts)

        # Split accounts
        n_long = n // 2
        n_short = n - n_long

        # Calculate sizes
        size_per_long, size_per_short = self.calculate_per_account_size(
            total_size, n_long, n_short
        )

        allocations = []

        for i, account in enumerate(accounts):
            if i < n_long:
                # Long accounts
                allocations.append(
                    StrategyAllocation(
                        account_id=account.account_id,
                        side="LONG",
                        size=size_per_long,
                        leverage=Decimal(str(account.leverage)),
                    )
                )
                # Update account role
                self.account_manager.assign_role(account.account_id, AccountRole.LONG)
            else:
                # Short accounts
                allocations.append(
                    StrategyAllocation(
                        account_id=account.account_id,
                        side="SHORT",
                        size=size_per_short,
                        leverage=Decimal(str(account.leverage)),
                    )
                )
                self.account_manager.assign_role(account.account_id, AccountRole.SHORT)

        # Validate
        if not self.validate_delta_neutrality(allocations):
            logger.error("50/50 allocations failed neutrality check!")

        logger.info(
            f"50/50 Strategy: {n_long} longs @ {size_per_long}, "
            f"{n_short} shorts @ {size_per_short}"
        )

        return allocations

    def get_rebalance_allocations(
        self,
        market: str,
        delta_report: DeltaReport,
    ) -> List[StrategyAllocation]:
        """Calculate rebalancing allocations.

        Strategy: Adjust sizes proportionally to restore balance.

        Args:
            market: Market symbol
            delta_report: Current delta report

        Returns:
            Adjustment allocations
        """
        if delta_report.is_neutral:
            return []

        allocations = []
        net_delta = delta_report.net_delta

        if net_delta > 0:
            # Too much long - reduce longs
            adjustment_per_account = net_delta / Decimal(
                len(delta_report.accounts_long)
            )
            for account_id in delta_report.accounts_long:
                allocations.append(
                    StrategyAllocation(
                        account_id=account_id,
                        side="LONG",
                        size=-adjustment_per_account,  # Negative = reduce
                    )
                )
        else:
            # Too much short - reduce shorts
            adjustment_per_account = abs(net_delta) / Decimal(
                len(delta_report.accounts_short)
            )
            for account_id in delta_report.accounts_short:
                allocations.append(
                    StrategyAllocation(
                        account_id=account_id,
                        side="SHORT",
                        size=-adjustment_per_account,  # Negative = reduce
                    )
                )

        logger.info(
            f"50/50 Rebalance: Net delta {net_delta}, "
            f"adjusting {len(allocations)} positions"
        )

        return allocations

    def get_new_position_allocations(
        self,
        market: str,
        position_size: Decimal,
    ) -> List[StrategyAllocation]:
        """Get allocations for opening new positions.

        Opens matched long/short positions to maintain neutrality.

        Args:
            market: Market symbol
            position_size: Size for the new position pair

        Returns:
            Allocations for one long and one short position
        """
        accounts = self.get_active_accounts()
        if len(accounts) < 2:
            raise ValueError("Need at least 2 accounts")

        # Find accounts without positions in this market
        available_long = []
        available_short = []

        for account in accounts:
            pos = self.position_manager.get_position(account.account_id, market)
            if pos is None:
                if account.role == AccountRole.LONG:
                    available_long.append(account)
                elif account.role == AccountRole.SHORT:
                    available_short.append(account)
                else:
                    # Neutral - assign based on count
                    if len(available_long) <= len(available_short):
                        available_long.append(account)
                    else:
                        available_short.append(account)

        if not available_long or not available_short:
            logger.warning("No available accounts for new positions")
            return []

        # Use first available from each side
        return [
            StrategyAllocation(
                account_id=available_long[0].account_id,
                side="LONG",
                size=position_size,
            ),
            StrategyAllocation(
                account_id=available_short[0].account_id,
                side="SHORT",
                size=position_size,
            ),
        ]
