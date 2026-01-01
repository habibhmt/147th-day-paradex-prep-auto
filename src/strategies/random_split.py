"""Random split strategy for pattern obfuscation."""

import logging
import random
from decimal import Decimal
from typing import List, Tuple

from src.core.account_manager import AccountManager, AccountRole
from src.core.delta_calculator import DeltaReport
from src.core.position_manager import PositionManager
from src.strategies.base import BaseStrategy, StrategyAllocation

logger = logging.getLogger(__name__)


class RandomSplitStrategy(BaseStrategy):
    """Random distribution strategy for anti-detection.

    Features:
    - Randomizes which accounts go long vs short
    - Adds variance to position sizes
    - Still maintains delta neutrality (sum long = sum short)

    Useful for:
    - Breaking predictable trading patterns
    - Avoiding detection as coordinated accounts
    - Varying XP distribution across accounts

    Note: Size variance is normalized to ensure delta neutrality.
    """

    def __init__(
        self,
        account_manager: AccountManager,
        position_manager: PositionManager,
        size_variance: float = 0.2,  # +/- 20% variance
        min_long_ratio: float = 0.3,  # At least 30% accounts long
        max_long_ratio: float = 0.7,  # At most 70% accounts long
    ):
        """Initialize random split strategy.

        Args:
            account_manager: Account manager
            position_manager: Position manager
            size_variance: Position size variance (0-0.5)
            min_long_ratio: Minimum ratio of long accounts
            max_long_ratio: Maximum ratio of long accounts
        """
        super().__init__(account_manager, position_manager)
        self.size_variance = max(0, min(0.5, size_variance))
        self.min_long_ratio = max(0.2, min_long_ratio)
        self.max_long_ratio = min(0.8, max_long_ratio)

    def _randomize_split(self, n_accounts: int) -> Tuple[int, int]:
        """Randomly determine long/short split.

        Args:
            n_accounts: Total number of accounts

        Returns:
            Tuple of (n_long, n_short)
        """
        min_long = max(1, int(n_accounts * self.min_long_ratio))
        max_long = min(n_accounts - 1, int(n_accounts * self.max_long_ratio))

        n_long = random.randint(min_long, max_long)
        n_short = n_accounts - n_long

        return n_long, n_short

    def _apply_variance(self, base_size: Decimal) -> Decimal:
        """Apply random variance to size.

        Args:
            base_size: Base position size

        Returns:
            Size with variance applied
        """
        variance = Decimal(
            str(1 + random.uniform(-self.size_variance, self.size_variance))
        )
        return base_size * variance

    def _normalize_sizes(
        self,
        sizes: List[Decimal],
        target_total: Decimal,
    ) -> List[Decimal]:
        """Normalize sizes to sum to target.

        Args:
            sizes: List of sizes to normalize
            target_total: Target sum

        Returns:
            Normalized sizes
        """
        current_total = sum(sizes)
        if current_total == 0:
            return sizes

        factor = target_total / current_total
        return [s * factor for s in sizes]

    def calculate_allocations(
        self,
        market: str,
        total_size: Decimal,
    ) -> List[StrategyAllocation]:
        """Calculate randomized allocations.

        Args:
            market: Market to trade
            total_size: Total position size

        Returns:
            List of randomized allocations
        """
        if not self.validate_account_count(2):
            raise ValueError("Need at least 2 accounts")

        accounts = self.get_active_accounts()
        n = len(accounts)

        # Randomly determine split
        n_long, n_short = self._randomize_split(n)

        # Shuffle accounts
        shuffled = accounts.copy()
        random.shuffle(shuffled)

        # Calculate base size per side
        per_side = total_size / 2
        base_long = per_side / Decimal(n_long)
        base_short = per_side / Decimal(n_short)

        # Generate sizes with variance
        long_sizes = [self._apply_variance(base_long) for _ in range(n_long)]
        short_sizes = [self._apply_variance(base_short) for _ in range(n_short)]

        # Normalize to maintain delta neutrality
        long_sizes = self._normalize_sizes(long_sizes, per_side)
        short_sizes = self._normalize_sizes(short_sizes, per_side)

        allocations = []

        # Assign long positions
        for i in range(n_long):
            account = shuffled[i]
            allocations.append(
                StrategyAllocation(
                    account_id=account.account_id,
                    side="LONG",
                    size=long_sizes[i],
                    leverage=Decimal(str(account.leverage)),
                )
            )
            self.account_manager.assign_role(account.account_id, AccountRole.LONG)

        # Assign short positions
        for i in range(n_short):
            account = shuffled[n_long + i]
            allocations.append(
                StrategyAllocation(
                    account_id=account.account_id,
                    side="SHORT",
                    size=short_sizes[i],
                    leverage=Decimal(str(account.leverage)),
                )
            )
            self.account_manager.assign_role(account.account_id, AccountRole.SHORT)

        # Validate
        if not self.validate_delta_neutrality(allocations):
            logger.error("Random allocations failed neutrality check!")

        logger.info(
            f"Random Strategy: {n_long} longs, {n_short} shorts "
            f"(variance={self.size_variance})"
        )

        return allocations

    def get_rebalance_allocations(
        self,
        market: str,
        delta_report: DeltaReport,
    ) -> List[StrategyAllocation]:
        """Calculate randomized rebalancing.

        Args:
            market: Market symbol
            delta_report: Current delta report

        Returns:
            Adjustment allocations
        """
        if delta_report.is_neutral:
            return []

        net_delta = delta_report.net_delta
        allocations = []

        if net_delta > 0:
            # Too much long - randomly select accounts to reduce
            accounts = delta_report.accounts_long.copy()
            random.shuffle(accounts)

            # Distribute adjustment with variance
            n_accounts = len(accounts)
            base_adjustment = net_delta / Decimal(n_accounts)
            adjustments = [
                self._apply_variance(base_adjustment) for _ in range(n_accounts)
            ]
            adjustments = self._normalize_sizes(adjustments, net_delta)

            for account_id, adj in zip(accounts, adjustments):
                allocations.append(
                    StrategyAllocation(
                        account_id=account_id,
                        side="LONG",
                        size=-adj,  # Negative = reduce
                    )
                )
        else:
            # Too much short
            accounts = delta_report.accounts_short.copy()
            random.shuffle(accounts)

            n_accounts = len(accounts)
            abs_delta = abs(net_delta)
            base_adjustment = abs_delta / Decimal(n_accounts)
            adjustments = [
                self._apply_variance(base_adjustment) for _ in range(n_accounts)
            ]
            adjustments = self._normalize_sizes(adjustments, abs_delta)

            for account_id, adj in zip(accounts, adjustments):
                allocations.append(
                    StrategyAllocation(
                        account_id=account_id,
                        side="SHORT",
                        size=-adj,
                    )
                )

        logger.info(
            f"Random Rebalance: Delta {net_delta}, "
            f"adjusting {len(allocations)} positions randomly"
        )

        return allocations

    def reshuffle_sides(
        self,
        market: str,
    ) -> List[StrategyAllocation]:
        """Generate new random side assignments.

        Useful for periodically reshuffling to break patterns.
        Returns allocations that would close existing and open new positions.

        Args:
            market: Market symbol

        Returns:
            Allocations to reshuffle positions
        """
        # Get current positions
        positions = self.position_manager.get_all_positions(market)
        if not positions:
            return []

        # Calculate total exposure
        total_long = sum(p.size for p in positions if p.side == "LONG")
        total_short = sum(p.size for p in positions if p.side == "SHORT")
        total_size = total_long + total_short

        # Generate new allocations
        return self.calculate_allocations(market, total_size)

    def summary(self) -> dict:
        """Get strategy summary."""
        base = super().summary()
        base.update(
            {
                "size_variance": self.size_variance,
                "min_long_ratio": self.min_long_ratio,
                "max_long_ratio": self.max_long_ratio,
            }
        )
        return base
