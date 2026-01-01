"""Funding rate based delta-neutral strategy."""

import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from src.core.account_manager import AccountManager, AccountRole
from src.core.delta_calculator import DeltaReport
from src.core.position_manager import PositionManager
from src.strategies.base import BaseStrategy, StrategyAllocation

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    """Funding rate data for a market."""

    market: str
    rate: Decimal  # Positive = longs pay shorts
    timestamp: float = 0.0
    next_funding_time: float = 0.0

    @property
    def is_positive(self) -> bool:
        """Check if funding is positive (longs pay shorts)."""
        return self.rate > 0

    @property
    def is_negative(self) -> bool:
        """Check if funding is negative (shorts pay longs)."""
        return self.rate < 0

    @property
    def profitable_side(self) -> str:
        """Get the side that receives funding."""
        return "SHORT" if self.is_positive else "LONG"


class FundingBasedStrategy(BaseStrategy):
    """Funding rate based strategy.

    Biases position distribution based on funding rate:
    - Positive funding: More accounts short (receive funding)
    - Negative funding: More accounts long (receive funding)

    Still maintains delta neutrality, but with unequal account distribution.
    Can generate additional profit from funding payments.

    Best for:
    - Markets with consistent funding bias
    - Maximizing returns beyond just XP
    - Longer position durations
    """

    def __init__(
        self,
        account_manager: AccountManager,
        position_manager: PositionManager,
        funding_bias: float = 0.6,  # 60/40 split toward profitable side
        min_funding_threshold: Decimal = Decimal("0.0001"),  # 0.01%
    ):
        """Initialize funding-based strategy.

        Args:
            account_manager: Account manager
            position_manager: Position manager
            funding_bias: Ratio of accounts on profitable side (0.5-0.8)
            min_funding_threshold: Minimum funding rate to trigger bias
        """
        super().__init__(account_manager, position_manager)
        self.funding_bias = max(0.5, min(0.8, funding_bias))  # Clamp to valid range
        self.min_funding_threshold = min_funding_threshold
        self._funding_rates: Dict[str, FundingRate] = {}

    def update_funding_rate(self, funding: FundingRate) -> None:
        """Update funding rate for market.

        Args:
            funding: New funding rate data
        """
        self._funding_rates[funding.market] = funding
        logger.debug(f"Updated funding rate for {funding.market}: {funding.rate}")

    def get_funding_rate(self, market: str) -> Optional[FundingRate]:
        """Get current funding rate for market.

        Args:
            market: Market symbol

        Returns:
            FundingRate if available
        """
        return self._funding_rates.get(market)

    def calculate_allocations(
        self,
        market: str,
        total_size: Decimal,
    ) -> List[StrategyAllocation]:
        """Calculate funding-biased allocations.

        Args:
            market: Market to trade
            total_size: Total position size

        Returns:
            List of allocations
        """
        if not self.validate_account_count(2):
            raise ValueError("Need at least 2 accounts")

        funding = self._funding_rates.get(market)

        # Fall back to 50/50 if no funding data or insignificant
        if funding is None or abs(funding.rate) < self.min_funding_threshold:
            logger.info(f"No significant funding for {market}, using 50/50")
            from src.strategies.simple_5050 import Simple5050Strategy

            fallback = Simple5050Strategy(self.account_manager, self.position_manager)
            return fallback.calculate_allocations(market, total_size)

        accounts = self.get_active_accounts()
        n = len(accounts)

        # Determine split based on funding
        profitable_side = funding.profitable_side
        unprofitable_side = "LONG" if profitable_side == "SHORT" else "SHORT"

        # Calculate account counts with bias
        n_profitable = int(n * self.funding_bias)
        n_unprofitable = n - n_profitable

        # Ensure at least 1 on each side
        n_profitable = max(1, min(n_profitable, n - 1))
        n_unprofitable = n - n_profitable

        # Calculate sizes (must balance for delta neutrality)
        per_side = total_size / 2
        size_per_profitable = per_side / Decimal(n_profitable)
        size_per_unprofitable = per_side / Decimal(n_unprofitable)

        # Shuffle accounts for randomness
        shuffled = accounts.copy()
        random.shuffle(shuffled)

        allocations = []

        for i, account in enumerate(shuffled):
            if i < n_profitable:
                allocations.append(
                    StrategyAllocation(
                        account_id=account.account_id,
                        side=profitable_side,
                        size=size_per_profitable,
                        leverage=Decimal(str(account.leverage)),
                    )
                )
                role = (
                    AccountRole.SHORT
                    if profitable_side == "SHORT"
                    else AccountRole.LONG
                )
                self.account_manager.assign_role(account.account_id, role)
            else:
                allocations.append(
                    StrategyAllocation(
                        account_id=account.account_id,
                        side=unprofitable_side,
                        size=size_per_unprofitable,
                        leverage=Decimal(str(account.leverage)),
                    )
                )
                role = (
                    AccountRole.LONG
                    if profitable_side == "SHORT"
                    else AccountRole.SHORT
                )
                self.account_manager.assign_role(account.account_id, role)

        # Validate
        if not self.validate_delta_neutrality(allocations):
            logger.error("Funding allocations failed neutrality check!")

        logger.info(
            f"Funding Strategy: {n_profitable} {profitable_side}s (profitable), "
            f"{n_unprofitable} {unprofitable_side}s, rate={funding.rate}"
        )

        return allocations

    def get_rebalance_allocations(
        self,
        market: str,
        delta_report: DeltaReport,
    ) -> List[StrategyAllocation]:
        """Calculate rebalancing with funding awareness.

        Args:
            market: Market symbol
            delta_report: Current delta report

        Returns:
            Adjustment allocations
        """
        if delta_report.is_neutral:
            return []

        funding = self._funding_rates.get(market)
        net_delta = delta_report.net_delta
        allocations = []

        # Prefer adjusting the unprofitable side
        if net_delta > 0:
            # Too much long
            if funding and funding.is_positive:
                # Longs are unprofitable, reduce them
                adjustment_per = net_delta / Decimal(len(delta_report.accounts_long))
                for account_id in delta_report.accounts_long:
                    allocations.append(
                        StrategyAllocation(
                            account_id=account_id,
                            side="LONG",
                            size=-adjustment_per,
                        )
                    )
            else:
                # Shorts are unprofitable, increase them
                adjustment_per = net_delta / Decimal(len(delta_report.accounts_short))
                for account_id in delta_report.accounts_short:
                    allocations.append(
                        StrategyAllocation(
                            account_id=account_id,
                            side="SHORT",
                            size=adjustment_per,
                        )
                    )
        else:
            # Too much short
            abs_delta = abs(net_delta)
            if funding and funding.is_negative:
                # Shorts are unprofitable, reduce them
                adjustment_per = abs_delta / Decimal(len(delta_report.accounts_short))
                for account_id in delta_report.accounts_short:
                    allocations.append(
                        StrategyAllocation(
                            account_id=account_id,
                            side="SHORT",
                            size=-adjustment_per,
                        )
                    )
            else:
                # Longs are unprofitable, increase them
                adjustment_per = abs_delta / Decimal(len(delta_report.accounts_long))
                for account_id in delta_report.accounts_long:
                    allocations.append(
                        StrategyAllocation(
                            account_id=account_id,
                            side="LONG",
                            size=adjustment_per,
                        )
                    )

        logger.info(
            f"Funding Rebalance: Delta {net_delta}, "
            f"funding rate {funding.rate if funding else 'N/A'}"
        )

        return allocations

    def estimate_funding_profit(
        self,
        market: str,
        hours: float = 24,
    ) -> Optional[Decimal]:
        """Estimate funding profit over time period.

        Args:
            market: Market symbol
            hours: Time period in hours

        Returns:
            Estimated profit or None
        """
        funding = self._funding_rates.get(market)
        if not funding:
            return None

        positions = self.position_manager.get_all_positions(market)
        if not positions:
            return None

        # Funding typically settles every 8 hours
        funding_periods = Decimal(str(hours / 8))
        total_profit = Decimal("0")

        for pos in positions:
            # Funding paid/received per period
            funding_amount = pos.notional_value * abs(funding.rate)

            if (pos.side == "LONG" and funding.is_positive) or (
                pos.side == "SHORT" and funding.is_negative
            ):
                # Paying funding
                total_profit -= funding_amount * funding_periods
            else:
                # Receiving funding
                total_profit += funding_amount * funding_periods

        return total_profit

    def summary(self) -> dict:
        """Get strategy summary."""
        base = super().summary()
        base.update(
            {
                "funding_bias": self.funding_bias,
                "min_threshold": str(self.min_funding_threshold),
                "tracked_markets": list(self._funding_rates.keys()),
            }
        )
        return base
