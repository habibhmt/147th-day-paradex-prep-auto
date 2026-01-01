"""Delta exposure calculation and monitoring."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from src.core.position_manager import PositionManager

logger = logging.getLogger(__name__)


@dataclass
class DeltaReport:
    """Report of delta exposure for a market."""

    market: str
    total_long: Decimal
    total_short: Decimal
    net_delta: Decimal
    delta_percentage: float
    is_neutral: bool
    accounts_long: List[str]
    accounts_short: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "total_long": str(self.total_long),
            "total_short": str(self.total_short),
            "net_delta": str(self.net_delta),
            "delta_percentage": self.delta_percentage,
            "is_neutral": self.is_neutral,
            "accounts_long": self.accounts_long,
            "accounts_short": self.accounts_short,
        }


@dataclass
class DeltaCalculator:
    """Calculates and monitors delta exposure.

    Delta-neutral means net exposure = 0:
    - Sum of long positions = Sum of short positions
    - Delta % = |Net Delta| / Gross Exposure * 100

    Target: Delta % < threshold (default 5%)
    """

    position_manager: PositionManager
    neutrality_threshold: float = 5.0  # 5% deviation allowed

    def calculate_delta(self, market: str) -> DeltaReport:
        """Calculate current delta exposure for a market.

        Args:
            market: Market symbol (e.g., BTC-USD-PERP)

        Returns:
            DeltaReport with exposure details
        """
        positions = self.position_manager.get_all_positions(market)

        total_long = Decimal("0")
        total_short = Decimal("0")
        accounts_long = []
        accounts_short = []

        for pos in positions:
            if pos.side == "LONG":
                total_long += pos.size
                accounts_long.append(pos.account_id)
            else:  # SHORT
                total_short += pos.size
                accounts_short.append(pos.account_id)

        net_delta = total_long - total_short
        gross = total_long + total_short

        # Calculate delta percentage
        if gross > 0:
            delta_pct = float(abs(net_delta) / gross * 100)
        else:
            delta_pct = 0.0

        is_neutral = delta_pct <= self.neutrality_threshold

        return DeltaReport(
            market=market,
            total_long=total_long,
            total_short=total_short,
            net_delta=net_delta,
            delta_percentage=delta_pct,
            is_neutral=is_neutral,
            accounts_long=accounts_long,
            accounts_short=accounts_short,
        )

    def needs_rebalancing(self, market: str) -> bool:
        """Check if rebalancing is needed for market.

        Args:
            market: Market symbol

        Returns:
            True if delta exceeds threshold
        """
        report = self.calculate_delta(market)
        return not report.is_neutral

    def get_imbalance(self, market: str) -> Decimal:
        """Get the imbalance amount that needs correction.

        Args:
            market: Market symbol

        Returns:
            Imbalance amount (positive = too much long, negative = too much short)
        """
        report = self.calculate_delta(market)
        return report.net_delta

    def calculate_rebalance_size(self, market: str) -> Decimal:
        """Calculate size needed to rebalance.

        To achieve delta neutrality:
        - If net_delta > 0: Need to reduce longs OR increase shorts
        - If net_delta < 0: Need to reduce shorts OR increase longs

        The rebalance size is |net_delta| / 2 (split between both sides).

        Args:
            market: Market symbol

        Returns:
            Size to adjust on each side
        """
        report = self.calculate_delta(market)
        # Each side needs to move by half the imbalance
        return abs(report.net_delta) / 2

    def get_rebalance_suggestion(self, market: str) -> Optional[dict]:
        """Get specific rebalance suggestion.

        Args:
            market: Market symbol

        Returns:
            Suggestion dict or None if balanced
        """
        report = self.calculate_delta(market)

        if report.is_neutral:
            return None

        rebalance_size = abs(report.net_delta) / 2

        if report.net_delta > 0:
            # Too much long exposure
            return {
                "action": "reduce_long_or_increase_short",
                "net_delta": str(report.net_delta),
                "rebalance_size": str(rebalance_size),
                "suggestion": f"Close {rebalance_size} longs OR open {rebalance_size} shorts",
                "accounts_long": report.accounts_long,
                "accounts_short": report.accounts_short,
            }
        else:
            # Too much short exposure
            return {
                "action": "reduce_short_or_increase_long",
                "net_delta": str(report.net_delta),
                "rebalance_size": str(rebalance_size),
                "suggestion": f"Close {rebalance_size} shorts OR open {rebalance_size} longs",
                "accounts_long": report.accounts_long,
                "accounts_short": report.accounts_short,
            }

    def calculate_target_sizes(
        self,
        market: str,
        total_size: Decimal,
        n_long: int,
        n_short: int,
    ) -> dict:
        """Calculate target sizes to achieve delta neutrality.

        For delta neutrality: total_long = total_short = total_size / 2

        Args:
            market: Market symbol
            total_size: Total desired exposure (long + short)
            n_long: Number of long accounts
            n_short: Number of short accounts

        Returns:
            Dict with per-account target sizes
        """
        if n_long == 0 or n_short == 0:
            raise ValueError("Need at least 1 account on each side")

        size_per_side = total_size / 2
        size_per_long = size_per_side / Decimal(n_long)
        size_per_short = size_per_side / Decimal(n_short)

        return {
            "total_size": str(total_size),
            "per_side": str(size_per_side),
            "per_long_account": str(size_per_long),
            "per_short_account": str(size_per_short),
            "n_long": n_long,
            "n_short": n_short,
            "expected_delta": "0",
        }

    def validate_allocations(
        self,
        allocations: List[dict],
    ) -> bool:
        """Validate that allocations result in delta neutrality.

        Args:
            allocations: List of {side: str, size: Decimal}

        Returns:
            True if allocations are delta neutral
        """
        total_long = Decimal("0")
        total_short = Decimal("0")

        for alloc in allocations:
            size = Decimal(str(alloc.get("size", 0)))
            side = alloc.get("side", "").upper()

            if side == "LONG":
                total_long += size
            elif side == "SHORT":
                total_short += size

        # Check if balanced (within small tolerance)
        diff = abs(total_long - total_short)
        tolerance = Decimal("0.0001")

        return diff <= tolerance

    def get_all_market_reports(self) -> List[DeltaReport]:
        """Get delta reports for all markets with positions.

        Returns:
            List of DeltaReports
        """
        markets = self.position_manager.get_markets()
        return [self.calculate_delta(market) for market in markets]

    def summary(self) -> dict:
        """Get summary of delta status across all markets.

        Returns:
            Summary dict
        """
        reports = self.get_all_market_reports()

        all_neutral = all(r.is_neutral for r in reports)
        markets_needing_rebalance = [r.market for r in reports if not r.is_neutral]

        return {
            "all_neutral": all_neutral,
            "total_markets": len(reports),
            "neutral_markets": sum(1 for r in reports if r.is_neutral),
            "markets_needing_rebalance": markets_needing_rebalance,
            "reports": [r.to_dict() for r in reports],
        }

    def set_threshold(self, threshold: float) -> None:
        """Update neutrality threshold.

        Args:
            threshold: New threshold percentage
        """
        self.neutrality_threshold = threshold
        logger.info(f"Delta threshold set to {threshold}%")
