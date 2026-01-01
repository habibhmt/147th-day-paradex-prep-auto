"""XP optimization for airdrop farming."""

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional

from src.core.position_manager import PositionManager
from src.xp.volume_tracker import VolumeTracker

logger = logging.getLogger(__name__)


@dataclass
class XPRecommendation:
    """Recommendation for XP optimization."""

    action: str
    reason: str
    priority: str  # "high", "medium", "low"
    details: Dict


class XPOptimizer:
    """Optimizes trading behavior for maximum XP.

    Key XP factors based on Paradex docs:
    - Trading volume (primary)
    - Position duration (24-48h optimal for bonus)
    - Market making (tight spreads)
    - Vault deposits

    Weekly distribution: 4,000,000 XP every Friday
    """

    def __init__(
        self,
        position_manager: PositionManager,
        volume_tracker: VolumeTracker,
        min_position_duration: float = 24.0,  # hours
        optimal_position_duration: float = 48.0,  # hours
        target_daily_volume: float = 100000.0,  # USD
    ):
        """Initialize XP optimizer.

        Args:
            position_manager: Position manager
            volume_tracker: Volume tracker
            min_position_duration: Minimum position hold time
            optimal_position_duration: Optimal hold time for max XP
            target_daily_volume: Daily volume target
        """
        self.position_manager = position_manager
        self.volume_tracker = volume_tracker
        self.min_duration = min_position_duration
        self.optimal_duration = optimal_position_duration
        self.target_volume = Decimal(str(target_daily_volume))

    def should_close_position(
        self,
        account_id: str,
        market: str,
    ) -> tuple[bool, str]:
        """Determine if position should be closed.

        Considers position duration for XP optimization.

        Args:
            account_id: Account ID
            market: Market symbol

        Returns:
            Tuple of (should_close, reason)
        """
        position = self.position_manager.get_position(account_id, market)
        if position is None:
            return False, "No position"

        hours_held = position.duration_hours

        if hours_held < self.min_duration:
            return False, f"Hold for XP (only {hours_held:.1f}h, need {self.min_duration}h)"

        if hours_held >= self.optimal_duration:
            return True, f"Optimal duration reached ({hours_held:.1f}h)"

        # Between min and optimal - depends on other factors
        return False, f"In optimal range ({hours_held:.1f}h)"

    def get_positions_to_close(
        self,
        market: Optional[str] = None,
    ) -> List[Dict]:
        """Get positions that should be closed for XP optimization.

        Args:
            market: Optional market filter

        Returns:
            List of positions to close
        """
        to_close = []

        for position in self.position_manager.get_all_positions(market):
            should_close, reason = self.should_close_position(
                position.account_id,
                position.market,
            )
            if should_close:
                to_close.append({
                    "account_id": position.account_id,
                    "market": position.market,
                    "size": str(position.size),
                    "duration_hours": position.duration_hours,
                    "reason": reason,
                })

        return to_close

    def get_volume_progress(self) -> Dict:
        """Get progress toward volume target.

        Returns:
            Progress dictionary
        """
        volume_24h = self.volume_tracker.get_total_volume_24h()
        progress = float(volume_24h / self.target_volume * 100) if self.target_volume > 0 else 0

        return {
            "volume_24h": str(volume_24h),
            "target": str(self.target_volume),
            "progress_pct": min(progress, 100),
            "remaining": str(max(Decimal("0"), self.target_volume - volume_24h)),
            "on_track": progress >= 100,
        }

    def get_recommendations(self) -> List[XPRecommendation]:
        """Get XP optimization recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check volume progress
        progress = self.get_volume_progress()
        if not progress["on_track"]:
            recommendations.append(
                XPRecommendation(
                    action="increase_volume",
                    reason=f"Only {progress['progress_pct']:.1f}% of daily volume target",
                    priority="high",
                    details={
                        "current": progress["volume_24h"],
                        "target": progress["target"],
                        "remaining": progress["remaining"],
                    },
                )
            )

        # Check position durations
        positions_to_close = self.get_positions_to_close()
        if positions_to_close:
            recommendations.append(
                XPRecommendation(
                    action="rotate_positions",
                    reason=f"{len(positions_to_close)} positions ready for rotation",
                    priority="medium",
                    details={"positions": positions_to_close},
                )
            )

        # Check for positions held too short
        short_positions = []
        for pos in self.position_manager.get_all_positions():
            if 0 < pos.duration_hours < self.min_duration:
                short_positions.append({
                    "account_id": pos.account_id,
                    "market": pos.market,
                    "duration_hours": pos.duration_hours,
                    "hours_remaining": self.min_duration - pos.duration_hours,
                })

        if short_positions:
            recommendations.append(
                XPRecommendation(
                    action="hold_positions",
                    reason=f"{len(short_positions)} positions need more hold time",
                    priority="low",
                    details={"positions": short_positions},
                )
            )

        return recommendations

    def calculate_optimal_trade_frequency(self) -> Dict:
        """Calculate optimal trade frequency for XP.

        Returns:
            Frequency recommendation
        """
        # Balance: More trades = more volume, but need position duration
        # Optimal: Trade every optimal_duration hours

        trades_per_day = 24 / self.optimal_duration
        volume_per_trade = self.target_volume / Decimal(str(trades_per_day))

        return {
            "trades_per_day": trades_per_day,
            "trade_interval_hours": self.optimal_duration,
            "volume_per_trade": str(volume_per_trade),
            "explanation": (
                f"Trade every {self.optimal_duration}h to balance "
                f"volume generation with position duration bonus"
            ),
        }

    def estimate_weekly_xp(
        self,
        account_id: Optional[str] = None,
        weekly_pool: int = 4_000_000,
    ) -> Dict:
        """Estimate weekly XP based on current performance.

        Args:
            account_id: Optional account filter
            weekly_pool: Total weekly XP pool

        Returns:
            XP estimate dictionary
        """
        if account_id:
            xp = self.volume_tracker.estimate_xp_share(account_id, weekly_pool)
            return {
                "account_id": account_id,
                "estimated_xp": xp,
                "weekly_pool": weekly_pool,
            }

        # All accounts
        estimates = []
        total_xp = 0

        for stats in self.volume_tracker.get_all_stats():
            xp = self.volume_tracker.estimate_xp_share(
                stats.account_id, weekly_pool
            )
            estimates.append({
                "account_id": stats.account_id,
                "estimated_xp": xp,
                "volume_7d": str(stats.volume_7d),
            })
            total_xp += xp

        return {
            "total_estimated_xp": total_xp,
            "weekly_pool": weekly_pool,
            "accounts": estimates,
        }

    def get_distribution_status(self) -> Dict:
        """Get status for upcoming XP distribution.

        XP distributes every Friday at 00:00 UTC.

        Returns:
            Distribution status
        """
        now = time.time()

        # Calculate seconds until Friday 00:00 UTC
        # Python's weekday: Monday=0, Friday=4
        import datetime

        utc_now = datetime.datetime.utcnow()
        days_until_friday = (4 - utc_now.weekday()) % 7
        if days_until_friday == 0 and utc_now.hour >= 0:
            days_until_friday = 7

        next_friday = utc_now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + datetime.timedelta(days=days_until_friday)

        seconds_until = (next_friday - utc_now).total_seconds()

        return {
            "next_distribution": next_friday.isoformat(),
            "hours_remaining": seconds_until / 3600,
            "days_remaining": seconds_until / 86400,
            "weekly_pool": 4_000_000,
        }

    def summary(self) -> Dict:
        """Get XP optimization summary.

        Returns:
            Summary dictionary
        """
        return {
            "volume_progress": self.get_volume_progress(),
            "distribution_status": self.get_distribution_status(),
            "recommendations_count": len(self.get_recommendations()),
            "estimated_xp": self.estimate_weekly_xp(),
            "trade_frequency": self.calculate_optimal_trade_frequency(),
        }
