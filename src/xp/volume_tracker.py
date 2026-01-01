"""Volume tracking for XP estimation."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade."""

    account_id: str
    market: str
    size: Decimal
    price: Decimal
    side: str
    timestamp: float = field(default_factory=time.time)

    @property
    def volume(self) -> Decimal:
        """Get trade volume (notional value)."""
        return self.size * self.price


@dataclass
class VolumeStats:
    """Volume statistics for an account."""

    account_id: str
    total_volume: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")
    volume_7d: Decimal = Decimal("0")
    trade_count: int = 0
    trade_count_24h: int = 0
    avg_trade_size: Decimal = Decimal("0")
    last_trade_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "total_volume": str(self.total_volume),
            "volume_24h": str(self.volume_24h),
            "volume_7d": str(self.volume_7d),
            "trade_count": self.trade_count,
            "trade_count_24h": self.trade_count_24h,
            "avg_trade_size": str(self.avg_trade_size),
        }


@dataclass
class VolumeTracker:
    """Tracks trading volume for XP estimation.

    XP is primarily based on trading volume, so tracking
    helps estimate and optimize XP earnings.

    Weekly pool: 4,000,000 XP distributed based on:
    - Trading volume (primary factor)
    - Position duration
    - Market making (tight spreads)
    """

    # Trade history (keep 7 days)
    _trade_history: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=10000))
    _account_stats: Dict[str, VolumeStats] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._trade_history = deque(maxlen=10000)
        self._account_stats = {}

    def record_trade(
        self,
        account_id: str,
        market: str,
        size: Decimal,
        price: Decimal,
        side: str,
    ) -> None:
        """Record a trade for volume tracking.

        Args:
            account_id: Account that traded
            market: Market symbol
            size: Trade size
            price: Trade price
            side: BUY or SELL
        """
        record = TradeRecord(
            account_id=account_id,
            market=market,
            size=size,
            price=price,
            side=side,
        )
        self._trade_history.append(record)

        # Update account stats
        if account_id not in self._account_stats:
            self._account_stats[account_id] = VolumeStats(account_id=account_id)

        stats = self._account_stats[account_id]
        volume = record.volume

        stats.total_volume += volume
        stats.trade_count += 1
        stats.last_trade_time = record.timestamp

        # Recalculate average
        if stats.trade_count > 0:
            stats.avg_trade_size = stats.total_volume / stats.trade_count

        self._update_time_based_stats()

        logger.debug(
            f"Recorded trade: {account_id} {side} {size} {market} @ {price} "
            f"(volume: {volume})"
        )

    def _update_time_based_stats(self) -> None:
        """Update 24h and 7d volume statistics."""
        now = time.time()
        cutoff_24h = now - 86400  # 24 hours
        cutoff_7d = now - 604800  # 7 days

        # Reset time-based stats
        for stats in self._account_stats.values():
            stats.volume_24h = Decimal("0")
            stats.volume_7d = Decimal("0")
            stats.trade_count_24h = 0

        # Recalculate from history
        for trade in self._trade_history:
            if trade.timestamp < cutoff_7d:
                continue

            stats = self._account_stats.get(trade.account_id)
            if not stats:
                continue

            if trade.timestamp >= cutoff_24h:
                stats.volume_24h += trade.volume
                stats.trade_count_24h += 1

            stats.volume_7d += trade.volume

    def get_account_stats(self, account_id: str) -> VolumeStats:
        """Get volume statistics for account.

        Args:
            account_id: Account ID

        Returns:
            VolumeStats for account
        """
        self._update_time_based_stats()
        return self._account_stats.get(
            account_id,
            VolumeStats(account_id=account_id),
        )

    def get_all_stats(self) -> List[VolumeStats]:
        """Get stats for all accounts.

        Returns:
            List of VolumeStats
        """
        self._update_time_based_stats()
        return list(self._account_stats.values())

    def get_total_volume_24h(self) -> Decimal:
        """Get total 24h volume across all accounts.

        Returns:
            Total 24h volume
        """
        self._update_time_based_stats()
        return sum(s.volume_24h for s in self._account_stats.values())

    def get_total_volume_7d(self) -> Decimal:
        """Get total 7d volume across all accounts.

        Returns:
            Total 7d volume
        """
        self._update_time_based_stats()
        return sum(s.volume_7d for s in self._account_stats.values())

    def estimate_xp_share(
        self,
        account_id: str,
        weekly_pool: int = 4_000_000,
    ) -> float:
        """Estimate XP share for account.

        Very rough estimate - actual XP depends on many factors.

        Args:
            account_id: Account to estimate
            weekly_pool: Total weekly XP pool

        Returns:
            Estimated XP
        """
        self._update_time_based_stats()
        stats = self._account_stats.get(account_id)
        if not stats:
            return 0.0

        total_volume = self.get_total_volume_7d()
        if total_volume == 0:
            return 0.0

        # Simple proportional estimate (actual formula is more complex)
        share = float(stats.volume_7d / total_volume)
        return share * weekly_pool

    def get_volume_by_market(
        self,
        account_id: str = None,
    ) -> Dict[str, Decimal]:
        """Get volume breakdown by market.

        Args:
            account_id: Optional account filter

        Returns:
            Dict of {market: volume}
        """
        by_market: Dict[str, Decimal] = {}

        for trade in self._trade_history:
            if account_id and trade.account_id != account_id:
                continue

            if trade.market not in by_market:
                by_market[trade.market] = Decimal("0")
            by_market[trade.market] += trade.volume

        return by_market

    def get_recent_trades(
        self,
        account_id: str = None,
        limit: int = 20,
    ) -> List[TradeRecord]:
        """Get recent trades.

        Args:
            account_id: Optional account filter
            limit: Maximum trades to return

        Returns:
            List of recent trades
        """
        trades = list(self._trade_history)
        if account_id:
            trades = [t for t in trades if t.account_id == account_id]
        return trades[-limit:]

    def clear_old_history(self, days: int = 7) -> int:
        """Clear history older than specified days.

        Args:
            days: Days to keep

        Returns:
            Number of records cleared
        """
        cutoff = time.time() - (days * 86400)
        old_count = len(self._trade_history)

        # Filter to keep recent only
        self._trade_history = deque(
            (t for t in self._trade_history if t.timestamp >= cutoff),
            maxlen=10000,
        )

        cleared = old_count - len(self._trade_history)
        if cleared > 0:
            logger.info(f"Cleared {cleared} old trade records")

        return cleared

    def summary(self) -> dict:
        """Get volume tracker summary.

        Returns:
            Summary dictionary
        """
        self._update_time_based_stats()
        return {
            "total_trades": len(self._trade_history),
            "accounts_tracked": len(self._account_stats),
            "total_volume_24h": str(self.get_total_volume_24h()),
            "total_volume_7d": str(self.get_total_volume_7d()),
            "accounts": [s.to_dict() for s in self._account_stats.values()],
        }
