"""Position tracking and management across accounts."""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""

    account_id: str
    market: str
    side: str  # "LONG" or "SHORT"
    size: Decimal
    entry_price: Decimal
    mark_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    liquidation_price: Optional[Decimal] = None
    opened_at: float = 0.0
    last_updated: float = 0.0

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position."""
        return self.size * self.mark_price

    @property
    def signed_size(self) -> Decimal:
        """Get signed size (positive for long, negative for short)."""
        return self.size if self.side == "LONG" else -self.size

    @property
    def duration_hours(self) -> float:
        """Get position duration in hours."""
        if self.opened_at == 0:
            return 0.0
        return (time.time() - self.opened_at) / 3600

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "market": self.market,
            "side": self.side,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "mark_price": str(self.mark_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "leverage": str(self.leverage),
            "duration_hours": self.duration_hours,
        }


@dataclass
class PositionManager:
    """Manages position state across all accounts.

    Tracks:
    - Current positions per account/market
    - Aggregate exposure
    - Position changes from WebSocket updates
    """

    # {account_id: {market: Position}}
    _positions: Dict[str, Dict[str, Position]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._positions = {}

    def update_position(self, position: Position) -> None:
        """Update position from exchange data or WebSocket.

        Args:
            position: Position to update
        """
        account_id = position.account_id
        market = position.market

        if account_id not in self._positions:
            self._positions[account_id] = {}

        position.last_updated = time.time()

        # If position is closed (size 0), remove it
        if position.size == 0:
            if market in self._positions[account_id]:
                del self._positions[account_id][market]
                logger.debug(f"Position closed: {account_id} {market}")
        else:
            self._positions[account_id][market] = position
            logger.debug(
                f"Position updated: {account_id} {market} "
                f"{position.side} {position.size}"
            )

    def update_from_api(
        self,
        account_id: str,
        api_positions: List[dict],
    ) -> None:
        """Update positions from API response.

        Args:
            account_id: Account ID
            api_positions: List of position dicts from API
        """
        # Clear existing positions for account
        self._positions[account_id] = {}

        for pos_data in api_positions:
            size = Decimal(str(pos_data.get("size", "0")))
            if size == 0:
                continue

            position = Position(
                account_id=account_id,
                market=pos_data.get("market", ""),
                side=pos_data.get("side", "").upper(),
                size=abs(size),
                entry_price=Decimal(str(pos_data.get("avg_entry_price", "0"))),
                mark_price=Decimal(str(pos_data.get("mark_price", "0"))),
                unrealized_pnl=Decimal(str(pos_data.get("unrealized_pnl", "0"))),
                realized_pnl=Decimal(str(pos_data.get("realized_pnl", "0"))),
                leverage=Decimal(str(pos_data.get("leverage", "1"))),
                liquidation_price=Decimal(str(pos_data.get("liquidation_price", "0")))
                if pos_data.get("liquidation_price")
                else None,
                opened_at=float(pos_data.get("created_at", 0)) / 1000
                if pos_data.get("created_at")
                else time.time(),
            )
            position.last_updated = time.time()

            self._positions[account_id][position.market] = position

    def get_position(
        self,
        account_id: str,
        market: str,
    ) -> Optional[Position]:
        """Get specific position.

        Args:
            account_id: Account ID
            market: Market symbol

        Returns:
            Position if exists, None otherwise
        """
        return self._positions.get(account_id, {}).get(market)

    def get_account_positions(self, account_id: str) -> List[Position]:
        """Get all positions for an account.

        Args:
            account_id: Account ID

        Returns:
            List of positions
        """
        return list(self._positions.get(account_id, {}).values())

    def get_all_positions(self, market: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by market.

        Args:
            market: Optional market filter

        Returns:
            List of positions
        """
        positions = []
        for account_positions in self._positions.values():
            for pos in account_positions.values():
                if market is None or pos.market == market:
                    positions.append(pos)
        return positions

    def get_market_positions(self, market: str) -> Dict[str, Position]:
        """Get positions for a specific market across all accounts.

        Args:
            market: Market symbol

        Returns:
            Dict of {account_id: Position}
        """
        result = {}
        for account_id, positions in self._positions.items():
            if market in positions:
                result[account_id] = positions[market]
        return result

    def get_net_exposure(self, market: str) -> Decimal:
        """Calculate net exposure for market (long - short).

        Args:
            market: Market symbol

        Returns:
            Net exposure (positive = net long, negative = net short)
        """
        net = Decimal("0")
        for pos in self.get_all_positions(market):
            net += pos.signed_size
        return net

    def get_gross_exposure(self, market: str) -> Decimal:
        """Calculate gross exposure for market (long + short).

        Args:
            market: Market symbol

        Returns:
            Gross exposure (total absolute position sizes)
        """
        gross = Decimal("0")
        for pos in self.get_all_positions(market):
            gross += pos.size
        return gross

    def get_long_exposure(self, market: str) -> Decimal:
        """Get total long exposure for market.

        Args:
            market: Market symbol

        Returns:
            Total long exposure
        """
        total = Decimal("0")
        for pos in self.get_all_positions(market):
            if pos.side == "LONG":
                total += pos.size
        return total

    def get_short_exposure(self, market: str) -> Decimal:
        """Get total short exposure for market.

        Args:
            market: Market symbol

        Returns:
            Total short exposure
        """
        total = Decimal("0")
        for pos in self.get_all_positions(market):
            if pos.side == "SHORT":
                total += pos.size
        return total

    def get_accounts_by_side(
        self,
        market: str,
        side: str,
    ) -> List[str]:
        """Get account IDs with positions on specified side.

        Args:
            market: Market symbol
            side: "LONG" or "SHORT"

        Returns:
            List of account IDs
        """
        accounts = []
        for account_id, positions in self._positions.items():
            if market in positions and positions[market].side == side:
                accounts.append(account_id)
        return accounts

    def get_total_pnl(self, market: Optional[str] = None) -> Dict[str, Decimal]:
        """Get total PnL across positions.

        Args:
            market: Optional market filter

        Returns:
            Dict with 'unrealized' and 'realized' PnL
        """
        unrealized = Decimal("0")
        realized = Decimal("0")

        for pos in self.get_all_positions(market):
            unrealized += pos.unrealized_pnl
            realized += pos.realized_pnl

        return {
            "unrealized": unrealized,
            "realized": realized,
            "total": unrealized + realized,
        }

    def get_markets(self) -> List[str]:
        """Get list of all markets with positions.

        Returns:
            List of market symbols
        """
        markets = set()
        for account_positions in self._positions.values():
            markets.update(account_positions.keys())
        return list(markets)

    def has_position(self, account_id: str, market: str) -> bool:
        """Check if account has position in market.

        Args:
            account_id: Account ID
            market: Market symbol

        Returns:
            True if position exists
        """
        return market in self._positions.get(account_id, {})

    def clear_account(self, account_id: str) -> None:
        """Clear all positions for account.

        Args:
            account_id: Account to clear
        """
        if account_id in self._positions:
            del self._positions[account_id]

    def clear_all(self) -> None:
        """Clear all tracked positions."""
        self._positions.clear()

    @property
    def total_positions(self) -> int:
        """Get total number of tracked positions."""
        return sum(
            len(positions)
            for positions in self._positions.values()
        )

    @property
    def accounts_with_positions(self) -> List[str]:
        """Get accounts that have open positions."""
        return [
            account_id
            for account_id, positions in self._positions.items()
            if positions
        ]

    def summary(self) -> dict:
        """Get summary of all positions.

        Returns:
            Summary dict
        """
        markets = self.get_markets()

        market_summaries = {}
        for market in markets:
            market_summaries[market] = {
                "long_exposure": str(self.get_long_exposure(market)),
                "short_exposure": str(self.get_short_exposure(market)),
                "net_exposure": str(self.get_net_exposure(market)),
                "position_count": len(self.get_market_positions(market)),
            }

        pnl = self.get_total_pnl()

        return {
            "total_positions": self.total_positions,
            "accounts_with_positions": len(self.accounts_with_positions),
            "markets": market_summaries,
            "pnl": {
                "unrealized": str(pnl["unrealized"]),
                "realized": str(pnl["realized"]),
                "total": str(pnl["total"]),
            },
        }
