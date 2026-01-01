"""Rebalancing engine for delta-neutral maintenance."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable

from src.core.delta_calculator import DeltaCalculator, DeltaReport
from src.core.order_manager import OrderManager, OrderRequest, OrderSide
from src.strategies.base import BaseStrategy, StrategyAllocation

logger = logging.getLogger(__name__)


class RebalanceTrigger(Enum):
    """Reasons for triggering a rebalance."""

    THRESHOLD_EXCEEDED = "threshold"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FUNDING_CHANGE = "funding"
    POSITION_CLOSED = "position_closed"


@dataclass
class RebalanceAction:
    """Describes a single rebalancing action."""

    account_id: str
    market: str
    action: str  # "increase", "decrease", "close"
    side: str  # "LONG" or "SHORT"
    size_change: Decimal
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "market": self.market,
            "action": self.action,
            "side": self.side,
            "size_change": str(self.size_change),
            "reason": self.reason,
        }


@dataclass
class RebalanceResult:
    """Result of a rebalancing operation."""

    success: bool
    trigger: RebalanceTrigger
    actions_planned: int
    actions_executed: int
    delta_before: Decimal
    delta_after: Optional[Decimal] = None
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RebalancingEngine:
    """Manages automatic position rebalancing.

    Monitors delta exposure and executes trades to maintain neutrality.
    Supports automatic and manual rebalancing modes.
    """

    delta_calculator: DeltaCalculator
    order_manager: OrderManager
    strategy: BaseStrategy
    threshold_pct: float = 5.0  # Trigger rebalance at 5% deviation
    min_rebalance_interval: float = 300.0  # 5 minutes minimum between rebalances
    max_slippage_pct: float = 0.5  # Maximum allowed slippage

    # Callbacks
    on_rebalance_start: Optional[Callable] = None
    on_rebalance_complete: Optional[Callable] = None

    # State tracking
    _last_rebalance: Dict[str, float] = field(default_factory=dict)
    _rebalance_history: List[RebalanceResult] = field(default_factory=list)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._last_rebalance = {}
        self._rebalance_history = []

    async def check_and_rebalance(
        self,
        market: str,
        trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD_EXCEEDED,
        force: bool = False,
    ) -> Optional[RebalanceResult]:
        """Check if rebalancing needed and execute if so.

        Args:
            market: Market to check
            trigger: Reason for check
            force: Force rebalance even if not needed

        Returns:
            RebalanceResult if rebalance was executed
        """
        delta_report = self.delta_calculator.calculate_delta(market)

        # Check if rebalance needed
        if not force and delta_report.is_neutral:
            logger.debug(f"No rebalance needed for {market}: delta={delta_report.delta_percentage}%")
            return None

        # Check minimum interval
        last = self._last_rebalance.get(market, 0)
        if not force and time.time() - last < self.min_rebalance_interval:
            logger.debug(f"Rebalance interval not met for {market}")
            return None

        # Execute rebalance
        return await self._execute_rebalance(market, delta_report, trigger)

    async def _execute_rebalance(
        self,
        market: str,
        delta_report: DeltaReport,
        trigger: RebalanceTrigger,
    ) -> RebalanceResult:
        """Execute rebalancing trades.

        Args:
            market: Market to rebalance
            delta_report: Current delta state
            trigger: Reason for rebalance

        Returns:
            RebalanceResult
        """
        logger.info(
            f"Starting rebalance for {market}: "
            f"delta={delta_report.net_delta} ({delta_report.delta_percentage:.2f}%)"
        )

        # Callback
        if self.on_rebalance_start:
            await self._safe_callback(
                self.on_rebalance_start, market, delta_report, trigger
            )

        # Get rebalance allocations from strategy
        allocations = self.strategy.get_rebalance_allocations(market, delta_report)

        if not allocations:
            logger.warning(f"No rebalance allocations generated for {market}")
            return RebalanceResult(
                success=False,
                trigger=trigger,
                actions_planned=0,
                actions_executed=0,
                delta_before=delta_report.net_delta,
                errors=["No allocations generated"],
            )

        # Convert allocations to orders
        orders = self._allocations_to_orders(market, allocations)

        # Execute orders
        results = await self.order_manager.submit_batch(orders)

        # Check results
        executed = sum(1 for r in results if r.success)
        errors = [r.error for r in results if r.error]

        # Update last rebalance time
        self._last_rebalance[market] = time.time()

        # Check new delta
        await asyncio.sleep(1)  # Wait for positions to update
        new_report = self.delta_calculator.calculate_delta(market)

        result = RebalanceResult(
            success=executed > 0 and new_report.is_neutral,
            trigger=trigger,
            actions_planned=len(orders),
            actions_executed=executed,
            delta_before=delta_report.net_delta,
            delta_after=new_report.net_delta,
            errors=errors,
        )

        self._rebalance_history.append(result)

        # Callback
        if self.on_rebalance_complete:
            await self._safe_callback(self.on_rebalance_complete, market, result)

        logger.info(
            f"Rebalance complete for {market}: "
            f"{executed}/{len(orders)} orders, "
            f"delta {delta_report.net_delta} -> {new_report.net_delta}"
        )

        return result

    def _allocations_to_orders(
        self,
        market: str,
        allocations: List[StrategyAllocation],
    ) -> List[OrderRequest]:
        """Convert strategy allocations to order requests.

        Args:
            market: Market symbol
            allocations: List of allocations

        Returns:
            List of order requests
        """
        orders = []

        for alloc in allocations:
            # Determine order side based on allocation
            if alloc.size > 0:
                # Increasing position
                side = OrderSide.BUY if alloc.side == "LONG" else OrderSide.SELL
            else:
                # Decreasing position
                side = OrderSide.SELL if alloc.side == "LONG" else OrderSide.BUY

            orders.append(
                OrderRequest(
                    account_id=alloc.account_id,
                    market=market,
                    side=side,
                    size=abs(alloc.size),
                )
            )

        return orders

    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback.

        Args:
            callback: Function to call
            *args: Arguments to pass
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def manual_rebalance(self, market: str) -> RebalanceResult:
        """Trigger manual rebalance.

        Args:
            market: Market to rebalance

        Returns:
            RebalanceResult
        """
        return await self.check_and_rebalance(
            market,
            trigger=RebalanceTrigger.MANUAL,
            force=True,
        )

    def get_rebalance_history(
        self,
        market: Optional[str] = None,
        limit: int = 10,
    ) -> List[RebalanceResult]:
        """Get recent rebalance history.

        Args:
            market: Optional market filter
            limit: Maximum results

        Returns:
            List of recent results
        """
        history = self._rebalance_history[-limit:]
        return list(reversed(history))

    def get_time_until_next_rebalance(self, market: str) -> float:
        """Get seconds until next rebalance allowed.

        Args:
            market: Market to check

        Returns:
            Seconds remaining (0 if immediate allowed)
        """
        last = self._last_rebalance.get(market, 0)
        elapsed = time.time() - last
        remaining = self.min_rebalance_interval - elapsed
        return max(0, remaining)

    def set_threshold(self, threshold_pct: float) -> None:
        """Update rebalance threshold.

        Args:
            threshold_pct: New threshold percentage
        """
        self.threshold_pct = threshold_pct
        self.delta_calculator.set_threshold(threshold_pct)
        logger.info(f"Rebalance threshold set to {threshold_pct}%")

    def reset_timers(self) -> None:
        """Reset all rebalance timers."""
        self._last_rebalance.clear()
        logger.info("Rebalance timers reset")
