"""Threshold monitoring for delta exposure."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.core.delta_calculator import DeltaCalculator, DeltaReport
from src.rebalancing.engine import RebalancingEngine, RebalanceTrigger

logger = logging.getLogger(__name__)


@dataclass
class ThresholdAlert:
    """Alert for threshold breach."""

    market: str
    delta_pct: float
    threshold_pct: float
    net_delta: str
    is_critical: bool  # > 2x threshold
    timestamp: float = 0.0


@dataclass
class ThresholdMonitor:
    """Monitors delta thresholds and triggers alerts/rebalancing.

    Runs continuous monitoring loop checking all markets.
    Can trigger automatic rebalancing or just alerts.
    """

    delta_calculator: DeltaCalculator
    rebalancing_engine: RebalancingEngine
    check_interval: float = 10.0  # Check every 10 seconds
    auto_rebalance: bool = True  # Auto rebalance when threshold exceeded

    # Callbacks
    alert_callback: Optional[Callable] = None  # Called on threshold breach

    # State
    _running: bool = False
    _monitored_markets: List[str] = field(default_factory=list)
    _tasks: List[asyncio.Task] = field(default_factory=list)
    _alerts: Dict[str, ThresholdAlert] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._running = False
        self._monitored_markets = []
        self._tasks = []
        self._alerts = {}

    async def start(self, markets: List[str]) -> None:
        """Start monitoring specified markets.

        Args:
            markets: List of market symbols to monitor
        """
        if self._running:
            logger.warning("Monitor already running")
            return

        self._monitored_markets = markets
        self._running = True

        # Start monitor task
        task = asyncio.create_task(self._monitor_loop())
        self._tasks.append(task)

        logger.info(f"Started monitoring {len(markets)} markets")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Stopped threshold monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                for market in self._monitored_markets:
                    await self._check_market(market)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_market(self, market: str) -> None:
        """Check delta for a single market.

        Args:
            market: Market to check
        """
        report = self.delta_calculator.calculate_delta(market)

        if report.is_neutral:
            # Clear any existing alert
            if market in self._alerts:
                del self._alerts[market]
            return

        # Threshold exceeded
        threshold = self.delta_calculator.neutrality_threshold
        is_critical = report.delta_percentage > (threshold * 2)

        alert = ThresholdAlert(
            market=market,
            delta_pct=report.delta_percentage,
            threshold_pct=threshold,
            net_delta=str(report.net_delta),
            is_critical=is_critical,
        )
        self._alerts[market] = alert

        logger.warning(
            f"{'CRITICAL ' if is_critical else ''}Threshold exceeded for {market}: "
            f"{report.delta_percentage:.2f}% (threshold: {threshold}%)"
        )

        # Trigger alert callback
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(alert)
                else:
                    self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Trigger auto rebalance
        if self.auto_rebalance:
            await self.rebalancing_engine.check_and_rebalance(
                market,
                trigger=RebalanceTrigger.THRESHOLD_EXCEEDED,
            )

    def add_market(self, market: str) -> None:
        """Add market to monitoring list.

        Args:
            market: Market symbol to add
        """
        if market not in self._monitored_markets:
            self._monitored_markets.append(market)
            logger.info(f"Added {market} to monitoring")

    def remove_market(self, market: str) -> None:
        """Remove market from monitoring.

        Args:
            market: Market to remove
        """
        if market in self._monitored_markets:
            self._monitored_markets.remove(market)
            if market in self._alerts:
                del self._alerts[market]
            logger.info(f"Removed {market} from monitoring")

    def get_active_alerts(self) -> List[ThresholdAlert]:
        """Get all active threshold alerts.

        Returns:
            List of active alerts
        """
        return list(self._alerts.values())

    def get_critical_alerts(self) -> List[ThresholdAlert]:
        """Get only critical alerts.

        Returns:
            List of critical alerts
        """
        return [a for a in self._alerts.values() if a.is_critical]

    def clear_alert(self, market: str) -> None:
        """Clear alert for market.

        Args:
            market: Market to clear
        """
        if market in self._alerts:
            del self._alerts[market]

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def monitored_markets(self) -> List[str]:
        """Get list of monitored markets."""
        return self._monitored_markets.copy()

    @property
    def alert_count(self) -> int:
        """Get number of active alerts."""
        return len(self._alerts)

    def set_auto_rebalance(self, enabled: bool) -> None:
        """Enable or disable automatic rebalancing.

        Args:
            enabled: Whether to auto rebalance
        """
        self.auto_rebalance = enabled
        logger.info(f"Auto rebalance {'enabled' if enabled else 'disabled'}")

    def status(self) -> dict:
        """Get monitor status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "monitored_markets": self._monitored_markets,
            "check_interval": self.check_interval,
            "auto_rebalance": self.auto_rebalance,
            "active_alerts": len(self._alerts),
            "critical_alerts": len(self.get_critical_alerts()),
        }
