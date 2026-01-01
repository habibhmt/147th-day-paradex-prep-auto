"""Main entry point for Paradex Delta Bot."""

import asyncio
import logging
import signal
import sys
from decimal import Decimal
from typing import Optional

from config.settings import BotConfig, StrategyType, get_config
from src.anti_detection.pattern_breaker import PatternBreaker
from src.anti_detection.randomizer import AntiDetectionRandomizer
from src.core.account_manager import AccountManager
from src.core.delta_calculator import DeltaCalculator
from src.core.order_manager import OrderManager, OrderRequest, OrderSide
from src.core.position_manager import PositionManager
from src.network.client_factory import ParadexClientFactory
from src.network.rate_limiter import RateLimiter
from src.network.websocket_manager import WebSocketManager
from src.notifications.notifier import Notifier
from src.rebalancing.engine import RebalancingEngine
from src.rebalancing.threshold_monitor import ThresholdMonitor
from src.security.keychain import KeychainManager
from src.strategies import (
    BaseStrategy,
    FundingBasedStrategy,
    RandomSplitStrategy,
    Simple5050Strategy,
)
from src.xp.optimizer import XPOptimizer
from src.xp.volume_tracker import VolumeTracker

logger = logging.getLogger(__name__)


class ParadexDeltaBot:
    """Main bot class orchestrating all components."""

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the bot.

        Args:
            config: Bot configuration (uses default if None)
        """
        self.config = config or get_config()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all bot components."""
        # Security
        self.keychain = KeychainManager()

        # Network
        self.client_factory = ParadexClientFactory(self.config.environment.value)
        self.rate_limiter = RateLimiter()
        self.ws_manager = WebSocketManager()

        # Core
        self.account_manager = AccountManager(
            keychain=self.keychain,
            client_factory=self.client_factory,
            ws_manager=self.ws_manager,
        )
        self.position_manager = PositionManager()
        self.delta_calculator = DeltaCalculator(
            self.position_manager,
            neutrality_threshold=self.config.rebalancing.threshold_pct,
        )
        self.order_manager = OrderManager(
            account_manager=self.account_manager,
            rate_limiter=self.rate_limiter,
        )

        # Strategy (default, can be changed)
        self.strategy: Optional[BaseStrategy] = None

        # Rebalancing
        self.rebalancing_engine: Optional[RebalancingEngine] = None
        self.threshold_monitor: Optional[ThresholdMonitor] = None

        # XP
        self.volume_tracker = VolumeTracker()
        self.xp_optimizer = XPOptimizer(
            self.position_manager,
            self.volume_tracker,
            min_position_duration=self.config.xp_optimization.min_position_duration_hours,
            optimal_position_duration=self.config.xp_optimization.optimal_position_duration_hours,
        )

        # Anti-detection
        self.randomizer = AntiDetectionRandomizer(
            size_variance=self.config.anti_detection.size_variance,
            timing_variance=self.config.anti_detection.timing_variance,
            min_delay=self.config.anti_detection.min_delay_seconds,
            max_delay=self.config.anti_detection.max_delay_seconds,
        )
        self.pattern_breaker = PatternBreaker()

        # Notifications
        self.notifier = Notifier(
            console_enabled=self.config.notifications.console_enabled,
        )

    def set_strategy(self, strategy_type: StrategyType) -> None:
        """Set the trading strategy.

        Args:
            strategy_type: Strategy to use
        """
        if strategy_type == StrategyType.SIMPLE_5050:
            self.strategy = Simple5050Strategy(
                self.account_manager,
                self.position_manager,
            )
        elif strategy_type == StrategyType.FUNDING_BASED:
            self.strategy = FundingBasedStrategy(
                self.account_manager,
                self.position_manager,
            )
        elif strategy_type == StrategyType.RANDOM_SPLIT:
            self.strategy = RandomSplitStrategy(
                self.account_manager,
                self.position_manager,
            )

        if self.strategy:
            logger.info(f"Strategy set: {self.strategy.name}")

            # Initialize rebalancing with strategy
            self.rebalancing_engine = RebalancingEngine(
                delta_calculator=self.delta_calculator,
                order_manager=self.order_manager,
                strategy=self.strategy,
                threshold_pct=self.config.rebalancing.threshold_pct,
                min_rebalance_interval=self.config.rebalancing.min_interval_seconds,
            )

            self.threshold_monitor = ThresholdMonitor(
                delta_calculator=self.delta_calculator,
                rebalancing_engine=self.rebalancing_engine,
                check_interval=self.config.position_check_interval_seconds,
                auto_rebalance=self.config.rebalancing.enabled,
            )

    async def initialize(self) -> bool:
        """Initialize the bot and accounts.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing Paradex Delta Bot...")

        # Initialize accounts
        n_accounts = await self.account_manager.initialize_accounts()

        if n_accounts < 2:
            await self.notifier.error(
                "Initialization Failed",
                f"Need at least 2 accounts, found {n_accounts}",
            )
            return False

        await self.notifier.success(
            "Bot Initialized",
            f"Loaded {n_accounts} accounts",
        )

        # Set default strategy if not set
        if not self.strategy:
            self.set_strategy(self.config.default_strategy)

        return True

    async def start(
        self,
        market: str,
        total_size: Decimal,
    ) -> None:
        """Start the trading bot.

        Args:
            market: Market to trade
            total_size: Total position size
        """
        if not self.strategy:
            raise ValueError("Strategy not set")

        self._running = True

        await self.notifier.info(
            "Starting Bot",
            f"Market: {market}, Size: ${total_size}",
        )

        # Calculate initial allocations
        allocations = self.strategy.calculate_allocations(market, total_size)

        if not allocations:
            await self.notifier.error("No Allocations", "Strategy returned no allocations")
            return

        # Start threshold monitoring
        if self.threshold_monitor:
            await self.threshold_monitor.start([market])

        # Execute initial positions
        await self._execute_allocations(market, allocations)

        # Main loop
        await self._trading_loop(market)

    async def _execute_allocations(
        self,
        market: str,
        allocations: list,
    ) -> None:
        """Execute strategy allocations.

        Args:
            market: Market to trade
            allocations: List of allocations
        """
        orders = []

        for alloc in allocations:
            # Apply randomization
            size = self.randomizer.randomize_size(alloc.size)

            side = OrderSide.BUY if alloc.side == "LONG" else OrderSide.SELL

            orders.append(
                OrderRequest(
                    account_id=alloc.account_id,
                    market=market,
                    side=side,
                    size=size,
                )
            )

        # Shuffle order to avoid patterns
        account_order = self.pattern_breaker.get_account_order(
            [o.account_id for o in orders]
        )

        # Execute with delays
        for account_id in account_order:
            order = next(o for o in orders if o.account_id == account_id)

            # Random delay
            await self.randomizer.random_delay()

            result = await self.order_manager.submit_order(order)

            if result.success:
                # Track for volume
                self.volume_tracker.record_trade(
                    account_id=account_id,
                    market=market,
                    size=order.size,
                    price=result.avg_price,
                    side=order.side.value,
                )
                self.pattern_breaker.record_trade(account_id)
            else:
                await self.notifier.notify_order_error(
                    account_id, market, result.error or "Unknown error"
                )

    async def _trading_loop(self, market: str) -> None:
        """Main trading loop.

        Args:
            market: Market being traded
        """
        interval = self.config.trading_interval_seconds

        while self._running:
            try:
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break

                # Apply interval variance
                actual_interval = self.pattern_breaker.get_varied_interval(interval)

                # Randomly skip intervals
                if self.pattern_breaker.should_skip_interval():
                    logger.debug("Skipping trading interval")
                    await asyncio.sleep(actual_interval)
                    continue

                # Sync positions
                await self.account_manager.sync_all_positions()

                # Check XP optimization
                recommendations = self.xp_optimizer.get_recommendations()
                for rec in recommendations:
                    if rec.priority == "high":
                        logger.info(f"XP Recommendation: {rec.action} - {rec.reason}")

                # Sleep until next interval
                await asyncio.sleep(actual_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)

    async def stop(self, close_positions: bool = False) -> None:
        """Stop the bot.

        Args:
            close_positions: Whether to close all positions
        """
        logger.info("Stopping bot...")
        self._running = False
        self._shutdown_event.set()

        # Stop monitoring
        if self.threshold_monitor:
            await self.threshold_monitor.stop()

        if close_positions:
            await self._close_all_positions()

        # Shutdown
        await self.account_manager.shutdown()

        await self.notifier.info("Bot Stopped", "Graceful shutdown complete")

    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        positions = self.position_manager.get_all_positions()

        for pos in positions:
            side = OrderSide.SELL if pos.side == "LONG" else OrderSide.BUY
            order = OrderRequest(
                account_id=pos.account_id,
                market=pos.market,
                side=side,
                size=pos.size,
                reduce_only=True,
            )
            await self.order_manager.submit_order(order)

        logger.info(f"Closed {len(positions)} positions")

    def status(self) -> dict:
        """Get bot status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "accounts": self.account_manager.account_count,
            "active_accounts": self.account_manager.active_count,
            "strategy": self.strategy.name if self.strategy else None,
            "exposure": self.account_manager.get_total_exposure(),
            "delta_summary": self.delta_calculator.summary(),
            "volume": self.volume_tracker.summary(),
            "xp": self.xp_optimizer.summary(),
        }


def main():
    """CLI entry point."""
    from src.cli.app import app
    app()


if __name__ == "__main__":
    main()
