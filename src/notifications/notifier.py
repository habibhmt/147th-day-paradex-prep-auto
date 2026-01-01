"""Notification dispatcher for alerts and status updates."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


@dataclass
class Notification:
    """A notification message."""

    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class Notifier:
    """Dispatches notifications to various handlers.

    Supports:
    - Console output (Rich formatting)
    - Custom callbacks
    - Notification history
    """

    console_enabled: bool = True
    max_history: int = 100

    _console: Console = field(default_factory=Console)
    _history: List[Notification] = field(default_factory=list)
    _callbacks: List[Callable] = field(default_factory=list)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._console = Console()
        self._history = []
        self._callbacks = []

    def add_callback(self, callback: Callable) -> None:
        """Add notification callback.

        Args:
            callback: Function to call on notifications
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove notification callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def notify(
        self,
        level: NotificationLevel,
        title: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """Send a notification.

        Args:
            level: Notification level
            title: Notification title
            message: Notification message
            data: Optional additional data
        """
        notification = Notification(
            level=level,
            title=title,
            message=message,
            data=data,
        )

        # Add to history
        self._history.append(notification)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        # Console output
        if self.console_enabled:
            self._print_notification(notification)

        # Callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    def _print_notification(self, notification: Notification) -> None:
        """Print notification to console.

        Args:
            notification: Notification to print
        """
        # Color based on level
        colors = {
            NotificationLevel.INFO: "blue",
            NotificationLevel.WARNING: "yellow",
            NotificationLevel.ERROR: "red",
            NotificationLevel.CRITICAL: "red bold",
            NotificationLevel.SUCCESS: "green",
        }
        color = colors.get(notification.level, "white")

        # Create panel
        panel = Panel(
            notification.message,
            title=f"[{color}]{notification.title}[/{color}]",
            border_style=color,
        )
        self._console.print(panel)

    # Convenience methods
    async def info(self, title: str, message: str, data: dict = None) -> None:
        """Send info notification."""
        await self.notify(NotificationLevel.INFO, title, message, data)

    async def warning(self, title: str, message: str, data: dict = None) -> None:
        """Send warning notification."""
        await self.notify(NotificationLevel.WARNING, title, message, data)

    async def error(self, title: str, message: str, data: dict = None) -> None:
        """Send error notification."""
        await self.notify(NotificationLevel.ERROR, title, message, data)

    async def critical(self, title: str, message: str, data: dict = None) -> None:
        """Send critical notification."""
        await self.notify(NotificationLevel.CRITICAL, title, message, data)

    async def success(self, title: str, message: str, data: dict = None) -> None:
        """Send success notification."""
        await self.notify(NotificationLevel.SUCCESS, title, message, data)

    # Specific notification types
    async def notify_rebalance(
        self,
        market: str,
        delta_before: str,
        delta_after: str,
        success: bool,
    ) -> None:
        """Notify about rebalancing.

        Args:
            market: Market that was rebalanced
            delta_before: Delta before rebalance
            delta_after: Delta after rebalance
            success: Whether rebalance succeeded
        """
        if success:
            await self.success(
                "Rebalance Complete",
                f"Market: {market}\nDelta: {delta_before} -> {delta_after}",
                {
                    "market": market,
                    "delta_before": delta_before,
                    "delta_after": delta_after,
                },
            )
        else:
            await self.error(
                "Rebalance Failed",
                f"Market: {market}\nDelta remained: {delta_before}",
                {"market": market, "delta": delta_before},
            )

    async def notify_threshold_breach(
        self,
        market: str,
        delta_pct: float,
        threshold: float,
    ) -> None:
        """Notify about threshold breach.

        Args:
            market: Market with breach
            delta_pct: Current delta percentage
            threshold: Threshold that was breached
        """
        level = (
            NotificationLevel.CRITICAL
            if delta_pct > threshold * 2
            else NotificationLevel.WARNING
        )
        await self.notify(
            level,
            "Threshold Breach",
            f"Market: {market}\nDelta: {delta_pct:.2f}% (threshold: {threshold}%)",
            {"market": market, "delta_pct": delta_pct, "threshold": threshold},
        )

    async def notify_order_error(
        self,
        account_id: str,
        market: str,
        error: str,
    ) -> None:
        """Notify about order error.

        Args:
            account_id: Account with error
            market: Market
            error: Error message
        """
        await self.error(
            "Order Error",
            f"Account: {account_id}\nMarket: {market}\nError: {error}",
            {"account_id": account_id, "market": market, "error": error},
        )

    def get_history(
        self,
        level: Optional[NotificationLevel] = None,
        limit: int = 20,
    ) -> List[Notification]:
        """Get notification history.

        Args:
            level: Optional level filter
            limit: Maximum notifications

        Returns:
            List of notifications
        """
        history = self._history
        if level:
            history = [n for n in history if n.level == level]
        return history[-limit:]

    def clear_history(self) -> None:
        """Clear notification history."""
        self._history.clear()

    @property
    def unread_count(self) -> int:
        """Get number of recent notifications."""
        return len(self._history)
