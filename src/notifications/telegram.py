"""Telegram notification integration for trading bot."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import aiohttp

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    rate_limit: int = 30  # Messages per minute
    retry_attempts: int = 3
    retry_delay: float = 1.0
    parse_mode: str = "HTML"  # HTML or Markdown
    disable_notification: bool = False  # Silent notifications

    def to_dict(self) -> dict:
        """Convert to dictionary (without sensitive data)."""
        return {
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "retry_attempts": self.retry_attempts,
            "parse_mode": self.parse_mode,
            "has_token": bool(self.bot_token),
            "has_chat_id": bool(self.chat_id),
        }

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)


@dataclass
class TelegramMessage:
    """A Telegram message."""

    text: str
    priority: MessagePriority = MessagePriority.NORMAL
    parse_mode: Optional[str] = None
    disable_notification: bool = False
    timestamp: float = field(default_factory=time.time)
    sent: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "priority": self.priority.value,
            "sent": self.sent,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class TelegramNotifier:
    """Telegram notification handler.

    Features:
    - Send messages to Telegram bot
    - Rate limiting to avoid API bans
    - Message queuing for high volume
    - Retry logic for failed messages
    - Formatted messages with HTML/Markdown
    """

    config: TelegramConfig = field(default_factory=TelegramConfig)
    _message_queue: List[TelegramMessage] = field(default_factory=list)
    _sent_timestamps: List[float] = field(default_factory=list)
    _session: Optional[aiohttp.ClientSession] = None
    _is_running: bool = False

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._message_queue = []
        self._sent_timestamps = []
        self._session = None
        self._is_running = False

    @property
    def api_url(self) -> str:
        """Get Telegram Bot API URL."""
        return f"https://api.telegram.org/bot{self.config.bot_token}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limit."""
        now = time.time()
        # Remove timestamps older than 1 minute
        self._sent_timestamps = [
            t for t in self._sent_timestamps
            if now - t < 60
        ]
        return len(self._sent_timestamps) < self.config.rate_limit

    def _record_send(self) -> None:
        """Record a message send timestamp."""
        self._sent_timestamps.append(time.time())

    async def send_message(
        self,
        text: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        parse_mode: Optional[str] = None,
        disable_notification: Optional[bool] = None,
    ) -> bool:
        """Send a message to Telegram.

        Args:
            text: Message text
            priority: Message priority
            parse_mode: Override parse mode (HTML/Markdown)
            disable_notification: Send silently

        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            logger.debug("Telegram notifications disabled")
            return False

        if not self.config.is_configured():
            logger.warning("Telegram not configured")
            return False

        message = TelegramMessage(
            text=text,
            priority=priority,
            parse_mode=parse_mode or self.config.parse_mode,
            disable_notification=(
                disable_notification
                if disable_notification is not None
                else self.config.disable_notification
            ),
        )

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Telegram rate limit reached, queueing message")
            self._message_queue.append(message)
            return False

        return await self._send_message(message)

    async def _send_message(self, message: TelegramMessage) -> bool:
        """Internal method to send message with retries."""
        session = await self._get_session()
        url = f"{self.api_url}/sendMessage"

        payload = {
            "chat_id": self.config.chat_id,
            "text": message.text,
            "parse_mode": message.parse_mode,
            "disable_notification": message.disable_notification,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        message.sent = True
                        self._record_send()
                        logger.debug("Telegram message sent successfully")
                        return True
                    elif response.status == 429:
                        # Rate limited by Telegram
                        data = await response.json()
                        retry_after = data.get("parameters", {}).get("retry_after", 60)
                        logger.warning(f"Telegram rate limited, retry after {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        error_text = await response.text()
                        message.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Telegram API error: {message.error}")

            except aiohttp.ClientError as e:
                message.error = str(e)
                logger.error(f"Telegram connection error: {e}")

            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return False

    async def process_queue(self) -> int:
        """Process queued messages.

        Returns:
            Number of messages successfully sent
        """
        sent = 0
        while self._message_queue and self._check_rate_limit():
            message = self._message_queue.pop(0)
            if await self._send_message(message):
                sent += 1
            await asyncio.sleep(0.1)  # Small delay between messages
        return sent

    def queue_message(
        self,
        text: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Queue a message for later sending.

        Args:
            text: Message text
            priority: Message priority
        """
        message = TelegramMessage(text=text, priority=priority)

        # Insert based on priority
        if priority == MessagePriority.URGENT:
            self._message_queue.insert(0, message)
        else:
            self._message_queue.append(message)

    # Convenience methods for common notifications

    async def notify_trade(
        self,
        account_id: str,
        market: str,
        side: str,
        size: str,
        price: str,
    ) -> bool:
        """Send trade notification."""
        icon = "📈" if side.upper() == "BUY" else "📉"
        text = (
            f"{icon} <b>Trade Executed</b>\n\n"
            f"Account: <code>{account_id}</code>\n"
            f"Market: {market}\n"
            f"Side: {side}\n"
            f"Size: {size}\n"
            f"Price: {price}"
        )
        return await self.send_message(text)

    async def notify_rebalance(
        self,
        market: str,
        delta_before: str,
        delta_after: str,
        success: bool,
    ) -> bool:
        """Send rebalance notification."""
        if success:
            text = (
                f"⚖️ <b>Rebalance Complete</b>\n\n"
                f"Market: {market}\n"
                f"Delta: {delta_before} → {delta_after}"
            )
            return await self.send_message(text)
        else:
            text = (
                f"❌ <b>Rebalance Failed</b>\n\n"
                f"Market: {market}\n"
                f"Delta: {delta_before}"
            )
            return await self.send_message(text, priority=MessagePriority.HIGH)

    async def notify_alert(
        self,
        title: str,
        message: str,
        level: str = "warning",
    ) -> bool:
        """Send alert notification."""
        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
            "success": "✅",
        }
        icon = icons.get(level, "📢")

        priority = (
            MessagePriority.URGENT if level == "critical"
            else MessagePriority.HIGH if level == "error"
            else MessagePriority.NORMAL
        )

        text = f"{icon} <b>{title}</b>\n\n{message}"
        return await self.send_message(text, priority=priority)

    async def notify_daily_summary(
        self,
        total_trades: int,
        total_volume: str,
        total_pnl: str,
        win_rate: float,
    ) -> bool:
        """Send daily summary."""
        pnl_icon = "📈" if not total_pnl.startswith("-") else "📉"
        text = (
            f"📊 <b>Daily Summary</b>\n\n"
            f"Trades: {total_trades}\n"
            f"Volume: ${total_volume}\n"
            f"PnL: {pnl_icon} ${total_pnl}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        return await self.send_message(text)

    async def notify_startup(self, accounts: int, markets: List[str]) -> bool:
        """Send startup notification."""
        text = (
            f"🚀 <b>Bot Started</b>\n\n"
            f"Accounts: {accounts}\n"
            f"Markets: {', '.join(markets)}"
        )
        return await self.send_message(text)

    async def notify_shutdown(self, reason: str = "Manual") -> bool:
        """Send shutdown notification."""
        text = f"🔴 <b>Bot Stopped</b>\n\nReason: {reason}"
        return await self.send_message(text, priority=MessagePriority.HIGH)

    async def notify_error(self, error_type: str, error_msg: str) -> bool:
        """Send error notification."""
        text = (
            f"❌ <b>Error: {error_type}</b>\n\n"
            f"<code>{error_msg[:500]}</code>"
        )
        return await self.send_message(text, priority=MessagePriority.HIGH)

    async def test_connection(self) -> Dict[str, Any]:
        """Test Telegram connection.

        Returns:
            Connection test result
        """
        if not self.config.is_configured():
            return {
                "success": False,
                "error": "Not configured",
            }

        try:
            session = await self._get_session()
            url = f"{self.api_url}/getMe"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    bot_info = data.get("result", {})
                    return {
                        "success": True,
                        "bot_name": bot_info.get("username"),
                        "bot_id": bot_info.get("id"),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                    }

        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_queue_size(self) -> int:
        """Get number of queued messages."""
        return len(self._message_queue)

    def get_rate_limit_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage.

        Returns:
            Rate limit status
        """
        now = time.time()
        recent = [t for t in self._sent_timestamps if now - t < 60]

        return {
            "messages_sent": len(recent),
            "limit": self.config.rate_limit,
            "usage_pct": (len(recent) / self.config.rate_limit) * 100,
            "available": self.config.rate_limit - len(recent),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get notifier status.

        Returns:
            Status dictionary
        """
        return {
            "config": self.config.to_dict(),
            "is_configured": self.config.is_configured(),
            "queue_size": self.get_queue_size(),
            "rate_limit": self.get_rate_limit_usage(),
        }

    def clear_queue(self) -> int:
        """Clear the message queue.

        Returns:
            Number of messages cleared
        """
        count = len(self._message_queue)
        self._message_queue.clear()
        return count
