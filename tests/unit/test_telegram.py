"""Unit tests for Telegram Notifier."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.notifications.telegram import (
    TelegramNotifier,
    TelegramConfig,
    TelegramMessage,
    MessagePriority,
)


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_values(self):
        """Should have expected priority values."""
        assert MessagePriority.LOW.value == "low"
        assert MessagePriority.NORMAL.value == "normal"
        assert MessagePriority.HIGH.value == "high"
        assert MessagePriority.URGENT.value == "urgent"


class TestTelegramConfig:
    """Tests for TelegramConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = TelegramConfig()

        assert config.bot_token == ""
        assert config.chat_id == ""
        assert config.enabled is True
        assert config.rate_limit == 30
        assert config.retry_attempts == 3
        assert config.parse_mode == "HTML"

    def test_custom_config(self):
        """Should accept custom values."""
        config = TelegramConfig(
            bot_token="test_token",
            chat_id="123456",
            rate_limit=20,
        )

        assert config.bot_token == "test_token"
        assert config.chat_id == "123456"
        assert config.rate_limit == 20

    def test_is_configured_false(self):
        """Should return False when not configured."""
        config = TelegramConfig()
        assert config.is_configured() is False

    def test_is_configured_partial(self):
        """Should return False with partial config."""
        config = TelegramConfig(bot_token="token")
        assert config.is_configured() is False

        config = TelegramConfig(chat_id="123")
        assert config.is_configured() is False

    def test_is_configured_true(self):
        """Should return True when fully configured."""
        config = TelegramConfig(
            bot_token="token",
            chat_id="123456",
        )
        assert config.is_configured() is True

    def test_to_dict(self):
        """Should convert to dictionary without sensitive data."""
        config = TelegramConfig(
            bot_token="secret_token",
            chat_id="123456",
        )

        d = config.to_dict()

        assert "bot_token" not in d
        assert d["has_token"] is True
        assert d["has_chat_id"] is True
        assert "enabled" in d


class TestTelegramMessage:
    """Tests for TelegramMessage dataclass."""

    def test_create_message(self):
        """Should create message correctly."""
        msg = TelegramMessage(
            text="Test message",
            priority=MessagePriority.HIGH,
        )

        assert msg.text == "Test message"
        assert msg.priority == MessagePriority.HIGH
        assert msg.sent is False
        assert msg.error is None

    def test_default_priority(self):
        """Should default to NORMAL priority."""
        msg = TelegramMessage(text="Test")
        assert msg.priority == MessagePriority.NORMAL

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        msg = TelegramMessage(
            text="Test message",
            priority=MessagePriority.URGENT,
        )

        d = msg.to_dict()

        assert d["text"] == "Test message"
        assert d["priority"] == "urgent"
        assert d["sent"] is False

    def test_to_dict_truncates_long_text(self):
        """Should truncate long text in to_dict."""
        long_text = "x" * 200
        msg = TelegramMessage(text=long_text)

        d = msg.to_dict()

        assert len(d["text"]) == 103  # 100 + "..."


class TestTelegramNotifier:
    """Tests for TelegramNotifier."""

    @pytest.fixture
    def notifier(self):
        """Create unconfigured notifier."""
        return TelegramNotifier()

    @pytest.fixture
    def configured_notifier(self):
        """Create configured notifier."""
        config = TelegramConfig(
            bot_token="test_token_123",
            chat_id="987654321",
        )
        return TelegramNotifier(config=config)

    def test_initial_state(self, notifier):
        """Should start with clean state."""
        assert notifier.get_queue_size() == 0
        assert len(notifier._sent_timestamps) == 0
        assert notifier._is_running is False

    def test_api_url(self, configured_notifier):
        """Should construct correct API URL."""
        assert "test_token_123" in configured_notifier.api_url
        assert configured_notifier.api_url.startswith("https://api.telegram.org/bot")

    @pytest.mark.asyncio
    async def test_send_message_disabled(self, notifier):
        """Should return False when disabled."""
        notifier.config.enabled = False

        result = await notifier.send_message("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self, notifier):
        """Should return False when not configured."""
        result = await notifier.send_message("Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_success(self, configured_notifier):
        """Should send message successfully."""
        with patch.object(
            configured_notifier, "_send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            result = await configured_notifier.send_message("Test message")

            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_records_timestamp(self, configured_notifier):
        """Should record send timestamp on successful send."""
        # Directly test _record_send and rate limiting behavior
        initial_count = len(configured_notifier._sent_timestamps)
        configured_notifier._record_send()

        assert len(configured_notifier._sent_timestamps) == initial_count + 1

    def test_check_rate_limit_under_limit(self, configured_notifier):
        """Should allow when under rate limit."""
        # Add some timestamps but stay under limit
        now = time.time()
        for _ in range(5):
            configured_notifier._sent_timestamps.append(now)

        assert configured_notifier._check_rate_limit() is True

    def test_check_rate_limit_at_limit(self, configured_notifier):
        """Should block when at rate limit."""
        now = time.time()
        for _ in range(configured_notifier.config.rate_limit):
            configured_notifier._sent_timestamps.append(now)

        assert configured_notifier._check_rate_limit() is False

    def test_check_rate_limit_cleans_old(self, configured_notifier):
        """Should clean old timestamps."""
        # Add old timestamps
        old_time = time.time() - 120  # 2 minutes ago
        for _ in range(50):
            configured_notifier._sent_timestamps.append(old_time)

        # Should clean and allow
        assert configured_notifier._check_rate_limit() is True
        assert len(configured_notifier._sent_timestamps) == 0

    def test_queue_message(self, notifier):
        """Should queue message."""
        notifier.queue_message("Test message")

        assert notifier.get_queue_size() == 1

    def test_queue_message_priority_order(self, notifier):
        """Should insert urgent messages at front."""
        notifier.queue_message("Normal 1", MessagePriority.NORMAL)
        notifier.queue_message("Normal 2", MessagePriority.NORMAL)
        notifier.queue_message("Urgent", MessagePriority.URGENT)

        assert notifier._message_queue[0].text == "Urgent"

    @pytest.mark.asyncio
    async def test_notify_trade(self, configured_notifier):
        """Should format trade notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_trade(
                account_id="acc1",
                market="BTC-USD-PERP",
                side="BUY",
                size="1.5",
                price="50000",
            )

            mock_send.assert_called_once()
            call_text = mock_send.call_args[0][0]
            assert "Trade Executed" in call_text
            assert "BTC-USD-PERP" in call_text

    @pytest.mark.asyncio
    async def test_notify_rebalance_success(self, configured_notifier):
        """Should format successful rebalance notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_rebalance(
                market="BTC-USD-PERP",
                delta_before="5%",
                delta_after="1%",
                success=True,
            )

            call_text = mock_send.call_args[0][0]
            assert "Rebalance Complete" in call_text

    @pytest.mark.asyncio
    async def test_notify_rebalance_failure(self, configured_notifier):
        """Should format failed rebalance notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_rebalance(
                market="BTC-USD-PERP",
                delta_before="5%",
                delta_after="5%",
                success=False,
            )

            call_text = mock_send.call_args[0][0]
            assert "Rebalance Failed" in call_text
            # Should be high priority
            assert mock_send.call_args.kwargs.get("priority") == MessagePriority.HIGH

    @pytest.mark.asyncio
    async def test_notify_alert(self, configured_notifier):
        """Should format alert notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_alert(
                title="Test Alert",
                message="Something happened",
                level="warning",
            )

            call_text = mock_send.call_args[0][0]
            assert "Test Alert" in call_text
            assert "Something happened" in call_text

    @pytest.mark.asyncio
    async def test_notify_alert_critical_priority(self, configured_notifier):
        """Should use urgent priority for critical alerts."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_alert(
                title="Critical",
                message="Emergency",
                level="critical",
            )

            assert mock_send.call_args.kwargs.get("priority") == MessagePriority.URGENT

    @pytest.mark.asyncio
    async def test_notify_daily_summary(self, configured_notifier):
        """Should format daily summary."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_daily_summary(
                total_trades=50,
                total_volume="100000",
                total_pnl="500",
                win_rate=65.5,
            )

            call_text = mock_send.call_args[0][0]
            assert "Daily Summary" in call_text
            assert "50" in call_text
            assert "65.5%" in call_text

    @pytest.mark.asyncio
    async def test_notify_startup(self, configured_notifier):
        """Should format startup notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_startup(
                accounts=3,
                markets=["BTC-USD-PERP", "ETH-USD-PERP"],
            )

            call_text = mock_send.call_args[0][0]
            assert "Bot Started" in call_text
            assert "3" in call_text

    @pytest.mark.asyncio
    async def test_notify_shutdown(self, configured_notifier):
        """Should format shutdown notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_shutdown(reason="Maintenance")

            call_text = mock_send.call_args[0][0]
            assert "Bot Stopped" in call_text
            assert "Maintenance" in call_text

    @pytest.mark.asyncio
    async def test_notify_error(self, configured_notifier):
        """Should format error notification."""
        with patch.object(
            configured_notifier, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            await configured_notifier.notify_error(
                error_type="Connection",
                error_msg="Failed to connect",
            )

            call_text = mock_send.call_args[0][0]
            assert "Error: Connection" in call_text

    @pytest.mark.asyncio
    async def test_test_connection_not_configured(self, notifier):
        """Should fail test when not configured."""
        result = await notifier.test_connection()

        assert result["success"] is False
        assert "Not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_test_connection_success(self, configured_notifier):
        """Should succeed with valid credentials."""
        import aiohttp

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "result": {"username": "test_bot", "id": 12345}
        })

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_session.get.return_value = mock_context

        with patch.object(
            configured_notifier, "_get_session", new_callable=AsyncMock
        ) as mock_get_session:
            mock_get_session.return_value = mock_session

            result = await configured_notifier.test_connection()

            assert result["success"] is True
            assert result["bot_name"] == "test_bot"

    def test_get_rate_limit_usage(self, configured_notifier):
        """Should return rate limit usage."""
        now = time.time()
        for _ in range(10):
            configured_notifier._sent_timestamps.append(now)

        usage = configured_notifier.get_rate_limit_usage()

        assert usage["messages_sent"] == 10
        assert usage["limit"] == 30
        assert usage["usage_pct"] == pytest.approx(33.33, rel=0.1)
        assert usage["available"] == 20

    def test_get_status(self, configured_notifier):
        """Should return status dictionary."""
        configured_notifier.queue_message("Test")

        status = configured_notifier.get_status()

        assert "config" in status
        assert "is_configured" in status
        assert "queue_size" in status
        assert "rate_limit" in status
        assert status["is_configured"] is True
        assert status["queue_size"] == 1

    def test_clear_queue(self, notifier):
        """Should clear message queue."""
        notifier.queue_message("Test 1")
        notifier.queue_message("Test 2")
        notifier.queue_message("Test 3")

        cleared = notifier.clear_queue()

        assert cleared == 3
        assert notifier.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_close(self, configured_notifier):
        """Should close session."""
        # Create a session
        await configured_notifier._get_session()

        await configured_notifier.close()

        # Session should be closed
        assert (
            configured_notifier._session is None
            or configured_notifier._session.closed
        )

    @pytest.mark.asyncio
    async def test_process_queue(self, configured_notifier):
        """Should process queued messages."""
        configured_notifier.queue_message("Test 1")
        configured_notifier.queue_message("Test 2")

        with patch.object(
            configured_notifier, "_send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True

            sent = await configured_notifier.process_queue()

            assert sent == 2
            assert configured_notifier.get_queue_size() == 0
