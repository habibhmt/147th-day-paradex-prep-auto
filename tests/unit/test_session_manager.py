"""Unit tests for Session management."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from src.network.session_manager import (
    SessionState,
    SessionPriority,
    SessionConfig,
    SessionMetrics,
    Session,
    SessionPool,
    SessionManager,
    get_session_manager,
    reset_session_manager,
)


class TestSessionState:
    """Tests for SessionState enum."""

    def test_state_values(self):
        """Should have expected state values."""
        assert SessionState.INACTIVE.value == "inactive"
        assert SessionState.CONNECTING.value == "connecting"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.RECONNECTING.value == "reconnecting"
        assert SessionState.DISCONNECTED.value == "disconnected"
        assert SessionState.ERROR.value == "error"


class TestSessionPriority:
    """Tests for SessionPriority enum."""

    def test_priority_values(self):
        """Should have expected priority values."""
        assert SessionPriority.LOW.value == 1
        assert SessionPriority.NORMAL.value == 2
        assert SessionPriority.HIGH.value == 3
        assert SessionPriority.CRITICAL.value == 4


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = SessionConfig()

        assert config.max_sessions == 100
        assert config.session_timeout == 300.0
        assert config.heartbeat_interval == 30.0
        assert config.reconnect_delay == 5.0
        assert config.max_reconnect_attempts == 5

    def test_custom_config(self):
        """Should accept custom values."""
        config = SessionConfig(
            max_sessions=50,
            session_timeout=600.0,
        )

        assert config.max_sessions == 50
        assert config.session_timeout == 600.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = SessionConfig()

        d = config.to_dict()

        assert d["max_sessions"] == 100
        assert "heartbeat_interval" in d


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_default_metrics(self):
        """Should have correct defaults."""
        metrics = SessionMetrics()

        assert metrics.total_sessions_created == 0
        assert metrics.active_sessions == 0
        assert metrics.total_reconnects == 0
        assert metrics.failed_connections == 0

    def test_record_session_created(self):
        """Should record session creation."""
        metrics = SessionMetrics()

        metrics.record_session_created()
        metrics.record_session_created()

        assert metrics.total_sessions_created == 2
        assert metrics.active_sessions == 2

    def test_record_session_closed(self):
        """Should record session closure."""
        metrics = SessionMetrics()
        metrics.active_sessions = 2

        metrics.record_session_closed(60.0)

        assert metrics.active_sessions == 1
        assert metrics.avg_session_duration == 60.0

    def test_avg_session_duration(self):
        """Should calculate average duration."""
        metrics = SessionMetrics()
        metrics.active_sessions = 2

        metrics.record_session_closed(60.0)
        metrics.record_session_closed(120.0)

        assert metrics.avg_session_duration == 90.0

    def test_record_reconnect(self):
        """Should record reconnect."""
        metrics = SessionMetrics()

        metrics.record_reconnect()

        assert metrics.total_reconnects == 1

    def test_record_connection_failure(self):
        """Should record connection failure."""
        metrics = SessionMetrics()

        metrics.record_connection_failure()

        assert metrics.failed_connections == 1

    def test_record_messages(self):
        """Should record messages."""
        metrics = SessionMetrics()

        metrics.record_message_sent()
        metrics.record_message_received()

        assert metrics.total_messages_sent == 1
        assert metrics.total_messages_received == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = SessionMetrics()
        metrics.record_session_created()

        d = metrics.to_dict()

        assert d["total_sessions_created"] == 1


class TestSession:
    """Tests for Session dataclass."""

    def test_create_session(self):
        """Should create session with defaults."""
        session = Session(account_id="acc1")

        assert session.account_id == "acc1"
        assert session.state == SessionState.INACTIVE
        assert session.priority == SessionPriority.NORMAL
        assert session.reconnect_attempts == 0

    def test_session_id_generated(self):
        """Should generate unique session ID."""
        session1 = Session(account_id="acc1")
        session2 = Session(account_id="acc1")

        assert session1.session_id != session2.session_id

    def test_session_duration(self):
        """Should calculate duration."""
        session = Session(
            account_id="acc1",
            created_at=time.time() - 60,
        )

        assert session.duration >= 60

    def test_session_idle_time(self):
        """Should calculate idle time."""
        session = Session(
            account_id="acc1",
            last_activity=time.time() - 30,
        )

        assert session.idle_time >= 30

    def test_is_active(self):
        """Should detect active state."""
        session = Session(account_id="acc1")
        assert session.is_active is False

        session.state = SessionState.ACTIVE
        assert session.is_active is True

    def test_is_connected(self):
        """Should detect connected states."""
        session = Session(account_id="acc1")

        session.state = SessionState.ACTIVE
        assert session.is_connected is True

        session.state = SessionState.RECONNECTING
        assert session.is_connected is True

        session.state = SessionState.DISCONNECTED
        assert session.is_connected is False

    def test_update_activity(self):
        """Should update activity timestamp."""
        session = Session(account_id="acc1")
        old_activity = session.last_activity

        time.sleep(0.01)
        session.update_activity()

        assert session.last_activity > old_activity

    def test_update_heartbeat(self):
        """Should update heartbeat timestamp."""
        session = Session(account_id="acc1")
        old_heartbeat = session.last_heartbeat

        time.sleep(0.01)
        session.update_heartbeat()

        assert session.last_heartbeat > old_heartbeat

    def test_record_message_sent(self):
        """Should record sent message."""
        session = Session(account_id="acc1")

        session.record_message_sent()

        assert session.messages_sent == 1

    def test_record_message_received(self):
        """Should record received message."""
        session = Session(account_id="acc1")

        session.record_message_received()

        assert session.messages_received == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        session = Session(account_id="acc1")

        d = session.to_dict()

        assert d["account_id"] == "acc1"
        assert "state" in d
        assert "duration" in d


class TestSessionPool:
    """Tests for SessionPool dataclass."""

    def test_create_pool(self):
        """Should create pool."""
        pool = SessionPool(account_id="acc1")

        assert pool.account_id == "acc1"
        assert pool.max_size == 5
        assert len(pool.sessions) == 0

    def test_active_count(self):
        """Should count active sessions."""
        pool = SessionPool(account_id="acc1")

        session1 = Session(account_id="acc1", state=SessionState.ACTIVE)
        session2 = Session(account_id="acc1", state=SessionState.INACTIVE)

        pool.sessions = [session1, session2]

        assert pool.active_count == 1

    def test_available_count(self):
        """Should count available slots."""
        pool = SessionPool(account_id="acc1", max_size=3)

        session = Session(account_id="acc1")
        pool.sessions.append(session)

        assert pool.available_count == 2

    def test_get_session(self):
        """Should get active session."""
        pool = SessionPool(account_id="acc1")

        session1 = Session(account_id="acc1", state=SessionState.INACTIVE)
        session2 = Session(account_id="acc1", state=SessionState.ACTIVE)

        pool.sessions = [session1, session2]

        result = pool.get_session()
        assert result == session2

    def test_get_session_none_active(self):
        """Should return None if no active session."""
        pool = SessionPool(account_id="acc1")

        result = pool.get_session()

        assert result is None

    def test_add_session(self):
        """Should add session to pool."""
        pool = SessionPool(account_id="acc1", max_size=2)
        session = Session(account_id="acc1")

        result = pool.add_session(session)

        assert result is True
        assert len(pool.sessions) == 1

    def test_add_session_full(self):
        """Should reject when pool is full."""
        pool = SessionPool(account_id="acc1", max_size=1)
        pool.sessions.append(Session(account_id="acc1"))

        result = pool.add_session(Session(account_id="acc1"))

        assert result is False

    def test_remove_session(self):
        """Should remove session from pool."""
        pool = SessionPool(account_id="acc1")
        session = Session(account_id="acc1")
        pool.sessions.append(session)

        result = pool.remove_session(session.session_id)

        assert result is True
        assert len(pool.sessions) == 0

    def test_remove_session_not_found(self):
        """Should return False if session not found."""
        pool = SessionPool(account_id="acc1")

        result = pool.remove_session("nonexistent")

        assert result is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        pool = SessionPool(account_id="acc1")

        d = pool.to_dict()

        assert d["account_id"] == "acc1"
        assert d["max_size"] == 5


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def manager(self):
        """Create fresh session manager."""
        return SessionManager()

    @pytest.mark.asyncio
    async def test_create_manager(self, manager):
        """Should create manager."""
        assert len(manager._sessions) == 0
        assert manager._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Should start and stop background tasks."""
        await manager.start()
        assert manager._heartbeat_task is not None
        assert manager._cleanup_task is not None

        await manager.stop()
        assert len(manager._sessions) == 0

    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """Should create new session."""
        session = await manager.create_session("acc1")

        assert session.account_id == "acc1"
        assert session.session_id in manager._sessions
        assert manager._metrics.total_sessions_created == 1

    @pytest.mark.asyncio
    async def test_create_session_with_priority(self, manager):
        """Should create session with priority."""
        session = await manager.create_session(
            "acc1",
            priority=SessionPriority.HIGH,
        )

        assert session.priority == SessionPriority.HIGH

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, manager):
        """Should create session with metadata."""
        session = await manager.create_session(
            "acc1",
            metadata={"key": "value"},
        )

        assert session.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_create_session_max_reached(self, manager):
        """Should raise when max sessions reached."""
        manager.config.max_sessions = 1
        await manager.create_session("acc1")

        with pytest.raises(RuntimeError):
            await manager.create_session("acc2")

    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        """Should get session by ID."""
        session = await manager.create_session("acc1")

        result = await manager.get_session(session.session_id)

        assert result == session

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager):
        """Should return None for unknown session."""
        result = await manager.get_session("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_sessions_for_account(self, manager):
        """Should get all sessions for account."""
        await manager.create_session("acc1")
        await manager.create_session("acc1")
        await manager.create_session("acc2")

        sessions = await manager.get_sessions_for_account("acc1")

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, manager):
        """Should get active sessions."""
        session1 = await manager.create_session("acc1")
        session2 = await manager.create_session("acc2")

        await manager.connect_session(session1.session_id)

        active = await manager.get_active_sessions()

        assert len(active) == 1
        assert active[0] == session1

    @pytest.mark.asyncio
    async def test_connect_session(self, manager):
        """Should connect session."""
        session = await manager.create_session("acc1")

        result = await manager.connect_session(session.session_id)

        assert result is True
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_connect_session_with_connection(self, manager):
        """Should store connection object."""
        session = await manager.create_session("acc1")
        connection = MagicMock()

        await manager.connect_session(session.session_id, connection)

        assert session._connection == connection

    @pytest.mark.asyncio
    async def test_connect_session_callback(self, manager):
        """Should call on_connect callback."""
        callback = AsyncMock()
        manager.on_connect(callback)

        session = await manager.create_session("acc1")
        await manager.connect_session(session.session_id)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_session(self, manager):
        """Should disconnect session."""
        session = await manager.create_session("acc1")
        await manager.connect_session(session.session_id)

        result = await manager.disconnect_session(session.session_id)

        assert result is True
        assert session.state == SessionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_session_callback(self, manager):
        """Should call on_disconnect callback."""
        callback = AsyncMock()
        manager.on_disconnect(callback)

        session = await manager.create_session("acc1")
        await manager.connect_session(session.session_id)
        await manager.disconnect_session(session.session_id)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session(self, manager):
        """Should close and remove session."""
        session = await manager.create_session("acc1")

        result = await manager.close_session(session.session_id)

        assert result is True
        assert session.session_id not in manager._sessions
        assert manager._metrics.active_sessions == 0

    @pytest.mark.asyncio
    async def test_reconnect_session(self, manager):
        """Should attempt reconnection."""
        session = await manager.create_session("acc1")
        session.state = SessionState.DISCONNECTED

        result = await manager.reconnect_session(session.session_id)

        assert result is True
        assert session.state == SessionState.RECONNECTING
        assert session.reconnect_attempts == 1
        assert manager._metrics.total_reconnects == 1

    @pytest.mark.asyncio
    async def test_reconnect_session_max_attempts(self, manager):
        """Should fail after max reconnect attempts."""
        manager.config.max_reconnect_attempts = 3
        session = await manager.create_session("acc1")
        session.reconnect_attempts = 3

        result = await manager.reconnect_session(session.session_id)

        assert result is False
        assert session.state == SessionState.ERROR

    @pytest.mark.asyncio
    async def test_set_session_error(self, manager):
        """Should set session to error state."""
        session = await manager.create_session("acc1")

        result = await manager.set_session_error(session.session_id, "Test error")

        assert result is True
        assert session.state == SessionState.ERROR
        assert session.metadata["last_error"] == "Test error"

    @pytest.mark.asyncio
    async def test_set_session_error_callback(self, manager):
        """Should call on_error callback."""
        callback = AsyncMock()
        manager.on_error(callback)

        session = await manager.create_session("acc1")
        await manager.set_session_error(session.session_id, "Error")

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pool(self, manager):
        """Should create session pool."""
        pool = manager.create_pool("acc1", max_size=3)

        assert pool.account_id == "acc1"
        assert pool.max_size == 3
        assert "acc1" in manager._pools

    @pytest.mark.asyncio
    async def test_get_pool(self, manager):
        """Should get session pool."""
        manager.create_pool("acc1")

        pool = manager.get_pool("acc1")

        assert pool is not None
        assert pool.account_id == "acc1"

    @pytest.mark.asyncio
    async def test_session_added_to_pool(self, manager):
        """Should add session to existing pool."""
        pool = manager.create_pool("acc1")
        session = await manager.create_session("acc1")

        assert session in pool.sessions

    @pytest.mark.asyncio
    async def test_send_message(self, manager):
        """Should record sent message."""
        session = await manager.create_session("acc1")
        await manager.connect_session(session.session_id)

        result = await manager.send_message(session.session_id, "test")

        assert result is True
        assert session.messages_sent == 1
        assert manager._metrics.total_messages_sent == 1

    @pytest.mark.asyncio
    async def test_send_message_inactive(self, manager):
        """Should fail if session not active."""
        session = await manager.create_session("acc1")

        result = await manager.send_message(session.session_id, "test")

        assert result is False

    @pytest.mark.asyncio
    async def test_receive_message(self, manager):
        """Should record received message."""
        session = await manager.create_session("acc1")

        result = await manager.receive_message(session.session_id)

        assert result is True
        assert session.messages_received == 1
        assert manager._metrics.total_messages_received == 1

    @pytest.mark.asyncio
    async def test_get_metrics(self, manager):
        """Should return metrics."""
        await manager.create_session("acc1")

        metrics = manager.get_metrics()

        assert metrics.total_sessions_created == 1

    @pytest.mark.asyncio
    async def test_get_status(self, manager):
        """Should return status."""
        session = await manager.create_session("acc1")
        await manager.connect_session(session.session_id)

        status = manager.get_status()

        assert status["total_sessions"] == 1
        assert status["active_sessions"] == 1
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_get_sessions_by_priority(self, manager):
        """Should filter sessions by priority."""
        await manager.create_session("acc1", priority=SessionPriority.HIGH)
        await manager.create_session("acc2", priority=SessionPriority.LOW)
        await manager.create_session("acc3", priority=SessionPriority.HIGH)

        high_priority = manager.get_sessions_by_priority(SessionPriority.HIGH)

        assert len(high_priority) == 2

    @pytest.mark.asyncio
    async def test_get_sessions_by_state(self, manager):
        """Should filter sessions by state."""
        session1 = await manager.create_session("acc1")
        session2 = await manager.create_session("acc2")
        await manager.connect_session(session1.session_id)

        active = manager.get_sessions_by_state(SessionState.ACTIVE)
        inactive = manager.get_sessions_by_state(SessionState.INACTIVE)

        assert len(active) == 1
        assert len(inactive) == 1


class TestGlobalSessionManager:
    """Tests for global session manager."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_session_manager()

    def test_get_session_manager_creates_singleton(self):
        """Should create singleton manager."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_reset_session_manager(self):
        """Should reset global manager."""
        manager1 = get_session_manager()
        reset_session_manager()
        manager2 = get_session_manager()

        assert manager1 is not manager2
