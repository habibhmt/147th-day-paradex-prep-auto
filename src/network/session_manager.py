"""Session management for multi-account trading."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session states."""

    INACTIVE = "inactive"
    CONNECTING = "connecting"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SessionPriority(Enum):
    """Session priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SessionConfig:
    """Configuration for session management."""

    max_sessions: int = 100
    session_timeout: float = 300.0  # 5 minutes
    heartbeat_interval: float = 30.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    cleanup_interval: float = 60.0
    connection_timeout: float = 30.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "reconnect_delay": self.reconnect_delay,
            "max_reconnect_attempts": self.max_reconnect_attempts,
        }


@dataclass
class SessionMetrics:
    """Metrics for session management."""

    total_sessions_created: int = 0
    active_sessions: int = 0
    total_reconnects: int = 0
    failed_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    avg_session_duration: float = 0.0
    _total_duration: float = 0.0
    _closed_sessions: int = 0

    def record_session_created(self) -> None:
        """Record new session."""
        self.total_sessions_created += 1
        self.active_sessions += 1

    def record_session_closed(self, duration: float) -> None:
        """Record session closure."""
        self.active_sessions -= 1
        self._closed_sessions += 1
        self._total_duration += duration
        if self._closed_sessions > 0:
            self.avg_session_duration = self._total_duration / self._closed_sessions

    def record_reconnect(self) -> None:
        """Record reconnection attempt."""
        self.total_reconnects += 1

    def record_connection_failure(self) -> None:
        """Record connection failure."""
        self.failed_connections += 1

    def record_message_sent(self) -> None:
        """Record sent message."""
        self.total_messages_sent += 1

    def record_message_received(self) -> None:
        """Record received message."""
        self.total_messages_received += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_sessions_created": self.total_sessions_created,
            "active_sessions": self.active_sessions,
            "total_reconnects": self.total_reconnects,
            "failed_connections": self.failed_connections,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "avg_session_duration": round(self.avg_session_duration, 2),
        }


@dataclass
class Session:
    """A single session instance."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id: str = ""
    state: SessionState = SessionState.INACTIVE
    priority: SessionPriority = SessionPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    reconnect_attempts: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _connection: Any = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_activity == 0:
            self.last_activity = self.created_at
        if self.last_heartbeat == 0:
            self.last_heartbeat = self.created_at

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Get idle time since last activity."""
        return time.time() - self.last_activity

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self.state in (SessionState.ACTIVE, SessionState.RECONNECTING)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def record_message_sent(self) -> None:
        """Record sent message."""
        self.messages_sent += 1
        self.update_activity()

    def record_message_received(self) -> None:
        """Record received message."""
        self.messages_received += 1
        self.update_activity()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "account_id": self.account_id,
            "state": self.state.value,
            "priority": self.priority.value,
            "duration": round(self.duration, 2),
            "idle_time": round(self.idle_time, 2),
            "reconnect_attempts": self.reconnect_attempts,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


@dataclass
class SessionPool:
    """Pool of sessions for an account."""

    account_id: str
    max_size: int = 5
    sessions: List[Session] = field(default_factory=list)

    def __post_init__(self):
        """Initialize sessions list."""
        if self.sessions is None:
            self.sessions = []

    @property
    def active_count(self) -> int:
        """Count active sessions."""
        return sum(1 for s in self.sessions if s.is_active)

    @property
    def available_count(self) -> int:
        """Count available session slots."""
        return self.max_size - len(self.sessions)

    def get_session(self) -> Optional[Session]:
        """Get an active session from pool."""
        for session in self.sessions:
            if session.is_active:
                return session
        return None

    def add_session(self, session: Session) -> bool:
        """Add session to pool."""
        if len(self.sessions) >= self.max_size:
            return False
        self.sessions.append(session)
        return True

    def remove_session(self, session_id: str) -> bool:
        """Remove session from pool."""
        for i, session in enumerate(self.sessions):
            if session.session_id == session_id:
                self.sessions.pop(i)
                return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "max_size": self.max_size,
            "total_sessions": len(self.sessions),
            "active_count": self.active_count,
            "sessions": [s.to_dict() for s in self.sessions],
        }


@dataclass
class SessionManager:
    """Manages sessions for multiple accounts.

    Features:
    - Session lifecycle management
    - Automatic reconnection
    - Session pooling
    - Priority-based scheduling
    - Heartbeat monitoring
    """

    config: SessionConfig = field(default_factory=SessionConfig)
    _sessions: Dict[str, Session] = field(default_factory=dict)
    _pools: Dict[str, SessionPool] = field(default_factory=dict)
    _metrics: SessionMetrics = field(default_factory=SessionMetrics)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _heartbeat_task: Optional[asyncio.Task] = None
    _cleanup_task: Optional[asyncio.Task] = None
    _on_connect: Optional[Callable] = None
    _on_disconnect: Optional[Callable] = None
    _on_error: Optional[Callable] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._sessions = {}
        self._pools = {}
        self._metrics = SessionMetrics()
        self._lock = asyncio.Lock()
        self._heartbeat_task = None
        self._cleanup_task = None

    async def start(self) -> None:
        """Start session manager background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop session manager and close all sessions."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all sessions
        for session in list(self._sessions.values()):
            await self.close_session(session.session_id)

        logger.info("Session manager stopped")

    async def create_session(
        self,
        account_id: str,
        priority: SessionPriority = SessionPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        async with self._lock:
            if len(self._sessions) >= self.config.max_sessions:
                raise RuntimeError("Maximum sessions reached")

            session = Session(
                account_id=account_id,
                priority=priority,
                metadata=metadata or {},
            )

            self._sessions[session.session_id] = session
            self._metrics.record_session_created()

            # Add to pool if exists
            if account_id in self._pools:
                self._pools[account_id].add_session(session)

            logger.debug(f"Created session {session.session_id} for {account_id}")
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def get_sessions_for_account(self, account_id: str) -> List[Session]:
        """Get all sessions for an account."""
        return [s for s in self._sessions.values() if s.account_id == account_id]

    async def get_active_sessions(self) -> List[Session]:
        """Get all active sessions."""
        return [s for s in self._sessions.values() if s.is_active]

    async def connect_session(
        self,
        session_id: str,
        connection: Any = None,
    ) -> bool:
        """Mark session as connected."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.state = SessionState.ACTIVE
        session._connection = connection
        session.update_activity()

        if self._on_connect:
            try:
                if asyncio.iscoroutinefunction(self._on_connect):
                    await self._on_connect(session)
                else:
                    self._on_connect(session)
            except Exception as e:
                logger.error(f"Error in on_connect callback: {e}")

        logger.debug(f"Session {session_id} connected")
        return True

    async def disconnect_session(self, session_id: str) -> bool:
        """Mark session as disconnected."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.state = SessionState.DISCONNECTED
        session._connection = None

        if self._on_disconnect:
            try:
                if asyncio.iscoroutinefunction(self._on_disconnect):
                    await self._on_disconnect(session)
                else:
                    self._on_disconnect(session)
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}")

        logger.debug(f"Session {session_id} disconnected")
        return True

    async def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            duration = session.duration
            session.state = SessionState.INACTIVE
            session._connection = None

            # Remove from pool
            if session.account_id in self._pools:
                self._pools[session.account_id].remove_session(session_id)

            del self._sessions[session_id]
            self._metrics.record_session_closed(duration)

            logger.debug(f"Closed session {session_id}")
            return True

    async def reconnect_session(self, session_id: str) -> bool:
        """Attempt to reconnect a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        if session.reconnect_attempts >= self.config.max_reconnect_attempts:
            session.state = SessionState.ERROR
            self._metrics.record_connection_failure()
            return False

        session.state = SessionState.RECONNECTING
        session.reconnect_attempts += 1
        self._metrics.record_reconnect()

        logger.debug(
            f"Reconnecting session {session_id} "
            f"(attempt {session.reconnect_attempts})"
        )
        return True

    async def set_session_error(
        self,
        session_id: str,
        error: str = None,
    ) -> bool:
        """Set session to error state."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.state = SessionState.ERROR
        if error:
            session.metadata["last_error"] = error

        if self._on_error:
            try:
                if asyncio.iscoroutinefunction(self._on_error):
                    await self._on_error(session, error)
                else:
                    self._on_error(session, error)
            except Exception as e:
                logger.error(f"Error in on_error callback: {e}")

        logger.debug(f"Session {session_id} error: {error}")
        return True

    def create_pool(
        self,
        account_id: str,
        max_size: int = 5,
    ) -> SessionPool:
        """Create a session pool for an account."""
        pool = SessionPool(account_id=account_id, max_size=max_size)
        self._pools[account_id] = pool
        return pool

    def get_pool(self, account_id: str) -> Optional[SessionPool]:
        """Get session pool for account."""
        return self._pools.get(account_id)

    def on_connect(self, callback: Callable) -> None:
        """Set connection callback."""
        self._on_connect = callback

    def on_disconnect(self, callback: Callable) -> None:
        """Set disconnection callback."""
        self._on_disconnect = callback

    def on_error(self, callback: Callable) -> None:
        """Set error callback."""
        self._on_error = callback

    async def send_message(
        self,
        session_id: str,
        message: Any,
    ) -> bool:
        """Record message sent on session."""
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return False

        session.record_message_sent()
        self._metrics.record_message_sent()
        return True

    async def receive_message(
        self,
        session_id: str,
        message: Any = None,
    ) -> bool:
        """Record message received on session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.record_message_received()
        self._metrics.record_message_received()
        return True

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to active sessions."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                for session in list(self._sessions.values()):
                    if session.is_active:
                        session.update_heartbeat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _cleanup_loop(self) -> None:
        """Cleanup timed out sessions."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                timed_out = []
                for session in self._sessions.values():
                    if session.idle_time > self.config.session_timeout:
                        timed_out.append(session.session_id)

                for session_id in timed_out:
                    logger.warning(f"Session {session_id} timed out")
                    await self.close_session(session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_metrics(self) -> SessionMetrics:
        """Get session metrics."""
        return self._metrics

    def get_status(self) -> dict:
        """Get manager status."""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active),
            "pools": len(self._pools),
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
        }

    def get_sessions_by_priority(
        self,
        priority: SessionPriority,
    ) -> List[Session]:
        """Get sessions by priority."""
        return [s for s in self._sessions.values() if s.priority == priority]

    def get_sessions_by_state(
        self,
        state: SessionState,
    ) -> List[Session]:
        """Get sessions by state."""
        return [s for s in self._sessions.values() if s.state == state]


# Global session manager
_global_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SessionManager()
    return _global_manager


def reset_session_manager() -> None:
    """Reset global session manager."""
    global _global_manager
    _global_manager = None
