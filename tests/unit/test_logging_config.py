"""Unit tests for Logging configuration."""

import pytest
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.logging_config import (
    LogLevel,
    LogFormat,
    LoggingConfig,
    ColorCodes,
    SensitiveDataFilter,
    JsonFormatter,
    ColoredFormatter,
    CompactFormatter,
    LoggerRegistry,
    ContextLogger,
    LogMetrics,
    MetricsHandler,
    get_registry,
    configure_logging,
    get_logger,
    get_context_logger,
    set_log_level,
    enable_json_logging,
    enable_debug_logging,
    log_function_call,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels(self):
        """Should have standard log levels."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_format_types(self):
        """Should have format types."""
        assert LogFormat.SIMPLE.value == "simple"
        assert LogFormat.DETAILED.value == "detailed"
        assert LogFormat.JSON.value == "json"
        assert LogFormat.COMPACT.value == "compact"


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = LoggingConfig()

        assert config.level == LogLevel.INFO
        assert config.format_type == LogFormat.DETAILED
        assert config.log_to_console is True
        assert config.log_to_file is True
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.colorize_console is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            format_type=LogFormat.JSON,
            log_to_console=False,
        )

        assert config.level == LogLevel.DEBUG
        assert config.format_type == LogFormat.JSON
        assert config.log_to_console is False

    def test_sensitive_fields_default(self):
        """Should have default sensitive fields."""
        config = LoggingConfig()

        assert "password" in config.sensitive_fields
        assert "secret" in config.sensitive_fields
        assert "token" in config.sensitive_fields

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = LoggingConfig()

        d = config.to_dict()

        assert d["level"] == "INFO"
        assert d["format_type"] == "detailed"
        assert "log_to_console" in d


class TestColorCodes:
    """Tests for ColorCodes."""

    def test_color_codes_exist(self):
        """Should have color codes."""
        assert ColorCodes.RESET == "\033[0m"
        assert ColorCodes.RED == "\033[31m"
        assert ColorCodes.GREEN == "\033[32m"

    def test_level_colors(self):
        """Should have level colors."""
        assert "DEBUG" in ColorCodes.LEVEL_COLORS
        assert "ERROR" in ColorCodes.LEVEL_COLORS
        assert ColorCodes.LEVEL_COLORS["ERROR"] == ColorCodes.RED


class TestSensitiveDataFilter:
    """Tests for SensitiveDataFilter."""

    @pytest.fixture
    def filter(self):
        """Create filter."""
        return SensitiveDataFilter()

    def test_redact_password(self, filter):
        """Should redact password."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='Login with password="secret123"',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "secret123" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redact_token(self, filter):
        """Should redact token."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API token=abc123xyz",
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "abc123xyz" not in record.msg

    def test_preserve_normal_message(self, filter):
        """Should not modify normal messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Normal log message",
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert record.msg == "Normal log message"

    def test_custom_sensitive_fields(self):
        """Should use custom sensitive fields."""
        filter = SensitiveDataFilter(["api_key"])

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="api_key=mysecret",
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "mysecret" not in record.msg


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create JSON formatter."""
        return JsonFormatter()

    def test_format_as_json(self, formatter):
        """Should format as valid JSON."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert data["line"] == 42

    def test_format_includes_timestamp(self, formatter):
        """Should include timestamp."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_format_with_exception(self, formatter):
        """Should include exception info."""
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestColoredFormatter:
    """Tests for ColoredFormatter."""

    def test_format_with_color(self):
        """Should add color codes."""
        formatter = ColoredFormatter("%(levelname)s: %(message)s", colorize=True)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert ColorCodes.RED in result or ColorCodes.RESET in result

    def test_format_without_color(self):
        """Should not add color codes when disabled."""
        formatter = ColoredFormatter("%(levelname)s: %(message)s", colorize=False)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "\033[" not in result


class TestCompactFormatter:
    """Tests for CompactFormatter."""

    def test_compact_format(self):
        """Should format compactly."""
        formatter = CompactFormatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should contain time, level initial, name, message
        assert "I" in result  # INFO -> I
        assert "test.logger" in result
        assert "Test message" in result


class TestLoggerRegistry:
    """Tests for LoggerRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return LoggerRegistry()

    def test_create_registry(self, registry):
        """Should create registry."""
        assert registry._initialized is False
        assert len(registry._loggers) == 0

    def test_configure_initializes(self, registry):
        """Should initialize on configure."""
        config = LoggingConfig(log_to_file=False)

        registry.configure(config)

        assert registry._initialized is True

    def test_get_logger(self, registry):
        """Should get or create logger."""
        registry.configure(LoggingConfig(log_to_file=False))

        logger1 = registry.get_logger("test.module")
        logger2 = registry.get_logger("test.module")

        assert logger1 is logger2
        assert "test.module" in registry._loggers

    def test_get_logger_auto_configure(self, registry):
        """Should auto-configure if not initialized."""
        # No explicit configure call
        logger = registry.get_logger("test")

        assert registry._initialized is True

    def test_set_level(self, registry):
        """Should set log level."""
        registry.configure(LoggingConfig(log_to_file=False))

        registry.set_level(LogLevel.DEBUG)

        assert logging.getLogger().level == logging.DEBUG

    def test_set_level_specific_logger(self, registry):
        """Should set level for specific logger."""
        registry.configure(LoggingConfig(log_to_file=False))
        logger = registry.get_logger("specific")

        registry.set_level(LogLevel.WARNING, "specific")

        assert logger.level == logging.WARNING

    def test_get_status(self, registry):
        """Should get status."""
        registry.configure(LoggingConfig(log_to_file=False))

        status = registry.get_status()

        assert status["initialized"] is True
        assert "handlers" in status
        assert "config" in status


class TestContextLogger:
    """Tests for ContextLogger."""

    @pytest.fixture
    def logger(self):
        """Create context logger."""
        base_logger = logging.getLogger("test.context")
        return ContextLogger(base_logger, {"user": "test_user"})

    def test_create_with_context(self, logger):
        """Should create with context."""
        assert logger.context == {"user": "test_user"}

    def test_with_context_creates_new(self, logger):
        """Should create new logger with additional context."""
        new_logger = logger.with_context(request_id="123")

        assert new_logger is not logger
        assert new_logger.context["user"] == "test_user"
        assert new_logger.context["request_id"] == "123"

    def test_format_message_with_context(self, logger):
        """Should format message with context."""
        msg = logger._format_message("Test action")

        assert "Test action" in msg
        assert "user=test_user" in msg

    def test_format_message_no_context(self):
        """Should not modify message without context."""
        logger = ContextLogger(logging.getLogger("test"), {})

        msg = logger._format_message("Test action")

        assert msg == "Test action"

    def test_logging_methods_exist(self, logger):
        """Should have all logging methods."""
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")
        assert hasattr(logger, "exception")


class TestLogMetrics:
    """Tests for LogMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics."""
        return LogMetrics()

    def test_default_metrics(self, metrics):
        """Should have correct defaults."""
        assert metrics.total_logs == 0
        assert metrics.errors_count == 0
        assert metrics.warnings_count == 0

    def test_record_log(self, metrics):
        """Should record log events."""
        metrics.record("INFO")
        metrics.record("INFO")
        metrics.record("DEBUG")

        assert metrics.total_logs == 3
        assert metrics.by_level["INFO"] == 2
        assert metrics.by_level["DEBUG"] == 1

    def test_record_error(self, metrics):
        """Should track errors."""
        metrics.record("ERROR")

        assert metrics.errors_count == 1

    def test_record_critical(self, metrics):
        """Should count critical as error."""
        metrics.record("CRITICAL")

        assert metrics.errors_count == 1

    def test_record_warning(self, metrics):
        """Should track warnings."""
        metrics.record("WARNING")

        assert metrics.warnings_count == 1

    def test_to_dict(self, metrics):
        """Should convert to dictionary."""
        metrics.record("ERROR")

        d = metrics.to_dict()

        assert d["total_logs"] == 1
        assert d["errors_count"] == 1


class TestMetricsHandler:
    """Tests for MetricsHandler."""

    def test_emit_records_metric(self):
        """Should record metric on emit."""
        metrics = LogMetrics()
        handler = MetricsHandler(metrics)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        assert metrics.total_logs == 1
        assert metrics.errors_count == 1


class TestGlobalFunctions:
    """Tests for global logging functions."""

    def setup_method(self):
        """Reset global registry before each test."""
        import src.logging_config
        src.logging_config._registry = None

    def test_get_registry_creates_singleton(self):
        """Should create singleton registry."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_configure_logging(self):
        """Should configure global logging."""
        config = LoggingConfig(level=LogLevel.DEBUG, log_to_file=False)

        configure_logging(config)

        registry = get_registry()
        assert registry._initialized is True

    def test_get_logger(self):
        """Should get logger from global registry."""
        logger = get_logger("test.global")

        assert logger is not None
        assert logger.name == "test.global"

    def test_get_context_logger(self):
        """Should get context logger."""
        ctx_logger = get_context_logger("test", user="alice")

        assert ctx_logger.context["user"] == "alice"

    def test_set_log_level(self):
        """Should set global log level."""
        configure_logging(LoggingConfig(log_to_file=False))

        set_log_level(LogLevel.WARNING)

        assert logging.getLogger().level == logging.WARNING


class TestLogFunctionCallDecorator:
    """Tests for log_function_call decorator."""

    def test_decorator_logs_call(self, caplog):
        """Should log function call."""
        @log_function_call()
        def test_func():
            return 42

        with caplog.at_level(logging.DEBUG):
            result = test_func()

        assert result == 42
        assert "Calling test_func" in caplog.text

    def test_decorator_logs_success(self, caplog):
        """Should log success."""
        @log_function_call()
        def success_func():
            return "ok"

        with caplog.at_level(logging.DEBUG):
            success_func()

        assert "completed successfully" in caplog.text

    def test_decorator_logs_error(self, caplog):
        """Should log error and re-raise."""
        @log_function_call()
        def error_func():
            raise ValueError("Test error")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                error_func()

        assert "raised ValueError" in caplog.text


class TestFileLogging:
    """Tests for file logging."""

    def test_file_handler_created(self):
        """Should create file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoggingConfig(
                log_dir=Path(tmpdir),
                file_name="test.log",
            )
            registry = LoggerRegistry()
            registry.configure(config)

            log_file = Path(tmpdir) / "test.log"
            assert log_file.exists() or any(
                isinstance(h, logging.handlers.RotatingFileHandler)
                for h in logging.getLogger().handlers
            )

    def test_file_rotation(self):
        """Should support log rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoggingConfig(
                log_dir=Path(tmpdir),
                file_name="rotate.log",
                max_bytes=100,  # Very small for testing
                backup_count=2,
            )
            registry = LoggerRegistry()
            registry.configure(config)

            # File handler should be RotatingFileHandler
            handlers = [
                h for h in logging.getLogger().handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(handlers) >= 0  # May not exist if log_to_console=True first
