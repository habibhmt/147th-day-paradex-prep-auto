"""Centralized logging configuration for the trading bot."""

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log format types."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    COMPACT = "compact"


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.DETAILED
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    file_name: str = "trading_bot.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    include_timestamps: bool = True
    include_source: bool = True
    colorize_console: bool = True
    sensitive_fields: List[str] = field(
        default_factory=lambda: ["password", "secret", "key", "token", "credential"]
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "format_type": self.format_type.value,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
            "log_dir": str(self.log_dir),
            "file_name": self.file_name,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
            "colorize_console": self.colorize_console,
        }


class ColorCodes:
    """ANSI color codes for console output."""

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    LEVEL_COLORS = {
        "DEBUG": CYAN,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": f"{BOLD}{RED}",
    }


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log records."""

    def __init__(self, sensitive_fields: List[str] = None):
        super().__init__()
        self.sensitive_fields = sensitive_fields or [
            "password", "secret", "key", "token", "credential", "private_key"
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log message."""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for field in self.sensitive_fields:
                if field.lower() in record.msg.lower():
                    record.msg = self._redact_field(record.msg, field)
        return True

    def _redact_field(self, msg: str, field: str) -> str:
        """Redact a specific field from message."""
        import re
        pattern = rf'({field}["\']?\s*[:=]\s*["\']?)([^"\'\s,}}\]]+)'
        return re.sub(pattern, r'\1[REDACTED]', msg, flags=re.IGNORECASE)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self._standard_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "asctime"
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if self.include_extra:
            extra = {
                k: v for k, v in record.__dict__.items()
                if k not in self._standard_keys and not k.startswith("_")
            }
            if extra:
                log_data["extra"] = extra

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for console."""

    def __init__(self, fmt: str = None, datefmt: str = None, colorize: bool = True):
        super().__init__(fmt, datefmt)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        if not self.colorize:
            return super().format(record)

        color = ColorCodes.LEVEL_COLORS.get(record.levelname, "")
        reset = ColorCodes.RESET

        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{reset}"

        result = super().format(record)
        record.levelname = original_levelname

        return result


class CompactFormatter(logging.Formatter):
    """Compact log format for high-volume logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format compactly."""
        ts = datetime.utcfromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname[0]  # First letter only
        return f"{ts} {level} [{record.name}] {record.getMessage()}"


@dataclass
class LoggerRegistry:
    """Registry for managing loggers."""

    _loggers: Dict[str, logging.Logger] = field(default_factory=dict)
    _config: LoggingConfig = field(default_factory=LoggingConfig)
    _initialized: bool = False

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._loggers = {}
        self._config = LoggingConfig()
        self._initialized = False

    def configure(self, config: LoggingConfig = None) -> None:
        """Configure logging system."""
        if config:
            self._config = config

        self._setup_root_logger()
        self._initialized = True

    def _setup_root_logger(self) -> None:
        """Setup root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self._config.level.value))

        root_logger.handlers.clear()

        if self._config.log_to_console:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)

        if self._config.log_to_file:
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)

        filter = SensitiveDataFilter(self._config.sensitive_fields)
        for handler in root_logger.handlers:
            handler.addFilter(filter)

    def _create_console_handler(self) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, self._config.level.value))
        handler.setFormatter(self._create_formatter(for_console=True))
        return handler

    def _create_file_handler(self) -> logging.Handler:
        """Create rotating file handler."""
        log_dir = self._config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / self._config.file_name

        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._config.max_bytes,
            backupCount=self._config.backup_count,
            encoding="utf-8",
        )
        handler.setLevel(getattr(logging, self._config.level.value))
        handler.setFormatter(self._create_formatter(for_console=False))
        return handler

    def _create_formatter(self, for_console: bool) -> logging.Formatter:
        """Create formatter based on config."""
        format_type = self._config.format_type

        if format_type == LogFormat.JSON:
            return JsonFormatter()

        if format_type == LogFormat.COMPACT:
            return CompactFormatter()

        if format_type == LogFormat.SIMPLE:
            fmt = "%(levelname)s: %(message)s"
        else:  # DETAILED
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

        if for_console and self._config.colorize_console:
            return ColoredFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S", colorize=True)
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger."""
        if not self._initialized:
            self.configure()

        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)

        return self._loggers[name]

    def set_level(self, level: LogLevel, logger_name: str = None) -> None:
        """Set log level for logger or all loggers."""
        level_value = getattr(logging, level.value)

        if logger_name:
            logging.getLogger(logger_name).setLevel(level_value)
        else:
            logging.getLogger().setLevel(level_value)
            for handler in logging.getLogger().handlers:
                handler.setLevel(level_value)

    def add_handler(
        self,
        handler: logging.Handler,
        logger_name: str = None,
    ) -> None:
        """Add custom handler."""
        if logger_name:
            logging.getLogger(logger_name).addHandler(handler)
        else:
            logging.getLogger().addHandler(handler)

    def get_status(self) -> Dict[str, Any]:
        """Get logging status."""
        root = logging.getLogger()
        return {
            "initialized": self._initialized,
            "level": logging.getLevelName(root.level),
            "handlers": [type(h).__name__ for h in root.handlers],
            "loggers": list(self._loggers.keys()),
            "config": self._config.to_dict(),
        }


@dataclass
class ContextLogger:
    """Logger with context for structured logging."""

    logger: logging.Logger
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize context."""
        self.context = self.context.copy() if self.context else {}

    def with_context(self, **kwargs) -> "ContextLogger":
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return ContextLogger(self.logger, new_context)

    def _format_message(self, msg: str) -> str:
        """Format message with context."""
        if not self.context:
            return msg
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{msg} | {context_str}"

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(msg), *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(msg), *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(msg), *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(msg), *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(msg), *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception."""
        self.logger.exception(self._format_message(msg), *args, **kwargs)


@dataclass
class LogMetrics:
    """Metrics for log output."""

    total_logs: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    errors_count: int = 0
    warnings_count: int = 0

    def __post_init__(self):
        """Initialize counters."""
        self.by_level = {}

    def record(self, level: str) -> None:
        """Record a log event."""
        self.total_logs += 1
        self.by_level[level] = self.by_level.get(level, 0) + 1

        if level == "ERROR" or level == "CRITICAL":
            self.errors_count += 1
        elif level == "WARNING":
            self.warnings_count += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_logs": self.total_logs,
            "by_level": self.by_level,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
        }


class MetricsHandler(logging.Handler):
    """Handler that tracks log metrics."""

    def __init__(self, metrics: LogMetrics):
        super().__init__()
        self.metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        """Record metric for log event."""
        self.metrics.record(record.levelname)


# Global registry
_registry: Optional[LoggerRegistry] = None


def get_registry() -> LoggerRegistry:
    """Get or create global logger registry."""
    global _registry
    if _registry is None:
        _registry = LoggerRegistry()
    return _registry


def configure_logging(config: LoggingConfig = None) -> None:
    """Configure global logging."""
    get_registry().configure(config)


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return get_registry().get_logger(name)


def get_context_logger(name: str, **context) -> ContextLogger:
    """Get context logger with initial context."""
    logger = get_registry().get_logger(name)
    return ContextLogger(logger, context)


def set_log_level(level: LogLevel, logger_name: str = None) -> None:
    """Set log level globally or for specific logger."""
    get_registry().set_level(level, logger_name)


def enable_json_logging() -> None:
    """Enable JSON log format."""
    config = LoggingConfig(format_type=LogFormat.JSON)
    configure_logging(config)


def enable_debug_logging() -> None:
    """Enable debug level logging."""
    set_log_level(LogLevel.DEBUG)


def log_function_call(logger: logging.Logger = None):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator
