"""Bot configuration using Pydantic Settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrategyType(str, Enum):
    """Available delta-neutral strategies."""

    SIMPLE_5050 = "5050"
    FUNDING_BASED = "funding"
    RANDOM_SPLIT = "random"


class Environment(str, Enum):
    """Paradex environment."""

    TESTNET = "testnet"
    MAINNET = "mainnet"


class RebalancingConfig(BaseSettings):
    """Rebalancing configuration."""

    enabled: bool = True
    threshold_pct: float = Field(default=5.0, description="Delta deviation threshold")
    min_interval_seconds: float = Field(default=300.0, description="Minimum time between rebalances")
    max_slippage_pct: float = Field(default=0.5, description="Maximum allowed slippage")


class AntiDetectionConfig(BaseSettings):
    """Anti-detection configuration."""

    size_variance: float = Field(default=0.15, description="Order size variance ±15%")
    timing_variance: float = Field(default=0.3, description="Timing variance ±30%")
    min_delay_seconds: float = Field(default=0.5, description="Minimum delay between orders")
    max_delay_seconds: float = Field(default=5.0, description="Maximum delay between orders")
    skip_interval_probability: float = Field(default=0.1, description="Probability to skip interval")


class XPOptimizationConfig(BaseSettings):
    """XP optimization configuration."""

    min_position_duration_hours: float = Field(default=24.0, description="Minimum position hold time")
    optimal_position_duration_hours: float = Field(default=48.0, description="Optimal hold time for max XP")
    target_daily_volume: float = Field(default=100000.0, description="Target daily trading volume USD")


class NotificationConfig(BaseSettings):
    """Notification configuration."""

    console_enabled: bool = True
    webhook_url: Optional[str] = None
    alert_on_rebalance: bool = True
    alert_on_error: bool = True
    alert_on_threshold: bool = True


class MarketConfig(BaseSettings):
    """Single market configuration."""

    symbol: str
    enabled: bool = True
    max_position_size: float = Field(default=10000.0, description="Max position size in USD")
    default_leverage: float = Field(default=1.0, description="Default leverage")


class BotConfig(BaseSettings):
    """Main bot configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PARADEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Environment = Environment.MAINNET

    # Trading
    default_strategy: StrategyType = StrategyType.SIMPLE_5050
    default_market: str = "BTC-USD-PERP"
    default_leverage: float = Field(default=1.0, ge=1.0, le=50.0)
    default_position_size: float = Field(default=1000.0, description="Default position size USD")

    # Timing
    trading_interval_seconds: float = Field(default=60.0, description="Trading loop interval")
    position_check_interval_seconds: float = Field(default=10.0, description="Position check interval")

    # Rate limits
    orders_per_second: int = Field(default=800, description="Max orders per second")
    requests_per_minute_ip: int = Field(default=1500, description="Max requests per minute per IP")

    # Sub-configs
    rebalancing: RebalancingConfig = Field(default_factory=RebalancingConfig)
    anti_detection: AntiDetectionConfig = Field(default_factory=AntiDetectionConfig)
    xp_optimization: XPOptimizationConfig = Field(default_factory=XPOptimizationConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    # Paths
    state_dir: Path = Field(default=Path.home() / ".paradex-delta")
    log_file: Optional[Path] = None

    @field_validator("state_dir", mode="after")
    @classmethod
    def ensure_state_dir(cls, v: Path) -> Path:
        """Ensure state directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_config() -> BotConfig:
    """Get cached bot configuration."""
    return BotConfig()
