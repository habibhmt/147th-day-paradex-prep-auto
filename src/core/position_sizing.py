"""Position sizing calculator for risk-adjusted trading."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""

    FIXED = "fixed"  # Fixed position size
    PERCENT_EQUITY = "percent_equity"  # Percentage of account equity
    PERCENT_RISK = "percent_risk"  # Based on risk per trade
    KELLY = "kelly"  # Kelly criterion
    OPTIMAL_F = "optimal_f"  # Optimal f sizing
    VOLATILITY = "volatility"  # Volatility-based sizing
    ATR = "atr"  # ATR-based sizing
    MARTINGALE = "martingale"  # Martingale progression
    ANTI_MARTINGALE = "anti_martingale"  # Anti-martingale


class RiskLevel(Enum):
    """Risk level presets."""

    CONSERVATIVE = "conservative"  # 0.5% risk per trade
    MODERATE = "moderate"  # 1% risk per trade
    AGGRESSIVE = "aggressive"  # 2% risk per trade
    VERY_AGGRESSIVE = "very_aggressive"  # 5% risk per trade


@dataclass
class SizingConfig:
    """Configuration for position sizing."""

    method: SizingMethod = SizingMethod.PERCENT_EQUITY
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_position_pct: float = 25.0  # Max 25% of equity per position
    max_total_exposure_pct: float = 100.0  # Max 100% total exposure
    min_position_size: Decimal = Decimal("0.001")
    max_position_size: Optional[Decimal] = None
    use_leverage: bool = True
    max_leverage: float = 10.0
    risk_per_trade_pct: float = 1.0  # Default 1% risk per trade
    kelly_fraction: float = 0.5  # Half Kelly for safety
    atr_multiplier: float = 2.0  # ATR stop distance multiplier
    volatility_lookback: int = 20  # Days for volatility calculation


@dataclass
class SizingInput:
    """Input data for position sizing calculation."""

    account_equity: Decimal
    entry_price: Decimal
    stop_loss_price: Optional[Decimal] = None
    current_exposure: Decimal = Decimal("0")
    win_rate: Optional[float] = None  # For Kelly criterion
    avg_win: Optional[Decimal] = None  # Average win amount
    avg_loss: Optional[Decimal] = None  # Average loss amount
    atr: Optional[Decimal] = None  # Average True Range
    volatility: Optional[float] = None  # Annualized volatility
    consecutive_wins: int = 0
    consecutive_losses: int = 0


@dataclass
class SizingResult:
    """Result of position sizing calculation."""

    size: Decimal
    size_usd: Decimal
    risk_amount: Decimal
    risk_percent: float
    method: SizingMethod
    leverage_used: float = 1.0
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize notes list."""
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "size": str(self.size),
            "size_usd": str(self.size_usd),
            "risk_amount": str(self.risk_amount),
            "risk_percent": round(self.risk_percent, 4),
            "method": self.method.value,
            "leverage_used": self.leverage_used,
            "notes": self.notes,
        }


@dataclass
class PositionSizer:
    """Calculator for optimal position sizes.

    Features:
    - Multiple sizing methods
    - Risk management integration
    - Leverage calculation
    - Kelly criterion support
    - Volatility-based sizing
    """

    config: SizingConfig = field(default_factory=SizingConfig)

    def calculate_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate position size based on configured method."""
        method = self.config.method

        if method == SizingMethod.FIXED:
            return self._fixed_size(input_data)
        elif method == SizingMethod.PERCENT_EQUITY:
            return self._percent_equity_size(input_data)
        elif method == SizingMethod.PERCENT_RISK:
            return self._percent_risk_size(input_data)
        elif method == SizingMethod.KELLY:
            return self._kelly_size(input_data)
        elif method == SizingMethod.OPTIMAL_F:
            return self._optimal_f_size(input_data)
        elif method == SizingMethod.VOLATILITY:
            return self._volatility_size(input_data)
        elif method == SizingMethod.ATR:
            return self._atr_size(input_data)
        elif method == SizingMethod.MARTINGALE:
            return self._martingale_size(input_data)
        elif method == SizingMethod.ANTI_MARTINGALE:
            return self._anti_martingale_size(input_data)
        else:
            return self._percent_equity_size(input_data)

    def _fixed_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate fixed position size."""
        size = self.config.min_position_size
        if self.config.max_position_size:
            size = min(size, self.config.max_position_size)

        size = self._apply_limits(size, input_data)
        size_usd = size * input_data.entry_price

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=Decimal("0"),
            risk_percent=0.0,
            method=SizingMethod.FIXED,
            notes=["Fixed size used"],
        )

    def _percent_equity_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size as percentage of equity."""
        equity = input_data.account_equity
        price = input_data.entry_price

        # Calculate position value based on equity percentage
        position_value = equity * Decimal(str(self.config.max_position_pct / 100))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price
        risk_amount = size_usd * Decimal("0.01")  # Assume 1% risk

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=risk_amount,
            risk_percent=float(risk_amount / equity * 100) if equity > 0 else 0,
            method=SizingMethod.PERCENT_EQUITY,
            notes=[f"Using {self.config.max_position_pct}% of equity"],
        )

    def _percent_risk_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size based on risk percentage."""
        equity = input_data.account_equity
        price = input_data.entry_price
        stop_loss = input_data.stop_loss_price

        notes = []

        if not stop_loss:
            # Use default risk distance if no stop loss provided
            stop_distance_pct = Decimal("0.02")  # 2% default
            notes.append("Using default 2% stop distance")
        else:
            stop_distance_pct = abs(price - stop_loss) / price

        # Risk amount based on risk level
        risk_pct = self._get_risk_percent()
        risk_amount = equity * Decimal(str(risk_pct / 100))

        # Position size = Risk Amount / Stop Distance %
        if stop_distance_pct > 0:
            size_usd = risk_amount / stop_distance_pct
            size = size_usd / price
        else:
            size = self.config.min_position_size
            size_usd = size * price

        size = self._apply_limits(size, input_data)
        size_usd = size * price

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            method=SizingMethod.PERCENT_RISK,
            notes=notes + [f"Risk {risk_pct}% per trade"],
        )

    def _kelly_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size using Kelly criterion."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        # Default values if not provided
        win_rate = input_data.win_rate or 0.5
        avg_win = input_data.avg_win or Decimal("100")
        avg_loss = input_data.avg_loss or Decimal("100")

        # Kelly formula: f = (p * b - q) / b
        # Where p = win probability, q = loss probability, b = win/loss ratio
        if avg_loss > 0:
            b = float(avg_win / avg_loss)
        else:
            b = 1.0

        p = win_rate
        q = 1 - win_rate

        kelly_pct = (p * b - q) / b if b > 0 else 0

        # Apply Kelly fraction for safety (half Kelly default)
        kelly_pct *= self.config.kelly_fraction

        # Clamp to reasonable bounds
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Max 25%

        position_value = equity * Decimal(str(kelly_pct))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price
        risk_amount = size_usd * Decimal("0.01")

        notes.append(f"Kelly fraction: {kelly_pct * 100:.2f}%")
        notes.append(f"Win rate: {win_rate * 100:.1f}%")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=risk_amount,
            risk_percent=float(kelly_pct * 100),
            method=SizingMethod.KELLY,
            notes=notes,
        )

    def _optimal_f_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size using Optimal f method."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        # Optimal f = largest loss / account
        avg_loss = input_data.avg_loss or Decimal("100")

        if avg_loss > 0:
            # Position size where max loss = f * account
            f = float(avg_loss / equity) if equity > 0 else 0.01
            f = min(f, 0.25)  # Cap at 25%
        else:
            f = 0.01

        position_value = equity * Decimal(str(f))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price

        notes.append(f"Optimal f: {f * 100:.2f}%")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=avg_loss,
            risk_percent=f * 100,
            method=SizingMethod.OPTIMAL_F,
            notes=notes,
        )

    def _volatility_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size based on volatility."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        volatility = input_data.volatility or 0.20  # Default 20% annual vol

        # Lower size for higher volatility
        # Base allocation adjusted by inverse volatility
        base_alloc = self.config.max_position_pct / 100
        vol_adjustment = 0.20 / volatility if volatility > 0 else 1.0  # Target 20% vol
        vol_adjustment = min(2.0, max(0.25, vol_adjustment))  # Clamp

        adjusted_alloc = base_alloc * vol_adjustment
        adjusted_alloc = min(adjusted_alloc, 0.25)  # Max 25%

        position_value = equity * Decimal(str(adjusted_alloc))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price
        risk_amount = size_usd * Decimal(str(volatility / 100))

        notes.append(f"Volatility: {volatility * 100:.1f}%")
        notes.append(f"Vol adjustment: {vol_adjustment:.2f}x")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=risk_amount,
            risk_percent=adjusted_alloc * 100,
            method=SizingMethod.VOLATILITY,
            notes=notes,
        )

    def _atr_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size based on ATR."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        atr = input_data.atr or price * Decimal("0.02")  # Default 2% of price

        # Stop distance = ATR * multiplier
        stop_distance = atr * Decimal(str(self.config.atr_multiplier))
        stop_distance_pct = float(stop_distance / price)

        # Risk amount
        risk_pct = self._get_risk_percent()
        risk_amount = equity * Decimal(str(risk_pct / 100))

        # Position size
        if stop_distance_pct > 0:
            size_usd = risk_amount / Decimal(str(stop_distance_pct))
            size = size_usd / price
        else:
            size = self.config.min_position_size
            size_usd = size * price

        size = self._apply_limits(size, input_data)
        size_usd = size * price

        notes.append(f"ATR: {atr}")
        notes.append(f"Stop distance: {stop_distance_pct * 100:.2f}%")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=risk_amount,
            risk_percent=risk_pct,
            method=SizingMethod.ATR,
            notes=notes,
        )

    def _martingale_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size using Martingale progression."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        # Double after each loss
        consecutive_losses = input_data.consecutive_losses
        multiplier = 2 ** consecutive_losses

        # Base size
        base_pct = 0.01  # 1% base
        position_pct = base_pct * multiplier

        # Cap multiplier
        max_multiplier = 8  # Max 8x
        if multiplier > max_multiplier:
            multiplier = max_multiplier
            position_pct = base_pct * max_multiplier
            notes.append(f"Capped at {max_multiplier}x")

        position_value = equity * Decimal(str(position_pct))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price

        notes.append(f"Consecutive losses: {consecutive_losses}")
        notes.append(f"Multiplier: {multiplier}x")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=size_usd * Decimal("0.01"),
            risk_percent=position_pct * 100,
            method=SizingMethod.MARTINGALE,
            notes=notes,
        )

    def _anti_martingale_size(self, input_data: SizingInput) -> SizingResult:
        """Calculate size using Anti-Martingale progression."""
        equity = input_data.account_equity
        price = input_data.entry_price
        notes = []

        # Increase after wins, reset after loss
        consecutive_wins = input_data.consecutive_wins

        # Progressive increase (1.5x per win)
        multiplier = 1.5 ** consecutive_wins

        # Cap multiplier
        max_multiplier = 4.0  # Max 4x
        if multiplier > max_multiplier:
            multiplier = max_multiplier
            notes.append(f"Capped at {max_multiplier}x")

        base_pct = 0.01  # 1% base
        position_pct = base_pct * multiplier
        position_pct = min(position_pct, 0.10)  # Max 10%

        position_value = equity * Decimal(str(position_pct))
        size = position_value / price

        size = self._apply_limits(size, input_data)
        size_usd = size * price

        notes.append(f"Consecutive wins: {consecutive_wins}")
        notes.append(f"Multiplier: {multiplier:.2f}x")

        return SizingResult(
            size=size,
            size_usd=size_usd,
            risk_amount=size_usd * Decimal("0.01"),
            risk_percent=position_pct * 100,
            method=SizingMethod.ANTI_MARTINGALE,
            notes=notes,
        )

    def _get_risk_percent(self) -> float:
        """Get risk percentage based on risk level."""
        risk_map = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 2.0,
            RiskLevel.VERY_AGGRESSIVE: 5.0,
        }
        return risk_map.get(self.config.risk_level, self.config.risk_per_trade_pct)

    def _apply_limits(self, size: Decimal, input_data: SizingInput) -> Decimal:
        """Apply size limits and constraints."""
        # Min size
        size = max(size, self.config.min_position_size)

        # Max size
        if self.config.max_position_size:
            size = min(size, self.config.max_position_size)

        # Max position percentage
        max_size_from_equity = (
            input_data.account_equity
            * Decimal(str(self.config.max_position_pct / 100))
            / input_data.entry_price
        )
        size = min(size, max_size_from_equity)

        # Max total exposure
        remaining_exposure = (
            input_data.account_equity
            * Decimal(str(self.config.max_total_exposure_pct / 100))
            - input_data.current_exposure
        )
        if remaining_exposure > 0:
            max_size_from_exposure = remaining_exposure / input_data.entry_price
            size = min(size, max_size_from_exposure)
        else:
            size = Decimal("0")

        # Round to reasonable precision
        size = size.quantize(Decimal("0.00001"), rounding=ROUND_DOWN)

        return size

    def calculate_leverage(
        self,
        position_value: Decimal,
        account_equity: Decimal,
    ) -> float:
        """Calculate leverage for a position."""
        if account_equity <= 0:
            return 0.0

        leverage = float(position_value / account_equity)

        if self.config.use_leverage:
            leverage = min(leverage, self.config.max_leverage)
        else:
            leverage = min(leverage, 1.0)

        return round(leverage, 2)

    def validate_size(
        self,
        size: Decimal,
        input_data: SizingInput,
    ) -> Tuple[bool, List[str]]:
        """Validate a position size."""
        errors = []

        # Check minimum
        if size < self.config.min_position_size:
            errors.append(f"Size below minimum: {self.config.min_position_size}")

        # Check maximum
        if self.config.max_position_size and size > self.config.max_position_size:
            errors.append(f"Size above maximum: {self.config.max_position_size}")

        # Check equity percentage
        position_value = size * input_data.entry_price
        equity_pct = float(position_value / input_data.account_equity * 100)
        if equity_pct > self.config.max_position_pct:
            errors.append(f"Exceeds max position %: {self.config.max_position_pct}%")

        # Check total exposure
        total_exposure = input_data.current_exposure + position_value
        exposure_pct = float(total_exposure / input_data.account_equity * 100)
        if exposure_pct > self.config.max_total_exposure_pct:
            errors.append(f"Exceeds max exposure: {self.config.max_total_exposure_pct}%")

        return len(errors) == 0, errors


@dataclass
class DeltaNeutralSizer:
    """Position sizer specifically for delta-neutral strategies."""

    base_sizer: PositionSizer = field(default_factory=PositionSizer)
    variance_pct: float = 15.0  # Size variance for anti-detection

    def calculate_pair_sizes(
        self,
        input_data: SizingInput,
        num_accounts: int,
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """Calculate sizes for long and short accounts.

        Returns:
            Tuple of (long_sizes, short_sizes)
        """
        # Get base size
        result = self.base_sizer.calculate_size(input_data)
        base_size = result.size

        # Split accounts
        num_long = num_accounts // 2
        num_short = num_accounts - num_long

        # Calculate sizes with variance
        long_sizes = []
        short_sizes = []

        total_size = base_size * num_accounts

        for i in range(num_long):
            variance = self._get_variance()
            size = (total_size / num_accounts) * Decimal(str(1 + variance))
            long_sizes.append(size.quantize(Decimal("0.00001")))

        for i in range(num_short):
            variance = self._get_variance()
            size = (total_size / num_accounts) * Decimal(str(1 + variance))
            short_sizes.append(size.quantize(Decimal("0.00001")))

        # Balance total long and short
        long_sizes, short_sizes = self._balance_sizes(long_sizes, short_sizes)

        return long_sizes, short_sizes

    def _get_variance(self) -> float:
        """Get random variance within configured range."""
        import random
        return random.uniform(-self.variance_pct / 100, self.variance_pct / 100)

    def _balance_sizes(
        self,
        long_sizes: List[Decimal],
        short_sizes: List[Decimal],
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """Balance total long and short sizes."""
        total_long = sum(long_sizes)
        total_short = sum(short_sizes)

        if total_long == 0 or total_short == 0:
            return long_sizes, short_sizes

        # Adjust shorter side to match
        if total_long > total_short:
            ratio = total_long / total_short
            short_sizes = [s * ratio for s in short_sizes]
        elif total_short > total_long:
            ratio = total_short / total_long
            long_sizes = [s * ratio for s in long_sizes]

        return long_sizes, short_sizes

    def calculate_rebalance_size(
        self,
        long_exposure: Decimal,
        short_exposure: Decimal,
        target_delta: Decimal = Decimal("0"),
    ) -> Tuple[str, Decimal]:
        """Calculate size needed to rebalance to target delta.

        Returns:
            Tuple of (side, size) where side is 'long' or 'short'
        """
        current_delta = long_exposure - short_exposure
        delta_diff = target_delta - current_delta

        if delta_diff > 0:
            # Need more long
            return "long", abs(delta_diff)
        elif delta_diff < 0:
            # Need more short
            return "short", abs(delta_diff)
        else:
            return "none", Decimal("0")


@dataclass
class RiskAdjustedSizer:
    """Position sizer with dynamic risk adjustment."""

    base_sizer: PositionSizer = field(default_factory=PositionSizer)
    drawdown_threshold: float = 10.0  # % drawdown to trigger reduction
    drawdown_reduction: float = 50.0  # % size reduction during drawdown
    profit_increase_threshold: float = 20.0  # % profit to increase size
    profit_increase: float = 25.0  # % size increase during profit

    def calculate_size(
        self,
        input_data: SizingInput,
        peak_equity: Decimal,
        initial_equity: Decimal,
    ) -> SizingResult:
        """Calculate risk-adjusted position size."""
        # Get base size
        result = self.base_sizer.calculate_size(input_data)

        # Calculate drawdown
        drawdown_pct = float((peak_equity - input_data.account_equity) / peak_equity * 100)

        # Calculate profit
        profit_pct = float(
            (input_data.account_equity - initial_equity) / initial_equity * 100
        )

        # Apply adjustments
        if drawdown_pct > self.drawdown_threshold:
            # Reduce size during drawdown
            reduction = Decimal(str(1 - self.drawdown_reduction / 100))
            result.size *= reduction
            result.size_usd *= reduction
            result.notes.append(f"Drawdown reduction: {self.drawdown_reduction}%")

        elif profit_pct > self.profit_increase_threshold:
            # Increase size during profit
            increase = Decimal(str(1 + self.profit_increase / 100))
            result.size *= increase
            result.size_usd *= increase
            result.notes.append(f"Profit increase: {self.profit_increase}%")

        return result
