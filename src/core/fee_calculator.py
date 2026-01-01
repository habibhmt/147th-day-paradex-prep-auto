"""Fee calculation for trading operations."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeeType(Enum):
    """Types of trading fees."""

    MAKER = "maker"  # Limit orders that add liquidity
    TAKER = "taker"  # Market orders that take liquidity
    FUNDING = "funding"  # Perpetual funding fees
    LIQUIDATION = "liquidation"  # Liquidation penalty
    WITHDRAWAL = "withdrawal"  # Withdrawal fees
    GAS = "gas"  # On-chain gas fees


class FeeTier(Enum):
    """Fee tier levels based on volume."""

    TIER_0 = "tier_0"  # Default/starting tier
    TIER_1 = "tier_1"  # First volume discount
    TIER_2 = "tier_2"  # Second volume discount
    TIER_3 = "tier_3"  # Third volume discount
    TIER_4 = "tier_4"  # Fourth volume discount
    VIP = "vip"  # VIP tier
    MARKET_MAKER = "market_maker"  # Market maker program


@dataclass
class FeeRate:
    """Fee rate configuration."""

    maker: Decimal = Decimal("0.0002")  # 0.02% maker
    taker: Decimal = Decimal("0.0005")  # 0.05% taker
    funding_multiplier: Decimal = Decimal("1.0")  # Funding rate multiplier
    liquidation: Decimal = Decimal("0.005")  # 0.5% liquidation penalty
    min_fee: Decimal = Decimal("0")  # Minimum fee

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "maker": str(self.maker),
            "taker": str(self.taker),
            "funding_multiplier": str(self.funding_multiplier),
            "liquidation": str(self.liquidation),
        }


@dataclass
class FeeTierConfig:
    """Configuration for fee tier thresholds."""

    tier: FeeTier
    volume_threshold: Decimal  # 30-day volume threshold
    maker_rate: Decimal
    taker_rate: Decimal
    rebate: Optional[Decimal] = None  # Maker rebate for high tiers

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "volume_threshold": str(self.volume_threshold),
            "maker_rate": str(self.maker_rate),
            "taker_rate": str(self.taker_rate),
            "rebate": str(self.rebate) if self.rebate else None,
        }


@dataclass
class FeeEstimate:
    """Estimated fees for a trade."""

    fee_type: FeeType
    fee_amount: Decimal
    fee_rate: Decimal
    notional_value: Decimal
    fee_asset: str = "USDC"
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize notes list."""
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fee_type": self.fee_type.value,
            "fee_amount": str(self.fee_amount),
            "fee_rate": str(self.fee_rate),
            "notional_value": str(self.notional_value),
            "fee_asset": self.fee_asset,
            "notes": self.notes,
        }


@dataclass
class TotalFeeBreakdown:
    """Complete breakdown of all fees for a trade."""

    total_fee: Decimal
    maker_fee: Optional[Decimal] = None
    taker_fee: Optional[Decimal] = None
    funding_fee: Optional[Decimal] = None
    gas_fee: Optional[Decimal] = None
    estimates: List[FeeEstimate] = field(default_factory=list)
    tier: FeeTier = FeeTier.TIER_0
    volume_30d: Decimal = Decimal("0")

    def __post_init__(self):
        """Initialize estimates list."""
        if self.estimates is None:
            self.estimates = []

    @property
    def total_fee_pct(self) -> float:
        """Calculate total fee as percentage of notional."""
        if not self.estimates:
            return 0.0
        total_notional = sum(e.notional_value for e in self.estimates)
        if total_notional == 0:
            return 0.0
        return float(self.total_fee / total_notional * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_fee": str(self.total_fee),
            "total_fee_pct": round(self.total_fee_pct, 6),
            "maker_fee": str(self.maker_fee) if self.maker_fee else None,
            "taker_fee": str(self.taker_fee) if self.taker_fee else None,
            "funding_fee": str(self.funding_fee) if self.funding_fee else None,
            "gas_fee": str(self.gas_fee) if self.gas_fee else None,
            "tier": self.tier.value,
            "estimates": [e.to_dict() for e in self.estimates],
        }


@dataclass
class FundingPayment:
    """Funding payment details."""

    market: str
    rate: Decimal
    position_size: Decimal
    notional_value: Decimal
    payment: Decimal  # Positive = receive, Negative = pay
    next_funding_time: float
    is_long: bool

    @property
    def is_paying(self) -> bool:
        """Check if position is paying funding."""
        return self.payment < 0

    @property
    def annualized_rate(self) -> Decimal:
        """Estimate annualized funding rate."""
        # Funding every 8 hours = 3x per day = 1095x per year
        return self.rate * Decimal("1095")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market": self.market,
            "rate": str(self.rate),
            "position_size": str(self.position_size),
            "notional_value": str(self.notional_value),
            "payment": str(self.payment),
            "is_paying": self.is_paying,
            "annualized_rate": str(self.annualized_rate),
        }


@dataclass
class FeeCalculator:
    """Calculator for trading fees.

    Features:
    - Multiple fee types
    - Volume-based tier discounts
    - Funding rate calculations
    - Fee projections
    """

    base_rates: FeeRate = field(default_factory=FeeRate)
    tier_configs: List[FeeTierConfig] = field(default_factory=list)
    current_tier: FeeTier = FeeTier.TIER_0
    volume_30d: Decimal = Decimal("0")

    def __post_init__(self):
        """Initialize tier configurations."""
        if not self.tier_configs:
            self.tier_configs = self._default_tier_configs()

    def _default_tier_configs(self) -> List[FeeTierConfig]:
        """Create default tier configurations."""
        return [
            FeeTierConfig(
                tier=FeeTier.TIER_0,
                volume_threshold=Decimal("0"),
                maker_rate=Decimal("0.0002"),  # 0.02%
                taker_rate=Decimal("0.0005"),  # 0.05%
            ),
            FeeTierConfig(
                tier=FeeTier.TIER_1,
                volume_threshold=Decimal("100000"),  # $100k
                maker_rate=Decimal("0.00018"),
                taker_rate=Decimal("0.00045"),
            ),
            FeeTierConfig(
                tier=FeeTier.TIER_2,
                volume_threshold=Decimal("1000000"),  # $1M
                maker_rate=Decimal("0.00015"),
                taker_rate=Decimal("0.0004"),
            ),
            FeeTierConfig(
                tier=FeeTier.TIER_3,
                volume_threshold=Decimal("10000000"),  # $10M
                maker_rate=Decimal("0.0001"),
                taker_rate=Decimal("0.00035"),
            ),
            FeeTierConfig(
                tier=FeeTier.TIER_4,
                volume_threshold=Decimal("50000000"),  # $50M
                maker_rate=Decimal("0.00005"),
                taker_rate=Decimal("0.0003"),
            ),
            FeeTierConfig(
                tier=FeeTier.VIP,
                volume_threshold=Decimal("100000000"),  # $100M
                maker_rate=Decimal("0"),  # No maker fee
                taker_rate=Decimal("0.00025"),
                rebate=Decimal("0.0001"),  # 0.01% rebate
            ),
            FeeTierConfig(
                tier=FeeTier.MARKET_MAKER,
                volume_threshold=Decimal("0"),  # By application
                maker_rate=Decimal("-0.0001"),  # Negative = rebate
                taker_rate=Decimal("0.0002"),
            ),
        ]

    def set_volume(self, volume_30d: Decimal) -> FeeTier:
        """Set 30-day volume and update tier."""
        self.volume_30d = volume_30d
        self.current_tier = self._determine_tier(volume_30d)
        return self.current_tier

    def _determine_tier(self, volume: Decimal) -> FeeTier:
        """Determine fee tier based on volume."""
        current_tier = FeeTier.TIER_0

        for config in self.tier_configs:
            if config.tier == FeeTier.MARKET_MAKER:
                continue  # Skip market maker (by application only)
            if volume >= config.volume_threshold:
                current_tier = config.tier

        return current_tier

    def get_tier_config(self, tier: FeeTier = None) -> Optional[FeeTierConfig]:
        """Get configuration for a tier."""
        tier = tier or self.current_tier
        for config in self.tier_configs:
            if config.tier == tier:
                return config
        return None

    def calculate_maker_fee(
        self,
        notional: Decimal,
        tier: FeeTier = None,
    ) -> FeeEstimate:
        """Calculate maker fee for a trade."""
        tier = tier or self.current_tier
        config = self.get_tier_config(tier)

        if config:
            rate = config.maker_rate
        else:
            rate = self.base_rates.maker

        fee = notional * rate
        fee = fee.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        # Apply minimum fee
        if fee > 0 and fee < self.base_rates.min_fee:
            fee = self.base_rates.min_fee

        notes = []
        if rate < 0:
            notes.append("Maker rebate applied")
        if tier != FeeTier.TIER_0:
            notes.append(f"Tier discount: {tier.value}")

        return FeeEstimate(
            fee_type=FeeType.MAKER,
            fee_amount=fee,
            fee_rate=rate,
            notional_value=notional,
            notes=notes,
        )

    def calculate_taker_fee(
        self,
        notional: Decimal,
        tier: FeeTier = None,
    ) -> FeeEstimate:
        """Calculate taker fee for a trade."""
        tier = tier or self.current_tier
        config = self.get_tier_config(tier)

        if config:
            rate = config.taker_rate
        else:
            rate = self.base_rates.taker

        fee = notional * rate
        fee = fee.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        # Apply minimum fee
        if fee < self.base_rates.min_fee:
            fee = self.base_rates.min_fee

        notes = []
        if tier != FeeTier.TIER_0:
            notes.append(f"Tier discount: {tier.value}")

        return FeeEstimate(
            fee_type=FeeType.TAKER,
            fee_amount=fee,
            fee_rate=rate,
            notional_value=notional,
            notes=notes,
        )

    def calculate_funding_fee(
        self,
        notional: Decimal,
        funding_rate: Decimal,
        is_long: bool,
    ) -> FeeEstimate:
        """Calculate funding fee for a position.

        Funding is paid from longs to shorts when rate is positive,
        and from shorts to longs when rate is negative.
        """
        # Apply funding multiplier
        rate = funding_rate * self.base_rates.funding_multiplier

        # Long positions pay positive funding, receive negative
        # Short positions receive positive funding, pay negative
        if is_long:
            fee = notional * rate  # Positive = pay, Negative = receive
        else:
            fee = -notional * rate  # Opposite for shorts

        fee = fee.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        notes = []
        if fee > 0:
            notes.append("Paying funding")
        elif fee < 0:
            notes.append("Receiving funding")
            fee = abs(fee)  # Return absolute value, note indicates direction
        else:
            notes.append("Neutral funding")

        return FeeEstimate(
            fee_type=FeeType.FUNDING,
            fee_amount=fee,
            fee_rate=rate,
            notional_value=notional,
            notes=notes,
        )

    def calculate_liquidation_fee(
        self,
        notional: Decimal,
    ) -> FeeEstimate:
        """Calculate liquidation penalty fee."""
        rate = self.base_rates.liquidation
        fee = notional * rate
        fee = fee.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        return FeeEstimate(
            fee_type=FeeType.LIQUIDATION,
            fee_amount=fee,
            fee_rate=rate,
            notional_value=notional,
            notes=["Liquidation penalty"],
        )

    def estimate_trade_fees(
        self,
        notional: Decimal,
        is_maker: bool = False,
        include_funding: bool = False,
        funding_rate: Decimal = Decimal("0"),
        is_long: bool = True,
        tier: FeeTier = None,
    ) -> TotalFeeBreakdown:
        """Estimate total fees for a trade."""
        tier = tier or self.current_tier
        estimates = []
        total = Decimal("0")

        # Trading fee
        if is_maker:
            fee_estimate = self.calculate_maker_fee(notional, tier)
            estimates.append(fee_estimate)
            maker_fee = fee_estimate.fee_amount
            taker_fee = None
        else:
            fee_estimate = self.calculate_taker_fee(notional, tier)
            estimates.append(fee_estimate)
            taker_fee = fee_estimate.fee_amount
            maker_fee = None

        total += fee_estimate.fee_amount

        # Funding fee (if requested)
        funding_fee = None
        if include_funding and funding_rate != 0:
            funding_estimate = self.calculate_funding_fee(
                notional, funding_rate, is_long
            )
            estimates.append(funding_estimate)
            funding_fee = funding_estimate.fee_amount
            if "Paying" in funding_estimate.notes[0]:
                total += funding_fee

        return TotalFeeBreakdown(
            total_fee=total,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            funding_fee=funding_fee,
            estimates=estimates,
            tier=tier,
            volume_30d=self.volume_30d,
        )

    def estimate_round_trip_fees(
        self,
        notional: Decimal,
        entry_is_maker: bool = False,
        exit_is_maker: bool = False,
        hold_periods: int = 0,  # Number of 8-hour funding periods
        funding_rate: Decimal = Decimal("0"),
        is_long: bool = True,
    ) -> TotalFeeBreakdown:
        """Estimate total fees for entry, hold, and exit."""
        estimates = []
        total = Decimal("0")

        # Entry fee
        if entry_is_maker:
            entry = self.calculate_maker_fee(notional)
        else:
            entry = self.calculate_taker_fee(notional)
        entry.notes.append("Entry trade")
        estimates.append(entry)
        total += entry.fee_amount

        # Funding fees
        funding_total = Decimal("0")
        if hold_periods > 0 and funding_rate != 0:
            for _ in range(hold_periods):
                funding = self.calculate_funding_fee(notional, funding_rate, is_long)
                if "Paying" in funding.notes[0]:
                    funding_total += funding.fee_amount

            if funding_total > 0:
                funding_estimate = FeeEstimate(
                    fee_type=FeeType.FUNDING,
                    fee_amount=funding_total,
                    fee_rate=funding_rate * hold_periods,
                    notional_value=notional,
                    notes=[f"Total funding over {hold_periods} periods"],
                )
                estimates.append(funding_estimate)
                total += funding_total

        # Exit fee
        if exit_is_maker:
            exit_fee = self.calculate_maker_fee(notional)
        else:
            exit_fee = self.calculate_taker_fee(notional)
        exit_fee.notes.append("Exit trade")
        estimates.append(exit_fee)
        total += exit_fee.fee_amount

        return TotalFeeBreakdown(
            total_fee=total,
            maker_fee=entry.fee_amount if entry_is_maker else exit_fee.fee_amount if exit_is_maker else None,
            taker_fee=entry.fee_amount if not entry_is_maker else exit_fee.fee_amount if not exit_is_maker else None,
            funding_fee=funding_total if funding_total > 0 else None,
            estimates=estimates,
            tier=self.current_tier,
            volume_30d=self.volume_30d,
        )

    def calculate_breakeven_move(
        self,
        notional: Decimal,
        entry_is_maker: bool = False,
        exit_is_maker: bool = False,
    ) -> Decimal:
        """Calculate price move needed to break even on fees."""
        # Get fees for round trip
        entry_fee = self.calculate_maker_fee(notional) if entry_is_maker else self.calculate_taker_fee(notional)
        exit_fee = self.calculate_maker_fee(notional) if exit_is_maker else self.calculate_taker_fee(notional)

        total_fee = entry_fee.fee_amount + exit_fee.fee_amount
        breakeven_pct = total_fee / notional

        return breakeven_pct.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

    def project_daily_fees(
        self,
        daily_volume: Decimal,
        maker_ratio: float = 0.5,  # Ratio of maker orders
        avg_position: Decimal = Decimal("0"),
        avg_funding_rate: Decimal = Decimal("0"),
        is_long: bool = True,
    ) -> Dict[str, Decimal]:
        """Project daily fee costs."""
        # Trading fees
        maker_volume = daily_volume * Decimal(str(maker_ratio))
        taker_volume = daily_volume * Decimal(str(1 - maker_ratio))

        maker_fee = self.calculate_maker_fee(maker_volume).fee_amount
        taker_fee = self.calculate_taker_fee(taker_volume).fee_amount

        # Funding fees (3 payments per day)
        funding_fee = Decimal("0")
        if avg_position > 0 and avg_funding_rate != 0:
            for _ in range(3):
                f = self.calculate_funding_fee(avg_position, avg_funding_rate, is_long)
                if "Paying" in f.notes[0]:
                    funding_fee += f.fee_amount

        return {
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "funding_fee": funding_fee,
            "total_fee": maker_fee + taker_fee + funding_fee,
            "volume": daily_volume,
        }

    def project_monthly_fees(
        self,
        daily_volume: Decimal,
        maker_ratio: float = 0.5,
        avg_position: Decimal = Decimal("0"),
        avg_funding_rate: Decimal = Decimal("0"),
        is_long: bool = True,
    ) -> Dict[str, Decimal]:
        """Project monthly fee costs."""
        daily = self.project_daily_fees(
            daily_volume, maker_ratio, avg_position, avg_funding_rate, is_long
        )

        return {
            "maker_fee": daily["maker_fee"] * 30,
            "taker_fee": daily["taker_fee"] * 30,
            "funding_fee": daily["funding_fee"] * 30,
            "total_fee": daily["total_fee"] * 30,
            "volume": daily_volume * 30,
        }


@dataclass
class FundingCalculator:
    """Calculator specifically for funding payments."""

    def calculate_funding_payment(
        self,
        market: str,
        position_size: Decimal,
        entry_price: Decimal,
        funding_rate: Decimal,
        is_long: bool,
        next_funding_time: float,
    ) -> FundingPayment:
        """Calculate funding payment for a position."""
        notional = position_size * entry_price

        # Long pays positive funding, receives negative
        # Short receives positive funding, pays negative
        if is_long:
            payment = -notional * funding_rate  # Negative = paying
        else:
            payment = notional * funding_rate  # Positive = receiving

        payment = payment.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

        return FundingPayment(
            market=market,
            rate=funding_rate,
            position_size=position_size,
            notional_value=notional,
            payment=payment,
            next_funding_time=next_funding_time,
            is_long=is_long,
        )

    def estimate_daily_funding(
        self,
        market: str,
        position_size: Decimal,
        entry_price: Decimal,
        avg_funding_rate: Decimal,
        is_long: bool,
    ) -> Decimal:
        """Estimate daily funding cost/income."""
        notional = position_size * entry_price

        # 3 funding payments per day (every 8 hours)
        daily_rate = avg_funding_rate * 3

        if is_long:
            daily_payment = -notional * daily_rate
        else:
            daily_payment = notional * daily_rate

        return daily_payment.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

    def estimate_annual_funding(
        self,
        market: str,
        position_size: Decimal,
        entry_price: Decimal,
        avg_funding_rate: Decimal,
        is_long: bool,
    ) -> Decimal:
        """Estimate annual funding cost/income."""
        daily = self.estimate_daily_funding(
            market, position_size, entry_price, avg_funding_rate, is_long
        )
        return daily * 365

    def find_optimal_side(
        self,
        market: str,
        position_size: Decimal,
        entry_price: Decimal,
        funding_rate: Decimal,
    ) -> Tuple[str, Decimal]:
        """Find optimal side based on funding rate.

        Returns:
            Tuple of (side, expected_payment)
            - side: 'long' or 'short'
            - payment: positive = income, negative = cost
        """
        long_payment = self.calculate_funding_payment(
            market, position_size, entry_price, funding_rate, True, 0
        )
        short_payment = self.calculate_funding_payment(
            market, position_size, entry_price, funding_rate, False, 0
        )

        # Choose side with higher payment (more income or less cost)
        if long_payment.payment >= short_payment.payment:
            return "long", long_payment.payment
        else:
            return "short", short_payment.payment


@dataclass
class FeeOptimizer:
    """Optimizer for minimizing trading fees."""

    calculator: FeeCalculator = field(default_factory=FeeCalculator)

    def optimize_order_type(
        self,
        notional: Decimal,
        urgency: float = 0.5,  # 0 = patient, 1 = urgent
    ) -> Tuple[str, FeeEstimate]:
        """Recommend order type based on urgency and fees.

        Returns:
            Tuple of (order_type, fee_estimate)
        """
        maker_fee = self.calculator.calculate_maker_fee(notional)
        taker_fee = self.calculator.calculate_taker_fee(notional)

        # Consider execution probability and fees
        # Higher urgency = more likely to use taker
        if urgency > 0.7:
            return "market", taker_fee
        elif urgency < 0.3:
            return "limit", maker_fee
        else:
            # Compare fees and decide
            if maker_fee.fee_amount < taker_fee.fee_amount * Decimal("0.5"):
                return "limit", maker_fee
            else:
                return "market", taker_fee

    def find_breakeven_volume_for_tier(
        self,
        target_tier: FeeTier,
        current_volume: Decimal = Decimal("0"),
    ) -> Decimal:
        """Calculate volume needed to reach a tier."""
        config = self.calculator.get_tier_config(target_tier)
        if not config:
            return Decimal("0")

        return config.volume_threshold - current_volume

    def estimate_tier_savings(
        self,
        monthly_volume: Decimal,
        current_tier: FeeTier = FeeTier.TIER_0,
        target_tier: FeeTier = FeeTier.TIER_1,
        maker_ratio: float = 0.5,
    ) -> Decimal:
        """Estimate monthly savings from upgrading tiers."""
        current_config = self.calculator.get_tier_config(current_tier)
        target_config = self.calculator.get_tier_config(target_tier)

        if not current_config or not target_config:
            return Decimal("0")

        maker_volume = monthly_volume * Decimal(str(maker_ratio))
        taker_volume = monthly_volume * Decimal(str(1 - maker_ratio))

        current_fees = (
            maker_volume * current_config.maker_rate +
            taker_volume * current_config.taker_rate
        )
        target_fees = (
            maker_volume * target_config.maker_rate +
            taker_volume * target_config.taker_rate
        )

        savings = current_fees - target_fees
        return savings.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# Default calculator instance
_default_calculator: Optional[FeeCalculator] = None


def get_fee_calculator() -> FeeCalculator:
    """Get or create default fee calculator."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = FeeCalculator()
    return _default_calculator


def reset_fee_calculator() -> None:
    """Reset default fee calculator."""
    global _default_calculator
    _default_calculator = None
