"""Unit tests for Fee Calculator."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.core.fee_calculator import (
    FeeType,
    FeeTier,
    FeeRate,
    FeeTierConfig,
    FeeEstimate,
    TotalFeeBreakdown,
    FundingPayment,
    FeeCalculator,
    FundingCalculator,
    FeeOptimizer,
    get_fee_calculator,
    reset_fee_calculator,
)


class TestFeeType:
    """Tests for FeeType enum."""

    def test_fee_type_values(self):
        """Should have expected fee types."""
        assert FeeType.MAKER.value == "maker"
        assert FeeType.TAKER.value == "taker"
        assert FeeType.FUNDING.value == "funding"
        assert FeeType.LIQUIDATION.value == "liquidation"
        assert FeeType.WITHDRAWAL.value == "withdrawal"
        assert FeeType.GAS.value == "gas"


class TestFeeTier:
    """Tests for FeeTier enum."""

    def test_tier_values(self):
        """Should have expected tier values."""
        assert FeeTier.TIER_0.value == "tier_0"
        assert FeeTier.TIER_1.value == "tier_1"
        assert FeeTier.VIP.value == "vip"
        assert FeeTier.MARKET_MAKER.value == "market_maker"


class TestFeeRate:
    """Tests for FeeRate dataclass."""

    def test_default_rates(self):
        """Should have default rate values."""
        rate = FeeRate()

        assert rate.maker == Decimal("0.0002")
        assert rate.taker == Decimal("0.0005")
        assert rate.liquidation == Decimal("0.005")

    def test_custom_rates(self):
        """Should accept custom rates."""
        rate = FeeRate(
            maker=Decimal("0.0001"),
            taker=Decimal("0.0003"),
        )

        assert rate.maker == Decimal("0.0001")
        assert rate.taker == Decimal("0.0003")

    def test_to_dict(self):
        """Should convert to dictionary."""
        rate = FeeRate()
        d = rate.to_dict()

        assert "maker" in d
        assert "taker" in d


class TestFeeTierConfig:
    """Tests for FeeTierConfig dataclass."""

    def test_create_config(self):
        """Should create tier config."""
        config = FeeTierConfig(
            tier=FeeTier.TIER_1,
            volume_threshold=Decimal("100000"),
            maker_rate=Decimal("0.00018"),
            taker_rate=Decimal("0.00045"),
        )

        assert config.tier == FeeTier.TIER_1
        assert config.volume_threshold == Decimal("100000")

    def test_config_with_rebate(self):
        """Should include rebate."""
        config = FeeTierConfig(
            tier=FeeTier.VIP,
            volume_threshold=Decimal("100000000"),
            maker_rate=Decimal("0"),
            taker_rate=Decimal("0.00025"),
            rebate=Decimal("0.0001"),
        )

        assert config.rebate == Decimal("0.0001")

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = FeeTierConfig(
            tier=FeeTier.TIER_1,
            volume_threshold=Decimal("100000"),
            maker_rate=Decimal("0.00018"),
            taker_rate=Decimal("0.00045"),
        )

        d = config.to_dict()

        assert d["tier"] == "tier_1"
        assert d["volume_threshold"] == "100000"


class TestFeeEstimate:
    """Tests for FeeEstimate dataclass."""

    def test_create_estimate(self):
        """Should create fee estimate."""
        estimate = FeeEstimate(
            fee_type=FeeType.MAKER,
            fee_amount=Decimal("10"),
            fee_rate=Decimal("0.0002"),
            notional_value=Decimal("50000"),
        )

        assert estimate.fee_amount == Decimal("10")
        assert estimate.fee_asset == "USDC"

    def test_estimate_with_notes(self):
        """Should include notes."""
        estimate = FeeEstimate(
            fee_type=FeeType.TAKER,
            fee_amount=Decimal("25"),
            fee_rate=Decimal("0.0005"),
            notional_value=Decimal("50000"),
            notes=["Tier discount applied"],
        )

        assert "Tier discount applied" in estimate.notes

    def test_to_dict(self):
        """Should convert to dictionary."""
        estimate = FeeEstimate(
            fee_type=FeeType.MAKER,
            fee_amount=Decimal("10"),
            fee_rate=Decimal("0.0002"),
            notional_value=Decimal("50000"),
        )

        d = estimate.to_dict()

        assert d["fee_type"] == "maker"
        assert d["fee_amount"] == "10"


class TestTotalFeeBreakdown:
    """Tests for TotalFeeBreakdown dataclass."""

    def test_create_breakdown(self):
        """Should create breakdown."""
        breakdown = TotalFeeBreakdown(
            total_fee=Decimal("35"),
            maker_fee=Decimal("10"),
            taker_fee=Decimal("25"),
        )

        assert breakdown.total_fee == Decimal("35")

    def test_total_fee_pct(self):
        """Should calculate fee percentage."""
        breakdown = TotalFeeBreakdown(
            total_fee=Decimal("50"),
            estimates=[
                FeeEstimate(
                    fee_type=FeeType.TAKER,
                    fee_amount=Decimal("50"),
                    fee_rate=Decimal("0.0005"),
                    notional_value=Decimal("100000"),
                )
            ],
        )

        assert breakdown.total_fee_pct == 0.05  # 0.05%

    def test_to_dict(self):
        """Should convert to dictionary."""
        breakdown = TotalFeeBreakdown(
            total_fee=Decimal("35"),
            tier=FeeTier.TIER_1,
        )

        d = breakdown.to_dict()

        assert d["total_fee"] == "35"
        assert d["tier"] == "tier_1"


class TestFundingPayment:
    """Tests for FundingPayment dataclass."""

    def test_create_payment(self):
        """Should create funding payment."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            rate=Decimal("0.0001"),
            position_size=Decimal("1"),
            notional_value=Decimal("50000"),
            payment=Decimal("-5"),
            next_funding_time=1000000,
            is_long=True,
        )

        assert payment.market == "BTC-USD-PERP"
        assert payment.is_paying is True

    def test_receiving_payment(self):
        """Should detect receiving funding."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            rate=Decimal("-0.0001"),
            position_size=Decimal("1"),
            notional_value=Decimal("50000"),
            payment=Decimal("5"),
            next_funding_time=1000000,
            is_long=True,
        )

        assert payment.is_paying is False

    def test_annualized_rate(self):
        """Should calculate annualized rate."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            rate=Decimal("0.0001"),  # 0.01% per 8 hours
            position_size=Decimal("1"),
            notional_value=Decimal("50000"),
            payment=Decimal("-5"),
            next_funding_time=1000000,
            is_long=True,
        )

        # 0.01% * 1095 = ~10.95% annual
        assert payment.annualized_rate == Decimal("0.1095")

    def test_to_dict(self):
        """Should convert to dictionary."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            rate=Decimal("0.0001"),
            position_size=Decimal("1"),
            notional_value=Decimal("50000"),
            payment=Decimal("-5"),
            next_funding_time=1000000,
            is_long=True,
        )

        d = payment.to_dict()

        assert d["market"] == "BTC-USD-PERP"
        assert d["is_paying"] is True


class TestFeeCalculator:
    """Tests for FeeCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create fee calculator."""
        return FeeCalculator()

    def test_default_tier_configs(self, calculator):
        """Should have default tier configurations."""
        assert len(calculator.tier_configs) > 0

        # Check first tier
        tier_0 = calculator.get_tier_config(FeeTier.TIER_0)
        assert tier_0 is not None
        assert tier_0.maker_rate == Decimal("0.0002")

    def test_set_volume_updates_tier(self, calculator):
        """Should update tier based on volume."""
        # Low volume = TIER_0
        tier = calculator.set_volume(Decimal("50000"))
        assert tier == FeeTier.TIER_0

        # High volume = higher tier
        tier = calculator.set_volume(Decimal("1500000"))
        assert tier in [FeeTier.TIER_2, FeeTier.TIER_3]

    def test_calculate_maker_fee(self, calculator):
        """Should calculate maker fee correctly."""
        estimate = calculator.calculate_maker_fee(Decimal("100000"))

        assert estimate.fee_type == FeeType.MAKER
        # 0.02% of 100000 = 20
        assert estimate.fee_amount == Decimal("20")

    def test_calculate_taker_fee(self, calculator):
        """Should calculate taker fee correctly."""
        estimate = calculator.calculate_taker_fee(Decimal("100000"))

        assert estimate.fee_type == FeeType.TAKER
        # 0.05% of 100000 = 50
        assert estimate.fee_amount == Decimal("50")

    def test_tier_discount_applied(self, calculator):
        """Should apply tier discount."""
        # Set to higher tier
        calculator.set_volume(Decimal("2000000"))

        estimate = calculator.calculate_taker_fee(Decimal("100000"))

        # Should be less than default 50
        assert estimate.fee_amount < Decimal("50")
        assert "Tier discount" in estimate.notes[0]

    def test_calculate_funding_fee_long_positive(self, calculator):
        """Should calculate positive funding for long."""
        estimate = calculator.calculate_funding_fee(
            notional=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        assert estimate.fee_type == FeeType.FUNDING
        # Long pays positive funding
        assert "Paying" in estimate.notes[0]
        assert estimate.fee_amount == Decimal("10")

    def test_calculate_funding_fee_short_positive(self, calculator):
        """Should calculate positive funding for short."""
        estimate = calculator.calculate_funding_fee(
            notional=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            is_long=False,
        )

        # Short receives positive funding
        assert "Receiving" in estimate.notes[0]

    def test_calculate_funding_fee_negative(self, calculator):
        """Should handle negative funding rate."""
        estimate = calculator.calculate_funding_fee(
            notional=Decimal("100000"),
            funding_rate=Decimal("-0.0001"),
            is_long=True,
        )

        # Long receives negative funding
        assert "Receiving" in estimate.notes[0]

    def test_calculate_liquidation_fee(self, calculator):
        """Should calculate liquidation penalty."""
        estimate = calculator.calculate_liquidation_fee(Decimal("100000"))

        assert estimate.fee_type == FeeType.LIQUIDATION
        # 0.5% of 100000 = 500
        assert estimate.fee_amount == Decimal("500")

    def test_estimate_trade_fees_taker(self, calculator):
        """Should estimate taker trade fees."""
        breakdown = calculator.estimate_trade_fees(
            notional=Decimal("100000"),
            is_maker=False,
        )

        assert breakdown.total_fee == Decimal("50")
        assert breakdown.taker_fee == Decimal("50")
        assert breakdown.maker_fee is None

    def test_estimate_trade_fees_maker(self, calculator):
        """Should estimate maker trade fees."""
        breakdown = calculator.estimate_trade_fees(
            notional=Decimal("100000"),
            is_maker=True,
        )

        assert breakdown.total_fee == Decimal("20")
        assert breakdown.maker_fee == Decimal("20")
        assert breakdown.taker_fee is None

    def test_estimate_trade_fees_with_funding(self, calculator):
        """Should include funding in estimate."""
        breakdown = calculator.estimate_trade_fees(
            notional=Decimal("100000"),
            is_maker=False,
            include_funding=True,
            funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        assert breakdown.funding_fee is not None
        assert breakdown.total_fee > Decimal("50")  # Trading + funding

    def test_estimate_round_trip_fees(self, calculator):
        """Should estimate round trip fees."""
        breakdown = calculator.estimate_round_trip_fees(
            notional=Decimal("100000"),
            entry_is_maker=False,
            exit_is_maker=True,
        )

        # Taker entry (50) + Maker exit (20) = 70
        assert breakdown.total_fee == Decimal("70")
        assert len(breakdown.estimates) == 2

    def test_round_trip_with_funding(self, calculator):
        """Should include funding in round trip."""
        breakdown = calculator.estimate_round_trip_fees(
            notional=Decimal("100000"),
            entry_is_maker=False,
            exit_is_maker=False,
            hold_periods=3,  # 3 funding periods
            funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        # 50 + 50 + 30 (3 * 10 funding) = 130
        assert breakdown.total_fee == Decimal("130")

    def test_calculate_breakeven_move(self, calculator):
        """Should calculate breakeven price move."""
        breakeven = calculator.calculate_breakeven_move(
            notional=Decimal("100000"),
            entry_is_maker=False,
            exit_is_maker=False,
        )

        # Two taker fees = 0.1% total
        assert breakeven == Decimal("0.001")

    def test_project_daily_fees(self, calculator):
        """Should project daily fees."""
        projection = calculator.project_daily_fees(
            daily_volume=Decimal("1000000"),
            maker_ratio=0.5,
        )

        assert "maker_fee" in projection
        assert "taker_fee" in projection
        assert "total_fee" in projection
        assert projection["total_fee"] == projection["maker_fee"] + projection["taker_fee"]

    def test_project_daily_fees_with_funding(self, calculator):
        """Should include funding in daily projection."""
        projection = calculator.project_daily_fees(
            daily_volume=Decimal("1000000"),
            maker_ratio=0.5,
            avg_position=Decimal("100000"),
            avg_funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        assert projection["funding_fee"] > 0

    def test_project_monthly_fees(self, calculator):
        """Should project monthly fees."""
        projection = calculator.project_monthly_fees(
            daily_volume=Decimal("1000000"),
            maker_ratio=0.5,
        )

        # Monthly should be 30x daily
        daily = calculator.project_daily_fees(Decimal("1000000"), 0.5)
        assert projection["total_fee"] == daily["total_fee"] * 30


class TestFundingCalculator:
    """Tests for FundingCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create funding calculator."""
        return FundingCalculator()

    def test_calculate_funding_payment_long(self, calculator):
        """Should calculate funding for long position."""
        payment = calculator.calculate_funding_payment(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            funding_rate=Decimal("0.0001"),
            is_long=True,
            next_funding_time=1000000,
        )

        assert payment.market == "BTC-USD-PERP"
        assert payment.is_paying is True
        assert payment.payment == Decimal("-5")  # 50000 * 0.0001

    def test_calculate_funding_payment_short(self, calculator):
        """Should calculate funding for short position."""
        payment = calculator.calculate_funding_payment(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            funding_rate=Decimal("0.0001"),
            is_long=False,
            next_funding_time=1000000,
        )

        assert payment.is_paying is False
        assert payment.payment == Decimal("5")  # Receiving

    def test_estimate_daily_funding(self, calculator):
        """Should estimate daily funding."""
        daily = calculator.estimate_daily_funding(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            avg_funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        # 3 payments * 5 = 15
        assert daily == Decimal("-15")

    def test_estimate_annual_funding(self, calculator):
        """Should estimate annual funding."""
        annual = calculator.estimate_annual_funding(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            avg_funding_rate=Decimal("0.0001"),
            is_long=True,
        )

        # 365 * 15 = 5475
        assert annual == Decimal("-5475")

    def test_find_optimal_side_positive_rate(self, calculator):
        """Should recommend short for positive funding."""
        side, payment = calculator.find_optimal_side(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            funding_rate=Decimal("0.0001"),
        )

        # Positive funding = shorts receive
        assert side == "short"
        assert payment > 0

    def test_find_optimal_side_negative_rate(self, calculator):
        """Should recommend long for negative funding."""
        side, payment = calculator.find_optimal_side(
            market="BTC-USD-PERP",
            position_size=Decimal("1"),
            entry_price=Decimal("50000"),
            funding_rate=Decimal("-0.0001"),
        )

        # Negative funding = longs receive
        assert side == "long"
        assert payment > 0


class TestFeeOptimizer:
    """Tests for FeeOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create fee optimizer."""
        return FeeOptimizer()

    def test_optimize_order_type_urgent(self, optimizer):
        """Should recommend market order for urgent trades."""
        order_type, estimate = optimizer.optimize_order_type(
            notional=Decimal("100000"),
            urgency=0.9,
        )

        assert order_type == "market"
        assert estimate.fee_type == FeeType.TAKER

    def test_optimize_order_type_patient(self, optimizer):
        """Should recommend limit order for patient trades."""
        order_type, estimate = optimizer.optimize_order_type(
            notional=Decimal("100000"),
            urgency=0.1,
        )

        assert order_type == "limit"
        assert estimate.fee_type == FeeType.MAKER

    def test_find_breakeven_volume_for_tier(self, optimizer):
        """Should calculate volume needed for tier."""
        volume_needed = optimizer.find_breakeven_volume_for_tier(
            target_tier=FeeTier.TIER_1,
            current_volume=Decimal("50000"),
        )

        assert volume_needed == Decimal("50000")  # 100000 - 50000

    def test_estimate_tier_savings(self, optimizer):
        """Should estimate tier upgrade savings."""
        savings = optimizer.estimate_tier_savings(
            monthly_volume=Decimal("1000000"),
            current_tier=FeeTier.TIER_0,
            target_tier=FeeTier.TIER_1,
            maker_ratio=0.5,
        )

        assert savings > Decimal("0")


class TestGlobalCalculator:
    """Tests for global calculator functions."""

    def test_get_fee_calculator(self):
        """Should get or create calculator."""
        reset_fee_calculator()

        calc1 = get_fee_calculator()
        calc2 = get_fee_calculator()

        assert calc1 is calc2  # Same instance

    def test_reset_fee_calculator(self):
        """Should reset calculator."""
        calc1 = get_fee_calculator()
        reset_fee_calculator()
        calc2 = get_fee_calculator()

        assert calc1 is not calc2  # Different instances


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_notional(self):
        """Should handle zero notional."""
        calculator = FeeCalculator()

        estimate = calculator.calculate_maker_fee(Decimal("0"))

        assert estimate.fee_amount == Decimal("0")

    def test_very_small_notional(self):
        """Should handle very small notional."""
        calculator = FeeCalculator()

        estimate = calculator.calculate_taker_fee(Decimal("0.01"))

        assert estimate.fee_amount >= Decimal("0")

    def test_very_large_notional(self):
        """Should handle very large notional."""
        calculator = FeeCalculator()

        estimate = calculator.calculate_taker_fee(Decimal("1000000000"))

        assert estimate.fee_amount > Decimal("0")

    def test_zero_funding_rate(self):
        """Should handle zero funding rate."""
        calculator = FeeCalculator()

        estimate = calculator.calculate_funding_fee(
            notional=Decimal("100000"),
            funding_rate=Decimal("0"),
            is_long=True,
        )

        assert estimate.fee_amount == Decimal("0")
        assert "Neutral" in estimate.notes[0]

    def test_negative_maker_rate_rebate(self):
        """Should handle maker rebates (negative rates)."""
        calculator = FeeCalculator()

        # Set to market maker tier
        config = calculator.get_tier_config(FeeTier.MARKET_MAKER)
        if config:
            estimate = calculator.calculate_maker_fee(
                Decimal("100000"),
                tier=FeeTier.MARKET_MAKER,
            )

            # Negative fee = rebate
            assert "rebate" in estimate.notes[0].lower()
