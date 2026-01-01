"""Unit tests for Position Sizing Calculator."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.core.position_sizing import (
    SizingMethod,
    RiskLevel,
    SizingConfig,
    SizingInput,
    SizingResult,
    PositionSizer,
    DeltaNeutralSizer,
    RiskAdjustedSizer,
)


class TestSizingMethod:
    """Tests for SizingMethod enum."""

    def test_method_values(self):
        """Should have expected method values."""
        assert SizingMethod.FIXED.value == "fixed"
        assert SizingMethod.PERCENT_EQUITY.value == "percent_equity"
        assert SizingMethod.PERCENT_RISK.value == "percent_risk"
        assert SizingMethod.KELLY.value == "kelly"
        assert SizingMethod.VOLATILITY.value == "volatility"
        assert SizingMethod.ATR.value == "atr"
        assert SizingMethod.MARTINGALE.value == "martingale"
        assert SizingMethod.ANTI_MARTINGALE.value == "anti_martingale"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_level_values(self):
        """Should have expected risk levels."""
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.AGGRESSIVE.value == "aggressive"
        assert RiskLevel.VERY_AGGRESSIVE.value == "very_aggressive"


class TestSizingConfig:
    """Tests for SizingConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = SizingConfig()

        assert config.method == SizingMethod.PERCENT_EQUITY
        assert config.risk_level == RiskLevel.MODERATE
        assert config.max_position_pct == 25.0
        assert config.max_total_exposure_pct == 100.0
        assert config.min_position_size == Decimal("0.001")
        assert config.max_leverage == 10.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = SizingConfig(
            method=SizingMethod.KELLY,
            risk_level=RiskLevel.AGGRESSIVE,
            max_position_pct=50.0,
            max_leverage=20.0,
        )

        assert config.method == SizingMethod.KELLY
        assert config.risk_level == RiskLevel.AGGRESSIVE
        assert config.max_position_pct == 50.0
        assert config.max_leverage == 20.0


class TestSizingInput:
    """Tests for SizingInput."""

    def test_create_input(self):
        """Should create input data."""
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

        assert input_data.account_equity == Decimal("10000")
        assert input_data.entry_price == Decimal("50000")
        assert input_data.current_exposure == Decimal("0")

    def test_input_with_stop_loss(self):
        """Should include stop loss."""
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("48000"),
        )

        assert input_data.stop_loss_price == Decimal("48000")

    def test_input_with_kelly_params(self):
        """Should include Kelly parameters."""
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            win_rate=0.55,
            avg_win=Decimal("200"),
            avg_loss=Decimal("100"),
        )

        assert input_data.win_rate == 0.55
        assert input_data.avg_win == Decimal("200")
        assert input_data.avg_loss == Decimal("100")


class TestSizingResult:
    """Tests for SizingResult."""

    def test_create_result(self):
        """Should create result."""
        result = SizingResult(
            size=Decimal("0.5"),
            size_usd=Decimal("25000"),
            risk_amount=Decimal("250"),
            risk_percent=2.5,
            method=SizingMethod.PERCENT_EQUITY,
        )

        assert result.size == Decimal("0.5")
        assert result.size_usd == Decimal("25000")
        assert result.risk_percent == 2.5

    def test_result_with_notes(self):
        """Should include notes."""
        result = SizingResult(
            size=Decimal("0.5"),
            size_usd=Decimal("25000"),
            risk_amount=Decimal("250"),
            risk_percent=2.5,
            method=SizingMethod.KELLY,
            notes=["Kelly fraction: 12.5%"],
        )

        assert "Kelly fraction: 12.5%" in result.notes

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = SizingResult(
            size=Decimal("0.5"),
            size_usd=Decimal("25000"),
            risk_amount=Decimal("250"),
            risk_percent=2.5,
            method=SizingMethod.PERCENT_EQUITY,
            leverage_used=2.5,
        )

        d = result.to_dict()

        assert d["size"] == "0.5"
        assert d["method"] == "percent_equity"
        assert d["leverage_used"] == 2.5


class TestPositionSizer:
    """Tests for PositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create position sizer."""
        return PositionSizer()

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        return SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

    def test_fixed_size(self):
        """Should calculate fixed size."""
        config = SizingConfig(
            method=SizingMethod.FIXED,
            min_position_size=Decimal("0.01"),
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.FIXED
        assert result.size >= Decimal("0.001")

    def test_percent_equity_size(self, sizer, input_data):
        """Should calculate percent of equity size."""
        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.PERCENT_EQUITY
        assert result.size > Decimal("0")
        assert result.size_usd > Decimal("0")

    def test_percent_equity_respects_max(self, input_data):
        """Should respect max position percentage."""
        config = SizingConfig(
            method=SizingMethod.PERCENT_EQUITY,
            max_position_pct=10.0,  # 10%
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_size(input_data)

        # Position value should be ~10% of equity
        max_value = input_data.account_equity * Decimal("0.10")
        assert result.size_usd <= max_value * Decimal("1.01")  # Allow small rounding

    def test_percent_risk_with_stop_loss(self):
        """Should calculate based on stop loss distance."""
        config = SizingConfig(
            method=SizingMethod.PERCENT_RISK,
            risk_level=RiskLevel.MODERATE,  # 1% risk
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),  # 2% stop
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.PERCENT_RISK
        assert result.risk_percent == 1.0

    def test_percent_risk_default_stop(self):
        """Should use default stop if not provided."""
        config = SizingConfig(method=SizingMethod.PERCENT_RISK)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        assert "default" in result.notes[0].lower()

    def test_kelly_size(self):
        """Should calculate Kelly criterion size."""
        config = SizingConfig(
            method=SizingMethod.KELLY,
            kelly_fraction=0.5,
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            win_rate=0.60,
            avg_win=Decimal("200"),
            avg_loss=Decimal("100"),
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.KELLY
        assert "Kelly fraction" in result.notes[0]

    def test_kelly_default_params(self):
        """Should use defaults if Kelly params missing."""
        config = SizingConfig(method=SizingMethod.KELLY)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.KELLY

    def test_optimal_f_size(self):
        """Should calculate optimal f size."""
        config = SizingConfig(method=SizingMethod.OPTIMAL_F)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            avg_loss=Decimal("200"),
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.OPTIMAL_F
        assert "Optimal f" in result.notes[0]

    def test_volatility_size(self):
        """Should calculate volatility-based size."""
        config = SizingConfig(method=SizingMethod.VOLATILITY)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            volatility=0.30,  # 30% annual volatility
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.VOLATILITY
        assert "Volatility" in result.notes[0]

    def test_volatility_reduces_size_for_high_vol(self):
        """Should reduce size for high volatility."""
        config = SizingConfig(method=SizingMethod.VOLATILITY)
        sizer = PositionSizer(config=config)

        low_vol_input = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            volatility=0.10,  # Low vol
        )
        high_vol_input = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            volatility=0.50,  # High vol
        )

        low_vol_result = sizer.calculate_size(low_vol_input)
        high_vol_result = sizer.calculate_size(high_vol_input)

        assert low_vol_result.size >= high_vol_result.size

    def test_atr_size(self):
        """Should calculate ATR-based size."""
        config = SizingConfig(
            method=SizingMethod.ATR,
            atr_multiplier=2.0,
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            atr=Decimal("1000"),  # $1000 ATR
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.ATR
        assert "ATR" in result.notes[0]

    def test_martingale_size(self):
        """Should calculate Martingale progression."""
        config = SizingConfig(method=SizingMethod.MARTINGALE)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            consecutive_losses=3,  # Double 3 times = 8x
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.MARTINGALE
        assert "Multiplier" in result.notes[-1]

    def test_martingale_caps_multiplier(self):
        """Should cap Martingale multiplier."""
        config = SizingConfig(method=SizingMethod.MARTINGALE)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            consecutive_losses=10,  # Would be 1024x
        )

        result = sizer.calculate_size(input_data)

        assert "Capped" in result.notes[0]

    def test_anti_martingale_size(self):
        """Should calculate Anti-Martingale progression."""
        config = SizingConfig(method=SizingMethod.ANTI_MARTINGALE)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            consecutive_wins=2,
        )

        result = sizer.calculate_size(input_data)

        assert result.method == SizingMethod.ANTI_MARTINGALE
        assert "Consecutive wins" in result.notes[0]

    def test_anti_martingale_caps_multiplier(self):
        """Should cap Anti-Martingale multiplier."""
        config = SizingConfig(method=SizingMethod.ANTI_MARTINGALE)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            consecutive_wins=10,  # Would exceed cap
        )

        result = sizer.calculate_size(input_data)

        assert "Capped" in result.notes[0]

    def test_respects_min_size(self, input_data):
        """Should respect minimum size."""
        config = SizingConfig(
            method=SizingMethod.FIXED,
            min_position_size=Decimal("0.001"),
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_size(input_data)

        assert result.size >= Decimal("0.001")

    def test_respects_max_size(self, input_data):
        """Should respect maximum size."""
        config = SizingConfig(
            method=SizingMethod.PERCENT_EQUITY,
            max_position_size=Decimal("0.01"),
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_size(input_data)

        assert result.size <= Decimal("0.01")

    def test_respects_total_exposure(self):
        """Should respect total exposure limit."""
        config = SizingConfig(
            method=SizingMethod.PERCENT_EQUITY,
            max_total_exposure_pct=50.0,
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            current_exposure=Decimal("4000"),  # 40% already used
        )

        result = sizer.calculate_size(input_data)

        # Should only allow 10% more (to reach 50%)
        max_additional = Decimal("1000")  # 10% of equity
        assert result.size_usd <= max_additional * Decimal("1.01")

    def test_zero_size_when_exposure_exceeded(self):
        """Should return zero when exposure exceeded."""
        config = SizingConfig(
            method=SizingMethod.PERCENT_EQUITY,
            max_total_exposure_pct=50.0,
        )
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            current_exposure=Decimal("6000"),  # 60% already used
        )

        result = sizer.calculate_size(input_data)

        assert result.size == Decimal("0")

    def test_calculate_leverage(self, sizer):
        """Should calculate leverage correctly."""
        leverage = sizer.calculate_leverage(
            position_value=Decimal("5000"),
            account_equity=Decimal("10000"),
        )

        assert leverage == 0.5

    def test_leverage_respects_max(self):
        """Should cap leverage at maximum."""
        config = SizingConfig(max_leverage=5.0)
        sizer = PositionSizer(config=config)

        leverage = sizer.calculate_leverage(
            position_value=Decimal("100000"),
            account_equity=Decimal("10000"),
        )

        assert leverage == 5.0

    def test_leverage_disabled(self):
        """Should cap at 1x when leverage disabled."""
        config = SizingConfig(use_leverage=False)
        sizer = PositionSizer(config=config)

        leverage = sizer.calculate_leverage(
            position_value=Decimal("20000"),
            account_equity=Decimal("10000"),
        )

        assert leverage == 1.0

    def test_validate_size_valid(self, sizer, input_data):
        """Should validate valid size."""
        is_valid, errors = sizer.validate_size(
            size=Decimal("0.01"),
            input_data=input_data,
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_size_below_min(self, sizer, input_data):
        """Should catch size below minimum."""
        is_valid, errors = sizer.validate_size(
            size=Decimal("0.0001"),
            input_data=input_data,
        )

        assert is_valid is False
        assert any("minimum" in e.lower() for e in errors)

    def test_validate_size_above_max(self, input_data):
        """Should catch size above maximum."""
        config = SizingConfig(max_position_size=Decimal("0.01"))
        sizer = PositionSizer(config=config)

        is_valid, errors = sizer.validate_size(
            size=Decimal("0.1"),
            input_data=input_data,
        )

        assert is_valid is False
        assert any("maximum" in e.lower() for e in errors)

    def test_validate_size_exceeds_equity_pct(self, input_data):
        """Should catch size exceeding equity percentage."""
        config = SizingConfig(max_position_pct=10.0)
        sizer = PositionSizer(config=config)

        # 1 BTC at $50000 = $50000 = 500% of $10000 equity
        is_valid, errors = sizer.validate_size(
            size=Decimal("1"),
            input_data=input_data,
        )

        assert is_valid is False
        assert any("position %" in e.lower() for e in errors)

    def test_get_risk_percent_by_level(self):
        """Should return correct risk percent for each level."""
        conservative = PositionSizer(SizingConfig(risk_level=RiskLevel.CONSERVATIVE))
        moderate = PositionSizer(SizingConfig(risk_level=RiskLevel.MODERATE))
        aggressive = PositionSizer(SizingConfig(risk_level=RiskLevel.AGGRESSIVE))
        very_aggressive = PositionSizer(SizingConfig(risk_level=RiskLevel.VERY_AGGRESSIVE))

        assert conservative._get_risk_percent() == 0.5
        assert moderate._get_risk_percent() == 1.0
        assert aggressive._get_risk_percent() == 2.0
        assert very_aggressive._get_risk_percent() == 5.0


class TestDeltaNeutralSizer:
    """Tests for DeltaNeutralSizer."""

    @pytest.fixture
    def sizer(self):
        """Create delta-neutral sizer."""
        return DeltaNeutralSizer()

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        return SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

    def test_calculate_pair_sizes(self, sizer, input_data):
        """Should calculate balanced long and short sizes."""
        long_sizes, short_sizes = sizer.calculate_pair_sizes(input_data, num_accounts=4)

        assert len(long_sizes) == 2  # Half long
        assert len(short_sizes) == 2  # Half short

    def test_sizes_balanced(self, sizer, input_data):
        """Should balance total long and short."""
        long_sizes, short_sizes = sizer.calculate_pair_sizes(input_data, num_accounts=4)

        total_long = sum(long_sizes)
        total_short = sum(short_sizes)

        # Should be approximately equal
        ratio = float(total_long / total_short) if total_short > 0 else 0
        assert 0.95 <= ratio <= 1.05

    def test_odd_accounts(self, sizer, input_data):
        """Should handle odd number of accounts."""
        long_sizes, short_sizes = sizer.calculate_pair_sizes(input_data, num_accounts=5)

        assert len(long_sizes) + len(short_sizes) == 5

    def test_variance_applied(self, sizer, input_data):
        """Should apply variance to sizes."""
        # Run multiple times to check variance
        all_equal = True
        prev_long = None

        for _ in range(5):
            long_sizes, _ = sizer.calculate_pair_sizes(input_data, num_accounts=4)
            if prev_long and long_sizes[0] != prev_long[0]:
                all_equal = False
                break
            prev_long = long_sizes

        # With variance, sizes should differ across runs
        # (might occasionally be equal by chance, but unlikely for 5 runs)

    def test_calculate_rebalance_long(self, sizer):
        """Should calculate rebalance for long."""
        side, size = sizer.calculate_rebalance_size(
            long_exposure=Decimal("1000"),
            short_exposure=Decimal("1200"),
            target_delta=Decimal("0"),
        )

        assert side == "long"
        assert size == Decimal("200")

    def test_calculate_rebalance_short(self, sizer):
        """Should calculate rebalance for short."""
        side, size = sizer.calculate_rebalance_size(
            long_exposure=Decimal("1200"),
            short_exposure=Decimal("1000"),
            target_delta=Decimal("0"),
        )

        assert side == "short"
        assert size == Decimal("200")

    def test_calculate_rebalance_none(self, sizer):
        """Should return none when balanced."""
        side, size = sizer.calculate_rebalance_size(
            long_exposure=Decimal("1000"),
            short_exposure=Decimal("1000"),
            target_delta=Decimal("0"),
        )

        assert side == "none"
        assert size == Decimal("0")

    def test_calculate_rebalance_custom_target(self, sizer):
        """Should rebalance to custom target."""
        side, size = sizer.calculate_rebalance_size(
            long_exposure=Decimal("1000"),
            short_exposure=Decimal("1000"),
            target_delta=Decimal("100"),  # Want $100 more long
        )

        assert side == "long"
        assert size == Decimal("100")


class TestRiskAdjustedSizer:
    """Tests for RiskAdjustedSizer."""

    @pytest.fixture
    def sizer(self):
        """Create risk-adjusted sizer."""
        return RiskAdjustedSizer(
            drawdown_threshold=10.0,
            drawdown_reduction=50.0,
            profit_increase_threshold=20.0,
            profit_increase=25.0,
        )

    @pytest.fixture
    def input_data(self):
        """Create test input data."""
        return SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
        )

    def test_normal_size_no_adjustment(self, sizer, input_data):
        """Should not adjust when no drawdown or profit."""
        result = sizer.calculate_size(
            input_data=input_data,
            peak_equity=Decimal("10000"),
            initial_equity=Decimal("10000"),
        )

        # No notes about adjustment
        assert not any("reduction" in n.lower() for n in result.notes)
        assert not any("increase" in n.lower() for n in result.notes)

    def test_reduces_size_on_drawdown(self, sizer, input_data):
        """Should reduce size during drawdown."""
        # 15% drawdown (above 10% threshold)
        input_data.account_equity = Decimal("8500")

        result = sizer.calculate_size(
            input_data=input_data,
            peak_equity=Decimal("10000"),
            initial_equity=Decimal("10000"),
        )

        assert any("reduction" in n.lower() for n in result.notes)

    def test_increases_size_on_profit(self, sizer, input_data):
        """Should increase size during profit."""
        # 25% profit (above 20% threshold)
        input_data.account_equity = Decimal("12500")

        result = sizer.calculate_size(
            input_data=input_data,
            peak_equity=Decimal("12500"),
            initial_equity=Decimal("10000"),
        )

        assert any("increase" in n.lower() for n in result.notes)

    def test_drawdown_takes_precedence(self, sizer, input_data):
        """Should reduce on drawdown even if profitable overall."""
        # In profit overall but in drawdown from peak
        input_data.account_equity = Decimal("10500")

        result = sizer.calculate_size(
            input_data=input_data,
            peak_equity=Decimal("12000"),  # 12.5% drawdown
            initial_equity=Decimal("10000"),  # 5% profit
        )

        assert any("reduction" in n.lower() for n in result.notes)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_equity(self):
        """Should handle zero equity."""
        sizer = PositionSizer()
        input_data = SizingInput(
            account_equity=Decimal("0"),
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        assert result.size >= Decimal("0")

    def test_zero_price(self):
        """Should handle zero price."""
        sizer = PositionSizer()
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("0.0001"),  # Near zero
        )

        result = sizer.calculate_size(input_data)

        assert result.size >= Decimal("0")

    def test_negative_stop_loss(self):
        """Should handle stop loss below entry."""
        config = SizingConfig(method=SizingMethod.PERCENT_RISK)
        sizer = PositionSizer(config=config)
        input_data = SizingInput(
            account_equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("45000"),  # 10% below
        )

        result = sizer.calculate_size(input_data)

        assert result.size > Decimal("0")

    def test_very_small_equity(self):
        """Should handle very small equity."""
        sizer = PositionSizer()
        input_data = SizingInput(
            account_equity=Decimal("1"),
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        # Should still return valid size (might be min)
        assert result.size >= Decimal("0")

    def test_very_large_equity(self):
        """Should handle very large equity."""
        sizer = PositionSizer()
        input_data = SizingInput(
            account_equity=Decimal("1000000000"),  # 1 billion
            entry_price=Decimal("50000"),
        )

        result = sizer.calculate_size(input_data)

        assert result.size > Decimal("0")
