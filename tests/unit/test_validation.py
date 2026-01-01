"""Unit tests for Configuration validation."""

import pytest
from unittest.mock import MagicMock

from src.config.validation import (
    ValidationLevel,
    ValidationType,
    ValidationError,
    ValidationResult,
    ValidationRule,
    ConfigValidator,
    TradingConfigValidator,
    AccountConfigValidator,
    is_positive,
    is_non_negative,
    is_percentage,
    is_valid_market,
    is_valid_address,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_level_values(self):
        """Should have expected level values."""
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.INFO.value == "info"


class TestValidationType:
    """Tests for ValidationType enum."""

    def test_type_values(self):
        """Should have expected type values."""
        assert ValidationType.REQUIRED.value == "required"
        assert ValidationType.TYPE.value == "type"
        assert ValidationType.RANGE.value == "range"
        assert ValidationType.PATTERN.value == "pattern"


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_create_error(self):
        """Should create validation error."""
        error = ValidationError(
            field="test_field",
            message="Test error message",
        )

        assert error.field == "test_field"
        assert error.message == "Test error message"
        assert error.level == ValidationLevel.ERROR

    def test_error_with_value(self):
        """Should include value."""
        error = ValidationError(
            field="size",
            message="Invalid size",
            value=-5,
        )

        assert error.value == -5

    def test_to_dict(self):
        """Should convert to dictionary."""
        error = ValidationError(
            field="field1",
            message="Error 1",
            level=ValidationLevel.WARNING,
        )

        d = error.to_dict()

        assert d["field"] == "field1"
        assert d["level"] == "warning"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result_valid(self):
        """Should create valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_add_error(self):
        """Should add error and mark invalid."""
        result = ValidationResult(is_valid=True)

        result.add_error(ValidationError(
            field="field1",
            message="Error",
            level=ValidationLevel.ERROR,
        ))

        assert result.is_valid is False
        assert result.error_count == 1

    def test_add_warning(self):
        """Should add warning without invalidating."""
        result = ValidationResult(is_valid=True)

        result.add_error(ValidationError(
            field="field1",
            message="Warning",
            level=ValidationLevel.WARNING,
        ))

        assert result.is_valid is True
        assert result.warning_count == 1

    def test_merge(self):
        """Should merge results."""
        result1 = ValidationResult(is_valid=True)
        result1.add_error(ValidationError("f1", "Error 1"))

        result2 = ValidationResult(is_valid=True)
        result2.add_error(ValidationError("f2", "Error 2", level=ValidationLevel.WARNING))

        result1.merge(result2)

        assert result1.error_count == 1
        assert result1.warning_count == 1

    def test_merge_invalidates(self):
        """Should invalidate on merge with invalid."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=False)

        result1.merge(result2)

        assert result1.is_valid is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ValidationResult(is_valid=True)
        result.add_error(ValidationError("f1", "Error"))

        d = result.to_dict()

        assert d["is_valid"] is False
        assert d["error_count"] == 1


class TestValidationRule:
    """Tests for ValidationRule dataclass."""

    def test_create_rule(self):
        """Should create rule."""
        rule = ValidationRule(
            field="name",
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None,
            message="Name required",
        )

        assert rule.field == "name"
        assert rule.rule_type == ValidationType.REQUIRED

    def test_validate_pass(self):
        """Should return None when valid."""
        rule = ValidationRule(
            field="name",
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None,
            message="Name required",
        )

        error = rule.validate("test_value")

        assert error is None

    def test_validate_fail(self):
        """Should return error when invalid."""
        rule = ValidationRule(
            field="name",
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None,
            message="Name required",
        )

        error = rule.validate(None)

        assert error is not None
        assert error.field == "name"

    def test_validate_with_condition_skip(self):
        """Should skip when condition not met."""
        rule = ValidationRule(
            field="field1",
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None,
            message="Required",
            condition=lambda ctx: ctx.get("enabled"),
        )

        error = rule.validate(None, {"enabled": False})

        assert error is None

    def test_validate_with_condition_run(self):
        """Should run when condition met."""
        rule = ValidationRule(
            field="field1",
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None,
            message="Required",
            condition=lambda ctx: ctx.get("enabled"),
        )

        error = rule.validate(None, {"enabled": True})

        assert error is not None


class TestConfigValidator:
    """Tests for ConfigValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator."""
        return ConfigValidator()

    def test_required_valid(self, validator):
        """Should pass when required field present."""
        validator.required("name")

        result = validator.validate({"name": "test"})

        assert result.is_valid is True

    def test_required_missing(self, validator):
        """Should fail when required field missing."""
        validator.required("name")

        result = validator.validate({"other": "value"})

        assert result.is_valid is False
        assert "name" in result.errors[0].field

    def test_required_empty(self, validator):
        """Should fail when required field empty."""
        validator.required("name")

        result = validator.validate({"name": ""})

        assert result.is_valid is False

    def test_type_check_valid(self, validator):
        """Should pass when type matches."""
        validator.type_check("count", int)

        result = validator.validate({"count": 42})

        assert result.is_valid is True

    def test_type_check_invalid(self, validator):
        """Should fail when type doesn't match."""
        validator.type_check("count", int)

        result = validator.validate({"count": "not_int"})

        assert result.is_valid is False

    def test_type_check_none_ok(self, validator):
        """Should pass when None."""
        validator.type_check("count", int)

        result = validator.validate({"count": None})

        assert result.is_valid is True

    def test_range_valid(self, validator):
        """Should pass when in range."""
        validator.range("value", min_val=0, max_val=100)

        result = validator.validate({"value": 50})

        assert result.is_valid is True

    def test_range_below_min(self, validator):
        """Should fail when below min."""
        validator.range("value", min_val=10)

        result = validator.validate({"value": 5})

        assert result.is_valid is False

    def test_range_above_max(self, validator):
        """Should fail when above max."""
        validator.range("value", max_val=100)

        result = validator.validate({"value": 150})

        assert result.is_valid is False

    def test_pattern_valid(self, validator):
        """Should pass when pattern matches."""
        validator.pattern("email", r"^[a-z]+@[a-z]+\.[a-z]+$")

        result = validator.validate({"email": "test@example.com"})

        assert result.is_valid is True

    def test_pattern_invalid(self, validator):
        """Should fail when pattern doesn't match."""
        validator.pattern("email", r"^[a-z]+@[a-z]+\.[a-z]+$")

        result = validator.validate({"email": "invalid-email"})

        assert result.is_valid is False

    def test_custom_validator(self, validator):
        """Should use custom validator."""
        validator.custom("value", lambda v: v > 0, "Must be positive")

        result = validator.validate({"value": 10})
        assert result.is_valid is True

        result = validator.validate({"value": -5})
        assert result.is_valid is False

    def test_custom_warning_level(self, validator):
        """Should add warning instead of error."""
        validator.custom(
            "value",
            lambda v: v < 100,
            "Consider using smaller value",
            level=ValidationLevel.WARNING,
        )

        result = validator.validate({"value": 150})

        assert result.is_valid is True  # Still valid
        assert result.warning_count == 1

    def test_conditional_when(self, validator):
        """Should apply conditional rules."""
        validator.when(lambda c: c.get("enabled")).required("config")

        # Condition not met - no error
        result = validator.validate({"enabled": False})
        assert result.is_valid is True

        # Condition met - error
        result = validator.validate({"enabled": True})
        assert result.is_valid is False

    def test_nested_field(self, validator):
        """Should validate nested fields."""
        validator.required("database.host")

        result = validator.validate({"database": {"host": "localhost"}})
        assert result.is_valid is True

        result = validator.validate({"database": {}})
        assert result.is_valid is False

    def test_register_custom_validator(self, validator):
        """Should register and use custom validator."""
        validator.register_custom_validator("is_even", lambda v: v % 2 == 0)
        validator.use_custom("value", "is_even", "Must be even")

        result = validator.validate({"value": 4})
        assert result.is_valid is True

        result = validator.validate({"value": 3})
        assert result.is_valid is False

    def test_chain_rules(self, validator):
        """Should chain multiple rules."""
        validator.required("name").type_check("name", str).pattern("name", r"^[A-Z]")

        result = validator.validate({"name": "Test"})
        assert result.is_valid is True

        result = validator.validate({"name": "test"})
        assert result.is_valid is False

    def test_multiple_errors(self, validator):
        """Should collect multiple errors."""
        validator.required("field1")
        validator.required("field2")
        validator.required("field3")

        result = validator.validate({})

        assert result.is_valid is False
        assert result.error_count == 3


class TestPreBuiltValidators:
    """Tests for pre-built validators."""

    def test_is_positive(self):
        """Should check positive values."""
        assert is_positive(5) is True
        assert is_positive(0.1) is True
        assert is_positive(0) is False
        assert is_positive(-5) is False
        assert is_positive(None) is True

    def test_is_non_negative(self):
        """Should check non-negative values."""
        assert is_non_negative(5) is True
        assert is_non_negative(0) is True
        assert is_non_negative(-5) is False
        assert is_non_negative(None) is True

    def test_is_percentage(self):
        """Should check percentage range."""
        assert is_percentage(0) is True
        assert is_percentage(50) is True
        assert is_percentage(100) is True
        assert is_percentage(-1) is False
        assert is_percentage(101) is False
        assert is_percentage(None) is True

    def test_is_valid_market(self):
        """Should check market format."""
        assert is_valid_market("BTC-USD-PERP") is True
        assert is_valid_market("ETH-USDC-PERP") is True
        assert is_valid_market("btc-usd-perp") is False
        assert is_valid_market("BTCUSD") is False
        assert is_valid_market(None) is True

    def test_is_valid_address(self):
        """Should check address format."""
        assert is_valid_address("0x" + "a" * 40) is True
        assert is_valid_address("0x" + "A" * 64) is True
        assert is_valid_address("0x123") is False
        assert is_valid_address("invalid") is False
        assert is_valid_address(None) is True


class TestTradingConfigValidator:
    """Tests for TradingConfigValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator."""
        return TradingConfigValidator()

    def test_valid_config(self, validator):
        """Should validate complete config."""
        config = {
            "market": "BTC-USD-PERP",
            "position_size": 1000,
            "delta_threshold": 5.0,
            "max_slippage": 0.5,
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is True

    def test_missing_market(self, validator):
        """Should require market."""
        config = {
            "position_size": 1000,
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is False
        assert any("market" in e.field for e in result.errors)

    def test_invalid_market_format(self, validator):
        """Should validate market format."""
        config = {
            "market": "invalid-market",
            "position_size": 1000,
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is False

    def test_negative_position_size(self, validator):
        """Should reject negative position size."""
        config = {
            "market": "BTC-USD-PERP",
            "position_size": -100,
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is False

    def test_delta_threshold_range(self, validator):
        """Should validate delta threshold range."""
        config = {
            "market": "BTC-USD-PERP",
            "position_size": 1000,
            "delta_threshold": 60,  # Too high
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is False

    def test_funding_strategy_requires_threshold(self, validator):
        """Should require funding_threshold for funding strategy."""
        config = {
            "market": "BTC-USD-PERP",
            "position_size": 1000,
            "strategy": "funding",
        }

        result = validator.validate_trading_config(config)

        assert result.is_valid is False
        assert any("funding_threshold" in e.field for e in result.errors)


class TestAccountConfigValidator:
    """Tests for AccountConfigValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator."""
        return AccountConfigValidator()

    def test_valid_account(self, validator):
        """Should validate complete account config."""
        config = {
            "alias": "account1",
            "address": "0x" + "a" * 40,
            "enabled": True,
        }

        result = validator.validate_account(config)

        assert result.is_valid is True

    def test_missing_alias(self, validator):
        """Should require alias."""
        config = {
            "address": "0x" + "a" * 40,
        }

        result = validator.validate_account(config)

        assert result.is_valid is False

    def test_invalid_alias_format(self, validator):
        """Should validate alias format."""
        config = {
            "alias": "1invalid",  # Can't start with number
            "address": "0x" + "a" * 40,
        }

        result = validator.validate_account(config)

        assert result.is_valid is False

    def test_alias_too_short(self, validator):
        """Should reject short alias."""
        config = {
            "alias": "ab",  # Too short
            "address": "0x" + "a" * 40,
        }

        result = validator.validate_account(config)

        assert result.is_valid is False

    def test_invalid_address(self, validator):
        """Should validate address format."""
        config = {
            "alias": "account1",
            "address": "invalid_address",
        }

        result = validator.validate_account(config)

        assert result.is_valid is False

    def test_enabled_type(self, validator):
        """Should check enabled is boolean."""
        config = {
            "alias": "account1",
            "address": "0x" + "a" * 40,
            "enabled": "yes",  # Should be bool
        }

        result = validator.validate_account(config)

        assert result.is_valid is False
