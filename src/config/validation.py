"""Configuration validation for trading bot."""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationType(Enum):
    """Types of validation rules."""

    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    rule_type: ValidationType = ValidationType.CUSTOM
    value: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "level": self.level.value,
            "rule_type": self.rule_type.value,
        }


@dataclass
class ValidationResult:
    """Result of validation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def __post_init__(self):
        """Initialize lists."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    @property
    def error_count(self) -> int:
        """Count errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count warnings."""
        return len(self.warnings)

    def add_error(self, error: ValidationError) -> None:
        """Add error."""
        if error.level == ValidationLevel.ERROR:
            self.errors.append(error)
            self.is_valid = False
        else:
            self.warnings.append(error)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass
class ValidationRule:
    """A validation rule."""

    field: str
    rule_type: ValidationType
    validator: Callable[[Any], bool]
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    condition: Optional[Callable[[Dict], bool]] = None

    def validate(self, value: Any, context: Dict = None) -> Optional[ValidationError]:
        """Run validation."""
        # Check condition
        if self.condition and context:
            if not self.condition(context):
                return None  # Skip if condition not met

        if not self.validator(value):
            return ValidationError(
                field=self.field,
                message=self.message,
                level=self.level,
                rule_type=self.rule_type,
                value=value,
            )
        return None


@dataclass
class ConfigValidator:
    """Validator for configuration objects.

    Features:
    - Required field validation
    - Type checking
    - Range validation
    - Pattern matching
    - Custom validators
    - Conditional rules
    """

    _rules: List[ValidationRule] = field(default_factory=list)
    _custom_validators: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize."""
        self._rules = []
        self._custom_validators = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self._rules.append(rule)

    def required(
        self,
        field: str,
        message: str = None,
    ) -> "ConfigValidator":
        """Add required field rule."""
        self.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None and v != "",
            message=message or f"Field '{field}' is required",
        ))
        return self

    def type_check(
        self,
        field: str,
        expected_type: type,
        message: str = None,
    ) -> "ConfigValidator":
        """Add type check rule."""
        self.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.TYPE,
            validator=lambda v: isinstance(v, expected_type) if v is not None else True,
            message=message or f"Field '{field}' must be {expected_type.__name__}",
        ))
        return self

    def range(
        self,
        field: str,
        min_val: Any = None,
        max_val: Any = None,
        message: str = None,
    ) -> "ConfigValidator":
        """Add range validation rule."""
        def check_range(value):
            if value is None:
                return True
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True

        msg = message
        if not msg:
            if min_val is not None and max_val is not None:
                msg = f"Field '{field}' must be between {min_val} and {max_val}"
            elif min_val is not None:
                msg = f"Field '{field}' must be at least {min_val}"
            else:
                msg = f"Field '{field}' must be at most {max_val}"

        self.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.RANGE,
            validator=check_range,
            message=msg,
        ))
        return self

    def pattern(
        self,
        field: str,
        regex: str,
        message: str = None,
    ) -> "ConfigValidator":
        """Add pattern validation rule."""
        pattern = re.compile(regex)
        self.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.PATTERN,
            validator=lambda v: bool(pattern.match(str(v))) if v else True,
            message=message or f"Field '{field}' does not match required pattern",
        ))
        return self

    def custom(
        self,
        field: str,
        validator: Callable[[Any], bool],
        message: str,
        level: ValidationLevel = ValidationLevel.ERROR,
    ) -> "ConfigValidator":
        """Add custom validation rule."""
        self.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.CUSTOM,
            validator=validator,
            message=message,
            level=level,
        ))
        return self

    def when(
        self,
        condition: Callable[[Dict], bool],
    ) -> "_ConditionalValidator":
        """Create conditional validator."""
        return _ConditionalValidator(self, condition)

    def register_custom_validator(
        self,
        name: str,
        validator: Callable[[Any], bool],
    ) -> None:
        """Register reusable custom validator."""
        self._custom_validators[name] = validator

    def use_custom(
        self,
        field: str,
        validator_name: str,
        message: str,
    ) -> "ConfigValidator":
        """Use registered custom validator."""
        validator = self._custom_validators.get(validator_name)
        if not validator:
            raise ValueError(f"Unknown validator: {validator_name}")
        return self.custom(field, validator, message)

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration."""
        result = ValidationResult(is_valid=True)

        for rule in self._rules:
            value = self._get_value(config, rule.field)
            error = rule.validate(value, config)
            if error:
                result.add_error(error)

        return result

    def _get_value(self, config: Dict, field: str) -> Any:
        """Get nested field value."""
        parts = field.split(".")
        value = config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value


@dataclass
class _ConditionalValidator:
    """Builder for conditional validation rules."""

    _parent: ConfigValidator
    _condition: Callable[[Dict], bool]

    def required(self, field: str, message: str = None) -> "ConfigValidator":
        """Add conditional required rule."""
        self._parent.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.REQUIRED,
            validator=lambda v: v is not None and v != "",
            message=message or f"Field '{field}' is required",
            condition=self._condition,
        ))
        return self._parent

    def custom(
        self,
        field: str,
        validator: Callable[[Any], bool],
        message: str,
    ) -> "ConfigValidator":
        """Add conditional custom rule."""
        self._parent.add_rule(ValidationRule(
            field=field,
            rule_type=ValidationType.CUSTOM,
            validator=validator,
            message=message,
            condition=self._condition,
        ))
        return self._parent


# Pre-built validators
def is_positive(value: Any) -> bool:
    """Check if value is positive."""
    if value is None:
        return True
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def is_non_negative(value: Any) -> bool:
    """Check if value is non-negative."""
    if value is None:
        return True
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def is_percentage(value: Any) -> bool:
    """Check if value is a valid percentage (0-100)."""
    if value is None:
        return True
    try:
        v = float(value)
        return 0 <= v <= 100
    except (TypeError, ValueError):
        return False


def is_valid_market(value: Any) -> bool:
    """Check if value is valid market format."""
    if value is None:
        return True
    pattern = re.compile(r"^[A-Z]+-[A-Z]+-PERP$")
    return bool(pattern.match(str(value)))


def is_valid_address(value: Any) -> bool:
    """Check if value is valid hex address."""
    if value is None:
        return True
    pattern = re.compile(r"^0x[a-fA-F0-9]{40,64}$")
    return bool(pattern.match(str(value)))


@dataclass
class TradingConfigValidator(ConfigValidator):
    """Specialized validator for trading configuration."""

    def __post_init__(self):
        """Setup trading-specific rules."""
        super().__post_init__()

        # Register custom validators
        self.register_custom_validator("positive", is_positive)
        self.register_custom_validator("non_negative", is_non_negative)
        self.register_custom_validator("percentage", is_percentage)
        self.register_custom_validator("market", is_valid_market)
        self.register_custom_validator("address", is_valid_address)

    def validate_trading_config(self, config: Dict) -> ValidationResult:
        """Validate complete trading configuration."""
        # Clear existing rules
        self._rules = []

        # Build validation rules
        self.required("market")
        self.custom("market", is_valid_market, "Invalid market format (e.g., BTC-USD-PERP)")

        self.required("position_size")
        self.custom("position_size", is_positive, "Position size must be positive")

        self.range("delta_threshold", min_val=0.1, max_val=50.0)

        self.range("max_slippage", min_val=0.0, max_val=5.0)

        # Conditional rules
        self.when(lambda c: c.get("strategy") == "funding")\
            .required("funding_threshold", "Funding threshold required for funding strategy")

        return self.validate(config)


@dataclass
class AccountConfigValidator(ConfigValidator):
    """Specialized validator for account configuration."""

    def validate_account(self, config: Dict) -> ValidationResult:
        """Validate account configuration."""
        self._rules = []

        self.required("alias")
        self.pattern("alias", r"^[a-zA-Z][a-zA-Z0-9_-]{2,20}$",
                    "Alias must be 3-21 chars, start with letter")

        self.required("address")
        self.custom("address", is_valid_address, "Invalid address format")

        self.type_check("enabled", bool)

        return self.validate(config)
