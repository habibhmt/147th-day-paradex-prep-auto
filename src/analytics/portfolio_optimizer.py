"""
Portfolio Optimizer Module.

Optimizes portfolio allocation across multiple assets
using various strategies including mean-variance, risk parity,
and constraint-based optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable
import math


class OptimizationStrategy(Enum):
    """Portfolio optimization strategy."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    TARGET_RISK = "target_risk"
    CUSTOM = "custom"


class RebalanceStrategy(Enum):
    """Rebalancing strategy."""

    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold
    PERIODIC = "periodic"  # Rebalance at fixed intervals
    HYBRID = "hybrid"  # Both threshold and periodic
    NEVER = "never"  # No automatic rebalancing


class ConstraintType(Enum):
    """Portfolio constraint type."""

    MIN_WEIGHT = "min_weight"
    MAX_WEIGHT = "max_weight"
    MIN_EXPOSURE = "min_exposure"
    MAX_EXPOSURE = "max_exposure"
    SECTOR_LIMIT = "sector_limit"
    POSITION_LIMIT = "position_limit"


@dataclass
class AssetMetrics:
    """Metrics for a single asset."""

    symbol: str
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    current_weight: float = 0.0
    target_weight: float = 0.0
    correlation_to_portfolio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "correlation_to_portfolio": self.correlation_to_portfolio,
        }


@dataclass
class PortfolioAllocation:
    """Target portfolio allocation."""

    weights: dict[str, float]  # symbol -> weight
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: OptimizationStrategy = OptimizationStrategy.EQUAL_WEIGHT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "expected_sharpe": self.expected_sharpe,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
        }


@dataclass
class PortfolioConstraint:
    """Portfolio optimization constraint."""

    constraint_type: ConstraintType
    symbol: str | None  # None for portfolio-level constraints
    value: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_type": self.constraint_type.value,
            "symbol": self.symbol,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class RebalanceAction:
    """Rebalance action for a single asset."""

    symbol: str
    current_weight: float
    target_weight: float
    weight_change: float
    current_value: Decimal
    target_value: Decimal
    value_change: Decimal
    action: str  # 'buy' or 'sell'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "weight_change": self.weight_change,
            "current_value": float(self.current_value),
            "target_value": float(self.target_value),
            "value_change": float(self.value_change),
            "action": self.action,
        }


@dataclass
class RebalancePlan:
    """Complete rebalancing plan."""

    actions: list[RebalanceAction]
    total_portfolio_value: Decimal
    total_turnover: float
    estimated_cost: Decimal
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actions": [a.to_dict() for a in self.actions],
            "total_portfolio_value": float(self.total_portfolio_value),
            "total_turnover": self.total_turnover,
            "estimated_cost": float(self.estimated_cost),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics."""

    total_variance: float
    total_volatility: float
    value_at_risk_95: float  # 95% VaR
    value_at_risk_99: float  # 99% VaR
    expected_shortfall: float
    max_drawdown: float
    beta: float
    correlation_matrix: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_variance": self.total_variance,
            "total_volatility": self.total_volatility,
            "value_at_risk_95": self.value_at_risk_95,
            "value_at_risk_99": self.value_at_risk_99,
            "expected_shortfall": self.expected_shortfall,
            "max_drawdown": self.max_drawdown,
            "beta": self.beta,
            "correlation_matrix": self.correlation_matrix,
        }


@dataclass
class DriftAnalysis:
    """Portfolio drift analysis."""

    timestamp: datetime
    total_drift: float
    max_drift_symbol: str
    max_drift_value: float
    needs_rebalance: bool
    drift_by_symbol: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_drift": self.total_drift,
            "max_drift_symbol": self.max_drift_symbol,
            "max_drift_value": self.max_drift_value,
            "needs_rebalance": self.needs_rebalance,
            "drift_by_symbol": self.drift_by_symbol,
        }


class PortfolioOptimizer:
    """Optimizes portfolio allocation."""

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.EQUAL_WEIGHT,
        rebalance_strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
        rebalance_threshold: float = 0.05,  # 5% drift
        risk_free_rate: float = 0.02,  # 2% annual
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        transaction_cost_bps: float = 5.0,
    ):
        """
        Initialize optimizer.

        Args:
            strategy: Optimization strategy
            rebalance_strategy: Rebalancing strategy
            rebalance_threshold: Threshold for threshold-based rebalancing
            risk_free_rate: Risk-free rate for Sharpe calculations
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            transaction_cost_bps: Transaction cost in basis points
        """
        self.strategy = strategy
        self.rebalance_strategy = rebalance_strategy
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.transaction_cost_bps = transaction_cost_bps

        # Asset data
        self._assets: dict[str, AssetMetrics] = {}

        # Current weights
        self._current_weights: dict[str, float] = {}

        # Target allocation
        self._target_allocation: PortfolioAllocation | None = None

        # Correlation matrix
        self._correlations: dict[str, dict[str, float]] = {}

        # Constraints
        self._constraints: list[PortfolioConstraint] = []

        # Callbacks
        self._callbacks: list[Callable[[PortfolioAllocation], None]] = []

    def add_asset(self, metrics: AssetMetrics) -> None:
        """
        Add or update asset metrics.

        Args:
            metrics: Asset metrics
        """
        self._assets[metrics.symbol] = metrics
        if metrics.symbol not in self._current_weights:
            self._current_weights[metrics.symbol] = 0.0

    def update_asset(
        self,
        symbol: str,
        expected_return: float,
        volatility: float,
        current_weight: float = 0.0,
    ) -> None:
        """
        Update asset metrics.

        Args:
            symbol: Asset symbol
            expected_return: Expected annual return
            volatility: Annual volatility
            current_weight: Current portfolio weight
        """
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        metrics = AssetMetrics(
            symbol=symbol,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            current_weight=current_weight,
        )
        self.add_asset(metrics)
        self._current_weights[symbol] = current_weight

    def set_correlation(self, symbol_a: str, symbol_b: str, correlation: float) -> None:
        """
        Set correlation between two assets.

        Args:
            symbol_a: First asset
            symbol_b: Second asset
            correlation: Correlation coefficient (-1 to 1)
        """
        if symbol_a not in self._correlations:
            self._correlations[symbol_a] = {}
        if symbol_b not in self._correlations:
            self._correlations[symbol_b] = {}

        self._correlations[symbol_a][symbol_b] = correlation
        self._correlations[symbol_b][symbol_a] = correlation

    def set_current_weights(self, weights: dict[str, float]) -> None:
        """Set current portfolio weights."""
        self._current_weights = weights.copy()
        for symbol, weight in weights.items():
            if symbol in self._assets:
                self._assets[symbol].current_weight = weight

    def add_constraint(self, constraint: PortfolioConstraint) -> None:
        """Add optimization constraint."""
        self._constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Clear all constraints."""
        self._constraints.clear()

    def optimize(
        self,
        strategy: OptimizationStrategy | None = None,
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation.

        Args:
            strategy: Override default strategy

        Returns:
            Optimized allocation
        """
        if strategy is None:
            strategy = self.strategy

        if not self._assets:
            return PortfolioAllocation(
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                expected_sharpe=0.0,
                strategy=strategy,
            )

        symbols = list(self._assets.keys())

        if strategy == OptimizationStrategy.EQUAL_WEIGHT:
            weights = self._equal_weight(symbols)
        elif strategy == OptimizationStrategy.RISK_PARITY:
            weights = self._risk_parity(symbols)
        elif strategy == OptimizationStrategy.MIN_VARIANCE:
            weights = self._min_variance(symbols)
        elif strategy == OptimizationStrategy.MAX_SHARPE:
            weights = self._max_sharpe(symbols)
        elif strategy == OptimizationStrategy.MAX_RETURN:
            weights = self._max_return(symbols)
        else:
            weights = self._equal_weight(symbols)

        # Apply constraints
        weights = self._apply_constraints(weights)

        # Calculate portfolio metrics
        exp_return = sum(
            weights.get(s, 0) * self._assets[s].expected_return
            for s in symbols
        )

        exp_vol = self._calculate_portfolio_volatility(weights)

        exp_sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0

        # Update target weights in assets
        for symbol, weight in weights.items():
            if symbol in self._assets:
                self._assets[symbol].target_weight = weight

        allocation = PortfolioAllocation(
            weights=weights,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            expected_sharpe=exp_sharpe,
            strategy=strategy,
        )

        self._target_allocation = allocation

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(allocation)
            except Exception:
                pass

        return allocation

    def _equal_weight(self, symbols: list[str]) -> dict[str, float]:
        """Equal weight allocation."""
        n = len(symbols)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {s: weight for s in symbols}

    def _risk_parity(self, symbols: list[str]) -> dict[str, float]:
        """Risk parity allocation (inverse volatility)."""
        if not symbols:
            return {}

        # Inverse volatility weights
        inv_vols = {}
        for symbol in symbols:
            vol = self._assets[symbol].volatility
            if vol > 0:
                inv_vols[symbol] = 1.0 / vol
            else:
                inv_vols[symbol] = 1.0

        total = sum(inv_vols.values())
        if total == 0:
            return self._equal_weight(symbols)

        return {s: inv_vols[s] / total for s in symbols}

    def _min_variance(self, symbols: list[str]) -> dict[str, float]:
        """Minimum variance portfolio."""
        # Simplified: use risk parity as approximation
        # Full implementation would require quadratic optimization
        return self._risk_parity(symbols)

    def _max_sharpe(self, symbols: list[str]) -> dict[str, float]:
        """Maximum Sharpe ratio portfolio."""
        if not symbols:
            return {}

        # Weight by Sharpe ratio (simplified)
        sharpes = {}
        for symbol in symbols:
            sharpe = max(0, self._assets[symbol].sharpe_ratio)
            sharpes[symbol] = sharpe

        total = sum(sharpes.values())
        if total == 0:
            return self._equal_weight(symbols)

        return {s: sharpes[s] / total for s in symbols}

    def _max_return(self, symbols: list[str]) -> dict[str, float]:
        """Maximum return portfolio (highest return asset)."""
        if not symbols:
            return {}

        # Find highest return asset
        best = max(symbols, key=lambda s: self._assets[s].expected_return)
        return {s: (1.0 if s == best else 0.0) for s in symbols}

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply portfolio constraints."""
        result = weights.copy()

        # Apply min/max weight constraints
        for symbol in result:
            result[symbol] = max(self.min_weight, min(self.max_weight, result[symbol]))

        # Apply custom constraints
        for constraint in self._constraints:
            if constraint.constraint_type == ConstraintType.MAX_WEIGHT:
                if constraint.symbol and constraint.symbol in result:
                    result[constraint.symbol] = min(result[constraint.symbol], constraint.value)
            elif constraint.constraint_type == ConstraintType.MIN_WEIGHT:
                if constraint.symbol and constraint.symbol in result:
                    result[constraint.symbol] = max(result[constraint.symbol], constraint.value)

        # Normalize to sum to 1
        total = sum(result.values())
        if total > 0:
            result = {s: w / total for s, w in result.items()}

        return result

    def _calculate_portfolio_volatility(self, weights: dict[str, float]) -> float:
        """Calculate portfolio volatility."""
        symbols = list(weights.keys())

        # Simplified: assume zero correlation for basic calculation
        # Full implementation would use correlation matrix
        variance = sum(
            (weights.get(s, 0) ** 2) * (self._assets[s].volatility ** 2)
            for s in symbols
        )

        # Add covariance terms if correlations available
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                corr = self._get_correlation(s1, s2)
                cov = corr * self._assets[s1].volatility * self._assets[s2].volatility
                variance += 2 * weights.get(s1, 0) * weights.get(s2, 0) * cov

        return math.sqrt(variance)

    def _get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get correlation between two assets."""
        if symbol_a == symbol_b:
            return 1.0
        if symbol_a in self._correlations:
            return self._correlations[symbol_a].get(symbol_b, 0.0)
        return 0.0

    def analyze_drift(self) -> DriftAnalysis:
        """
        Analyze portfolio drift from target allocation.

        Returns:
            Drift analysis
        """
        if not self._target_allocation:
            return DriftAnalysis(
                timestamp=datetime.now(),
                total_drift=0.0,
                max_drift_symbol="",
                max_drift_value=0.0,
                needs_rebalance=False,
                drift_by_symbol={},
            )

        drift_by_symbol = {}
        total_drift = 0.0
        max_drift = 0.0
        max_drift_symbol = ""

        for symbol in self._current_weights:
            current = self._current_weights.get(symbol, 0.0)
            target = self._target_allocation.weights.get(symbol, 0.0)
            drift = abs(current - target)
            drift_by_symbol[symbol] = drift
            total_drift += drift

            if drift > max_drift:
                max_drift = drift
                max_drift_symbol = symbol

        needs_rebalance = max_drift > self.rebalance_threshold

        return DriftAnalysis(
            timestamp=datetime.now(),
            total_drift=total_drift,
            max_drift_symbol=max_drift_symbol,
            max_drift_value=max_drift,
            needs_rebalance=needs_rebalance,
            drift_by_symbol=drift_by_symbol,
        )

    def generate_rebalance_plan(
        self,
        portfolio_value: Decimal,
    ) -> RebalancePlan:
        """
        Generate rebalancing plan.

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Rebalancing plan
        """
        if not self._target_allocation:
            return RebalancePlan(
                actions=[],
                total_portfolio_value=portfolio_value,
                total_turnover=0.0,
                estimated_cost=Decimal("0"),
            )

        actions = []
        total_turnover = 0.0

        for symbol in set(list(self._current_weights.keys()) + list(self._target_allocation.weights.keys())):
            current_weight = self._current_weights.get(symbol, 0.0)
            target_weight = self._target_allocation.weights.get(symbol, 0.0)
            weight_change = target_weight - current_weight

            if abs(weight_change) < 0.001:  # Skip tiny changes
                continue

            current_value = portfolio_value * Decimal(str(current_weight))
            target_value = portfolio_value * Decimal(str(target_weight))
            value_change = target_value - current_value

            action_type = "buy" if weight_change > 0 else "sell"

            actions.append(RebalanceAction(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_change=weight_change,
                current_value=current_value,
                target_value=target_value,
                value_change=value_change,
                action=action_type,
            ))

            total_turnover += abs(weight_change)

        # Estimate transaction cost
        turnover_value = portfolio_value * Decimal(str(total_turnover / 2))
        estimated_cost = turnover_value * Decimal(str(self.transaction_cost_bps / 10000))

        return RebalancePlan(
            actions=actions,
            total_portfolio_value=portfolio_value,
            total_turnover=total_turnover / 2,  # One-way turnover
            estimated_cost=estimated_cost,
        )

    def calculate_risk_metrics(
        self,
        weights: dict[str, float] | None = None,
    ) -> PortfolioRiskMetrics:
        """
        Calculate portfolio risk metrics.

        Args:
            weights: Portfolio weights (uses target if None)

        Returns:
            Risk metrics
        """
        if weights is None:
            if self._target_allocation:
                weights = self._target_allocation.weights
            else:
                weights = self._current_weights

        volatility = self._calculate_portfolio_volatility(weights)
        variance = volatility ** 2

        # VaR calculations (assuming normal distribution)
        # 95% VaR: 1.65 standard deviations
        var_95 = 1.65 * volatility
        # 99% VaR: 2.33 standard deviations
        var_99 = 2.33 * volatility

        # Expected shortfall (simplified)
        expected_shortfall = var_99 * 1.1

        # Build correlation matrix
        symbols = list(weights.keys())
        corr_matrix = {}
        for s1 in symbols:
            corr_matrix[s1] = {}
            for s2 in symbols:
                corr_matrix[s1][s2] = self._get_correlation(s1, s2)

        return PortfolioRiskMetrics(
            total_variance=variance,
            total_volatility=volatility,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=0.0,  # Would need historical data
            beta=1.0,  # Would need benchmark
            correlation_matrix=corr_matrix,
        )

    def get_allocation(self) -> PortfolioAllocation | None:
        """Get current target allocation."""
        return self._target_allocation

    def get_asset(self, symbol: str) -> AssetMetrics | None:
        """Get asset metrics."""
        return self._assets.get(symbol)

    def get_assets(self) -> list[str]:
        """Get list of asset symbols."""
        return list(self._assets.keys())

    def get_current_weights(self) -> dict[str, float]:
        """Get current weights."""
        return self._current_weights.copy()

    def add_callback(self, callback: Callable[[PortfolioAllocation], None]) -> None:
        """Add optimization callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[PortfolioAllocation], None]) -> bool:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def clear_asset(self, symbol: str) -> None:
        """Remove an asset."""
        self._assets.pop(symbol, None)
        self._current_weights.pop(symbol, None)
        self._correlations.pop(symbol, None)

    def clear_all(self) -> None:
        """Clear all data."""
        self._assets.clear()
        self._current_weights.clear()
        self._correlations.clear()
        self._target_allocation = None
        self._constraints.clear()


# Global instance
_optimizer: PortfolioOptimizer | None = None


def get_portfolio_optimizer() -> PortfolioOptimizer:
    """Get global portfolio optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    return _optimizer


def reset_portfolio_optimizer() -> None:
    """Reset global optimizer."""
    global _optimizer
    _optimizer = None
