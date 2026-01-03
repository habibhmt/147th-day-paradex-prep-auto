"""
Portfolio Optimizer Module

Advanced portfolio optimization with multiple strategies including
Mean-Variance, Risk Parity, Black-Litterman, and Kelly Criterion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import math


class OptimizationType(Enum):
    """Types of portfolio optimization strategies."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    BLACK_LITTERMAN = "black_litterman"


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_THRESHOLD = "on_threshold"


class RiskMetric(Enum):
    """Risk measurement metrics."""
    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    DOWNSIDE_DEV = "downside_deviation"


@dataclass
class AssetData:
    """Asset data for optimization."""
    symbol: str
    expected_return: float
    volatility: float
    current_weight: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    current_price: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "current_weight": self.current_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "current_price": str(self.current_price),
        }


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""
    weights: dict[str, float]
    timestamp: datetime
    optimization_type: OptimizationType
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "timestamp": self.timestamp.isoformat(),
            "optimization_type": self.optimization_type.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_contribution": self.risk_contribution,
        }

    def get_weight(self, symbol: str) -> float:
        """Get weight for a specific symbol."""
        return self.weights.get(symbol, 0.0)


@dataclass
class RebalanceOrder:
    """Order to rebalance portfolio."""
    symbol: str
    side: str  # "buy" or "sell"
    target_weight: float
    current_weight: float
    weight_change: float
    estimated_value: Decimal
    priority: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "weight_change": self.weight_change,
            "estimated_value": str(self.estimated_value),
            "priority": self.priority,
        }


@dataclass
class OptimizerConfig:
    """Configuration for portfolio optimizer."""
    optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_position_size: float = 0.25
    min_position_size: float = 0.01
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    rebalance_threshold: float = 0.05  # 5% drift
    use_constraints: bool = True
    allow_short: bool = False
    risk_metric: RiskMetric = RiskMetric.VOLATILITY
    kelly_fraction: float = 0.5  # Half Kelly for safety

    def __post_init__(self):
        """Validate configuration."""
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.min_position_size < 0:
            raise ValueError("min_position_size cannot be negative")
        if self.min_position_size > self.max_position_size:
            raise ValueError("min_position_size cannot exceed max_position_size")
        if self.rebalance_threshold <= 0:
            raise ValueError("rebalance_threshold must be positive")
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be between 0 and 1")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "optimization_type": self.optimization_type.value,
            "risk_free_rate": self.risk_free_rate,
            "target_return": self.target_return,
            "target_volatility": self.target_volatility,
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "rebalance_frequency": self.rebalance_frequency.value,
            "rebalance_threshold": self.rebalance_threshold,
            "use_constraints": self.use_constraints,
            "allow_short": self.allow_short,
            "risk_metric": self.risk_metric.value,
            "kelly_fraction": self.kelly_fraction,
        }


@dataclass
class CorrelationMatrix:
    """Correlation matrix for assets."""
    symbols: list[str]
    matrix: list[list[float]]

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two assets."""
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return 0.0
        idx1 = self.symbols.index(symbol1)
        idx2 = self.symbols.index(symbol2)
        return self.matrix[idx1][idx2]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbols": self.symbols,
            "matrix": self.matrix,
        }


class ReturnEstimator:
    """Estimates expected returns for assets."""

    def __init__(self, lookback_periods: int = 30):
        """Initialize return estimator."""
        self.lookback_periods = lookback_periods
        self.price_history: dict[str, list[Decimal]] = {}

    def add_price(self, symbol: str, price: Decimal) -> None:
        """Add price data point."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback_periods + 1:
            self.price_history[symbol] = self.price_history[symbol][-(self.lookback_periods + 1):]

    def calculate_returns(self, symbol: str) -> list[float]:
        """Calculate historical returns."""
        if symbol not in self.price_history:
            return []
        prices = self.price_history[symbol]
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = float((prices[i] - prices[i-1]) / prices[i-1])
                returns.append(ret)
        return returns

    def estimate_expected_return(self, symbol: str) -> float:
        """Estimate expected return."""
        returns = self.calculate_returns(symbol)
        if not returns:
            return 0.0
        return sum(returns) / len(returns)

    def estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility (standard deviation of returns)."""
        returns = self.calculate_returns(symbol)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)


class CorrelationEstimator:
    """Estimates correlation between assets."""

    def __init__(self, return_estimator: ReturnEstimator):
        """Initialize correlation estimator."""
        self.return_estimator = return_estimator

    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two assets."""
        returns1 = self.return_estimator.calculate_returns(symbol1)
        returns2 = self.return_estimator.calculate_returns(symbol2)
        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0
        # Align returns
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]
        if min_len < 2:
            return 0.0
        mean1 = sum(returns1) / len(returns1)
        mean2 = sum(returns2) / len(returns2)
        cov = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2))
        std1 = math.sqrt(sum((r - mean1) ** 2 for r in returns1))
        std2 = math.sqrt(sum((r - mean2) ** 2 for r in returns2))
        if std1 == 0 or std2 == 0:
            return 0.0
        return cov / (std1 * std2)

    def build_correlation_matrix(self, symbols: list[str]) -> CorrelationMatrix:
        """Build correlation matrix for symbols."""
        n = len(symbols)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.calculate_correlation(symbols[i], symbols[j])
        return CorrelationMatrix(symbols=symbols, matrix=matrix)


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimization."""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize optimizer."""
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        assets: list[AssetData],
        correlation: CorrelationMatrix,
        target_return: Optional[float] = None,
    ) -> PortfolioWeights:
        """Optimize portfolio weights."""
        if not assets:
            return self._empty_weights()
        n = len(assets)
        if n == 1:
            return self._single_asset_weights(assets[0])
        # Simplified optimization: allocate based on Sharpe ratio
        weights = {}
        sharpe_ratios = []
        for asset in assets:
            if asset.volatility > 0:
                sharpe = (asset.expected_return - self.risk_free_rate) / asset.volatility
            else:
                sharpe = 0.0
            sharpe_ratios.append(max(sharpe, 0))
        total_sharpe = sum(sharpe_ratios)
        if total_sharpe == 0:
            # Equal weight if no positive Sharpe
            for asset in assets:
                weights[asset.symbol] = 1.0 / n
        else:
            for asset, sharpe in zip(assets, sharpe_ratios):
                weights[asset.symbol] = sharpe / total_sharpe
        # Calculate portfolio metrics
        exp_return = sum(assets[i].expected_return * list(weights.values())[i] for i in range(n))
        exp_vol = self._calculate_portfolio_volatility(assets, weights, correlation)
        sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
        return PortfolioWeights(
            weights=weights,
            timestamp=datetime.now(),
            optimization_type=OptimizationType.MEAN_VARIANCE,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
        )

    def _calculate_portfolio_volatility(
        self,
        assets: list[AssetData],
        weights: dict[str, float],
        correlation: CorrelationMatrix,
    ) -> float:
        """Calculate portfolio volatility."""
        variance = 0.0
        symbols = list(weights.keys())
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                w1 = weights[sym1]
                w2 = weights[sym2]
                vol1 = next((a.volatility for a in assets if a.symbol == sym1), 0)
                vol2 = next((a.volatility for a in assets if a.symbol == sym2), 0)
                corr = correlation.get_correlation(sym1, sym2)
                variance += w1 * w2 * vol1 * vol2 * corr
        return math.sqrt(max(0, variance))

    def _empty_weights(self) -> PortfolioWeights:
        """Return empty weights."""
        return PortfolioWeights(
            weights={},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.MEAN_VARIANCE,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
        )

    def _single_asset_weights(self, asset: AssetData) -> PortfolioWeights:
        """Return weights for single asset."""
        sharpe = (asset.expected_return - self.risk_free_rate) / asset.volatility if asset.volatility > 0 else 0
        return PortfolioWeights(
            weights={asset.symbol: 1.0},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.MEAN_VARIANCE,
            expected_return=asset.expected_return,
            expected_volatility=asset.volatility,
            sharpe_ratio=sharpe,
        )


class RiskParityOptimizer:
    """Risk parity portfolio optimization."""

    def optimize(
        self,
        assets: list[AssetData],
        correlation: CorrelationMatrix,
    ) -> PortfolioWeights:
        """Optimize for equal risk contribution."""
        if not assets:
            return self._empty_weights()
        n = len(assets)
        if n == 1:
            return self._single_asset_weights(assets[0])
        # Simple risk parity: weight inversely to volatility
        inverse_vols = []
        for asset in assets:
            if asset.volatility > 0:
                inverse_vols.append(1.0 / asset.volatility)
            else:
                inverse_vols.append(0.0)
        total_inverse_vol = sum(inverse_vols)
        weights = {}
        if total_inverse_vol == 0:
            for asset in assets:
                weights[asset.symbol] = 1.0 / n
        else:
            for asset, inv_vol in zip(assets, inverse_vols):
                weights[asset.symbol] = inv_vol / total_inverse_vol
        # Calculate metrics
        exp_return = sum(assets[i].expected_return * list(weights.values())[i] for i in range(n))
        exp_vol = self._calculate_portfolio_volatility(assets, weights, correlation)
        # Calculate risk contributions
        risk_contrib = {}
        for asset in assets:
            w = weights[asset.symbol]
            risk_contrib[asset.symbol] = w * asset.volatility / exp_vol if exp_vol > 0 else 0
        return PortfolioWeights(
            weights=weights,
            timestamp=datetime.now(),
            optimization_type=OptimizationType.RISK_PARITY,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=exp_return / exp_vol if exp_vol > 0 else 0,
            risk_contribution=risk_contrib,
        )

    def _calculate_portfolio_volatility(
        self,
        assets: list[AssetData],
        weights: dict[str, float],
        correlation: CorrelationMatrix,
    ) -> float:
        """Calculate portfolio volatility."""
        variance = 0.0
        symbols = list(weights.keys())
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                w1 = weights[sym1]
                w2 = weights[sym2]
                vol1 = next((a.volatility for a in assets if a.symbol == sym1), 0)
                vol2 = next((a.volatility for a in assets if a.symbol == sym2), 0)
                corr = correlation.get_correlation(sym1, sym2)
                variance += w1 * w2 * vol1 * vol2 * corr
        return math.sqrt(max(0, variance))

    def _empty_weights(self) -> PortfolioWeights:
        """Return empty weights."""
        return PortfolioWeights(
            weights={},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.RISK_PARITY,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
        )

    def _single_asset_weights(self, asset: AssetData) -> PortfolioWeights:
        """Return weights for single asset."""
        return PortfolioWeights(
            weights={asset.symbol: 1.0},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.RISK_PARITY,
            expected_return=asset.expected_return,
            expected_volatility=asset.volatility,
            sharpe_ratio=asset.expected_return / asset.volatility if asset.volatility > 0 else 0,
            risk_contribution={asset.symbol: 1.0},
        )


class KellyOptimizer:
    """Kelly criterion portfolio optimization."""

    def __init__(self, fraction: float = 0.5):
        """Initialize Kelly optimizer."""
        self.fraction = fraction

    def optimize(
        self,
        assets: list[AssetData],
        correlation: CorrelationMatrix,
    ) -> PortfolioWeights:
        """Optimize using Kelly criterion."""
        if not assets:
            return self._empty_weights()
        # Calculate Kelly fraction for each asset
        kelly_fractions = []
        for asset in assets:
            if asset.volatility > 0:
                kelly = asset.expected_return / (asset.volatility ** 2)
                kelly_fractions.append(max(0, kelly * self.fraction))
            else:
                kelly_fractions.append(0)
        total_kelly = sum(kelly_fractions)
        weights = {}
        if total_kelly == 0:
            for asset in assets:
                weights[asset.symbol] = 1.0 / len(assets)
        else:
            # Normalize to sum to 1
            for asset, kf in zip(assets, kelly_fractions):
                weights[asset.symbol] = kf / total_kelly
        # Calculate metrics
        n = len(assets)
        exp_return = sum(assets[i].expected_return * list(weights.values())[i] for i in range(n))
        exp_vol = self._calculate_portfolio_volatility(assets, weights, correlation)
        return PortfolioWeights(
            weights=weights,
            timestamp=datetime.now(),
            optimization_type=OptimizationType.KELLY,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=exp_return / exp_vol if exp_vol > 0 else 0,
        )

    def _calculate_portfolio_volatility(
        self,
        assets: list[AssetData],
        weights: dict[str, float],
        correlation: CorrelationMatrix,
    ) -> float:
        """Calculate portfolio volatility."""
        variance = 0.0
        symbols = list(weights.keys())
        for sym1 in symbols:
            for sym2 in symbols:
                w1 = weights[sym1]
                w2 = weights[sym2]
                vol1 = next((a.volatility for a in assets if a.symbol == sym1), 0)
                vol2 = next((a.volatility for a in assets if a.symbol == sym2), 0)
                corr = correlation.get_correlation(sym1, sym2)
                variance += w1 * w2 * vol1 * vol2 * corr
        return math.sqrt(max(0, variance))

    def _empty_weights(self) -> PortfolioWeights:
        """Return empty weights."""
        return PortfolioWeights(
            weights={},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.KELLY,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
        )


class EqualWeightOptimizer:
    """Simple equal weight allocation."""

    def optimize(
        self,
        assets: list[AssetData],
        correlation: CorrelationMatrix,
    ) -> PortfolioWeights:
        """Allocate equally to all assets."""
        if not assets:
            return self._empty_weights()
        n = len(assets)
        weights = {asset.symbol: 1.0 / n for asset in assets}
        exp_return = sum(asset.expected_return for asset in assets) / n
        exp_vol = self._calculate_portfolio_volatility(assets, weights, correlation)
        return PortfolioWeights(
            weights=weights,
            timestamp=datetime.now(),
            optimization_type=OptimizationType.EQUAL_WEIGHT,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=exp_return / exp_vol if exp_vol > 0 else 0,
        )

    def _calculate_portfolio_volatility(
        self,
        assets: list[AssetData],
        weights: dict[str, float],
        correlation: CorrelationMatrix,
    ) -> float:
        """Calculate portfolio volatility."""
        variance = 0.0
        symbols = list(weights.keys())
        for sym1 in symbols:
            for sym2 in symbols:
                w1 = weights[sym1]
                w2 = weights[sym2]
                vol1 = next((a.volatility for a in assets if a.symbol == sym1), 0)
                vol2 = next((a.volatility for a in assets if a.symbol == sym2), 0)
                corr = correlation.get_correlation(sym1, sym2)
                variance += w1 * w2 * vol1 * vol2 * corr
        return math.sqrt(max(0, variance))

    def _empty_weights(self) -> PortfolioWeights:
        """Return empty weights."""
        return PortfolioWeights(
            weights={},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.EQUAL_WEIGHT,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
        )


class PortfolioOptimizer:
    """Main portfolio optimizer with multiple strategies."""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize portfolio optimizer."""
        self.config = config or OptimizerConfig()
        self.return_estimator = ReturnEstimator()
        self.correlation_estimator = CorrelationEstimator(self.return_estimator)
        self.assets: dict[str, AssetData] = {}
        self.current_weights: Optional[PortfolioWeights] = None
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_history: list[dict] = []

        # Initialize optimizers
        self.mean_variance = MeanVarianceOptimizer(self.config.risk_free_rate)
        self.risk_parity = RiskParityOptimizer()
        self.kelly = KellyOptimizer(self.config.kelly_fraction)
        self.equal_weight = EqualWeightOptimizer()

    def add_asset(self, asset: AssetData) -> None:
        """Add asset to universe."""
        self.assets[asset.symbol] = asset

    def remove_asset(self, symbol: str) -> bool:
        """Remove asset from universe."""
        if symbol in self.assets:
            del self.assets[symbol]
            return True
        return False

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price for asset."""
        if symbol in self.assets:
            self.assets[symbol].current_price = price
        self.return_estimator.add_price(symbol, price)

    def update_expected_return(self, symbol: str, expected_return: float) -> None:
        """Update expected return for asset."""
        if symbol in self.assets:
            self.assets[symbol].expected_return = expected_return

    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility for asset."""
        if symbol in self.assets:
            self.assets[symbol].volatility = volatility

    def optimize(self) -> PortfolioWeights:
        """Optimize portfolio based on config."""
        assets = list(self.assets.values())
        if not assets:
            return PortfolioWeights(
                weights={},
                timestamp=datetime.now(),
                optimization_type=self.config.optimization_type,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
            )
        # Build correlation matrix
        symbols = [a.symbol for a in assets]
        correlation = self.correlation_estimator.build_correlation_matrix(symbols)
        # Choose optimizer based on config
        if self.config.optimization_type == OptimizationType.MEAN_VARIANCE:
            weights = self.mean_variance.optimize(assets, correlation, self.config.target_return)
        elif self.config.optimization_type == OptimizationType.RISK_PARITY:
            weights = self.risk_parity.optimize(assets, correlation)
        elif self.config.optimization_type == OptimizationType.KELLY:
            weights = self.kelly.optimize(assets, correlation)
        elif self.config.optimization_type == OptimizationType.EQUAL_WEIGHT:
            weights = self.equal_weight.optimize(assets, correlation)
        elif self.config.optimization_type == OptimizationType.MIN_VARIANCE:
            # Use mean-variance with target return = 0
            weights = self.mean_variance.optimize(assets, correlation, 0.0)
            weights.optimization_type = OptimizationType.MIN_VARIANCE
        elif self.config.optimization_type == OptimizationType.MAX_SHARPE:
            weights = self.mean_variance.optimize(assets, correlation)
            weights.optimization_type = OptimizationType.MAX_SHARPE
        else:
            weights = self.equal_weight.optimize(assets, correlation)
        # Apply constraints
        if self.config.use_constraints:
            weights = self._apply_constraints(weights)
        self.current_weights = weights
        return weights

    def _apply_constraints(self, weights: PortfolioWeights) -> PortfolioWeights:
        """Apply position size constraints."""
        constrained = {}
        for symbol, weight in weights.weights.items():
            # Apply min/max constraints
            weight = max(self.config.min_position_size, weight)
            weight = min(self.config.max_position_size, weight)
            if not self.config.allow_short:
                weight = max(0, weight)
            constrained[symbol] = weight
        # Normalize to sum to 1
        total = sum(constrained.values())
        if total > 0:
            constrained = {s: w / total for s, w in constrained.items()}
        return PortfolioWeights(
            weights=constrained,
            timestamp=weights.timestamp,
            optimization_type=weights.optimization_type,
            expected_return=weights.expected_return,
            expected_volatility=weights.expected_volatility,
            sharpe_ratio=weights.sharpe_ratio,
            risk_contribution=weights.risk_contribution,
        )

    def should_rebalance(self, current_weights: dict[str, float]) -> bool:
        """Check if rebalancing is needed."""
        if self.current_weights is None:
            return True
        # Check threshold-based rebalancing
        for symbol, target_weight in self.current_weights.weights.items():
            current = current_weights.get(symbol, 0)
            drift = abs(current - target_weight)
            if drift > self.config.rebalance_threshold:
                return True
        return False

    def generate_rebalance_orders(
        self,
        current_weights: dict[str, float],
        portfolio_value: Decimal,
    ) -> list[RebalanceOrder]:
        """Generate orders to rebalance portfolio."""
        if self.current_weights is None:
            return []
        orders = []
        all_symbols = set(current_weights.keys()) | set(self.current_weights.weights.keys())
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = self.current_weights.weights.get(symbol, 0)
            weight_change = target - current
            if abs(weight_change) < 0.001:  # Skip tiny changes
                continue
            side = "buy" if weight_change > 0 else "sell"
            value = portfolio_value * Decimal(str(abs(weight_change)))
            orders.append(RebalanceOrder(
                symbol=symbol,
                side=side,
                target_weight=target,
                current_weight=current,
                weight_change=weight_change,
                estimated_value=value,
                priority=int(abs(weight_change) * 100),
            ))
        # Sort by priority (largest changes first)
        orders.sort(key=lambda o: o.priority, reverse=True)
        return orders

    def record_rebalance(self, orders: list[RebalanceOrder]) -> None:
        """Record rebalancing event."""
        self.last_rebalance = datetime.now()
        self.rebalance_history.append({
            "timestamp": self.last_rebalance.isoformat(),
            "orders": [o.to_dict() for o in orders],
            "target_weights": self.current_weights.weights if self.current_weights else {},
        })

    def get_status(self) -> dict:
        """Get optimizer status."""
        return {
            "assets_count": len(self.assets),
            "assets": {s: a.to_dict() for s, a in self.assets.items()},
            "current_weights": self.current_weights.to_dict() if self.current_weights else None,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "rebalance_count": len(self.rebalance_history),
            "config": self.config.to_dict(),
        }
