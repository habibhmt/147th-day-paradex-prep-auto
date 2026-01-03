"""
Tests for Portfolio Optimizer Module
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.portfolio.optimizer import (
    OptimizationType,
    RebalanceFrequency,
    RiskMetric,
    AssetData,
    PortfolioWeights,
    RebalanceOrder,
    OptimizerConfig,
    CorrelationMatrix,
    ReturnEstimator,
    CorrelationEstimator,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    KellyOptimizer,
    EqualWeightOptimizer,
    PortfolioOptimizer,
)


class TestOptimizationType:
    def test_all_types_defined(self):
        assert OptimizationType.MEAN_VARIANCE.value == "mean_variance"
        assert OptimizationType.RISK_PARITY.value == "risk_parity"
        assert OptimizationType.EQUAL_WEIGHT.value == "equal_weight"
        assert OptimizationType.KELLY.value == "kelly"


class TestAssetData:
    def test_creation(self):
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        assert asset.symbol == "ETH"
        assert asset.expected_return == 0.1

    def test_to_dict(self):
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        result = asset.to_dict()
        assert result["symbol"] == "ETH"


class TestPortfolioWeights:
    def test_creation(self):
        weights = PortfolioWeights(
            weights={"ETH": 0.6, "BTC": 0.4},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.MEAN_VARIANCE,
            expected_return=0.12,
            expected_volatility=0.25,
            sharpe_ratio=0.4,
        )
        assert weights.weights["ETH"] == 0.6

    def test_get_weight(self):
        weights = PortfolioWeights(
            weights={"ETH": 0.6},
            timestamp=datetime.now(),
            optimization_type=OptimizationType.MEAN_VARIANCE,
            expected_return=0.12,
            expected_volatility=0.25,
            sharpe_ratio=0.4,
        )
        assert weights.get_weight("ETH") == 0.6
        assert weights.get_weight("INVALID") == 0.0


class TestOptimizerConfig:
    def test_valid_config(self):
        config = OptimizerConfig(risk_free_rate=0.03)
        assert config.risk_free_rate == 0.03

    def test_invalid_max_position_size(self):
        with pytest.raises(ValueError):
            OptimizerConfig(max_position_size=0)

    def test_invalid_kelly_fraction(self):
        with pytest.raises(ValueError):
            OptimizerConfig(kelly_fraction=1.5)


class TestCorrelationMatrix:
    def test_get_correlation(self):
        matrix = CorrelationMatrix(
            symbols=["ETH", "BTC"],
            matrix=[[1.0, 0.7], [0.7, 1.0]],
        )
        assert matrix.get_correlation("ETH", "BTC") == 0.7
        assert matrix.get_correlation("INVALID", "BTC") == 0.0


class TestReturnEstimator:
    def test_add_price(self):
        estimator = ReturnEstimator()
        estimator.add_price("ETH", Decimal("2000"))
        assert "ETH" in estimator.price_history

    def test_calculate_returns(self):
        estimator = ReturnEstimator()
        for p in [100, 110, 105]:
            estimator.add_price("ETH", Decimal(str(p)))
        returns = estimator.calculate_returns("ETH")
        assert len(returns) == 2

    def test_estimate_volatility(self):
        estimator = ReturnEstimator()
        for p in [100, 110, 90, 120]:
            estimator.add_price("ETH", Decimal(str(p)))
        vol = estimator.estimate_volatility("ETH")
        assert vol > 0


class TestMeanVarianceOptimizer:
    def test_optimize_empty(self):
        optimizer = MeanVarianceOptimizer()
        correlation = CorrelationMatrix(symbols=[], matrix=[])
        weights = optimizer.optimize([], correlation)
        assert len(weights.weights) == 0

    def test_optimize_single(self):
        optimizer = MeanVarianceOptimizer()
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        correlation = CorrelationMatrix(symbols=["ETH"], matrix=[[1.0]])
        weights = optimizer.optimize([asset], correlation)
        assert weights.weights["ETH"] == 1.0

    def test_optimize_multiple(self):
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        assets = [
            AssetData(symbol="ETH", expected_return=0.15, volatility=0.4),
            AssetData(symbol="BTC", expected_return=0.12, volatility=0.35),
        ]
        correlation = CorrelationMatrix(
            symbols=["ETH", "BTC"],
            matrix=[[1.0, 0.7], [0.7, 1.0]],
        )
        weights = optimizer.optimize(assets, correlation)
        assert len(weights.weights) == 2


class TestRiskParityOptimizer:
    def test_optimize_empty(self):
        optimizer = RiskParityOptimizer()
        correlation = CorrelationMatrix(symbols=[], matrix=[])
        weights = optimizer.optimize([], correlation)
        assert len(weights.weights) == 0

    def test_optimize_risk_parity(self):
        optimizer = RiskParityOptimizer()
        assets = [
            AssetData(symbol="LOW", expected_return=0.08, volatility=0.1),
            AssetData(symbol="HIGH", expected_return=0.15, volatility=0.4),
        ]
        correlation = CorrelationMatrix(
            symbols=["LOW", "HIGH"],
            matrix=[[1.0, 0.3], [0.3, 1.0]],
        )
        weights = optimizer.optimize(assets, correlation)
        assert weights.weights["LOW"] > weights.weights["HIGH"]


class TestKellyOptimizer:
    def test_optimize_empty(self):
        optimizer = KellyOptimizer()
        correlation = CorrelationMatrix(symbols=[], matrix=[])
        weights = optimizer.optimize([], correlation)
        assert len(weights.weights) == 0

    def test_optimize_kelly(self):
        optimizer = KellyOptimizer(fraction=0.5)
        assets = [
            AssetData(symbol="ETH", expected_return=0.2, volatility=0.3),
            AssetData(symbol="BTC", expected_return=0.15, volatility=0.35),
        ]
        correlation = CorrelationMatrix(
            symbols=["ETH", "BTC"],
            matrix=[[1.0, 0.6], [0.6, 1.0]],
        )
        weights = optimizer.optimize(assets, correlation)
        assert len(weights.weights) == 2


class TestEqualWeightOptimizer:
    def test_optimize_equal(self):
        optimizer = EqualWeightOptimizer()
        assets = [
            AssetData(symbol="ETH", expected_return=0.1, volatility=0.3),
            AssetData(symbol="BTC", expected_return=0.12, volatility=0.35),
        ]
        correlation = CorrelationMatrix(
            symbols=["ETH", "BTC"],
            matrix=[[1.0, 0.7], [0.7, 1.0]],
        )
        weights = optimizer.optimize(assets, correlation)
        assert weights.weights["ETH"] == pytest.approx(0.5, rel=0.01)


class TestPortfolioOptimizer:
    def test_creation(self):
        optimizer = PortfolioOptimizer()
        assert optimizer.config is not None

    def test_add_asset(self):
        optimizer = PortfolioOptimizer()
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        optimizer.add_asset(asset)
        assert "ETH" in optimizer.assets

    def test_remove_asset(self):
        optimizer = PortfolioOptimizer()
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        optimizer.add_asset(asset)
        assert optimizer.remove_asset("ETH")
        assert "ETH" not in optimizer.assets

    def test_update_price(self):
        optimizer = PortfolioOptimizer()
        asset = AssetData(symbol="ETH", expected_return=0.1, volatility=0.3)
        optimizer.add_asset(asset)
        optimizer.update_price("ETH", Decimal("2000"))
        assert optimizer.assets["ETH"].current_price == Decimal("2000")

    def test_optimize_empty(self):
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize()
        assert len(weights.weights) == 0

    def test_optimize_mean_variance(self):
        config = OptimizerConfig(optimization_type=OptimizationType.MEAN_VARIANCE)
        optimizer = PortfolioOptimizer(config)
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        optimizer.add_asset(AssetData(symbol="BTC", expected_return=0.12, volatility=0.3))
        weights = optimizer.optimize()
        assert weights.optimization_type == OptimizationType.MEAN_VARIANCE

    def test_optimize_risk_parity(self):
        config = OptimizerConfig(optimization_type=OptimizationType.RISK_PARITY)
        optimizer = PortfolioOptimizer(config)
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        weights = optimizer.optimize()
        assert weights.optimization_type == OptimizationType.RISK_PARITY

    def test_optimize_kelly(self):
        config = OptimizerConfig(optimization_type=OptimizationType.KELLY)
        optimizer = PortfolioOptimizer(config)
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        weights = optimizer.optimize()
        assert weights.optimization_type == OptimizationType.KELLY

    def test_optimize_equal_weight(self):
        config = OptimizerConfig(optimization_type=OptimizationType.EQUAL_WEIGHT)
        optimizer = PortfolioOptimizer(config)
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        weights = optimizer.optimize()
        assert weights.optimization_type == OptimizationType.EQUAL_WEIGHT

    def test_should_rebalance_no_current(self):
        optimizer = PortfolioOptimizer()
        assert optimizer.should_rebalance({"ETH": 0.5})

    def test_generate_rebalance_orders(self):
        optimizer = PortfolioOptimizer()
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        optimizer.add_asset(AssetData(symbol="BTC", expected_return=0.12, volatility=0.3))
        optimizer.optimize()
        orders = optimizer.generate_rebalance_orders({"ETH": 0.3, "BTC": 0.7}, Decimal("10000"))
        assert isinstance(orders, list)

    def test_record_rebalance(self):
        optimizer = PortfolioOptimizer()
        orders = [
            RebalanceOrder(
                symbol="ETH", side="buy", target_weight=0.5,
                current_weight=0.3, weight_change=0.2, estimated_value=Decimal("1000"),
            )
        ]
        optimizer.record_rebalance(orders)
        assert optimizer.last_rebalance is not None

    def test_get_status(self):
        optimizer = PortfolioOptimizer()
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        status = optimizer.get_status()
        assert status["assets_count"] == 1


class TestPortfolioOptimizerIntegration:
    def test_full_cycle(self):
        config = OptimizerConfig(optimization_type=OptimizationType.MEAN_VARIANCE)
        optimizer = PortfolioOptimizer(config)
        optimizer.add_asset(AssetData(symbol="ETH", expected_return=0.15, volatility=0.35))
        optimizer.add_asset(AssetData(symbol="BTC", expected_return=0.12, volatility=0.3))
        for i in range(10):
            optimizer.update_price("ETH", Decimal(str(2000 + i * 10)))
            optimizer.update_price("BTC", Decimal(str(50000 + i * 250)))
        weights = optimizer.optimize()
        assert len(weights.weights) == 2
        status = optimizer.get_status()
        assert status["assets_count"] == 2
