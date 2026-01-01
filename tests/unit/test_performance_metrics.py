"""Unit tests for Performance Metrics."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.performance_metrics import (
    MetricType,
    EquityPoint,
    DrawdownPeriod,
    ReturnMetrics,
    RiskMetrics,
    RiskAdjustedMetrics,
    EfficiencyMetrics,
    ConsistencyMetrics,
    PerformanceReport,
    PerformanceCalculator,
    get_performance_calculator,
    reset_performance_calculator,
)


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self):
        """Should have expected metric type values."""
        assert MetricType.RETURN.value == "return"
        assert MetricType.RISK.value == "risk"
        assert MetricType.RISK_ADJUSTED.value == "risk_adjusted"
        assert MetricType.EFFICIENCY.value == "efficiency"
        assert MetricType.CONSISTENCY.value == "consistency"


class TestEquityPoint:
    """Tests for EquityPoint dataclass."""

    def test_create_equity_point(self):
        """Should create equity point."""
        point = EquityPoint(
            timestamp=time.time(),
            equity=Decimal("10000"),
            drawdown=Decimal("500"),
            drawdown_pct=5.0,
        )

        assert point.equity == Decimal("10000")
        assert point.drawdown_pct == 5.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        point = EquityPoint(
            timestamp=time.time(),
            equity=Decimal("10000"),
        )

        d = point.to_dict()

        assert "equity" in d
        assert "drawdown_pct" in d


class TestDrawdownPeriod:
    """Tests for DrawdownPeriod dataclass."""

    def test_create_drawdown_period(self):
        """Should create drawdown period."""
        period = DrawdownPeriod(
            start_time=time.time() - 86400,
            end_time=time.time(),
            peak_equity=Decimal("10000"),
            trough_equity=Decimal("9000"),
            max_drawdown=Decimal("1000"),
            max_drawdown_pct=10.0,
        )

        assert period.max_drawdown_pct == 10.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        period = DrawdownPeriod(
            start_time=time.time(),
            max_drawdown_pct=5.0,
        )

        d = period.to_dict()

        assert "max_drawdown_pct" in d


class TestReturnMetrics:
    """Tests for ReturnMetrics dataclass."""

    def test_create_return_metrics(self):
        """Should create return metrics."""
        metrics = ReturnMetrics(
            total_return=Decimal("1000"),
            total_return_pct=10.0,
            annualized_return=25.0,
        )

        assert metrics.total_return_pct == 10.0

    def test_positive_days_pct(self):
        """Should calculate positive days percentage."""
        metrics = ReturnMetrics(
            positive_days=60,
            negative_days=40,
        )

        assert metrics.positive_days_pct == 60.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = ReturnMetrics(annualized_return=25.0)

        d = metrics.to_dict()

        assert "annualized_return" in d


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_create_risk_metrics(self):
        """Should create risk metrics."""
        metrics = RiskMetrics(
            volatility=2.5,
            annualized_volatility=40.0,
            max_drawdown_pct=15.0,
        )

        assert metrics.max_drawdown_pct == 15.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = RiskMetrics(var_95=2.5)

        d = metrics.to_dict()

        assert "var_95" in d


class TestRiskAdjustedMetrics:
    """Tests for RiskAdjustedMetrics dataclass."""

    def test_create_risk_adjusted(self):
        """Should create risk-adjusted metrics."""
        metrics = RiskAdjustedMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
        )

        assert metrics.sharpe_ratio == 1.5

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = RiskAdjustedMetrics(omega_ratio=1.5)

        d = metrics.to_dict()

        assert "omega_ratio" in d


class TestEfficiencyMetrics:
    """Tests for EfficiencyMetrics dataclass."""

    def test_create_efficiency(self):
        """Should create efficiency metrics."""
        metrics = EfficiencyMetrics(
            profit_factor=2.0,
            win_rate=60.0,
            payoff_ratio=1.5,
        )

        assert metrics.profit_factor == 2.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = EfficiencyMetrics(kelly_criterion=0.25)

        d = metrics.to_dict()

        assert "kelly_criterion" in d


class TestConsistencyMetrics:
    """Tests for ConsistencyMetrics dataclass."""

    def test_create_consistency(self):
        """Should create consistency metrics."""
        metrics = ConsistencyMetrics(
            return_stability=0.95,
            recovery_factor=3.0,
            system_quality=2.5,
        )

        assert metrics.return_stability == 0.95

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = ConsistencyMetrics(k_ratio=0.5)

        d = metrics.to_dict()

        assert "k_ratio" in d


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_create_report(self):
        """Should create performance report."""
        report = PerformanceReport(
            period_start=time.time() - 86400,
            period_end=time.time(),
            initial_equity=Decimal("10000"),
            final_equity=Decimal("12000"),
            return_metrics=ReturnMetrics(),
            risk_metrics=RiskMetrics(),
            risk_adjusted=RiskAdjustedMetrics(),
            efficiency=EfficiencyMetrics(),
            consistency=ConsistencyMetrics(),
        )

        assert report.initial_equity == Decimal("10000")

    def test_period_days(self):
        """Should calculate period in days."""
        report = PerformanceReport(
            period_start=time.time() - 86400 * 30,  # 30 days ago
            period_end=time.time(),
            initial_equity=Decimal("10000"),
            final_equity=Decimal("12000"),
            return_metrics=ReturnMetrics(),
            risk_metrics=RiskMetrics(),
            risk_adjusted=RiskAdjustedMetrics(),
            efficiency=EfficiencyMetrics(),
            consistency=ConsistencyMetrics(),
        )

        assert report.period_days >= 29
        assert report.period_days <= 31

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = PerformanceReport(
            period_start=time.time(),
            period_end=time.time(),
            initial_equity=Decimal("10000"),
            final_equity=Decimal("11000"),
            return_metrics=ReturnMetrics(),
            risk_metrics=RiskMetrics(),
            risk_adjusted=RiskAdjustedMetrics(),
            efficiency=EfficiencyMetrics(),
            consistency=ConsistencyMetrics(),
        )

        d = report.to_dict()

        assert "return_metrics" in d
        assert "risk_metrics" in d


class TestPerformanceCalculator:
    """Tests for PerformanceCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator."""
        return PerformanceCalculator()

    @pytest.fixture
    def sample_equity_curve(self, calculator):
        """Add sample equity curve."""
        base_time = time.time()
        equities = [10000, 10200, 10100, 10300, 10150, 10400, 10350, 10500, 10450, 10600]

        for i, equity in enumerate(equities):
            calculator.add_equity_point(
                base_time + i * 86400,  # Daily points
                Decimal(str(equity)),
            )

        return calculator

    def test_add_equity_point(self, calculator):
        """Should add equity point."""
        calculator.add_equity_point(time.time(), Decimal("10000"))

        assert len(calculator._equity_curve) == 1

    def test_add_equity_point_calculates_drawdown(self, calculator):
        """Should calculate drawdown."""
        calculator.add_equity_point(time.time(), Decimal("10000"))
        calculator.add_equity_point(time.time() + 1, Decimal("9500"))

        assert calculator._equity_curve[-1].drawdown == Decimal("500")

    def test_add_equity_point_calculates_returns(self, calculator):
        """Should calculate daily returns."""
        calculator.add_equity_point(time.time(), Decimal("10000"))
        calculator.add_equity_point(time.time() + 1, Decimal("10100"))

        assert len(calculator._daily_returns) == 1
        assert calculator._daily_returns[0] == pytest.approx(0.01, rel=0.001)

    def test_calculate_return_metrics(self, sample_equity_curve):
        """Should calculate return metrics."""
        metrics = sample_equity_curve.calculate_return_metrics()

        assert metrics.total_return == Decimal("600")
        assert metrics.total_return_pct == 6.0

    def test_calculate_return_metrics_empty(self, calculator):
        """Should handle empty equity curve."""
        metrics = calculator.calculate_return_metrics()

        assert metrics.total_return == Decimal("0")

    def test_calculate_risk_metrics(self, sample_equity_curve):
        """Should calculate risk metrics."""
        metrics = sample_equity_curve.calculate_risk_metrics()

        assert metrics.volatility > 0
        assert metrics.max_drawdown_pct > 0

    def test_calculate_risk_metrics_empty(self, calculator):
        """Should handle empty returns."""
        metrics = calculator.calculate_risk_metrics()

        assert metrics.volatility == 0

    def test_calculate_risk_adjusted(self, sample_equity_curve):
        """Should calculate risk-adjusted metrics."""
        metrics = sample_equity_curve.calculate_risk_adjusted_metrics()

        # Should have calculated various ratios
        assert isinstance(metrics.sharpe_ratio, float)

    def test_calculate_efficiency_metrics(self, calculator):
        """Should calculate efficiency metrics."""
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 150},
            {"pnl": -30},
            {"pnl": 80},
        ]

        metrics = calculator.calculate_efficiency_metrics(trades)

        assert metrics.win_rate == 60.0  # 3/5

    def test_calculate_efficiency_profit_factor(self, calculator):
        """Should calculate profit factor."""
        trades = [
            {"pnl": 100},
            {"pnl": 100},
            {"pnl": -50},
        ]

        metrics = calculator.calculate_efficiency_metrics(trades)

        assert metrics.profit_factor == 4.0  # 200/50

    def test_calculate_consistency_metrics(self, sample_equity_curve):
        """Should calculate consistency metrics."""
        metrics = sample_equity_curve.calculate_consistency_metrics()

        assert isinstance(metrics.return_stability, float)
        assert isinstance(metrics.recovery_factor, float)

    def test_calculate_consistency_metrics_few_points(self, calculator):
        """Should handle few equity points."""
        for i in range(5):
            calculator.add_equity_point(time.time() + i, Decimal("10000"))

        metrics = calculator.calculate_consistency_metrics()

        # Should return default values
        assert metrics.return_stability == 0.0

    def test_get_full_report(self, sample_equity_curve):
        """Should generate full report."""
        trades = [{"pnl": 100}, {"pnl": -50}]

        report = sample_equity_curve.get_full_report(trades)

        assert report.initial_equity == Decimal("10000")
        assert report.final_equity == Decimal("10600")
        assert report.return_metrics is not None
        assert report.risk_metrics is not None

    def test_get_full_report_empty(self, calculator):
        """Should handle empty data."""
        report = calculator.get_full_report()

        assert report.initial_equity == Decimal("0")

    def test_reset(self, sample_equity_curve):
        """Should reset calculator."""
        sample_equity_curve.reset()

        assert len(sample_equity_curve._equity_curve) == 0
        assert len(sample_equity_curve._daily_returns) == 0


class TestGlobalCalculator:
    """Tests for global calculator functions."""

    def test_get_performance_calculator(self):
        """Should get or create calculator."""
        reset_performance_calculator()

        c1 = get_performance_calculator()
        c2 = get_performance_calculator()

        assert c1 is c2

    def test_reset_performance_calculator(self):
        """Should reset calculator."""
        c1 = get_performance_calculator()
        reset_performance_calculator()
        c2 = get_performance_calculator()

        assert c1 is not c2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_equity_point(self):
        """Should handle single point."""
        calculator = PerformanceCalculator()
        calculator.add_equity_point(time.time(), Decimal("10000"))

        metrics = calculator.calculate_return_metrics()

        assert metrics.total_return == Decimal("0")

    def test_constant_equity(self):
        """Should handle constant equity."""
        calculator = PerformanceCalculator()

        for i in range(10):
            calculator.add_equity_point(time.time() + i, Decimal("10000"))

        metrics = calculator.calculate_return_metrics()

        assert metrics.total_return_pct == 0.0

    def test_all_positive_returns(self):
        """Should handle all positive returns."""
        calculator = PerformanceCalculator()

        equity = 10000
        for i in range(20):
            calculator.add_equity_point(time.time() + i * 86400, Decimal(str(equity)))
            equity += 100

        metrics = calculator.calculate_risk_metrics()

        assert metrics.max_drawdown_pct == 0.0

    def test_large_drawdown(self):
        """Should handle large drawdown."""
        calculator = PerformanceCalculator()

        calculator.add_equity_point(time.time(), Decimal("10000"))
        calculator.add_equity_point(time.time() + 1, Decimal("5000"))

        metrics = calculator.calculate_risk_metrics()

        assert metrics.max_drawdown_pct == 50.0

    def test_no_trades_efficiency(self):
        """Should handle no trades."""
        calculator = PerformanceCalculator()

        metrics = calculator.calculate_efficiency_metrics([])

        assert metrics.profit_factor == 0.0
        assert metrics.win_rate == 0.0

    def test_all_winning_trades(self):
        """Should handle all wins."""
        calculator = PerformanceCalculator()

        trades = [{"pnl": 100} for _ in range(5)]

        metrics = calculator.calculate_efficiency_metrics(trades)

        assert metrics.win_rate == 100.0

    def test_all_losing_trades(self):
        """Should handle all losses."""
        calculator = PerformanceCalculator()

        trades = [{"pnl": -100} for _ in range(5)]

        metrics = calculator.calculate_efficiency_metrics(trades)

        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
