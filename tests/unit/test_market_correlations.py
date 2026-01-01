"""Unit tests for Market Correlations module."""

import pytest
import time
import math
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.market_correlations import (
    CorrelationType,
    RelationshipStrength,
    RelationshipDirection,
    CointegrationResult,
    PriceData,
    CorrelationPair,
    RollingCorrelation,
    LeadLagAnalysis,
    CointegrationTest,
    CorrelationMatrix,
    PairScore,
    CorrelationCalculator,
    MarketCorrelationAnalyzer,
    get_correlation_analyzer,
    reset_correlation_analyzer,
)


class TestCorrelationType:
    """Tests for CorrelationType enum."""

    def test_type_values(self):
        """Should have expected correlation type values."""
        assert CorrelationType.PEARSON.value == "pearson"
        assert CorrelationType.SPEARMAN.value == "spearman"
        assert CorrelationType.KENDALL.value == "kendall"


class TestRelationshipStrength:
    """Tests for RelationshipStrength enum."""

    def test_strength_values(self):
        """Should have expected strength values."""
        assert RelationshipStrength.VERY_STRONG.value == "very_strong"
        assert RelationshipStrength.STRONG.value == "strong"
        assert RelationshipStrength.MODERATE.value == "moderate"
        assert RelationshipStrength.WEAK.value == "weak"
        assert RelationshipStrength.NEGLIGIBLE.value == "negligible"


class TestRelationshipDirection:
    """Tests for RelationshipDirection enum."""

    def test_direction_values(self):
        """Should have expected direction values."""
        assert RelationshipDirection.POSITIVE.value == "positive"
        assert RelationshipDirection.NEGATIVE.value == "negative"
        assert RelationshipDirection.NEUTRAL.value == "neutral"


class TestCointegrationResult:
    """Tests for CointegrationResult enum."""

    def test_result_values(self):
        """Should have expected result values."""
        assert CointegrationResult.COINTEGRATED.value == "cointegrated"
        assert CointegrationResult.NOT_COINTEGRATED.value == "not_cointegrated"
        assert CointegrationResult.INCONCLUSIVE.value == "inconclusive"


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_create_price_data(self):
        """Should create price data."""
        data = PriceData(
            market="BTC-USD-PERP",
            timestamp=time.time(),
            price=Decimal("50000"),
            volume=Decimal("100"),
        )

        assert data.market == "BTC-USD-PERP"
        assert data.price == Decimal("50000")

    def test_to_dict(self):
        """Should convert to dictionary."""
        data = PriceData(
            market="ETH-USD-PERP",
            timestamp=time.time(),
            price=Decimal("3000"),
            returns=0.05,
        )

        d = data.to_dict()

        assert d["market"] == "ETH-USD-PERP"
        assert d["returns"] == 0.05


class TestCorrelationPair:
    """Tests for CorrelationPair dataclass."""

    def test_create_pair(self):
        """Should create correlation pair."""
        pair = CorrelationPair(
            market_a="BTC-USD-PERP",
            market_b="ETH-USD-PERP",
            correlation=0.85,
            sample_size=100,
        )

        assert pair.correlation == 0.85

    def test_strength_very_strong(self):
        """Should detect very strong correlation."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.85,
        )

        assert pair.strength == RelationshipStrength.VERY_STRONG

    def test_strength_strong(self):
        """Should detect strong correlation."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.65,
        )

        assert pair.strength == RelationshipStrength.STRONG

    def test_strength_moderate(self):
        """Should detect moderate correlation."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.45,
        )

        assert pair.strength == RelationshipStrength.MODERATE

    def test_strength_weak(self):
        """Should detect weak correlation."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.25,
        )

        assert pair.strength == RelationshipStrength.WEAK

    def test_strength_negligible(self):
        """Should detect negligible correlation."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.1,
        )

        assert pair.strength == RelationshipStrength.NEGLIGIBLE

    def test_direction_positive(self):
        """Should detect positive direction."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.5,
        )

        assert pair.direction == RelationshipDirection.POSITIVE

    def test_direction_negative(self):
        """Should detect negative direction."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=-0.5,
        )

        assert pair.direction == RelationshipDirection.NEGATIVE

    def test_direction_neutral(self):
        """Should detect neutral direction."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.05,
        )

        assert pair.direction == RelationshipDirection.NEUTRAL

    def test_to_dict(self):
        """Should convert to dictionary."""
        pair = CorrelationPair(
            market_a="A",
            market_b="B",
            correlation=0.7,
        )

        d = pair.to_dict()

        assert d["strength"] == "strong"
        assert d["direction"] == "positive"


class TestRollingCorrelation:
    """Tests for RollingCorrelation dataclass."""

    def test_create_rolling(self):
        """Should create rolling correlation."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.5, 0.6, 0.7],
        )

        assert rolling.window_size == 20
        assert len(rolling.correlations) == 3

    def test_current(self):
        """Should get current correlation."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.5, 0.6, 0.7],
        )

        assert rolling.current == 0.7

    def test_current_empty(self):
        """Should handle empty correlations."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
        )

        assert rolling.current == 0.0

    def test_mean(self):
        """Should calculate mean."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.4, 0.5, 0.6],
        )

        assert rolling.mean == 0.5

    def test_std(self):
        """Should calculate std."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.5, 0.5, 0.5],
        )

        assert rolling.std == 0.0

    def test_min_max(self):
        """Should get min and max."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.3, 0.5, 0.8],
        )

        assert rolling.min == 0.3
        assert rolling.max == 0.8

    def test_is_stable(self):
        """Should check stability."""
        stable = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.5, 0.52, 0.48, 0.51],
        )

        unstable = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.2, 0.8, 0.3, 0.9],
        )

        assert stable.is_stable() is True
        assert unstable.is_stable() is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        rolling = RollingCorrelation(
            market_a="A",
            market_b="B",
            window_size=20,
            correlations=[0.5, 0.6],
        )

        d = rolling.to_dict()

        assert "current" in d
        assert "mean" in d
        assert "is_stable" in d


class TestLeadLagAnalysis:
    """Tests for LeadLagAnalysis dataclass."""

    def test_create_analysis(self):
        """Should create lead-lag analysis."""
        analysis = LeadLagAnalysis(
            leader="BTC",
            follower="ETH",
            lag_periods=3,
            correlation_at_lag=0.75,
            optimal_lag=3,
        )

        assert analysis.leader == "BTC"
        assert analysis.lag_periods == 3

    def test_to_dict(self):
        """Should convert to dictionary."""
        analysis = LeadLagAnalysis(
            leader="A",
            follower="B",
            lag_periods=2,
            correlation_at_lag=0.6,
        )

        d = analysis.to_dict()

        assert d["leader"] == "A"
        assert d["lag_periods"] == 2


class TestCointegrationTest:
    """Tests for CointegrationTest dataclass."""

    def test_create_test(self):
        """Should create cointegration test."""
        test = CointegrationTest(
            market_a="BTC",
            market_b="ETH",
            test_statistic=-4.5,
            result=CointegrationResult.COINTEGRATED,
        )

        assert test.is_cointegrated is True

    def test_not_cointegrated(self):
        """Should detect not cointegrated."""
        test = CointegrationTest(
            market_a="A",
            market_b="B",
            result=CointegrationResult.NOT_COINTEGRATED,
        )

        assert test.is_cointegrated is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        test = CointegrationTest(
            market_a="A",
            market_b="B",
            hedge_ratio=0.8,
        )

        d = test.to_dict()

        assert d["hedge_ratio"] == 0.8


class TestCorrelationMatrix:
    """Tests for CorrelationMatrix dataclass."""

    def test_create_matrix(self):
        """Should create correlation matrix."""
        matrix = CorrelationMatrix(
            markets=["A", "B", "C"],
            matrix={
                "A": {"A": 1.0, "B": 0.8, "C": 0.5},
                "B": {"B": 1.0, "C": 0.6},
                "C": {"C": 1.0},
            },
        )

        assert len(matrix.markets) == 3

    def test_get_correlation(self):
        """Should get correlation between markets."""
        matrix = CorrelationMatrix(
            markets=["A", "B"],
            matrix={
                "A": {"A": 1.0, "B": 0.8},
            },
        )

        assert matrix.get_correlation("A", "B") == 0.8
        assert matrix.get_correlation("B", "A") == 0.8  # Symmetric
        assert matrix.get_correlation("A", "A") == 1.0

    def test_get_most_correlated(self):
        """Should get most correlated markets."""
        matrix = CorrelationMatrix(
            markets=["A", "B", "C"],
            matrix={
                "A": {"A": 1.0, "B": 0.9, "C": 0.5},
            },
        )

        most = matrix.get_most_correlated("A", n=2)

        assert len(most) == 2
        assert most[0][0] == "B"
        assert most[0][1] == 0.9

    def test_get_least_correlated(self):
        """Should get least correlated markets."""
        matrix = CorrelationMatrix(
            markets=["A", "B", "C"],
            matrix={
                "A": {"A": 1.0, "B": 0.9, "C": 0.2},
            },
        )

        least = matrix.get_least_correlated("A", n=1)

        assert least[0][0] == "C"

    def test_to_dict(self):
        """Should convert to dictionary."""
        matrix = CorrelationMatrix(markets=["A", "B"])

        d = matrix.to_dict()

        assert "markets" in d
        assert "matrix" in d


class TestPairScore:
    """Tests for PairScore dataclass."""

    def test_create_score(self):
        """Should create pair score."""
        score = PairScore(
            market_a="A",
            market_b="B",
            correlation_score=80.0,
            cointegration_score=100.0,
            total_score=85.0,
            rank=1,
        )

        assert score.total_score == 85.0
        assert score.rank == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        score = PairScore(
            market_a="A",
            market_b="B",
            total_score=75.0,
        )

        d = score.to_dict()

        assert d["total_score"] == 75.0


class TestCorrelationCalculator:
    """Tests for CorrelationCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator."""
        return CorrelationCalculator()

    def test_calculate_pearson_perfect_positive(self, calculator):
        """Should calculate perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        corr, p_value = calculator.calculate_pearson(x, y)

        assert corr == pytest.approx(1.0, abs=0.01)

    def test_calculate_pearson_perfect_negative(self, calculator):
        """Should calculate perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        corr, p_value = calculator.calculate_pearson(x, y)

        assert corr == pytest.approx(-1.0, abs=0.01)

    def test_calculate_pearson_no_correlation(self, calculator):
        """Should handle no correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 1.0, 4.0, 2.0, 3.0]  # Random

        corr, p_value = calculator.calculate_pearson(x, y)

        assert abs(corr) < 0.5  # Low correlation

    def test_calculate_pearson_empty(self, calculator):
        """Should handle empty data."""
        corr, p_value = calculator.calculate_pearson([], [])

        assert corr == 0.0
        assert p_value == 1.0

    def test_calculate_pearson_single_point(self, calculator):
        """Should handle single point."""
        corr, p_value = calculator.calculate_pearson([1.0], [2.0])

        assert corr == 0.0

    def test_calculate_spearman(self, calculator):
        """Should calculate Spearman correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 3.0, 2.0, 5.0, 4.0]

        corr, p_value = calculator.calculate_spearman(x, y)

        assert isinstance(corr, float)

    def test_calculate_kendall(self, calculator):
        """Should calculate Kendall's tau."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        corr, p_value = calculator.calculate_kendall(x, y)

        assert corr == 1.0  # Perfect concordance


class TestMarketCorrelationAnalyzer:
    """Tests for MarketCorrelationAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return MarketCorrelationAnalyzer(
            window_size=50,
            min_sample_size=10,
        )

    @pytest.fixture
    def sample_prices(self, analyzer):
        """Add sample prices."""
        base_time = time.time()

        # Create correlated and negatively correlated price series
        # Use alternating pattern for negative correlation
        for i in range(50):
            btc_price = Decimal(str(50000 + i * 100))
            eth_price = Decimal(str(3000 + i * 6))  # Positively correlated

            # For negative correlation, when BTC goes up, SOL goes down and vice versa
            if i % 2 == 0:
                sol_price = Decimal(str(100 + i * 2))
            else:
                sol_price = Decimal(str(100 - i * 2 + 50))

            analyzer.add_price("BTC-USD-PERP", base_time + i, btc_price)
            analyzer.add_price("ETH-USD-PERP", base_time + i, eth_price)
            analyzer.add_price("SOL-USD-PERP", base_time + i, sol_price)

        return analyzer

    def test_add_price(self, analyzer):
        """Should add price data."""
        result = analyzer.add_price("BTC-USD-PERP", time.time(), Decimal("50000"))

        assert result == 0.0  # First price has no return

    def test_add_price_calculates_return(self, analyzer):
        """Should calculate return."""
        analyzer.add_price("BTC-USD-PERP", time.time(), Decimal("50000"))
        result = analyzer.add_price("BTC-USD-PERP", time.time() + 1, Decimal("51000"))

        assert result == pytest.approx(0.02, abs=0.001)

    def test_add_prices_batch(self, analyzer):
        """Should add batch prices."""
        prices = {
            "BTC": [(time.time(), Decimal("50000")), (time.time() + 1, Decimal("51000"))],
            "ETH": [(time.time(), Decimal("3000")), (time.time() + 1, Decimal("3100"))],
        }

        analyzer.add_prices_batch(prices)

        assert analyzer.get_sample_size("BTC") == 1  # 1 return from 2 prices
        assert analyzer.get_sample_size("ETH") == 1

    def test_calculate_correlation(self, sample_prices):
        """Should calculate correlation."""
        pair = sample_prices.calculate_correlation("BTC-USD-PERP", "ETH-USD-PERP")

        assert pair.correlation > 0.9  # Highly correlated
        assert pair.sample_size > 0

    def test_calculate_correlation_different(self, sample_prices):
        """Should detect different correlation than BTC-ETH."""
        pair_btc_eth = sample_prices.calculate_correlation("BTC-USD-PERP", "ETH-USD-PERP")
        pair_btc_sol = sample_prices.calculate_correlation("BTC-USD-PERP", "SOL-USD-PERP")

        # BTC-ETH should be more correlated than BTC-SOL
        assert abs(pair_btc_eth.correlation) > abs(pair_btc_sol.correlation)

    def test_calculate_correlation_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        analyzer.add_price("A", time.time(), Decimal("100"))
        analyzer.add_price("B", time.time(), Decimal("200"))

        pair = analyzer.calculate_correlation("A", "B")

        assert pair.correlation == 0.0

    def test_calculate_correlation_matrix(self, sample_prices):
        """Should calculate correlation matrix."""
        matrix = sample_prices.calculate_correlation_matrix()

        assert len(matrix.markets) == 3
        assert matrix.get_correlation("BTC-USD-PERP", "ETH-USD-PERP") > 0.8

    def test_calculate_rolling_correlation(self, sample_prices):
        """Should calculate rolling correlation."""
        rolling = sample_prices.calculate_rolling_correlation(
            "BTC-USD-PERP",
            "ETH-USD-PERP",
            window_size=20,
        )

        assert len(rolling.correlations) > 0
        assert rolling.current != 0.0

    def test_analyze_lead_lag(self, sample_prices):
        """Should analyze lead-lag relationship."""
        analysis = sample_prices.analyze_lead_lag(
            "BTC-USD-PERP",
            "ETH-USD-PERP",
            max_lag=5,
        )

        assert analysis.leader in ["BTC-USD-PERP", "ETH-USD-PERP"]
        assert analysis.follower in ["BTC-USD-PERP", "ETH-USD-PERP"]
        assert len(analysis.lag_correlations) > 0

    def test_test_cointegration(self, sample_prices):
        """Should test cointegration."""
        result = sample_prices.test_cointegration("BTC-USD-PERP", "ETH-USD-PERP")

        assert result.result in [
            CointegrationResult.COINTEGRATED,
            CointegrationResult.NOT_COINTEGRATED,
            CointegrationResult.INCONCLUSIVE,
        ]
        assert result.hedge_ratio != 0.0

    def test_test_cointegration_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        result = analyzer.test_cointegration("A", "B")

        assert result.result == CointegrationResult.INCONCLUSIVE

    def test_find_best_pairs(self, sample_prices):
        """Should find best trading pairs."""
        pairs = sample_prices.find_best_pairs(top_n=3)

        assert len(pairs) <= 3
        if pairs:
            assert pairs[0].rank == 1
            assert pairs[0].total_score >= pairs[-1].total_score

    def test_find_best_pairs_min_correlation(self, sample_prices):
        """Should filter by minimum correlation."""
        pairs = sample_prices.find_best_pairs(min_correlation=0.95)

        for pair in pairs:
            corr = sample_prices.calculate_correlation(pair.market_a, pair.market_b)
            assert abs(corr.correlation) >= 0.95

    def test_get_highly_correlated_pairs(self, sample_prices):
        """Should get highly correlated pairs."""
        pairs = sample_prices.get_highly_correlated_pairs(threshold=0.8)

        for pair in pairs:
            assert abs(pair.correlation) >= 0.8

    def test_get_uncorrelated_pairs(self, sample_prices):
        """Should get uncorrelated pairs."""
        pairs = sample_prices.get_uncorrelated_pairs(threshold=0.5)

        for pair in pairs:
            assert abs(pair.correlation) <= 0.5

    def test_add_callback(self, analyzer):
        """Should add callback."""
        callback = MagicMock()
        analyzer.add_callback(callback)

        assert callback in analyzer._callbacks

    def test_remove_callback(self, analyzer):
        """Should remove callback."""
        callback = MagicMock()
        analyzer.add_callback(callback)
        analyzer.remove_callback(callback)

        assert callback not in analyzer._callbacks

    def test_get_markets(self, sample_prices):
        """Should get list of markets."""
        markets = sample_prices.get_markets()

        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_get_sample_size(self, sample_prices):
        """Should get sample size."""
        size = sample_prices.get_sample_size("BTC-USD-PERP")

        assert size > 0

    def test_clear_market(self, sample_prices):
        """Should clear market data."""
        sample_prices.clear_market("BTC-USD-PERP")

        assert sample_prices.get_sample_size("BTC-USD-PERP") == 0

    def test_clear_all(self, sample_prices):
        """Should clear all data."""
        sample_prices.clear_all()

        assert len(sample_prices.get_markets()) == 0


class TestGlobalCorrelationAnalyzer:
    """Tests for global analyzer functions."""

    def test_get_correlation_analyzer(self):
        """Should get or create analyzer."""
        reset_correlation_analyzer()

        a1 = get_correlation_analyzer()
        a2 = get_correlation_analyzer()

        assert a1 is a2

    def test_reset_correlation_analyzer(self):
        """Should reset analyzer."""
        a1 = get_correlation_analyzer()
        reset_correlation_analyzer()
        a2 = get_correlation_analyzer()

        assert a1 is not a2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_market(self):
        """Should handle single market."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(10):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))

        matrix = analyzer.calculate_correlation_matrix()

        assert len(matrix.markets) == 1

    def test_constant_prices(self):
        """Should handle constant prices."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(10):
            analyzer.add_price("A", time.time() + i, Decimal("100"))
            analyzer.add_price("B", time.time() + i, Decimal("200"))

        pair = analyzer.calculate_correlation("A", "B")

        # All returns are 0, correlation undefined
        assert pair.correlation == 0.0

    def test_identical_series(self):
        """Should handle identical series."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(20):
            price = Decimal(str(100 + i * 5))
            analyzer.add_price("A", time.time() + i, price)
            analyzer.add_price("B", time.time() + i, price)

        pair = analyzer.calculate_correlation("A", "B")

        assert pair.correlation == pytest.approx(1.0, abs=0.01)

    def test_opposite_series(self):
        """Should handle opposite series (negatively correlated returns)."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        # Create series with opposite movements in returns
        # A goes: 100, 110, 100, 110, 100 (alternating up/down)
        # B goes: 100, 90, 100, 90, 100 (opposite alternating)
        base = time.time()
        for i in range(20):
            if i % 2 == 0:
                a_price = Decimal("100")
                b_price = Decimal("100")
            else:
                a_price = Decimal("110")
                b_price = Decimal("90")

            analyzer.add_price("A", base + i, a_price)
            analyzer.add_price("B", base + i, b_price)

        pair = analyzer.calculate_correlation("A", "B")

        # Should be negatively correlated
        assert pair.correlation < 0

    def test_missing_market(self):
        """Should handle missing market."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(10):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))

        pair = analyzer.calculate_correlation("A", "MISSING")

        assert pair.sample_size == 0

    def test_window_trimming(self):
        """Should trim data to window size."""
        analyzer = MarketCorrelationAnalyzer(window_size=10, min_sample_size=5)

        for i in range(50):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))

        # Data should be trimmed to window_size * 2
        assert len(analyzer._price_data["A"]) <= 20

    def test_different_sample_sizes(self):
        """Should handle different sample sizes."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(20):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))

        for i in range(10):
            analyzer.add_price("B", time.time() + i, Decimal(str(200 + i)))

        pair = analyzer.calculate_correlation("A", "B")

        # Should use minimum of both
        assert pair.sample_size <= 10

    def test_spearman_correlation(self):
        """Should calculate Spearman correlation."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(20):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))
            analyzer.add_price("B", time.time() + i, Decimal(str(200 + i * 2)))

        pair = analyzer.calculate_correlation("A", "B", CorrelationType.SPEARMAN)

        assert pair.correlation_type == CorrelationType.SPEARMAN

    def test_kendall_correlation(self):
        """Should calculate Kendall correlation."""
        analyzer = MarketCorrelationAnalyzer(min_sample_size=5)

        for i in range(20):
            analyzer.add_price("A", time.time() + i, Decimal(str(100 + i)))
            analyzer.add_price("B", time.time() + i, Decimal(str(200 + i * 2)))

        pair = analyzer.calculate_correlation("A", "B", CorrelationType.KENDALL)

        assert pair.correlation_type == CorrelationType.KENDALL
