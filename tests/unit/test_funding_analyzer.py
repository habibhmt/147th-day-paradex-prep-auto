"""Tests for Funding Rate Analyzer module."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.analytics.funding_analyzer import (
    CrossMarketFunding,
    FundingAnalysis,
    FundingDirection,
    FundingHistory,
    FundingOpportunity,
    FundingPayment,
    FundingRate,
    FundingRateAnalyzer,
    FundingRegime,
    FundingTrend,
    get_funding_analyzer,
    reset_funding_analyzer,
)


class TestFundingDirectionEnum:
    """Tests for FundingDirection enum."""

    def test_all_directions(self):
        """Test all funding directions."""
        directions = [
            FundingDirection.LONG_PAYS_SHORT,
            FundingDirection.SHORT_PAYS_LONG,
            FundingDirection.NEUTRAL,
        ]
        assert len(directions) == 3

    def test_direction_values(self):
        """Test direction values."""
        assert FundingDirection.LONG_PAYS_SHORT.value == "long_pays_short"


class TestFundingTrendEnum:
    """Tests for FundingTrend enum."""

    def test_all_trends(self):
        """Test all trends."""
        trends = [
            FundingTrend.RISING,
            FundingTrend.FALLING,
            FundingTrend.STABLE,
            FundingTrend.VOLATILE,
        ]
        assert len(trends) == 4


class TestFundingRegimeEnum:
    """Tests for FundingRegime enum."""

    def test_all_regimes(self):
        """Test all regimes."""
        regimes = [
            FundingRegime.HIGH_POSITIVE,
            FundingRegime.POSITIVE,
            FundingRegime.NEUTRAL,
            FundingRegime.NEGATIVE,
            FundingRegime.HIGH_NEGATIVE,
        ]
        assert len(regimes) == 5


class TestFundingRate:
    """Tests for FundingRate dataclass."""

    def test_create_rate(self):
        """Test creating funding rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),  # 0.03%
            timestamp=datetime.now(),
        )
        assert rate.market == "BTC-USD-PERP"
        assert rate.rate == Decimal("0.0003")

    def test_rate_bps(self):
        """Test rate in basis points."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),  # 0.03%
            timestamp=datetime.now(),
        )
        assert rate.rate_bps == 3.0  # 3 bps

    def test_direction_long_pays(self):
        """Test direction when positive."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.003"),  # 0.3%
            timestamp=datetime.now(),
        )
        assert rate.direction == FundingDirection.LONG_PAYS_SHORT

    def test_direction_short_pays(self):
        """Test direction when negative."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("-0.003"),
            timestamp=datetime.now(),
        )
        assert rate.direction == FundingDirection.SHORT_PAYS_LONG

    def test_direction_neutral(self):
        """Test direction when near zero."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0001"),
            timestamp=datetime.now(),
        )
        assert rate.direction == FundingDirection.NEUTRAL

    def test_annualized_rate(self):
        """Test annualized rate calculation."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),  # 0.03%
            timestamp=datetime.now(),
        )
        # 0.03% * 3 * 365 = 32.85%
        assert rate.annualized_rate == pytest.approx(32.85, rel=0.01)

    def test_to_dict(self):
        """Test converting to dict."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),
            timestamp=datetime.now(),
        )
        d = rate.to_dict()
        assert d["market"] == "BTC-USD-PERP"
        assert "rate_bps" in d


class TestFundingHistory:
    """Tests for FundingHistory dataclass."""

    @pytest.fixture
    def sample_rates(self):
        """Create sample rates."""
        now = datetime.now()
        return [
            FundingRate(market="BTC-USD-PERP", rate=Decimal("0.0001"), timestamp=now - timedelta(hours=2)),
            FundingRate(market="BTC-USD-PERP", rate=Decimal("0.0002"), timestamp=now - timedelta(hours=1)),
            FundingRate(market="BTC-USD-PERP", rate=Decimal("0.0003"), timestamp=now),
        ]

    def test_create_history(self, sample_rates):
        """Test creating history."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        assert history.market == "BTC-USD-PERP"
        assert len(history.rates) == 3

    def test_avg_rate(self, sample_rates):
        """Test average rate calculation."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        assert history.avg_rate == Decimal("0.0002")

    def test_max_rate(self, sample_rates):
        """Test max rate."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        assert history.max_rate == Decimal("0.0003")

    def test_min_rate(self, sample_rates):
        """Test min rate."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        assert history.min_rate == Decimal("0.0001")

    def test_cumulative_rate(self, sample_rates):
        """Test cumulative rate."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        assert history.cumulative_rate == Decimal("0.0006")

    def test_to_dict(self, sample_rates):
        """Test converting to dict."""
        history = FundingHistory(
            market="BTC-USD-PERP",
            rates=sample_rates,
            start_time=sample_rates[0].timestamp,
            end_time=sample_rates[-1].timestamp,
        )
        d = history.to_dict()
        assert d["rates_count"] == 3


class TestFundingPayment:
    """Tests for FundingPayment dataclass."""

    def test_create_payment(self):
        """Test creating payment."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            position_size=Decimal("10"),
            position_side="long",
            funding_rate=Decimal("0.0003"),
            payment=Decimal("-0.15"),
            timestamp=datetime.now(),
        )
        assert payment.market == "BTC-USD-PERP"
        assert payment.payment == Decimal("-0.15")

    def test_is_receiving_true(self):
        """Test is_receiving when positive."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            position_size=Decimal("10"),
            position_side="short",
            funding_rate=Decimal("0.0003"),
            payment=Decimal("0.15"),
            timestamp=datetime.now(),
        )
        assert payment.is_receiving is True

    def test_is_receiving_false(self):
        """Test is_receiving when negative."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            position_size=Decimal("10"),
            position_side="long",
            funding_rate=Decimal("0.0003"),
            payment=Decimal("-0.15"),
            timestamp=datetime.now(),
        )
        assert payment.is_receiving is False

    def test_to_dict(self):
        """Test converting to dict."""
        payment = FundingPayment(
            market="BTC-USD-PERP",
            position_size=Decimal("10"),
            position_side="long",
            funding_rate=Decimal("0.0003"),
            payment=Decimal("-0.15"),
            timestamp=datetime.now(),
        )
        d = payment.to_dict()
        assert d["position_side"] == "long"


class TestFundingOpportunity:
    """Tests for FundingOpportunity dataclass."""

    def test_create_opportunity(self):
        """Test creating opportunity."""
        opp = FundingOpportunity(
            market="BTC-USD-PERP",
            current_rate=Decimal("0.0005"),
            avg_rate_24h=Decimal("0.0004"),
            predicted_rate=Decimal("0.0003"),
            recommended_side="short",
            expected_payment_8h=Decimal("5"),
            expected_payment_24h=Decimal("15"),
            confidence=0.8,
            regime=FundingRegime.HIGH_POSITIVE,
            trend=FundingTrend.STABLE,
        )
        assert opp.recommended_side == "short"
        assert opp.confidence == 0.8

    def test_to_dict(self):
        """Test converting to dict."""
        opp = FundingOpportunity(
            market="BTC-USD-PERP",
            current_rate=Decimal("0.0005"),
            avg_rate_24h=Decimal("0.0004"),
            predicted_rate=Decimal("0.0003"),
            recommended_side="short",
            expected_payment_8h=Decimal("5"),
            expected_payment_24h=Decimal("15"),
            confidence=0.8,
            regime=FundingRegime.HIGH_POSITIVE,
            trend=FundingTrend.STABLE,
        )
        d = opp.to_dict()
        assert d["regime"] == "high_positive"


class TestFundingRateAnalyzer:
    """Tests for FundingRateAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    @pytest.fixture
    def analyzer_with_data(self, analyzer):
        """Create analyzer with sample data."""
        now = datetime.now()
        for i in range(24):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003") + Decimal(str(i * 0.00001)),
                timestamp=now - timedelta(hours=24 - i),
            )
            analyzer.add_rate(rate)
        return analyzer

    def test_init_defaults(self):
        """Test default initialization."""
        analyzer = FundingRateAnalyzer()
        assert analyzer.history_window_hours == 168

    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = FundingRateAnalyzer(
            history_window_hours=72,
            prediction_lookback=12,
        )
        assert analyzer.history_window_hours == 72

    def test_add_rate(self, analyzer):
        """Test adding a rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        current = analyzer.get_current_rate("BTC-USD-PERP")
        assert current is not None
        assert current.rate == Decimal("0.0003")

    def test_update_rates(self, analyzer):
        """Test updating rates."""
        now = datetime.now()
        rates = [
            (now - timedelta(hours=2), Decimal("0.0001")),
            (now - timedelta(hours=1), Decimal("0.0002")),
            (now, Decimal("0.0003")),
        ]
        analyzer.update_rates("BTC-USD-PERP", rates)

        current = analyzer.get_current_rate("BTC-USD-PERP")
        assert current.rate == Decimal("0.0003")

    def test_get_current_rate(self, analyzer_with_data):
        """Test getting current rate."""
        rate = analyzer_with_data.get_current_rate("BTC-USD-PERP")
        assert rate is not None

    def test_get_current_rate_missing(self, analyzer):
        """Test getting rate for unknown market."""
        rate = analyzer.get_current_rate("UNKNOWN")
        assert rate is None

    def test_get_history(self, analyzer_with_data):
        """Test getting history."""
        history = analyzer_with_data.get_history("BTC-USD-PERP")
        assert history is not None
        assert len(history.rates) == 24

    def test_get_history_limited(self, analyzer_with_data):
        """Test getting limited history."""
        history = analyzer_with_data.get_history("BTC-USD-PERP", hours=4)
        assert history is not None
        assert len(history.rates) <= 5

    def test_get_history_missing(self, analyzer):
        """Test getting history for unknown market."""
        history = analyzer.get_history("UNKNOWN")
        assert history is None


class TestRegimeDetection:
    """Tests for funding regime detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_detect_high_positive(self, analyzer):
        """Test detecting high positive regime."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.001"),  # 0.1%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.HIGH_POSITIVE

    def test_detect_positive(self, analyzer):
        """Test detecting positive regime."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),  # 0.03%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.POSITIVE

    def test_detect_neutral(self, analyzer):
        """Test detecting neutral regime."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.00005"),  # 0.005%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.NEUTRAL

    def test_detect_negative(self, analyzer):
        """Test detecting negative regime."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("-0.0003"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.NEGATIVE

    def test_detect_high_negative(self, analyzer):
        """Test detecting high negative regime."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("-0.001"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.HIGH_NEGATIVE

    def test_detect_unknown_market(self, analyzer):
        """Test detecting regime for unknown market."""
        regime = analyzer.detect_regime("UNKNOWN")
        assert regime == FundingRegime.NEUTRAL


class TestTrendDetection:
    """Tests for funding trend detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_detect_rising_trend(self, analyzer):
        """Test detecting rising trend."""
        now = datetime.now()
        for i in range(10):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0001") * (i + 1),  # Increasing
                timestamp=now - timedelta(hours=10 - i),
            )
            analyzer.add_rate(rate)

        trend = analyzer.detect_trend("BTC-USD-PERP")
        assert trend == FundingTrend.RISING

    def test_detect_falling_trend(self, analyzer):
        """Test detecting falling trend."""
        now = datetime.now()
        for i in range(10):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.001") - Decimal("0.0001") * i,  # Decreasing
                timestamp=now - timedelta(hours=10 - i),
            )
            analyzer.add_rate(rate)

        trend = analyzer.detect_trend("BTC-USD-PERP")
        assert trend == FundingTrend.FALLING

    def test_detect_stable_trend(self, analyzer):
        """Test detecting stable trend."""
        now = datetime.now()
        for i in range(10):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003"),  # Constant
                timestamp=now - timedelta(hours=10 - i),
            )
            analyzer.add_rate(rate)

        trend = analyzer.detect_trend("BTC-USD-PERP")
        assert trend == FundingTrend.STABLE

    def test_detect_unknown_market(self, analyzer):
        """Test trend for unknown market."""
        trend = analyzer.detect_trend("UNKNOWN")
        assert trend == FundingTrend.STABLE


class TestAverageCalculation:
    """Tests for average calculation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with data."""
        analyzer = FundingRateAnalyzer()
        now = datetime.now()
        for i in range(48):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003"),
                timestamp=now - timedelta(hours=48 - i),
            )
            analyzer.add_rate(rate)
        return analyzer

    def test_calculate_average(self, analyzer):
        """Test calculating average."""
        avg = analyzer.calculate_average("BTC-USD-PERP", 24)
        assert avg == Decimal("0.0003")

    def test_calculate_average_no_data(self):
        """Test average with no data."""
        analyzer = FundingRateAnalyzer()
        avg = analyzer.calculate_average("UNKNOWN", 24)
        assert avg == Decimal("0")


class TestPrediction:
    """Tests for rate prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with data."""
        analyzer = FundingRateAnalyzer()
        now = datetime.now()
        for i in range(24):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003"),
                timestamp=now - timedelta(hours=24 - i),
            )
            analyzer.add_rate(rate)
        return analyzer

    def test_predict_rate(self, analyzer):
        """Test predicting rate."""
        predicted = analyzer.predict_rate("BTC-USD-PERP")
        # Should be close to average with mean reversion
        assert predicted is not None
        assert abs(predicted - Decimal("0.0003")) < Decimal("0.001")

    def test_predict_rate_no_data(self):
        """Test prediction with no data."""
        analyzer = FundingRateAnalyzer()
        predicted = analyzer.predict_rate("UNKNOWN")
        assert predicted == Decimal("0")


class TestVolatility:
    """Tests for volatility calculation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_calculate_volatility(self, analyzer):
        """Test calculating volatility."""
        now = datetime.now()
        for i in range(24):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003") + Decimal(str((i % 2) * 0.0001)),
                timestamp=now - timedelta(hours=24 - i),
            )
            analyzer.add_rate(rate)

        volatility = analyzer.calculate_volatility("BTC-USD-PERP")
        assert volatility > 0

    def test_volatility_no_data(self, analyzer):
        """Test volatility with no data."""
        volatility = analyzer.calculate_volatility("UNKNOWN")
        assert volatility == 0.0


class TestMeanReversion:
    """Tests for mean reversion score."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_mean_reversion_score(self, analyzer):
        """Test mean reversion score calculation."""
        now = datetime.now()
        for i in range(24):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0003"),
                timestamp=now - timedelta(hours=24 - i),
            )
            analyzer.add_rate(rate)

        score = analyzer.calculate_mean_reversion_score("BTC-USD-PERP")
        assert -1 <= score <= 1

    def test_mean_reversion_no_data(self, analyzer):
        """Test with no data."""
        score = analyzer.calculate_mean_reversion_score("UNKNOWN")
        assert score == 0.0


class TestPaymentEstimation:
    """Tests for payment estimation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        analyzer = FundingRateAnalyzer()
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),
            timestamp=datetime.now(),
            mark_price=Decimal("50000"),
        )
        analyzer.add_rate(rate)
        return analyzer

    def test_estimate_payment_long(self, analyzer):
        """Test payment estimate for long position."""
        payment = analyzer.estimate_payment(
            "BTC-USD-PERP",
            Decimal("10"),  # 10 BTC
            "long",
            Decimal("50000"),
        )
        assert payment is not None
        # Long pays when rate is positive
        assert payment.payment < 0

    def test_estimate_payment_short(self, analyzer):
        """Test payment estimate for short position."""
        payment = analyzer.estimate_payment(
            "BTC-USD-PERP",
            Decimal("10"),
            "short",
            Decimal("50000"),
        )
        assert payment is not None
        # Short receives when rate is positive
        assert payment.payment > 0

    def test_estimate_payment_no_data(self, analyzer):
        """Test payment with no market data."""
        payment = analyzer.estimate_payment("UNKNOWN", Decimal("10"), "long")
        assert payment is None


class TestOpportunityFinding:
    """Tests for opportunity finding."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_find_opportunity_positive(self, analyzer):
        """Test finding opportunity with positive rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0005"),  # 0.05%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        opp = analyzer.find_opportunity("BTC-USD-PERP")
        assert opp is not None
        assert opp.recommended_side == "short"

    def test_find_opportunity_negative(self, analyzer):
        """Test finding opportunity with negative rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("-0.0005"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        opp = analyzer.find_opportunity("BTC-USD-PERP")
        assert opp is not None
        assert opp.recommended_side == "long"

    def test_no_opportunity_neutral(self, analyzer):
        """Test no opportunity when neutral."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.00005"),  # Very small
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        opp = analyzer.find_opportunity("BTC-USD-PERP")
        assert opp is None


class TestAnalysis:
    """Tests for complete analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with data."""
        analyzer = FundingRateAnalyzer()
        now = datetime.now()
        for i in range(24):
            rate = FundingRate(
                market="BTC-USD-PERP",
                rate=Decimal("0.0005"),
                timestamp=now - timedelta(hours=24 - i),
            )
            analyzer.add_rate(rate)
        return analyzer

    def test_analyze(self, analyzer):
        """Test complete analysis."""
        analysis = analyzer.analyze("BTC-USD-PERP")
        assert analysis is not None
        assert analysis.market == "BTC-USD-PERP"
        assert analysis.current_rate is not None
        assert analysis.regime is not None
        assert analysis.trend is not None

    def test_analyze_no_data(self):
        """Test analysis with no data."""
        analyzer = FundingRateAnalyzer()
        analysis = analyzer.analyze("UNKNOWN")
        assert analysis is None


class TestCrossMarketComparison:
    """Tests for cross-market comparison."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with multi-market data."""
        analyzer = FundingRateAnalyzer()

        # BTC - high positive
        analyzer.add_rate(FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0005"),
            timestamp=datetime.now(),
        ))

        # ETH - low positive
        analyzer.add_rate(FundingRate(
            market="ETH-USD-PERP",
            rate=Decimal("0.0002"),
            timestamp=datetime.now(),
        ))

        # SOL - negative
        analyzer.add_rate(FundingRate(
            market="SOL-USD-PERP",
            rate=Decimal("-0.0003"),
            timestamp=datetime.now(),
        ))

        return analyzer

    def test_compare_markets(self, analyzer):
        """Test comparing markets."""
        comparison = analyzer.compare_markets()
        assert comparison is not None
        assert len(comparison.markets) == 3

    def test_best_long_market(self, analyzer):
        """Test finding best market for long."""
        comparison = analyzer.compare_markets()
        assert comparison.best_long_market == "SOL-USD-PERP"  # Lowest rate

    def test_best_short_market(self, analyzer):
        """Test finding best market for short."""
        comparison = analyzer.compare_markets()
        assert comparison.best_short_market == "BTC-USD-PERP"  # Highest rate

    def test_spread(self, analyzer):
        """Test spread calculation."""
        comparison = analyzer.compare_markets()
        # BTC (0.0005) - SOL (-0.0003) = 0.0008
        assert comparison.spread == Decimal("0.0008")

    def test_get_best_funding_market(self, analyzer):
        """Test getting best market by side."""
        best_long = analyzer.get_best_funding_market("long")
        assert best_long == "SOL-USD-PERP"

        best_short = analyzer.get_best_funding_market("short")
        assert best_short == "BTC-USD-PERP"


class TestCallbacks:
    """Tests for callbacks."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_add_callback(self, analyzer):
        """Test adding callback."""
        results = []

        def callback(market, analysis):
            results.append((market, analysis))

        analyzer.add_callback(callback)

        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)
        analyzer.analyze("BTC-USD-PERP")

        assert len(results) == 1

    def test_remove_callback(self, analyzer):
        """Test removing callback."""
        def callback(market, analysis):
            pass

        analyzer.add_callback(callback)
        removed = analyzer.remove_callback(callback)
        assert removed is True

    def test_remove_nonexistent_callback(self, analyzer):
        """Test removing non-existent callback."""
        def callback(market, analysis):
            pass

        removed = analyzer.remove_callback(callback)
        assert removed is False


class TestUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with data."""
        analyzer = FundingRateAnalyzer()
        for market in ["BTC-USD-PERP", "ETH-USD-PERP"]:
            rate = FundingRate(
                market=market,
                rate=Decimal("0.0003"),
                timestamp=datetime.now(),
            )
            analyzer.add_rate(rate)
        return analyzer

    def test_get_markets(self, analyzer):
        """Test getting markets list."""
        markets = analyzer.get_markets()
        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_clear_market(self, analyzer):
        """Test clearing market."""
        analyzer.clear_market("BTC-USD-PERP")
        rate = analyzer.get_current_rate("BTC-USD-PERP")
        assert rate is None

    def test_clear_all(self, analyzer):
        """Test clearing all."""
        analyzer.clear_all()
        markets = analyzer.get_markets()
        assert len(markets) == 0


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_funding_analyzer(self):
        """Test getting global analyzer."""
        reset_funding_analyzer()
        analyzer = get_funding_analyzer()
        assert analyzer is not None

    def test_get_funding_analyzer_singleton(self):
        """Test analyzer is singleton."""
        reset_funding_analyzer()
        a1 = get_funding_analyzer()
        a2 = get_funding_analyzer()
        assert a1 is a2

    def test_reset_funding_analyzer(self):
        """Test resetting analyzer."""
        a1 = get_funding_analyzer()
        reset_funding_analyzer()
        a2 = get_funding_analyzer()
        assert a1 is not a2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FundingRateAnalyzer()

    def test_zero_rate(self, analyzer):
        """Test with zero rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.NEUTRAL

    def test_very_high_rate(self, analyzer):
        """Test with very high rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.01"),  # 1%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.HIGH_POSITIVE

    def test_very_negative_rate(self, analyzer):
        """Test with very negative rate."""
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("-0.01"),  # -1%
            timestamp=datetime.now(),
        )
        analyzer.add_rate(rate)

        regime = analyzer.detect_regime("BTC-USD-PERP")
        assert regime == FundingRegime.HIGH_NEGATIVE

    def test_old_data_trimmed(self, analyzer):
        """Test old data is trimmed."""
        # Add old data
        old_time = datetime.now() - timedelta(hours=200)
        rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0003"),
            timestamp=old_time,
        )
        analyzer.add_rate(rate)

        # Add new data to trigger trim
        new_rate = FundingRate(
            market="BTC-USD-PERP",
            rate=Decimal("0.0004"),
            timestamp=datetime.now(),
        )
        analyzer.add_rate(new_rate)

        history = analyzer.get_history("BTC-USD-PERP")
        # Old data should be trimmed
        assert len(history.rates) == 1
        assert history.rates[0].rate == Decimal("0.0004")
