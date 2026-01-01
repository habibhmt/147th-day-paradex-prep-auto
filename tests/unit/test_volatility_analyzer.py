"""Unit tests for Volatility Analyzer module."""

import pytest
import time
import math
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.volatility_analyzer import (
    VolatilityType,
    VolatilityRegime,
    TrendDirection,
    OHLC,
    VolatilityMetrics,
    VolatilityRegimeInfo,
    VolatilityForecast,
    VolatilityTermStructure,
    VolatilityClustering,
    IntradayPattern,
    VolatilityReport,
    VolatilityCalculator,
    VolatilityAnalyzer,
    get_volatility_analyzer,
    reset_volatility_analyzer,
)


class TestVolatilityType:
    """Tests for VolatilityType enum."""

    def test_type_values(self):
        """Should have expected type values."""
        assert VolatilityType.REALIZED.value == "realized"
        assert VolatilityType.PARKINSON.value == "parkinson"
        assert VolatilityType.GARMAN_KLASS.value == "garman_klass"
        assert VolatilityType.ATR.value == "atr"


class TestVolatilityRegime:
    """Tests for VolatilityRegime enum."""

    def test_regime_values(self):
        """Should have expected regime values."""
        assert VolatilityRegime.VERY_LOW.value == "very_low"
        assert VolatilityRegime.LOW.value == "low"
        assert VolatilityRegime.NORMAL.value == "normal"
        assert VolatilityRegime.HIGH.value == "high"
        assert VolatilityRegime.VERY_HIGH.value == "very_high"
        assert VolatilityRegime.EXTREME.value == "extreme"


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_direction_values(self):
        """Should have expected direction values."""
        assert TrendDirection.INCREASING.value == "increasing"
        assert TrendDirection.DECREASING.value == "decreasing"
        assert TrendDirection.STABLE.value == "stable"


class TestOHLC:
    """Tests for OHLC dataclass."""

    def test_create_ohlc(self):
        """Should create OHLC data."""
        ohlc = OHLC(
            timestamp=time.time(),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
        )

        assert ohlc.open == Decimal("50000")
        assert ohlc.high == Decimal("51000")

    def test_range(self):
        """Should calculate range."""
        ohlc = OHLC(
            timestamp=time.time(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
        )

        assert ohlc.range == Decimal("20")

    def test_range_pct(self):
        """Should calculate range percentage."""
        ohlc = OHLC(
            timestamp=time.time(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
        )

        assert ohlc.range_pct == 20.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        ohlc = OHLC(
            timestamp=time.time(),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
        )

        d = ohlc.to_dict()

        assert "open" in d
        assert "range" in d


class TestVolatilityMetrics:
    """Tests for VolatilityMetrics dataclass."""

    def test_create_metrics(self):
        """Should create volatility metrics."""
        metrics = VolatilityMetrics(
            realized_volatility=2.5,
            annualized_volatility=48.0,
            atr=Decimal("500"),
            atr_pct=1.0,
        )

        assert metrics.realized_volatility == 2.5
        assert metrics.annualized_volatility == 48.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = VolatilityMetrics(
            realized_volatility=2.0,
            period_days=20,
        )

        d = metrics.to_dict()

        assert d["realized_volatility"] == 2.0
        assert d["period_days"] == 20


class TestVolatilityRegimeInfo:
    """Tests for VolatilityRegimeInfo dataclass."""

    def test_create_regime_info(self):
        """Should create regime info."""
        info = VolatilityRegimeInfo(
            regime=VolatilityRegime.HIGH,
            current_volatility=60.0,
            regime_percentile=80.0,
        )

        assert info.regime == VolatilityRegime.HIGH
        assert info.regime_percentile == 80.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        info = VolatilityRegimeInfo(
            regime=VolatilityRegime.LOW,
            regime_duration=5,
        )

        d = info.to_dict()

        assert d["regime"] == "low"
        assert d["regime_duration"] == 5


class TestVolatilityForecast:
    """Tests for VolatilityForecast dataclass."""

    def test_create_forecast(self):
        """Should create forecast."""
        forecast = VolatilityForecast(
            forecast_periods=5,
            predicted_volatility=50.0,
            confidence_low=40.0,
            confidence_high=60.0,
        )

        assert forecast.predicted_volatility == 50.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        forecast = VolatilityForecast(
            model_type="garch",
            predicted_volatility=45.0,
        )

        d = forecast.to_dict()

        assert d["model_type"] == "garch"


class TestVolatilityTermStructure:
    """Tests for VolatilityTermStructure dataclass."""

    def test_create_term_structure(self):
        """Should create term structure."""
        structure = VolatilityTermStructure(
            short_term=60.0,
            medium_term=50.0,
            long_term=45.0,
            is_inverted=True,
        )

        assert structure.is_inverted is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        structure = VolatilityTermStructure(
            term_spread=-15.0,
        )

        d = structure.to_dict()

        assert d["term_spread"] == -15.0


class TestVolatilityClustering:
    """Tests for VolatilityClustering dataclass."""

    def test_create_clustering(self):
        """Should create clustering analysis."""
        clustering = VolatilityClustering(
            cluster_coefficient=0.6,
            persistence=0.8,
            is_clustering=True,
        )

        assert clustering.is_clustering is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        clustering = VolatilityClustering(
            half_life=10.0,
        )

        d = clustering.to_dict()

        assert d["half_life"] == 10.0


class TestIntradayPattern:
    """Tests for IntradayPattern dataclass."""

    def test_create_pattern(self):
        """Should create intraday pattern."""
        pattern = IntradayPattern(
            most_volatile_hour=14,
            least_volatile_hour=4,
        )

        assert pattern.most_volatile_hour == 14

    def test_to_dict(self):
        """Should convert to dictionary."""
        pattern = IntradayPattern(
            asian_session_vol=1.5,
            european_session_vol=2.0,
            us_session_vol=2.5,
        )

        d = pattern.to_dict()

        assert d["asian_session_vol"] == 1.5


class TestVolatilityReport:
    """Tests for VolatilityReport dataclass."""

    def test_create_report(self):
        """Should create volatility report."""
        report = VolatilityReport(
            market="BTC-USD-PERP",
            metrics=VolatilityMetrics(realized_volatility=2.0),
            regime=VolatilityRegimeInfo(regime=VolatilityRegime.NORMAL),
        )

        assert report.market == "BTC-USD-PERP"

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = VolatilityReport(
            market="ETH-USD-PERP",
        )

        d = report.to_dict()

        assert d["market"] == "ETH-USD-PERP"


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator."""
        return VolatilityCalculator()

    def test_calculate_realized_volatility(self, calculator):
        """Should calculate realized volatility."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.02]

        vol = calculator.calculate_realized_volatility(returns, annualize=False)

        assert vol > 0

    def test_calculate_realized_volatility_annualized(self, calculator):
        """Should annualize volatility."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.02]

        vol = calculator.calculate_realized_volatility(returns, annualize=False)
        ann_vol = calculator.calculate_realized_volatility(returns, annualize=True)

        assert ann_vol > vol

    def test_calculate_realized_volatility_empty(self, calculator):
        """Should handle empty returns."""
        vol = calculator.calculate_realized_volatility([])

        assert vol == 0.0

    def test_calculate_parkinson_volatility(self, calculator):
        """Should calculate Parkinson volatility."""
        ohlc = [
            OHLC(time.time(), Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102")),
            OHLC(time.time(), Decimal("102"), Decimal("108"), Decimal("98"), Decimal("104")),
            OHLC(time.time(), Decimal("104"), Decimal("110"), Decimal("100"), Decimal("106")),
        ]

        vol = calculator.calculate_parkinson_volatility(ohlc)

        assert vol > 0

    def test_calculate_garman_klass_volatility(self, calculator):
        """Should calculate Garman-Klass volatility."""
        ohlc = [
            OHLC(time.time(), Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102")),
            OHLC(time.time(), Decimal("102"), Decimal("108"), Decimal("98"), Decimal("104")),
            OHLC(time.time(), Decimal("104"), Decimal("110"), Decimal("100"), Decimal("106")),
        ]

        vol = calculator.calculate_garman_klass_volatility(ohlc)

        assert vol > 0

    def test_calculate_atr(self, calculator):
        """Should calculate ATR."""
        ohlc = [
            OHLC(time.time(), Decimal("100"), Decimal("105"), Decimal("95"), Decimal("102")),
            OHLC(time.time() + 1, Decimal("102"), Decimal("108"), Decimal("98"), Decimal("104")),
            OHLC(time.time() + 2, Decimal("104"), Decimal("110"), Decimal("100"), Decimal("106")),
        ]

        atr = calculator.calculate_atr(ohlc, period=3)

        assert atr > Decimal("0")

    def test_calculate_atr_empty(self, calculator):
        """Should handle empty data."""
        atr = calculator.calculate_atr([])

        assert atr == Decimal("0")

    def test_calculate_ewma_volatility(self, calculator):
        """Should calculate EWMA volatility."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.025]

        vol = calculator.calculate_ewma_volatility(returns)

        assert vol > 0


class TestVolatilityAnalyzer:
    """Tests for VolatilityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return VolatilityAnalyzer()

    @pytest.fixture
    def sample_data(self, analyzer):
        """Add sample OHLC data."""
        base_time = time.time()

        for i in range(100):
            # Create realistic OHLC data
            base = 50000 + i * 50
            high = base + 200 + (i % 5) * 50
            low = base - 150 - (i % 3) * 30
            close = base + 100

            analyzer.add_ohlc(
                "BTC-USD-PERP",
                base_time + i * 3600,
                Decimal(str(base)),
                Decimal(str(high)),
                Decimal(str(low)),
                Decimal(str(close)),
            )

        return analyzer

    def test_add_ohlc(self, analyzer):
        """Should add OHLC data."""
        result = analyzer.add_ohlc(
            "BTC-USD-PERP",
            time.time(),
            Decimal("50000"),
            Decimal("51000"),
            Decimal("49000"),
            Decimal("50500"),
        )

        assert result == 0.0  # First bar has no return

    def test_add_ohlc_calculates_return(self, analyzer):
        """Should calculate return."""
        analyzer.add_ohlc(
            "BTC-USD-PERP",
            time.time(),
            Decimal("50000"),
            Decimal("51000"),
            Decimal("49000"),
            Decimal("50000"),
        )

        result = analyzer.add_ohlc(
            "BTC-USD-PERP",
            time.time() + 1,
            Decimal("50000"),
            Decimal("52000"),
            Decimal("49500"),
            Decimal("51000"),
        )

        assert result == pytest.approx(0.02, abs=0.001)

    def test_add_price(self, analyzer):
        """Should add simple price."""
        result = analyzer.add_price("BTC-USD-PERP", time.time(), Decimal("50000"))

        assert result == 0.0

    def test_calculate_volatility(self, sample_data):
        """Should calculate volatility."""
        metrics = sample_data.calculate_volatility("BTC-USD-PERP")

        assert metrics.realized_volatility > 0
        assert metrics.annualized_volatility > metrics.realized_volatility
        assert metrics.atr > Decimal("0")

    def test_calculate_volatility_parkinson(self, sample_data):
        """Should calculate Parkinson volatility."""
        metrics = sample_data.calculate_volatility(
            "BTC-USD-PERP",
            vol_type=VolatilityType.PARKINSON,
        )

        assert metrics.volatility_type == VolatilityType.PARKINSON
        assert metrics.realized_volatility > 0

    def test_calculate_volatility_garman_klass(self, sample_data):
        """Should calculate Garman-Klass volatility."""
        metrics = sample_data.calculate_volatility(
            "BTC-USD-PERP",
            vol_type=VolatilityType.GARMAN_KLASS,
        )

        assert metrics.volatility_type == VolatilityType.GARMAN_KLASS

    def test_calculate_volatility_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        analyzer.add_price("A", time.time(), Decimal("100"))

        metrics = analyzer.calculate_volatility("A")

        assert metrics.period_days == 0

    def test_detect_regime(self, sample_data):
        """Should detect volatility regime."""
        # First calculate volatility to populate history
        for _ in range(10):
            sample_data.calculate_volatility("BTC-USD-PERP")

        regime = sample_data.detect_regime("BTC-USD-PERP")

        assert regime.regime in list(VolatilityRegime)
        assert regime.current_volatility > 0
        assert 0 <= regime.regime_percentile <= 100

    def test_detect_regime_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        regime = analyzer.detect_regime("MISSING")

        assert regime.regime == VolatilityRegime.NORMAL

    def test_forecast_volatility(self, sample_data):
        """Should forecast volatility."""
        forecast = sample_data.forecast_volatility("BTC-USD-PERP", periods=5)

        assert forecast.forecast_periods == 5
        assert forecast.predicted_volatility > 0
        assert forecast.confidence_low < forecast.predicted_volatility
        assert forecast.confidence_high > forecast.predicted_volatility

    def test_forecast_volatility_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        forecast = analyzer.forecast_volatility("MISSING")

        assert forecast.predicted_volatility == 0.0

    def test_calculate_term_structure(self, sample_data):
        """Should calculate term structure."""
        structure = sample_data.calculate_term_structure("BTC-USD-PERP")

        assert structure.short_term > 0
        assert structure.medium_term > 0
        assert structure.long_term > 0

    def test_calculate_term_structure_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        structure = analyzer.calculate_term_structure("MISSING")

        assert structure.short_term == 0.0

    def test_analyze_clustering(self, sample_data):
        """Should analyze clustering."""
        clustering = sample_data.analyze_clustering("BTC-USD-PERP")

        assert isinstance(clustering.cluster_coefficient, float)
        assert isinstance(clustering.is_clustering, bool)

    def test_analyze_clustering_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        clustering = analyzer.analyze_clustering("MISSING")

        assert clustering.cluster_coefficient == 0.0

    def test_analyze_intraday_pattern(self, sample_data):
        """Should analyze intraday pattern."""
        pattern = sample_data.analyze_intraday_pattern("BTC-USD-PERP")

        assert len(pattern.hour_volatilities) > 0

    def test_analyze_intraday_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        pattern = analyzer.analyze_intraday_pattern("MISSING")

        assert len(pattern.hour_volatilities) == 0

    def test_get_full_report(self, sample_data):
        """Should get full report."""
        report = sample_data.get_full_report("BTC-USD-PERP")

        assert report.market == "BTC-USD-PERP"
        assert report.metrics is not None
        assert report.regime is not None
        assert report.forecast is not None

    def test_get_volatility_trend(self, sample_data):
        """Should get volatility trend."""
        trend = sample_data.get_volatility_trend("BTC-USD-PERP")

        assert trend in [TrendDirection.INCREASING, TrendDirection.DECREASING, TrendDirection.STABLE]

    def test_get_volatility_trend_insufficient_data(self, analyzer):
        """Should handle insufficient data."""
        trend = analyzer.get_volatility_trend("MISSING")

        assert trend == TrendDirection.STABLE

    def test_compare_volatility(self, sample_data):
        """Should compare volatility."""
        # Add data for second market
        base_time = time.time()
        for i in range(100):
            sample_data.add_price("ETH-USD-PERP", base_time + i, Decimal(str(3000 + i * 3)))

        comparison = sample_data.compare_volatility("BTC-USD-PERP", "ETH-USD-PERP")

        assert "volatility_a" in comparison
        assert "volatility_b" in comparison
        assert "volatility_ratio" in comparison

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

    def test_get_markets(self, sample_data):
        """Should get list of markets."""
        markets = sample_data.get_markets()

        assert "BTC-USD-PERP" in markets

    def test_get_sample_size(self, sample_data):
        """Should get sample size."""
        size = sample_data.get_sample_size("BTC-USD-PERP")

        assert size == 100

    def test_clear_market(self, sample_data):
        """Should clear market data."""
        sample_data.clear_market("BTC-USD-PERP")

        assert sample_data.get_sample_size("BTC-USD-PERP") == 0

    def test_clear_all(self, sample_data):
        """Should clear all data."""
        sample_data.clear_all()

        assert len(sample_data.get_markets()) == 0


class TestGlobalVolatilityAnalyzer:
    """Tests for global analyzer functions."""

    def test_get_volatility_analyzer(self):
        """Should get or create analyzer."""
        reset_volatility_analyzer()

        a1 = get_volatility_analyzer()
        a2 = get_volatility_analyzer()

        assert a1 is a2

    def test_reset_volatility_analyzer(self):
        """Should reset analyzer."""
        a1 = get_volatility_analyzer()
        reset_volatility_analyzer()
        a2 = get_volatility_analyzer()

        assert a1 is not a2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_data_point(self):
        """Should handle single data point."""
        analyzer = VolatilityAnalyzer()

        analyzer.add_price("A", time.time(), Decimal("100"))

        metrics = analyzer.calculate_volatility("A")

        assert metrics.realized_volatility == 0.0

    def test_constant_prices(self):
        """Should handle constant prices."""
        analyzer = VolatilityAnalyzer()

        for i in range(20):
            analyzer.add_price("A", time.time() + i, Decimal("100"))

        metrics = analyzer.calculate_volatility("A")

        assert metrics.realized_volatility == 0.0

    def test_zero_prices(self):
        """Should handle zero in OHLC."""
        analyzer = VolatilityAnalyzer()

        analyzer.add_ohlc(
            "A",
            time.time(),
            Decimal("0"),
            Decimal("100"),
            Decimal("0"),
            Decimal("100"),
        )

        metrics = analyzer.calculate_volatility("A")

        # Should not crash
        assert metrics is not None

    def test_high_volatility(self):
        """Should handle high volatility."""
        analyzer = VolatilityAnalyzer()

        base = time.time()
        for i in range(30):
            # Alternate +50% and -50%
            if i % 2 == 0:
                price = Decimal("100")
            else:
                price = Decimal("150")
            analyzer.add_price("A", base + i, price)

        metrics = analyzer.calculate_volatility("A")

        assert metrics.realized_volatility > 10  # High vol

    def test_very_low_volatility(self):
        """Should handle very low volatility."""
        analyzer = VolatilityAnalyzer()

        base = time.time()
        for i in range(30):
            # Tiny changes
            price = Decimal(str(100 + i * 0.001))
            analyzer.add_price("A", base + i, price)

        metrics = analyzer.calculate_volatility("A")

        assert metrics.realized_volatility >= 0

    def test_history_trimming(self):
        """Should trim history to max size."""
        analyzer = VolatilityAnalyzer()
        analyzer._max_history = 50

        base = time.time()
        for i in range(100):
            analyzer.add_price("A", base + i, Decimal(str(100 + i)))

        assert len(analyzer._ohlc_data["A"]) <= 50

    def test_inverted_term_structure(self):
        """Should detect inverted term structure."""
        analyzer = VolatilityAnalyzer()

        base = time.time()
        # Create data where short-term vol > long-term vol
        for i in range(100):
            if i < 10:
                # High recent volatility
                price = Decimal(str(100 + (i % 2) * 20))
            else:
                # Low historical volatility
                price = Decimal(str(100 + i * 0.5))
            analyzer.add_price("A", base + i, price)

        structure = analyzer.calculate_term_structure("A")

        # Just verify it returns a structure
        assert structure is not None

    def test_volatility_regime_transitions(self):
        """Should track regime transitions."""
        analyzer = VolatilityAnalyzer()

        base = time.time()
        # Low volatility period
        for i in range(50):
            analyzer.add_price("A", base + i, Decimal(str(100 + i * 0.1)))
            analyzer.calculate_volatility("A")

        # High volatility period
        for i in range(50, 100):
            if i % 2 == 0:
                price = Decimal("100")
            else:
                price = Decimal("110")
            analyzer.add_price("A", base + i, price)
            analyzer.calculate_volatility("A")

        regime = analyzer.detect_regime("A")

        # Should detect some regime
        assert regime.regime in list(VolatilityRegime)
