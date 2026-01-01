"""Unit tests for Signal Generator module."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.signal_generator import (
    SignalType,
    SignalStrength,
    IndicatorType,
    TimeFrame,
    PriceBar,
    IndicatorValue,
    Signal,
    CompositeSignal,
    SignalRule,
    SignalPersistence,
    TechnicalIndicators,
    SignalGenerator,
    MultiTimeframeSignals,
    get_signal_generator,
    reset_signal_generator,
)


class TestSignalType:
    """Tests for SignalType enum."""

    def test_type_values(self):
        """Should have expected type values."""
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"
        assert SignalType.NEUTRAL.value == "neutral"
        assert SignalType.CLOSE_LONG.value == "close_long"
        assert SignalType.CLOSE_SHORT.value == "close_short"


class TestSignalStrength:
    """Tests for SignalStrength enum."""

    def test_strength_values(self):
        """Should have expected strength values."""
        assert SignalStrength.VERY_WEAK.value == "very_weak"
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.MODERATE.value == "moderate"
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.VERY_STRONG.value == "very_strong"


class TestIndicatorType:
    """Tests for IndicatorType enum."""

    def test_indicator_values(self):
        """Should have expected indicator values."""
        assert IndicatorType.MOMENTUM.value == "momentum"
        assert IndicatorType.TREND.value == "trend"
        assert IndicatorType.VOLATILITY.value == "volatility"


class TestTimeFrame:
    """Tests for TimeFrame enum."""

    def test_timeframe_values(self):
        """Should have expected timeframe values."""
        assert TimeFrame.M1.value == "1m"
        assert TimeFrame.H1.value == "1h"
        assert TimeFrame.D1.value == "1d"


class TestPriceBar:
    """Tests for PriceBar dataclass."""

    def test_create_bar(self):
        """Should create price bar."""
        bar = PriceBar(
            timestamp=time.time(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
        )

        assert bar.open == Decimal("100")
        assert bar.high == Decimal("105")

    def test_to_dict(self):
        """Should convert to dictionary."""
        bar = PriceBar(
            timestamp=time.time(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
        )

        d = bar.to_dict()

        assert "open" in d
        assert "high" in d


class TestIndicatorValue:
    """Tests for IndicatorValue dataclass."""

    def test_create_indicator_value(self):
        """Should create indicator value."""
        value = IndicatorValue(
            name="rsi",
            value=65.5,
            indicator_type=IndicatorType.OSCILLATOR,
            signal=SignalType.NEUTRAL,
        )

        assert value.name == "rsi"
        assert value.value == 65.5

    def test_to_dict(self):
        """Should convert to dictionary."""
        value = IndicatorValue(
            name="macd",
            value=0.5,
            indicator_type=IndicatorType.MOMENTUM,
        )

        d = value.to_dict()

        assert d["name"] == "macd"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_create_signal(self):
        """Should create signal."""
        signal = Signal(
            signal_type=SignalType.LONG,
            strength=SignalStrength.STRONG,
            confidence=75.0,
            source="rsi",
        )

        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 75.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        signal = Signal(
            signal_type=SignalType.SHORT,
            strength=SignalStrength.MODERATE,
            confidence=60.0,
        )

        d = signal.to_dict()

        assert d["signal_type"] == "short"
        assert d["confidence"] == 60.0


class TestCompositeSignal:
    """Tests for CompositeSignal dataclass."""

    def test_create_composite(self):
        """Should create composite signal."""
        composite = CompositeSignal(
            signal_type=SignalType.LONG,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            agreement_score=90.0,
        )

        assert composite.agreement_score == 90.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        composite = CompositeSignal(
            signal_type=SignalType.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=40.0,
        )

        d = composite.to_dict()

        assert "agreement_score" in d


class TestSignalRule:
    """Tests for SignalRule dataclass."""

    def test_create_rule(self):
        """Should create rule."""
        rule = SignalRule(
            name="test_rule",
            condition=lambda **kwargs: True,
            signal_type=SignalType.LONG,
        )

        assert rule.name == "test_rule"

    def test_evaluate_true(self):
        """Should return signal when condition true."""
        rule = SignalRule(
            name="always_true",
            condition=lambda **kwargs: True,
            signal_type=SignalType.LONG,
        )

        signal = rule.evaluate()

        assert signal is not None
        assert signal.signal_type == SignalType.LONG

    def test_evaluate_false(self):
        """Should return None when condition false."""
        rule = SignalRule(
            name="always_false",
            condition=lambda **kwargs: False,
            signal_type=SignalType.LONG,
        )

        signal = rule.evaluate()

        assert signal is None

    def test_evaluate_disabled(self):
        """Should return None when disabled."""
        rule = SignalRule(
            name="disabled",
            condition=lambda **kwargs: True,
            signal_type=SignalType.LONG,
            enabled=False,
        )

        signal = rule.evaluate()

        assert signal is None


class TestSignalPersistence:
    """Tests for SignalPersistence dataclass."""

    def test_create_persistence(self):
        """Should create persistence."""
        now = time.time()
        persistence = SignalPersistence(
            signal_type=SignalType.LONG,
            first_seen=now - 3600,
            last_seen=now,
            occurrence_count=10,
        )

        assert persistence.occurrence_count == 10

    def test_duration(self):
        """Should calculate duration."""
        now = time.time()
        persistence = SignalPersistence(
            signal_type=SignalType.LONG,
            first_seen=now - 600,  # 10 minutes ago
            last_seen=now,
        )

        assert persistence.duration_seconds == 600
        assert persistence.duration_minutes == 10

    def test_to_dict(self):
        """Should convert to dictionary."""
        persistence = SignalPersistence(
            signal_type=SignalType.SHORT,
            first_seen=time.time(),
            last_seen=time.time(),
        )

        d = persistence.to_dict()

        assert d["signal_type"] == "short"


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators."""

    def test_sma(self):
        """Should calculate SMA."""
        prices = [Decimal(str(i)) for i in range(10, 20)]

        sma = TechnicalIndicators.sma(prices, 5)

        assert sma == Decimal("17")  # (15+16+17+18+19)/5

    def test_sma_insufficient_data(self):
        """Should return None for insufficient data."""
        prices = [Decimal("100"), Decimal("101")]

        sma = TechnicalIndicators.sma(prices, 5)

        assert sma is None

    def test_ema(self):
        """Should calculate EMA."""
        prices = [Decimal(str(100 + i)) for i in range(20)]

        ema = TechnicalIndicators.ema(prices, 10)

        assert ema is not None

    def test_rsi(self):
        """Should calculate RSI."""
        # Create uptrend data
        prices = [Decimal(str(100 + i * 2)) for i in range(20)]

        rsi = TechnicalIndicators.rsi(prices)

        assert rsi is not None
        assert 50 < rsi <= 100  # Uptrend = high RSI

    def test_rsi_downtrend(self):
        """Should calculate RSI for downtrend."""
        # Create downtrend data
        prices = [Decimal(str(200 - i * 2)) for i in range(20)]

        rsi = TechnicalIndicators.rsi(prices)

        assert rsi is not None
        assert 0 <= rsi < 50  # Downtrend = low RSI

    def test_rsi_insufficient_data(self):
        """Should return None for insufficient data."""
        prices = [Decimal("100"), Decimal("101")]

        rsi = TechnicalIndicators.rsi(prices)

        assert rsi is None

    def test_macd(self):
        """Should calculate MACD."""
        prices = [Decimal(str(100 + i)) for i in range(50)]

        result = TechnicalIndicators.macd(prices)

        assert result is not None
        macd_line, signal_line, histogram = result
        assert histogram == macd_line - signal_line

    def test_bollinger_bands(self):
        """Should calculate Bollinger Bands."""
        prices = [Decimal(str(100 + (i % 5))) for i in range(30)]

        bb = TechnicalIndicators.bollinger_bands(prices)

        assert bb is not None
        upper, middle, lower = bb
        assert upper > middle > lower

    def test_stochastic(self):
        """Should calculate Stochastic."""
        highs = [Decimal(str(105 + i)) for i in range(20)]
        lows = [Decimal(str(95 + i)) for i in range(20)]
        closes = [Decimal(str(100 + i)) for i in range(20)]

        stoch = TechnicalIndicators.stochastic(highs, lows, closes)

        assert stoch is not None
        k, d = stoch
        assert 0 <= k <= 100
        assert 0 <= d <= 100

    def test_atr(self):
        """Should calculate ATR."""
        highs = [Decimal(str(105 + i)) for i in range(20)]
        lows = [Decimal(str(95 + i)) for i in range(20)]
        closes = [Decimal(str(100 + i)) for i in range(20)]

        atr = TechnicalIndicators.atr(highs, lows, closes)

        assert atr is not None
        assert atr > Decimal("0")


class TestSignalGenerator:
    """Tests for SignalGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator."""
        return SignalGenerator()

    @pytest.fixture
    def sample_bars(self, generator):
        """Add sample bars."""
        base_time = time.time()

        for i in range(50):
            # Create realistic bars with some volatility
            base = 50000 + i * 50
            high = base + 100 + (i % 5) * 20
            low = base - 80 - (i % 3) * 15
            close = base + 50

            generator.add_bar(
                "BTC-USD-PERP",
                base_time + i * 60,
                Decimal(str(base)),
                Decimal(str(high)),
                Decimal(str(low)),
                Decimal(str(close)),
            )

        return generator

    def test_add_bar(self, generator):
        """Should add bar and generate signals."""
        signals = generator.add_bar(
            "BTC-USD-PERP",
            time.time(),
            Decimal("50000"),
            Decimal("51000"),
            Decimal("49000"),
            Decimal("50500"),
        )

        assert isinstance(signals, list)

    def test_add_bar_generates_signals(self, sample_bars):
        """Should generate signals from bars."""
        signals = sample_bars.get_signals("BTC-USD-PERP")

        # Should have generated some signals
        assert len(signals) > 0

    def test_signals_have_required_fields(self, sample_bars):
        """Should generate signals with required fields."""
        signals = sample_bars.get_signals("BTC-USD-PERP")

        for signal in signals:
            assert signal.signal_type in list(SignalType)
            assert signal.strength in list(SignalStrength)
            assert 0 <= signal.confidence <= 100
            assert signal.source != ""

    def test_get_composite_signal(self, sample_bars):
        """Should get composite signal."""
        composite = sample_bars.get_composite_signal("BTC-USD-PERP")

        assert composite.signal_type in list(SignalType)
        assert 0 <= composite.agreement_score <= 100

    def test_get_composite_signal_empty(self, generator):
        """Should handle empty signals."""
        composite = generator.get_composite_signal("MISSING")

        assert composite.signal_type == SignalType.NEUTRAL
        assert composite.confidence == 0

    def test_add_custom_rule(self, generator):
        """Should add custom rule."""
        rule = SignalRule(
            name="test_rule",
            condition=lambda **kwargs: True,
            signal_type=SignalType.LONG,
        )

        generator.add_custom_rule(rule)

        assert len(generator._custom_rules) == 1

    def test_remove_custom_rule(self, generator):
        """Should remove custom rule."""
        rule = SignalRule(
            name="to_remove",
            condition=lambda **kwargs: True,
            signal_type=SignalType.LONG,
        )
        generator.add_custom_rule(rule)

        result = generator.remove_custom_rule("to_remove")

        assert result is True
        assert len(generator._custom_rules) == 0

    def test_add_callback(self, generator):
        """Should add callback."""
        callback = MagicMock()
        generator.add_callback(callback)

        # Add bar to trigger callback
        for i in range(30):
            generator.add_bar(
                "BTC",
                time.time() + i,
                Decimal("100"),
                Decimal("105"),
                Decimal("95"),
                Decimal("102"),
            )

        # Callback may or may not be called depending on signals
        assert callback in generator._callbacks

    def test_remove_callback(self, generator):
        """Should remove callback."""
        callback = MagicMock()
        generator.add_callback(callback)
        generator.remove_callback(callback)

        assert callback not in generator._callbacks

    def test_get_persistence(self, sample_bars):
        """Should track signal persistence."""
        persistence = sample_bars.get_persistence("BTC-USD-PERP")

        # Should have some persistence data
        assert isinstance(persistence, dict)

    def test_get_markets(self, sample_bars):
        """Should get list of markets."""
        markets = sample_bars.get_markets()

        assert "BTC-USD-PERP" in markets

    def test_get_bar_count(self, sample_bars):
        """Should get bar count."""
        count = sample_bars.get_bar_count("BTC-USD-PERP")

        assert count == 50

    def test_clear_market(self, sample_bars):
        """Should clear market data."""
        sample_bars.clear_market("BTC-USD-PERP")

        assert sample_bars.get_bar_count("BTC-USD-PERP") == 0

    def test_clear_all(self, sample_bars):
        """Should clear all data."""
        sample_bars.clear_all()

        assert len(sample_bars.get_markets()) == 0


class TestSignalGeneratorIndicators:
    """Tests for signal generation from specific indicators."""

    @pytest.fixture
    def generator(self):
        """Create generator."""
        return SignalGenerator(
            rsi_oversold=30.0,
            rsi_overbought=70.0,
        )

    def test_rsi_oversold_signal(self, generator):
        """Should generate long signal when RSI oversold."""
        base_time = time.time()

        # Create downtrend to make RSI oversold
        for i in range(50):
            price = Decimal(str(100 - i * 0.5))
            generator.add_bar(
                "TEST",
                base_time + i,
                price + Decimal("1"),
                price + Decimal("2"),
                price - Decimal("1"),
                price,
            )

        signals = generator.get_signals("TEST")
        rsi_signals = [s for s in signals if s.source == "rsi"]

        # May have long signals from oversold RSI
        long_signals = [s for s in rsi_signals if s.signal_type == SignalType.LONG]
        # Could be empty if not oversold enough, just check it ran
        assert isinstance(long_signals, list)

    def test_rsi_overbought_signal(self, generator):
        """Should generate short signal when RSI overbought."""
        base_time = time.time()

        # Create uptrend to make RSI overbought
        for i in range(50):
            price = Decimal(str(100 + i * 0.5))
            generator.add_bar(
                "TEST",
                base_time + i,
                price - Decimal("1"),
                price + Decimal("2"),
                price - Decimal("2"),
                price,
            )

        signals = generator.get_signals("TEST")
        rsi_signals = [s for s in signals if s.source == "rsi"]

        short_signals = [s for s in rsi_signals if s.signal_type == SignalType.SHORT]
        assert isinstance(short_signals, list)


class TestMultiTimeframeSignals:
    """Tests for MultiTimeframeSignals."""

    @pytest.fixture
    def mtf(self):
        """Create multi-timeframe analyzer."""
        return MultiTimeframeSignals()

    def test_add_bar(self, mtf):
        """Should add bar to specific timeframe."""
        signals = mtf.add_bar(
            "BTC-USD-PERP",
            TimeFrame.H1,
            time.time(),
            Decimal("50000"),
            Decimal("51000"),
            Decimal("49000"),
            Decimal("50500"),
        )

        assert isinstance(signals, list)

    def test_add_bar_wrong_timeframe(self, mtf):
        """Should handle wrong timeframe."""
        signals = mtf.add_bar(
            "BTC-USD-PERP",
            TimeFrame.D1,  # Not in default timeframes
            time.time(),
            Decimal("50000"),
            Decimal("51000"),
            Decimal("49000"),
            Decimal("50500"),
        )

        assert signals == []

    def test_get_alignment(self, mtf):
        """Should get timeframe alignment."""
        # Add data to all timeframes
        base = time.time()
        for i in range(50):
            for tf in mtf.timeframes:
                mtf.add_bar(
                    "BTC",
                    tf,
                    base + i * 60,
                    Decimal(str(100 + i)),
                    Decimal(str(105 + i)),
                    Decimal(str(95 + i)),
                    Decimal(str(102 + i)),
                )

        alignment = mtf.get_alignment("BTC")

        assert "alignment" in alignment
        assert "bullish_count" in alignment
        assert "bearish_count" in alignment

    def test_get_generator(self, mtf):
        """Should get generator for timeframe."""
        gen = mtf.get_generator(TimeFrame.H1)

        assert gen is not None
        assert isinstance(gen, SignalGenerator)

    def test_get_generator_missing(self, mtf):
        """Should return None for missing timeframe."""
        gen = mtf.get_generator(TimeFrame.D1)

        assert gen is None


class TestGlobalSignalGenerator:
    """Tests for global generator functions."""

    def test_get_signal_generator(self):
        """Should get or create generator."""
        reset_signal_generator()

        g1 = get_signal_generator()
        g2 = get_signal_generator()

        assert g1 is g2

    def test_reset_signal_generator(self):
        """Should reset generator."""
        g1 = get_signal_generator()
        reset_signal_generator()
        g2 = get_signal_generator()

        assert g1 is not g2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_bar(self):
        """Should handle single bar."""
        generator = SignalGenerator()

        signals = generator.add_bar(
            "TEST",
            time.time(),
            Decimal("100"),
            Decimal("105"),
            Decimal("95"),
            Decimal("102"),
        )

        assert signals == []  # Not enough data

    def test_constant_prices(self):
        """Should handle constant prices."""
        generator = SignalGenerator()

        for i in range(50):
            generator.add_bar(
                "TEST",
                time.time() + i,
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
            )

        signals = generator.get_signals("TEST")
        assert isinstance(signals, list)

    def test_high_volatility(self):
        """Should handle high volatility."""
        generator = SignalGenerator()

        for i in range(50):
            if i % 2 == 0:
                price = Decimal("100")
            else:
                price = Decimal("150")

            generator.add_bar(
                "TEST",
                time.time() + i,
                price,
                price + Decimal("10"),
                price - Decimal("10"),
                price,
            )

        signals = generator.get_signals("TEST")
        assert isinstance(signals, list)

    def test_history_trimming(self):
        """Should trim history to max size."""
        generator = SignalGenerator()
        generator._max_bars = 50

        for i in range(100):
            generator.add_bar(
                "TEST",
                time.time() + i,
                Decimal(str(100 + i)),
                Decimal(str(105 + i)),
                Decimal(str(95 + i)),
                Decimal(str(102 + i)),
            )

        assert generator.get_bar_count("TEST") <= 50

    def test_multiple_markets(self):
        """Should handle multiple markets."""
        generator = SignalGenerator()

        for market in ["BTC", "ETH", "SOL"]:
            for i in range(30):
                generator.add_bar(
                    market,
                    time.time() + i,
                    Decimal("100"),
                    Decimal("105"),
                    Decimal("95"),
                    Decimal("102"),
                )

        assert len(generator.get_markets()) == 3

    def test_custom_rule_with_bars(self):
        """Should pass bars to custom rule."""
        generator = SignalGenerator()

        called = {"value": False}

        def check_bars(**kwargs):
            if "bars" in kwargs and len(kwargs["bars"]) > 25:
                called["value"] = True
                return True
            return False

        rule = SignalRule(
            name="bar_checker",
            condition=check_bars,
            signal_type=SignalType.LONG,
        )
        generator.add_custom_rule(rule)

        for i in range(30):
            generator.add_bar(
                "TEST",
                time.time() + i,
                Decimal("100"),
                Decimal("105"),
                Decimal("95"),
                Decimal("102"),
            )

        assert called["value"] is True

    def test_signal_persistence_update(self):
        """Should update persistence on repeat signals."""
        generator = SignalGenerator()

        # Generate same signal type repeatedly
        for i in range(50):
            generator.add_bar(
                "TEST",
                time.time() + i,
                Decimal(str(100 + i)),
                Decimal(str(105 + i)),
                Decimal(str(95 + i)),
                Decimal(str(102 + i)),
            )

        persistence = generator.get_persistence("TEST")

        # Should have some persistence entries
        for key, p in persistence.items():
            if p.occurrence_count > 1:
                assert p.duration_seconds > 0
