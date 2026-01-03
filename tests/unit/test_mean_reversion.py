"""
Tests for Mean Reversion Trading Strategy Module.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.strategies.mean_reversion import (
    MeanReversionType,
    DeviationState,
    EntrySignal,
    ExitSignal,
    MeanReversionSignal,
    MeanReversionConfig,
    MeanReversionMetrics,
    StatisticalCalculator,
    MeanReversionAnalyzer,
    MeanReversionStrategy,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestMeanReversionType:
    """Tests for MeanReversionType enum."""

    def test_all_types_defined(self):
        """Test all types are defined."""
        assert MeanReversionType.BOLLINGER.value == "bollinger"
        assert MeanReversionType.RSI.value == "rsi"
        assert MeanReversionType.ZSCORE.value == "zscore"
        assert MeanReversionType.KELTNER.value == "keltner"
        assert MeanReversionType.PERCENTAGE.value == "percentage"


class TestDeviationState:
    """Tests for DeviationState enum."""

    def test_all_states_defined(self):
        """Test all states are defined."""
        assert DeviationState.EXTREMELY_OVERBOUGHT.value == "extremely_overbought"
        assert DeviationState.OVERBOUGHT.value == "overbought"
        assert DeviationState.ABOVE_MEAN.value == "above_mean"
        assert DeviationState.AT_MEAN.value == "at_mean"
        assert DeviationState.BELOW_MEAN.value == "below_mean"
        assert DeviationState.OVERSOLD.value == "oversold"
        assert DeviationState.EXTREMELY_OVERSOLD.value == "extremely_oversold"


class TestEntrySignal:
    """Tests for EntrySignal enum."""

    def test_all_signals_defined(self):
        """Test all entry signals are defined."""
        assert EntrySignal.LONG_ENTRY.value == "long_entry"
        assert EntrySignal.SHORT_ENTRY.value == "short_entry"
        assert EntrySignal.NO_SIGNAL.value == "no_signal"


class TestExitSignal:
    """Tests for ExitSignal enum."""

    def test_all_signals_defined(self):
        """Test all exit signals are defined."""
        assert ExitSignal.EXIT_LONG.value == "exit_long"
        assert ExitSignal.EXIT_SHORT.value == "exit_short"
        assert ExitSignal.HOLD.value == "hold"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestMeanReversionSignal:
    """Tests for MeanReversionSignal dataclass."""

    def test_creation(self):
        """Test signal creation."""
        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol="BTC-USD-PERP",
            entry_signal=EntrySignal.LONG_ENTRY,
            exit_signal=ExitSignal.HOLD,
            deviation_state=DeviationState.OVERSOLD,
            current_price=Decimal("48000"),
            mean_price=Decimal("50000"),
            deviation_pct=Decimal("-4")
        )
        assert signal.entry_signal == EntrySignal.LONG_ENTRY
        assert signal.deviation_pct == Decimal("-4")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = MeanReversionSignal(
            timestamp=datetime.now(),
            symbol="BTC-USD-PERP",
            entry_signal=EntrySignal.SHORT_ENTRY,
            exit_signal=ExitSignal.HOLD,
            deviation_state=DeviationState.OVERBOUGHT,
            current_price=Decimal("52000"),
            mean_price=Decimal("50000"),
            deviation_pct=Decimal("4"),
            z_score=Decimal("2.5")
        )
        result = signal.to_dict()
        assert result["entry_signal"] == "short_entry"
        assert result["deviation_state"] == "overbought"


class TestMeanReversionConfig:
    """Tests for MeanReversionConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.BOLLINGER
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_ma_period(self):
        """Test invalid MA period."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.BOLLINGER,
            ma_period=1
        )
        errors = config.validate()
        assert any("MA period" in e for e in errors)

    def test_invalid_bb_std(self):
        """Test invalid Bollinger std dev."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.BOLLINGER,
            bb_std_dev=Decimal("-1")
        )
        errors = config.validate()
        assert any("std dev" in e for e in errors)

    def test_invalid_rsi_thresholds(self):
        """Test invalid RSI thresholds."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.RSI,
            rsi_overbought=Decimal("30"),
            rsi_oversold=Decimal("70")
        )
        errors = config.validate()
        assert any("overbought" in e for e in errors)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.ZSCORE
        )
        result = config.to_dict()
        assert result["symbol"] == "BTC-USD-PERP"
        assert result["strategy_type"] == "zscore"


class TestMeanReversionMetrics:
    """Tests for MeanReversionMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = MeanReversionMetrics()
        assert metrics.total_signals == 0
        assert metrics.total_pnl == Decimal("0")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MeanReversionMetrics(
            total_signals=50,
            successful_reversions=35,
            win_rate=Decimal("70")
        )
        result = metrics.to_dict()
        assert result["total_signals"] == 50
        assert result["win_rate"] == "70"


# =============================================================================
# Statistical Calculator Tests
# =============================================================================

class TestStatisticalCalculator:
    """Tests for StatisticalCalculator."""

    def test_calculate_mean(self):
        """Test mean calculation."""
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        mean = StatisticalCalculator.calculate_mean(values)
        assert mean == Decimal("20")

    def test_calculate_mean_empty(self):
        """Test mean with empty list."""
        mean = StatisticalCalculator.calculate_mean([])
        assert mean == Decimal("0")

    def test_calculate_std_dev(self):
        """Test standard deviation calculation."""
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        std = StatisticalCalculator.calculate_std_dev(values)
        assert std > Decimal("0")

    def test_calculate_std_dev_single_value(self):
        """Test std dev with single value."""
        std = StatisticalCalculator.calculate_std_dev([Decimal("100")])
        assert std == Decimal("0")

    def test_calculate_z_score(self):
        """Test z-score calculation."""
        z = StatisticalCalculator.calculate_z_score(
            Decimal("110"),
            Decimal("100"),
            Decimal("10")
        )
        assert z == Decimal("1")

    def test_calculate_z_score_zero_std(self):
        """Test z-score with zero std dev."""
        z = StatisticalCalculator.calculate_z_score(
            Decimal("110"),
            Decimal("100"),
            Decimal("0")
        )
        assert z == Decimal("0")

    def test_calculate_ema(self):
        """Test EMA calculation."""
        values = [Decimal(str(50000 + i * 10)) for i in range(20)]
        ema = StatisticalCalculator.calculate_ema(values, 10)
        assert ema is not None
        assert ema > Decimal("50000")

    def test_calculate_ema_insufficient_data(self):
        """Test EMA with insufficient data."""
        values = [Decimal("100")]
        ema = StatisticalCalculator.calculate_ema(values, 10)
        assert ema is None

    def test_calculate_sma(self):
        """Test SMA calculation."""
        values = [Decimal(str(i)) for i in range(1, 11)]
        sma = StatisticalCalculator.calculate_sma(values, 5)
        assert sma == Decimal("8")

    def test_calculate_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        values = [Decimal("100")]
        sma = StatisticalCalculator.calculate_sma(values, 10)
        assert sma is None

    def test_calculate_rsi_bullish(self):
        """Test RSI in bullish market."""
        prices = [Decimal(str(50000 + i * 100)) for i in range(20)]
        rsi = StatisticalCalculator.calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi > Decimal("50")

    def test_calculate_rsi_bearish(self):
        """Test RSI in bearish market."""
        prices = [Decimal(str(60000 - i * 100)) for i in range(20)]
        rsi = StatisticalCalculator.calculate_rsi(prices, 14)
        assert rsi is not None
        assert rsi < Decimal("50")

    def test_calculate_atr(self):
        """Test ATR calculation."""
        highs = [Decimal(str(51000 + i * 10)) for i in range(20)]
        lows = [Decimal(str(49000 + i * 10)) for i in range(20)]
        closes = [Decimal(str(50000 + i * 10)) for i in range(20)]
        atr = StatisticalCalculator.calculate_atr(highs, lows, closes)
        assert atr is not None
        assert atr > Decimal("0")

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = [Decimal(str(50000 + (i % 10 - 5) * 50)) for i in range(30)]
        upper, middle, lower = StatisticalCalculator.calculate_bollinger_bands(prices)
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert upper > middle > lower

    def test_calculate_keltner_channels(self):
        """Test Keltner Channels calculation."""
        closes = [Decimal(str(50000 + i * 10)) for i in range(30)]
        highs = [Decimal(str(50100 + i * 10)) for i in range(30)]
        lows = [Decimal(str(49900 + i * 10)) for i in range(30)]
        upper, middle, lower = StatisticalCalculator.calculate_keltner_channels(
            closes, highs, lows
        )
        assert upper is not None
        assert middle is not None
        assert lower is not None


# =============================================================================
# Mean Reversion Analyzer Tests
# =============================================================================

class TestMeanReversionAnalyzerBollinger:
    """Tests for Bollinger Bands analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create Bollinger analyzer."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.BOLLINGER,
            ma_period=20,
            bb_std_dev=Decimal("2")
        )
        return MeanReversionAnalyzer(config)

    def test_add_price(self, analyzer):
        """Test adding price."""
        analyzer.add_price(Decimal("50000"))
        assert len(analyzer._price_history) == 1

    def test_analyze_insufficient_data(self, analyzer):
        """Test analysis with insufficient data."""
        analyzer.add_price(Decimal("50000"))
        signal = analyzer.analyze()
        assert signal is None

    def test_analyze_overbought(self, analyzer):
        """Test overbought detection."""
        # Add stable prices then spike
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # Add overbought price
        analyzer.add_price(Decimal("52000"))

        signal = analyzer.analyze()
        assert signal is not None
        assert signal.deviation_state in [DeviationState.OVERBOUGHT, DeviationState.EXTREMELY_OVERBOUGHT]

    def test_analyze_oversold(self, analyzer):
        """Test oversold detection."""
        # Add stable prices then drop
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # Add oversold price
        analyzer.add_price(Decimal("48000"))

        signal = analyzer.analyze()
        assert signal is not None
        assert signal.deviation_state in [DeviationState.OVERSOLD, DeviationState.EXTREMELY_OVERSOLD]


class TestMeanReversionAnalyzerRSI:
    """Tests for RSI analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create RSI analyzer."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.RSI,
            rsi_period=14
        )
        return MeanReversionAnalyzer(config)

    def test_analyze_overbought_rsi(self, analyzer):
        """Test RSI overbought detection."""
        # Rising prices
        for i in range(20):
            analyzer.add_price(Decimal(str(50000 + i * 200)))

        signal = analyzer.analyze()
        assert signal is not None

    def test_analyze_oversold_rsi(self, analyzer):
        """Test RSI oversold detection."""
        # Falling prices
        for i in range(20):
            analyzer.add_price(Decimal(str(60000 - i * 200)))

        signal = analyzer.analyze()
        assert signal is not None


class TestMeanReversionAnalyzerZScore:
    """Tests for Z-score analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create Z-score analyzer."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.ZSCORE,
            zscore_lookback=20,
            zscore_entry_threshold=Decimal("2")
        )
        return MeanReversionAnalyzer(config)

    def test_analyze_high_zscore(self, analyzer):
        """Test high z-score detection."""
        # Add stable prices
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # Add high deviation price
        analyzer.add_price(Decimal("52500"))

        signal = analyzer.analyze()
        assert signal is not None
        assert signal.z_score is not None

    def test_analyze_low_zscore(self, analyzer):
        """Test low z-score detection."""
        # Add stable prices
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # Add low deviation price
        analyzer.add_price(Decimal("47500"))

        signal = analyzer.analyze()
        assert signal is not None


class TestMeanReversionAnalyzerKeltner:
    """Tests for Keltner analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create Keltner analyzer."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.KELTNER
        )
        return MeanReversionAnalyzer(config)

    def test_analyze_keltner(self, analyzer):
        """Test Keltner channel analysis."""
        for i in range(30):
            price = Decimal("50000") + (i % 5 - 2) * 50
            analyzer.add_price(price, price + 100, price - 100)

        signal = analyzer.analyze()
        assert signal is not None


class TestMeanReversionAnalyzerPercentage:
    """Tests for percentage deviation analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create percentage analyzer."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.PERCENTAGE,
            entry_deviation_pct=Decimal("3")
        )
        return MeanReversionAnalyzer(config)

    def test_analyze_above_threshold(self, analyzer):
        """Test deviation above threshold."""
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # 4% deviation
        analyzer.add_price(Decimal("52000"))

        signal = analyzer.analyze()
        assert signal is not None
        assert signal.entry_signal == EntrySignal.SHORT_ENTRY

    def test_analyze_below_threshold(self, analyzer):
        """Test deviation below threshold."""
        for i in range(25):
            analyzer.add_price(Decimal("50000"))

        # -4% deviation
        analyzer.add_price(Decimal("48000"))

        signal = analyzer.analyze()
        assert signal is not None
        assert signal.entry_signal == EntrySignal.LONG_ENTRY


# =============================================================================
# Mean Reversion Strategy Tests
# =============================================================================

class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""

    @pytest.fixture
    def config(self):
        """Create config."""
        return MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.BOLLINGER,
            stop_loss_pct=Decimal("3"),
            take_profit_pct=Decimal("2")
        )

    @pytest.fixture
    def strategy(self, config):
        """Create strategy instance."""
        return MeanReversionStrategy(config)

    def test_strategy_creation(self, strategy):
        """Test strategy creation."""
        assert strategy.config.symbol == "BTC-USD-PERP"
        assert not strategy._in_position

    def test_on_price(self, strategy):
        """Test processing price."""
        for i in range(25):
            strategy.on_price(Decimal("50000"))

        signal = strategy.on_price(Decimal("52000"))
        assert signal is not None

    def test_enter_long_position(self, strategy):
        """Test entering long position."""
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        assert strategy._in_position
        assert strategy._position_side == "long"
        assert strategy.metrics.long_entries == 1

    def test_enter_short_position(self, strategy):
        """Test entering short position."""
        strategy.enter_position("short", Decimal("52000"), Decimal("1"), Decimal("4"))
        assert strategy._in_position
        assert strategy._position_side == "short"
        assert strategy.metrics.short_entries == 1

    def test_exit_long_profit(self, strategy):
        """Test exiting long with profit."""
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        pnl = strategy.exit_position(Decimal("50000"), reverted_to_mean=True)
        assert pnl == Decimal("2000")
        assert strategy.metrics.successful_reversions == 1

    def test_exit_long_loss(self, strategy):
        """Test exiting long with loss."""
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        pnl = strategy.exit_position(Decimal("46000"), reverted_to_mean=False)
        assert pnl == Decimal("-2000")
        assert strategy.metrics.failed_reversions == 1

    def test_exit_short_profit(self, strategy):
        """Test exiting short with profit."""
        strategy.enter_position("short", Decimal("52000"), Decimal("1"), Decimal("4"))
        pnl = strategy.exit_position(Decimal("50000"), reverted_to_mean=True)
        assert pnl == Decimal("2000")
        assert strategy.metrics.successful_reversions == 1

    def test_exit_short_loss(self, strategy):
        """Test exiting short with loss."""
        strategy.enter_position("short", Decimal("52000"), Decimal("1"), Decimal("4"))
        pnl = strategy.exit_position(Decimal("54000"), reverted_to_mean=False)
        assert pnl == Decimal("-2000")
        assert strategy.metrics.failed_reversions == 1

    def test_should_enter(self, strategy):
        """Test should_enter logic."""
        for i in range(25):
            strategy.on_price(Decimal("50000"))

        signal = strategy.on_price(Decimal("52000"))
        if signal and signal.confidence >= Decimal("60"):
            assert strategy.should_enter(signal)

    def test_should_exit(self, strategy):
        """Test should_exit logic."""
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))

        for i in range(25):
            strategy.on_price(Decimal("50000"))

        signal = strategy.on_price(Decimal("50000"))
        if signal:
            # At mean or above should trigger exit for long
            if signal.deviation_state in [DeviationState.AT_MEAN, DeviationState.ABOVE_MEAN]:
                assert strategy.should_exit(signal)

    def test_check_stop_loss_long(self, strategy):
        """Test stop loss check for long position."""
        strategy.enter_position("long", Decimal("50000"), Decimal("1"), Decimal("-3"))
        # 3% stop loss = 48500
        assert strategy.check_stop_loss(Decimal("48000"))
        assert not strategy.check_stop_loss(Decimal("49000"))

    def test_check_stop_loss_short(self, strategy):
        """Test stop loss check for short position."""
        strategy.enter_position("short", Decimal("50000"), Decimal("1"), Decimal("3"))
        # 3% stop loss = 51500
        assert strategy.check_stop_loss(Decimal("52000"))
        assert not strategy.check_stop_loss(Decimal("51000"))

    def test_get_status(self, strategy):
        """Test getting strategy status."""
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        status = strategy.get_status()
        assert status["in_position"]
        assert status["position_side"] == "long"
        assert "metrics" in status

    def test_win_rate_calculation(self, strategy):
        """Test win rate calculation."""
        # 2 successful, 1 failed
        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        strategy.exit_position(Decimal("50000"), reverted_to_mean=True)

        strategy.enter_position("short", Decimal("52000"), Decimal("1"), Decimal("4"))
        strategy.exit_position(Decimal("50000"), reverted_to_mean=True)

        strategy.enter_position("long", Decimal("48000"), Decimal("1"), Decimal("-4"))
        strategy.exit_position(Decimal("46000"), reverted_to_mean=False)

        # Win rate should be 66.67%
        assert strategy.metrics.win_rate > Decimal("66")
        assert strategy.metrics.win_rate < Decimal("67")


# =============================================================================
# Integration Tests
# =============================================================================

class TestMeanReversionIntegration:
    """Integration tests for mean reversion strategy."""

    def test_full_reversion_cycle(self):
        """Test complete mean reversion cycle."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.PERCENTAGE,
            entry_deviation_pct=Decimal("3"),
            exit_deviation_pct=Decimal("0.5")
        )

        strategy = MeanReversionStrategy(config)

        # Build price history at mean
        for i in range(25):
            strategy.on_price(Decimal("50000"))

        # Price drops - entry signal
        signal = strategy.on_price(Decimal("48000"))
        assert signal is not None
        if strategy.should_enter(signal):
            strategy.enter_position(
                "long",
                Decimal("48000"),
                Decimal("1"),
                signal.deviation_pct
            )

        # Price reverts to mean
        for i in range(5):
            price = Decimal("48000") + (i * 400)
            signal = strategy.on_price(price)

        # Exit at mean
        signal = strategy.on_price(Decimal("50000"))
        if signal and strategy.should_exit(signal):
            pnl = strategy.exit_position(Decimal("50000"), reverted_to_mean=True)
            assert pnl == Decimal("2000")

        assert strategy.metrics.successful_reversions >= 1

    def test_multiple_reversions(self):
        """Test multiple reversion trades."""
        config = MeanReversionConfig(
            symbol="BTC-USD-PERP",
            strategy_type=MeanReversionType.ZSCORE,
            zscore_entry_threshold=Decimal("2")
        )

        strategy = MeanReversionStrategy(config)

        # Execute multiple trades
        trades = [
            ("long", Decimal("48000"), Decimal("50000")),  # Win
            ("short", Decimal("52000"), Decimal("50000")), # Win
            ("long", Decimal("48500"), Decimal("47000")),  # Loss
        ]

        for side, entry, exit_p in trades:
            strategy.enter_position(side, entry, Decimal("1"), Decimal("3"))
            reverted = exit_p == Decimal("50000")
            strategy.exit_position(exit_p, reverted_to_mean=reverted)

        assert strategy.metrics.successful_reversions == 2
        assert strategy.metrics.failed_reversions == 1
        assert strategy.metrics.total_pnl > Decimal("0")
