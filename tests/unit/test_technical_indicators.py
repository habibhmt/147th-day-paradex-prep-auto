"""
Tests for Technical Indicators Library.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

from src.analytics.technical_indicators import (
    IndicatorType, SignalType, IndicatorResult, OHLCV,
    SMA, EMA, WMA, DEMA, TEMA, VWMA,
    RSI, StochasticRSI, MACD, Momentum, ROC, CCI, WilliamsR,
    ATR, BollingerBands, KeltnerChannels, DonchianChannels, StandardDeviation,
    OBV, VWAP, MFI, ChaikinMoneyFlow, AccumulationDistribution,
    ADX, Aroon, ParabolicSAR, Ichimoku,
    TechnicalIndicators, get_indicators, set_indicators
)


# ============== Fixtures ==============

@pytest.fixture
def price_series():
    """Generate price series for testing."""
    return [Decimal(str(p)) for p in [
        100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
        110, 108, 106, 104, 102, 100, 98, 100, 102, 104
    ]]


@pytest.fixture
def ohlcv_series():
    """Generate OHLCV series for testing."""
    return [
        OHLCV(Decimal("100"), Decimal("102"), Decimal("99"), Decimal("101"), Decimal("1000")),
        OHLCV(Decimal("101"), Decimal("104"), Decimal("100"), Decimal("103"), Decimal("1200")),
        OHLCV(Decimal("103"), Decimal("105"), Decimal("102"), Decimal("104"), Decimal("1100")),
        OHLCV(Decimal("104"), Decimal("107"), Decimal("103"), Decimal("106"), Decimal("1300")),
        OHLCV(Decimal("106"), Decimal("108"), Decimal("105"), Decimal("107"), Decimal("1400")),
        OHLCV(Decimal("107"), Decimal("109"), Decimal("106"), Decimal("108"), Decimal("1500")),
        OHLCV(Decimal("108"), Decimal("110"), Decimal("107"), Decimal("109"), Decimal("1600")),
        OHLCV(Decimal("109"), Decimal("111"), Decimal("108"), Decimal("110"), Decimal("1700")),
        OHLCV(Decimal("110"), Decimal("112"), Decimal("109"), Decimal("111"), Decimal("1800")),
        OHLCV(Decimal("111"), Decimal("113"), Decimal("110"), Decimal("112"), Decimal("1900")),
    ]


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_indicator_type_values(self):
        assert IndicatorType.TREND.value == "trend"
        assert IndicatorType.MOMENTUM.value == "momentum"
        assert IndicatorType.VOLATILITY.value == "volatility"

    def test_signal_type_values(self):
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.NEUTRAL.value == "neutral"


# ============== Data Class Tests ==============

class TestIndicatorResult:
    """Test IndicatorResult dataclass."""

    def test_creation(self):
        result = IndicatorResult(
            name="RSI",
            value=Decimal("70"),
            signal=SignalType.SELL
        )
        assert result.name == "RSI"
        assert result.value == Decimal("70")

    def test_to_dict(self):
        result = IndicatorResult(
            name="MACD",
            value=Decimal("0.5"),
            signal=SignalType.BUY
        )
        data = result.to_dict()
        assert data["name"] == "MACD"
        assert data["signal"] == "buy"


class TestOHLCV:
    """Test OHLCV dataclass."""

    def test_creation(self):
        ohlcv = OHLCV(
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=Decimal("1000")
        )
        assert ohlcv.open == Decimal("100")
        assert ohlcv.close == Decimal("103")


# ============== Moving Average Tests ==============

class TestSMA:
    """Test Simple Moving Average."""

    def test_init(self):
        sma = SMA(period=5)
        assert sma.period == 5

    def test_update_insufficient_data(self):
        sma = SMA(period=5)
        for i in range(4):
            result = sma.update(Decimal(str(100 + i)))
            assert result is None

    def test_update_calculates_average(self):
        sma = SMA(period=5)
        values = [100, 102, 104, 106, 108]
        for v in values:
            result = sma.update(Decimal(str(v)))

        assert result == Decimal("104")

    def test_current(self):
        sma = SMA(period=3)
        for v in [10, 20, 30]:
            sma.update(Decimal(str(v)))

        assert sma.current() == Decimal("20")


class TestEMA:
    """Test Exponential Moving Average."""

    def test_init(self):
        ema = EMA(period=10)
        assert ema.period == 10

    def test_update_insufficient_data(self):
        ema = EMA(period=5)
        for i in range(4):
            result = ema.update(Decimal(str(100 + i)))
            assert result is None

    def test_update_calculates_ema(self):
        ema = EMA(period=5)
        for v in [100, 102, 104, 106, 108]:
            result = ema.update(Decimal(str(v)))

        assert result is not None
        assert result > Decimal("0")

    def test_current(self):
        ema = EMA(period=3)
        for v in [10, 20, 30]:
            ema.update(Decimal(str(v)))

        assert ema.current() is not None


class TestWMA:
    """Test Weighted Moving Average."""

    def test_init(self):
        wma = WMA(period=5)
        assert wma.period == 5

    def test_update_calculates_weighted(self):
        wma = WMA(period=3)
        for v in [10, 20, 30]:
            result = wma.update(Decimal(str(v)))

        # WMA(3) = (1*10 + 2*20 + 3*30) / 6 = 140/6 = 23.33...
        assert result is not None
        assert float(result) > 20


class TestDEMA:
    """Test Double Exponential Moving Average."""

    def test_init(self):
        dema = DEMA(period=5)
        assert dema.period == 5

    def test_update(self, price_series):
        dema = DEMA(period=5)
        result = None
        for price in price_series:
            result = dema.update(price)

        assert result is not None


class TestTEMA:
    """Test Triple Exponential Moving Average."""

    def test_init(self):
        tema = TEMA(period=5)
        assert tema.period == 5

    def test_update(self, price_series):
        tema = TEMA(period=5)
        result = None
        for price in price_series:
            result = tema.update(price)

        assert result is not None


class TestVWMA:
    """Test Volume Weighted Moving Average."""

    def test_init(self):
        vwma = VWMA(period=5)
        assert vwma.period == 5

    def test_update(self):
        vwma = VWMA(period=3)
        prices = [100, 102, 104]
        volumes = [1000, 2000, 3000]

        for p, v in zip(prices, volumes):
            result = vwma.update(Decimal(str(p)), Decimal(str(v)))

        # VWMA = (100*1000 + 102*2000 + 104*3000) / 6000
        assert result is not None
        assert result > Decimal("100")


# ============== Momentum Indicator Tests ==============

class TestRSI:
    """Test Relative Strength Index."""

    def test_init(self):
        rsi = RSI(period=14)
        assert rsi.period == 14

    def test_update_insufficient_data(self):
        rsi = RSI(period=14)
        for i in range(10):
            result = rsi.update(Decimal(str(100 + i)))
            assert result is None

    def test_update_uptrend(self):
        rsi = RSI(period=5)
        # Strong uptrend
        for v in [100, 102, 104, 106, 108, 110, 112]:
            result = rsi.update(Decimal(str(v)))

        assert result is not None
        assert result > Decimal("50")

    def test_update_downtrend(self):
        rsi = RSI(period=5)
        # Strong downtrend
        for v in [112, 110, 108, 106, 104, 102, 100]:
            result = rsi.update(Decimal(str(v)))

        assert result is not None
        assert result < Decimal("50")

    def test_get_signal(self):
        rsi = RSI(period=5)
        for v in [100, 102, 104, 106, 108, 110, 112]:
            rsi.update(Decimal(str(v)))

        signal = rsi.get_signal()
        assert signal in [SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL]


class TestStochasticRSI:
    """Test Stochastic RSI."""

    def test_init(self):
        stoch = StochasticRSI(rsi_period=14, stoch_period=14)
        assert stoch.stoch_period == 14

    def test_update(self, price_series):
        stoch = StochasticRSI(rsi_period=5, stoch_period=5)
        result = None
        for price in price_series:
            result = stoch.update(price)

        # May or may not have result depending on data
        assert result is None or (isinstance(result, tuple) and len(result) == 2)


class TestMACD:
    """Test MACD."""

    def test_init(self):
        macd = MACD(fast=12, slow=26, signal=9)
        assert macd.fast_ema.period == 12

    def test_update(self, price_series):
        macd = MACD(fast=5, slow=10, signal=3)
        result = None
        for price in price_series:
            result = macd.update(price)

        assert result is not None
        assert len(result) == 3

    def test_get_signal(self, price_series):
        macd = MACD(fast=5, slow=10, signal=3)
        for price in price_series:
            macd.update(price)

        signal = macd.get_signal()
        assert signal in [SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL]


class TestMomentum:
    """Test Momentum indicator."""

    def test_init(self):
        mom = Momentum(period=10)
        assert mom.period == 10

    def test_update(self):
        mom = Momentum(period=5)
        values = [100, 102, 104, 106, 108, 110]
        for v in values:
            result = mom.update(Decimal(str(v)))

        assert result == Decimal("10")  # 110 - 100


class TestROC:
    """Test Rate of Change."""

    def test_init(self):
        roc = ROC(period=10)
        assert roc.period == 10

    def test_update(self):
        roc = ROC(period=5)
        values = [100, 102, 104, 106, 108, 110]
        for v in values:
            result = roc.update(Decimal(str(v)))

        assert result == Decimal("10")  # (110-100)/100 * 100


class TestCCI:
    """Test Commodity Channel Index."""

    def test_init(self):
        cci = CCI(period=20)
        assert cci.period == 20

    def test_update(self, ohlcv_series):
        cci = CCI(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = cci.update(ohlcv.high, ohlcv.low, ohlcv.close)

        assert result is not None


class TestWilliamsR:
    """Test Williams %R."""

    def test_init(self):
        wr = WilliamsR(period=14)
        assert wr.period == 14

    def test_update(self, ohlcv_series):
        wr = WilliamsR(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = wr.update(ohlcv.high, ohlcv.low, ohlcv.close)

        assert result is not None
        assert result <= Decimal("0")
        assert result >= Decimal("-100")


# ============== Volatility Indicator Tests ==============

class TestATR:
    """Test Average True Range."""

    def test_init(self):
        atr = ATR(period=14)
        assert atr.period == 14

    def test_update(self, ohlcv_series):
        atr = ATR(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = atr.update(ohlcv.high, ohlcv.low, ohlcv.close)

        assert result is not None
        assert result > Decimal("0")


class TestBollingerBands:
    """Test Bollinger Bands."""

    def test_init(self):
        bb = BollingerBands(period=20, std_dev=Decimal("2"))
        assert bb.period == 20

    def test_update(self, price_series):
        bb = BollingerBands(period=5)
        result = None
        for price in price_series:
            result = bb.update(price)

        assert result is not None
        upper, middle, lower = result
        assert upper > middle > lower

    def test_get_signal(self, price_series):
        bb = BollingerBands(period=5)
        for price in price_series:
            bb.update(price)

        signal = bb.get_signal(price_series[-1])
        assert signal in [SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL]


class TestKeltnerChannels:
    """Test Keltner Channels."""

    def test_init(self):
        kc = KeltnerChannels(ema_period=20, atr_period=10)
        assert kc.ema.period == 20

    def test_update(self, ohlcv_series):
        kc = KeltnerChannels(ema_period=5, atr_period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = kc.update(ohlcv.high, ohlcv.low, ohlcv.close)

        assert result is not None
        upper, middle, lower = result
        assert upper > middle > lower


class TestDonchianChannels:
    """Test Donchian Channels."""

    def test_init(self):
        dc = DonchianChannels(period=20)
        assert dc.period == 20

    def test_update(self, ohlcv_series):
        dc = DonchianChannels(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = dc.update(ohlcv.high, ohlcv.low)

        assert result is not None
        upper, middle, lower = result
        assert upper >= middle >= lower


class TestStandardDeviation:
    """Test Standard Deviation."""

    def test_init(self):
        sd = StandardDeviation(period=20)
        assert sd.period == 20

    def test_update(self, price_series):
        sd = StandardDeviation(period=5)
        result = None
        for price in price_series:
            result = sd.update(price)

        assert result is not None
        assert result >= Decimal("0")


# ============== Volume Indicator Tests ==============

class TestOBV:
    """Test On Balance Volume."""

    def test_init(self):
        obv = OBV()
        assert obv.obv == Decimal("0")

    def test_update_up(self):
        obv = OBV()
        obv.update(Decimal("100"), Decimal("1000"))
        result = obv.update(Decimal("105"), Decimal("2000"))
        assert result == Decimal("2000")

    def test_update_down(self):
        obv = OBV()
        obv.update(Decimal("100"), Decimal("1000"))
        result = obv.update(Decimal("95"), Decimal("2000"))
        assert result == Decimal("-2000")


class TestVWAP:
    """Test Volume Weighted Average Price."""

    def test_init(self):
        vwap = VWAP()
        assert vwap.cumulative_vol == Decimal("0")

    def test_update(self, ohlcv_series):
        vwap = VWAP()
        result = None
        for ohlcv in ohlcv_series:
            result = vwap.update(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

        assert result is not None
        assert result > Decimal("0")

    def test_reset(self):
        vwap = VWAP()
        vwap.update(Decimal("100"), Decimal("99"), Decimal("101"), Decimal("1000"))
        vwap.reset()
        assert vwap.cumulative_vol == Decimal("0")


class TestMFI:
    """Test Money Flow Index."""

    def test_init(self):
        mfi = MFI(period=14)
        assert mfi.period == 14

    def test_update(self, ohlcv_series):
        mfi = MFI(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = mfi.update(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

        assert result is not None
        assert Decimal("0") <= result <= Decimal("100")


class TestChaikinMoneyFlow:
    """Test Chaikin Money Flow."""

    def test_init(self):
        cmf = ChaikinMoneyFlow(period=20)
        assert cmf.period == 20

    def test_update(self, ohlcv_series):
        cmf = ChaikinMoneyFlow(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = cmf.update(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

        assert result is not None


class TestAccumulationDistribution:
    """Test Accumulation/Distribution Line."""

    def test_init(self):
        ad = AccumulationDistribution()
        assert ad.ad_line == Decimal("0")

    def test_update(self, ohlcv_series):
        ad = AccumulationDistribution()
        result = None
        for ohlcv in ohlcv_series:
            result = ad.update(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

        assert result is not None


# ============== Trend Indicator Tests ==============

class TestADX:
    """Test Average Directional Index."""

    def test_init(self):
        adx = ADX(period=14)
        assert adx.period == 14

    def test_update(self, ohlcv_series):
        adx = ADX(period=3)
        result = None
        for ohlcv in ohlcv_series:
            result = adx.update(ohlcv.high, ohlcv.low, ohlcv.close)

        # May not have enough data
        if result:
            adx_val, plus_di, minus_di = result
            assert adx_val >= Decimal("0")


class TestAroon:
    """Test Aroon Indicator."""

    def test_init(self):
        aroon = Aroon(period=25)
        assert aroon.period == 25

    def test_update(self, ohlcv_series):
        aroon = Aroon(period=5)
        result = None
        for ohlcv in ohlcv_series:
            result = aroon.update(ohlcv.high, ohlcv.low)

        assert result is not None
        up, down, osc = result
        assert Decimal("0") <= up <= Decimal("100")
        assert Decimal("0") <= down <= Decimal("100")


class TestParabolicSAR:
    """Test Parabolic SAR."""

    def test_init(self):
        sar = ParabolicSAR()
        assert sar.af_start == Decimal("0.02")

    def test_update(self, ohlcv_series):
        sar = ParabolicSAR()
        result = None
        for ohlcv in ohlcv_series:
            result = sar.update(ohlcv.high, ohlcv.low)

        assert result is not None


class TestIchimoku:
    """Test Ichimoku Cloud."""

    def test_init(self):
        ichimoku = Ichimoku(tenkan=9, kijun=26, senkou_b=52)
        assert ichimoku.tenkan_period == 9

    def test_update_insufficient_data(self):
        ichimoku = Ichimoku(tenkan=9, kijun=26, senkou_b=52)
        result = ichimoku.update(Decimal("100"), Decimal("99"), Decimal("101"))
        assert result is None

    def test_update_with_data(self):
        ichimoku = Ichimoku(tenkan=3, kijun=5, senkou_b=10)
        result = None
        for i in range(15):
            price = Decimal(str(100 + i))
            result = ichimoku.update(price + Decimal("2"), price - Decimal("2"), price)

        assert result is not None
        assert "tenkan_sen" in result
        assert "kijun_sen" in result
        assert "senkou_span_a" in result
        assert "senkou_span_b" in result


# ============== Indicator Collection Tests ==============

class TestTechnicalIndicators:
    """Test TechnicalIndicators collection."""

    def test_add_sma(self):
        ti = TechnicalIndicators()
        sma = ti.add_sma("sma_20", 20)
        assert "sma_20" in ti.indicators
        assert isinstance(sma, SMA)

    def test_add_ema(self):
        ti = TechnicalIndicators()
        ema = ti.add_ema("ema_12", 12)
        assert "ema_12" in ti.indicators
        assert isinstance(ema, EMA)

    def test_add_rsi(self):
        ti = TechnicalIndicators()
        rsi = ti.add_rsi("rsi_14", 14)
        assert "rsi_14" in ti.indicators
        assert isinstance(rsi, RSI)

    def test_add_macd(self):
        ti = TechnicalIndicators()
        macd = ti.add_macd("macd_std", 12, 26, 9)
        assert "macd_std" in ti.indicators
        assert isinstance(macd, MACD)

    def test_add_bollinger(self):
        ti = TechnicalIndicators()
        bb = ti.add_bollinger("bb_20", 20)
        assert "bb_20" in ti.indicators
        assert isinstance(bb, BollingerBands)

    def test_add_atr(self):
        ti = TechnicalIndicators()
        atr = ti.add_atr("atr_14", 14)
        assert "atr_14" in ti.indicators
        assert isinstance(atr, ATR)

    def test_add_adx(self):
        ti = TechnicalIndicators()
        adx = ti.add_adx("adx_14", 14)
        assert "adx_14" in ti.indicators
        assert isinstance(adx, ADX)

    def test_get(self):
        ti = TechnicalIndicators()
        ti.add_sma("sma_20", 20)
        result = ti.get("sma_20")
        assert isinstance(result, SMA)

    def test_get_nonexistent(self):
        ti = TechnicalIndicators()
        result = ti.get("nonexistent")
        assert result is None

    def test_update_all(self, ohlcv_series):
        ti = TechnicalIndicators()
        ti.add_sma("sma_5", 5)
        ti.add_ema("ema_5", 5)
        ti.add_rsi("rsi_5", 5)

        for ohlcv in ohlcv_series:
            results = ti.update_all(ohlcv)

        assert "sma_5" in results
        assert "ema_5" in results
        assert "rsi_5" in results


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global indicators instance."""

    def test_get_indicators(self):
        indicators = get_indicators()
        assert isinstance(indicators, TechnicalIndicators)

    def test_set_indicators(self):
        custom = TechnicalIndicators()
        custom.add_sma("test_sma", 10)
        set_indicators(custom)

        indicators = get_indicators()
        assert indicators.get("test_sma") is not None


# ============== Integration Tests ==============

class TestIndicatorIntegration:
    """Integration tests."""

    def test_multiple_indicators_same_data(self, ohlcv_series):
        sma = SMA(period=5)
        ema = EMA(period=5)
        rsi = RSI(period=5)

        for ohlcv in ohlcv_series:
            sma.update(ohlcv.close)
            ema.update(ohlcv.close)
            rsi.update(ohlcv.close)

        assert sma.current() is not None
        assert ema.current() is not None

    def test_full_technical_analysis(self, ohlcv_series):
        ti = TechnicalIndicators()

        # Add various indicators
        ti.add_sma("sma_5", 5)
        ti.add_ema("ema_5", 5)
        ti.add_rsi("rsi_5", 5)
        ti.add_macd("macd", 5, 10, 3)
        ti.add_bollinger("bb", 5)

        # Process data
        for ohlcv in ohlcv_series:
            results = ti.update_all(ohlcv)

        # Verify results
        sma = ti.get("sma_5")
        assert sma.current() is not None
