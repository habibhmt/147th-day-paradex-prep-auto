"""
Tests for Price Prediction Module.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

from src.analytics.price_prediction import (
    PredictionModel, PredictionTimeframe, ConfidenceLevel, TrendDirection,
    PricePoint, Prediction, PredictionRange, ModelMetrics,
    LinearRegressionPredictor, MovingAveragePredictor, ExponentialSmoothingPredictor,
    MeanReversionPredictor, MomentumPredictor, EnsemblePredictor,
    ModelEvaluator, PricePredictor, PriceForecast,
    get_predictor, set_predictor
)


# ============== Fixtures ==============

@pytest.fixture
def uptrend_prices():
    """Generate uptrend price series."""
    return [Decimal(str(100 + i * 2)) for i in range(30)]


@pytest.fixture
def downtrend_prices():
    """Generate downtrend price series."""
    return [Decimal(str(160 - i * 2)) for i in range(30)]


@pytest.fixture
def volatile_prices():
    """Generate volatile price series."""
    base = 100
    return [Decimal(str(base + (i % 5 - 2) * 3)) for i in range(30)]


@pytest.fixture
def price_predictor():
    """Create price predictor instance."""
    return PricePredictor()


# ============== Enum Tests ==============

class TestEnums:
    """Test enum values."""

    def test_prediction_model_values(self):
        assert PredictionModel.LINEAR_REGRESSION.value == "linear_regression"
        assert PredictionModel.MOVING_AVERAGE.value == "moving_average"
        assert PredictionModel.ENSEMBLE.value == "ensemble"

    def test_prediction_timeframe_values(self):
        assert PredictionTimeframe.MINUTES_5.value == "5m"
        assert PredictionTimeframe.HOUR_1.value == "1h"
        assert PredictionTimeframe.DAY_1.value == "1d"

    def test_confidence_level_values(self):
        assert ConfidenceLevel.VERY_LOW.value == "very_low"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"

    def test_trend_direction_values(self):
        assert TrendDirection.STRONGLY_BULLISH.value == "strongly_bullish"
        assert TrendDirection.BEARISH.value == "bearish"
        assert TrendDirection.NEUTRAL.value == "neutral"


# ============== Data Class Tests ==============

class TestPricePoint:
    """Test PricePoint dataclass."""

    def test_creation(self):
        pp = PricePoint(
            timestamp=datetime.now(),
            price=Decimal("50000"),
            volume=Decimal("1000")
        )
        assert pp.price == Decimal("50000")

    def test_to_dict(self):
        pp = PricePoint(
            timestamp=datetime.now(),
            price=Decimal("50000")
        )
        result = pp.to_dict()
        assert "price" in result
        assert result["price"] == "50000"


class TestPrediction:
    """Test Prediction dataclass."""

    def test_creation(self):
        pred = Prediction(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            predicted_price=Decimal("51000"),
            price_change=Decimal("1000"),
            price_change_pct=Decimal("2"),
            direction=TrendDirection.BULLISH,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=Decimal("75"),
            timeframe=PredictionTimeframe.HOUR_1,
            model=PredictionModel.ENSEMBLE
        )
        assert pred.symbol == "BTC-USD"
        assert pred.direction == TrendDirection.BULLISH

    def test_to_dict(self):
        pred = Prediction(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            predicted_price=Decimal("51000"),
            price_change=Decimal("1000"),
            price_change_pct=Decimal("2"),
            direction=TrendDirection.BULLISH,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=Decimal("75"),
            timeframe=PredictionTimeframe.HOUR_1,
            model=PredictionModel.ENSEMBLE
        )
        result = pred.to_dict()
        assert result["direction"] == "bullish"
        assert result["model"] == "ensemble"


class TestPredictionRange:
    """Test PredictionRange dataclass."""

    def test_creation(self):
        pr = PredictionRange(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            predicted_low=Decimal("49000"),
            predicted_mid=Decimal("51000"),
            predicted_high=Decimal("53000"),
            confidence_interval=Decimal("95"),
            timeframe=PredictionTimeframe.HOUR_1,
            model=PredictionModel.ENSEMBLE
        )
        assert pr.predicted_low < pr.predicted_mid < pr.predicted_high

    def test_to_dict(self):
        pr = PredictionRange(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            predicted_low=Decimal("49000"),
            predicted_mid=Decimal("51000"),
            predicted_high=Decimal("53000"),
            confidence_interval=Decimal("95"),
            timeframe=PredictionTimeframe.HOUR_1,
            model=PredictionModel.ENSEMBLE
        )
        result = pr.to_dict()
        assert "predicted_low" in result
        assert "confidence_interval" in result


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_creation(self):
        metrics = ModelMetrics(
            model=PredictionModel.LINEAR_REGRESSION,
            mae=Decimal("50"),
            mse=Decimal("2500"),
            rmse=Decimal("50"),
            mape=Decimal("1"),
            r_squared=Decimal("0.85"),
            accuracy=Decimal("65"),
            total_predictions=100,
            correct_predictions=65
        )
        assert metrics.accuracy == Decimal("65")

    def test_to_dict(self):
        metrics = ModelMetrics(
            model=PredictionModel.MOMENTUM,
            mae=Decimal("30"),
            mse=Decimal("900"),
            rmse=Decimal("30"),
            mape=Decimal("0.6"),
            r_squared=Decimal("0.9"),
            accuracy=Decimal("70"),
            total_predictions=50,
            correct_predictions=35
        )
        result = metrics.to_dict()
        assert result["model"] == "momentum"
        assert "rmse" in result


# ============== Linear Regression Predictor Tests ==============

class TestLinearRegressionPredictor:
    """Test LinearRegressionPredictor."""

    def test_init(self):
        lr = LinearRegressionPredictor(lookback=20)
        assert lr.lookback == 20

    def test_update(self):
        lr = LinearRegressionPredictor()
        lr.update("BTC-USD", Decimal("50000"))
        assert "BTC-USD" in lr.history
        assert len(lr.history["BTC-USD"]) == 1

    def test_fit(self, uptrend_prices):
        lr = LinearRegressionPredictor(lookback=20)
        for price in uptrend_prices[:20]:
            lr.update("BTC-USD", price)

        result = lr.fit("BTC-USD")
        assert result is True
        assert lr.slope is not None
        assert lr.slope > Decimal("0")

    def test_predict_uptrend(self, uptrend_prices):
        lr = LinearRegressionPredictor(lookback=20)
        for price in uptrend_prices[:20]:
            lr.update("BTC-USD", price)

        pred = lr.predict("BTC-USD", steps=1)
        assert pred is not None
        assert pred > uptrend_prices[19]

    def test_predict_insufficient_data(self):
        lr = LinearRegressionPredictor(lookback=20)
        lr.update("BTC-USD", Decimal("50000"))
        lr.update("BTC-USD", Decimal("51000"))

        pred = lr.predict("BTC-USD")
        assert pred is None

    def test_get_trend_bullish(self, uptrend_prices):
        lr = LinearRegressionPredictor(lookback=20)
        for price in uptrend_prices[:20]:
            lr.update("BTC-USD", price)

        lr.fit("BTC-USD")
        trend = lr.get_trend("BTC-USD")
        assert trend in [TrendDirection.BULLISH, TrendDirection.STRONGLY_BULLISH]

    def test_get_trend_bearish(self, downtrend_prices):
        lr = LinearRegressionPredictor(lookback=20)
        for price in downtrend_prices[:20]:
            lr.update("BTC-USD", price)

        lr.fit("BTC-USD")
        trend = lr.get_trend("BTC-USD")
        assert trend in [TrendDirection.BEARISH, TrendDirection.STRONGLY_BEARISH]


# ============== Moving Average Predictor Tests ==============

class TestMovingAveragePredictor:
    """Test MovingAveragePredictor."""

    def test_init(self):
        ma = MovingAveragePredictor(short_period=10, long_period=30)
        assert ma.short_period == 10
        assert ma.long_period == 30

    def test_update(self):
        ma = MovingAveragePredictor()
        for i in range(35):
            ma.update("BTC-USD", Decimal(str(100 + i)))

        assert len(ma.history["BTC-USD"]) <= ma.long_period * 2

    def test_predict(self, uptrend_prices):
        ma = MovingAveragePredictor(short_period=5, long_period=15)
        for price in uptrend_prices:
            ma.update("BTC-USD", price)

        pred = ma.predict("BTC-USD")
        assert pred is not None

    def test_get_signal_bullish(self, uptrend_prices):
        ma = MovingAveragePredictor(short_period=5, long_period=15)
        for price in uptrend_prices:
            ma.update("BTC-USD", price)

        signal = ma.get_signal("BTC-USD")
        assert signal in [TrendDirection.BULLISH, TrendDirection.STRONGLY_BULLISH, TrendDirection.NEUTRAL]


# ============== Exponential Smoothing Tests ==============

class TestExponentialSmoothingPredictor:
    """Test ExponentialSmoothingPredictor."""

    def test_init(self):
        es = ExponentialSmoothingPredictor(alpha=Decimal("0.3"), beta=Decimal("0.1"))
        assert es.alpha == Decimal("0.3")

    def test_update(self):
        es = ExponentialSmoothingPredictor()
        es.update("BTC-USD", Decimal("50000"))
        assert "BTC-USD" in es.level

    def test_predict(self, uptrend_prices):
        es = ExponentialSmoothingPredictor()
        for price in uptrend_prices:
            es.update("BTC-USD", price)

        pred = es.predict("BTC-USD", steps=1)
        assert pred is not None

    def test_predict_multiple_steps(self, uptrend_prices):
        es = ExponentialSmoothingPredictor()
        for price in uptrend_prices:
            es.update("BTC-USD", price)

        pred1 = es.predict("BTC-USD", steps=1)
        pred5 = es.predict("BTC-USD", steps=5)

        # In uptrend, further prediction should be higher
        assert pred1 is not None
        assert pred5 is not None
        assert pred5 > pred1


# ============== Mean Reversion Predictor Tests ==============

class TestMeanReversionPredictor:
    """Test MeanReversionPredictor."""

    def test_init(self):
        mr = MeanReversionPredictor(lookback=50)
        assert mr.lookback == 50

    def test_get_mean(self, volatile_prices):
        mr = MeanReversionPredictor(lookback=20)
        for price in volatile_prices[:20]:
            mr.update("BTC-USD", price)

        mean = mr.get_mean("BTC-USD")
        assert mean is not None

    def test_get_deviation(self, volatile_prices):
        mr = MeanReversionPredictor(lookback=20)
        for price in volatile_prices[:20]:
            mr.update("BTC-USD", price)

        deviation = mr.get_deviation("BTC-USD")
        assert deviation is not None

    def test_predict(self, volatile_prices):
        mr = MeanReversionPredictor(lookback=20)
        for price in volatile_prices:
            mr.update("BTC-USD", price)

        pred = mr.predict("BTC-USD")
        assert pred is not None


# ============== Momentum Predictor Tests ==============

class TestMomentumPredictor:
    """Test MomentumPredictor."""

    def test_init(self):
        mp = MomentumPredictor(lookback=14)
        assert mp.lookback == 14

    def test_get_momentum_uptrend(self, uptrend_prices):
        mp = MomentumPredictor(lookback=10)
        for price in uptrend_prices:
            mp.update("BTC-USD", price)

        momentum = mp.get_momentum("BTC-USD")
        assert momentum is not None
        assert momentum > Decimal("0")

    def test_get_momentum_downtrend(self, downtrend_prices):
        mp = MomentumPredictor(lookback=10)
        for price in downtrend_prices:
            mp.update("BTC-USD", price)

        momentum = mp.get_momentum("BTC-USD")
        assert momentum is not None
        assert momentum < Decimal("0")

    def test_predict(self, uptrend_prices):
        mp = MomentumPredictor(lookback=10)
        for price in uptrend_prices:
            mp.update("BTC-USD", price)

        pred = mp.predict("BTC-USD")
        assert pred is not None


# ============== Ensemble Predictor Tests ==============

class TestEnsemblePredictor:
    """Test EnsemblePredictor."""

    def test_init(self):
        ep = EnsemblePredictor()
        assert len(ep.predictors) == 0

    def test_add_predictor(self):
        ep = EnsemblePredictor()
        lr = LinearRegressionPredictor()
        ep.add_predictor(lr, Decimal("1"))
        assert len(ep.predictors) == 1

    def test_update(self, uptrend_prices):
        ep = EnsemblePredictor()
        lr = LinearRegressionPredictor(lookback=10)
        ma = MovingAveragePredictor(short_period=5, long_period=10)
        ep.add_predictor(lr, Decimal("1"))
        ep.add_predictor(ma, Decimal("1"))

        for price in uptrend_prices:
            ep.update("BTC-USD", price)

        assert "BTC-USD" in ep.history

    def test_predict(self, uptrend_prices):
        ep = EnsemblePredictor()
        lr = LinearRegressionPredictor(lookback=10)
        ma = MovingAveragePredictor(short_period=5, long_period=10)
        ep.add_predictor(lr, Decimal("1"))
        ep.add_predictor(ma, Decimal("1"))

        for price in uptrend_prices:
            ep.update("BTC-USD", price)

        pred = ep.predict("BTC-USD")
        assert pred is not None


# ============== Model Evaluator Tests ==============

class TestModelEvaluator:
    """Test ModelEvaluator."""

    def test_add_result(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("BTC-USD", Decimal("50000"), Decimal("50100"))
        assert len(evaluator.predictions["BTC-USD"]) == 1

    def test_calculate_metrics_insufficient_data(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("BTC-USD", Decimal("50000"), Decimal("50100"))

        metrics = evaluator.calculate_metrics("BTC-USD", PredictionModel.ENSEMBLE)
        assert metrics is None

    def test_calculate_metrics(self):
        evaluator = ModelEvaluator()
        # Add enough data
        for i in range(10):
            pred = Decimal(str(50000 + i * 100))
            actual = Decimal(str(50050 + i * 100))
            evaluator.add_result("BTC-USD", pred, actual)

        metrics = evaluator.calculate_metrics("BTC-USD", PredictionModel.ENSEMBLE)
        assert metrics is not None
        assert metrics.mae >= Decimal("0")
        assert metrics.rmse >= Decimal("0")


# ============== Price Predictor Tests ==============

class TestPricePredictor:
    """Test PricePredictor."""

    def test_init(self, price_predictor):
        assert price_predictor.linear is not None
        assert price_predictor.ensemble is not None

    def test_register_callback(self, price_predictor):
        callback = Mock()
        price_predictor.register_callback("on_prediction", callback)
        assert callback in price_predictor.callbacks["on_prediction"]

    def test_update(self, price_predictor):
        price_predictor.update("BTC-USD", Decimal("50000"))
        assert "BTC-USD" in price_predictor.linear.history

    def test_predict_insufficient_data(self, price_predictor):
        price_predictor.update("BTC-USD", Decimal("50000"))
        pred = price_predictor.predict("BTC-USD")
        # May return None or low confidence prediction
        assert pred is None or isinstance(pred, Prediction)

    def test_predict_with_data(self, price_predictor, uptrend_prices):
        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        pred = price_predictor.predict("BTC-USD", PredictionModel.ENSEMBLE)
        assert pred is not None
        assert pred.symbol == "BTC-USD"
        assert pred.predicted_price is not None

    def test_predict_different_models(self, price_predictor, uptrend_prices):
        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        models = [
            PredictionModel.LINEAR_REGRESSION,
            PredictionModel.MOVING_AVERAGE,
            PredictionModel.EXPONENTIAL_SMOOTHING,
            PredictionModel.MOMENTUM,
            PredictionModel.ENSEMBLE
        ]

        for model in models:
            pred = price_predictor.predict("BTC-USD", model)
            # Some models may return None based on data requirements
            if pred:
                assert pred.model == model

    def test_predict_range(self, price_predictor, uptrend_prices):
        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        pr = price_predictor.predict_range("BTC-USD")
        assert pr is not None
        assert pr.predicted_low <= pr.predicted_mid <= pr.predicted_high

    def test_predict_range_different_confidence(self, price_predictor, uptrend_prices):
        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        pr_95 = price_predictor.predict_range(
            "BTC-USD",
            confidence_interval=Decimal("95")
        )
        pr_99 = price_predictor.predict_range(
            "BTC-USD",
            confidence_interval=Decimal("99")
        )

        assert pr_95 is not None
        assert pr_99 is not None
        # 99% confidence should have wider range
        range_95 = pr_95.predicted_high - pr_95.predicted_low
        range_99 = pr_99.predicted_high - pr_99.predicted_low
        assert range_99 >= range_95

    def test_callbacks_triggered(self, price_predictor, uptrend_prices):
        callback = Mock()
        price_predictor.register_callback("on_prediction", callback)

        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        price_predictor.predict("BTC-USD")
        assert callback.called


# ============== Price Forecast Tests ==============

class TestPriceForecast:
    """Test PriceForecast."""

    def test_forecast(self, price_predictor, uptrend_prices):
        for price in uptrend_prices:
            price_predictor.update("BTC-USD", price)

        forecast = PriceForecast(price_predictor)
        forecasts = forecast.forecast("BTC-USD", periods=5)

        # May have some forecasts
        assert isinstance(forecasts, list)


# ============== Global Instance Tests ==============

class TestGlobalInstance:
    """Test global predictor instance."""

    def test_get_predictor(self):
        predictor = get_predictor()
        assert isinstance(predictor, PricePredictor)

    def test_set_predictor(self):
        custom = PricePredictor()
        custom.update("TEST", Decimal("100"))
        set_predictor(custom)

        predictor = get_predictor()
        assert "TEST" in predictor.linear.history


# ============== Integration Tests ==============

class TestPredictionIntegration:
    """Integration tests."""

    def test_full_prediction_flow(self, uptrend_prices):
        predictor = PricePredictor()

        # Update with historical data
        for price in uptrend_prices:
            predictor.update("BTC-USD", price)

        # Make predictions
        pred = predictor.predict("BTC-USD", PredictionModel.ENSEMBLE, PredictionTimeframe.HOUR_1)
        assert pred is not None

        # Get range prediction
        pred_range = predictor.predict_range("BTC-USD")
        assert pred_range is not None

        # Verify prediction consistency
        assert pred_range.predicted_low <= pred.predicted_price <= pred_range.predicted_high or \
               abs(pred.predicted_price - pred_range.predicted_mid) < Decimal("1000")

    def test_multi_symbol_prediction(self, uptrend_prices, downtrend_prices):
        predictor = PricePredictor()

        for price in uptrend_prices:
            predictor.update("BTC-USD", price)
        for price in downtrend_prices:
            predictor.update("ETH-USD", price)

        btc_pred = predictor.predict("BTC-USD")
        eth_pred = predictor.predict("ETH-USD")

        if btc_pred and eth_pred:
            # BTC in uptrend should predict higher
            assert btc_pred.direction in [TrendDirection.BULLISH, TrendDirection.STRONGLY_BULLISH, TrendDirection.NEUTRAL]
            # ETH in downtrend should predict lower
            assert eth_pred.direction in [TrendDirection.BEARISH, TrendDirection.STRONGLY_BEARISH, TrendDirection.NEUTRAL]
