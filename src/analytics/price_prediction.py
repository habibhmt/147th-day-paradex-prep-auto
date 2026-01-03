"""
Price Prediction Module.

Statistical and ML-based price prediction methods
for trading signal generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import math
import random


class PredictionModel(Enum):
    """Prediction model types."""
    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ENSEMBLE = "ensemble"


class PredictionTimeframe(Enum):
    """Prediction timeframe."""
    MINUTES_5 = "5m"
    MINUTES_15 = "15m"
    HOUR_1 = "1h"
    HOURS_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class ConfidenceLevel(Enum):
    """Prediction confidence level."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrendDirection(Enum):
    """Predicted trend direction."""
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


@dataclass
class PricePoint:
    """Price data point."""
    timestamp: datetime
    price: Decimal
    volume: Optional[Decimal] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "volume": str(self.volume) if self.volume else None
        }


@dataclass
class Prediction:
    """Price prediction result."""
    symbol: str
    current_price: Decimal
    predicted_price: Decimal
    price_change: Decimal
    price_change_pct: Decimal
    direction: TrendDirection
    confidence: ConfidenceLevel
    confidence_score: Decimal  # 0-100
    timeframe: PredictionTimeframe
    model: PredictionModel
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": str(self.current_price),
            "predicted_price": str(self.predicted_price),
            "price_change": str(self.price_change),
            "price_change_pct": str(self.price_change_pct),
            "direction": self.direction.value,
            "confidence": self.confidence.value,
            "confidence_score": str(self.confidence_score),
            "timeframe": self.timeframe.value,
            "model": self.model.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class PredictionRange:
    """Price prediction with confidence interval."""
    symbol: str
    current_price: Decimal
    predicted_low: Decimal
    predicted_mid: Decimal
    predicted_high: Decimal
    confidence_interval: Decimal  # e.g., 95%
    timeframe: PredictionTimeframe
    model: PredictionModel
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": str(self.current_price),
            "predicted_low": str(self.predicted_low),
            "predicted_mid": str(self.predicted_mid),
            "predicted_high": str(self.predicted_high),
            "confidence_interval": str(self.confidence_interval),
            "timeframe": self.timeframe.value,
            "model": self.model.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model: PredictionModel
    mae: Decimal  # Mean Absolute Error
    mse: Decimal  # Mean Squared Error
    rmse: Decimal  # Root Mean Squared Error
    mape: Decimal  # Mean Absolute Percentage Error
    r_squared: Decimal  # R-squared
    accuracy: Decimal  # Direction accuracy
    total_predictions: int
    correct_predictions: int

    def to_dict(self) -> dict:
        return {
            "model": self.model.value,
            "mae": str(self.mae),
            "mse": str(self.mse),
            "rmse": str(self.rmse),
            "mape": str(self.mape),
            "r_squared": str(self.r_squared),
            "accuracy": str(self.accuracy),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions
        }


class LinearRegressionPredictor:
    """Linear regression based prediction."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.history: dict[str, list[Decimal]] = {}
        self.slope: Optional[Decimal] = None
        self.intercept: Optional[Decimal] = None

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.lookback:
            self.history[symbol].pop(0)

    def fit(self, symbol: str) -> bool:
        """Fit linear regression model."""
        prices = self.history.get(symbol, [])
        if len(prices) < 3:
            return False

        n = len(prices)
        x = list(range(n))
        y = [float(p) for p in prices]

        # Calculate slope and intercept
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return False

        self.slope = Decimal(str(numerator / denominator))
        self.intercept = Decimal(str(y_mean - float(self.slope) * x_mean))

        return True

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict future price."""
        if not self.fit(symbol):
            return None

        n = len(self.history.get(symbol, []))
        future_x = n + steps - 1

        predicted = self.intercept + self.slope * Decimal(str(future_x))
        return predicted

    def get_trend(self, symbol: str) -> Optional[TrendDirection]:
        """Get trend direction based on slope."""
        if self.slope is None:
            return None

        prices = self.history.get(symbol, [])
        if not prices:
            return None

        avg_price = sum(prices) / Decimal(str(len(prices)))
        slope_pct = (self.slope / avg_price) * Decimal("100")

        if slope_pct > Decimal("2"):
            return TrendDirection.STRONGLY_BULLISH
        elif slope_pct > Decimal("0.5"):
            return TrendDirection.BULLISH
        elif slope_pct < Decimal("-2"):
            return TrendDirection.STRONGLY_BEARISH
        elif slope_pct < Decimal("-0.5"):
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL


class MovingAveragePredictor:
    """Moving average based prediction."""

    def __init__(self, short_period: int = 10, long_period: int = 30):
        self.short_period = short_period
        self.long_period = long_period
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.long_period * 2:
            self.history[symbol].pop(0)

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict using MA crossover momentum."""
        prices = self.history.get(symbol, [])
        if len(prices) < self.long_period:
            return None

        short_ma = sum(prices[-self.short_period:]) / Decimal(str(self.short_period))
        long_ma = sum(prices[-self.long_period:]) / Decimal(str(self.long_period))

        # Project based on MA difference
        momentum = short_ma - long_ma
        current = prices[-1]

        # Scale momentum by steps
        predicted = current + momentum * Decimal(str(steps)) * Decimal("0.5")
        return predicted

    def get_signal(self, symbol: str) -> TrendDirection:
        """Get trend signal from MA crossover."""
        prices = self.history.get(symbol, [])
        if len(prices) < self.long_period:
            return TrendDirection.NEUTRAL

        short_ma = sum(prices[-self.short_period:]) / Decimal(str(self.short_period))
        long_ma = sum(prices[-self.long_period:]) / Decimal(str(self.long_period))

        diff_pct = (short_ma - long_ma) / long_ma * Decimal("100")

        if diff_pct > Decimal("2"):
            return TrendDirection.STRONGLY_BULLISH
        elif diff_pct > Decimal("0.5"):
            return TrendDirection.BULLISH
        elif diff_pct < Decimal("-2"):
            return TrendDirection.STRONGLY_BEARISH
        elif diff_pct < Decimal("-0.5"):
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL


class ExponentialSmoothingPredictor:
    """Exponential smoothing (Holt-Winters) prediction."""

    def __init__(self, alpha: Decimal = Decimal("0.3"), beta: Decimal = Decimal("0.1")):
        self.alpha = alpha
        self.beta = beta
        self.level: dict[str, Decimal] = {}
        self.trend: dict[str, Decimal] = {}
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update with new price."""
        if symbol not in self.history:
            self.history[symbol] = []
            self.level[symbol] = price
            self.trend[symbol] = Decimal("0")
        else:
            prev_level = self.level[symbol]
            prev_trend = self.trend[symbol]

            # Update level
            self.level[symbol] = self.alpha * price + (Decimal("1") - self.alpha) * (prev_level + prev_trend)

            # Update trend
            self.trend[symbol] = self.beta * (self.level[symbol] - prev_level) + (Decimal("1") - self.beta) * prev_trend

        self.history[symbol].append(price)
        if len(self.history[symbol]) > 100:
            self.history[symbol].pop(0)

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict future price."""
        if symbol not in self.level:
            return None

        level = self.level[symbol]
        trend = self.trend[symbol]

        return level + trend * Decimal(str(steps))


class MeanReversionPredictor:
    """Mean reversion based prediction."""

    def __init__(self, lookback: int = 50, reversion_speed: Decimal = Decimal("0.5")):
        self.lookback = lookback
        self.reversion_speed = reversion_speed
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.lookback:
            self.history[symbol].pop(0)

    def get_mean(self, symbol: str) -> Optional[Decimal]:
        """Get historical mean."""
        prices = self.history.get(symbol, [])
        if not prices:
            return None
        return sum(prices) / Decimal(str(len(prices)))

    def get_deviation(self, symbol: str) -> Optional[Decimal]:
        """Get deviation from mean."""
        prices = self.history.get(symbol, [])
        if not prices:
            return None

        mean = self.get_mean(symbol)
        current = prices[-1]

        return current - mean

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict price reverting to mean."""
        prices = self.history.get(symbol, [])
        if len(prices) < 5:
            return None

        current = prices[-1]
        mean = self.get_mean(symbol)
        deviation = current - mean

        # Predict reversion toward mean
        reversion = deviation * self.reversion_speed * Decimal(str(steps)) / Decimal("10")
        predicted = current - reversion

        return predicted


class MomentumPredictor:
    """Momentum based prediction."""

    def __init__(self, lookback: int = 14):
        self.lookback = lookback
        self.history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)
        if len(self.history[symbol]) > self.lookback + 10:
            self.history[symbol].pop(0)

    def get_momentum(self, symbol: str) -> Optional[Decimal]:
        """Calculate momentum."""
        prices = self.history.get(symbol, [])
        if len(prices) < self.lookback:
            return None

        current = prices[-1]
        past = prices[-self.lookback]

        if past == 0:
            return None

        return (current - past) / past * Decimal("100")

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict based on momentum continuation."""
        prices = self.history.get(symbol, [])
        if len(prices) < self.lookback:
            return None

        momentum = self.get_momentum(symbol)
        if momentum is None:
            return None

        current = prices[-1]
        # Project momentum forward, with decay
        decay = Decimal("0.8") ** Decimal(str(steps))
        change = momentum / Decimal("100") * current * decay * Decimal(str(steps)) / Decimal(str(self.lookback))

        return current + change


class EnsemblePredictor:
    """Ensemble of multiple predictors."""

    def __init__(self):
        self.predictors: list[tuple[Any, Decimal]] = []  # (predictor, weight)
        self.history: dict[str, list[Decimal]] = {}

    def add_predictor(self, predictor: Any, weight: Decimal = Decimal("1")):
        """Add predictor with weight."""
        self.predictors.append((predictor, weight))

    def update(self, symbol: str, price: Decimal):
        """Update all predictors."""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(price)

        for predictor, _ in self.predictors:
            predictor.update(symbol, price)

    def predict(self, symbol: str, steps: int = 1) -> Optional[Decimal]:
        """Predict using weighted average of all predictors."""
        predictions = []
        total_weight = Decimal("0")

        for predictor, weight in self.predictors:
            pred = predictor.predict(symbol, steps)
            if pred is not None:
                predictions.append((pred, weight))
                total_weight += weight

        if not predictions or total_weight == 0:
            return None

        weighted_sum = sum(p * w for p, w in predictions)
        return weighted_sum / total_weight


class ModelEvaluator:
    """Evaluate prediction model performance."""

    def __init__(self):
        self.predictions: dict[str, list[tuple[Decimal, Decimal]]] = {}  # (predicted, actual)

    def add_result(self, symbol: str, predicted: Decimal, actual: Decimal):
        """Add prediction result."""
        if symbol not in self.predictions:
            self.predictions[symbol] = []
        self.predictions[symbol].append((predicted, actual))

    def calculate_metrics(self, symbol: str, model: PredictionModel) -> Optional[ModelMetrics]:
        """Calculate model metrics."""
        results = self.predictions.get(symbol, [])
        if len(results) < 5:
            return None

        n = len(results)
        errors = []
        squared_errors = []
        abs_pct_errors = []
        correct_direction = 0

        for i, (pred, actual) in enumerate(results):
            error = float(actual - pred)
            errors.append(error)
            squared_errors.append(error ** 2)

            if actual != 0:
                abs_pct_errors.append(abs(error / float(actual)) * 100)

            # Check direction prediction
            if i > 0:
                prev_actual = float(results[i-1][1])
                actual_direction = float(actual) - prev_actual
                pred_direction = float(pred) - prev_actual
                if (actual_direction > 0 and pred_direction > 0) or \
                   (actual_direction < 0 and pred_direction < 0) or \
                   (actual_direction == 0 and pred_direction == 0):
                    correct_direction += 1

        mae = sum(abs(e) for e in errors) / n
        mse = sum(squared_errors) / n
        rmse = math.sqrt(mse)
        mape = sum(abs_pct_errors) / len(abs_pct_errors) if abs_pct_errors else 0

        # R-squared
        actual_values = [float(a) for _, a in results]
        actual_mean = sum(actual_values) / n
        ss_tot = sum((a - actual_mean) ** 2 for a in actual_values)
        ss_res = sum(squared_errors)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        accuracy = correct_direction / (n - 1) * 100 if n > 1 else 0

        return ModelMetrics(
            model=model,
            mae=Decimal(str(mae)),
            mse=Decimal(str(mse)),
            rmse=Decimal(str(rmse)),
            mape=Decimal(str(mape)),
            r_squared=Decimal(str(r_squared)),
            accuracy=Decimal(str(accuracy)),
            total_predictions=n,
            correct_predictions=correct_direction
        )


class PricePredictor:
    """Main price prediction engine."""

    def __init__(self):
        self.linear = LinearRegressionPredictor()
        self.ma = MovingAveragePredictor()
        self.exp_smooth = ExponentialSmoothingPredictor()
        self.mean_rev = MeanReversionPredictor()
        self.momentum = MomentumPredictor()
        self.ensemble = EnsemblePredictor()
        self.evaluator = ModelEvaluator()
        self.callbacks: dict[str, list[Callable]] = {
            "on_prediction": []
        }

        # Initialize ensemble
        self.ensemble.add_predictor(self.linear, Decimal("1"))
        self.ensemble.add_predictor(self.ma, Decimal("1"))
        self.ensemble.add_predictor(self.exp_smooth, Decimal("1.2"))
        self.ensemble.add_predictor(self.momentum, Decimal("0.8"))

    def register_callback(self, event: str, callback: Callable):
        """Register callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def update(self, symbol: str, price: Decimal):
        """Update all models with new price."""
        self.linear.update(symbol, price)
        self.ma.update(symbol, price)
        self.exp_smooth.update(symbol, price)
        self.mean_rev.update(symbol, price)
        self.momentum.update(symbol, price)
        self.ensemble.update(symbol, price)

    def predict(
        self,
        symbol: str,
        model: PredictionModel = PredictionModel.ENSEMBLE,
        timeframe: PredictionTimeframe = PredictionTimeframe.HOUR_1
    ) -> Optional[Prediction]:
        """Make price prediction."""
        # Get steps based on timeframe
        steps = self._timeframe_to_steps(timeframe)

        # Get current price
        history = self.linear.history.get(symbol, [])
        if not history:
            return None
        current_price = history[-1]

        # Get prediction based on model
        predicted_price = self._get_prediction(symbol, model, steps)
        if predicted_price is None:
            return None

        # Calculate change
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * Decimal("100") if current_price != 0 else Decimal("0")

        # Determine direction
        direction = self._get_direction(price_change_pct)

        # Calculate confidence
        confidence_score = self._calculate_confidence(symbol, model)
        confidence = self._score_to_confidence(confidence_score)

        prediction = Prediction(
            symbol=symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            direction=direction,
            confidence=confidence,
            confidence_score=confidence_score,
            timeframe=timeframe,
            model=model
        )

        for cb in self.callbacks["on_prediction"]:
            cb(prediction)

        return prediction

    def predict_range(
        self,
        symbol: str,
        model: PredictionModel = PredictionModel.ENSEMBLE,
        timeframe: PredictionTimeframe = PredictionTimeframe.HOUR_1,
        confidence_interval: Decimal = Decimal("95")
    ) -> Optional[PredictionRange]:
        """Predict price range with confidence interval."""
        steps = self._timeframe_to_steps(timeframe)

        history = self.linear.history.get(symbol, [])
        if len(history) < 10:
            return None

        current_price = history[-1]
        predicted_mid = self._get_prediction(symbol, model, steps)
        if predicted_mid is None:
            return None

        # Calculate volatility for range
        returns = []
        for i in range(1, min(20, len(history))):
            ret = (float(history[i]) - float(history[i-1])) / float(history[i-1])
            returns.append(ret)

        if not returns:
            return None

        volatility = math.sqrt(sum(r**2 for r in returns) / len(returns))

        # Z-score for confidence interval
        if confidence_interval >= Decimal("99"):
            z = Decimal("2.576")
        elif confidence_interval >= Decimal("95"):
            z = Decimal("1.96")
        elif confidence_interval >= Decimal("90"):
            z = Decimal("1.645")
        else:
            z = Decimal("1.28")

        margin = current_price * Decimal(str(volatility)) * z * Decimal(str(math.sqrt(steps)))

        return PredictionRange(
            symbol=symbol,
            current_price=current_price,
            predicted_low=predicted_mid - margin,
            predicted_mid=predicted_mid,
            predicted_high=predicted_mid + margin,
            confidence_interval=confidence_interval,
            timeframe=timeframe,
            model=model
        )

    def _get_prediction(self, symbol: str, model: PredictionModel, steps: int) -> Optional[Decimal]:
        """Get prediction from specified model."""
        if model == PredictionModel.LINEAR_REGRESSION:
            return self.linear.predict(symbol, steps)
        elif model == PredictionModel.MOVING_AVERAGE:
            return self.ma.predict(symbol, steps)
        elif model == PredictionModel.EXPONENTIAL_SMOOTHING:
            return self.exp_smooth.predict(symbol, steps)
        elif model == PredictionModel.MEAN_REVERSION:
            return self.mean_rev.predict(symbol, steps)
        elif model == PredictionModel.MOMENTUM:
            return self.momentum.predict(symbol, steps)
        elif model == PredictionModel.ENSEMBLE:
            return self.ensemble.predict(symbol, steps)
        return None

    def _timeframe_to_steps(self, timeframe: PredictionTimeframe) -> int:
        """Convert timeframe to prediction steps."""
        mapping = {
            PredictionTimeframe.MINUTES_5: 1,
            PredictionTimeframe.MINUTES_15: 3,
            PredictionTimeframe.HOUR_1: 12,
            PredictionTimeframe.HOURS_4: 48,
            PredictionTimeframe.DAY_1: 288,
            PredictionTimeframe.WEEK_1: 2016
        }
        return mapping.get(timeframe, 12)

    def _get_direction(self, change_pct: Decimal) -> TrendDirection:
        """Get trend direction from price change."""
        if change_pct > Decimal("5"):
            return TrendDirection.STRONGLY_BULLISH
        elif change_pct > Decimal("1"):
            return TrendDirection.BULLISH
        elif change_pct < Decimal("-5"):
            return TrendDirection.STRONGLY_BEARISH
        elif change_pct < Decimal("-1"):
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def _calculate_confidence(self, symbol: str, model: PredictionModel) -> Decimal:
        """Calculate prediction confidence score."""
        # Base confidence on data availability
        history = self.linear.history.get(symbol, [])
        data_score = min(Decimal("40"), Decimal(str(len(history))) / Decimal("50") * Decimal("40"))

        # Model agreement score
        predictions = []
        for m in [self.linear, self.ma, self.exp_smooth, self.momentum]:
            pred = m.predict(symbol, 1)
            if pred:
                predictions.append(pred)

        if len(predictions) >= 2:
            avg = sum(predictions) / Decimal(str(len(predictions)))
            if avg != 0:
                variance = sum((p - avg) ** 2 for p in predictions) / Decimal(str(len(predictions)))
                cv = Decimal(str(math.sqrt(float(variance)))) / avg * Decimal("100")
                agreement_score = max(Decimal("0"), Decimal("30") - cv)
            else:
                agreement_score = Decimal("15")
        else:
            agreement_score = Decimal("10")

        # Trend consistency score
        trend_score = Decimal("20")  # Base

        return min(Decimal("100"), data_score + agreement_score + trend_score)

    def _score_to_confidence(self, score: Decimal) -> ConfidenceLevel:
        """Convert score to confidence level."""
        if score >= Decimal("80"):
            return ConfidenceLevel.VERY_HIGH
        elif score >= Decimal("60"):
            return ConfidenceLevel.HIGH
        elif score >= Decimal("40"):
            return ConfidenceLevel.MEDIUM
        elif score >= Decimal("20"):
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    def get_model_performance(self, symbol: str) -> dict[str, ModelMetrics]:
        """Get performance metrics for all models."""
        metrics = {}
        for model in PredictionModel:
            m = self.evaluator.calculate_metrics(symbol, model)
            if m:
                metrics[model.value] = m
        return metrics


class PriceForecast:
    """Multi-step price forecasting."""

    def __init__(self, predictor: PricePredictor):
        self.predictor = predictor

    def forecast(
        self,
        symbol: str,
        periods: int = 10,
        model: PredictionModel = PredictionModel.ENSEMBLE
    ) -> list[Prediction]:
        """Generate multi-step forecast."""
        forecasts = []

        for i in range(1, periods + 1):
            # Adjust timeframe based on period
            if i <= 3:
                timeframe = PredictionTimeframe.MINUTES_15
            elif i <= 12:
                timeframe = PredictionTimeframe.HOUR_1
            else:
                timeframe = PredictionTimeframe.HOURS_4

            pred = self.predictor.predict(symbol, model, timeframe)
            if pred:
                forecasts.append(pred)

        return forecasts


# Global instance
_predictor: Optional[PricePredictor] = None


def get_predictor() -> PricePredictor:
    """Get global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor


def set_predictor(predictor: PricePredictor):
    """Set global predictor instance."""
    global _predictor
    _predictor = predictor
