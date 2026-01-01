"""
Revenue Forecasting Engine.
Predicts future revenue using multiple forecasting models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import statistics

from src.utils.structured_logging import get_logger

logger = get_logger("revenue_forecast")


class ForecastModel(str, Enum):
    """Available forecasting models."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL = "seasonal"
    ENSEMBLE = "ensemble"


class ForecastPeriod(str, Enum):
    """Forecast time periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class DataPoint:
    """Single revenue data point."""
    date: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastPoint:
    """Single forecast point."""
    date: datetime
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.95


@dataclass
class ForecastResult:
    """Complete forecast result."""
    business_id: str
    model: ForecastModel
    period: ForecastPeriod
    forecast_points: List[ForecastPoint]
    historical_data_points: int
    accuracy_score: Optional[float] = None
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    trend: str = "stable"  # "up", "down", "stable"
    seasonality_detected: bool = False
    confidence_interval: float = 0.95
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_forecasted(self) -> float:
        """Sum of all forecasted values."""
        return sum(fp.predicted_value for fp in self.forecast_points)
    
    @property
    def average_forecasted(self) -> float:
        """Average forecasted value."""
        if not self.forecast_points:
            return 0.0
        return self.total_forecasted / len(self.forecast_points)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "model": self.model.value,
            "period": self.period.value,
            "forecast_points": [
                {
                    "date": fp.date.isoformat(),
                    "predicted": round(fp.predicted_value, 2),
                    "lower": round(fp.lower_bound, 2),
                    "upper": round(fp.upper_bound, 2),
                }
                for fp in self.forecast_points
            ],
            "summary": {
                "total": round(self.total_forecasted, 2),
                "average": round(self.average_forecasted, 2),
                "trend": self.trend,
                "accuracy": round(self.accuracy_score, 2) if self.accuracy_score else None,
                "mape": round(self.mape, 2) if self.mape else None,
            },
            "metadata": {
                "historical_points": self.historical_data_points,
                "seasonality_detected": self.seasonality_detected,
                "confidence": self.confidence_interval,
                "generated_at": self.generated_at.isoformat(),
            },
        }


class SimpleForecaster:
    """Simple average forecasting."""
    
    @staticmethod
    def forecast(
        data: List[float],
        periods: int,
        confidence: float = 0.95,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast using simple average."""
        if not data:
            return [], []
        
        avg = statistics.mean(data)
        std = statistics.stdev(data) if len(data) > 1 else avg * 0.1
        
        # Z-score for confidence interval
        z = 1.96 if confidence == 0.95 else 1.645
        
        predictions = [avg] * periods
        bounds = [(avg - z * std, avg + z * std)] * periods
        
        return predictions, bounds


class WeightedForecaster:
    """Weighted average with more weight on recent data."""
    
    @staticmethod
    def forecast(
        data: List[float],
        periods: int,
        decay: float = 0.9,
        confidence: float = 0.95,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast with exponential weights."""
        if not data:
            return [], []
        
        # Calculate weights (most recent = highest weight)
        weights = [decay ** (len(data) - 1 - i) for i in range(len(data))]
        total_weight = sum(weights)
        
        weighted_avg = sum(d * w for d, w in zip(data, weights)) / total_weight
        
        # Weighted variance
        variance = sum(w * (d - weighted_avg) ** 2 for d, w in zip(data, weights)) / total_weight
        std = math.sqrt(variance)
        
        z = 1.96 if confidence == 0.95 else 1.645
        
        predictions = [weighted_avg] * periods
        bounds = [(weighted_avg - z * std, weighted_avg + z * std)] * periods
        
        return predictions, bounds


class LinearRegressionForecaster:
    """Linear regression trend forecasting."""
    
    @staticmethod
    def forecast(
        data: List[float],
        periods: int,
        confidence: float = 0.95,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast using linear regression."""
        if len(data) < 2:
            return SimpleForecaster.forecast(data, periods, confidence)
        
        n = len(data)
        x = list(range(n))
        
        # Calculate slope and intercept
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(data)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, data))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        if denominator == 0:
            return SimpleForecaster.forecast(data, periods, confidence)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate standard error
        predictions_historical = [slope * xi + intercept for xi in x]
        residuals = [y - p for y, p in zip(data, predictions_historical)]
        std_error = math.sqrt(sum(r ** 2 for r in residuals) / (n - 2)) if n > 2 else 0
        
        z = 1.96 if confidence == 0.95 else 1.645
        
        # Forecast future periods
        predictions = []
        bounds = []
        
        for i in range(periods):
            future_x = n + i
            pred = slope * future_x + intercept
            pred = max(0, pred)  # Revenue can't be negative
            
            # Confidence interval widens with distance
            margin = z * std_error * math.sqrt(1 + 1/n + (future_x - x_mean)**2 / denominator)
            
            predictions.append(pred)
            bounds.append((max(0, pred - margin), pred + margin))
        
        return predictions, bounds


class ExponentialSmoothingForecaster:
    """Exponential smoothing (Holt's method)."""
    
    @staticmethod
    def forecast(
        data: List[float],
        periods: int,
        alpha: float = 0.3,
        beta: float = 0.1,
        confidence: float = 0.95,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast using Holt's exponential smoothing."""
        if len(data) < 2:
            return SimpleForecaster.forecast(data, periods, confidence)
        
        # Initialize
        level = data[0]
        trend = data[1] - data[0]
        
        # Smooth the series
        levels = [level]
        trends = [trend]
        
        for i in range(1, len(data)):
            prev_level = level
            level = alpha * data[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            levels.append(level)
            trends.append(trend)
        
        # Calculate residual error
        fitted = [levels[0]]
        for i in range(1, len(data)):
            fitted.append(levels[i-1] + trends[i-1])
        
        residuals = [a - f for a, f in zip(data, fitted)]
        std = statistics.stdev(residuals) if len(residuals) > 1 else abs(level) * 0.1
        
        z = 1.96 if confidence == 0.95 else 1.645
        
        # Forecast
        predictions = []
        bounds = []
        
        current_level = levels[-1]
        current_trend = trends[-1]
        
        for h in range(1, periods + 1):
            pred = max(0, current_level + h * current_trend)
            margin = z * std * math.sqrt(h)
            
            predictions.append(pred)
            bounds.append((max(0, pred - margin), pred + margin))
        
        return predictions, bounds


class SeasonalForecaster:
    """Seasonal decomposition forecasting."""
    
    @staticmethod
    def detect_seasonality(
        data: List[float],
        period: int = 12,
    ) -> Tuple[bool, List[float]]:
        """Detect seasonal patterns."""
        if len(data) < period * 2:
            return False, [1.0] * period
        
        # Calculate seasonal indices
        n = len(data)
        num_complete_periods = n // period
        
        if num_complete_periods < 2:
            return False, [1.0] * period
        
        # Average for each position in the period
        seasonal_sums = [0.0] * period
        seasonal_counts = [0] * period
        
        for i, val in enumerate(data):
            pos = i % period
            seasonal_sums[pos] += val
            seasonal_counts[pos] += 1
        
        seasonal_avgs = [
            s / c if c > 0 else 1.0
            for s, c in zip(seasonal_sums, seasonal_counts)
        ]
        
        # Normalize
        overall_avg = statistics.mean(seasonal_avgs) if seasonal_avgs else 1.0
        seasonal_indices = [s / overall_avg if overall_avg > 0 else 1.0 for s in seasonal_avgs]
        
        # Check if seasonality is significant
        variance = statistics.variance(seasonal_indices) if len(seasonal_indices) > 1 else 0
        is_seasonal = variance > 0.01
        
        return is_seasonal, seasonal_indices
    
    @staticmethod
    def forecast(
        data: List[float],
        periods: int,
        season_length: int = 12,
        confidence: float = 0.95,
    ) -> Tuple[List[float], List[Tuple[float, float]], bool]:
        """Generate seasonal forecast."""
        is_seasonal, seasonal_indices = SeasonalForecaster.detect_seasonality(
            data, season_length
        )
        
        if not is_seasonal:
            preds, bounds = ExponentialSmoothingForecaster.forecast(data, periods, confidence=confidence)
            return preds, bounds, False
        
        # Deseasonalize
        deseasonalized = []
        for i, val in enumerate(data):
            idx = i % season_length
            deseasonalized.append(val / seasonal_indices[idx] if seasonal_indices[idx] > 0 else val)
        
        # Forecast deseasonalized
        base_preds, base_bounds = ExponentialSmoothingForecaster.forecast(
            deseasonalized, periods, confidence=confidence
        )
        
        # Reseasonalize
        predictions = []
        bounds = []
        
        current_pos = len(data) % season_length
        
        for i in range(periods):
            idx = (current_pos + i) % season_length
            seasonal_factor = seasonal_indices[idx]
            
            pred = base_preds[i] * seasonal_factor
            lower = base_bounds[i][0] * seasonal_factor
            upper = base_bounds[i][1] * seasonal_factor
            
            predictions.append(max(0, pred))
            bounds.append((max(0, lower), upper))
        
        return predictions, bounds, True


class RevenueForecastingEngine:
    """
    Revenue forecasting with multiple models and ensemble methods.
    
    Features:
    - Multiple forecasting models
    - Automatic model selection
    - Ensemble forecasting
    - Confidence intervals
    - Seasonality detection
    """
    
    def __init__(self):
        self._cached_forecasts: Dict[str, ForecastResult] = {}
        self._model_performance: Dict[str, Dict[ForecastModel, float]] = {}
    
    async def forecast(
        self,
        business_id: str,
        historical_data: List[DataPoint],
        periods: int = 12,
        period_type: ForecastPeriod = ForecastPeriod.MONTHLY,
        model: Optional[ForecastModel] = None,
        confidence: float = 0.95,
    ) -> ForecastResult:
        """
        Generate revenue forecast.
        
        Args:
            business_id: Business identifier
            historical_data: Historical revenue data points
            periods: Number of periods to forecast
            period_type: Type of period (daily, weekly, monthly)
            model: Specific model to use (auto-select if None)
            confidence: Confidence interval level
            
        Returns:
            Complete forecast result
        """
        if not historical_data:
            logger.warning(f"No historical data for {business_id}")
            return self._empty_forecast(business_id, periods, period_type, confidence)
        
        # Sort by date
        sorted_data = sorted(historical_data, key=lambda x: x.date)
        values = [dp.value for dp in sorted_data]
        
        # Auto-select model if not specified
        if model is None:
            model = self._select_best_model(values)
        
        logger.info(
            f"Forecasting {periods} {period_type.value} periods for {business_id} using {model.value}"
        )
        
        # Generate forecast based on model
        if model == ForecastModel.ENSEMBLE:
            return await self._ensemble_forecast(
                business_id, values, sorted_data[-1].date,
                periods, period_type, confidence
            )
        
        predictions, bounds, is_seasonal = self._run_model(
            model, values, periods, confidence
        )
        
        # Create forecast points
        forecast_points = self._create_forecast_points(
            sorted_data[-1].date, predictions, bounds, period_type, confidence
        )
        
        # Calculate accuracy if we have enough data
        accuracy, mape = self._calculate_model_accuracy(model, values)
        
        # Detect trend
        trend = self._detect_trend(values)
        
        result = ForecastResult(
            business_id=business_id,
            model=model,
            period=period_type,
            forecast_points=forecast_points,
            historical_data_points=len(values),
            accuracy_score=accuracy,
            mape=mape,
            trend=trend,
            seasonality_detected=is_seasonal,
            confidence_interval=confidence,
        )
        
        # Cache result
        cache_key = f"{business_id}:{period_type.value}:{model.value}"
        self._cached_forecasts[cache_key] = result
        
        return result
    
    def _select_best_model(self, values: List[float]) -> ForecastModel:
        """Auto-select the best model based on data characteristics."""
        n = len(values)
        
        if n < 3:
            return ForecastModel.SIMPLE_AVERAGE
        elif n < 12:
            return ForecastModel.WEIGHTED_AVERAGE
        elif n < 24:
            # Check for trend
            trend_strength = self._calculate_trend_strength(values)
            if abs(trend_strength) > 0.3:
                return ForecastModel.LINEAR_REGRESSION
            return ForecastModel.EXPONENTIAL_SMOOTHING
        else:
            # Enough data for seasonal or ensemble
            is_seasonal, _ = SeasonalForecaster.detect_seasonality(values)
            if is_seasonal:
                return ForecastModel.SEASONAL
            return ForecastModel.ENSEMBLE
    
    def _run_model(
        self,
        model: ForecastModel,
        values: List[float],
        periods: int,
        confidence: float,
    ) -> Tuple[List[float], List[Tuple[float, float]], bool]:
        """Run the specified forecasting model."""
        is_seasonal = False
        
        if model == ForecastModel.SIMPLE_AVERAGE:
            preds, bounds = SimpleForecaster.forecast(values, periods, confidence)
        elif model == ForecastModel.WEIGHTED_AVERAGE:
            preds, bounds = WeightedForecaster.forecast(values, periods, confidence=confidence)
        elif model == ForecastModel.LINEAR_REGRESSION:
            preds, bounds = LinearRegressionForecaster.forecast(values, periods, confidence)
        elif model == ForecastModel.EXPONENTIAL_SMOOTHING:
            preds, bounds = ExponentialSmoothingForecaster.forecast(values, periods, confidence=confidence)
        elif model == ForecastModel.SEASONAL:
            preds, bounds, is_seasonal = SeasonalForecaster.forecast(values, periods, confidence=confidence)
        else:
            preds, bounds = SimpleForecaster.forecast(values, periods, confidence)
        
        return preds, bounds, is_seasonal
    
    async def _ensemble_forecast(
        self,
        business_id: str,
        values: List[float],
        last_date: datetime,
        periods: int,
        period_type: ForecastPeriod,
        confidence: float,
    ) -> ForecastResult:
        """Combine multiple models for better accuracy."""
        models = [
            ForecastModel.WEIGHTED_AVERAGE,
            ForecastModel.LINEAR_REGRESSION,
            ForecastModel.EXPONENTIAL_SMOOTHING,
        ]
        
        if len(values) >= 24:
            models.append(ForecastModel.SEASONAL)
        
        all_predictions: List[List[float]] = []
        all_bounds: List[List[Tuple[float, float]]] = []
        is_seasonal = False
        
        for model in models:
            preds, bounds, seasonal = self._run_model(model, values, periods, confidence)
            all_predictions.append(preds)
            all_bounds.append(bounds)
            if seasonal:
                is_seasonal = True
        
        # Average predictions
        ensemble_predictions = []
        ensemble_bounds = []
        
        for i in range(periods):
            period_preds = [p[i] for p in all_predictions]
            period_bounds = [b[i] for b in all_bounds]
            
            avg_pred = statistics.mean(period_preds)
            avg_lower = statistics.mean(b[0] for b in period_bounds)
            avg_upper = statistics.mean(b[1] for b in period_bounds)
            
            ensemble_predictions.append(avg_pred)
            ensemble_bounds.append((avg_lower, avg_upper))
        
        forecast_points = self._create_forecast_points(
            last_date, ensemble_predictions, ensemble_bounds, period_type, confidence
        )
        
        accuracy, mape = self._calculate_model_accuracy(ForecastModel.ENSEMBLE, values)
        trend = self._detect_trend(values)
        
        return ForecastResult(
            business_id=business_id,
            model=ForecastModel.ENSEMBLE,
            period=period_type,
            forecast_points=forecast_points,
            historical_data_points=len(values),
            accuracy_score=accuracy,
            mape=mape,
            trend=trend,
            seasonality_detected=is_seasonal,
            confidence_interval=confidence,
        )
    
    def _create_forecast_points(
        self,
        last_date: datetime,
        predictions: List[float],
        bounds: List[Tuple[float, float]],
        period_type: ForecastPeriod,
        confidence: float,
    ) -> List[ForecastPoint]:
        """Create forecast point objects."""
        delta = self._get_period_delta(period_type)
        
        points = []
        for i, (pred, (lower, upper)) in enumerate(zip(predictions, bounds)):
            forecast_date = last_date + delta * (i + 1)
            points.append(ForecastPoint(
                date=forecast_date,
                predicted_value=pred,
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
            ))
        
        return points
    
    def _get_period_delta(self, period_type: ForecastPeriod) -> timedelta:
        """Get timedelta for period type."""
        if period_type == ForecastPeriod.DAILY:
            return timedelta(days=1)
        elif period_type == ForecastPeriod.WEEKLY:
            return timedelta(weeks=1)
        elif period_type == ForecastPeriod.MONTHLY:
            return timedelta(days=30)
        elif period_type == ForecastPeriod.QUARTERLY:
            return timedelta(days=90)
        return timedelta(days=30)
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate linear trend strength (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in values))
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
        
        return numerator / (denom_x * denom_y)
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detect overall trend direction."""
        strength = self._calculate_trend_strength(values)
        
        if strength > 0.3:
            return "up"
        elif strength < -0.3:
            return "down"
        return "stable"
    
    def _calculate_model_accuracy(
        self,
        model: ForecastModel,
        values: List[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate model accuracy using backtesting."""
        if len(values) < 6:
            return None, None
        
        # Use last 20% as test set
        split = int(len(values) * 0.8)
        train = values[:split]
        test = values[split:]
        
        if len(test) < 1:
            return None, None
        
        # Generate forecast for test period
        preds, _, _ = self._run_model(model, train, len(test), 0.95)
        
        # Calculate MAPE
        errors = []
        for actual, predicted in zip(test, preds):
            if actual != 0:
                errors.append(abs((actual - predicted) / actual))
        
        if not errors:
            return None, None
        
        mape = statistics.mean(errors) * 100
        accuracy = max(0, 100 - mape)
        
        return accuracy, mape
    
    def _empty_forecast(
        self,
        business_id: str,
        periods: int,
        period_type: ForecastPeriod,
        confidence: float,
    ) -> ForecastResult:
        """Return empty forecast when no data available."""
        return ForecastResult(
            business_id=business_id,
            model=ForecastModel.SIMPLE_AVERAGE,
            period=period_type,
            forecast_points=[],
            historical_data_points=0,
            trend="stable",
            confidence_interval=confidence,
        )
    
    async def compare_models(
        self,
        business_id: str,
        historical_data: List[DataPoint],
        periods: int = 12,
    ) -> Dict[ForecastModel, Dict[str, Any]]:
        """Compare all models and their accuracy."""
        values = [dp.value for dp in sorted(historical_data, key=lambda x: x.date)]
        
        results = {}
        
        for model in ForecastModel:
            if model == ForecastModel.ENSEMBLE:
                continue
            
            accuracy, mape = self._calculate_model_accuracy(model, values)
            preds, _, _ = self._run_model(model, values, periods, 0.95)
            
            results[model] = {
                "accuracy": round(accuracy, 2) if accuracy else None,
                "mape": round(mape, 2) if mape else None,
                "total_forecast": round(sum(preds), 2),
                "avg_forecast": round(statistics.mean(preds), 2) if preds else None,
            }
        
        return results


# Global engine instance
forecasting_engine = RevenueForecastingEngine()


def get_forecasting_engine() -> RevenueForecastingEngine:
    """Get the global forecasting engine instance."""
    return forecasting_engine
