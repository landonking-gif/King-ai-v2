"""
Predictive Analytics Engine.

Provides time-series forecasting and risk prediction using Prophet.
Enables proactive decision-making by forecasting:
- Revenue and profit trends
- Market conditions
- Integration failure probability
- Business health indicators
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.utils.structured_logging import get_logger

logger = get_logger("predictive_analytics")


class ForecastMetric(str, Enum):
    """Metrics that can be forecasted."""
    REVENUE = "revenue"
    PROFIT = "profit"
    TRAFFIC = "traffic"
    CONVERSION_RATE = "conversion_rate"
    CUSTOMER_COUNT = "customer_count"
    ORDER_COUNT = "order_count"
    AVG_ORDER_VALUE = "avg_order_value"
    CHURN_RATE = "churn_rate"
    CAC = "customer_acquisition_cost"
    LTV = "lifetime_value"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ForecastPoint:
    """A single forecast data point."""
    timestamp: datetime
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.95


@dataclass
class ForecastResult:
    """Complete forecast result."""
    metric: ForecastMetric
    current_value: float
    forecast_points: List[ForecastPoint]
    trend: str  # "up", "down", "stable"
    trend_strength: float  # 0-1
    seasonality: Optional[Dict[str, Any]] = None
    changepoints: List[datetime] = field(default_factory=list)
    model_accuracy: Optional[float] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def forecast_end_value(self) -> float:
        """Get the final forecasted value."""
        if self.forecast_points:
            return self.forecast_points[-1].predicted_value
        return self.current_value
    
    @property
    def expected_change_pct(self) -> float:
        """Calculate expected percentage change."""
        if self.current_value == 0:
            return 0
        return ((self.forecast_end_value - self.current_value) / self.current_value) * 100


@dataclass
class RiskForecast:
    """Risk forecast for a specific area."""
    category: str
    current_risk: RiskLevel
    predicted_risk: RiskLevel
    probability: float
    contributing_factors: List[str]
    mitigation_suggestions: List[str]
    forecast_horizon_days: int
    confidence: float


@dataclass
class MarketConditionForecast:
    """Market condition forecast."""
    sector: str
    current_sentiment: str
    predicted_sentiment: str
    volatility_index: float
    key_events: List[Dict[str, Any]]
    competitor_activity: Optional[str] = None
    recommendation: Optional[str] = None


class PredictiveAnalyticsEngine:
    """
    Time-series forecasting and risk prediction engine.
    
    Uses Prophet for forecasting when available, with fallback
    to simpler statistical methods.
    
    Features:
    - Revenue/profit forecasting
    - Trend detection
    - Seasonality analysis
    - Risk prediction
    - Anomaly detection
    """
    
    def __init__(self):
        self._prophet_available = False
        self._check_prophet()
        self._historical_data: Dict[str, List[Tuple[datetime, float]]] = {}
        self._forecast_cache: Dict[str, ForecastResult] = {}
    
    def _check_prophet(self):
        """Check if Prophet is available."""
        try:
            from prophet import Prophet
            self._prophet_available = True
            logger.info("Prophet forecasting library available")
        except ImportError:
            logger.warning(
                "Prophet not installed. Using fallback forecasting. "
                "Install with: pip install prophet"
            )
    
    def add_historical_data(
        self,
        metric: ForecastMetric,
        data: List[Tuple[datetime, float]]
    ):
        """
        Add historical data for a metric.
        
        Args:
            metric: The metric type
            data: List of (timestamp, value) tuples
        """
        key = metric.value
        if key not in self._historical_data:
            self._historical_data[key] = []
        
        self._historical_data[key].extend(data)
        # Sort by timestamp
        self._historical_data[key].sort(key=lambda x: x[0])
        
        # Invalidate cache
        if key in self._forecast_cache:
            del self._forecast_cache[key]
        
        logger.info(
            f"Added {len(data)} data points for {metric.value}",
            total_points=len(self._historical_data[key])
        )
    
    async def forecast(
        self,
        metric: ForecastMetric,
        horizon_days: int = 30,
        include_seasonality: bool = True
    ) -> ForecastResult:
        """
        Generate forecast for a metric.
        
        Args:
            metric: Metric to forecast
            horizon_days: Days to forecast ahead
            include_seasonality: Whether to model seasonality
            
        Returns:
            ForecastResult with predictions
        """
        key = metric.value
        
        # Check cache
        cache_key = f"{key}_{horizon_days}"
        if cache_key in self._forecast_cache:
            cached = self._forecast_cache[cache_key]
            age = (datetime.utcnow() - cached.generated_at).total_seconds()
            if age < 3600:  # Cache for 1 hour
                return cached
        
        data = self._historical_data.get(key, [])
        
        if len(data) < 7:
            # Not enough data - return simple projection
            return self._simple_forecast(metric, data, horizon_days)
        
        if self._prophet_available:
            result = await self._prophet_forecast(metric, data, horizon_days, include_seasonality)
        else:
            result = self._statistical_forecast(metric, data, horizon_days)
        
        # Cache result
        self._forecast_cache[cache_key] = result
        
        return result
    
    async def _prophet_forecast(
        self,
        metric: ForecastMetric,
        data: List[Tuple[datetime, float]],
        horizon_days: int,
        include_seasonality: bool
    ) -> ForecastResult:
        """Generate forecast using Prophet."""
        from prophet import Prophet
        import pandas as pd
        
        # Prepare data for Prophet
        df = pd.DataFrame(data, columns=['ds', 'y'])
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=include_seasonality,
            weekly_seasonality=include_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Fit model
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon_days)
        
        # Predict
        forecast = model.predict(future)
        
        # Extract forecast points (only future dates)
        last_historical = data[-1][0]
        forecast_points = []
        
        for _, row in forecast.iterrows():
            if row['ds'] > last_historical:
                forecast_points.append(ForecastPoint(
                    timestamp=row['ds'].to_pydatetime(),
                    predicted_value=float(row['yhat']),
                    lower_bound=float(row['yhat_lower']),
                    upper_bound=float(row['yhat_upper']),
                    confidence=0.95
                ))
        
        # Determine trend
        if len(forecast_points) >= 2:
            start_val = forecast_points[0].predicted_value
            end_val = forecast_points[-1].predicted_value
            change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
            
            if change_pct > 5:
                trend = "up"
                trend_strength = min(abs(change_pct) / 20, 1.0)
            elif change_pct < -5:
                trend = "down"
                trend_strength = min(abs(change_pct) / 20, 1.0)
            else:
                trend = "stable"
                trend_strength = 0.2
        else:
            trend = "stable"
            trend_strength = 0.0
        
        # Extract seasonality info
        seasonality = None
        if include_seasonality:
            seasonality = {
                "weekly": True,
                "yearly": True
            }
        
        # Get changepoints
        changepoints = [cp.to_pydatetime() for cp in model.changepoints]
        
        return ForecastResult(
            metric=metric,
            current_value=data[-1][1],
            forecast_points=forecast_points,
            trend=trend,
            trend_strength=trend_strength,
            seasonality=seasonality,
            changepoints=changepoints,
            model_accuracy=0.85  # Would calculate from cross-validation
        )
    
    def _statistical_forecast(
        self,
        metric: ForecastMetric,
        data: List[Tuple[datetime, float]],
        horizon_days: int
    ) -> ForecastResult:
        """Fallback statistical forecasting without Prophet."""
        values = [v for _, v in data]
        
        # Calculate trend using simple linear regression
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Linear regression coefficients
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Calculate standard error
        y_pred = slope * x + intercept
        residuals = y - y_pred
        std_error = np.std(residuals)
        
        # Generate forecast points
        forecast_points = []
        last_date = data[-1][0]
        
        for i in range(1, horizon_days + 1):
            future_x = n + i
            predicted = slope * future_x + intercept
            
            # Confidence interval (95%)
            margin = 1.96 * std_error * np.sqrt(1 + 1/n + (future_x - x_mean)**2 / denominator)
            
            forecast_points.append(ForecastPoint(
                timestamp=last_date + timedelta(days=i),
                predicted_value=float(predicted),
                lower_bound=float(predicted - margin),
                upper_bound=float(predicted + margin),
                confidence=0.95
            ))
        
        # Determine trend
        if slope > 0.01 * y_mean:
            trend = "up"
            trend_strength = min(abs(slope) / y_mean, 1.0)
        elif slope < -0.01 * y_mean:
            trend = "down"
            trend_strength = min(abs(slope) / y_mean, 1.0)
        else:
            trend = "stable"
            trend_strength = 0.1
        
        return ForecastResult(
            metric=metric,
            current_value=data[-1][1],
            forecast_points=forecast_points,
            trend=trend,
            trend_strength=trend_strength,
            model_accuracy=0.7  # Lower accuracy for simple model
        )
    
    def _simple_forecast(
        self,
        metric: ForecastMetric,
        data: List[Tuple[datetime, float]],
        horizon_days: int
    ) -> ForecastResult:
        """Simple forecast when data is insufficient."""
        current_value = data[-1][1] if data else 0
        
        # Just project flat with wide confidence interval
        forecast_points = []
        last_date = data[-1][0] if data else datetime.utcnow()
        
        for i in range(1, horizon_days + 1):
            margin = current_value * 0.2  # 20% margin
            forecast_points.append(ForecastPoint(
                timestamp=last_date + timedelta(days=i),
                predicted_value=current_value,
                lower_bound=current_value - margin,
                upper_bound=current_value + margin,
                confidence=0.5
            ))
        
        return ForecastResult(
            metric=metric,
            current_value=current_value,
            forecast_points=forecast_points,
            trend="stable",
            trend_strength=0.0,
            model_accuracy=0.3
        )
    
    async def predict_risk(
        self,
        category: str,
        historical_incidents: List[Dict[str, Any]] = None,
        current_metrics: Dict[str, float] = None
    ) -> RiskForecast:
        """
        Predict risk level for a category.
        
        Args:
            category: Risk category (e.g., "payment_failure", "inventory", "market")
            historical_incidents: Past incidents for this category
            current_metrics: Current relevant metrics
            
        Returns:
            RiskForecast with prediction and recommendations
        """
        historical_incidents = historical_incidents or []
        current_metrics = current_metrics or {}
        
        # Calculate base risk from historical data
        recent_incidents = [
            i for i in historical_incidents
            if (datetime.utcnow() - i.get("timestamp", datetime.utcnow())).days < 30
        ]
        
        incident_rate = len(recent_incidents) / 30  # incidents per day
        
        # Determine current risk level
        if incident_rate >= 0.5:
            current_risk = RiskLevel.CRITICAL
        elif incident_rate >= 0.2:
            current_risk = RiskLevel.HIGH
        elif incident_rate >= 0.05:
            current_risk = RiskLevel.MODERATE
        else:
            current_risk = RiskLevel.LOW
        
        # Predict future risk based on trend
        contributing_factors = []
        mitigation_suggestions = []
        
        # Analyze patterns
        if category == "payment_failure":
            error_rate = current_metrics.get("payment_error_rate", 0)
            if error_rate > 0.05:
                contributing_factors.append(f"High payment error rate: {error_rate*100:.1f}%")
                mitigation_suggestions.append("Review payment provider status")
                mitigation_suggestions.append("Enable PayPal fallback")
        
        elif category == "inventory":
            stock_level = current_metrics.get("stock_level", 100)
            reorder_point = current_metrics.get("reorder_point", 20)
            if stock_level < reorder_point * 1.5:
                contributing_factors.append(f"Low stock level: {stock_level}")
                mitigation_suggestions.append("Initiate reorder from supplier")
        
        elif category == "market":
            volatility = current_metrics.get("price_volatility", 0)
            if volatility > 0.15:
                contributing_factors.append(f"High price volatility: {volatility*100:.1f}%")
                mitigation_suggestions.append("Consider hedging strategies")
                mitigation_suggestions.append("Diversify supplier base")
        
        # Predict future risk (simple decay/growth model)
        if len(contributing_factors) > 2:
            predicted_risk = RiskLevel.CRITICAL
            probability = 0.8
        elif len(contributing_factors) > 0:
            predicted_risk = RiskLevel.HIGH
            probability = 0.6
        elif incident_rate > 0:
            predicted_risk = RiskLevel.MODERATE
            probability = 0.4
        else:
            predicted_risk = RiskLevel.LOW
            probability = 0.2
        
        return RiskForecast(
            category=category,
            current_risk=current_risk,
            predicted_risk=predicted_risk,
            probability=probability,
            contributing_factors=contributing_factors,
            mitigation_suggestions=mitigation_suggestions or ["Continue monitoring"],
            forecast_horizon_days=14,
            confidence=0.75
        )
    
    async def detect_anomalies(
        self,
        metric: ForecastMetric,
        sensitivity: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in historical data.
        
        Args:
            metric: Metric to analyze
            sensitivity: Standard deviations for anomaly threshold
            
        Returns:
            List of detected anomalies
        """
        data = self._historical_data.get(metric.value, [])
        
        if len(data) < 10:
            return []
        
        values = np.array([v for _, v in data])
        timestamps = [t for t, _ in data]
        
        # Calculate rolling statistics
        window = min(7, len(values) // 3)
        
        anomalies = []
        for i in range(window, len(values)):
            window_data = values[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std == 0:
                continue
            
            z_score = abs(values[i] - mean) / std
            
            if z_score > sensitivity:
                anomalies.append({
                    "timestamp": timestamps[i],
                    "value": float(values[i]),
                    "expected": float(mean),
                    "z_score": float(z_score),
                    "direction": "high" if values[i] > mean else "low"
                })
        
        return anomalies
    
    def get_business_health_forecast(
        self,
        business_id: str,
        metrics: Dict[ForecastMetric, float]
    ) -> Dict[str, Any]:
        """
        Generate overall business health forecast.
        
        Args:
            business_id: Business unit ID
            metrics: Current metric values
            
        Returns:
            Health forecast summary
        """
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Revenue trend
        if metrics.get(ForecastMetric.REVENUE, 0) < metrics.get("revenue_target", float("inf")):
            health_score -= 15
            issues.append("Revenue below target")
            recommendations.append("Review pricing strategy")
        
        # Profit margin
        revenue = metrics.get(ForecastMetric.REVENUE, 1)
        profit = metrics.get(ForecastMetric.PROFIT, 0)
        margin = profit / revenue if revenue > 0 else 0
        
        if margin < 0.1:
            health_score -= 20
            issues.append(f"Low profit margin: {margin*100:.1f}%")
            recommendations.append("Reduce costs or increase prices")
        
        # Customer metrics
        churn = metrics.get(ForecastMetric.CHURN_RATE, 0)
        if churn > 0.05:
            health_score -= 10
            issues.append(f"High churn rate: {churn*100:.1f}%")
            recommendations.append("Improve customer retention")
        
        # Determine health status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        elif health_score >= 40:
            status = "at_risk"
        else:
            status = "critical"
        
        return {
            "business_id": business_id,
            "health_score": max(0, health_score),
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "forecast_confidence": 0.75,
            "assessed_at": datetime.utcnow().isoformat()
        }


# Global instance
predictive_engine = PredictiveAnalyticsEngine()


async def run_daily_risk_simulation():
    """
    Scheduled task to run daily risk simulations.
    
    Integrates with scheduler for autonomous risk monitoring.
    """
    logger.info("Running daily risk simulation")
    
    categories = ["payment_failure", "inventory", "market", "integration"]
    results = []
    
    for category in categories:
        forecast = await predictive_engine.predict_risk(category)
        results.append({
            "category": category,
            "current_risk": forecast.current_risk.value,
            "predicted_risk": forecast.predicted_risk.value,
            "factors": forecast.contributing_factors
        })
        
        # Alert on high/critical risk
        if forecast.predicted_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(
                f"Risk alert: {category}",
                current=forecast.current_risk.value,
                predicted=forecast.predicted_risk.value,
                factors=forecast.contributing_factors
            )
    
    return results
