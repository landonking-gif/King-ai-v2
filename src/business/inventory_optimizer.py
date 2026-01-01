"""
Inventory Optimizer.
Optimal stock levels and reorder point calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import statistics

from src.utils.structured_logging import get_logger

logger = get_logger("inventory_optimizer")


class StockStatus(str, Enum):
    """Stock status levels."""
    OVERSTOCK = "overstock"
    OPTIMAL = "optimal"
    LOW = "low"
    CRITICAL = "critical"
    OUT_OF_STOCK = "out_of_stock"


class ReplenishmentStrategy(str, Enum):
    """Replenishment strategies."""
    FIXED_ORDER_QUANTITY = "fixed_order_quantity"  # EOQ model
    FIXED_PERIOD = "fixed_period"  # Periodic review
    MIN_MAX = "min_max"  # Min-max system
    JUST_IN_TIME = "just_in_time"  # JIT


class DemandPattern(str, Enum):
    """Demand pattern types."""
    STEADY = "steady"
    SEASONAL = "seasonal"
    TRENDING = "trending"
    SPORADIC = "sporadic"


@dataclass
class DemandForecast:
    """Demand forecast for a product."""
    product_id: str
    daily_demand: float
    weekly_demand: float
    monthly_demand: float
    demand_std: float
    pattern: DemandPattern
    seasonality_factor: float = 1.0
    trend_factor: float = 0.0
    forecast_accuracy: float = 0.0


@dataclass
class InventoryMetrics:
    """Inventory metrics for a product."""
    product_id: str
    current_stock: int
    on_order: int = 0
    reserved: int = 0
    
    # Costs
    unit_cost: float = 0.0
    holding_cost_rate: float = 0.25  # Annual holding cost as % of unit cost
    ordering_cost: float = 50.0  # Fixed cost per order
    stockout_cost: float = 0.0  # Cost per unit stockout
    
    # Lead time
    lead_time_days: float = 7.0
    lead_time_std: float = 1.0
    
    # Sales data
    sales_history: List[float] = field(default_factory=list)
    
    @property
    def available_stock(self) -> int:
        return max(0, self.current_stock + self.on_order - self.reserved)
    
    @property
    def inventory_value(self) -> float:
        return self.current_stock * self.unit_cost


@dataclass
class ReorderRecommendation:
    """Reorder recommendation for a product."""
    product_id: str
    should_reorder: bool
    order_quantity: int
    reorder_point: int
    safety_stock: int
    days_of_supply: float
    projected_stockout_date: Optional[datetime] = None
    urgency: str = "normal"
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "should_reorder": self.should_reorder,
            "order_quantity": self.order_quantity,
            "reorder_point": self.reorder_point,
            "safety_stock": self.safety_stock,
            "days_of_supply": round(self.days_of_supply, 1),
            "projected_stockout_date": (
                self.projected_stockout_date.isoformat()
                if self.projected_stockout_date else None
            ),
            "urgency": self.urgency,
            "reason": self.reason,
        }


@dataclass
class InventoryAnalysis:
    """Complete inventory analysis result."""
    product_id: str
    status: StockStatus
    metrics: InventoryMetrics
    forecast: DemandForecast
    recommendation: ReorderRecommendation
    optimization_savings: float = 0.0
    turnover_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "status": self.status.value,
            "current_stock": self.metrics.current_stock,
            "available_stock": self.metrics.available_stock,
            "inventory_value": round(self.metrics.inventory_value, 2),
            "daily_demand": round(self.forecast.daily_demand, 1),
            "demand_pattern": self.forecast.pattern.value,
            "turnover_rate": round(self.turnover_rate, 2),
            "recommendation": self.recommendation.to_dict(),
        }


class DemandForecaster:
    """Forecast demand for products."""
    
    def forecast(
        self,
        product_id: str,
        sales_history: List[float],
        periods: int = 30,
    ) -> DemandForecast:
        """
        Forecast demand based on historical sales.
        
        Args:
            product_id: Product identifier
            sales_history: Historical daily sales
            periods: Number of days to forecast
            
        Returns:
            Demand forecast
        """
        if not sales_history:
            return DemandForecast(
                product_id=product_id,
                daily_demand=0,
                weekly_demand=0,
                monthly_demand=0,
                demand_std=0,
                pattern=DemandPattern.STEADY,
            )
        
        # Calculate basic statistics
        daily_demand = statistics.mean(sales_history)
        demand_std = statistics.stdev(sales_history) if len(sales_history) > 1 else 0
        
        # Detect pattern
        pattern = self._detect_pattern(sales_history)
        
        # Calculate seasonality and trend
        seasonality = self._calculate_seasonality(sales_history)
        trend = self._calculate_trend(sales_history)
        
        # Forecast accuracy (simplified)
        if len(sales_history) > 7:
            predicted = [daily_demand] * 7
            actual = sales_history[-7:]
            mape = sum(abs(p - a) / max(a, 0.01) for p, a in zip(predicted, actual)) / 7
            accuracy = max(0, 1 - mape)
        else:
            accuracy = 0.5
        
        return DemandForecast(
            product_id=product_id,
            daily_demand=daily_demand,
            weekly_demand=daily_demand * 7,
            monthly_demand=daily_demand * 30,
            demand_std=demand_std,
            pattern=pattern,
            seasonality_factor=seasonality,
            trend_factor=trend,
            forecast_accuracy=accuracy,
        )
    
    def _detect_pattern(self, sales: List[float]) -> DemandPattern:
        """Detect the demand pattern type."""
        if len(sales) < 14:
            return DemandPattern.STEADY
        
        # Calculate coefficient of variation
        mean = statistics.mean(sales)
        if mean == 0:
            return DemandPattern.SPORADIC
        
        cv = statistics.stdev(sales) / mean if mean > 0 else 0
        
        # Check for trend
        first_half = statistics.mean(sales[:len(sales)//2])
        second_half = statistics.mean(sales[len(sales)//2:])
        trend_ratio = second_half / first_half if first_half > 0 else 1
        
        if abs(trend_ratio - 1) > 0.2:
            return DemandPattern.TRENDING
        
        # Check for seasonality (weekly pattern)
        if len(sales) >= 14:
            week1 = sales[:7]
            week2 = sales[7:14]
            correlation = self._correlation(week1, week2)
            if correlation > 0.7:
                return DemandPattern.SEASONAL
        
        if cv > 1.0:
            return DemandPattern.SPORADIC
        
        return DemandPattern.STEADY
    
    def _calculate_seasonality(self, sales: List[float]) -> float:
        """Calculate seasonality factor."""
        if len(sales) < 7:
            return 1.0
        
        # Weekly seasonality
        weekly_avg = statistics.mean(sales[-7:])
        overall_avg = statistics.mean(sales)
        
        if overall_avg == 0:
            return 1.0
        
        return weekly_avg / overall_avg
    
    def _calculate_trend(self, sales: List[float]) -> float:
        """Calculate trend factor (daily change)."""
        if len(sales) < 7:
            return 0.0
        
        # Simple linear regression slope
        n = len(sales)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(sales)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(sales))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
        
        return numerator / (denom_x * denom_y)


class InventoryOptimizer:
    """
    Inventory Optimization Engine.
    
    Features:
    - Economic Order Quantity (EOQ) calculation
    - Reorder point determination
    - Safety stock optimization
    - Demand forecasting
    - Stock status monitoring
    """
    
    def __init__(self, service_level: float = 0.95):
        """
        Initialize inventory optimizer.
        
        Args:
            service_level: Target service level (probability of no stockout)
        """
        self.service_level = service_level
        self.forecaster = DemandForecaster()
        
        # Z-score for service level
        self.z_score = self._get_z_score(service_level)
    
    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for service level."""
        # Common service level z-scores
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.98: 2.05,
            0.99: 2.33,
        }
        
        # Find closest match
        closest = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores.get(closest, 1.65)
    
    async def analyze_product(
        self,
        metrics: InventoryMetrics,
    ) -> InventoryAnalysis:
        """
        Analyze inventory for a single product.
        
        Args:
            metrics: Product inventory metrics
            
        Returns:
            Complete inventory analysis
        """
        # Forecast demand
        forecast = self.forecaster.forecast(
            metrics.product_id,
            metrics.sales_history,
        )
        
        # Calculate optimal levels
        eoq = self._calculate_eoq(metrics, forecast)
        safety_stock = self._calculate_safety_stock(metrics, forecast)
        reorder_point = self._calculate_reorder_point(metrics, forecast, safety_stock)
        
        # Determine stock status
        status = self._determine_status(metrics, forecast, reorder_point)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            metrics, forecast, eoq, safety_stock, reorder_point, status
        )
        
        # Calculate metrics
        turnover = self._calculate_turnover_rate(metrics, forecast)
        savings = self._calculate_potential_savings(metrics, forecast, eoq)
        
        return InventoryAnalysis(
            product_id=metrics.product_id,
            status=status,
            metrics=metrics,
            forecast=forecast,
            recommendation=recommendation,
            optimization_savings=savings,
            turnover_rate=turnover,
        )
    
    def _calculate_eoq(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
    ) -> int:
        """Calculate Economic Order Quantity."""
        annual_demand = forecast.daily_demand * 365
        
        if annual_demand == 0 or metrics.unit_cost == 0:
            return 0
        
        holding_cost = metrics.unit_cost * metrics.holding_cost_rate
        
        # EOQ formula: sqrt(2 * D * S / H)
        eoq = math.sqrt(
            (2 * annual_demand * metrics.ordering_cost) / holding_cost
        )
        
        return max(1, int(math.ceil(eoq)))
    
    def _calculate_safety_stock(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
    ) -> int:
        """Calculate safety stock level."""
        # Safety stock = Z * sqrt(LT * σ_d^2 + d^2 * σ_LT^2)
        demand_var = forecast.demand_std ** 2
        lead_time_var = metrics.lead_time_std ** 2
        
        variance = (
            metrics.lead_time_days * demand_var +
            (forecast.daily_demand ** 2) * lead_time_var
        )
        
        safety_stock = self.z_score * math.sqrt(variance)
        
        return max(0, int(math.ceil(safety_stock)))
    
    def _calculate_reorder_point(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
        safety_stock: int,
    ) -> int:
        """Calculate reorder point."""
        # ROP = d * LT + SS
        lead_time_demand = forecast.daily_demand * metrics.lead_time_days
        reorder_point = lead_time_demand + safety_stock
        
        return max(0, int(math.ceil(reorder_point)))
    
    def _determine_status(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
        reorder_point: int,
    ) -> StockStatus:
        """Determine current stock status."""
        available = metrics.available_stock
        
        if available == 0:
            return StockStatus.OUT_OF_STOCK
        
        days_of_supply = available / forecast.daily_demand if forecast.daily_demand > 0 else 999
        
        if days_of_supply < metrics.lead_time_days:
            return StockStatus.CRITICAL
        
        if available <= reorder_point:
            return StockStatus.LOW
        
        # Overstock: more than 90 days of supply
        if days_of_supply > 90:
            return StockStatus.OVERSTOCK
        
        return StockStatus.OPTIMAL
    
    def _generate_recommendation(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
        eoq: int,
        safety_stock: int,
        reorder_point: int,
        status: StockStatus,
    ) -> ReorderRecommendation:
        """Generate reorder recommendation."""
        available = metrics.available_stock
        daily_demand = forecast.daily_demand
        
        days_of_supply = available / daily_demand if daily_demand > 0 else 999
        
        # Projected stockout date
        if daily_demand > 0:
            stockout_date = datetime.utcnow() + timedelta(days=days_of_supply)
        else:
            stockout_date = None
        
        # Determine if reorder needed
        should_reorder = available <= reorder_point
        
        # Calculate order quantity
        if should_reorder:
            # Order up to (reorder point + EOQ) - current position
            target = reorder_point + eoq
            order_quantity = max(0, target - available)
        else:
            order_quantity = 0
        
        # Determine urgency
        if status == StockStatus.OUT_OF_STOCK:
            urgency = "critical"
            reason = "Product is out of stock"
        elif status == StockStatus.CRITICAL:
            urgency = "urgent"
            reason = f"Stock will run out in {days_of_supply:.1f} days, before lead time"
        elif status == StockStatus.LOW:
            urgency = "normal"
            reason = "Stock is below reorder point"
        elif status == StockStatus.OVERSTOCK:
            urgency = "low"
            reason = "Stock levels are high, consider promotions"
        else:
            urgency = "none"
            reason = "Inventory levels are optimal"
        
        return ReorderRecommendation(
            product_id=metrics.product_id,
            should_reorder=should_reorder,
            order_quantity=order_quantity,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            days_of_supply=days_of_supply,
            projected_stockout_date=stockout_date,
            urgency=urgency,
            reason=reason,
        )
    
    def _calculate_turnover_rate(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
    ) -> float:
        """Calculate inventory turnover rate (annual)."""
        if metrics.current_stock == 0:
            return 0.0
        
        annual_demand = forecast.daily_demand * 365
        return annual_demand / metrics.current_stock
    
    def _calculate_potential_savings(
        self,
        metrics: InventoryMetrics,
        forecast: DemandForecast,
        eoq: int,
    ) -> float:
        """Calculate potential cost savings from optimization."""
        # Current average order size (assumed)
        if not metrics.sales_history or len(metrics.sales_history) < 30:
            return 0.0
        
        # Compare current vs optimal total costs
        annual_demand = forecast.daily_demand * 365
        holding_cost = metrics.unit_cost * metrics.holding_cost_rate
        
        # Optimal total cost with EOQ
        if eoq > 0:
            optimal_cost = math.sqrt(2 * annual_demand * metrics.ordering_cost * holding_cost)
        else:
            optimal_cost = 0
        
        # Estimated current cost (assuming suboptimal ordering)
        # Using 2x the optimal as a rough estimate of current
        current_cost = optimal_cost * 1.5
        
        return max(0, current_cost - optimal_cost)
    
    async def optimize_portfolio(
        self,
        products: List[InventoryMetrics],
    ) -> Dict[str, Any]:
        """
        Optimize inventory across all products.
        
        Args:
            products: List of product inventory metrics
            
        Returns:
            Portfolio-level optimization results
        """
        analyses = []
        total_value = 0.0
        total_savings = 0.0
        reorders_needed = 0
        
        for metrics in products:
            analysis = await self.analyze_product(metrics)
            analyses.append(analysis)
            
            total_value += metrics.inventory_value
            total_savings += analysis.optimization_savings
            if analysis.recommendation.should_reorder:
                reorders_needed += 1
        
        # ABC analysis
        abc_classification = self._abc_analysis(products)
        
        return {
            "total_products": len(products),
            "total_inventory_value": round(total_value, 2),
            "potential_annual_savings": round(total_savings, 2),
            "reorders_needed": reorders_needed,
            "status_summary": self._summarize_status(analyses),
            "abc_classification": abc_classification,
            "recommendations": [
                a.recommendation.to_dict()
                for a in analyses
                if a.recommendation.should_reorder
            ][:10],  # Top 10 recommendations
        }
    
    def _abc_analysis(
        self,
        products: List[InventoryMetrics],
    ) -> Dict[str, List[str]]:
        """Perform ABC classification."""
        if not products:
            return {"A": [], "B": [], "C": []}
        
        # Sort by value
        sorted_products = sorted(
            products,
            key=lambda p: p.inventory_value,
            reverse=True,
        )
        
        total_value = sum(p.inventory_value for p in products)
        
        a_items = []
        b_items = []
        c_items = []
        
        cumulative = 0
        for product in sorted_products:
            cumulative += product.inventory_value
            percentage = cumulative / total_value if total_value > 0 else 0
            
            if percentage <= 0.8:  # Top 80% of value
                a_items.append(product.product_id)
            elif percentage <= 0.95:  # Next 15% of value
                b_items.append(product.product_id)
            else:  # Bottom 5% of value
                c_items.append(product.product_id)
        
        return {
            "A": a_items[:10],  # Limit for display
            "B": b_items[:10],
            "C": c_items[:10],
        }
    
    def _summarize_status(
        self,
        analyses: List[InventoryAnalysis],
    ) -> Dict[str, int]:
        """Summarize status across all products."""
        summary = {status.value: 0 for status in StockStatus}
        
        for analysis in analyses:
            summary[analysis.status.value] += 1
        
        return summary


# Global inventory optimizer instance
inventory_optimizer = InventoryOptimizer()


def get_inventory_optimizer() -> InventoryOptimizer:
    """Get the global inventory optimizer."""
    return inventory_optimizer
