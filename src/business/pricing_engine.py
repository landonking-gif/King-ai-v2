"""
Dynamic Pricing Engine.
Automated pricing optimization based on market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import math
import statistics

from src.utils.structured_logging import get_logger

logger = get_logger("pricing_engine")


class PricingStrategy(str, Enum):
    """Pricing strategies."""
    COST_PLUS = "cost_plus"
    COMPETITIVE = "competitive"
    VALUE_BASED = "value_based"
    DYNAMIC = "dynamic"
    PENETRATION = "penetration"
    SKIMMING = "skimming"


class PriceChangeReason(str, Enum):
    """Reasons for price changes."""
    DEMAND_INCREASE = "demand_increase"
    DEMAND_DECREASE = "demand_decrease"
    COMPETITOR_CHANGE = "competitor_change"
    INVENTORY_HIGH = "inventory_high"
    INVENTORY_LOW = "inventory_low"
    COST_CHANGE = "cost_change"
    PROMOTIONAL = "promotional"
    SEASONAL = "seasonal"
    MARGIN_OPTIMIZATION = "margin_optimization"


class ElasticityType(str, Enum):
    """Price elasticity types."""
    ELASTIC = "elastic"  # |E| > 1
    INELASTIC = "inelastic"  # |E| < 1
    UNIT_ELASTIC = "unit_elastic"  # |E| = 1


@dataclass
class ProductCosts:
    """Cost structure for a product."""
    product_id: str
    unit_cost: float
    shipping_cost: float = 0.0
    platform_fee_rate: float = 0.0
    payment_fee_rate: float = 0.03  # Default 3%
    overhead_rate: float = 0.10  # 10% overhead
    
    @property
    def total_cost(self) -> float:
        return self.unit_cost + self.shipping_cost
    
    def break_even_price(self, with_overhead: bool = True) -> float:
        """Calculate break-even price."""
        base = self.total_cost
        if with_overhead:
            base *= (1 + self.overhead_rate)
        
        # Account for fees
        fee_rate = self.platform_fee_rate + self.payment_fee_rate
        if fee_rate >= 1:
            return base * 2  # Fallback
        
        return base / (1 - fee_rate)


@dataclass
class MarketData:
    """Market conditions for pricing."""
    product_id: str
    competitor_prices: List[float] = field(default_factory=list)
    demand_trend: float = 0.0  # -1 to 1, positive = increasing
    inventory_level: int = 0
    inventory_days_supply: float = 30.0
    conversion_rate: float = 0.02
    views: int = 0
    sales_velocity: float = 0.0  # Units per day
    
    @property
    def avg_competitor_price(self) -> float:
        if not self.competitor_prices:
            return 0.0
        return statistics.mean(self.competitor_prices)
    
    @property
    def min_competitor_price(self) -> float:
        if not self.competitor_prices:
            return 0.0
        return min(self.competitor_prices)
    
    @property
    def max_competitor_price(self) -> float:
        if not self.competitor_prices:
            return 0.0
        return max(self.competitor_prices)


@dataclass
class PriceHistory:
    """Historical pricing data."""
    price: float
    quantity_sold: int
    revenue: float
    timestamp: datetime
    
    @property
    def effective_aov(self) -> float:
        if self.quantity_sold == 0:
            return self.price
        return self.revenue / self.quantity_sold


@dataclass
class PriceRecommendation:
    """Price recommendation result."""
    product_id: str
    current_price: float
    recommended_price: float
    min_price: float
    max_price: float
    confidence: float
    strategy: PricingStrategy
    reason: PriceChangeReason
    expected_margin: float
    expected_volume_change: float
    expected_revenue_change: float
    
    @property
    def price_change(self) -> float:
        return self.recommended_price - self.current_price
    
    @property
    def price_change_percent(self) -> float:
        if self.current_price == 0:
            return 0.0
        return (self.price_change / self.current_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "current_price": round(self.current_price, 2),
            "recommended_price": round(self.recommended_price, 2),
            "price_change": round(self.price_change, 2),
            "price_change_percent": round(self.price_change_percent, 1),
            "min_price": round(self.min_price, 2),
            "max_price": round(self.max_price, 2),
            "confidence": round(self.confidence, 2),
            "strategy": self.strategy.value,
            "reason": self.reason.value,
            "expected_margin": round(self.expected_margin * 100, 1),
            "expected_volume_change": round(self.expected_volume_change * 100, 1),
            "expected_revenue_change": round(self.expected_revenue_change * 100, 1),
        }


@dataclass
class PricingConfig:
    """Configuration for pricing engine."""
    target_margin: float = 0.30  # 30% target margin
    min_margin: float = 0.10  # 10% minimum margin
    max_price_change: float = 0.15  # Max 15% change per update
    price_rounding: float = 0.99  # Round to x.99
    competitor_weight: float = 0.3  # Weight for competitive pricing
    demand_weight: float = 0.3  # Weight for demand-based pricing
    inventory_weight: float = 0.2  # Weight for inventory-based pricing
    margin_weight: float = 0.2  # Weight for margin optimization


class ElasticityCalculator:
    """Calculate price elasticity of demand."""
    
    def calculate(
        self,
        price_history: List[PriceHistory],
    ) -> tuple:
        """
        Calculate price elasticity.
        
        Returns:
            (elasticity, elasticity_type)
        """
        if len(price_history) < 2:
            return -1.0, ElasticityType.ELASTIC  # Default assumption
        
        # Calculate elasticity using midpoint method
        elasticities = []
        
        for i in range(1, len(price_history)):
            prev = price_history[i - 1]
            curr = price_history[i]
            
            if prev.price == 0 or curr.quantity_sold + prev.quantity_sold == 0:
                continue
            
            # Midpoint elasticity formula
            price_change = (curr.price - prev.price) / ((curr.price + prev.price) / 2)
            qty_change = (curr.quantity_sold - prev.quantity_sold) / ((curr.quantity_sold + prev.quantity_sold) / 2)
            
            if price_change != 0:
                elasticity = qty_change / price_change
                elasticities.append(elasticity)
        
        if not elasticities:
            return -1.0, ElasticityType.ELASTIC
        
        avg_elasticity = statistics.mean(elasticities)
        
        if abs(avg_elasticity) > 1:
            etype = ElasticityType.ELASTIC
        elif abs(avg_elasticity) < 1:
            etype = ElasticityType.INELASTIC
        else:
            etype = ElasticityType.UNIT_ELASTIC
        
        return avg_elasticity, etype


class DynamicPricingEngine:
    """
    Dynamic Pricing Engine.
    
    Features:
    - Cost-plus pricing
    - Competitive pricing
    - Demand-based pricing
    - Inventory-based pricing
    - Price elasticity analysis
    - Margin optimization
    """
    
    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()
        self.elasticity_calculator = ElasticityCalculator()
    
    async def recommend_price(
        self,
        product_id: str,
        current_price: float,
        costs: ProductCosts,
        market_data: MarketData,
        price_history: List[PriceHistory] = None,
        strategy: PricingStrategy = None,
    ) -> PriceRecommendation:
        """
        Generate price recommendation.
        
        Args:
            product_id: Product identifier
            current_price: Current selling price
            costs: Product cost structure
            market_data: Market conditions
            price_history: Historical pricing data
            strategy: Override pricing strategy
            
        Returns:
            Price recommendation
        """
        price_history = price_history or []
        
        # Calculate floor and ceiling prices
        min_price = self._calculate_min_price(costs)
        max_price = self._calculate_max_price(costs, market_data, current_price)
        
        # Determine strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(market_data, price_history)
        
        # Calculate price based on strategy
        if strategy == PricingStrategy.COST_PLUS:
            recommended = self._cost_plus_price(costs)
            reason = PriceChangeReason.MARGIN_OPTIMIZATION
        
        elif strategy == PricingStrategy.COMPETITIVE:
            recommended = self._competitive_price(costs, market_data)
            reason = PriceChangeReason.COMPETITOR_CHANGE
        
        elif strategy == PricingStrategy.VALUE_BASED:
            recommended = self._value_based_price(costs, market_data, price_history)
            reason = PriceChangeReason.MARGIN_OPTIMIZATION
        
        elif strategy == PricingStrategy.DYNAMIC:
            recommended, reason = self._dynamic_price(
                current_price, costs, market_data, price_history
            )
        
        elif strategy == PricingStrategy.PENETRATION:
            recommended = self._penetration_price(costs, market_data)
            reason = PriceChangeReason.DEMAND_INCREASE
        
        else:  # SKIMMING
            recommended = self._skimming_price(costs, market_data)
            reason = PriceChangeReason.MARGIN_OPTIMIZATION
        
        # Apply constraints
        recommended = self._apply_constraints(
            recommended, current_price, min_price, max_price
        )
        
        # Round price
        recommended = self._round_price(recommended)
        
        # Calculate expected impacts
        elasticity, _ = self.elasticity_calculator.calculate(price_history)
        expected_margin = self._calculate_margin(recommended, costs)
        expected_volume_change = self._estimate_volume_change(
            current_price, recommended, elasticity
        )
        expected_revenue_change = self._estimate_revenue_change(
            current_price, recommended, expected_volume_change
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(market_data, price_history)
        
        return PriceRecommendation(
            product_id=product_id,
            current_price=current_price,
            recommended_price=recommended,
            min_price=min_price,
            max_price=max_price,
            confidence=confidence,
            strategy=strategy,
            reason=reason,
            expected_margin=expected_margin,
            expected_volume_change=expected_volume_change,
            expected_revenue_change=expected_revenue_change,
        )
    
    def _calculate_min_price(self, costs: ProductCosts) -> float:
        """Calculate minimum price (break-even + min margin)."""
        break_even = costs.break_even_price()
        return break_even * (1 + self.config.min_margin)
    
    def _calculate_max_price(
        self,
        costs: ProductCosts,
        market_data: MarketData,
        current_price: float,
    ) -> float:
        """Calculate maximum price."""
        # Base on cost
        cost_based_max = costs.break_even_price() * 3  # 200% margin max
        
        # Consider competitor ceiling
        if market_data.competitor_prices:
            competitor_max = market_data.max_competitor_price * 1.2
            cost_based_max = min(cost_based_max, competitor_max)
        
        return max(cost_based_max, current_price * 1.3)
    
    def _select_strategy(
        self,
        market_data: MarketData,
        price_history: List[PriceHistory],
    ) -> PricingStrategy:
        """Select best pricing strategy."""
        # High inventory - use penetration
        if market_data.inventory_days_supply > 60:
            return PricingStrategy.PENETRATION
        
        # Low inventory - use skimming
        if market_data.inventory_days_supply < 7:
            return PricingStrategy.SKIMMING
        
        # Strong competitors - use competitive
        if len(market_data.competitor_prices) >= 3:
            return PricingStrategy.COMPETITIVE
        
        # Good conversion data - use dynamic
        if len(price_history) >= 5:
            return PricingStrategy.DYNAMIC
        
        # Default - cost plus
        return PricingStrategy.COST_PLUS
    
    def _cost_plus_price(self, costs: ProductCosts) -> float:
        """Calculate cost-plus price."""
        break_even = costs.break_even_price()
        return break_even * (1 + self.config.target_margin)
    
    def _competitive_price(
        self,
        costs: ProductCosts,
        market_data: MarketData,
    ) -> float:
        """Calculate competitive price."""
        if not market_data.competitor_prices:
            return self._cost_plus_price(costs)
        
        avg_price = market_data.avg_competitor_price
        min_price = self._calculate_min_price(costs)
        
        # Position slightly below average
        competitive_price = avg_price * 0.95
        
        return max(competitive_price, min_price)
    
    def _value_based_price(
        self,
        costs: ProductCosts,
        market_data: MarketData,
        price_history: List[PriceHistory],
    ) -> float:
        """Calculate value-based price."""
        # Use conversion rate as value indicator
        if market_data.conversion_rate > 0.03:
            # High conversion = high perceived value
            value_multiplier = 1.2
        elif market_data.conversion_rate > 0.02:
            value_multiplier = 1.1
        else:
            value_multiplier = 1.0
        
        base_price = self._cost_plus_price(costs)
        return base_price * value_multiplier
    
    def _dynamic_price(
        self,
        current_price: float,
        costs: ProductCosts,
        market_data: MarketData,
        price_history: List[PriceHistory],
    ) -> tuple:
        """Calculate dynamic price based on multiple factors."""
        adjustments = []
        
        # Demand-based adjustment
        demand_adj = market_data.demand_trend * 0.1 * self.config.demand_weight
        adjustments.append(demand_adj)
        
        # Inventory-based adjustment
        if market_data.inventory_days_supply < 14:
            inv_adj = 0.1 * self.config.inventory_weight  # Price up
            reason = PriceChangeReason.INVENTORY_LOW
        elif market_data.inventory_days_supply > 45:
            inv_adj = -0.1 * self.config.inventory_weight  # Price down
            reason = PriceChangeReason.INVENTORY_HIGH
        else:
            inv_adj = 0
            reason = PriceChangeReason.DEMAND_INCREASE if demand_adj > 0 else PriceChangeReason.DEMAND_DECREASE
        adjustments.append(inv_adj)
        
        # Competitive adjustment
        if market_data.competitor_prices:
            avg_comp = market_data.avg_competitor_price
            if avg_comp > 0:
                comp_diff = (avg_comp - current_price) / current_price
                comp_adj = comp_diff * 0.3 * self.config.competitor_weight
                adjustments.append(comp_adj)
                if abs(comp_adj) > abs(inv_adj):
                    reason = PriceChangeReason.COMPETITOR_CHANGE
        
        # Margin adjustment
        current_margin = self._calculate_margin(current_price, costs)
        if current_margin < self.config.min_margin:
            margin_adj = 0.1 * self.config.margin_weight
            reason = PriceChangeReason.MARGIN_OPTIMIZATION
        elif current_margin > self.config.target_margin * 1.5:
            margin_adj = -0.05 * self.config.margin_weight
        else:
            margin_adj = 0
        adjustments.append(margin_adj)
        
        total_adjustment = sum(adjustments)
        recommended = current_price * (1 + total_adjustment)
        
        return recommended, reason
    
    def _penetration_price(
        self,
        costs: ProductCosts,
        market_data: MarketData,
    ) -> float:
        """Calculate penetration price (aggressive low price)."""
        min_price = self._calculate_min_price(costs)
        
        if market_data.competitor_prices:
            # Price at or below lowest competitor
            target = market_data.min_competitor_price * 0.95
            return max(target, min_price)
        
        return min_price * 1.1
    
    def _skimming_price(
        self,
        costs: ProductCosts,
        market_data: MarketData,
    ) -> float:
        """Calculate skimming price (premium pricing)."""
        base = self._cost_plus_price(costs)
        
        if market_data.competitor_prices:
            # Price above average competitor
            target = market_data.avg_competitor_price * 1.15
            return max(target, base * 1.2)
        
        return base * 1.3
    
    def _apply_constraints(
        self,
        price: float,
        current_price: float,
        min_price: float,
        max_price: float,
    ) -> float:
        """Apply price constraints."""
        # Limit change percentage
        max_change = current_price * self.config.max_price_change
        if price > current_price + max_change:
            price = current_price + max_change
        elif price < current_price - max_change:
            price = current_price - max_change
        
        # Apply floor and ceiling
        price = max(min_price, min(max_price, price))
        
        return price
    
    def _round_price(self, price: float) -> float:
        """Round price to psychological price point."""
        # Round to .99 ending
        decimal_part = self.config.price_rounding
        return math.floor(price) + decimal_part
    
    def _calculate_margin(
        self,
        price: float,
        costs: ProductCosts,
    ) -> float:
        """Calculate margin at a given price."""
        if price == 0:
            return 0.0
        
        # Deduct fees
        net_price = price * (1 - costs.platform_fee_rate - costs.payment_fee_rate)
        profit = net_price - costs.total_cost * (1 + costs.overhead_rate)
        
        return profit / price
    
    def _estimate_volume_change(
        self,
        current_price: float,
        new_price: float,
        elasticity: float,
    ) -> float:
        """Estimate volume change based on price change."""
        if current_price == 0:
            return 0.0
        
        price_change_pct = (new_price - current_price) / current_price
        
        # Volume change = elasticity * price change
        return elasticity * price_change_pct
    
    def _estimate_revenue_change(
        self,
        current_price: float,
        new_price: float,
        volume_change: float,
    ) -> float:
        """Estimate revenue change."""
        if current_price == 0:
            return 0.0
        
        price_change_pct = (new_price - current_price) / current_price
        
        # Revenue change = (1 + price change) * (1 + volume change) - 1
        return (1 + price_change_pct) * (1 + volume_change) - 1
    
    def _calculate_confidence(
        self,
        market_data: MarketData,
        price_history: List[PriceHistory],
    ) -> float:
        """Calculate confidence in recommendation."""
        confidence = 0.5  # Base confidence
        
        # More competitor data = higher confidence
        if len(market_data.competitor_prices) >= 5:
            confidence += 0.15
        elif len(market_data.competitor_prices) >= 3:
            confidence += 0.1
        
        # More historical data = higher confidence
        if len(price_history) >= 10:
            confidence += 0.2
        elif len(price_history) >= 5:
            confidence += 0.1
        
        # Higher conversion rate = more reliable signals
        if market_data.conversion_rate > 0.03:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def optimize_portfolio(
        self,
        products: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Optimize pricing across product portfolio.
        
        Args:
            products: List of product data with costs and market data
            
        Returns:
            Portfolio optimization results
        """
        recommendations = []
        total_expected_revenue_change = 0.0
        
        for product in products:
            costs = ProductCosts(
                product_id=product["product_id"],
                unit_cost=product.get("unit_cost", 0),
                shipping_cost=product.get("shipping_cost", 0),
                platform_fee_rate=product.get("platform_fee_rate", 0),
            )
            
            market_data = MarketData(
                product_id=product["product_id"],
                competitor_prices=product.get("competitor_prices", []),
                demand_trend=product.get("demand_trend", 0),
                inventory_level=product.get("inventory_level", 0),
                inventory_days_supply=product.get("inventory_days_supply", 30),
                conversion_rate=product.get("conversion_rate", 0.02),
            )
            
            recommendation = await self.recommend_price(
                product_id=product["product_id"],
                current_price=product.get("current_price", 0),
                costs=costs,
                market_data=market_data,
            )
            
            recommendations.append(recommendation)
            total_expected_revenue_change += recommendation.expected_revenue_change
        
        # Sort by expected impact
        recommendations.sort(
            key=lambda r: abs(r.expected_revenue_change),
            reverse=True,
        )
        
        return {
            "total_products": len(products),
            "avg_expected_revenue_change": (
                round(total_expected_revenue_change / len(products) * 100, 1)
                if products else 0
            ),
            "recommendations": [r.to_dict() for r in recommendations[:10]],
            "price_increases": sum(1 for r in recommendations if r.price_change > 0),
            "price_decreases": sum(1 for r in recommendations if r.price_change < 0),
            "unchanged": sum(1 for r in recommendations if r.price_change == 0),
        }


# Global pricing engine instance
pricing_engine = DynamicPricingEngine()


def get_pricing_engine() -> DynamicPricingEngine:
    """Get the global pricing engine."""
    return pricing_engine
