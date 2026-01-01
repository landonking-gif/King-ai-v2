"""
Customer Segmentation.
Automatic customer segmentation for targeted marketing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import statistics
import math

from src.utils.structured_logging import get_logger

logger = get_logger("segmentation")


class SegmentType(str, Enum):
    """Types of customer segments."""
    RFM = "rfm"  # Recency, Frequency, Monetary
    BEHAVIORAL = "behavioral"
    DEMOGRAPHIC = "demographic"
    VALUE = "value"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"


class LifecycleStage(str, Enum):
    """Customer lifecycle stages."""
    NEW = "new"
    ACTIVE = "active"
    AT_RISK = "at_risk"
    CHURNED = "churned"
    REACTIVATED = "reactivated"
    VIP = "vip"


class ValueTier(str, Enum):
    """Customer value tiers."""
    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    STANDARD = "standard"


@dataclass
class CustomerMetrics:
    """Metrics for a single customer."""
    customer_id: str
    total_orders: int = 0
    total_revenue: float = 0.0
    first_order_date: Optional[datetime] = None
    last_order_date: Optional[datetime] = None
    average_order_value: float = 0.0
    days_since_last_order: int = 0
    order_frequency_days: float = 0.0
    products_purchased: Set[str] = field(default_factory=set)
    categories_purchased: Set[str] = field(default_factory=set)
    return_rate: float = 0.0
    customer_lifetime_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_orders(
        cls,
        customer_id: str,
        orders: List[Dict[str, Any]],
    ) -> "CustomerMetrics":
        """Create metrics from order history."""
        if not orders:
            return cls(customer_id=customer_id)
        
        total_revenue = sum(o.get("total", 0) for o in orders)
        total_orders = len(orders)
        
        order_dates = [
            datetime.fromisoformat(o["date"]) if isinstance(o["date"], str) else o["date"]
            for o in orders if o.get("date")
        ]
        order_dates.sort()
        
        first_order = order_dates[0] if order_dates else None
        last_order = order_dates[-1] if order_dates else None
        
        days_since = (datetime.utcnow() - last_order).days if last_order else 999
        
        # Calculate frequency
        if len(order_dates) > 1:
            total_days = (order_dates[-1] - order_dates[0]).days
            frequency = total_days / (len(order_dates) - 1) if total_days > 0 else 0
        else:
            frequency = 0
        
        # Products and categories
        products = set()
        categories = set()
        for order in orders:
            for item in order.get("items", []):
                if item.get("product_id"):
                    products.add(item["product_id"])
                if item.get("category"):
                    categories.add(item["category"])
        
        return cls(
            customer_id=customer_id,
            total_orders=total_orders,
            total_revenue=total_revenue,
            first_order_date=first_order,
            last_order_date=last_order,
            average_order_value=total_revenue / total_orders if total_orders > 0 else 0,
            days_since_last_order=days_since,
            order_frequency_days=frequency,
            products_purchased=products,
            categories_purchased=categories,
        )


@dataclass
class RFMScore:
    """RFM (Recency, Frequency, Monetary) score."""
    recency_score: int  # 1-5, 5 = most recent
    frequency_score: int  # 1-5, 5 = most frequent
    monetary_score: int  # 1-5, 5 = highest spending
    
    @property
    def total_score(self) -> int:
        return self.recency_score + self.frequency_score + self.monetary_score
    
    @property
    def segment_code(self) -> str:
        return f"R{self.recency_score}F{self.frequency_score}M{self.monetary_score}"
    
    @property
    def segment_name(self) -> str:
        """Get human-readable segment name from RFM scores."""
        if self.total_score >= 12:
            return "Champions"
        elif self.recency_score >= 4 and self.frequency_score >= 3:
            return "Loyal Customers"
        elif self.recency_score >= 4 and self.monetary_score >= 3:
            return "Potential Loyalists"
        elif self.recency_score >= 3 and self.frequency_score <= 2:
            return "New Customers"
        elif self.recency_score <= 2 and self.frequency_score >= 4:
            return "At Risk"
        elif self.recency_score <= 2 and self.total_score >= 8:
            return "Cant Lose Them"
        elif self.recency_score <= 2 and self.total_score <= 6:
            return "Lost"
        elif self.frequency_score >= 3 and self.monetary_score >= 3:
            return "Need Attention"
        else:
            return "About to Sleep"


@dataclass
class Segment:
    """A customer segment."""
    id: str
    name: str
    type: SegmentType
    description: str = ""
    customer_ids: List[str] = field(default_factory=list)
    criteria: Dict[str, Any] = field(default_factory=dict)
    size: int = 0
    avg_value: float = 0.0
    total_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "size": self.size,
            "avg_value": round(self.avg_value, 2),
            "total_value": round(self.total_value, 2),
            "criteria": self.criteria,
        }


@dataclass
class SegmentationResult:
    """Result of segmentation analysis."""
    segments: List[Segment]
    total_customers: int
    unassigned_customers: int
    analysis_date: datetime = field(default_factory=datetime.utcnow)
    insights: List[str] = field(default_factory=list)


class RFMSegmenter:
    """RFM-based segmentation."""
    
    def __init__(self, quintiles: int = 5):
        self.quintiles = quintiles
    
    def calculate_scores(
        self,
        customers: List[CustomerMetrics],
    ) -> Dict[str, RFMScore]:
        """Calculate RFM scores for all customers."""
        if not customers:
            return {}
        
        # Extract values for percentile calculation
        recencies = [c.days_since_last_order for c in customers]
        frequencies = [c.total_orders for c in customers]
        monetary = [c.total_revenue for c in customers]
        
        # Calculate percentile thresholds
        r_thresholds = self._calculate_thresholds(recencies, reverse=True)
        f_thresholds = self._calculate_thresholds(frequencies)
        m_thresholds = self._calculate_thresholds(monetary)
        
        scores = {}
        for customer in customers:
            r_score = self._score_value(customer.days_since_last_order, r_thresholds, reverse=True)
            f_score = self._score_value(customer.total_orders, f_thresholds)
            m_score = self._score_value(customer.total_revenue, m_thresholds)
            
            scores[customer.customer_id] = RFMScore(
                recency_score=r_score,
                frequency_score=f_score,
                monetary_score=m_score,
            )
        
        return scores
    
    def _calculate_thresholds(
        self,
        values: List[float],
        reverse: bool = False,
    ) -> List[float]:
        """Calculate percentile thresholds."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        thresholds = []
        for i in range(1, self.quintiles):
            idx = int((i / self.quintiles) * n)
            thresholds.append(sorted_values[idx])
        
        return thresholds
    
    def _score_value(
        self,
        value: float,
        thresholds: List[float],
        reverse: bool = False,
    ) -> int:
        """Score a value based on thresholds."""
        score = 1
        for threshold in thresholds:
            if reverse:
                if value <= threshold:
                    score += 1
            else:
                if value >= threshold:
                    score += 1
        
        return min(score, self.quintiles)


class LifecycleSegmenter:
    """Customer lifecycle segmentation."""
    
    def __init__(
        self,
        new_days: int = 30,
        active_days: int = 90,
        at_risk_days: int = 180,
        churned_days: int = 365,
    ):
        self.new_days = new_days
        self.active_days = active_days
        self.at_risk_days = at_risk_days
        self.churned_days = churned_days
    
    def classify(
        self,
        customer: CustomerMetrics,
    ) -> LifecycleStage:
        """Classify customer into lifecycle stage."""
        if customer.first_order_date is None:
            return LifecycleStage.NEW
        
        days_as_customer = (datetime.utcnow() - customer.first_order_date).days
        days_since_last = customer.days_since_last_order
        
        # VIP: High value and active
        if customer.total_revenue > 1000 and days_since_last <= self.active_days:
            return LifecycleStage.VIP
        
        # New customer
        if days_as_customer <= self.new_days:
            return LifecycleStage.NEW
        
        # Active customer
        if days_since_last <= self.active_days:
            return LifecycleStage.ACTIVE
        
        # At risk
        if days_since_last <= self.at_risk_days:
            return LifecycleStage.AT_RISK
        
        # Check for reactivation
        if customer.total_orders > 1:
            # Calculate average frequency
            avg_gap = customer.order_frequency_days
            if days_since_last > avg_gap * 2:
                return LifecycleStage.CHURNED
        
        # Churned
        if days_since_last > self.churned_days:
            return LifecycleStage.CHURNED
        
        return LifecycleStage.AT_RISK


class ValueSegmenter:
    """Customer value tier segmentation."""
    
    def __init__(
        self,
        platinum_threshold: float = 0.95,  # Top 5%
        gold_threshold: float = 0.80,  # Top 20%
        silver_threshold: float = 0.50,  # Top 50%
        bronze_threshold: float = 0.25,  # Top 75%
    ):
        self.platinum_threshold = platinum_threshold
        self.gold_threshold = gold_threshold
        self.silver_threshold = silver_threshold
        self.bronze_threshold = bronze_threshold
    
    def classify(
        self,
        customer: CustomerMetrics,
        all_customers: List[CustomerMetrics],
    ) -> ValueTier:
        """Classify customer into value tier."""
        if not all_customers:
            return ValueTier.STANDARD
        
        # Calculate percentile
        values = sorted([c.total_revenue for c in all_customers])
        rank = sum(1 for v in values if v <= customer.total_revenue)
        percentile = rank / len(values)
        
        if percentile >= self.platinum_threshold:
            return ValueTier.PLATINUM
        elif percentile >= self.gold_threshold:
            return ValueTier.GOLD
        elif percentile >= self.silver_threshold:
            return ValueTier.SILVER
        elif percentile >= self.bronze_threshold:
            return ValueTier.BRONZE
        else:
            return ValueTier.STANDARD


class CustomerSegmentation:
    """
    Customer segmentation engine.
    
    Features:
    - RFM analysis
    - Lifecycle stages
    - Value tiers
    - Behavioral segments
    - Custom segmentation rules
    """
    
    def __init__(self):
        self.rfm_segmenter = RFMSegmenter()
        self.lifecycle_segmenter = LifecycleSegmenter()
        self.value_segmenter = ValueSegmenter()
    
    async def segment_customers(
        self,
        customers: List[CustomerMetrics],
        segment_types: List[SegmentType] = None,
    ) -> SegmentationResult:
        """
        Segment customers using multiple methods.
        
        Args:
            customers: List of customer metrics
            segment_types: Types of segmentation to apply
            
        Returns:
            Segmentation result with all segments
        """
        segment_types = segment_types or [
            SegmentType.RFM,
            SegmentType.LIFECYCLE,
            SegmentType.VALUE,
        ]
        
        all_segments = []
        
        if SegmentType.RFM in segment_types:
            rfm_segments = self._create_rfm_segments(customers)
            all_segments.extend(rfm_segments)
        
        if SegmentType.LIFECYCLE in segment_types:
            lifecycle_segments = self._create_lifecycle_segments(customers)
            all_segments.extend(lifecycle_segments)
        
        if SegmentType.VALUE in segment_types:
            value_segments = self._create_value_segments(customers)
            all_segments.extend(value_segments)
        
        # Generate insights
        insights = self._generate_insights(customers, all_segments)
        
        return SegmentationResult(
            segments=all_segments,
            total_customers=len(customers),
            unassigned_customers=0,
            insights=insights,
        )
    
    def _create_rfm_segments(
        self,
        customers: List[CustomerMetrics],
    ) -> List[Segment]:
        """Create RFM-based segments."""
        rfm_scores = self.rfm_segmenter.calculate_scores(customers)
        
        # Group by segment name
        segment_groups: Dict[str, List[str]] = {}
        customer_values: Dict[str, float] = {}
        
        for customer in customers:
            score = rfm_scores.get(customer.customer_id)
            if score:
                segment_name = score.segment_name
                if segment_name not in segment_groups:
                    segment_groups[segment_name] = []
                segment_groups[segment_name].append(customer.customer_id)
                customer_values[customer.customer_id] = customer.total_revenue
        
        segments = []
        for name, customer_ids in segment_groups.items():
            values = [customer_values.get(cid, 0) for cid in customer_ids]
            
            segments.append(Segment(
                id=f"rfm_{name.lower().replace(' ', '_')}",
                name=name,
                type=SegmentType.RFM,
                description=f"RFM segment: {name}",
                customer_ids=customer_ids,
                size=len(customer_ids),
                avg_value=statistics.mean(values) if values else 0,
                total_value=sum(values),
            ))
        
        return segments
    
    def _create_lifecycle_segments(
        self,
        customers: List[CustomerMetrics],
    ) -> List[Segment]:
        """Create lifecycle-based segments."""
        stage_groups: Dict[LifecycleStage, List[str]] = {}
        customer_values: Dict[str, float] = {}
        
        for customer in customers:
            stage = self.lifecycle_segmenter.classify(customer)
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(customer.customer_id)
            customer_values[customer.customer_id] = customer.total_revenue
        
        segments = []
        for stage, customer_ids in stage_groups.items():
            values = [customer_values.get(cid, 0) for cid in customer_ids]
            
            segments.append(Segment(
                id=f"lifecycle_{stage.value}",
                name=stage.value.replace("_", " ").title(),
                type=SegmentType.LIFECYCLE,
                description=f"Lifecycle stage: {stage.value}",
                customer_ids=customer_ids,
                size=len(customer_ids),
                avg_value=statistics.mean(values) if values else 0,
                total_value=sum(values),
            ))
        
        return segments
    
    def _create_value_segments(
        self,
        customers: List[CustomerMetrics],
    ) -> List[Segment]:
        """Create value-based segments."""
        tier_groups: Dict[ValueTier, List[str]] = {}
        customer_values: Dict[str, float] = {}
        
        for customer in customers:
            tier = self.value_segmenter.classify(customer, customers)
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(customer.customer_id)
            customer_values[customer.customer_id] = customer.total_revenue
        
        segments = []
        for tier, customer_ids in tier_groups.items():
            values = [customer_values.get(cid, 0) for cid in customer_ids]
            
            segments.append(Segment(
                id=f"value_{tier.value}",
                name=f"{tier.value.title()} Tier",
                type=SegmentType.VALUE,
                description=f"Value tier: {tier.value}",
                customer_ids=customer_ids,
                size=len(customer_ids),
                avg_value=statistics.mean(values) if values else 0,
                total_value=sum(values),
            ))
        
        return segments
    
    def _generate_insights(
        self,
        customers: List[CustomerMetrics],
        segments: List[Segment],
    ) -> List[str]:
        """Generate insights from segmentation."""
        insights = []
        
        # Value concentration
        value_segments = [s for s in segments if s.type == SegmentType.VALUE]
        if value_segments:
            platinum = next((s for s in value_segments if "platinum" in s.id), None)
            if platinum:
                total_value = sum(s.total_value for s in value_segments)
                if total_value > 0:
                    concentration = platinum.total_value / total_value * 100
                    insights.append(
                        f"Top {platinum.size} customers ({(platinum.size/len(customers)*100):.1f}%) "
                        f"generate {concentration:.1f}% of revenue"
                    )
        
        # At-risk customers
        lifecycle_segments = [s for s in segments if s.type == SegmentType.LIFECYCLE]
        at_risk = next((s for s in lifecycle_segments if "at_risk" in s.id), None)
        if at_risk and at_risk.size > 0:
            insights.append(
                f"{at_risk.size} customers ({(at_risk.size/len(customers)*100):.1f}%) are at risk of churning, "
                f"representing ${at_risk.total_value:,.2f} in potential lost revenue"
            )
        
        # Champions segment
        rfm_segments = [s for s in segments if s.type == SegmentType.RFM]
        champions = next((s for s in rfm_segments if "champion" in s.id.lower()), None)
        if champions:
            insights.append(
                f"Champions segment: {champions.size} customers with ${champions.avg_value:,.2f} average value"
            )
        
        return insights[:5]  # Limit to top 5 insights


# Global segmentation engine instance
segmentation_engine = CustomerSegmentation()


def get_segmentation_engine() -> CustomerSegmentation:
    """Get the global segmentation engine."""
    return segmentation_engine
