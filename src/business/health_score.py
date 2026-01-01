"""
Business Health Score Calculator.
Aggregates KPIs and metrics into a comprehensive health score.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("health_score")


class HealthCategory(str, Enum):
    """Categories of business health."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    GROWTH = "growth"
    RISK = "risk"


class HealthGrade(str, Enum):
    """Health grade levels."""
    EXCELLENT = "A"
    GOOD = "B"
    FAIR = "C"
    POOR = "D"
    CRITICAL = "F"


@dataclass
class MetricScore:
    """Score for a single metric."""
    name: str
    value: float
    score: float  # 0.0 to 1.0
    weight: float
    target: Optional[float] = None
    benchmark: Optional[float] = None
    trend: Optional[str] = None  # "up", "down", "stable"
    category: HealthCategory = HealthCategory.OPERATIONAL
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight
    
    @property
    def status(self) -> str:
        if self.score >= 0.8:
            return "healthy"
        elif self.score >= 0.6:
            return "warning"
        else:
            return "critical"


@dataclass
class CategoryScore:
    """Aggregated score for a category."""
    category: HealthCategory
    score: float
    grade: HealthGrade
    metrics: List[MetricScore]
    insights: List[str] = field(default_factory=list)
    
    @classmethod
    def from_metrics(cls, category: HealthCategory, metrics: List[MetricScore]) -> "CategoryScore":
        if not metrics:
            return cls(category=category, score=0.5, grade=HealthGrade.FAIR, metrics=[])
        
        total_weight = sum(m.weight for m in metrics)
        if total_weight == 0:
            score = 0.5
        else:
            score = sum(m.weighted_score for m in metrics) / total_weight
        
        grade = cls._score_to_grade(score)
        
        return cls(
            category=category,
            score=score,
            grade=grade,
            metrics=metrics,
        )
    
    @staticmethod
    def _score_to_grade(score: float) -> HealthGrade:
        if score >= 0.9:
            return HealthGrade.EXCELLENT
        elif score >= 0.75:
            return HealthGrade.GOOD
        elif score >= 0.6:
            return HealthGrade.FAIR
        elif score >= 0.4:
            return HealthGrade.POOR
        else:
            return HealthGrade.CRITICAL


@dataclass
class BusinessHealthScore:
    """Complete health score for a business."""
    business_id: str
    overall_score: float
    overall_grade: HealthGrade
    categories: Dict[HealthCategory, CategoryScore]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trend: str = "stable"  # "improving", "declining", "stable"
    percentile: Optional[int] = None  # Compared to similar businesses
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "overall_score": round(self.overall_score, 2),
            "overall_grade": self.overall_grade.value,
            "trend": self.trend,
            "percentile": self.percentile,
            "timestamp": self.timestamp.isoformat(),
            "categories": {
                cat.value: {
                    "score": round(cs.score, 2),
                    "grade": cs.grade.value,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "score": round(m.score, 2),
                            "status": m.status,
                            "trend": m.trend,
                        }
                        for m in cs.metrics
                    ],
                }
                for cat, cs in self.categories.items()
            },
            "insights": self.insights,
            "recommendations": self.recommendations,
        }


class MetricScorer:
    """Scores individual metrics."""
    
    @staticmethod
    def score_against_target(
        value: float,
        target: float,
        higher_is_better: bool = True,
        tolerance: float = 0.1,
    ) -> float:
        """Score a value against a target."""
        if target == 0:
            return 0.5
        
        ratio = value / target
        
        if higher_is_better:
            if ratio >= 1.0:
                return min(1.0, 0.9 + (ratio - 1) * 0.1)
            else:
                return max(0.0, ratio * 0.9)
        else:
            if ratio <= 1.0:
                return min(1.0, 0.9 + (1 - ratio) * 0.1)
            else:
                return max(0.0, (2 - ratio) * 0.5)
    
    @staticmethod
    def score_percentage(value: float, inverse: bool = False) -> float:
        """Score a percentage value (0-100)."""
        normalized = max(0, min(100, value)) / 100
        return 1 - normalized if inverse else normalized
    
    @staticmethod
    def score_growth_rate(rate: float, target_rate: float = 0.1) -> float:
        """Score a growth rate."""
        if rate >= target_rate:
            return min(1.0, 0.7 + (rate / target_rate) * 0.3)
        elif rate >= 0:
            return 0.5 + (rate / target_rate) * 0.2
        else:
            return max(0.0, 0.5 + rate)  # Negative growth reduces score
    
    @staticmethod
    def score_ratio(
        value: float,
        healthy_min: float,
        healthy_max: float,
    ) -> float:
        """Score a ratio within a healthy range."""
        if healthy_min <= value <= healthy_max:
            return 1.0
        elif value < healthy_min:
            return max(0.0, value / healthy_min)
        else:
            return max(0.0, 1 - (value - healthy_max) / healthy_max)


class HealthScoreCalculator:
    """
    Calculates comprehensive business health scores.
    
    Features:
    - Multi-category scoring
    - Trend analysis
    - Benchmark comparison
    - Actionable insights
    """
    
    # Default category weights
    CATEGORY_WEIGHTS = {
        HealthCategory.FINANCIAL: 0.30,
        HealthCategory.OPERATIONAL: 0.20,
        HealthCategory.CUSTOMER: 0.25,
        HealthCategory.GROWTH: 0.15,
        HealthCategory.RISK: 0.10,
    }
    
    def __init__(
        self,
        category_weights: Optional[Dict[HealthCategory, float]] = None,
    ):
        self.category_weights = category_weights or self.CATEGORY_WEIGHTS.copy()
        self._historical_scores: Dict[str, List[Tuple[datetime, float]]] = {}
    
    async def calculate(
        self,
        business_id: str,
        metrics: Dict[str, Any],
        targets: Optional[Dict[str, float]] = None,
        benchmarks: Optional[Dict[str, float]] = None,
    ) -> BusinessHealthScore:
        """
        Calculate health score for a business.
        
        Args:
            business_id: Business identifier
            metrics: Dictionary of metric names to values
            targets: Optional targets for metrics
            benchmarks: Optional industry benchmarks
            
        Returns:
            Complete health score
        """
        targets = targets or {}
        benchmarks = benchmarks or {}
        
        # Score each category
        category_scores = {}
        
        # Financial Health
        category_scores[HealthCategory.FINANCIAL] = self._score_financial(
            metrics, targets, benchmarks
        )
        
        # Operational Health
        category_scores[HealthCategory.OPERATIONAL] = self._score_operational(
            metrics, targets, benchmarks
        )
        
        # Customer Health
        category_scores[HealthCategory.CUSTOMER] = self._score_customer(
            metrics, targets, benchmarks
        )
        
        # Growth Health
        category_scores[HealthCategory.GROWTH] = self._score_growth(
            metrics, targets, benchmarks
        )
        
        # Risk Health
        category_scores[HealthCategory.RISK] = self._score_risk(
            metrics, targets, benchmarks
        )
        
        # Calculate overall score
        overall_score = sum(
            cs.score * self.category_weights.get(cat, 0.2)
            for cat, cs in category_scores.items()
        )
        
        # Determine grade
        if overall_score >= 0.9:
            overall_grade = HealthGrade.EXCELLENT
        elif overall_score >= 0.75:
            overall_grade = HealthGrade.GOOD
        elif overall_score >= 0.6:
            overall_grade = HealthGrade.FAIR
        elif overall_score >= 0.4:
            overall_grade = HealthGrade.POOR
        else:
            overall_grade = HealthGrade.CRITICAL
        
        # Calculate trend
        trend = self._calculate_trend(business_id, overall_score)
        
        # Generate insights and recommendations
        insights = self._generate_insights(category_scores)
        recommendations = self._generate_recommendations(category_scores, metrics)
        
        # Store for trend analysis
        self._store_score(business_id, overall_score)
        
        return BusinessHealthScore(
            business_id=business_id,
            overall_score=overall_score,
            overall_grade=overall_grade,
            categories=category_scores,
            trend=trend,
            insights=insights,
            recommendations=recommendations,
        )
    
    def _score_financial(
        self,
        metrics: Dict[str, Any],
        targets: Dict[str, float],
        benchmarks: Dict[str, float],
    ) -> CategoryScore:
        """Score financial health metrics."""
        scored_metrics = []
        
        # Revenue
        if "revenue" in metrics:
            target = targets.get("revenue", metrics.get("revenue", 1) * 1.1)
            score = MetricScorer.score_against_target(
                metrics["revenue"], target, higher_is_better=True
            )
            scored_metrics.append(MetricScore(
                name="revenue",
                value=metrics["revenue"],
                score=score,
                weight=0.25,
                target=target,
                category=HealthCategory.FINANCIAL,
            ))
        
        # Profit Margin
        if "profit_margin" in metrics:
            target = targets.get("profit_margin", 0.2)  # 20% target
            score = MetricScorer.score_against_target(
                metrics["profit_margin"], target, higher_is_better=True
            )
            scored_metrics.append(MetricScore(
                name="profit_margin",
                value=metrics["profit_margin"],
                score=score,
                weight=0.30,
                target=target,
                category=HealthCategory.FINANCIAL,
            ))
        
        # Cash Flow
        if "cash_flow" in metrics:
            # Positive cash flow is good
            score = 0.8 if metrics["cash_flow"] > 0 else 0.3
            scored_metrics.append(MetricScore(
                name="cash_flow",
                value=metrics["cash_flow"],
                score=score,
                weight=0.20,
                category=HealthCategory.FINANCIAL,
            ))
        
        # Expenses Ratio
        if "expenses" in metrics and "revenue" in metrics:
            ratio = metrics["expenses"] / max(metrics["revenue"], 1)
            score = MetricScorer.score_ratio(ratio, 0.5, 0.8)
            scored_metrics.append(MetricScore(
                name="expense_ratio",
                value=ratio,
                score=score,
                weight=0.25,
                category=HealthCategory.FINANCIAL,
            ))
        
        return CategoryScore.from_metrics(HealthCategory.FINANCIAL, scored_metrics)
    
    def _score_operational(
        self,
        metrics: Dict[str, Any],
        targets: Dict[str, float],
        benchmarks: Dict[str, float],
    ) -> CategoryScore:
        """Score operational health metrics."""
        scored_metrics = []
        
        # Fulfillment Rate
        if "fulfillment_rate" in metrics:
            score = MetricScorer.score_percentage(metrics["fulfillment_rate"] * 100)
            scored_metrics.append(MetricScore(
                name="fulfillment_rate",
                value=metrics["fulfillment_rate"],
                score=score,
                weight=0.30,
                category=HealthCategory.OPERATIONAL,
            ))
        
        # Average Processing Time
        if "processing_time" in metrics:
            target = targets.get("processing_time", 24)  # 24 hours
            score = MetricScorer.score_against_target(
                metrics["processing_time"], target, higher_is_better=False
            )
            scored_metrics.append(MetricScore(
                name="processing_time",
                value=metrics["processing_time"],
                score=score,
                weight=0.25,
                target=target,
                category=HealthCategory.OPERATIONAL,
            ))
        
        # Inventory Turnover
        if "inventory_turnover" in metrics:
            target = targets.get("inventory_turnover", 12)  # 12x per year
            score = MetricScorer.score_against_target(
                metrics["inventory_turnover"], target, higher_is_better=True
            )
            scored_metrics.append(MetricScore(
                name="inventory_turnover",
                value=metrics["inventory_turnover"],
                score=score,
                weight=0.25,
                category=HealthCategory.OPERATIONAL,
            ))
        
        # Automation Rate
        if "automation_rate" in metrics:
            score = MetricScorer.score_percentage(metrics["automation_rate"] * 100)
            scored_metrics.append(MetricScore(
                name="automation_rate",
                value=metrics["automation_rate"],
                score=score,
                weight=0.20,
                category=HealthCategory.OPERATIONAL,
            ))
        
        return CategoryScore.from_metrics(HealthCategory.OPERATIONAL, scored_metrics)
    
    def _score_customer(
        self,
        metrics: Dict[str, Any],
        targets: Dict[str, float],
        benchmarks: Dict[str, float],
    ) -> CategoryScore:
        """Score customer health metrics."""
        scored_metrics = []
        
        # Customer Satisfaction (NPS or similar)
        if "customer_satisfaction" in metrics:
            score = MetricScorer.score_percentage(metrics["customer_satisfaction"])
            scored_metrics.append(MetricScore(
                name="customer_satisfaction",
                value=metrics["customer_satisfaction"],
                score=score,
                weight=0.30,
                category=HealthCategory.CUSTOMER,
            ))
        
        # Churn Rate (lower is better)
        if "churn_rate" in metrics:
            score = MetricScorer.score_percentage(metrics["churn_rate"] * 100, inverse=True)
            scored_metrics.append(MetricScore(
                name="churn_rate",
                value=metrics["churn_rate"],
                score=score,
                weight=0.25,
                category=HealthCategory.CUSTOMER,
            ))
        
        # Customer Lifetime Value
        if "customer_ltv" in metrics:
            target = targets.get("customer_ltv", metrics.get("customer_ltv", 100) * 1.2)
            score = MetricScorer.score_against_target(
                metrics["customer_ltv"], target, higher_is_better=True
            )
            scored_metrics.append(MetricScore(
                name="customer_ltv",
                value=metrics["customer_ltv"],
                score=score,
                weight=0.25,
                target=target,
                category=HealthCategory.CUSTOMER,
            ))
        
        # Repeat Customer Rate
        if "repeat_rate" in metrics:
            score = MetricScorer.score_percentage(metrics["repeat_rate"] * 100)
            scored_metrics.append(MetricScore(
                name="repeat_rate",
                value=metrics["repeat_rate"],
                score=score,
                weight=0.20,
                category=HealthCategory.CUSTOMER,
            ))
        
        return CategoryScore.from_metrics(HealthCategory.CUSTOMER, scored_metrics)
    
    def _score_growth(
        self,
        metrics: Dict[str, Any],
        targets: Dict[str, float],
        benchmarks: Dict[str, float],
    ) -> CategoryScore:
        """Score growth health metrics."""
        scored_metrics = []
        
        # Revenue Growth
        if "revenue_growth" in metrics:
            target_rate = targets.get("revenue_growth", 0.15)  # 15% growth
            score = MetricScorer.score_growth_rate(metrics["revenue_growth"], target_rate)
            scored_metrics.append(MetricScore(
                name="revenue_growth",
                value=metrics["revenue_growth"],
                score=score,
                weight=0.35,
                target=target_rate,
                category=HealthCategory.GROWTH,
            ))
        
        # Customer Growth
        if "customer_growth" in metrics:
            target_rate = targets.get("customer_growth", 0.10)
            score = MetricScorer.score_growth_rate(metrics["customer_growth"], target_rate)
            scored_metrics.append(MetricScore(
                name="customer_growth",
                value=metrics["customer_growth"],
                score=score,
                weight=0.30,
                target=target_rate,
                category=HealthCategory.GROWTH,
            ))
        
        # Market Share
        if "market_share" in metrics:
            score = MetricScorer.score_percentage(metrics["market_share"] * 100)
            scored_metrics.append(MetricScore(
                name="market_share",
                value=metrics["market_share"],
                score=score,
                weight=0.20,
                category=HealthCategory.GROWTH,
            ))
        
        # Product Expansion
        if "product_count" in metrics:
            target = targets.get("product_count", 10)
            score = min(1.0, metrics["product_count"] / target)
            scored_metrics.append(MetricScore(
                name="product_count",
                value=metrics["product_count"],
                score=score,
                weight=0.15,
                target=target,
                category=HealthCategory.GROWTH,
            ))
        
        return CategoryScore.from_metrics(HealthCategory.GROWTH, scored_metrics)
    
    def _score_risk(
        self,
        metrics: Dict[str, Any],
        targets: Dict[str, float],
        benchmarks: Dict[str, float],
    ) -> CategoryScore:
        """Score risk health metrics."""
        scored_metrics = []
        
        # Supplier Concentration
        if "supplier_concentration" in metrics:
            # Lower concentration is less risky
            score = MetricScorer.score_percentage(
                metrics["supplier_concentration"] * 100, inverse=True
            )
            scored_metrics.append(MetricScore(
                name="supplier_concentration",
                value=metrics["supplier_concentration"],
                score=score,
                weight=0.25,
                category=HealthCategory.RISK,
            ))
        
        # Customer Concentration
        if "customer_concentration" in metrics:
            score = MetricScorer.score_percentage(
                metrics["customer_concentration"] * 100, inverse=True
            )
            scored_metrics.append(MetricScore(
                name="customer_concentration",
                value=metrics["customer_concentration"],
                score=score,
                weight=0.25,
                category=HealthCategory.RISK,
            ))
        
        # Debt Ratio
        if "debt_ratio" in metrics:
            score = MetricScorer.score_ratio(metrics["debt_ratio"], 0.0, 0.5)
            scored_metrics.append(MetricScore(
                name="debt_ratio",
                value=metrics["debt_ratio"],
                score=score,
                weight=0.25,
                category=HealthCategory.RISK,
            ))
        
        # Legal/Compliance Issues
        if "compliance_score" in metrics:
            score = MetricScorer.score_percentage(metrics["compliance_score"])
            scored_metrics.append(MetricScore(
                name="compliance_score",
                value=metrics["compliance_score"],
                score=score,
                weight=0.25,
                category=HealthCategory.RISK,
            ))
        
        return CategoryScore.from_metrics(HealthCategory.RISK, scored_metrics)
    
    def _calculate_trend(self, business_id: str, current_score: float) -> str:
        """Calculate score trend based on history."""
        history = self._historical_scores.get(business_id, [])
        
        if len(history) < 3:
            return "stable"
        
        # Get recent scores
        recent = [score for _, score in history[-7:]]
        
        # Simple trend detection
        if len(recent) >= 3:
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            if second_half > first_half + 0.05:
                return "improving"
            elif second_half < first_half - 0.05:
                return "declining"
        
        return "stable"
    
    def _store_score(self, business_id: str, score: float) -> None:
        """Store score for trend analysis."""
        if business_id not in self._historical_scores:
            self._historical_scores[business_id] = []
        
        self._historical_scores[business_id].append((datetime.utcnow(), score))
        
        # Keep last 90 days
        cutoff = datetime.utcnow() - timedelta(days=90)
        self._historical_scores[business_id] = [
            (dt, s) for dt, s in self._historical_scores[business_id]
            if dt > cutoff
        ]
    
    def _generate_insights(
        self, category_scores: Dict[HealthCategory, CategoryScore]
    ) -> List[str]:
        """Generate insights from scores."""
        insights = []
        
        for cat, cs in category_scores.items():
            if cs.score < 0.6:
                insights.append(
                    f"{cat.value.title()} health is critical (grade: {cs.grade.value})"
                )
            elif cs.score >= 0.9:
                insights.append(
                    f"Excellent {cat.value} health performance"
                )
            
            # Find worst performing metric
            if cs.metrics:
                worst = min(cs.metrics, key=lambda m: m.score)
                if worst.score < 0.5:
                    insights.append(
                        f"{worst.name} needs attention (score: {worst.score:.0%})"
                    )
        
        return insights[:5]  # Limit to top 5
    
    def _generate_recommendations(
        self,
        category_scores: Dict[HealthCategory, CategoryScore],
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check each category
        for cat, cs in category_scores.items():
            if cs.score < 0.6:
                if cat == HealthCategory.FINANCIAL:
                    recommendations.append(
                        "Review pricing strategy and reduce unnecessary expenses"
                    )
                elif cat == HealthCategory.CUSTOMER:
                    recommendations.append(
                        "Implement customer feedback loop and loyalty program"
                    )
                elif cat == HealthCategory.OPERATIONAL:
                    recommendations.append(
                        "Automate manual processes and optimize fulfillment"
                    )
                elif cat == HealthCategory.GROWTH:
                    recommendations.append(
                        "Explore new customer acquisition channels"
                    )
                elif cat == HealthCategory.RISK:
                    recommendations.append(
                        "Diversify supplier and customer base"
                    )
        
        return recommendations[:3]  # Limit to top 3


# Global calculator instance
health_calculator = HealthScoreCalculator()


def get_health_calculator() -> HealthScoreCalculator:
    """Get the global health calculator instance."""
    return health_calculator
