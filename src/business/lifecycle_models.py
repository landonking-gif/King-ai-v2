"""
Business Lifecycle Models.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional, Callable


class LifecycleStage(Enum):
    """Business lifecycle stages."""
    IDEATION = "ideation"
    VALIDATION = "validation"
    LAUNCH = "launch"
    GROWTH = "growth"
    SCALE = "scale"
    MATURITY = "maturity"
    DECLINE = "decline"
    EXIT = "exit"


class MilestoneType(Enum):
    """Types of business milestones."""
    REVENUE = "revenue"
    CUSTOMER = "customer"
    PRODUCT = "product"
    OPERATIONAL = "operational"
    FUNDING = "funding"
    TEAM = "team"


class TransitionTrigger(Enum):
    """What triggers a lifecycle transition."""
    MANUAL = "manual"
    MILESTONE = "milestone"
    METRIC = "metric"
    TIME = "time"
    AUTOMATED = "automated"


@dataclass
class Milestone:
    """Business milestone definition."""
    id: str
    name: str
    milestone_type: MilestoneType
    description: str
    target_value: float
    current_value: float = 0.0
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    target_date: Optional[date] = None
    metadata: dict = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 0.0
        return min(100, (self.current_value / self.target_value) * 100)

    @property
    def is_overdue(self) -> bool:
        if not self.target_date or self.achieved:
            return False
        return date.today() > self.target_date


@dataclass
class StageRequirement:
    """Requirements to enter a lifecycle stage."""
    metric_name: str
    operator: str  # gte, lte, eq, gt, lt
    value: float
    description: str = ""

    def evaluate(self, current_value: float) -> bool:
        ops = {
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
        }
        return ops.get(self.operator, lambda a, b: False)(current_value, self.value)


@dataclass
class LifecycleTransition:
    """Record of a lifecycle transition."""
    id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    trigger: TransitionTrigger
    triggered_by: str  # milestone_id, metric_name, or user_id
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LifecycleState:
    """Current lifecycle state of a business."""
    business_id: str
    current_stage: LifecycleStage
    entered_at: datetime
    milestones: list[Milestone] = field(default_factory=list)
    transitions: list[LifecycleTransition] = field(default_factory=list)
    health_score: float = 100.0
    next_stage: Optional[LifecycleStage] = None
    blockers: list[str] = field(default_factory=list)


# Stage configuration with requirements and milestones
STAGE_CONFIG = {
    LifecycleStage.IDEATION: {
        "description": "Idea development and initial planning",
        "typical_duration_days": 30,
        "requirements": [],
        "milestones": [
            {"name": "Business Plan Complete", "type": MilestoneType.OPERATIONAL, "target": 1},
            {"name": "Market Research Done", "type": MilestoneType.OPERATIONAL, "target": 1},
        ],
        "next_stages": [LifecycleStage.VALIDATION],
    },
    LifecycleStage.VALIDATION: {
        "description": "Market validation and MVP development",
        "typical_duration_days": 60,
        "requirements": [
            StageRequirement("business_plan_complete", "eq", 1, "Business plan required"),
        ],
        "milestones": [
            {"name": "MVP Launched", "type": MilestoneType.PRODUCT, "target": 1},
            {"name": "First 10 Customers", "type": MilestoneType.CUSTOMER, "target": 10},
            {"name": "First Revenue", "type": MilestoneType.REVENUE, "target": 1},
        ],
        "next_stages": [LifecycleStage.LAUNCH, LifecycleStage.IDEATION],
    },
    LifecycleStage.LAUNCH: {
        "description": "Official launch and initial traction",
        "typical_duration_days": 90,
        "requirements": [
            StageRequirement("customers", "gte", 10, "Minimum 10 customers"),
            StageRequirement("has_revenue", "eq", 1, "Must have revenue"),
        ],
        "milestones": [
            {"name": "100 Customers", "type": MilestoneType.CUSTOMER, "target": 100},
            {"name": "$1K MRR", "type": MilestoneType.REVENUE, "target": 1000},
        ],
        "next_stages": [LifecycleStage.GROWTH],
    },
    LifecycleStage.GROWTH: {
        "description": "Rapid growth and market expansion",
        "typical_duration_days": 180,
        "requirements": [
            StageRequirement("mrr", "gte", 1000, "Minimum $1K MRR"),
            StageRequirement("customers", "gte", 100, "Minimum 100 customers"),
        ],
        "milestones": [
            {"name": "1000 Customers", "type": MilestoneType.CUSTOMER, "target": 1000},
            {"name": "$10K MRR", "type": MilestoneType.REVENUE, "target": 10000},
            {"name": "Team of 5", "type": MilestoneType.TEAM, "target": 5},
        ],
        "next_stages": [LifecycleStage.SCALE],
    },
    LifecycleStage.SCALE: {
        "description": "Scaling operations and market dominance",
        "typical_duration_days": 365,
        "requirements": [
            StageRequirement("mrr", "gte", 10000, "Minimum $10K MRR"),
            StageRequirement("growth_rate", "gte", 10, "Minimum 10% monthly growth"),
        ],
        "milestones": [
            {"name": "$100K MRR", "type": MilestoneType.REVENUE, "target": 100000},
            {"name": "10K Customers", "type": MilestoneType.CUSTOMER, "target": 10000},
        ],
        "next_stages": [LifecycleStage.MATURITY],
    },
    LifecycleStage.MATURITY: {
        "description": "Stable, profitable operations",
        "typical_duration_days": None,
        "requirements": [
            StageRequirement("mrr", "gte", 100000, "Minimum $100K MRR"),
            StageRequirement("profit_margin", "gte", 20, "Minimum 20% margin"),
        ],
        "milestones": [],
        "next_stages": [LifecycleStage.EXIT, LifecycleStage.DECLINE],
    },
    LifecycleStage.DECLINE: {
        "description": "Declining metrics, requires intervention",
        "typical_duration_days": None,
        "requirements": [],
        "milestones": [],
        "next_stages": [LifecycleStage.GROWTH, LifecycleStage.EXIT],
    },
    LifecycleStage.EXIT: {
        "description": "Business exit (sale, shutdown, or transition)",
        "typical_duration_days": None,
        "requirements": [],
        "milestones": [],
        "next_stages": [],
    },
}
