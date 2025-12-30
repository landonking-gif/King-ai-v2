"""
Portfolio Data Models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    EQUAL = "equal"  # Distribute equally
    PERFORMANCE = "performance"  # Based on performance
    GROWTH = "growth"  # Prioritize growth potential
    REVENUE = "revenue"  # Based on revenue
    CUSTOM = "custom"  # Custom weights


class PortfolioStatus(Enum):
    """Portfolio status."""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class ResourceAllocation:
    """Resource allocation for a business."""
    business_id: str
    weight: float  # 0.0 to 1.0
    budget_percent: float
    attention_hours: float  # Hours per week
    priority: int  # 1-10
    locked: bool = False  # Prevent auto-rebalancing


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio metrics."""
    total_revenue: float = 0.0
    total_expenses: float = 0.0
    total_profit: float = 0.0
    average_margin: float = 0.0
    total_customers: int = 0
    active_businesses: int = 0
    growth_rate: float = 0.0
    best_performer_id: Optional[str] = None
    worst_performer_id: Optional[str] = None
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def net_margin(self) -> float:
        if self.total_revenue > 0:
            return (self.total_profit / self.total_revenue) * 100
        return 0.0


@dataclass
class BusinessPerformance:
    """Performance data for a single business."""
    business_id: str
    business_name: str
    revenue: float
    profit: float
    margin: float
    growth_rate: float
    health_score: float
    stage: str
    rank: int = 0


@dataclass
class Portfolio:
    """A portfolio of businesses."""
    id: str
    name: str
    owner_id: str
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    strategy: AllocationStrategy = AllocationStrategy.EQUAL
    business_ids: list[str] = field(default_factory=list)
    allocations: dict[str, ResourceAllocation] = field(default_factory=dict)
    total_budget: float = 0.0
    target_roi: float = 20.0  # Target ROI percentage
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def business_count(self) -> int:
        return len(self.business_ids)

    def add_business(self, business_id: str, weight: float = None):
        if business_id not in self.business_ids:
            self.business_ids.append(business_id)
            if weight is None:
                weight = 1.0 / len(self.business_ids)
            self.allocations[business_id] = ResourceAllocation(
                business_id=business_id,
                weight=weight,
                budget_percent=weight * 100,
                attention_hours=weight * 40,
                priority=5,
            )
            self.updated_at = datetime.utcnow()

    def remove_business(self, business_id: str):
        if business_id in self.business_ids:
            self.business_ids.remove(business_id)
            self.allocations.pop(business_id, None)
            self.updated_at = datetime.utcnow()


@dataclass
class RebalanceAction:
    """A rebalancing action."""
    business_id: str
    action: str  # increase, decrease, maintain
    current_weight: float
    recommended_weight: float
    reason: str


@dataclass
class RebalanceReport:
    """Report from portfolio rebalancing."""
    portfolio_id: str
    timestamp: datetime
    strategy_used: AllocationStrategy
    actions: list[RebalanceAction] = field(default_factory=list)
    total_reallocation: float = 0.0
    applied: bool = False
