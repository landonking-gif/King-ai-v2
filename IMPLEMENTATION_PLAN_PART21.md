# Implementation Plan Part 21: Business Portfolio Management

| Field | Value |
|-------|-------|
| Module | Portfolio Management System |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 3 (Database), Part 4 (Business Unit), Part 19 (Lifecycle) |

---

## 1. Scope

This module implements portfolio management for multiple businesses:

- **Portfolio Container** - Manage collections of businesses
- **Resource Allocation** - Distribute resources across businesses
- **Performance Aggregation** - Roll-up metrics across portfolio
- **Rebalancing** - Optimize portfolio based on performance
- **Cross-Business Analytics** - Compare and analyze businesses

---

## 2. Tasks

### Task 21.1: Portfolio Models

**File: `src/business/portfolio_models.py`**

```python
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
```

---

### Task 21.2: Portfolio Manager

**File: `src/business/portfolio.py`**

```python
"""
Portfolio Manager - Manage business portfolios.
"""
import uuid
from datetime import datetime
from typing import Optional
from src.business.portfolio_models import (
    Portfolio, PortfolioStatus, AllocationStrategy,
    ResourceAllocation, PortfolioMetrics, BusinessPerformance,
    RebalanceAction, RebalanceReport
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """Manage portfolios of businesses."""

    def __init__(self):
        self._portfolios: dict[str, Portfolio] = {}
        self._business_data: dict[str, dict] = {}  # Mock business data

    async def create_portfolio(
        self,
        name: str,
        owner_id: str,
        strategy: AllocationStrategy = AllocationStrategy.EQUAL,
        total_budget: float = 0.0,
    ) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name=name,
            owner_id=owner_id,
            strategy=strategy,
            total_budget=total_budget,
        )
        self._portfolios[portfolio.id] = portfolio
        logger.info(f"Created portfolio: {portfolio.id}")
        return portfolio

    async def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get a portfolio by ID."""
        return self._portfolios.get(portfolio_id)

    async def get_portfolios_for_owner(self, owner_id: str) -> list[Portfolio]:
        """Get all portfolios for an owner."""
        return [p for p in self._portfolios.values() if p.owner_id == owner_id]

    async def update_portfolio(
        self,
        portfolio_id: str,
        name: str = None,
        status: PortfolioStatus = None,
        strategy: AllocationStrategy = None,
        total_budget: float = None,
    ) -> Optional[Portfolio]:
        """Update portfolio settings."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        if name:
            portfolio.name = name
        if status:
            portfolio.status = status
        if strategy:
            portfolio.strategy = strategy
        if total_budget is not None:
            portfolio.total_budget = total_budget
        
        portfolio.updated_at = datetime.utcnow()
        return portfolio

    async def add_business(
        self, portfolio_id: str, business_id: str, weight: float = None
    ) -> bool:
        """Add a business to a portfolio."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return False
        
        portfolio.add_business(business_id, weight)
        
        # Rebalance if using equal strategy
        if portfolio.strategy == AllocationStrategy.EQUAL:
            await self._rebalance_equal(portfolio)
        
        return True

    async def remove_business(self, portfolio_id: str, business_id: str) -> bool:
        """Remove a business from a portfolio."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return False
        
        portfolio.remove_business(business_id)
        
        # Rebalance remaining
        if portfolio.strategy == AllocationStrategy.EQUAL:
            await self._rebalance_equal(portfolio)
        
        return True

    async def set_allocation(
        self,
        portfolio_id: str,
        business_id: str,
        weight: float = None,
        priority: int = None,
        locked: bool = None,
    ) -> bool:
        """Set allocation for a business."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio or business_id not in portfolio.business_ids:
            return False

        alloc = portfolio.allocations.get(business_id)
        if not alloc:
            return False

        if weight is not None:
            alloc.weight = weight
            alloc.budget_percent = weight * 100
            alloc.attention_hours = weight * 40
        if priority is not None:
            alloc.priority = priority
        if locked is not None:
            alloc.locked = locked

        portfolio.updated_at = datetime.utcnow()
        return True

    async def calculate_metrics(self, portfolio_id: str) -> Optional[PortfolioMetrics]:
        """Calculate aggregated portfolio metrics."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        metrics = PortfolioMetrics()
        performances = []

        for biz_id in portfolio.business_ids:
            biz_data = await self._get_business_metrics(biz_id)
            if biz_data:
                metrics.total_revenue += biz_data.get("revenue", 0)
                metrics.total_expenses += biz_data.get("expenses", 0)
                metrics.total_customers += biz_data.get("customers", 0)
                
                perf = BusinessPerformance(
                    business_id=biz_id,
                    business_name=biz_data.get("name", "Unknown"),
                    revenue=biz_data.get("revenue", 0),
                    profit=biz_data.get("profit", 0),
                    margin=biz_data.get("margin", 0),
                    growth_rate=biz_data.get("growth_rate", 0),
                    health_score=biz_data.get("health_score", 50),
                    stage=biz_data.get("stage", "unknown"),
                )
                performances.append(perf)

        metrics.total_profit = metrics.total_revenue - metrics.total_expenses
        metrics.active_businesses = len(portfolio.business_ids)

        if performances:
            metrics.average_margin = sum(p.margin for p in performances) / len(performances)
            metrics.growth_rate = sum(p.growth_rate for p in performances) / len(performances)
            
            # Find best and worst performers
            sorted_perf = sorted(performances, key=lambda x: x.profit, reverse=True)
            if sorted_perf:
                metrics.best_performer_id = sorted_perf[0].business_id
                metrics.worst_performer_id = sorted_perf[-1].business_id

        return metrics

    async def get_performance_ranking(
        self, portfolio_id: str
    ) -> list[BusinessPerformance]:
        """Get businesses ranked by performance."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return []

        performances = []
        for biz_id in portfolio.business_ids:
            biz_data = await self._get_business_metrics(biz_id)
            if biz_data:
                perf = BusinessPerformance(
                    business_id=biz_id,
                    business_name=biz_data.get("name", "Unknown"),
                    revenue=biz_data.get("revenue", 0),
                    profit=biz_data.get("profit", 0),
                    margin=biz_data.get("margin", 0),
                    growth_rate=biz_data.get("growth_rate", 0),
                    health_score=biz_data.get("health_score", 50),
                    stage=biz_data.get("stage", "unknown"),
                )
                performances.append(perf)

        # Rank by composite score
        for i, perf in enumerate(sorted(
            performances,
            key=lambda x: (x.profit * 0.4 + x.growth_rate * 0.3 + x.health_score * 0.3),
            reverse=True
        )):
            perf.rank = i + 1

        return sorted(performances, key=lambda x: x.rank)

    async def recommend_rebalance(self, portfolio_id: str) -> Optional[RebalanceReport]:
        """Generate rebalancing recommendations."""
        portfolio = self._portfolios.get(portfolio_id)
        if not portfolio:
            return None

        report = RebalanceReport(
            portfolio_id=portfolio_id,
            timestamp=datetime.utcnow(),
            strategy_used=portfolio.strategy,
        )

        performances = await self.get_performance_ranking(portfolio_id)
        if not performances:
            return report

        # Calculate recommended weights based on strategy
        if portfolio.strategy == AllocationStrategy.PERFORMANCE:
            weights = await self._calculate_performance_weights(performances)
        elif portfolio.strategy == AllocationStrategy.GROWTH:
            weights = await self._calculate_growth_weights(performances)
        elif portfolio.strategy == AllocationStrategy.REVENUE:
            weights = await self._calculate_revenue_weights(performances)
        else:
            weights = {p.business_id: 1.0 / len(performances) for p in performances}

        for perf in performances:
            current = portfolio.allocations.get(perf.business_id)
            if not current:
                continue

            recommended = weights.get(perf.business_id, current.weight)
            diff = recommended - current.weight

            if current.locked:
                action = "maintain"
                reason = "Allocation is locked"
            elif abs(diff) < 0.05:
                action = "maintain"
                reason = "Within acceptable range"
            elif diff > 0:
                action = "increase"
                reason = f"Performance rank: {perf.rank}"
            else:
                action = "decrease"
                reason = f"Underperforming (rank {perf.rank})"

            report.actions.append(RebalanceAction(
                business_id=perf.business_id,
                action=action,
                current_weight=current.weight,
                recommended_weight=recommended,
                reason=reason,
            ))
            report.total_reallocation += abs(diff)

        return report

    async def apply_rebalance(self, report: RebalanceReport) -> bool:
        """Apply a rebalancing report."""
        portfolio = self._portfolios.get(report.portfolio_id)
        if not portfolio:
            return False

        for action in report.actions:
            if action.action != "maintain":
                alloc = portfolio.allocations.get(action.business_id)
                if alloc and not alloc.locked:
                    alloc.weight = action.recommended_weight
                    alloc.budget_percent = action.recommended_weight * 100
                    alloc.attention_hours = action.recommended_weight * 40

        report.applied = True
        portfolio.updated_at = datetime.utcnow()
        return True

    async def _rebalance_equal(self, portfolio: Portfolio):
        """Rebalance to equal weights."""
        if not portfolio.business_ids:
            return
        
        weight = 1.0 / len(portfolio.business_ids)
        for biz_id in portfolio.business_ids:
            alloc = portfolio.allocations.get(biz_id)
            if alloc and not alloc.locked:
                alloc.weight = weight
                alloc.budget_percent = weight * 100
                alloc.attention_hours = weight * 40

    async def _calculate_performance_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on performance."""
        total_score = sum(
            p.profit * 0.5 + p.health_score * 0.5
            for p in performances
        )
        if total_score <= 0:
            return {p.business_id: 1.0 / len(performances) for p in performances}
        
        return {
            p.business_id: (p.profit * 0.5 + p.health_score * 0.5) / total_score
            for p in performances
        }

    async def _calculate_growth_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on growth potential."""
        # Favor high growth + early stage
        scores = {}
        for p in performances:
            stage_bonus = {"ideation": 1.5, "validation": 1.3, "growth": 1.2}.get(p.stage, 1.0)
            scores[p.business_id] = max(0, p.growth_rate) * stage_bonus + 10
        
        total = sum(scores.values())
        return {bid: s / total for bid, s in scores.items()}

    async def _calculate_revenue_weights(
        self, performances: list[BusinessPerformance]
    ) -> dict[str, float]:
        """Calculate weights based on revenue."""
        total = sum(max(0, p.revenue) for p in performances)
        if total <= 0:
            return {p.business_id: 1.0 / len(performances) for p in performances}
        return {p.business_id: max(0, p.revenue) / total for p in performances}

    async def _get_business_metrics(self, business_id: str) -> dict:
        """Get metrics for a business (mock implementation)."""
        # In real implementation, fetch from business unit or analytics
        return self._business_data.get(business_id, {
            "name": f"Business {business_id[:8]}",
            "revenue": 10000,
            "expenses": 7000,
            "profit": 3000,
            "margin": 30,
            "customers": 100,
            "growth_rate": 5.0,
            "health_score": 70,
            "stage": "growth",
        })

    def set_business_data(self, business_id: str, data: dict):
        """Set mock business data for testing."""
        self._business_data[business_id] = data
```

---

### Task 21.3: Portfolio API Routes

**File: `src/api/routes/portfolio.py`**

```python
"""
Portfolio API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.business.portfolio_models import AllocationStrategy, PortfolioStatus
from src.business.portfolio import PortfolioManager
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/portfolios", tags=["portfolios"])

_manager: Optional[PortfolioManager] = None


def get_manager() -> PortfolioManager:
    global _manager
    if _manager is None:
        _manager = PortfolioManager()
    return _manager


class CreatePortfolioRequest(BaseModel):
    name: str
    owner_id: str
    strategy: str = "equal"
    total_budget: float = 0.0


class UpdatePortfolioRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    strategy: Optional[str] = None
    total_budget: Optional[float] = None


class AddBusinessRequest(BaseModel):
    business_id: str
    weight: Optional[float] = None


class SetAllocationRequest(BaseModel):
    weight: Optional[float] = None
    priority: Optional[int] = None
    locked: Optional[bool] = None


@router.post("/")
async def create_portfolio(
    req: CreatePortfolioRequest,
    manager: PortfolioManager = Depends(get_manager),
):
    """Create a new portfolio."""
    try:
        strategy = AllocationStrategy(req.strategy)
    except ValueError:
        strategy = AllocationStrategy.EQUAL
    
    portfolio = await manager.create_portfolio(
        name=req.name,
        owner_id=req.owner_id,
        strategy=strategy,
        total_budget=req.total_budget,
    )
    
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "status": portfolio.status.value,
    }


@router.get("/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Get a portfolio by ID."""
    portfolio = await manager.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")
    
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "status": portfolio.status.value,
        "strategy": portfolio.strategy.value,
        "business_count": portfolio.business_count,
        "total_budget": portfolio.total_budget,
        "businesses": portfolio.business_ids,
        "allocations": {
            bid: {
                "weight": a.weight,
                "budget_percent": a.budget_percent,
                "priority": a.priority,
                "locked": a.locked,
            }
            for bid, a in portfolio.allocations.items()
        },
    }


@router.get("/owner/{owner_id}")
async def get_owner_portfolios(
    owner_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Get all portfolios for an owner."""
    portfolios = await manager.get_portfolios_for_owner(owner_id)
    return {
        "portfolios": [
            {
                "id": p.id,
                "name": p.name,
                "status": p.status.value,
                "business_count": p.business_count,
            }
            for p in portfolios
        ]
    }


@router.patch("/{portfolio_id}")
async def update_portfolio(
    portfolio_id: str,
    req: UpdatePortfolioRequest,
    manager: PortfolioManager = Depends(get_manager),
):
    """Update portfolio settings."""
    status = None
    if req.status:
        try:
            status = PortfolioStatus(req.status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {req.status}")
    
    strategy = None
    if req.strategy:
        try:
            strategy = AllocationStrategy(req.strategy)
        except ValueError:
            raise HTTPException(400, f"Invalid strategy: {req.strategy}")
    
    portfolio = await manager.update_portfolio(
        portfolio_id,
        name=req.name,
        status=status,
        strategy=strategy,
        total_budget=req.total_budget,
    )
    
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")
    
    return {"status": "updated"}


@router.post("/{portfolio_id}/businesses")
async def add_business(
    portfolio_id: str,
    req: AddBusinessRequest,
    manager: PortfolioManager = Depends(get_manager),
):
    """Add a business to a portfolio."""
    success = await manager.add_business(
        portfolio_id, req.business_id, req.weight
    )
    if not success:
        raise HTTPException(404, "Portfolio not found")
    return {"status": "added"}


@router.delete("/{portfolio_id}/businesses/{business_id}")
async def remove_business(
    portfolio_id: str,
    business_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Remove a business from a portfolio."""
    success = await manager.remove_business(portfolio_id, business_id)
    if not success:
        raise HTTPException(404, "Portfolio or business not found")
    return {"status": "removed"}


@router.put("/{portfolio_id}/businesses/{business_id}/allocation")
async def set_allocation(
    portfolio_id: str,
    business_id: str,
    req: SetAllocationRequest,
    manager: PortfolioManager = Depends(get_manager),
):
    """Set allocation for a business."""
    success = await manager.set_allocation(
        portfolio_id,
        business_id,
        weight=req.weight,
        priority=req.priority,
        locked=req.locked,
    )
    if not success:
        raise HTTPException(404, "Portfolio or business not found")
    return {"status": "updated"}


@router.get("/{portfolio_id}/metrics")
async def get_metrics(
    portfolio_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Get aggregated portfolio metrics."""
    metrics = await manager.calculate_metrics(portfolio_id)
    if not metrics:
        raise HTTPException(404, "Portfolio not found")
    
    return {
        "total_revenue": metrics.total_revenue,
        "total_expenses": metrics.total_expenses,
        "total_profit": metrics.total_profit,
        "net_margin": metrics.net_margin,
        "average_margin": metrics.average_margin,
        "total_customers": metrics.total_customers,
        "active_businesses": metrics.active_businesses,
        "growth_rate": metrics.growth_rate,
        "best_performer": metrics.best_performer_id,
        "worst_performer": metrics.worst_performer_id,
    }


@router.get("/{portfolio_id}/ranking")
async def get_ranking(
    portfolio_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Get performance ranking of businesses."""
    rankings = await manager.get_performance_ranking(portfolio_id)
    return {
        "rankings": [
            {
                "rank": r.rank,
                "business_id": r.business_id,
                "name": r.business_name,
                "revenue": r.revenue,
                "profit": r.profit,
                "margin": r.margin,
                "growth_rate": r.growth_rate,
                "health_score": r.health_score,
                "stage": r.stage,
            }
            for r in rankings
        ]
    }


@router.get("/{portfolio_id}/rebalance")
async def recommend_rebalance(
    portfolio_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Get rebalancing recommendations."""
    report = await manager.recommend_rebalance(portfolio_id)
    if not report:
        raise HTTPException(404, "Portfolio not found")
    
    return {
        "portfolio_id": report.portfolio_id,
        "strategy": report.strategy_used.value,
        "total_reallocation": report.total_reallocation,
        "actions": [
            {
                "business_id": a.business_id,
                "action": a.action,
                "current_weight": a.current_weight,
                "recommended_weight": a.recommended_weight,
                "reason": a.reason,
            }
            for a in report.actions
        ],
    }


@router.post("/{portfolio_id}/rebalance")
async def apply_rebalance(
    portfolio_id: str,
    manager: PortfolioManager = Depends(get_manager),
):
    """Apply rebalancing recommendations."""
    report = await manager.recommend_rebalance(portfolio_id)
    if not report:
        raise HTTPException(404, "Portfolio not found")
    
    success = await manager.apply_rebalance(report)
    if not success:
        raise HTTPException(500, "Failed to apply rebalancing")
    
    return {"status": "rebalanced", "changes": len(report.actions)}
```

---

### Task 21.4: Tests

**File: `tests/test_portfolio.py`**

```python
"""Tests for Portfolio Management."""
import pytest
from src.business.portfolio_models import (
    Portfolio, PortfolioStatus, AllocationStrategy
)
from src.business.portfolio import PortfolioManager


@pytest.fixture
def manager():
    return PortfolioManager()


class TestPortfolioManager:
    @pytest.mark.asyncio
    async def test_create_portfolio(self, manager):
        portfolio = await manager.create_portfolio(
            name="Test Portfolio",
            owner_id="user_1",
        )
        
        assert portfolio.id is not None
        assert portfolio.name == "Test Portfolio"
        assert portfolio.status == PortfolioStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_add_business(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        
        success = await manager.add_business(portfolio.id, "biz_1")
        
        assert success is True
        assert "biz_1" in portfolio.business_ids

    @pytest.mark.asyncio
    async def test_equal_allocation(self, manager):
        portfolio = await manager.create_portfolio(
            "Test", "user_1", AllocationStrategy.EQUAL
        )
        
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        assert portfolio.allocations["biz_1"].weight == 0.5
        assert portfolio.allocations["biz_2"].weight == 0.5

    @pytest.mark.asyncio
    async def test_remove_business(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        success = await manager.remove_business(portfolio.id, "biz_1")
        
        assert success is True
        assert "biz_1" not in portfolio.business_ids

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        manager.set_business_data("biz_1", {
            "name": "Business 1",
            "revenue": 50000,
            "expenses": 30000,
            "profit": 20000,
            "margin": 40,
            "customers": 500,
            "growth_rate": 10,
            "health_score": 85,
            "stage": "growth",
        })
        
        metrics = await manager.calculate_metrics(portfolio.id)
        
        assert metrics.total_revenue == 50000
        assert metrics.total_profit == 20000
        assert metrics.total_customers == 500

    @pytest.mark.asyncio
    async def test_performance_ranking(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        manager.set_business_data("biz_1", {
            "name": "Business 1",
            "revenue": 10000,
            "profit": 5000,
            "margin": 50,
            "growth_rate": 5,
            "health_score": 70,
            "stage": "growth",
        })
        manager.set_business_data("biz_2", {
            "name": "Business 2",
            "revenue": 20000,
            "profit": 8000,
            "margin": 40,
            "growth_rate": 15,
            "health_score": 90,
            "stage": "scaling",
        })
        
        rankings = await manager.get_performance_ranking(portfolio.id)
        
        assert len(rankings) == 2
        assert rankings[0].rank == 1

    @pytest.mark.asyncio
    async def test_recommend_rebalance(self, manager):
        portfolio = await manager.create_portfolio(
            "Test", "user_1", AllocationStrategy.PERFORMANCE
        )
        await manager.add_business(portfolio.id, "biz_1")
        await manager.add_business(portfolio.id, "biz_2")
        
        report = await manager.recommend_rebalance(portfolio.id)
        
        assert report is not None
        assert len(report.actions) == 2

    @pytest.mark.asyncio
    async def test_locked_allocation(self, manager):
        portfolio = await manager.create_portfolio("Test", "user_1")
        await manager.add_business(portfolio.id, "biz_1")
        
        await manager.set_allocation(portfolio.id, "biz_1", locked=True)
        
        assert portfolio.allocations["biz_1"].locked is True

    @pytest.mark.asyncio
    async def test_get_portfolios_for_owner(self, manager):
        await manager.create_portfolio("P1", "user_1")
        await manager.create_portfolio("P2", "user_1")
        await manager.create_portfolio("P3", "user_2")
        
        portfolios = await manager.get_portfolios_for_owner("user_1")
        
        assert len(portfolios) == 2
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Portfolio creation | Create with strategy and budget |
| Business management | Add/remove businesses |
| Allocation strategies | Equal, performance, growth, revenue |
| Metrics aggregation | Roll-up across businesses |
| Performance ranking | Rank by composite score |
| Rebalancing | Recommend and apply changes |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/business/portfolio_models.py` | Data models for portfolios |
| `src/business/portfolio.py` | Portfolio management logic |
| `src/api/routes/portfolio.py` | REST API endpoints |
| `tests/test_portfolio.py` | Unit tests |
