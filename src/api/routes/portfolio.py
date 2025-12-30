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
