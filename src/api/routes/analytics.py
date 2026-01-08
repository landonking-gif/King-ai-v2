"""
Analytics API Routes.
"""
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from src.agents.analytics import AnalyticsAgent
from src.utils.logging import get_logger
from threading import Lock

logger = get_logger(__name__)
router = APIRouter(tags=["analytics"])

_agent: Optional[AnalyticsAgent] = None
_agent_lock = Lock()


def get_agent() -> AnalyticsAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            # Double-check locking pattern
            if _agent is None:
                _agent = AnalyticsAgent()
    return _agent


class RecordMetricRequest(BaseModel):
    metric_name: str
    value: float
    dimensions: Optional[dict] = None


class SetTargetRequest(BaseModel):
    business_id: str
    kpi_name: str
    target: float = Field(..., gt=0)


@router.get("/dashboard/{business_id}")
async def get_dashboard(business_id: str, agent: AnalyticsAgent = Depends(get_agent)):
    """Get dashboard metrics."""
    result = await agent.get_dashboard_metrics(business_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.get("/kpis/{business_id}")
async def get_kpis(business_id: str, agent: AnalyticsAgent = Depends(get_agent)):
    """Get all KPIs."""
    result = await agent.get_kpis(business_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.post("/kpis/target")
async def set_kpi_target(req: SetTargetRequest, agent: AnalyticsAgent = Depends(get_agent)):
    """Set a KPI target."""
    agent.set_kpi_target(req.business_id, req.kpi_name, req.target)
    return {"status": "ok"}


@router.get("/trends/{business_id}/{metric_name}")
async def get_trends(
    business_id: str,
    metric_name: str,
    days: int = 30,
    granularity: str = "daily",
    agent: AnalyticsAgent = Depends(get_agent)
):
    """Get trend data."""
    result = await agent.get_trends(business_id, metric_name, days, granularity)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.get("/compare/{business_id}")
async def compare_periods(
    business_id: str,
    metrics: str = Query(..., description="Comma-separated metric names"),
    days: int = 30,
    agent: AnalyticsAgent = Depends(get_agent)
):
    """Compare current vs previous period."""
    metric_list = [m.strip() for m in metrics.split(",")]
    result = await agent.compare_periods(business_id, metric_list, days)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.get("/alerts/{business_id}")
async def check_alerts(business_id: str, agent: AnalyticsAgent = Depends(get_agent)):
    """Check for metric alerts."""
    result = await agent.check_alerts(business_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.get("/report/{business_id}")
async def generate_report(
    business_id: str,
    days: int = 30,
    title: Optional[str] = None,
    agent: AnalyticsAgent = Depends(get_agent)
):
    """Generate a business report."""
    result = await agent.generate_report(business_id, days, title)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return result.get("data")


@router.get("/pl")
async def get_pl_report(period: str = "30d", agent: AnalyticsAgent = Depends(get_agent)):
    """
    Get Profit & Loss report across the entire empire.
    Aggregates revenue and expenses from all business units.
    """
    from src.database.connection import get_db_yield
    from src.database.models import BusinessUnit
    from sqlalchemy import select, func
    
    # We'll use get_db_yield manually here if not using Depends in the route signature
    # But since we're in a route, we can just add db: AsyncSession = Depends(get_db_yield)
    # I'll update the signature to be cleaner.
    return await _get_pl_data()

async def _get_pl_data():
    from src.database.connection import get_db_ctx
    from src.database.models import BusinessUnit
    from sqlalchemy import select, func
    
    async with get_db_ctx() as db:
        result = await db.execute(
            select(
                func.sum(BusinessUnit.total_revenue).label("revenue"),
                func.sum(BusinessUnit.total_expenses).label("expenses")
            )
        )
        row = result.first()
        
        revenue = (row.revenue or 0.0) if row else 0.0
        expenses = (row.expenses or 0.0) if row else 0.0
        profit = revenue - expenses
        
        return {
            "success": True,
            "data": {
                "revenue": revenue,
                "expenses": expenses,
                "profit": profit,
                "margin": (profit / revenue * 100) if revenue > 0 else 0,
                "period": "30d", # Placeholder for now as per dashboard request
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        }

@router.post("/metrics")
async def record_metric(req: RecordMetricRequest, agent: AnalyticsAgent = Depends(get_agent)):
    """Record a metric value."""
    result = await agent.record_metric(req.metric_name, req.value, req.dimensions)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Unknown error"))
    return {"status": "recorded"}
