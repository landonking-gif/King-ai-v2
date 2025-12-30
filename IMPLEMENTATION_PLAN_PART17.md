# Implementation Plan Part 17: Analytics Sub-Agent

| Field | Value |
|-------|-------|
| Module | Analytics & Metrics Collection Sub-Agent |
| Priority | High |
| Estimated Effort | 4-5 hours |
| Dependencies | Part 3 (Database), Part 13-16 (Commerce/Finance) |

---

## 1. Scope

This module implements comprehensive analytics and metrics tracking:

- **Metrics Collector** - Aggregate data from all business sources
- **KPI Tracking** - Define and monitor key performance indicators
- **Trend Analysis** - Time-series analysis, forecasting
- **Report Generation** - Automated business reports
- **Alerts** - Threshold-based notifications

---

## 2. Tasks

### Task 17.1: Metrics Models

**File: `src/analytics/models.py`**

```python
"""
Analytics Data Models.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional


class MetricType(Enum):
    COUNTER = "counter"      # Cumulative count
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution
    RATE = "rate"            # Per-time-unit


class MetricCategory(Enum):
    REVENUE = "revenue"
    TRAFFIC = "traffic"
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    OPERATIONS = "operations"
    FINANCIAL = "financial"


class TimeGranularity(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class MetricDefinition:
    name: str
    display_name: str
    category: MetricCategory
    metric_type: MetricType
    unit: str = ""
    description: str = ""
    higher_is_better: bool = True
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class MetricValue:
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class TimeSeries:
    metric_name: str
    values: list[tuple[datetime, float]]
    granularity: TimeGranularity
    start_time: datetime
    end_time: datetime

    @property
    def count(self) -> int:
        return len(self.values)

    def average(self) -> float:
        if not self.values:
            return 0.0
        return sum(v for _, v in self.values) / len(self.values)

    def trend(self) -> float:
        """Calculate trend as percentage change."""
        if len(self.values) < 2:
            return 0.0
        first = self.values[0][1]
        last = self.values[-1][1]
        if first == 0:
            return 0.0
        return ((last - first) / first) * 100


@dataclass
class KPI:
    name: str
    current_value: float
    target_value: float
    previous_value: float
    unit: str = ""
    trend_percent: float = 0.0
    status: str = "on_track"  # on_track, at_risk, behind

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 0.0
        return (self.current_value / self.target_value) * 100

    @property
    def change_percent(self) -> float:
        if self.previous_value == 0:
            return 0.0
        return ((self.current_value - self.previous_value) / self.previous_value) * 100


@dataclass
class Alert:
    id: str
    metric_name: str
    level: str  # warning, critical
    message: str
    current_value: float
    threshold: float
    triggered_at: datetime
    acknowledged: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Report:
    id: str
    title: str
    business_id: str
    period_start: date
    period_end: date
    generated_at: datetime
    sections: list[dict] = field(default_factory=list)
    kpis: list[KPI] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
```

---

### Task 17.2: Metrics Collector

**File: `src/analytics/collector.py`**

```python
"""
Metrics Collector - Aggregate data from various sources.
"""
import asyncio
from datetime import datetime, timedelta, date
from typing import Any, Optional
from collections import defaultdict
from src.analytics.models import (
    MetricDefinition, MetricValue, TimeSeries, MetricType,
    MetricCategory, TimeGranularity
)
from src.database.connection import get_db_session
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Standard metric definitions
STANDARD_METRICS: dict[str, MetricDefinition] = {
    "revenue_total": MetricDefinition(
        name="revenue_total",
        display_name="Total Revenue",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.COUNTER,
        unit="USD",
        higher_is_better=True,
    ),
    "orders_count": MetricDefinition(
        name="orders_count",
        display_name="Order Count",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.COUNTER,
        higher_is_better=True,
    ),
    "average_order_value": MetricDefinition(
        name="average_order_value",
        display_name="Average Order Value",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.GAUGE,
        unit="USD",
        higher_is_better=True,
    ),
    "conversion_rate": MetricDefinition(
        name="conversion_rate",
        display_name="Conversion Rate",
        category=MetricCategory.CONVERSION,
        metric_type=MetricType.GAUGE,
        unit="%",
        higher_is_better=True,
        warning_threshold=2.0,
        critical_threshold=1.0,
    ),
    "page_views": MetricDefinition(
        name="page_views",
        display_name="Page Views",
        category=MetricCategory.TRAFFIC,
        metric_type=MetricType.COUNTER,
        higher_is_better=True,
    ),
    "unique_visitors": MetricDefinition(
        name="unique_visitors",
        display_name="Unique Visitors",
        category=MetricCategory.TRAFFIC,
        metric_type=MetricType.COUNTER,
        higher_is_better=True,
    ),
    "bounce_rate": MetricDefinition(
        name="bounce_rate",
        display_name="Bounce Rate",
        category=MetricCategory.ENGAGEMENT,
        metric_type=MetricType.GAUGE,
        unit="%",
        higher_is_better=False,
        warning_threshold=60.0,
        critical_threshold=80.0,
    ),
    "customer_count": MetricDefinition(
        name="customer_count",
        display_name="Total Customers",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.GAUGE,
        higher_is_better=True,
    ),
    "churn_rate": MetricDefinition(
        name="churn_rate",
        display_name="Churn Rate",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.GAUGE,
        unit="%",
        higher_is_better=False,
        warning_threshold=5.0,
        critical_threshold=10.0,
    ),
    "mrr": MetricDefinition(
        name="mrr",
        display_name="Monthly Recurring Revenue",
        category=MetricCategory.REVENUE,
        metric_type=MetricType.GAUGE,
        unit="USD",
        higher_is_better=True,
    ),
}


class MetricsCollector:
    """Collect and aggregate metrics from various data sources."""

    def __init__(self):
        self.metrics = STANDARD_METRICS.copy()
        self._cache: dict[str, list[MetricValue]] = defaultdict(list)
        self._sources: dict[str, callable] = {}

    def register_metric(self, definition: MetricDefinition):
        """Register a custom metric definition."""
        self.metrics[definition.name] = definition

    def register_source(self, name: str, collector_fn: callable):
        """Register a data source collector function."""
        self._sources[name] = collector_fn

    async def collect(
        self, business_id: str, metric_names: list[str] = None
    ) -> dict[str, MetricValue]:
        """Collect current values for specified metrics."""
        results = {}
        names = metric_names or list(self.metrics.keys())

        for source_name, collector_fn in self._sources.items():
            try:
                source_data = await collector_fn(business_id)
                for name, value in source_data.items():
                    if name in names:
                        results[name] = MetricValue(
                            metric_name=name,
                            value=value,
                            timestamp=datetime.utcnow(),
                            dimensions={"source": source_name},
                        )
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")

        return results

    async def record(self, metric_name: str, value: float, dimensions: dict = None):
        """Record a metric value."""
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric: {metric_name}")
            return

        mv = MetricValue(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.utcnow(),
            dimensions=dimensions or {},
        )
        self._cache[metric_name].append(mv)

        # Trim cache to last 1000 values per metric
        if len(self._cache[metric_name]) > 1000:
            self._cache[metric_name] = self._cache[metric_name][-1000:]

    async def get_time_series(
        self,
        business_id: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        granularity: TimeGranularity = TimeGranularity.DAILY,
    ) -> TimeSeries:
        """Get time series data for a metric."""
        # In production, this would query from database
        cached = self._cache.get(metric_name, [])
        
        filtered = [
            (mv.timestamp, mv.value)
            for mv in cached
            if start_time <= mv.timestamp <= end_time
        ]

        return TimeSeries(
            metric_name=metric_name,
            values=sorted(filtered, key=lambda x: x[0]),
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
        )

    async def aggregate(
        self,
        business_id: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "sum",  # sum, avg, min, max, count
    ) -> float:
        """Aggregate metric values over a time period."""
        ts = await self.get_time_series(business_id, metric_name, start_time, end_time)
        
        if not ts.values:
            return 0.0

        values = [v for _, v in ts.values]
        
        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return float(len(values))
        
        return 0.0

    def get_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get metric definition."""
        return self.metrics.get(metric_name)

    def list_metrics(self, category: MetricCategory = None) -> list[MetricDefinition]:
        """List all metric definitions."""
        if category:
            return [m for m in self.metrics.values() if m.category == category]
        return list(self.metrics.values())
```

---

### Task 17.3: Analytics Agent

**File: `src/agents/analytics.py`**

```python
"""
Analytics Agent - KPI tracking, trend analysis, reporting.
"""
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, date
from typing import Any, Optional
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.analytics.models import (
    KPI, Alert, Report, TimeSeries, TimeGranularity, MetricCategory
)
from src.analytics.collector import MetricsCollector, STANDARD_METRICS
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnalyticsAgent(BaseAgent):
    """Agent for analytics, KPI tracking, and reporting."""

    def __init__(self):
        super().__init__(
            name="Analytics Agent",
            capabilities=[AgentCapability.ANALYSIS]
        )
        self.collector = MetricsCollector()
        self._kpi_targets: dict[str, dict[str, float]] = {}  # business_id -> {kpi: target}
        self._alerts: list[Alert] = []

    def set_kpi_target(self, business_id: str, kpi_name: str, target: float):
        """Set a KPI target for a business."""
        if business_id not in self._kpi_targets:
            self._kpi_targets[business_id] = {}
        self._kpi_targets[business_id][kpi_name] = target

    async def get_dashboard_metrics(self, business_id: str) -> AgentResult:
        """Get key metrics for dashboard display."""
        try:
            metrics = await self.collector.collect(business_id)
            
            dashboard = {
                "revenue": {
                    "total": metrics.get("revenue_total", {}).value if "revenue_total" in metrics else 0,
                    "orders": metrics.get("orders_count", {}).value if "orders_count" in metrics else 0,
                    "aov": metrics.get("average_order_value", {}).value if "average_order_value" in metrics else 0,
                },
                "traffic": {
                    "page_views": metrics.get("page_views", {}).value if "page_views" in metrics else 0,
                    "visitors": metrics.get("unique_visitors", {}).value if "unique_visitors" in metrics else 0,
                    "bounce_rate": metrics.get("bounce_rate", {}).value if "bounce_rate" in metrics else 0,
                },
                "conversion": {
                    "rate": metrics.get("conversion_rate", {}).value if "conversion_rate" in metrics else 0,
                },
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            return AgentResult(success=True, data=dashboard)
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def get_kpis(self, business_id: str) -> AgentResult:
        """Get all KPIs with current status."""
        try:
            metrics = await self.collector.collect(business_id)
            targets = self._kpi_targets.get(business_id, {})
            
            # Calculate comparison period (previous period)
            now = datetime.utcnow()
            period_start = now - timedelta(days=30)
            prev_start = period_start - timedelta(days=30)
            
            kpis = []
            for name, definition in STANDARD_METRICS.items():
                current = metrics.get(name)
                if not current:
                    continue
                
                target = targets.get(name, current.value * 1.1)  # Default 10% growth target
                
                # Get previous period value
                prev_ts = await self.collector.get_time_series(
                    business_id, name, prev_start, period_start
                )
                prev_value = prev_ts.average() if prev_ts.values else current.value
                
                # Determine status
                progress = (current.value / target * 100) if target > 0 else 0
                if progress >= 90:
                    status = "on_track"
                elif progress >= 70:
                    status = "at_risk"
                else:
                    status = "behind"
                
                kpi = KPI(
                    name=definition.display_name,
                    current_value=current.value,
                    target_value=target,
                    previous_value=prev_value,
                    unit=definition.unit,
                    status=status,
                )
                kpis.append({
                    "name": kpi.name,
                    "current": kpi.current_value,
                    "target": kpi.target_value,
                    "previous": kpi.previous_value,
                    "unit": kpi.unit,
                    "progress": round(kpi.progress_percent, 1),
                    "change": round(kpi.change_percent, 1),
                    "status": kpi.status,
                })
            
            return AgentResult(success=True, data={"kpis": kpis})
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def get_trends(
        self,
        business_id: str,
        metric_name: str,
        days: int = 30,
        granularity: str = "daily",
    ) -> AgentResult:
        """Get trend data for a metric."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            gran = TimeGranularity(granularity)
            
            ts = await self.collector.get_time_series(
                business_id, metric_name, start_time, end_time, gran
            )
            
            definition = self.collector.get_definition(metric_name)
            
            return AgentResult(
                success=True,
                data={
                    "metric": metric_name,
                    "display_name": definition.display_name if definition else metric_name,
                    "values": [
                        {"timestamp": t.isoformat(), "value": v}
                        for t, v in ts.values
                    ],
                    "average": round(ts.average(), 2),
                    "trend_percent": round(ts.trend(), 2),
                    "count": ts.count,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def compare_periods(
        self,
        business_id: str,
        metric_names: list[str],
        current_days: int = 30,
    ) -> AgentResult:
        """Compare current period vs previous period."""
        try:
            now = datetime.utcnow()
            current_start = now - timedelta(days=current_days)
            prev_start = current_start - timedelta(days=current_days)
            
            comparisons = []
            for name in metric_names:
                current_ts = await self.collector.get_time_series(
                    business_id, name, current_start, now
                )
                prev_ts = await self.collector.get_time_series(
                    business_id, name, prev_start, current_start
                )
                
                current_val = current_ts.average()
                prev_val = prev_ts.average()
                
                change = 0.0
                if prev_val > 0:
                    change = ((current_val - prev_val) / prev_val) * 100
                
                definition = self.collector.get_definition(name)
                
                comparisons.append({
                    "metric": name,
                    "display_name": definition.display_name if definition else name,
                    "current": round(current_val, 2),
                    "previous": round(prev_val, 2),
                    "change_percent": round(change, 2),
                    "improved": (change > 0) == (definition.higher_is_better if definition else True),
                })
            
            return AgentResult(success=True, data={"comparisons": comparisons})
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def check_alerts(self, business_id: str) -> AgentResult:
        """Check metrics against thresholds and generate alerts."""
        try:
            metrics = await self.collector.collect(business_id)
            new_alerts = []
            
            for name, value in metrics.items():
                definition = self.collector.get_definition(name)
                if not definition:
                    continue
                
                alert = None
                if definition.critical_threshold is not None:
                    if definition.higher_is_better:
                        if value.value < definition.critical_threshold:
                            alert = ("critical", definition.critical_threshold)
                    else:
                        if value.value > definition.critical_threshold:
                            alert = ("critical", definition.critical_threshold)
                
                if not alert and definition.warning_threshold is not None:
                    if definition.higher_is_better:
                        if value.value < definition.warning_threshold:
                            alert = ("warning", definition.warning_threshold)
                    else:
                        if value.value > definition.warning_threshold:
                            alert = ("warning", definition.warning_threshold)
                
                if alert:
                    level, threshold = alert
                    new_alert = Alert(
                        id=str(uuid.uuid4()),
                        metric_name=name,
                        level=level,
                        message=f"{definition.display_name} is {level}: {value.value} (threshold: {threshold})",
                        current_value=value.value,
                        threshold=threshold,
                        triggered_at=datetime.utcnow(),
                    )
                    new_alerts.append(new_alert)
                    self._alerts.append(new_alert)
            
            return AgentResult(
                success=True,
                data={
                    "alerts": [
                        {
                            "id": a.id,
                            "metric": a.metric_name,
                            "level": a.level,
                            "message": a.message,
                            "value": a.current_value,
                            "threshold": a.threshold,
                            "triggered_at": a.triggered_at.isoformat(),
                        }
                        for a in new_alerts
                    ],
                    "count": len(new_alerts),
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def generate_report(
        self,
        business_id: str,
        period_days: int = 30,
        title: str = None,
    ) -> AgentResult:
        """Generate a business report."""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=period_days)
            
            # Gather data
            dashboard = await self.get_dashboard_metrics(business_id)
            kpis_result = await self.get_kpis(business_id)
            
            # Build report
            report = Report(
                id=str(uuid.uuid4()),
                title=title or f"Business Report - {start_date} to {end_date}",
                business_id=business_id,
                period_start=start_date,
                period_end=end_date,
                generated_at=datetime.utcnow(),
                sections=[
                    {"name": "Overview", "data": dashboard.data if dashboard.success else {}},
                    {"name": "KPIs", "data": kpis_result.data if kpis_result.success else {}},
                ],
                insights=self._generate_insights(
                    dashboard.data if dashboard.success else {},
                    kpis_result.data.get("kpis", []) if kpis_result.success else []
                ),
            )
            
            return AgentResult(
                success=True,
                data={
                    "report_id": report.id,
                    "title": report.title,
                    "period": f"{start_date} to {end_date}",
                    "sections": report.sections,
                    "insights": report.insights,
                    "generated_at": report.generated_at.isoformat(),
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    def _generate_insights(self, dashboard: dict, kpis: list) -> list[str]:
        """Generate insights from data."""
        insights = []
        
        # Revenue insights
        revenue = dashboard.get("revenue", {})
        if revenue.get("total", 0) > 0:
            aov = revenue.get("aov", 0)
            if aov > 100:
                insights.append(f"Strong average order value at ${aov:.2f}")
            elif aov < 30:
                insights.append("Consider upselling strategies to increase order value")
        
        # Traffic insights
        traffic = dashboard.get("traffic", {})
        bounce = traffic.get("bounce_rate", 0)
        if bounce > 70:
            insights.append(f"High bounce rate ({bounce}%) - review landing pages")
        
        # KPI insights
        behind_kpis = [k for k in kpis if k.get("status") == "behind"]
        if behind_kpis:
            names = ", ".join(k["name"] for k in behind_kpis[:3])
            insights.append(f"KPIs needing attention: {names}")
        
        if not insights:
            insights.append("Business metrics are performing within expected ranges")
        
        return insights

    async def record_metric(
        self, metric_name: str, value: float, dimensions: dict = None
    ) -> AgentResult:
        """Record a metric value."""
        try:
            await self.collector.record(metric_name, value, dimensions)
            return AgentResult(success=True, message="Metric recorded")
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def execute(self, task: str, context: dict) -> AgentResult:
        action = context.get("action", "")
        business_id = context.get("business_id", "")
        
        if action == "dashboard":
            return await self.get_dashboard_metrics(business_id)
        elif action == "kpis":
            return await self.get_kpis(business_id)
        elif action == "trends":
            return await self.get_trends(
                business_id, context["metric"], context.get("days", 30)
            )
        elif action == "compare":
            return await self.compare_periods(
                business_id, context["metrics"], context.get("days", 30)
            )
        elif action == "alerts":
            return await self.check_alerts(business_id)
        elif action == "report":
            return await self.generate_report(business_id, context.get("days", 30))
        
        return AgentResult(success=False, message=f"Unknown action: {action}")
```

---

### Task 17.4: Analytics API Routes

**File: `src/api/routes/analytics.py`**

```python
"""
Analytics API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from src.agents.analytics import AnalyticsAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

_agent: Optional[AnalyticsAgent] = None


def get_agent() -> AnalyticsAgent:
    global _agent
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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/kpis/{business_id}")
async def get_kpis(business_id: str, agent: AnalyticsAgent = Depends(get_agent)):
    """Get all KPIs."""
    result = await agent.get_kpis(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/alerts/{business_id}")
async def check_alerts(business_id: str, agent: AnalyticsAgent = Depends(get_agent)):
    """Check for metric alerts."""
    result = await agent.check_alerts(business_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/report/{business_id}")
async def generate_report(
    business_id: str,
    days: int = 30,
    title: Optional[str] = None,
    agent: AnalyticsAgent = Depends(get_agent)
):
    """Generate a business report."""
    result = await agent.generate_report(business_id, days, title)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/metrics")
async def record_metric(req: RecordMetricRequest, agent: AnalyticsAgent = Depends(get_agent)):
    """Record a metric value."""
    result = await agent.record_metric(req.metric_name, req.value, req.dimensions)
    if not result.success:
        raise HTTPException(400, result.message)
    return {"status": "recorded"}
```

---

### Task 17.5: Tests

**File: `tests/test_analytics.py`**

```python
"""Tests for Analytics Agent."""
import pytest
from datetime import datetime, timedelta
from src.analytics.models import KPI, TimeSeries, TimeGranularity, MetricValue
from src.analytics.collector import MetricsCollector, STANDARD_METRICS
from src.agents.analytics import AnalyticsAgent


class TestKPI:
    def test_progress_percent(self):
        kpi = KPI(name="Revenue", current_value=8000, target_value=10000, previous_value=7000)
        assert kpi.progress_percent == 80.0

    def test_change_percent(self):
        kpi = KPI(name="Revenue", current_value=8000, target_value=10000, previous_value=7000)
        assert round(kpi.change_percent, 2) == 14.29


class TestTimeSeries:
    def test_average(self):
        ts = TimeSeries(
            metric_name="test",
            values=[(datetime.now(), 10), (datetime.now(), 20), (datetime.now(), 30)],
            granularity=TimeGranularity.DAILY,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert ts.average() == 20.0

    def test_trend(self):
        ts = TimeSeries(
            metric_name="test",
            values=[(datetime.now(), 100), (datetime.now(), 150)],
            granularity=TimeGranularity.DAILY,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert ts.trend() == 50.0


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_standard_metrics_loaded(self, collector):
        assert "revenue_total" in collector.metrics
        assert "conversion_rate" in collector.metrics

    @pytest.mark.asyncio
    async def test_record_metric(self, collector):
        await collector.record("revenue_total", 1000)
        assert len(collector._cache["revenue_total"]) == 1

    def test_list_metrics_by_category(self, collector):
        from src.analytics.models import MetricCategory
        revenue_metrics = collector.list_metrics(MetricCategory.REVENUE)
        assert len(revenue_metrics) > 0


class TestAnalyticsAgent:
    @pytest.fixture
    def agent(self):
        return AnalyticsAgent()

    def test_set_kpi_target(self, agent):
        agent.set_kpi_target("biz_1", "revenue_total", 100000)
        assert agent._kpi_targets["biz_1"]["revenue_total"] == 100000

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, agent):
        result = await agent.get_dashboard_metrics("biz_1")
        assert result.success
        assert "revenue" in result.data
        assert "traffic" in result.data

    @pytest.mark.asyncio
    async def test_generate_report(self, agent):
        result = await agent.generate_report("biz_1", 30, "Test Report")
        assert result.success
        assert result.data["title"] == "Test Report"
        assert "sections" in result.data
        assert "insights" in result.data

    @pytest.mark.asyncio
    async def test_record_metric(self, agent):
        result = await agent.record_metric("revenue_total", 5000)
        assert result.success
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Metrics collected | Standard business metrics tracked |
| KPIs calculated | Current/target/previous with status |
| Trends analyzed | Time series with averages and trends |
| Alerts generated | Threshold-based warnings/criticals |
| Reports created | Automated business reports |
| API functional | All endpoints return valid data |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/analytics/models.py` | Data models for metrics/KPIs |
| `src/analytics/collector.py` | Metrics collection/aggregation |
| `src/agents/analytics.py` | Analytics agent |
| `src/api/routes/analytics.py` | REST API endpoints |
| `tests/test_analytics.py` | Unit tests |
