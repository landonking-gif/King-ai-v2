"""
Analytics Agent - Data-driven insight generation.
Expert in pattern recognition, market trends, and KPI monitoring.
Integrates with Google Analytics 4 for web analytics data.
"""

import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, date, timezone
from typing import Any, Optional
from src.agents.base import SubAgent
from src.utils.metrics import TASKS_EXECUTED
from src.analytics.models import (
    KPI, Alert, Report, TimeSeries, TimeGranularity, MetricCategory
)
from src.analytics.collector import MetricsCollector, STANDARD_METRICS
from src.integrations.google_analytics_client import GoogleAnalyticsClient, get_ga_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnalyticsAgent(SubAgent):
    """
    Data scientist agent for business intelligence.
    Integrates with internal metrics and Google Analytics 4.
    """
    name = "analytics"
    description = "Performs data analysis and provides actionable business insights with Google Analytics integration."
    
    # Function calling schema for LLM integration
    FUNCTION_SCHEMA = {
        "name": "analytics",
        "description": "Analyze business metrics, KPIs, and web analytics. Integrates with Google Analytics 4 for traffic and e-commerce data.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "dashboard", "kpis", "trends", "compare_periods",
                        "alerts", "generate_report", "record_metric",
                        "ga_traffic", "ga_ecommerce", "ga_top_products",
                        "ga_sources", "ga_funnel", "ga_realtime"
                    ],
                    "description": "The analytics operation to perform"
                },
                "business_id": {
                    "type": "string",
                    "description": "The business unit ID"
                },
                "metric_name": {
                    "type": "string",
                    "description": "Name of the metric to analyze"
                },
                "days": {
                    "type": "integer",
                    "default": 30,
                    "description": "Number of days to analyze"
                },
                "granularity": {
                    "type": "string",
                    "enum": ["hourly", "daily", "weekly", "monthly"],
                    "default": "daily"
                }
            },
            "required": ["action", "business_id"]
        }
    }
    
    def __init__(self, ga_property_id: str = None):
        super().__init__()
        self.collector = MetricsCollector()
        self._kpi_targets: dict[str, dict[str, float]] = {}  # business_id -> {kpi: target}
        self._alerts: list[Alert] = []
        self._ga_clients: dict[str, GoogleAnalyticsClient] = {}
        self._default_ga_property = ga_property_id

    def get_ga_client(self, business_id: str) -> GoogleAnalyticsClient:
        """Get or create GA4 client for a business."""
        if business_id not in self._ga_clients:
            # In production, property ID would come from business config
            self._ga_clients[business_id] = get_ga_client(self._default_ga_property)
        return self._ga_clients[business_id]

    def set_kpi_target(self, business_id: str, kpi_name: str, target: float):
        """Set a KPI target for a business."""
        if business_id not in self._kpi_targets:
            self._kpi_targets[business_id] = {}
        self._kpi_targets[business_id][kpi_name] = target

    # =========================================================================
    # Google Analytics 4 Integration Methods
    # =========================================================================
    
    async def get_ga_traffic_metrics(
        self,
        business_id: str,
        days: int = 30,
    ) -> dict:
        """Get Google Analytics traffic metrics."""
        try:
            ga = self.get_ga_client(business_id)
            start_date = date.today() - timedelta(days=days)
            result = await ga.get_traffic_metrics(start_date=start_date)
            
            if result.get("success"):
                TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return result
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

    async def get_ga_ecommerce_metrics(
        self,
        business_id: str,
        days: int = 30,
    ) -> dict:
        """Get Google Analytics e-commerce metrics."""
        try:
            ga = self.get_ga_client(business_id)
            start_date = date.today() - timedelta(days=days)
            result = await ga.get_ecommerce_metrics(start_date=start_date)
            
            if result.get("success"):
                TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return result
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

    async def get_ga_top_products(
        self,
        business_id: str,
        days: int = 30,
        limit: int = 10,
    ) -> dict:
        """Get top performing products from Google Analytics."""
        try:
            ga = self.get_ga_client(business_id)
            start_date = date.today() - timedelta(days=days)
            return await ga.get_top_products(start_date=start_date, limit=limit)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_ga_traffic_sources(
        self,
        business_id: str,
        days: int = 30,
    ) -> dict:
        """Get traffic acquisition breakdown from Google Analytics."""
        try:
            ga = self.get_ga_client(business_id)
            start_date = date.today() - timedelta(days=days)
            return await ga.get_traffic_sources(start_date=start_date)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_ga_conversion_funnel(
        self,
        business_id: str,
        days: int = 30,
    ) -> dict:
        """Get e-commerce conversion funnel from Google Analytics."""
        try:
            ga = self.get_ga_client(business_id)
            start_date = date.today() - timedelta(days=days)
            return await ga.get_conversion_funnel(start_date=start_date)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_ga_realtime(self, business_id: str) -> dict:
        """Get real-time active users from Google Analytics."""
        try:
            ga = self.get_ga_client(business_id)
            return await ga.get_realtime_users()
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Internal Analytics Methods
    # =========================================================================

    async def get_dashboard_metrics(self, business_id: str) -> dict:
        """Get key metrics for dashboard display."""
        try:
            metrics = await self.collector.collect(business_id)
            
            dashboard = {
                "revenue": {
                    "total": metrics.get("revenue_total").value if "revenue_total" in metrics else 0,
                    "orders": metrics.get("orders_count").value if "orders_count" in metrics else 0,
                    "aov": metrics.get("average_order_value").value if "average_order_value" in metrics else 0,
                },
                "traffic": {
                    "page_views": metrics.get("page_views").value if "page_views" in metrics else 0,
                    "visitors": metrics.get("unique_visitors").value if "unique_visitors" in metrics else 0,
                    "bounce_rate": metrics.get("bounce_rate").value if "bounce_rate" in metrics else 0,
                },
                "conversion": {
                    "rate": metrics.get("conversion_rate").value if "conversion_rate" in metrics else 0,
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            return {"success": True, "data": dashboard}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_kpis(self, business_id: str) -> dict:
        """Get all KPIs with current status."""
        try:
            metrics = await self.collector.collect(business_id)
            targets = self._kpi_targets.get(business_id, {})
            
            # Calculate comparison period (previous period)
            now = datetime.now(timezone.utc)
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
            
            return {"success": True, "data": {"kpis": kpis}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_trends(
        self,
        business_id: str,
        metric_name: str,
        days: int = 30,
        granularity: str = "daily",
    ) -> dict:
        """Get trend data for a metric."""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            gran = TimeGranularity(granularity)
            
            ts = await self.collector.get_time_series(
                business_id, metric_name, start_time, end_time, gran
            )
            
            definition = self.collector.get_definition(metric_name)
            
            return {
                "success": True,
                "data": {
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
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def compare_periods(
        self,
        business_id: str,
        metric_names: list[str],
        current_days: int = 30,
    ) -> dict:
        """Compare current period vs previous period."""
        try:
            now = datetime.now(timezone.utc)
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
            
            return {"success": True, "data": {"comparisons": comparisons}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check_alerts(self, business_id: str) -> dict:
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
                        triggered_at=datetime.now(timezone.utc),
                    )
                    new_alerts.append(new_alert)
                    self._alerts.append(new_alert)
            
            return {
                "success": True,
                "data": {
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
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_report(
        self,
        business_id: str,
        period_days: int = 30,
        title: str = None,
    ) -> dict:
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
                generated_at=datetime.now(timezone.utc),
                sections=[
                    {"name": "Overview", "data": dashboard.get("data", {})},
                    {"name": "KPIs", "data": kpis_result.get("data", {})},
                ],
                insights=self._generate_insights(
                    dashboard.get("data", {}),
                    kpis_result.get("data", {}).get("kpis", [])
                ),
            )
            
            return {
                "success": True,
                "data": {
                    "report_id": report.id,
                    "title": report.title,
                    "period": f"{start_date} to {end_date}",
                    "sections": report.sections,
                    "insights": report.insights,
                    "generated_at": report.generated_at.isoformat(),
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
    ) -> dict:
        """Record a metric value."""
        try:
            await self.collector.record(metric_name, value, dimensions)
            return {"success": True, "message": "Metric recorded"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute(self, task: dict) -> dict:
        """
        Executes a data analysis task.
        Supports both legacy format and new action-based format.
        """
        # Check if it's a new action-based task
        action = task.get("action")
        if action:
            business_id = task.get("business_id", "")
            
            if action == "dashboard":
                return await self.get_dashboard_metrics(business_id)
            elif action == "kpis":
                return await self.get_kpis(business_id)
            elif action == "trends":
                metric = task.get("metric")
                if not metric:
                    return {"success": False, "error": "Missing required parameter: metric"}
                return await self.get_trends(
                    business_id, metric, task.get("days", 30)
                )
            elif action == "compare":
                metrics = task.get("metrics")
                if not metrics:
                    return {"success": False, "error": "Missing required parameter: metrics"}
                return await self.compare_periods(
                    business_id, metrics, task.get("days", 30)
                )
            elif action == "alerts":
                return await self.check_alerts(business_id)
            elif action == "report":
                return await self.generate_report(business_id, task.get("days", 30))
            
            return {"success": False, "error": f"Unknown action: {action}"}
        
        # Legacy format - use LLM for general analysis
        description = task.get("description", "Analysis task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: DATA ANALYSIS
        {description}
        
        ### RAW DATA:
        {input_data}
        
        ### INSTRUCTION:
        Identify trends, anomalies, and actionable insights.
        Provide data-backed conclusions for strategic decision making.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "data_analysis"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
