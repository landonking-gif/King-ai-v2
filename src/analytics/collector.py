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
from src.database.connection import get_db
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
