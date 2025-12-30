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
