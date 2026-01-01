"""
Prometheus Metrics Exporter.
Exposes application metrics in Prometheus format.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import time
import asyncio
from functools import wraps

from src.utils.structured_logging import get_logger

logger = get_logger("metrics")


class MetricType(str, Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """Label for a metric."""
    name: str
    value: str


@dataclass
class MetricSample:
    """A single metric sample."""
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None


@dataclass
class HistogramBucket:
    """Histogram bucket."""
    le: float  # Less than or equal
    count: int = 0


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    help_text: str
    label_names: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # For histograms


class Counter:
    """Prometheus Counter metric."""
    
    def __init__(self, name: str, help_text: str, label_names: List[str] = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values: Dict[tuple, float] = {}
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def inc(self, labels: Dict[str, str] = None, value: float = 1.0) -> None:
        """Increment the counter."""
        labels = labels or {}
        key = self._label_key(labels)
        self._values[key] = self._values.get(key, 0.0) + value
    
    def get(self, labels: Dict[str, str] = None) -> float:
        """Get current value."""
        labels = labels or {}
        return self._values.get(self._label_key(labels), 0.0)
    
    def samples(self) -> List[MetricSample]:
        """Get all samples."""
        result = []
        for key, value in self._values.items():
            labels = dict(key)
            result.append(MetricSample(
                name=self.name,
                labels=labels,
                value=value,
            ))
        return result


class Gauge:
    """Prometheus Gauge metric."""
    
    def __init__(self, name: str, help_text: str, label_names: List[str] = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values: Dict[tuple, float] = {}
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """Set the gauge value."""
        labels = labels or {}
        self._values[self._label_key(labels)] = value
    
    def inc(self, labels: Dict[str, str] = None, value: float = 1.0) -> None:
        """Increment the gauge."""
        labels = labels or {}
        key = self._label_key(labels)
        self._values[key] = self._values.get(key, 0.0) + value
    
    def dec(self, labels: Dict[str, str] = None, value: float = 1.0) -> None:
        """Decrement the gauge."""
        labels = labels or {}
        key = self._label_key(labels)
        self._values[key] = self._values.get(key, 0.0) - value
    
    def get(self, labels: Dict[str, str] = None) -> float:
        """Get current value."""
        labels = labels or {}
        return self._values.get(self._label_key(labels), 0.0)
    
    def samples(self) -> List[MetricSample]:
        """Get all samples."""
        result = []
        for key, value in self._values.items():
            labels = dict(key)
            result.append(MetricSample(
                name=self.name,
                labels=labels,
                value=value,
            ))
        return result


class Histogram:
    """Prometheus Histogram metric."""
    
    DEFAULT_BUCKETS = [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    ]
    
    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
        buckets: List[float] = None,
    ):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._observations: Dict[tuple, List[float]] = {}
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value."""
        labels = labels or {}
        key = self._label_key(labels)
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append(value)
    
    def time(self, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        labels = labels or {}
        return _HistogramTimer(self, labels)
    
    def samples(self) -> List[MetricSample]:
        """Get all samples including bucket counts."""
        result = []
        
        for key, observations in self._observations.items():
            labels = dict(key)
            
            # Bucket counts
            for bucket in self.buckets:
                count = sum(1 for o in observations if o <= bucket)
                result.append(MetricSample(
                    name=f"{self.name}_bucket",
                    labels={**labels, "le": str(bucket)},
                    value=float(count),
                ))
            
            # +Inf bucket
            result.append(MetricSample(
                name=f"{self.name}_bucket",
                labels={**labels, "le": "+Inf"},
                value=float(len(observations)),
            ))
            
            # Sum
            result.append(MetricSample(
                name=f"{self.name}_sum",
                labels=labels,
                value=sum(observations),
            ))
            
            # Count
            result.append(MetricSample(
                name=f"{self.name}_count",
                labels=labels,
                value=float(len(observations)),
            ))
        
        return result


class _HistogramTimer:
    """Timer context manager for histograms."""
    
    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration = time.perf_counter() - self._start
        self._histogram.observe(duration, self._labels)


class Summary:
    """Prometheus Summary metric with quantiles."""
    
    DEFAULT_QUANTILES = [0.5, 0.9, 0.95, 0.99]
    
    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
        quantiles: List[float] = None,
    ):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self._observations: Dict[tuple, List[float]] = {}
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value."""
        labels = labels or {}
        key = self._label_key(labels)
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append(value)
    
    def samples(self) -> List[MetricSample]:
        """Get all samples including quantiles."""
        result = []
        
        for key, observations in self._observations.items():
            labels = dict(key)
            sorted_obs = sorted(observations)
            n = len(sorted_obs)
            
            if n > 0:
                # Quantiles
                for q in self.quantiles:
                    idx = int(q * n)
                    if idx >= n:
                        idx = n - 1
                    result.append(MetricSample(
                        name=self.name,
                        labels={**labels, "quantile": str(q)},
                        value=sorted_obs[idx],
                    ))
            
            # Sum
            result.append(MetricSample(
                name=f"{self.name}_sum",
                labels=labels,
                value=sum(observations) if observations else 0.0,
            ))
            
            # Count
            result.append(MetricSample(
                name=f"{self.name}_count",
                labels=labels,
                value=float(len(observations)),
            ))
        
        return result


class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self, namespace: str = ""):
        self.namespace = namespace
        self._metrics: Dict[str, Any] = {}
        self._definitions: Dict[str, MetricDefinition] = {}
    
    def _full_name(self, name: str) -> str:
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name
    
    def counter(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
    ) -> Counter:
        """Create or get a counter."""
        full_name = self._full_name(name)
        
        if full_name not in self._metrics:
            self._metrics[full_name] = Counter(full_name, help_text, label_names)
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                type=MetricType.COUNTER,
                help_text=help_text,
                label_names=label_names or [],
            )
        
        return self._metrics[full_name]
    
    def gauge(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
    ) -> Gauge:
        """Create or get a gauge."""
        full_name = self._full_name(name)
        
        if full_name not in self._metrics:
            self._metrics[full_name] = Gauge(full_name, help_text, label_names)
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                type=MetricType.GAUGE,
                help_text=help_text,
                label_names=label_names or [],
            )
        
        return self._metrics[full_name]
    
    def histogram(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
        buckets: List[float] = None,
    ) -> Histogram:
        """Create or get a histogram."""
        full_name = self._full_name(name)
        
        if full_name not in self._metrics:
            self._metrics[full_name] = Histogram(
                full_name, help_text, label_names, buckets
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                type=MetricType.HISTOGRAM,
                help_text=help_text,
                label_names=label_names or [],
                buckets=buckets or Histogram.DEFAULT_BUCKETS,
            )
        
        return self._metrics[full_name]
    
    def summary(
        self,
        name: str,
        help_text: str,
        label_names: List[str] = None,
        quantiles: List[float] = None,
    ) -> Summary:
        """Create or get a summary."""
        full_name = self._full_name(name)
        
        if full_name not in self._metrics:
            self._metrics[full_name] = Summary(
                full_name, help_text, label_names, quantiles
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                type=MetricType.SUMMARY,
                help_text=help_text,
                label_names=label_names or [],
            )
        
        return self._metrics[full_name]
    
    def get_all_samples(self) -> List[MetricSample]:
        """Get all metric samples."""
        samples = []
        for metric in self._metrics.values():
            samples.extend(metric.samples())
        return samples


class PrometheusExporter:
    """
    Exports metrics in Prometheus text format.
    
    Features:
    - Counter, Gauge, Histogram, Summary support
    - Label support
    - Namespace prefixing
    - FastAPI route integration
    """
    
    def __init__(self, namespace: str = "kingai"):
        self.registry = MetricsRegistry(namespace=namespace)
        self._collectors: List[Callable[[], None]] = []
    
    def add_collector(self, collector: Callable[[], None]) -> None:
        """Add a collector function that populates metrics."""
        self._collectors.append(collector)
    
    def collect(self) -> None:
        """Run all collectors."""
        for collector in self._collectors:
            try:
                collector()
            except Exception as e:
                logger.error(f"Collector error: {e}")
    
    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        self.collect()
        
        lines = []
        samples = self.registry.get_all_samples()
        
        # Group by metric name
        by_metric: Dict[str, List[MetricSample]] = {}
        for sample in samples:
            base_name = sample.name.rsplit("_", 1)[0] if "_bucket" in sample.name or "_sum" in sample.name or "_count" in sample.name else sample.name
            if base_name not in by_metric:
                by_metric[base_name] = []
            by_metric[base_name].append(sample)
        
        for metric_name, metric_samples in by_metric.items():
            definition = self.registry._definitions.get(metric_name)
            
            if definition:
                lines.append(f"# HELP {metric_name} {definition.help_text}")
                lines.append(f"# TYPE {metric_name} {definition.type.value}")
            
            for sample in metric_samples:
                label_str = ""
                if sample.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in sample.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{sample.name}{label_str} {sample.value}")
        
        return "\n".join(lines) + "\n"
    
    def get_fastapi_router(self):
        """Get FastAPI router with /metrics endpoint."""
        from fastapi import APIRouter
        from fastapi.responses import PlainTextResponse
        
        router = APIRouter()
        
        @router.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            return self.export()
        
        return router


# Global exporter instance
prometheus_exporter = PrometheusExporter()


def get_prometheus_exporter() -> PrometheusExporter:
    """Get the global Prometheus exporter instance."""
    return prometheus_exporter


# Convenience decorators
def count_calls(counter: Counter, labels: Dict[str, str] = None):
    """Decorator to count function calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            counter.inc(labels)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            counter.inc(labels)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def time_execution(histogram: Histogram, labels: Dict[str, str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                histogram.observe(time.perf_counter() - start, labels)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(time.perf_counter() - start, labels)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Pre-define common metrics
def setup_default_metrics():
    """Set up common application metrics."""
    registry = prometheus_exporter.registry
    
    # HTTP metrics
    registry.counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    
    registry.histogram(
        "http_request_duration_seconds",
        "HTTP request duration",
        ["method", "endpoint"],
    )
    
    # Agent metrics
    registry.counter(
        "agent_executions_total",
        "Total agent executions",
        ["agent_type", "status"],
    )
    
    registry.histogram(
        "agent_execution_duration_seconds",
        "Agent execution duration",
        ["agent_type"],
    )
    
    # LLM metrics
    registry.counter(
        "llm_requests_total",
        "Total LLM requests",
        ["provider", "model"],
    )
    
    registry.counter(
        "llm_tokens_total",
        "Total LLM tokens used",
        ["provider", "model", "type"],
    )
    
    # Business metrics
    registry.gauge(
        "active_businesses",
        "Number of active businesses",
    )
    
    registry.gauge(
        "pending_approvals",
        "Number of pending approvals",
        ["priority"],
    )


# Auto-setup default metrics
setup_default_metrics()
