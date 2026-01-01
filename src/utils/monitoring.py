"""
Datadog monitoring integration.
Provides metrics, APM, and alerting capabilities.
"""

import os
import time
import functools
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass
import asyncio

from config.settings import settings

# Conditional import for Datadog
try:
    from ddtrace import tracer, patch_all
    from datadog import initialize, statsd
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    tracer = None
    statsd = None


@dataclass
class MetricTags:
    """Standard tags for metrics."""
    environment: str
    service: str = "king-ai"
    version: str = "2.0.0"


class DatadogMonitor:
    """
    Centralized Datadog monitoring client.
    Provides metrics, tracing, and custom events.
    """
    
    def __init__(self):
        """Initialize Datadog if API key is available."""
        self.enabled = False
        self.tags = MetricTags(
            environment=settings.environment or "development"
        )
        
        dd_api_key = settings.dd_api_key
        if DATADOG_AVAILABLE and dd_api_key:
            initialize(
                api_key=dd_api_key,
                app_key=settings.dd_app_key,
            )
            patch_all()  # Auto-instrument common libraries
            self.enabled = True
    
    def _get_tags(self, extra_tags: Dict[str, str] = None) -> list[str]:
        """Build tag list for metrics."""
        tags = [
            f"env:{self.tags.environment}",
            f"service:{self.tags.service}",
            f"version:{self.tags.version}",
        ]
        if extra_tags:
            tags.extend([f"{k}:{v}" for k, v in extra_tags.items()])
        return tags
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        if self.enabled and statsd:
            statsd.increment(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Set a gauge metric."""
        if self.enabled and statsd:
            statsd.gauge(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Record a histogram value (timing, size, etc.)."""
        if self.enabled and statsd:
            statsd.histogram(
                f"king_ai.{metric}",
                value=value,
                tags=self._get_tags(tags)
            )
    
    def timing(
        self,
        metric: str,
        value_ms: float,
        tags: Dict[str, str] = None
    ):
        """Record a timing metric in milliseconds."""
        self.histogram(metric, value_ms, tags)
    
    @contextmanager
    def timed(self, metric: str, tags: Dict[str, str] = None):
        """Context manager for timing a block of code."""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.timing(metric, elapsed_ms, tags)
    
    def trace(
        self,
        operation_name: str,
        service: str = None,
        resource: str = None
    ):
        """Decorator for tracing a function."""
        def decorator(func: Callable) -> Callable:
            if not self.enabled or not tracer:
                return func
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.trace(
                    operation_name,
                    service=service or self.tags.service,
                    resource=resource or func.__name__
                ):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.trace(
                    operation_name,
                    service=service or self.tags.service,
                    resource=resource or func.__name__
                ):
                    return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Dict[str, str] = None
    ):
        """Send a custom event to Datadog."""
        if self.enabled and statsd:
            statsd.event(
                title=title,
                text=text,
                alert_type=alert_type,
                tags=self._get_tags(tags)
            )


# Global monitor instance
monitor = DatadogMonitor()


# Convenience decorators
def trace_llm(func: Callable) -> Callable:
    """Trace LLM inference calls."""
    return monitor.trace("llm.inference", resource=func.__name__)(func)


def trace_agent(agent_name: str):
    """Trace agent execution."""
    def decorator(func: Callable) -> Callable:
        return monitor.trace(
            "agent.execute",
            resource=agent_name
        )(func)
    return decorator


def trace_db(func: Callable) -> Callable:
    """Trace database operations."""
    return monitor.trace("db.query", resource=func.__name__)(func)
