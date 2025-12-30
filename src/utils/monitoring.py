"""
Monitoring utilities for tracking system metrics and performance.
Provides a simple interface for increment, gauge, and timing metrics.
"""

from typing import Dict, Any, Optional
from src.utils.structured_logging import get_logger
import time
from contextlib import contextmanager

logger = get_logger("monitoring")


class Monitor:
    """
    Simple monitoring client for tracking metrics.
    Logs metrics to structured logs for now, with extensibility for Datadog/Prometheus.
    """
    
    def __init__(self):
        """Initialize the monitor."""
        self.enabled = True
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        if self.enabled:
            logger.info(
                "metric.increment",
                metric=metric,
                value=value,
                tags=tags or {}
            )
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Dict[str, str] = None
    ):
        """Set a gauge metric."""
        if self.enabled:
            logger.info(
                "metric.gauge",
                metric=metric,
                value=value,
                tags=tags or {}
            )
    
    def timing(
        self,
        metric: str,
        value_ms: float,
        tags: Dict[str, str] = None
    ):
        """Record a timing metric in milliseconds."""
        if self.enabled:
            logger.info(
                "metric.timing",
                metric=metric,
                value_ms=value_ms,
                tags=tags or {}
            )
    
    @contextmanager
    def timed(self, metric: str, tags: Dict[str, str] = None):
        """Context manager for timing a block of code."""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.timing(metric, elapsed_ms, tags)


# Global monitor instance
monitor = Monitor()
