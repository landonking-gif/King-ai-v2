"""
Monitoring utilities for King AI.
Provides metrics collection and performance tracking.
"""

import time
from typing import Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager


class MonitorStub:
    """
    Stub monitoring client for tracking metrics.
    Can be replaced with Datadog or other monitoring solutions.
    """
    
    def __init__(self):
        self.enabled = False
    
    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        pass
    
    def gauge(self, metric: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        pass
    
    def histogram(self, metric: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        pass
    
    def timing(self, metric: str, value_ms: float, tags: Dict[str, str] = None):
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


# Global monitor instance
monitor = MonitorStub()


# Convenience decorators
def trace_llm(func: Callable) -> Callable:
    """Trace LLM inference calls."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def trace_agent(agent_name: str):
    """Trace agent execution."""
    def decorator(func: Callable) -> Callable:
        return func
    return decorator


def trace_db(func: Callable) -> Callable:
    """Trace database operations."""
    return func
