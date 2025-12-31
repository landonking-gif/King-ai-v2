# FILE: src/monitoring/datadog_integration.py (CREATE NEW FILE)
"""
Datadog Integration - Full observability for King AI.
Implements metrics, traces, and logs shipping to Datadog.
"""

import os
import time
import functools
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger("datadog")


# Check if Datadog is available
try:
    from datadog import initialize, statsd
    from ddtrace import tracer, patch_all
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    logger.warning("Datadog packages not installed")


@dataclass
class DatadogConfig:
    """Datadog configuration."""
    api_key: str
    app_key: str
    service_name: str = "king-ai"
    env: str = "production"
    version: str = "2.0.0"
    statsd_host: str = "localhost"
    statsd_port: int = 8125


class DatadogMonitor:
    """
    Comprehensive Datadog monitoring integration.
    
    Features:
    - Custom metrics (gauges, counters, histograms)
    - Distributed tracing for requests
    - Structured log shipping
    - APM integration
    """
    
    def __init__(self, config: DatadogConfig = None):
        self.enabled = DATADOG_AVAILABLE and os.getenv("DATADOG_API_KEY")
        
        if not self.enabled:
            logger.info("Datadog monitoring disabled (no API key)")
            return
        
        self.config = config or DatadogConfig(
            api_key=os.getenv("DATADOG_API_KEY", ""),
            app_key=os.getenv("DATADOG_APP_KEY", ""),
            env=os.getenv("ENVIRONMENT", "development")
        )
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Datadog clients."""
        if not self.enabled:
            return
        
        # Initialize main client
        initialize(
            api_key=self.config.api_key,
            app_key=self.config.app_key,
            statsd_host=self.config.statsd_host,
            statsd_port=self.config.statsd_port
        )
        
        # Configure tracer
        tracer.configure(
            hostname=self.config.statsd_host,
            port=8126,
            service=self.config.service_name,
            env=self.config.env,
            version=self.config.version
        )
        
        # Auto-patch common libraries
        patch_all()
        
        logger.info("Datadog monitoring initialized")
    
    # --- Metrics ---
    
    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        if not self.enabled:
            return
        
        tag_list = self._format_tags(tags)
        statsd.increment(f"king_ai.{metric}", value, tags=tag_list)
    
    def gauge(self, metric: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        if not self.enabled:
            return
        
        tag_list = self._format_tags(tags)
        statsd.gauge(f"king_ai.{metric}", value, tags=tag_list)
    
    def histogram(self, metric: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        if not self.enabled:
            return
        
        tag_list = self._format_tags(tags)
        statsd.histogram(f"king_ai.{metric}", value, tags=tag_list)
    
    def timing(self, metric: str, value_ms: float, tags: Dict[str, str] = None):
        """Record timing in milliseconds."""
        if not self.enabled:
            return
        
        tag_list = self._format_tags(tags)
        statsd.timing(f"king_ai.{metric}", value_ms, tags=tag_list)
    
    def _format_tags(self, tags: Dict[str, str] = None) -> list:
        """Format tags for Datadog."""
        base_tags = [
            f"service:{self.config.service_name}",
            f"env:{self.config.env}",
            f"version:{self.config.version}"
        ]
        if tags:
            base_tags.extend([f"{k}:{v}" for k, v in tags.items()])
        return base_tags
    
    # --- Tracing ---
    
    def trace(self, operation: str, resource: str = None):
        """Create a trace span decorator."""
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.trace(operation, resource=resource or func.__name__):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.trace(operation, resource=resource or func.__name__):
                    return func(*args, **kwargs)
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    # --- Business Metrics ---
    
    def record_business_metric(
        self,
        business_id: str,
        metric_name: str,
        value: float,
        unit: str = None
    ):
        """Record a business-specific metric."""
        self.gauge(
            f"business.{metric_name}",
            value,
            tags={
                "business_id": business_id,
                "unit": unit or "count"
            }
        )
    
    def record_llm_call(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        success: bool
    ):
        """Record LLM API call metrics."""
        tags = {
            "provider": provider,
            "model": model,
            "success": str(success)
        }
        
        self.increment("llm.calls", tags=tags)
        self.histogram("llm.tokens_in", tokens_in, tags=tags)
        self.histogram("llm.tokens_out", tokens_out, tags=tags)
        self.timing("llm.latency", latency_ms, tags=tags)
    
    def record_agent_task(
        self,
        agent: str,
        task_type: str,
        duration_ms: float,
        success: bool
    ):
        """Record agent task metrics."""
        tags = {
            "agent": agent,
            "task_type": task_type,
            "success": str(success)
        }
        
        self.increment("agent.tasks", tags=tags)
        self.timing("agent.duration", duration_ms, tags=tags)
    
    def record_evolution_proposal(
        self,
        proposal_type: str,
        risk_level: str,
        approved: bool,
        executed: bool = False
    ):
        """Record evolution proposal metrics."""
        self.increment(
            "evolution.proposals",
            tags={
                "type": proposal_type,
                "risk_level": risk_level,
                "approved": str(approved),
                "executed": str(executed)
            }
        )


# Singleton
datadog_monitor = DatadogMonitor()


# Convenience decorators
def trace_llm(func):
    """Decorator to trace LLM calls."""
    return datadog_monitor.trace("llm.inference")(func)


def trace_agent(agent_name: str):
    """Decorator to trace agent operations."""
    def decorator(func):
        return datadog_monitor.trace(f"agent.{agent_name}")(func)
    return decorator


def trace_api(endpoint: str):
    """Decorator to trace API endpoints."""
    def decorator(func):
        return datadog_monitor.trace("api.request", resource=endpoint)(func)
    return decorator